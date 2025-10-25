import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
import time
from ccpy.energy.cc_energy import get_cc_energy
from ccpy.lib.core import hbar_cc3, cc3_loops
from ccpy.models.integrals import Integral
from ccpy.models.operators import ClusterOperator

def build_hbar_ccsdta(T, H0, RHF_symmetry, system, *args):
    """Calculate the one- and two-body components of the CCSD(T)(a) similarity-transformed
     Hamiltonian (H_N e^(T1+T2))_C + Delta, where T3 = <ijkabc|(V_N*T2)_C|0>/-D_MP, where
     D_MP = e_a+e_b+e_c-e_i-e_j-e_k."""
    from copy import deepcopy

    save_t3 = True

    nua, noa = T.a.shape
    nub, nob = T.b.shape

    resid_a = np.zeros((nua, noa))
    resid_b = np.zeros((nub, nob))
    resid_aa = np.zeros((nua, nua, noa, noa))
    resid_ab = np.zeros((nua, nub, noa, nob))
    resid_bb = np.zeros((nub, nub, nob, nob))

    # Copy the Bare Hamiltonian object for T1/T2-similarity transformed HBar
    H = deepcopy(H0)

    # Add in the t3-dependent terms to Hbar computed on-the-fly
    H.aa.vooo, H.aa.vvov, H.ab.vooo, H.ab.ovoo, H.ab.vvov, H.ab.vvvo, H.bb.vooo, H.bb.vvov = hbar_cc3.build_hbar(
            H.aa.vooo, H.aa.vvov,
            H.ab.vooo, H.ab.ovoo, H.ab.vvov, H.ab.vvvo,
            H.bb.vooo, H.bb.vvov,
            T.aa, T.ab, T.bb,
            H0.aa.vooo, H0.aa.vvov,
            H0.ab.vooo, H0.ab.ovoo, H0.ab.vvov, H0.ab.vvvo,
            H0.bb.vooo, H0.bb.vvov,
            H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
            H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
    )

    # It is sometimes useful to store T3 for future use
    if save_t3:
        t_start = time.perf_counter()
        print("   >> Constructing and storing T3[2] = R3 (V_N*T2)_C |0> <<")
        # Prepare MP denominatores
        n = np.newaxis
        eps_o_a = np.diagonal(H0.a.oo); eps_v_a = np.diagonal(H0.a.vv);
        eps_o_b = np.diagonal(H0.b.oo); eps_v_b = np.diagonal(H0.b.vv);
        #
        e3_aaa = (eps_o_a[n, n, n, :, n, n] + eps_o_a[n, n, n, n, :, n] + eps_o_a[n, n, n, n, n, :]
                 -eps_v_a[:, n, n, n, n, n] - eps_v_a[n, :, n ,n, n, n] - eps_v_a[n, n, :, n, n, n])
        e3_aab = (eps_o_a[n, n, n, :, n, n] + eps_o_a[n, n, n, n, :, n] + eps_o_b[n, n, n, n, n, :]
                 -eps_v_a[:, n, n, n, n, n] - eps_v_a[n, :, n ,n, n, n] - eps_v_b[n, n, :, n, n, n])
        e3_abb = (eps_o_a[n, n, n, :, n, n] + eps_o_b[n, n, n, n, :, n] + eps_o_b[n, n, n, n, n, :]
                 -eps_v_a[:, n, n, n, n, n] - eps_v_b[n, :, n ,n, n, n] - eps_v_b[n, n, :, n, n, n])
        e3_bbb = (eps_o_b[n, n, n, :, n, n] + eps_o_b[n, n, n, n, :, n] + eps_o_b[n, n, n, n, n, :]
                 -eps_v_b[:, n, n, n, n, n] - eps_v_b[n, :, n ,n, n, n] - eps_v_b[n, n, :, n, n, n])
        # Prepare new cluster operator
        T_new = ClusterOperator(system, order=3)
        T_new.a = T.a.copy()
        T_new.b = T.b.copy()
        T_new.aa = T.aa.copy()
        T_new.ab = T.ab.copy()
        T_new.bb = T.bb.copy()
        # t(aaa)
        T_new.aaa = -0.25 * ccpy_einsum("amij,bcmk->abcijk", H0.aa.vooo, T.aa)
        T_new.aaa += 0.25 * ccpy_einsum("abie,ecjk->abcijk", H0.aa.vvov, T.aa)
        T_new.aaa /= e3_aaa
        T_new.aaa -= np.transpose(T_new.aaa, (0, 1, 2, 3, 5, 4)) # (jk)
        T_new.aaa -= np.transpose(T_new.aaa, (0, 1, 2, 4, 3, 5)) + np.transpose(T_new.aaa, (0, 1, 2, 5, 4, 3)) # (i/jk)
        T_new.aaa -= np.transpose(T_new.aaa, (0, 2, 1, 3, 4, 5)) # (bc)
        T_new.aaa -= np.transpose(T_new.aaa, (2, 1, 0, 3, 4, 5)) + np.transpose(T_new.aaa, (1, 0, 2, 3, 4, 5)) # (a/bc)
        # t(aab)
        T_new.aab = 0.5 * ccpy_einsum("bcek,aeij->abcijk", H0.ab.vvvo, T.aa)
        T_new.aab -= 0.5 * ccpy_einsum("mcjk,abim->abcijk", H0.ab.ovoo, T.aa)
        T_new.aab += ccpy_einsum("acie,bejk->abcijk", H0.ab.vvov, T.ab)
        T_new.aab -= ccpy_einsum("amik,bcjm->abcijk", H0.ab.vooo, T.ab)
        T_new.aab += 0.5 * ccpy_einsum("abie,ecjk->abcijk", H0.aa.vvov, T.ab)
        T_new.aab -= 0.5 * ccpy_einsum("amij,bcmk->abcijk", H0.aa.vooo, T.ab)
        T_new.aab /= e3_aab
        T_new.aab -= np.transpose(T_new.aab, (1, 0, 2, 3, 4, 5))
        T_new.aab -= np.transpose(T_new.aab, (0, 1, 2, 4, 3, 5))
        # t(abb)
        T_new.abb = 0.5 * ccpy_einsum("abie,ecjk->abcijk", H0.ab.vvov, T.bb)
        T_new.abb -= 0.5 * ccpy_einsum("amij,bcmk->abcijk", H0.ab.vooo, T.bb)
        T_new.abb += 0.5 * ccpy_einsum("cbke,aeij->abcijk", H0.bb.vvov, T.ab)
        T_new.abb -= 0.5 * ccpy_einsum("cmkj,abim->abcijk", H0.bb.vooo, T.ab)
        T_new.abb += ccpy_einsum("abej,ecik->abcijk", H0.ab.vvvo, T.ab)
        T_new.abb -= ccpy_einsum("mbij,acmk->abcijk", H0.ab.ovoo, T.ab)
        T_new.abb /= e3_abb
        T_new.abb -= np.transpose(T_new.abb, (0, 2, 1, 3, 4, 5))
        T_new.abb -= np.transpose(T_new.abb, (0, 1, 2, 3, 5, 4))
        # t(bbb)
        T_new.bbb = -0.25 * ccpy_einsum("amij,bcmk->abcijk", H0.bb.vooo, T.bb)
        T_new.bbb += 0.25 * ccpy_einsum("abie,ecjk->abcijk", H0.bb.vvov, T.bb)
        T_new.bbb /= e3_bbb
        T_new.bbb -= np.transpose(T_new.bbb, (0, 1, 2, 3, 5, 4)) # (jk)
        T_new.bbb -= np.transpose(T_new.bbb, (0, 1, 2, 4, 3, 5)) + np.transpose(T_new.bbb, (0, 1, 2, 5, 4, 3)) # (i/jk)
        T_new.bbb -= np.transpose(T_new.bbb, (0, 2, 1, 3, 4, 5)) # (bc)
        T_new.bbb -= np.transpose(T_new.bbb, (2, 1, 0, 3, 4, 5)) + np.transpose(T_new.bbb, (1, 0, 2, 3, 4, 5)) # (a/bc)
        # replace T
        T = T_new
        t_end = time.perf_counter()
        print(f"   Completed in {t_end - t_start} seconds")

    # Update T1/T2 with T3[2]
    T.a, T.b, T.aa, T.ab, T.bb, _, _, _, _, _ = cc3_loops.update_t(
        T.a, T.b, T.aa, T.ab, T.bb,
        resid_a, resid_b, resid_aa, resid_ab, resid_bb,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        H0.a.ov, H0.b.ov,
        H0.aa.oovv, H0.ab.oovv, H0.bb.oovv,
        H0.aa.ooov, H0.aa.vovv,
        H0.ab.ooov, H0.ab.oovo, H0.ab.vovv, H0.ab.ovvv,
        H0.bb.ooov, H0.bb.vovv,
        H0.aa.vooo, H0.aa.vvov,
        H0.ab.vooo, H0.ab.ovoo, H0.ab.vvov, H0.ab.vvvo,
        H0.bb.vooo, H0.bb.vvov,
        0.0)

    cc_energy = get_cc_energy(T, H0)
    print(f"   CCSD(T)(a) Correlation Energy: {cc_energy}")
    print(f"   CCSD(T)(a) Total Energy: {cc_energy + system.reference_energy}")

    H.a.ov += (
            ccpy_einsum("imae,em->ia", H0.aa.oovv, T.a)
            + ccpy_einsum("imae,em->ia", H0.ab.oovv, T.b)
    )

    H.a.oo += (
            ccpy_einsum("je,ei->ji", H.a.ov, T.a)
            + ccpy_einsum("jmie,em->ji", H0.aa.ooov, T.a)
            + ccpy_einsum("jmie,em->ji", H0.ab.ooov, T.b)
            + 0.5 * ccpy_einsum("jnef,efin->ji", H0.aa.oovv, T.aa)
            + ccpy_einsum("jnef,efin->ji", H0.ab.oovv, T.ab)
    )

    H.a.vv += (
            - ccpy_einsum("mb,am->ab", H.a.ov, T.a)
            + ccpy_einsum("ambe,em->ab", H0.aa.vovv, T.a)
            + ccpy_einsum("ambe,em->ab", H0.ab.vovv, T.b)
            - 0.5 * ccpy_einsum("mnbf,afmn->ab", H0.aa.oovv, T.aa)
            - ccpy_einsum("mnbf,afmn->ab", H0.ab.oovv, T.ab)
    )

    H.b.ov += (
            ccpy_einsum("imae,em->ia", H0.bb.oovv, T.b)
            + ccpy_einsum("miea,em->ia", H0.ab.oovv, T.a)
    )

    H.b.oo += (
            ccpy_einsum("je,ei->ji", H.b.ov, T.b)
            + ccpy_einsum("jmie,em->ji", H0.bb.ooov, T.b)
            + ccpy_einsum("mjei,em->ji", H0.ab.oovo, T.a)
            + 0.5 * ccpy_einsum("jnef,efin->ji", H0.bb.oovv, T.bb)
            + ccpy_einsum("njfe,feni->ji", H0.ab.oovv, T.ab)
    )

    H.b.vv += (
            - ccpy_einsum("mb,am->ab", H.b.ov, T.b)
            + ccpy_einsum("ambe,em->ab", H0.bb.vovv, T.b)
            + ccpy_einsum("maeb,em->ab", H0.ab.ovvv, T.a)
            - 0.5 * ccpy_einsum("mnbf,afmn->ab", H0.bb.oovv, T.bb)
            - ccpy_einsum("nmfb,fanm->ab", H0.ab.oovv, T.ab)
    )

    Q1 = -ccpy_einsum("mnfe,an->amef", H0.aa.oovv, T.a)
    I2A_vovv = H0.aa.vovv + 0.5 * Q1
    H.aa.vovv = I2A_vovv + 0.5 * Q1

    Q1 = ccpy_einsum("mnfe,fi->mnie", H0.aa.oovv, T.a)
    I2A_ooov = H0.aa.ooov + 0.5 * Q1
    H.aa.ooov = I2A_ooov + 0.5 * Q1

    Q1 = -ccpy_einsum("nmef,an->amef", H0.ab.oovv, T.a)
    I2B_vovv = H0.ab.vovv + 0.5 * Q1
    H.ab.vovv = I2B_vovv + 0.5 * Q1

    Q1 = ccpy_einsum("mnfe,fi->mnie", H0.ab.oovv, T.a)
    I2B_ooov = H0.ab.ooov + 0.5 * Q1
    H.ab.ooov = I2B_ooov + 0.5 * Q1

    Q1 = -ccpy_einsum("mnef,an->maef", H0.ab.oovv, T.b)
    I2B_ovvv = H0.ab.ovvv + 0.5 * Q1
    H.ab.ovvv = I2B_ovvv + 0.5 * Q1

    Q1 = ccpy_einsum("nmef,fi->nmei", H0.ab.oovv, T.b)
    I2B_oovo = H0.ab.oovo + 0.5 * Q1
    H.ab.oovo = I2B_oovo + 0.5 * Q1

    Q1 = -ccpy_einsum("nmef,an->amef", H0.bb.oovv, T.b)
    I2C_vovv = H0.bb.vovv + 0.5 * Q1
    H.bb.vovv = I2C_vovv + 0.5 * Q1

    Q1 = ccpy_einsum("mnfe,fi->mnie", H0.bb.oovv, T.b)
    I2C_ooov = H0.bb.ooov + 0.5 * Q1
    H.bb.ooov = I2C_ooov + 0.5 * Q1

    Q1 = -ccpy_einsum("bmfe,am->abef", I2A_vovv, T.a)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvvv += 0.5 * ccpy_einsum("mnef,abmn->abef", H0.aa.oovv, T.aa) + Q1

    H.ab.vvvv += (
            - ccpy_einsum("mbef,am->abef", I2B_ovvv, T.a)
            - ccpy_einsum("amef,bm->abef", I2B_vovv, T.b)
            + ccpy_einsum("mnef,abmn->abef", H0.ab.oovv, T.ab)
    )

    Q1 = -ccpy_einsum("bmfe,am->abef", I2C_vovv, T.b)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvvv += 0.5 * ccpy_einsum("mnef,abmn->abef", H0.bb.oovv, T.bb) + Q1

    Q1 = +ccpy_einsum("nmje,ei->mnij", I2A_ooov, T.a)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.oooo += 0.5 * ccpy_einsum("mnef,efij->mnij", H0.aa.oovv, T.aa) + Q1

    H.ab.oooo += (
            ccpy_einsum("mnej,ei->mnij", I2B_oovo, T.a)
            + ccpy_einsum("mnie,ej->mnij", I2B_ooov, T.b)
            + ccpy_einsum("mnef,efij->mnij", H0.ab.oovv, T.ab)
    )

    Q1 = +ccpy_einsum("nmje,ei->mnij", I2C_ooov, T.b)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.oooo += 0.5 * ccpy_einsum("mnef,efij->mnij", H0.bb.oovv, T.bb) + Q1

    H.aa.voov += (
            ccpy_einsum("amfe,fi->amie", I2A_vovv, T.a)
            - ccpy_einsum("nmie,an->amie", I2A_ooov, T.a)
            + ccpy_einsum("nmfe,afin->amie", H0.aa.oovv, T.aa)
            + ccpy_einsum("mnef,afin->amie", H0.ab.oovv, T.ab)
    )

    H.ab.voov += (
            ccpy_einsum("amfe,fi->amie", I2B_vovv, T.a)
            - ccpy_einsum("nmie,an->amie", I2B_ooov, T.a)
            + ccpy_einsum("nmfe,afin->amie", H0.ab.oovv, T.aa)
            + ccpy_einsum("nmfe,afin->amie", H0.bb.oovv, T.ab)
    )

    H.ab.ovvo += (
            ccpy_einsum("maef,fi->maei", I2B_ovvv, T.b)
            - ccpy_einsum("mnei,an->maei", I2B_oovo, T.b)
            + ccpy_einsum("mnef,afin->maei", H0.ab.oovv, T.bb)
            + ccpy_einsum("mnef,fani->maei", H0.aa.oovv, T.ab)
    )

    H.ab.ovov += (
            ccpy_einsum("mafe,fi->maie", I2B_ovvv, T.a)
            - ccpy_einsum("mnie,an->maie", I2B_ooov, T.b)
            - ccpy_einsum("mnfe,fain->maie", H0.ab.oovv, T.ab)
    )

    H.ab.vovo += (
            - ccpy_einsum("nmei,an->amei", I2B_oovo, T.a)
            + ccpy_einsum("amef,fi->amei", I2B_vovv, T.b)
            - ccpy_einsum("nmef,afni->amei", H0.ab.oovv, T.ab)
    )

    H.bb.voov += (
            ccpy_einsum("amfe,fi->amie", I2C_vovv, T.b)
            - ccpy_einsum("nmie,an->amie", I2C_ooov, T.b)
            + ccpy_einsum("nmfe,afin->amie", H0.bb.oovv, T.bb)
            + ccpy_einsum("nmfe,fani->amie", H0.ab.oovv, T.ab)
    )

    Q1 = (
            ccpy_einsum("mnjf,afin->amij", H.aa.ooov, T.aa)
            + ccpy_einsum("mnjf,afin->amij", H.ab.ooov, T.ab)
    )
    Q2 = H0.aa.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.aa.vovv, T.a)
    Q2 = ccpy_einsum("amif,fj->amij", Q2, T.a)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.vooo += Q1 + (
            ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
            - ccpy_einsum("nmij,an->amij", H.aa.oooo, T.a)
            + 0.5 * ccpy_einsum("amef,efij->amij", H0.aa.vovv, T.aa)
    )

    Q1 = H0.ab.voov + ccpy_einsum("amfe,fi->amie", H0.ab.vovv, T.a)
    H.ab.vooo += (
            ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
            - ccpy_einsum("nmij,an->amij", H.ab.oooo, T.a)
            + ccpy_einsum("mnjf,afin->amij", H.bb.ooov, T.ab)
            + ccpy_einsum("nmfj,afin->amij", H.ab.oovo, T.aa)
            - ccpy_einsum("nmif,afnj->amij", H.ab.ooov, T.ab)
            + ccpy_einsum("amej,ei->amij", H0.ab.vovo, T.a)
            + ccpy_einsum("amie,ej->amij", Q1, T.b)
            + ccpy_einsum("amef,efij->amij", H0.ab.vovv, T.ab)
    )

    Q1 = H0.ab.ovov + ccpy_einsum("mafe,fj->maje", H0.ab.ovvv, T.a)
    H.ab.ovoo += (
            ccpy_einsum("me,eaji->maji", H.a.ov, T.ab)
            - ccpy_einsum("mnji,an->maji", H.ab.oooo, T.b)
            + ccpy_einsum("mnjf,fani->maji", H.aa.ooov, T.ab)
            + ccpy_einsum("mnjf,fani->maji", H.ab.ooov, T.bb)
            - ccpy_einsum("mnfi,fajn->maji", H.ab.oovo, T.ab)
            + ccpy_einsum("maje,ei->maji", Q1, T.b)
            + ccpy_einsum("maei,ej->maji", H0.ab.ovvo, T.a)
            + ccpy_einsum("mafe,feji->maji", H0.ab.ovvv, T.ab)
    )

    Q1 = (
            ccpy_einsum("mnjf,afin->amij", H.bb.ooov, T.bb)
            + ccpy_einsum("nmfj,fani->amij", H.ab.oovo, T.ab)
    )
    Q2 = H0.bb.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.bb.vovv, T.b)
    Q2 = ccpy_einsum("amif,fj->amij", Q2, T.b)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.vooo += Q1 + (
            + ccpy_einsum("me,aeij->amij", H.b.ov, T.bb)
            - ccpy_einsum("nmij,an->amij", H.bb.oooo, T.b)
            + 0.5 * ccpy_einsum("amef,efij->amij", H0.bb.vovv, T.bb)
    )

    Q1 = (
            ccpy_einsum("bnef,afin->abie", H.aa.vovv, T.aa)
            + ccpy_einsum("bnef,afin->abie", H.ab.vovv, T.ab)
    )
    Q2 = H0.aa.ovov - 0.5 * ccpy_einsum("mnie,bn->mbie", H0.aa.ooov, T.a)
    Q2 = -ccpy_einsum("mbie,am->abie", Q2, T.a)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvov += Q1 + (
            - ccpy_einsum("me,abim->abie", H.a.ov, T.aa)
            + ccpy_einsum("abfe,fi->abie", H.aa.vvvv, T.a)
            + 0.5 * ccpy_einsum("mnie,abmn->abie", H0.aa.ooov, T.aa)
    )

    Q1 = H0.ab.ovov - ccpy_einsum("mnie,bn->mbie", H0.ab.ooov, T.b)
    Q1 = -ccpy_einsum("mbie,am->abie", Q1, T.a)
    H.ab.vvov += Q1 + (
            - ccpy_einsum("me,abim->abie", H.b.ov, T.ab)
            + ccpy_einsum("abfe,fi->abie", H.ab.vvvv, T.a)
            + ccpy_einsum("nbfe,afin->abie", H.ab.ovvv, T.aa)
            + ccpy_einsum("bnef,afin->abie", H.bb.vovv, T.ab)
            - ccpy_einsum("amfe,fbim->abie", H.ab.vovv, T.ab)
            - ccpy_einsum("amie,bm->abie", H0.ab.voov, T.b)
            + ccpy_einsum("nmie,abnm->abie", H0.ab.ooov, T.ab)
    )

    Q1 = H0.ab.vovo - ccpy_einsum("nmei,bn->bmei", H0.ab.oovo, T.a)
    Q1 = -ccpy_einsum("bmei,am->baei", Q1, T.b)
    H.ab.vvvo += Q1 + (
            - ccpy_einsum("me,bami->baei", H.a.ov, T.ab)
            + ccpy_einsum("baef,fi->baei", H.ab.vvvv, T.b)
            + ccpy_einsum("bnef,fani->baei", H.aa.vovv, T.ab)
            + ccpy_einsum("bnef,fani->baei", H.ab.vovv, T.bb)
            - ccpy_einsum("maef,bfmi->baei", H.ab.ovvv, T.ab)
            - ccpy_einsum("naei,bn->baei", H0.ab.ovvo, T.a)
            + ccpy_einsum("nmei,banm->baei", H0.ab.oovo, T.ab)
    )

    Q1 = (
            ccpy_einsum("bnef,afin->abie", H.bb.vovv, T.bb)
            + ccpy_einsum("nbfe,fani->abie", H.ab.ovvv, T.ab)
    )
    Q2 = H.bb.ovov - 0.5 * ccpy_einsum("mnie,bn->mbie", H0.bb.ooov, T.b)
    Q2 = -ccpy_einsum("mbie,am->abie", Q2, T.b)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvov += Q1 + (
            - ccpy_einsum("me,abim->abie", H.b.ov, T.bb)
            + ccpy_einsum("abfe,fi->abie", H.bb.vvvv, T.b)
            + 0.5 * ccpy_einsum("mnie,abmn->abie", H0.bb.ooov, T.bb)
    )

    # For RHF symmetry, copy a parts to b and aa parts to bb
    if RHF_symmetry:
        H.b.ov = H.a.ov.copy()
        H.b.oo = H.a.oo.copy()
        H.b.vv = H.a.vv.copy()
        H.bb.oooo = H.aa.oooo.copy()
        H.bb.ooov = H.aa.ooov.copy()
        H.bb.vooo = H.aa.vooo.copy()
        H.bb.oovv = H.aa.oovv.copy()
        H.bb.voov = H.aa.voov.copy()
        H.bb.vovv = H.aa.vovv.copy()
        H.bb.vvov = H.aa.vvov.copy()
        H.bb.vvvv = H.aa.vvvv.copy()

    return H, T, cc_energy
