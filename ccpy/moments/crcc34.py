"""Functions to calculate the ground-state CR-CC(3,4) quadruples correction to CCSDT."""
import time
from ccpy.utilities.linear_algebra import ccpy_einsum

import numpy as np
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal

def calc_crcc34(T, L, corr_energy, H, H0, system, use_RHF):
    """
    Calculate the ground-state CR-CC(3,4) correction to the CCSDT energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # build MP denoms in full
    oa = np.diagonal(H0.a.oo)
    ob = np.diagonal(H0.b.oo)
    va = np.diagonal(H0.a.vv)
    vb = np.diagonal(H0.b.vv)
    n = np.newaxis
    e4_aaaa = 1.0 / (- va[:, n, n, n, n, n, n, n] - va[n, :, n, n, n, n, n, n] - va[n, n, :, n, n, n, n, n] - va[n, n, n, :, n, n, n, n]
                        + oa[n, n, n, n, :, n, n, n] + oa[n, n, n, n, n, :, n, n] + oa[n, n, n, n, n, n, :, n] + oa[n, n, n, n, n, n, n, :])
    e4_aaab = 1.0 / (- va[:, n, n, n, n, n, n, n] - va[n, :, n, n, n, n, n, n] - va[n, n, :, n, n, n, n, n] - vb[n, n, n, :, n, n, n, n]
                        + oa[n, n, n, n, :, n, n, n] + oa[n, n, n, n, n, :, n, n] + oa[n, n, n, n, n, n, :, n] + ob[n, n, n, n, n, n, n, :])
    e4_aabb = 1.0 / (- va[:, n, n, n, n, n, n, n] - va[n, :, n, n, n, n, n, n] - vb[n, n, :, n, n, n, n, n] - vb[n, n, n, :, n, n, n, n]
                        + oa[n, n, n, n, :, n, n, n] + oa[n, n, n, n, n, :, n, n] + ob[n, n, n, n, n, n, :, n] + ob[n, n, n, n, n, n, n, :])
    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    # form the diagonal part of the h(vvvv) elements
    nua, nub, noa, nob = T.ab.shape
    h_aa_vvvv = np.zeros((nua, nua))
    for a in range(nua):
        for b in range(a, nua):
            h_aa_vvvv[a, b] = H.aa.vvvv[a, b, a, b]
            h_aa_vvvv[b, a] = h_aa_vvvv[a, b]
    h_ab_vvvv = np.zeros((nua, nub))
    for a in range(nua):
        for b in range(nub):
            h_ab_vvvv[a, b] = H.ab.vvvv[a, b, a, b]
    h_bb_vvvv = np.zeros((nub, nub))
    for a in range(nub):
        for b in range(a, nub):
            h_bb_vvvv[a, b] = H.bb.vvvv[a, b, a, b]
            h_bb_vvvv[b, a] = h_bb_vvvv[a, b]

    #### aaaa correction ####
    M4_aaaa = build_m4a(H, T)
    L4_aaaa = build_l4a(H, L)
    L4_aaaa *= e4_aaaa
    dA_aaaa = (1.0 / 576.0) * np.sum(M4_aaaa * L4_aaaa)
    #### aaab correction ####
    M4_aaab = build_m4b(H, T)
    L4_aaab = build_l4b(H, L)
    L4_aaab *= e4_aaab
    dA_aaab = (1.0 / 36.0) * np.sum(M4_aaab * L4_aaab)
    #### aabb correction ####
    M4_aabb = build_m4c(H, T)
    L4_aabb = build_l4c(H, L)
    L4_aabb *= e4_aabb
    dA_aabb = (1.0 / 16.0) * np.sum(M4_aabb * L4_aabb)
    if use_RHF:
        correction_A = 2.0 * dA_aaaa + 2.0 * dA_aaab + dA_aabb
        correction_B = 0.0
        correction_C = 0.0
        correction_D = 0.0
    else:
        pass

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    energy_A = corr_energy + correction_A
    energy_B = corr_energy + correction_B
    energy_C = corr_energy + correction_C
    energy_D = corr_energy + correction_D

    total_energy_A = system.reference_energy + energy_A
    total_energy_B = system.reference_energy + energy_B
    total_energy_C = system.reference_energy + energy_C
    total_energy_D = system.reference_energy + energy_D

    print('   CR-CC(3,4) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   CCSDT = {:>10.10f}".format(system.reference_energy + corr_energy))
    print(
        "   CR-CC(3,4)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CR-CC(3,4)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CR-CC(3,4)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CR-CC(3,4)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )

    Ecrcc34 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    delta34 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}
    
    return Ecrcc34, delta34

def build_m4a(H, T):
    # <ijklabcd | H(2) | 0 >
    m4a = -(144.0 / 576.0) * ccpy_einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.aa, T.aa)  # (jl/i/k)(bc/a/d) = 12 * 12 = 144
    m4a += (36.0 / 576.0) * ccpy_einsum("mnij,adml,bcnk->abcdijkl", H.aa.oooo, T.aa, T.aa)  # (ij/kl)(bc/ad) = 6 * 6 = 36
    m4a += (36.0 / 576.0) * ccpy_einsum("abef,fcjk,edil->abcdijkl", H.aa.vvvv, T.aa, T.aa)  # (jk/il)(ab/cd) = 6 * 6 = 36

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    m4a += (24.0 / 576.0) * ccpy_einsum("cdke,abeijl->abcdijkl", H.aa.vvov, T.aaa)  # (cd/ab)(k/ijl) = 6 * 4 = 24
    m4a -= (24.0 / 576.0) * ccpy_einsum("cmkl,abdijm->abcdijkl", H.aa.vooo, T.aaa)  # (c/abd)(kl/ij) = 6 * 4 = 24

    I3A_vooooo = ccpy_einsum("nmle,bejk->bmnjkl", H.aa.ooov, T.aa)
    I3A_vooooo -= np.transpose(I3A_vooooo, (0, 1, 2, 5, 4, 3)) + np.transpose(I3A_vooooo, (0, 1, 2, 3, 5, 4))
    I3A_vooooo += 0.5 * ccpy_einsum("mnef,befjkl->bmnjkl", H.aa.oovv, T.aaa)
    m4a += 0.5 * (16.0 / 576.0) * ccpy_einsum("bmnjkl,acdimn->abcdijkl", I3A_vooooo, T.aaa)  # (b/acd)(i/jkl) = 4 * 4 = 16

    I3A_vvvovv = -ccpy_einsum("dmfe,bcjm->bcdjef", H.aa.vovv, T.aa)
    I3A_vvvovv -= np.transpose(I3A_vvvovv, (2, 1, 0, 3, 4, 5)) + np.transpose(I3A_vvvovv, (0, 2, 1, 3, 4, 5))
    m4a += 0.5 * (16.0 / 576.0) * ccpy_einsum("bcdjef,aefikl->abcdijkl", I3A_vvvovv, T.aaa)  # (a/bcd)(j/ikl) = 4 * 4 = 16

    I3A_vvooov = (
            -0.5 * ccpy_einsum("nmke,cdnl->cdmkle", H.aa.ooov, T.aa)
            + 0.5 * ccpy_einsum("cmfe,fdkl->cdmkle", H.aa.vovv, T.aa)
            + 0.125 * ccpy_einsum("mnef,cdfkln->cdmkle", H.aa.oovv, T.aaa)  # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
            + 0.25 * ccpy_einsum("mnef,cdfkln->cdmkle", H.ab.oovv, T.aab)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    m4a += (36.0 / 576.0) * ccpy_einsum("cdmkle,abeijm->abcdijkl", I3A_vvooov, T.aaa)  # (cd/ab)(kl/ij) = 6 * 6 = 36

    I3B_vvooov = (
            -0.5 * ccpy_einsum("nmke,cdnl->cdmkle", H.ab.ooov, T.aa)
            + 0.5 * ccpy_einsum("cmfe,fdkl->cdmkle", H.ab.vovv, T.aa)
            + 0.125 * ccpy_einsum("mnef,cdfkln->cdmkle", H.bb.oovv, T.aab)
    # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    m4a += (36.0 / 576.0) * ccpy_einsum("cdmkle,abeijm->abcdijkl", I3B_vvooov, T.aab)  # (cd/ab)(kl/ij) = 6 * 6 = 36
    # antisymmetrize
    m4a -= np.transpose(m4a, (0, 1, 2, 3, 4, 6, 5, 7)) # (jk)
    m4a -= np.transpose(m4a, (0, 1, 2, 3, 4, 7, 6, 5)) + np.transpose(m4a, (0, 1, 2, 3, 4, 5, 7, 6)) # (l/jk)
    m4a -= np.transpose(m4a, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(m4a, (0, 1, 2, 3, 6, 5, 4, 7)) + np.transpose(m4a, (0, 1, 2, 3, 7, 5, 6, 4)) # (i/jkl)
    m4a -= np.transpose(m4a, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    m4a -= np.transpose(m4a, (0, 3, 2, 1, 4, 5, 6, 7)) + np.transpose(m4a, (0, 1, 3, 2, 4, 5, 6, 7)) # (d/bc)
    m4a -= np.transpose(m4a, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(m4a, (2, 1, 0, 3, 4, 5, 6, 7)) + np.transpose(m4a, (3, 1, 2, 0, 4, 5, 6, 7)) # (a/bcd)
    return m4a

def build_m4b(H, T):
    # <ijklabcd | H(2) | 0 >
    m4b = -(9.0 / 36.0) * ccpy_einsum("mdel,abim,ecjk->abcdijkl", H.ab.ovvo, T.aa, T.aa)    # (i/jk)(c/ab) = 9
    m4b += (9.0 / 36.0) * ccpy_einsum("mnij,bcnk,adml->abcdijkl", H.aa.oooo, T.aa, T.ab)    # (k/ij)(a/bc) = 9
    m4b -= (18.0 / 36.0) * ccpy_einsum("mdjf,abim,cfkl->abcdijkl", H.ab.ovov, T.aa, T.ab)   # (ijk)(c/ab) = (i/jk)(c/ab)(jk) = 18
    m4b -= ccpy_einsum("amie,bejl,cdkm->abcdijkl", H.ab.voov, T.ab, T.ab)                   # (ijk)(abc) = (i/jk)(a/bc)(jk)(bc) = 36
    m4b += (18.0 / 36.0) * ccpy_einsum("mnjl,bcmk,adin->abcdijkl", H.ab.oooo, T.aa, T.ab)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    m4b -= (18.0 / 36.0) * ccpy_einsum("bmel,ecjk,adim->abcdijkl", H.ab.vovo, T.aa, T.ab)   # (i/jk)(abc) = (i/jk)(a/bc)(bc) = 18
    m4b -= (18.0 / 36.0) * ccpy_einsum("amie,ecjk,bdml->abcdijkl", H.aa.voov, T.aa, T.ab)   # (i/kj)(abc) = (i/kj)(a/bc)(bc) = 18
    m4b += (9.0 / 36.0) * ccpy_einsum("abef,fcjk,edil->abcdijkl", H.aa.vvvv, T.aa, T.ab)    # (i/jk)(c/ab) = (i/jk)(c/ab) = 9
    m4b -= (18.0 / 36.0) * ccpy_einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.aa, T.ab)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    m4b += (18.0 / 36.0) * ccpy_einsum("adef,ebij,cfkl->abcdijkl", H.ab.vvvv, T.aa, T.ab)   # (k/ij)(abc) = (k/ij)(a/bc)(bc) = 18

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    m4b -= (1.0 / 12.0) * ccpy_einsum("mdkl,abcijm->abcdijkl", H.ab.ovoo, T.aaa)  # (k/ij) = 3
    m4b -= (9.0 / 36.0) * ccpy_einsum("amik,bcdjml->abcdijkl", H.aa.vooo, T.aab)  # (j/ik)(a/bc) = 9
    m4b -= (9.0 / 36.0) * ccpy_einsum("amil,bcdjkm->abcdijkl", H.ab.vooo, T.aab)  # (a/bc)(i/jk) = 9

    m4b += (1.0 / 12.0) * ccpy_einsum("cdel,abeijk->abcdijkl", H.ab.vvvo, T.aaa)  # (c/ab) = 3
    m4b += (9.0 / 36.0) * ccpy_einsum("acie,bedjkl->abcdijkl", H.aa.vvov, T.aab)  # (b/ac)(i/jk) = 9
    m4b += (9.0 / 36.0) * ccpy_einsum("adie,bcejkl->abcdijkl", H.ab.vvov, T.aab)  # (a/bc)(i/jk) = 9

    I3B_oovooo = (
                    ccpy_einsum("mnie,edjl->mndijl", H.aa.ooov, T.ab)
                   +0.25 * ccpy_einsum("mnef,efdijl->mndijl", H.aa.oovv, T.aab)
    )
    I3B_oovooo -= np.transpose(I3B_oovooo, (0, 1, 2, 4, 3, 5))
    m4b += (1.0 / 12.0) * 0.5 * ccpy_einsum("mndijl,abcmnk->abcdijkl", I3B_oovooo, T.aaa)  # (k/ij) = 3

    I3A_vooooo = ccpy_einsum("mnie,delj->dmnlij", H.aa.ooov, T.aa)
    I3A_vooooo -= np.transpose(I3A_vooooo, (0, 1, 2, 4, 3, 5)) + np.transpose(I3A_vooooo, (0, 1, 2, 5, 4, 3))
    I3A_vooooo += 0.5 * ccpy_einsum("mnef,efdijl->dmnlij", H.aa.oovv, T.aaa)
    m4b += (1.0 / 12.0) * 0.5 * ccpy_einsum("cmnkij,abdmnl->abcdijkl", I3A_vooooo, T.aab)  # (c/ab) = 3

    I3B_vooooo = (
                    0.5 * ccpy_einsum("mnel,aeik->amnikl", H.ab.oovo, T.aa)
                  + ccpy_einsum("mnke,aeil->amnikl", H.ab.ooov, T.ab)
                  + 0.5 * ccpy_einsum("mnef,aefikl->amnikl", H.ab.oovv, T.aab)
    )
    I3B_vooooo -= np.transpose(I3B_vooooo, (0, 1, 2, 4, 3, 5))
    m4b += (9.0 / 36.0) * ccpy_einsum("amnikl,bcdjmn->abcdijkl", I3B_vooooo, T.aab)  # (a/bc)(j/ik) = 9

    I3B_vvvvvo = -ccpy_einsum("amef,bdml->abdefl", H.aa.vovv, T.ab)
    I3B_vvvvvo -= np.transpose(I3B_vvvvvo, (1, 0, 2, 3, 4, 5))
    m4b += (1.0 / 12.0) * 0.5 * ccpy_einsum("abdefl,efcijk->abcdijkl", I3B_vvvvvo, T.aaa)  # (c/ab) = 3

    I3A_vvvvvo = -ccpy_einsum("amef,bcmk->abcefk", H.aa.vovv, T.aa)
    I3A_vvvvvo -= np.transpose(I3A_vvvvvo, (1, 0, 2, 3, 4, 5)) + np.transpose(I3A_vvvvvo, (2, 1, 0, 3, 4, 5))
    m4b += (1.0 / 12.0) * 0.5 * ccpy_einsum("abcefk,efdijl->abcdijkl", I3A_vvvvvo, T.aab)  # (k/ij) = 3

    I3B_vvvovv = (
                    -0.5 * ccpy_einsum("mdef,acim->acdief", H.ab.ovvv, T.aa)
                    - ccpy_einsum("cmef,adim->acdief", H.ab.vovv, T.ab)
    )
    I3B_vvvovv -= np.transpose(I3B_vvvovv, (1, 0, 2, 3, 4, 5))
    m4b += (9.0 / 36.0) * ccpy_einsum("acdief,befjkl->abcdijkl", I3B_vvvovv, T.aab)  # (b/ac)(i/jk) = 9

    I3B_vovovo = (
                    -ccpy_einsum("nmie,adnl->amdiel", H.aa.ooov, T.ab)
                    +ccpy_einsum("amfe,fdil->amdiel", H.aa.vovv, T.ab)
                    -ccpy_einsum("mnel,adin->amdiel", H.ab.oovo, T.ab)
                    +ccpy_einsum("mdef,afil->amdiel", H.ab.ovvv, T.ab)
                    +ccpy_einsum("mnef,afdinl->amdiel", H.aa.oovv, T.aab)
                    +ccpy_einsum("mnef,afdinl->amdiel", H.ab.oovv, T.abb)
    )
    m4b += (9.0 / 36.0) * ccpy_einsum("amdiel,bcejkm->abcdijkl", I3B_vovovo, T.aaa)  # (a/bc)(i/jk) = 9

    I3A_vvooov = (
                -0.5 * ccpy_einsum("nmje,abin->abmije", H.aa.ooov, T.aa)
                +0.5 * ccpy_einsum("bmfe,afij->abmije", H.aa.vovv, T.aa)
                +0.25 * ccpy_einsum("mnef,abfijn->abmije", H.ab.oovv, T.aab)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    m4b += (9.0 / 36.0) * ccpy_einsum("abmije,cedkml->abcdijkl", I3A_vvooov, T.aab)  # (c/ab)(k/ij) = 9

    I3B_vvoovo = (
                -0.5 * ccpy_einsum("nmel,acin->acmiel", H.ab.oovo, T.aa)
                + ccpy_einsum("cmef,afil->acmiel", H.ab.vovv, T.ab)
                - 0.5 * ccpy_einsum("nmef,acfinl->acmiel", H.ab.oovv, T.aab)
    )
    I3B_vvoovo -= np.transpose(I3B_vvoovo, (1, 0, 2, 3, 4, 5))
    m4b -= (9.0 / 36.0) * ccpy_einsum("acmiel,ebdkjm->abcdijkl", I3B_vvoovo, T.aab)  # (b/ac)(i/jk) = 9

    I3B_vovoov = (
                0.5 * ccpy_einsum("mdfe,afik->amdike", H.ab.ovvv, T.aa)
                -ccpy_einsum("mnke,adin->amdike", H.ab.ooov, T.ab)
    )
    I3B_vovoov -= np.transpose(I3B_vovoov, (0, 1, 2, 4, 3, 5))
    m4b -= (9.0 / 36.0) * ccpy_einsum("amdike,bcejml->abcdijkl", I3B_vovoov, T.aab)  # (a/bc)(j/ik) = 9

    I3C_vvooov = (
                -ccpy_einsum("nmie,adnl->admile", H.ab.ooov, T.ab)
                -ccpy_einsum("nmle,adin->admile", H.bb.ooov, T.ab)
                +ccpy_einsum("amfe,fdil->admile", H.ab.vovv, T.ab)
                +ccpy_einsum("dmfe,afil->admile", H.bb.vovv, T.ab)
                +ccpy_einsum("mnef,afdinl->admile", H.bb.oovv, T.abb)  # added 5/2/22
    )
    m4b += (9.0 / 36.0) * ccpy_einsum("admile,bcejkm->abcdijkl", I3C_vvooov, T.aab)  # (a/bc)(i/jk) = 9

    I3B_vvooov = (
                -0.5 * ccpy_einsum("nmje,abin->abmije", H.ab.ooov, T.aa)
                +0.5 * ccpy_einsum("bmfe,afij->abmije", H.ab.vovv, T.aa)
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    m4b += (9.0 / 36.0) * ccpy_einsum("abmije,cdeklm->abcdijkl", I3B_vvooov, T.abb)  # (c/ab)(k/ij) = 9
    # antisymmetrize
    m4b -= np.transpose(m4b, (0, 1, 2, 3, 4, 6, 5, 7))  # (jk)
    m4b -= np.transpose(m4b, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(m4b, (0, 1, 2, 3, 6, 5, 4, 7)) # (i/jk)
    m4b -= np.transpose(m4b, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    m4b -= np.transpose(m4b, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(m4b, (2, 1, 0, 3, 4, 5, 6, 7)) # (a/bc)
    return m4b

def build_m4c(H, T):
    # <ijklabcd | H(2) | 0 >
    m4c = -ccpy_einsum("cmke,adim,bejl->abcdijkl", H.bb.voov, T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    m4c -= ccpy_einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    m4c -= 0.5 * ccpy_einsum("mcek,aeij,bdml->abcdijkl", H.ab.ovvo, T.aa, T.ab)    # (kl)(ab)(cd) = 8
    m4c -= 0.5 * ccpy_einsum("amie,bdjm,cekl->abcdijkl", H.ab.voov, T.ab, T.bb)    # (ij)(ab)(cd) = 8
    m4c -= 0.5 * ccpy_einsum("mcek,abim,edjl->abcdijkl", H.ab.ovvo, T.aa, T.ab)    # (ij)(kl)(cd) = 8
    m4c -= 0.5 * ccpy_einsum("amie,cdkm,bejl->abcdijkl", H.ab.voov, T.bb, T.ab)    # (ij)(kl)(ab) = 8
    m4c -= ccpy_einsum("bmel,adim,ecjk->abcdijkl", H.ab.vovo, T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    m4c -= ccpy_einsum("mdje,bcmk,aeil->abcdijkl", H.ab.ovov, T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    m4c -= 0.25 * ccpy_einsum("mdje,abim,cekl->abcdijkl", H.ab.ovov, T.aa, T.bb)   # (ij)(cd) = 4
    m4c -= 0.25 * ccpy_einsum("bmel,cdkm,aeij->abcdijkl", H.ab.vovo, T.bb, T.aa)   # (kl)(ab) = 4
    m4c += 0.25 * ccpy_einsum("mnij,acmk,bdnl->abcdijkl", H.aa.oooo, T.ab, T.ab)   # (kl)(ab) = 4 !!! (tricky asym)
    m4c += 0.25 * ccpy_einsum("abef,ecik,fdjl->abcdijkl", H.aa.vvvv, T.ab, T.ab)   # (ij)(kl) = 4 !!! (tricky asym)
    m4c += 0.25 * ccpy_einsum("mnik,abmj,cdnl->abcdijkl", H.ab.oooo, T.aa, T.bb)   # (ij)(kl) = 4
    m4c += 0.25 * ccpy_einsum("acef,ebij,fdkl->abcdijkl", H.ab.vvvv, T.aa, T.bb)   # (ab)(cd) = 4
    m4c += ccpy_einsum("mnik,adml,bcjn->abcdijkl", H.ab.oooo, T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    m4c += ccpy_einsum("acef,edil,bfjk->abcdijkl", H.ab.vvvv, T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    m4c += 0.25 * ccpy_einsum("mnkl,adin,bcjm->abcdijkl", H.bb.oooo, T.ab, T.ab)   # (ij)(cd) = 4 !!! (tricky asym)
    m4c += 0.25 * ccpy_einsum("cdef,afil,bejk->abcdijkl", H.bb.vvvv, T.ab, T.ab)   # (ij)(kl) = 4 !!! (tricky asym)

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    m4c -= (8.0 / 16.0) * ccpy_einsum("mdil,abcmjk->abcdijkl", H.ab.ovoo, T.aab)  # [1]  (ij)(kl)(cd) = 8
    m4c -= (2.0 / 16.0) * ccpy_einsum("bmji,acdmkl->abcdijkl", H.aa.vooo, T.abb)  # [2]  (ab) = 2
    m4c -= (2.0 / 16.0) * ccpy_einsum("cmkl,abdijm->abcdijkl", H.bb.vooo, T.aab)  # [3]  (cd) = 2
    m4c -= (8.0 / 16.0) * ccpy_einsum("amil,bcdjkm->abcdijkl", H.ab.vooo, T.abb)  # [4]  (ij)(ab)(kl) = 8
    m4c += (8.0 / 16.0) * ccpy_einsum("adel,becjik->abcdijkl", H.ab.vvvo, T.aab)  # [5]  (ab)(kl)(cd) = 8
    m4c += (2.0 / 16.0) * ccpy_einsum("baje,ecdikl->abcdijkl", H.aa.vvov, T.abb)  # [6]  (ij) = 2
    m4c += (8.0 / 16.0) * ccpy_einsum("adie,bcejkl->abcdijkl", H.ab.vvov, T.abb)  # [7]  (ij)(ab)(cd) = 8
    m4c += (2.0 / 16.0) * ccpy_einsum("cdke,abeijl->abcdijkl", H.bb.vvov, T.aab)  # [8]  (kl) = 2

    I3B_oovooo = (
                ccpy_einsum("mnif,fdjl->mndijl", H.aa.ooov, T.ab)
               + 0.25 * ccpy_einsum("mnef,efdijl->mndijl", H.aa.oovv, T.aab)
    )
    I3B_oovooo -= np.transpose(I3B_oovooo, (0, 1, 2, 4, 3, 5))
    m4c += (4.0 / 16.0) * 0.5 * ccpy_einsum("mndijl,abcmnk->abcdijkl", I3B_oovooo, T.aab)  # [9]  (kl)(cd) = 4

    I3B_ovoooo = (
                ccpy_einsum("mnif,bfjl->mbnijl", H.ab.ooov, T.ab)
                + 0.5 * ccpy_einsum("mnfl,bfji->mbnijl", H.ab.oovo, T.aa)
                + 0.5 * ccpy_einsum("mnef,befjil->mbnijl", H.ab.oovv, T.aab)
    )
    I3B_ovoooo -= np.transpose(I3B_ovoooo, (0, 1, 2, 4, 3, 5))
    m4c += (4.0 / 16.0) * ccpy_einsum("mbnijl,acdmkn->abcdijkl", I3B_ovoooo, T.abb)  # [10]  (kl)(ab) = 4

    I3C_vooooo = (
                ccpy_einsum("nmlf,afik->amnikl", H.bb.ooov, T.ab)
                + 0.25 * ccpy_einsum("mnef,aefikl->amnikl", H.bb.oovv, T.abb)
    )
    I3C_vooooo -= np.transpose(I3C_vooooo, (0, 1, 2, 3, 5, 4))
    m4c += (4.0 / 16.0) * 0.5 * ccpy_einsum("amnikl,bcdjmn->abcdijkl", I3C_vooooo, T.abb)  # [11]  (ij)(ab) = 4

    I3C_oovooo = (
                0.5 * ccpy_einsum("mnif,cfkl->mncilk", H.ab.ooov, T.bb)
                + ccpy_einsum("mnfl,fcik->mncilk", H.ab.oovo, T.ab)
                + 0.5 * ccpy_einsum("mnef,efcilk->mncilk", H.ab.oovv, T.abb)
    )
    I3C_oovooo -= np.transpose(I3C_oovooo, (0, 1, 2, 3, 5, 4))
    m4c += (4.0 / 16.0) * ccpy_einsum("mncilk,abdmjn->abcdijkl", I3C_oovooo, T.aab)  # [12]  (ij)(cd) = 4

    I3B_vvvvvo = -ccpy_einsum("bmfe,acmk->abcefk", H.aa.vovv, T.ab)
    I3B_vvvvvo -= np.transpose(I3B_vvvvvo, (1, 0, 2, 3, 4, 5))
    m4c += (4.0 / 16.0) * 0.5 * ccpy_einsum("abcefk,efdijl->abcdijkl", I3B_vvvvvo, T.aab)  # [13]  (kl)(cd) = 4

    I3C_vvvvov = (
                -ccpy_einsum("mdef,acmk->acdekf", H.ab.ovvv, T.ab)
                - 0.5 * ccpy_einsum("amef,cdkm->acdekf", H.ab.vovv, T.bb)
    )
    I3C_vvvvov -= np.transpose(I3C_vvvvov, (0, 2, 1, 3, 4, 5))
    m4c += (4.0 / 16.0) * ccpy_einsum("acdekf,ebfijl->abcdijkl", I3C_vvvvov, T.aab)  # [14]  (kl)(ab) = 4

    I3B_vvvvov = (
                -0.5 * ccpy_einsum("mdef,abmj->abdejf", H.ab.ovvv, T.aa)
                -ccpy_einsum("amef,bdjm->abdejf", H.ab.vovv, T.ab)
    )
    I3B_vvvvov -= np.transpose(I3B_vvvvov, (1, 0, 2, 3, 4, 5))
    m4c += (4.0 / 16.0) * ccpy_einsum("abdejf,efcilk->abcdijkl", I3B_vvvvov, T.abb)  # [15]  (ij)(cd) = 4

    I3C_vvvovv = -ccpy_einsum("cmef,adim->acdief", H.bb.vovv, T.ab)
    I3C_vvvovv -= np.transpose(I3C_vvvovv, (0, 2, 1, 3, 4, 5))
    m4c += (4.0 / 16.0) * 0.5 * ccpy_einsum("acdief,befjkl->abcdijkl", I3C_vvvovv, T.abb)  # [16]  (ij)(ab) = 4

    I3A_vvooov = (
                -0.5 * ccpy_einsum("nmje,abin->abmije", H.aa.ooov, T.aa)
                +0.5 * ccpy_einsum("bmfe,afij->abmije", H.aa.vovv, T.aa)
                +0.25 * ccpy_einsum("mnef,abfijn->abmije", H.aa.oovv, T.aaa)
                +0.25 * ccpy_einsum("mnef,abfijn->abmije", H.ab.oovv, T.aab)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    m4c += (1.0 / 16.0) * ccpy_einsum("abmije,ecdmkl->abcdijkl", I3A_vvooov, T.abb)  # [17]  (1) = 1

    I3B_vvooov = (
                -0.5 * ccpy_einsum("nmje,abin->abmije", H.ab.ooov, T.aa)
                +0.5 * ccpy_einsum("bmfe,afij->abmije", H.ab.vovv, T.aa)
                +0.25 * ccpy_einsum("nmfe,abfijn->abmije", H.ab.oovv, T.aaa)
                +0.25 * ccpy_einsum("nmfe,abfijn->abmije", H.bb.oovv, T.aab)
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    m4c += (1.0 / 16.0) * ccpy_einsum("abmije,ecdmkl->abcdijkl", I3B_vvooov, T.bbb)  # [18]  (1) = 1

    I3C_ovvvoo = (
                -0.5 * ccpy_einsum("mnek,cdnl->mcdekl", H.ab.oovo, T.bb)
                +0.5 * ccpy_einsum("mcef,fdkl->mcdekl", H.ab.ovvv, T.bb)
    )
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 2, 1, 3, 4, 5))
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 1, 2, 3, 5, 4))
    m4c += (1.0 / 16.0) * ccpy_einsum("mcdekl,abeijm->abcdijkl", I3C_ovvvoo, T.aaa)  # [19]  (1) = 1

    I3D_vvooov = (
                -0.5 * ccpy_einsum("nmke,cdnl->cdmkle", H.bb.ooov, T.bb)
                +0.5 * ccpy_einsum("cmfe,fdkl->cdmkle", H.bb.vovv, T.bb)
    )
    I3D_vvooov -= np.transpose(I3D_vvooov, (1, 0, 2, 3, 4, 5))
    I3D_vvooov -= np.transpose(I3D_vvooov, (0, 1, 2, 4, 3, 5))
    m4c += (1.0 / 16.0) * ccpy_einsum("cdmkle,abeijm->abcdijkl", I3D_vvooov, T.aab)  # [20]  (1) = 1

    I3B_vovovo = (
                -ccpy_einsum("mnel,adin->amdiel", H.ab.oovo, T.ab)
                +ccpy_einsum("mdef,afil->amdiel", H.ab.ovvv, T.ab)
                +0.5 * ccpy_einsum("mnef,afdinl->amdiel", H.aa.oovv, T.aab) # !!! factor 1/2 to compensate asym
                +ccpy_einsum("mnef,afdinl->amdiel", H.ab.oovv, T.abb)
                -ccpy_einsum("nmie,adnl->amdiel", H.aa.ooov, T.ab)
                +ccpy_einsum("amfe,fdil->amdiel", H.aa.vovv, T.ab)
    )
    m4c += ccpy_einsum("amdiel,becjmk->abcdijkl", I3B_vovovo, T.aab)  # [21]  (ij)(kl)(ab)(cd) = 16

    I3C_vovovo = (
                -ccpy_einsum("nmie,adnl->amdiel", H.ab.ooov, T.ab)
                +ccpy_einsum("amfe,fdil->amdiel", H.ab.vovv, T.ab)
                -ccpy_einsum("nmle,adin->amdiel", H.bb.ooov, T.ab)
                +ccpy_einsum("dmfe,afil->amdiel", H.bb.vovv, T.ab)
                +0.5 * ccpy_einsum("mnef,afdinl->amdiel", H.bb.oovv, T.abb) # !!! factor 1/2 to compensate asym
    )
    m4c += ccpy_einsum("amdiel,becjmk->abcdijkl", I3C_vovovo, T.abb)  # [22]  (ij)(kl)(ab)(cd) = 16

    I3B_vovoov = (
                -ccpy_einsum("mnie,bdjn->bmdjie", H.ab.ooov, T.ab)
                +0.5 * ccpy_einsum("mdfe,bfji->bmdjie", H.ab.ovvv, T.aa)
                -0.5 * ccpy_einsum("mnfe,bfdjin->bmdjie", H.ab.oovv, T.aab)
    )
    I3B_vovoov -= np.transpose(I3B_vovoov, (0, 1, 2, 4, 3, 5))
    m4c -= (4.0 / 16.0) * ccpy_einsum("bmdjie,aecmlk->abcdijkl", I3B_vovoov, T.abb)  # [23]  (ab)(cd) = 4

    I3C_ovvoov = (
                -0.5 * ccpy_einsum("mnie,cdkn->mcdike", H.ab.ooov, T.bb)
                +ccpy_einsum("mdfe,fcik->mcdike", H.ab.ovvv, T.ab)
                -0.5 * ccpy_einsum("mnfe,fcdikn->mcdike", H.ab.oovv, T.abb)
    )
    I3C_ovvoov -= np.transpose(I3C_ovvoov, (0, 2, 1, 3, 4, 5))
    m4c -= (4.0 / 16.0) * ccpy_einsum("mcdike,abemjl->abcdijkl", I3C_ovvoov, T.aab)  # [24]  (ij)(kl) = 4

    I3B_vvovoo = (
                -0.5 * ccpy_einsum("nmel,abnj->abmejl", H.ab.oovo, T.aa)
                +ccpy_einsum("amef,bfjl->abmejl", H.ab.vovv, T.ab)
    )
    I3B_vvovoo -= np.transpose(I3B_vvovoo, (1, 0, 2, 3, 4, 5))
    m4c -= (4.0 / 16.0) * ccpy_einsum("abmejl,ecdikm->abcdijkl", I3B_vvovoo, T.abb)  # [25]  (ij)(kl) = 4

    I3C_vovvoo = (
                -ccpy_einsum("nmel,acnk->amcelk", H.ab.oovo, T.ab)
                +0.5 * ccpy_einsum("amef,fclk->amcelk", H.ab.vovv, T.bb)
    )
    I3C_vovvoo -= np.transpose(I3C_vovvoo, (0, 1, 2, 3, 5, 4))
    m4c -= (4.0 / 16.0) * ccpy_einsum("amcelk,bedjim->abcdijkl", I3C_vovvoo, T.aab)  # [26]  (ab)(cd) = 4
    # antisymmetrize
    m4c -= np.transpose(m4c, (1, 0, 2, 3, 4, 5, 6, 7)) # (ab)
    m4c -= np.transpose(m4c, (0, 1, 3, 2, 4, 5, 6, 7)) # (cd)
    m4c -= np.transpose(m4c, (0, 1, 2, 3, 5, 4, 6, 7)) # (ij)
    m4c -= np.transpose(m4c, (0, 1, 2, 3, 4, 5, 7, 6)) # (kl)
    return m4c

def build_l4a(H, L):
    # < 0 | (L2 + L3) H(3) | ijklabcd >
    l4a = (24.0 / 576.0) * ccpy_einsum("dkec,abeijl->abcdijkl", H.aa.vovv, L.aaa) # (cd/ab)(k/ijl) = 6 * 4 = 24
    l4a -= (24.0 / 576.0) * ccpy_einsum("lkmc,abdijm->abcdijkl", H.aa.ooov, L.aaa) # (c/abd)(kl/ij) = 6 * 4 = 24
    # Disconnected terms
    l4a += (36.0 / 576.0) * ccpy_einsum("klcd,abij->abcdijkl", H.aa.oovv, L.aa) # (ij/kl)(ab/cd) = 6 * 6 = 36
    l4a += (16.0 / 576.0) * ccpy_einsum("ia,bcdjkl->abcdijkl", H.a.ov, L.aaa) # (i/jkl)(a/bcd) = 4 * 4 = 16
    # antisymmetrize
    l4a -= np.transpose(l4a, (0, 1, 2, 3, 4, 6, 5, 7)) # (jk)
    l4a -= np.transpose(l4a, (0, 1, 2, 3, 4, 7, 6, 5)) + np.transpose(l4a, (0, 1, 2, 3, 4, 5, 7, 6)) # (l/jk)
    l4a -= np.transpose(l4a, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(l4a, (0, 1, 2, 3, 6, 5, 4, 7)) + np.transpose(l4a, (0, 1, 2, 3, 7, 5, 6, 4)) # (i/jkl)
    l4a -= np.transpose(l4a, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    l4a -= np.transpose(l4a, (0, 3, 2, 1, 4, 5, 6, 7)) + np.transpose(l4a, (0, 1, 3, 2, 4, 5, 6, 7)) # (d/bc)
    l4a -= np.transpose(l4a, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(l4a, (2, 1, 0, 3, 4, 5, 6, 7)) + np.transpose(l4a, (3, 1, 2, 0, 4, 5, 6, 7)) # (a/bcd)
    return l4a

def build_l4b(H, L):
    # < 0 | (L2 + L3) H(3) | ijkl~abcd~ >
    l4b = -(1.0 / 12.0) * ccpy_einsum("klmd,abcijm->abcdijkl", H.ab.ooov, L.aaa)  # (k/ij) = 3
    l4b -= (9.0 / 36.0) * ccpy_einsum("kima,bcdjml->abcdijkl", H.aa.ooov, L.aab)  # (j/ik)(a/bc) = 9
    l4b -= (9.0 / 36.0) * ccpy_einsum("ilam,bcdjkm->abcdijkl", H.ab.oovo, L.aab)  # (a/bc)(i/jk) = 9
    l4b += (1.0 / 12.0) * ccpy_einsum("elcd,abeijk->abcdijkl", H.ab.vovv, L.aaa)  # (c/ab) = 3
    l4b += (9.0 / 36.0) * ccpy_einsum("eica,bedjkl->abcdijkl", H.aa.vovv, L.aab)  # (b/ac)(i/jk) = 9
    l4b += (9.0 / 36.0) * ccpy_einsum("iead,bcejkl->abcdijkl", H.ab.ovvv, L.aab)  # (a/bc)(i/jk) = 9
    # Disconnected terms
    l4b += (9.0 / 36.0) * ccpy_einsum("ia,bcdjkl->abcdijkl", H.a.ov, L.aab) # (i/jk)(a/bc) = 9
    l4b += (1.0 / 36.0) * ccpy_einsum("ld,abcijk->abcdijkl", H.b.ov, L.aaa)
    l4b += (9.0 / 36.0) * ccpy_einsum("ijab,cdkl->abcdijkl", H.aa.oovv, L.ab) # (c/ab)(k/ij) = 9
    l4b += (9.0 / 36.0) * ccpy_einsum("klcd,abij->abcdijkl", H.ab.oovv, L.aa) # (c/ab)(k/ij) = 9
    # antisymmetrize
    l4b -= np.transpose(l4b, (0, 1, 2, 3, 4, 6, 5, 7))  # (jk)
    l4b -= np.transpose(l4b, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(l4b, (0, 1, 2, 3, 6, 5, 4, 7)) # (i/jk)
    l4b -= np.transpose(l4b, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    l4b -= np.transpose(l4b, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(l4b, (2, 1, 0, 3, 4, 5, 6, 7)) # (a/bc)
    return l4b

def build_l4c(H, L):
    # < 0 | (L2 + L3) H(3) | ijk~l~abc~d~ >
    l4c = -(8.0 / 16.0) * ccpy_einsum("ilmd,abcmjk->abcdijkl", H.ab.ooov, L.aab)  # [1]  (ij)(kl)(cd) = 8
    l4c -= (2.0 / 16.0) * ccpy_einsum("ijmb,acdmkl->abcdijkl", H.aa.ooov, L.abb)  # [2]  (ab) = 2
    l4c -= (2.0 / 16.0) * ccpy_einsum("lkmc,abdijm->abcdijkl", H.bb.ooov, L.aab)  # [3]  (cd) = 2
    l4c -= (8.0 / 16.0) * ccpy_einsum("ilam,bcdjkm->abcdijkl", H.ab.oovo, L.abb)  # [4]  (ij)(ab)(kl) = 8
    l4c += (8.0 / 16.0) * ccpy_einsum("elad,becjik->abcdijkl", H.ab.vovv, L.aab)  # [5]  (ab)(kl)(cd) = 8
    l4c += (2.0 / 16.0) * ccpy_einsum("ejab,ecdikl->abcdijkl", H.aa.vovv, L.abb)  # [6]  (ij) = 2
    l4c += (8.0 / 16.0) * ccpy_einsum("iead,bcejkl->abcdijkl", H.ab.ovvv, L.abb)  # [7]  (ij)(ab)(cd) = 8
    l4c += (2.0 / 16.0) * ccpy_einsum("ekdc,abeijl->abcdijkl", H.bb.vovv, L.aab)  # [8]  (kl) = 2
    # Disconnected terms
    l4c += (4.0 / 16.0) * ccpy_einsum("ia,bcdjkl->abcdijkl", H.a.ov, L.abb)
    l4c += (4.0 / 16.0) * ccpy_einsum("kc,abdijl->abcdijkl", H.b.ov, L.aab)
    l4c += (1.0 / 16.0) * ccpy_einsum("ijab,cdkl->abcdijkl", H.aa.oovv, L.bb)
    l4c += ccpy_einsum("ikac,bdjl->abcdijkl", H.ab.oovv, L.ab)
    l4c += (1.0 / 16.0) * ccpy_einsum("klcd,abij->abcdijkl", H.bb.oovv, L.aa) 
    # antisymmetrize
    l4c -= np.transpose(l4c, (1, 0, 2, 3, 4, 5, 6, 7)) # (ab)
    l4c -= np.transpose(l4c, (0, 1, 3, 2, 4, 5, 6, 7)) # (cd)
    l4c -= np.transpose(l4c, (0, 1, 2, 3, 5, 4, 6, 7)) # (ij)
    l4c -= np.transpose(l4c, (0, 1, 2, 3, 4, 5, 7, 6)) # (kl)
    return l4c
