import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.cholesky.cholesky_builders import build_2index_batch_vvvv_aa, build_2index_batch_vvvv_bb, build_3index_batch_vvvv_ab

def build_hbar_ccsd_chol(T, H0, RHF_symmetry, *args):
    """Calculate the CCSD similarity-transformed Hamiltonian (H_N e^(T1+T2))_C.
    Copied as-is from original CCpy implementation."""
    from copy import deepcopy
    #from ccpy.models.integrals import Integral

    # Copy the Bare Hamiltonian object for T1/T2-similarity transformed HBar
    H = deepcopy(H0)
    #H = Integral.from_empty(system, 2, use_none=True)

    # Orbital dimensions
    nua, nub, noa, nob = T.ab.shape

    # Make useful intermediates
    tau_aa = 0.5 * T.aa + ccpy_einsum("ai,bj->abij", T.a, T.a)
    tau_aa -= np.transpose(tau_aa, (0, 1, 3, 2))
    if RHF_symmetry:
        tau_bb = tau_aa
    else:
        tau_bb = 0.5 * T.bb + ccpy_einsum("ai,bj->abij", T.b, T.b)
        tau_bb -= np.transpose(tau_bb, (0, 1, 3, 2))
    tau_ab = T.ab + ccpy_einsum("ai,bj->abij", T.a, T.b)

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

    #x1 = time.time()
    # H.aa.vvvv = (
    #        0.5 * H0.aa.vvvv
    #        + 0.25 * ccpy_einsum("mnef,abmn->abef", H0.aa.oovv, tau_aa)
    #        - ccpy_einsum("amef,bm->abef", H0.aa.vovv, T.a)
    # )
    # H.aa.vvvv -= np.transpose(H.aa.vvvv, (1, 0, 2, 3))
    # if RHF_symmetry:
    #    H.bb.vvvv = H.aa.vvvv
    # else:
    #    H.bb.vvvv = (
    #            0.5 * H0.bb.vvvv
    #            + 0.25 * ccpy_einsum("mnef,abmn->abef", H0.bb.oovv, tau_bb)
    #            - ccpy_einsum("amef,bm->abef", H0.bb.vovv, T.b)
    #    )
    #    H.bb.vvvv -= np.transpose(H.bb.vvvv, (1, 0, 2, 3))
    # H.ab.vvvv = (
    #        H0.ab.vvvv
    #        - ccpy_einsum("mbef,am->abef", H0.ab.ovvv, T.a)
    #        - ccpy_einsum("amef,bm->abef", H0.ab.vovv, T.b)
    #        + ccpy_einsum("mnef,abmn->abef", H0.ab.oovv, tau_ab)
    # )
    #x2 = time.time()
    #print("vvvv:", x2 - x1)

    #x1 = time.perf_counter()
    H.aa.oooo = (
            0.5 * H0.aa.oooo
            + ccpy_einsum("nmje,ei->mnij", H0.aa.ooov, T.a)
            + 0.25 * ccpy_einsum("mnef,efij->mnij", H0.aa.oovv, tau_aa)
    )
    H.aa.oooo -= np.transpose(H.aa.oooo, (0, 1, 3, 2))
    if RHF_symmetry:
        H.bb.oooo = H.aa.oooo
    else:
        H.bb.oooo = (
                0.5 * H0.bb.oooo
                + ccpy_einsum("nmje,ei->mnij", H0.bb.ooov, T.b)
                + 0.25 * ccpy_einsum("mnef,efij->mnij", H0.bb.oovv, tau_bb)
        )
        H.bb.oooo -= np.transpose(H.bb.oooo, (0, 1, 3, 2))
    H.ab.oooo = (
            H0.ab.oooo
            + ccpy_einsum("mnej,ei->mnij", H0.ab.oovo, T.a)
            + ccpy_einsum("mnie,ej->mnij", H0.ab.ooov, T.b)
            + ccpy_einsum("mnef,efij->mnij", H0.ab.oovv, tau_ab)
    )
    #x2 = time.perf_counter()
    #print("oooo:", x2 - x1)

    #x1 = time.perf_counter()
    H.aa.voov = (
            H0.aa.voov
            + ccpy_einsum("amfe,fi->amie", H0.aa.vovv, T.a)
            - ccpy_einsum("nmie,an->amie", H.aa.ooov, T.a)
            + ccpy_einsum("nmfe,afin->amie", H0.aa.oovv, T.aa)
            + ccpy_einsum("mnef,afin->amie", H0.ab.oovv, T.ab)
    )
    if RHF_symmetry:
        H.bb.voov = H.aa.voov
    else:
        H.bb.voov = (
                H0.bb.voov
                + ccpy_einsum("amfe,fi->amie", H0.bb.vovv, T.b)
                - ccpy_einsum("nmie,an->amie", H.bb.ooov, T.b)
                + ccpy_einsum("nmfe,afin->amie", H0.bb.oovv, T.bb)
                + ccpy_einsum("nmfe,fani->amie", H0.ab.oovv, T.ab)
        )
    H.ab.voov = (
            H0.ab.voov
            + ccpy_einsum("amfe,fi->amie", H0.ab.vovv, T.a)
            - ccpy_einsum("nmie,an->amie", H.ab.ooov, T.a)
            + ccpy_einsum("nmfe,afin->amie", H0.ab.oovv, T.aa)
            + ccpy_einsum("nmfe,afin->amie", H0.bb.oovv, T.ab)
    )
    H.ab.ovvo = (
            H0.ab.ovvo
            + ccpy_einsum("maef,fi->maei", H0.ab.ovvv, T.b)
            - ccpy_einsum("mnei,an->maei", H.ab.oovo, T.b)
            + ccpy_einsum("mnef,afin->maei", H0.ab.oovv, T.bb)
            + ccpy_einsum("mnef,fani->maei", H0.aa.oovv, T.ab)
    )
    H.ab.ovov = (
            H0.ab.ovov
            + ccpy_einsum("mafe,fi->maie", H0.ab.ovvv, T.a)
            - ccpy_einsum("mnie,an->maie", H.ab.ooov, T.b)
            - ccpy_einsum("mnfe,fain->maie", H0.ab.oovv, T.ab)
    )
    H.ab.vovo = (
            H0.ab.vovo
            - ccpy_einsum("nmei,an->amei", H.ab.oovo, T.a)
            + ccpy_einsum("amef,fi->amei", H0.ab.vovv, T.b)
            - ccpy_einsum("nmef,afni->amei", H0.ab.oovv, T.ab)
    )
    #x2 = time.perf_counter()
    #print("voov:", x2 - x1)

    #x1 = time.perf_counter()
    Q1 = (
            ccpy_einsum("mnjf,afin->amij", H.aa.ooov, T.aa)
            + ccpy_einsum("mnjf,afin->amij", H.ab.ooov, T.ab)
    )
    Q2 = H0.aa.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.aa.vovv, T.a)
    Q2 = ccpy_einsum("amif,fj->amij", Q2, T.a)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.vooo = H0.aa.vooo + Q1 + (
            ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
            - ccpy_einsum("nmij,an->amij", H.aa.oooo, T.a)
            + 0.5 * ccpy_einsum("amef,efij->amij", H0.aa.vovv, T.aa)
    )
    if RHF_symmetry:
        H.bb.vooo = H.aa.vooo
    else:
        Q1 = (
                ccpy_einsum("mnjf,afin->amij", H.bb.ooov, T.bb)
                + ccpy_einsum("nmfj,fani->amij", H.ab.oovo, T.ab)
        )
        Q2 = H0.bb.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.bb.vovv, T.b)
        Q2 = ccpy_einsum("amif,fj->amij", Q2, T.b)
        Q1 += Q2
        Q1 -= np.transpose(Q1, (0, 1, 3, 2))
        H.bb.vooo = H0.bb.vooo + Q1 + (
                + ccpy_einsum("me,aeij->amij", H.b.ov, T.bb)
                - ccpy_einsum("nmij,an->amij", H.bb.oooo, T.b)
                + 0.5 * ccpy_einsum("amef,efij->amij", H0.bb.vovv, T.bb)
        )
    Q1 = H0.ab.voov + ccpy_einsum("amfe,fi->amie", H0.ab.vovv, T.a)
    H.ab.vooo = H0.ab.vooo + (
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
    H.ab.ovoo = H0.ab.ovoo + (
            ccpy_einsum("me,eaji->maji", H.a.ov, T.ab)
            - ccpy_einsum("mnji,an->maji", H.ab.oooo, T.b)
            + ccpy_einsum("mnjf,fani->maji", H.aa.ooov, T.ab)
            + ccpy_einsum("mnjf,fani->maji", H.ab.ooov, T.bb)
            - ccpy_einsum("mnfi,fajn->maji", H.ab.oovo, T.ab)
            + ccpy_einsum("maje,ei->maji", Q1, T.b)
            + ccpy_einsum("maei,ej->maji", H0.ab.ovvo, T.a)
            + ccpy_einsum("mafe,feji->maji", H0.ab.ovvv, T.ab)
    )
    #x2 = time.perf_counter()
    #print("vooo:", x2 - x1)

    #x1 = time.perf_counter()
    x2a_voov = H.aa.voov + 0.5 * ccpy_einsum("nmie,an->amie", H.aa.ooov, T.a) # defined to avoid double-counting from A(ab) on [8] and [15]
    H.aa.vvov = (
            0.5 * H0.aa.vvov # [1]
            - 0.5 * ccpy_einsum("me,abim->abie", H.a.ov, T.aa) # [4]+[12]+[13]
            - ccpy_einsum("amie,bm->abie", x2a_voov, T.a) # [3]+[8']+[9]+[11]+[13]+[15']
            + 0.25 * ccpy_einsum("mnie,abmn->abie", H.aa.ooov, T.aa) # [6]+[10]
            # Terms we have to deal with directly that are nu^4: [2], [5], and [7]
            #+ 0.5 * ccpy_einsum("abfe,fi->abie", H0.aa.vvvv, T.a) # [2]
            + ccpy_einsum("bnef,afin->abie", H0.aa.vovv, T.aa) # [5]
            + ccpy_einsum("bnef,afin->abie", H0.ab.vovv, T.ab) # [7]
    )
    # Add on h0(vvvv) term using Cholesky
    for a in range(nua):
        for b in range(a + 1, nua):
            batch_ints = build_2index_batch_vvvv_aa(a, b, H0)
            H.aa.vvov[a, b, :, :] += ccpy_einsum("fe,fi->ie", batch_ints, T.a)
    H.aa.vvov -= np.transpose(H.aa.vvov, (1, 0, 2, 3))
    if RHF_symmetry:
        H.bb.vvov = H.aa.vvov
    else:
        x2c_voov = H.bb.voov + 0.5 * ccpy_einsum("nmie,an->amie", H.bb.ooov, T.b)  # defined to avoid double-counting from A(ab) on [8] and [15]
        H.bb.vvov = (
                0.5 * H0.bb.vvov  # [1]
                - 0.5 * ccpy_einsum("me,abim->abie", H.b.ov, T.bb)  # [4]+[12]+[13]
                - ccpy_einsum("amie,bm->abie", x2c_voov, T.b)  # [3]+[8']+[9]+[11]+[13]+[15']
                + 0.25 * ccpy_einsum("mnie,abmn->abie", H.bb.ooov, T.bb)  # [6]+[10]
                # Terms we have to deal with directly that are nu^4: [2], [5], and [7]
                #+ 0.5 * ccpy_einsum("abfe,fi->abie", H0.bb.vvvv, T.b)  # [2]
                + ccpy_einsum("bnef,afin->abie", H0.bb.vovv, T.bb)  # [5]
                + ccpy_einsum("nbfe,fani->abie", H0.ab.ovvv, T.ab)  # [7]
        )
        # Add on h0(vvvv) term using Cholesky
        for a in range(nub):
            for b in range(a + 1, nub):
                batch_ints = build_2index_batch_vvvv_bb(a, b, H0)
                H.bb.vvov[a, b, :, :] += ccpy_einsum("fe,fi->ie", batch_ints, T.b)
        H.bb.vvov -= np.transpose(H.bb.vvov, (1, 0, 2, 3))

    # need to define x2b_voov and x2b_ovov such that [10] and [18] are not double counted
    x2b_voov = H.ab.voov + 0.5 * ccpy_einsum("nmie,an->amie", H.ab.ooov, T.a) # nu2no3
    x2b_ovov = H.ab.ovov + 0.5 * ccpy_einsum("nmie,bm->nbie", H.ab.ooov, T.b) # nu2no3
    H.ab.vvov = (
            H0.ab.vvov # [1]
            - ccpy_einsum("me,abim->abie", H.b.ov, T.ab) # [8] + [13] + [16]
            - ccpy_einsum("amie,bm->abie", x2b_voov, T.b) # [4] + 1/2*[10] + [11] + [12] + [17] + 1/2*[18]
            - ccpy_einsum("mbie,am->abie", x2b_ovov, T.a) # [2] + [9] + 1/2*[10] + [15] + 1/2*[18]
            + ccpy_einsum("nmie,abnm->abie", H.ab.ooov, T.ab) # [7] + [14]
            # Terms we have to deal with directly that are nu^4: [2], [5], [6], and [19]
            #+ ccpy_einsum("abfe,fi->abie", H0.ab.vvvv, T.a) # [2]
            + ccpy_einsum("mbfe,afim->abie", H0.ab.ovvv, T.aa) # [5]
            + ccpy_einsum("bmef,afim->abie", H0.bb.vovv, T.ab) # [6]
            - ccpy_einsum("amfe,fbim->abie", H0.ab.vovv, T.ab) # [19]
    )
    x2b_ovvo = H.ab.ovvo + 0.5 * ccpy_einsum("nmei,am->naei", H.ab.oovo, T.b)
    x2b_vovo = H.ab.vovo + 0.5 * ccpy_einsum("nmei,bn->bmei", H.ab.oovo, T.a)
    H.ab.vvvo = (
            H0.ab.vvvo # [1]
            - ccpy_einsum("me,bami->baei", H.a.ov, T.ab) # [8] + [13] + [16]
            - ccpy_einsum("maei,bm->baei", x2b_ovvo, T.a) # [4] + 1/2*[10] + [11] + [12] + [14] + 1/2*[17]
            - ccpy_einsum("bmei,am->baei", x2b_vovo, T.b) # [3] + [9] + 1/2*[10] + 1/2*[17] + [18]
            + ccpy_einsum("nmei,banm->baei", H.ab.oovo, T.ab) # [6] + [15]
            # Terms we have to deal with directly that are nu^4: [2], [5], [7], and [19]
            #+ ccpy_einsum("baef,fi->baei", H0.ab.vvvv, T.b) # [2]
            + ccpy_einsum("bmef,afim->baei", H0.ab.vovv, T.bb) # [5]
            + ccpy_einsum("bmef,fami->baei", H0.aa.vovv, T.ab) # [7]
            - ccpy_einsum("maef,bfmi->baei", H0.ab.ovvv, T.ab) # [19]
    )
    #x2 = time.perf_counter()
    #print("vvov:", x2 - x1)
    # add on h0(vvvv) term using Cholesky
    for a in range(nua):
        batch_ints = build_3index_batch_vvvv_ab(a, H0)
        H.ab.vvov[a, :, :, :] += ccpy_einsum("bfe,fi->bie", batch_ints, T.a)
        H.ab.vvvo[a, :, :, :] += ccpy_einsum("bef,fi->bei", batch_ints, T.b)
    return H

