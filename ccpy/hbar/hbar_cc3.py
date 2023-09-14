import numpy as np
from ccpy.utilities.updates import cc3_loops

def build_hbar_cc3(T, H0, system, *args):
    """Calculate the one- and two-body components of the CC3 similarity-transformed
     Hamiltonian (H_N e^(T1+T2+T3))_C, where T3 = <ijkabc|(V_N*T2)_C|0>/-D_MP, where
     D_MP = e_a+e_b+e_c-e_i-e_j-e_k."""
    from copy import deepcopy
    from ccpy.models.integrals import Integral

    # Copy the Bare Hamiltonian object for T1/T2-similarity transformed HBar
    H = deepcopy(H0)

    H.a.ov += (
            np.einsum("imae,em->ia", H0.aa.oovv, T.a, optimize=True)
            + np.einsum("imae,em->ia", H0.ab.oovv, T.b, optimize=True)
    )

    H.a.oo += (
            np.einsum("je,ei->ji", H.a.ov, T.a, optimize=True)
            + np.einsum("jmie,em->ji", H0.aa.ooov, T.a, optimize=True)
            + np.einsum("jmie,em->ji", H0.ab.ooov, T.b, optimize=True)
            + 0.5 * np.einsum("jnef,efin->ji", H0.aa.oovv, T.aa, optimize=True)
            + np.einsum("jnef,efin->ji", H0.ab.oovv, T.ab, optimize=True)
    )

    H.a.vv += (
            - np.einsum("mb,am->ab", H.a.ov, T.a, optimize=True)
            + np.einsum("ambe,em->ab", H0.aa.vovv, T.a, optimize=True)
            + np.einsum("ambe,em->ab", H0.ab.vovv, T.b, optimize=True)
            - 0.5 * np.einsum("mnbf,afmn->ab", H0.aa.oovv, T.aa, optimize=True)
            - np.einsum("mnbf,afmn->ab", H0.ab.oovv, T.ab, optimize=True)
    )

    H.b.ov += (
            np.einsum("imae,em->ia", H0.bb.oovv, T.b, optimize=True)
            + np.einsum("miea,em->ia", H0.ab.oovv, T.a, optimize=True)
    )

    H.b.oo += (
            np.einsum("je,ei->ji", H.b.ov, T.b, optimize=True)
            + np.einsum("jmie,em->ji", H0.bb.ooov, T.b, optimize=True)
            + np.einsum("mjei,em->ji", H0.ab.oovo, T.a, optimize=True)
            + 0.5 * np.einsum("jnef,efin->ji", H0.bb.oovv, T.bb, optimize=True)
            + np.einsum("njfe,feni->ji", H0.ab.oovv, T.ab, optimize=True)
    )

    H.b.vv += (
            - np.einsum("mb,am->ab", H.b.ov, T.b, optimize=True)
            + np.einsum("ambe,em->ab", H0.bb.vovv, T.b, optimize=True)
            + np.einsum("maeb,em->ab", H0.ab.ovvv, T.a, optimize=True)
            - 0.5 * np.einsum("mnbf,afmn->ab", H0.bb.oovv, T.bb, optimize=True)
            - np.einsum("nmfb,fanm->ab", H0.ab.oovv, T.ab, optimize=True)
    )

    Q1 = -np.einsum("mnfe,an->amef", H0.aa.oovv, T.a, optimize=True)
    I2A_vovv = H0.aa.vovv + 0.5 * Q1
    H.aa.vovv = I2A_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H0.aa.oovv, T.a, optimize=True)
    I2A_ooov = H0.aa.ooov + 0.5 * Q1
    H.aa.ooov = I2A_ooov + 0.5 * Q1

    Q1 = -np.einsum("nmef,an->amef", H0.ab.oovv, T.a, optimize=True)
    I2B_vovv = H0.ab.vovv + 0.5 * Q1
    H.ab.vovv = I2B_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H0.ab.oovv, T.a, optimize=True)
    I2B_ooov = H0.ab.ooov + 0.5 * Q1
    H.ab.ooov = I2B_ooov + 0.5 * Q1

    Q1 = -np.einsum("mnef,an->maef", H0.ab.oovv, T.b, optimize=True)
    I2B_ovvv = H0.ab.ovvv + 0.5 * Q1
    H.ab.ovvv = I2B_ovvv + 0.5 * Q1

    Q1 = np.einsum("nmef,fi->nmei", H0.ab.oovv, T.b, optimize=True)
    I2B_oovo = H0.ab.oovo + 0.5 * Q1
    H.ab.oovo = I2B_oovo + 0.5 * Q1

    Q1 = -np.einsum("nmef,an->amef", H0.bb.oovv, T.b, optimize=True)
    I2C_vovv = H0.bb.vovv + 0.5 * Q1
    H.bb.vovv = I2C_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H0.bb.oovv, T.b, optimize=True)
    I2C_ooov = H0.bb.ooov + 0.5 * Q1
    H.bb.ooov = I2C_ooov + 0.5 * Q1

    Q1 = -np.einsum("bmfe,am->abef", I2A_vovv, T.a, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvvv += 0.5 * np.einsum("mnef,abmn->abef", H0.aa.oovv, T.aa, optimize=True) + Q1

    H.ab.vvvv += (
            - np.einsum("mbef,am->abef", I2B_ovvv, T.a, optimize=True)
            - np.einsum("amef,bm->abef", I2B_vovv, T.b, optimize=True)
            + np.einsum("mnef,abmn->abef", H0.ab.oovv, T.ab, optimize=True)
    )

    Q1 = -np.einsum("bmfe,am->abef", I2C_vovv, T.b, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvvv += 0.5 * np.einsum("mnef,abmn->abef", H0.bb.oovv, T.bb, optimize=True) + Q1

    Q1 = +np.einsum("nmje,ei->mnij", I2A_ooov, T.a, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.oooo += 0.5 * np.einsum("mnef,efij->mnij", H0.aa.oovv, T.aa, optimize=True) + Q1

    H.ab.oooo += (
            np.einsum("mnej,ei->mnij", I2B_oovo, T.a, optimize=True)
            + np.einsum("mnie,ej->mnij", I2B_ooov, T.b, optimize=True)
            + np.einsum("mnef,efij->mnij", H0.ab.oovv, T.ab, optimize=True)
    )

    Q1 = +np.einsum("nmje,ei->mnij", I2C_ooov, T.b, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.oooo += 0.5 * np.einsum("mnef,efij->mnij", H0.bb.oovv, T.bb, optimize=True) + Q1

    H.aa.voov += (
            np.einsum("amfe,fi->amie", I2A_vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", I2A_ooov, T.a, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.aa.oovv, T.aa, optimize=True)
            + np.einsum("mnef,afin->amie", H0.ab.oovv, T.ab, optimize=True)
    )

    H.ab.voov += (
            np.einsum("amfe,fi->amie", I2B_vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", I2B_ooov, T.a, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.ab.oovv, T.aa, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.bb.oovv, T.ab, optimize=True)
    )

    H.ab.ovvo += (
            np.einsum("maef,fi->maei", I2B_ovvv, T.b, optimize=True)
            - np.einsum("mnei,an->maei", I2B_oovo, T.b, optimize=True)
            + np.einsum("mnef,afin->maei", H0.ab.oovv, T.bb, optimize=True)
            + np.einsum("mnef,fani->maei", H0.aa.oovv, T.ab, optimize=True)
    )

    H.ab.ovov += (
            np.einsum("mafe,fi->maie", I2B_ovvv, T.a, optimize=True)
            - np.einsum("mnie,an->maie", I2B_ooov, T.b, optimize=True)
            - np.einsum("mnfe,fain->maie", H0.ab.oovv, T.ab, optimize=True)
    )

    H.ab.vovo += (
            - np.einsum("nmei,an->amei", I2B_oovo, T.a, optimize=True)
            + np.einsum("amef,fi->amei", I2B_vovv, T.b, optimize=True)
            - np.einsum("nmef,afni->amei", H0.ab.oovv, T.ab, optimize=True)
    )

    H.bb.voov += (
            np.einsum("amfe,fi->amie", I2C_vovv, T.b, optimize=True)
            - np.einsum("nmie,an->amie", I2C_ooov, T.b, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.bb.oovv, T.bb, optimize=True)
            + np.einsum("nmfe,fani->amie", H0.ab.oovv, T.ab, optimize=True)
    )

    Q1 = (
            np.einsum("mnjf,afin->amij", H.aa.ooov, T.aa, optimize=True)
            + np.einsum("mnjf,afin->amij", H.ab.ooov, T.ab, optimize=True)
    )
    Q2 = H0.aa.voov + 0.5 * np.einsum("amef,ei->amif", H0.aa.vovv, T.a, optimize=True)
    Q2 = np.einsum("amif,fj->amij", Q2, T.a, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.vooo += Q1 + (
            np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
            - np.einsum("nmij,an->amij", H.aa.oooo, T.a, optimize=True)
            + 0.5 * np.einsum("amef,efij->amij", H0.aa.vovv, T.aa, optimize=True)
    )

    Q1 = H0.ab.voov + np.einsum("amfe,fi->amie", H0.ab.vovv, T.a, optimize=True)
    H.ab.vooo += (
            np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
            - np.einsum("nmij,an->amij", H.ab.oooo, T.a, optimize=True)
            + np.einsum("mnjf,afin->amij", H.bb.ooov, T.ab, optimize=True)
            + np.einsum("nmfj,afin->amij", H.ab.oovo, T.aa, optimize=True)
            - np.einsum("nmif,afnj->amij", H.ab.ooov, T.ab, optimize=True)
            + np.einsum("amej,ei->amij", H0.ab.vovo, T.a, optimize=True)
            + np.einsum("amie,ej->amij", Q1, T.b, optimize=True)
            + np.einsum("amef,efij->amij", H0.ab.vovv, T.ab, optimize=True)
    )

    Q1 = H0.ab.ovov + np.einsum("mafe,fj->maje", H0.ab.ovvv, T.a, optimize=True)
    H.ab.ovoo += (
            np.einsum("me,eaji->maji", H.a.ov, T.ab, optimize=True)
            - np.einsum("mnji,an->maji", H.ab.oooo, T.b, optimize=True)
            + np.einsum("mnjf,fani->maji", H.aa.ooov, T.ab, optimize=True)
            + np.einsum("mnjf,fani->maji", H.ab.ooov, T.bb, optimize=True)
            - np.einsum("mnfi,fajn->maji", H.ab.oovo, T.ab, optimize=True)
            + np.einsum("maje,ei->maji", Q1, T.b, optimize=True)
            + np.einsum("maei,ej->maji", H0.ab.ovvo, T.a, optimize=True)
            + np.einsum("mafe,feji->maji", H0.ab.ovvv, T.ab, optimize=True)
    )

    Q1 = (
            np.einsum("mnjf,afin->amij", H.bb.ooov, T.bb, optimize=True)
            + np.einsum("nmfj,fani->amij", H.ab.oovo, T.ab, optimize=True)
    )
    Q2 = H0.bb.voov + 0.5 * np.einsum("amef,ei->amif", H0.bb.vovv, T.b, optimize=True)
    Q2 = np.einsum("amif,fj->amij", Q2, T.b, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.vooo += Q1 + (
            + np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
            - np.einsum("nmij,an->amij", H.bb.oooo, T.b, optimize=True)
            + 0.5 * np.einsum("amef,efij->amij", H0.bb.vovv, T.bb, optimize=True)
    )

    Q1 = (
            np.einsum("bnef,afin->abie", H.aa.vovv, T.aa, optimize=True)
            + np.einsum("bnef,afin->abie", H.ab.vovv, T.ab, optimize=True)
    )
    Q2 = H0.aa.ovov - 0.5 * np.einsum("mnie,bn->mbie", H0.aa.ooov, T.a, optimize=True)
    Q2 = -np.einsum("mbie,am->abie", Q2, T.a, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvov += Q1 + (
            - np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
            + np.einsum("abfe,fi->abie", H.aa.vvvv, T.a, optimize=True)
            + 0.5 * np.einsum("mnie,abmn->abie", H0.aa.ooov, T.aa, optimize=True)
    )

    Q1 = H0.ab.ovov - np.einsum("mnie,bn->mbie", H0.ab.ooov, T.b, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, T.a, optimize=True)
    H.ab.vvov += Q1 + (
            - np.einsum("me,abim->abie", H.b.ov, T.ab, optimize=True)
            + np.einsum("abfe,fi->abie", H.ab.vvvv, T.a, optimize=True)
            + np.einsum("nbfe,afin->abie", H.ab.ovvv, T.aa, optimize=True)
            + np.einsum("bnef,afin->abie", H.bb.vovv, T.ab, optimize=True)
            - np.einsum("amfe,fbim->abie", H.ab.vovv, T.ab, optimize=True)
            - np.einsum("amie,bm->abie", H0.ab.voov, T.b, optimize=True)
            + np.einsum("nmie,abnm->abie", H0.ab.ooov, T.ab, optimize=True)
    )

    Q1 = H0.ab.vovo - np.einsum("nmei,bn->bmei", H0.ab.oovo, T.a, optimize=True)
    Q1 = -np.einsum("bmei,am->baei", Q1, T.b, optimize=True)
    H.ab.vvvo += Q1 + (
            - np.einsum("me,bami->baei", H.a.ov, T.ab, optimize=True)
            + np.einsum("baef,fi->baei", H.ab.vvvv, T.b, optimize=True)
            + np.einsum("bnef,fani->baei", H.aa.vovv, T.ab, optimize=True)
            + np.einsum("bnef,fani->baei", H.ab.vovv, T.bb, optimize=True)
            - np.einsum("maef,bfmi->baei", H.ab.ovvv, T.ab, optimize=True)
            - np.einsum("naei,bn->baei", H0.ab.ovvo, T.a, optimize=True)
            + np.einsum("nmei,banm->baei", H0.ab.oovo, T.ab, optimize=True)
    )

    Q1 = (
            np.einsum("bnef,afin->abie", H.bb.vovv, T.bb, optimize=True)
            + np.einsum("nbfe,fani->abie", H.ab.ovvv, T.ab, optimize=True)
    )
    Q2 = H.bb.ovov - 0.5 * np.einsum("mnie,bn->mbie", H0.bb.ooov, T.b, optimize=True)
    Q2 = -np.einsum("mbie,am->abie", Q2, T.b, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvov += Q1 + (
            - np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
            + np.einsum("abfe,fi->abie", H.bb.vvvv, T.b, optimize=True)
            + 0.5 * np.einsum("mnie,abmn->abie", H0.bb.ooov, T.bb, optimize=True)
    )

    # Now, compute the CC3 intermediates
    X = Integral.from_empty(system, 2, data_type=H.a.oo.dtype)

    Q1 = -np.einsum("bmfe,am->abef", I2A_vovv, T.a, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    X.aa.vvvv = H0.aa.vvvv + Q1

    X.ab.vvvv = H0.ab.vvvv + (
            - np.einsum("mbef,am->abef", I2B_ovvv, T.a, optimize=True)
            - np.einsum("amef,bm->abef", I2B_vovv, T.b, optimize=True)
    )

    Q1 = -np.einsum("bmfe,am->abef", I2C_vovv, T.b, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    X.bb.vvvv = H0.bb.vvvv + Q1

    Q1 = +np.einsum("nmje,ei->mnij", I2A_ooov, T.a, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    X.aa.oooo = H0.aa.oooo + Q1

    X.ab.oooo = H0.ab.oooo + (
            np.einsum("mnej,ei->mnij", I2B_oovo, T.a, optimize=True)
            + np.einsum("mnie,ej->mnij", I2B_ooov, T.b, optimize=True)
    )

    Q1 = +np.einsum("nmje,ei->mnij", I2C_ooov, T.b, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    X.bb.oooo = H0.bb.oooo + Q1

    X.aa.voov = H0.aa.voov + (
            np.einsum("amfe,fi->amie", I2A_vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", I2A_ooov, T.a, optimize=True)
    )

    X.ab.voov = H0.ab.voov + (
            np.einsum("amfe,fi->amie", I2B_vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", I2B_ooov, T.a, optimize=True)
    )

    X.ab.ovvo = H0.ab.ovvo + (
            np.einsum("maef,fi->maei", I2B_ovvv, T.b, optimize=True)
            - np.einsum("mnei,an->maei", I2B_oovo, T.b, optimize=True)
    )

    X.ab.ovov = H0.ab.ovov + (
            np.einsum("mafe,fi->maie", I2B_ovvv, T.a, optimize=True)
            - np.einsum("mnie,an->maie", I2B_ooov, T.b, optimize=True)
    )

    X.ab.vovo = H0.ab.vovo + (
            - np.einsum("nmei,an->amei", I2B_oovo, T.a, optimize=True)
            + np.einsum("amef,fi->amei", I2B_vovv, T.b, optimize=True)
    )

    X.bb.voov = H0.bb.voov + (
            np.einsum("amfe,fi->amie", I2C_vovv, T.b, optimize=True)
            - np.einsum("nmie,an->amie", I2C_ooov, T.b, optimize=True)
    )

    Q1 = H0.aa.voov + 0.5 * np.einsum("amef,ei->amif", H0.aa.vovv, T.a, optimize=True)
    Q1 = np.einsum("amif,fj->amij", Q1, T.a, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    X.aa.vooo = H0.aa.vooo + Q1 - np.einsum("nmij,an->amij", X.aa.oooo, T.a, optimize=True)

    Q1 = H0.aa.voov + 0.5 * np.einsum("amef,ei->amif", H0.aa.vovv, T.a, optimize=True)
    Q1 = np.einsum("amif,fj->amij", Q1, T.a, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    X.ab.vooo = H0.aa.vooo + Q1 - np.einsum("nmij,an->amij", X.aa.oooo, T.a, optimize=True)

    Q1 = H0.ab.voov + np.einsum("amfe,fi->amie", H0.ab.vovv, T.a, optimize=True)
    X.ab.vooo = H0.ab.vooo + (
            - np.einsum("nmij,an->amij", X.ab.oooo, T.a, optimize=True)
            + np.einsum("amej,ei->amij", H0.ab.vovo, T.a, optimize=True)
            + np.einsum("amie,ej->amij", Q1, T.b, optimize=True)
    )

    Q1 = H0.ab.ovov + np.einsum("mafe,fj->maje", H0.ab.ovvv, T.a, optimize=True)
    X.ab.ovoo = H0.ab.ovoo + (
            - np.einsum("mnji,an->maji", X.ab.oooo, T.b, optimize=True)
            + np.einsum("maje,ei->maji", Q1, T.b, optimize=True)
            + np.einsum("maei,ej->maji", H0.ab.ovvo, T.a, optimize=True)
    )

    Q1 = H0.bb.voov + 0.5 * np.einsum("amef,ei->amif", H0.bb.vovv, T.b, optimize=True)
    Q1 = np.einsum("amif,fj->amij", Q1, T.b, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    X.bb.vooo = H0.bb.vooo + Q1 - np.einsum("nmij,an->amij", X.bb.oooo, T.b, optimize=True)

    Q1 = H0.aa.ovov - 0.5 * np.einsum("mnie,bn->mbie", H0.aa.ooov, T.a, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, T.a, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    X.aa.vvov = H0.aa.vvov + Q1 + np.einsum("abfe,fi->abie", X.aa.vvvv, T.a, optimize=True)

    Q1 = H0.ab.ovov - np.einsum("mnie,bn->mbie", H0.ab.ooov, T.b, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, T.a, optimize=True)
    X.ab.vvov = H0.ab.vvov + Q1 + (
            + np.einsum("abfe,fi->abie", X.ab.vvvv, T.a, optimize=True)
            - np.einsum("amie,bm->abie", H0.ab.voov, T.b, optimize=True)
    )

    Q1 = H0.ab.vovo - np.einsum("nmei,bn->bmei", H0.ab.oovo, T.a, optimize=True)
    Q1 = -np.einsum("bmei,am->baei", Q1, T.b, optimize=True)
    X.ab.vvvo = H0.ab.vvvo + Q1 + (
            + np.einsum("baef,fi->baei", X.ab.vvvv, T.b, optimize=True)
            - np.einsum("naei,bn->baei", H0.ab.ovvo, T.a, optimize=True)
    )

    Q1 = H0.bb.ovov - 0.5 * np.einsum("mnie,bn->mbie", H0.bb.ooov, T.b, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, T.b, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    X.bb.vvov = H0.bb.vvov + Q1 + np.einsum("abfe,fi->abie", X.bb.vvvv, T.b, optimize=True)

    # Compute t3 from CC3 perturbative expressions
    t3_aaa = compute_t3a(T, X, H0)
    t3_aab = compute_t3b(T, X, H0)
    t3_abb = compute_t3c(T, X, H0)
    t3_bbb = compute_t3d(T, X, H0)
    # Add in the t3-dependent terms to Hbar
    H.aa.vooo += (
            + 0.5 * np.einsum("mnef,aefijn->amij", H0.aa.oovv, t3_aaa, optimize=True)
            + np.einsum("mnef,aefijn->amij", H0.ab.oovv, t3_aab, optimize=True)
    )
    H.ab.vooo += (
            + np.einsum("nmfe,afeinj->amij", H0.ab.oovv, t3_aab, optimize=True)
            + 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, t3_abb, optimize=True)
    )
    H.ab.ovoo += (
            + 0.5 * np.einsum("mnef,efajni->maji", H0.aa.oovv, t3_aab, optimize=True)
            + np.einsum("mnef,efajni->maji", H0.ab.oovv, t3_abb, optimize=True)
    )
    H.bb.vooo += (
            + 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, t3_bbb, optimize=True)
            + np.einsum("nmfe,faenij->amij", H0.ab.oovv, t3_abb, optimize=True)
    )
    H.aa.vvov += (
            - 0.5 * np.einsum("mnef,abfimn->abie", H0.aa.oovv, t3_aaa, optimize=True)
            - np.einsum("mnef,abfimn->abie", H0.ab.oovv, t3_aab, optimize=True)
    )
    H.ab.vvov += (
            - np.einsum("nmfe,afbinm->abie", H0.ab.oovv, t3_aab, optimize=True)
            - 0.5 * np.einsum("mnef,afbinm->abie", H0.bb.oovv, t3_abb, optimize=True)
    )
    H.ab.vvvo += (
            - 0.5 * np.einsum("mnef,bfamni->baei", H0.aa.oovv, t3_aab, optimize=True)
            - np.einsum("mnef,bfamni->baei", H0.ab.oovv, t3_abb, optimize=True)
    )
    H.bb.vvov += (
            - 0.5 * np.einsum("mnef,abfimn->abie", H0.bb.oovv, t3_bbb, optimize=True)
            - np.einsum("nmfe,fabnim->abie", H0.ab.oovv, t3_abb, optimize=True)
    )

    return H, X

def compute_t3a(T, X, H0):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3))_C|0>.
    """
    # MM(2,3)A
    x3a = -0.25 * np.einsum("amij,bcmk->abcijk", X.aa.vooo, T.aa, optimize=True)
    x3a += 0.25 * np.einsum("abie,ecjk->abcijk", X.aa.vvov, T.aa, optimize=True)
    t3_aaa = cc3_loops.cc3_loops.compute_t3a(x3a, H0.a.oo, H0.a.vv)
    return t3_aaa

# @profile
def compute_t3b(T, X, H0):
    """
    Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N e^(T1+T2+T3))_C|0>.
    """
    # MM(2,3)B
    x3b = 0.5 * np.einsum("bcek,aeij->abcijk", X.ab.vvvo, T.aa, optimize=True)
    x3b -= 0.5 * np.einsum("mcjk,abim->abcijk", X.ab.ovoo, T.aa, optimize=True)
    x3b += np.einsum("acie,bejk->abcijk", X.ab.vvov, T.ab, optimize=True)
    x3b -= np.einsum("amik,bcjm->abcijk", X.ab.vooo, T.ab, optimize=True)
    x3b += 0.5 * np.einsum("abie,ecjk->abcijk", X.aa.vvov, T.ab, optimize=True)
    x3b -= 0.5 * np.einsum("amij,bcmk->abcijk", X.aa.vooo, T.ab, optimize=True)
    t3_aab = cc3_loops.cc3_loops.compute_t3b(x3b, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv)
    return t3_aab

# @profile
def compute_t3c(T, X, H0):
    """
    Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N e^(T1+T2+T3))_C|0>.
    """
    # MM(2,3)C
    x3c = 0.5 * np.einsum("abie,ecjk->abcijk", X.ab.vvov, T.bb, optimize=True)
    x3c -= 0.5 * np.einsum("amij,bcmk->abcijk", X.ab.vooo, T.bb, optimize=True)
    x3c += 0.5 * np.einsum("cbke,aeij->abcijk", X.bb.vvov, T.ab, optimize=True)
    x3c -= 0.5 * np.einsum("cmkj,abim->abcijk", X.bb.vooo, T.ab, optimize=True)
    x3c += np.einsum("abej,ecik->abcijk", X.ab.vvvo, T.ab, optimize=True)
    x3c -= np.einsum("mbij,acmk->abcijk", X.ab.ovoo, T.ab, optimize=True)
    t3_abb = cc3_loops.cc3_loops.compute_t3c(x3c, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv)
    return t3_abb

# @profile
def compute_t3d(T, X, H0):
    """
    Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N e^(T1+T2+T3))_C|0>.
    """
    # MM(2,3)D
    x3d = -0.25 * np.einsum("amij,bcmk->abcijk", X.bb.vooo, T.bb, optimize=True)
    x3d += 0.25 * np.einsum("abie,ecjk->abcijk", X.bb.vvov, T.bb, optimize=True)
    t3_bbb = cc3_loops.cc3_loops.compute_t3d(x3d, H0.b.oo, H0.b.vv)
    return t3_bbb