import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.models.integrals import Integral

def build_hbar_cc2(T, H0, RHF_symmetry, system, *args):
    """Calculate the one- and two-body components of the CC2 similarity-transformed
     Hamiltonian."""
    from copy import deepcopy

    # Copy the Bare Hamiltonian object for T1/T2-similarity transformed HBar
    H = deepcopy(H0)

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
    H.aa.vvvv += Q1

    H.ab.vvvv += (
            - ccpy_einsum("mbef,am->abef", I2B_ovvv, T.a)
            - ccpy_einsum("amef,bm->abef", I2B_vovv, T.b)
    )

    Q1 = -ccpy_einsum("bmfe,am->abef", I2C_vovv, T.b)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvvv +=  Q1

    Q1 = +ccpy_einsum("nmje,ei->mnij", I2A_ooov, T.a)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.oooo += Q1

    H.ab.oooo += (
            ccpy_einsum("mnej,ei->mnij", I2B_oovo, T.a)
            + ccpy_einsum("mnie,ej->mnij", I2B_ooov, T.b)
    )

    Q1 = +ccpy_einsum("nmje,ei->mnij", I2C_ooov, T.b)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.oooo += Q1

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

    Q1 = H0.aa.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.aa.vovv, T.a)
    Q1 = ccpy_einsum("amif,fj->amij", Q1, T.a)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.vooo += Q1 + (
            - ccpy_einsum("nmij,an->amij", H.aa.oooo, T.a)
    )

    Q1 = H0.ab.voov + ccpy_einsum("amfe,fi->amie", H0.ab.vovv, T.a)
    H.ab.vooo += (
            - ccpy_einsum("nmij,an->amij", H.ab.oooo, T.a)
            + ccpy_einsum("amej,ei->amij", H0.ab.vovo, T.a)
            + ccpy_einsum("amie,ej->amij", Q1, T.b)
    )

    Q1 = H0.ab.ovov + ccpy_einsum("mafe,fj->maje", H0.ab.ovvv, T.a)
    H.ab.ovoo += (
            - ccpy_einsum("mnji,an->maji", H.ab.oooo, T.b)
            + ccpy_einsum("maje,ei->maji", Q1, T.b)
            + ccpy_einsum("maei,ej->maji", H0.ab.ovvo, T.a)
    )

    Q1 = H0.bb.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.bb.vovv, T.b)
    Q1 = ccpy_einsum("amif,fj->amij", Q1, T.b)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.vooo += Q1 + (
            - ccpy_einsum("nmij,an->amij", H.bb.oooo, T.b)
    )

    Q1 = H0.aa.ovov - 0.5 * ccpy_einsum("mnie,bn->mbie", H0.aa.ooov, T.a)
    Q1 = -ccpy_einsum("mbie,am->abie", Q1, T.a)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvov += Q1 + (
            + ccpy_einsum("abfe,fi->abie", H.aa.vvvv, T.a)
    )

    Q1 = H0.ab.ovov - ccpy_einsum("mnie,bn->mbie", H0.ab.ooov, T.b)
    Q1 = -ccpy_einsum("mbie,am->abie", Q1, T.a)
    H.ab.vvov += Q1 + (
            + ccpy_einsum("abfe,fi->abie", H.ab.vvvv, T.a)
            - ccpy_einsum("amie,bm->abie", H0.ab.voov, T.b)
    )

    Q1 = H0.ab.vovo - ccpy_einsum("nmei,bn->bmei", H0.ab.oovo, T.a)
    Q1 = -ccpy_einsum("bmei,am->baei", Q1, T.b)
    H.ab.vvvo += Q1 + (
            + ccpy_einsum("baef,fi->baei", H.ab.vvvv, T.b)
            - ccpy_einsum("naei,bn->baei", H0.ab.ovvo, T.a)
    )

    Q1 = H.bb.ovov - 0.5 * ccpy_einsum("mnie,bn->mbie", H0.bb.ooov, T.b)
    Q1 = -ccpy_einsum("mbie,am->abie", Q1, T.b)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvov += Q1 + (
            + ccpy_einsum("abfe,fi->abie", H.bb.vvvv, T.b)
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

    # fock = Integral.from_empty(system, 1, data_type=H.a.oo.dtype)
    # fock.a.oo = H0.a.oo.copy()
    # fock.a.vv = H0.a.vv.copy()
    # fock.b.oo = H0.b.oo.copy()
    # fock.b.vv = H0.b.vv.copy()

    return H
