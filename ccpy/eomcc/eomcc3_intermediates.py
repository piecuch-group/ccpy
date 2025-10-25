import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

from ccpy.models.integrals import Integral

def get_eomccsd_intermediates(H, R, system):
    """Calculate the H*(R1+R2) intermediates for EOMCCSD."""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 1, data_type=H.a.oo.dtype)

    X.a.ov = (
        ccpy_einsum("mnef,fn->me", H.aa.oovv, R.a)
        + ccpy_einsum("mnef,fn->me", H.ab.oovv, R.b)
    )

    X.b.ov = (
        ccpy_einsum("nmfe,fn->me", H.ab.oovv, R.a)
        + ccpy_einsum("nmfe,fn->me", H.bb.oovv, R.b)
    )

    X.a.oo = (
            + ccpy_einsum("mnjf,fn->mj", H.aa.ooov, R.a)
            + ccpy_einsum("mnjf,fn->mj", H.ab.ooov, R.b)
            + 0.5 * ccpy_einsum("mnef,efjn->mj", H.aa.oovv, R.aa)
            + ccpy_einsum("mnef,efjn->mj", H.ab.oovv, R.ab)
    )

    X.a.vv = (
            + ccpy_einsum("bnef,fn->be", H.aa.vovv, R.a)
            + ccpy_einsum("bnef,fn->be", H.ab.vovv, R.b)
            - 0.5 * ccpy_einsum("mnef,bfmn->be", H.aa.oovv, R.aa)
            - ccpy_einsum("mnef,bfmn->be", H.ab.oovv, R.ab)
    )

    X.b.oo = (
            + ccpy_einsum("nmfk,fn->mk", H.ab.oovo, R.a)
            + ccpy_einsum("mnkf,fn->mk", H.bb.ooov, R.b)
            + ccpy_einsum("nmfe,fenk->mk", H.ab.oovv, R.ab)
            + 0.5 * ccpy_einsum("mnef,efkn->mk", H.bb.oovv, R.bb)
    )

    X.b.vv = (
            + ccpy_einsum("ncfe,fn->ce", H.ab.ovvv, R.a)
            + ccpy_einsum("cnef,fn->ce", H.bb.vovv, R.b)
            -1.0 * ccpy_einsum("nmfe,fcnm->ce", H.ab.oovv, R.ab)
            - 0.5 * ccpy_einsum("mnef,fcnm->ce", H.bb.oovv, R.bb)
    )
    return X

def get_HR1_intermediates(H, R, system):
    """Calculate the (H(1)*R1)_C intermediates for EOMCC3, where
    H(1) = exp(-T1) H exp(T1) is the CCS-like similarity transformed
    Hamiltonian."""

    HR1 = Integral.from_empty(system, 2, data_type=np.float64, use_none=True)

    HR1.aa.vvov = (
        -ccpy_einsum("anie,bn->abie", H.aa.voov, R.a)
        +0.5 * ccpy_einsum("abfe,fi->abie", H.aa.vvvv, R.a)
    )
    HR1.aa.vvov -= np.transpose(HR1.aa.vvov, (1, 0, 2, 3))
    HR1.aa.vooo = (
         ccpy_einsum("amif,fj->amij", H.aa.voov, R.a)
        -0.5 * ccpy_einsum("nmij,an->amij", H.aa.oooo, R.a)
    )
    HR1.aa.vooo -= np.transpose(HR1.aa.vooo, (0, 1, 3, 2))
    HR1.ab.vvov = (
        -ccpy_einsum("nbie,an->abie", H.ab.ovov, R.a)
        +ccpy_einsum("abfe,fi->abie", H.ab.vvvv, R.a)
        -ccpy_einsum("amie,bm->abie", H.ab.voov, R.b)
    )
    HR1.ab.vvvo = (
        -ccpy_einsum("amej,bm->abej", H.ab.vovo, R.b)
        +ccpy_einsum("abef,fj->abej", H.ab.vvvv, R.b)
        -ccpy_einsum("mbej,am->abej", H.ab.ovvo, R.a)
    )
    HR1.ab.vooo = (
         ccpy_einsum("amej,ei->amij", H.ab.vovo, R.a)
        -ccpy_einsum("nmij,an->amij", H.ab.oooo, R.a)
        +ccpy_einsum("amie,ej->amij", H.ab.voov, R.b)
    )
    HR1.ab.ovoo = (
         ccpy_einsum("mbie,ej->mbij", H.ab.ovov, R.b)
        -ccpy_einsum("mnij,bn->mbij", H.ab.oooo, R.b)
        +ccpy_einsum("mbej,ei->mbij", H.ab.ovvo, R.a)
    )
    HR1.bb.vvov = (
            -ccpy_einsum("anie,bn->abie", H.bb.voov, R.b)
            + 0.5 * ccpy_einsum("abfe,fi->abie", H.bb.vvvv, R.b)
    )
    HR1.bb.vvov -= np.transpose(HR1.bb.vvov, (1, 0, 2, 3))
    HR1.bb.vooo = (
            ccpy_einsum("amif,fj->amij", H.bb.voov, R.b)
            - 0.5 * ccpy_einsum("nmij,an->amij", H.bb.oooo, R.b)
    )
    HR1.bb.vooo -= np.transpose(HR1.bb.vooo, (0, 1, 3, 2))

    return HR1