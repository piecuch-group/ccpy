import numpy as np

from ccpy.models.integrals import Integral

def get_eomccsd_intermediates(H, R, system):
    """Calculate the H*(R1+R2) intermediates for EOMCCSD."""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 1, data_type=H.a.oo.dtype)

    X.a.ov = (
        np.einsum("mnef,fn->me", H.aa.oovv, R.a, optimize=True)
        + np.einsum("mnef,fn->me", H.ab.oovv, R.b, optimize=True)
    )

    X.b.ov = (
        np.einsum("nmfe,fn->me", H.ab.oovv, R.a, optimize=True)
        + np.einsum("nmfe,fn->me", H.bb.oovv, R.b, optimize=True)
    )

    X.a.oo = (
            + np.einsum("mnjf,fn->mj", H.aa.ooov, R.a, optimize=True)
            + np.einsum("mnjf,fn->mj", H.ab.ooov, R.b, optimize=True)
            + 0.5 * np.einsum("mnef,efjn->mj", H.aa.oovv, R.aa, optimize=True)
            + np.einsum("mnef,efjn->mj", H.ab.oovv, R.ab, optimize=True)
    )

    X.a.vv = (
            + np.einsum("bnef,fn->be", H.aa.vovv, R.a, optimize=True)
            + np.einsum("bnef,fn->be", H.ab.vovv, R.b, optimize=True)
            - 0.5 * np.einsum("mnef,bfmn->be", H.aa.oovv, R.aa, optimize=True)
            - np.einsum("mnef,bfmn->be", H.ab.oovv, R.ab, optimize=True)
    )

    X.b.oo = (
            + np.einsum("nmfk,fn->mk", H.ab.oovo, R.a, optimize=True)
            + np.einsum("mnkf,fn->mk", H.bb.ooov, R.b, optimize=True)
            + np.einsum("nmfe,fenk->mk", H.ab.oovv, R.ab, optimize=True)
            + 0.5 * np.einsum("mnef,efkn->mk", H.bb.oovv, R.bb, optimize=True)
    )

    X.b.vv = (
            + np.einsum("ncfe,fn->ce", H.ab.ovvv, R.a, optimize=True)
            + np.einsum("cnef,fn->ce", H.bb.vovv, R.b, optimize=True)
            -1.0 * np.einsum("nmfe,fcnm->ce", H.ab.oovv, R.ab, optimize=True)
            - 0.5 * np.einsum("mnef,fcnm->ce", H.bb.oovv, R.bb, optimize=True)
    )
    return X

def get_HR1_intermediates(H, R, system):
    """Calculate the (H(1)*R1)_C intermediates for EOMCC3, where
    H(1) = exp(-T1) H exp(T1) is the CCS-like similarity transformed
    Hamiltonian."""

    HR1 = Integral.from_empty(system, 2, data_type=np.float64, use_none=True)

    HR1.aa.vvov = (
        -np.einsum("anie,bn->abie", H.aa.voov, R.a, optimize=True)
        +0.5 * np.einsum("abfe,fi->abie", H.aa.vvvv, R.a, optimize=True)
    )
    HR1.aa.vvov -= np.transpose(HR1.aa.vvov, (1, 0, 2, 3))
    HR1.aa.vooo = (
         np.einsum("amif,fj->amij", H.aa.voov, R.a, optimize=True)
        -0.5 * np.einsum("nmij,an->amij", H.aa.oooo, R.a, optimize=True)
    )
    HR1.aa.vooo -= np.transpose(HR1.aa.vooo, (0, 1, 3, 2))
    HR1.ab.vvov = (
        -np.einsum("nbie,an->abie", H.ab.ovov, R.a, optimize=True)
        +np.einsum("abfe,fi->abie", H.ab.vvvv, R.a, optimize=True)
        -np.einsum("amie,bm->abie", H.ab.voov, R.b, optimize=True)
    )
    HR1.ab.vvvo = (
        -np.einsum("amej,bm->abej", H.ab.vovo, R.b, optimize=True)
        +np.einsum("abef,fj->abej", H.ab.vvvv, R.b, optimize=True)
        -np.einsum("mbej,am->abej", H.ab.ovvo, R.a, optimize=True)
    )
    HR1.ab.vooo = (
         np.einsum("amej,ei->amij", H.ab.vovo, R.a, optimize=True)
        -np.einsum("nmij,an->amij", H.ab.oooo, R.a, optimize=True)
        +np.einsum("amie,ej->amij", H.ab.voov, R.b, optimize=True)
    )
    HR1.ab.ovoo = (
         np.einsum("mbie,ej->mbij", H.ab.ovov, R.b, optimize=True)
        -np.einsum("mnij,bn->mbij", H.ab.oooo, R.b, optimize=True)
        +np.einsum("mbej,ei->mbij", H.ab.ovvo, R.a, optimize=True)
    )
    HR1.bb.vvov = (
            -np.einsum("anie,bn->abie", H.bb.voov, R.b, optimize=True)
            + 0.5 * np.einsum("abfe,fi->abie", H.bb.vvvv, R.b, optimize=True)
    )
    HR1.bb.vvov -= np.transpose(HR1.bb.vvov, (1, 0, 2, 3))
    HR1.bb.vooo = (
            np.einsum("amif,fj->amij", H.bb.voov, R.b, optimize=True)
            - 0.5 * np.einsum("nmij,an->amij", H.bb.oooo, R.b, optimize=True)
    )
    HR1.bb.vooo -= np.transpose(HR1.bb.vooo, (0, 1, 3, 2))

    return HR1