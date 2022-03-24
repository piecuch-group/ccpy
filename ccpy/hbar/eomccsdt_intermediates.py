import numpy as np

from ccpy.models.integrals import Integral


def get_eomccsd_intermediates(H, R, system):
    """Calculate the CCSD-like intermediates for CCSDT. This routine
    should only calculate terms with T2 and any remaining terms outside of the CCS intermediate
    routine."""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 2, data_type=H.a.oo.dtype)

    X.a.oo = (
            np.einsum("me,ej->mj", H.a.ov, R.a, optimize=True)
            + np.einsum("mnjf,fn->mj", H.aa.ooov, R.a, optimize=True)
            + np.einsum("mnjf,fn->mj", H.ab.ooov, R.b, optimize=True)
            + 0.5 * np.einsum("mnef,efjn->mj", H.aa.oovv, R.aa, optimize=True)
            + np.einsum("mnef,efjn->mj", H.ab.oovv, R.ab, optimize=True)
    )

    X.a.vv = (
            -1.0 * np.einsum("me,bm->be", H.a.ov, R.a, optimize=True)
            + np.einsum("bnef,fn->be", H.aa.vovv, R.a, optimize=True)
            + np.einsum("bnef,fn->be", H.ab.vovv, R.b, optimize=True)
            - 0.5 * np.einsum("mnef,bfmn->be", H.aa.oovv, R.aa, optimize=True)
            - np.einsum("mnef,bfmn->be", H.ab.oovv, R.ab, optimize=True)
    )

    X.b.oo = (
            np.einsum("me,ek->mk", H.b.ov, R.b, optimize=True)
            + np.einsum("nmfk,fn->mk", H.ab.oovo, R.a, optimize=True)
            + np.einsum("mnkf,fn->mk", H.bb.ooov, R.b, optimize=True)
            + np.einsum("nmfe,fenk->mk", H.ab.oovv, R.ab, optimize=True)
            + 0.5 * np.einsum("mnef,efkn->mk", H.bb.oovv, R.bb, optimize=True)
    )

    X.b.vv = (
            -1.0 * np.einsum("me,cm->ce", H.b.ov, R.b, optimize=True)
            + np.einsum("ncfe,fn->ce", H.ab.ovvv, R.a, optimize=True)
            + np.einsum("cnef,fn->ce", H.bb.vovv, R.b, optimize=True)
            -1.0 * np.einsum("nmfe,fcnm->ce", H.ab.oovv, R.ab, optimize=True)
            - 0.5 * np.einsum("mnef,fcnm->ce", H.bb.oovv, R.bb, optimize=True)
    )

    X.aa.oooo = (
        np.einsum("nmje,ei->mnij", H.aa.ooov, R.a, optimize=True)
        + 0.25 * np.einsum("mnef,efij->mnij", H.aa.oovv, R.aa, optimize=True)
    )
    X.aa.oooo -= np.transpose(X.aa.oooo, (0, 1, 3, 2))

    X.ab.oooo = (
        np.einsum("nmje,ek->nmjk", H.ab.ooov, R.b, optimize=True)
        + np.einsum("nmek,ej->nmjk", H.ab.oovo, R.a, optimize=True)
        + np.einsum("mnef,efjk->mnjk", H.ab.oovv, R.ab, optimize=True)
    )

    X.bb.oooo = (
        np.einsum("mnie,ej->mnij", H.bb.ooov, R.b, optimize=True)
        + 0.25 * np.einsum("mnef,efij->mnij", H.bb.oovv, R.bb, optimize=True)
    )
    X.bb.oooo -= np.transpose(X.bb.oooo, (0, 1, 3, 2))

    X.aa.vvvv = (
        -1.0 * np.einsum("amef,bm->abef", H.aa.vovv, R.a, optimize=True)
        + 0.25 * np.einsum("mnef,abmn->abef", H.aa.oovv, R.aa, optimize=True)
    )
    X.aa.vvvv -= np.transpose(X.aa.vvvv, (1, 0, 2, 3))

    X.ab.vvvv = (
        - np.einsum("bmfe,cm->bcfe", H.ab.vovv, R.b, optimize=True)
        - np.einsum("mcfe,bm->bcfe", H.ab.ovvv, R.a, optimize=True)
        + np.einsum("mnef,bcmn->bcef", H.ab.oovv, R.ab, optimize=True)
    )

    X.bb.vvvv = (
        - np.einsum("amef,bm->abef", H.bb.vovv, R.b, optimize=True)
        + 0.25 * np.einsum("mnef,abmn->abef", H.bb.oovv, R.bb, optimize=True)
    )
    X.bb.vvvv -= np.transpose(X.bb.vvvv, (1, 0, 2, 3))

    X.aa.voov = (
        -1.0 * np.einsum("nmje,bn->bmje", H.aa.ooov, R.a, optimize=True)
        + np.einsum("bmfe,fj->bmje", H.aa.vovv, R.a, optimize=True)
        + np.einsum("mnef,fcnk->cmke", H.aa.oovv, R.aa, optimize=True)
        + np.einsum("mnef,cfkn->cmke", H.ab.oovv, R.ab, optimize=True)
    )

    X.ab.voov = (
        -1.0 * np.einsum("nmje,bn->bmje", H.ab.ooov, R.a, optimize=True)
        + np.einsum("bmfe,fj->bmje", H.ab.vovv, R.a, optimize=True)
        + np.einsum("nmfe,fcnk->cmke", H.ab.oovv, R.aa, optimize=True)
        + np.einsum("mnef,cfkn->cmke", H.bb.oovv, R.ab, optimize=True)
    )

    X.ab.ovvo = (
        - np.einsum("nmfk,cm->ncfk", H.ab.oovo, R.b, optimize=True)
        + np.einsum("ncfe,ek->ncfk", H.ab.ovvv, R.b, optimize=True)
        + np.einsum("mnef,ecmk->ncfk", H.aa.oovv, R.ab, optimize=True)
        + np.einsum("nmfe,ecmk->ncfk", H.ab.oovv, R.bb, optimize=True)
    )

    X.ab.vovo = (
        np.einsum("bmfe,ek->bmfk", H.ab.vovv, R.b, optimize=True)
        - np.einsum("nmfk,bn->bmfk", H.ab.oovo, R.a, optimize=True)
        - np.einsum("mnef,bfmk->bnek", H.ab.oovv, R.ab, optimize=True)
    )

    X.ab.ovov = (
        - np.einsum("nmje,cm->ncje", H.ab.ooov, R.b, optimize=True)
        + np.einsum("ncfe,fj->ncje", H.ab.ovvv, R.a, optimize=True)
        - np.einsum("mnef,ecjn->mcjf", H.ab.oovv, R.ab, optimize=True)
    )

    X.bb.voov = (
            - np.einsum("mnkf,cm->cnkf", H.bb.ooov, R.b, optimize=True)
            + np.einsum("cnef,ek->cnkf", H.bb.vovv, R.b, optimize=True)
            + np.einsum("mnef,ecmk->cnkf", H.ab.oovv, R.ab, optimize=True)
            + np.einsum("mnef,ecmk->cnkf", H.bb.oovv, R.bb, optimize=True)
    )

    return X
