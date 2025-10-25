import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

from ccpy.models.integrals import Integral


def get_eomccsd_intermediates(H, R, system):
    """Calculate the H*(R1+R2) intermediates for EOMCCSD."""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 1, data_type=H.a.oo.dtype)

    X.a.oo = (
            #ccpy_einsum("me,ej->mj", H.a.ov, R.a)
            + ccpy_einsum("mnjf,fn->mj", H.aa.ooov, R.a)
            + ccpy_einsum("mnjf,fn->mj", H.ab.ooov, R.b)
            + 0.5 * ccpy_einsum("mnef,efjn->mj", H.aa.oovv, R.aa)
            + ccpy_einsum("mnef,efjn->mj", H.ab.oovv, R.ab)
    )

    X.a.vv = (
            #-1.0 * ccpy_einsum("me,bm->be", H.a.ov, R.a)
            + ccpy_einsum("bnef,fn->be", H.aa.vovv, R.a)
            + ccpy_einsum("bnef,fn->be", H.ab.vovv, R.b)
            - 0.5 * ccpy_einsum("mnef,bfmn->be", H.aa.oovv, R.aa)
            - ccpy_einsum("mnef,bfmn->be", H.ab.oovv, R.ab)
    )

    X.b.oo = (
            #ccpy_einsum("me,ek->mk", H.b.ov, R.b)
            + ccpy_einsum("nmfk,fn->mk", H.ab.oovo, R.a)
            + ccpy_einsum("mnkf,fn->mk", H.bb.ooov, R.b)
            + ccpy_einsum("nmfe,fenk->mk", H.ab.oovv, R.ab)
            + 0.5 * ccpy_einsum("mnef,efkn->mk", H.bb.oovv, R.bb)
    )

    X.b.vv = (
            #-1.0 * ccpy_einsum("me,cm->ce", H.b.ov, R.b)
            + ccpy_einsum("ncfe,fn->ce", H.ab.ovvv, R.a)
            + ccpy_einsum("cnef,fn->ce", H.bb.vovv, R.b)
            -1.0 * ccpy_einsum("nmfe,fcnm->ce", H.ab.oovv, R.ab)
            - 0.5 * ccpy_einsum("mnef,fcnm->ce", H.bb.oovv, R.bb)
    )
    return X

def get_eomccsd_chol_intermediates(H, T, R, system):
    """Calculate the H*(R1+R2) intermediates for EOMCCSD."""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 2, data_type=H.a.oo.dtype, use_none=True)

    X.a.oo = (
            #ccpy_einsum("me,ej->mj", H.a.ov, R.a)
            + ccpy_einsum("mnjf,fn->mj", H.aa.ooov, R.a)
            + ccpy_einsum("mnjf,fn->mj", H.ab.ooov, R.b)
            + 0.5 * ccpy_einsum("mnef,efjn->mj", H.aa.oovv, R.aa)
            + ccpy_einsum("mnef,efjn->mj", H.ab.oovv, R.ab)
    )

    X.a.vv = (
            #-1.0 * ccpy_einsum("me,bm->be", H.a.ov, R.a)
            + ccpy_einsum("bnef,fn->be", H.aa.vovv, R.a)
            + ccpy_einsum("bnef,fn->be", H.ab.vovv, R.b)
            - 0.5 * ccpy_einsum("mnef,bfmn->be", H.aa.oovv, R.aa)
            - ccpy_einsum("mnef,bfmn->be", H.ab.oovv, R.ab)
    )

    X.b.oo = (
            #ccpy_einsum("me,ek->mk", H.b.ov, R.b)
            + ccpy_einsum("nmfk,fn->mk", H.ab.oovo, R.a)
            + ccpy_einsum("mnkf,fn->mk", H.bb.ooov, R.b)
            + ccpy_einsum("nmfe,fenk->mk", H.ab.oovv, R.ab)
            + 0.5 * ccpy_einsum("mnef,efkn->mk", H.bb.oovv, R.bb)
    )

    X.b.vv = (
            #-1.0 * ccpy_einsum("me,cm->ce", H.b.ov, R.b)
            + ccpy_einsum("ncfe,fn->ce", H.ab.ovvv, R.a)
            + ccpy_einsum("cnef,fn->ce", H.bb.vovv, R.b)
            -1.0 * ccpy_einsum("nmfe,fcnm->ce", H.ab.oovv, R.ab)
            - 0.5 * ccpy_einsum("mnef,fcnm->ce", H.bb.oovv, R.bb)
    )

    X.aa.oooo = (
        0.5 * ccpy_einsum("mnef,efij->mnij", H.aa.oovv, R.aa)
    )
    X.ab.oooo = (
        ccpy_einsum("mnef,efij->mnij", H.ab.oovv, R.ab)
    )
    X.bb.oooo = (
        0.5 * ccpy_einsum("mnef,efij->mnij", H.bb.oovv, R.bb)
    )
    return X