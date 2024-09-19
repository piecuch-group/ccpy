import numpy as np

from ccpy.models.integrals import Integral


def get_eomccsd_intermediates(H, R, system):
    """Calculate the H*(R1+R2) intermediates for EOMCCSD."""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 1, data_type=H.a.oo.dtype)

    X.a.oo = (
            #np.einsum("me,ej->mj", H.a.ov, R.a, optimize=True)
            + np.einsum("mnjf,fn->mj", H.aa.ooov, R.a, optimize=True)
            + np.einsum("mnjf,fn->mj", H.ab.ooov, R.b, optimize=True)
            + 0.5 * np.einsum("mnef,efjn->mj", H.aa.oovv, R.aa, optimize=True)
            + np.einsum("mnef,efjn->mj", H.ab.oovv, R.ab, optimize=True)
    )

    X.a.vv = (
            #-1.0 * np.einsum("me,bm->be", H.a.ov, R.a, optimize=True)
            + np.einsum("bnef,fn->be", H.aa.vovv, R.a, optimize=True)
            + np.einsum("bnef,fn->be", H.ab.vovv, R.b, optimize=True)
            - 0.5 * np.einsum("mnef,bfmn->be", H.aa.oovv, R.aa, optimize=True)
            - np.einsum("mnef,bfmn->be", H.ab.oovv, R.ab, optimize=True)
    )

    X.b.oo = (
            #np.einsum("me,ek->mk", H.b.ov, R.b, optimize=True)
            + np.einsum("nmfk,fn->mk", H.ab.oovo, R.a, optimize=True)
            + np.einsum("mnkf,fn->mk", H.bb.ooov, R.b, optimize=True)
            + np.einsum("nmfe,fenk->mk", H.ab.oovv, R.ab, optimize=True)
            + 0.5 * np.einsum("mnef,efkn->mk", H.bb.oovv, R.bb, optimize=True)
    )

    X.b.vv = (
            #-1.0 * np.einsum("me,cm->ce", H.b.ov, R.b, optimize=True)
            + np.einsum("ncfe,fn->ce", H.ab.ovvv, R.a, optimize=True)
            + np.einsum("cnef,fn->ce", H.bb.vovv, R.b, optimize=True)
            -1.0 * np.einsum("nmfe,fcnm->ce", H.ab.oovv, R.ab, optimize=True)
            - 0.5 * np.einsum("mnef,fcnm->ce", H.bb.oovv, R.bb, optimize=True)
    )
    return X

def get_eomccsd_chol_intermediates(H, T, R, system):
    """Calculate the H*(R1+R2) intermediates for EOMCCSD."""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 2, data_type=H.a.oo.dtype, use_none=True)

    X.a.oo = (
            #np.einsum("me,ej->mj", H.a.ov, R.a, optimize=True)
            + np.einsum("mnjf,fn->mj", H.aa.ooov, R.a, optimize=True)
            + np.einsum("mnjf,fn->mj", H.ab.ooov, R.b, optimize=True)
            + 0.5 * np.einsum("mnef,efjn->mj", H.aa.oovv, R.aa, optimize=True)
            + np.einsum("mnef,efjn->mj", H.ab.oovv, R.ab, optimize=True)
    )

    X.a.vv = (
            #-1.0 * np.einsum("me,bm->be", H.a.ov, R.a, optimize=True)
            + np.einsum("bnef,fn->be", H.aa.vovv, R.a, optimize=True)
            + np.einsum("bnef,fn->be", H.ab.vovv, R.b, optimize=True)
            - 0.5 * np.einsum("mnef,bfmn->be", H.aa.oovv, R.aa, optimize=True)
            - np.einsum("mnef,bfmn->be", H.ab.oovv, R.ab, optimize=True)
    )

    X.b.oo = (
            #np.einsum("me,ek->mk", H.b.ov, R.b, optimize=True)
            + np.einsum("nmfk,fn->mk", H.ab.oovo, R.a, optimize=True)
            + np.einsum("mnkf,fn->mk", H.bb.ooov, R.b, optimize=True)
            + np.einsum("nmfe,fenk->mk", H.ab.oovv, R.ab, optimize=True)
            + 0.5 * np.einsum("mnef,efkn->mk", H.bb.oovv, R.bb, optimize=True)
    )

    X.b.vv = (
            #-1.0 * np.einsum("me,cm->ce", H.b.ov, R.b, optimize=True)
            + np.einsum("ncfe,fn->ce", H.ab.ovvv, R.a, optimize=True)
            + np.einsum("cnef,fn->ce", H.bb.vovv, R.b, optimize=True)
            -1.0 * np.einsum("nmfe,fcnm->ce", H.ab.oovv, R.ab, optimize=True)
            - 0.5 * np.einsum("mnef,fcnm->ce", H.bb.oovv, R.bb, optimize=True)
    )

    X.aa.oooo = (
        0.5 * np.einsum("mnef,efij->mnij", H.aa.oovv, R.aa, optimize=True)
    )
    X.ab.oooo = (
        np.einsum("mnef,efij->mnij", H.ab.oovv, R.ab, optimize=True)
    )
    X.bb.oooo = (
        0.5 * np.einsum("mnef,efij->mnij", H.bb.oovv, R.bb, optimize=True)
    )

    # Recover bare (vovv) and (ooov) parts from hbar(vovv) and hbar(ooov)
    h0_aa_vovv = H.aa.vovv + np.einsum("mnfe,an->amef", H.aa.oovv, T.a, optimize=True)
    h0_ab_ovvv = H.ab.ovvv + np.einsum("mnef,an->maef", H.ab.oovv, T.b, optimize=True)
    h0_ab_vovv = H.ab.vovv + np.einsum("nmef,an->amef", H.ab.oovv, T.a, optimize=True)
    h0_bb_vovv = H.bb.vovv + np.einsum("mnfe,an->amef", H.bb.oovv, T.b, optimize=True)
    X.aa.vooo = (
        0.5 * np.einsum("anef,efij->anij", h0_aa_vovv, R.aa, optimize=True)
    )
    X.bb.vooo = (
        0.5 * np.einsum("anef,efij->anij", h0_bb_vovv, R.bb, optimize=True)
    )
    X.ab.vooo = (
        np.einsum("amef,efij->amij", h0_ab_vovv, R.ab, optimize=True)
    )
    X.ab.ovoo = (
        np.einsum("mbef,efij->mbij", h0_ab_ovvv, R.ab, optimize=True)
    )

    return X