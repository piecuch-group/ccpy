import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

from ccpy.lib.core import cc_loops2


def update(L, LH, T, H, omega, shift, is_ground, flag_RHF, system):

    # build L1
    LH = build_LH_1A(L, LH, T, H)
    # build L2
    LH = build_LH_2A(L, LH, T, H)
    LH = build_LH_2B(L, LH, T, H)
    # Update the L vector
    L.a, L.aa, L.ab, LH.a, LH.aa, LH.ab = cc_loops2.update_l_2h1p(L.a, L.aa, L.ab,
                                                                            LH.a, LH.aa, LH.ab,
                                                                            omega,
                                                                            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                                            shift)
    return L, LH

def update_l(L, omega, H, RHF_symmetry, system):
    L.a, L.aa, L.ab = cc_loops2.update_r_2h1p(
            L.a,
            L.aa,
            L.ab,
            omega,
            H.a.oo,
            H.a.vv,
            H.b.oo,
            H.b.vv,
            0.0
    )
    return L

def LH_fun(LH, L, T, H, flag_RHF, system):
    # build L1
    LH = build_LH_1A(L, LH, T, H)
    # build L2
    LH = build_LH_2A(L, LH, T, H)
    LH = build_LH_2B(L, LH, T, H)
    return LH.flatten()

def build_LH_1A(L, LH, T, H):

    LH.a = -1.0 * ccpy_einsum("m,im->i", L.a, H.a.oo)
    LH.a -= 0.5 * ccpy_einsum("mfn,finm->i", L.aa, H.aa.vooo)
    LH.a -= ccpy_einsum("mfn,ifmn->i", L.ab, H.ab.ovoo)
    return LH

def build_LH_2A(L, LH, T, H):

    LH.aa = ccpy_einsum("i,jb->ibj", L.a, H.a.ov)
    LH.aa -= 0.5 * ccpy_einsum("m,ijmb->ibj", L.a, H.aa.ooov)
    LH.aa += 0.5 * ccpy_einsum("iej,eb->ibj", L.aa, H.a.vv)
    LH.aa -= ccpy_einsum("ibm,jm->ibj", L.aa, H.a.oo)
    LH.aa += 0.25 * ccpy_einsum("mbn,ijmn->ibj", L.aa, H.aa.oooo)
    LH.aa += ccpy_einsum("iem,ejmb->ibj", L.aa, H.aa.voov)
    LH.aa += ccpy_einsum("iem,jebm->ibj", L.ab, H.ab.ovvo)
    I1 = (
        -0.5 * ccpy_einsum("mfn,efmn->e", L.aa, T.aa)
        - ccpy_einsum("mfn,efmn->e", L.ab, T.ab)
    )
    LH.aa += 0.5 * ccpy_einsum("e,ijeb->ibj", I1, H.aa.oovv)
    LH.aa -= np.transpose(LH.aa, (2, 1, 0))
    return LH

def build_LH_2B(L, LH, T, H):

    LH.ab = ccpy_einsum("i,jb->ibj", L.a, H.b.ov)
    LH.ab -= ccpy_einsum("m,ijmb->ibj", L.a, H.ab.ooov)
    LH.ab -= ccpy_einsum("ibm,jm->ibj", L.ab, H.b.oo)
    LH.ab -= ccpy_einsum("mbj,im->ibj", L.ab, H.a.oo)
    LH.ab += ccpy_einsum("iej,eb->ibj", L.ab, H.b.vv)
    LH.ab += ccpy_einsum("mbn,ijmn->ibj", L.ab, H.ab.oooo)
    LH.ab += ccpy_einsum("iem,ejmb->ibj", L.aa, H.ab.voov)
    LH.ab += ccpy_einsum("iem,ejmb->ibj", L.ab, H.bb.voov)
    LH.ab -= ccpy_einsum("mej,iemb->ibj", L.ab, H.ab.ovov)
    I1 = (
        -0.5 * ccpy_einsum("mfn,efmn->e", L.aa, T.aa)
        - ccpy_einsum("mfn,efmn->e", L.ab, T.ab)
    )
    LH.ab += ccpy_einsum("e,ijeb->ibj", I1, H.ab.oovv)
    return LH
