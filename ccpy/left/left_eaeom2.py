import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

from ccpy.lib.core import cc_loops2

def update_l(L, omega, H, RHF_symmetry, system):
    L.a, L.aa, L.ab = cc_loops2.update_r_2p1h(
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

    LH.a = ccpy_einsum("e,ea->a", L.a, H.a.vv)
    LH.a += 0.5 * ccpy_einsum("efn,fena->a", L.aa, H.aa.vvov)
    LH.a += ccpy_einsum("efn,efan->a", L.ab, H.ab.vvvo)
    return LH

def build_LH_2A(L, LH, T, H):

    LH.aa = ccpy_einsum("a,jb->abj", L.a, H.a.ov)
    LH.aa += 0.5 * ccpy_einsum("e,ejab->abj", L.a, H.aa.vovv)
    LH.aa += ccpy_einsum("ebj,ea->abj", L.aa, H.a.vv)
    LH.aa -= 0.5 * ccpy_einsum("abm,jm->abj", L.aa, H.a.oo)
    LH.aa += ccpy_einsum("afn,fjnb->abj", L.aa, H.aa.voov)
    LH.aa += ccpy_einsum("afn,jfbn->abj", L.ab, H.ab.ovvo)
    LH.aa += 0.25 * ccpy_einsum("efj,efab->abj", L.aa, H.aa.vvvv)
    I1 = (
        0.5 * ccpy_einsum("efn,efmn->m", L.aa, T.aa)
        + ccpy_einsum("efn,efmn->m", L.ab, T.ab)
    )
    LH.aa -= 0.5 * ccpy_einsum("mjab,m->abj", H.aa.oovv, I1)
    LH.aa -= np.transpose(LH.aa, (1, 0, 2))
    return LH

def build_LH_2B(L, LH, T, H):

    LH.ab = ccpy_einsum("a,jb->abj", L.a, H.b.ov)
    LH.ab += ccpy_einsum("e,ejab->abj", L.a, H.ab.vovv)
    LH.ab -= ccpy_einsum("abm,jm->abj", L.ab, H.b.oo)
    LH.ab += ccpy_einsum("aej,eb->abj", L.ab, H.b.vv)
    LH.ab += ccpy_einsum("ebj,ea->abj", L.ab, H.a.vv)
    LH.ab += ccpy_einsum("afn,fjnb->abj", L.aa, H.ab.voov)
    LH.ab += ccpy_einsum("afn,fjnb->abj", L.ab, H.bb.voov)
    LH.ab -= ccpy_einsum("ebm,ejam->abj", L.ab, H.ab.vovo)
    LH.ab += ccpy_einsum("efj,efab->abj", L.ab, H.ab.vvvv)
    I1 = (
        0.5 * ccpy_einsum("efn,efmn->m", L.aa, T.aa)
        + ccpy_einsum("efn,efmn->m", L.ab, T.ab)
    )
    LH.ab -= ccpy_einsum("mjab,m->abj", H.ab.oovv, I1)
    return LH
