import numpy as np

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

    LH.a = np.einsum("e,ea->a", L.a, H.a.vv, optimize=True)
    LH.a += 0.5 * np.einsum("efn,fena->a", L.aa, H.aa.vvov, optimize=True)
    LH.a += np.einsum("efn,efan->a", L.ab, H.ab.vvvo, optimize=True)
    return LH

def build_LH_2A(L, LH, T, H):

    LH.aa = np.einsum("a,jb->abj", L.a, H.a.ov, optimize=True)
    LH.aa += 0.5 * np.einsum("e,ejab->abj", L.a, H.aa.vovv, optimize=True)
    LH.aa += np.einsum("ebj,ea->abj", L.aa, H.a.vv, optimize=True)
    LH.aa -= 0.5 * np.einsum("abm,jm->abj", L.aa, H.a.oo, optimize=True)
    LH.aa += np.einsum("afn,fjnb->abj", L.aa, H.aa.voov, optimize=True)
    LH.aa += np.einsum("afn,jfbn->abj", L.ab, H.ab.ovvo, optimize=True)
    LH.aa += 0.25 * np.einsum("efj,efab->abj", L.aa, H.aa.vvvv, optimize=True)
    I1 = (
        0.5 * np.einsum("efn,efmn->m", L.aa, T.aa, optimize=True)
        + np.einsum("efn,efmn->m", L.ab, T.ab, optimize=True)
    )
    LH.aa -= 0.5 * np.einsum("mjab,m->abj", H.aa.oovv, I1, optimize=True)
    LH.aa -= np.transpose(LH.aa, (1, 0, 2))
    return LH

def build_LH_2B(L, LH, T, H):

    LH.ab = np.einsum("a,jb->abj", L.a, H.b.ov, optimize=True)
    LH.ab += np.einsum("e,ejab->abj", L.a, H.ab.vovv, optimize=True)
    LH.ab -= np.einsum("abm,jm->abj", L.ab, H.b.oo, optimize=True)
    LH.ab += np.einsum("aej,eb->abj", L.ab, H.b.vv, optimize=True)
    LH.ab += np.einsum("ebj,ea->abj", L.ab, H.a.vv, optimize=True)
    LH.ab += np.einsum("afn,fjnb->abj", L.aa, H.ab.voov, optimize=True)
    LH.ab += np.einsum("afn,fjnb->abj", L.ab, H.bb.voov, optimize=True)
    LH.ab -= np.einsum("ebm,ejam->abj", L.ab, H.ab.vovo, optimize=True)
    LH.ab += np.einsum("efj,efab->abj", L.ab, H.ab.vvvv, optimize=True)
    I1 = (
        0.5 * np.einsum("efn,efmn->m", L.aa, T.aa, optimize=True)
        + np.einsum("efn,efmn->m", L.ab, T.ab, optimize=True)
    )
    LH.ab -= np.einsum("mjab,m->abj", H.ab.oovv, I1, optimize=True)
    return LH
