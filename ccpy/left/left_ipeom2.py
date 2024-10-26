import numpy as np

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

    LH.a = -1.0 * np.einsum("m,im->i", L.a, H.a.oo, optimize=True)
    LH.a -= 0.5 * np.einsum("mfn,finm->i", L.aa, H.aa.vooo, optimize=True)
    LH.a -= np.einsum("mfn,ifmn->i", L.ab, H.ab.ovoo, optimize=True)
    return LH

def build_LH_2A(L, LH, T, H):

    LH.aa = np.einsum("i,jb->ibj", L.a, H.a.ov, optimize=True)
    LH.aa -= 0.5 * np.einsum("m,ijmb->ibj", L.a, H.aa.ooov, optimize=True)
    LH.aa += 0.5 * np.einsum("iej,eb->ibj", L.aa, H.a.vv, optimize=True)
    LH.aa -= np.einsum("ibm,jm->ibj", L.aa, H.a.oo, optimize=True)
    LH.aa += 0.25 * np.einsum("mbn,ijmn->ibj", L.aa, H.aa.oooo, optimize=True)
    LH.aa += np.einsum("iem,ejmb->ibj", L.aa, H.aa.voov, optimize=True)
    LH.aa += np.einsum("iem,jebm->ibj", L.ab, H.ab.ovvo, optimize=True)
    I1 = (
        -0.5 * np.einsum("mfn,efmn->e", L.aa, T.aa, optimize=True)
        - np.einsum("mfn,efmn->e", L.ab, T.ab, optimize=True)
    )
    LH.aa += 0.5 * np.einsum("e,ijeb->ibj", I1, H.aa.oovv, optimize=True)
    LH.aa -= np.transpose(LH.aa, (2, 1, 0))
    return LH

def build_LH_2B(L, LH, T, H):

    LH.ab = np.einsum("i,jb->ibj", L.a, H.b.ov, optimize=True)
    LH.ab -= np.einsum("m,ijmb->ibj", L.a, H.ab.ooov, optimize=True)
    LH.ab -= np.einsum("ibm,jm->ibj", L.ab, H.b.oo, optimize=True)
    LH.ab -= np.einsum("mbj,im->ibj", L.ab, H.a.oo, optimize=True)
    LH.ab += np.einsum("iej,eb->ibj", L.ab, H.b.vv, optimize=True)
    LH.ab += np.einsum("mbn,ijmn->ibj", L.ab, H.ab.oooo, optimize=True)
    LH.ab += np.einsum("iem,ejmb->ibj", L.aa, H.ab.voov, optimize=True)
    LH.ab += np.einsum("iem,ejmb->ibj", L.ab, H.bb.voov, optimize=True)
    LH.ab -= np.einsum("mej,iemb->ibj", L.ab, H.ab.ovov, optimize=True)
    I1 = (
        -0.5 * np.einsum("mfn,efmn->e", L.aa, T.aa, optimize=True)
        - np.einsum("mfn,efmn->e", L.ab, T.ab, optimize=True)
    )
    LH.ab += np.einsum("e,ijeb->ibj", I1, H.ab.oovv, optimize=True)
    return LH
