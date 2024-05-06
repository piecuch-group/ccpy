import numpy as np

from ccpy.utilities.updates import cc_loops2
from ccpy.left.left_ipeom_intermediates import get_leftipeom3_intermediates

def update_l(L, omega, H, RHF_symmetry, system):
    L.a, L.aa, L.ab, L.aaa, L.aab, L.abb = cc_loops2.cc_loops2.update_r_3h2p(
            L.a,
            L.aa,
            L.ab,
            L.aaa,
            L.aab,
            L.abb,
            omega,
            H.a.oo,
            H.a.vv,
            H.b.oo,
            H.b.vv,
            0.0
    )
    return L

def LH_fun(LH, L, T, H, flag_RHF, system):

    # get LT intermediates
    X = get_leftipeom3_intermediates(L, T, system)
    # build L1
    LH = build_LH_1A(L, LH, H, X)
    # build L2
    LH = build_LH_2A(L, LH, H, X)
    LH = build_LH_2B(L, LH, H, X)
    # build L3
    LH = build_LH_3A(L, LH, H, X)
    LH = build_LH_3B(L, LH, H, X)
    LH = build_LH_3C(L, LH, H, X)
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

def build_LH_3A(L, LH, H, X):
    """Calculate the projection < 0 | (L1h+L2h1p+L3h2p)*(H_N e^(T1+T2))_C | ijkbc >."""
    # moment-like terms
    LH.aaa = (3.0 / 12.0) * np.einsum("i,jkbc->ibcjk", L.a, H.aa.oovv, optimize=True)
    LH.aaa += (6.0 / 12.0) * np.einsum("ibj,kc->ibcjk", L.aa, H.a.ov, optimize=True)
    LH.aaa += (3.0 / 12.0) * np.einsum("iej,ekbc->ibcjk", L.aa, H.aa.vovv, optimize=True)
    LH.aaa -= (6.0 / 12.0) * np.einsum("mck,ijmb->ibcjk", L.aa, H.aa.ooov, optimize=True)
    #
    LH.aaa -= np.transpose(LH.aaa, (3, 1, 2, 0, 4)) + np.transpose(LH.aaa, (4, 1, 2, 3, 0)) # antisymmetrize A(i/jk)
    LH.aaa -= np.transpose(LH.aaa, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    LH.aaa -= np.transpose(LH.aaa, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    return LH.aaa

def build_LH_3B(L, LH, H, X):
    """Calculate the projection < 0 | (L1h+L2h1p+L3h2p)(H_N e^(T1+T2))_C | ijk~bc~ >."""
    # moment-like terms
    LH.aab = np.einsum("i,jkbc->ibcjk", L.a, H.ab.oovv, optimize=True)
    LH.aab += (1.0 / 2.0) * np.einsum("ibj,kc->ibcjk", L.aa, H.b.ov, optimize=True)
    LH.aab += np.einsum("ick,jb->ibcjk", L.ab, H.a.ov, optimize=True)
    LH.aab += (1.0 / 2.0) * np.einsum("iej,ekbc->ibcjk", L.aa, H.ab.vovv, optimize=True)
    LH.aab += np.einsum("iek,jebc->ibcjk", L.ab, H.ab.ovvv, optimize=True)
    LH.aab -= np.einsum("mbj,ikmc->ibcjk", L.aa, H.ab.ooov, optimize=True)
    LH.aab -= (1.0 / 2.0) * np.einsum("mck,ijmb->ibcjk", L.ab, H.aa.ooov, optimize=True)
    LH.aab -= np.einsum("icm,jkbm->ibcjk", L.ab, H.ab.oovo, optimize=True)
    #
    LH.aab -= np.transpose(LH.aab, (3, 1, 2, 0, 4)) # antisymmetrize (ij)
    return LH.aab

def build_LH_3C(L, LH, H, X):
    """Calculate the projection < 0 | (L1h+L2h1p+L3h2p)(H_N e^(T1+T2))_C | ij~k~b~c~ >."""
    # moment-like terms
    LH.abb = (1.0 / 4.0) * np.einsum("i,jkbc->ibcjk", L.a, H.bb.oovv, optimize=True)
    LH.abb += np.einsum("ibj,kc->ibcjk", L.ab, H.b.ov, optimize=True)
    LH.abb += (2.0 / 4.0) * np.einsum("iej,ekbc->ibcjk", L.ab, H.bb.vovv, optimize=True)
    LH.abb -= np.einsum("mck,ijmb->ibcjk", L.ab, H.ab.ooov, optimize=True)
    LH.abb -= (2.0 / 4.0) * np.einsum("ibm,jkmc->ibcjk", L.ab, H.bb.ooov, optimize=True)
    #
    LH.abb -= np.transpose(LH.abb, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    LH.abb -= np.transpose(LH.abb, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return LH.abb
