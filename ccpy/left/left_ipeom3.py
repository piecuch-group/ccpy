import numpy as np

from ccpy.lib.core import cc_loops2
from ccpy.left.left_ipeom_intermediates import get_leftipeom3_intermediates

def update_l(L, omega, H, RHF_symmetry, system):
    L.a, L.aa, L.ab, L.aaa, L.aab, L.abb = cc_loops2.update_r_3h2p(
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

def build_LH_1A(L, LH, H, X):

    LH.a = -1.0 * np.einsum("m,im->i", L.a, H.a.oo, optimize=True)
    LH.a -= 0.5 * np.einsum("mfn,finm->i", L.aa, H.aa.vooo, optimize=True)
    LH.a -= np.einsum("mfn,ifmn->i", L.ab, H.ab.ovoo, optimize=True)
    # parts contracted with L3 (verified against explicit 3-body hbars)
    LH.a += np.einsum("ibaj,abj->i", H.ab.ovvo, X["ab"]["vvo"], optimize=True)
    LH.a += np.einsum("bija,abj->i", H.aa.voov, X["aa"]["vvo"], optimize=True)
    LH.a += 0.5 * np.einsum("ljk,iklj->i", X["aa"]["ooo"], H.aa.oooo, optimize=True)
    LH.a += np.einsum("jcb,ibjc->i", X["ab"]["ovv"], H.ab.ovov, optimize=True)
    LH.a += np.einsum("ljk,iklj->i", X["ab"]["ooo"], H.ab.oooo, optimize=True)
    return LH

def build_LH_2A(L, LH, H, X):

    LH.aa = np.einsum("i,jb->ibj", L.a, H.a.ov, optimize=True)
    LH.aa -= 0.5 * np.einsum("m,ijmb->ibj", L.a, H.aa.ooov, optimize=True)
    LH.aa += 0.5 * np.einsum("iej,eb->ibj", L.aa, H.a.vv, optimize=True)
    LH.aa -= np.einsum("ibm,jm->ibj", L.aa, H.a.oo, optimize=True)
    LH.aa += 0.25 * np.einsum("mbn,ijmn->ibj", L.aa, H.aa.oooo, optimize=True)
    LH.aa += np.einsum("iem,ejmb->ibj", L.aa, H.aa.voov, optimize=True)
    LH.aa += np.einsum("iem,jebm->ibj", L.ab, H.ab.ovvo, optimize=True)
    LH.aa += 0.5 * np.einsum("e,ijeb->ibj", X["a"]["v"], H.aa.oovv, optimize=True)
    # parts contracted with L3
    LH.aa -= 0.5 * np.einsum("finm,mbfjn->ibj", H.aa.vooo, L.aaa, optimize=True)
    LH.aa -= np.einsum("ifmn,mbfjn->ibj", H.ab.ovoo, L.aab, optimize=True)
    LH.aa += 0.25 * np.einsum("fenb,iefjn->ibj", H.aa.vvov, L.aaa, optimize=True)
    LH.aa += 0.5 * np.einsum("efbn,iefjn->ibj", H.ab.vvvo, L.aab, optimize=True)
    # 3-body hbar terms (verified against explicit 3-body hbars)
    LH.aa += np.einsum("fej,eibf->ibj", X["aa"]["vvo"], H.aa.vovv, optimize=True)
    LH.aa -= 0.5 * np.einsum("fbm,jimf->ibj", X["aa"]["vvo"], H.aa.ooov, optimize=True)
    LH.aa -= np.einsum("imn,njmb->ibj", X["aa"]["ooo"], H.aa.ooov, optimize=True)
    LH.aa -= np.einsum("imn,jnbm->ibj", X["ab"]["ooo"], H.ab.oovo, optimize=True)
    LH.aa -= np.einsum("ife,jebf->ibj", X["ab"]["ovv"], H.ab.ovvv, optimize=True)
    LH.aa -= np.transpose(LH.aa, (2, 1, 0))
    return LH

def build_LH_2B(L, LH, H, X):

    LH.ab = np.einsum("i,jb->ibj", L.a, H.b.ov, optimize=True)
    LH.ab -= np.einsum("m,ijmb->ibj", L.a, H.ab.ooov, optimize=True)
    LH.ab -= np.einsum("ibm,jm->ibj", L.ab, H.b.oo, optimize=True)
    LH.ab -= np.einsum("mbj,im->ibj", L.ab, H.a.oo, optimize=True)
    LH.ab += np.einsum("iej,eb->ibj", L.ab, H.b.vv, optimize=True)
    LH.ab += np.einsum("mbn,ijmn->ibj", L.ab, H.ab.oooo, optimize=True)
    LH.ab += np.einsum("iem,ejmb->ibj", L.aa, H.ab.voov, optimize=True)
    LH.ab += np.einsum("iem,ejmb->ibj", L.ab, H.bb.voov, optimize=True)
    LH.ab -= np.einsum("mej,iemb->ibj", L.ab, H.ab.ovov, optimize=True)
    LH.ab += np.einsum("e,ijeb->ibj", X["a"]["v"], H.ab.oovv, optimize=True)
    # parts contracted with L3
    LH.ab -= 0.5 * np.einsum("mfbnj,finm->ibj", L.aab, H.aa.vooo, optimize=True)
    LH.ab -= np.einsum("mfbnj,ifmn->ibj", L.abb, H.ab.ovoo, optimize=True)
    LH.ab += np.einsum("ifenj,fenb->ibj", L.aab, H.ab.vvov, optimize=True)
    LH.ab += 0.5 * np.einsum("ifenj,fenb->ibj", L.abb, H.bb.vvov, optimize=True)
    LH.ab -= np.einsum("ifbnm,fjnm->ibj", L.aab, H.ab.vooo, optimize=True)
    LH.ab -= 0.5 * np.einsum("ifbnm,fjnm->ibj", L.abb, H.bb.vooo, optimize=True)
    # 3-body hbar terms
    LH.ab += np.einsum("fei,ejfb->ibj", X["aa"]["vvo"], H.ab.vovv, optimize=True)
    LH.ab -= np.einsum("ife,ejfb->ibj", X["ab"]["ovv"], H.bb.vovv, optimize=True)
    LH.ab -= np.einsum("ebm,ijem->ibj", X["ab"]["vvo"], H.ab.oovo, optimize=True)
    LH.ab += np.einsum("fej,iefb->ibj", X["ab"]["vvo"], H.ab.ovvv, optimize=True)
    LH.ab -= np.einsum("imn,njmb->ibj", X["aa"]["ooo"], H.ab.ooov, optimize=True)
    LH.ab -= np.einsum("imn,njmb->ibj", X["ab"]["ooo"], H.bb.ooov, optimize=True)
    LH.ab += np.einsum("njm,imnb->ibj", X["ab"]["ooo"], H.ab.ooov, optimize=True)
    LH.ab -= np.einsum("meb,ijme->ibj", X["ab"]["ovv"], H.ab.ooov, optimize=True)
    return LH

def build_LH_3A(L, LH, H, X):
    """Calculate the projection < 0 | (L1h+L2h1p+L3h2p)*(H_N e^(T1+T2))_C | ijkbc >."""
    # moment-like terms
    LH.aaa = (3.0 / 12.0) * np.einsum("i,jkbc->ibcjk", L.a, H.aa.oovv, optimize=True)
    LH.aaa += (6.0 / 12.0) * np.einsum("ibj,kc->ibcjk", L.aa, H.a.ov, optimize=True)
    LH.aaa += (3.0 / 12.0) * np.einsum("iej,ekbc->ibcjk", L.aa, H.aa.vovv, optimize=True)
    LH.aaa -= (6.0 / 12.0) * np.einsum("mck,ijmb->ibcjk", L.aa, H.aa.ooov, optimize=True)
    #
    LH.aaa -= (3.0 / 12.0) * np.einsum("im,mbcjk->ibcjk", H.a.oo, L.aaa, optimize=True)
    LH.aaa += (2.0 / 12.0) * np.einsum("eb,iecjk->ibcjk", H.a.vv, L.aaa, optimize=True)
    LH.aaa += (3.0 / 24.0) * np.einsum("jkmn,ibcmn->ibcjk", H.aa.oooo, L.aaa, optimize=True)
    LH.aaa += (1.0 / 24.0) * np.einsum("efbc,iefjk->ibcjk", H.aa.vvvv, L.aaa, optimize=True)
    LH.aaa += (6.0 / 12.0) * np.einsum("ekmc,ibejm->ibcjk", H.aa.voov, L.aaa, optimize=True)
    LH.aaa += (6.0 / 12.0) * np.einsum("kecm,ibejm->ibcjk", H.ab.ovvo, L.aab, optimize=True)
    # 3-body hbar terms
    LH.aaa += (6.0 / 12.0) * np.einsum("eck,ijeb->ibcjk", X["aa"]["vvo"], H.aa.oovv, optimize=True)
    LH.aaa -= (3.0 / 12.0) * np.einsum("ikm,mjcb->ibcjk", X["aa"]["ooo"], H.aa.oovv, optimize=True)
    #
    LH.aaa -= np.transpose(LH.aaa, (3, 1, 2, 0, 4)) + np.transpose(LH.aaa, (4, 1, 2, 3, 0)) # antisymmetrize A(i/jk)
    LH.aaa -= np.transpose(LH.aaa, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    LH.aaa -= np.transpose(LH.aaa, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    return LH

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
    LH.aab -= np.einsum("im,mbcjk->ibcjk", H.a.oo, L.aab, optimize=True)
    LH.aab -= (1.0 / 2.0) * np.einsum("km,ibcjm->ibcjk", H.b.oo, L.aab, optimize=True)
    LH.aab += (1.0 / 2.0) * np.einsum("eb,iecjk->ibcjk", H.a.vv, L.aab, optimize=True)
    LH.aab += (1.0 / 2.0) * np.einsum("ec,ibejk->ibcjk", H.b.vv, L.aab, optimize=True)
    LH.aab += (1.0 / 4.0) * np.einsum("ijmn,mbcnk->ibcjk", H.aa.oooo, L.aab, optimize=True)
    LH.aab += np.einsum("jkmn,ibcmn->ibcjk", H.ab.oooo, L.aab, optimize=True)
    LH.aab += (1.0 / 2.0) * np.einsum("efbc,iefjk->ibcjk", H.ab.vvvv, L.aab, optimize=True)
    LH.aab += np.einsum("ejmb,iecmk->ibcjk", H.aa.voov, L.aab, optimize=True)
    LH.aab += np.einsum("jebm,iecmk->ibcjk", H.ab.ovvo, L.abb, optimize=True)
    LH.aab += (1.0 / 2.0) * np.einsum("ekmc,ibejm->ibcjk", H.ab.voov, L.aaa, optimize=True)
    LH.aab += (1.0 / 2.0) * np.einsum("ekmc,ibejm->ibcjk", H.bb.voov, L.aab, optimize=True)
    LH.aab -= np.einsum("jemc,ibemk->ibcjk", H.ab.ovov, L.aab, optimize=True)
    LH.aab -= (1.0 / 2.0) * np.einsum("ekbm,iecjm->ibcjk", H.ab.vovo, L.aab, optimize=True)
    # 3-body hbar terms
    LH.aab -= (1.0 / 2.0) * np.einsum("ijm,mkbc->ibcjk", X["aa"]["ooo"], H.ab.oovv, optimize=True)
    LH.aab -= np.einsum("ikm,jmbc->ibcjk", X["ab"]["ooo"], H.ab.oovv, optimize=True)
    LH.aab += np.einsum("ebj,ikec->ibcjk", X["aa"]["vvo"], H.ab.oovv, optimize=True)
    LH.aab += (1.0 / 2.0) * np.einsum("eck,ijeb->ibcjk", X["ab"]["vvo"], H.aa.oovv, optimize=True)
    LH.aab += np.einsum("iec,jkbe->ibcjk", X["ab"]["ovv"], H.ab.oovv, optimize=True)
    #
    LH.aab -= np.transpose(LH.aab, (3, 1, 2, 0, 4)) # antisymmetrize (ij)
    return LH

def build_LH_3C(L, LH, H, X):
    """Calculate the projection < 0 | (L1h+L2h1p+L3h2p)(H_N e^(T1+T2))_C | ij~k~b~c~ >."""
    # moment-like terms
    LH.abb = (1.0 / 4.0) * np.einsum("i,jkbc->ibcjk", L.a, H.bb.oovv, optimize=True)
    LH.abb += np.einsum("ibj,kc->ibcjk", L.ab, H.b.ov, optimize=True)
    LH.abb += (2.0 / 4.0) * np.einsum("iej,ekbc->ibcjk", L.ab, H.bb.vovv, optimize=True)
    LH.abb -= np.einsum("mck,ijmb->ibcjk", L.ab, H.ab.ooov, optimize=True)
    LH.abb -= (2.0 / 4.0) * np.einsum("ibm,jkmc->ibcjk", L.ab, H.bb.ooov, optimize=True)
    #
    LH.abb -= (1.0 / 4.0) * np.einsum("im,mbcjk->ibcjk", H.a.oo, L.abb, optimize=True)
    LH.abb -= (2.0 / 4.0) * np.einsum("jm,ibcmk->ibcjk", H.b.oo, L.abb, optimize=True)
    LH.abb += (2.0 / 4.0) * np.einsum("eb,iecjk->ibcjk", H.b.vv, L.abb, optimize=True)
    LH.abb += (1.0 / 8.0) * np.einsum("jkmn,ibcmn->ibcjk", H.bb.oooo, L.abb, optimize=True)
    LH.abb += (2.0 / 4.0) * np.einsum("ijmn,mbcnk->ibcjk", H.ab.oooo, L.abb, optimize=True)
    LH.abb += (1.0 / 8.0) * np.einsum("efbc,iefjk->ibcjk", H.bb.vvvv, L.abb, optimize=True)
    LH.abb += np.einsum("ejmb,iecmk->ibcjk", H.ab.voov, L.aab, optimize=True)
    LH.abb += np.einsum("ejmb,iecmk->ibcjk", H.bb.voov, L.abb, optimize=True)
    LH.abb -= (2.0 / 4.0) * np.einsum("iemb,mecjk->ibcjk", H.ab.ovov, L.abb, optimize=True)
    # 3-body hbar terms
    LH.abb -= (2.0 / 4.0) * np.einsum("ijm,mkbc->ibcjk", X["ab"]["ooo"], H.bb.oovv, optimize=True)
    LH.abb += np.einsum("eck,ijeb->ibcjk", X["ab"]["vvo"], H.ab.oovv, optimize=True)
    LH.abb += (2.0 / 4.0) * np.einsum("iec,jkbe->ibcjk", X["ab"]["ovv"], H.bb.oovv, optimize=True)
    #
    LH.abb -= np.transpose(LH.abb, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    LH.abb -= np.transpose(LH.abb, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return LH
