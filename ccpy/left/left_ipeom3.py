import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

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

    LH.a = -1.0 * ccpy_einsum("m,im->i", L.a, H.a.oo)
    LH.a -= 0.5 * ccpy_einsum("mfn,finm->i", L.aa, H.aa.vooo)
    LH.a -= ccpy_einsum("mfn,ifmn->i", L.ab, H.ab.ovoo)
    # parts contracted with L3 (verified against explicit 3-body hbars)
    LH.a += ccpy_einsum("ibaj,abj->i", H.ab.ovvo, X["ab"]["vvo"])
    LH.a += ccpy_einsum("bija,abj->i", H.aa.voov, X["aa"]["vvo"])
    LH.a += 0.5 * ccpy_einsum("ljk,iklj->i", X["aa"]["ooo"], H.aa.oooo)
    LH.a += ccpy_einsum("jcb,ibjc->i", X["ab"]["ovv"], H.ab.ovov)
    LH.a += ccpy_einsum("ljk,iklj->i", X["ab"]["ooo"], H.ab.oooo)
    return LH

def build_LH_2A(L, LH, H, X):

    LH.aa = ccpy_einsum("i,jb->ibj", L.a, H.a.ov)
    LH.aa -= 0.5 * ccpy_einsum("m,ijmb->ibj", L.a, H.aa.ooov)
    LH.aa += 0.5 * ccpy_einsum("iej,eb->ibj", L.aa, H.a.vv)
    LH.aa -= ccpy_einsum("ibm,jm->ibj", L.aa, H.a.oo)
    LH.aa += 0.25 * ccpy_einsum("mbn,ijmn->ibj", L.aa, H.aa.oooo)
    LH.aa += ccpy_einsum("iem,ejmb->ibj", L.aa, H.aa.voov)
    LH.aa += ccpy_einsum("iem,jebm->ibj", L.ab, H.ab.ovvo)
    LH.aa += 0.5 * ccpy_einsum("e,ijeb->ibj", X["a"]["v"], H.aa.oovv)
    # parts contracted with L3
    LH.aa -= 0.5 * ccpy_einsum("finm,mbfjn->ibj", H.aa.vooo, L.aaa)
    LH.aa -= ccpy_einsum("ifmn,mbfjn->ibj", H.ab.ovoo, L.aab)
    LH.aa += 0.25 * ccpy_einsum("fenb,iefjn->ibj", H.aa.vvov, L.aaa)
    LH.aa += 0.5 * ccpy_einsum("efbn,iefjn->ibj", H.ab.vvvo, L.aab)
    # 3-body hbar terms (verified against explicit 3-body hbars)
    LH.aa += ccpy_einsum("fej,eibf->ibj", X["aa"]["vvo"], H.aa.vovv)
    LH.aa -= 0.5 * ccpy_einsum("fbm,jimf->ibj", X["aa"]["vvo"], H.aa.ooov)
    LH.aa -= ccpy_einsum("imn,njmb->ibj", X["aa"]["ooo"], H.aa.ooov)
    LH.aa -= ccpy_einsum("imn,jnbm->ibj", X["ab"]["ooo"], H.ab.oovo)
    LH.aa -= ccpy_einsum("ife,jebf->ibj", X["ab"]["ovv"], H.ab.ovvv)
    LH.aa -= np.transpose(LH.aa, (2, 1, 0))
    return LH

def build_LH_2B(L, LH, H, X):

    LH.ab = ccpy_einsum("i,jb->ibj", L.a, H.b.ov)
    LH.ab -= ccpy_einsum("m,ijmb->ibj", L.a, H.ab.ooov)
    LH.ab -= ccpy_einsum("ibm,jm->ibj", L.ab, H.b.oo)
    LH.ab -= ccpy_einsum("mbj,im->ibj", L.ab, H.a.oo)
    LH.ab += ccpy_einsum("iej,eb->ibj", L.ab, H.b.vv)
    LH.ab += ccpy_einsum("mbn,ijmn->ibj", L.ab, H.ab.oooo)
    LH.ab += ccpy_einsum("iem,ejmb->ibj", L.aa, H.ab.voov)
    LH.ab += ccpy_einsum("iem,ejmb->ibj", L.ab, H.bb.voov)
    LH.ab -= ccpy_einsum("mej,iemb->ibj", L.ab, H.ab.ovov)
    LH.ab += ccpy_einsum("e,ijeb->ibj", X["a"]["v"], H.ab.oovv)
    # parts contracted with L3
    LH.ab -= 0.5 * ccpy_einsum("mfbnj,finm->ibj", L.aab, H.aa.vooo)
    LH.ab -= ccpy_einsum("mfbnj,ifmn->ibj", L.abb, H.ab.ovoo)
    LH.ab += ccpy_einsum("ifenj,fenb->ibj", L.aab, H.ab.vvov)
    LH.ab += 0.5 * ccpy_einsum("ifenj,fenb->ibj", L.abb, H.bb.vvov)
    LH.ab -= ccpy_einsum("ifbnm,fjnm->ibj", L.aab, H.ab.vooo)
    LH.ab -= 0.5 * ccpy_einsum("ifbnm,fjnm->ibj", L.abb, H.bb.vooo)
    # 3-body hbar terms
    LH.ab += ccpy_einsum("fei,ejfb->ibj", X["aa"]["vvo"], H.ab.vovv)
    LH.ab -= ccpy_einsum("ife,ejfb->ibj", X["ab"]["ovv"], H.bb.vovv)
    LH.ab -= ccpy_einsum("ebm,ijem->ibj", X["ab"]["vvo"], H.ab.oovo)
    LH.ab += ccpy_einsum("fej,iefb->ibj", X["ab"]["vvo"], H.ab.ovvv)
    LH.ab -= ccpy_einsum("imn,njmb->ibj", X["aa"]["ooo"], H.ab.ooov)
    LH.ab -= ccpy_einsum("imn,njmb->ibj", X["ab"]["ooo"], H.bb.ooov)
    LH.ab += ccpy_einsum("njm,imnb->ibj", X["ab"]["ooo"], H.ab.ooov)
    LH.ab -= ccpy_einsum("meb,ijme->ibj", X["ab"]["ovv"], H.ab.ooov)
    return LH

def build_LH_3A(L, LH, H, X):
    """Calculate the projection < 0 | (L1h+L2h1p+L3h2p)*(H_N e^(T1+T2))_C | ijkbc >."""
    # moment-like terms
    LH.aaa = (3.0 / 12.0) * ccpy_einsum("i,jkbc->ibcjk", L.a, H.aa.oovv)
    LH.aaa += (6.0 / 12.0) * ccpy_einsum("ibj,kc->ibcjk", L.aa, H.a.ov)
    LH.aaa += (3.0 / 12.0) * ccpy_einsum("iej,ekbc->ibcjk", L.aa, H.aa.vovv)
    LH.aaa -= (6.0 / 12.0) * ccpy_einsum("mck,ijmb->ibcjk", L.aa, H.aa.ooov)
    #
    LH.aaa -= (3.0 / 12.0) * ccpy_einsum("im,mbcjk->ibcjk", H.a.oo, L.aaa)
    LH.aaa += (2.0 / 12.0) * ccpy_einsum("eb,iecjk->ibcjk", H.a.vv, L.aaa)
    LH.aaa += (3.0 / 24.0) * ccpy_einsum("jkmn,ibcmn->ibcjk", H.aa.oooo, L.aaa)
    LH.aaa += (1.0 / 24.0) * ccpy_einsum("efbc,iefjk->ibcjk", H.aa.vvvv, L.aaa)
    LH.aaa += (6.0 / 12.0) * ccpy_einsum("ekmc,ibejm->ibcjk", H.aa.voov, L.aaa)
    LH.aaa += (6.0 / 12.0) * ccpy_einsum("kecm,ibejm->ibcjk", H.ab.ovvo, L.aab)
    # 3-body hbar terms
    LH.aaa += (6.0 / 12.0) * ccpy_einsum("eck,ijeb->ibcjk", X["aa"]["vvo"], H.aa.oovv)
    LH.aaa -= (3.0 / 12.0) * ccpy_einsum("ikm,mjcb->ibcjk", X["aa"]["ooo"], H.aa.oovv)
    #
    LH.aaa -= np.transpose(LH.aaa, (3, 1, 2, 0, 4)) + np.transpose(LH.aaa, (4, 1, 2, 3, 0)) # antisymmetrize A(i/jk)
    LH.aaa -= np.transpose(LH.aaa, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    LH.aaa -= np.transpose(LH.aaa, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    return LH

def build_LH_3B(L, LH, H, X):
    """Calculate the projection < 0 | (L1h+L2h1p+L3h2p)(H_N e^(T1+T2))_C | ijk~bc~ >."""
    # moment-like terms
    LH.aab = ccpy_einsum("i,jkbc->ibcjk", L.a, H.ab.oovv)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("ibj,kc->ibcjk", L.aa, H.b.ov)
    LH.aab += ccpy_einsum("ick,jb->ibcjk", L.ab, H.a.ov)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("iej,ekbc->ibcjk", L.aa, H.ab.vovv)
    LH.aab += ccpy_einsum("iek,jebc->ibcjk", L.ab, H.ab.ovvv)
    LH.aab -= ccpy_einsum("mbj,ikmc->ibcjk", L.aa, H.ab.ooov)
    LH.aab -= (1.0 / 2.0) * ccpy_einsum("mck,ijmb->ibcjk", L.ab, H.aa.ooov)
    LH.aab -= ccpy_einsum("icm,jkbm->ibcjk", L.ab, H.ab.oovo)
    #
    LH.aab -= ccpy_einsum("im,mbcjk->ibcjk", H.a.oo, L.aab)
    LH.aab -= (1.0 / 2.0) * ccpy_einsum("km,ibcjm->ibcjk", H.b.oo, L.aab)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("eb,iecjk->ibcjk", H.a.vv, L.aab)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("ec,ibejk->ibcjk", H.b.vv, L.aab)
    LH.aab += (1.0 / 4.0) * ccpy_einsum("ijmn,mbcnk->ibcjk", H.aa.oooo, L.aab)
    LH.aab += ccpy_einsum("jkmn,ibcmn->ibcjk", H.ab.oooo, L.aab)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("efbc,iefjk->ibcjk", H.ab.vvvv, L.aab)
    LH.aab += ccpy_einsum("ejmb,iecmk->ibcjk", H.aa.voov, L.aab)
    LH.aab += ccpy_einsum("jebm,iecmk->ibcjk", H.ab.ovvo, L.abb)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("ekmc,ibejm->ibcjk", H.ab.voov, L.aaa)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("ekmc,ibejm->ibcjk", H.bb.voov, L.aab)
    LH.aab -= ccpy_einsum("jemc,ibemk->ibcjk", H.ab.ovov, L.aab)
    LH.aab -= (1.0 / 2.0) * ccpy_einsum("ekbm,iecjm->ibcjk", H.ab.vovo, L.aab)
    # 3-body hbar terms
    LH.aab -= (1.0 / 2.0) * ccpy_einsum("ijm,mkbc->ibcjk", X["aa"]["ooo"], H.ab.oovv)
    LH.aab -= ccpy_einsum("ikm,jmbc->ibcjk", X["ab"]["ooo"], H.ab.oovv)
    LH.aab += ccpy_einsum("ebj,ikec->ibcjk", X["aa"]["vvo"], H.ab.oovv)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("eck,ijeb->ibcjk", X["ab"]["vvo"], H.aa.oovv)
    LH.aab += ccpy_einsum("iec,jkbe->ibcjk", X["ab"]["ovv"], H.ab.oovv)
    #
    LH.aab -= np.transpose(LH.aab, (3, 1, 2, 0, 4)) # antisymmetrize (ij)
    return LH

def build_LH_3C(L, LH, H, X):
    """Calculate the projection < 0 | (L1h+L2h1p+L3h2p)(H_N e^(T1+T2))_C | ij~k~b~c~ >."""
    # moment-like terms
    LH.abb = (1.0 / 4.0) * ccpy_einsum("i,jkbc->ibcjk", L.a, H.bb.oovv)
    LH.abb += ccpy_einsum("ibj,kc->ibcjk", L.ab, H.b.ov)
    LH.abb += (2.0 / 4.0) * ccpy_einsum("iej,ekbc->ibcjk", L.ab, H.bb.vovv)
    LH.abb -= ccpy_einsum("mck,ijmb->ibcjk", L.ab, H.ab.ooov)
    LH.abb -= (2.0 / 4.0) * ccpy_einsum("ibm,jkmc->ibcjk", L.ab, H.bb.ooov)
    #
    LH.abb -= (1.0 / 4.0) * ccpy_einsum("im,mbcjk->ibcjk", H.a.oo, L.abb)
    LH.abb -= (2.0 / 4.0) * ccpy_einsum("jm,ibcmk->ibcjk", H.b.oo, L.abb)
    LH.abb += (2.0 / 4.0) * ccpy_einsum("eb,iecjk->ibcjk", H.b.vv, L.abb)
    LH.abb += (1.0 / 8.0) * ccpy_einsum("jkmn,ibcmn->ibcjk", H.bb.oooo, L.abb)
    LH.abb += (2.0 / 4.0) * ccpy_einsum("ijmn,mbcnk->ibcjk", H.ab.oooo, L.abb)
    LH.abb += (1.0 / 8.0) * ccpy_einsum("efbc,iefjk->ibcjk", H.bb.vvvv, L.abb)
    LH.abb += ccpy_einsum("ejmb,iecmk->ibcjk", H.ab.voov, L.aab)
    LH.abb += ccpy_einsum("ejmb,iecmk->ibcjk", H.bb.voov, L.abb)
    LH.abb -= (2.0 / 4.0) * ccpy_einsum("iemb,mecjk->ibcjk", H.ab.ovov, L.abb)
    # 3-body hbar terms
    LH.abb -= (2.0 / 4.0) * ccpy_einsum("ijm,mkbc->ibcjk", X["ab"]["ooo"], H.bb.oovv)
    LH.abb += ccpy_einsum("eck,ijeb->ibcjk", X["ab"]["vvo"], H.ab.oovv)
    LH.abb += (2.0 / 4.0) * ccpy_einsum("iec,jkbe->ibcjk", X["ab"]["ovv"], H.bb.oovv)
    #
    LH.abb -= np.transpose(LH.abb, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    LH.abb -= np.transpose(LH.abb, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return LH
