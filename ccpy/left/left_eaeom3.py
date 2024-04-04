import numpy as np

from ccpy.utilities.updates import cc_loops2
from ccpy.left.left_eaeom_intermediates import get_lefteaeom3_intermediates

def update_l(L, omega, H, RHF_symmetry, system):
    L.a, L.aa, L.ab, L.aaa, L.aab, L.abb = cc_loops2.cc_loops2.update_r_3p2h(
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
    X = get_lefteaeom3_intermediates(L, T, system)
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
    """Calculate the projection < 0 | (L1p+L2p1h+L3p2h)*(H_N e^(T1+T2))_C | a >."""
    LH.a = np.einsum("e,ea->a", L.a, H.a.vv, optimize=True)
    LH.a += 0.5 * np.einsum("efn,fena->a", L.aa, H.aa.vvov, optimize=True)
    LH.a += np.einsum("efn,efan->a", L.ab, H.ab.vvvo, optimize=True)
    # parts contracted with L3
    LH.a -= np.einsum("fmna,mfn->a", H.aa.voov, X["aa"]["ovo"], optimize=True)
    LH.a -= np.einsum("mfna,mfn->a", H.ab.ovvo, X["ab"]["ovo"], optimize=True)
    LH.a += 0.5 * np.einsum("abe,abfe->f", X["aa"]["vvv"], H.aa.vvvv, optimize=True)
    LH.a += np.einsum("abe,abfe->f", X["ab"]["vvv"], H.ab.vvvv, optimize=True)
    return LH

def build_LH_2A(L, LH, H, X):
    """Calculate the projection < 0 | (L1p+L2p1h+L3p2h)*(H_N e^(T1+T2))_C | abj >."""
    LH.aa = np.einsum("a,jb->abj", L.a, H.a.ov, optimize=True)
    LH.aa += 0.5 * np.einsum("e,ejab->abj", L.a, H.aa.vovv, optimize=True)
    LH.aa += np.einsum("ebj,ea->abj", L.aa, H.a.vv, optimize=True)
    LH.aa -= 0.5 * np.einsum("abm,jm->abj", L.aa, H.a.oo, optimize=True)
    LH.aa += np.einsum("afn,fjnb->abj", L.aa, H.aa.voov, optimize=True)
    LH.aa += np.einsum("afn,jfbn->abj", L.ab, H.ab.ovvo, optimize=True)
    LH.aa += 0.25 * np.einsum("efj,efab->abj", L.aa, H.aa.vvvv, optimize=True)
    LH.aa -= 0.5 * np.einsum("mjab,m->abj", H.aa.oovv, X["a"]["o"], optimize=True)
    # parts contracted with L3
    LH.aa -= np.einsum("mbn,jmna->abj", X["aa"]["ovo"], H.aa.ooov, optimize=True)
    LH.aa += np.einsum("aef,fjeb->abj", X["aa"]["vvv"], H.aa.vovv, optimize=True)
    LH.aa -= np.transpose(LH.aa, (1, 0, 2))
    return LH

def build_LH_2B(L, LH, H, X):
    """Calculate the projection < 0 | (L1p+L2p1h+L3p2h)*(H_N e^(T1+T2))_C | ab~j~ >."""
    LH.ab = np.einsum("a,jb->abj", L.a, H.b.ov, optimize=True)
    LH.ab += np.einsum("e,ejab->abj", L.a, H.ab.vovv, optimize=True)
    LH.ab -= np.einsum("abm,jm->abj", L.ab, H.b.oo, optimize=True)
    LH.ab += np.einsum("aej,eb->abj", L.ab, H.b.vv, optimize=True)
    LH.ab += np.einsum("ebj,ea->abj", L.ab, H.a.vv, optimize=True)
    LH.ab += np.einsum("afn,fjnb->abj", L.aa, H.ab.voov, optimize=True)
    LH.ab += np.einsum("afn,fjnb->abj", L.ab, H.bb.voov, optimize=True)
    LH.ab -= np.einsum("ebm,ejam->abj", L.ab, H.ab.vovo, optimize=True)
    LH.ab += np.einsum("efj,efab->abj", L.ab, H.ab.vvvv, optimize=True)
    LH.ab -= np.einsum("mjab,m->abj", H.ab.oovv, X["a"]["o"], optimize=True)
    # parts contracted with L3
    return LH

def build_LH_3A(L, LH, H, X):
    """Calculate the projection < 0 | (L1p+L2p1h+L3p2h)*(H_N e^(T1+T2))_C | jkabc >."""
    # moment-like terms < 0 | (L1p+L2p1h)*(H_N e^(T1+T2))_C | jkabc >
    LH.aaa = (3.0 / 12.0) * np.einsum("a,jkbc->abcjk", L.a, H.aa.oovv, optimize=True)
    LH.aaa += (6.0 / 12.0) * np.einsum("abj,kc->abcjk", L.aa, H.a.ov, optimize=True)
    LH.aaa -= (3.0 / 12.0) * np.einsum("abm,jkmc->abcjk", L.aa, H.aa.ooov, optimize=True)
    LH.aaa += (6.0 / 12.0) * np.einsum("eck,ejab->abcjk", L.aa, H.aa.vovv, optimize=True)
    # <0|L3p2h*(H_N e^(T1+T2))_C | jkabc>
    LH.aaa -= (2.0 / 12.0) * np.einsum("jm,abcmk->abcjk", H.a.oo, L.aaa, optimize=True)
    LH.aaa += (3.0 / 12.0) * np.einsum("eb,aecjk->abcjk", H.a.vv, L.aaa, optimize=True)
    LH.aaa += (1.0 / 24.0) * np.einsum("jkmn,abcmn->abcjk", H.aa.oooo, L.aaa, optimize=True)
    LH.aaa += (3.0 / 24.0) * np.einsum("efbc,aefjk->abcjk", H.aa.vvvv, L.aaa, optimize=True)
    LH.aaa += (6.0 / 12.0) * np.einsum("ejmb,acekm->abcjk", H.aa.voov, L.aaa, optimize=True)
    LH.aaa += (6.0 / 12.0) * np.einsum("jebm,acek,->abcjk", H.ab.ovvo, L.aab, optimize=True)
    # three-body Hbar terms
    LH.aaa -= (6.0 / 12.0) * np.einsum("mck,mjab->abcjk", X["aa"]["ovo"], H.aa.oovv, optimize=True)
    LH.aaa += (3.0 / 12.0) * np.einsum("aeb,jkec->abcjk", X["aa"]["vvv"], H.aa.oovv, optimize=True)
    LH.aaa -= np.transpose(LH.aaa, (1, 0, 2, 3, 4)) + np.transpose(LH.aaa, (2, 1, 0, 3, 4)) # antisymmetrize A(a/bc)
    LH.aaa -= np.transpose(LH.aaa, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    LH.aaa -= np.transpose(LH.aaa, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return LH

def build_LH_3B(L, LH, H, X):
    """Calculate the projection < 0 | (L1p+L2p1h+L3p2h)(H_N e^(T1+T2))_C | jk~abc~ >."""
    # moment-like terms < 0 | (L1p+L2p1h)*(H_N e^(T1+T2))_C | jk~abc~ >
    LH.aab = np.einsum("a,jkbc->abcjk", L.a, H.ab.oovv, optimize=True)
    LH.aab += (1.0 / 2.0) * np.einsum("abj,kc->abcjk", L.aa, H.b.ov, optimize=True)
    LH.aab += np.einsum("ack,jb->abcjk", L.ab, H.a.ov, optimize=True)
    LH.aab -= (1.0 / 2.0) * np.einsum("abm,jkmc->abcjk", L.aa, H.ab.ooov, optimize=True)
    LH.aab -= np.einsum("acm,jkbm->abcjk", L.ab, H.ab.oovo, optimize=True)
    LH.aab += np.einsum("aej,ekbc->abcjk", L.aa, H.ab.vovv, optimize=True)
    LH.aab += np.einsum("aek,jebc->abcjk", L.ab, H.ab.ovvv, optimize=True)
    LH.aab += (1.0 / 2.0) * np.einsum("eck,ejab->abcjk", L.ab, H.aa.vovv, optimize=True)
    # < 0 | L3p2h*(H_N e^(T1+T2))_C | jk~abc~ >
    LH.aab -= (1.0 / 2.0) * np.einsum("jm,abcmk->abcjk", H.a.oo, L.aab, optimize=True) # (1)
    LH.aab -= (1.0 / 2.0) * np.einsum("km,abcjm->abcjk", H.b.oo, L.aab, optimize=True) # (2)
    LH.aab += np.einsum("ea,ebcjk->abcjk", H.a.vv, L.aab, optimize=True) # (3)
    LH.aab += (1.0 / 2.0) * np.einsum("ec,abejk->abcjk", H.b.vv, L.aab, optimize=True) # (4)
    LH.aab += (1.0 / 2.0) * np.einsum("jkmn,abcmn->abcjk", H.ab.oooo, L.aab, optimize=True) # (5)
    LH.aab += (1.0 / 4.0) * np.einsum("efab,efcjk->abcjk", H.aa.vvvv, L.aab, optimize=True) # (6)
    LH.aab += np.einsum("efbc,aefjk->abcjk", H.ab.vvvv, L.aab, optimize=True) # (7)
    LH.aab += np.einsum("ejmb,aecmk->abcjk", H.ab.voov, L.aab, optimize=True) # (8)
    LH.aab += np.einsum("jebm,aecmk->abcjk", H.ab.ovvo, L.abb, optimize=True) # (9)
    LH.aab += (1.0 / 2.0) * np.einsum("ekmc,abejm->abcjk", H.ab.voov, L.aaa, optimize=True) # (10)
    LH.aab += (1.0 / 2.0) * np.einsum("ekmc,abejm->abcjk", H.bb.voov, L.aab, optimize=True) # (11)
    LH.aab -= (1.0 / 2.0) * np.einsum("jemc,abemk->abcjk", H.ab.ovov, L.aab, optimize=True) # (12)
    LH.aab -= np.einsum("ekbm,aecjm->abcjk", H.ab.vovo, L.aab, optimize=True) # (13)
    # three-body Hbar terms
    LH.aab -= np.einsum("akm,jmbc->abcjk", X["ab"]["voo"], H.ab.oovv, optimize=True) # (1)
    LH.aab -= (1.0 / 2.0) * np.einsum("mck,mjab->abcjk", X["ab"]["ovo"], H.aa.oovv, optimize=True) # (2)
    LH.aab -= np.einsum("mbj,mkac->abcjk", X["aa"]["ovo"], H.ab.oovv, optimize=True) # (3)
    LH.aab += (1.0 / 2.0) * np.einsum("aeb,jkec->abcjk", X["aa"]["vvv"], H.ab.oovv, optimize=True) # (4)
    LH.aab += np.einsum("aec,jkbe->abcjk", X["ab"]["vvv"], H.ab.oovv, optimize=True) # (5)
    LH.aab -= np.transpose(LH.aab, (1, 0, 2, 3, 4)) # antisymmetrize A(ab)
    return LH

def build_LH_3C(L, LH, H, X):
    """Calculate the projection < 0 | (L1p+L2p1h+L3p2h)(H_N e^(T1+T2))_C | j~k~ab~c~ >."""
    # moment-like terms < 0 | (L1p+L2p1h)*(H_N e^(T1+T2))_C | j~k~ab~c~ >
    LH.abb = (1.0 / 4.0) * np.einsum("a,jkbc->abcjk", L.a, H.bb.oovv, optimize=True)
    LH.abb += np.einsum("abj,kc->abcjk", L.ab, H.b.ov, optimize=True)
    LH.abb -= (2.0 / 4.0) * np.einsum("abm,jkmc->abcjk", L.ab, H.bb.ooov, optimize=True)
    LH.abb += (2.0 / 4.0) * np.einsum("aej,ekbc->abcjk", L.ab, H.bb.vovv, optimize=True)
    LH.abb += (2.0 / 4.0) * np.einsum("eck,ejab->abcjk", L.ab, H.ab.vovv, optimize=True)
    # < 0 | L3p2h*(H_N e^(T1+T2))_C | j!k~ab!c~ >
    LH.abb -= (2.0 / 4.0) * np.einsum("jm,abcmk->abcjk", H.b.oo, L.abb, optimize=True)
    LH.abb += (2.0 / 4.0) * np.einsum("eb,aecjk->abcjk", H.b.vv, L.abb, optimize=True)
    LH.abb += (1.0 / 4.0) * np.einsum("ea,ebcjk->abcjk", H.a.vv, L.abb, optimize=True)
    LH.abb += (1.0 / 8.0) * np.einsum("jkmn,abcmn->abcjk", H.bb.oooo, L.abb, optimize=True)
    LH.abb += (2.0 / 4.0) * np.einsum("efab,efcjk->abcjk", H.ab.vvvv, L.abb, optimize=True)
    LH.abb += (1.0 / 8.0) * np.einsum("efbc,aefjk->abcjk", H.bb.vvvv, L.abb, optimize=True)
    LH.abb += np.einsum("ejmb,aecmk->abcjk", H.ab.voov, L.aab, optimize=True)
    LH.abb += np.einsum("ejmb,aecmk->abcjk", H.bb.voov, L.abb, optimize=True)
    LH.abb -= (2.0 / 4.0) * np.einsum("ejam,ebcmk->abcjk", H.ab.vovo, L.abb, optimize=True)
    # three-body Hbar terms
    LH.abb -= np.einsum("mck,mjab->abcjk", X["ab"]["ovo"], H.ab.oovv, optimize=True)
    LH.abb -= (2.0 / 4.0) * np.einsum("ajm,mkbc->abcjk", X["ab"]["voo"], H.bb.oovv, optimize=True)
    LH.abb += (2.0 / 4.0) * np.einsum("aeb,jkec->abcjk", X["ab"]["vvv"], H.bb.oovv, optimize=True)
    LH.abb -= np.transpose(LH.abb, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    LH.abb -= np.transpose(LH.abb, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return LH
