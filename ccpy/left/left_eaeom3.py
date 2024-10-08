import numpy as np

from ccpy.lib.core import cc_loops2
from ccpy.left.left_eaeom_intermediates import get_lefteaeom3_intermediates

def update_l(L, omega, H, RHF_symmetry, system):
    L.a, L.aa, L.ab, L.aaa, L.aab, L.abb = cc_loops2.update_r_3p2h(
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
    # parts contracted with L3 (verified against explicit 3-body hbars)
    LH.a -= np.einsum("mfan,mfn->a", H.ab.ovvo, X["ab"]["ovo"], optimize=True)
    LH.a -= np.einsum("fmna,mfn->a", H.aa.voov, X["aa"]["ovo"], optimize=True)
    LH.a -= 0.5 * np.einsum("fge,feag->a", X["aa"]["vvv"], H.aa.vvvv, optimize=True)
    LH.a -= np.einsum("eman,enm->a", H.ab.vovo, X["ab"]["voo"], optimize=True)
    LH.a -= np.einsum("efg,egaf->a", X["ab"]["vvv"], H.ab.vvvv, optimize=True)
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
    LH.aa += 0.5 * np.einsum("fena,ebfjn->abj", H.aa.vvov, L.aaa, optimize=True)
    LH.aa += np.einsum("efan,ebfjn->abj", H.ab.vvvo, L.aab, optimize=True)
    LH.aa -= 0.25 * np.einsum("fjnm,abfmn->abj", H.aa.vooo, L.aaa, optimize=True)
    LH.aa -= 0.5 * np.einsum("jfmn,abfmn->abj", H.ab.ovoo, L.aab, optimize=True)
    # 3-body hbar terms (verified against explicit 3-body hbars)
    LH.aa += np.einsum("mbn,jmna->abj", X["aa"]["ovo"], H.aa.ooov, optimize=True) #
    LH.aa -= np.einsum("amn,jnbm->abj", X["ab"]["voo"], H.ab.oovo, optimize=True) #
    LH.aa -= np.einsum("aef,fjeb->abj", X["aa"]["vvv"], H.aa.vovv, optimize=True) #
    LH.aa -= np.einsum("aef,jfbe->abj", X["ab"]["vvv"], H.ab.ovvv, optimize=True) #
    LH.aa -= 0.5 * np.einsum("mej,emba->abj", X["aa"]["ovo"], H.aa.vovv, optimize=True)
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
    LH.ab += 0.5 * np.einsum("fena,efbnj->abj", H.aa.vvov, L.aab, optimize=True)
    LH.ab += np.einsum("efan,efbnj->abj", H.ab.vvvo, L.abb, optimize=True)
    LH.ab += np.einsum("fenb,afenj->abj", H.ab.vvov, L.aab, optimize=True)
    LH.ab += 0.5 * np.einsum("fenb,afenj->abj", H.bb.vvov, L.abb, optimize=True)
    LH.ab -= np.einsum("fjnm,afbnm->abj", H.ab.vooo, L.aab, optimize=True)
    LH.ab -= 0.5 * np.einsum("fjnm,abfmn->abj", H.bb.vooo, L.abb, optimize=True)
    # 3-body hbar terms
    LH.ab += np.einsum("man,mjnb->abj", X["aa"]["ovo"], H.ab.ooov, optimize=True) # [Ia]
    LH.ab -= np.einsum("amn,njmb->abj", X["ab"]["voo"], H.bb.ooov, optimize=True) # [Ib]
    LH.ab -= np.einsum("aef,fjeb->abj", X["aa"]["vvv"], H.ab.vovv, optimize=True) # [IIa]
    LH.ab -= np.einsum("aef,fjeb->abj", X["ab"]["vvv"], H.bb.vovv, optimize=True) # [IIb]
    LH.ab += np.einsum("nbm,njam->abj", X["ab"]["ovo"], H.ab.oovo, optimize=True) # [Iab]
    LH.ab += np.einsum("efb,ejaf->abj", X["ab"]["vvv"], H.ab.vovv, optimize=True) # [IIab]
    LH.ab -= np.einsum("ejn,enab->abj", X["ab"]["voo"], H.ab.vovv, optimize=True) # [III]
    LH.ab -= np.einsum("nfj,nfab->abj", X["ab"]["ovo"], H.ab.ovvv, optimize=True) # [IV]
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
    LH.aaa += (6.0 / 12.0) * np.einsum("jebm,acekm->abcjk", H.ab.ovvo, L.aab, optimize=True)
    # three-body Hbar terms (verified against explicit 3-body hbars)
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
    LH.aab += np.einsum("ejmb,aecmk->abcjk", H.aa.voov, L.aab, optimize=True) # (8) !
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
    LH.abb += np.einsum("eck,ejab->abcjk", L.ab, H.ab.vovv, optimize=True) # !
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

    ### L1A
    #h3a_vvvvoo = (
       #-(6.0 / 12.0) * np.einsum("bmje,acmk->abcejk", H.aa.voov, T.aa, optimize=True) # [I]
       #+(3.0 / 12.0) * np.einsum("abef,fcjk->abcejk", H.aa.vvvv, T.aa, optimize=True) # [II]
    #)
    #h3a_vvvvoo -= np.transpose(h3a_vvvvoo, (1, 0, 2, 3, 4, 5)) + np.transpose(h3a_vvvvoo, (2, 1, 0, 3, 4, 5)) # antisymmetrize A(a/bc)
    #h3a_vvvvoo -= np.transpose(h3a_vvvvoo, (0, 2, 1, 3, 4, 5)) # antisymmetrize A(bc)
    #h3a_vvvvoo -= np.transpose(h3a_vvvvoo, (0, 1, 2, 3, 5, 4)) # antisymmetrize A(jk)
    #LH.a += (1.0 / 12.0) * np.einsum("efgno,efgano->a", L.aaa, h3a_vvvvoo, optimize=True)
    #
    #h3b_vvvvoo = (
        #+ 0.5 * np.einsum("abef,fcjk->abcejk", H.aa.vvvv, T.ab, optimize=True) # [II]
        #- np.einsum("amek,bcjm->abcejk", H.ab.vovo, T.ab, optimize=True) # [III]
        #+ np.einsum("acef,bfjk->abcejk", H.ab.vvvv, T.ab, optimize=True) # [IV]
        #- np.einsum("bmje,acmk->abcejk", H.aa.voov, T.ab, optimize=True) # [I]
        #- 0.5 * np.einsum("mcek,abmj->abcejk", H.ab.ovvo, T.aa, optimize=True) # [V]
    #)
    #h3b_vvvvoo -= np.transpose(h3b_vvvvoo, (1, 0, 2, 3, 4, 5)) # antisymmetrize A(ab)
    #LH.a += 0.5 * np.einsum("efgno,efgano->a", L.aab, h3b_vvvvoo, optimize=True)
    #
    #h3c_vvvvoo = (
        #- 0.5 * np.einsum("amej,bcmk->abcejk", H.ab.vovo, T.bb, optimize=True) # [III]
        #+ 0.5 * np.einsum("abef,fcjk->abcejk", H.ab.vvvv, T.bb, optimize=True) # [IV]
        #- np.einsum("mbej,acmk->abcejk", H.ab.ovvo, T.ab, optimize=True) # [V]
    #)
    #h3c_vvvvoo -= np.transpose(h3c_vvvvoo, (0, 2, 1, 3, 4, 5)) # antisymmetrize A(bc)
    #h3c_vvvvoo -= np.transpose(h3c_vvvvoo, (0, 1, 2, 3, 5, 4)) # antisymmetrize A(jk)
    #LH.a += 0.25 * np.einsum("efgno,efgano->a", L.abb, h3c_vvvvoo, optimize=True)
    ###

    ### L2A
    #h3a_vvooov = (
    #        -0.5 * np.einsum("nmje,abin->abmije", H.aa.ooov, T.aa, optimize=True)
    #        + 0.5 * np.einsum("bmfe,afij->abmije", H.aa.vovv, T.aa, optimize=True)
    #)
    #h3a_vvooov -= np.transpose(h3a_vvooov, (1, 0, 2, 3, 4, 5)) # (ab)
    #h3a_vvooov -= np.transpose(h3a_vvooov, (0, 1, 2, 4, 3, 5)) # (ij)
    #LH.aa += 0.25 * np.einsum("efjmnb,aefmn->abj", h3a_vvooov, L.aaa, optimize=True)
    #h3b_ovvvoo = (
    #        -np.einsum("mnej,abin->mabeij", H.ab.oovo, T.ab, optimize=True)
    #        +np.einsum("mbef,afij->mabeij", H.ab.ovvv, T.ab, optimize=True)
    #        -np.einsum("nmie,abnj->mabeij", H.aa.ooov, T.ab, optimize=True)
    #        +np.einsum("amfe,fbij->mabeij", H.aa.vovv, T.ab, optimize=True)
    #)
    #LH.aa += np.einsum("jefbmn,aefmn->abj", h3b_ovvvoo, L.aab, optimize=True)
    #h3c_ovvvoo = (
    #       -0.5 * np.einsum("mnej,abin->mabeij", H.ab.oovo, T.bb, optimize=True)
    #       +0.5 * np.einsum("mbef,afij->mabeij", H.ab.ovvv, T.bb, optimize=True)
    #)
    #h3c_ovvvoo -= np.transpose(h3c_ovvvoo, (0, 2, 1, 3, 4, 5)) # (ab)
    #h3c_ovvvoo -= np.transpose(h3c_ovvvoo, (0, 1, 2, 3, 5, 4)) # (ij)
    #LH.aa += 0.25 * np.einsum("jefbmn,aefmn->abj", h3c_ovvvoo, L.abb, optimize=True)
    # h3a_vvvvvo = (
    #     -(3.0 / 6.0) * np.einsum("anef,bcnk->abcefk", H.aa.vovv, T.aa, optimize=True)
    # )
    # h3a_vvvvvo -= np.transpose(h3a_vvvvvo, (1, 0, 2, 3, 4, 5)) + np.transpose(h3a_vvvvvo, (2, 1, 0, 3, 4, 5)) # (a/bc)
    # h3a_vvvvvo -= np.transpose(h3a_vvvvvo, (0, 2, 1, 3, 4, 5)) # (bc)
    # LH.aa += (1.0 / 12.0) * np.einsum("efgabo,efgjo->abj", h3a_vvvvvo, L.aaa, optimize=True)
    # h3b_vvvvvo = (
    #     -np.einsum("anef,bcnk->abcefk", H.aa.vovv, T.ab, optimize=True)
    # )
    # h3b_vvvvvo -= np.transpose(h3b_vvvvvo, (1, 0, 2, 3, 4, 5)) # (ab)
    # LH.aa += 0.25 * np.einsum("efgabo,efgjo->abj", h3b_vvvvvo, L.aab, optimize=True)
    ###

    ### L2B
    ## vooovv-type ###
    # (a)
    #h3b_vvooov = (
    #      -0.5 * np.einsum("nmje,abin->abmije", H.ab.ooov, T.aa, optimize=True) # [Ia]
    #      + 0.5 * np.einsum("bmfe,afij->abmije", H.ab.vovv, T.aa, optimize=True) # [IIa]
    #)
    #h3b_vvooov -= np.transpose(h3b_vvooov, (1, 0, 2, 3, 4, 5)) # (ab)
    #h3b_vvooov -= np.transpose(h3b_vvooov, (0, 1, 2, 4, 3, 5)) # (ij)
    #LH.ab += 0.25 * np.einsum("efjmnb,aefmn->abj", h3b_vvooov, L.aaa, optimize=True)
    # (b)
    #h3c_vvooov = (
    #       -np.einsum("nmie,abnj->abmije", H.ab.ooov, T.ab, optimize=True) # [Ia]
    #       -np.einsum("nmje,abin->abmije", H.bb.ooov, T.ab, optimize=True) # [Ib]
    #       +np.einsum("amfe,fbij->abmije", H.ab.vovv, T.ab, optimize=True) # [IIa]
    #       +np.einsum("bmfe,afij->abmije", H.bb.vovv, T.ab, optimize=True) # [IIb]
    #)
    #LH.ab += np.einsum("efjmnb,aefmn->abj", h3c_vvooov, L.aab, optimize=True)
    # (c)
    #h3d_vvooov = (
    #     -0.5 * np.einsum("nmje,abin->abmije", H.bb.ooov, T.bb, optimize=True) # [Ib]
    #     +0.5 * np.einsum("bmfe,afij->abmije", H.bb.vovv, T.bb, optimize=True) # [IIb]
    #)
    #h3d_vvooov -= np.transpose(h3d_vvooov, (1, 0, 2, 3, 4, 5)) # (ab)
    #h3d_vvooov -= np.transpose(h3d_vvooov, (0, 1, 2, 4, 3, 5)) # (ij)
    #LH.ab += 0.25 * np.einsum("efjmnb,aefmn->abj", h3d_vvooov, L.abb, optimize=True)
    # (d)
    #h3b_vvovoo = (
    #   -0.5 * np.einsum("nmej,acnk->acmekj", H.ab.oovo, T.aa, optimize=True) # [Iab]
    #   +np.einsum("amef,cfkj->acmekj", H.ab.vovv, T.ab, optimize=True) # [IIab]
    #)
    #h3b_vvovoo -= np.transpose(h3b_vvovoo, (1, 0, 2, 3, 4, 5)) # (ac)
    #LH.ab -= 0.5 * np.einsum("efjanm,efbnm->abj", h3b_vvovoo, L.aab, optimize=True)
    # (e)
    #h3c_vvovoo =(
    #   -np.einsum("nmej,acnk->acmekj", H.ab.oovo, T.ab, optimize=True) # [Iab]
    #   +0.5 * np.einsum("amef,fcjk->acmekj", H.ab.vovv, T.bb, optimize=True) # [IIab]
    #)
    #h3c_vvovoo -= np.transpose(h3c_vvovoo, (0, 1, 2, 3, 5, 4)) # (jk)
    #LH.ab -= 0.5 * np.einsum("efjanm,efbnm->abj", h3c_vvovoo, L.abb, optimize=True)
    ### vvvvvo-type ###
    # (f)
    #h3b_vvvvov = (
    #    -np.einsum("anef,cbkn->acbekf", H.ab.vovv, T.ab, optimize=True) # [III]
    #    -0.5 * np.einsum("nbef,acnk->acbekf", H.ab.ovvv, T.aa, optimize=True) # [IV]
    #)
    #h3b_vvvvov -= np.transpose(h3b_vvvvov, (1, 0, 2, 3, 4, 5)) # (ac)
    #LH.ab += 0.5 * np.einsum("egfaob,egfoj->abj", h3b_vvvvov, L.aab, optimize=True)
    # (g)
    #h3c_vvvvov = (
    #    -0.5 * np.einsum("anef,bcnk->acbekf", H.ab.vovv, T.bb, optimize=True) # [III]
    #    -np.einsum("nbef,acnk->acbekf", H.ab.ovvv, T.ab, optimize=True) # [IV]
    #)
    #h3c_vvvvov -= np.transpose(h3c_vvvvov, (0, 2, 1, 3, 4, 5)) # (bc)
    #LH.ab += 0.5 * np.einsum("egfaob,egfoj->abj", h3c_vvvvov, L.abb, optimize=True)

    ### L3A
    # (a)
    #h3a_vvovov = -np.einsum("mnef,abmj->abnejf", H.aa.oovv, T.aa, optimize=True)
    #LH.aaa += (6.0 / 24.0) * np.einsum("efjanb,efcnk->abcjk", h3a_vvovov, L.aaa, optimize=True)
    # (b)
    #h3b_vovvvo = -np.einsum("mnef,abmj->anbefj", H.aa.oovv, T.ab, optimize=True)
    #LH.aaa += (6.0 / 12.0) * np.einsum("ejfabn,ecfkn->abcjk", h3b_vovvvo, L.aab, optimize=True)
    # (c)
    #h3a_oovovo = np.einsum("mnef,ecjk->mncjfk", H.aa.oovv, T.aa, optimize=True)
    #LH.aaa -= (3.0 / 24.0) * np.einsum("jkfmcn,abfmn->abcjk", h3a_oovovo, L.aaa, optimize=True)
    # (d)
    #h3b_oovovo = np.einsum("mnef,ecjk->mncjfk", H.aa.oovv, T.ab, optimize=True)
    #LH.aaa -= (3.0 / 12.0) * np.einsum("jkfmcn,abfmn->abcjk", h3b_oovovo, L.aab, optimize=True)
    #

    ### L3B (there is a bug here I think)
    ## (a)
    #h3a_vvovov = -np.einsum("mnef,abmj->abnejf", H.aa.oovv, T.aa, optimize=True)
    #LH.aab += (1.0 / 4.0) * np.einsum("efjanb,efcnk->abcjk", h3a_vvovov, L.aab, optimize=True)
    ## (b)
    #h3b_vovvvo = -np.einsum("mnef,abmj->anbefj", H.aa.oovv, T.ab, optimize=True)
    #LH.aab += (1.0 / 2.0) * np.einsum("ejfabn,ecfkn->abcjk", h3b_vovvvo, L.abb, optimize=True)
    ## (c)
    #h3b_ovooov = np.einsum("mnef,ecjk->mcnjkf", H.ab.oovv, T.aa, optimize=True)
    #LH.aab -= (1.0 / 4.0) * np.einsum("jfkmnc,abfmn->abcjk", h3b_ovooov, L.aaa, optimize=True)
    ## (d)
    #h3c_ovooov = np.einsum("mnef,ecjk->mcnjkf", H.ab.oovv, T.ab, optimize=True)
    #LH.aab -= (1.0 / 2.0) * np.einsum("jfkmnc,abfmn->abcjk", h3c_ovooov, L.aab, optimize=True)
    ## (e)
    #h3b_vvovov = -np.einsum("mnef,abmj->abnejf", H.ab.oovv, T.aa, optimize=True)
    #LH.aab += 0.5 * np.einsum("efkanc,ebfjn->abcjk", h3b_vvovov, L.aaa, optimize=True)
    ## (f)
    #h3c_vvovov = -np.einsum("mnef,acmk->acnekf", H.ab.oovv, T.ab, optimize=True)
    #LH.aab += np.einsum("efkanc,ebfjn->abcjk", h3c_vvovov, L.aab, optimize=True)
    ## (g)
    #h3b_ovovoo = np.einsum("nmfe,bejk->jfkbnm", H.ab.oovv, T.ab, optimize=True)
    #LH.aab += np.einsum("jfkbnm,afcnm->abcjk", h3b_ovovoo, L.aab, optimize=True)
    ## (h)
    #h3c_oovvoo = np.einsum("nmfe,bejk->nmbfkj", H.ab.oovv, T.bb, optimize=True)
    #LH.abb -= (1.0 / 2.0) * np.einsum("jkfbmn,afcnm->abcjk", h3c_oovvoo, L.abb, optimize=True)
    ## (i)
    #h3b_ovvvov = -np.einsum("nmfe,bcjm->nbcfje", H.ab.oovv, T.ab, optimize=True)
    #LH.abb += np.einsum("jfebnc,afenk->abcjk", h3b_ovvvov, L.aab, optimize=True)
    ## (j)
    #h3c_ovvvov = -np.einsum("nmfe,bcjm->nbefjc", H.ab.oovv, T.bb, optimize=True)
    #LH.abb += (1.0 / 2.0) * np.einsum("jfebnc,afenk->abcjk", h3c_ovvvov, L.abb, optimize=True)
    #

    ### L3C
    # (a)
    #h3b_vvovov = -np.einsum("mnef,abmj->abnejf", H.ab.oovv, T.aa, optimize=True)
    #LH.abb += 0.5 * np.einsum("efjanb,efcnk->abcjk", h3b_vvovov, L.aab, optimize=True)
    # (b)
    #h3c_vvovov = -np.einsum("mnef,acmk->acnekf", H.ab.oovv, T.ab, optimize=True)
    #LH.abb += np.einsum("efjanb,efcnk->abcjk", h3c_vvovov, L.abb, optimize=True)
    # (c)
    #h3c_vvoovv = -np.einsum("mnef,bcjm->bcnjef", H.bb.oovv, T.ab, optimize=True)
    #LH.abb += 0.5 * np.einsum("feknbc,afenj->abcjk", h3c_vvoovv, L.aab, optimize=True)
    # (d)
    #h3d_vvoovv = -np.einsum("mnef,bcjm->cbnejf", H.bb.oovv, T.bb, optimize=True)
    #LH.abb += 0.25 * np.einsum("efkbnc,aefjn->abcjk", h3d_vvoovv, L.abb, optimize=True)
    # (e)
    #h3c_voooov = np.einsum("mnef,bejk->bmnjkf", H.bb.oovv, T.ab, optimize=True)
    #LH.abb -= 0.5 * np.einsum("fjknmc,afbnm->abcjk", h3c_voooov, L.aab, optimize=True)
    # (f)
    #h3d_voooov = np.einsum("mnef,bejk->bmnjkf", H.bb.oovv, T.bb, optimize=True)
    #LH.abb -= 0.25 * np.einsum("fjknmc,abfmn->abcjk", h3d_voooov, L.abb, optimize=True)