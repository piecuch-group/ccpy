import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

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
    LH.a = ccpy_einsum("e,ea->a", L.a, H.a.vv)
    LH.a += 0.5 * ccpy_einsum("efn,fena->a", L.aa, H.aa.vvov)
    LH.a += ccpy_einsum("efn,efan->a", L.ab, H.ab.vvvo)
    # parts contracted with L3 (verified against explicit 3-body hbars)
    LH.a -= ccpy_einsum("mfan,mfn->a", H.ab.ovvo, X["ab"]["ovo"])
    LH.a -= ccpy_einsum("fmna,mfn->a", H.aa.voov, X["aa"]["ovo"])
    LH.a -= 0.5 * ccpy_einsum("fge,feag->a", X["aa"]["vvv"], H.aa.vvvv)
    LH.a -= ccpy_einsum("eman,enm->a", H.ab.vovo, X["ab"]["voo"])
    LH.a -= ccpy_einsum("efg,egaf->a", X["ab"]["vvv"], H.ab.vvvv)
    return LH

def build_LH_2A(L, LH, H, X):
    """Calculate the projection < 0 | (L1p+L2p1h+L3p2h)*(H_N e^(T1+T2))_C | abj >."""
    LH.aa = ccpy_einsum("a,jb->abj", L.a, H.a.ov)
    LH.aa += 0.5 * ccpy_einsum("e,ejab->abj", L.a, H.aa.vovv)
    LH.aa += ccpy_einsum("ebj,ea->abj", L.aa, H.a.vv)
    LH.aa -= 0.5 * ccpy_einsum("abm,jm->abj", L.aa, H.a.oo)
    LH.aa += ccpy_einsum("afn,fjnb->abj", L.aa, H.aa.voov)
    LH.aa += ccpy_einsum("afn,jfbn->abj", L.ab, H.ab.ovvo)
    LH.aa += 0.25 * ccpy_einsum("efj,efab->abj", L.aa, H.aa.vvvv)
    LH.aa -= 0.5 * ccpy_einsum("mjab,m->abj", H.aa.oovv, X["a"]["o"])
    # parts contracted with L3
    LH.aa += 0.5 * ccpy_einsum("fena,ebfjn->abj", H.aa.vvov, L.aaa)
    LH.aa += ccpy_einsum("efan,ebfjn->abj", H.ab.vvvo, L.aab)
    LH.aa -= 0.25 * ccpy_einsum("fjnm,abfmn->abj", H.aa.vooo, L.aaa)
    LH.aa -= 0.5 * ccpy_einsum("jfmn,abfmn->abj", H.ab.ovoo, L.aab)
    # 3-body hbar terms (verified against explicit 3-body hbars)
    LH.aa += ccpy_einsum("mbn,jmna->abj", X["aa"]["ovo"], H.aa.ooov) #
    LH.aa -= ccpy_einsum("amn,jnbm->abj", X["ab"]["voo"], H.ab.oovo) #
    LH.aa -= ccpy_einsum("aef,fjeb->abj", X["aa"]["vvv"], H.aa.vovv) #
    LH.aa -= ccpy_einsum("aef,jfbe->abj", X["ab"]["vvv"], H.ab.ovvv) #
    LH.aa -= 0.5 * ccpy_einsum("mej,emba->abj", X["aa"]["ovo"], H.aa.vovv)
    LH.aa -= np.transpose(LH.aa, (1, 0, 2))
    return LH

def build_LH_2B(L, LH, H, X):
    """Calculate the projection < 0 | (L1p+L2p1h+L3p2h)*(H_N e^(T1+T2))_C | ab~j~ >."""
    LH.ab = ccpy_einsum("a,jb->abj", L.a, H.b.ov)
    LH.ab += ccpy_einsum("e,ejab->abj", L.a, H.ab.vovv)
    LH.ab -= ccpy_einsum("abm,jm->abj", L.ab, H.b.oo)
    LH.ab += ccpy_einsum("aej,eb->abj", L.ab, H.b.vv)
    LH.ab += ccpy_einsum("ebj,ea->abj", L.ab, H.a.vv)
    LH.ab += ccpy_einsum("afn,fjnb->abj", L.aa, H.ab.voov)
    LH.ab += ccpy_einsum("afn,fjnb->abj", L.ab, H.bb.voov)
    LH.ab -= ccpy_einsum("ebm,ejam->abj", L.ab, H.ab.vovo)
    LH.ab += ccpy_einsum("efj,efab->abj", L.ab, H.ab.vvvv)
    LH.ab -= ccpy_einsum("mjab,m->abj", H.ab.oovv, X["a"]["o"])
    # parts contracted with L3
    LH.ab += 0.5 * ccpy_einsum("fena,efbnj->abj", H.aa.vvov, L.aab)
    LH.ab += ccpy_einsum("efan,efbnj->abj", H.ab.vvvo, L.abb)
    LH.ab += ccpy_einsum("fenb,afenj->abj", H.ab.vvov, L.aab)
    LH.ab += 0.5 * ccpy_einsum("fenb,afenj->abj", H.bb.vvov, L.abb)
    LH.ab -= ccpy_einsum("fjnm,afbnm->abj", H.ab.vooo, L.aab)
    LH.ab -= 0.5 * ccpy_einsum("fjnm,abfmn->abj", H.bb.vooo, L.abb)
    # 3-body hbar terms
    LH.ab += ccpy_einsum("man,mjnb->abj", X["aa"]["ovo"], H.ab.ooov) # [Ia]
    LH.ab -= ccpy_einsum("amn,njmb->abj", X["ab"]["voo"], H.bb.ooov) # [Ib]
    LH.ab -= ccpy_einsum("aef,fjeb->abj", X["aa"]["vvv"], H.ab.vovv) # [IIa]
    LH.ab -= ccpy_einsum("aef,fjeb->abj", X["ab"]["vvv"], H.bb.vovv) # [IIb]
    LH.ab += ccpy_einsum("nbm,njam->abj", X["ab"]["ovo"], H.ab.oovo) # [Iab]
    LH.ab += ccpy_einsum("efb,ejaf->abj", X["ab"]["vvv"], H.ab.vovv) # [IIab]
    LH.ab -= ccpy_einsum("ejn,enab->abj", X["ab"]["voo"], H.ab.vovv) # [III]
    LH.ab -= ccpy_einsum("nfj,nfab->abj", X["ab"]["ovo"], H.ab.ovvv) # [IV]
    return LH

def build_LH_3A(L, LH, H, X):
    """Calculate the projection < 0 | (L1p+L2p1h+L3p2h)*(H_N e^(T1+T2))_C | jkabc >."""
    # moment-like terms < 0 | (L1p+L2p1h)*(H_N e^(T1+T2))_C | jkabc >
    LH.aaa = (3.0 / 12.0) * ccpy_einsum("a,jkbc->abcjk", L.a, H.aa.oovv)
    LH.aaa += (6.0 / 12.0) * ccpy_einsum("abj,kc->abcjk", L.aa, H.a.ov)
    LH.aaa -= (3.0 / 12.0) * ccpy_einsum("abm,jkmc->abcjk", L.aa, H.aa.ooov)
    LH.aaa += (6.0 / 12.0) * ccpy_einsum("eck,ejab->abcjk", L.aa, H.aa.vovv)
    # <0|L3p2h*(H_N e^(T1+T2))_C | jkabc>
    LH.aaa -= (2.0 / 12.0) * ccpy_einsum("jm,abcmk->abcjk", H.a.oo, L.aaa)
    LH.aaa += (3.0 / 12.0) * ccpy_einsum("eb,aecjk->abcjk", H.a.vv, L.aaa)
    LH.aaa += (1.0 / 24.0) * ccpy_einsum("jkmn,abcmn->abcjk", H.aa.oooo, L.aaa)
    LH.aaa += (3.0 / 24.0) * ccpy_einsum("efbc,aefjk->abcjk", H.aa.vvvv, L.aaa)
    LH.aaa += (6.0 / 12.0) * ccpy_einsum("ejmb,acekm->abcjk", H.aa.voov, L.aaa)
    LH.aaa += (6.0 / 12.0) * ccpy_einsum("jebm,acekm->abcjk", H.ab.ovvo, L.aab)
    # three-body Hbar terms (verified against explicit 3-body hbars)
    LH.aaa -= (6.0 / 12.0) * ccpy_einsum("mck,mjab->abcjk", X["aa"]["ovo"], H.aa.oovv)
    LH.aaa += (3.0 / 12.0) * ccpy_einsum("aeb,jkec->abcjk", X["aa"]["vvv"], H.aa.oovv)
    LH.aaa -= np.transpose(LH.aaa, (1, 0, 2, 3, 4)) + np.transpose(LH.aaa, (2, 1, 0, 3, 4)) # antisymmetrize A(a/bc)
    LH.aaa -= np.transpose(LH.aaa, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    LH.aaa -= np.transpose(LH.aaa, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return LH

def build_LH_3B(L, LH, H, X):
    """Calculate the projection < 0 | (L1p+L2p1h+L3p2h)(H_N e^(T1+T2))_C | jk~abc~ >."""
    # moment-like terms < 0 | (L1p+L2p1h)*(H_N e^(T1+T2))_C | jk~abc~ >
    LH.aab = ccpy_einsum("a,jkbc->abcjk", L.a, H.ab.oovv)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("abj,kc->abcjk", L.aa, H.b.ov)
    LH.aab += ccpy_einsum("ack,jb->abcjk", L.ab, H.a.ov)
    LH.aab -= (1.0 / 2.0) * ccpy_einsum("abm,jkmc->abcjk", L.aa, H.ab.ooov)
    LH.aab -= ccpy_einsum("acm,jkbm->abcjk", L.ab, H.ab.oovo)
    LH.aab += ccpy_einsum("aej,ekbc->abcjk", L.aa, H.ab.vovv)
    LH.aab += ccpy_einsum("aek,jebc->abcjk", L.ab, H.ab.ovvv)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("eck,ejab->abcjk", L.ab, H.aa.vovv)
    # < 0 | L3p2h*(H_N e^(T1+T2))_C | jk~abc~ >
    LH.aab -= (1.0 / 2.0) * ccpy_einsum("jm,abcmk->abcjk", H.a.oo, L.aab) # (1)
    LH.aab -= (1.0 / 2.0) * ccpy_einsum("km,abcjm->abcjk", H.b.oo, L.aab) # (2)
    LH.aab += ccpy_einsum("ea,ebcjk->abcjk", H.a.vv, L.aab) # (3)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("ec,abejk->abcjk", H.b.vv, L.aab) # (4)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("jkmn,abcmn->abcjk", H.ab.oooo, L.aab) # (5)
    LH.aab += (1.0 / 4.0) * ccpy_einsum("efab,efcjk->abcjk", H.aa.vvvv, L.aab) # (6)
    LH.aab += ccpy_einsum("efbc,aefjk->abcjk", H.ab.vvvv, L.aab) # (7)
    LH.aab += ccpy_einsum("ejmb,aecmk->abcjk", H.aa.voov, L.aab) # (8) !
    LH.aab += ccpy_einsum("jebm,aecmk->abcjk", H.ab.ovvo, L.abb) # (9)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("ekmc,abejm->abcjk", H.ab.voov, L.aaa) # (10)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("ekmc,abejm->abcjk", H.bb.voov, L.aab) # (11)
    LH.aab -= (1.0 / 2.0) * ccpy_einsum("jemc,abemk->abcjk", H.ab.ovov, L.aab) # (12)
    LH.aab -= ccpy_einsum("ekbm,aecjm->abcjk", H.ab.vovo, L.aab) # (13)
    # three-body Hbar terms
    LH.aab -= ccpy_einsum("akm,jmbc->abcjk", X["ab"]["voo"], H.ab.oovv) # (1)
    LH.aab -= (1.0 / 2.0) * ccpy_einsum("mck,mjab->abcjk", X["ab"]["ovo"], H.aa.oovv) # (2)
    LH.aab -= ccpy_einsum("mbj,mkac->abcjk", X["aa"]["ovo"], H.ab.oovv) # (3)
    LH.aab += (1.0 / 2.0) * ccpy_einsum("aeb,jkec->abcjk", X["aa"]["vvv"], H.ab.oovv) # (4)
    LH.aab += ccpy_einsum("aec,jkbe->abcjk", X["ab"]["vvv"], H.ab.oovv) # (5)
    LH.aab -= np.transpose(LH.aab, (1, 0, 2, 3, 4)) # antisymmetrize A(ab)
    return LH

def build_LH_3C(L, LH, H, X):
    """Calculate the projection < 0 | (L1p+L2p1h+L3p2h)(H_N e^(T1+T2))_C | j~k~ab~c~ >."""
    # moment-like terms < 0 | (L1p+L2p1h)*(H_N e^(T1+T2))_C | j~k~ab~c~ >
    LH.abb = (1.0 / 4.0) * ccpy_einsum("a,jkbc->abcjk", L.a, H.bb.oovv)
    LH.abb += ccpy_einsum("abj,kc->abcjk", L.ab, H.b.ov)
    LH.abb -= (2.0 / 4.0) * ccpy_einsum("abm,jkmc->abcjk", L.ab, H.bb.ooov)
    LH.abb += (2.0 / 4.0) * ccpy_einsum("aej,ekbc->abcjk", L.ab, H.bb.vovv)
    LH.abb += ccpy_einsum("eck,ejab->abcjk", L.ab, H.ab.vovv) # !
    # < 0 | L3p2h*(H_N e^(T1+T2))_C | j!k~ab!c~ >
    LH.abb -= (2.0 / 4.0) * ccpy_einsum("jm,abcmk->abcjk", H.b.oo, L.abb)
    LH.abb += (2.0 / 4.0) * ccpy_einsum("eb,aecjk->abcjk", H.b.vv, L.abb)
    LH.abb += (1.0 / 4.0) * ccpy_einsum("ea,ebcjk->abcjk", H.a.vv, L.abb)
    LH.abb += (1.0 / 8.0) * ccpy_einsum("jkmn,abcmn->abcjk", H.bb.oooo, L.abb)
    LH.abb += (2.0 / 4.0) * ccpy_einsum("efab,efcjk->abcjk", H.ab.vvvv, L.abb)
    LH.abb += (1.0 / 8.0) * ccpy_einsum("efbc,aefjk->abcjk", H.bb.vvvv, L.abb)
    LH.abb += ccpy_einsum("ejmb,aecmk->abcjk", H.ab.voov, L.aab)
    LH.abb += ccpy_einsum("ejmb,aecmk->abcjk", H.bb.voov, L.abb)
    LH.abb -= (2.0 / 4.0) * ccpy_einsum("ejam,ebcmk->abcjk", H.ab.vovo, L.abb)
    # three-body Hbar terms
    LH.abb -= ccpy_einsum("mck,mjab->abcjk", X["ab"]["ovo"], H.ab.oovv)
    LH.abb -= (2.0 / 4.0) * ccpy_einsum("ajm,mkbc->abcjk", X["ab"]["voo"], H.bb.oovv)
    LH.abb += (2.0 / 4.0) * ccpy_einsum("aeb,jkec->abcjk", X["ab"]["vvv"], H.bb.oovv)
    LH.abb -= np.transpose(LH.abb, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    LH.abb -= np.transpose(LH.abb, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return LH

    ### L1A
    #h3a_vvvvoo = (
       #-(6.0 / 12.0) * ccpy_einsum("bmje,acmk->abcejk", H.aa.voov, T.aa) # [I]
       #+(3.0 / 12.0) * ccpy_einsum("abef,fcjk->abcejk", H.aa.vvvv, T.aa) # [II]
    #)
    #h3a_vvvvoo -= np.transpose(h3a_vvvvoo, (1, 0, 2, 3, 4, 5)) + np.transpose(h3a_vvvvoo, (2, 1, 0, 3, 4, 5)) # antisymmetrize A(a/bc)
    #h3a_vvvvoo -= np.transpose(h3a_vvvvoo, (0, 2, 1, 3, 4, 5)) # antisymmetrize A(bc)
    #h3a_vvvvoo -= np.transpose(h3a_vvvvoo, (0, 1, 2, 3, 5, 4)) # antisymmetrize A(jk)
    #LH.a += (1.0 / 12.0) * ccpy_einsum("efgno,efgano->a", L.aaa, h3a_vvvvoo)
    #
    #h3b_vvvvoo = (
        #+ 0.5 * ccpy_einsum("abef,fcjk->abcejk", H.aa.vvvv, T.ab) # [II]
        #- ccpy_einsum("amek,bcjm->abcejk", H.ab.vovo, T.ab) # [III]
        #+ ccpy_einsum("acef,bfjk->abcejk", H.ab.vvvv, T.ab) # [IV]
        #- ccpy_einsum("bmje,acmk->abcejk", H.aa.voov, T.ab) # [I]
        #- 0.5 * ccpy_einsum("mcek,abmj->abcejk", H.ab.ovvo, T.aa) # [V]
    #)
    #h3b_vvvvoo -= np.transpose(h3b_vvvvoo, (1, 0, 2, 3, 4, 5)) # antisymmetrize A(ab)
    #LH.a += 0.5 * ccpy_einsum("efgno,efgano->a", L.aab, h3b_vvvvoo)
    #
    #h3c_vvvvoo = (
        #- 0.5 * ccpy_einsum("amej,bcmk->abcejk", H.ab.vovo, T.bb) # [III]
        #+ 0.5 * ccpy_einsum("abef,fcjk->abcejk", H.ab.vvvv, T.bb) # [IV]
        #- ccpy_einsum("mbej,acmk->abcejk", H.ab.ovvo, T.ab) # [V]
    #)
    #h3c_vvvvoo -= np.transpose(h3c_vvvvoo, (0, 2, 1, 3, 4, 5)) # antisymmetrize A(bc)
    #h3c_vvvvoo -= np.transpose(h3c_vvvvoo, (0, 1, 2, 3, 5, 4)) # antisymmetrize A(jk)
    #LH.a += 0.25 * ccpy_einsum("efgno,efgano->a", L.abb, h3c_vvvvoo)
    ###

    ### L2A
    #h3a_vvooov = (
    #        -0.5 * ccpy_einsum("nmje,abin->abmije", H.aa.ooov, T.aa)
    #        + 0.5 * ccpy_einsum("bmfe,afij->abmije", H.aa.vovv, T.aa)
    #)
    #h3a_vvooov -= np.transpose(h3a_vvooov, (1, 0, 2, 3, 4, 5)) # (ab)
    #h3a_vvooov -= np.transpose(h3a_vvooov, (0, 1, 2, 4, 3, 5)) # (ij)
    #LH.aa += 0.25 * ccpy_einsum("efjmnb,aefmn->abj", h3a_vvooov, L.aaa)
    #h3b_ovvvoo = (
    #        -ccpy_einsum("mnej,abin->mabeij", H.ab.oovo, T.ab)
    #        +ccpy_einsum("mbef,afij->mabeij", H.ab.ovvv, T.ab)
    #        -ccpy_einsum("nmie,abnj->mabeij", H.aa.ooov, T.ab)
    #        +ccpy_einsum("amfe,fbij->mabeij", H.aa.vovv, T.ab)
    #)
    #LH.aa += ccpy_einsum("jefbmn,aefmn->abj", h3b_ovvvoo, L.aab)
    #h3c_ovvvoo = (
    #       -0.5 * ccpy_einsum("mnej,abin->mabeij", H.ab.oovo, T.bb)
    #       +0.5 * ccpy_einsum("mbef,afij->mabeij", H.ab.ovvv, T.bb)
    #)
    #h3c_ovvvoo -= np.transpose(h3c_ovvvoo, (0, 2, 1, 3, 4, 5)) # (ab)
    #h3c_ovvvoo -= np.transpose(h3c_ovvvoo, (0, 1, 2, 3, 5, 4)) # (ij)
    #LH.aa += 0.25 * ccpy_einsum("jefbmn,aefmn->abj", h3c_ovvvoo, L.abb)
    # h3a_vvvvvo = (
    #     -(3.0 / 6.0) * ccpy_einsum("anef,bcnk->abcefk", H.aa.vovv, T.aa)
    # )
    # h3a_vvvvvo -= np.transpose(h3a_vvvvvo, (1, 0, 2, 3, 4, 5)) + np.transpose(h3a_vvvvvo, (2, 1, 0, 3, 4, 5)) # (a/bc)
    # h3a_vvvvvo -= np.transpose(h3a_vvvvvo, (0, 2, 1, 3, 4, 5)) # (bc)
    # LH.aa += (1.0 / 12.0) * ccpy_einsum("efgabo,efgjo->abj", h3a_vvvvvo, L.aaa)
    # h3b_vvvvvo = (
    #     -ccpy_einsum("anef,bcnk->abcefk", H.aa.vovv, T.ab)
    # )
    # h3b_vvvvvo -= np.transpose(h3b_vvvvvo, (1, 0, 2, 3, 4, 5)) # (ab)
    # LH.aa += 0.25 * ccpy_einsum("efgabo,efgjo->abj", h3b_vvvvvo, L.aab)
    ###

    ### L2B
    ## vooovv-type ###
    # (a)
    #h3b_vvooov = (
    #      -0.5 * ccpy_einsum("nmje,abin->abmije", H.ab.ooov, T.aa) # [Ia]
    #      + 0.5 * ccpy_einsum("bmfe,afij->abmije", H.ab.vovv, T.aa) # [IIa]
    #)
    #h3b_vvooov -= np.transpose(h3b_vvooov, (1, 0, 2, 3, 4, 5)) # (ab)
    #h3b_vvooov -= np.transpose(h3b_vvooov, (0, 1, 2, 4, 3, 5)) # (ij)
    #LH.ab += 0.25 * ccpy_einsum("efjmnb,aefmn->abj", h3b_vvooov, L.aaa)
    # (b)
    #h3c_vvooov = (
    #       -ccpy_einsum("nmie,abnj->abmije", H.ab.ooov, T.ab) # [Ia]
    #       -ccpy_einsum("nmje,abin->abmije", H.bb.ooov, T.ab) # [Ib]
    #       +ccpy_einsum("amfe,fbij->abmije", H.ab.vovv, T.ab) # [IIa]
    #       +ccpy_einsum("bmfe,afij->abmije", H.bb.vovv, T.ab) # [IIb]
    #)
    #LH.ab += ccpy_einsum("efjmnb,aefmn->abj", h3c_vvooov, L.aab)
    # (c)
    #h3d_vvooov = (
    #     -0.5 * ccpy_einsum("nmje,abin->abmije", H.bb.ooov, T.bb) # [Ib]
    #     +0.5 * ccpy_einsum("bmfe,afij->abmije", H.bb.vovv, T.bb) # [IIb]
    #)
    #h3d_vvooov -= np.transpose(h3d_vvooov, (1, 0, 2, 3, 4, 5)) # (ab)
    #h3d_vvooov -= np.transpose(h3d_vvooov, (0, 1, 2, 4, 3, 5)) # (ij)
    #LH.ab += 0.25 * ccpy_einsum("efjmnb,aefmn->abj", h3d_vvooov, L.abb)
    # (d)
    #h3b_vvovoo = (
    #   -0.5 * ccpy_einsum("nmej,acnk->acmekj", H.ab.oovo, T.aa) # [Iab]
    #   +ccpy_einsum("amef,cfkj->acmekj", H.ab.vovv, T.ab) # [IIab]
    #)
    #h3b_vvovoo -= np.transpose(h3b_vvovoo, (1, 0, 2, 3, 4, 5)) # (ac)
    #LH.ab -= 0.5 * ccpy_einsum("efjanm,efbnm->abj", h3b_vvovoo, L.aab)
    # (e)
    #h3c_vvovoo =(
    #   -ccpy_einsum("nmej,acnk->acmekj", H.ab.oovo, T.ab) # [Iab]
    #   +0.5 * ccpy_einsum("amef,fcjk->acmekj", H.ab.vovv, T.bb) # [IIab]
    #)
    #h3c_vvovoo -= np.transpose(h3c_vvovoo, (0, 1, 2, 3, 5, 4)) # (jk)
    #LH.ab -= 0.5 * ccpy_einsum("efjanm,efbnm->abj", h3c_vvovoo, L.abb)
    ### vvvvvo-type ###
    # (f)
    #h3b_vvvvov = (
    #    -ccpy_einsum("anef,cbkn->acbekf", H.ab.vovv, T.ab) # [III]
    #    -0.5 * ccpy_einsum("nbef,acnk->acbekf", H.ab.ovvv, T.aa) # [IV]
    #)
    #h3b_vvvvov -= np.transpose(h3b_vvvvov, (1, 0, 2, 3, 4, 5)) # (ac)
    #LH.ab += 0.5 * ccpy_einsum("egfaob,egfoj->abj", h3b_vvvvov, L.aab)
    # (g)
    #h3c_vvvvov = (
    #    -0.5 * ccpy_einsum("anef,bcnk->acbekf", H.ab.vovv, T.bb) # [III]
    #    -ccpy_einsum("nbef,acnk->acbekf", H.ab.ovvv, T.ab) # [IV]
    #)
    #h3c_vvvvov -= np.transpose(h3c_vvvvov, (0, 2, 1, 3, 4, 5)) # (bc)
    #LH.ab += 0.5 * ccpy_einsum("egfaob,egfoj->abj", h3c_vvvvov, L.abb)

    ### L3A
    # (a)
    #h3a_vvovov = -ccpy_einsum("mnef,abmj->abnejf", H.aa.oovv, T.aa)
    #LH.aaa += (6.0 / 24.0) * ccpy_einsum("efjanb,efcnk->abcjk", h3a_vvovov, L.aaa)
    # (b)
    #h3b_vovvvo = -ccpy_einsum("mnef,abmj->anbefj", H.aa.oovv, T.ab)
    #LH.aaa += (6.0 / 12.0) * ccpy_einsum("ejfabn,ecfkn->abcjk", h3b_vovvvo, L.aab)
    # (c)
    #h3a_oovovo = ccpy_einsum("mnef,ecjk->mncjfk", H.aa.oovv, T.aa)
    #LH.aaa -= (3.0 / 24.0) * ccpy_einsum("jkfmcn,abfmn->abcjk", h3a_oovovo, L.aaa)
    # (d)
    #h3b_oovovo = ccpy_einsum("mnef,ecjk->mncjfk", H.aa.oovv, T.ab)
    #LH.aaa -= (3.0 / 12.0) * ccpy_einsum("jkfmcn,abfmn->abcjk", h3b_oovovo, L.aab)
    #

    ### L3B (there is a bug here I think)
    ## (a)
    #h3a_vvovov = -ccpy_einsum("mnef,abmj->abnejf", H.aa.oovv, T.aa)
    #LH.aab += (1.0 / 4.0) * ccpy_einsum("efjanb,efcnk->abcjk", h3a_vvovov, L.aab)
    ## (b)
    #h3b_vovvvo = -ccpy_einsum("mnef,abmj->anbefj", H.aa.oovv, T.ab)
    #LH.aab += (1.0 / 2.0) * ccpy_einsum("ejfabn,ecfkn->abcjk", h3b_vovvvo, L.abb)
    ## (c)
    #h3b_ovooov = ccpy_einsum("mnef,ecjk->mcnjkf", H.ab.oovv, T.aa)
    #LH.aab -= (1.0 / 4.0) * ccpy_einsum("jfkmnc,abfmn->abcjk", h3b_ovooov, L.aaa)
    ## (d)
    #h3c_ovooov = ccpy_einsum("mnef,ecjk->mcnjkf", H.ab.oovv, T.ab)
    #LH.aab -= (1.0 / 2.0) * ccpy_einsum("jfkmnc,abfmn->abcjk", h3c_ovooov, L.aab)
    ## (e)
    #h3b_vvovov = -ccpy_einsum("mnef,abmj->abnejf", H.ab.oovv, T.aa)
    #LH.aab += 0.5 * ccpy_einsum("efkanc,ebfjn->abcjk", h3b_vvovov, L.aaa)
    ## (f)
    #h3c_vvovov = -ccpy_einsum("mnef,acmk->acnekf", H.ab.oovv, T.ab)
    #LH.aab += ccpy_einsum("efkanc,ebfjn->abcjk", h3c_vvovov, L.aab)
    ## (g)
    #h3b_ovovoo = ccpy_einsum("nmfe,bejk->jfkbnm", H.ab.oovv, T.ab)
    #LH.aab += ccpy_einsum("jfkbnm,afcnm->abcjk", h3b_ovovoo, L.aab)
    ## (h)
    #h3c_oovvoo = ccpy_einsum("nmfe,bejk->nmbfkj", H.ab.oovv, T.bb)
    #LH.abb -= (1.0 / 2.0) * ccpy_einsum("jkfbmn,afcnm->abcjk", h3c_oovvoo, L.abb)
    ## (i)
    #h3b_ovvvov = -ccpy_einsum("nmfe,bcjm->nbcfje", H.ab.oovv, T.ab)
    #LH.abb += ccpy_einsum("jfebnc,afenk->abcjk", h3b_ovvvov, L.aab)
    ## (j)
    #h3c_ovvvov = -ccpy_einsum("nmfe,bcjm->nbefjc", H.ab.oovv, T.bb)
    #LH.abb += (1.0 / 2.0) * ccpy_einsum("jfebnc,afenk->abcjk", h3c_ovvvov, L.abb)
    #

    ### L3C
    # (a)
    #h3b_vvovov = -ccpy_einsum("mnef,abmj->abnejf", H.ab.oovv, T.aa)
    #LH.abb += 0.5 * ccpy_einsum("efjanb,efcnk->abcjk", h3b_vvovov, L.aab)
    # (b)
    #h3c_vvovov = -ccpy_einsum("mnef,acmk->acnekf", H.ab.oovv, T.ab)
    #LH.abb += ccpy_einsum("efjanb,efcnk->abcjk", h3c_vvovov, L.abb)
    # (c)
    #h3c_vvoovv = -ccpy_einsum("mnef,bcjm->bcnjef", H.bb.oovv, T.ab)
    #LH.abb += 0.5 * ccpy_einsum("feknbc,afenj->abcjk", h3c_vvoovv, L.aab)
    # (d)
    #h3d_vvoovv = -ccpy_einsum("mnef,bcjm->cbnejf", H.bb.oovv, T.bb)
    #LH.abb += 0.25 * ccpy_einsum("efkbnc,aefjn->abcjk", h3d_vvoovv, L.abb)
    # (e)
    #h3c_voooov = ccpy_einsum("mnef,bejk->bmnjkf", H.bb.oovv, T.ab)
    #LH.abb -= 0.5 * ccpy_einsum("fjknmc,afbnm->abcjk", h3c_voooov, L.aab)
    # (f)
    #h3d_voooov = ccpy_einsum("mnef,bejk->bmnjkf", H.bb.oovv, T.bb)
    #LH.abb -= 0.25 * ccpy_einsum("fjknmc,abfmn->abcjk", h3d_voooov, L.abb)