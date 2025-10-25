'''
Coupled-Cluster Method with Singles, Doubles, and Triples (CCSDT)
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.lib.core import cc_loops2

def update(T, dT, H, X, shift, flag_RHF):

    X = get_pre_ccs_intermediates(X, T, H, flag_RHF)

    # update T1
    T, dT = update_t1a(T, dT, H, X, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, X, shift)

    # CCS intermediates
    X = get_ccs_intermediates_opt(X, T, H, flag_RHF)

    # update T2
    T, dT = update_t2a(T, dT, X, H, shift)
    T, dT = update_t2b(T, dT, X, H, shift)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, X, H, shift)

    # CCSD intermediates
    #[TODO]: Should accept CCS HBar as input and build only terms with T2 in it
    X = get_ccsd_intermediates(T, X, H, flag_RHF)

    # update T3
    T, dT = update_t3a(T, dT, X, H, shift)
    T, dT = update_t3b(T, dT, X, H, shift)
    if flag_RHF:
        T.abb = np.transpose(T.aab, (2, 1, 0, 5, 4, 3))
        dT.abb = np.transpose(dT.aab, (2, 1, 0, 5, 4, 3))
        T.bbb = T.aaa.copy()
        dT.bbb = dT.aaa.copy()
    else:
        T, dT = update_t3c(T, dT, X, H, shift)
        T, dT = update_t3d(T, dT, X, H, shift)

    return T, dT

def update_t1a(T, dT, H, X, shift):
    """
    Update t1a amplitudes by calculating the projection <ia|(H_N exp(T1+T2+T3))_C|0>.
    """
    dT.a = -ccpy_einsum("mi,am->ai", X.a.oo, T.a)
    dT.a += ccpy_einsum("ae,ei->ai", X.a.vv, T.a)
    dT.a += ccpy_einsum("me,aeim->ai", X.a.ov, T.aa) # [+]
    dT.a += ccpy_einsum("me,aeim->ai", X.b.ov, T.ab) # [+]
    dT.a += ccpy_einsum("anif,fn->ai", H.aa.voov, T.a)
    dT.a += ccpy_einsum("anif,fn->ai", H.ab.voov, T.b)
    dT.a -= 0.5 * ccpy_einsum("mnif,afmn->ai", H.aa.ooov, T.aa)
    dT.a -= ccpy_einsum("mnif,afmn->ai", H.ab.ooov, T.ab)
    dT.a += 0.5 * ccpy_einsum("anef,efin->ai", H.aa.vovv, T.aa)
    dT.a += ccpy_einsum("anef,efin->ai", H.ab.vovv, T.ab)
    # T3 parts
    dT.a += 0.25 * ccpy_einsum("mnef,aefimn->ai", H.aa.oovv, T.aaa)
    dT.a += ccpy_einsum("mnef,aefimn->ai", H.ab.oovv, T.aab)
    dT.a += 0.25 * ccpy_einsum("mnef,aefimn->ai", H.bb.oovv, T.abb)
    T.a, dT.a = cc_loops2.update_t1a(
        T.a,
        dT.a + H.a.vo,
        H.a.oo,
        H.a.vv,
        shift,
    )
    return T, dT

def update_t1b(T, dT, H, X, shift):
    """
    Update t1b amplitudes by calculating the projection <i~a~|(H_N exp(T1+T2+T3))_C|0>.
    """
    dT.b = -ccpy_einsum("mi,am->ai", X.b.oo, T.b)
    dT.b += ccpy_einsum("ae,ei->ai", X.b.vv, T.b)
    dT.b += ccpy_einsum("anif,fn->ai", H.bb.voov, T.b)
    dT.b += ccpy_einsum("nafi,fn->ai", H.ab.ovvo, T.a)
    dT.b += ccpy_einsum("me,eami->ai", X.a.ov, T.ab)
    dT.b += ccpy_einsum("me,aeim->ai", X.b.ov, T.bb)
    dT.b -= 0.5 * ccpy_einsum("mnif,afmn->ai", H.bb.ooov, T.bb)
    dT.b -= ccpy_einsum("nmfi,fanm->ai", H.ab.oovo, T.ab)
    dT.b += 0.5 * ccpy_einsum("anef,efin->ai", H.bb.vovv, T.bb)
    dT.b += ccpy_einsum("nafe,feni->ai", H.ab.ovvv, T.ab)
    # T3 parts
    dT.b += 0.25 * ccpy_einsum("mnef,aefimn->ai", H.bb.oovv, T.bbb)
    dT.b += 0.25 * ccpy_einsum("mnef,efamni->ai", H.aa.oovv, T.aab)
    dT.b += ccpy_einsum("mnef,efamni->ai", H.ab.oovv, T.abb)
    T.b, dT.b = cc_loops2.update_t1b(
        T.b,
        dT.b + H.b.vo,
        H.b.oo,
        H.b.vv,
        shift,
    )
    return T, dT

def update_t2a(T, dT, H, H0, shift):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N exp(T1+T2+T3))_C|0>.
    """
    # intermediates
    I2A_voov = H.aa.voov + (
        + 0.5 * ccpy_einsum("mnef,afin->amie", H0.aa.oovv, T.aa)
        + ccpy_einsum("mnef,afin->amie", H0.ab.oovv, T.ab)
    )
    I2A_oooo = H.aa.oooo + 0.5 * ccpy_einsum("mnef,efij->mnij", H0.aa.oovv, T.aa)
    I2B_voov = H.ab.voov + 0.5 * ccpy_einsum("mnef,afin->amie", H0.bb.oovv, T.ab)
    I2A_vooo = H.aa.vooo + 0.5*ccpy_einsum('anef,efij->anij', H0.aa.vovv + 0.5 * H.aa.vovv, T.aa)

    tau = 0.5 * T.aa + ccpy_einsum('ai,bj->abij', T.a, T.a)

    dT.aa = -0.5 * ccpy_einsum("amij,bm->abij", I2A_vooo, T.a)
    dT.aa += 0.5 * ccpy_einsum("abie,ej->abij", H.aa.vvov, T.a)
    dT.aa += 0.5 * ccpy_einsum("ae,ebij->abij", H.a.vv, T.aa)
    dT.aa -= 0.5 * ccpy_einsum("mi,abmj->abij", H.a.oo, T.aa)
    dT.aa += ccpy_einsum("amie,ebmj->abij", I2A_voov, T.aa)
    dT.aa += ccpy_einsum("amie,bejm->abij", I2B_voov, T.ab)
    dT.aa += 0.25 * ccpy_einsum("abef,efij->abij", H0.aa.vvvv, tau)
    dT.aa += 0.125 * ccpy_einsum("mnij,abmn->abij", I2A_oooo, T.aa)
    # T3 parts
    dT.aa += 0.25 * ccpy_einsum("me,abeijm->abij", H.a.ov, T.aaa)
    dT.aa += 0.25 * ccpy_einsum("me,abeijm->abij", H.b.ov, T.aab)
    dT.aa -= 0.5 * ccpy_einsum("mnif,abfmjn->abij", H0.ab.ooov + H.ab.ooov, T.aab)
    dT.aa -= 0.25 * ccpy_einsum("mnif,abfmjn->abij", H0.aa.ooov + H.aa.ooov, T.aaa)
    dT.aa += 0.25 * ccpy_einsum("anef,ebfijn->abij", H0.aa.vovv + H.aa.vovv, T.aaa)
    dT.aa += 0.5 * ccpy_einsum("anef,ebfijn->abij", H0.ab.vovv + H.ab.vovv, T.aab)
    T.aa, dT.aa = cc_loops2.update_t2a(
        T.aa, dT.aa + 0.25 * H0.aa.vvoo, H0.a.oo, H0.a.vv, shift
    )
    return T, dT

def update_t2b(T, dT, H, H0, shift):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N exp(T1+T2+T3))_C|0>.
    """
    # intermediates
    I2A_voov = H.aa.voov + (
        + ccpy_einsum("mnef,aeim->anif", H0.aa.oovv, T.aa)
        + ccpy_einsum("nmfe,aeim->anif", H0.ab.oovv, T.ab)
    )
    I2B_voov = H.ab.voov + (
        + ccpy_einsum("mnef,aeim->anif", H0.ab.oovv, T.aa)
        + ccpy_einsum("mnef,aeim->anif", H0.bb.oovv, T.ab)
    )
    I2B_oooo = H.ab.oooo + ccpy_einsum("mnef,efij->mnij", H0.ab.oovv, T.ab)
    I2B_vovo = H.ab.vovo - ccpy_einsum("mnef,afmj->anej", H0.ab.oovv, T.ab)
    I2B_ovoo = H.ab.ovoo + ccpy_einsum("maef,efij->maij", H0.ab.ovvv + 0.5 * H.ab.ovvv, T.ab)
    I2B_vooo = H.ab.vooo + ccpy_einsum("amef,efij->amij", H0.ab.vovv + 0.5 * H.ab.vovv, T.ab)

    tau = T.ab + ccpy_einsum('ai,bj->abij', T.a, T.b)

    dT.ab = -ccpy_einsum("mbij,am->abij", I2B_ovoo, T.a)
    dT.ab -= ccpy_einsum("amij,bm->abij", I2B_vooo, T.b)
    dT.ab += ccpy_einsum("abej,ei->abij", H.ab.vvvo, T.a)
    dT.ab += ccpy_einsum("abie,ej->abij", H.ab.vvov, T.b)
    dT.ab += ccpy_einsum("ae,ebij->abij", H.a.vv, T.ab)
    dT.ab += ccpy_einsum("be,aeij->abij", H.b.vv, T.ab)
    dT.ab -= ccpy_einsum("mi,abmj->abij", H.a.oo, T.ab)
    dT.ab -= ccpy_einsum("mj,abim->abij", H.b.oo, T.ab)
    dT.ab += ccpy_einsum("amie,ebmj->abij", I2A_voov, T.ab)
    dT.ab += ccpy_einsum("amie,ebmj->abij", I2B_voov, T.bb)
    dT.ab += ccpy_einsum("mbej,aeim->abij", H.ab.ovvo, T.aa)
    dT.ab += ccpy_einsum("bmje,aeim->abij", H.bb.voov, T.ab)
    dT.ab -= ccpy_einsum("mbie,aemj->abij", H.ab.ovov, T.ab)
    dT.ab -= ccpy_einsum("amej,ebim->abij", I2B_vovo, T.ab)
    dT.ab += ccpy_einsum("mnij,abmn->abij", I2B_oooo, T.ab)
    dT.ab += ccpy_einsum("abef,efij->abij", H0.ab.vvvv, tau)
    # T3 parts
    dT.ab -= 0.5 * ccpy_einsum("mnif,afbmnj->abij", H0.aa.ooov + H.aa.ooov, T.aab)
    dT.ab -= ccpy_einsum("nmfj,afbinm->abij", H0.ab.oovo + H.ab.oovo, T.aab)
    dT.ab -= 0.5 * ccpy_einsum("mnjf,afbinm->abij", H0.bb.ooov + H.bb.ooov, T.abb)
    dT.ab -= ccpy_einsum("mnif,afbmnj->abij", H0.ab.ooov + H.ab.ooov, T.abb)
    dT.ab += 0.5 * ccpy_einsum("anef,efbinj->abij", H0.aa.vovv + H.aa.vovv, T.aab)
    dT.ab += ccpy_einsum("anef,efbinj->abij", H0.ab.vovv + H.ab.vovv, T.abb)
    dT.ab += ccpy_einsum("nbfe,afeinj->abij", H0.ab.ovvv + H.ab.ovvv, T.aab)
    dT.ab += 0.5 * ccpy_einsum("bnef,afeinj->abij", H0.bb.vovv + H.bb.vovv, T.abb)
    dT.ab += ccpy_einsum("me,aebimj->abij", H.a.ov, T.aab)
    dT.ab += ccpy_einsum("me,aebimj->abij", H.b.ov, T.abb)
    T.ab, dT.ab = cc_loops2.update_t2b(
        T.ab, dT.ab + H0.ab.vvoo, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv, shift
    )
    return T, dT

def update_t2c(T, dT, H, H0, shift):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N exp(T1+T2+T3))_C|0>.
    """
    # intermediates
    I2C_oooo = H.bb.oooo + 0.5 * ccpy_einsum("mnef,efij->mnij", H0.bb.oovv, T.bb)

    I2B_ovvo = H.ab.ovvo + (
        + ccpy_einsum("mnef,afin->maei", H0.ab.oovv, T.bb)
        + 0.5 * ccpy_einsum("mnef,fani->maei", H0.aa.oovv, T.ab)
    )
    I2C_voov = H.bb.voov + 0.5 * ccpy_einsum("mnef,afin->amie", H0.bb.oovv, T.bb)
    I2C_vooo = H.bb.vooo + 0.5 * ccpy_einsum('anef,efij->anij', H0.bb.vovv + 0.5 * H.bb.vovv, T.bb)

    tau = 0.5 * T.bb + ccpy_einsum('ai,bj->abij', T.b, T.b)

    dT.bb = -0.5 * ccpy_einsum("amij,bm->abij", I2C_vooo, T.b)
    dT.bb += 0.5 * ccpy_einsum("abie,ej->abij", H.bb.vvov, T.b)
    dT.bb += 0.5 * ccpy_einsum("ae,ebij->abij", H.b.vv, T.bb)
    dT.bb -= 0.5 * ccpy_einsum("mi,abmj->abij", H.b.oo, T.bb)
    dT.bb += ccpy_einsum("amie,ebmj->abij", I2C_voov, T.bb)
    dT.bb += ccpy_einsum("maei,ebmj->abij", I2B_ovvo, T.ab)
    dT.bb += 0.25 * ccpy_einsum("abef,efij->abij", H0.bb.vvvv, tau)
    dT.bb += 0.125 * ccpy_einsum("mnij,abmn->abij", I2C_oooo, T.bb)
    # T3 parts
    dT.bb += 0.25 * ccpy_einsum("me,eabmij->abij", H.a.ov, T.abb)
    dT.bb += 0.25 * ccpy_einsum("me,abeijm->abij", H.b.ov, T.bbb)
    dT.bb += 0.25 * ccpy_einsum("anef,ebfijn->abij", H0.bb.vovv + H.bb.vovv, T.bbb)
    dT.bb += 0.5 * ccpy_einsum("nafe,febnij->abij", H0.ab.ovvv + H.ab.ovvv, T.abb)
    dT.bb -= 0.25 * ccpy_einsum("mnif,abfmjn->abij", H0.bb.ooov + H.bb.ooov, T.bbb)
    dT.bb -= 0.5 * ccpy_einsum("nmfi,fabnmj->abij", H0.ab.oovo + H.ab.oovo, T.abb)
    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H0.bb.vvoo, H0.b.oo, H0.b.vv, shift
    )
    return T, dT

def update_t3a(T, dT, H, H0, shift):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N exp(T1+T2+T3))_C|0>.
    """
    # <ijkabc | H(2) | 0 > + (VT3)_C intermediates
    I2A_vvov = -0.5 * ccpy_einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa)
    I2A_vvov -= ccpy_einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab)
    I2A_vvov += H.aa.vvov
    
    I2A_vooo = 0.5 * ccpy_einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa)
    I2A_vooo += H.aa.vooo + ccpy_einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab)
    I2A_vooo -= ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)

    # MM(2,3)A
    dT.aaa = -0.25 * ccpy_einsum("amij,bcmk->abcijk", I2A_vooo, T.aa)
    dT.aaa += 0.25 * ccpy_einsum("abie,ecjk->abcijk", I2A_vvov, T.aa)
    # (HBar*T3)_C
    dT.aaa -= (1.0 / 12.0) * ccpy_einsum("mk,abcijm->abcijk", H.a.oo, T.aaa)
    dT.aaa += (1.0 / 12.0) * ccpy_einsum("ce,abeijk->abcijk", H.a.vv, T.aaa)
    dT.aaa += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa)
    dT.aaa += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa)
    dT.aaa += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa)
    dT.aaa += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab)
    T.aaa, dT.aaa = cc_loops2.update_t3a_v2(
        T.aaa,
        dT.aaa,
        H0.a.oo,
        H0.a.vv,
        shift,
    )
    return T, dT

def update_t3b(T, dT, H, H0, shift):
    """
    Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N exp(T1+T2+T3))_C|0>.
    """
    # <ijk~abc~ | H(2) | 0 > + (VT3)_C intermediates
    I2A_vvov = -0.5 * ccpy_einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa)
    I2A_vvov += -ccpy_einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab)
    I2A_vvov += H.aa.vvov
    
    I2A_vooo = 0.5 * ccpy_einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa)
    I2A_vooo += ccpy_einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab)
    I2A_vooo += -ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
    I2A_vooo += H.aa.vooo

    I2B_vvvo = -0.5 * ccpy_einsum("mnef,afbmnj->abej", H0.aa.oovv, T.aab)
    I2B_vvvo += -ccpy_einsum("mnef,afbmnj->abej", H0.ab.oovv, T.abb)
    I2B_vvvo += H.ab.vvvo
    
    I2B_ovoo = 0.5 * ccpy_einsum("mnef,efbinj->mbij", H0.aa.oovv, T.aab)
    I2B_ovoo += ccpy_einsum("mnef,efbinj->mbij", H0.ab.oovv, T.abb)
    I2B_ovoo += -ccpy_einsum("me,ecjk->mcjk", H.a.ov, T.ab)
    I2B_ovoo += H.ab.ovoo

    I2B_vvov = -ccpy_einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab)
    I2B_vvov += -0.5 * ccpy_einsum("nmfe,afbinm->abie", H0.bb.oovv, T.abb)
    I2B_vvov += H.ab.vvov
    
    I2B_vooo = ccpy_einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab)
    I2B_vooo += 0.5 * ccpy_einsum("nmfe,afeinj->amij", H0.bb.oovv, T.abb)
    I2B_vooo += -ccpy_einsum("me,aeik->amik", H.b.ov, T.ab)
    I2B_vooo += H.ab.vooo

    # MM(2,3)B
    dT.aab = 0.5 * ccpy_einsum("bcek,aeij->abcijk", I2B_vvvo, T.aa)
    dT.aab -= 0.5 * ccpy_einsum("mcjk,abim->abcijk", I2B_ovoo, T.aa)
    dT.aab += ccpy_einsum("acie,bejk->abcijk", I2B_vvov, T.ab)
    dT.aab -= ccpy_einsum("amik,bcjm->abcijk", I2B_vooo, T.ab)
    dT.aab += 0.5 * ccpy_einsum("abie,ecjk->abcijk", I2A_vvov, T.ab)
    dT.aab -= 0.5 * ccpy_einsum("amij,bcmk->abcijk", I2A_vooo, T.ab)
    # (HBar*T3)_C
    dT.aab -= 0.5 * ccpy_einsum("mi,abcmjk->abcijk", H.a.oo, T.aab)
    dT.aab -= 0.25 * ccpy_einsum("mk,abcijm->abcijk", H.b.oo, T.aab)
    dT.aab += 0.5 * ccpy_einsum("ae,ebcijk->abcijk", H.a.vv, T.aab)
    dT.aab += 0.25 * ccpy_einsum("ce,abeijk->abcijk", H.b.vv, T.aab)
    dT.aab += 0.125 * ccpy_einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aab)
    dT.aab += 0.5 * ccpy_einsum("mnjk,abcimn->abcijk", H.ab.oooo, T.aab)
    dT.aab += 0.125 * ccpy_einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aab)
    dT.aab += 0.5 * ccpy_einsum("bcef,aefijk->abcijk", H.ab.vvvv, T.aab)
    dT.aab += ccpy_einsum("amie,ebcmjk->abcijk", H.aa.voov, T.aab)
    dT.aab += ccpy_einsum("amie,becjmk->abcijk", H.ab.voov, T.abb)
    dT.aab += 0.25 * ccpy_einsum("mcek,abeijm->abcijk", H.ab.ovvo, T.aaa)
    dT.aab += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.bb.voov, T.aab)
    dT.aab -= 0.5 * ccpy_einsum("amek,ebcijm->abcijk", H.ab.vovo, T.aab)
    dT.aab -= 0.5 * ccpy_einsum("mcie,abemjk->abcijk", H.ab.ovov, T.aab)
    T.aab, dT.aab = cc_loops2.update_t3b_v2(
        T.aab,
        dT.aab,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift,
    )
    return T, dT

def update_t3c(T, dT, H, H0, shift):
    """
    Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N exp(T1+T2+T3))_C|0>.
    """
    # <ij~k~ab~c~ | H(2) | 0 > + (VT3)_C intermediates
    I2B_vvvo = -0.5 * ccpy_einsum("mnef,afbmnj->abej", H0.aa.oovv, T.aab)
    I2B_vvvo += -ccpy_einsum("mnef,afbmnj->abej", H0.ab.oovv, T.abb)
    I2B_vvvo += H.ab.vvvo
    
    I2B_ovoo = 0.5 * ccpy_einsum("mnef,efbinj->mbij", H0.aa.oovv, T.aab)
    I2B_ovoo += ccpy_einsum("mnef,efbinj->mbij", H0.ab.oovv, T.abb)
    I2B_ovoo -= ccpy_einsum("me,ebij->mbij", H.a.ov, T.ab)
    I2B_ovoo += H.ab.ovoo

    I2B_vvov = -ccpy_einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab)
    I2B_vvov += -0.5 * ccpy_einsum("nmfe,afbinm->abie", H0.bb.oovv, T.abb)
    I2B_vvov += H.ab.vvov
    
    I2B_vooo = ccpy_einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab)
    I2B_vooo += 0.5 * ccpy_einsum("nmfe,afeinj->amij", H0.bb.oovv, T.abb)
    I2B_vooo -= ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
    I2B_vooo += H.ab.vooo

    I2C_vvov = -0.5 * ccpy_einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb)
    I2C_vvov += -ccpy_einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb)
    I2C_vvov += H.bb.vvov
    
    I2C_vooo = ccpy_einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb)
    I2C_vooo += 0.5 * ccpy_einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb)
    I2C_vooo -= ccpy_einsum("me,cekj->cmkj", H.b.ov, T.bb)
    I2C_vooo += H.bb.vooo

    # MM(2,3)C
    dT.abb = 0.5 * ccpy_einsum("abie,ecjk->abcijk", I2B_vvov, T.bb)
    dT.abb -= 0.5 * ccpy_einsum("amij,bcmk->abcijk", I2B_vooo, T.bb)
    dT.abb += 0.5 * ccpy_einsum("cbke,aeij->abcijk", I2C_vvov, T.ab)
    dT.abb -= 0.5 * ccpy_einsum("cmkj,abim->abcijk", I2C_vooo, T.ab)
    dT.abb += ccpy_einsum("abej,ecik->abcijk", I2B_vvvo, T.ab)
    dT.abb -= ccpy_einsum("mbij,acmk->abcijk", I2B_ovoo, T.ab)
    # (HBar*T3)_C
    dT.abb -= 0.25 * ccpy_einsum("mi,abcmjk->abcijk", H.a.oo, T.abb)
    dT.abb -= 0.5 * ccpy_einsum("mj,abcimk->abcijk", H.b.oo, T.abb)
    dT.abb += 0.25 * ccpy_einsum("ae,ebcijk->abcijk", H.a.vv, T.abb)
    dT.abb += 0.5 * ccpy_einsum("be,aecijk->abcijk", H.b.vv, T.abb)
    dT.abb += 0.125 * ccpy_einsum("mnjk,abcimn->abcijk", H.bb.oooo, T.abb)
    dT.abb += 0.5 * ccpy_einsum("mnij,abcmnk->abcijk", H.ab.oooo, T.abb)
    dT.abb += 0.125 * ccpy_einsum("bcef,aefijk->abcijk", H.bb.vvvv, T.abb)
    dT.abb += 0.5 * ccpy_einsum("abef,efcijk->abcijk", H.ab.vvvv, T.abb)
    dT.abb += 0.25 * ccpy_einsum("amie,ebcmjk->abcijk", H.aa.voov, T.abb)
    dT.abb += 0.25 * ccpy_einsum("amie,ebcmjk->abcijk", H.ab.voov, T.bbb)
    dT.abb += ccpy_einsum("mbej,aecimk->abcijk", H.ab.ovvo, T.aab)
    dT.abb += ccpy_einsum("bmje,aecimk->abcijk", H.bb.voov, T.abb)
    dT.abb -= 0.5 * ccpy_einsum("mbie,aecmjk->abcijk", H.ab.ovov, T.abb)
    dT.abb -= 0.5 * ccpy_einsum("amej,ebcimk->abcijk", H.ab.vovo, T.abb)
    T.abb, dT.abb = cc_loops2.update_t3c_v2(
        T.abb,
        dT.abb,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift,
    )
    return T, dT

def update_t3d(T, dT, H, H0, shift):
    """
    Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N exp(T1+T2+T3))_C|0>.
    """
    #  <i~j~k~a~b~c~ | H(2) | 0 > + (VT3)_C intermediates
    I2C_vvov = -0.5 * ccpy_einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb)
    I2C_vvov -= ccpy_einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb)
    I2C_vvov += H.bb.vvov
    
    I2C_vooo = 0.5 * ccpy_einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb)
    I2C_vooo += ccpy_einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb)
    I2C_vooo -= ccpy_einsum("me,aeij->amij", H.b.ov, T.bb)
    I2C_vooo += H.bb.vooo

    # MM(2,3)D
    dT.bbb = -0.25 * ccpy_einsum("amij,bcmk->abcijk", I2C_vooo, T.bb)
    dT.bbb += 0.25 * ccpy_einsum("abie,ecjk->abcijk", I2C_vvov, T.bb)
    # (HBar*T3)_C
    dT.bbb -= (1.0 / 12.0) * ccpy_einsum("mk,abcijm->abcijk", H.b.oo, T.bbb)
    dT.bbb += (1.0 / 12.0) * ccpy_einsum("ce,abeijk->abcijk", H.b.vv, T.bbb)
    dT.bbb += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", H.bb.oooo, T.bbb)
    dT.bbb += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", H.bb.vvvv, T.bbb)
    dT.bbb += 0.25 * ccpy_einsum("maei,ebcmjk->abcijk", H.ab.ovvo, T.abb)
    dT.bbb += 0.25 * ccpy_einsum("amie,ebcmjk->abcijk", H.bb.voov, T.bbb)
    T.bbb, dT.bbb = cc_loops2.update_t3d_v2(
        T.bbb, 
        dT.bbb, 
        H0.b.oo, 
        H0.b.vv, 
        shift,
    )
    return T, dT
