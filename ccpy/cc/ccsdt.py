"""Module with functions that perform the CC with singles, doubles,
and triples (CCSDT) calculation for a molecular system."""

import numpy as np

from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.lib.core import cc_loops2

def update(T, dT, H, X, shift, flag_RHF, system):

    X = get_pre_ccs_intermediates(X, T, H, system, flag_RHF)

    # update T1
    T, dT = update_t1a(T, dT, H, X, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, X, shift)

    # CCS intermediates
    X = get_ccs_intermediates_opt(X, T, H, system, flag_RHF)

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
    dT.a = -np.einsum("mi,am->ai", X.a.oo, T.a, optimize=True)
    dT.a += np.einsum("ae,ei->ai", X.a.vv, T.a, optimize=True)
    dT.a += np.einsum("me,aeim->ai", X.a.ov, T.aa, optimize=True) # [+]
    dT.a += np.einsum("me,aeim->ai", X.b.ov, T.ab, optimize=True) # [+]
    dT.a += np.einsum("anif,fn->ai", H.aa.voov, T.a, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.ab.voov, T.b, optimize=True)
    dT.a -= 0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, T.aa, optimize=True)
    dT.a -= np.einsum("mnif,afmn->ai", H.ab.ooov, T.ab, optimize=True)
    dT.a += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, T.aa, optimize=True)
    dT.a += np.einsum("anef,efin->ai", H.ab.vovv, T.ab, optimize=True)
    # T3 parts
    dT.a += 0.25 * np.einsum("mnef,aefimn->ai", H.aa.oovv, T.aaa, optimize=True)
    dT.a += np.einsum("mnef,aefimn->ai", H.ab.oovv, T.aab, optimize=True)
    dT.a += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, T.abb, optimize=True)
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
    dT.b = -np.einsum("mi,am->ai", X.b.oo, T.b, optimize=True)
    dT.b += np.einsum("ae,ei->ai", X.b.vv, T.b, optimize=True)
    dT.b += np.einsum("anif,fn->ai", H.bb.voov, T.b, optimize=True)
    dT.b += np.einsum("nafi,fn->ai", H.ab.ovvo, T.a, optimize=True)
    dT.b += np.einsum("me,eami->ai", X.a.ov, T.ab, optimize=True)
    dT.b += np.einsum("me,aeim->ai", X.b.ov, T.bb, optimize=True)
    dT.b -= 0.5 * np.einsum("mnif,afmn->ai", H.bb.ooov, T.bb, optimize=True)
    dT.b -= np.einsum("nmfi,fanm->ai", H.ab.oovo, T.ab, optimize=True)
    dT.b += 0.5 * np.einsum("anef,efin->ai", H.bb.vovv, T.bb, optimize=True)
    dT.b += np.einsum("nafe,feni->ai", H.ab.ovvv, T.ab, optimize=True)
    # T3 parts
    dT.b += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, T.bbb, optimize=True)
    dT.b += 0.25 * np.einsum("mnef,efamni->ai", H.aa.oovv, T.aab, optimize=True)
    dT.b += np.einsum("mnef,efamni->ai", H.ab.oovv, T.abb, optimize=True)
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
        + 0.5 * np.einsum("mnef,afin->amie", H0.aa.oovv, T.aa, optimize=True)
        + np.einsum("mnef,afin->amie", H0.ab.oovv, T.ab, optimize=True)
    )
    I2A_oooo = H.aa.oooo + 0.5 * np.einsum("mnef,efij->mnij", H0.aa.oovv, T.aa, optimize=True)
    I2B_voov = H.ab.voov + 0.5 * np.einsum("mnef,afin->amie", H0.bb.oovv, T.ab, optimize=True)
    I2A_vooo = H.aa.vooo + 0.5*np.einsum('anef,efij->anij', H0.aa.vovv + 0.5 * H.aa.vovv, T.aa, optimize=True)

    tau = 0.5 * T.aa + np.einsum('ai,bj->abij', T.a, T.a, optimize=True)

    dT.aa = -0.5 * np.einsum("amij,bm->abij", I2A_vooo, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("abie,ej->abij", H.aa.vvov, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", H.a.vv, T.aa, optimize=True)
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", H.a.oo, T.aa, optimize=True)
    dT.aa += np.einsum("amie,ebmj->abij", I2A_voov, T.aa, optimize=True)
    dT.aa += np.einsum("amie,bejm->abij", I2B_voov, T.ab, optimize=True)
    dT.aa += 0.25 * np.einsum("abef,efij->abij", H0.aa.vvvv, tau, optimize=True)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", I2A_oooo, T.aa, optimize=True)
    # T3 parts
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.a.ov, T.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T.aab, optimize=True)
    dT.aa -= 0.5 * np.einsum("mnif,abfmjn->abij", H0.ab.ooov + H.ab.ooov, T.aab, optimize=True)
    dT.aa -= 0.25 * np.einsum("mnif,abfmjn->abij", H0.aa.ooov + H.aa.ooov, T.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("anef,ebfijn->abij", H0.aa.vovv + H.aa.vovv, T.aaa, optimize=True)
    dT.aa += 0.5 * np.einsum("anef,ebfijn->abij", H0.ab.vovv + H.ab.vovv, T.aab, optimize=True)
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
        + np.einsum("mnef,aeim->anif", H0.aa.oovv, T.aa, optimize=True)
        + np.einsum("nmfe,aeim->anif", H0.ab.oovv, T.ab, optimize=True)
    )
    I2B_voov = H.ab.voov + (
        + np.einsum("mnef,aeim->anif", H0.ab.oovv, T.aa, optimize=True)
        + np.einsum("mnef,aeim->anif", H0.bb.oovv, T.ab, optimize=True)
    )
    I2B_oooo = H.ab.oooo + np.einsum("mnef,efij->mnij", H0.ab.oovv, T.ab, optimize=True)
    I2B_vovo = H.ab.vovo - np.einsum("mnef,afmj->anej", H0.ab.oovv, T.ab, optimize=True)
    I2B_ovoo = H.ab.ovoo + np.einsum("maef,efij->maij", H0.ab.ovvv + 0.5 * H.ab.ovvv, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo + np.einsum("amef,efij->amij", H0.ab.vovv + 0.5 * H.ab.vovv, T.ab, optimize=True)

    tau = T.ab + np.einsum('ai,bj->abij', T.a, T.b, optimize=True)

    dT.ab = -np.einsum("mbij,am->abij", I2B_ovoo, T.a, optimize=True)
    dT.ab -= np.einsum("amij,bm->abij", I2B_vooo, T.b, optimize=True)
    dT.ab += np.einsum("abej,ei->abij", H.ab.vvvo, T.a, optimize=True)
    dT.ab += np.einsum("abie,ej->abij", H.ab.vvov, T.b, optimize=True)
    dT.ab += np.einsum("ae,ebij->abij", H.a.vv, T.ab, optimize=True)
    dT.ab += np.einsum("be,aeij->abij", H.b.vv, T.ab, optimize=True)
    dT.ab -= np.einsum("mi,abmj->abij", H.a.oo, T.ab, optimize=True)
    dT.ab -= np.einsum("mj,abim->abij", H.b.oo, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", I2A_voov, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", I2B_voov, T.bb, optimize=True)
    dT.ab += np.einsum("mbej,aeim->abij", H.ab.ovvo, T.aa, optimize=True)
    dT.ab += np.einsum("bmje,aeim->abij", H.bb.voov, T.ab, optimize=True)
    dT.ab -= np.einsum("mbie,aemj->abij", H.ab.ovov, T.ab, optimize=True)
    dT.ab -= np.einsum("amej,ebim->abij", I2B_vovo, T.ab, optimize=True)
    dT.ab += np.einsum("mnij,abmn->abij", I2B_oooo, T.ab, optimize=True)
    dT.ab += np.einsum("abef,efij->abij", H0.ab.vvvv, tau, optimize=True)
    # T3 parts
    dT.ab -= 0.5 * np.einsum("mnif,afbmnj->abij", H0.aa.ooov + H.aa.ooov, T.aab, optimize=True)
    dT.ab -= np.einsum("nmfj,afbinm->abij", H0.ab.oovo + H.ab.oovo, T.aab, optimize=True)
    dT.ab -= 0.5 * np.einsum("mnjf,afbinm->abij", H0.bb.ooov + H.bb.ooov, T.abb, optimize=True)
    dT.ab -= np.einsum("mnif,afbmnj->abij", H0.ab.ooov + H.ab.ooov, T.abb, optimize=True)
    dT.ab += 0.5 * np.einsum("anef,efbinj->abij", H0.aa.vovv + H.aa.vovv, T.aab, optimize=True)
    dT.ab += np.einsum("anef,efbinj->abij", H0.ab.vovv + H.ab.vovv, T.abb, optimize=True)
    dT.ab += np.einsum("nbfe,afeinj->abij", H0.ab.ovvv + H.ab.ovvv, T.aab, optimize=True)
    dT.ab += 0.5 * np.einsum("bnef,afeinj->abij", H0.bb.vovv + H.bb.vovv, T.abb, optimize=True)
    dT.ab += np.einsum("me,aebimj->abij", H.a.ov, T.aab, optimize=True)
    dT.ab += np.einsum("me,aebimj->abij", H.b.ov, T.abb, optimize=True)
    T.ab, dT.ab = cc_loops2.update_t2b(
        T.ab, dT.ab + H0.ab.vvoo, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv, shift
    )
    return T, dT

def update_t2c(T, dT, H, H0, shift):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N exp(T1+T2+T3))_C|0>.
    """
    # intermediates
    I2C_oooo = H.bb.oooo + 0.5 * np.einsum("mnef,efij->mnij", H0.bb.oovv, T.bb, optimize=True)

    I2B_ovvo = H.ab.ovvo + (
        + np.einsum("mnef,afin->maei", H0.ab.oovv, T.bb, optimize=True)
        + 0.5 * np.einsum("mnef,fani->maei", H0.aa.oovv, T.ab, optimize=True)
    )
    I2C_voov = H.bb.voov + 0.5 * np.einsum("mnef,afin->amie", H0.bb.oovv, T.bb, optimize=True)
    I2C_vooo = H.bb.vooo + 0.5 * np.einsum('anef,efij->anij', H0.bb.vovv + 0.5 * H.bb.vovv, T.bb, optimize=True)

    tau = 0.5 * T.bb + np.einsum('ai,bj->abij', T.b, T.b, optimize=True)

    dT.bb = -0.5 * np.einsum("amij,bm->abij", I2C_vooo, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("abie,ej->abij", H.bb.vvov, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("ae,ebij->abij", H.b.vv, T.bb, optimize=True)
    dT.bb -= 0.5 * np.einsum("mi,abmj->abij", H.b.oo, T.bb, optimize=True)
    dT.bb += np.einsum("amie,ebmj->abij", I2C_voov, T.bb, optimize=True)
    dT.bb += np.einsum("maei,ebmj->abij", I2B_ovvo, T.ab, optimize=True)
    dT.bb += 0.25 * np.einsum("abef,efij->abij", H0.bb.vvvv, tau, optimize=True)
    dT.bb += 0.125 * np.einsum("mnij,abmn->abij", I2C_oooo, T.bb, optimize=True)
    # T3 parts
    dT.bb += 0.25 * np.einsum("me,eabmij->abij", H.a.ov, T.abb, optimize=True)
    dT.bb += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T.bbb, optimize=True)
    dT.bb += 0.25 * np.einsum("anef,ebfijn->abij", H0.bb.vovv + H.bb.vovv, T.bbb, optimize=True)
    dT.bb += 0.5 * np.einsum("nafe,febnij->abij", H0.ab.ovvv + H.ab.ovvv, T.abb, optimize=True)
    dT.bb -= 0.25 * np.einsum("mnif,abfmjn->abij", H0.bb.ooov + H.bb.ooov, T.bbb, optimize=True)
    dT.bb -= 0.5 * np.einsum("nmfi,fabnmj->abij", H0.ab.oovo + H.ab.oovo, T.abb, optimize=True)
    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H0.bb.vvoo, H0.b.oo, H0.b.vv, shift
    )
    return T, dT

def update_t3a(T, dT, H, H0, shift):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N exp(T1+T2+T3))_C|0>.
    """
    # <ijkabc | H(2) | 0 > + (VT3)_C intermediates
    I2A_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vvov -= np.einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab, optimize=True)
    I2A_vvov += H.aa.vvov
    
    I2A_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vooo += H.aa.vooo + np.einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab, optimize=True)
    I2A_vooo -= np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)

    # MM(2,3)A
    dT.aaa = -0.25 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.aa, optimize=True)
    dT.aaa += 0.25 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.aa, optimize=True)
    # (HBar*T3)_C
    dT.aaa -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.a.oo, T.aaa, optimize=True)
    dT.aaa += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.a.vv, T.aaa, optimize=True)
    dT.aaa += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa, optimize=True)
    dT.aaa += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa, optimize=True)
    dT.aaa += 0.25 * np.einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa, optimize=True)
    dT.aaa += 0.25 * np.einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab, optimize=True)
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
    I2A_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vvov += -np.einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab, optimize=True)
    I2A_vvov += H.aa.vvov
    
    I2A_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vooo += np.einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab, optimize=True)
    I2A_vooo += -np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2A_vooo += H.aa.vooo

    I2B_vvvo = -0.5 * np.einsum("mnef,afbmnj->abej", H0.aa.oovv, T.aab, optimize=True)
    I2B_vvvo += -np.einsum("mnef,afbmnj->abej", H0.ab.oovv, T.abb, optimize=True)
    I2B_vvvo += H.ab.vvvo
    
    I2B_ovoo = 0.5 * np.einsum("mnef,efbinj->mbij", H0.aa.oovv, T.aab, optimize=True)
    I2B_ovoo += np.einsum("mnef,efbinj->mbij", H0.ab.oovv, T.abb, optimize=True)
    I2B_ovoo += -np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_ovoo += H.ab.ovoo

    I2B_vvov = -np.einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab, optimize=True)
    I2B_vvov += -0.5 * np.einsum("nmfe,afbinm->abie", H0.bb.oovv, T.abb, optimize=True)
    I2B_vvov += H.ab.vvov
    
    I2B_vooo = np.einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab, optimize=True)
    I2B_vooo += 0.5 * np.einsum("nmfe,afeinj->amij", H0.bb.oovv, T.abb, optimize=True)
    I2B_vooo += -np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2B_vooo += H.ab.vooo

    # MM(2,3)B
    dT.aab = 0.5 * np.einsum("bcek,aeij->abcijk", I2B_vvvo, T.aa, optimize=True)
    dT.aab -= 0.5 * np.einsum("mcjk,abim->abcijk", I2B_ovoo, T.aa, optimize=True)
    dT.aab += np.einsum("acie,bejk->abcijk", I2B_vvov, T.ab, optimize=True)
    dT.aab -= np.einsum("amik,bcjm->abcijk", I2B_vooo, T.ab, optimize=True)
    dT.aab += 0.5 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.ab, optimize=True)
    dT.aab -= 0.5 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.ab, optimize=True)
    # (HBar*T3)_C
    dT.aab -= 0.5 * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.aab, optimize=True)
    dT.aab -= 0.25 * np.einsum("mk,abcijm->abcijk", H.b.oo, T.aab, optimize=True)
    dT.aab += 0.5 * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.aab, optimize=True)
    dT.aab += 0.25 * np.einsum("ce,abeijk->abcijk", H.b.vv, T.aab, optimize=True)
    dT.aab += 0.125 * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aab, optimize=True)
    dT.aab += 0.5 * np.einsum("mnjk,abcimn->abcijk", H.ab.oooo, T.aab, optimize=True)
    dT.aab += 0.125 * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aab, optimize=True)
    dT.aab += 0.5 * np.einsum("bcef,aefijk->abcijk", H.ab.vvvv, T.aab, optimize=True)
    dT.aab += np.einsum("amie,ebcmjk->abcijk", H.aa.voov, T.aab, optimize=True)
    dT.aab += np.einsum("amie,becjmk->abcijk", H.ab.voov, T.abb, optimize=True)
    dT.aab += 0.25 * np.einsum("mcek,abeijm->abcijk", H.ab.ovvo, T.aaa, optimize=True)
    dT.aab += 0.25 * np.einsum("cmke,abeijm->abcijk", H.bb.voov, T.aab, optimize=True)
    dT.aab -= 0.5 * np.einsum("amek,ebcijm->abcijk", H.ab.vovo, T.aab, optimize=True)
    dT.aab -= 0.5 * np.einsum("mcie,abemjk->abcijk", H.ab.ovov, T.aab, optimize=True)
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
    I2B_vvvo = -0.5 * np.einsum("mnef,afbmnj->abej", H0.aa.oovv, T.aab, optimize=True)
    I2B_vvvo += -np.einsum("mnef,afbmnj->abej", H0.ab.oovv, T.abb, optimize=True)
    I2B_vvvo += H.ab.vvvo
    
    I2B_ovoo = 0.5 * np.einsum("mnef,efbinj->mbij", H0.aa.oovv, T.aab, optimize=True)
    I2B_ovoo += np.einsum("mnef,efbinj->mbij", H0.ab.oovv, T.abb, optimize=True)
    I2B_ovoo -= np.einsum("me,ebij->mbij", H.a.ov, T.ab, optimize=True)
    I2B_ovoo += H.ab.ovoo

    I2B_vvov = -np.einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab, optimize=True)
    I2B_vvov += -0.5 * np.einsum("nmfe,afbinm->abie", H0.bb.oovv, T.abb, optimize=True)
    I2B_vvov += H.ab.vvov
    
    I2B_vooo = np.einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab, optimize=True)
    I2B_vooo += 0.5 * np.einsum("nmfe,afeinj->amij", H0.bb.oovv, T.abb, optimize=True)
    I2B_vooo -= np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
    I2B_vooo += H.ab.vooo

    I2C_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vvov += -np.einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb, optimize=True)
    I2C_vvov += H.bb.vvov
    
    I2C_vooo = np.einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb, optimize=True)
    I2C_vooo += 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vooo -= np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)
    I2C_vooo += H.bb.vooo

    # MM(2,3)C
    dT.abb = 0.5 * np.einsum("abie,ecjk->abcijk", I2B_vvov, T.bb, optimize=True)
    dT.abb -= 0.5 * np.einsum("amij,bcmk->abcijk", I2B_vooo, T.bb, optimize=True)
    dT.abb += 0.5 * np.einsum("cbke,aeij->abcijk", I2C_vvov, T.ab, optimize=True)
    dT.abb -= 0.5 * np.einsum("cmkj,abim->abcijk", I2C_vooo, T.ab, optimize=True)
    dT.abb += np.einsum("abej,ecik->abcijk", I2B_vvvo, T.ab, optimize=True)
    dT.abb -= np.einsum("mbij,acmk->abcijk", I2B_ovoo, T.ab, optimize=True)
    # (HBar*T3)_C
    dT.abb -= 0.25 * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.abb, optimize=True)
    dT.abb -= 0.5 * np.einsum("mj,abcimk->abcijk", H.b.oo, T.abb, optimize=True)
    dT.abb += 0.25 * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.abb, optimize=True)
    dT.abb += 0.5 * np.einsum("be,aecijk->abcijk", H.b.vv, T.abb, optimize=True)
    dT.abb += 0.125 * np.einsum("mnjk,abcimn->abcijk", H.bb.oooo, T.abb, optimize=True)
    dT.abb += 0.5 * np.einsum("mnij,abcmnk->abcijk", H.ab.oooo, T.abb, optimize=True)
    dT.abb += 0.125 * np.einsum("bcef,aefijk->abcijk", H.bb.vvvv, T.abb, optimize=True)
    dT.abb += 0.5 * np.einsum("abef,efcijk->abcijk", H.ab.vvvv, T.abb, optimize=True)
    dT.abb += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.aa.voov, T.abb, optimize=True)
    dT.abb += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.ab.voov, T.bbb, optimize=True)
    dT.abb += np.einsum("mbej,aecimk->abcijk", H.ab.ovvo, T.aab, optimize=True)
    dT.abb += np.einsum("bmje,aecimk->abcijk", H.bb.voov, T.abb, optimize=True)
    dT.abb -= 0.5 * np.einsum("mbie,aecmjk->abcijk", H.ab.ovov, T.abb, optimize=True)
    dT.abb -= 0.5 * np.einsum("amej,ebcimk->abcijk", H.ab.vovo, T.abb, optimize=True)
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
    I2C_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vvov -= np.einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb, optimize=True)
    I2C_vvov += H.bb.vvov
    
    I2C_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vooo += np.einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb, optimize=True)
    I2C_vooo -= np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    I2C_vooo += H.bb.vooo

    # MM(2,3)D
    dT.bbb = -0.25 * np.einsum("amij,bcmk->abcijk", I2C_vooo, T.bb, optimize=True)
    dT.bbb += 0.25 * np.einsum("abie,ecjk->abcijk", I2C_vvov, T.bb, optimize=True)
    # (HBar*T3)_C
    dT.bbb -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.b.oo, T.bbb, optimize=True)
    dT.bbb += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.b.vv, T.bbb, optimize=True)
    dT.bbb += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.bb.oooo, T.bbb, optimize=True)
    dT.bbb += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.bb.vvvv, T.bbb, optimize=True)
    dT.bbb += 0.25 * np.einsum("maei,ebcmjk->abcijk", H.ab.ovvo, T.abb, optimize=True)
    dT.bbb += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.bb.voov, T.bbb, optimize=True)
    T.bbb, dT.bbb = cc_loops2.update_t3d_v2(
        T.bbb, 
        dT.bbb, 
        H0.b.oo, 
        H0.b.vv, 
        shift,
    )
    return T, dT
