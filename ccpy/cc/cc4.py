"""Module with functions that perform the approximate CC method
with singles, doubles, triples and quadruples (CC4)."""
import numpy as np

from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.hbar.hbar_cc4 import get_cc4_intermediates
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

    # Get T1-transformed intermediates for CC4
    HT1 = get_cc4_intermediates(T, H)
    # Build T4 contributions to T2 and T3 residuals
    dT = add_t4_aaaa_contributions(T, dT, HT1, H, X)
    dT = add_t4_aaab_contributions(T, dT, HT1, H, X)
    dT = add_t4_aabb_contributions(T, dT, HT1, H, X)

    # update T2
    T, dT = update_t2a(T, dT, X, H, shift)
    T, dT = update_t2b(T, dT, X, H, shift)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        pass

    # CCSD intermediates
    # [TODO]: Should accept CCS HBar as input and build only terms with T2 in it
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
        pass

    return T, dT

def update_t1a(T, dT, H, X, shift):
    """
    Update t1a amplitudes by calculating the projection <ia|(H_N exp(T1+T2+T3))_C|0>.
    """
    dT.a = -np.einsum("mi,am->ai", X.a.oo, T.a, optimize=True)
    dT.a += np.einsum("ae,ei->ai", X.a.vv, T.a, optimize=True)
    dT.a += np.einsum("me,aeim->ai", X.a.ov, T.aa, optimize=True)  # [+]
    dT.a += np.einsum("me,aeim->ai", X.b.ov, T.ab, optimize=True)  # [+]
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
    I2A_vooo = H.aa.vooo + 0.5 * np.einsum('anef,efij->anij', H0.aa.vovv + 0.5 * H.aa.vovv, T.aa, optimize=True)

    tau = 0.5 * T.aa + np.einsum('ai,bj->abij', T.a, T.a, optimize=True)

    dT.aa -= 0.5 * np.einsum("amij,bm->abij", I2A_vooo, T.a, optimize=True)
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
    # T4 parts
    # dT.aa += (1.0 / 4.0) * 0.25 * np.einsum("mnef,abefijmn->abij", H0.aa.oovv, t4["aaaa"], optimize=True)
    # dT.aa += (1.0 / 4.0) * np.einsum("mnef,abefijmn->abij", H0.ab.oovv, t4["aaab"], optimize=True)
    # dT.aa += (1.0 / 4.0) * 0.25 * np.einsum("mnef,abefijmn->abij", H0.bb.oovv, t4["aabb"], optimize=True)

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

    dT.ab -= np.einsum("mbij,am->abij", I2B_ovoo, T.a, optimize=True)
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
    # T4 parts
    # dT.ab += 0.25 * np.einsum("mnef,aefbimnj->abij", H0.aa.oovv, t4["aaab"], optimize=True)
    # dT.ab += np.einsum("mnef,aefbimnj->abij", H0.ab.oovv, t4["aabb"], optimize=True)
    # dT.ab += 0.25 * np.einsum("mnef,abefijmn->abij", H0.bb.oovv, t4["abbb"], optimize=True)

    T.ab, dT.ab = cc_loops2.update_t2b(
        T.ab, dT.ab + H0.ab.vvoo, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv, shift
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
    dT.aaa -= 0.25 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.aa, optimize=True)
    dT.aaa += 0.25 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.aa, optimize=True)
    # (HBar*T3)_C
    dT.aaa -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.a.oo, T.aaa, optimize=True)
    dT.aaa += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.a.vv, T.aaa, optimize=True)
    dT.aaa += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa, optimize=True)
    dT.aaa += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa, optimize=True)
    dT.aaa += 0.25 * np.einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa, optimize=True)
    dT.aaa += 0.25 * np.einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab, optimize=True)
    # (HBar*T4)_C
    ## dT.aaa += (1.0 / 36.0) * np.einsum("me,abceijkm->abcijk", H.a.ov, t4["aaaa"], optimize=True) # (1) = 1
    ## dT.aaa += (1.0 / 36.0) * np.einsum("me,abceijkm->abcijk", H.b.ov, t4["aaab"], optimize=True) # (1) = 1
    # dT.aaa += (1.0 / 24.0) * np.einsum("cnef,abefijkn->abcijk", H.aa.vovv, t4["aaaa"], optimize=True) # (c/ab) = 3
    # dT.aaa += (1.0 / 12.0) * np.einsum("cnef,abefijkn->abcijk", H.ab.vovv, t4["aaab"], optimize=True) # (c/ab) = 3
    # dT.aaa -= (1.0 / 24.0) * np.einsum("mnkf,abcfijmn->abcijk", H.aa.ooov, t4["aaaa"], optimize=True) # (k/ij) = 3
    # dT.aaa -= (1.0 / 12.0) * np.einsum("mnkf,abcfijmn->abcijk", H.ab.ooov, t4["aaab"], optimize=True) # (k/ij) = 3

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
    dT.aab += 0.5 * np.einsum("bcek,aeij->abcijk", I2B_vvvo, T.aa, optimize=True)
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
    # (HBar*T4)_C
    ## dT.aab += 0.25 * np.einsum("me,abecijmk->abcijk", H.a.ov, t4["aaab"], optimize=True) # (1) = 1
    ## dT.aab += 0.25 * np.einsum("me,abecijmk->abcijk", H.b.ov, t4["aabb"], optimize=True) # (1) = 1
    # dT.aab -= 0.25 * np.einsum("mnjf,abfcimnk->abcijk", H.aa.ooov, t4["aaab"], optimize=True) # (ij) = 2
    # dT.aab -= 0.5 * np.einsum("mnjf,abfcimnk->abcijk", H.ab.ooov, t4["aabb"], optimize=True) # (ij) = 2
    # dT.aab -= 0.25 * np.einsum("nmfk,abfcijnm->abcijk", H.ab.oovo, t4["aaab"], optimize=True) # (1) = 1
    # dT.aab -= 0.125 * np.einsum("mnkf,abfcijnm->abcijk", H.bb.ooov, t4["aabb"], optimize=True) # (1) = 1
    # dT.aab += 0.25 * np.einsum("bnef,aefcijnk->abcijk", H.aa.vovv, t4["aaab"], optimize=True) # (ab) = 2
    # dT.aab += 0.5 * np.einsum("bnef,aefcijnk->abcijk", H.ab.vovv, t4["aabb"], optimize=True) # (ab) = 2
    # dT.aab += 0.25 * np.einsum("ncfe,abfeijnk->abcijk", H.ab.ovvv, t4["aaab"], optimize=True) # (1) = 1
    # dT.aab += 0.125 * np.einsum("cnef,abfeijnk->abcijk", H.bb.vovv, t4["aabb"], optimize=True) # (1) = 1

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

def compute_t4(T, HT1, H0, flag_RHF=True):
    t4 = {}
    t4["aaaa"] = update_t4a(T, HT1, H0)
    t4["aaab"] = update_t4b(T, HT1, H0)
    t4["aabb"] = update_t4c(T, HT1, H0)
    # using RHF symmetry
    if flag_RHF:
        t4["abbb"] = np.transpose(t4["aaab"], (3, 2, 1, 0, 7, 6, 5, 4))
        t4["bbbb"] = t4["aaaa"].copy()
    return t4

def t4_aaaa_ijkl(i, j, k, l, HT1, T):
    # Diagram 1: -A(jl/i/k)A(bc/a/d) h2a(amie) * t2a(bcmk) * t2a(edjl) -> A(i/jl)A(k/ijl)
    # (1)
    x4a = -0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, i, :], T.aa[:, :, :, k], T.aa[:, :, j, l], optimize=True) # (1)
    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, j, :], T.aa[:, :, :, k], T.aa[:, :, i, l], optimize=True) # (ij)
    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, l, :], T.aa[:, :, :, k], T.aa[:, :, j, i], optimize=True) # (il)
    # (ik)
    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, k, :], T.aa[:, :, :, i], T.aa[:, :, j, l], optimize=True) # (ik)
    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, j, :], T.aa[:, :, :, i], T.aa[:, :, k, l], optimize=True) # (ij)(ik)
    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, l, :], T.aa[:, :, :, i], T.aa[:, :, j, k], optimize=True) # (il)(ik)
    # (jk)
    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, i, :], T.aa[:, :, :, j], T.aa[:, :, k, l], optimize=True) # (jk)
    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, k, :], T.aa[:, :, :, j], T.aa[:, :, i, l], optimize=True) # (ij)(jk)
    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, l, :], T.aa[:, :, :, j], T.aa[:, :, k, i], optimize=True) # (il)(jk)
    # (kl)
    x4a += 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, i, :], T.aa[:, :, :, l], T.aa[:, :, j, k], optimize=True) # (kl)
    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, j, :], T.aa[:, :, :, l], T.aa[:, :, i, k], optimize=True) # (ij)(kl)
    x4a -= 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, k, :], T.aa[:, :, :, l], T.aa[:, :, j, i], optimize=True) # (il)(kl)
    # Diagram 2: A(ij/kl)A(bc/ad) h2a(mnij) * t2a(adml) * t2a(bcnk) [checked]
    x4a += 0.25 * np.einsum("mn,adm,bcn->abcd", HT1["aa"]["oooo"][:, :, i, j], T.aa[:, :, :, l], T.aa[:, :, :, k], optimize=True)  # (1)
    x4a -= 0.25 * np.einsum("mn,adm,bcn->abcd", HT1["aa"]["oooo"][:, :, k, j], T.aa[:, :, :, l], T.aa[:, :, :, i], optimize=True)  # (ik)
    x4a -= 0.25 * np.einsum("mn,adm,bcn->abcd", HT1["aa"]["oooo"][:, :, l, j], T.aa[:, :, :, i], T.aa[:, :, :, k], optimize=True)  # (il)
    x4a -= 0.25 * np.einsum("mn,adm,bcn->abcd", HT1["aa"]["oooo"][:, :, i, k], T.aa[:, :, :, l], T.aa[:, :, :, j], optimize=True)  # (jk)
    x4a -= 0.25 * np.einsum("mn,adm,bcn->abcd", HT1["aa"]["oooo"][:, :, i, l], T.aa[:, :, :, j], T.aa[:, :, :, k], optimize=True)  # (jl)
    x4a += 0.25 * np.einsum("mn,adm,bcn->abcd", HT1["aa"]["oooo"][:, :, k, l], T.aa[:, :, :, j], T.aa[:, :, :, i], optimize=True)  # (ik)(jl)
    # Diagram 3: A(jk/il)A(ab/cd) h2a(abef) * t2a(fcjk) * t2a(edil) [checked]
    x4a += 0.25 * np.einsum("abef,fc,ed->abcd", HT1["aa"]["vvvv"], T.aa[:, :, j, k], T.aa[:, :, i, l], optimize=True)  # (1)
    x4a -= 0.25 * np.einsum("abef,fc,ed->abcd", HT1["aa"]["vvvv"], T.aa[:, :, i, k], T.aa[:, :, j, l], optimize=True)  # (ij)
    x4a -= 0.25 * np.einsum("abef,fc,ed->abcd", HT1["aa"]["vvvv"], T.aa[:, :, l, k], T.aa[:, :, i, j], optimize=True)  # (jl)
    x4a -= 0.25 * np.einsum("abef,fc,ed->abcd", HT1["aa"]["vvvv"], T.aa[:, :, j, i], T.aa[:, :, k, l], optimize=True)  # (ik)
    x4a -= 0.25 * np.einsum("abef,fc,ed->abcd", HT1["aa"]["vvvv"], T.aa[:, :, j, l], T.aa[:, :, i, k], optimize=True)  # (kl)
    x4a += 0.25 * np.einsum("abef,fc,ed->abcd", HT1["aa"]["vvvv"], T.aa[:, :, i, l], T.aa[:, :, j, k], optimize=True)  # (ij)(kl)
    # Diagram 4: A(ab/cd)A(k/ijl) h2a(cdke) * t3a(abeijl), weight = 6/24 = (1.0 / 4.0) [checked]
    x4a += (1.0 / 4.0) * np.einsum("cde,abe->abcd", HT1["aa"]["vvov"][:, :, k, :], T.aaa[:, :, :, i, j, l], optimize=True) # (1)
    x4a -= (1.0 / 4.0) * np.einsum("cde,abe->abcd", HT1["aa"]["vvov"][:, :, i, :], T.aaa[:, :, :, k, j, l], optimize=True) # (ik)
    x4a -= (1.0 / 4.0) * np.einsum("cde,abe->abcd", HT1["aa"]["vvov"][:, :, j, :], T.aaa[:, :, :, i, k, l], optimize=True) # (jk)
    x4a -= (1.0 / 4.0) * np.einsum("cde,abe->abcd", HT1["aa"]["vvov"][:, :, l, :], T.aaa[:, :, :, i, j, k], optimize=True) # (kl)
    # Diagram 5: -A(c/abd)A(kl/ij) h2a(cmkl) * t3a(abdijm), weight = 4/24 = (1.0 / 6.0) [checked]
    x4a -= (1.0 / 6.0) * np.einsum("cm,abdm->abcd", HT1["aa"]["vooo"][:, :, k, l], T.aaa[:, :, :, i, j, :], optimize=True) # (1)
    x4a += (1.0 / 6.0) * np.einsum("cm,abdm->abcd", HT1["aa"]["vooo"][:, :, i, l], T.aaa[:, :, :, k, j, :], optimize=True) # (ik)
    x4a += (1.0 / 6.0) * np.einsum("cm,abdm->abcd", HT1["aa"]["vooo"][:, :, j, l], T.aaa[:, :, :, i, k, :], optimize=True) # (jk)
    x4a += (1.0 / 6.0) * np.einsum("cm,abdm->abcd", HT1["aa"]["vooo"][:, :, k, i], T.aaa[:, :, :, l, j, :], optimize=True) # (il)
    x4a += (1.0 / 6.0) * np.einsum("cm,abdm->abcd", HT1["aa"]["vooo"][:, :, k, j], T.aaa[:, :, :, i, l, :], optimize=True) # (jl)
    x4a -= (1.0 / 6.0) * np.einsum("cm,abdm->abcd", HT1["aa"]["vooo"][:, :, i, j], T.aaa[:, :, :, k, l, :], optimize=True) # (ik)(jl)

    # antisymmetrize A(abcd)
    x4a -= np.transpose(x4a, (1, 0, 2, 3)) + np.transpose(x4a, (2, 1, 0, 3)) + np.transpose(x4a, (3, 1, 2, 0))  # (a/bcd)
    x4a -= np.transpose(x4a, (0, 2, 1, 3)) + np.transpose(x4a, (0, 3, 2, 1))  # (b/cd)
    x4a -= np.transpose(x4a, (0, 1, 3, 2))  # (cd)

    for a in range(x4a.shape[0]):
        x4a[a, a, :, :] *= 0.0
        x4a[a, :, a, :] *= 0.0
        x4a[a, :, :, a] *= 0.0
        x4a[:, a, a, :] *= 0.0
        x4a[:, a, :, a] *= 0.0
        x4a[:, :, a, a] *= 0.0
        x4a[a, a, a, :] *= 0.0
        x4a[a, a, :, a] *= 0.0
        x4a[a, :, a, a] *= 0.0
        x4a[:, a, a, a] *= 0.0
        x4a[a, a, a, a] *= 0.0
    return x4a

def t4_aaab_ijkl(i, j, k, l, HT1, T):
    # Diagram 1:  -A(i/jk)A(c/ab) h2b(mdel) * t2a(abim) * t2a(ecjk)
    x4b = -0.5 * np.einsum("mde,abm,ec->abcd", HT1["ab"]["ovvo"][:, :, :, l], T.aa[:, :, i, :], T.aa[:, :, j, k], optimize=True) # (1)
    x4b += 0.5 * np.einsum("mde,abm,ec->abcd", HT1["ab"]["ovvo"][:, :, :, l], T.aa[:, :, j, :], T.aa[:, :, i, k], optimize=True) # (ij)
    x4b += 0.5 * np.einsum("mde,abm,ec->abcd", HT1["ab"]["ovvo"][:, :, :, l], T.aa[:, :, k, :], T.aa[:, :, j, i], optimize=True) # (ik)
    # Diagram 2:   A(k/ij)A(a/bc) h2a(mnij) * t2a(bcnk) * t2b(adml)
    x4b += 0.5 * np.einsum("mn,bcn,adm->abcd", HT1["aa"]["oooo"][:, :, i, j], T.aa[:, :, :, k], T.ab[:, :, :, l], optimize=True) # (1)
    x4b -= 0.5 * np.einsum("mn,bcn,adm->abcd", HT1["aa"]["oooo"][:, :, k, j], T.aa[:, :, :, i], T.ab[:, :, :, l], optimize=True) # (ik)
    x4b -= 0.5 * np.einsum("mn,bcn,adm->abcd", HT1["aa"]["oooo"][:, :, i, k], T.aa[:, :, :, j], T.ab[:, :, :, l], optimize=True) # (jk)
    # Diagram 3:  -A(ijk)A(c/ab) h2b(mdjf) * t2a(abim) * t2b(cfkl)
    x4b -= 0.5 * np.einsum("mdf,abm,cf->abcd", HT1["ab"]["ovov"][:, :, j, :], T.aa[:, :, i, :], T.ab[:, :, k, l], optimize=True) # (1)
    x4b += 0.5 * np.einsum("mdf,abm,cf->abcd", HT1["ab"]["ovov"][:, :, i, :], T.aa[:, :, j, :], T.ab[:, :, k, l], optimize=True) # (ij)
    x4b += 0.5 * np.einsum("mdf,abm,cf->abcd", HT1["ab"]["ovov"][:, :, j, :], T.aa[:, :, k, :], T.ab[:, :, i, l], optimize=True) # (ik)
    x4b += 0.5 * np.einsum("mdf,abm,cf->abcd", HT1["ab"]["ovov"][:, :, k, :], T.aa[:, :, i, :], T.ab[:, :, j, l], optimize=True) # (jk)
    x4b -= 0.5 * np.einsum("mdf,abm,cf->abcd", HT1["ab"]["ovov"][:, :, i, :], T.aa[:, :, k, :], T.ab[:, :, j, l], optimize=True) # (ij)(jk)
    x4b -= 0.5 * np.einsum("mdf,abm,cf->abcd", HT1["ab"]["ovov"][:, :, k, :], T.aa[:, :, j, :], T.ab[:, :, i, l], optimize=True) # (ik)(jk)
    # Diagram 4:  -A(ijk)A(abc) h2b(amie) * t2b(bejl) * t2b(cdkm)
    x4b -= np.einsum("ame,be,cdm->abcd", HT1["ab"]["voov"][:, :, i, :], T.ab[:, :, j, l], T.ab[:, :, k, :], optimize=True) # (1)
    x4b += np.einsum("ame,be,cdm->abcd", HT1["ab"]["voov"][:, :, j, :], T.ab[:, :, i, l], T.ab[:, :, k, :], optimize=True) # (ij)
    x4b += np.einsum("ame,be,cdm->abcd", HT1["ab"]["voov"][:, :, k, :], T.ab[:, :, j, l], T.ab[:, :, i, :], optimize=True) # (ik)
    x4b += np.einsum("ame,be,cdm->abcd", HT1["ab"]["voov"][:, :, i, :], T.ab[:, :, k, l], T.ab[:, :, j, :], optimize=True) # (jk)
    x4b -= np.einsum("ame,be,cdm->abcd", HT1["ab"]["voov"][:, :, k, :], T.ab[:, :, i, l], T.ab[:, :, j, :], optimize=True) # (ij)(jk)
    x4b -= np.einsum("ame,be,cdm->abcd", HT1["ab"]["voov"][:, :, j, :], T.ab[:, :, k, l], T.ab[:, :, i, :], optimize=True) # (ik)(jk)
    # Diagram 5:   A(ijk)A(a/bc) h2b(mnjl) * t2a(bcmk) * t2b(adin)
    x4b += 0.5 * np.einsum("mn,bcm,adn->abcd", HT1["ab"]["oooo"][:, :, j, l], T.aa[:, :, :, k], T.ab[:, :, i, :], optimize=True) # (1)
    x4b -= 0.5 * np.einsum("mn,bcm,adn->abcd", HT1["ab"]["oooo"][:, :, i, l], T.aa[:, :, :, k], T.ab[:, :, j, :], optimize=True) # (ij)
    x4b -= 0.5 * np.einsum("mn,bcm,adn->abcd", HT1["ab"]["oooo"][:, :, j, l], T.aa[:, :, :, i], T.ab[:, :, k, :], optimize=True) # (ik)
    x4b -= 0.5 * np.einsum("mn,bcm,adn->abcd", HT1["ab"]["oooo"][:, :, k, l], T.aa[:, :, :, j], T.ab[:, :, i, :], optimize=True) # (jk)
    x4b += 0.5 * np.einsum("mn,bcm,adn->abcd", HT1["ab"]["oooo"][:, :, i, l], T.aa[:, :, :, j], T.ab[:, :, k, :], optimize=True) # (ij)(jk)
    x4b += 0.5 * np.einsum("mn,bcm,adn->abcd", HT1["ab"]["oooo"][:, :, k, l], T.aa[:, :, :, i], T.ab[:, :, j, :], optimize=True) # (ik)(jk)
    # Diagram 6:  -A(i/jk)A(abc) h2b(bmel) * t2a(ecjk) * t2b(adim)
    x4b -= np.einsum("bme,ec,adm->abcd", HT1["ab"]["vovo"][:, :, :, l], T.aa[:, :, j, k], T.ab[:, :, i, :], optimize=True) # (1)
    x4b += np.einsum("bme,ec,adm->abcd", HT1["ab"]["vovo"][:, :, :, l], T.aa[:, :, i, k], T.ab[:, :, j, :], optimize=True) # (ij)
    x4b += np.einsum("bme,ec,adm->abcd", HT1["ab"]["vovo"][:, :, :, l], T.aa[:, :, j, i], T.ab[:, :, k, :], optimize=True) # (ik)
    # Diagram 7:  -A(i/jk)A(abc) h2a(amie) * t2a(ecjk) * t2b(bdml)
    x4b -= np.einsum("ame,ec,bdm->abcd", HT1["aa"]["voov"][:, :, i, :], T.aa[:, :, j, k], T.ab[:, :, :, l], optimize=True) # (1)
    x4b += np.einsum("ame,ec,bdm->abcd", HT1["aa"]["voov"][:, :, j, :], T.aa[:, :, i, k], T.ab[:, :, :, l], optimize=True) # (ij)
    x4b += np.einsum("ame,ec,bdm->abcd", HT1["aa"]["voov"][:, :, k, :], T.aa[:, :, j, i], T.ab[:, :, :, l], optimize=True) # (ik)
    # Diagram 8:   A(i/jk)A(c/ab) h2a(abef) * t2a(fcjk) * t2b(edil)
    x4b += 0.5 * np.einsum("abef,fc,ed->abcd", HT1["aa"]["vvvv"], T.aa[:, :, j, k], T.ab[:, :, i, l], optimize=True) # (1)
    x4b -= 0.5 * np.einsum("abef,fc,ed->abcd", HT1["aa"]["vvvv"], T.aa[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (ij)
    x4b -= 0.5 * np.einsum("abef,fc,ed->abcd", HT1["aa"]["vvvv"], T.aa[:, :, j, i], T.ab[:, :, k, l], optimize=True) # (ik)
    # Diagram 9:  -A(ijk)A(a/bc) h2a(amie) * t2a(bcmk) * t2b(edjl)
    x4b -= 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, i, :], T.aa[:, :, :, k], T.ab[:, :, j, l], optimize=True) # (1)
    x4b += 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, j, :], T.aa[:, :, :, k], T.ab[:, :, i, l], optimize=True) # (ij)
    x4b += 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, k, :], T.aa[:, :, :, i], T.ab[:, :, j, l], optimize=True) # (ik)
    x4b += 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, i, :], T.aa[:, :, :, j], T.ab[:, :, k, l], optimize=True) # (jk)
    x4b -= 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, k, :], T.aa[:, :, :, j], T.ab[:, :, i, l], optimize=True) # (ij)(jk)
    x4b -= 0.5 * np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, j, :], T.aa[:, :, :, i], T.ab[:, :, k, l], optimize=True) # (ik)(jk)
    # Diagram 10:  A(k/ij)A(abc) h2b(adef) * t2a(ebij) * t2b(cfkl)
    x4b += np.einsum("adef,eb,cf->abcd", HT1["ab"]["vvvv"], T.aa[:, :, i, j], T.ab[:, :, k, l], optimize=True) # (1)
    x4b -= np.einsum("adef,eb,cf->abcd", HT1["ab"]["vvvv"], T.aa[:, :, k, j], T.ab[:, :, i, l], optimize=True) # (ik)
    x4b -= np.einsum("adef,eb,cf->abcd", HT1["ab"]["vvvv"], T.aa[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (jk)
    # Diagram 11: -A(k/ij) h2b(mdkl) * t3a(abcijm)
    x4b -= (1.0 / 6.0) * np.einsum("md,abcm->abcd", HT1["ab"]["ovoo"][:, :, k, l], T.aaa[:, :, :, i, j, :], optimize=True) # (1)
    x4b += (1.0 / 6.0) * np.einsum("md,abcm->abcd", HT1["ab"]["ovoo"][:, :, i, l], T.aaa[:, :, :, k, j, :], optimize=True) # (ik)
    x4b += (1.0 / 6.0) * np.einsum("md,abcm->abcd", HT1["ab"]["ovoo"][:, :, j, l], T.aaa[:, :, :, i, k, :], optimize=True) # (jk)
    # Diagram 12: -A(j/ik)A(a/bc) h2a(amik) * t3b(bcdjml)
    x4b -= 0.5 * np.einsum("am,bcdm->abcd", HT1["aa"]["vooo"][:, :, i, k], T.aab[:, :, :, j, :, l], optimize=True) # (1)
    x4b += 0.5 * np.einsum("am,bcdm->abcd", HT1["aa"]["vooo"][:, :, j, k], T.aab[:, :, :, i, :, l], optimize=True) # (ij)
    x4b += 0.5 * np.einsum("am,bcdm->abcd", HT1["aa"]["vooo"][:, :, i, j], T.aab[:, :, :, k, :, l], optimize=True) # (jk)
    # Diagram 13: -A(a/bc)A(i/jk) h2b(amil) * t3b(bcdjkm)
    x4b -= 0.5 * np.einsum("am,bcdm->abcd", HT1["ab"]["vooo"][:, :, i, l], T.aab[:, :, :, j, k, :], optimize=True) # (1)
    x4b += 0.5 * np.einsum("am,bcdm->abcd", HT1["ab"]["vooo"][:, :, j, l], T.aab[:, :, :, i, k, :], optimize=True) # (ij)
    x4b += 0.5 * np.einsum("am,bcdm->abcd", HT1["ab"]["vooo"][:, :, k, l], T.aab[:, :, :, j, i, :], optimize=True) # (ik)
    # Diagram 14: A(c/ab) h2b(cdel) * t3a(abeijk)
    x4b += 0.5 * np.einsum("cde,abe->abcd", HT1["ab"]["vvvo"][:, :, :, l], T.aaa[:, :, :, i, j, k], optimize=True) # (1)
    # Diagram 15: A(b/ac)A(i/jk) h2a(acie) * t3b(bedjkl)
    x4b += 0.5 * np.einsum("ace,bed->abcd", HT1["aa"]["vvov"][:, :, i, :], T.aab[:, :, :, j, k, l], optimize=True) # (1)
    x4b -= 0.5 * np.einsum("ace,bed->abcd", HT1["aa"]["vvov"][:, :, j, :], T.aab[:, :, :, i, k, l], optimize=True) # (ij)
    x4b -= 0.5 * np.einsum("ace,bed->abcd", HT1["aa"]["vvov"][:, :, k, :], T.aab[:, :, :, j, i, l], optimize=True) # (ik)
    # Diagram 16: A(a/bc)A(i/jk) h2b(adie) * t3b(bcejkl)
    x4b += 0.5 * np.einsum("ade,bce->abcd", HT1["ab"]["vvov"][:, :, i, :], T.aab[:, :, :, j, k, l], optimize=True) # (1)
    x4b -= 0.5 * np.einsum("ade,bce->abcd", HT1["ab"]["vvov"][:, :, j, :], T.aab[:, :, :, i, k, l], optimize=True) # (ij)
    x4b -= 0.5 * np.einsum("ade,bce->abcd", HT1["ab"]["vvov"][:, :, k, :], T.aab[:, :, :, j, i, l], optimize=True) # (ik)

    # Antisymmetrize
    x4b -= np.transpose(x4b, (0, 2, 1, 3)) # (bc)
    x4b -= np.transpose(x4b, (2, 1, 0, 3)) + np.transpose(x4b, (1, 0, 2, 3)) # (a/bc)

    for a in range(x4b.shape[0]):
        x4b[a, a, a, :] *= 0.0
        x4b[a, a, :, :] *= 0.0
        x4b[a, :, a, :] *= 0.0
        x4b[:, a, a, :] *= 0.0
    return x4b

def t4_aabb_ijkl(i, j, k, l, HT1, T):
    # Diagram 1:  -A(ij)A(kl)A(ab)A(cd) h2c(cmke) * t2b(adim) * t2b(bejl)
    x4c = -np.einsum("cme,adm,be->abcd", HT1["bb"]["voov"][:, :, k, :], T.ab[:, :, i, :], T.ab[:, :, j, l], optimize=True) # (1)
    x4c += np.einsum("cme,adm,be->abcd", HT1["bb"]["voov"][:, :, k, :], T.ab[:, :, j, :], T.ab[:, :, i, l], optimize=True) # (ij)
    x4c += np.einsum("cme,adm,be->abcd", HT1["bb"]["voov"][:, :, l, :], T.ab[:, :, i, :], T.ab[:, :, j, k], optimize=True) # (kl)
    x4c -= np.einsum("cme,adm,be->abcd", HT1["bb"]["voov"][:, :, l, :], T.ab[:, :, j, :], T.ab[:, :, i, k], optimize=True) # (ij)(kl)
    # Diagram 2:  -A(ij)A(kl)A(ab)A(cd) h2a(amie) * t2b(bcmk) * t2b(edjl)
    x4c -= np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, i, :], T.ab[:, :, :, k], T.ab[:, :, j, l], optimize=True) # (1)
    x4c += np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, j, :], T.ab[:, :, :, k], T.ab[:, :, i, l], optimize=True) # (ij)
    x4c += np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, i, :], T.ab[:, :, :, l], T.ab[:, :, j, k], optimize=True) # (kl)
    x4c -= np.einsum("ame,bcm,ed->abcd", HT1["aa"]["voov"][:, :, j, :], T.ab[:, :, :, l], T.ab[:, :, i, k], optimize=True) # (ij)(kl)
    # Diagram 3:  -A(kl)A(ab)A(cd) h2b(mcek) * t2a(aeij) * t2b(bdml)
    x4c -= np.einsum("mce,ae,bdm->abcd", HT1["ab"]["ovvo"][:, :, :, k], T.aa[:, :, i, j], T.ab[:, :, :, l], optimize=True) # (1)
    x4c += np.einsum("mce,ae,bdm->abcd", HT1["ab"]["ovvo"][:, :, :, l], T.aa[:, :, i, j], T.ab[:, :, :, k], optimize=True) # (2)
    # Diagram 4:  -A(ij)A(ab)A(cd) h2b(amie) * t2b(bdjm) * t2c(cekl)
    x4c -= np.einsum("ame,bdm,ce->abcd", HT1["ab"]["voov"][:, :, i, :], T.ab[:, :, j, :], T.bb[:, :, k, l], optimize=True) # (1)
    x4c += np.einsum("ame,bdm,ce->abcd", HT1["ab"]["voov"][:, :, j, :], T.ab[:, :, i, :], T.bb[:, :, k, l], optimize=True) # (ij)
    # Diagram 5:  -A(ij)A(kl)A(cd) h2b(mcek) * t2a(abim) * t2b(edjl)
    x4c -= 0.5 * np.einsum("mce,abm,ed->abcd", HT1["ab"]["ovvo"][:, :, :, k], T.aa[:, :, i, :], T.ab[:, :, j, l], optimize=True) # (1)
    x4c += 0.5 * np.einsum("mce,abm,ed->abcd", HT1["ab"]["ovvo"][:, :, :, k], T.aa[:, :, j, :], T.ab[:, :, i, l], optimize=True) # (ij)
    x4c += 0.5 * np.einsum("mce,abm,ed->abcd", HT1["ab"]["ovvo"][:, :, :, l], T.aa[:, :, i, :], T.ab[:, :, j, k], optimize=True) # (kl)
    x4c -= 0.5 * np.einsum("mce,abm,ed->abcd", HT1["ab"]["ovvo"][:, :, :, l], T.aa[:, :, j, :], T.ab[:, :, i, k], optimize=True) # (ij)(kl)
    # Diagram 6:  -A(ij)A(kl)A(ab) h2b(amie) * t2c(cdkm) * t2b(bejl)
    x4c -= 0.5 * np.einsum("ame,cdm,be->abcd", HT1["ab"]["voov"][:, :, i, :], T.bb[:, :, k, :], T.ab[:, :, j, l], optimize=True) # (1)
    x4c += 0.5 * np.einsum("ame,cdm,be->abcd", HT1["ab"]["voov"][:, :, j, :], T.bb[:, :, k, :], T.ab[:, :, i, l], optimize=True) # (ij)
    x4c += 0.5 * np.einsum("ame,cdm,be->abcd", HT1["ab"]["voov"][:, :, i, :], T.bb[:, :, l, :], T.ab[:, :, j, k], optimize=True) # (kl)
    x4c -= 0.5 * np.einsum("ame,cdm,be->abcd", HT1["ab"]["voov"][:, :, j, :], T.bb[:, :, l, :], T.ab[:, :, i, k], optimize=True) # (ij)(kl)
    # Diagram 7:  -A(ij)A(kl)A(ab)A(cd) h2b(bmel) * t2b(adim) * t2b(ecjk)
    x4c -= np.einsum("bme,adm,ec->abcd", HT1["ab"]["vovo"][:, :, :, l], T.ab[:, :, i, :], T.ab[:, :, j, k], optimize=True) # (1)
    x4c += np.einsum("bme,adm,ec->abcd", HT1["ab"]["vovo"][:, :, :, l], T.ab[:, :, j, :], T.ab[:, :, i, k], optimize=True) # (ij)
    x4c += np.einsum("bme,adm,ec->abcd", HT1["ab"]["vovo"][:, :, :, k], T.ab[:, :, i, :], T.ab[:, :, j, l], optimize=True) # (kl)
    x4c -= np.einsum("bme,adm,ec->abcd", HT1["ab"]["vovo"][:, :, :, k], T.ab[:, :, j, :], T.ab[:, :, i, l], optimize=True) # (ij)(kl)
    # Diagram 8:  -A(ij)A(kl)A(ab)A(cd) h2b(mdje) * t2b(bcmk) * t2b(aeil)
    x4c -= np.einsum("mde,bcm,ae->abcd", HT1["ab"]["ovov"][:, :, j, :], T.ab[:, :, :, k], T.ab[:, :, i, l], optimize=True) # (1)
    x4c += np.einsum("mde,bcm,ae->abcd", HT1["ab"]["ovov"][:, :, i, :], T.ab[:, :, :, k], T.ab[:, :, j, l], optimize=True) # (ij)
    x4c += np.einsum("mde,bcm,ae->abcd", HT1["ab"]["ovov"][:, :, j, :], T.ab[:, :, :, l], T.ab[:, :, i, k], optimize=True) # (kl)
    x4c -= np.einsum("mde,bcm,ae->abcd", HT1["ab"]["ovov"][:, :, i, :], T.ab[:, :, :, l], T.ab[:, :, j, k], optimize=True) # (ij)(kl)
    # Diagram 9:  -A(ij)A(cd) h2b(mdje) * t2a(abim) * t2c(cekl)
    x4c -= 0.5 * np.einsum("mde,abm,ce->abcd", HT1["ab"]["ovov"][:, :, j, :], T.aa[:, :, i, :], T.bb[:, :, k, l], optimize=True) # (1)
    x4c += 0.5 * np.einsum("mde,abm,ce->abcd", HT1["ab"]["ovov"][:, :, i, :], T.aa[:, :, j, :], T.bb[:, :, k, l], optimize=True) # (ij)
    # Diagram 10: -A(kl)A(ab) h2b(bmel) * t2a(aeij) * t2c(cdkm)
    x4c -= 0.5 * np.einsum("bme,ae,cdm->abcd", HT1["ab"]["vovo"][:, :, :, l], T.aa[:, :, i, j], T.bb[:, :, k, :], optimize=True) # (1)
    x4c += 0.5 * np.einsum("bme,ae,cdm->abcd", HT1["ab"]["vovo"][:, :, :, k], T.aa[:, :, i, j], T.bb[:, :, l, :], optimize=True) # (kl)
    # Diagram 11:  A(kl)A(ab) h2a(mnij) * t2b(acmk) * t2b(bdnl)
    x4c += 0.5 * np.einsum("mn,acm,bdn->abcd", HT1["aa"]["oooo"][:, :, i, j], T.ab[:, :, :, k], T.ab[:, :, :, l], optimize=True) # (1)
    x4c -= 0.5 * np.einsum("mn,acm,bdn->abcd", HT1["aa"]["oooo"][:, :, i, j], T.ab[:, :, :, l], T.ab[:, :, :, k], optimize=True) # (kl)
    # Diagram 12:  A(ij)A(kl) h2a(abef) * t2b(ecik) * t2b(fdjl)
    x4c += 0.25 * np.einsum("abef,ec,fd->abcd", HT1["aa"]["vvvv"], T.ab[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (1)
    x4c -= 0.25 * np.einsum("abef,ec,fd->abcd", HT1["aa"]["vvvv"], T.ab[:, :, j, k], T.ab[:, :, i, l], optimize=True) # (ij)
    x4c -= 0.25 * np.einsum("abef,ec,fd->abcd", HT1["aa"]["vvvv"], T.ab[:, :, i, l], T.ab[:, :, j, k], optimize=True) # (kl)
    x4c += 0.25 * np.einsum("abef,ec,fd->abcd", HT1["aa"]["vvvv"], T.ab[:, :, j, l], T.ab[:, :, i, k], optimize=True) # (ij)(kl)
    # Diagram 13:  A(ij)A(kl) h2b(mnik) * t2a(abmj) * t2c(cdnl)
    x4c += 0.25 * np.einsum("mn,abm,cdn->abcd", HT1["ab"]["oooo"][:, :, i, k], T.aa[:, :, :, j], T.bb[:, :, :, l], optimize=True) # (1)
    x4c -= 0.25 * np.einsum("mn,abm,cdn->abcd", HT1["ab"]["oooo"][:, :, j, k], T.aa[:, :, :, i], T.bb[:, :, :, l], optimize=True) # (ij)
    x4c -= 0.25 * np.einsum("mn,abm,cdn->abcd", HT1["ab"]["oooo"][:, :, i, l], T.aa[:, :, :, j], T.bb[:, :, :, k], optimize=True) # (kl)
    x4c += 0.25 * np.einsum("mn,abm,cdn->abcd", HT1["ab"]["oooo"][:, :, j, l], T.aa[:, :, :, i], T.bb[:, :, :, k], optimize=True) # (ij)(kl)
    # Diagram 14:  A(ab)A(cd) h2b(acef) * t2a(ebij) * t2c(fdkl)
    x4c += np.einsum("acef,eb,fd->abcd", HT1["ab"]["vvvv"], T.aa[:, :, i, j], T.bb[:, :, k, l], optimize=True) # (1)
    # Diagram 15:  A(ij)A(kl)A(ab)A(cd) h2b(mnik) * t2b(adml) * t2b(bcjn)
    x4c += np.einsum("mn,adm,bcn->abcd", HT1["ab"]["oooo"][:, :, i, k], T.ab[:, :, :, l], T.ab[:, :, j, :], optimize=True) # (1)
    x4c -= np.einsum("mn,adm,bcn->abcd", HT1["ab"]["oooo"][:, :, j, k], T.ab[:, :, :, l], T.ab[:, :, i, :], optimize=True) # (ij)
    x4c -= np.einsum("mn,adm,bcn->abcd", HT1["ab"]["oooo"][:, :, i, l], T.ab[:, :, :, k], T.ab[:, :, j, :], optimize=True) # (kl)
    x4c += np.einsum("mn,adm,bcn->abcd", HT1["ab"]["oooo"][:, :, j, l], T.ab[:, :, :, k], T.ab[:, :, i, :], optimize=True) # (ij)(kl)
    # Diagram 16:  A(ij)A(kl)A(ab)A(cd) h2b(acef) * t2b(edil) * t2b(bfjk)
    x4c += np.einsum("acef,ed,bf->abcd", HT1["ab"]["vvvv"], T.ab[:, :, i, l], T.ab[:, :, j, k], optimize=True) # (1)
    x4c -= np.einsum("acef,ed,bf->abcd", HT1["ab"]["vvvv"], T.ab[:, :, j, l], T.ab[:, :, i, k], optimize=True) # (ij)
    x4c -= np.einsum("acef,ed,bf->abcd", HT1["ab"]["vvvv"], T.ab[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (kl)
    x4c += np.einsum("acef,ed,bf->abcd", HT1["ab"]["vvvv"], T.ab[:, :, j, k], T.ab[:, :, i, l], optimize=True) # (ij)(kl)
    # Diagram 17:  A(ij)A(cd) h2c(mnkl) * t2b(adin) * t2b(bcjm)
    x4c += 0.5 * np.einsum("mn,adn,bcm->abcd", HT1["bb"]["oooo"][:, :, k, l], T.ab[:, :, i, :], T.ab[:, :, j, :], optimize=True) # (1)
    x4c -= 0.5 * np.einsum("mn,adn,bcm->abcd", HT1["bb"]["oooo"][:, :, k, l], T.ab[:, :, j, :], T.ab[:, :, i, :], optimize=True) # (ij)
    # Diagram 18:  A(ij)A(kl) h2c(cdef) * t2b(afil) * t2b(bejk)
    x4c += 0.25 * np.einsum("cdef,af,be->abcd", HT1["bb"]["vvvv"], T.ab[:, :, i, l], T.ab[:, :, j, k], optimize=True) # (1)
    x4c -= 0.25 * np.einsum("cdef,af,be->abcd", HT1["bb"]["vvvv"], T.ab[:, :, j, l], T.ab[:, :, i, k], optimize=True) # (ij)
    x4c -= 0.25 * np.einsum("cdef,af,be->abcd", HT1["bb"]["vvvv"], T.ab[:, :, i, k], T.ab[:, :, j, l], optimize=True) # (kl)
    x4c += 0.25 * np.einsum("cdef,af,be->abcd", HT1["bb"]["vvvv"], T.ab[:, :, j, k], T.ab[:, :, i, l], optimize=True) # (ij)(kl)
    # Diagram 19: -A(ij)A(kl)A(cd) h2b(mdil) * t3b(abcmjk)
    x4c -= 0.5 * np.einsum("md,abcm->abcd", HT1["ab"]["ovoo"][:, :, i, l], T.aab[:, :, :, :, j, k], optimize=True) # (1)
    x4c += 0.5 * np.einsum("md,abcm->abcd", HT1["ab"]["ovoo"][:, :, j, l], T.aab[:, :, :, :, i, k], optimize=True) # (ij)
    x4c += 0.5 * np.einsum("md,abcm->abcd", HT1["ab"]["ovoo"][:, :, i, k], T.aab[:, :, :, :, j, l], optimize=True) # (kl)
    x4c -= 0.5 * np.einsum("md,abcm->abcd", HT1["ab"]["ovoo"][:, :, j, k], T.aab[:, :, :, :, i, l], optimize=True) # (ij)(kl)
    # Diagram 20: -A(ab) h2a(bmji) * t3c(acdmkl)
    x4c -= 0.5 * np.einsum("bm,acdm->abcd", HT1["aa"]["vooo"][:, :, j, i], T.abb[:, :, :, :, k, l], optimize=True) # (1)
    # Diagram 21: -A(cd) h2c(cmkl) * t3b(abdijm)
    x4c -= 0.5 * np.einsum("cm,abdm->abcd", HT1["bb"]["vooo"][:, :, k, l], T.aab[:, :, :, i, j, :], optimize=True) # (1)
    # Diagram 22: -A(ij)A(ab)A(kl) h2b(amil) * t3c(bcdjkm)
    x4c -= 0.5 * np.einsum("am,bcdm->abcd", HT1["ab"]["vooo"][:, :, i, l], T.abb[:, :, :, j, k, :], optimize=True) # (1)
    x4c += 0.5 * np.einsum("am,bcdm->abcd", HT1["ab"]["vooo"][:, :, j, l], T.abb[:, :, :, i, k, :], optimize=True) # (ij)
    x4c += 0.5 * np.einsum("am,bcdm->abcd", HT1["ab"]["vooo"][:, :, i, k], T.abb[:, :, :, j, l, :], optimize=True) # (kl)
    x4c -= 0.5 * np.einsum("am,bcdm->abcd", HT1["ab"]["vooo"][:, :, j, k], T.abb[:, :, :, i, l, :], optimize=True) # (ij)(kl)
    # Diagram 23: A(ab)A(kl)A(cd) h2b(adel) * t3b(becjik)
    x4c += np.einsum("ade,bec->abcd", HT1["ab"]["vvvo"][:, :, :, l], T.aab[:, :, :, j, i, k], optimize=True) # (1)
    x4c -= np.einsum("ade,bec->abcd", HT1["ab"]["vvvo"][:, :, :, k], T.aab[:, :, :, j, i, l], optimize=True) # (kl)
    # Diagram 24: A(ij) h2a(baje) * t3c(ecdikl)
    x4c += 0.25 * np.einsum("bae,ecd->abcd", HT1["aa"]["vvov"][:, :, j, :], T.abb[:, :, :, i, k, l], optimize=True) # (1)
    x4c -= 0.25 * np.einsum("bae,ecd->abcd", HT1["aa"]["vvov"][:, :, i, :], T.abb[:, :, :, j, k, l], optimize=True) # (ij)
    # Diagram 25: A(ij)A(ab)A(cd) h2b(adie) * t3c(bcejkl)
    x4c += np.einsum("ade,bce->abcd", HT1["ab"]["vvov"][:, :, i, :], T.abb[:, :, :, j, k, l], optimize=True) # (1)
    x4c -= np.einsum("ade,bce->abcd", HT1["ab"]["vvov"][:, :, j, :], T.abb[:, :, :, i, k, l], optimize=True) # (ij)
    # Diagram 26: A(kl) h2c(cdke) * t3b(abeijl)
    x4c += 0.25 * np.einsum("cde,abe->abcd", HT1["bb"]["vvov"][:, :, k, :], T.aab[:, :, :, i, j, l], optimize=True) # (1)
    x4c -= 0.25 * np.einsum("cde,abe->abcd", HT1["bb"]["vvov"][:, :, l, :], T.aab[:, :, :, i, j, k], optimize=True) # (kl)

    # antisymmetrize
    x4c -= np.transpose(x4c, (1, 0, 2, 3)) # (ab)
    x4c -= np.transpose(x4c, (0, 1, 3, 2)) # (cd)
    for a in range(x4c.shape[0]):
        x4c[a, a, :, :] *= 0.0
    for a in range(x4c.shape[2]):
        x4c[:, :, a, a] *= 0.0
    return x4c

def update_t4a(T, HT1, H0):
    oa = np.diagonal(H0.a.oo)
    va = np.diagonal(H0.a.vv)
    n = np.newaxis
    e_abcdijkl = 1.0 / (- va[:, n, n, n, n, n, n, n] - va[n, :, n, n, n, n, n, n] - va[n, n, :, n, n, n, n, n] - va[n, n, n, :, n, n, n, n]
                        + oa[n, n, n, n, :, n, n, n] + oa[n, n, n, n, n, :, n, n] + oa[n, n, n, n, n, n, :, n] + oa[n, n, n, n, n, n, n, :])

    # <ijklabcd | H(2) | 0 >
    t4a = -(144.0 / 576.0) * np.einsum("amie,bcmk,edjl->abcdijkl", HT1["aa"]["voov"], T.aa, T.aa, optimize=True)  # (jl/i/k)(bc/a/d) = 12 * 12 = 144
    t4a += (36.0 / 576.0) * np.einsum("mnij,adml,bcnk->abcdijkl", HT1["aa"]["oooo"], T.aa, T.aa, optimize=True)   # (ij/kl)(bc/ad) = 6 * 6 = 36
    t4a += (36.0 / 576.0) * np.einsum("abef,fcjk,edil->abcdijkl", HT1["aa"]["vvvv"], T.aa, T.aa, optimize=True)   # (jk/il)(ab/cd) = 6 * 6 = 36

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    t4a += (24.0 / 576.0) * np.einsum("cdke,abeijl->abcdijkl", HT1["aa"]["vvov"], T.aaa, optimize=True) # (cd/ab)(k/ijl) = 6 * 4 = 24
    t4a -= (24.0 / 576.0) * np.einsum("cmkl,abdijm->abcdijkl", HT1["aa"]["vooo"], T.aaa, optimize=True) # (c/abd)(kl/ij) = 6 * 4 = 24

    # Divide by MP denominator
    t4a -= np.transpose(t4a, (0, 1, 2, 3, 4, 6, 5, 7)) # (jk)
    t4a -= np.transpose(t4a, (0, 1, 2, 3, 4, 7, 6, 5)) + np.transpose(t4a, (0, 1, 2, 3, 4, 5, 7, 6)) # (l/jk)
    t4a -= np.transpose(t4a, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(t4a, (0, 1, 2, 3, 6, 5, 4, 7)) + np.transpose(t4a, (0, 1, 2, 3, 7, 5, 6, 4)) # (i/jkl)

    t4a -= np.transpose(t4a, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    t4a -= np.transpose(t4a, (0, 3, 2, 1, 4, 5, 6, 7)) + np.transpose(t4a, (0, 1, 3, 2, 4, 5, 6, 7)) # (d/bc)
    t4a -= np.transpose(t4a, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(t4a, (2, 1, 0, 3, 4, 5, 6, 7)) + np.transpose(t4a, (3, 1, 2, 0, 4, 5, 6, 7)) # (a/bcd)
    return t4a * e_abcdijkl

def update_t4b(T, HT1, H0):
    oa = np.diagonal(H0.a.oo)
    ob = np.diagonal(H0.b.oo)
    va = np.diagonal(H0.a.vv)
    vb = np.diagonal(H0.b.vv)
    n = np.newaxis
    e_abcdijkl = 1.0 / (- va[:, n, n, n, n, n, n, n] - va[n, :, n, n, n, n, n, n] - va[n, n, :, n, n, n, n, n] - vb[n, n, n, :, n, n, n, n]
                        + oa[n, n, n, n, :, n, n, n] + oa[n, n, n, n, n, :, n, n] + oa[n, n, n, n, n, n, :, n] + ob[n, n, n, n, n, n, n, :])

    # <ijklabcd | H(2) | 0 >
    t4b = -(9.0 / 36.0) * np.einsum("mdel,abim,ecjk->abcdijkl", HT1["ab"]["ovvo"], T.aa, T.aa, optimize=True)    # (i/jk)(c/ab) = 9
    t4b += (9.0 / 36.0) * np.einsum("mnij,bcnk,adml->abcdijkl", HT1["aa"]["oooo"], T.aa, T.ab, optimize=True)    # (k/ij)(a/bc) = 9
    t4b -= (18.0 / 36.0) * np.einsum("mdjf,abim,cfkl->abcdijkl", HT1["ab"]["ovov"], T.aa, T.ab, optimize=True)   # (ijk)(c/ab) = (i/jk)(c/ab)(jk) = 18
    t4b -= np.einsum("amie,bejl,cdkm->abcdijkl", HT1["ab"]["voov"], T.ab, T.ab, optimize=True)                   # (ijk)(abc) = (i/jk)(a/bc)(jk)(bc) = 36
    t4b += (18.0 / 36.0) * np.einsum("mnjl,bcmk,adin->abcdijkl", HT1["ab"]["oooo"], T.aa, T.ab, optimize=True)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    t4b -= (18.0 / 36.0) * np.einsum("bmel,ecjk,adim->abcdijkl", HT1["ab"]["vovo"], T.aa, T.ab, optimize=True)   # (i/jk)(abc) = (i/jk)(a/bc)(bc) = 18
    t4b -= (18.0 / 36.0) * np.einsum("amie,ecjk,bdml->abcdijkl", HT1["aa"]["voov"], T.aa, T.ab, optimize=True)   # (i/kj)(abc) = (i/kj)(a/bc)(bc) = 18
    t4b += (9.0 / 36.0) * np.einsum("abef,fcjk,edil->abcdijkl", HT1["aa"]["vvvv"], T.aa, T.ab, optimize=True)    # (i/jk)(c/ab) = (i/jk)(c/ab) = 9
    t4b -= (18.0 / 36.0) * np.einsum("amie,bcmk,edjl->abcdijkl", HT1["aa"]["voov"], T.aa, T.ab, optimize=True)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    t4b += (18.0 / 36.0) * np.einsum("adef,ebij,cfkl->abcdijkl", HT1["ab"]["vvvv"], T.aa, T.ab, optimize=True)   # (k/ij)(abc) = (k/ij)(a/bc)(bc) = 18
    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    t4b -= (1.0 / 12.0) * np.einsum("mdkl,abcijm->abcdijkl", HT1["ab"]["ovoo"], T.aaa, optimize=True)  # (k/ij) = 3
    t4b -= (9.0 / 36.0) * np.einsum("amik,bcdjml->abcdijkl", HT1["aa"]["vooo"], T.aab, optimize=True)  # (j/ik)(a/bc) = 9
    t4b -= (9.0 / 36.0) * np.einsum("amil,bcdjkm->abcdijkl", HT1["ab"]["vooo"], T.aab, optimize=True)  # (a/bc)(i/jk) = 9
    t4b += (1.0 / 12.0) * np.einsum("cdel,abeijk->abcdijkl", HT1["ab"]["vvvo"], T.aaa, optimize=True)  # (c/ab) = 3
    t4b += (9.0 / 36.0) * np.einsum("acie,bedjkl->abcdijkl", HT1["aa"]["vvov"], T.aab, optimize=True)  # (b/ac)(i/jk) = 9
    t4b += (9.0 / 36.0) * np.einsum("adie,bcejkl->abcdijkl", HT1["ab"]["vvov"], T.aab, optimize=True)  # (a/bc)(i/jk) = 9

    # Divide by MP denominator
    t4b -= np.transpose(t4b, (0, 1, 2, 3, 4, 6, 5, 7))  # (jk)
    t4b -= np.transpose(t4b, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(t4b, (0, 1, 2, 3, 6, 5, 4, 7)) # (i/jk)
    t4b -= np.transpose(t4b, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    t4b -= np.transpose(t4b, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(t4b, (2, 1, 0, 3, 4, 5, 6, 7)) # (a/bc)
    return t4b * e_abcdijkl

def update_t4c(T, HT1, H0):
    oa = np.diagonal(H0.a.oo)
    ob = np.diagonal(H0.b.oo)
    va = np.diagonal(H0.a.vv)
    vb = np.diagonal(H0.b.vv)
    n = np.newaxis
    e_abcdijkl = 1.0 / (- va[:, n, n, n, n, n, n, n] - va[n, :, n, n, n, n, n, n] - vb[n, n, :, n, n, n, n, n] - vb[n, n, n, :, n, n, n, n]
                        + oa[n, n, n, n, :, n, n, n] + oa[n, n, n, n, n, :, n, n] + ob[n, n, n, n, n, n, :, n] + ob[n, n, n, n, n, n, n, :])
    # <ijklabcd | H(2) | 0 >
    t4c = -np.einsum("cmke,adim,bejl->abcdijkl", HT1["bb"]["voov"], T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    t4c -= np.einsum("amie,bcmk,edjl->abcdijkl", HT1["aa"]["voov"], T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    t4c -= 0.5 * np.einsum("mcek,aeij,bdml->abcdijkl", HT1["ab"]["ovvo"], T.aa, T.ab, optimize=True)    # (kl)(ab)(cd) = 8
    t4c -= 0.5 * np.einsum("amie,bdjm,cekl->abcdijkl", HT1["ab"]["voov"], T.ab, T.bb, optimize=True)    # (ij)(ab)(cd) = 8
    t4c -= 0.5 * np.einsum("mcek,abim,edjl->abcdijkl", HT1["ab"]["ovvo"], T.aa, T.ab, optimize=True)    # (ij)(kl)(cd) = 8
    t4c -= 0.5 * np.einsum("amie,cdkm,bejl->abcdijkl", HT1["ab"]["voov"], T.bb, T.ab, optimize=True)    # (ij)(kl)(ab) = 8
    t4c -= np.einsum("bmel,adim,ecjk->abcdijkl", HT1["ab"]["vovo"], T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    t4c -= np.einsum("mdje,bcmk,aeil->abcdijkl", HT1["ab"]["ovov"], T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    t4c -= 0.25 * np.einsum("mdje,abim,cekl->abcdijkl", HT1["ab"]["ovov"], T.aa, T.bb, optimize=True)   # (ij)(cd) = 4
    t4c -= 0.25 * np.einsum("bmel,cdkm,aeij->abcdijkl", HT1["ab"]["vovo"], T.bb, T.aa, optimize=True)   # (kl)(ab) = 4
    t4c += 0.25 * np.einsum("mnij,acmk,bdnl->abcdijkl", HT1["aa"]["oooo"], T.ab, T.ab, optimize=True)   # (kl)(ab) = 4 !!! (tricky asym)
    t4c += 0.25 * np.einsum("abef,ecik,fdjl->abcdijkl", HT1["aa"]["vvvv"], T.ab, T.ab, optimize=True)   # (ij)(kl) = 4 !!! (tricky asym)
    t4c += 0.25 * np.einsum("mnik,abmj,cdnl->abcdijkl", HT1["ab"]["oooo"], T.aa, T.bb, optimize=True)   # (ij)(kl) = 4
    t4c += 0.25 * np.einsum("acef,ebij,fdkl->abcdijkl", HT1["ab"]["vvvv"], T.aa, T.bb, optimize=True)   # (ab)(cd) = 4
    t4c += np.einsum("mnik,adml,bcjn->abcdijkl", HT1["ab"]["oooo"], T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    t4c += np.einsum("acef,edil,bfjk->abcdijkl", HT1["ab"]["vvvv"], T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    t4c += 0.25 * np.einsum("mnkl,adin,bcjm->abcdijkl", HT1["bb"]["oooo"], T.ab, T.ab, optimize=True)   # (ij)(cd) = 4 !!! (tricky asym)
    t4c += 0.25 * np.einsum("cdef,afil,bejk->abcdijkl", HT1["bb"]["vvvv"], T.ab, T.ab, optimize=True)   # (ij)(kl) = 4 !!! (tricky asym)
    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    t4c -= (8.0 / 16.0) * np.einsum("mdil,abcmjk->abcdijkl", HT1["ab"]["ovoo"], T.aab, optimize=True)  # [1]  (ij)(kl)(cd) = 8
    t4c -= (2.0 / 16.0) * np.einsum("bmji,acdmkl->abcdijkl", HT1["aa"]["vooo"], T.abb, optimize=True)  # [2]  (ab) = 2
    t4c -= (2.0 / 16.0) * np.einsum("cmkl,abdijm->abcdijkl", HT1["bb"]["vooo"], T.aab, optimize=True)  # [3]  (cd) = 2
    t4c -= (8.0 / 16.0) * np.einsum("amil,bcdjkm->abcdijkl", HT1["ab"]["vooo"], T.abb, optimize=True)  # [4]  (ij)(ab)(kl) = 8
    t4c += (8.0 / 16.0) * np.einsum("adel,becjik->abcdijkl", HT1["ab"]["vvvo"], T.aab, optimize=True)  # [5]  (ab)(kl)(cd) = 8
    t4c += (2.0 / 16.0) * np.einsum("baje,ecdikl->abcdijkl", HT1["aa"]["vvov"], T.abb, optimize=True)  # [6]  (ij) = 2
    t4c += (8.0 / 16.0) * np.einsum("adie,bcejkl->abcdijkl", HT1["ab"]["vvov"], T.abb, optimize=True)  # [7]  (ij)(ab)(cd) = 8
    t4c += (2.0 / 16.0) * np.einsum("cdke,abeijl->abcdijkl", HT1["bb"]["vvov"], T.aab, optimize=True)  # [8]  (kl) = 2

    # Divide by MP denominator
    t4c -= np.transpose(t4c, (1, 0, 2, 3, 4, 5, 6, 7)) # (ab)
    t4c -= np.transpose(t4c, (0, 1, 3, 2, 4, 5, 6, 7)) # (cd)
    t4c -= np.transpose(t4c, (0, 1, 2, 3, 5, 4, 6, 7)) # (ij)
    t4c -= np.transpose(t4c, (0, 1, 2, 3, 4, 5, 7, 6)) # (kl)
    return t4c * e_abcdijkl

def add_t4_aaaa_contributions(T, dT, HT1, H0, X):
    nua, nub, noa, nob = T.ab.shape

    va = np.diagonal(H0.a.vv)
    n = np.newaxis
    e_abcd = -va[:, n, n, n] - va[n, :, n, n] - va[n, n, :, n] - va[n, n, n, :]
    
    # Allocate residuals
    dT.aa = np.zeros((nua, nua, noa, noa))
    dT.aaa = np.zeros((nua, nua, nua, noa, noa, noa))
    #
    # Exact expressions
    #
    # exact_aa = 0.25 * np.einsum("mnef,abefijmn->abij", H0.aa.oovv, t4["aaaa"], optimize=True)
    # exact_aaa = (
    #                (1.0 / 24.0) * np.einsum("cnef,abefijkn->abcijk", H0.aa.vovv, t4["aaaa"], optimize=True) # (c/ab) = 3
    #                - (1.0 / 24.0) * np.einsum("mnkf,abcfijmn->abcijk", H0.aa.ooov, t4["aaaa"], optimize=True) # (k/ij) = 3
    # )
    # exact_aaa -= np.transpose(exact_aaa, (0, 1, 2, 3, 5, 4)) # (jk)
    # exact_aaa -= np.transpose(exact_aaa, (0, 1, 2, 4, 3, 5)) + np.transpose(exact_aaa, (0, 1, 2, 5, 4, 3)) # (i/jk)
    # exact_aaa -= np.transpose(exact_aaa, (0, 2, 1, 3, 4, 5)) # (bc)
    # exact_aaa -= np.transpose(exact_aaa, (2, 1, 0, 3, 4, 5)) + np.transpose(exact_aaa, (1, 0, 2, 3, 4, 5)) # (a/bc)
    #
    for i in range(noa):
        for j in range(i + 1, noa):
            for k in range(j + 1, noa):
                for l in range(k + 1, noa):
                    denom_occ = H0.a.oo[i, i] + H0.a.oo[j, j] + H0.a.oo[k, k] + H0.a.oo[l, l]
                    t4a = t4_aaaa_ijkl(i, j, k, l, HT1, T)
                    t4a /= (denom_occ + e_abcd)
                    #
                    # Debug t4a
                    #
                    # print('error in t4a = ', np.linalg.norm(t4a.flatten() - t4["aaaa"][:, :, :, :, i, j, k, l].flatten()))
                    # print(i, j, k, l)
                    # print('----------')
                    # for a in range(nua):
                    #    for b in range(a + 1, nua):
                    #        for c in range(b + 1, nua):
                    #            for d in range(c + 1, nua):
                    #                if abs(t4["aaaa"][a, b, c, d, i, j, k, l] - t4a[a, b, c, d]) > 1.0e-08:
                    #                    print(a, b, c, d, "expected:", t4["aaaa"][a, b, c, d, i, j, k, l], "got:", t4a[a, b, c, d])
                    #
                    # A(ij/kl) x(abij) <- h2a(klef)*t4a(abefijkl)
                    dT.aa[:, :, i, j] += 0.25 * np.einsum("ef,abef->ab", H0.aa.oovv[k, l, :, :], t4a, optimize=True) # (1)
                    dT.aa[:, :, k, j] -= 0.25 * np.einsum("ef,abef->ab", H0.aa.oovv[i, l, :, :], t4a, optimize=True) # (ik)
                    dT.aa[:, :, l, j] -= 0.25 * np.einsum("ef,abef->ab", H0.aa.oovv[k, i, :, :], t4a, optimize=True) # (il)
                    dT.aa[:, :, i, k] -= 0.25 * np.einsum("ef,abef->ab", H0.aa.oovv[j, l, :, :], t4a, optimize=True) # (jk)
                    dT.aa[:, :, i, l] -= 0.25 * np.einsum("ef,abef->ab", H0.aa.oovv[k, j, :, :], t4a, optimize=True) # (jl)
                    dT.aa[:, :, k, l] += 0.25 * np.einsum("ef,abef->ab", H0.aa.oovv[i, j, :, :], t4a, optimize=True) # (ik)(jl)
                    # A(l/ijk) x(abcijk) <- h2a(:lef)*t4a(abefijkl)
                    dT.aaa[:, :, :, i, j, k] += 0.25 * np.einsum("cef,abef->abc", H0.aa.vovv[:, l, :, :] + X.aa.vovv[:, l, :, :], t4a, optimize=True) # (1)
                    dT.aaa[:, :, :, l, j, k] -= 0.25 * np.einsum("cef,abef->abc", H0.aa.vovv[:, i, :, :] + X.aa.vovv[:, i, :, :], t4a, optimize=True) # (il)
                    dT.aaa[:, :, :, i, l, k] -= 0.25 * np.einsum("cef,abef->abc", H0.aa.vovv[:, j, :, :] + X.aa.vovv[:, j, :, :], t4a, optimize=True) # (jl)
                    dT.aaa[:, :, :, i, j, l] -= 0.25 * np.einsum("cef,abef->abc", H0.aa.vovv[:, k, :, :] + X.aa.vovv[:, k, :, :], t4a, optimize=True) # (kl)
                    # A(ij/kl) x(abcijk) <- -h2a(klmd)*t4a(abcdijkl)
                    dT.aaa[:, :, :, i, j, :] -= (1.0 / 6.0) * np.einsum("md,abcd->abcm", H0.aa.ooov[k, l, :, :] + X.aa.ooov[k, l, :, :], t4a, optimize=True) # (1)
                    dT.aaa[:, :, :, k, j, :] += (1.0 / 6.0) * np.einsum("md,abcd->abcm", H0.aa.ooov[i, l, :, :] + X.aa.ooov[i, l, :, :], t4a, optimize=True) # (ik)
                    dT.aaa[:, :, :, l, j, :] += (1.0 / 6.0) * np.einsum("md,abcd->abcm", H0.aa.ooov[k, i, :, :] + X.aa.ooov[k, i, :, :], t4a, optimize=True) # (il)
                    dT.aaa[:, :, :, i, k, :] += (1.0 / 6.0) * np.einsum("md,abcd->abcm", H0.aa.ooov[j, l, :, :] + X.aa.ooov[j, l, :, :], t4a, optimize=True) # (jk)
                    dT.aaa[:, :, :, i, l, :] += (1.0 / 6.0) * np.einsum("md,abcd->abcm", H0.aa.ooov[k, j, :, :] + X.aa.ooov[k, j, :, :], t4a, optimize=True) # (jl)
                    dT.aaa[:, :, :, k, l, :] -= (1.0 / 6.0) * np.einsum("md,abcd->abcm", H0.aa.ooov[i, j, :, :] + X.aa.ooov[i, j, :, :], t4a, optimize=True) # (ik)(jl)
    # antisymmetrize
    # dT.aa -= np.transpose(dT.aa, (1, 0, 2, 3)) # (ab)
    # dT.aa -= np.transpose(dT.aa, (0, 1, 3, 2)) # (ij)
    # #
    # dT.aaa -= np.transpose(dT.aaa, (0, 1, 2, 3, 5, 4)) # (jk)
    # dT.aaa -= np.transpose(dT.aaa, (0, 1, 2, 4, 3, 5)) + np.transpose(dT.aaa, (0, 1, 2, 5, 4, 3)) # (i/jk)
    # dT.aaa -= np.transpose(dT.aaa, (0, 2, 1, 3, 4, 5)) # (bc)
    # dT.aaa -= np.transpose(dT.aaa, (2, 1, 0, 3, 4, 5)) + np.transpose(dT.aaa, (1, 0, 2, 3, 4, 5)) # (a/bc)
    # clear diagonal elements
    for a in range(nua):
        dT.aa[a, a, :, :] *= 0.0
        dT.aaa[a, a, a, :, :, :] *= 0.0
        dT.aaa[a, a, :, :, :, :] *= 0.0
        dT.aaa[a, :, a, :, :, :] *= 0.0
        dT.aaa[:, a, a, :, :, :] *= 0.0
    for i in range(noa):
        dT.aa[:, :, i, i] *= 0.0
        dT.aaa[:, :, :, i, i, i] *= 0.0
        dT.aaa[:, :, :, i, i, :] *= 0.0
        dT.aaa[:, :, :, i, :, i] *= 0.0
        dT.aaa[:, :, :, :, i, i] *= 0.0
    #
    # Check the results
    #
    # print(np.linalg.norm(exact_aa.flatten() - dT.aa.flatten()))
    # for i in range(noa):
    #    for j in range(i + 1, noa):
    #        for a in range(nua):
    #            for b in range(a + 1, nua):
    #                if abs(dT.aa[a, b, i, j] - exact_aa[a, b, i, j]) > 1.0e-08:
    #                    print(a, b, i, j, "expected:", exact_aa[a, b, i, j], "got:", dT.aa[a, b, i, j], "factor:", exact_aa[a, b, i, j]/dT.aa[a, b, i, j])
    # print(np.linalg.norm(exact_aaa.flatten() - dT.aaa.flatten()))
    # for i in range(noa):
    #    for j in range(i + 1, noa):
    #        for k in range(j + 1, noa):
    #            for a in range(nua):
    #                for b in range(a + 1, nua):
    #                    for c in range(b + 1, nua):
    #                        if abs(dT.aaa[a, b, c, i, j, k] - exact_aaa[a, b, c, i, j, k]) > 1.0e-08:
    #                            print(a, b, c, i, j, k, "expected:", exact_aaa[a, b, c, i, j, k], "got:", dT.aaa[a, b, c, i, j, k], "factor:", exact_aaa[a, b, c, i, j, k]/dT.aaa[a, b, c, i, j, k])
    return dT

def add_t4_aaab_contributions(T, dT, HT1, H0, X):
    nua, nub, noa, nob = T.ab.shape

    va = np.diagonal(H0.a.vv)
    vb = np.diagonal(H0.b.vv)
    n = np.newaxis
    e_abcd = -va[:, n, n, n] - va[n, :, n, n] - va[n, n, :, n] - vb[n, n, n, :]
    # Allocate residuals
    # dT.aa = np.zeros((nua, nua, noa, noa))
    dT.ab = np.zeros((nua, nub, noa, nob))
    # dT.aaa = np.zeros((nua, nua, nua, noa, noa, noa))
    dT.aab = np.zeros((nua, nua, nub, noa, noa, nob))
    #
    # Exact expressions
    #
    # exact_aa = np.einsum("mnef,abefijmn->abij", H0.ab.oovv, t4["aaab"], optimize=True)
    # exact_ab = (
    #             0.25 * np.einsum("mnef,aefbimnj->abij", H0.aa.oovv, t4["aaab"], optimize=True)
    #             + 0.25 * np.einsum("mnef,abefijmn->abij", H0.bb.oovv, t4["abbb"], optimize=True)
    # )
    # exact_aaa = (
    #             (1.0 / 12.0) * np.einsum("cnef,abefijkn->abcijk", H0.ab.vovv, t4["aaab"], optimize=True)  # (c/ab) = 3
    #             - (1.0 / 12.0) * np.einsum("mnkf,abcfijmn->abcijk", H0.ab.ooov, t4["aaab"], optimize=True)  # (k/ij) = 3
    # )
    # exact_aaa -= np.transpose(exact_aaa, (0, 1, 2, 3, 5, 4)) # (jk)
    # exact_aaa -= np.transpose(exact_aaa, (0, 1, 2, 4, 3, 5)) + np.transpose(exact_aaa, (0, 1, 2, 5, 4, 3)) # (i/jk)
    # exact_aaa -= np.transpose(exact_aaa, (0, 2, 1, 3, 4, 5)) # (bc)
    # exact_aaa -= np.transpose(exact_aaa, (2, 1, 0, 3, 4, 5)) + np.transpose(exact_aaa, (1, 0, 2, 3, 4, 5)) # (a/bc)
    #
    # exact_aab = (
    #         - 0.25 * np.einsum("mnjf,abfcimnk->abcijk", H0.aa.ooov, t4["aaab"], optimize=True)  # (ij) = 2
    #         - 0.25 * np.einsum("nmfk,abfcijnm->abcijk", H0.ab.oovo, t4["aaab"], optimize=True)  # (1) = 1
    #         + 0.25 * np.einsum("bnef,aefcijnk->abcijk", H0.aa.vovv, t4["aaab"], optimize=True)  # (ab) = 2
    #         + 0.25 * np.einsum("ncfe,abfeijnk->abcijk", H0.ab.ovvv, t4["aaab"], optimize=True)  # (1) = 1
    # )
    # exact_aab -= np.transpose(exact_aab, (1, 0, 2, 3, 4, 5)) # (ab)
    # exact_aab -= np.transpose(exact_aab, (0, 1, 2, 4, 3, 5)) # (ij)
    #
    for i in range(noa):
        for j in range(i + 1, noa):
            for k in range(j + 1, noa):
                for l in range(nob):
                    denom_occ = H0.a.oo[i, i] + H0.a.oo[j, j] + H0.a.oo[k, k] + H0.b.oo[l, l]
                    t4b = t4_aaab_ijkl(i, j, k, l, HT1, T)
                    t4b /= (denom_occ + e_abcd)
                    #
                    # Debug t4b
                    #
                    # print('error in t4b = ', np.linalg.norm(t4b.flatten() - t4["aaab"][:, :, :, :, i, j, k, l].flatten()))
                    # print(i, j, k, l)
                    # print('----------')
                    # for a in range(nua):
                    #    for b in range(a + 1, nua):
                    #        for c in range(b + 1, nua):
                    #            for d in range(nub):
                    #                if abs(t4["aaab"][a, b, c, d, i, j, k, l] - t4b[a, b, c, d]) > 1.0e-08:
                    #                    print(a, b, c, d, "expected:", t4["aaab"][a, b, c, d, i, j, k, l], "got:", t4b[a, b, c, d])
                    #
                    # A(k/ij) h2b(klef) * t4b(abefijkl) -> x2a(abij)
                    dT.aa[:, :, i, j] += 0.5 * np.einsum("ef,abef->ab", H0.ab.oovv[k, l, :, :], t4b, optimize=True) # (1)
                    dT.aa[:, :, k, j] -= 0.5 * np.einsum("ef,abef->ab", H0.ab.oovv[i, l, :, :], t4b, optimize=True) # (ik)
                    dT.aa[:, :, i, k] -= 0.5 * np.einsum("ef,abef->ab", H0.ab.oovv[j, l, :, :], t4b, optimize=True) # (jk)
                    # A(i/jk) h2a(jkef) * t4b(aefbijkl) -> x2b(abil)
                    dT.ab[:, :, i, l] += 0.5 * np.einsum("ef,aefb->ab", H0.aa.oovv[j, k, :, :], t4b, optimize=True) # (1)
                    dT.ab[:, :, j, l] -= 0.5 * np.einsum("ef,aefb->ab", H0.aa.oovv[i, k, :, :], t4b, optimize=True) # (1)
                    dT.ab[:, :, k, l] -= 0.5 * np.einsum("ef,aefb->ab", H0.aa.oovv[j, i, :, :], t4b, optimize=True) # (1)
                    # h2b(clef) * t4b(abefijkl) -> x3a(abcijk)
                    dT.aaa[:, :, :, i, j, k] += 0.5 * np.einsum("cef,abef->abc", H0.ab.vovv[:, l, :, :] + X.ab.vovv[:, l, :, :], t4b, optimize=True) # (1)
                    # A(k/ij) -h2b(klmf) * t4b(abcfijkl) -> x3a(abcijk)
                    dT.aaa[:, :, :, i, j, :] -= (1.0 / 6.0) * np.einsum("mf,abcf->abcm", H0.ab.ooov[k, l, :, :] + X.ab.ooov[k, l, :, :], t4b, optimize=True) # (1)
                    dT.aaa[:, :, :, k, j, :] += (1.0 / 6.0) * np.einsum("mf,abcf->abcm", H0.ab.ooov[i, l, :, :] + X.ab.ooov[i, l, :, :], t4b, optimize=True) # (ik)
                    dT.aaa[:, :, :, i, k, :] += (1.0 / 6.0) * np.einsum("mf,abcf->abcm", H0.ab.ooov[j, l, :, :] + X.ab.ooov[j, l, :, :], t4b, optimize=True) # (jk)
                    # A(i/jk) -h2a(jkmf) * t4b(abfcijkl) -> x3b(abciml)
                    dT.aab[:, :, :, i, :, l] -= 0.5 * np.einsum("mf,abfc->abcm", H0.aa.ooov[j, k, :, :] + X.aa.ooov[j, k, :, :], t4b, optimize=True) # (1)
                    dT.aab[:, :, :, j, :, l] += 0.5 * np.einsum("mf,abfc->abcm", H0.aa.ooov[i, k, :, :] + X.aa.ooov[i, k, :, :], t4b, optimize=True) # (ij)
                    dT.aab[:, :, :, k, :, l] += 0.5 * np.einsum("mf,abfc->abcm", H0.aa.ooov[j, i, :, :] + X.aa.ooov[j, i, :, :], t4b, optimize=True) # (ik)
                    # A(k/ij) -h2b(klfn) * t4b(abfcijkl) -> x3b(abcijn)
                    dT.aab[:, :, :, i, j, :] -= 0.5 * np.einsum("fn,abfc->abcn", H0.ab.oovo[k, l, :, :] + X.ab.oovo[k, l, :, :], t4b, optimize=True) # (1)
                    dT.aab[:, :, :, k, j, :] += 0.5 * np.einsum("fn,abfc->abcn", H0.ab.oovo[i, l, :, :] + X.ab.oovo[i, l, :, :], t4b, optimize=True) # (ik)
                    dT.aab[:, :, :, i, k, :] += 0.5 * np.einsum("fn,abfc->abcn", H0.ab.oovo[j, l, :, :] + X.ab.oovo[j, l, :, :], t4b, optimize=True) # (jk)
                    # A(k/ij) h2a(bkef) * t4b(aefcijkl) -> x3b(abcijl)
                    dT.aab[:, :, :, i, j, l] += 0.5 * np.einsum("bef,aefc->abc", H0.aa.vovv[:, k, :, :] + X.aa.vovv[:, k, :, :], t4b, optimize=True) # (1)
                    dT.aab[:, :, :, k, j, l] -= 0.5 * np.einsum("bef,aefc->abc", H0.aa.vovv[:, i, :, :] + X.aa.vovv[:, i, :, :], t4b, optimize=True) # (ik)
                    dT.aab[:, :, :, i, k, l] -= 0.5 * np.einsum("bef,aefc->abc", H0.aa.vovv[:, j, :, :] + X.aa.vovv[:, j, :, :], t4b, optimize=True) # (jk)
                    # A(k/ij) h2b(kcfe) * t4b(abfeijkl) -> x3b(abcijl)
                    dT.aab[:, :, :, i, j, l] += 0.5 * np.einsum("cfe,abfe->abc", H0.ab.ovvv[k, :, :, :] + X.ab.ovvv[k, :, :, :], t4b, optimize=True) # (1)
                    dT.aab[:, :, :, k, j, l] -= 0.5 * np.einsum("cfe,abfe->abc", H0.ab.ovvv[i, :, :, :] + X.ab.ovvv[i, :, :, :], t4b, optimize=True) # (ik)
                    dT.aab[:, :, :, i, k, l] -= 0.5 * np.einsum("cfe,abfe->abc", H0.ab.ovvv[j, :, :, :] + X.ab.ovvv[j, :, :, :], t4b, optimize=True) # (jk)
    # Add in the V*T4C part
    dT.ab += dT.ab.transpose(1, 0, 3, 2)
    # antisymmetrize
    # dT.aa -= np.transpose(dT.aa, (1, 0, 2, 3)) # (ab)
    # dT.aa -= np.transpose(dT.aa, (0, 1, 3, 2)) # (ij)
    # #
    # dT.aaa -= np.transpose(dT.aaa, (0, 1, 2, 3, 5, 4)) # (jk)
    # dT.aaa -= np.transpose(dT.aaa, (0, 1, 2, 4, 3, 5)) + np.transpose(dT.aaa, (0, 1, 2, 5, 4, 3)) # (i/jk)
    # dT.aaa -= np.transpose(dT.aaa, (0, 2, 1, 3, 4, 5)) # (bc)
    # dT.aaa -= np.transpose(dT.aaa, (2, 1, 0, 3, 4, 5)) + np.transpose(dT.aaa, (1, 0, 2, 3, 4, 5)) # (a/bc)
    # #
    # dT.aab -= np.transpose(dT.aab, (1, 0, 2, 3, 4, 5)) # (ab)
    # dT.aab -= np.transpose(dT.aab, (0, 1, 2, 4, 3, 5)) # (ij)
    # clear diagonal elements
    for a in range(nua):
        dT.aa[a, a, :, :] *= 0.0
        dT.aaa[a, a, a, :, :, :] *= 0.0
        dT.aaa[a, a, :, :, :, :] *= 0.0
        dT.aaa[a, :, a, :, :, :] *= 0.0
        dT.aaa[:, a, a, :, :, :] *= 0.0
        dT.aab[a, a, :, :, :, :] *= 0.0
    for i in range(noa):
        dT.aa[:, :, i, i] *= 0.0
        dT.aaa[:, :, :, i, i, i] *= 0.0
        dT.aaa[:, :, :, i, i, :] *= 0.0
        dT.aaa[:, :, :, i, :, i] *= 0.0
        dT.aaa[:, :, :, :, i, i] *= 0.0
        dT.aab[:, :, :, i, i, :] *= 0.0
    #
    # Check the results
    #
    # print(np.linalg.norm(exact_aa.flatten() - dT.aa.flatten()))
    # for i in range(noa):
    #    for j in range(i + 1, noa):
    #        for a in range(nua):
    #            for b in range(a + 1, nua):
    #                if abs(dT.aa[a, b, i, j] - exact_aa[a, b, i, j]) > 1.0e-08:
    #                    print(a, b, i, j, "expected:", exact_aa[a, b, i, j], "got:", dT.aa[a, b, i, j], "factor:", exact_aa[a, b, i, j]/dT.aa[a, b, i, j])
    # print(np.linalg.norm(exact_ab.flatten() - dT.ab.flatten()))
    # for i in range(noa):
    #    for j in range(nob):
    #        for a in range(nua):
    #            for b in range(nub):
    #                if abs(dT.ab[a, b, i, j] - exact_ab[a, b, i, j]) > 1.0e-08:
    #                    print(a, b, i, j, "expected:", exact_ab[a, b, i, j], "got:", dT.ab[a, b, i, j], "factor:", exact_ab[a, b, i, j]/dT.ab[a, b, i, j])
    # print(np.linalg.norm(exact_aaa.flatten() - dT.aaa.flatten()))
    # for i in range(noa):
    #    for j in range(i + 1, noa):
    #        for k in range(j + 1, noa):
    #            for a in range(nua):
    #                for b in range(a + 1, nua):
    #                    for c in range(b + 1, nua):
    #                        if abs(dT.aaa[a, b, c, i, j, k] - exact_aaa[a, b, c, i, j, k]) > 1.0e-08:
    #                            print(a, b, c, i, j, k, "expected:", exact_aaa[a, b, c, i, j, k], "got:", dT.aaa[a, b, c, i, j, k], "factor:", exact_aaa[a, b, c, i, j, k]/dT.aaa[a, b, c, i, j, k])
    # print(np.linalg.norm(exact_aab.flatten() - dT.aab.flatten()))
    # for i in range(noa):
    #    for j in range(i + 1, noa):
    #        for k in range(nob):
    #            for a in range(nua):
    #                for b in range(a + 1, nua):
    #                    for c in range(nub):
    #                        if abs(dT.aab[a, b, c, i, j, k] - exact_aab[a, b, c, i, j, k]) > 1.0e-08:
    #                            print(a, b, c, i, j, k, "expected:", exact_aab[a, b, c, i, j, k], "got:", dT.aab[a, b, c, i, j, k], "factor:", exact_aab[a, b, c, i, j, k]/dT.aab[a, b, c, i, j, k])
    return dT 

def add_t4_aabb_contributions(T, dT, HT1, H0, X):
    nua, nub, noa, nob = T.ab.shape

    va = np.diagonal(H0.a.vv)
    vb = np.diagonal(H0.b.vv)
    n = np.newaxis
    e_abcd = -va[:, n, n, n] - va[n, :, n, n] - vb[n, n, :, n] - vb[n, n, n, :]
    # Allocate residuals
    # dT.aa = np.zeros((nua, nua, noa, noa))
    # dT.ab = np.zeros((nua, nub, noa, nob))
    # dT.aab = np.zeros((nua, nua, nub, noa, noa, nob))
    #
    # Exact expressions
    #
    # exact_aa = 0.25 * np.einsum("mnef,abefijmn->abij", H0.bb.oovv, t4["aabb"], optimize=True)
    # exact_ab = (
    #     np.einsum("mnef,aefbimnj->abij", H0.ab.oovv, t4["aabb"], optimize=True)
    # )
    #
    # exact_aab = (
    #         -0.5 * np.einsum("mnjf,abfcimnk->abcijk", H0.ab.ooov, t4["aabb"], optimize=True) # (ij) = 2
    #         -0.125 * np.einsum("mnkf,abfcijnm->abcijk", H0.bb.ooov, t4["aabb"], optimize=True) # (1) = 1
    #         +0.5 * np.einsum("bnef,aefcijnk->abcijk", H0.ab.vovv, t4["aabb"], optimize=True) # (ab) = 2
    #         +0.125 * np.einsum("cnef,abfeijnk->abcijk", H0.bb.vovv, t4["aabb"], optimize=True) # (1) = 1
    # )
    # exact_aab -= np.transpose(exact_aab, (1, 0, 2, 3, 4, 5)) # (ab)
    # exact_aab -= np.transpose(exact_aab, (0, 1, 2, 4, 3, 5)) # (ij)
    #
    for i in range(noa):
        for j in range(i + 1, noa):
            for k in range(nob):
                for l in range(k + 1, nob):
                    denom_occ = H0.a.oo[i, i] + H0.a.oo[j, j] + H0.b.oo[k, k] + H0.b.oo[l, l]
                    t4c = t4_aabb_ijkl(i, j, k, l, HT1, T)
                    t4c /= (denom_occ + e_abcd)
                    #
                    # Debug t4b
                    #
                    # print('error in t4c = ', np.linalg.norm(t4c.flatten() - t4["aabb"][:, :, :, :, i, j, k, l].flatten()))
                    # print(i, j, k, l)
                    # print('----------')
                    # for a in range(nua):
                    #    for b in range(a + 1, nua):
                    #        for c in range(nub):
                    #            for d in range(c + 1, nub):
                    #                if abs(t4["aabb"][a, b, c, d, i, j, k, l] - t4c[a, b, c, d]) > 1.0e-08:
                    #                    print(a, b, c, d, "expected:", t4["aabb"][a, b, c, d, i, j, k, l], "got:", t4c[a, b, c, d])
                    #
                    # h2c(klef) * t4c(abefijkl) -> x2a(abij)
                    dT.aa[:, :, i, j] += 0.25 * np.einsum("ef,abef->ab", H0.bb.oovv[k, l, :, :], t4c, optimize=True) # (1)
                    # A(ij)A(kl) h2b(jkef) * t4c(aefbijkl) -> x2b(abil)
                    dT.ab[:, :, i, l] += np.einsum("ef,aefb->ab", H0.ab.oovv[j, k, :, :], t4c, optimize=True) # (1)
                    dT.ab[:, :, j, l] -= np.einsum("ef,aefb->ab", H0.ab.oovv[i, k, :, :], t4c, optimize=True) # (ij)
                    dT.ab[:, :, i, k] -= np.einsum("ef,aefb->ab", H0.ab.oovv[j, l, :, :], t4c, optimize=True) # (kl)
                    dT.ab[:, :, j, k] += np.einsum("ef,aefb->ab", H0.ab.oovv[i, l, :, :], t4c, optimize=True) # (ij)(kl)
                    # A(ij)A(kl)A(fc) -h2b(jkmf) * t4c(abfcijkl) -> x3b(abciml)
                    dT.aab[:, :, :, i, :, l] -= 0.5 * np.einsum("mf,abfc->abcm", H0.ab.ooov[j, k, :, :] + X.ab.ooov[j, k, :, :], t4c, optimize=True) # (1)
                    dT.aab[:, :, :, j, :, l] += 0.5 * np.einsum("mf,abfc->abcm", H0.ab.ooov[i, k, :, :] + X.ab.ooov[i, k, :, :], t4c, optimize=True) # (ij)
                    dT.aab[:, :, :, i, :, k] += 0.5 * np.einsum("mf,abfc->abcm", H0.ab.ooov[j, l, :, :] + X.ab.ooov[j, l, :, :], t4c, optimize=True) # (kl)
                    dT.aab[:, :, :, j, :, k] -= 0.5 * np.einsum("mf,abfc->abcm", H0.ab.ooov[i, l, :, :] + X.ab.ooov[i, l, :, :], t4c, optimize=True) # (ij)(kl)
                    # -h2c(lknf) * t4c(abfcijkl) -> x3b(abcijn)
                    dT.aab[:, :, :, i, j, :] -= 0.5 * np.einsum("nf,abfc->abcn", H0.bb.ooov[l, k, :, :] + X.bb.ooov[l, k, :, :], t4c, optimize=True)
                    # A(ab)A(kl) h2b(bkef) * t4c(aefcijkl) -> x3b(abcijl)
                    dT.aab[:, :, :, i, j, l] += np.einsum("bef,aefc->abc", H0.ab.vovv[:, k, :, :] + X.ab.vovv[:, k, :, :], t4c, optimize=True) # (1)
                    dT.aab[:, :, :, i, j, k] -= np.einsum("bef,aefc->abc", H0.ab.vovv[:, l, :, :] + X.ab.vovv[:, l, :, :], t4c, optimize=True) # (kl)
                    # A(kl) h2c(ckef) * t4c(abfeijkl) -> x3b(abcijl)
                    dT.aab[:, :, :, i, j, l] += 0.25 * np.einsum("cef,abfe->abc", H0.bb.vovv[:, k, :, :] + X.bb.vovv[:, k, :, :], t4c, optimize=True) # (1)
                    dT.aab[:, :, :, i, j, k] -= 0.25 * np.einsum("cef,abfe->abc", H0.bb.vovv[:, l, :, :] + X.bb.vovv[:, l, :, :], t4c, optimize=True) # (1)
    # antisymmetrize
    # dT.aa -= np.transpose(dT.aa, (1, 0, 2, 3)) # (ab)
    # dT.aa -= np.transpose(dT.aa, (0, 1, 3, 2)) # (ij)
    # #
    # dT.aab -= np.transpose(dT.aab, (1, 0, 2, 3, 4, 5)) # (ab)
    # dT.aab -= np.transpose(dT.aab, (0, 1, 2, 4, 3, 5)) # (ij)
    # clear diagonal elements
    for a in range(nua):
        dT.aa[a, a, :, :] *= 0.0
        dT.aab[a, a, :, :, :, :] *= 0.0
    for i in range(noa):
        dT.aa[:, :, i, i] *= 0.0
        dT.aab[:, :, :, i, i, :] *= 0.0
    #
    # Check the results
    #
    # print(np.linalg.norm(exact_aa.flatten() - dT.aa.flatten()))
    # for i in range(noa):
    #    for j in range(i + 1, noa):
    #        for a in range(nua):
    #            for b in range(a + 1, nua):
    #                if abs(dT.aa[a, b, i, j] - exact_aa[a, b, i, j]) > 1.0e-08:
    #                    print(a, b, i, j, "expected:", exact_aa[a, b, i, j], "got:", dT.aa[a, b, i, j], "factor:", exact_aa[a, b, i, j]/dT.aa[a, b, i, j])
    # print(np.linalg.norm(exact_ab.flatten() - dT.ab.flatten()))
    # for i in range(noa):
    #    for j in range(nob):
    #        for a in range(nua):
    #            for b in range(nub):
    #                if abs(dT.ab[a, b, i, j] - exact_ab[a, b, i, j]) > 1.0e-08:
    #                    print(a, b, i, j, "expected:", exact_ab[a, b, i, j], "got:", dT.ab[a, b, i, j], "factor:", exact_ab[a, b, i, j]/dT.ab[a, b, i, j])
    # print(np.linalg.norm(exact_aab.flatten() - dT.aab.flatten()))
    # for i in range(noa):
    #    for j in range(i + 1, noa):
    #        for k in range(nob):
    #            for a in range(nua):
    #                for b in range(a + 1, nua):
    #                    for c in range(nub):
    #                        if abs(dT.aab[a, b, c, i, j, k] - exact_aab[a, b, c, i, j, k]) > 1.0e-08:
    #                            print(a, b, c, i, j, k, "expected:", exact_aab[a, b, c, i, j, k], "got:", dT.aab[a, b, c, i, j, k], "factor:", exact_aab[a, b, c, i, j, k]/dT.aab[a, b, c, i, j, k])
    return dT
