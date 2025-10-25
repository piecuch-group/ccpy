'''
Coupled-Cluster Method with Singles, Doubles, Triples, and Quadruples (CCSDTQ)
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.hbar.hbar_ccsdt import add_VT3_intermediates
from ccpy.lib.core import cc_loops2
from ccpy.lib.core import cc_loops_t4

#@profile
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

    # add VT3 intermediates to two-body part of HBar
    X = add_VT3_intermediates(T, X, H, flag_RHF)

    # update T4
    T, dT = update_t4a(T, dT, X, H, shift)
    T, dT = update_t4b(T, dT, X, H, shift)
    T, dT = update_t4c(T, dT, X, H, shift)
    if flag_RHF:
        T.abbb = np.transpose(T.aaab, (3, 2, 1, 0, 7, 6, 5, 4))
        dT.abbb = np.transpose(dT.aaab, (3, 2, 1, 0, 7, 6, 5, 4))
        T.bbbb = T.aaaa.copy()
        dT.bbbb = dT.aaaa.copy()
    else:
        T, dT = update_t4d(T, dT, X, H, shift)
        T, dT = update_t4e(T, dT, X, H, shift)

    return T, dT

def update_t1a(T, dT, H, X, shift):
    """
    Update t1a amplitudes by calculating the projection <ia|(H_N e^(T1+T2+T3+T4))_C|0>.
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
    Update t1b amplitudes by calculating the projection <i~a~|(H_N e^(T1+T2+T3+T4))_C|0>.
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
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2+T3+T4))_C|0>.
    """
    # intermediates
    I2A_voov = H.aa.voov + (
        + 0.5 * ccpy_einsum("mnef,afin->amie", H0.aa.oovv, T.aa)
        + ccpy_einsum("mnef,afin->amie", H0.ab.oovv, T.ab)
    )
    I2A_oooo = H.aa.oooo + 0.5 * ccpy_einsum("mnef,efij->mnij", H0.aa.oovv, T.aa)
    I2B_voov = H.ab.voov + 0.5 * ccpy_einsum("mnef,afin->amie", H0.bb.oovv, T.ab)
    I2A_vooo = H.aa.vooo + 0.5 * ccpy_einsum('anef,efij->anij', H0.aa.vovv + 0.5 * H.aa.vovv, T.aa)

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
    # T4 parts
    dT.aa += (1.0 / 4.0) * 0.25 * ccpy_einsum("mnef,abefijmn->abij", H0.aa.oovv, T.aaaa)
    dT.aa += (1.0 / 4.0) * ccpy_einsum("mnef,abefijmn->abij", H0.ab.oovv, T.aaab)
    dT.aa += (1.0 / 4.0) * 0.25 * ccpy_einsum("mnef,abefijmn->abij", H0.bb.oovv, T.aabb)
    T.aa, dT.aa = cc_loops2.update_t2a(
        T.aa,
        dT.aa + 0.25 * H0.aa.vvoo,
        H0.a.oo,
        H0.a.vv,
        shift
    )
    return T, dT

def update_t2b(T, dT, H, H0, shift):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2+T3+T4))_C|0>.
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
    # T4 parts
    dT.ab += 0.25 * ccpy_einsum("mnef,aefbimnj->abij", H0.aa.oovv, T.aaab)
    dT.ab += ccpy_einsum("mnef,aefbimnj->abij", H0.ab.oovv, T.aabb)
    dT.ab += 0.25 * ccpy_einsum("mnef,abefijmn->abij", H0.bb.oovv, T.abbb)
    T.ab, dT.ab = cc_loops2.update_t2b(
        T.ab,
        dT.ab + H0.ab.vvoo,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift
    )
    return T, dT

def update_t2c(T, dT, H, H0, shift):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2+T3+T4))_C|0>.
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
    # T4 parts
    dT.bb += 0.0625 * ccpy_einsum("mnef,abefijmn->abij", H0.bb.oovv, T.bbbb)
    dT.bb += 0.25 * ccpy_einsum("nmfe,febanmji->abij", H0.ab.oovv, T.abbb)
    dT.bb += 0.0625 * ccpy_einsum("mnef,febanmji->abij", H0.aa.oovv, T.aabb)
    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb,
        dT.bb + 0.25 * H0.bb.vvoo,
        H0.b.oo,
        H0.b.vv,
        shift
    )
    return T, dT

def update_t3a(T, dT, H, H0, shift):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3+T4))_C|0>.
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
    dT.aaa -= (1.0 / 12.0) * ccpy_einsum("mk,abcijm->abcijk", H.a.oo, T.aaa) # (k/ij) = 3
    dT.aaa += (1.0 / 12.0) * ccpy_einsum("ce,abeijk->abcijk", H.a.vv, T.aaa) # (c/ab) = 3
    dT.aaa += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa) # (k/ij) = 3
    dT.aaa += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa) # (c/ab) = 3
    dT.aaa += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa) # (c/ij)(k/ij) = 9
    dT.aaa += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab) # (c/ij)(k/ij) = 9
    # (HBar*T4)_C
    dT.aaa += (1.0 / 36.0) * ccpy_einsum("me,abceijkm->abcijk", H.a.ov, T.aaaa) # (1) = 1
    dT.aaa += (1.0 / 36.0) * ccpy_einsum("me,abceijkm->abcijk", H.b.ov, T.aaab) # (1) = 1
    dT.aaa += (1.0 / 24.0) * ccpy_einsum("cnef,abefijkn->abcijk", H.aa.vovv, T.aaaa) # (c/ab) = 3
    dT.aaa += (1.0 / 12.0) * ccpy_einsum("cnef,abefijkn->abcijk", H.ab.vovv, T.aaab) # (c/ab) = 3
    dT.aaa -= (1.0 / 24.0) * ccpy_einsum("mnkf,abcfijmn->abcijk", H.aa.ooov, T.aaaa) # (k/ij) = 3
    dT.aaa -= (1.0 / 12.0) * ccpy_einsum("mnkf,abcfijmn->abcijk", H.ab.ooov, T.aaab) # (k/ij) = 3
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
    Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N e^(T1+T2+T3+T4))_C|0>.
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
    # (HBar*T4)_C
    dT.aab += 0.25 * ccpy_einsum("me,abecijmk->abcijk", H.a.ov, T.aaab) # (1) = 1
    dT.aab += 0.25 * ccpy_einsum("me,abecijmk->abcijk", H.b.ov, T.aabb) # (1) = 1
    dT.aab -= 0.25 * ccpy_einsum("mnjf,abfcimnk->abcijk", H.aa.ooov, T.aaab) # (ij) = 2
    dT.aab -= 0.5 * ccpy_einsum("mnjf,abfcimnk->abcijk", H.ab.ooov, T.aabb) # (ij) = 2
    dT.aab -= 0.25 * ccpy_einsum("nmfk,abfcijnm->abcijk", H.ab.oovo, T.aaab) # (1) = 1
    dT.aab -= 0.125 * ccpy_einsum("mnkf,abfcijnm->abcijk", H.bb.ooov, T.aabb) # (1) = 1
    dT.aab += 0.25 * ccpy_einsum("bnef,aefcijnk->abcijk", H.aa.vovv, T.aaab) # (ab) = 2
    dT.aab += 0.5 * ccpy_einsum("bnef,aefcijnk->abcijk", H.ab.vovv, T.aabb) # (ab) = 2
    dT.aab += 0.25 * ccpy_einsum("ncfe,abfeijnk->abcijk", H.ab.ovvv, T.aaab) # (1) = 1
    dT.aab += 0.125 * ccpy_einsum("cnef,abfeijnk->abcijk", H.bb.vovv, T.aabb) # (1) = 1
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
    Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N e^(T1+T2+T3+T4))_C|0>.
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
    # (HBar*T4)_C
    dT.abb += 0.25 * ccpy_einsum("me,cebakmji->cbakji", H.b.ov, T.abbb)
    dT.abb += 0.25 * ccpy_einsum("me,cebakmji->cbakji", H.a.ov, T.aabb)
    dT.abb -= 0.25 * ccpy_einsum("mnjf,cfbaknmi->cbakji", H.bb.ooov, T.abbb)
    dT.abb -= 0.5 * ccpy_einsum("nmfj,cfbaknmi->cbakji", H.ab.oovo, T.aabb)
    dT.abb -= 0.25 * ccpy_einsum("mnkf,cfbamnji->cbakji", H.ab.ooov, T.abbb)
    dT.abb -= 0.125 * ccpy_einsum("mnkf,cfbamnji->cbakji", H.aa.ooov, T.aabb)
    dT.abb += 0.25 * ccpy_einsum("bnef,cfeaknji->cbakji", H.bb.vovv, T.abbb)
    dT.abb += 0.5 * ccpy_einsum("nbfe,cfeaknji->cbakji", H.ab.ovvv, T.aabb)
    dT.abb += 0.25 * ccpy_einsum("cnef,efbaknji->cbakji", H.ab.vovv, T.abbb)
    dT.abb += 0.125 * ccpy_einsum("cnef,efbaknji->cbakji", H.aa.vovv, T.aabb)

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
    Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N e^(T1+T2+T3+T4))_C|0>.
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
    # <ijkabc | (H(2) * T3)_C | 0 >
    dT.bbb -= (1.0 / 12.0) * ccpy_einsum("mk,abcijm->abcijk", H.b.oo, T.bbb)
    dT.bbb += (1.0 / 12.0) * ccpy_einsum("ce,abeijk->abcijk", H.b.vv, T.bbb)
    dT.bbb += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", H.bb.oooo, T.bbb)
    dT.bbb += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", H.bb.vvvv, T.bbb)
    dT.bbb += 0.25 * ccpy_einsum("maei,ebcmjk->abcijk", H.ab.ovvo, T.abb)
    dT.bbb += 0.25 * ccpy_einsum("amie,ebcmjk->abcijk", H.bb.voov, T.bbb)
    # <ijkabc | (H(2) * T4)_C | 0 >
    dT.bbb += (1.0 / 36.0) * ccpy_einsum("me,abceijkm->abcijk", H.b.ov, T.bbbb)
    dT.bbb += (1.0 / 36.0) * ccpy_einsum("me,ecbamkji->abcijk", H.a.ov, T.abbb)
    dT.bbb += (1.0 / 24.0) * ccpy_einsum("cnef,abefijkn->abcijk", H.bb.vovv, T.bbbb) # (c/ab) = 3
    dT.bbb += (1.0 / 12.0) * ccpy_einsum("ncfe,febankji->abcijk", H.ab.ovvv, T.abbb) # (c/ab) = 3
    dT.bbb -= (1.0 / 24.0) * ccpy_einsum("mnkf,abcfijmn->abcijk", H.bb.ooov, T.bbbb) # (k/ij) = 3
    dT.bbb -= (1.0 / 12.0) * ccpy_einsum("nmfk,fcbanmji->abcijk", H.ab.oovo, T.abbb) # (k/ij) = 3

    T.bbb, dT.bbb = cc_loops2.update_t3d_v2(
        T.bbb,
        dT.bbb,
        H0.b.oo,
        H0.b.vv,
        shift,
    )
    return T, dT

def update_t4a(T, dT, H, H0, shift):
    """
    Update t4a amplitudes by calculating the projection <ijklabcd|(H_N e^(T1+T2+T3+T4))_C|0>.
    """
    # <ijklabcd | H(2) | 0 >
    dT.aaaa = -(144.0 / 576.0) * ccpy_einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.aa, T.aa)  # (jl/i/k)(bc/a/d) = 12 * 12 = 144
    dT.aaaa += (36.0 / 576.0) * ccpy_einsum("mnij,adml,bcnk->abcdijkl", H.aa.oooo, T.aa, T.aa)   # (ij/kl)(bc/ad) = 6 * 6 = 36
    dT.aaaa += (36.0 / 576.0) * ccpy_einsum("abef,fcjk,edil->abcdijkl", H.aa.vvvv, T.aa, T.aa)   # (jk/il)(ab/cd) = 6 * 6 = 36

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    dT.aaaa += (24.0 / 576.0) * ccpy_einsum("cdke,abeijl->abcdijkl", H.aa.vvov, T.aaa) # (cd/ab)(k/ijl) = 6 * 4 = 24
    dT.aaaa -= (24.0 / 576.0) * ccpy_einsum("cmkl,abdijm->abcdijkl", H.aa.vooo, T.aaa) # (c/abd)(kl/ij) = 6 * 4 = 24

    I3A_vooooo = ccpy_einsum("nmle,bejk->bmnjkl", H.aa.ooov, T.aa)
    I3A_vooooo -= np.transpose(I3A_vooooo, (0, 1, 2, 5, 4, 3)) + np.transpose(I3A_vooooo, (0, 1, 2, 3, 5, 4))
    I3A_vooooo += 0.5 * ccpy_einsum("mnef,befjkl->bmnjkl", H0.aa.oovv, T.aaa)
    dT.aaaa += 0.5 * (16.0 / 576.0) * ccpy_einsum("bmnjkl,acdimn->abcdijkl", I3A_vooooo, T.aaa) # (b/acd)(i/jkl) = 4 * 4 = 16

    I3A_vvvovv = -ccpy_einsum("dmfe,bcjm->bcdjef", H.aa.vovv, T.aa)
    I3A_vvvovv -= np.transpose(I3A_vvvovv, (2, 1, 0, 3, 4, 5)) + np.transpose(I3A_vvvovv, (0, 2, 1, 3, 4, 5))
    dT.aaaa += 0.5 * (16.0 / 576.0) * ccpy_einsum("bcdjef,aefikl->abcdijkl", I3A_vvvovv, T.aaa) # (a/bcd)(j/ikl) = 4 * 4 = 16

    I3A_vvooov = (
                    -0.5 * ccpy_einsum("nmke,cdnl->cdmkle", H.aa.ooov, T.aa)
                    +0.5 * ccpy_einsum("cmfe,fdkl->cdmkle", H.aa.vovv, T.aa)
                    +0.125 * ccpy_einsum("mnef,cdfkln->cdmkle", H0.aa.oovv, T.aaa) # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
                    +0.25 * ccpy_einsum("mnef,cdfkln->cdmkle", H0.ab.oovv, T.aab)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    dT.aaaa += (36.0 / 576.0) * ccpy_einsum("cdmkle,abeijm->abcdijkl", I3A_vvooov, T.aaa) # (cd/ab)(kl/ij) = 6 * 6 = 36

    I3B_vvooov = (
                    -0.5 * ccpy_einsum("nmke,cdnl->cdmkle", H.ab.ooov, T.aa)
                    +0.5 * ccpy_einsum("cmfe,fdkl->cdmkle", H.ab.vovv, T.aa)
                    +0.125 * ccpy_einsum("mnef,cdfkln->cdmkle", H0.bb.oovv, T.aab) # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aaaa += (36.0 / 576.0) * ccpy_einsum("cdmkle,abeijm->abcdijkl", I3B_vvooov, T.aab) # (cd/ab)(kl/ij) = 6 * 6 = 36

    # <ijklabcd | (H(2)*T4)_C | 0 >
    dT.aaaa -= (4.0 / 576.0) * ccpy_einsum("mi,abcdmjkl->abcdijkl", H.a.oo, T.aaaa) # (l/ijk) = 4
    dT.aaaa += (4.0 / 576.0) * ccpy_einsum("ae,ebcdijkl->abcdijkl", H.a.vv, T.aaaa) # (d/abc) = 4
    dT.aaaa += (6.0 / 576.0) * 0.5 * ccpy_einsum("mnij,abcdmnkl->abcdijkl", H.aa.oooo, T.aaaa) # (kl/ij) = 6
    dT.aaaa += (6.0 / 576.0) * 0.5 * ccpy_einsum("abef,efcdijkl->abcdijkl", H.aa.vvvv, T.aaaa) # (cd/ab) = 6
    dT.aaaa += (16.0 / 576.0) * ccpy_einsum("amie,ebcdmjkl->abcdijkl", H.aa.voov, T.aaaa) # (d/abc)(l/ijk) = 16
    dT.aaaa += (16.0 / 576.0) * ccpy_einsum("amie,bcdejklm->abcdijkl", H.ab.voov, T.aaab) # (d/abc)(l/ijk) = 16

    I3A_vvvoov = (
                    -0.5 * ccpy_einsum("mnef,bcdfjkmn->bcdjke", H0.aa.oovv, T.aaaa)
                    -ccpy_einsum("mnef,bcdfjkmn->bcdjke", H0.ab.oovv, T.aaab)
    )
    dT.aaaa += (24.0 / 576.0) * ccpy_einsum("bcdjke,aeil->abcdijkl", I3A_vvvoov, T.aa) # (a/bcd)(jk/il) = 4 * 6 = 24

    I3A_vvoooo = (
                    0.5 * ccpy_einsum("mnef,bcefjkln->bcmjkl", H0.aa.oovv, T.aaaa)
                    +ccpy_einsum("mnef,bcefjkln->bcmjkl", H0.ab.oovv, T.aaab)
    )
    dT.aaaa -= (24.0 / 576.0) * ccpy_einsum("bcmjkl,adim->abcdijkl", I3A_vvoooo, T.aa) # (bc/ad)(i/jkl) = 6 * 4 = 24


    T.aaaa, dT.aaaa = cc_loops_t4.update_t4a(
        T.aaaa,
        dT.aaaa,
        H0.a.oo,
        H0.a.vv,
        shift,
    )
    return T, dT

def update_t4b(T, dT, H, H0, shift):
    """
    Update t4b amplitudes by calculating the projection <ijkl~abcd~|(H_N e^(T1+T2+T3+T4))_C|0>.
    """
    # <ijklabcd | H(2) | 0 >
    dT.aaab = -(9.0 / 36.0) * ccpy_einsum("mdel,abim,ecjk->abcdijkl", H.ab.ovvo, T.aa, T.aa)    # (i/jk)(c/ab) = 9
    dT.aaab += (9.0 / 36.0) * ccpy_einsum("mnij,bcnk,adml->abcdijkl", H.aa.oooo, T.aa, T.ab)    # (k/ij)(a/bc) = 9
    dT.aaab -= (18.0 / 36.0) * ccpy_einsum("mdjf,abim,cfkl->abcdijkl", H.ab.ovov, T.aa, T.ab)   # (ijk)(c/ab) = (i/jk)(c/ab)(jk) = 18
    dT.aaab -= ccpy_einsum("amie,bejl,cdkm->abcdijkl", H.ab.voov, T.ab, T.ab)                   # (ijk)(abc) = (i/jk)(a/bc)(jk)(bc) = 36
    dT.aaab += (18.0 / 36.0) * ccpy_einsum("mnjl,bcmk,adin->abcdijkl", H.ab.oooo, T.aa, T.ab)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    dT.aaab -= (18.0 / 36.0) * ccpy_einsum("bmel,ecjk,adim->abcdijkl", H.ab.vovo, T.aa, T.ab)   # (i/jk)(abc) = (i/jk)(a/bc)(bc) = 18
    dT.aaab -= (18.0 / 36.0) * ccpy_einsum("amie,ecjk,bdml->abcdijkl", H.aa.voov, T.aa, T.ab)   # (i/kj)(abc) = (i/kj)(a/bc)(bc) = 18
    dT.aaab += (9.0 / 36.0) * ccpy_einsum("abef,fcjk,edil->abcdijkl", H.aa.vvvv, T.aa, T.ab)    # (i/jk)(c/ab) = (i/jk)(c/ab) = 9
    dT.aaab -= (18.0 / 36.0) * ccpy_einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.aa, T.ab)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    dT.aaab += (18.0 / 36.0) * ccpy_einsum("adef,ebij,cfkl->abcdijkl", H.ab.vvvv, T.aa, T.ab)   # (k/ij)(abc) = (k/ij)(a/bc)(bc) = 18

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    dT.aaab -= (1.0 / 12.0) * ccpy_einsum("mdkl,abcijm->abcdijkl", H.ab.ovoo, T.aaa)  # (k/ij) = 3
    dT.aaab -= (9.0 / 36.0) * ccpy_einsum("amik,bcdjml->abcdijkl", H.aa.vooo, T.aab)  # (j/ik)(a/bc) = 9
    dT.aaab -= (9.0 / 36.0) * ccpy_einsum("amil,bcdjkm->abcdijkl", H.ab.vooo, T.aab)  # (a/bc)(i/jk) = 9

    dT.aaab += (1.0 / 12.0) * ccpy_einsum("cdel,abeijk->abcdijkl", H.ab.vvvo, T.aaa)  # (c/ab) = 3
    dT.aaab += (9.0 / 36.0) * ccpy_einsum("acie,bedjkl->abcdijkl", H.aa.vvov, T.aab)  # (b/ac)(i/jk) = 9
    dT.aaab += (9.0 / 36.0) * ccpy_einsum("adie,bcejkl->abcdijkl", H.ab.vvov, T.aab)  # (a/bc)(i/jk) = 9

    I3B_oovooo = (
                    ccpy_einsum("mnie,edjl->mndijl", H.aa.ooov, T.ab)
                   +0.25 * ccpy_einsum("mnef,efdijl->mndijl", H0.aa.oovv, T.aab)
    )
    I3B_oovooo -= np.transpose(I3B_oovooo, (0, 1, 2, 4, 3, 5))
    dT.aaab += (1.0 / 12.0) * 0.5 * ccpy_einsum("mndijl,abcmnk->abcdijkl", I3B_oovooo, T.aaa)  # (k/ij) = 3

    I3A_vooooo = ccpy_einsum("mnie,delj->dmnlij", H.aa.ooov, T.aa)
    I3A_vooooo -= np.transpose(I3A_vooooo, (0, 1, 2, 4, 3, 5)) + np.transpose(I3A_vooooo, (0, 1, 2, 5, 4, 3))
    I3A_vooooo += 0.5 * ccpy_einsum("mnef,efdijl->dmnlij", H0.aa.oovv, T.aaa)
    dT.aaab += (1.0 / 12.0) * 0.5 * ccpy_einsum("cmnkij,abdmnl->abcdijkl", I3A_vooooo, T.aab)  # (c/ab) = 3

    I3B_vooooo = (
                    0.5 * ccpy_einsum("mnel,aeik->amnikl", H.ab.oovo, T.aa)
                  + ccpy_einsum("mnke,aeil->amnikl", H.ab.ooov, T.ab)
                  + 0.5 * ccpy_einsum("mnef,aefikl->amnikl", H0.ab.oovv, T.aab)
    )
    I3B_vooooo -= np.transpose(I3B_vooooo, (0, 1, 2, 4, 3, 5))
    dT.aaab += (9.0 / 36.0) * ccpy_einsum("amnikl,bcdjmn->abcdijkl", I3B_vooooo, T.aab)  # (a/bc)(j/ik) = 9

    I3B_vvvvvo = -ccpy_einsum("amef,bdml->abdefl", H.aa.vovv, T.ab)
    I3B_vvvvvo -= np.transpose(I3B_vvvvvo, (1, 0, 2, 3, 4, 5))
    dT.aaab += (1.0 / 12.0) * 0.5 * ccpy_einsum("abdefl,efcijk->abcdijkl", I3B_vvvvvo, T.aaa)  # (c/ab) = 3

    I3A_vvvvvo = -ccpy_einsum("amef,bcmk->abcefk", H.aa.vovv, T.aa)
    I3A_vvvvvo -= np.transpose(I3A_vvvvvo, (1, 0, 2, 3, 4, 5)) + np.transpose(I3A_vvvvvo, (2, 1, 0, 3, 4, 5))
    dT.aaab += (1.0 / 12.0) * 0.5 * ccpy_einsum("abcefk,efdijl->abcdijkl", I3A_vvvvvo, T.aab)  # (k/ij) = 3

    I3B_vvvovv = (
                    -0.5 * ccpy_einsum("mdef,acim->acdief", H.ab.ovvv, T.aa)
                    - ccpy_einsum("cmef,adim->acdief", H.ab.vovv, T.ab)
    )
    I3B_vvvovv -= np.transpose(I3B_vvvovv, (1, 0, 2, 3, 4, 5))
    dT.aaab += (9.0 / 36.0) * ccpy_einsum("acdief,befjkl->abcdijkl", I3B_vvvovv, T.aab)  # (b/ac)(i/jk) = 9

    I3B_vovovo = (
                    -ccpy_einsum("nmie,adnl->amdiel", H.aa.ooov, T.ab)
                    +ccpy_einsum("amfe,fdil->amdiel", H.aa.vovv, T.ab)
                    -ccpy_einsum("mnel,adin->amdiel", H.ab.oovo, T.ab)
                    +ccpy_einsum("mdef,afil->amdiel", H.ab.ovvv, T.ab)
                    +ccpy_einsum("mnef,afdinl->amdiel", H0.aa.oovv, T.aab)
                    +ccpy_einsum("mnef,afdinl->amdiel", H0.ab.oovv, T.abb)
    )
    dT.aaab += (9.0 / 36.0) * ccpy_einsum("amdiel,bcejkm->abcdijkl", I3B_vovovo, T.aaa)  # (a/bc)(i/jk) = 9

    I3A_vvooov = (
                -0.5 * ccpy_einsum("nmje,abin->abmije", H.aa.ooov, T.aa)
                +0.5 * ccpy_einsum("bmfe,afij->abmije", H.aa.vovv, T.aa)
                +0.25 * ccpy_einsum("mnef,abfijn->abmije", H0.ab.oovv, T.aab)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aaab += (9.0 / 36.0) * ccpy_einsum("abmije,cedkml->abcdijkl", I3A_vvooov, T.aab)  # (c/ab)(k/ij) = 9

    I3B_vvoovo = (
                -0.5 * ccpy_einsum("nmel,acin->acmiel", H.ab.oovo, T.aa)
                + ccpy_einsum("cmef,afil->acmiel", H.ab.vovv, T.ab)
                - 0.5 * ccpy_einsum("nmef,acfinl->acmiel", H0.ab.oovv, T.aab)
    )
    I3B_vvoovo -= np.transpose(I3B_vvoovo, (1, 0, 2, 3, 4, 5))
    dT.aaab -= (9.0 / 36.0) * ccpy_einsum("acmiel,ebdkjm->abcdijkl", I3B_vvoovo, T.aab)  # (b/ac)(i/jk) = 9

    I3B_vovoov = (
                0.5 * ccpy_einsum("mdfe,afik->amdike", H.ab.ovvv, T.aa)
                -ccpy_einsum("mnke,adin->amdike", H.ab.ooov, T.ab)
    )
    I3B_vovoov -= np.transpose(I3B_vovoov, (0, 1, 2, 4, 3, 5))
    dT.aaab -= (9.0 / 36.0) * ccpy_einsum("amdike,bcejml->abcdijkl", I3B_vovoov, T.aab)  # (a/bc)(j/ik) = 9

    I3C_vvooov = (
                -ccpy_einsum("nmie,adnl->admile", H.ab.ooov, T.ab)
                -ccpy_einsum("nmle,adin->admile", H.bb.ooov, T.ab)
                +ccpy_einsum("amfe,fdil->admile", H.ab.vovv, T.ab)
                +ccpy_einsum("dmfe,afil->admile", H.bb.vovv, T.ab)
                +ccpy_einsum("mnef,afdinl->admile", H0.bb.oovv, T.abb)  # added 5/2/22
    )
    dT.aaab += (9.0 / 36.0) * ccpy_einsum("admile,bcejkm->abcdijkl", I3C_vvooov, T.aab)  # (a/bc)(i/jk) = 9

    I3B_vvooov = (
                -0.5 * ccpy_einsum("nmje,abin->abmije", H.ab.ooov, T.aa)
                +0.5 * ccpy_einsum("bmfe,afij->abmije", H.ab.vovv, T.aa)
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aaab += (9.0 / 36.0) * ccpy_einsum("abmije,cdeklm->abcdijkl", I3B_vvooov, T.abb)  # (c/ab)(k/ij) = 9

    # <ijklabcd | (H(2)*T4)_C | 0 >
    dT.aaab -= (1.0 / 12.0) * ccpy_einsum("mi,abcdmjkl->abcdijkl", H.a.oo, T.aaab)  # (i/jk) = 3
    dT.aaab -= (1.0 / 36.0) * ccpy_einsum("ml,abcdijkm->abcdijkl", H.b.oo, T.aaab)  # (1) = 1
    dT.aaab += (1.0 / 12.0) * ccpy_einsum("ae,ebcdijkl->abcdijkl", H.a.vv, T.aaab)  # (a/bc) = 3
    dT.aaab += (1.0 / 36.0) * ccpy_einsum("de,abceijkl->abcdijkl", H.b.vv, T.aaab)  # (1) = 1

    dT.aaab += (1.0 / 12.0) * 0.5 * ccpy_einsum("mnij,abcdmnkl->abcdijkl", H.aa.oooo, T.aaab)  # (k/ij) = 3
    dT.aaab += (1.0 / 12.0) * ccpy_einsum("mnil,abcdmjkn->abcdijkl", H.ab.oooo, T.aaab)  # (i/jk) = 3
    dT.aaab += (1.0 / 12.0) * 0.5 * ccpy_einsum("abef,efcdijkl->abcdijkl", H.aa.vvvv, T.aaab)  # (c/ab) = 3
    dT.aaab += (1.0 / 12.0) * ccpy_einsum("adef,ebcfijkl->abcdijkl", H.ab.vvvv, T.aaab)  # (a/bc) = 3

    dT.aaab += (9.0 / 36.0) * ccpy_einsum("amie,ebcdmjkl->abcdijkl", H.aa.voov, T.aaab)  # (a/bc)(i/jk) = 9
    dT.aaab += (9.0 / 36.0) * ccpy_einsum("amie,bcedjkml->abcdijkl", H.ab.voov, T.aabb)  # (a/bc)(i/jk) = 9
    dT.aaab += (1.0 / 36.0) * ccpy_einsum("mdel,abceijkm->abcdijkl", H.ab.ovvo, T.aaaa)  # (1) = 1
    dT.aaab += (1.0 / 36.0) * ccpy_einsum("dmle,abceijkm->abcdijkl", H.bb.voov, T.aaab)  # (1) = 1
    dT.aaab -= (1.0 / 12.0) * ccpy_einsum("amel,ebcdijkm->abcdijkl", H.ab.vovo, T.aaab)  # (a/bc) = 3
    dT.aaab -= (1.0 / 12.0) * ccpy_einsum("mdie,abcemjkl->abcdijkl", H.ab.ovov, T.aaab)  # (i/jk) = 3

    I3B_vvvvoo = (
        -0.5 * ccpy_einsum("mnef,acfdmknl->acdekl", H0.aa.oovv, T.aaab)
        - ccpy_einsum("mnef,acfdmknl->acdekl", H0.ab.oovv, T.aabb)
    )
    dT.aaab += (9.0 / 36.0) * ccpy_einsum("acdekl,ebij->abcdijkl", I3B_vvvvoo, T.aa)  # (b/ac)(k/ij) = 9

    I3A_vvvvoo = (
        -0.5 * ccpy_einsum("mnef,abcfmjkn->abcejk", H0.aa.oovv, T.aaaa)
        - ccpy_einsum("mnef,abcfmjkn->abcejk", H0.ab.oovv, T.aaab)
    )
    dT.aaab += (1.0 / 12.0) * ccpy_einsum("abcejk,edil->abcdijkl", I3A_vvvvoo, T.ab)  # (i/jk) = 3

    I3B_vvvoov = (
        - ccpy_einsum("nmfe,abfdijnm->abdije", H0.ab.oovv, T.aaab)
        - 0.5 * ccpy_einsum("nmfe,abfdijnm->abdije", H0.bb.oovv, T.aabb)
    )
    dT.aaab += (9.0 / 36.0) * ccpy_einsum("abdije,cekl->abcdijkl", I3B_vvvoov, T.ab)  # (c/ab)(k/ij) = 9

    I3B_vovooo = (
        0.5 * ccpy_einsum("mnef,cefdkinl->cmdkil", H0.aa.oovv, T.aaab)
        + ccpy_einsum("mnef,cefdkinl->cmdkil", H0.ab.oovv, T.aabb)
    )
    dT.aaab -= (9.0 / 36.0) * ccpy_einsum("cmdkil,abmj->abcdijkl", I3B_vovooo, T.aa)  # (c/ab)(j/ik) = 9

    I3A_vovooo = (
        0.5 * ccpy_einsum("mnef,bcefjkin->bmcjik", H0.aa.oovv, T.aaaa)
        + ccpy_einsum("mnef,bcefjkin->bmcjik", H0.ab.oovv, T.aaab)
    )
    dT.aaab -= (1.0 / 12.0) * ccpy_einsum("bmcjik,adml->abcdijkl", I3A_vovooo, T.ab)  # (a/bc) = 3

    I3B_vvoooo = (
        ccpy_einsum("nmfe,bcfejknl->bcmjkl", H0.ab.oovv, T.aaab)
        + 0.5 * ccpy_einsum("nmfe,bcfejknl->bcmjkl", H0.bb.oovv, T.aabb)
    )
    dT.aaab -= (9.0 / 36.0) * ccpy_einsum("bcmjkl,adim->abcdijkl", I3B_vvoooo, T.ab)  # (a/bc)(i/jk) = 9


    T.aaab, dT.aaab = cc_loops_t4.update_t4b(
        T.aaab,
        dT.aaab,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift,
    )
    return T, dT


def update_t4c(T, dT, H, H0, shift):
    """
    Update t4c amplitudes by calculating the projection <ijk~l~abc~d~|(H_N e^(T1+T2+T3+T4))_C|0>.
    """
    # <ijklabcd | H(2) | 0 >
    dT.aabb = -ccpy_einsum("cmke,adim,bejl->abcdijkl", H.bb.voov, T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= ccpy_einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= 0.5 * ccpy_einsum("mcek,aeij,bdml->abcdijkl", H.ab.ovvo, T.aa, T.ab)    # (kl)(ab)(cd) = 8
    dT.aabb -= 0.5 * ccpy_einsum("amie,bdjm,cekl->abcdijkl", H.ab.voov, T.ab, T.bb)    # (ij)(ab)(cd) = 8
    dT.aabb -= 0.5 * ccpy_einsum("mcek,abim,edjl->abcdijkl", H.ab.ovvo, T.aa, T.ab)    # (ij)(kl)(cd) = 8
    dT.aabb -= 0.5 * ccpy_einsum("amie,cdkm,bejl->abcdijkl", H.ab.voov, T.bb, T.ab)    # (ij)(kl)(ab) = 8
    dT.aabb -= ccpy_einsum("bmel,adim,ecjk->abcdijkl", H.ab.vovo, T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= ccpy_einsum("mdje,bcmk,aeil->abcdijkl", H.ab.ovov, T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= 0.25 * ccpy_einsum("mdje,abim,cekl->abcdijkl", H.ab.ovov, T.aa, T.bb)   # (ij)(cd) = 4
    dT.aabb -= 0.25 * ccpy_einsum("bmel,cdkm,aeij->abcdijkl", H.ab.vovo, T.bb, T.aa)   # (kl)(ab) = 4
    dT.aabb += 0.25 * ccpy_einsum("mnij,acmk,bdnl->abcdijkl", H.aa.oooo, T.ab, T.ab)   # (kl)(ab) = 4 !!! (tricky asym)
    dT.aabb += 0.25 * ccpy_einsum("abef,ecik,fdjl->abcdijkl", H.aa.vvvv, T.ab, T.ab)   # (ij)(kl) = 4 !!! (tricky asym)
    dT.aabb += 0.25 * ccpy_einsum("mnik,abmj,cdnl->abcdijkl", H.ab.oooo, T.aa, T.bb)   # (ij)(kl) = 4
    dT.aabb += 0.25 * ccpy_einsum("acef,ebij,fdkl->abcdijkl", H.ab.vvvv, T.aa, T.bb)   # (ab)(cd) = 4
    dT.aabb += ccpy_einsum("mnik,adml,bcjn->abcdijkl", H.ab.oooo, T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb += ccpy_einsum("acef,edil,bfjk->abcdijkl", H.ab.vvvv, T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb += 0.25 * ccpy_einsum("mnkl,adin,bcjm->abcdijkl", H.bb.oooo, T.ab, T.ab)   # (ij)(cd) = 4 !!! (tricky asym)
    dT.aabb += 0.25 * ccpy_einsum("cdef,afil,bejk->abcdijkl", H.bb.vvvv, T.ab, T.ab)   # (ij)(kl) = 4 !!! (tricky asym)

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    dT.aabb -= (8.0 / 16.0) * ccpy_einsum("mdil,abcmjk->abcdijkl", H.ab.ovoo, T.aab)  # [1]  (ij)(kl)(cd) = 8
    dT.aabb -= (2.0 / 16.0) * ccpy_einsum("bmji,acdmkl->abcdijkl", H.aa.vooo, T.abb)  # [2]  (ab) = 2
    dT.aabb -= (2.0 / 16.0) * ccpy_einsum("cmkl,abdijm->abcdijkl", H.bb.vooo, T.aab)  # [3]  (cd) = 2
    dT.aabb -= (8.0 / 16.0) * ccpy_einsum("amil,bcdjkm->abcdijkl", H.ab.vooo, T.abb)  # [4]  (ij)(ab)(kl) = 8
    dT.aabb += (8.0 / 16.0) * ccpy_einsum("adel,becjik->abcdijkl", H.ab.vvvo, T.aab)  # [5]  (ab)(kl)(cd) = 8
    dT.aabb += (2.0 / 16.0) * ccpy_einsum("baje,ecdikl->abcdijkl", H.aa.vvov, T.abb)  # [6]  (ij) = 2
    dT.aabb += (8.0 / 16.0) * ccpy_einsum("adie,bcejkl->abcdijkl", H.ab.vvov, T.abb)  # [7]  (ij)(ab)(cd) = 8
    dT.aabb += (2.0 / 16.0) * ccpy_einsum("cdke,abeijl->abcdijkl", H.bb.vvov, T.aab)  # [8]  (kl) = 2

    I3B_oovooo = (
                ccpy_einsum("mnif,fdjl->mndijl", H.aa.ooov, T.ab)
               + 0.25 * ccpy_einsum("mnef,efdijl->mndijl", H0.aa.oovv, T.aab)
    )
    I3B_oovooo -= np.transpose(I3B_oovooo, (0, 1, 2, 4, 3, 5))
    dT.aabb += (4.0 / 16.0) * 0.5 * ccpy_einsum("mndijl,abcmnk->abcdijkl", I3B_oovooo, T.aab)  # [9]  (kl)(cd) = 4

    I3B_ovoooo = (
                ccpy_einsum("mnif,bfjl->mbnijl", H.ab.ooov, T.ab)
                + 0.5 * ccpy_einsum("mnfl,bfji->mbnijl", H.ab.oovo, T.aa)
                + 0.5 * ccpy_einsum("mnef,befjil->mbnijl", H0.ab.oovv, T.aab)
    )
    I3B_ovoooo -= np.transpose(I3B_ovoooo, (0, 1, 2, 4, 3, 5))
    dT.aabb += (4.0 / 16.0) * ccpy_einsum("mbnijl,acdmkn->abcdijkl", I3B_ovoooo, T.abb)  # [10]  (kl)(ab) = 4

    I3C_vooooo = (
                ccpy_einsum("nmlf,afik->amnikl", H.bb.ooov, T.ab)
                + 0.25 * ccpy_einsum("mnef,aefikl->amnikl", H0.bb.oovv, T.abb)
    )
    I3C_vooooo -= np.transpose(I3C_vooooo, (0, 1, 2, 3, 5, 4))
    dT.aabb += (4.0 / 16.0) * 0.5 * ccpy_einsum("amnikl,bcdjmn->abcdijkl", I3C_vooooo, T.abb)  # [11]  (ij)(ab) = 4

    I3C_oovooo = (
                0.5 * ccpy_einsum("mnif,cfkl->mncilk", H.ab.ooov, T.bb)
                + ccpy_einsum("mnfl,fcik->mncilk", H.ab.oovo, T.ab)
                + 0.5 * ccpy_einsum("mnef,efcilk->mncilk", H0.ab.oovv, T.abb)
    )
    I3C_oovooo -= np.transpose(I3C_oovooo, (0, 1, 2, 3, 5, 4))
    dT.aabb += (4.0 / 16.0) * ccpy_einsum("mncilk,abdmjn->abcdijkl", I3C_oovooo, T.aab)  # [12]  (ij)(cd) = 4

    I3B_vvvvvo = -ccpy_einsum("bmfe,acmk->abcefk", H.aa.vovv, T.ab)
    I3B_vvvvvo -= np.transpose(I3B_vvvvvo, (1, 0, 2, 3, 4, 5))
    dT.aabb += (4.0 / 16.0) * 0.5 * ccpy_einsum("abcefk,efdijl->abcdijkl", I3B_vvvvvo, T.aab)  # [13]  (kl)(cd) = 4

    I3C_vvvvov = (
                -ccpy_einsum("mdef,acmk->acdekf", H.ab.ovvv, T.ab)
                - 0.5 * ccpy_einsum("amef,cdkm->acdekf", H.ab.vovv, T.bb)
    )
    I3C_vvvvov -= np.transpose(I3C_vvvvov, (0, 2, 1, 3, 4, 5))
    dT.aabb += (4.0 / 16.0) * ccpy_einsum("acdekf,ebfijl->abcdijkl", I3C_vvvvov, T.aab)  # [14]  (kl)(ab) = 4

    I3B_vvvvov = (
                -0.5 * ccpy_einsum("mdef,abmj->abdejf", H.ab.ovvv, T.aa)
                -ccpy_einsum("amef,bdjm->abdejf", H.ab.vovv, T.ab)
    )
    I3B_vvvvov -= np.transpose(I3B_vvvvov, (1, 0, 2, 3, 4, 5))
    dT.aabb += (4.0 / 16.0) * ccpy_einsum("abdejf,efcilk->abcdijkl", I3B_vvvvov, T.abb)  # [15]  (ij)(cd) = 4

    I3C_vvvovv = -ccpy_einsum("cmef,adim->acdief", H.bb.vovv, T.ab)
    I3C_vvvovv -= np.transpose(I3C_vvvovv, (0, 2, 1, 3, 4, 5))
    dT.aabb += (4.0 / 16.0) * 0.5 * ccpy_einsum("acdief,befjkl->abcdijkl", I3C_vvvovv, T.abb)  # [16]  (ij)(ab) = 4

    I3A_vvooov = (
                -0.5 * ccpy_einsum("nmje,abin->abmije", H.aa.ooov, T.aa)
                +0.5 * ccpy_einsum("bmfe,afij->abmije", H.aa.vovv, T.aa)
                +0.25 * ccpy_einsum("mnef,abfijn->abmije", H0.aa.oovv, T.aaa)
                +0.25 * ccpy_einsum("mnef,abfijn->abmije", H0.ab.oovv, T.aab)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aabb += (1.0 / 16.0) * ccpy_einsum("abmije,ecdmkl->abcdijkl", I3A_vvooov, T.abb)  # [17]  (1) = 1

    I3B_vvooov = (
                -0.5 * ccpy_einsum("nmje,abin->abmije", H.ab.ooov, T.aa)
                +0.5 * ccpy_einsum("bmfe,afij->abmije", H.ab.vovv, T.aa)
                +0.25 * ccpy_einsum("nmfe,abfijn->abmije", H0.ab.oovv, T.aaa)
                +0.25 * ccpy_einsum("nmfe,abfijn->abmije", H0.bb.oovv, T.aab)
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aabb += (1.0 / 16.0) * ccpy_einsum("abmije,ecdmkl->abcdijkl", I3B_vvooov, T.bbb)  # [18]  (1) = 1

    I3C_ovvvoo = (
                -0.5 * ccpy_einsum("mnek,cdnl->mcdekl", H.ab.oovo, T.bb)
                +0.5 * ccpy_einsum("mcef,fdkl->mcdekl", H.ab.ovvv, T.bb)
    )
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 2, 1, 3, 4, 5))
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 1, 2, 3, 5, 4))
    dT.aabb += (1.0 / 16.0) * ccpy_einsum("mcdekl,abeijm->abcdijkl", I3C_ovvvoo, T.aaa)  # [19]  (1) = 1

    I3D_vvooov = (
                -0.5 * ccpy_einsum("nmke,cdnl->cdmkle", H.bb.ooov, T.bb)
                +0.5 * ccpy_einsum("cmfe,fdkl->cdmkle", H.bb.vovv, T.bb)
    )
    I3D_vvooov -= np.transpose(I3D_vvooov, (1, 0, 2, 3, 4, 5))
    I3D_vvooov -= np.transpose(I3D_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aabb += (1.0 / 16.0) * ccpy_einsum("cdmkle,abeijm->abcdijkl", I3D_vvooov, T.aab)  # [20]  (1) = 1

    I3B_vovovo = (
                -ccpy_einsum("mnel,adin->amdiel", H.ab.oovo, T.ab)
                +ccpy_einsum("mdef,afil->amdiel", H.ab.ovvv, T.ab)
                +0.5 * ccpy_einsum("mnef,afdinl->amdiel", H0.aa.oovv, T.aab) # !!! factor 1/2 to compensate asym
                +ccpy_einsum("mnef,afdinl->amdiel", H0.ab.oovv, T.abb)
                -ccpy_einsum("nmie,adnl->amdiel", H.aa.ooov, T.ab)
                +ccpy_einsum("amfe,fdil->amdiel", H.aa.vovv, T.ab)
    )
    dT.aabb += ccpy_einsum("amdiel,becjmk->abcdijkl", I3B_vovovo, T.aab)  # [21]  (ij)(kl)(ab)(cd) = 16

    I3C_vovovo = (
                -ccpy_einsum("nmie,adnl->amdiel", H.ab.ooov, T.ab)
                +ccpy_einsum("amfe,fdil->amdiel", H.ab.vovv, T.ab)
                -ccpy_einsum("nmle,adin->amdiel", H.bb.ooov, T.ab)
                +ccpy_einsum("dmfe,afil->amdiel", H.bb.vovv, T.ab)
                +0.5 * ccpy_einsum("mnef,afdinl->amdiel", H0.bb.oovv, T.abb) # !!! factor 1/2 to compensate asym
    )
    dT.aabb += ccpy_einsum("amdiel,becjmk->abcdijkl", I3C_vovovo, T.abb)  # [22]  (ij)(kl)(ab)(cd) = 16

    I3B_vovoov = (
                -ccpy_einsum("mnie,bdjn->bmdjie", H.ab.ooov, T.ab)
                +0.5 * ccpy_einsum("mdfe,bfji->bmdjie", H.ab.ovvv, T.aa)
                -0.5 * ccpy_einsum("mnfe,bfdjin->bmdjie", H0.ab.oovv, T.aab)
    )
    I3B_vovoov -= np.transpose(I3B_vovoov, (0, 1, 2, 4, 3, 5))
    dT.aabb -= (4.0 / 16.0) * ccpy_einsum("bmdjie,aecmlk->abcdijkl", I3B_vovoov, T.abb)  # [23]  (ab)(cd) = 4

    I3C_ovvoov = (
                -0.5 * ccpy_einsum("mnie,cdkn->mcdike", H.ab.ooov, T.bb)
                +ccpy_einsum("mdfe,fcik->mcdike", H.ab.ovvv, T.ab)
                -0.5 * ccpy_einsum("mnfe,fcdikn->mcdike", H0.ab.oovv, T.abb)
    )
    I3C_ovvoov -= np.transpose(I3C_ovvoov, (0, 2, 1, 3, 4, 5))
    dT.aabb -= (4.0 / 16.0) * ccpy_einsum("mcdike,abemjl->abcdijkl", I3C_ovvoov, T.aab)  # [24]  (ij)(kl) = 4

    I3B_vvovoo = (
                -0.5 * ccpy_einsum("nmel,abnj->abmejl", H.ab.oovo, T.aa)
                +ccpy_einsum("amef,bfjl->abmejl", H.ab.vovv, T.ab)
    )
    I3B_vvovoo -= np.transpose(I3B_vvovoo, (1, 0, 2, 3, 4, 5))
    dT.aabb -= (4.0 / 16.0) * ccpy_einsum("abmejl,ecdikm->abcdijkl", I3B_vvovoo, T.abb)  # [25]  (ij)(kl) = 4

    I3C_vovvoo = (
                -ccpy_einsum("nmel,acnk->amcelk", H.ab.oovo, T.ab)
                +0.5 * ccpy_einsum("amef,fclk->amcelk", H.ab.vovv, T.bb)
    )
    I3C_vovvoo -= np.transpose(I3C_vovvoo, (0, 1, 2, 3, 5, 4))
    dT.aabb -= (4.0 / 16.0) * ccpy_einsum("amcelk,bedjim->abcdijkl", I3C_vovvoo, T.aab)  # [26]  (ab)(cd) = 4

    # <ijklabcd | (H(2)*T4)_C | 0 >
    dT.aabb -= (2.0 / 16.0) * ccpy_einsum("mi,abcdmjkl->abcdijkl", H.a.oo, T.aabb)  # [1]  (ij) = 2
    dT.aabb -= (2.0 / 16.0) * ccpy_einsum("ml,abcdijkm->abcdijkl", H.b.oo, T.aabb)  # [2]  (kl) = 2
    dT.aabb += (2.0 / 16.0) * ccpy_einsum("ae,ebcdijkl->abcdijkl", H.a.vv, T.aabb)  # [3]  (ab) = 2
    dT.aabb += (2.0 / 16.0) * ccpy_einsum("de,abceijkl->abcdijkl", H.b.vv, T.aabb)  # [4]  (cd) = 2
    dT.aabb += (1.0 / 16.0) * 0.5 * ccpy_einsum("mnij,abcdmnkl->abcdijkl", H.aa.oooo, T.aabb)  # [5]  (1) = 1
    dT.aabb += (4.0 / 16.0) * ccpy_einsum("mnil,abcdmjkn->abcdijkl", H.ab.oooo, T.aabb)  # [6]  (ij)(kl) = 4
    dT.aabb += (1.0 / 16.0) * 0.5 * ccpy_einsum("mnkl,abcdijmn->abcdijkl", H.bb.oooo, T.aabb)  #  [7]  (1) = 1
    dT.aabb += (1.0 / 16.0) * 0.5 * ccpy_einsum("abef,efcdijkl->abcdijkl", H.aa.vvvv, T.aabb)  #  [8]  (1) = 1
    dT.aabb += (4.0 / 16.0) * ccpy_einsum("adef,ebcfijkl->abcdijkl", H.ab.vvvv, T.aabb)  #  [9]  (ab)(cd) = 4
    dT.aabb += (1.0 / 16.0) * 0.5 * ccpy_einsum("cdef,abefijkl->abcdijkl", H.bb.vvvv, T.aabb)  #  [10]  (1) = 1
    dT.aabb += (4.0 / 16.0) * ccpy_einsum("amie,ebcdmjkl->abcdijkl", H.aa.voov, T.aabb)  #  [11]  (ij)(ab) = 4
    dT.aabb += (4.0 / 16.0) * ccpy_einsum("amie,becdjmkl->abcdijkl", H.ab.voov, T.abbb)  #  [12]  (ij)(ab) = 4
    dT.aabb += (4.0 / 16.0) * ccpy_einsum("mdel,aebcimjk->abcdijkl", H.ab.ovvo, T.aaab)  #  [13]  (kl)(cd) = 4
    dT.aabb += (4.0 / 16.0) * ccpy_einsum("dmle,abceijkm->abcdijkl", H.bb.voov, T.aabb)  #  [14]  (kl)(cd) = 4
    dT.aabb -= (4.0 / 16.0) * ccpy_einsum("mdie,abcemjkl->abcdijkl", H.ab.ovov, T.aabb)  #  [15]  (ij)(cd) = 4
    dT.aabb -= (4.0 / 16.0) * ccpy_einsum("amel,ebcdijkm->abcdijkl", H.ab.vovo, T.aabb)  #  [16]  (kl)(ab) = 4

    I3C_vvvvoo = (
                -0.5 * ccpy_einsum("mnef,afcdmnkl->acdekl", H0.aa.oovv, T.aabb)
                -ccpy_einsum("mnef,afcdmnkl->acdekl", H0.ab.oovv, T.abbb)
    )
    dT.aabb += (2.0 / 16.0) * ccpy_einsum("acdekl,beji->abcdijkl", I3C_vvvvoo, T.aa)  #  [17]  (ab) = 2

    I3B_vvvvoo = (
                -0.5 * ccpy_einsum("mnef,abfcmjnk->abcejk", H0.aa.oovv, T.aaab)
                -ccpy_einsum("mnef,abfcmjnk->abcejk", H0.ab.oovv, T.aabb)
    )
    dT.aabb += (8.0 / 16.0) * ccpy_einsum("abcejk,edil->abcdijkl", I3B_vvvvoo, T.ab)  #  [18]  (ij)(kl)(cd) = 8

    I3C_vvvoov = (
                -ccpy_einsum("nmfe,bfcdjnkm->bcdjke", H0.ab.oovv, T.aabb)
                -0.5 * ccpy_einsum("mnef,bcdfjkmn->bcdjke", H0.bb.oovv, T.abbb)
    )
    dT.aabb += (8.0 / 16.0) * ccpy_einsum("bcdjke,aeil->abcdijkl", I3C_vvvoov, T.ab)  #  [19]  (ij)(kl)(ab) = 8

    I3B_vvvoov = (
                -ccpy_einsum("nmfe,abfdijnm->abdije", H0.ab.oovv, T.aaab)
                -0.5 * ccpy_einsum("mnef,abfdijnm->abdije", H0.bb.oovv, T.aabb)
    )
    dT.aabb += (2.0 / 16.0) * ccpy_einsum("abdije,eclk->abcdijkl", I3B_vvvoov, T.bb)  #  [20]  (cd) = 2

    I3C_ovvooo = (
                0.5 * ccpy_einsum("mnef,efcdinkl->mcdikl", H0.aa.oovv, T.aabb)
                +ccpy_einsum("mnef,efcdinkl->mcdikl", H0.ab.oovv, T.abbb)
    )
    dT.aabb -= (2.0 / 16.0) * ccpy_einsum("mcdikl,abmj->abcdijkl", I3C_ovvooo, T.aa)  #  [21]  (ij) = 2

    I3B_vovooo = (
                0.5 * ccpy_einsum("mnef,befcjink->bmcjik", H0.aa.oovv, T.aaab)
                +ccpy_einsum("mnef,befcjink->bmcjik", H0.ab.oovv, T.aabb)
    )
    dT.aabb -= (8.0 / 16.0) * ccpy_einsum("bmcjik,adml->abcdijkl", I3B_vovooo, T.ab)  #  [22]  (ab)(kl)(cd) = 8

    I3C_vovooo = (
                ccpy_einsum("nmfe,bfecjnlk->bmcjlk", H0.ab.oovv, T.aabb)
                +0.5 * ccpy_einsum("mnef,bfecjnlk->bmcjlk", H0.bb.oovv, T.abbb)
    )
    dT.aabb -= (8.0 / 16.0) * ccpy_einsum("bmcjlk,adim->abcdijkl", I3C_vovooo, T.ab)  #  [23]  (ij)(ab)(cd) = 8

    I3B_vvoooo = (
                ccpy_einsum("nmfe,abfeijnl->abmijl", H0.ab.oovv, T.aaab)
                +0.5 * ccpy_einsum("mnef,abfeijnl->abmijl", H0.bb.oovv, T.aabb)
    )
    dT.aabb -= (2.0 / 16.0) * ccpy_einsum("abmijl,cdkm->abcdijkl", I3B_vvoooo, T.bb)  #  [24]  (kl) = 2

    T.aabb, dT.aabb = cc_loops_t4.update_t4c(
        T.aabb,
        dT.aabb,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift,
    )
    return T, dT

def update_t4d(T, dT, H, H0, shift):
    """
    Update t4d amplitudes by calculating the projection <ij~k~l~ab~c~d~|(H_N e^(T1+T2+T3+T4))_C|0>.
    """
    # <ijklabcd | H(2) | 0 >
    dT.abbb = -(9.0 / 36.0) * ccpy_einsum("dmle,abim,ecjk->dcbalkji", H.ab.voov, T.bb, T.bb)    # (i/jk)(c/ab) = 9
    dT.abbb += (9.0 / 36.0) * ccpy_einsum("mnij,bcnk,dalm->dcbalkji", H.bb.oooo, T.bb, T.ab)    # (k/ij)(a/bc) = 9
    dT.abbb -= (18.0 / 36.0) * ccpy_einsum("dmfj,abim,fclk->dcbalkji", H.ab.vovo, T.bb, T.ab)   # (ijk)(c/ab) = (i/jk)(c/ab)(jk) = 18
    dT.abbb -= ccpy_einsum("maei,eblj,dcmk->dcbalkji", H.ab.ovvo, T.ab, T.ab)                   # (ijk)(abc) = (i/jk)(a/bc)(jk)(bc) = 36
    dT.abbb += (18.0 / 36.0) * ccpy_einsum("nmlj,bcmk,dani->dcbalkji", H.ab.oooo, T.bb, T.ab)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    dT.abbb -= (18.0 / 36.0) * ccpy_einsum("mble,ecjk,dami->dcbalkji", H.ab.ovov, T.bb, T.ab)   # (i/jk)(abc) = (i/jk)(a/bc)(bc) = 18
    dT.abbb -= (18.0 / 36.0) * ccpy_einsum("amie,ecjk,dblm->dcbalkji", H.bb.voov, T.bb, T.ab)   # (i/kj)(abc) = (i/kj)(a/bc)(bc) = 18
    dT.abbb += (9.0 / 36.0) * ccpy_einsum("abef,fcjk,deli->dcbalkji", H.bb.vvvv, T.bb, T.ab)    # (i/jk)(c/ab) = (i/jk)(c/ab) = 9
    dT.abbb -= (18.0 / 36.0) * ccpy_einsum("amie,bcmk,delj->dcbalkji", H.bb.voov, T.bb, T.ab)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    dT.abbb += (18.0 / 36.0) * ccpy_einsum("dafe,ebij,fclk->dcbalkji", H.ab.vvvv, T.bb, T.ab)   # (k/ij)(abc) = (k/ij)(a/bc)(bc) = 18

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    dT.abbb -= (1.0 / 12.0) * ccpy_einsum("dmlk,abcijm->dcbalkji", H.ab.vooo, T.bbb)  # (k/ij) = 3
    dT.abbb -= (9.0 / 36.0) * ccpy_einsum("amik,dcblmj->dcbalkji", H.bb.vooo, T.abb)  # (j/ik)(a/bc) = 9
    dT.abbb -= (9.0 / 36.0) * ccpy_einsum("mali,dcbmkj->dcbalkji", H.ab.ovoo, T.abb)  # (a/bc)(i/jk) = 9

    dT.abbb += (1.0 / 12.0) * ccpy_einsum("dcle,abeijk->dcbalkji", H.ab.vvov, T.bbb)  # (c/ab) = 3
    dT.abbb += (9.0 / 36.0) * ccpy_einsum("acie,deblkj->dcbalkji", H.bb.vvov, T.abb)  # (b/ac)(i/jk) = 9 #
    dT.abbb += (9.0 / 36.0) * ccpy_einsum("daei,ecblkj->dcbalkji", H.ab.vvvo, T.abb)  # (a/bc)(i/jk) = 9

    I3C_vooooo = (
                    ccpy_einsum("mnie,delj->dnmlji", H.bb.ooov, T.ab)
                   +0.25 * ccpy_einsum("mnef,dfelji->dnmlji", H0.bb.oovv, T.abb)
    )
    I3C_vooooo -= np.transpose(I3C_vooooo, (0, 1, 2, 3, 5, 4))
    dT.abbb += (1.0 / 12.0) * 0.5 * ccpy_einsum("dnmlji,abcmnk->dcbalkji", I3C_vooooo, T.bbb)  # (k/ij) = 3

    I3D_vooooo = ccpy_einsum("mnie,delj->dmnlij", H.bb.ooov, T.bb)
    I3D_vooooo -= np.transpose(I3D_vooooo, (0, 1, 2, 4, 3, 5)) + np.transpose(I3D_vooooo, (0, 1, 2, 5, 4, 3))
    I3D_vooooo += 0.5 * ccpy_einsum("mnef,efdijl->dmnlij", H0.bb.oovv, T.bbb)
    dT.abbb += (1.0 / 12.0) * 0.5 * ccpy_einsum("cmnkij,dbalnm->dcbalkji", I3D_vooooo, T.abb)  # (c/ab) = 3

    I3C_oovooo = (
                    0.5 * ccpy_einsum("nmle,aeik->nmalki", H.ab.ooov, T.bb)
                  + ccpy_einsum("nmek,eali->nmalki", H.ab.oovo, T.ab)
                  + 0.5 * ccpy_einsum("nmfe,fealki->nmalki", H0.ab.oovv, T.abb)
    )
    I3C_oovooo -= np.transpose(I3C_oovooo, (0, 1, 2, 3, 5, 4))
    dT.abbb += (9.0 / 36.0) * ccpy_einsum("nmalki,dcbnmj->dcbalkji", I3C_oovooo, T.abb)  # (a/bc)(j/ik) = 9

    I3C_vvvvov = -ccpy_einsum("amef,dblm->dbalfe", H.bb.vovv, T.ab)
    I3C_vvvvov -= np.transpose(I3C_vvvvov, (0, 2, 1, 3, 4, 5))
    dT.abbb += (1.0 / 12.0) * 0.5 * ccpy_einsum("dbalfe,efcijk->dcbalkji", I3C_vvvvov, T.bbb)  # (c/ab) = 3

    I3D_vvvvvo = -ccpy_einsum("amef,bcmk->abcefk", H.bb.vovv, T.bb)
    I3D_vvvvvo -= np.transpose(I3D_vvvvvo, (1, 0, 2, 3, 4, 5)) + np.transpose(I3D_vvvvvo, (2, 1, 0, 3, 4, 5))
    dT.abbb += (1.0 / 12.0) * 0.5 * ccpy_einsum("abcefk,dfelji->dcbalkji", I3D_vvvvvo, T.abb)  # (k/ij) = 3

    I3C_vvvvvo = (
                    -0.5 * ccpy_einsum("dmfe,acim->dcafei", H.ab.vovv, T.bb)
                    - ccpy_einsum("mcfe,dami->dcafei", H.ab.ovvv, T.ab)
    )
    I3C_vvvvvo -= np.transpose(I3C_vvvvvo, (0, 2, 1, 3, 4, 5))
    dT.abbb += (9.0 / 36.0) * ccpy_einsum("dcafei,feblkj->dcbalkji", I3C_vvvvvo, T.abb)  # (b/ac)(i/jk) = 9

    I3C_vovovo = (
                    -ccpy_einsum("nmie,daln->dmalei", H.bb.ooov, T.ab)
                    +ccpy_einsum("amfe,dfli->dmalei", H.bb.vovv, T.ab)
                    -ccpy_einsum("nmle,dani->dmalei", H.ab.ooov, T.ab)
                    +ccpy_einsum("dmfe,fali->dmalei", H.ab.vovv, T.ab)
                    +ccpy_einsum("mnef,dfalni->dmalei", H0.bb.oovv, T.abb)
                    +ccpy_einsum("nmfe,dfalni->dmalei", H0.ab.oovv, T.aab)
    )
    dT.abbb += (9.0 / 36.0) * ccpy_einsum("dmalei,bcejkm->dcbalkji", I3C_vovovo, T.bbb)  # (a/bc)(i/jk) = 9

    I3D_vvooov = (
                -0.5 * ccpy_einsum("nmje,abin->abmije", H.bb.ooov, T.bb)
                +0.5 * ccpy_einsum("bmfe,afij->abmije", H.bb.vovv, T.bb)
                +0.25 * ccpy_einsum("nmfe,fbanji->abmije", H0.ab.oovv, T.abb)
    )
    I3D_vvooov -= np.transpose(I3D_vvooov, (1, 0, 2, 3, 4, 5))
    I3D_vvooov -= np.transpose(I3D_vvooov, (0, 1, 2, 4, 3, 5))
    dT.abbb += (9.0 / 36.0) * ccpy_einsum("abmije,declmk->dcbalkji", I3D_vvooov, T.abb)  # (c/ab)(k/ij) = 9

    I3C_ovvovo = (
                -0.5 * ccpy_einsum("mnle,acin->mcalei", H.ab.ooov, T.bb)
                + ccpy_einsum("mcfe,fali->mcalei", H.ab.ovvv, T.ab)
                - 0.5 * ccpy_einsum("mnfe,fcalni->mcalei", H0.ab.oovv, T.abb)
    )
    I3C_ovvovo -= np.transpose(I3C_ovvovo, (0, 2, 1, 3, 4, 5))
    dT.abbb -= (9.0 / 36.0) * ccpy_einsum("mcalei,dbemjk->dcbalkji", I3C_ovvovo, T.abb)  # (b/ac)(i/jk) = 9

    I3C_vovvoo = (
                0.5 * ccpy_einsum("dmef,afik->dmaeki", H.ab.vovv, T.bb)
                -ccpy_einsum("nmek,dani->dmaeki", H.ab.oovo, T.ab)
    )
    I3C_vovvoo -= np.transpose(I3C_vovvoo, (0, 1, 2, 3, 5, 4))
    dT.abbb -= (9.0 / 36.0) * ccpy_einsum("dmaeki,ecblmj->dcbalkji", I3C_vovvoo, T.abb)  # (a/bc)(j/ik) = 9

    I3B_ovvvoo = (
                -ccpy_einsum("mnei,daln->mdaeli", H.ab.oovo, T.ab)
                -ccpy_einsum("nmle,dani->mdaeli", H.aa.ooov, T.ab)
                +ccpy_einsum("maef,dfli->mdaeli", H.ab.ovvv, T.ab)
                +ccpy_einsum("dmfe,fali->mdaeli", H.aa.vovv, T.ab)
                +ccpy_einsum("mnef,dfalni->mdaeli", H0.aa.oovv, T.aab)  # added 5/2/22
    )
    dT.abbb += (9.0 / 36.0) * ccpy_einsum("mdaeli,ecbmkj->dcbalkji", I3B_ovvvoo, T.abb)  # (a/bc)(i/jk) = 9

    I3C_ovvvoo = (
                -0.5 * ccpy_einsum("mnej,abin->mbaeji", H.ab.oovo, T.bb)
                +0.5 * ccpy_einsum("mbef,afij->mbaeji", H.ab.ovvv, T.bb)
    )
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 2, 1, 3, 4, 5))
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 1, 2, 3, 5, 4))
    dT.abbb += (9.0 / 36.0) * ccpy_einsum("mbaeji,edcmlk->dcbalkji", I3C_ovvvoo, T.aab)  # (c/ab)(k/ij) = 9

    # <ijklabcd | (H(2)*T4)_C | 0 >
    dT.abbb -= (1.0 / 12.0) * ccpy_einsum("mi,dcbalkjm->dcbalkji", H.b.oo, T.abbb)  # (i/jk) = 3
    dT.abbb -= (1.0 / 36.0) * ccpy_einsum("ml,dcbamkji->dcbalkji", H.a.oo, T.abbb)  # (1) = 1
    dT.abbb += (1.0 / 12.0) * ccpy_einsum("ae,dcbelkji->dcbalkji", H.b.vv, T.abbb)  # (a/bc) = 3
    dT.abbb += (1.0 / 36.0) * ccpy_einsum("de,ecbalkji->dcbalkji", H.a.vv, T.abbb)  # (1) = 1

    dT.abbb += (1.0 / 12.0) * 0.5 * ccpy_einsum("mnij,dcbalknm->dcbalkji", H.bb.oooo, T.abbb)  # (k/ij) = 3
    dT.abbb += (1.0 / 12.0) * ccpy_einsum("nmli,dcbankjm->dcbalkji", H.ab.oooo, T.abbb)  # (i/jk) = 3
    dT.abbb += (1.0 / 12.0) * 0.5 * ccpy_einsum("abef,dcfelkji->dcbalkji", H.bb.vvvv, T.abbb)  # (c/ab) = 3
    dT.abbb += (1.0 / 12.0) * ccpy_einsum("dafe,fcbelkji->dcbalkji", H.ab.vvvv, T.abbb)  # (a/bc) = 3

    dT.abbb += (9.0 / 36.0) * ccpy_einsum("amie,dcbelkjm->dcbalkji", H.bb.voov, T.abbb)  # (a/bc)(i/jk) = 9
    dT.abbb += (9.0 / 36.0) * ccpy_einsum("maei,decblmkj->dcbalkji", H.ab.ovvo, T.aabb)  # (a/bc)(i/jk) = 9
    dT.abbb += (1.0 / 36.0) * ccpy_einsum("dmle,abceijkm->dcbalkji", H.ab.voov, T.bbbb)  # (1) = 1
    dT.abbb += (1.0 / 36.0) * ccpy_einsum("dmle,ecbamkji->dcbalkji", H.aa.voov, T.abbb)  # (1) = 1
    dT.abbb -= (1.0 / 12.0) * ccpy_einsum("male,dcbemkji->dcbalkji", H.ab.ovov, T.abbb)  # (a/bc) = 3
    dT.abbb -= (1.0 / 12.0) * ccpy_einsum("dmei,ecbalkjm->dcbalkji", H.ab.vovo, T.abbb)  # (i/jk) = 3

    I3C_vvvoov = (
        -0.5 * ccpy_einsum("mnef,dfcalnkm->dcalke", H0.bb.oovv, T.abbb)
        - ccpy_einsum("nmfe,dfcalnkm->dcalke", H0.ab.oovv, T.aabb)
    )
    dT.abbb += (9.0 / 36.0) * ccpy_einsum("dcalke,ebij->dcbalkji", I3C_vvvoov, T.bb)  # (b/ac)(k/ij) = 9

    I3D_vvvvoo = (
        -0.5 * ccpy_einsum("mnef,abcfmjkn->abcejk", H0.bb.oovv, T.bbbb)
        - ccpy_einsum("nmfe,fcbankjm->abcejk", H0.ab.oovv, T.abbb)
    )
    dT.abbb += (1.0 / 12.0) * ccpy_einsum("abcejk,deli->dcbalkji", I3D_vvvvoo, T.ab)  # (i/jk) = 3

    I3C_vvvvoo = (
        - ccpy_einsum("mnef,dfbamnji->dbaeji", H0.ab.oovv, T.abbb)
        - 0.5 * ccpy_einsum("nmfe,dfbamnji->dbaeji", H0.aa.oovv, T.aabb)
    )
    dT.abbb += (9.0 / 36.0) * ccpy_einsum("dbaeji,eclk->dcbalkji", I3C_vvvvoo, T.ab)  # (c/ab)(k/ij) = 9

    I3C_vovooo = (
        0.5 * ccpy_einsum("mnef,dfeclnik->dmclik", H0.bb.oovv, T.abbb)
        + ccpy_einsum("nmfe,dfeclnik->dmclik", H0.ab.oovv, T.aabb)
    )
    dT.abbb -= (9.0 / 36.0) * ccpy_einsum("dmclik,abmj->dcbalkji", I3C_vovooo, T.bb)  # (c/ab)(j/ik) = 9

    I3D_vovooo = (
        0.5 * ccpy_einsum("mnef,bcefjkin->bmcjik", H0.bb.oovv, T.bbbb)
        + ccpy_einsum("nmfe,fecbnikj->bmcjik", H0.ab.oovv, T.abbb)
    )
    dT.abbb -= (1.0 / 12.0) * ccpy_einsum("bmcjik,dalm->dcbalkji", I3D_vovooo, T.ab)  # (a/bc) = 3

    I3C_ovvooo = (
        ccpy_einsum("mnef,efcblnkj->mcblkj", H0.ab.oovv, T.abbb)
        + 0.5 * ccpy_einsum("nmfe,efcblnkj->mcblkj", H0.aa.oovv, T.aabb)
    )
    dT.abbb -= (9.0 / 36.0) * ccpy_einsum("mcblkj,dami->dcbalkji", I3C_ovvooo, T.ab)  # (a/bc)(i/jk) = 9

    T.abbb, dT.abbb = cc_loops_t4.update_t4d(
        T.abbb,
        dT.abbb,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift,
    )
    return T, dT

def update_t4e(T, dT, H, H0, shift):
    """
    Update t4e amplitudes by calculating the projection <i~j~k~l~a~b~c~d~|(H_N e^(T1+T2+T3+T4))_C|0>.
    """
    # <ijklabcd | H(2) | 0 >
    dT.bbbb = -(144.0 / 576.0) * ccpy_einsum("amie,bcmk,edjl->abcdijkl", H.bb.voov, T.bb, T.bb)  # (jl/i/k)(bc/a/d) = 12 * 12 = 144
    dT.bbbb += (36.0 / 576.0) * ccpy_einsum("mnij,adml,bcnk->abcdijkl", H.bb.oooo, T.bb, T.bb)  # (ij/kl)(bc/ad) = 6 * 6 = 36
    dT.bbbb += (36.0 / 576.0) * ccpy_einsum("abef,fcjk,edil->abcdijkl", H.bb.vvvv, T.bb, T.bb)  # (jk/il)(ab/cd) = 6 * 6 = 36

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    dT.bbbb += (24.0 / 576.0) * ccpy_einsum("cdke,abeijl->abcdijkl", H.bb.vvov, T.bbb)  # (cd/ab)(k/ijl) = 6 * 4 = 24
    dT.bbbb -= (24.0 / 576.0) * ccpy_einsum("cmkl,abdijm->abcdijkl", H.bb.vooo, T.bbb)  # (c/abd)(kl/ij) = 6 * 4 = 24

    I3D_vooooo = ccpy_einsum("nmle,bejk->bmnjkl", H.bb.ooov, T.bb)
    I3D_vooooo -= np.transpose(I3D_vooooo, (0, 1, 2, 5, 4, 3)) + np.transpose(I3D_vooooo, (0, 1, 2, 3, 5, 4))
    I3D_vooooo += 0.5 * ccpy_einsum("mnef,befjkl->bmnjkl", H0.bb.oovv, T.bbb)
    dT.bbbb += 0.5 * (16.0 / 576.0) * ccpy_einsum("bmnjkl,acdimn->abcdijkl", I3D_vooooo, T.bbb)  # (b/acd)(i/jkl) = 4 * 4 = 16

    I3D_vvvovv = -ccpy_einsum("dmfe,bcjm->bcdjef", H.bb.vovv, T.bb)
    I3D_vvvovv -= np.transpose(I3D_vvvovv, (2, 1, 0, 3, 4, 5)) + np.transpose(I3D_vvvovv, (0, 2, 1, 3, 4, 5))
    dT.bbbb += 0.5 * (16.0 / 576.0) * ccpy_einsum("bcdjef,aefikl->abcdijkl", I3D_vvvovv, T.bbb)  # (a/bcd)(j/ikl) = 4 * 4 = 16

    I3D_vvooov = (
            -0.5 * ccpy_einsum("nmke,cdnl->cdmkle", H.bb.ooov, T.bb)
            + 0.5 * ccpy_einsum("cmfe,fdkl->cdmkle", H.bb.vovv, T.bb)
            + 0.125 * ccpy_einsum("mnef,cdfkln->cdmkle", H0.bb.oovv, T.bbb)  # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
            + 0.25 * ccpy_einsum("nmfe,fdcnlk->cdmkle", H0.ab.oovv, T.abb)
    )
    I3D_vvooov -= np.transpose(I3D_vvooov, (0, 1, 2, 4, 3, 5))
    I3D_vvooov -= np.transpose(I3D_vvooov, (1, 0, 2, 3, 4, 5))
    dT.bbbb += (36.0 / 576.0) * ccpy_einsum("cdmkle,abeijm->abcdijkl", I3D_vvooov, T.bbb)  # (cd/ab)(kl/ij) = 6 * 6 = 36

    I3C_ovvvoo = (
            -0.5 * ccpy_einsum("mnek,cdnl->mdcelk", H.ab.oovo, T.bb)
            + 0.5 * ccpy_einsum("mcef,fdkl->mdcelk", H.ab.ovvv, T.bb)
            + 0.125 * ccpy_einsum("mnef,fdcnlk->mdcelk", H0.aa.oovv, T.abb)
    # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
    )
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 2, 1, 3, 4, 5))
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 1, 2, 3, 5, 4))
    dT.bbbb += (36.0 / 576.0) * ccpy_einsum("mdcelk,ebamji->dcbalkji", I3C_ovvvoo, T.abb)  # (cd/ab)(kl/ij) = 6 * 6 = 36

    # <ijklabcd | (H(2)*T4)_C | 0 >
    dT.bbbb -= (4.0 / 576.0) * ccpy_einsum("mi,abcdmjkl->abcdijkl", H.b.oo, T.bbbb)  # (l/ijk) = 4
    dT.bbbb += (4.0 / 576.0) * ccpy_einsum("ae,ebcdijkl->abcdijkl", H.b.vv, T.bbbb)  # (d/abc) = 4
    dT.bbbb += (6.0 / 576.0) * 0.5 * ccpy_einsum("mnij,abcdmnkl->abcdijkl", H.bb.oooo, T.bbbb)  # (kl/ij) = 6
    dT.bbbb += (6.0 / 576.0) * 0.5 * ccpy_einsum("abef,efcdijkl->abcdijkl", H.bb.vvvv, T.bbbb)  # (cd/ab) = 6
    dT.bbbb += (16.0 / 576.0) * ccpy_einsum("amie,ebcdmjkl->abcdijkl", H.bb.voov, T.bbbb)  # (d/abc)(l/ijk) = 16
    dT.bbbb += (16.0 / 576.0) * ccpy_einsum("maei,edcbmlkj->dcbalkji", H.ab.ovvo, T.abbb)  # (d/abc)(l/ijk) = 16

    I3D_vvvoov = (
            -0.5 * ccpy_einsum("mnef,bcdfjkmn->bcdjke", H0.bb.oovv, T.bbbb)
            - ccpy_einsum("nmfe,fdcbnmkj->bcdjke", H0.ab.oovv, T.abbb)
    )
    dT.bbbb += (24.0 / 576.0) * ccpy_einsum("bcdjke,aeil->abcdijkl", I3D_vvvoov, T.bb)  # (a/bcd)(jk/il) = 4 * 6 = 24

    I3D_vvoooo = (
            0.5 * ccpy_einsum("mnef,bcefjkln->bcmjkl", H0.bb.oovv, T.bbbb)
            + ccpy_einsum("nmfe,fecbnlkj->bcmjkl", H0.ab.oovv, T.abbb)
    )
    dT.bbbb -= (24.0 / 576.0) * ccpy_einsum("bcmjkl,adim->abcdijkl", I3D_vvoooo, T.bb)  # (bc/ad)(i/jkl) = 6 * 4 = 24

    T.bbbb, dT.bbbb = cc_loops_t4.update_t4e(
        T.bbbb,
        dT.bbbb,
        H0.b.oo,
        H0.b.vv,
        shift,
    )
    return T, dT
