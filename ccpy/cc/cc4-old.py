"""Module with functions that perform the CC with singles, doubles,
triples, and quadruples (CCSDTQ) calculation for a molecular system.
***This is a working version that was restored from a commit on GitHub
 dated 05/05/2022. The former commit ID is c585654.***"""

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

from ccpy.lib.core import cc_loops2, cc4_loops
from ccpy.hbar.hbar_cc4 import get_cc4_intermediates
#from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
#from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates

def update(T, dT, H, X, shift, flag_RHF, system):

    #X = get_pre_ccs_intermediates(X, T, H, system, flag_RHF)

    # CCS-like transformed intermediates for CC3
    HT1 = get_cc4_intermediates(T, H)
    # Build the CC4 T4
    t4 = compute_t4(T, HT1, H, flag_RHF)

    # update T1
    T, dT = update_t1a(T, dT, H, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, shift)

    # CCS intermediates
    X = get_ccs_intermediates_opt(T, H)
    #X = get_ccs_intermediates_opt(X, T, H, system, flag_RHF)

    # update T2
    T, dT = update_t2a(T, t4, dT, X, H, shift)
    T, dT = update_t2b(T, t4, dT, X, H, shift)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, t4, dT, X, H, shift)

    # CCSD intermediates
    # [TODO]: Should accept CCS HBar as input and build only terms with T2 in it
    X = get_ccsd_intermediates(T, H)
    #X = get_ccsd_intermediates(T, X, H, flag_RHF)

    # update T3
    T, dT = update_t3a(T, t4, dT, X, H, shift)
    T, dT = update_t3b(T, t4, dT, X, H, shift)
    if flag_RHF:
        T.abb = np.transpose(T.aab, (2, 1, 0, 5, 4, 3))
        dT.abb = np.transpose(dT.aab, (2, 1, 0, 5, 4, 3))
        T.bbb = T.aaa.copy()
        dT.bbb = dT.aaa.copy()
    else:
        T, dT = update_t3c(T, t4, dT, X, H, shift)
        T, dT = update_t3d(T, t4, dT, X, H, shift)


    return T, dT


def update_t1a(T, dT, H, shift):
    """
    Update t1a amplitudes by calculating the projection <ia|(H_N e^(T1+T2+T3))_C|0>.
    """
    chi1A_vv = H.a.vv.copy()
    chi1A_vv += ccpy_einsum("anef,fn->ae", H.aa.vovv, T.a)
    chi1A_vv += ccpy_einsum("anef,fn->ae", H.ab.vovv, T.b)

    chi1A_oo = H.a.oo.copy()
    chi1A_oo += ccpy_einsum("mnif,fn->mi", H.aa.ooov, T.a)
    chi1A_oo += ccpy_einsum("mnif,fn->mi", H.ab.ooov, T.b)

    h1A_ov = H.a.ov.copy()
    h1A_ov += ccpy_einsum("mnef,fn->me", H.aa.oovv, T.a)
    h1A_ov += ccpy_einsum("mnef,fn->me", H.ab.oovv, T.b)

    h1B_ov = H.b.ov.copy()
    h1B_ov += ccpy_einsum("nmfe,fn->me", H.ab.oovv, T.a)
    h1B_ov += ccpy_einsum("mnef,fn->me", H.bb.oovv, T.b)

    h1A_oo = chi1A_oo.copy()
    h1A_oo += ccpy_einsum("me,ei->mi", h1A_ov, T.a)

    h2A_ooov = H.aa.ooov + ccpy_einsum("mnfe,fi->mnie", H.aa.oovv, T.a)
    h2B_ooov = H.ab.ooov + ccpy_einsum("mnfe,fi->mnie", H.ab.oovv, T.a)
    h2A_vovv = H.aa.vovv - ccpy_einsum("mnfe,an->amef", H.aa.oovv, T.a)
    h2B_vovv = H.ab.vovv - ccpy_einsum("nmef,an->amef", H.ab.oovv, T.a)

    dT.a = -ccpy_einsum("mi,am->ai", h1A_oo, T.a)
    dT.a += ccpy_einsum("ae,ei->ai", chi1A_vv, T.a)
    dT.a += ccpy_einsum("anif,fn->ai", H.aa.voov, T.a)
    dT.a += ccpy_einsum("anif,fn->ai", H.ab.voov, T.b)
    dT.a += ccpy_einsum("me,aeim->ai", h1A_ov, T.aa)
    dT.a += ccpy_einsum("me,aeim->ai", h1B_ov, T.ab)
    dT.a -= 0.5 * ccpy_einsum("mnif,afmn->ai", h2A_ooov, T.aa)
    dT.a -= ccpy_einsum("mnif,afmn->ai", h2B_ooov, T.ab)
    dT.a += 0.5 * ccpy_einsum("anef,efin->ai", h2A_vovv, T.aa)
    dT.a += ccpy_einsum("anef,efin->ai", h2B_vovv, T.ab)
    # T3 parts
    dT.a += 0.25 * ccpy_einsum("mnef,aefimn->ai", H.aa.oovv, T.aaa)
    dT.a += ccpy_einsum("mnef,aefimn->ai", H.ab.oovv, T.aab)
    dT.a += 0.25 * ccpy_einsum("mnef,aefimn->ai", H.bb.oovv, T.abb)

    T.a, dT.a = cc_loops2.cc_loops2.update_t1a(
        T.a,
        dT.a + H.a.vo,
        H.a.oo,
        H.a.vv,
        shift,
    )
    return T, dT


def update_t1b(T, dT, H, shift):
    """
    Update t1b amplitudes by calculating the projection <i~a~|(H_N e^(T1+T2+T3))_C|0>.
    """
    # Intermediates
    chi1B_vv = H.b.vv.copy()
    chi1B_vv += ccpy_einsum("anef,fn->ae", H.bb.vovv, T.b)
    chi1B_vv += ccpy_einsum("nafe,fn->ae", H.ab.ovvv, T.a)

    chi1B_oo = H.b.oo.copy()
    chi1B_oo += ccpy_einsum("mnif,fn->mi", H.bb.ooov, T.b)
    chi1B_oo += ccpy_einsum("nmfi,fn->mi", H.ab.oovo, T.a)

    h1A_ov = H.a.ov.copy()
    h1A_ov += ccpy_einsum("mnef,fn->me", H.aa.oovv, T.a)
    h1A_ov += ccpy_einsum("mnef,fn->me", H.ab.oovv, T.b)

    h1B_ov = H.b.ov.copy()
    h1B_ov += ccpy_einsum("nmfe,fn->me", H.ab.oovv, T.a)
    h1B_ov += ccpy_einsum("mnef,fn->me", H.bb.oovv, T.b)

    h1B_oo = chi1B_oo + ccpy_einsum("me,ei->mi", h1B_ov, T.b)

    h2C_ooov = H.bb.ooov + ccpy_einsum("mnfe,fi->mnie", H.bb.oovv, T.b)
    h2B_oovo = H.ab.oovo + ccpy_einsum("nmef,fi->nmei", H.ab.oovv, T.b)
    h2C_vovv = H.bb.vovv - ccpy_einsum("mnfe,an->amef", H.bb.oovv, T.b)
    h2B_ovvv = H.ab.ovvv - ccpy_einsum("mnfe,an->mafe", H.ab.oovv, T.b)

    dT.b = -ccpy_einsum("mi,am->ai", h1B_oo, T.b)
    dT.b += ccpy_einsum("ae,ei->ai", chi1B_vv, T.b)
    dT.b += ccpy_einsum("anif,fn->ai", H.bb.voov, T.b)
    dT.b += ccpy_einsum("nafi,fn->ai", H.ab.ovvo, T.a)
    dT.b += ccpy_einsum("me,eami->ai", h1A_ov, T.ab)
    dT.b += ccpy_einsum("me,aeim->ai", h1B_ov, T.bb)
    dT.b -= 0.5 * ccpy_einsum("mnif,afmn->ai", h2C_ooov, T.bb)
    dT.b -= ccpy_einsum("nmfi,fanm->ai", h2B_oovo, T.ab)
    dT.b += 0.5 * ccpy_einsum("anef,efin->ai", h2C_vovv, T.bb)
    dT.b += ccpy_einsum("nafe,feni->ai", h2B_ovvv, T.ab)
    # T3 parts
    dT.b += 0.25 * ccpy_einsum("mnef,aefimn->ai", H.bb.oovv, T.bbb)
    dT.b += 0.25 * ccpy_einsum("mnef,efamni->ai", H.aa.oovv, T.aab)
    dT.b += ccpy_einsum("mnef,efamni->ai", H.ab.oovv, T.abb)

    T.b, dT.b = cc_loops2.cc_loops2.update_t1b(
        T.b,
        dT.b + H.b.vo,
        H.b.oo,
        H.b.vv,
        shift,
    )
    return T, dT


# @profile
def update_t2a(T, t4, dT, H, H0, shift):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2))_C|0>.
    """
    # intermediates
    I1A_oo = (
            H.a.oo
            + 0.5 * ccpy_einsum("mnef,efin->mi", H0.aa.oovv, T.aa)
            + ccpy_einsum("mnef,efin->mi", H0.ab.oovv, T.ab)
    )

    I1A_vv = (
            H.a.vv
            - 0.5 * ccpy_einsum("mnef,afmn->ae", H0.aa.oovv, T.aa)
            - ccpy_einsum("mnef,afmn->ae", H0.ab.oovv, T.ab)
    )

    I2A_voov = (
            H.aa.voov
            + 0.5 * ccpy_einsum("mnef,afin->amie", H0.aa.oovv, T.aa)
            + ccpy_einsum("mnef,afin->amie", H0.ab.oovv, T.ab)
    )

    I2A_oooo = H.aa.oooo + 0.5 * np.einsum(
        "mnef,efij->mnij", H0.aa.oovv, T.aa, optimize=True
    )

    I2B_voov = H.ab.voov + 0.5 * np.einsum(
        "mnef,afin->amie", H0.bb.oovv, T.ab, optimize=True
    )

    I2A_vooo = H.aa.vooo + 0.5 * ccpy_einsum('anef,efij->anij', H0.aa.vovv + 0.5 * H.aa.vovv, T.aa)

    tau = 0.5 * T.aa + ccpy_einsum('ai,bj->abij', T.a, T.a)

    dT.aa = -0.5 * ccpy_einsum("amij,bm->abij", I2A_vooo, T.a)
    dT.aa += 0.5 * ccpy_einsum("abie,ej->abij", H.aa.vvov, T.a)
    dT.aa += 0.5 * ccpy_einsum("ae,ebij->abij", I1A_vv, T.aa)
    dT.aa -= 0.5 * ccpy_einsum("mi,abmj->abij", I1A_oo, T.aa)
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
    dT.aa += (1.0 / 4.0) * 0.25 * ccpy_einsum("mnef,abefijmn->abij", H0.aa.oovv, t4["aaaa"])
    dT.aa += (1.0 / 4.0) * ccpy_einsum("mnef,abefijmn->abij", H0.ab.oovv, t4["aaab"])
    dT.aa += (1.0 / 4.0) * 0.25 * ccpy_einsum("mnef,abefijmn->abij", H0.bb.oovv, t4["aabb"])

    T.aa, dT.aa = cc_loops2.cc_loops2.update_t2a(
        T.aa,
        dT.aa + 0.25 * H0.aa.vvoo,
        H0.a.oo,
        H0.a.vv,
        shift
    )
    return T, dT


# @profile
def update_t2b(T, t4, dT, H, H0, shift):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2))_C|0>.
    """
    # intermediates
    I1A_vv = (
            H.a.vv
            - 0.5 * ccpy_einsum("mnef,afmn->ae", H0.aa.oovv, T.aa)
            - ccpy_einsum("mnef,afmn->ae", H0.ab.oovv, T.ab)
    )

    I1B_vv = (
            H.b.vv
            - ccpy_einsum("nmfe,fbnm->be", H0.ab.oovv, T.ab)
            - 0.5 * ccpy_einsum("mnef,fbnm->be", H0.bb.oovv, T.bb)
    )

    I1A_oo = (
            H.a.oo
            + 0.5 * ccpy_einsum("mnef,efin->mi", H0.aa.oovv, T.aa)
            + ccpy_einsum("mnef,efin->mi", H0.ab.oovv, T.ab)
    )

    I1B_oo = (
            H.b.oo
            + ccpy_einsum("nmfe,fenj->mj", H0.ab.oovv, T.ab)
            + 0.5 * ccpy_einsum("mnef,efjn->mj", H0.bb.oovv, T.bb)
    )

    I2A_voov = (
            H.aa.voov
            + ccpy_einsum("mnef,aeim->anif", H0.aa.oovv, T.aa)
            + ccpy_einsum("nmfe,aeim->anif", H0.ab.oovv, T.ab)
    )

    I2B_voov = (
            H.ab.voov
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
    dT.ab += ccpy_einsum("ae,ebij->abij", I1A_vv, T.ab)
    dT.ab += ccpy_einsum("be,aeij->abij", I1B_vv, T.ab)
    dT.ab -= ccpy_einsum("mi,abmj->abij", I1A_oo, T.ab)
    dT.ab -= ccpy_einsum("mj,abim->abij", I1B_oo, T.ab)
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
    dT.ab += 0.25 * ccpy_einsum("mnef,aefbimnj->abij", H0.aa.oovv, t4["aaab"])
    dT.ab += ccpy_einsum("mnef,aefbimnj->abij", H0.ab.oovv, t4["aabb"])
    dT.ab += 0.25 * ccpy_einsum("mnef,abefijmn->abij", H0.bb.oovv, t4["abbb"])

    T.ab, dT.ab = cc_loops2.cc_loops2.update_t2b(
        T.ab,
        dT.ab + H0.ab.vvoo,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift
    )
    return T, dT


# @profile
def update_t2c(T, t4, dT, H, H0, shift):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2))_C|0>.
    """
    # intermediates
    I1B_oo = (
            H.b.oo
            + 0.5 * ccpy_einsum("mnef,efin->mi", H0.bb.oovv, T.bb)
            + ccpy_einsum("nmfe,feni->mi", H0.ab.oovv, T.ab)
    )

    I1B_vv = (
            H.b.vv
            - 0.5 * ccpy_einsum("mnef,afmn->ae", H0.bb.oovv, T.bb)
            - ccpy_einsum("nmfe,fanm->ae", H0.ab.oovv, T.ab)
    )

    I2C_oooo = H.bb.oooo + 0.5 * np.einsum(
        "mnef,efij->mnij", H0.bb.oovv, T.bb, optimize=True
    )

    I2B_ovvo = (
            H.ab.ovvo
            + ccpy_einsum("mnef,afin->maei", H0.ab.oovv, T.bb)
            + 0.5 * ccpy_einsum("mnef,fani->maei", H0.aa.oovv, T.ab)
    )

    I2C_voov = H.bb.voov + 0.5 * np.einsum(
        "mnef,afin->amie", H0.bb.oovv, T.bb, optimize=True
    )

    I2C_vooo = H.bb.vooo + 0.5 * ccpy_einsum('anef,efij->anij', H0.bb.vovv + 0.5 * H.bb.vovv, T.bb)

    tau = 0.5 * T.bb + ccpy_einsum('ai,bj->abij', T.b, T.b)

    dT.bb = -0.5 * ccpy_einsum("amij,bm->abij", I2C_vooo, T.b)
    dT.bb += 0.5 * ccpy_einsum("abie,ej->abij", H.bb.vvov, T.b)
    dT.bb += 0.5 * ccpy_einsum("ae,ebij->abij", I1B_vv, T.bb)
    dT.bb -= 0.5 * ccpy_einsum("mi,abmj->abij", I1B_oo, T.bb)
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
    dT.bb += 0.0625 * ccpy_einsum("mnef,abefijmn->abij", H0.bb.oovv, t4["bbbb"])
    dT.bb += 0.25 * ccpy_einsum("nmfe,febanmji->abij", H0.ab.oovv, t4["abbb"])
    dT.bb += 0.0625 * ccpy_einsum("mnef,febanmji->abij", H0.aa.oovv, t4["aabb"])

    T.bb, dT.bb = cc_loops2.cc_loops2.update_t2c(
        T.bb,
        dT.bb + 0.25 * H0.bb.vvoo,
        H0.b.oo,
        H0.b.vv,
        shift
    )
    return T, dT


# @profile
def update_t3a(T, t4, dT, H, H0, shift):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3))_C|0>.
    """
    # <ijkabc | H(2) | 0 > + (VT3)_C intermediates
    I2A_vvov = -0.5 * ccpy_einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa)
    I2A_vvov -= ccpy_einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab)
    I2A_vvov += H.aa.vvov + ccpy_einsum("me,abim->abie", H.a.ov, T.aa)

    I2A_vooo = 0.5 * ccpy_einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa)
    I2A_vooo += H.aa.vooo + ccpy_einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab)

    # MM(2,3)A
    dT.aaa = -0.25 * ccpy_einsum("amij,bcmk->abcijk", I2A_vooo, T.aa) # (a/bc)(k/ij) = 9
    dT.aaa += 0.25 * ccpy_einsum("abie,ecjk->abcijk", I2A_vvov, T.aa) # (c/ab)(i/jk) = 9
    # (HBar*T3)_C
    dT.aaa -= (1.0 / 12.0) * ccpy_einsum("mk,abcijm->abcijk", H.a.oo, T.aaa) # (k/ij) = 3
    dT.aaa += (1.0 / 12.0) * ccpy_einsum("ce,abeijk->abcijk", H.a.vv, T.aaa) # (c/ab) = 3
    dT.aaa += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa) # (k/ij) = 3
    dT.aaa += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa) # (c/ab) = 3
    dT.aaa += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa) # (c/ij)(k/ij) = 9
    dT.aaa += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab) # (c/ij)(k/ij) = 9
    # (HBar*T4)_C
    # dT.aaa += (1.0 / 36.0) * ccpy_einsum("me,abceijkm->abcijk", H.a.ov, t4["aaaa"]) # (1) = 1
    # dT.aaa += (1.0 / 36.0) * ccpy_einsum("me,abceijkm->abcijk", H.b.ov, t4["aaab"]) # (1) = 1
    dT.aaa += (1.0 / 24.0) * ccpy_einsum("cnef,abefijkn->abcijk", H.aa.vovv, t4["aaaa"]) # (c/ab) = 3
    dT.aaa += (1.0 / 12.0) * ccpy_einsum("cnef,abefijkn->abcijk", H.ab.vovv, t4["aaab"]) # (c/ab) = 3
    dT.aaa -= (1.0 / 24.0) * ccpy_einsum("mnkf,abcfijmn->abcijk", H.aa.ooov, t4["aaaa"]) # (k/ij) = 3
    dT.aaa -= (1.0 / 12.0) * ccpy_einsum("mnkf,abcfijmn->abcijk", H.ab.ooov, t4["aaab"]) # (k/ij) = 3

    T.aaa, dT.aaa = cc_loops2.cc_loops2.update_t3a_v2(
        T.aaa,
        dT.aaa,
        H0.a.oo,
        H0.a.vv,
        shift,
    )
    return T, dT


# @profile
def update_t3b(T, t4, dT, H, H0, shift):
    """
    Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N e^(T1+T2+T3))_C|0>.
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
    # dT.aab += 0.25 * ccpy_einsum("me,abecijmk->abcijk", H.a.ov, t4["aaab"]) # (1) = 1
    # dT.aab += 0.25 * ccpy_einsum("me,abecijmk->abcijk", H.b.ov, t4["aabb"]) # (1) = 1
    dT.aab -= 0.25 * ccpy_einsum("mnjf,abfcimnk->abcijk", H.aa.ooov, t4["aaab"]) # (ij) = 2
    dT.aab -= 0.5 * ccpy_einsum("mnjf,abfcimnk->abcijk", H.ab.ooov, t4["aabb"]) # (ij) = 2
    dT.aab -= 0.25 * ccpy_einsum("nmfk,abfcijnm->abcijk", H.ab.oovo, t4["aaab"]) # (1) = 1
    dT.aab -= 0.125 * ccpy_einsum("mnkf,abfcijnm->abcijk", H.bb.ooov, t4["aabb"]) # (1) = 1
    dT.aab += 0.25 * ccpy_einsum("bnef,aefcijnk->abcijk", H.aa.vovv, t4["aaab"]) # (ab) = 2
    dT.aab += 0.5 * ccpy_einsum("bnef,aefcijnk->abcijk", H.ab.vovv, t4["aabb"]) # (ab) = 2
    dT.aab += 0.25 * ccpy_einsum("ncfe,abfeijnk->abcijk", H.ab.ovvv, t4["aaab"]) # (1) = 1
    dT.aab += 0.125 * ccpy_einsum("cnef,abfeijnk->abcijk", H.bb.vovv, t4["aabb"]) # (1) = 1

    T.aab, dT.aab = cc_loops2.cc_loops2.update_t3b_v2(
        T.aab,
        dT.aab,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift,
    )
    return T, dT


# @profile
def update_t3c(T, t4, dT, H, H0, shift):
    """
    Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N e^(T1+T2+T3))_C|0>.
    """
    # <ij~k~ab~c~ | H(2) | 0 > + (VT3)_C intermediates
    I2B_vvvo = -0.5 * ccpy_einsum("mnef,afbmnj->abej", H0.aa.oovv, T.aab)
    I2B_vvvo += -ccpy_einsum("mnef,afbmnj->abej", H0.ab.oovv, T.abb)
    I2B_vvvo += H.ab.vvvo

    I2B_ovoo = 0.5 * ccpy_einsum("mnef,efbinj->mbij", H0.aa.oovv, T.aab)
    I2B_ovoo += ccpy_einsum("mnef,efbinj->mbij", H0.ab.oovv, T.abb)
    I2B_ovoo += H.ab.ovoo

    I2B_ovoo -= ccpy_einsum("me,ebij->mbij", H.a.ov, T.ab)
    I2B_vvov = -ccpy_einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab)
    I2B_vvov += -0.5 * ccpy_einsum("nmfe,afbinm->abie", H0.bb.oovv, T.abb)
    I2B_vvov += H.ab.vvov

    I2B_vooo = ccpy_einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab)
    I2B_vooo += 0.5 * ccpy_einsum("nmfe,afeinj->amij", H0.bb.oovv, T.abb)
    I2B_vooo += H.ab.vooo

    I2B_vooo -= ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
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
    # dT.abb += 0.25 * ccpy_einsum("me,cebakmji->cbakji", H.b.ov, t4["abbb"])
    # dT.abb += 0.25 * ccpy_einsum("me,cebakmji->cbakji", H.a.ov, t4["aabb"])
    dT.abb -= 0.25 * ccpy_einsum("mnjf,cfbaknmi->cbakji", H.bb.ooov, t4["abbb"])
    dT.abb -= 0.5 * ccpy_einsum("nmfj,cfbaknmi->cbakji", H.ab.oovo, t4["aabb"])
    dT.abb -= 0.25 * ccpy_einsum("mnkf,cfbamnji->cbakji", H.ab.ooov, t4["abbb"])
    dT.abb -= 0.125 * ccpy_einsum("mnkf,cfbamnji->cbakji", H.aa.ooov, t4["aabb"])
    dT.abb += 0.25 * ccpy_einsum("bnef,cfeaknji->cbakji", H.bb.vovv, t4["abbb"])
    dT.abb += 0.5 * ccpy_einsum("nbfe,cfeaknji->cbakji", H.ab.ovvv, t4["aabb"])
    dT.abb += 0.25 * ccpy_einsum("cnef,efbaknji->cbakji", H.ab.vovv, t4["abbb"])
    dT.abb += 0.125 * ccpy_einsum("cnef,efbaknji->cbakji", H.aa.vovv, t4["aabb"])

    T.abb, dT.abb = cc_loops2.cc_loops2.update_t3c_v2(
        T.abb,
        dT.abb,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift,
    )
    return T, dT


# @profile
def update_t3d(T, t4, dT, H, H0, shift):
    """
    Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N e^(T1+T2+T3))_C|0>.
    """
    #  <ijkabc | H(2) | 0 > + (VT3)_C intermediates
    I2C_vvov = -0.5 * ccpy_einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb)
    I2C_vvov -= ccpy_einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb)
    I2C_vvov += ccpy_einsum("me,abim->abie", H.b.ov, T.bb)
    I2C_vvov += H.bb.vvov

    I2C_vooo = 0.5 * ccpy_einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb)
    I2C_vooo += ccpy_einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb)
    I2C_vooo += H.bb.vooo
    # <ijkabc | H(2) | 0 >
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
    # dT.bbb += (1.0 / 36.0) * ccpy_einsum("me,abceijkm->abcijk", H.b.ov, t4["bbbb"])
    # dT.bbb += (1.0 / 36.0) * ccpy_einsum("me,ecbamkji->abcijk", H.a.ov, t4["abbb"])
    dT.bbb += (1.0 / 24.0) * ccpy_einsum("cnef,abefijkn->abcijk", H.bb.vovv, t4["bbbb"]) # (c/ab) = 3
    dT.bbb += (1.0 / 12.0) * ccpy_einsum("ncfe,febankji->abcijk", H.ab.ovvv, t4["abbb"]) # (c/ab) = 3
    dT.bbb -= (1.0 / 24.0) * ccpy_einsum("mnkf,abcfijmn->abcijk", H.bb.ooov, t4["bbbb"]) # (k/ij) = 3
    dT.bbb -= (1.0 / 12.0) * ccpy_einsum("nmfk,fcbanmji->abcijk", H.ab.oovo, t4["abbb"]) # (k/ij) = 3

    T.bbb, dT.bbb = cc_loops2.cc_loops2.update_t3d_v2(
        T.bbb,
        dT.bbb,
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

def update_t4a(T, HT1, H0):
    oa = np.diagonal(H0.a.oo)
    va = np.diagonal(H0.a.vv)
    n = np.newaxis
    e_abcdijkl = 1.0 / (- va[:, n, n, n, n, n, n, n] - va[n, :, n, n, n, n, n, n] - va[n, n, :, n, n, n, n, n] - va[n, n, n, :, n, n, n, n]
                        + oa[n, n, n, n, :, n, n, n] + oa[n, n, n, n, n, :, n, n] + oa[n, n, n, n, n, n, :, n] + oa[n, n, n, n, n, n, n, :])

    # <ijklabcd | H(2) | 0 >
    t4a = -(144.0 / 576.0) * ccpy_einsum("amie,bcmk,edjl->abcdijkl", HT1["aa"]["voov"], T.aa, T.aa)  # (jl/i/k)(bc/a/d) = 12 * 12 = 144
    t4a += (36.0 / 576.0) * ccpy_einsum("mnij,adml,bcnk->abcdijkl", HT1["aa"]["oooo"], T.aa, T.aa)   # (ij/kl)(bc/ad) = 6 * 6 = 36
    t4a += (36.0 / 576.0) * ccpy_einsum("abef,fcjk,edil->abcdijkl", HT1["aa"]["vvvv"], T.aa, T.aa)   # (jk/il)(ab/cd) = 6 * 6 = 36

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    t4a += (24.0 / 576.0) * ccpy_einsum("cdke,abeijl->abcdijkl", HT1["aa"]["vvov"], T.aaa) # (cd/ab)(k/ijl) = 6 * 4 = 24
    t4a -= (24.0 / 576.0) * ccpy_einsum("cmkl,abdijm->abcdijkl", HT1["aa"]["vooo"], T.aaa) # (c/abd)(kl/ij) = 6 * 4 = 24

    # Divide by MP denominator
    #t4a = cc4_loops.cc4_loops.update_t4a(t4a, H0.a.oo, H0.a.vv)
    t4a -= np.transpose(t4a, (0, 1, 2, 3, 4, 6, 5, 7)) # (jk)
    t4a -= np.transpose(t4a, (0, 1, 2, 3, 4, 7, 6, 5)) + np.transpose(t4a, (0, 1, 2, 3, 4, 5, 7, 6)) # (l/jk)
    t4a -= np.transpose(t4a, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(t4a, (0, 1, 2, 3, 6, 5, 4, 7)) + np.transpose(t4a, (0, 1, 2, 3, 7, 5, 6, 4)) # (i/jkl)

    t4a -= np.transpose(t4a, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    t4a -= np.transpose(t4a, (0, 3, 2, 1, 4, 5, 6, 7)) + np.transpose(t4a, (0, 1, 3, 2, 4, 5, 6, 7)) # (d/bc)
    t4a -= np.transpose(t4a, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(t4a, (2, 1, 0, 3, 4, 5, 6, 7)) + np.transpose(t4a, (3, 1, 2, 0, 4, 5, 6, 7)) # (a/bcd)
    return t4a * e_abcdijkl
    #return t4a


def update_t4b(T, HT1, H0):
    oa = np.diagonal(H0.a.oo)
    ob = np.diagonal(H0.b.oo)
    va = np.diagonal(H0.a.vv)
    vb = np.diagonal(H0.b.vv)
    n = np.newaxis
    e_abcdijkl = 1.0 / (- va[:, n, n, n, n, n, n, n] - va[n, :, n, n, n, n, n, n] - va[n, n, :, n, n, n, n, n] - vb[n, n, n, :, n, n, n, n]
                        + oa[n, n, n, n, :, n, n, n] + oa[n, n, n, n, n, :, n, n] + oa[n, n, n, n, n, n, :, n] + ob[n, n, n, n, n, n, n, :])

    # <ijklabcd | H(2) | 0 >
    t4b = -(9.0 / 36.0) * ccpy_einsum("mdel,abim,ecjk->abcdijkl", HT1["ab"]["ovvo"], T.aa, T.aa)    # (i/jk)(c/ab) = 9
    t4b += (9.0 / 36.0) * ccpy_einsum("mnij,bcnk,adml->abcdijkl", HT1["aa"]["oooo"], T.aa, T.ab)    # (k/ij)(a/bc) = 9
    t4b -= (18.0 / 36.0) * ccpy_einsum("mdjf,abim,cfkl->abcdijkl", HT1["ab"]["ovov"], T.aa, T.ab)   # (ijk)(c/ab) = (i/jk)(c/ab)(jk) = 18
    t4b -= ccpy_einsum("amie,bejl,cdkm->abcdijkl", HT1["ab"]["voov"], T.ab, T.ab)                   # (ijk)(abc) = (i/jk)(a/bc)(jk)(bc) = 36
    t4b += (18.0 / 36.0) * ccpy_einsum("mnjl,bcmk,adin->abcdijkl", HT1["ab"]["oooo"], T.aa, T.ab)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    t4b -= (18.0 / 36.0) * ccpy_einsum("bmel,ecjk,adim->abcdijkl", HT1["ab"]["vovo"], T.aa, T.ab)   # (i/jk)(abc) = (i/jk)(a/bc)(bc) = 18
    t4b -= (18.0 / 36.0) * ccpy_einsum("amie,ecjk,bdml->abcdijkl", HT1["aa"]["voov"], T.aa, T.ab)   # (i/kj)(abc) = (i/kj)(a/bc)(bc) = 18
    t4b += (9.0 / 36.0) * ccpy_einsum("abef,fcjk,edil->abcdijkl", HT1["aa"]["vvvv"], T.aa, T.ab)    # (i/jk)(c/ab) = (i/jk)(c/ab) = 9
    t4b -= (18.0 / 36.0) * ccpy_einsum("amie,bcmk,edjl->abcdijkl", HT1["aa"]["voov"], T.aa, T.ab)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    t4b += (18.0 / 36.0) * ccpy_einsum("adef,ebij,cfkl->abcdijkl", HT1["ab"]["vvvv"], T.aa, T.ab)   # (k/ij)(abc) = (k/ij)(a/bc)(bc) = 18
    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    t4b -= (1.0 / 12.0) * ccpy_einsum("mdkl,abcijm->abcdijkl", HT1["ab"]["ovoo"], T.aaa)  # (k/ij) = 3
    t4b -= (9.0 / 36.0) * ccpy_einsum("amik,bcdjml->abcdijkl", HT1["aa"]["vooo"], T.aab)  # (j/ik)(a/bc) = 9
    t4b -= (9.0 / 36.0) * ccpy_einsum("amil,bcdjkm->abcdijkl", HT1["ab"]["vooo"], T.aab)  # (a/bc)(i/jk) = 9
    t4b += (1.0 / 12.0) * ccpy_einsum("cdel,abeijk->abcdijkl", HT1["ab"]["vvvo"], T.aaa)  # (c/ab) = 3
    t4b += (9.0 / 36.0) * ccpy_einsum("acie,bedjkl->abcdijkl", HT1["aa"]["vvov"], T.aab)  # (b/ac)(i/jk) = 9
    t4b += (9.0 / 36.0) * ccpy_einsum("adie,bcejkl->abcdijkl", HT1["ab"]["vvov"], T.aab)  # (a/bc)(i/jk) = 9

    # Divide by MP denominator
    #t4b = cc4_loops.cc4_loops.update_t4b(t4b, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv)
    t4b -= np.transpose(t4b, (0, 1, 2, 3, 4, 6, 5, 7))  # (jk)
    t4b -= np.transpose(t4b, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(t4b, (0, 1, 2, 3, 6, 5, 4, 7)) # (i/jk)

    t4b -= np.transpose(t4b, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    t4b -= np.transpose(t4b, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(t4b, (2, 1, 0, 3, 4, 5, 6, 7)) # (a/bc)
    return t4b * e_abcdijkl
    #return t4b


def update_t4c(T, HT1, H0):
    oa = np.diagonal(H0.a.oo)
    ob = np.diagonal(H0.b.oo)
    va = np.diagonal(H0.a.vv)
    vb = np.diagonal(H0.b.vv)
    n = np.newaxis
    e_abcdijkl = 1.0 / (- va[:, n, n, n, n, n, n, n] - va[n, :, n, n, n, n, n, n] - vb[n, n, :, n, n, n, n, n] - vb[n, n, n, :, n, n, n, n]
                        + oa[n, n, n, n, :, n, n, n] + oa[n, n, n, n, n, :, n, n] + ob[n, n, n, n, n, n, :, n] + ob[n, n, n, n, n, n, n, :])
    # <ijklabcd | H(2) | 0 >
    t4c = -ccpy_einsum("cmke,adim,bejl->abcdijkl", HT1["bb"]["voov"], T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    t4c -= ccpy_einsum("amie,bcmk,edjl->abcdijkl", HT1["aa"]["voov"], T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    t4c -= 0.5 * ccpy_einsum("mcek,aeij,bdml->abcdijkl", HT1["ab"]["ovvo"], T.aa, T.ab)    # (kl)(ab)(cd) = 8
    t4c -= 0.5 * ccpy_einsum("amie,bdjm,cekl->abcdijkl", HT1["ab"]["voov"], T.ab, T.bb)    # (ij)(ab)(cd) = 8
    t4c -= 0.5 * ccpy_einsum("mcek,abim,edjl->abcdijkl", HT1["ab"]["ovvo"], T.aa, T.ab)    # (ij)(kl)(cd) = 8
    t4c -= 0.5 * ccpy_einsum("amie,cdkm,bejl->abcdijkl", HT1["ab"]["voov"], T.bb, T.ab)    # (ij)(kl)(ab) = 8
    t4c -= ccpy_einsum("bmel,adim,ecjk->abcdijkl", HT1["ab"]["vovo"], T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    t4c -= ccpy_einsum("mdje,bcmk,aeil->abcdijkl", HT1["ab"]["ovov"], T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    t4c -= 0.25 * ccpy_einsum("mdje,abim,cekl->abcdijkl", HT1["ab"]["ovov"], T.aa, T.bb)   # (ij)(cd) = 4
    t4c -= 0.25 * ccpy_einsum("bmel,cdkm,aeij->abcdijkl", HT1["ab"]["vovo"], T.bb, T.aa)   # (kl)(ab) = 4
    t4c += 0.25 * ccpy_einsum("mnij,acmk,bdnl->abcdijkl", HT1["aa"]["oooo"], T.ab, T.ab)   # (kl)(ab) = 4 !!! (tricky asym)
    t4c += 0.25 * ccpy_einsum("abef,ecik,fdjl->abcdijkl", HT1["aa"]["vvvv"], T.ab, T.ab)   # (ij)(kl) = 4 !!! (tricky asym)
    t4c += 0.25 * ccpy_einsum("mnik,abmj,cdnl->abcdijkl", HT1["ab"]["oooo"], T.aa, T.bb)   # (ij)(kl) = 4
    t4c += 0.25 * ccpy_einsum("acef,ebij,fdkl->abcdijkl", HT1["ab"]["vvvv"], T.aa, T.bb)   # (ab)(cd) = 4
    t4c += ccpy_einsum("mnik,adml,bcjn->abcdijkl", HT1["ab"]["oooo"], T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    t4c += ccpy_einsum("acef,edil,bfjk->abcdijkl", HT1["ab"]["vvvv"], T.ab, T.ab)          # (ij)(kl)(ab)(cd) = 16
    t4c += 0.25 * ccpy_einsum("mnkl,adin,bcjm->abcdijkl", HT1["bb"]["oooo"], T.ab, T.ab)   # (ij)(cd) = 4 !!! (tricky asym)
    t4c += 0.25 * ccpy_einsum("cdef,afil,bejk->abcdijkl", HT1["bb"]["vvvv"], T.ab, T.ab)   # (ij)(kl) = 4 !!! (tricky asym)
    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    t4c -= (8.0 / 16.0) * ccpy_einsum("mdil,abcmjk->abcdijkl", HT1["ab"]["ovoo"], T.aab)  # [1]  (ij)(kl)(cd) = 8
    t4c -= (2.0 / 16.0) * ccpy_einsum("bmji,acdmkl->abcdijkl", HT1["aa"]["vooo"], T.abb)  # [2]  (ab) = 2
    t4c -= (2.0 / 16.0) * ccpy_einsum("cmkl,abdijm->abcdijkl", HT1["bb"]["vooo"], T.aab)  # [3]  (cd) = 2
    t4c -= (8.0 / 16.0) * ccpy_einsum("amil,bcdjkm->abcdijkl", HT1["ab"]["vooo"], T.abb)  # [4]  (ij)(ab)(kl) = 8
    t4c += (8.0 / 16.0) * ccpy_einsum("adel,becjik->abcdijkl", HT1["ab"]["vvvo"], T.aab)  # [5]  (ab)(kl)(cd) = 8
    t4c += (2.0 / 16.0) * ccpy_einsum("baje,ecdikl->abcdijkl", HT1["aa"]["vvov"], T.abb)  # [6]  (ij) = 2
    t4c += (8.0 / 16.0) * ccpy_einsum("adie,bcejkl->abcdijkl", HT1["ab"]["vvov"], T.abb)  # [7]  (ij)(ab)(cd) = 8
    t4c += (2.0 / 16.0) * ccpy_einsum("cdke,abeijl->abcdijkl", HT1["bb"]["vvov"], T.aab)  # [8]  (kl) = 2

    # Divide by MP denominator
    #t4c = cc4_loops.cc4_loops.update_t4c(t4c, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv)
    t4c -= np.transpose(t4c, (1, 0, 2, 3, 4, 5, 6, 7)) # (ab)
    t4c -= np.transpose(t4c, (0, 1, 3, 2, 4, 5, 6, 7)) # (cd)
    t4c -= np.transpose(t4c, (0, 1, 2, 3, 5, 4, 6, 7)) # (ij)
    t4c -= np.transpose(t4c, (0, 1, 2, 3, 4, 5, 7, 6)) # (kl)
    return t4c * e_abcdijkl
    #return t4c

def get_ccs_intermediates_opt(T, H0):
    """
    Calculate the CCS-like similarity-transformed HBar intermediates (H_N e^T1)_C.
    """

    # [TODO]: Copying large arrays is slow! We should pass in Hbar and simply update its elements.
    from copy import deepcopy
    # Copy the Bare Hamiltonian object for T1-transforemd HBar
    H = deepcopy(H0)

    # 1-body components
    # -------------------#
    H.a.ov = H0.a.ov + (
        ccpy_einsum("mnef,fn->me", H0.aa.oovv, T.a)
        + ccpy_einsum("mnef,fn->me", H0.ab.oovv, T.b)
    ) # no(2)nu(2)

    H.b.ov = H0.b.ov + (
            ccpy_einsum("nmfe,fn->me", H0.ab.oovv, T.a)
            + ccpy_einsum("mnef,fn->me", H0.bb.oovv, T.b)
    ) # no(2)nu(2)

    H.a.vv = H0.a.vv + (
        ccpy_einsum("anef,fn->ae", H0.aa.vovv, T.a)
        + ccpy_einsum("anef,fn->ae", H0.ab.vovv, T.b)
        - ccpy_einsum("me,am->ae", H.a.ov, T.a)
    ) # no(1)nu(3)

    H.a.oo = H0.a.oo + (
        ccpy_einsum("mnif,fn->mi", H0.aa.ooov, T.a)
        + ccpy_einsum("mnif,fn->mi", H0.ab.ooov, T.b)
        + ccpy_einsum("me,ei->mi", H.a.ov, T.a)
    ) # no(3)nu(1)

    H.b.vv = H0.b.vv + (
        ccpy_einsum("anef,fn->ae", H0.bb.vovv, T.b)
        + ccpy_einsum("nafe,fn->ae", H0.ab.ovvv, T.a)
        - ccpy_einsum("me,am->ae", H.b.ov, T.b)
    ) # no(1)nu(3)

    H.b.oo = H0.b.oo + (
        ccpy_einsum("mnif,fn->mi", H0.bb.ooov, T.b)
        + ccpy_einsum("nmfi,fn->mi", H0.ab.oovo, T.a)
        + ccpy_einsum("me,ei->mi", H.b.ov, T.b)
    ) # no(3)nu(1)

    # 2-body components
    # -------------------#
    # AA parts
    # -------------------#
    H.aa.ooov = ccpy_einsum("mnfe,fi->mnie", H0.aa.oovv, T.a) # no(3)nu(2)

    H.aa.oooo = 0.5 * H0.aa.oooo + ccpy_einsum("nmje,ei->mnij", H0.aa.ooov + 0.5 * H.aa.ooov, T.a) # no(4)nu(1)
    H.aa.oooo -= np.transpose(H.aa.oooo, (0, 1, 3, 2))

    H.aa.vovv = -ccpy_einsum("mnfe,an->amef", H0.aa.oovv, T.a) # no(2)nu(3)

    H.aa.voov = H0.aa.voov + (
            ccpy_einsum("amfe,fi->amie", H0.aa.vovv + 0.5 * H.aa.vovv, T.a)
            - ccpy_einsum("nmie,an->amie", H0.aa.ooov + 0.5 * H.aa.ooov, T.a)
    ) # no(2)nu(3)

    L_amie = H0.aa.voov + 0.5 * ccpy_einsum('amef,ei->amif', H0.aa.vovv, T.a) # no(2)nu(3)
    X_mnij = H0.aa.oooo + ccpy_einsum('mnie,ej->mnij', H.aa.ooov, T.a) # no(4)nu(1)
    H.aa.vooo = 0.5 * H0.aa.vooo + (
        ccpy_einsum('amie,ej->amij', L_amie, T.a)
       -0.25 * ccpy_einsum('mnij,am->anij', X_mnij, T.a)
    ) # no(3)nu(2)
    H.aa.vooo -= np.transpose(H.aa.vooo, (0, 1, 3, 2))

    L_amie = ccpy_einsum('mnie,am->anie', H0.aa.ooov, T.a)
    H.aa.vvov = H0.aa.vvov + ccpy_einsum("anie,bn->abie", L_amie, T.a) # no(1)nu(4)
    #H.aa.vvov -= np.transpose(H.aa.vvov, (1, 0, 2, 3)) # WHY IS THIS NOT NEEDED???

    # -------------------#
    # AB parts
    # -------------------#
    H.ab.ooov = ccpy_einsum("mnfe,fi->mnie", H0.ab.oovv, T.a)

    H.ab.oovo = ccpy_einsum("nmef,fi->nmei", H0.ab.oovv, T.b)

    H.ab.oooo = H0.ab.oooo + (
        ccpy_einsum("mnej,ei->mnij", H0.ab.oovo + 0.5 * H.ab.oovo, T.a)
        + ccpy_einsum("mnie,ej->mnij", H0.ab.ooov + 0.5 * H.ab.ooov, T.b)
    )

    H.ab.vovv = -ccpy_einsum("nmef,an->amef", H0.ab.oovv, T.a)

    H.ab.ovvv = -ccpy_einsum("mnef,an->maef", H0.ab.oovv, T.b)

    H.ab.voov = H0.ab.voov + (
        ccpy_einsum("amfe,fi->amie", H0.ab.vovv + 0.5 * H.ab.vovv, T.a)
        - ccpy_einsum("nmie,an->amie", H0.ab.ooov + 0.5 * H.ab.ooov, T.a)
    )

    H.ab.ovvo = H0.ab.ovvo + (
        ccpy_einsum("maef,fi->maei", H0.ab.ovvv + 0.5 * H.ab.ovvv, T.b)
        - ccpy_einsum("mnei,an->maei", H0.ab.oovo + 0.5 * H.ab.oovo, T.b)
    )

    H.ab.ovov = H0.ab.ovov + (
        ccpy_einsum("mafe,fi->maie", H0.ab.ovvv + 0.5 * H.ab.ovvv, T.a)
        - ccpy_einsum("mnie,an->maie", H0.ab.ooov + 0.5 * H.ab.ooov, T.b)
    )

    H.ab.vovo = H0.ab.vovo - (
        ccpy_einsum("nmei,an->amei", H0.ab.oovo + 0.5 * H.ab.oovo, T.a)
        - ccpy_einsum("amef,fi->amei", H0.ab.vovv + 0.5 * H.ab.vovv, T.b)
    )

    X_mnij = H0.ab.oooo + (
        ccpy_einsum("mnif,fj->mnij", H0.ab.ooov, T.b)
        +ccpy_einsum("mnej,ei->mnij", H0.ab.oovo, T.a)
    )
    L_mbej = H0.ab.ovvo + ccpy_einsum("mbef,fj->mbej", H0.ab.ovvv, T.b)
    H.ab.ovoo = H0.ab.ovoo + (
        ccpy_einsum("mbej,ei->mbij", L_mbej, T.a)
        -ccpy_einsum("mnij,bn->mbij", X_mnij, T.b)
    )

    L_amie = ccpy_einsum("amef,ei->amif", H0.ab.vovv + H.ab.vovv, T.a)
    H.ab.vooo = H0.ab.vooo + ccpy_einsum("amif,fj->amij", H0.ab.voov + L_amie, T.b)

    H.ab.vvvo = H0.ab.vvvo - ccpy_einsum("anej,bn->abej", H0.ab.vovo, T.b)

    H.ab.vvov = H0.ab.vvov - ccpy_einsum("mbie,am->abie", H0.ab.ovov, T.a)
    # -------------------#
    # BB parts
    # -------------------#
    H.bb.ooov = ccpy_einsum("mnfe,fi->mnie", H0.bb.oovv, T.b)

    H.bb.oooo = 0.5 * H0.bb.oooo + ccpy_einsum("nmje,ei->mnij", H0.bb.ooov + 0.5 * H.bb.ooov, T.b)
    H.bb.oooo -= np.transpose(H.bb.oooo, (0, 1, 3, 2))

    H.bb.vovv = -ccpy_einsum("mnfe,an->amef", H0.bb.oovv, T.b)

    H.bb.voov = H0.bb.voov + (
        ccpy_einsum("amfe,fi->amie", H0.bb.vovv + 0.5 * H.bb.vovv, T.b)
        - ccpy_einsum("nmie,an->amie", H0.bb.ooov + 0.5 * H.bb.ooov, T.b)
    )

    L_amie = H0.bb.voov + 0.5 * ccpy_einsum('amef,ei->amif', H0.bb.vovv, T.b)
    X_mnij = H0.bb.oooo + ccpy_einsum('mnie,ej->mnij', H.bb.ooov, T.b)
    H.bb.vooo = 0.5 * H0.bb.vooo + (
        ccpy_einsum('amie,ej->amij', L_amie, T.b)
       -0.25 * ccpy_einsum('mnij,am->anij', X_mnij, T.b)
    )
    H.bb.vooo -= np.transpose(H.bb.vooo, (0, 1, 3, 2))

    L_amie = ccpy_einsum('mnie,am->anie', H0.bb.ooov, T.b)
    H.bb.vvov = H0.bb.vvov + + ccpy_einsum("anie,bn->abie", L_amie, T.b)
    #H.bb.vvov -= np.transpose(H.bb.vvov, (1, 0, 2, 3))
    return H

def get_ccsd_intermediates(T, H0):
    """Calculate the CCSD-like intermediates for CCSDT. This routine
    should only calculate terms with T2 and any remaining terms outside of the CCS intermediate
    routine."""
    from copy import deepcopy

    # Copy the Bare Hamiltonian object for T1/T2-similarity transformed HBar
    H = deepcopy(H0)

    H.a.ov += (
            ccpy_einsum("imae,em->ia", H0.aa.oovv, T.a)
            + ccpy_einsum("imae,em->ia", H0.ab.oovv, T.b)
    )

    H.a.oo += (
            ccpy_einsum("je,ei->ji", H.a.ov, T.a)
            + ccpy_einsum("jmie,em->ji", H0.aa.ooov, T.a)
            + ccpy_einsum("jmie,em->ji", H0.ab.ooov, T.b)
            + 0.5 * ccpy_einsum("jnef,efin->ji", H0.aa.oovv, T.aa)
            + ccpy_einsum("jnef,efin->ji", H0.ab.oovv, T.ab)
    )

    H.a.vv += (
            - ccpy_einsum("mb,am->ab", H.a.ov, T.a)
            + ccpy_einsum("ambe,em->ab", H0.aa.vovv, T.a)
            + ccpy_einsum("ambe,em->ab", H0.ab.vovv, T.b)
            - 0.5 * ccpy_einsum("mnbf,afmn->ab", H0.aa.oovv, T.aa)
            - ccpy_einsum("mnbf,afmn->ab", H0.ab.oovv, T.ab)
    )

    H.b.ov += (
            ccpy_einsum("imae,em->ia", H0.bb.oovv, T.b)
            + ccpy_einsum("miea,em->ia", H0.ab.oovv, T.a)
    )

    H.b.oo += (
            ccpy_einsum("je,ei->ji", H.b.ov, T.b)
            + ccpy_einsum("jmie,em->ji", H0.bb.ooov, T.b)
            + ccpy_einsum("mjei,em->ji", H0.ab.oovo, T.a)
            + 0.5 * ccpy_einsum("jnef,efin->ji", H0.bb.oovv, T.bb)
            + ccpy_einsum("njfe,feni->ji", H0.ab.oovv, T.ab)
    )

    H.b.vv += (
            - ccpy_einsum("mb,am->ab", H.b.ov, T.b)
            + ccpy_einsum("ambe,em->ab", H0.bb.vovv, T.b)
            + ccpy_einsum("maeb,em->ab", H0.ab.ovvv, T.a)
            - 0.5 * ccpy_einsum("mnbf,afmn->ab", H0.bb.oovv, T.bb)
            - ccpy_einsum("nmfb,fanm->ab", H0.ab.oovv, T.ab)
    )

    Q1 = -ccpy_einsum("mnfe,an->amef", H0.aa.oovv, T.a)
    I2A_vovv = H0.aa.vovv + 0.5 * Q1
    H.aa.vovv = I2A_vovv + 0.5 * Q1

    Q1 = ccpy_einsum("mnfe,fi->mnie", H0.aa.oovv, T.a)
    I2A_ooov = H0.aa.ooov + 0.5 * Q1
    H.aa.ooov = I2A_ooov + 0.5 * Q1

    Q1 = -ccpy_einsum("nmef,an->amef", H0.ab.oovv, T.a)
    I2B_vovv = H0.ab.vovv + 0.5 * Q1
    H.ab.vovv = I2B_vovv + 0.5 * Q1

    Q1 = ccpy_einsum("mnfe,fi->mnie", H0.ab.oovv, T.a)
    I2B_ooov = H0.ab.ooov + 0.5 * Q1
    H.ab.ooov = I2B_ooov + 0.5 * Q1

    Q1 = -ccpy_einsum("mnef,an->maef", H0.ab.oovv, T.b)
    I2B_ovvv = H0.ab.ovvv + 0.5 * Q1
    H.ab.ovvv = I2B_ovvv + 0.5 * Q1

    Q1 = ccpy_einsum("nmef,fi->nmei", H0.ab.oovv, T.b)
    I2B_oovo = H0.ab.oovo + 0.5 * Q1
    H.ab.oovo = I2B_oovo + 0.5 * Q1

    Q1 = -ccpy_einsum("nmef,an->amef", H0.bb.oovv, T.b)
    I2C_vovv = H0.bb.vovv + 0.5 * Q1
    H.bb.vovv = I2C_vovv + 0.5 * Q1

    Q1 = ccpy_einsum("mnfe,fi->mnie", H0.bb.oovv, T.b)
    I2C_ooov = H0.bb.ooov + 0.5 * Q1
    H.bb.ooov = I2C_ooov + 0.5 * Q1

    Q1 = -ccpy_einsum("bmfe,am->abef", I2A_vovv, T.a)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvvv += 0.5 * ccpy_einsum("mnef,abmn->abef", H0.aa.oovv, T.aa) + Q1

    H.ab.vvvv += (
            - ccpy_einsum("mbef,am->abef", I2B_ovvv, T.a)
            - ccpy_einsum("amef,bm->abef", I2B_vovv, T.b)
            + ccpy_einsum("mnef,abmn->abef", H0.ab.oovv, T.ab)
    )

    Q1 = -ccpy_einsum("bmfe,am->abef", I2C_vovv, T.b)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvvv += 0.5 * ccpy_einsum("mnef,abmn->abef", H0.bb.oovv, T.bb) + Q1

    Q1 = +ccpy_einsum("nmje,ei->mnij", I2A_ooov, T.a)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.oooo += 0.5 * ccpy_einsum("mnef,efij->mnij", H0.aa.oovv, T.aa) + Q1

    H.ab.oooo += (
            ccpy_einsum("mnej,ei->mnij", I2B_oovo, T.a)
            + ccpy_einsum("mnie,ej->mnij", I2B_ooov, T.b)
            + ccpy_einsum("mnef,efij->mnij", H0.ab.oovv, T.ab)
    )

    Q1 = +ccpy_einsum("nmje,ei->mnij", I2C_ooov, T.b)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.oooo += 0.5 * ccpy_einsum("mnef,efij->mnij", H0.bb.oovv, T.bb) + Q1

    H.aa.voov += (
            ccpy_einsum("amfe,fi->amie", I2A_vovv, T.a)
            - ccpy_einsum("nmie,an->amie", I2A_ooov, T.a)
            + ccpy_einsum("nmfe,afin->amie", H0.aa.oovv, T.aa)
            + ccpy_einsum("mnef,afin->amie", H0.ab.oovv, T.ab)
    )

    H.ab.voov += (
            ccpy_einsum("amfe,fi->amie", I2B_vovv, T.a)
            - ccpy_einsum("nmie,an->amie", I2B_ooov, T.a)
            + ccpy_einsum("nmfe,afin->amie", H0.ab.oovv, T.aa)
            + ccpy_einsum("nmfe,afin->amie", H0.bb.oovv, T.ab)
    )

    H.ab.ovvo += (
            ccpy_einsum("maef,fi->maei", I2B_ovvv, T.b)
            - ccpy_einsum("mnei,an->maei", I2B_oovo, T.b)
            + ccpy_einsum("mnef,afin->maei", H0.ab.oovv, T.bb)
            + ccpy_einsum("mnef,fani->maei", H0.aa.oovv, T.ab)
    )

    H.ab.ovov += (
            ccpy_einsum("mafe,fi->maie", I2B_ovvv, T.a)
            - ccpy_einsum("mnie,an->maie", I2B_ooov, T.b)
            - ccpy_einsum("mnfe,fain->maie", H0.ab.oovv, T.ab)
    )

    H.ab.vovo += (
            - ccpy_einsum("nmei,an->amei", I2B_oovo, T.a)
            + ccpy_einsum("amef,fi->amei", I2B_vovv, T.b)
            - ccpy_einsum("nmef,afni->amei", H0.ab.oovv, T.ab)
    )

    H.bb.voov += (
            ccpy_einsum("amfe,fi->amie", I2C_vovv, T.b)
            - ccpy_einsum("nmie,an->amie", I2C_ooov, T.b)
            + ccpy_einsum("nmfe,afin->amie", H0.bb.oovv, T.bb)
            + ccpy_einsum("nmfe,fani->amie", H0.ab.oovv, T.ab)
    )

    Q1 = (
            ccpy_einsum("mnjf,afin->amij", H.aa.ooov, T.aa)
            + ccpy_einsum("mnjf,afin->amij", H.ab.ooov, T.ab)
    )
    Q2 = H0.aa.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.aa.vovv, T.a)
    Q2 = ccpy_einsum("amif,fj->amij", Q2, T.a)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.vooo += Q1 + (
            ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
            - ccpy_einsum("nmij,an->amij", H.aa.oooo, T.a)
            + 0.5 * ccpy_einsum("amef,efij->amij", H0.aa.vovv, T.aa)
    )

    Q1 = H0.ab.voov + ccpy_einsum("amfe,fi->amie", H0.ab.vovv, T.a)
    H.ab.vooo += (
            ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
            - ccpy_einsum("nmij,an->amij", H.ab.oooo, T.a)
            + ccpy_einsum("mnjf,afin->amij", H.bb.ooov, T.ab)
            + ccpy_einsum("nmfj,afin->amij", H.ab.oovo, T.aa)
            - ccpy_einsum("nmif,afnj->amij", H.ab.ooov, T.ab)
            + ccpy_einsum("amej,ei->amij", H0.ab.vovo, T.a)
            + ccpy_einsum("amie,ej->amij", Q1, T.b)
            + ccpy_einsum("amef,efij->amij", H0.ab.vovv, T.ab)
    )

    Q1 = H0.ab.ovov + ccpy_einsum("mafe,fj->maje", H0.ab.ovvv, T.a)
    H.ab.ovoo += (
            ccpy_einsum("me,eaji->maji", H.a.ov, T.ab)
            - ccpy_einsum("mnji,an->maji", H.ab.oooo, T.b)
            + ccpy_einsum("mnjf,fani->maji", H.aa.ooov, T.ab)
            + ccpy_einsum("mnjf,fani->maji", H.ab.ooov, T.bb)
            - ccpy_einsum("mnfi,fajn->maji", H.ab.oovo, T.ab)
            + ccpy_einsum("maje,ei->maji", Q1, T.b)
            + ccpy_einsum("maei,ej->maji", H0.ab.ovvo, T.a)
            + ccpy_einsum("mafe,feji->maji", H0.ab.ovvv, T.ab)
    )

    Q1 = (
            ccpy_einsum("mnjf,afin->amij", H.bb.ooov, T.bb)
            + ccpy_einsum("nmfj,fani->amij", H.ab.oovo, T.ab)
    )
    Q2 = H0.bb.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.bb.vovv, T.b)
    Q2 = ccpy_einsum("amif,fj->amij", Q2, T.b)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.vooo += Q1 + (
            + ccpy_einsum("me,aeij->amij", H.b.ov, T.bb)
            - ccpy_einsum("nmij,an->amij", H.bb.oooo, T.b)
            + 0.5 * ccpy_einsum("amef,efij->amij", H0.bb.vovv, T.bb)
    )

    Q1 = (
            ccpy_einsum("bnef,afin->abie", H.aa.vovv, T.aa)
            + ccpy_einsum("bnef,afin->abie", H.ab.vovv, T.ab)
    )
    Q2 = H0.aa.ovov - 0.5 * ccpy_einsum("mnie,bn->mbie", H0.aa.ooov, T.a)
    Q2 = -ccpy_einsum("mbie,am->abie", Q2, T.a)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvov += Q1 + (
            - ccpy_einsum("me,abim->abie", H.a.ov, T.aa)
            + ccpy_einsum("abfe,fi->abie", H.aa.vvvv, T.a)
            + 0.5 * ccpy_einsum("mnie,abmn->abie", H0.aa.ooov, T.aa)
    )

    Q1 = H0.ab.ovov - ccpy_einsum("mnie,bn->mbie", H0.ab.ooov, T.b)
    Q1 = -ccpy_einsum("mbie,am->abie", Q1, T.a)
    H.ab.vvov += Q1 + (
            - ccpy_einsum("me,abim->abie", H.b.ov, T.ab)
            + ccpy_einsum("abfe,fi->abie", H.ab.vvvv, T.a)
            + ccpy_einsum("nbfe,afin->abie", H.ab.ovvv, T.aa)
            + ccpy_einsum("bnef,afin->abie", H.bb.vovv, T.ab)
            - ccpy_einsum("amfe,fbim->abie", H.ab.vovv, T.ab)
            - ccpy_einsum("amie,bm->abie", H0.ab.voov, T.b)
            + ccpy_einsum("nmie,abnm->abie", H0.ab.ooov, T.ab)
    )

    Q1 = H0.ab.vovo - ccpy_einsum("nmei,bn->bmei", H0.ab.oovo, T.a)
    Q1 = -ccpy_einsum("bmei,am->baei", Q1, T.b)
    H.ab.vvvo += Q1 + (
            - ccpy_einsum("me,bami->baei", H.a.ov, T.ab)
            + ccpy_einsum("baef,fi->baei", H.ab.vvvv, T.b)
            + ccpy_einsum("bnef,fani->baei", H.aa.vovv, T.ab)
            + ccpy_einsum("bnef,fani->baei", H.ab.vovv, T.bb)
            - ccpy_einsum("maef,bfmi->baei", H.ab.ovvv, T.ab)
            - ccpy_einsum("naei,bn->baei", H0.ab.ovvo, T.a)
            + ccpy_einsum("nmei,banm->baei", H0.ab.oovo, T.ab)
    )

    Q1 = (
            ccpy_einsum("bnef,afin->abie", H.bb.vovv, T.bb)
            + ccpy_einsum("nbfe,fani->abie", H.ab.ovvv, T.ab)
    )
    Q2 = H.bb.ovov - 0.5 * ccpy_einsum("mnie,bn->mbie", H0.bb.ooov, T.b)
    Q2 = -ccpy_einsum("mbie,am->abie", Q2, T.b)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvov += Q1 + (
            - ccpy_einsum("me,abim->abie", H.b.ov, T.bb)
            + ccpy_einsum("abfe,fi->abie", H.bb.vvvv, T.b)
            + 0.5 * ccpy_einsum("mnie,abmn->abie", H0.bb.ooov, T.bb)
    )


    return H
