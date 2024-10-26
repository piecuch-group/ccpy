"""Module with functions that perform the CC with singles, doubles,
triples, and quadruples (CCSDTQ) calculation for a molecular system.
***This is a working version that was restored from a commit on GitHub
 dated 05/05/2022. The former commit ID is c585654.***"""

import numpy as np

from ccpy.hbar.hbar_ccsdt import add_VT3_intermediates
from ccpy.lib.core import cc_loops2
from ccpy.lib.core import cc_loops_t4

def update(T, dT, H, X, shift, flag_RHF, system):

    # update T1
    T, dT = update_t1a(T, dT, H, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, shift)

    # CCS intermediates
    hbar = get_ccs_intermediates_opt(T, H)

    # update T2
    T, dT = update_t2a(T, dT, hbar, H, shift)
    T, dT = update_t2b(T, dT, hbar, H, shift)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, hbar, H, shift)

    # CCSD intermediates
    # [TODO]: Should accept CCS HBar as input and build only terms with T2 in it
    hbar = get_ccsd_intermediates(T, H)

    # update T3
    T, dT = update_t3a(T, dT, hbar, H, shift)
    T, dT = update_t3b(T, dT, hbar, H, shift)
    if flag_RHF:
        T.abb = np.transpose(T.aab, (2, 1, 0, 5, 4, 3))
        dT.abb = np.transpose(dT.aab, (2, 1, 0, 5, 4, 3))
        T.bbb = T.aaa.copy()
        dT.bbb = dT.aaa.copy()
    else:
        T, dT = update_t3c(T, dT, hbar, H, shift)
        T, dT = update_t3d(T, dT, hbar, H, shift)

    # add VT3 intermediates to two-body part of HBar
    hbar = add_VT3_intermediates(T, hbar)

    # update T4
    T, dT = update_t4a(T, dT, hbar, H, shift)
    T, dT = update_t4b(T, dT, hbar, H, shift)
    T, dT = update_t4c(T, dT, hbar, H, shift)
    if flag_RHF:
        T.abbb = np.transpose(T.aaab, (3, 2, 1, 0, 7, 6, 5, 4))
        dT.abbb = np.transpose(dT.aaab, (3, 2, 1, 0, 7, 6, 5, 4))
        T.bbbb = T.aaaa.copy()
        dT.bbbb = dT.aaaa.copy()
    else:
        T, dT = update_t4d(T, dT, hbar, H, shift)
        T, dT = update_t4e(T, dT, hbar, H, shift)

    return T, dT


def update_t1a(T, dT, H, shift):
    """
    Update t1a amplitudes by calculating the projection <ia|(H_N e^(T1+T2+T3))_C|0>.
    """
    chi1A_vv = H.a.vv.copy()
    chi1A_vv += np.einsum("anef,fn->ae", H.aa.vovv, T.a, optimize=True)
    chi1A_vv += np.einsum("anef,fn->ae", H.ab.vovv, T.b, optimize=True)

    chi1A_oo = H.a.oo.copy()
    chi1A_oo += np.einsum("mnif,fn->mi", H.aa.ooov, T.a, optimize=True)
    chi1A_oo += np.einsum("mnif,fn->mi", H.ab.ooov, T.b, optimize=True)

    h1A_ov = H.a.ov.copy()
    h1A_ov += np.einsum("mnef,fn->me", H.aa.oovv, T.a, optimize=True)
    h1A_ov += np.einsum("mnef,fn->me", H.ab.oovv, T.b, optimize=True)

    h1B_ov = H.b.ov.copy()
    h1B_ov += np.einsum("nmfe,fn->me", H.ab.oovv, T.a, optimize=True)
    h1B_ov += np.einsum("mnef,fn->me", H.bb.oovv, T.b, optimize=True)

    h1A_oo = chi1A_oo.copy()
    h1A_oo += np.einsum("me,ei->mi", h1A_ov, T.a, optimize=True)

    h2A_ooov = H.aa.ooov + np.einsum("mnfe,fi->mnie", H.aa.oovv, T.a, optimize=True)
    h2B_ooov = H.ab.ooov + np.einsum("mnfe,fi->mnie", H.ab.oovv, T.a, optimize=True)
    h2A_vovv = H.aa.vovv - np.einsum("mnfe,an->amef", H.aa.oovv, T.a, optimize=True)
    h2B_vovv = H.ab.vovv - np.einsum("nmef,an->amef", H.ab.oovv, T.a, optimize=True)

    dT.a = -np.einsum("mi,am->ai", h1A_oo, T.a, optimize=True)
    dT.a += np.einsum("ae,ei->ai", chi1A_vv, T.a, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.aa.voov, T.a, optimize=True)
    dT.a += np.einsum("anif,fn->ai", H.ab.voov, T.b, optimize=True)
    dT.a += np.einsum("me,aeim->ai", h1A_ov, T.aa, optimize=True)
    dT.a += np.einsum("me,aeim->ai", h1B_ov, T.ab, optimize=True)
    dT.a -= 0.5 * np.einsum("mnif,afmn->ai", h2A_ooov, T.aa, optimize=True)
    dT.a -= np.einsum("mnif,afmn->ai", h2B_ooov, T.ab, optimize=True)
    dT.a += 0.5 * np.einsum("anef,efin->ai", h2A_vovv, T.aa, optimize=True)
    dT.a += np.einsum("anef,efin->ai", h2B_vovv, T.ab, optimize=True)
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


def update_t1b(T, dT, H, shift):
    """
    Update t1b amplitudes by calculating the projection <i~a~|(H_N e^(T1+T2+T3))_C|0>.
    """
    # Intermediates
    chi1B_vv = H.b.vv.copy()
    chi1B_vv += np.einsum("anef,fn->ae", H.bb.vovv, T.b, optimize=True)
    chi1B_vv += np.einsum("nafe,fn->ae", H.ab.ovvv, T.a, optimize=True)

    chi1B_oo = H.b.oo.copy()
    chi1B_oo += np.einsum("mnif,fn->mi", H.bb.ooov, T.b, optimize=True)
    chi1B_oo += np.einsum("nmfi,fn->mi", H.ab.oovo, T.a, optimize=True)

    h1A_ov = H.a.ov.copy()
    h1A_ov += np.einsum("mnef,fn->me", H.aa.oovv, T.a, optimize=True)
    h1A_ov += np.einsum("mnef,fn->me", H.ab.oovv, T.b, optimize=True)

    h1B_ov = H.b.ov.copy()
    h1B_ov += np.einsum("nmfe,fn->me", H.ab.oovv, T.a, optimize=True)
    h1B_ov += np.einsum("mnef,fn->me", H.bb.oovv, T.b, optimize=True)

    h1B_oo = chi1B_oo + np.einsum("me,ei->mi", h1B_ov, T.b, optimize=True)

    h2C_ooov = H.bb.ooov + np.einsum("mnfe,fi->mnie", H.bb.oovv, T.b, optimize=True)
    h2B_oovo = H.ab.oovo + np.einsum("nmef,fi->nmei", H.ab.oovv, T.b, optimize=True)
    h2C_vovv = H.bb.vovv - np.einsum("mnfe,an->amef", H.bb.oovv, T.b, optimize=True)
    h2B_ovvv = H.ab.ovvv - np.einsum("mnfe,an->mafe", H.ab.oovv, T.b, optimize=True)

    dT.b = -np.einsum("mi,am->ai", h1B_oo, T.b, optimize=True)
    dT.b += np.einsum("ae,ei->ai", chi1B_vv, T.b, optimize=True)
    dT.b += np.einsum("anif,fn->ai", H.bb.voov, T.b, optimize=True)
    dT.b += np.einsum("nafi,fn->ai", H.ab.ovvo, T.a, optimize=True)
    dT.b += np.einsum("me,eami->ai", h1A_ov, T.ab, optimize=True)
    dT.b += np.einsum("me,aeim->ai", h1B_ov, T.bb, optimize=True)
    dT.b -= 0.5 * np.einsum("mnif,afmn->ai", h2C_ooov, T.bb, optimize=True)
    dT.b -= np.einsum("nmfi,fanm->ai", h2B_oovo, T.ab, optimize=True)
    dT.b += 0.5 * np.einsum("anef,efin->ai", h2C_vovv, T.bb, optimize=True)
    dT.b += np.einsum("nafe,feni->ai", h2B_ovvv, T.ab, optimize=True)
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


# @profile
def update_t2a(T, dT, H, H0, shift):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2))_C|0>.
    """
    # intermediates
    I1A_oo = (
            H.a.oo
            + 0.5 * np.einsum("mnef,efin->mi", H.aa.oovv, T.aa, optimize=True)
            + np.einsum("mnef,efin->mi", H.ab.oovv, T.ab, optimize=True)
    )

    I1A_vv = (
            H.a.vv
            - 0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, T.aa, optimize=True)
            - np.einsum("mnef,afmn->ae", H.ab.oovv, T.ab, optimize=True)
    )

    I2A_voov = (
            H.aa.voov
            + 0.5 * np.einsum("mnef,afin->amie", H.aa.oovv, T.aa, optimize=True)
            + np.einsum("mnef,afin->amie", H.ab.oovv, T.ab, optimize=True)
    )

    I2A_oooo = H.aa.oooo + 0.5 * np.einsum(
        "mnef,efij->mnij", H.aa.oovv, T.aa, optimize=True
    )

    I2B_voov = H.ab.voov + 0.5 * np.einsum(
        "mnef,afin->amie", H.bb.oovv, T.ab, optimize=True
    )

    I2A_vooo = H.aa.vooo + 0.5 * np.einsum('anef,efij->anij', H0.aa.vovv + 0.5 * H.aa.vovv, T.aa, optimize=True)

    tau = 0.5 * T.aa + np.einsum('ai,bj->abij', T.a, T.a, optimize=True)

    dT.aa = -0.5 * np.einsum("amij,bm->abij", I2A_vooo, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("abie,ej->abij", H.aa.vvov, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", I1A_vv, T.aa, optimize=True)
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", I1A_oo, T.aa, optimize=True)
    dT.aa += np.einsum("amie,ebmj->abij", I2A_voov, T.aa, optimize=True)
    dT.aa += np.einsum("amie,bejm->abij", I2B_voov, T.ab, optimize=True)
    dT.aa += 0.25 * np.einsum("abef,efij->abij", H.aa.vvvv, tau, optimize=True)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", I2A_oooo, T.aa, optimize=True)
    # T3 parts
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.a.ov, T.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T.aab, optimize=True)
    dT.aa -= 0.5 * np.einsum("mnif,abfmjn->abij", H0.ab.ooov + H.ab.ooov, T.aab, optimize=True)
    dT.aa -= 0.25 * np.einsum("mnif,abfmjn->abij", H0.aa.ooov + H.aa.ooov, T.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("anef,ebfijn->abij", H0.aa.vovv + H.aa.vovv, T.aaa, optimize=True)
    dT.aa += 0.5 * np.einsum("anef,ebfijn->abij", H0.ab.vovv + H.ab.vovv, T.aab, optimize=True)
    # T4 parts
    dT.aa += (1.0 / 4.0) * 0.25 * np.einsum("mnef,abefijmn->abij", H0.aa.oovv, T.aaaa, optimize=True)
    dT.aa += (1.0 / 4.0) * np.einsum("mnef,abefijmn->abij", H0.ab.oovv, T.aaab, optimize=True)
    dT.aa += (1.0 / 4.0) * 0.25 * np.einsum("mnef,abefijmn->abij", H0.bb.oovv, T.aabb, optimize=True)

    T.aa, dT.aa = cc_loops2.update_t2a(
        T.aa,
        dT.aa + 0.25 * H0.aa.vvoo,
        H0.a.oo,
        H0.a.vv,
        shift
    )
    return T, dT


# @profile
def update_t2b(T, dT, H, H0, shift):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2))_C|0>.
    """
    # intermediates
    I1A_vv = (
            H.a.vv
            - 0.5 * np.einsum("mnef,afmn->ae", H.aa.oovv, T.aa, optimize=True)
            - np.einsum("mnef,afmn->ae", H.ab.oovv, T.ab, optimize=True)
    )

    I1B_vv = (
            H.b.vv
            - np.einsum("nmfe,fbnm->be", H.ab.oovv, T.ab, optimize=True)
            - 0.5 * np.einsum("mnef,fbnm->be", H.bb.oovv, T.bb, optimize=True)
    )

    I1A_oo = (
            H.a.oo
            + 0.5 * np.einsum("mnef,efin->mi", H.aa.oovv, T.aa, optimize=True)
            + np.einsum("mnef,efin->mi", H.ab.oovv, T.ab, optimize=True)
    )

    I1B_oo = (
            H.b.oo
            + np.einsum("nmfe,fenj->mj", H.ab.oovv, T.ab, optimize=True)
            + 0.5 * np.einsum("mnef,efjn->mj", H.bb.oovv, T.bb, optimize=True)
    )

    I2A_voov = (
            H.aa.voov
            + np.einsum("mnef,aeim->anif", H.aa.oovv, T.aa, optimize=True)
            + np.einsum("nmfe,aeim->anif", H.ab.oovv, T.ab, optimize=True)
    )

    I2B_voov = (
            H.ab.voov
            + np.einsum("mnef,aeim->anif", H.ab.oovv, T.aa, optimize=True)
            + np.einsum("mnef,aeim->anif", H.bb.oovv, T.ab, optimize=True)
    )

    I2B_oooo = H.ab.oooo + np.einsum("mnef,efij->mnij", H.ab.oovv, T.ab, optimize=True)

    I2B_vovo = H.ab.vovo - np.einsum("mnef,afmj->anej", H.ab.oovv, T.ab, optimize=True)

    I2B_ovoo = H.ab.ovoo + np.einsum("maef,efij->maij", H0.ab.ovvv + 0.5 * H.ab.ovvv, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo + np.einsum("amef,efij->amij", H0.ab.vovv + 0.5 * H.ab.vovv, T.ab, optimize=True)

    tau = T.ab + np.einsum('ai,bj->abij', T.a, T.b, optimize=True)

    dT.ab = -np.einsum("mbij,am->abij", I2B_ovoo, T.a, optimize=True)
    dT.ab -= np.einsum("amij,bm->abij", I2B_vooo, T.b, optimize=True)
    dT.ab += np.einsum("abej,ei->abij", H.ab.vvvo, T.a, optimize=True)
    dT.ab += np.einsum("abie,ej->abij", H.ab.vvov, T.b, optimize=True)
    dT.ab += np.einsum("ae,ebij->abij", I1A_vv, T.ab, optimize=True)
    dT.ab += np.einsum("be,aeij->abij", I1B_vv, T.ab, optimize=True)
    dT.ab -= np.einsum("mi,abmj->abij", I1A_oo, T.ab, optimize=True)
    dT.ab -= np.einsum("mj,abim->abij", I1B_oo, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", I2A_voov, T.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", I2B_voov, T.bb, optimize=True)
    dT.ab += np.einsum("mbej,aeim->abij", H.ab.ovvo, T.aa, optimize=True)
    dT.ab += np.einsum("bmje,aeim->abij", H.bb.voov, T.ab, optimize=True)
    dT.ab -= np.einsum("mbie,aemj->abij", H.ab.ovov, T.ab, optimize=True)
    dT.ab -= np.einsum("amej,ebim->abij", I2B_vovo, T.ab, optimize=True)
    dT.ab += np.einsum("mnij,abmn->abij", I2B_oooo, T.ab, optimize=True)
    dT.ab += np.einsum("abef,efij->abij", H.ab.vvvv, tau, optimize=True)
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
    dT.ab += 0.25 * np.einsum("mnef,aefbimnj->abij", H0.aa.oovv, T.aaab, optimize=True)
    dT.ab += np.einsum("mnef,aefbimnj->abij", H0.ab.oovv, T.aabb, optimize=True)
    dT.ab += 0.25 * np.einsum("mnef,abefijmn->abij", H0.bb.oovv, T.abbb, optimize=True)

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


# @profile
def update_t2c(T, dT, H, H0, shift):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2))_C|0>.
    """
    # intermediates
    I1B_oo = (
            H.b.oo
            + 0.5 * np.einsum("mnef,efin->mi", H.bb.oovv, T.bb, optimize=True)
            + np.einsum("nmfe,feni->mi", H.ab.oovv, T.ab, optimize=True)
    )

    I1B_vv = (
            H.b.vv
            - 0.5 * np.einsum("mnef,afmn->ae", H.bb.oovv, T.bb, optimize=True)
            - np.einsum("nmfe,fanm->ae", H.ab.oovv, T.ab, optimize=True)
    )

    I2C_oooo = H.bb.oooo + 0.5 * np.einsum(
        "mnef,efij->mnij", H.bb.oovv, T.bb, optimize=True
    )

    I2B_ovvo = (
            H.ab.ovvo
            + np.einsum("mnef,afin->maei", H.ab.oovv, T.bb, optimize=True)
            + 0.5 * np.einsum("mnef,fani->maei", H.aa.oovv, T.ab, optimize=True)
    )

    I2C_voov = H.bb.voov + 0.5 * np.einsum(
        "mnef,afin->amie", H.bb.oovv, T.bb, optimize=True
    )

    I2C_vooo = H.bb.vooo + 0.5 * np.einsum('anef,efij->anij', H0.bb.vovv + 0.5 * H.bb.vovv, T.bb, optimize=True)

    tau = 0.5 * T.bb + np.einsum('ai,bj->abij', T.b, T.b, optimize=True)

    dT.bb = -0.5 * np.einsum("amij,bm->abij", I2C_vooo, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("abie,ej->abij", H.bb.vvov, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("ae,ebij->abij", I1B_vv, T.bb, optimize=True)
    dT.bb -= 0.5 * np.einsum("mi,abmj->abij", I1B_oo, T.bb, optimize=True)
    dT.bb += np.einsum("amie,ebmj->abij", I2C_voov, T.bb, optimize=True)
    dT.bb += np.einsum("maei,ebmj->abij", I2B_ovvo, T.ab, optimize=True)
    dT.bb += 0.25 * np.einsum("abef,efij->abij", H.bb.vvvv, tau, optimize=True)
    dT.bb += 0.125 * np.einsum("mnij,abmn->abij", I2C_oooo, T.bb, optimize=True)
    # T3 parts
    dT.bb += 0.25 * np.einsum("me,eabmij->abij", H.a.ov, T.abb, optimize=True)
    dT.bb += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T.bbb, optimize=True)
    dT.bb += 0.25 * np.einsum("anef,ebfijn->abij", H0.bb.vovv + H.bb.vovv, T.bbb, optimize=True)
    dT.bb += 0.5 * np.einsum("nafe,febnij->abij", H0.ab.ovvv + H.ab.ovvv, T.abb, optimize=True)
    dT.bb -= 0.25 * np.einsum("mnif,abfmjn->abij", H0.bb.ooov + H.bb.ooov, T.bbb, optimize=True)
    dT.bb -= 0.5 * np.einsum("nmfi,fabnmj->abij", H0.ab.oovo + H.ab.oovo, T.abb, optimize=True)
    # T4 parts
    dT.bb += 0.0625 * np.einsum("mnef,abefijmn->abij", H0.bb.oovv, T.bbbb, optimize=True)
    dT.bb += 0.25 * np.einsum("nmfe,febanmji->abij", H0.ab.oovv, T.abbb, optimize=True)
    dT.bb += 0.0625 * np.einsum("mnef,febanmji->abij", H0.aa.oovv, T.aabb, optimize=True)

    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb,
        dT.bb + 0.25 * H0.bb.vvoo,
        H0.b.oo,
        H0.b.vv,
        shift
    )
    return T, dT


# @profile
def update_t3a(T, dT, H, H0, shift):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3))_C|0>.
    """
    # <ijkabc | H(2) | 0 > + (VT3)_C intermediates
    I2A_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vvov -= np.einsum("mnef,abfimn->abie", H0.ab.oovv, T.aab, optimize=True)
    I2A_vvov += H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)

    I2A_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.aa.oovv, T.aaa, optimize=True)
    I2A_vooo += H.aa.vooo + np.einsum("mnef,aefijn->amij", H0.ab.oovv, T.aab, optimize=True)

    # MM(2,3)A
    dT.aaa = -0.25 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.aa, optimize=True) # (a/bc)(k/ij) = 9
    dT.aaa += 0.25 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.aa, optimize=True) # (c/ab)(i/jk) = 9
    # (HBar*T3)_C
    dT.aaa -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.a.oo, T.aaa, optimize=True) # (k/ij) = 3
    dT.aaa += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.a.vv, T.aaa, optimize=True) # (c/ab) = 3
    dT.aaa += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa, optimize=True) # (k/ij) = 3
    dT.aaa += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa, optimize=True) # (c/ab) = 3
    dT.aaa += 0.25 * np.einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa, optimize=True) # (c/ij)(k/ij) = 9
    dT.aaa += 0.25 * np.einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab, optimize=True) # (c/ij)(k/ij) = 9
    # (HBar*T4)_C
    dT.aaa += (1.0 / 36.0) * np.einsum("me,abceijkm->abcijk", H.a.ov, T.aaaa, optimize=True) # (1) = 1
    dT.aaa += (1.0 / 36.0) * np.einsum("me,abceijkm->abcijk", H.b.ov, T.aaab, optimize=True) # (1) = 1
    dT.aaa += (1.0 / 24.0) * np.einsum("cnef,abefijkn->abcijk", H.aa.vovv, T.aaaa, optimize=True) # (c/ab) = 3
    dT.aaa += (1.0 / 12.0) * np.einsum("cnef,abefijkn->abcijk", H.ab.vovv, T.aaab, optimize=True) # (c/ab) = 3
    dT.aaa -= (1.0 / 24.0) * np.einsum("mnkf,abcfijmn->abcijk", H.aa.ooov, T.aaaa, optimize=True) # (k/ij) = 3
    dT.aaa -= (1.0 / 12.0) * np.einsum("mnkf,abcfijmn->abcijk", H.ab.ooov, T.aaab, optimize=True) # (k/ij) = 3

    T.aaa, dT.aaa = cc_loops2.update_t3a_v2(
        T.aaa,
        dT.aaa,
        H0.a.oo,
        H0.a.vv,
        shift,
    )
    return T, dT


# @profile
def update_t3b(T, dT, H, H0, shift):
    """
    Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N e^(T1+T2+T3))_C|0>.
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
    # (HBar*T4)_C
    dT.aab += 0.25 * np.einsum("me,abecijmk->abcijk", H.a.ov, T.aaab, optimize=True) # (1) = 1
    dT.aab += 0.25 * np.einsum("me,abecijmk->abcijk", H.b.ov, T.aabb, optimize=True) # (1) = 1
    dT.aab -= 0.25 * np.einsum("mnjf,abfcimnk->abcijk", H.aa.ooov, T.aaab, optimize=True) # (ij) = 2
    dT.aab -= 0.5 * np.einsum("mnjf,abfcimnk->abcijk", H.ab.ooov, T.aabb, optimize=True) # (ij) = 2
    dT.aab -= 0.25 * np.einsum("nmfk,abfcijnm->abcijk", H.ab.oovo, T.aaab, optimize=True) # (1) = 1
    dT.aab -= 0.125 * np.einsum("mnkf,abfcijnm->abcijk", H.bb.ooov, T.aabb, optimize=True) # (1) = 1
    dT.aab += 0.25 * np.einsum("bnef,aefcijnk->abcijk", H.aa.vovv, T.aaab, optimize=True) # (ab) = 2
    dT.aab += 0.5 * np.einsum("bnef,aefcijnk->abcijk", H.ab.vovv, T.aabb, optimize=True) # (ab) = 2
    dT.aab += 0.25 * np.einsum("ncfe,abfeijnk->abcijk", H.ab.ovvv, T.aaab, optimize=True) # (1) = 1
    dT.aab += 0.125 * np.einsum("cnef,abfeijnk->abcijk", H.bb.vovv, T.aabb, optimize=True) # (1) = 1

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


# @profile
def update_t3c(T, dT, H, H0, shift):
    """
    Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N e^(T1+T2+T3))_C|0>.
    """
    # <ij~k~ab~c~ | H(2) | 0 > + (VT3)_C intermediates
    I2B_vvvo = -0.5 * np.einsum("mnef,afbmnj->abej", H0.aa.oovv, T.aab, optimize=True)
    I2B_vvvo += -np.einsum("mnef,afbmnj->abej", H0.ab.oovv, T.abb, optimize=True)
    I2B_vvvo += H.ab.vvvo

    I2B_ovoo = 0.5 * np.einsum("mnef,efbinj->mbij", H0.aa.oovv, T.aab, optimize=True)
    I2B_ovoo += np.einsum("mnef,efbinj->mbij", H0.ab.oovv, T.abb, optimize=True)
    I2B_ovoo += H.ab.ovoo

    I2B_ovoo -= np.einsum("me,ebij->mbij", H.a.ov, T.ab, optimize=True)
    I2B_vvov = -np.einsum("nmfe,afbinm->abie", H0.ab.oovv, T.aab, optimize=True)
    I2B_vvov += -0.5 * np.einsum("nmfe,afbinm->abie", H0.bb.oovv, T.abb, optimize=True)
    I2B_vvov += H.ab.vvov

    I2B_vooo = np.einsum("nmfe,afeinj->amij", H0.ab.oovv, T.aab, optimize=True)
    I2B_vooo += 0.5 * np.einsum("nmfe,afeinj->amij", H0.bb.oovv, T.abb, optimize=True)
    I2B_vooo += H.ab.vooo

    I2B_vooo -= np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
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
    # (HBar*T4)_C
    dT.abb += 0.25 * np.einsum("me,cebakmji->cbakji", H.b.ov, T.abbb, optimize=True)
    dT.abb += 0.25 * np.einsum("me,cebakmji->cbakji", H.a.ov, T.aabb, optimize=True)
    dT.abb -= 0.25 * np.einsum("mnjf,cfbaknmi->cbakji", H.bb.ooov, T.abbb, optimize=True)
    dT.abb -= 0.5 * np.einsum("nmfj,cfbaknmi->cbakji", H.ab.oovo, T.aabb, optimize=True)
    dT.abb -= 0.25 * np.einsum("mnkf,cfbamnji->cbakji", H.ab.ooov, T.abbb, optimize=True)
    dT.abb -= 0.125 * np.einsum("mnkf,cfbamnji->cbakji", H.aa.ooov, T.aabb, optimize=True)
    dT.abb += 0.25 * np.einsum("bnef,cfeaknji->cbakji", H.bb.vovv, T.abbb, optimize=True)
    dT.abb += 0.5 * np.einsum("nbfe,cfeaknji->cbakji", H.ab.ovvv, T.aabb, optimize=True)
    dT.abb += 0.25 * np.einsum("cnef,efbaknji->cbakji", H.ab.vovv, T.abbb, optimize=True)
    dT.abb += 0.125 * np.einsum("cnef,efbaknji->cbakji", H.aa.vovv, T.aabb, optimize=True)

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


# @profile
def update_t3d(T, dT, H, H0, shift):
    """
    Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N e^(T1+T2+T3))_C|0>.
    """
    #  <ijkabc | H(2) | 0 > + (VT3)_C intermediates
    I2C_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vvov -= np.einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb, optimize=True)
    I2C_vvov += np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
    I2C_vvov += H.bb.vvov

    I2C_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vooo += np.einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb, optimize=True)
    I2C_vooo += H.bb.vooo
    # <ijkabc | H(2) | 0 >
    dT.bbb = -0.25 * np.einsum("amij,bcmk->abcijk", I2C_vooo, T.bb, optimize=True)
    dT.bbb += 0.25 * np.einsum("abie,ecjk->abcijk", I2C_vvov, T.bb, optimize=True)
    # <ijkabc | (H(2) * T3)_C | 0 >
    dT.bbb -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.b.oo, T.bbb, optimize=True)
    dT.bbb += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.b.vv, T.bbb, optimize=True)
    dT.bbb += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.bb.oooo, T.bbb, optimize=True)
    dT.bbb += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.bb.vvvv, T.bbb, optimize=True)
    dT.bbb += 0.25 * np.einsum("maei,ebcmjk->abcijk", H.ab.ovvo, T.abb, optimize=True)
    dT.bbb += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.bb.voov, T.bbb, optimize=True)
    # <ijkabc | (H(2) * T4)_C | 0 >
    dT.bbb += (1.0 / 36.0) * np.einsum("me,abceijkm->abcijk", H.b.ov, T.bbbb, optimize=True)
    dT.bbb += (1.0 / 36.0) * np.einsum("me,ecbamkji->abcijk", H.a.ov, T.abbb, optimize=True)
    dT.bbb += (1.0 / 24.0) * np.einsum("cnef,abefijkn->abcijk", H.bb.vovv, T.bbbb, optimize=True) # (c/ab) = 3
    dT.bbb += (1.0 / 12.0) * np.einsum("ncfe,febankji->abcijk", H.ab.ovvv, T.abbb, optimize=True) # (c/ab) = 3
    dT.bbb -= (1.0 / 24.0) * np.einsum("mnkf,abcfijmn->abcijk", H.bb.ooov, T.bbbb, optimize=True) # (k/ij) = 3
    dT.bbb -= (1.0 / 12.0) * np.einsum("nmfk,fcbanmji->abcijk", H.ab.oovo, T.abbb, optimize=True) # (k/ij) = 3

    T.bbb, dT.bbb = cc_loops2.update_t3d_v2(
        T.bbb,
        dT.bbb,
        H0.b.oo,
        H0.b.vv,
        shift,
    )
    return T, dT

def update_t4a(T, dT, H, H0, shift):

    # <ijklabcd | H(2) | 0 >
    dT.aaaa = -(144.0 / 576.0) * np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.aa, T.aa, optimize=True)  # (jl/i/k)(bc/a/d) = 12 * 12 = 144
    dT.aaaa += (36.0 / 576.0) * np.einsum("mnij,adml,bcnk->abcdijkl", H.aa.oooo, T.aa, T.aa, optimize=True)   # (ij/kl)(bc/ad) = 6 * 6 = 36
    dT.aaaa += (36.0 / 576.0) * np.einsum("abef,fcjk,edil->abcdijkl", H.aa.vvvv, T.aa, T.aa, optimize=True)   # (jk/il)(ab/cd) = 6 * 6 = 36

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    dT.aaaa += (24.0 / 576.0) * np.einsum("cdke,abeijl->abcdijkl", H.aa.vvov, T.aaa, optimize=True) # (cd/ab)(k/ijl) = 6 * 4 = 24
    dT.aaaa -= (24.0 / 576.0) * np.einsum("cmkl,abdijm->abcdijkl", H.aa.vooo, T.aaa, optimize=True) # (c/abd)(kl/ij) = 6 * 4 = 24

    I3A_vooooo = np.einsum("nmle,bejk->bmnjkl", H.aa.ooov, T.aa, optimize=True)
    I3A_vooooo -= np.transpose(I3A_vooooo, (0, 1, 2, 5, 4, 3)) + np.transpose(I3A_vooooo, (0, 1, 2, 3, 5, 4))
    I3A_vooooo += 0.5 * np.einsum("mnef,befjkl->bmnjkl", H.aa.oovv, T.aaa, optimize=True)
    dT.aaaa += 0.5 * (16.0 / 576.0) * np.einsum("bmnjkl,acdimn->abcdijkl", I3A_vooooo, T.aaa, optimize=True) # (b/acd)(i/jkl) = 4 * 4 = 16

    I3A_vvvovv = -np.einsum("dmfe,bcjm->bcdjef", H.aa.vovv, T.aa, optimize=True)
    I3A_vvvovv -= np.transpose(I3A_vvvovv, (2, 1, 0, 3, 4, 5)) + np.transpose(I3A_vvvovv, (0, 2, 1, 3, 4, 5))
    dT.aaaa += 0.5 * (16.0 / 576.0) * np.einsum("bcdjef,aefikl->abcdijkl", I3A_vvvovv, T.aaa, optimize=True) # (a/bcd)(j/ikl) = 4 * 4 = 16

    I3A_vvooov = (
                    -0.5 * np.einsum("nmke,cdnl->cdmkle", H.aa.ooov, T.aa, optimize=True)
                    +0.5 * np.einsum("cmfe,fdkl->cdmkle", H.aa.vovv, T.aa, optimize=True)
                    +0.125 * np.einsum("mnef,cdfkln->cdmkle", H0.aa.oovv, T.aaa, optimize=True) # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
                    +0.25 * np.einsum("mnef,cdfkln->cdmkle", H0.ab.oovv, T.aab, optimize=True)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    dT.aaaa += (36.0 / 576.0) * np.einsum("cdmkle,abeijm->abcdijkl", I3A_vvooov, T.aaa, optimize=True) # (cd/ab)(kl/ij) = 6 * 6 = 36

    I3B_vvooov = (
                    -0.5 * np.einsum("nmke,cdnl->cdmkle", H.ab.ooov, T.aa, optimize=True)
                    +0.5 * np.einsum("cmfe,fdkl->cdmkle", H.ab.vovv, T.aa, optimize=True)
                    +0.125 * np.einsum("mnef,cdfkln->cdmkle", H0.bb.oovv, T.aab, optimize=True) # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aaaa += (36.0 / 576.0) * np.einsum("cdmkle,abeijm->abcdijkl", I3B_vvooov, T.aab, optimize=True) # (cd/ab)(kl/ij) = 6 * 6 = 36

    # <ijklabcd | (H(2)*T4)_C | 0 >
    dT.aaaa -= (4.0 / 576.0) * np.einsum("mi,abcdmjkl->abcdijkl", H.a.oo, T.aaaa, optimize=True) # (l/ijk) = 4
    dT.aaaa += (4.0 / 576.0) * np.einsum("ae,ebcdijkl->abcdijkl", H.a.vv, T.aaaa, optimize=True) # (d/abc) = 4
    dT.aaaa += (6.0 / 576.0) * 0.5 * np.einsum("mnij,abcdmnkl->abcdijkl", H.aa.oooo, T.aaaa, optimize=True) # (kl/ij) = 6
    dT.aaaa += (6.0 / 576.0) * 0.5 * np.einsum("abef,efcdijkl->abcdijkl", H.aa.vvvv, T.aaaa, optimize=True) # (cd/ab) = 6
    dT.aaaa += (16.0 / 576.0) * np.einsum("amie,ebcdmjkl->abcdijkl", H.aa.voov, T.aaaa, optimize=True) # (d/abc)(l/ijk) = 16
    dT.aaaa += (16.0 / 576.0) * np.einsum("amie,bcdejklm->abcdijkl", H.ab.voov, T.aaab, optimize=True) # (d/abc)(l/ijk) = 16

    I3A_vvvoov = (
                    -0.5 * np.einsum("mnef,bcdfjkmn->bcdjke", H0.aa.oovv, T.aaaa, optimize=True)
                    -np.einsum("mnef,bcdfjkmn->bcdjke", H0.ab.oovv, T.aaab, optimize=True)
    )
    dT.aaaa += (24.0 / 576.0) * np.einsum("bcdjke,aeil->abcdijkl", I3A_vvvoov, T.aa, optimize=True) # (a/bcd)(jk/il) = 4 * 6 = 24

    I3A_vvoooo = (
                    0.5 * np.einsum("mnef,bcefjkln->bcmjkl", H0.aa.oovv, T.aaaa, optimize=True)
                    +np.einsum("mnef,bcefjkln->bcmjkl", H.ab.oovv, T.aaab, optimize=True)
    )
    dT.aaaa -= (24.0 / 576.0) * np.einsum("bcmjkl,adim->abcdijkl", I3A_vvoooo, T.aa, optimize=True) # (bc/ad)(i/jkl) = 6 * 4 = 24


    T.aaaa, dT.aaaa = cc_loops_t4.update_t4a(
        T.aaaa,
        dT.aaaa,
        H0.a.oo,
        H0.a.vv,
        shift,
    )
    return T, dT


def update_t4b(T, dT, H, H0, shift):

    # <ijklabcd | H(2) | 0 >
    dT.aaab = -(9.0 / 36.0) * np.einsum("mdel,abim,ecjk->abcdijkl", H.ab.ovvo, T.aa, T.aa, optimize=True)    # (i/jk)(c/ab) = 9
    dT.aaab += (9.0 / 36.0) * np.einsum("mnij,bcnk,adml->abcdijkl", H.aa.oooo, T.aa, T.ab, optimize=True)    # (k/ij)(a/bc) = 9
    dT.aaab -= (18.0 / 36.0) * np.einsum("mdjf,abim,cfkl->abcdijkl", H.ab.ovov, T.aa, T.ab, optimize=True)   # (ijk)(c/ab) = (i/jk)(c/ab)(jk) = 18
    dT.aaab -= np.einsum("amie,bejl,cdkm->abcdijkl", H.ab.voov, T.ab, T.ab, optimize=True)                   # (ijk)(abc) = (i/jk)(a/bc)(jk)(bc) = 36
    dT.aaab += (18.0 / 36.0) * np.einsum("mnjl,bcmk,adin->abcdijkl", H.ab.oooo, T.aa, T.ab, optimize=True)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    dT.aaab -= (18.0 / 36.0) * np.einsum("bmel,ecjk,adim->abcdijkl", H.ab.vovo, T.aa, T.ab, optimize=True)   # (i/jk)(abc) = (i/jk)(a/bc)(bc) = 18
    dT.aaab -= (18.0 / 36.0) * np.einsum("amie,ecjk,bdml->abcdijkl", H.aa.voov, T.aa, T.ab, optimize=True)   # (i/kj)(abc) = (i/kj)(a/bc)(bc) = 18
    dT.aaab += (9.0 / 36.0) * np.einsum("abef,fcjk,edil->abcdijkl", H.aa.vvvv, T.aa, T.ab, optimize=True)    # (i/jk)(c/ab) = (i/jk)(c/ab) = 9
    dT.aaab -= (18.0 / 36.0) * np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.aa, T.ab, optimize=True)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    dT.aaab += (18.0 / 36.0) * np.einsum("adef,ebij,cfkl->abcdijkl", H.ab.vvvv, T.aa, T.ab, optimize=True)   # (k/ij)(abc) = (k/ij)(a/bc)(bc) = 18

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    dT.aaab -= (1.0 / 12.0) * np.einsum("mdkl,abcijm->abcdijkl", H.ab.ovoo, T.aaa, optimize=True)  # (k/ij) = 3
    dT.aaab -= (9.0 / 36.0) * np.einsum("amik,bcdjml->abcdijkl", H.aa.vooo, T.aab, optimize=True)  # (j/ik)(a/bc) = 9
    dT.aaab -= (9.0 / 36.0) * np.einsum("amil,bcdjkm->abcdijkl", H.ab.vooo, T.aab, optimize=True)  # (a/bc)(i/jk) = 9

    dT.aaab += (1.0 / 12.0) * np.einsum("cdel,abeijk->abcdijkl", H.ab.vvvo, T.aaa, optimize=True)  # (c/ab) = 3
    dT.aaab += (9.0 / 36.0) * np.einsum("acie,bedjkl->abcdijkl", H.aa.vvov, T.aab, optimize=True)  # (b/ac)(i/jk) = 9
    dT.aaab += (9.0 / 36.0) * np.einsum("adie,bcejkl->abcdijkl", H.ab.vvov, T.aab, optimize=True)  # (a/bc)(i/jk) = 9

    I3B_oovooo = (
                    np.einsum("mnie,edjl->mndijl", H.aa.ooov, T.ab, optimize=True)
                   +0.25 * np.einsum("mnef,efdijl->mndijl", H.aa.oovv, T.aab, optimize=True)
    )
    I3B_oovooo -= np.transpose(I3B_oovooo, (0, 1, 2, 4, 3, 5))
    dT.aaab += (1.0 / 12.0) * 0.5 * np.einsum("mndijl,abcmnk->abcdijkl", I3B_oovooo, T.aaa, optimize=True)  # (k/ij) = 3

    I3A_vooooo = np.einsum("mnie,delj->dmnlij", H.aa.ooov, T.aa, optimize=True)
    I3A_vooooo -= np.transpose(I3A_vooooo, (0, 1, 2, 4, 3, 5)) + np.transpose(I3A_vooooo, (0, 1, 2, 5, 4, 3))
    I3A_vooooo += 0.5 * np.einsum("mnef,efdijl->dmnlij", H.aa.oovv, T.aaa, optimize=True)
    dT.aaab += (1.0 / 12.0) * 0.5 * np.einsum("cmnkij,abdmnl->abcdijkl", I3A_vooooo, T.aab, optimize=True)  # (c/ab) = 3

    I3B_vooooo = (
                    0.5 * np.einsum("mnel,aeik->amnikl", H.ab.oovo, T.aa, optimize=True)
                  + np.einsum("mnke,aeil->amnikl", H.ab.ooov, T.ab, optimize=True)
                  + 0.5 * np.einsum("mnef,aefikl->amnikl", H.ab.oovv, T.aab, optimize=True)
    )
    I3B_vooooo -= np.transpose(I3B_vooooo, (0, 1, 2, 4, 3, 5))
    dT.aaab += (9.0 / 36.0) * np.einsum("amnikl,bcdjmn->abcdijkl", I3B_vooooo, T.aab, optimize=True)  # (a/bc)(j/ik) = 9

    I3B_vvvvvo = -np.einsum("amef,bdml->abdefl", H.aa.vovv, T.ab, optimize=True)
    I3B_vvvvvo -= np.transpose(I3B_vvvvvo, (1, 0, 2, 3, 4, 5))
    dT.aaab += (1.0 / 12.0) * 0.5 * np.einsum("abdefl,efcijk->abcdijkl", I3B_vvvvvo, T.aaa, optimize=True)  # (c/ab) = 3

    I3A_vvvvvo = -np.einsum("amef,bcmk->abcefk", H.aa.vovv, T.aa, optimize=True)
    I3A_vvvvvo -= np.transpose(I3A_vvvvvo, (1, 0, 2, 3, 4, 5)) + np.transpose(I3A_vvvvvo, (2, 1, 0, 3, 4, 5))
    dT.aaab += (1.0 / 12.0) * 0.5 * np.einsum("abcefk,efdijl->abcdijkl", I3A_vvvvvo, T.aab, optimize=True)  # (k/ij) = 3

    I3B_vvvovv = (
                    -0.5 * np.einsum("mdef,acim->acdief", H.ab.ovvv, T.aa, optimize=True)
                    - np.einsum("cmef,adim->acdief", H.ab.vovv, T.ab, optimize=True)
    )
    I3B_vvvovv -= np.transpose(I3B_vvvovv, (1, 0, 2, 3, 4, 5))
    dT.aaab += (9.0 / 36.0) * np.einsum("acdief,befjkl->abcdijkl", I3B_vvvovv, T.aab, optimize=True)  # (b/ac)(i/jk) = 9

    I3B_vovovo = (
                    -np.einsum("nmie,adnl->amdiel", H.aa.ooov, T.ab, optimize=True)
                    +np.einsum("amfe,fdil->amdiel", H.aa.vovv, T.ab, optimize=True)
                    -np.einsum("mnel,adin->amdiel", H.ab.oovo, T.ab, optimize=True)
                    +np.einsum("mdef,afil->amdiel", H.ab.ovvv, T.ab, optimize=True)
                    +np.einsum("mnef,afdinl->amdiel", H.aa.oovv, T.aab, optimize=True)
                    +np.einsum("mnef,afdinl->amdiel", H.ab.oovv, T.abb, optimize=True)
    )
    dT.aaab += (9.0 / 36.0) * np.einsum("amdiel,bcejkm->abcdijkl", I3B_vovovo, T.aaa, optimize=True)  # (a/bc)(i/jk) = 9

    I3A_vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.aa.ooov, T.aa, optimize=True)
                +0.5 * np.einsum("bmfe,afij->abmije", H.aa.vovv, T.aa, optimize=True)
                +0.25 * np.einsum("mnef,abfijn->abmije", H.ab.oovv, T.aab, optimize=True)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aaab += (9.0 / 36.0) * np.einsum("abmije,cedkml->abcdijkl", I3A_vvooov, T.aab, optimize=True)  # (c/ab)(k/ij) = 9

    I3B_vvoovo = (
                -0.5 * np.einsum("nmel,acin->acmiel", H.ab.oovo, T.aa, optimize=True)
                + np.einsum("cmef,afil->acmiel", H.ab.vovv, T.ab, optimize=True)
                - 0.5 * np.einsum("nmef,acfinl->acmiel", H.ab.oovv, T.aab, optimize=True)
    )
    I3B_vvoovo -= np.transpose(I3B_vvoovo, (1, 0, 2, 3, 4, 5))
    dT.aaab -= (9.0 / 36.0) * np.einsum("acmiel,ebdkjm->abcdijkl", I3B_vvoovo, T.aab, optimize=True)  # (b/ac)(i/jk) = 9

    I3B_vovoov = (
                0.5 * np.einsum("mdfe,afik->amdike", H.ab.ovvv, T.aa, optimize=True)
                -np.einsum("mnke,adin->amdike", H.ab.ooov, T.ab, optimize=True)
    )
    I3B_vovoov -= np.transpose(I3B_vovoov, (0, 1, 2, 4, 3, 5))
    dT.aaab -= (9.0 / 36.0) * np.einsum("amdike,bcejml->abcdijkl", I3B_vovoov, T.aab, optimize=True)  # (a/bc)(j/ik) = 9

    I3C_vvooov = (
                -np.einsum("nmie,adnl->admile", H.ab.ooov, T.ab, optimize=True)
                -np.einsum("nmle,adin->admile", H.bb.ooov, T.ab, optimize=True)
                +np.einsum("amfe,fdil->admile", H.ab.vovv, T.ab, optimize=True)
                +np.einsum("dmfe,afil->admile", H.bb.vovv, T.ab, optimize=True)
                +np.einsum("mnef,afdinl->admile", H.bb.oovv, T.abb, optimize=True)  # added 5/2/22
    )
    dT.aaab += (9.0 / 36.0) * np.einsum("admile,bcejkm->abcdijkl", I3C_vvooov, T.aab, optimize=True)  # (a/bc)(i/jk) = 9

    I3B_vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.ab.ooov, T.aa, optimize=True)
                +0.5 * np.einsum("bmfe,afij->abmije", H.ab.vovv, T.aa, optimize=True)
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aaab += (9.0 / 36.0) * np.einsum("abmije,cdeklm->abcdijkl", I3B_vvooov, T.abb, optimize=True)  # (c/ab)(k/ij) = 9

    # <ijklabcd | (H(2)*T4)_C | 0 >
    dT.aaab -= (1.0 / 12.0) * np.einsum("mi,abcdmjkl->abcdijkl", H.a.oo, T.aaab, optimize=True)  # (i/jk) = 3
    dT.aaab -= (1.0 / 36.0) * np.einsum("ml,abcdijkm->abcdijkl", H.b.oo, T.aaab, optimize=True)  # (1) = 1
    dT.aaab += (1.0 / 12.0) * np.einsum("ae,ebcdijkl->abcdijkl", H.a.vv, T.aaab, optimize=True)  # (a/bc) = 3
    dT.aaab += (1.0 / 36.0) * np.einsum("de,abceijkl->abcdijkl", H.b.vv, T.aaab, optimize=True)  # (1) = 1

    dT.aaab += (1.0 / 12.0) * 0.5 * np.einsum("mnij,abcdmnkl->abcdijkl", H.aa.oooo, T.aaab, optimize=True)  # (k/ij) = 3
    dT.aaab += (1.0 / 12.0) * np.einsum("mnil,abcdmjkn->abcdijkl", H.ab.oooo, T.aaab, optimize=True)  # (i/jk) = 3
    dT.aaab += (1.0 / 12.0) * 0.5 * np.einsum("abef,efcdijkl->abcdijkl", H.aa.vvvv, T.aaab, optimize=True)  # (c/ab) = 3
    dT.aaab += (1.0 / 12.0) * np.einsum("adef,ebcfijkl->abcdijkl", H.ab.vvvv, T.aaab, optimize=True)  # (a/bc) = 3

    dT.aaab += (9.0 / 36.0) * np.einsum("amie,ebcdmjkl->abcdijkl", H.aa.voov, T.aaab, optimize=True)  # (a/bc)(i/jk) = 9
    dT.aaab += (9.0 / 36.0) * np.einsum("amie,bcedjkml->abcdijkl", H.ab.voov, T.aabb, optimize=True)  # (a/bc)(i/jk) = 9
    dT.aaab += (1.0 / 36.0) * np.einsum("mdel,abceijkm->abcdijkl", H.ab.ovvo, T.aaaa, optimize=True)  # (1) = 1
    dT.aaab += (1.0 / 36.0) * np.einsum("dmle,abceijkm->abcdijkl", H.bb.voov, T.aaab, optimize=True)  # (1) = 1
    dT.aaab -= (1.0 / 12.0) * np.einsum("amel,ebcdijkm->abcdijkl", H.ab.vovo, T.aaab, optimize=True)  # (a/bc) = 3
    dT.aaab -= (1.0 / 12.0) * np.einsum("mdie,abcemjkl->abcdijkl", H.ab.ovov, T.aaab, optimize=True)  # (i/jk) = 3

    I3B_vvvvoo = (
        -0.5 * np.einsum("mnef,acfdmknl->acdekl", H.aa.oovv, T.aaab, optimize=True)
        - np.einsum("mnef,acfdmknl->acdekl", H.ab.oovv, T.aabb, optimize=True)
    )
    dT.aaab += (9.0 / 36.0) * np.einsum("acdekl,ebij->abcdijkl", I3B_vvvvoo, T.aa, optimize=True)  # (b/ac)(k/ij) = 9

    I3A_vvvvoo = (
        -0.5 * np.einsum("mnef,abcfmjkn->abcejk", H.aa.oovv, T.aaaa, optimize=True)
        - np.einsum("mnef,abcfmjkn->abcejk", H.ab.oovv, T.aaab, optimize=True)
    )
    dT.aaab += (1.0 / 12.0) * np.einsum("abcejk,edil->abcdijkl", I3A_vvvvoo, T.ab, optimize=True)  # (i/jk) = 3

    I3B_vvvoov = (
        - np.einsum("nmfe,abfdijnm->abdije", H.ab.oovv, T.aaab, optimize=True)
        - 0.5 * np.einsum("nmfe,abfdijnm->abdije", H.bb.oovv, T.aabb, optimize=True)
    )
    dT.aaab += (9.0 / 36.0) * np.einsum("abdije,cekl->abcdijkl", I3B_vvvoov, T.ab, optimize=True)  # (c/ab)(k/ij) = 9

    I3B_vovooo = (
        0.5 * np.einsum("mnef,cefdkinl->cmdkil", H.aa.oovv, T.aaab, optimize=True)
        + np.einsum("mnef,cefdkinl->cmdkil", H.ab.oovv, T.aabb, optimize=True)
    )
    dT.aaab -= (9.0 / 36.0) * np.einsum("cmdkil,abmj->abcdijkl", I3B_vovooo, T.aa, optimize=True)  # (c/ab)(j/ik) = 9

    I3A_vovooo = (
        0.5 * np.einsum("mnef,bcefjkin->bmcjik", H.aa.oovv, T.aaaa, optimize=True)
        + np.einsum("mnef,bcefjkin->bmcjik", H.ab.oovv, T.aaab, optimize=True)
    )
    dT.aaab -= (1.0 / 12.0) * np.einsum("bmcjik,adml->abcdijkl", I3A_vovooo, T.ab, optimize=True)  # (a/bc) = 3

    I3B_vvoooo = (
        np.einsum("nmfe,bcfejknl->bcmjkl", H.ab.oovv, T.aaab, optimize=True)
        + 0.5 * np.einsum("nmfe,bcfejknl->bcmjkl", H.bb.oovv, T.aabb, optimize=True)
    )
    dT.aaab -= (9.0 / 36.0) * np.einsum("bcmjkl,adim->abcdijkl", I3B_vvoooo, T.ab, optimize=True)  # (a/bc)(i/jk) = 9


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

    # <ijklabcd | H(2) | 0 >
    dT.aabb = -np.einsum("cmke,adim,bejl->abcdijkl", H.bb.voov, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= 0.5 * np.einsum("mcek,aeij,bdml->abcdijkl", H.ab.ovvo, T.aa, T.ab, optimize=True)    # (kl)(ab)(cd) = 8
    dT.aabb -= 0.5 * np.einsum("amie,bdjm,cekl->abcdijkl", H.ab.voov, T.ab, T.bb, optimize=True)    # (ij)(ab)(cd) = 8
    dT.aabb -= 0.5 * np.einsum("mcek,abim,edjl->abcdijkl", H.ab.ovvo, T.aa, T.ab, optimize=True)    # (ij)(kl)(cd) = 8
    dT.aabb -= 0.5 * np.einsum("amie,cdkm,bejl->abcdijkl", H.ab.voov, T.bb, T.ab, optimize=True)    # (ij)(kl)(ab) = 8
    dT.aabb -= np.einsum("bmel,adim,ecjk->abcdijkl", H.ab.vovo, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= np.einsum("mdje,bcmk,aeil->abcdijkl", H.ab.ovov, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= 0.25 * np.einsum("mdje,abim,cekl->abcdijkl", H.ab.ovov, T.aa, T.bb, optimize=True)   # (ij)(cd) = 4
    dT.aabb -= 0.25 * np.einsum("bmel,cdkm,aeij->abcdijkl", H.ab.vovo, T.bb, T.aa, optimize=True)   # (kl)(ab) = 4
    dT.aabb += 0.25 * np.einsum("mnij,acmk,bdnl->abcdijkl", H.aa.oooo, T.ab, T.ab, optimize=True)   # (kl)(ab) = 4 !!! (tricky asym)
    dT.aabb += 0.25 * np.einsum("abef,ecik,fdjl->abcdijkl", H.aa.vvvv, T.ab, T.ab, optimize=True)   # (ij)(kl) = 4 !!! (tricky asym)
    dT.aabb += 0.25 * np.einsum("mnik,abmj,cdnl->abcdijkl", H.ab.oooo, T.aa, T.bb, optimize=True)   # (ij)(kl) = 4
    dT.aabb += 0.25 * np.einsum("acef,ebij,fdkl->abcdijkl", H.ab.vvvv, T.aa, T.bb, optimize=True)   # (ab)(cd) = 4
    dT.aabb += np.einsum("mnik,adml,bcjn->abcdijkl", H.ab.oooo, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb += np.einsum("acef,edil,bfjk->abcdijkl", H.ab.vvvv, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    dT.aabb += 0.25 * np.einsum("mnkl,adin,bcjm->abcdijkl", H.bb.oooo, T.ab, T.ab, optimize=True)   # (ij)(cd) = 4 !!! (tricky asym)
    dT.aabb += 0.25 * np.einsum("cdef,afil,bejk->abcdijkl", H.bb.vvvv, T.ab, T.ab, optimize=True)   # (ij)(kl) = 4 !!! (tricky asym)

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    dT.aabb -= (8.0 / 16.0) * np.einsum("mdil,abcmjk->abcdijkl", H.ab.ovoo, T.aab, optimize=True)  # [1]  (ij)(kl)(cd) = 8
    dT.aabb -= (2.0 / 16.0) * np.einsum("bmji,acdmkl->abcdijkl", H.aa.vooo, T.abb, optimize=True)  # [2]  (ab) = 2
    dT.aabb -= (2.0 / 16.0) * np.einsum("cmkl,abdijm->abcdijkl", H.bb.vooo, T.aab, optimize=True)  # [3]  (cd) = 2
    dT.aabb -= (8.0 / 16.0) * np.einsum("amil,bcdjkm->abcdijkl", H.ab.vooo, T.abb, optimize=True)  # [4]  (ij)(ab)(kl) = 8
    dT.aabb += (8.0 / 16.0) * np.einsum("adel,becjik->abcdijkl", H.ab.vvvo, T.aab, optimize=True)  # [5]  (ab)(kl)(cd) = 8
    dT.aabb += (2.0 / 16.0) * np.einsum("baje,ecdikl->abcdijkl", H.aa.vvov, T.abb, optimize=True)  # [6]  (ij) = 2
    dT.aabb += (8.0 / 16.0) * np.einsum("adie,bcejkl->abcdijkl", H.ab.vvov, T.abb, optimize=True)  # [7]  (ij)(ab)(cd) = 8
    dT.aabb += (2.0 / 16.0) * np.einsum("cdke,abeijl->abcdijkl", H.bb.vvov, T.aab, optimize=True)  # [8]  (kl) = 2

    I3B_oovooo = (
                np.einsum("mnif,fdjl->mndijl", H.aa.ooov, T.ab, optimize=True)
               + 0.25 * np.einsum("mnef,efdijl->mndijl", H.aa.oovv, T.aab, optimize=True)
    )
    I3B_oovooo -= np.transpose(I3B_oovooo, (0, 1, 2, 4, 3, 5))
    dT.aabb += (4.0 / 16.0) * 0.5 * np.einsum("mndijl,abcmnk->abcdijkl", I3B_oovooo, T.aab, optimize=True)  # [9]  (kl)(cd) = 4

    I3B_ovoooo = (
                np.einsum("mnif,bfjl->mbnijl", H.ab.ooov, T.ab, optimize=True)
                + 0.5 * np.einsum("mnfl,bfji->mbnijl", H.ab.oovo, T.aa, optimize=True)
                + 0.5 * np.einsum("mnef,befjil->mbnijl", H.ab.oovv, T.aab, optimize=True)
    )
    I3B_ovoooo -= np.transpose(I3B_ovoooo, (0, 1, 2, 4, 3, 5))
    dT.aabb += (4.0 / 16.0) * np.einsum("mbnijl,acdmkn->abcdijkl", I3B_ovoooo, T.abb, optimize=True)  # [10]  (kl)(ab) = 4

    I3C_vooooo = (
                np.einsum("nmlf,afik->amnikl", H.bb.ooov, T.ab, optimize=True)
                + 0.25 * np.einsum("mnef,aefikl->amnikl", H.bb.oovv, T.abb, optimize=True)
    )
    I3C_vooooo -= np.transpose(I3C_vooooo, (0, 1, 2, 3, 5, 4))
    dT.aabb += (4.0 / 16.0) * 0.5 * np.einsum("amnikl,bcdjmn->abcdijkl", I3C_vooooo, T.abb, optimize=True)  # [11]  (ij)(ab) = 4

    I3C_oovooo = (
                0.5 * np.einsum("mnif,cfkl->mncilk", H.ab.ooov, T.bb, optimize=True)
                + np.einsum("mnfl,fcik->mncilk", H.ab.oovo, T.ab, optimize=True)
                + 0.5 * np.einsum("mnef,efcilk->mncilk", H.ab.oovv, T.abb, optimize=True)
    )
    I3C_oovooo -= np.transpose(I3C_oovooo, (0, 1, 2, 3, 5, 4))
    dT.aabb += (4.0 / 16.0) * np.einsum("mncilk,abdmjn->abcdijkl", I3C_oovooo, T.aab, optimize=True)  # [12]  (ij)(cd) = 4

    I3B_vvvvvo = -np.einsum("bmfe,acmk->abcefk", H.aa.vovv, T.ab, optimize=True)
    I3B_vvvvvo -= np.transpose(I3B_vvvvvo, (1, 0, 2, 3, 4, 5))
    dT.aabb += (4.0 / 16.0) * 0.5 * np.einsum("abcefk,efdijl->abcdijkl", I3B_vvvvvo, T.aab, optimize=True)  # [13]  (kl)(cd) = 4

    I3C_vvvvov = (
                -np.einsum("mdef,acmk->acdekf", H.ab.ovvv, T.ab, optimize=True)
                - 0.5 * np.einsum("amef,cdkm->acdekf", H.ab.vovv, T.bb, optimize=True)
    )
    I3C_vvvvov -= np.transpose(I3C_vvvvov, (0, 2, 1, 3, 4, 5))
    dT.aabb += (4.0 / 16.0) * np.einsum("acdekf,ebfijl->abcdijkl", I3C_vvvvov, T.aab, optimize=True)  # [14]  (kl)(ab) = 4

    I3B_vvvvov = (
                -0.5 * np.einsum("mdef,abmj->abdejf", H.ab.ovvv, T.aa, optimize=True)
                -np.einsum("amef,bdjm->abdejf", H.ab.vovv, T.ab, optimize=True)
    )
    I3B_vvvvov -= np.transpose(I3B_vvvvov, (1, 0, 2, 3, 4, 5))
    dT.aabb += (4.0 / 16.0) * np.einsum("abdejf,efcilk->abcdijkl", I3B_vvvvov, T.abb, optimize=True)  # [15]  (ij)(cd) = 4

    I3C_vvvovv = -np.einsum("cmef,adim->acdief", H.bb.vovv, T.ab, optimize=True)
    I3C_vvvovv -= np.transpose(I3C_vvvovv, (0, 2, 1, 3, 4, 5))
    dT.aabb += (4.0 / 16.0) * 0.5 * np.einsum("acdief,befjkl->abcdijkl", I3C_vvvovv, T.abb, optimize=True)  # [16]  (ij)(ab) = 4

    I3A_vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.aa.ooov, T.aa, optimize=True)
                +0.5 * np.einsum("bmfe,afij->abmije", H.aa.vovv, T.aa, optimize=True)
                +0.25 * np.einsum("mnef,abfijn->abmije", H.aa.oovv, T.aaa, optimize=True)
                +0.25 * np.einsum("mnef,abfijn->abmije", H.ab.oovv, T.aab, optimize=True)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aabb += (1.0 / 16.0) * np.einsum("abmije,ecdmkl->abcdijkl", I3A_vvooov, T.abb, optimize=True)  # [17]  (1) = 1

    I3B_vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.ab.ooov, T.aa, optimize=True)
                +0.5 * np.einsum("bmfe,afij->abmije", H.ab.vovv, T.aa, optimize=True)
                +0.25 * np.einsum("nmfe,abfijn->abmije", H.ab.oovv, T.aaa, optimize=True)
                +0.25 * np.einsum("nmfe,abfijn->abmije", H.bb.oovv, T.aab, optimize=True)
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aabb += (1.0 / 16.0) * np.einsum("abmije,ecdmkl->abcdijkl", I3B_vvooov, T.bbb, optimize=True)  # [18]  (1) = 1

    I3C_ovvvoo = (
                -0.5 * np.einsum("mnek,cdnl->mcdekl", H.ab.oovo, T.bb, optimize=True)
                +0.5 * np.einsum("mcef,fdkl->mcdekl", H.ab.ovvv, T.bb, optimize=True)
    )
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 2, 1, 3, 4, 5))
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 1, 2, 3, 5, 4))
    dT.aabb += (1.0 / 16.0) * np.einsum("mcdekl,abeijm->abcdijkl", I3C_ovvvoo, T.aaa, optimize=True)  # [19]  (1) = 1

    I3D_vvooov = (
                -0.5 * np.einsum("nmke,cdnl->cdmkle", H.bb.ooov, T.bb, optimize=True)
                +0.5 * np.einsum("cmfe,fdkl->cdmkle", H.bb.vovv, T.bb, optimize=True)
    )
    I3D_vvooov -= np.transpose(I3D_vvooov, (1, 0, 2, 3, 4, 5))
    I3D_vvooov -= np.transpose(I3D_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aabb += (1.0 / 16.0) * np.einsum("cdmkle,abeijm->abcdijkl", I3D_vvooov, T.aab, optimize=True)  # [20]  (1) = 1

    I3B_vovovo = (
                -np.einsum("mnel,adin->amdiel", H.ab.oovo, T.ab, optimize=True)
                +np.einsum("mdef,afil->amdiel", H.ab.ovvv, T.ab, optimize=True)
                +0.5 * np.einsum("mnef,afdinl->amdiel", H.aa.oovv, T.aab, optimize=True) # !!! factor 1/2 to compensate asym
                +np.einsum("mnef,afdinl->amdiel", H.ab.oovv, T.abb, optimize=True)
                -np.einsum("nmie,adnl->amdiel", H.aa.ooov, T.ab, optimize=True)
                +np.einsum("amfe,fdil->amdiel", H.aa.vovv, T.ab, optimize=True)
    )
    dT.aabb += np.einsum("amdiel,becjmk->abcdijkl", I3B_vovovo, T.aab, optimize=True)  # [21]  (ij)(kl)(ab)(cd) = 16

    I3C_vovovo = (
                -np.einsum("nmie,adnl->amdiel", H.ab.ooov, T.ab, optimize=True)
                +np.einsum("amfe,fdil->amdiel", H.ab.vovv, T.ab, optimize=True)
                -np.einsum("nmle,adin->amdiel", H.bb.ooov, T.ab, optimize=True)
                +np.einsum("dmfe,afil->amdiel", H.bb.vovv, T.ab, optimize=True)
                +0.5 * np.einsum("mnef,afdinl->amdiel", H.bb.oovv, T.abb, optimize=True) # !!! factor 1/2 to compensate asym
    )
    dT.aabb += np.einsum("amdiel,becjmk->abcdijkl", I3C_vovovo, T.abb, optimize=True)  # [22]  (ij)(kl)(ab)(cd) = 16

    I3B_vovoov = (
                -np.einsum("mnie,bdjn->bmdjie", H.ab.ooov, T.ab, optimize=True)
                +0.5 * np.einsum("mdfe,bfji->bmdjie", H.ab.ovvv, T.aa, optimize=True)
                -0.5 * np.einsum("mnfe,bfdjin->bmdjie", H.ab.oovv, T.aab, optimize=True)
    )
    I3B_vovoov -= np.transpose(I3B_vovoov, (0, 1, 2, 4, 3, 5))
    dT.aabb -= (4.0 / 16.0) * np.einsum("bmdjie,aecmlk->abcdijkl", I3B_vovoov, T.abb, optimize=True)  # [23]  (ab)(cd) = 4

    I3C_ovvoov = (
                -0.5 * np.einsum("mnie,cdkn->mcdike", H.ab.ooov, T.bb, optimize=True)
                +np.einsum("mdfe,fcik->mcdike", H.ab.ovvv, T.ab, optimize=True)
                -0.5 * np.einsum("mnfe,fcdikn->mcdike", H.ab.oovv, T.abb, optimize=True)
    )
    I3C_ovvoov -= np.transpose(I3C_ovvoov, (0, 2, 1, 3, 4, 5))
    dT.aabb -= (4.0 / 16.0) * np.einsum("mcdike,abemjl->abcdijkl", I3C_ovvoov, T.aab, optimize=True)  # [24]  (ij)(kl) = 4

    I3B_vvovoo = (
                -0.5 * np.einsum("nmel,abnj->abmejl", H.ab.oovo, T.aa, optimize=True)
                +np.einsum("amef,bfjl->abmejl", H.ab.vovv, T.ab, optimize=True)
    )
    I3B_vvovoo -= np.transpose(I3B_vvovoo, (1, 0, 2, 3, 4, 5))
    dT.aabb -= (4.0 / 16.0) * np.einsum("abmejl,ecdikm->abcdijkl", I3B_vvovoo, T.abb, optimize=True)  # [25]  (ij)(kl) = 4

    I3C_vovvoo = (
                -np.einsum("nmel,acnk->amcelk", H.ab.oovo, T.ab, optimize=True)
                +0.5 * np.einsum("amef,fclk->amcelk", H.ab.vovv, T.bb, optimize=True)
    )
    I3C_vovvoo -= np.transpose(I3C_vovvoo, (0, 1, 2, 3, 5, 4))
    dT.aabb -= (4.0 / 16.0) * np.einsum("amcelk,bedjim->abcdijkl", I3C_vovvoo, T.aab, optimize=True)  # [26]  (ab)(cd) = 4

    # <ijklabcd | (H(2)*T4)_C | 0 >
    dT.aabb -= (2.0 / 16.0) * np.einsum("mi,abcdmjkl->abcdijkl", H.a.oo, T.aabb, optimize=True)  # [1]  (ij) = 2
    dT.aabb -= (2.0 / 16.0) * np.einsum("ml,abcdijkm->abcdijkl", H.b.oo, T.aabb, optimize=True)  # [2]  (kl) = 2
    dT.aabb += (2.0 / 16.0) * np.einsum("ae,ebcdijkl->abcdijkl", H.a.vv, T.aabb, optimize=True)  # [3]  (ab) = 2
    dT.aabb += (2.0 / 16.0) * np.einsum("de,abceijkl->abcdijkl", H.b.vv, T.aabb, optimize=True)  # [4]  (cd) = 2
    dT.aabb += (1.0 / 16.0) * 0.5 * np.einsum("mnij,abcdmnkl->abcdijkl", H.aa.oooo, T.aabb, optimize=True)  # [5]  (1) = 1
    dT.aabb += (4.0 / 16.0) * np.einsum("mnil,abcdmjkn->abcdijkl", H.ab.oooo, T.aabb, optimize=True)  # [6]  (ij)(kl) = 4
    dT.aabb += (1.0 / 16.0) * 0.5 * np.einsum("mnkl,abcdijmn->abcdijkl", H.bb.oooo, T.aabb, optimize=True)  #  [7]  (1) = 1
    dT.aabb += (1.0 / 16.0) * 0.5 * np.einsum("abef,efcdijkl->abcdijkl", H.aa.vvvv, T.aabb, optimize=True)  #  [8]  (1) = 1
    dT.aabb += (4.0 / 16.0) * np.einsum("adef,ebcfijkl->abcdijkl", H.ab.vvvv, T.aabb, optimize=True)  #  [9]  (ab)(cd) = 4
    dT.aabb += (1.0 / 16.0) * 0.5 * np.einsum("cdef,abefijkl->abcdijkl", H.bb.vvvv, T.aabb, optimize=True)  #  [10]  (1) = 1
    dT.aabb += (4.0 / 16.0) * np.einsum("amie,ebcdmjkl->abcdijkl", H.aa.voov, T.aabb, optimize=True)  #  [11]  (ij)(ab) = 4
    dT.aabb += (4.0 / 16.0) * np.einsum("amie,becdjmkl->abcdijkl", H.ab.voov, T.abbb, optimize=True)  #  [12]  (ij)(ab) = 4
    dT.aabb += (4.0 / 16.0) * np.einsum("mdel,aebcimjk->abcdijkl", H.ab.ovvo, T.aaab, optimize=True)  #  [13]  (kl)(cd) = 4
    dT.aabb += (4.0 / 16.0) * np.einsum("dmle,abceijkm->abcdijkl", H.bb.voov, T.aabb, optimize=True)  #  [14]  (kl)(cd) = 4
    dT.aabb -= (4.0 / 16.0) * np.einsum("mdie,abcemjkl->abcdijkl", H.ab.ovov, T.aabb, optimize=True)  #  [15]  (ij)(cd) = 4
    dT.aabb -= (4.0 / 16.0) * np.einsum("amel,ebcdijkm->abcdijkl", H.ab.vovo, T.aabb, optimize=True)  #  [16]  (kl)(ab) = 4

    I3C_vvvvoo = (
                -0.5 * np.einsum("mnef,afcdmnkl->acdekl", H.aa.oovv, T.aabb, optimize=True)
                -np.einsum("mnef,afcdmnkl->acdekl", H.ab.oovv, T.abbb, optimize=True)
    )
    dT.aabb += (2.0 / 16.0) * np.einsum("acdekl,beji->abcdijkl", I3C_vvvvoo, T.aa, optimize=True)  #  [17]  (ab) = 2

    I3B_vvvvoo = (
                -0.5 * np.einsum("mnef,abfcmjnk->abcejk", H.aa.oovv, T.aaab, optimize=True)
                -np.einsum("mnef,abfcmjnk->abcejk", H.ab.oovv, T.aabb, optimize=True)
    )
    dT.aabb += (8.0 / 16.0) * np.einsum("abcejk,edil->abcdijkl", I3B_vvvvoo, T.ab, optimize=True)  #  [18]  (ij)(kl)(cd) = 8

    I3C_vvvoov = (
                -np.einsum("nmfe,bfcdjnkm->bcdjke", H.ab.oovv, T.aabb, optimize=True)
                -0.5 * np.einsum("mnef,bcdfjkmn->bcdjke", H.bb.oovv, T.abbb, optimize=True)
    )
    dT.aabb += (8.0 / 16.0) * np.einsum("bcdjke,aeil->abcdijkl", I3C_vvvoov, T.ab, optimize=True)  #  [19]  (ij)(kl)(ab) = 8

    I3B_vvvoov = (
                -np.einsum("nmfe,abfdijnm->abdije", H.ab.oovv, T.aaab, optimize=True)
                -0.5 * np.einsum("mnef,abfdijnm->abdije", H.bb.oovv, T.aabb, optimize=True)
    )
    dT.aabb += (2.0 / 16.0) * np.einsum("abdije,eclk->abcdijkl", I3B_vvvoov, T.bb, optimize=True)  #  [20]  (cd) = 2

    I3C_ovvooo = (
                0.5 * np.einsum("mnef,efcdinkl->mcdikl", H.aa.oovv, T.aabb, optimize=True)
                +np.einsum("mnef,efcdinkl->mcdikl", H.ab.oovv, T.abbb, optimize=True)
    )
    dT.aabb -= (2.0 / 16.0) * np.einsum("mcdikl,abmj->abcdijkl", I3C_ovvooo, T.aa, optimize=True)  #  [21]  (ij) = 2

    I3B_vovooo = (
                0.5 * np.einsum("mnef,befcjink->bmcjik", H.aa.oovv, T.aaab, optimize=True)
                +np.einsum("mnef,befcjink->bmcjik", H.ab.oovv, T.aabb, optimize=True)
    )
    dT.aabb -= (8.0 / 16.0) * np.einsum("bmcjik,adml->abcdijkl", I3B_vovooo, T.ab, optimize=True)  #  [22]  (ab)(kl)(cd) = 8

    I3C_vovooo = (
                np.einsum("nmfe,bfecjnlk->bmcjlk", H.ab.oovv, T.aabb, optimize=True)
                +0.5 * np.einsum("mnef,bfecjnlk->bmcjlk", H.bb.oovv, T.abbb, optimize=True)
    )
    dT.aabb -= (8.0 / 16.0) * np.einsum("bmcjlk,adim->abcdijkl", I3C_vovooo, T.ab, optimize=True)  #  [23]  (ij)(ab)(cd) = 8

    I3B_vvoooo = (
                np.einsum("nmfe,abfeijnl->abmijl", H.ab.oovv, T.aaab, optimize=True)
                +0.5 * np.einsum("mnef,abfeijnl->abmijl", H.bb.oovv, T.aabb, optimize=True)
    )
    dT.aabb -= (2.0 / 16.0) * np.einsum("abmijl,cdkm->abcdijkl", I3B_vvoooo, T.bb, optimize=True)  #  [24]  (kl) = 2

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

def update_t4d(T, dT, hbar, H, shift):
    return T, dT

def update_t4e(T, dT, hbar, H, shift):
    return T, dT

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
        np.einsum("mnef,fn->me", H0.aa.oovv, T.a, optimize=True)
        + np.einsum("mnef,fn->me", H0.ab.oovv, T.b, optimize=True)
    ) # no(2)nu(2)

    H.b.ov = H0.b.ov + (
            np.einsum("nmfe,fn->me", H0.ab.oovv, T.a, optimize=True)
            + np.einsum("mnef,fn->me", H0.bb.oovv, T.b, optimize=True)
    ) # no(2)nu(2)

    H.a.vv = H0.a.vv + (
        np.einsum("anef,fn->ae", H0.aa.vovv, T.a, optimize=True)
        + np.einsum("anef,fn->ae", H0.ab.vovv, T.b, optimize=True)
        - np.einsum("me,am->ae", H.a.ov, T.a, optimize=True)
    ) # no(1)nu(3)

    H.a.oo = H0.a.oo + (
        np.einsum("mnif,fn->mi", H0.aa.ooov, T.a, optimize=True)
        + np.einsum("mnif,fn->mi", H0.ab.ooov, T.b, optimize=True)
        + np.einsum("me,ei->mi", H.a.ov, T.a, optimize=True)
    ) # no(3)nu(1)

    H.b.vv = H0.b.vv + (
        np.einsum("anef,fn->ae", H0.bb.vovv, T.b, optimize=True)
        + np.einsum("nafe,fn->ae", H0.ab.ovvv, T.a, optimize=True)
        - np.einsum("me,am->ae", H.b.ov, T.b, optimize=True)
    ) # no(1)nu(3)

    H.b.oo = H0.b.oo + (
        np.einsum("mnif,fn->mi", H0.bb.ooov, T.b, optimize=True)
        + np.einsum("nmfi,fn->mi", H0.ab.oovo, T.a, optimize=True)
        + np.einsum("me,ei->mi", H.b.ov, T.b, optimize=True)
    ) # no(3)nu(1)

    # 2-body components
    # -------------------#
    # AA parts
    # -------------------#
    H.aa.ooov = np.einsum("mnfe,fi->mnie", H0.aa.oovv, T.a, optimize=True) # no(3)nu(2)

    H.aa.oooo = 0.5 * H0.aa.oooo + np.einsum("nmje,ei->mnij", H0.aa.ooov + 0.5 * H.aa.ooov, T.a, optimize=True) # no(4)nu(1)
    H.aa.oooo -= np.transpose(H.aa.oooo, (0, 1, 3, 2))

    H.aa.vovv = -np.einsum("mnfe,an->amef", H0.aa.oovv, T.a, optimize=True) # no(2)nu(3)

    H.aa.voov = H0.aa.voov + (
            np.einsum("amfe,fi->amie", H0.aa.vovv + 0.5 * H.aa.vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", H0.aa.ooov + 0.5 * H.aa.ooov, T.a, optimize=True)
    ) # no(2)nu(3)

    L_amie = H0.aa.voov + 0.5 * np.einsum('amef,ei->amif', H0.aa.vovv, T.a, optimize=True) # no(2)nu(3)
    X_mnij = H0.aa.oooo + np.einsum('mnie,ej->mnij', H.aa.ooov, T.a, optimize=True) # no(4)nu(1)
    H.aa.vooo = 0.5 * H0.aa.vooo + (
        np.einsum('amie,ej->amij', L_amie, T.a, optimize=True)
       -0.25 * np.einsum('mnij,am->anij', X_mnij, T.a, optimize=True)
    ) # no(3)nu(2)
    H.aa.vooo -= np.transpose(H.aa.vooo, (0, 1, 3, 2))

    L_amie = np.einsum('mnie,am->anie', H0.aa.ooov, T.a, optimize=True)
    H.aa.vvov = H0.aa.vvov + np.einsum("anie,bn->abie", L_amie, T.a, optimize=True) # no(1)nu(4)
    #H.aa.vvov -= np.transpose(H.aa.vvov, (1, 0, 2, 3)) # WHY IS THIS NOT NEEDED???

    # -------------------#
    # AB parts
    # -------------------#
    H.ab.ooov = np.einsum("mnfe,fi->mnie", H0.ab.oovv, T.a, optimize=True)

    H.ab.oovo = np.einsum("nmef,fi->nmei", H0.ab.oovv, T.b, optimize=True)

    H.ab.oooo = H0.ab.oooo + (
        np.einsum("mnej,ei->mnij", H0.ab.oovo + 0.5 * H.ab.oovo, T.a, optimize=True)
        + np.einsum("mnie,ej->mnij", H0.ab.ooov + 0.5 * H.ab.ooov, T.b, optimize=True)
    )

    H.ab.vovv = -np.einsum("nmef,an->amef", H0.ab.oovv, T.a, optimize=True)

    H.ab.ovvv = -np.einsum("mnef,an->maef", H0.ab.oovv, T.b, optimize=True)

    H.ab.voov = H0.ab.voov + (
        np.einsum("amfe,fi->amie", H0.ab.vovv + 0.5 * H.ab.vovv, T.a, optimize=True)
        - np.einsum("nmie,an->amie", H0.ab.ooov + 0.5 * H.ab.ooov, T.a, optimize=True)
    )

    H.ab.ovvo = H0.ab.ovvo + (
        np.einsum("maef,fi->maei", H0.ab.ovvv + 0.5 * H.ab.ovvv, T.b, optimize=True)
        - np.einsum("mnei,an->maei", H0.ab.oovo + 0.5 * H.ab.oovo, T.b, optimize=True)
    )

    H.ab.ovov = H0.ab.ovov + (
        np.einsum("mafe,fi->maie", H0.ab.ovvv + 0.5 * H.ab.ovvv, T.a, optimize=True)
        - np.einsum("mnie,an->maie", H0.ab.ooov + 0.5 * H.ab.ooov, T.b, optimize=True)
    )

    H.ab.vovo = H0.ab.vovo - (
        np.einsum("nmei,an->amei", H0.ab.oovo + 0.5 * H.ab.oovo, T.a, optimize=True)
        - np.einsum("amef,fi->amei", H0.ab.vovv + 0.5 * H.ab.vovv, T.b, optimize=True)
    )

    X_mnij = H0.ab.oooo + (
        np.einsum("mnif,fj->mnij", H0.ab.ooov, T.b, optimize=True)
        +np.einsum("mnej,ei->mnij", H0.ab.oovo, T.a, optimize=True)
    )
    L_mbej = H0.ab.ovvo + np.einsum("mbef,fj->mbej", H0.ab.ovvv, T.b, optimize=True)
    H.ab.ovoo = H0.ab.ovoo + (
        np.einsum("mbej,ei->mbij", L_mbej, T.a, optimize=True)
        -np.einsum("mnij,bn->mbij", X_mnij, T.b, optimize=True)
    )

    L_amie = np.einsum("amef,ei->amif", H0.ab.vovv + H.ab.vovv, T.a, optimize=True)
    H.ab.vooo = H0.ab.vooo + np.einsum("amif,fj->amij", H0.ab.voov + L_amie, T.b, optimize=True)

    H.ab.vvvo = H0.ab.vvvo - np.einsum("anej,bn->abej", H0.ab.vovo, T.b, optimize=True)

    H.ab.vvov = H0.ab.vvov - np.einsum("mbie,am->abie", H0.ab.ovov, T.a, optimize=True)
    # -------------------#
    # BB parts
    # -------------------#
    H.bb.ooov = np.einsum("mnfe,fi->mnie", H0.bb.oovv, T.b, optimize=True)

    H.bb.oooo = 0.5 * H0.bb.oooo + np.einsum("nmje,ei->mnij", H0.bb.ooov + 0.5 * H.bb.ooov, T.b, optimize=True)
    H.bb.oooo -= np.transpose(H.bb.oooo, (0, 1, 3, 2))

    H.bb.vovv = -np.einsum("mnfe,an->amef", H0.bb.oovv, T.b, optimize=True)

    H.bb.voov = H0.bb.voov + (
        np.einsum("amfe,fi->amie", H0.bb.vovv + 0.5 * H.bb.vovv, T.b, optimize=True)
        - np.einsum("nmie,an->amie", H0.bb.ooov + 0.5 * H.bb.ooov, T.b, optimize=True)
    )

    L_amie = H0.bb.voov + 0.5 * np.einsum('amef,ei->amif', H0.bb.vovv, T.b, optimize=True)
    X_mnij = H0.bb.oooo + np.einsum('mnie,ej->mnij', H.bb.ooov, T.b, optimize=True)
    H.bb.vooo = 0.5 * H0.bb.vooo + (
        np.einsum('amie,ej->amij', L_amie, T.b, optimize=True)
       -0.25 * np.einsum('mnij,am->anij', X_mnij, T.b, optimize=True)
    )
    H.bb.vooo -= np.transpose(H.bb.vooo, (0, 1, 3, 2))

    L_amie = np.einsum('mnie,am->anie', H0.bb.ooov, T.b, optimize=True)
    H.bb.vvov = H0.bb.vvov + + np.einsum("anie,bn->abie", L_amie, T.b, optimize=True)
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
            np.einsum("imae,em->ia", H0.aa.oovv, T.a, optimize=True)
            + np.einsum("imae,em->ia", H0.ab.oovv, T.b, optimize=True)
    )

    H.a.oo += (
            np.einsum("je,ei->ji", H.a.ov, T.a, optimize=True)
            + np.einsum("jmie,em->ji", H0.aa.ooov, T.a, optimize=True)
            + np.einsum("jmie,em->ji", H0.ab.ooov, T.b, optimize=True)
            + 0.5 * np.einsum("jnef,efin->ji", H0.aa.oovv, T.aa, optimize=True)
            + np.einsum("jnef,efin->ji", H0.ab.oovv, T.ab, optimize=True)
    )

    H.a.vv += (
            - np.einsum("mb,am->ab", H.a.ov, T.a, optimize=True)
            + np.einsum("ambe,em->ab", H0.aa.vovv, T.a, optimize=True)
            + np.einsum("ambe,em->ab", H0.ab.vovv, T.b, optimize=True)
            - 0.5 * np.einsum("mnbf,afmn->ab", H0.aa.oovv, T.aa, optimize=True)
            - np.einsum("mnbf,afmn->ab", H0.ab.oovv, T.ab, optimize=True)
    )

    H.b.ov += (
            np.einsum("imae,em->ia", H0.bb.oovv, T.b, optimize=True)
            + np.einsum("miea,em->ia", H0.ab.oovv, T.a, optimize=True)
    )

    H.b.oo += (
            np.einsum("je,ei->ji", H.b.ov, T.b, optimize=True)
            + np.einsum("jmie,em->ji", H0.bb.ooov, T.b, optimize=True)
            + np.einsum("mjei,em->ji", H0.ab.oovo, T.a, optimize=True)
            + 0.5 * np.einsum("jnef,efin->ji", H0.bb.oovv, T.bb, optimize=True)
            + np.einsum("njfe,feni->ji", H0.ab.oovv, T.ab, optimize=True)
    )

    H.b.vv += (
            - np.einsum("mb,am->ab", H.b.ov, T.b, optimize=True)
            + np.einsum("ambe,em->ab", H0.bb.vovv, T.b, optimize=True)
            + np.einsum("maeb,em->ab", H0.ab.ovvv, T.a, optimize=True)
            - 0.5 * np.einsum("mnbf,afmn->ab", H0.bb.oovv, T.bb, optimize=True)
            - np.einsum("nmfb,fanm->ab", H0.ab.oovv, T.ab, optimize=True)
    )

    Q1 = -np.einsum("mnfe,an->amef", H0.aa.oovv, T.a, optimize=True)
    I2A_vovv = H0.aa.vovv + 0.5 * Q1
    H.aa.vovv = I2A_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H0.aa.oovv, T.a, optimize=True)
    I2A_ooov = H0.aa.ooov + 0.5 * Q1
    H.aa.ooov = I2A_ooov + 0.5 * Q1

    Q1 = -np.einsum("nmef,an->amef", H0.ab.oovv, T.a, optimize=True)
    I2B_vovv = H0.ab.vovv + 0.5 * Q1
    H.ab.vovv = I2B_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H0.ab.oovv, T.a, optimize=True)
    I2B_ooov = H0.ab.ooov + 0.5 * Q1
    H.ab.ooov = I2B_ooov + 0.5 * Q1

    Q1 = -np.einsum("mnef,an->maef", H0.ab.oovv, T.b, optimize=True)
    I2B_ovvv = H0.ab.ovvv + 0.5 * Q1
    H.ab.ovvv = I2B_ovvv + 0.5 * Q1

    Q1 = np.einsum("nmef,fi->nmei", H0.ab.oovv, T.b, optimize=True)
    I2B_oovo = H0.ab.oovo + 0.5 * Q1
    H.ab.oovo = I2B_oovo + 0.5 * Q1

    Q1 = -np.einsum("nmef,an->amef", H0.bb.oovv, T.b, optimize=True)
    I2C_vovv = H0.bb.vovv + 0.5 * Q1
    H.bb.vovv = I2C_vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H0.bb.oovv, T.b, optimize=True)
    I2C_ooov = H0.bb.ooov + 0.5 * Q1
    H.bb.ooov = I2C_ooov + 0.5 * Q1

    Q1 = -np.einsum("bmfe,am->abef", I2A_vovv, T.a, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvvv += 0.5 * np.einsum("mnef,abmn->abef", H0.aa.oovv, T.aa, optimize=True) + Q1

    H.ab.vvvv += (
            - np.einsum("mbef,am->abef", I2B_ovvv, T.a, optimize=True)
            - np.einsum("amef,bm->abef", I2B_vovv, T.b, optimize=True)
            + np.einsum("mnef,abmn->abef", H0.ab.oovv, T.ab, optimize=True)
    )

    Q1 = -np.einsum("bmfe,am->abef", I2C_vovv, T.b, optimize=True)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvvv += 0.5 * np.einsum("mnef,abmn->abef", H0.bb.oovv, T.bb, optimize=True) + Q1

    Q1 = +np.einsum("nmje,ei->mnij", I2A_ooov, T.a, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.oooo += 0.5 * np.einsum("mnef,efij->mnij", H0.aa.oovv, T.aa, optimize=True) + Q1

    H.ab.oooo += (
            np.einsum("mnej,ei->mnij", I2B_oovo, T.a, optimize=True)
            + np.einsum("mnie,ej->mnij", I2B_ooov, T.b, optimize=True)
            + np.einsum("mnef,efij->mnij", H0.ab.oovv, T.ab, optimize=True)
    )

    Q1 = +np.einsum("nmje,ei->mnij", I2C_ooov, T.b, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.oooo += 0.5 * np.einsum("mnef,efij->mnij", H0.bb.oovv, T.bb, optimize=True) + Q1

    H.aa.voov += (
            np.einsum("amfe,fi->amie", I2A_vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", I2A_ooov, T.a, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.aa.oovv, T.aa, optimize=True)
            + np.einsum("mnef,afin->amie", H0.ab.oovv, T.ab, optimize=True)
    )

    H.ab.voov += (
            np.einsum("amfe,fi->amie", I2B_vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", I2B_ooov, T.a, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.ab.oovv, T.aa, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.bb.oovv, T.ab, optimize=True)
    )

    H.ab.ovvo += (
            np.einsum("maef,fi->maei", I2B_ovvv, T.b, optimize=True)
            - np.einsum("mnei,an->maei", I2B_oovo, T.b, optimize=True)
            + np.einsum("mnef,afin->maei", H0.ab.oovv, T.bb, optimize=True)
            + np.einsum("mnef,fani->maei", H0.aa.oovv, T.ab, optimize=True)
    )

    H.ab.ovov += (
            np.einsum("mafe,fi->maie", I2B_ovvv, T.a, optimize=True)
            - np.einsum("mnie,an->maie", I2B_ooov, T.b, optimize=True)
            - np.einsum("mnfe,fain->maie", H0.ab.oovv, T.ab, optimize=True)
    )

    H.ab.vovo += (
            - np.einsum("nmei,an->amei", I2B_oovo, T.a, optimize=True)
            + np.einsum("amef,fi->amei", I2B_vovv, T.b, optimize=True)
            - np.einsum("nmef,afni->amei", H0.ab.oovv, T.ab, optimize=True)
    )

    H.bb.voov += (
            np.einsum("amfe,fi->amie", I2C_vovv, T.b, optimize=True)
            - np.einsum("nmie,an->amie", I2C_ooov, T.b, optimize=True)
            + np.einsum("nmfe,afin->amie", H0.bb.oovv, T.bb, optimize=True)
            + np.einsum("nmfe,fani->amie", H0.ab.oovv, T.ab, optimize=True)
    )

    Q1 = (
            np.einsum("mnjf,afin->amij", H.aa.ooov, T.aa, optimize=True)
            + np.einsum("mnjf,afin->amij", H.ab.ooov, T.ab, optimize=True)
    )
    Q2 = H0.aa.voov + 0.5 * np.einsum("amef,ei->amif", H0.aa.vovv, T.a, optimize=True)
    Q2 = np.einsum("amif,fj->amij", Q2, T.a, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.aa.vooo += Q1 + (
            np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
            - np.einsum("nmij,an->amij", H.aa.oooo, T.a, optimize=True)
            + 0.5 * np.einsum("amef,efij->amij", H0.aa.vovv, T.aa, optimize=True)
    )

    Q1 = H0.ab.voov + np.einsum("amfe,fi->amie", H0.ab.vovv, T.a, optimize=True)
    H.ab.vooo += (
            np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
            - np.einsum("nmij,an->amij", H.ab.oooo, T.a, optimize=True)
            + np.einsum("mnjf,afin->amij", H.bb.ooov, T.ab, optimize=True)
            + np.einsum("nmfj,afin->amij", H.ab.oovo, T.aa, optimize=True)
            - np.einsum("nmif,afnj->amij", H.ab.ooov, T.ab, optimize=True)
            + np.einsum("amej,ei->amij", H0.ab.vovo, T.a, optimize=True)
            + np.einsum("amie,ej->amij", Q1, T.b, optimize=True)
            + np.einsum("amef,efij->amij", H0.ab.vovv, T.ab, optimize=True)
    )

    Q1 = H0.ab.ovov + np.einsum("mafe,fj->maje", H0.ab.ovvv, T.a, optimize=True)
    H.ab.ovoo += (
            np.einsum("me,eaji->maji", H.a.ov, T.ab, optimize=True)
            - np.einsum("mnji,an->maji", H.ab.oooo, T.b, optimize=True)
            + np.einsum("mnjf,fani->maji", H.aa.ooov, T.ab, optimize=True)
            + np.einsum("mnjf,fani->maji", H.ab.ooov, T.bb, optimize=True)
            - np.einsum("mnfi,fajn->maji", H.ab.oovo, T.ab, optimize=True)
            + np.einsum("maje,ei->maji", Q1, T.b, optimize=True)
            + np.einsum("maei,ej->maji", H0.ab.ovvo, T.a, optimize=True)
            + np.einsum("mafe,feji->maji", H0.ab.ovvv, T.ab, optimize=True)
    )

    Q1 = (
            np.einsum("mnjf,afin->amij", H.bb.ooov, T.bb, optimize=True)
            + np.einsum("nmfj,fani->amij", H.ab.oovo, T.ab, optimize=True)
    )
    Q2 = H0.bb.voov + 0.5 * np.einsum("amef,ei->amif", H0.bb.vovv, T.b, optimize=True)
    Q2 = np.einsum("amif,fj->amij", Q2, T.b, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    H.bb.vooo += Q1 + (
            + np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
            - np.einsum("nmij,an->amij", H.bb.oooo, T.b, optimize=True)
            + 0.5 * np.einsum("amef,efij->amij", H0.bb.vovv, T.bb, optimize=True)
    )

    Q1 = (
            np.einsum("bnef,afin->abie", H.aa.vovv, T.aa, optimize=True)
            + np.einsum("bnef,afin->abie", H.ab.vovv, T.ab, optimize=True)
    )
    Q2 = H0.aa.ovov - 0.5 * np.einsum("mnie,bn->mbie", H0.aa.ooov, T.a, optimize=True)
    Q2 = -np.einsum("mbie,am->abie", Q2, T.a, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.aa.vvov += Q1 + (
            - np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
            + np.einsum("abfe,fi->abie", H.aa.vvvv, T.a, optimize=True)
            + 0.5 * np.einsum("mnie,abmn->abie", H0.aa.ooov, T.aa, optimize=True)
    )

    Q1 = H0.ab.ovov - np.einsum("mnie,bn->mbie", H0.ab.ooov, T.b, optimize=True)
    Q1 = -np.einsum("mbie,am->abie", Q1, T.a, optimize=True)
    H.ab.vvov += Q1 + (
            - np.einsum("me,abim->abie", H.b.ov, T.ab, optimize=True)
            + np.einsum("abfe,fi->abie", H.ab.vvvv, T.a, optimize=True)
            + np.einsum("nbfe,afin->abie", H.ab.ovvv, T.aa, optimize=True)
            + np.einsum("bnef,afin->abie", H.bb.vovv, T.ab, optimize=True)
            - np.einsum("amfe,fbim->abie", H.ab.vovv, T.ab, optimize=True)
            - np.einsum("amie,bm->abie", H0.ab.voov, T.b, optimize=True)
            + np.einsum("nmie,abnm->abie", H0.ab.ooov, T.ab, optimize=True)
    )

    Q1 = H0.ab.vovo - np.einsum("nmei,bn->bmei", H0.ab.oovo, T.a, optimize=True)
    Q1 = -np.einsum("bmei,am->baei", Q1, T.b, optimize=True)
    H.ab.vvvo += Q1 + (
            - np.einsum("me,bami->baei", H.a.ov, T.ab, optimize=True)
            + np.einsum("baef,fi->baei", H.ab.vvvv, T.b, optimize=True)
            + np.einsum("bnef,fani->baei", H.aa.vovv, T.ab, optimize=True)
            + np.einsum("bnef,fani->baei", H.ab.vovv, T.bb, optimize=True)
            - np.einsum("maef,bfmi->baei", H.ab.ovvv, T.ab, optimize=True)
            - np.einsum("naei,bn->baei", H0.ab.ovvo, T.a, optimize=True)
            + np.einsum("nmei,banm->baei", H0.ab.oovo, T.ab, optimize=True)
    )

    Q1 = (
            np.einsum("bnef,afin->abie", H.bb.vovv, T.bb, optimize=True)
            + np.einsum("nbfe,fani->abie", H.ab.ovvv, T.ab, optimize=True)
    )
    Q2 = H.bb.ovov - 0.5 * np.einsum("mnie,bn->mbie", H0.bb.ooov, T.b, optimize=True)
    Q2 = -np.einsum("mbie,am->abie", Q2, T.b, optimize=True)
    Q1 += Q2
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    H.bb.vvov += Q1 + (
            - np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
            + np.einsum("abfe,fi->abie", H.bb.vvvv, T.b, optimize=True)
            + 0.5 * np.einsum("mnie,abmn->abie", H0.bb.ooov, T.bb, optimize=True)
    )


    return H
