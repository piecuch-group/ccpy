"""Module with functions that perform the CC with singles, doubles,
triples, and quadruples (CCSDTQ) calculation for a molecular system."""

import numpy as np

from ccpy.hbar.hbar_ccs import get_ccs_intermediates_opt
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.hbar.hbar_ccsdt import add_VT3_intermediates
from ccpy.utilities.updates import cc_loops2
from ccpy.utilities.updates import cc_loops_t4


def update(T, dT, H, shift, flag_RHF, system):
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
        dT.abb = np.transpose(dT.abb, (2, 1, 0, 5, 4, 3))
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
        dT.abb = np.transpose(dT.aaab, (3, 2, 1, 0, 7, 6, 5, 4))
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

    T.b, dT.b = cc_loops2.cc_loops2.update_t1b(
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
    dT.aa += 0.0625 * np.einsum("mnef,abefijmn->abij", H0.aa.oovv, T.aaaa, optimize=True)
    dT.aa += 0.25 * np.einsum("mnef,abefijmn->abij", H0.ab.oovv, T.aaab, optimize=True)
    dT.aa += 0.0625 * np.einsum("mnef,abefijmn->abij", H0.bb.oovv, T.aabb, optimize=True)

    T.aa, dT.aa = cc_loops2.cc_loops2.update_t2a(
        T.aa, dT.aa + 0.25 * H0.aa.vvoo, H0.a.oo, H0.a.vv, shift
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

    T.ab, dT.ab = cc_loops2.cc_loops2.update_t2b(
        T.ab, dT.ab + H0.ab.vvoo, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv, shift
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

    T.bb, dT.bb = cc_loops2.cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H0.bb.vvoo, H0.b.oo, H0.b.vv, shift
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

    T.aaa, dT.aaa = cc_loops2.cc_loops2.update_t3a_v2(
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

    T.bbb, dT.bbb = cc_loops2.cc_loops2.update_t3d_v2(
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
    dT.aaaa += 0.5 * (16.0 / 576.0) * np.einsum("bmnjkl,acdinm->abcdijkl", I3A_vooooo, T.aaa, optimize=True) # (b/acd)(i/jkl) = 4 * 4 = 16

    I3A_vvvovv = -np.einsum("dmfe,bcjm->bcdjef", H.aa.vovv, T.aa, optimize=True)
    I3A_vvvovv -= np.transpose(I3A_vvvovv, (2, 1, 0, 3, 4, 5)) + np.transpose(I3A_vvvovv, (0, 2, 1, 3, 4, 5))
    dT.aaaa += 0.5 * (16.0 / 576.0) * np.einsum("bcdjef,aefikl->abcdijkl", I3A_vvvovv, T.aaa, optimize=True) # (a/bcd)(j/ikl) = 4 * 4 = 16

    I3A_vvooov = -0.5 * np.einsum("nmke,cdnl->cdmkle", H.aa.ooov, T.aa, optimize=True)
    I3A_vvooov += 0.5 * np.einsum("cmfe,fdkl->cdmkle", H.aa.vovv, T.aa, optimize=True)
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    I3A_vvooov += (
                    0.5 * np.einsum("mnef,cdfkln->cdmkle", H.aa.oovv, T.aaa, optimize=True) # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
                   +np.einsum("mnef,cdfkln->cdmkle", H.ab.oovv, T.aab, optimize=True)
    )
    dT.aaaa += (36.0 / 576.0) * np.einsum("cdmkle,abeijm->abcdijkl", I3A_vvooov, T.aaa, optimize=True) # (cd/ab)(kl/ij) = 6 * 6 = 36

    I3B_vvooov = -0.5 * np.einsum("nmke,cdnl->cdmkle", H.ab.ooov, T.aa, optimize=True)
    I3B_vvooov += 0.5 * np.einsum("cmfe,fdkl->cdmkle", H.ab.vovv, T.aa, optimize=True)
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    I3B_vvooov += (
                    #np.einsum("mnef,cdfkln->cdmkle", H.ab.oovv, T.aaa, optimize=True) this is redundant!
                   + 0.5 * np.einsum("mnef,cdfkln->cdmkle", H.bb.oovv, T.aab, optimize=True) # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
    )
    dT.aaaa += (36.0 / 576.0) * np.einsum("cdmkle,abeijm->abcdijkl", I3B_vvooov, T.aab, optimize=True) # (cd/ab)(kl/ij) = 6 * 6 = 36

    # <ijklabcd | (H(2)*T4)_C | 0 >
    dT.aaaa -= (4.0 / 576.0) * np.einsum("ml,abcdijkm->abcdijkl", H.a.oo, T.aaaa, optimize=True) # (l/ijk) = 4
    dT.aaaa += (4.0 / 576.0) * np.einsum("de,abceijkl->abcdijkl", H.a.vv, T.aaaa, optimize=True) # (d/abc) = 4
    dT.aaaa += 0.5 * (6.0 / 576.0) * np.einsum("mnkl,abcdijmn->abcdijkl", H.aa.oooo, T.aaaa, optimize=True) # (kl/ij) = 6
    dT.aaaa += 0.5 * (6.0 / 576.0) * np.einsum("cdef,abefijkl->abcdijkl", H.aa.vvvv, T.aaaa, optimize=True) # (cd/ab) = 6
    dT.aaaa += (16.0 / 576.0) * np.einsum("dmle,abceijkm->abcdijkl", H.aa.voov, T.aaaa, optimize=True) # (d/abc)(l/ijk) = 16
    dT.aaaa += (16.0 / 576.0) * np.einsum("dmle,abceijkm->abcdijkl", H.ab.voov, T.aaab, optimize=True) # (d/abc)(l/ijk) = 16

    I3A_vvvoov = (
                    -0.5 * np.einsum("mnef,bcdfjkmn->bcdjke", H.aa.oovv, T.aaaa, optimize=True)
                    -np.einsum("mnef,bcdfjkmn->bcdjke", H.ab.oovv, T.aaab, optimize=True)
    )
    dT.aaaa += (24.0 / 576.0) * np.einsum("bcdjke,aeil->abcdijkl", I3A_vvvoov, T.aa, optimize=True) # (a/bcd)(jk/il) = 4 * 6 = 24

    I3A_vvoooo = (
                    0.5 * np.einsum("mnef,bcefjkln->bcmjkl", H.aa.oovv, T.aaaa, optimize=True)
                    +np.einsum("mnef,bcefjkln->bcmjkl", H.ab.oovv, T.aaab, optimize=True)
    )
    dT.aaaa -= (24.0 / 576.0) * np.einsum("bcmjkl,adim->abcdijkl", I3A_vvoooo, T.aa, optimize=True) # (bc/ad)(i/jkl) = 6 * 4 = 24


    T.aaaa, dT.aaaa = cc_loops_t4.cc_loops_t4.update_t4a(
        T.aaaa,
        dT.aaaa,
        H0.a.oo,
        H0.a.vv,
        shift,
    )
    return T, dT


def update_t4b(T, dT, H, H0, shift):

    # <ijklabcd | H(2) | 0 >
    dT.aaab = -(9.0 / 36.0) * np.einsum("mdel,abim,ecjk->abcdijkl", H.ab.ovvo, T.aa, T.aa, optimize=True)   # (i/jk)(c/ab) = 9
    dT.aaab += (9.0 / 36.0) * np.einsum("mnij,bcnk,adml->abcdijkl", H.aa.oooo, T.aa, T.ab, optimize=True)   # (k/ij)(a/bc) = 9
    dT.aaab -= (18.0 / 36.0) * np.einsum("mdjf,abim,cfkl->abcdijkl", H.ab.ovov, T.aa, T.ab, optimize=True)   # (ijk)(c/ab) = (i/jk)(c/ab)(jk) = 18
    dT.aaab -= np.einsum("amie,bejl,cdkm->abcdijkl", H.ab.voov, T.ab, T.ab, optimize=True)   # (ijk)(abc) = (i/jk)(a/bc)(jk)(bc) = 36
    dT.aaab += (18.0 / 36.0) * np.einsum("mnjl,bcmk,adin->abcdijkl", H.ab.oooo, T.aa, T.ab, optimize=True)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    dT.aaab -= (18.0 / 36.0) * np.einsum("bmel,ecjk,adim->abcdijkl", H.ab.vovo, T.aa, T.ab, optimize=True)   # (i/jk)(abc) = (i/jk)(a/bc)(bc) = 18
    dT.aaab -= (18.0 / 36.0) * np.einsum("amie,ecjk,bdml->abcdijkl", H.aa.voov, T.aa, T.ab, optimize=True)   # (i/kj)(abc) = (i/kj)(a/bc)(bc) = 18
    dT.aaab += (9.0 / 36.0) * np.einsum("abef,fcjk,edil->abcdijkl", H.aa.vvvv, T.aa, T.ab, optimize=True)   # (i/jk)(c/ab) = (i/jk)(c/ab) = 9
    dT.aaab -= (18.0 / 36.0) * np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.aa, T.ab, optimize=True)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    dT.aaab += (18.0 / 36.0) * np.einsum("adef,ebij,cfkl->abcdijkl", H.ab.vvvv, T.aa, T.ab, optimize=True)  # (k/ij)(abc) = (k/ij)(a/bc)(bc) = 18

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    dT.aaab -= np.einsum("mdkl,abcijm->abcdijkl", H.ab.ovoo, T.aaa, optimize=True)
    dT.aaab -= np.einsum("amik,bcdjml->abcdijkl", H.aa.vooo, T.aab, optimize=True)
    dT.aaab -= np.einsum("amil,bcdjkm->abcdijkl", H.ab.vooo, T.aab, optimize=True)

    dT.aaab += np.einsum("cdel,abeijk->abcdijkl", H.ab.vvvo, T.aaa, optimize=True)
    dT.aaab += np.einsum("acie,bedjkl->abcdijkl", H.aa.vvov, T.aab, optimize=True)
    dT.aaab += np.einsum("adie,bcejkl->abcdijkl", H.ab.vvov, T.aab, optimize=True)

    I3B_oovooo = (
                    np.einsum("mnie,edjl->mndijl", H.aa.ooov, T.ab, optimize=True)
                   +0.25 * np.einsum("mnef,efdijl->mndijl", H.aa.oovv, T.aab, optimize=True)
    )
    I3B_oovooo -= np.transpose(I3B_oovooo, (0, 1, 2, 4, 3, 5))
    dT.aaab += 0.5 * np.einsum("mndijl,abcmnk->abcdijkl", I3B_oovooo, T.aaa, optimize=True)

    I3A_vooooo = np.einsum("mnie,delj->dmnlij", H.aa.ooov, T.aa, optimize=True)
    I3A_vooooo -= np.transpose(I3A_vooooo, (0, 1, 2, 4, 3, 5)) + np.transpose(I3A_vooooo, (0, 1, 2, 5, 4, 3))
    I3A_vooooo += 0.5 * np.einsum("mnef,efdijl->dmnlij", H.aa.oovv, T.aaa, optimize=True)
    dT.aaab += 0.5 * np.einsum("cmnkij,abdmnl->abcdijkl", I3A_vooooo, T.aab, optimize=True)

    I3B_vooooo = (
                    0.5 * np.einsum("mnel,aeik->amnikl", H.ab.oovo, T.aa, optimize=True)
                  + np.einsum("mnke,aeil->amnikl", H.ab.ooov, T.ab, optimize=True)
                  + 0.5 * np.einsum("mnef,aefikl->amnikl", H.ab.oovv, T.aab, optimize=True)
    )
    I3B_vooooo -= np.transpose(I3B_vooooo, (0, 1, 2, 4, 3, 5))
    dT.aaab += np.einsum("amnikl,bcdjmn->abcdijkl", I3B_vooooo, T.aab, optimize=True)

    I3B_vvvvvo = -np.einsum("amef,bdml->abdefl", H.aa.vovv, T.ab, optimize=True)
    I3B_vvvvvo -= np.transpose(I3B_vvvvvo, (1, 0, 2, 3, 4, 5))
    dT.aaab += 0.5 * np.einsum("abdefl,efcijk->abcdijkl", I3B_vvvvvo, T.aaa, optimize=True)

    I3A_vvvvvo = -np.einsum("amef,bcmk->abcefk", H.aa.vovv, T.aa, optimize=True)
    I3A_vvvvvo -= np.transpose(I3A_vvvvvo, (1, 0, 2, 3, 4, 5)) + np.transpose(I3A_vvvvvo, (2, 1, 0, 3, 4, 5))
    dT.aaab += 0.5 * np.einsum("abcefk,efdijl->abcdijkl", I3A_vvvvvo, T.aab, optimize=True)

    I3B_vvvovv = (
                    -0.5 * np.einsum("mdef,acim->acdief", H.ab.ovvv, T.aa, optimize=True)
                    - np.einsum("cmef,adim->acdief", H.ab.vovv, T.ab, optimize=True)
    )
    I3B_vvvovv -= np.transpose(I3B_vvvovv, (1, 0, 2, 3, 4, 5))
    dT.aaab += np.einsum("acdief,befjkl->abcdijkl", I3B_vvvovv, T.aab, optimize=True)

    I3B_vovovo = (
                    -np.einsum("nmie,adnl->amdiel", H.aa.ooov, T.ab, optimize=True)
                    +np.einsum("amfe,fdil->amdiel", H.aa.vovv, T.ab, optimize=True)
                    -np.einsum("mnel,adin->amdiel", H.ab.oovo, T.ab, optimize=True)
                    +np.einsum("mdef,afil->amdiel", H.ab.ovvv, T.ab, optimize=True)
                    +np.einsum("mnef,afdinl->amdiel", H.aa.ooov, T.aab, optimize=True)
                    +np.einsum("mnef,afdinl->amdiel", H.ab.oovv, T.abb, optimize=True)
    )
    dT.aaab += np.einsum("amdiel,bcejkm->abcdijkl", I3B_vovovo, T.aaa, optimize=True)

    I3A_vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.aa.ooov, T.aa, optimize=True)
                +0.5 * np.einsum("bmfe,afij->abmije", H.aa.vovv, T.aa, optimize=True)
                +0.25 * np.einsum("mnef,abfijn->abmije", H.aa.oovv, T.aaa, optimize=True)
                +0.25 * np.einsum("mnef,abfijn->abmije", H.ab.oovv, T.aab, optimize=True)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 4))
    dT.aaab += np.einsum("abmije,cedkml->abcdijkl", I3A_vvooov, T.aab, optimize=True)

    I3B_vvoovo = (
                -0.5 * np.einsum("nmel,acin->acmiel", H.ab.oovo, T.aa, optimize=True)
                + np.einsum("cmef,afil->acmiel", H.ab.vovv, T.ab, optimize=True)
                - 0.5 * np.einsum("nmef,acfinl->acmiel", H.ab.oovv, T.aab, optimize=True)
    )
    I3B_vvoovo -= np.transpose(I3B_vvoovo, (1, 0, 2, 3, 4, 5))
    dT.aaab -= np.einsum("acmiel,ebdkjm->abcdijkl", I3B_vvoovo, T.aab, optimize=True)

    I3B_vovoov = (
                0.5 * np.einsum("mdfe,afik->amdike", H.ab.ovvv, T.aa, optimize=True)
                - np.einsum("mnke,adin->amdike", H.ab.ooov, T.ab, optimize=True)
    )
    I3B_vovoov -= np.transpose(I3B_vovoov, (0, 1, 2, 4, 3, 5))
    dT.aaab -= np.einsum("amdike,bcejml->abcdijkl", I3B_vovoov, T.aab, optimize=True)

    I3C_vvooov = (
                -np.einsum("nmie,adnl->admile", H.ab.ooov, T.ab, optimize=True)
                -np.einsum("nmle,adin->admile", H.bb.ooov, T.ab, optimize=True)
                +np.einsum("amfe,fdil->admile", H.ab.vovv, T.ab, optimize=True)
                +np.einsum("dmfe,afil->admile", H.bb.vovv, T.ab, optimize=True)
    )
    dT.aaab += np.einsum("admile,bcejkm->abcdijkl", I3C_vvooov, T.aab, optimize=True)

    I3B_vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.ab.ooov, T.aa, optimize=True)
                +0.5 * np.einsum("bmfe,afij->abmije", H.ab.vovv, T.aa, optimize=True)
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    dT.aaab += np.einsum("abmije,cdeklm->abcdijkl", I3B_vvooov, T.abb, optimize=True)

    # <ijklabcd | (H(2)*T4)_C | 0 >


    T.aaab, dT.aaab = cc_loops_t4.cc_loops_t4.update_t4b(
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
    dT.aabb = -np.einsum("cmke,adim,bejl->abcdijkl", H.bb.voov, T.ab, T.ab, optimize=True)   # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.ab, T.ab, optimize=True)   # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= 0.5 * np.einsum("mcek,aeij,bdml->abcdijkl", H.ab.ovvo, T.aa, T.ab, optimize=True)   # (kl)(ab)(cd) = 8
    dT.aabb -= 0.5 * np.einsum("amie,bdjm,cekl->abcdijkl", H.ab.voov, T.ab, T.bb, optimize=True)   # (ij)(ab)(cd) = 8
    dT.aabb -= 0.5 * np.einsum("mcek,abim,edjl->abcdijkl", H.ab.ovvo, T.aa, T.ab, optimize=True)   # (ij)(kl)(cd) = 8
    dT.aabb -= 0.5 * np.einsum("amie,cdkm,bejl->abcdijkl", H.ab.voov, T.bb, T.ab, optimize=True)   # (ij)(kl)(ab) = 8
    dT.aabb -= np.einsum("bmel,adim,ecjk->abcdijkl", H.ab.vovo, T.ab, T.ab, optimize=True)   # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= np.einsum("mdje,bcmk,aeil->abcdijkl", H.ab.ovov, T.ab, T.ab, optimize=True)   # (ij)(kl)(ab)(cd) = 16
    dT.aabb -= 0.25 * np.einsum("mdje,abim,cekl->abcdijkl", H.ab.ovov, T.aa, T.bb, optimize=True)   # (ij)(cd) = 4
    dT.aabb -= 0.25 * np.einsum("bmel,cdkm,aeij->abcdijkl", H.ab.vovo, T.bb, T.aa, optimize=True)  # (kl)(ab) = 4
    dT.aabb += 0.25 * np.einsum("mnij,acmk,bdnl->abcdijkl", H.aa.oooo, T.ab, T.ab, optimize=True)   # (kl)(ab) = 4 !!! (tricky asym)
    dT.aabb += 0.25 * np.einsum("abef,ecik,fdjl->abcdijkl", H.aa.vvvv, T.ab, T.ab, optimize=True)   # (ij)(kl) = 4 !!! (tricky asym)
    dT.aabb += 0.25 * np.einsum("mnik,abmj,cdnl->abcdijkl", H.ab.oooo, T.aa, T.bb, optimize=True)   # (ij)(kl) = 4
    dT.aabb += 0.25 * np.einsum("acef,ebij,fdkl->abcdijkl", H.ab.vvvv, T.aa, T.bb, optimize=True)   # (ab)(cd) = 4
    dT.aabb += np.einsum("mnik,adml,bcjn->abcdijkl", H.ab.oooo, T.ab, T.ab, optimize=True)   # (ij)(kl)(ab)(cd) = 16
    dT.aabb += np.einsum("acef,edil,bfjk->abcdijkl", H.ab.vvvv, T.ab, T.ab, optimize=True)   # (ij)(kl)(ab)(cd) = 16
    dT.aabb += 0.25 * np.einsum("mnkl,adin,bcjm->abcdijkl", H.bb.oooo, T.ab, T.ab, optimize=True)   # (ij)(cd) = 4 !!! (tricky asym)
    dT.aabb += 0.25 * np.einsum("cdef,afil,bejk->abcdijkl", H.bb.vvvv, T.ab, T.ab, optimize=True)   # (ij)(kl) = 4 !!! (tricky asym)

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >

    # <ijklabcd | (H(2)*T4)_C | 0 >

    T.aabb, dT.aabb = cc_loops_t4.cc_loops_t4.update_t4c(
        T.aabb,
        dT.aabb,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift,
    )
    return T, dT
