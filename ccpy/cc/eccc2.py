"""Module with functions that help perform the externally corrected (ec)
CC (ec-CC) computations that solve for T1 and T2 in the presence of T3
and T4 extracted from an external non-CC source."""

import numpy as np
from ccpy.utilities.updates import cc_loops2

def update(T, dT, H, X, shift, flag_RHF, system, T_ext, VT_ext):

    # update T1
    T, dT = update_t1a(T, dT, H, VT_ext, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, VT_ext, shift)

    # CCS intermediates
    hbar = get_ccs_intermediates_opt(T, H)

    # update T2
    T, dT = update_t2a(T, dT, hbar, H, VT_ext, shift, T_ext)
    T, dT = update_t2b(T, dT, hbar, H, VT_ext, shift, T_ext)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, hbar, H, VT_ext, shift, T_ext)

    return T, dT


def update_t1a(T, dT, H, VT_ext, shift):
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

    T.a, dT.a = cc_loops2.cc_loops2.update_t1a(
        T.a,
        dT.a + H.a.vo + VT_ext.a.vo,
        H.a.oo,
        H.a.vv,
        shift,
    )
    return T, dT


def update_t1b(T, dT, H, VT_ext, shift):
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

    T.b, dT.b = cc_loops2.cc_loops2.update_t1b(
        T.b,
        dT.b + H.b.vo + VT_ext.b.vo,
        H.b.oo,
        H.b.vv,
        shift,
    )
    return T, dT


# @profile
def update_t2a(T, dT, H, H0, VT_ext, shift, T_ext):
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
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.a.ov, T_ext.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T_ext.aab, optimize=True)
    dT.aa -= 0.5 * np.einsum("mnif,abfmjn->abij", H0.ab.ooov + H.ab.ooov, T_ext.aab, optimize=True)
    dT.aa -= 0.25 * np.einsum("mnif,abfmjn->abij", H0.aa.ooov + H.aa.ooov, T_ext.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("anef,ebfijn->abij", H0.aa.vovv + H.aa.vovv, T_ext.aaa, optimize=True)
    dT.aa += 0.5 * np.einsum("anef,ebfijn->abij", H0.ab.vovv + H.ab.vovv, T_ext.aab, optimize=True)

    T.aa, dT.aa = cc_loops2.cc_loops2.update_t2a(
        T.aa,
        dT.aa + 0.25 * (H0.aa.vvoo + VT_ext.aa.vvoo),
        H0.a.oo,
        H0.a.vv,
        shift
    )
    return T, dT


# @profile
def update_t2b(T, dT, H, H0, VT_ext, shift, T_ext):
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
    dT.ab -= 0.5 * np.einsum("mnif,afbmnj->abij", H0.aa.ooov + H.aa.ooov, T_ext.aab, optimize=True)
    dT.ab -= np.einsum("nmfj,afbinm->abij", H0.ab.oovo + H.ab.oovo, T_ext.aab, optimize=True)
    dT.ab -= 0.5 * np.einsum("mnjf,afbinm->abij", H0.bb.ooov + H.bb.ooov, T_ext.abb, optimize=True)
    dT.ab -= np.einsum("mnif,afbmnj->abij", H0.ab.ooov + H.ab.ooov, T_ext.abb, optimize=True)
    dT.ab += 0.5 * np.einsum("anef,efbinj->abij", H0.aa.vovv + H.aa.vovv, T_ext.aab, optimize=True)
    dT.ab += np.einsum("anef,efbinj->abij", H0.ab.vovv + H.ab.vovv, T_ext.abb, optimize=True)
    dT.ab += np.einsum("nbfe,afeinj->abij", H0.ab.ovvv + H.ab.ovvv, T_ext.aab, optimize=True)
    dT.ab += 0.5 * np.einsum("bnef,afeinj->abij", H0.bb.vovv + H.bb.vovv, T_ext.abb, optimize=True)
    dT.ab += np.einsum("me,aebimj->abij", H.a.ov, T_ext.aab, optimize=True)
    dT.ab += np.einsum("me,aebimj->abij", H.b.ov, T_ext.abb, optimize=True)

    T.ab, dT.ab = cc_loops2.cc_loops2.update_t2b(
        T.ab,
        dT.ab + H0.ab.vvoo + VT_ext.ab.vvoo,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift
    )
    return T, dT


# @profile
def update_t2c(T, dT, H, H0, VT_ext, shift, T_ext):
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
    dT.bb += 0.25 * np.einsum("me,eabmij->abij", H.a.ov, T_ext.abb, optimize=True)
    dT.bb += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T_ext.bbb, optimize=True)
    dT.bb += 0.25 * np.einsum("anef,ebfijn->abij", H0.bb.vovv + H.bb.vovv, T_ext.bbb, optimize=True)
    dT.bb += 0.5 * np.einsum("nafe,febnij->abij", H0.ab.ovvv + H.ab.ovvv, T_ext.abb, optimize=True)
    dT.bb -= 0.25 * np.einsum("mnif,abfmjn->abij", H0.bb.ooov + H.bb.ooov, T_ext.bbb, optimize=True)
    dT.bb -= 0.5 * np.einsum("nmfi,fabnmj->abij", H0.ab.oovo + H.ab.oovo, T_ext.abb, optimize=True)

    T.bb, dT.bb = cc_loops2.cc_loops2.update_t2c(
        T.bb,
        dT.bb + 0.25 * (H0.bb.vvoo + VT_ext.bb.vvoo),
        H0.b.oo,
        H0.b.vv,
        shift
    )
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