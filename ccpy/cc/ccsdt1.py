"""Module with functions that perform the CC with singles, doubles,
and triples (CCSDT) calculation for a molecular system."""

import numpy as np

from ccpy.hbar.hbar_ccs import get_ccs_intermediates
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.utilities.updates import cc_loops2

# [TODO]: Add in optional arguments for active-space slicing vectors. Make sure this is compatible with
# the standard CC solvers!
def update(T, dT, H, shift, flag_RHF):
    # update T1
    T, dT = update_t1a(T, dT, H, shift)
    if flag_RHF:
        # TODO: Maybe copy isn't needed. Reference should suffice
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, shift)

    # CCS intermediates
    hbar = get_ccs_intermediates(T, H)

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

    # T, update_t1a_loop(T, X1A, fA_oo, fA_vv, shift)
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


def update_t2a(T, dT, H, H0, shift):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2+T3))_C|0>.
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

    dT.aa = -0.5 * np.einsum("amij,bm->abij", H.aa.vooo, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("abie,ej->abij", H.aa.vvov, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", I1A_vv, T.aa, optimize=True)
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", I1A_oo, T.aa, optimize=True)
    dT.aa += np.einsum("amie,ebmj->abij", I2A_voov, T.aa, optimize=True)
    dT.aa += np.einsum("amie,bejm->abij", I2B_voov, T.ab, optimize=True)
    dT.aa += 0.125 * np.einsum("abef,efij->abij", H.aa.vvvv, T.aa, optimize=True)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", I2A_oooo, T.aa, optimize=True)
    # T3 parts
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.a.ov, T.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T.aab, optimize=True)
    dT.aa -= 0.5 * np.einsum("mnif,abfmjn->abij", H.ab.ooov, T.aab, optimize=True)
    dT.aa -= 0.25 * np.einsum("mnif,abfmjn->abij", H.aa.ooov, T.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("anef,ebfijn->abij", H.aa.vovv, T.aaa, optimize=True)
    dT.aa += 0.5 * np.einsum("anef,ebfijn->abij", H.ab.vovv, T.aab, optimize=True)

    T.aa, dT.aa = cc_loops2.cc_loops2.update_t2a(
        T.aa,
        dT.aa + 0.25 * H0.aa.vvoo,
        H0.a.oo,
        H0.a.vv,
        shift,
    )
    return T, dT


def update_t2b(T, dT, H, H0, shift):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2+T3))_C|0>.
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

    dT.ab = -np.einsum("mbij,am->abij", H.ab.ovoo, T.a, optimize=True)
    dT.ab -= np.einsum("amij,bm->abij", H.ab.vooo, T.b, optimize=True)
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
    dT.ab += np.einsum("abef,efij->abij", H.ab.vvvv, T.ab, optimize=True)
    # T3 parts
    dT.ab -= 0.5 * np.einsum("mnif,afbmnj->abij", H.aa.ooov, T.aab, optimize=True)
    dT.ab -= np.einsum("nmfj,afbinm->abij", H.ab.oovo, T.aab, optimize=True)
    dT.ab -= 0.5 * np.einsum("mnjf,afbinm->abij", H.bb.ooov, T.abb, optimize=True)
    dT.ab -= np.einsum("mnif,afbmnj->abij", H.ab.ooov, T.abb, optimize=True)
    dT.ab += 0.5 * np.einsum("anef,efbinj->abij", H.aa.vovv, T.aab, optimize=True)
    dT.ab += np.einsum("anef,efbinj->abij", H.ab.vovv, T.abb, optimize=True)
    dT.ab += np.einsum("nbfe,afeinj->abij", H.ab.ovvv, T.aab, optimize=True)
    dT.ab += 0.5 * np.einsum("bnef,afeinj->abij", H.bb.vovv, T.abb, optimize=True)
    dT.ab += np.einsum("me,aebimj->abij", H.a.ov, T.aab, optimize=True)
    dT.ab += np.einsum("me,aebimj->abij", H.b.ov, T.abb, optimize=True)

    T.ab, dT.ab = cc_loops2.cc_loops2.update_t2b(
        T.ab,
        dT.ab + H0.ab.vvoo,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        shift,
    )

    return T, dT


def update_t2c(T, dT, H, H0, shift):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2+T3))_C|0>.
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

    dT.bb = -0.5 * np.einsum("amij,bm->abij", H.bb.vooo, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("abie,ej->abij", H.bb.vvov, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("ae,ebij->abij", I1B_vv, T.bb, optimize=True)
    dT.bb -= 0.5 * np.einsum("mi,abmj->abij", I1B_oo, T.bb, optimize=True)
    dT.bb += np.einsum("amie,ebmj->abij", I2C_voov, T.bb, optimize=True)
    dT.bb += np.einsum("maei,ebmj->abij", I2B_ovvo, T.ab, optimize=True)
    dT.bb += 0.125 * np.einsum("abef,efij->abij", H.bb.vvvv, T.bb, optimize=True)
    dT.bb += 0.125 * np.einsum("mnij,abmn->abij", I2C_oooo, T.bb, optimize=True)
    # T3 parts
    dT.bb += 0.25 * np.einsum("me,eabmij->abij", H.a.ov, T.abb, optimize=True)
    dT.bb += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T.bbb, optimize=True)
    dT.bb += 0.25 * np.einsum("anef,ebfijn->abij", H.bb.vovv, T.bbb, optimize=True)
    dT.bb += 0.5 * np.einsum("nafe,febnij->abij", H.ab.ovvv, T.abb, optimize=True)
    dT.bb -= 0.25 * np.einsum("mnif,abfmjn->abij", H.bb.ooov, T.bbb, optimize=True)
    dT.bb -= 0.5 * np.einsum("nmfi,fabnmj->abij", H.ab.oovo, T.abb, optimize=True)

    T.bb, dT.bb = cc_loops2.cc_loops2.update_t2c(
        T.bb,
        dT.bb + 0.25 * H0.bb.vvoo,
        H0.b.oo,
        H0.b.vv,
        shift,
    )

    return T, dT


# @profile
def update_t3a(T, dT, H, H0, shift, oa, Oa, ob, Ob, va, Va, vb, Vb):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3))_C|0>.
    """
    # active space slicing

    # <ijkabc | H(2) | 0 > + (VT3)_C intermediates

    # (HBar*T3)_C

    # PROJECTION 1: VVVOOO
    dT.aaa.VVVOOO += (3.0 / 36.0) * (
            +1.0 * np.einsum('mI,CBAmJK->ABCIJK', H.a.oo[oa, Oa], T.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('MI,CBAMJK->ABCIJK', H.a.oo[Oa, Oa], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVOOO += (3.0 / 36.0) * (
            -1.0 * np.einsum('Ae,CBeIJK->ABCIJK', H.a.vv[Va, va], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AE,CBEIJK->ABCIJK', H.a.vv[Va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('mnIJ,CBAmnK->ABCIJK', H.aa.oooo[oa, oa, Oa, Oa], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,CBAnMK->ABCIJK', H.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,CBAMNK->ABCIJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('ABef,CfeIJK->ABCIJK', H.aa.vvvv[Va, Va, va, va], T.aaa.VvvOOO, optimize=True)
            + 1.0 * np.einsum('ABEf,CEfIJK->ABCIJK', H.aa.vvvv[Va, Va, Va, va], T.aaa.VVvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEIJK->ABCIJK', H.aa.vvvv[Va, Va, Va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('AmIe,CBemJK->ABCIJK', H.aa.voov[Va, oa, Oa, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEmJK->ABCIJK', H.aa.voov[Va, oa, Oa, Va], T.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMIe,CBeMJK->ABCIJK', H.aa.voov[Va, Oa, Oa, va], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEMJK->ABCIJK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVOOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('AmIe,CBeJKm->ABCIJK', H.ab.voov[Va, ob, Oa, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEJKm->ABCIJK', H.ab.voov[Va, ob, Oa, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('AMIe,CBeJKM->ABCIJK', H.ab.voov[Va, Ob, Oa, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEJKM->ABCIJK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    # PROJECTION 2: VVvOOO
    dT.aaa.VVvOOO += (3.0 / 36.0) * (
            +1.0 * np.einsum('mI,BAcmJK->ABcIJK', H.a.oo[oa, Oa], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('MI,BAcMJK->ABcIJK', H.a.oo[Oa, Oa], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (2.0 / 36.0) * (
            +1.0 * np.einsum('Ae,BceIJK->ABcIJK', H.a.vv[Va, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AE,BEcIJK->ABcIJK', H.a.vv[Va, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (1.0 / 36.0) * (
            +1.0 * np.einsum('ce,ABeIJK->ABcIJK', H.a.vv[va, va], T.aaa.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cE,ABEIJK->ABcIJK', H.a.vv[va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('mnIJ,BAcmnK->ABcIJK', H.aa.oooo[oa, oa, Oa, Oa], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,BAcnMK->ABcIJK', H.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,BAcMNK->ABcIJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (1.0 / 36.0) * (
            -1.0 * np.einsum('ABEf,EcfIJK->ABcIJK', H.aa.vvvv[Va, Va, Va, va], T.aaa.VvvOOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcIJK->ABcIJK', H.aa.vvvv[Va, Va, Va, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (2.0 / 36.0) * (
            +0.5 * np.einsum('Acef,AfeIJK->ABcIJK', H.aa.vvvv[Va, va, va, va], T.aaa.VvvOOO, optimize=True)
            + 0.5 * np.einsum('acEf,EafIJK->ABcIJK', H.aa.vvvv[va, va, Va, va], T.aaa.VvvOOO, optimize=True)
            + 0.5 * np.einsum('acEF,FEaIJK->ABcIJK', H.aa.vvvv[va, va, Va, Va], T.aaa.VVvOOO, optimize=True)
            - 0.5 * np.einsum('AcEf,AEfIJK->ABcIJK', H.aa.vvvv[Va, va, Va, va], T.aaa.VVvOOO, optimize=True)
            + 0.5 * np.einsum('AcEF,AFEIJK->ABcIJK', H.aa.vvvv[Va, va, Va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (6.0 / 36.0) * (
            +1.0 * np.einsum('AmIe,BcemJK->ABcIJK', H.aa.voov[Va, oa, Oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmIE,BEcmJK->ABcIJK', H.aa.voov[Va, oa, Oa, Va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMIe,BceMJK->ABcIJK', H.aa.voov[Va, Oa, Oa, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,BEcMJK->ABcIJK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (3.0 / 36.0) * (
            +1.0 * np.einsum('cmIe,ABemJK->ABcIJK', H.aa.voov[va, oa, Oa, va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmIE,ABEmJK->ABcIJK', H.aa.voov[va, oa, Oa, Va], T.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('cMIe,ABeMJK->ABcIJK', H.aa.voov[va, Oa, Oa, va], T.aaa.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cMIE,ABEMJK->ABcIJK', H.aa.voov[va, Oa, Oa, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (6.0 / 36.0) * (
            +1.0 * np.einsum('AmIe,BceJKm->ABcIJK', H.ab.voov[Va, ob, Oa, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AmIE,BcEJKm->ABcIJK', H.ab.voov[Va, ob, Oa, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('AMIe,BceJKM->ABcIJK', H.ab.voov[Va, Ob, Oa, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AMIE,BcEJKM->ABcIJK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.aaa.VVvOOO += (3.0 / 36.0) * (
            +1.0 * np.einsum('cmIe,ABeJKm->ABcIJK', H.ab.voov[va, ob, Oa, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('cmIE,ABEJKm->ABcIJK', H.ab.voov[va, ob, Oa, Vb], T.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('cMIe,ABeJKM->ABcIJK', H.ab.voov[va, Ob, Oa, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cMIE,ABEJKM->ABcIJK', H.ab.voov[va, Ob, Oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    # PROJECTION 3:
    dT.aaa.VVVoOO += (1.0 / 36.0) * (
            +1.0 * np.einsum('mi,CBAmJK->ABCiJK', H.a.oo[oa, oa], T.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('Mi,CBAMJK->ABCiJK', H.a.oo[Oa, oa], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVoOO += (2.0 / 36.0) * (
            -1.0 * np.einsum('mJ,CBAmiK->ABCiJK', H.a.oo[oa, Oa], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('MJ,CBAiMK->ABCiJK', H.a.oo[Oa, Oa], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVoOO += (3.0 / 36.0) * (
            -1.0 * np.einsum('Ae,CBeiJK->ABCiJK', H.a.vv[Va, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AE,CBEiJK->ABCiJK', H.a.vv[Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVoOO += (2.0 / 36.0) * (
            -0.5 * np.einsum('mniJ,CBAmnK->ABCiJK', H.aa.oooo[oa, oa, oa, Oa], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('MniJ,CBAnMK->ABCiJK', H.aa.oooo[Oa, oa, oa, Oa], T.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,CBAMNK->ABCiJK', H.aa.oooo[Oa, Oa, oa, Oa], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVoOO += (1.0 / 36.0) * (
            +1.0 * np.einsum('MnKJ,CBAniM->ABCiJK', H.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,CBAiMN->ABCiJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVoOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('ABef,CfeiJK->ABCiJK', H.aa.vvvv[Va, Va, va, va], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('ABEf,CEfiJK->ABCiJK', H.aa.vvvv[Va, Va, Va, va], T.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEiJK->ABCiJK', H.aa.vvvv[Va, Va, Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVoOO += (3.0 / 36.0) * (
            -1.0 * np.einsum('Amie,CBemJK->ABCiJK', H.aa.voov[Va, oa, oa, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmJK->ABCiJK', H.aa.voov[Va, oa, oa, Va], T.aaa.VVVoOO, optimize=True)
            - 1.0 * np.einsum('AMie,CBeMJK->ABCiJK', H.aa.voov[Va, Oa, oa, va], T.aaa.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,CBEMJK->ABCiJK', H.aa.voov[Va, Oa, oa, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVoOO += (6.0 / 36.0) * (
            +1.0 * np.einsum('AmJe,CBemiK->ABCiJK', H.aa.voov[Va, oa, Oa, va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('AmJE,CBEmiK->ABCiJK', H.aa.voov[Va, oa, Oa, Va], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('AMJe,CBeiMK->ABCiJK', H.aa.voov[Va, Oa, Oa, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,CBEiMK->ABCiJK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVoOO += (3.0 / 36.0) * (
            -1.0 * np.einsum('Amie,CBeJKm->ABCiJK', H.ab.voov[Va, ob, oa, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEJKm->ABCiJK', H.ab.voov[Va, ob, oa, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('AMie,CBeJKM->ABCiJK', H.ab.voov[Va, Ob, oa, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,CBEJKM->ABCiJK', H.ab.voov[Va, Ob, oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dT.aaa.VVVoOO += (6.0 / 36.0) * (
            +1.0 * np.einsum('AmJe,CBeiKm->ABCiJK', H.ab.voov[Va, ob, Oa, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('AmJE,CBEiKm->ABCiJK', H.ab.voov[Va, ob, Oa, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('AMJe,CBeiKM->ABCiJK', H.ab.voov[Va, Ob, Oa, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMJE,CBEiKM->ABCiJK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.VVVoOO, optimize=True)
    )
    # PROJECTION 4:
    dT.aaa.VVvoOO += (1.0 / 36.0) * (
            +1.0 * np.einsum('mi,BAcmJK->ABciJK', H.a.oo[oa, oa], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('Mi,BAcMJK->ABciJK', H.a.oo[Oa, oa], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 36.0) * (
            -1.0 * np.einsum('mJ,BAcmiK->ABciJK', H.a.oo[oa, Oa], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MJ,BAciMK->ABciJK', H.a.oo[Oa, Oa], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (3.0 / 36.0) * (
            -1.0 * np.einsum('Ae,CBeiJK->ABciJK', H.a.vv[Va, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AE,CBEiJK->ABciJK', H.a.vv[Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 36.0) * (
            -1.0 * np.einsum('ce,BaeiJK->ABciJK', H.a.vv[va, va], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('cE,BEaiJK->ABciJK', H.a.vv[va, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 36.0) * (
            -0.5 * np.einsum('mniJ,BAcmnK->ABciJK', H.aa.oooo[oa, oa, oa, Oa], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('MniJ,BAcnMK->ABciJK', H.aa.oooo[Oa, oa, oa, Oa], T.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,BAcMNK->ABciJK', H.aa.oooo[Oa, Oa, oa, Oa], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvoOO += (1.0 / 36.0) * (
            +1.0 * np.einsum('MnKJ,BAcniM->ABciJK', H.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,BAciMN->ABciJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (1.0 / 36.0) * (
            -1.0 * np.einsum('ABEf,EcfiJK->ABciJK', H.aa.vvvv[Va, Va, Va, va], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEciJK->ABciJK', H.aa.vvvv[Va, Va, Va, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 36.0) * (
            +0.5 * np.einsum('Acef,BfeiJK->ABciJK', H.aa.vvvv[Va, va, va, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AcEf,BEfiJK->ABciJK', H.aa.vvvv[Va, va, Va, va], T.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('AcEF,BFEiJK->ABciJK', H.aa.vvvv[Va, va, Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 36.0) * (
            +1.0 * np.einsum('Amie,BcemJK->ABciJK', H.aa.voov[Va, oa, oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmJK->ABciJK', H.aa.voov[Va, oa, oa, Va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMie,BceMJK->ABciJK', H.aa.voov[Va, Oa, oa, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('AMiE,BEcMJK->ABciJK', H.aa.voov[Va, Oa, oa, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvoOO += (4.0 / 36.0) * (
            -1.0 * np.einsum('AmJe,BcemiK->ABciJK', H.aa.voov[Va, oa, Oa, va], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('AmJE,BEcmiK->ABciJK', H.aa.voov[Va, oa, Oa, Va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMJe,BceiMK->ABciJK', H.aa.voov[Va, Oa, Oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,BEciMK->ABciJK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (1.0 / 36.0) * (
            +1.0 * np.einsum('cmie,ABemJK->ABciJK', H.aa.voov[va, oa, oa, va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEmJK->ABciJK', H.aa.voov[va, oa, oa, Va], T.aaa.VVVoOO, optimize=True)
            + 1.0 * np.einsum('cMie,ABeMJK->ABciJK', H.aa.voov[va, Oa, oa, va], T.aaa.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cMiE,ABEMJK->ABciJK', H.aa.voov[va, Oa, oa, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 36.0) * (
            -1.0 * np.einsum('cmJe,ABemiK->ABciJK', H.aa.voov[va, oa, Oa, va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('cmJE,ABEmiK->ABciJK', H.aa.voov[va, oa, Oa, Va], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('cMJe,ABeiMK->ABciJK', H.aa.voov[va, Oa, Oa, va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMJE,ABEiMK->ABciJK', H.aa.voov[va, Oa, Oa, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 36.0) * (
            +1.0 * np.einsum('Amie,BceJKm->ABciJK', H.ab.voov[Va, ob, oa, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('AmiE,BcEJKm->ABciJK', H.ab.voov[Va, ob, oa, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('AMie,BceJKM->ABciJK', H.ab.voov[Va, Ob, oa, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('AMiE,BcEJKM->ABciJK', H.ab.voov[Va, Ob, oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.aaa.VVvoOO += (4.0 / 36.0) * (
            -1.0 * np.einsum('AmJe,BceiKm->ABciJK', H.ab.voov[Va, ob, Oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('AmJE,BcEiKm->ABciJK', H.ab.voov[Va, ob, Oa, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('AMJe,BceiKM->ABciJK', H.ab.voov[Va, Ob, Oa, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMJE,BcEiKM->ABciJK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.VvVoOO, optimize=True)
    )
    dT.aaa.VVvoOO += (1.0 / 36.0) * (
            +1.0 * np.einsum('cmie,ABeJKm->ABciJK', H.ab.voov[va, ob, oa, vb], T.aab.VVvOOo, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEJKm->ABciJK', H.ab.voov[va, ob, oa, Vb], T.aab.VVVOOo, optimize=True)
            + 1.0 * np.einsum('cMie,ABeJKM->ABciJK', H.ab.voov[va, Ob, oa, vb], T.aab.VVvOOO, optimize=True)
            + 1.0 * np.einsum('cMiE,ABEJKM->ABciJK', H.ab.voov[va, Ob, oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dT.aaa.VVvoOO += (2.0 / 36.0) * (
            -1.0 * np.einsum('cmJe,ABeiKm->ABciJK', H.ab.voov[va, ob, Oa, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('cmJE,ABEiKm->ABciJK', H.ab.voov[va, ob, Oa, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('cMJe,ABeiKM->ABciJK', H.ab.voov[va, Ob, Oa, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cMJE,ABEiKM->ABciJK', H.ab.voov[va, Ob, Oa, Vb], T.aab.VVVoOO, optimize=True)
    )
    # PROJECTION 5:
    dT.aaa.VvvOOO += (3.0 / 36.0) * (
            +1.0 * np.einsum('mI,AcbmJK->AbcIJK', H.a.oo[oa, Oa], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('MI,AcbMJK->AbcIJK', H.a.oo[Oa, Oa], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (1.0 / 36.0) * (
            -1.0 * np.einsum('AE,EcbIJK->AbcIJK', H.a.vv[Va, Va], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (2.0 / 36.0) * (
            +1.0 * np.einsum('ce,AbeIJK->AbcIJK', H.a.vv[va, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('cE,AEbIJK->AbcIJK', H.a.vv[va, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (3.0 / 36.0) * (
            -0.5 * np.einsum('mnIJ,AcbmnK->AbcIJK', H.aa.oooo[oa, oa, Oa, Oa], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('MnIJ,AcbnMK->AbcIJK', H.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNIJ,AcbMNK->AbcIJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (2.0 / 36.0) * (
            -1.0 * np.einsum('AbEf,EcfIJK->AbcIJK', H.aa.vvvv[Va, va, Va, va], T.aaa.VvvOOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcIJK->AbcIJK', H.aa.vvvv[Va, va, Va, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (1.0 / 36.0) * (
            +0.5 * np.einsum('cbef,AfeIJK->AbcIJK', H.aa.vvvv[va, va, va, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('cbEf,AEfIJK->AbcIJK', H.aa.vvvv[va, va, Va, va], T.aaa.VVvOOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEIJK->AbcIJK', H.aa.vvvv[va, va, Va, Va], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (3.0 / 36.0) * (
            -1.0 * np.einsum('AmIE,EcbmJK->AbcIJK', H.aa.voov[Va, oa, Oa, Va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMIE,EcbMJK->AbcIJK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (6.0 / 36.0) * (
            +1.0 * np.einsum('cmIe,AbemJK->AbcIJK', H.aa.voov[va, oa, Oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('cmIE,AEbmJK->AbcIJK', H.aa.voov[va, oa, Oa, Va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMIe,AbeMJK->AbcIJK', H.aa.voov[va, Oa, Oa, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('cMIE,AEbMJK->AbcIJK', H.aa.voov[va, Oa, Oa, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (3.0 / 36.0) * (
            -1.0 * np.einsum('AmIE,cbEJKm->AbcIJK', H.ab.voov[Va, ob, Oa, Vb], T.aab.vvVOOo, optimize=True)
            - 1.0 * np.einsum('AMIE,cbEJKM->AbcIJK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.vvVOOO, optimize=True)
    )
    dT.aaa.VvvOOO += (6.0 / 36.0) * (
            +1.0 * np.einsum('cmIe,AbeJKm->AbcIJK', H.ab.voov[va, ob, Oa, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('cmIE,AbEJKm->AbcIJK', H.ab.voov[va, ob, Oa, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('cMIe,AbeJKM->AbcIJK', H.ab.voov[va, Ob, Oa, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('cMIE,AbEJKM->AbcIJK', H.ab.voov[va, Ob, Oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    # PROJECTION 6:
    dT.aaa.VVVooO += (2.0 / 36.0) * (
            +1.0 * np.einsum('mi,CBAmjK->ABCijK', H.a.oo[oa, oa], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('Mi,CBAjMK->ABCijK', H.a.oo[Oa, oa], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVooO += (1.0 / 36.0) * (
            -1.0 * np.einsum('MK,CBAjiM->ABCijK', H.a.oo[Oa, Oa], T.aaa.VVVooO, optimize=True)
    )
    dT.aaa.VVVooO += (3.0 / 36.0) * (
            -1.0 * np.einsum('Ae,CBeijK->ABCijK', H.a.vv[Va, va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('AE,CBEijK->ABCijK', H.a.vv[Va, Va], T.aaa.VVVooO, optimize=True)
    )
    dT.aaa.VVVooO += (1.0 / 36.0) * (
            -0.5 * np.einsum('mnij,CBAmnK->ABCijK', H.aa.oooo[oa, oa, oa, oa], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('Mnij,CBAnMK->ABCijK', H.aa.oooo[Oa, oa, oa, oa], T.aaa.VVVoOO, optimize=True)
            - 0.5 * np.einsum('MNij,CBAMNK->ABCijK', H.aa.oooo[Oa, Oa, oa, oa], T.aaa.VVVOOO, optimize=True)
    )
    dT.aaa.VVVooO += (2.0 / 36.0) * (
            +1.0 * np.einsum('MniK,CBAnjM->ABCijK', H.aa.oooo[Oa, oa, oa, Oa], T.aaa.VVVooO, optimize=True)
            + 0.5 * np.einsum('MNiK,CBAjMN->ABCijK', H.aa.oooo[Oa, Oa, oa, Oa], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVooO += (3.0 / 36.0) * (
            -0.5 * np.einsum('ABef,CfeijK->ABCijK', H.aa.vvvv[Va, Va, va, va], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('ABEf,CEfijK->ABCijK', H.aa.vvvv[Va, Va, Va, va], T.aaa.VVvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,CFEijK->ABCijK', H.aa.vvvv[Va, Va, Va, Va], T.aaa.VVVooO, optimize=True)
    )
    dT.aaa.VVVooO += (6.0 / 36.0) * (
            -1.0 * np.einsum('Amie,CBemjK->ABCijK', H.aa.voov[Va, oa, oa, va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEmjK->ABCijK', H.aa.voov[Va, oa, oa, Va], T.aaa.VVVooO, optimize=True)
            + 1.0 * np.einsum('AMie,CBejMK->ABCijK', H.aa.voov[Va, Oa, oa, va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,CBEjMK->ABCijK', H.aa.voov[Va, Oa, oa, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVVooO += (3.0 / 36.0) * (
            +1.0 * np.einsum('AMKe,CBejiM->ABCijK', H.aa.voov[Va, Oa, Oa, va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMKE,CBEjiM->ABCijK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VVVooO, optimize=True)
    )
    dT.aaa.VVVooO += (6.0 / 36.0) * (
            -1.0 * np.einsum('Amie,CBejKm->ABCijK', H.ab.voov[Va, ob, oa, vb], T.aab.VVvoOo, optimize=True)
            - 1.0 * np.einsum('AmiE,CBEjKm->ABCijK', H.ab.voov[Va, ob, oa, Vb], T.aab.VVVoOo, optimize=True)
            - 1.0 * np.einsum('AMie,CBejKM->ABCijK', H.ab.voov[Va, Ob, oa, vb], T.aab.VVvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,CBEjKM->ABCijK', H.ab.voov[Va, Ob, oa, Vb], T.aab.VVVoOO, optimize=True)
    )
    dT.aaa.VVVooO += (3.0 / 36.0) * (
            +1.0 * np.einsum('AMKe,CBejiM->ABCijK', H.ab.voov[Va, Ob, Oa, vb], T.aab.VVvooO, optimize=True)
            + 1.0 * np.einsum('AMKE,CBEjiM->ABCijK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.VVVooO, optimize=True)
    )
    # PROJECTION 7:
    dT.aaa.VvvoOO += (1.0 / 36.0) * (
            +1.0 * np.einsum('mi,AcbmJK->AbciJK', H.a.oo[oa, oa], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('Mi,AcbMJK->AbciJK', H.a.oo[Oa, oa], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 36.0) * (
            +1.0 * np.einsum('mK,AcbmiJ->AbciJK', H.a.oo[oa, Oa], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('MK,AcbiMJ->AbciJK', H.a.oo[Oa, Oa], T.aaa.VvvoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (1.0 / 36.0) * (
            -1.0 * np.einsum('AE,EcbiJK->AbciJK', H.a.vv[Va, Va], T.aaa.VvvoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 36.0) * (
            +1.0 * np.einsum('ce,AbeiJK->AbciJK', H.a.vv[va, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('cE,AEbiJK->AbciJK', H.a.vv[va, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 36.0) * (
            -0.5 * np.einsum('mniJ,AcbmnK->AbciJK', H.aa.oooo[oa, oa, oa, Oa], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('MniJ,AcbnMK->AbciJK', H.aa.oooo[Oa, oa, oa, Oa], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNiJ,AcbMNK->AbciJK', H.aa.oooo[Oa, Oa, oa, Oa], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvoOO += (1.0 / 36.0) * (
            +1.0 * np.einsum('MnKJ,AcbniM->AbciJK', H.aa.oooo[Oa, oa, Oa, Oa], T.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNKJ,AcbiMN->AbciJK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VvvoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 36.0) * (
            -1.0 * np.einsum('AbEf,EcfiJK->AbciJK', H.aa.vvvv[Va, va, Va, va], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEciJK->AbciJK', H.aa.vvvv[Va, va, Va, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (1.0 / 36.0) * (
            +0.5 * np.einsum('cbef,AfeiJK->AbciJK', H.aa.vvvv[va, va, va, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('cbEf,AEfiJK->AbciJK', H.aa.vvvv[va, va, Va, va], T.aaa.VVvoOO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEiJK->AbciJK', H.aa.vvvv[va, va, Va, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (1.0 / 36.0) * (
            -1.0 * np.einsum('AmiE,EcbmJK->AbciJK', H.aa.voov[Va, oa, oa, Va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('AMiE,EcbMJK->AbciJK', H.aa.voov[Va, Oa, oa, Va], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 36.0) * (
            +1.0 * np.einsum('cmie,AbemJK->AbciJK', H.aa.voov[va, oa, oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('cmiE,AEbmJK->AbciJK', H.aa.voov[va, oa, oa, Va], T.aaa.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMie,AbeMJK->AbciJK', H.aa.voov[va, Oa, oa, va], T.aaa.VvvOOO, optimize=True)
            - 1.0 * np.einsum('cMiE,AEbMJK->AbciJK', H.aa.voov[va, Oa, oa, Va], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 36.0) * (
            +1.0 * np.einsum('AmJE,EcbmiK->AbciJK', H.aa.voov[Va, oa, Oa, Va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMJE,EcbiMK->AbciJK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VvvoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (4.0 / 36.0) * (
            -1.0 * np.einsum('cmJe,AbemiK->AbciJK', H.aa.voov[va, oa, Oa, va], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('cmJE,AEbmiK->AbciJK', H.aa.voov[va, oa, Oa, Va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('cMJe,AbeiMK->AbciJK', H.aa.voov[va, Oa, Oa, va], T.aaa.VvvoOO, optimize=True)
            - 1.0 * np.einsum('cMJE,AEbiMK->AbciJK', H.aa.voov[va, Oa, Oa, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (9.0 / 36.0) * (
            -1.0 * np.einsum('AmIe,CBeJKm->AbciJK', H.ab.voov[Va, ob, Oa, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEJKm->AbciJK', H.ab.voov[Va, ob, Oa, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('AMIe,CBeJKM->AbciJK', H.ab.voov[Va, Ob, Oa, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEJKM->AbciJK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 36.0) * (
            +1.0 * np.einsum('cmie,AbeJKm->AbciJK', H.ab.voov[va, ob, oa, vb], T.aab.VvvOOo, optimize=True)
            + 1.0 * np.einsum('cmiE,AbEJKm->AbciJK', H.ab.voov[va, ob, oa, Vb], T.aab.VvVOOo, optimize=True)
            + 1.0 * np.einsum('cMie,AbeJKM->AbciJK', H.ab.voov[va, Ob, oa, vb], T.aab.VvvOOO, optimize=True)
            + 1.0 * np.einsum('cMiE,AbEJKM->AbciJK', H.ab.voov[va, Ob, oa, Vb], T.aab.VvVOOO, optimize=True)
    )
    dT.aaa.VvvoOO += (2.0 / 36.0) * (
            +1.0 * np.einsum('AmJE,cbEiKm->AbciJK', H.ab.voov[Va, ob, Oa, Vb], T.aab.vvVoOo, optimize=True)
            + 1.0 * np.einsum('AMJE,cbEiKM->AbciJK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.vvVoOO, optimize=True)
    )
    dT.aaa.VvvoOO += (4.0 / 36.0) * (
            -1.0 * np.einsum('cmJe,AbeiKm->AbciJK', H.ab.voov[va, ob, Oa, vb], T.aab.VvvoOo, optimize=True)
            - 1.0 * np.einsum('cmJE,AbEiKm->AbciJK', H.ab.voov[va, ob, Oa, Vb], T.aab.VvVoOo, optimize=True)
            - 1.0 * np.einsum('cMJe,AbeiKM->AbciJK', H.ab.voov[va, Ob, Oa, vb], T.aab.VvvoOO, optimize=True)
            - 1.0 * np.einsum('cMJE,AbEiKM->AbciJK', H.ab.voov[va, Ob, Oa, Vb], T.aab.VvVoOO, optimize=True)
    )
    # PROJECTION 8
    dT.aaa.VVvooO += (2.0 / 36.0) * (
            +1.0 * np.einsum('mi,BAcmjK->ABcijK', H.a.oo[oa, oa], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('Mi,BAcjMK->ABcijK', H.a.oo[Oa, oa], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VVvooO += (1.0 / 36.0) * (
            -1.0 * np.einsum('MK,BAcjiM->ABcijK', H.a.oo[Oa, Oa], T.aaa.VVvooO, optimize=True)
    )
    dT.aaa.VVvooO += (2.0 / 36.0) * (
            +1.0 * np.einsum('Ae,BceijK->ABcijK', H.a.vv[Va, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('AE,BEcijK->ABcijK', H.a.vv[Va, Va], T.aaa.VVvooO, optimize=True)
    )
    dT.aaa.VVvooO += (1.0 / 36.0) * (
            +1.0 * np.einsum('ce,ABeijK->ABcijK', H.a.vv[va, va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('cE,ABEijK->ABcijK', H.a.vv[va, Va], T.aaa.VVVooO, optimize=True)
    )
    dT.aaa.VVvooO += (1.0 / 36.0) * (
            -0.5 * np.einsum('mnij,BAcmnK->ABcijK', H.aa.oooo[oa, oa, oa, oa], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('Mnij,BAcnMK->ABcijK', H.aa.oooo[Oa, oa, oa, oa], T.aaa.VVvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,BAcMNK->ABcijK', H.aa.oooo[Oa, Oa, oa, oa], T.aaa.VVvOOO, optimize=True)
    )
    dT.aaa.VVvooO += (2.0 / 36.0) * (
            +1.0 * np.einsum('MniK,BAcnjM->ABcijK', H.aa.oooo[Oa, oa, oa, Oa], T.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('MNiK,BAcjMN->ABcijK', H.aa.oooo[Oa, Oa, oa, Oa], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VVvooO += (1.0 / 36.0) * (
            -1.0 * np.einsum('ABEf,EcfijK->ABcijK', H.aa.vvvv[Va, Va, Va, va], T.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('ABEF,FEcijK->ABcijK', H.aa.vvvv[Va, Va, Va, Va], T.aaa.VVvooO, optimize=True)
    )
    dT.aaa.VVvooO += (2.0 / 36.0) * (
            +0.5 * np.einsum('Acef,BfeijK->ABcijK', H.aa.vvvv[Va, va, va, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('AcEf,BEfijK->ABcijK', H.aa.vvvv[Va, va, Va, va], T.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('AcEF,BFEijK->ABcijK', H.aa.vvvv[Va, va, Va, Va], T.aaa.VVVooO, optimize=True)
    )
    dT.aaa.VVvooO += (4.0 / 36.0) * (
            +1.0 * np.einsum('Amie,BcemjK->ABcijK', H.aa.voov[Va, oa, oa, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('AmiE,BEcmjK->ABcijK', H.aa.voov[Va, oa, oa, Va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('AMie,BcejMK->ABcijK', H.aa.voov[Va, Oa, oa, va], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('AMiE,BEcjMK->ABcijK', H.aa.voov[Va, Oa, oa, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VVvooO += (2.0 / 36.0) * (
            +1.0 * np.einsum('cmie,ABemjK->ABcijK', H.aa.voov[va, oa, oa, va], T.aaa.VVvooO, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEmjK->ABcijK', H.aa.voov[va, oa, oa, Va], T.aaa.VVVooO, optimize=True)
            - 1.0 * np.einsum('cMie,ABejMK->ABcijK', H.aa.voov[va, Oa, oa, va], T.aaa.VVvoOO, optimize=True)
            - 1.0 * np.einsum('cMiE,ABEjMK->ABcijK', H.aa.voov[va, Oa, oa, Va], T.aaa.VVVoOO, optimize=True)
    )
    dT.aaa.VVvooO += (2.0 / 36.0) * (
            -1.0 * np.einsum('AMKe,BcejiM->ABcijK', H.aa.voov[Va, Oa, Oa, va], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMKE,BEcjiM->ABcijK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VVvooO, optimize=True)
    )
    dT.aaa.VVvooO += (1.0 / 36.0) * (
            -1.0 * np.einsum('cMKe,ABejiM->ABcijK', H.aa.voov[va, Oa, Oa, va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('cMKE,ABEjiM->ABcijK', H.aa.voov[va, Oa, Oa, Va], T.aaa.VVVooO, optimize=True)
    )
    dT.aaa.VVvooO += (9.0 / 36.0) * (
            -1.0 * np.einsum('AmIe,CBeJKm->ABcijK', H.ab.voov[Va, ob, Oa, vb], T.aab.VVvOOo, optimize=True)
            - 1.0 * np.einsum('AmIE,CBEJKm->ABcijK', H.ab.voov[Va, ob, Oa, Vb], T.aab.VVVOOo, optimize=True)
            - 1.0 * np.einsum('AMIe,CBeJKM->ABcijK', H.ab.voov[Va, Ob, Oa, vb], T.aab.VVvOOO, optimize=True)
            - 1.0 * np.einsum('AMIE,CBEJKM->ABcijK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.VVVOOO, optimize=True)
    )
    dT.aaa.VVvooO += (2.0 / 36.0) * (
            +1.0 * np.einsum('cmie,ABejKm->ABcijK', H.ab.voov[va, ob, oa, vb], T.aab.VVvoOo, optimize=True)
            + 1.0 * np.einsum('cmiE,ABEjKm->ABcijK', H.ab.voov[va, ob, oa, Vb], T.aab.VVVoOo, optimize=True)
            + 1.0 * np.einsum('cMie,ABejKM->ABcijK', H.ab.voov[va, Ob, oa, vb], T.aab.VVvoOO, optimize=True)
            + 1.0 * np.einsum('cMiE,ABEjKM->ABcijK', H.ab.voov[va, Ob, oa, Vb], T.aab.VVVoOO, optimize=True)
    )
    dT.aaa.VVvooO += (2.0 / 36.0) * (
            -1.0 * np.einsum('AMKe,BcejiM->ABcijK', H.ab.voov[Va, Ob, Oa, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('AMKE,BcEjiM->ABcijK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.VvVooO, optimize=True)
    )
    dT.aaa.VVvooO += (1.0 / 36.0) * (
            -1.0 * np.einsum('cMKe,ABejiM->ABcijK', H.ab.voov[va, Ob, Oa, vb], T.aab.VVvooO, optimize=True)
            - 1.0 * np.einsum('cMKE,ABEjiM->ABcijK', H.ab.voov[va, Ob, Oa, Vb], T.aab.VVVooO, optimize=True)
    )
    # PROJECTION 9:
    dT.aaa.VvvooO += (2.0 / 36.0) * (
            +1.0 * np.einsum('mi,AcbmjK->AbcijK', H.a.oo[oa, oa], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('Mi,AcbjMK->AbcijK', H.a.oo[Oa, oa], T.aaa.VvvoOO, optimize=True)
    )
    dT.aaa.VvvooO += (1.0 / 36.0) * (
            -1.0 * np.einsum('MK,AcbjiM->AbcijK', H.a.oo[Oa, Oa], T.aaa.VvvooO, optimize=True)
    )
    dT.aaa.VvvooO += (1.0 / 36.0) * (
            -1.0 * np.einsum('AE,EcbijK->AbcijK', H.a.vv[Va, Va], T.aaa.VvvooO, optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 36.0) * (
            +1.0 * np.einsum('ce,AbeijK->AbcijK', H.a.vv[va, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('cE,AEbijK->AbcijK', H.a.vv[va, Va], T.aaa.VVvooO, optimize=True)
    )
    dT.aaa.VvvooO += (1.0 / 36.0) * (
            -0.5 * np.einsum('mnij,AcbmnK->AbcijK', H.aa.oooo[oa, oa, oa, oa], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('Mnij,AcbnMK->AbcijK', H.aa.oooo[Oa, oa, oa, oa], T.aaa.VvvoOO, optimize=True)
            - 0.5 * np.einsum('MNij,AcbMNK->AbcijK', H.aa.oooo[Oa, Oa, oa, oa], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvooO += (3.0 / 36.0) * (
            -0.5 * np.einsum('mNiK,AcbmiN->AbcijK', H.aa.oooo[oa, Oa, oa, Oa], T.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('mNIK,AcbmNI->AbcijK', H.aa.oooo[oa, Oa, Oa, Oa], T.aaa.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MniK,AcbniM->AbcijK', H.aa.oooo[Oa, oa, oa, Oa], T.aaa.VvvooO, optimize=True)
            + 0.5 * np.einsum('MNiK,AcbiMN->AbcijK', H.aa.oooo[Oa, Oa, oa, Oa], T.aaa.VvvoOO, optimize=True)
            + 0.5 * np.einsum('MNIK,AcbMNI->AbcijK', H.aa.oooo[Oa, Oa, Oa, Oa], T.aaa.VvvOOO, optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 36.0) * (
            -1.0 * np.einsum('AbEf,EcfijK->AbcijK', H.aa.vvvv[Va, va, Va, va], T.aaa.VvvooO, optimize=True)
            - 0.5 * np.einsum('AbEF,FEcijK->AbcijK', H.aa.vvvv[Va, va, Va, Va], T.aaa.VVvooO, optimize=True)
    )
    dT.aaa.VvvooO += (1.0 / 36.0) * (
            +0.5 * np.einsum('cbef,AfeijK->AbcijK', H.aa.vvvv[va, va, va, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('cbEf,AEfijK->AbcijK', H.aa.vvvv[va, va, Va, va], T.aaa.VVvooO, optimize=True)
            + 0.5 * np.einsum('cbEF,AFEijK->AbcijK', H.aa.vvvv[va, va, Va, Va], T.aaa.VVVooO, optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 36.0) * (
            -1.0 * np.einsum('AmiE,EcbmjK->AbcijK', H.aa.voov[Va, oa, oa, Va], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('AMiE,EcbjMK->AbcijK', H.aa.voov[Va, Oa, oa, Va], T.aaa.VvvoOO, optimize=True)
    )
    dT.aaa.VvvooO += (4.0 / 36.0) * (
            +1.0 * np.einsum('cmie,AbemjK->AbcijK', H.aa.voov[va, oa, oa, va], T.aaa.VvvooO, optimize=True)
            - 1.0 * np.einsum('cmiE,AEbmjK->AbcijK', H.aa.voov[va, oa, oa, Va], T.aaa.VVvooO, optimize=True)
            - 1.0 * np.einsum('cMie,AbejMK->AbcijK', H.aa.voov[va, Oa, oa, va], T.aaa.VvvoOO, optimize=True)
            + 1.0 * np.einsum('cMiE,AEbjMK->AbcijK', H.aa.voov[va, Oa, oa, Va], T.aaa.VVvoOO, optimize=True)
    )
    dT.aaa.VvvooO += (1.0 / 36.0) * (
            +1.0 * np.einsum('AMKE,EcbjiM->AbcijK', H.aa.voov[Va, Oa, Oa, Va], T.aaa.VvvooO, optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 36.0) * (
            -1.0 * np.einsum('cMKe,AbejiM->AbcijK', H.aa.voov[va, Oa, Oa, va], T.aaa.VvvooO, optimize=True)
            + 1.0 * np.einsum('cMKE,AEbjiM->AbcijK', H.aa.voov[va, Oa, Oa, Va], T.aaa.VVvooO, optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 36.0) * (
            -1.0 * np.einsum('AmiE,cbEjKm->AbcijK', H.ab.voov[Va, ob, oa, Vb], T.aab.vvVoOo, optimize=True)
            - 1.0 * np.einsum('AMiE,cbEjKM->AbcijK', H.ab.voov[Va, Ob, oa, Vb], T.aab.vvVoOO, optimize=True)
    )
    dT.aaa.VvvooO += (4.0 / 36.0) * (
            +1.0 * np.einsum('cmie,AbejKm->AbcijK', H.ab.voov[va, ob, oa, vb], T.aab.VvvoOo, optimize=True)
            + 1.0 * np.einsum('cmiE,AbEjKm->AbcijK', H.ab.voov[va, ob, oa, Vb], T.aab.VvVoOo, optimize=True)
            + 1.0 * np.einsum('cMie,AbejKM->AbcijK', H.ab.voov[va, Ob, oa, vb], T.aab.VvvoOO, optimize=True)
            + 1.0 * np.einsum('cMiE,AbEjKM->AbcijK', H.ab.voov[va, Ob, oa, Vb], T.aab.VvVoOO, optimize=True)
    )
    dT.aaa.VvvooO += (1.0 / 36.0) * (
            +1.0 * np.einsum('AMKE,cbEjiM->AbcijK', H.ab.voov[Va, Ob, Oa, Vb], T.aab.vvVooO, optimize=True)
    )
    dT.aaa.VvvooO += (2.0 / 36.0) * (
            -1.0 * np.einsum('cMKe,AbejiM->AbcijK', H.ab.voov[va, Ob, Oa, vb], T.aab.VvvooO, optimize=True)
            - 1.0 * np.einsum('cMKE,AbEjiM->AbcijK', H.ab.voov[va, Ob, Oa, Vb], T.aab.VvVooO, optimize=True)
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
    #  <i~j~k~a~b~c~ | H(2) | 0 > + (VT3)_C intermediates
    I2C_vvov = -0.5 * np.einsum("mnef,abfimn->abie", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vvov -= np.einsum("nmfe,fabnim->abie", H0.ab.oovv, T.abb, optimize=True)
    I2C_vvov += np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
    I2C_vvov += H.bb.vvov

    I2C_vooo = 0.5 * np.einsum("mnef,aefijn->amij", H0.bb.oovv, T.bbb, optimize=True)
    I2C_vooo += np.einsum("nmfe,faenij->amij", H0.ab.oovv, T.abb, optimize=True)
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

    T.bbb, dT.bbb = cc_loops2.cc_loops2.update_t3d_v2(
        T.bbb,
        dT.bbb,
        H0.b.oo,
        H0.b.vv,
        shift,
    )
    return T, dT
