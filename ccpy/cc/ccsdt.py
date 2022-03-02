"""Module with functions that perform the CC with singles, doubles,
and triples (CCSDT) calculation for a molecular system."""

import numpy as np

from ccpy.hbar.hbar_ccs import get_ccs_intermediates_opt
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.utilities.updates import cc_loops2


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
    #[TODO]: Should accept CCS HBar as input and build only terms with T2 in it
    hbar = get_ccsd_intermediates(T, H)

    # update T3
    T, dT = update_t3a(T, dT, hbar, H, shift)
    T, dT = update_t3b(T, dT, hbar, H, shift)
    if flag_RHF:
        T.abb = np.transpose(T.aab, (2, 0, 1, 5, 3, 4))
        dT.abb = np.transpose(dT.abb, (2, 1, 0, 5, 4, 3))
        T.bbb = T.aaa.copy()
        dT.bbb = dT.aaa.copy()
    else:
        T, dT = update_t3c(T, dT, hbar, H, shift)
        T, dT = update_t3d(T, dT, hbar, H, shift)

    return T, dT

def update_t1a(T, dT, H, shift):
    """Update t1a amplitudes by calculating the projection <ia|(H_N e^(T1+T2+T3))_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2
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

    dT.a += 0.25 * np.einsum("mnef,aefimn->ai", H.aa.oovv, T.aaa, optimize=True)
    dT.a += np.einsum("mnef,aefimn->ai", H.ab.oovv, T.aab, optimize=True)
    dT.a += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, T.abb, optimize=True)

    # T, update_t1a_loop(T, X1A, fA_oo, fA_vv, shift)
    T.a, dT.a = cc_loops2.cc_loops2.update_t1a(
        T.a, dT.a + H.a.vo, H.a.oo, H.a.vv, shift
    )
    return T, dT

def update_t1b(T, dT, H, shift):
    """Update t1b amplitudes by calculating the projection <i~a~|(H_N e^(T1+T2+T3))_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2
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

    dT.b += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, T.abb, optimize=True)
    dT.b += 0.25 * np.einsum("mnef,efamni->ai", H.aa.oovv, T.aab, optimize=True)
    dT.b += np.einsum("mnef,efamni->ai", H.ab.oovv, T.abb, optimize=True)

    T.b, dT.b = cc_loops2.cc_loops2.update_t1b(
        T.b, dT.b + H.b.vo, H.b.oo, H.b.vv, shift
    )
    return T, dT

def update_t2a(T, dT, H, H0, shift):
    """Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2+T3))_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2
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
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.a.ov, T.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T.aab, optimize=True)
    dT.aa -= 0.5 * np.einsum("mnif,abfmjn->abij", H.ab.ooov, T.aab, optimize=True)
    dT.aa -= 0.25 * np.einsum("mnif,abfmjn->abij", H.aa.ooov, T.aaa, optimize=True)
    dT.aa += 0.25 * np.einsum("anef,ebfijn->abij", H.aa.vovv, T.aaa, optimize=True)
    dT.aa += 0.5 * np.einsum("anef,ebfijn->abij", H.ab.vovv, T.aab, optimize=True)

    T.aa, dT.aa = cc_loops2.cc_loops2.update_t2a(
        T.aa, dT.aa + 0.25 * H0.aa.vvoo, H0.a.oo, H0.a.vv, shift
    )
    return T, dT

def update_t2b(T, dT, H, H0, shift):
    """Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2+T3))_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2
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
        T.ab, dT.ab + H0.ab.vvoo, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv, shift
    )

    return T, dT

def update_t2c(T, dT, H, H0, shift):
    """Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2+T3))_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2
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
    dT.bb += 0.25 * np.einsum("me,eabmij->abij", H.a.ov, T.abb, optimize=True)
    dT.bb += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, T.bbb, optimize=True)
    dT.bb += 0.25 * np.einsum("anef,ebfijn->abij", H.bb.vovv, T.bbb, optimize=True)
    dT.bb += 0.5 * np.einsum("nafe,febnij->abij", H.ab.ovvv, T.abb, optimize=True)
    dT.bb -= 0.25 * np.einsum("mnif,abfmjn->abij", H.bb.ooov, T.bbb, optimize=True)
    dT.bb -= 0.5 * np.einsum("nmfi,fabnmj->abij", H.ab.oovo, T.abb, optimize=True)

    T.bb, dT.bb = cc_loops2.cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H0.bb.vvoo, H0.b.oo, H0.b.vv, shift
    )

    return T, dT

# @profile
def update_t3a(T, dT, H, H0, shift):
    """Update t3a amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3))_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2, T3
    """
    # Intermediates
    I2A_vvov = -0.5 * np.einsum(
        "mnef,abfimn->abie", H0.aa.oovv, T.aaa, optimize=True
    )
    I2A_vvov -= np.einsum(
        "mnef,abfimn->abie", H0.ab.oovv, T.aab, optimize=True
    )
    I2A_vvov += H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
    I2A_vooo = 0.5 * np.einsum(
        "mnef,aefijn->amij", H0.aa.oovv, T.aaa, optimize=True
    )
    I2A_vooo = H.aa.vooo + np.einsum(
        "mnef,aefijn->amij", H0.ab.oovv, T.aab, optimize=True
    )

    # MM(2,3)A
    dT.aaa = -0.25 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.aa, optimize=True)
    dT.aaa += 0.25 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.aa, optimize=True)
    # (HBar*T3)_C
    dT.aaa -= (1.0 / 12.0) * np.einsum(
        "mk,abcijm->abcijk", H.a.oo, T.aaa, optimize=True
    )
    dT.aaa += (1.0 / 12.0) * np.einsum(
        "ce,abeijk->abcijk", H.a.vv, T.aaa, optimize=True
    )
    dT.aaa += (1.0 / 24.0) * np.einsum(
        "mnij,abcmnk->abcijk", H.aa.oooo, T.aaa, optimize=True
    )
    dT.aaa += (1.0 / 24.0) * np.einsum(
        "abef,efcijk->abcijk", H.aa.vvvv, T.aaa, optimize=True
    )
    dT.aaa += 0.25 * np.einsum(
        "cmke,abeijm->abcijk", H.aa.voov, T.aaa, optimize=True
    )
    dT.aaa += 0.25 * np.einsum(
        "cmke,abeijm->abcijk", H.ab.voov, T.aab, optimize=True
    )

    T.aaa, dT.aaa = cc_loops2.cc_loops2.update_t3a(
        T.aaa, dT.aaa, H0.a.oo, H0.a.vv, shift
    )
    return T, dT


# @profile
def update_t3b(cc_t, ints, H1A, H1B, H2A, H2B, H2C, sys, shift):
    """Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N e^(T1+T2+T3))_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2, T3
    """
    # MM23B + (VT3)_C intermediates
    I2A_vvov = -0.5 * np.einsum(
        "mnef,abfimn->abie", ints["vA"]["oovv"], cc_t["t3a"], optimize=True
    )
    I2A_vvov += -np.einsum(
        "mnef,abfimn->abie", ints["vB"]["oovv"], cc_t["t3b"], optimize=True
    )
    I2A_vvov += H2A["vvov"]
    I2A_vooo = 0.5 * np.einsum(
        "mnef,aefijn->amij", ints["vA"]["oovv"], cc_t["t3a"], optimize=True
    )
    I2A_vooo += np.einsum(
        "mnef,aefijn->amij", ints["vB"]["oovv"], cc_t["t3b"], optimize=True
    )
    I2A_vooo += H2A["vooo"]
    I2A_vooo += -np.einsum("me,aeij->amij", H1A["ov"], cc_t["t2a"], optimize=True)
    I2B_vvvo = -0.5 * np.einsum(
        "mnef,afbmnj->abej", ints["vA"]["oovv"], cc_t["t3b"], optimize=True
    )
    I2B_vvvo += -np.einsum(
        "mnef,afbmnj->abej", ints["vB"]["oovv"], cc_t["t3c"], optimize=True
    )
    I2B_vvvo += H2B["vvvo"]
    I2B_ovoo = 0.5 * np.einsum(
        "mnef,efbinj->mbij", ints["vA"]["oovv"], cc_t["t3b"], optimize=True
    )
    I2B_ovoo += np.einsum(
        "mnef,efbinj->mbij", ints["vB"]["oovv"], cc_t["t3c"], optimize=True
    )
    I2B_ovoo += H2B["ovoo"]
    I2B_ovoo += -np.einsum("me,ecjk->mcjk", H1A["ov"], cc_t["t2b"], optimize=True)
    I2B_vvov = -np.einsum(
        "nmfe,afbinm->abie", ints["vB"]["oovv"], cc_t["t3b"], optimize=True
    )
    I2B_vvov += -0.5 * np.einsum(
        "nmfe,afbinm->abie", ints["vC"]["oovv"], cc_t["t3c"], optimize=True
    )
    I2B_vvov += H2B["vvov"]
    I2B_vooo = np.einsum(
        "nmfe,afeinj->amij", ints["vB"]["oovv"], cc_t["t3b"], optimize=True
    )
    I2B_vooo += 0.5 * np.einsum(
        "nmfe,afeinj->amij", ints["vC"]["oovv"], cc_t["t3c"], optimize=True
    )
    I2B_vooo += H2B["vooo"]
    I2B_vooo += -np.einsum("me,aeik->amik", H1B["ov"], cc_t["t2b"], optimize=True)
    # MM(2,3)B
    X3B = 0.5 * np.einsum("bcek,aeij->abcijk", I2B_vvvo, cc_t["t2a"], optimize=True)
    X3B -= 0.5 * np.einsum("mcjk,abim->abcijk", I2B_ovoo, cc_t["t2a"], optimize=True)
    X3B += np.einsum("acie,bejk->abcijk", I2B_vvov, cc_t["t2b"], optimize=True)
    X3B -= np.einsum("amik,bcjm->abcijk", I2B_vooo, cc_t["t2b"], optimize=True)
    X3B += 0.5 * np.einsum("abie,ecjk->abcijk", I2A_vvov, cc_t["t2b"], optimize=True)
    X3B -= 0.5 * np.einsum("amij,bcmk->abcijk", I2A_vooo, cc_t["t2b"], optimize=True)
    # (HBar*T3)_C
    X3B -= 0.5 * np.einsum("mi,abcmjk->abcijk", H1A["oo"], cc_t["t3b"], optimize=True)
    X3B -= 0.25 * np.einsum("mk,abcijm->abcijk", H1B["oo"], cc_t["t3b"], optimize=True)
    X3B += 0.5 * np.einsum("ae,ebcijk->abcijk", H1A["vv"], cc_t["t3b"], optimize=True)
    X3B += 0.25 * np.einsum("ce,abeijk->abcijk", H1B["vv"], cc_t["t3b"], optimize=True)
    X3B += 0.125 * np.einsum(
        "mnij,abcmnk->abcijk", H2A["oooo"], cc_t["t3b"], optimize=True
    )
    X3B += 0.5 * np.einsum(
        "mnjk,abcimn->abcijk", H2B["oooo"], cc_t["t3b"], optimize=True
    )
    X3B += 0.125 * np.einsum(
        "abef,efcijk->abcijk", H2A["vvvv"], cc_t["t3b"], optimize=True
    )
    X3B += 0.5 * np.einsum(
        "bcef,aefijk->abcijk", H2B["vvvv"], cc_t["t3b"], optimize=True
    )
    X3B += np.einsum("amie,ebcmjk->abcijk", H2A["voov"], cc_t["t3b"], optimize=True)
    X3B += np.einsum("amie,becjmk->abcijk", H2B["voov"], cc_t["t3c"], optimize=True)
    X3B += 0.25 * np.einsum(
        "mcek,abeijm->abcijk", H2B["ovvo"], cc_t["t3a"], optimize=True
    )
    X3B += 0.25 * np.einsum(
        "cmke,abeijm->abcijk", H2C["voov"], cc_t["t3b"], optimize=True
    )
    X3B -= 0.5 * np.einsum(
        "amek,ebcijm->abcijk", H2B["vovo"], cc_t["t3b"], optimize=True
    )
    X3B -= 0.5 * np.einsum(
        "mcie,abemjk->abcijk", H2B["ovov"], cc_t["t3b"], optimize=True
    )

    cc_t["t3b"], resid = cc_loops2.cc_loops2.update_t3b_v2(
        cc_t["t3b"],
        X3B,
        ints["fA"]["oo"],
        ints["fA"]["vv"],
        ints["fB"]["oo"],
        ints["fB"]["vv"],
        shift,
    )
    return cc_t, resid


# @profile
def update_t3c(cc_t, ints, H1A, H1B, H2A, H2B, H2C, sys, shift):
    """Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N e^(T1+T2+T3))_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2, T3
    """
    # Intermediates
    I2B_vvvo = -0.5 * np.einsum(
        "mnef,afbmnj->abej", ints["vA"]["oovv"], cc_t["t3b"], optimize=True
    )
    I2B_vvvo += -np.einsum(
        "mnef,afbmnj->abej", ints["vB"]["oovv"], cc_t["t3c"], optimize=True
    )
    I2B_vvvo += H2B["vvvo"]
    I2B_ovoo = 0.5 * np.einsum(
        "mnef,efbinj->mbij", ints["vA"]["oovv"], cc_t["t3b"], optimize=True
    )
    I2B_ovoo += np.einsum(
        "mnef,efbinj->mbij", ints["vB"]["oovv"], cc_t["t3c"], optimize=True
    )
    I2B_ovoo += H2B["ovoo"]
    I2B_ovoo -= np.einsum("me,ebij->mbij", H1A["ov"], cc_t["t2b"], optimize=True)
    I2B_vvov = -np.einsum(
        "nmfe,afbinm->abie", ints["vB"]["oovv"], cc_t["t3b"], optimize=True
    )
    I2B_vvov += -0.5 * np.einsum(
        "nmfe,afbinm->abie", ints["vC"]["oovv"], cc_t["t3c"], optimize=True
    )
    I2B_vvov += H2B["vvov"]
    I2B_vooo = np.einsum(
        "nmfe,afeinj->amij", ints["vB"]["oovv"], cc_t["t3b"], optimize=True
    )
    I2B_vooo += 0.5 * np.einsum(
        "nmfe,afeinj->amij", ints["vC"]["oovv"], cc_t["t3c"], optimize=True
    )
    I2B_vooo += H2B["vooo"]
    I2B_vooo -= np.einsum("me,aeij->amij", H1B["ov"], cc_t["t2b"], optimize=True)
    I2C_vvov = -0.5 * np.einsum(
        "mnef,abfimn->abie", ints["vC"]["oovv"], cc_t["t3d"], optimize=True
    )
    I2C_vvov += -np.einsum(
        "nmfe,fabnim->abie", ints["vB"]["oovv"], cc_t["t3c"], optimize=True
    )
    I2C_vvov += H2C["vvov"]
    I2C_vooo = np.einsum(
        "nmfe,faenij->amij", ints["vB"]["oovv"], cc_t["t3c"], optimize=True
    )
    I2C_vooo += 0.5 * np.einsum(
        "mnef,aefijn->amij", ints["vC"]["oovv"], cc_t["t3d"], optimize=True
    )
    I2C_vooo += H2C["vooo"]
    I2C_vooo -= np.einsum("me,cekj->cmkj", H1B["ov"], cc_t["t2c"], optimize=True)
    # MM(2,3)C
    X3C = 0.5 * np.einsum("abie,ecjk->abcijk", I2B_vvov, cc_t["t2c"], optimize=True)
    X3C -= 0.5 * np.einsum("amij,bcmk->abcijk", I2B_vooo, cc_t["t2c"], optimize=True)
    X3C += 0.5 * np.einsum("cbke,aeij->abcijk", I2C_vvov, cc_t["t2b"], optimize=True)
    X3C -= 0.5 * np.einsum("cmkj,abim->abcijk", I2C_vooo, cc_t["t2b"], optimize=True)
    X3C += np.einsum("abej,ecik->abcijk", I2B_vvvo, cc_t["t2b"], optimize=True)
    X3C -= np.einsum("mbij,acmk->abcijk", I2B_ovoo, cc_t["t2b"], optimize=True)
    # (HBar*T3)_C
    X3C -= 0.25 * np.einsum("mi,abcmjk->abcijk", H1A["oo"], cc_t["t3c"], optimize=True)
    X3C -= 0.5 * np.einsum("mj,abcimk->abcijk", H1B["oo"], cc_t["t3c"], optimize=True)
    X3C += 0.25 * np.einsum("ae,ebcijk->abcijk", H1A["vv"], cc_t["t3c"], optimize=True)
    X3C += 0.5 * np.einsum("be,aecijk->abcijk", H1B["vv"], cc_t["t3c"], optimize=True)
    X3C += 0.125 * np.einsum(
        "mnjk,abcimn->abcijk", H2C["oooo"], cc_t["t3c"], optimize=True
    )
    X3C += 0.5 * np.einsum(
        "mnij,abcmnk->abcijk", H2B["oooo"], cc_t["t3c"], optimize=True
    )
    X3C += 0.125 * np.einsum(
        "bcef,aefijk->abcijk", H2C["vvvv"], cc_t["t3c"], optimize=True
    )
    X3C += 0.5 * np.einsum(
        "abef,efcijk->abcijk", H2B["vvvv"], cc_t["t3c"], optimize=True
    )
    X3C += 0.25 * np.einsum(
        "amie,ebcmjk->abcijk", H2A["voov"], cc_t["t3c"], optimize=True
    )
    X3C += 0.25 * np.einsum(
        "amie,ebcmjk->abcijk", H2B["voov"], cc_t["t3d"], optimize=True
    )
    X3C += np.einsum("mbej,aecimk->abcijk", H2B["ovvo"], cc_t["t3b"], optimize=True)
    X3C += np.einsum("bmje,aecimk->abcijk", H2C["voov"], cc_t["t3c"], optimize=True)
    X3C -= 0.5 * np.einsum(
        "mbie,aecmjk->abcijk", H2B["ovov"], cc_t["t3c"], optimize=True
    )
    X3C -= 0.5 * np.einsum(
        "amej,ebcimk->abcijk", H2B["vovo"], cc_t["t3c"], optimize=True
    )

    cc_t["t3c"], resid = cc_loops2.cc_loops2.update_t3c_v2(
        cc_t["t3c"],
        X3C,
        ints["fA"]["oo"],
        ints["fA"]["vv"],
        ints["fB"]["oo"],
        ints["fB"]["vv"],
        shift,
    )
    return cc_t, resid


# @profile
def update_t3d(cc_t, ints, H1A, H1B, H2A, H2B, H2C, sys, shift):
    """Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N e^(T1+T2+T3))_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2, T3
    """
    # Intermediates
    I2C_vvov = -0.5 * np.einsum(
        "mnef,abfimn->abie", ints["vC"]["oovv"], cc_t["t3d"], optimize=True
    )
    I2C_vvov -= np.einsum(
        "nmfe,fabnim->abie", ints["vB"]["oovv"], cc_t["t3c"], optimize=True
    )
    I2C_vvov += np.einsum("me,abim->abie", H1B["ov"], cc_t["t2c"], optimize=True)
    I2C_vvov += H2C["vvov"]
    I2C_vooo = 0.5 * np.einsum(
        "mnef,aefijn->amij", ints["vC"]["oovv"], cc_t["t3d"], optimize=True
    )
    I2C_vooo += np.einsum(
        "nmfe,faenij->amij", ints["vB"]["oovv"], cc_t["t3c"], optimize=True
    )
    I2C_vooo += H2C["vooo"]
    # MM(2,3)D
    X3D = -0.25 * np.einsum("amij,bcmk->abcijk", I2C_vooo, cc_t["t2c"], optimize=True)
    X3D += 0.25 * np.einsum("abie,ecjk->abcijk", I2C_vvov, cc_t["t2c"], optimize=True)
    # (HBar*T3)_C
    X3D -= (1.0 / 12.0) * np.einsum(
        "mk,abcijm->abcijk", H1B["oo"], cc_t["t3d"], optimize=True
    )
    X3D += (1.0 / 12.0) * np.einsum(
        "ce,abeijk->abcijk", H1B["vv"], cc_t["t3d"], optimize=True
    )
    X3D += (1.0 / 24.0) * np.einsum(
        "mnij,abcmnk->abcijk", H2C["oooo"], cc_t["t3d"], optimize=True
    )
    X3D += (1.0 / 24.0) * np.einsum(
        "abef,efcijk->abcijk", H2C["vvvv"], cc_t["t3d"], optimize=True
    )
    X3D += 0.25 * np.einsum(
        "maei,ebcmjk->abcijk", H2B["ovvo"], cc_t["t3c"], optimize=True
    )
    X3D += 0.25 * np.einsum(
        "amie,ebcmjk->abcijk", H2C["voov"], cc_t["t3d"], optimize=True
    )

    cc_t["t3d"], resid = cc_loops2.cc_loops2.update_t3d_v2(
        cc_t["t3d"], X3D, ints["fB"]["oo"], ints["fB"]["vv"], shift
    )
    return cc_t, resid
