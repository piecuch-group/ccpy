"""Module with functions that perform the CC with singles, doubles,
and P-space triples [CC(P)] calculation for a molecular system."""
import numpy as np

from ccpy.hbar.hbar_ccs import get_ccs_intermediates_opt
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.utilities.updates import ccsdt_p_loops

def update(T, dT, H, shift, flag_RHF, system, t3_excitations, pspace=None):

    # determine whether t3 updates should be done. Stupid compatibility with
    # empty sections of t3_excitations
    do_t3 = {"aaa" : True, "aab" : True, "abb" : True, "bbb" : True}
    if np.array_equal(t3_excitations["aaa"][0,:], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aaa"] = False
    if np.array_equal(t3_excitations["aab"][0,:], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aab"] = False
    if np.array_equal(t3_excitations["abb"][0,:], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["abb"] = False
    if np.array_equal(t3_excitations["bbb"][0,:], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["bbb"] = False
    build_hbar = do_t3["aaa"] or do_t3["aab"] or do_t3["abb"] or do_t3["bbb"]

    # update T1
    T, dT = update_t1a(T, dT, H, shift, t3_excitations)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, shift, t3_excitations)

    # CCS intermediates
    hbar = get_ccs_intermediates_opt(T, H)

    # update T2
    T, dT = update_t2a(T, dT, hbar, H, shift, t3_excitations)
    T, dT = update_t2b(T, dT, hbar, H, shift, t3_excitations)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, hbar, H, shift, t3_excitations)

    # CCSD intermediates
    #[TODO]: Should accept CCS HBar as input and build only terms with T2 in it
    if build_hbar:
        hbar = get_ccsd_intermediates(T, H)

    # update T3
    if do_t3["aaa"]:
        T, dT, t3_excitations = update_t3a(T, dT, hbar, H, shift, t3_excitations)
    if do_t3["aab"]:
        T, dT, t3_excitations = update_t3b(T, dT, hbar, H, shift, t3_excitations)
    if flag_RHF:
       T.abb = T.aab.copy()
       t3_excitations["abb"] = t3_excitations["aab"][:, np.array([2, 0, 1, 5, 3, 4])]
       T.bbb = T.aaa.copy()
       t3_excitations["bbb"] = t3_excitations["aaa"].copy()
    else:
        if do_t3["abb"]:
            T, dT, t3_excitations = update_t3c(T, dT, hbar, H, shift, t3_excitations)
        if do_t3["bbb"]:
            T, dT, t3_excitations = update_t3d(T, dT, hbar, H, shift, t3_excitations)

    return T, dT

def update_t1a(T, dT, H, shift, t3_excitations):
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

    T.a, dT.a = ccsdt_p_loops.ccsdt_p_loops.update_t1a(
        T.a, 
        dT.a + H.a.vo,
        t3_excitations["aaa"].T, t3_excitations["aab"].T, t3_excitations["abb"].T,
        T.aaa, T.aab, T.abb,
        H.aa.oovv, H.ab.oovv, H.bb.oovv,
        H.a.oo, H.a.vv,
        shift
    )

    return T, dT

def update_t1b(T, dT, H, shift, t3_excitations):
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

    T.b, dT.b = ccsdt_p_loops.ccsdt_p_loops.update_t1b(
        T.b,
        dT.b + H.b.vo,
        t3_excitations["aab"].T, t3_excitations["abb"].T, t3_excitations["bbb"].T,
        T.aab, T.abb, T.bbb,
        H.aa.oovv, H.ab.oovv, H.bb.oovv,
        H.b.oo, H.b.vv,
        shift
    )

    return T, dT

# @profile
def update_t2a(T, dT, H, H0, shift, t3_excitations):
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

    I2A_vooo = H.aa.vooo + 0.5*np.einsum('anef,efij->anij', H0.aa.vovv + 0.5 * H.aa.vovv, T.aa, optimize=True)

    tau = 0.5 * T.aa + np.einsum('ai,bj->abij', T.a, T.a, optimize=True)

    dT.aa = -0.5 * np.einsum("amij,bm->abij", I2A_vooo, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("abie,ej->abij", H.aa.vvov, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", I1A_vv, T.aa, optimize=True)
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", I1A_oo, T.aa, optimize=True)
    dT.aa += np.einsum("amie,ebmj->abij", I2A_voov, T.aa, optimize=True)
    dT.aa += np.einsum("amie,bejm->abij", I2B_voov, T.ab, optimize=True)
    dT.aa += 0.25 * np.einsum("abef,efij->abij", H.aa.vvvv, tau, optimize=True)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", I2A_oooo, T.aa, optimize=True)

    T.aa, dT.aa = ccsdt_p_loops.ccsdt_p_loops.update_t2a(
        T.aa,
        dT.aa + 0.25 * H0.aa.vvoo,
        t3_excitations["aaa"].T, t3_excitations["aab"].T,
        T.aaa, T.aab,
        H.a.ov, H.b.ov,
        H0.aa.ooov + H.aa.ooov, H0.aa.vovv + H.aa.vovv,
        H0.ab.ooov + H.ab.ooov, H0.ab.vovv + H.ab.vovv,
        H0.a.oo, H0.a.vv,
        shift
    )

    return T, dT

# @profile
def update_t2b(T, dT, H, H0, shift, t3_excitations):
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

    T.ab, dT.ab = ccsdt_p_loops.ccsdt_p_loops.update_t2b(
        T.ab,
        dT.ab + H0.ab.vvoo,
        t3_excitations["aab"].T, t3_excitations["abb"].T,
        T.aab, T.abb,
        H.a.ov, H.b.ov,
        H.aa.ooov + H0.aa.ooov, H.aa.vovv + H0.aa.vovv,
        H.ab.ooov + H0.ab.ooov, H.ab.oovo + H0.ab.oovo, H.ab.vovv + H0.ab.vovv, H.ab.ovvv + H0.ab.ovvv,
        H.bb.ooov + H0.bb.ooov, H.bb.vovv + H0.bb.vovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )

    return T, dT

# @profile
def update_t2c(T, dT, H, H0, shift, t3_excitations):
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

    I2C_vooo = H.bb.vooo + 0.5*np.einsum('anef,efij->anij', H0.bb.vovv + 0.5 * H.bb.vovv, T.bb, optimize=True)

    tau = 0.5 * T.bb + np.einsum('ai,bj->abij', T.b, T.b, optimize=True)

    dT.bb = -0.5 * np.einsum("amij,bm->abij", I2C_vooo, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("abie,ej->abij", H.bb.vvov, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("ae,ebij->abij", I1B_vv, T.bb, optimize=True)
    dT.bb -= 0.5 * np.einsum("mi,abmj->abij", I1B_oo, T.bb, optimize=True)
    dT.bb += np.einsum("amie,ebmj->abij", I2C_voov, T.bb, optimize=True)
    dT.bb += np.einsum("maei,ebmj->abij", I2B_ovvo, T.ab, optimize=True)
    dT.bb += 0.25 * np.einsum("abef,efij->abij", H.bb.vvvv, tau, optimize=True)
    dT.bb += 0.125 * np.einsum("mnij,abmn->abij", I2C_oooo, T.bb, optimize=True)

    T.bb, dT.bb = ccsdt_p_loops.ccsdt_p_loops.update_t2c(
        T.bb,
        dT.bb + 0.25 * H0.bb.vvoo,
        t3_excitations["abb"].T, t3_excitations["bbb"].T,
        T.abb, T.bbb,
        H.a.ov, H.b.ov,
        H0.ab.oovo + H.ab.oovo, H0.ab.ovvv + H.ab.ovvv,
        H0.bb.ooov + H.bb.ooov, H0.bb.vovv + H.bb.vovv,
        H0.b.oo, H0.b.vv,
        shift
    )

    return T, dT

# @profile
def update_t3a(T, dT, H, H0, shift, t3_excitations):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3))_C|0>.
    """
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)

    dT.aaa, T.aaa, t3_excitations["aaa"] = ccsdt_p_loops.ccsdt_p_loops.update_t3a_p(
        T.aaa, t3_excitations["aaa"].T, 
        T.aab, t3_excitations["aab"].T,
        T.aa,
        H.a.oo, H.a.vv,
        H0.aa.oovv, H.aa.vvov, I2A_vooo,
        H.aa.oooo, H.aa.voov, H.aa.vvvv,
        H0.ab.oovv, H.ab.voov,
        H0.a.oo, H0.a.vv,
        shift
    )
    # re-transpose t3_excitations to maintain consistency with other parts of code
    t3_excitations["aaa"] = t3_excitations["aaa"].T

    return T, dT, t3_excitations

# @profile
def update_t3b(T, dT, H, H0, shift, t3_excitations):
    """
    Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N e^(T1+T2+T3))_C|0>.
    """
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)

    dT.aab, T.aab, t3_excitations["aab"] = ccsdt_p_loops.ccsdt_p_loops.update_t3b_p(
        T.aaa, t3_excitations["aaa"].T,
        T.aab, t3_excitations["aab"].T,
        T.abb, t3_excitations["abb"].T,
        T.aa, T.ab,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H0.aa.oovv, H.aa.vvov, I2A_vooo, H.aa.oooo, H.aa.voov, H.aa.vvvv,
        H0.ab.oovv, H.ab.vvov, H.ab.vvvo, I2B_vooo, I2B_ovoo, 
        H.ab.oooo, H.ab.voov,H.ab.vovo, H.ab.ovov, H.ab.ovvo, H.ab.vvvv,
        H0.bb.oovv, H.bb.voov,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )
    # re-transpose t3_excitations to maintain consistency with other parts of code
    t3_excitations["aab"] = t3_excitations["aab"].T

    return T, dT, t3_excitations

# @profile
def update_t3c(T, dT, H, H0, shift, t3_excitations):
    """
    Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N e^(T1+T2+T3))_C|0>.
    """
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)

    dT.abb, T.abb, t3_excitations["abb"] = ccsdt_p_loops.ccsdt_p_loops.update_t3c_p(
        T.aab, t3_excitations["aab"].T,
        T.abb, t3_excitations["abb"].T,
        T.bbb, t3_excitations["bbb"].T,
        T.ab, T.bb,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H0.aa.oovv, H.aa.voov,
        H0.ab.oovv, I2B_vooo, I2B_ovoo, H.ab.vvov, H.ab.vvvo, H.ab.oooo,
        H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo, H.ab.vvvv,
        H0.bb.oovv, I2C_vooo, H.bb.vvov, H.bb.oooo, H.bb.voov, H.bb.vvvv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        shift
    )
    # re-transpose t3_excitations to maintain consistency with other parts of code
    t3_excitations["abb"] = t3_excitations["abb"].T

    return T, dT, t3_excitations

# @profile
def update_t3d(T, dT, H, H0, shift, t3_excitations):
    """
    Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N e^(T1+T2+T3))_C|0>.
    """
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)

    dT.bbb, T.bbb, t3_excitations["bbb"] = ccsdt_p_loops.ccsdt_p_loops.update_t3d_p(
        T.abb, t3_excitations["abb"].T,
        T.bbb, t3_excitations["bbb"].T,
        T.bb,
        H.b.oo, H.b.vv,
        H0.bb.oovv, H.bb.vvov, I2C_vooo,
        H.bb.oooo, H.bb.voov, H.bb.vvvv,
        H0.ab.oovv, H.ab.ovvo,
        H0.b.oo, H0.b.vv,
        shift
    )
    # re-transpose t3_excitations to maintain consistency with other parts of code
    t3_excitations["bbb"] = t3_excitations["bbb"].T

    return T, dT, t3_excitations