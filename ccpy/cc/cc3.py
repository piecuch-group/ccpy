"""Module with functions that perform the CC with singles, doubles,
and iterative perturbative triples (CC3) calculation for a molecular system."""

import numpy as np
from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.hbar.hbar_cc3 import get_cc3_intermediates
from ccpy.utilities.updates import cc3_loops

#@profile
def update(T, dT, H, hbar, shift, flag_RHF, system):

    # pre-CCS intermediates
    hbar = get_pre_ccs_intermediates(hbar, T, H, system, flag_RHF)

    # update T1
    dT = build_t1a(T, dT, H, hbar)
    if flag_RHF:
        dT.b = dT.a.copy()
    else:
        dT = build_t1b(T, dT, H, hbar)

    # CCS intermediates
    hbar = get_ccs_intermediates_opt(hbar, T, H, system, flag_RHF)

    # update T2
    dT = build_t2a(T, dT, hbar, H)
    dT = build_t2b(T, dT, hbar, H)
    if flag_RHF:
        dT.bb = dT.aa.copy()
    else:
        dT = build_t2c(T, dT, hbar, H)

    # Adjust CCS intermediate values for contractions with T3 in T2 updates
    hbar.aa.ooov += H.aa.ooov
    hbar.aa.vovv += H.aa.vovv
    hbar.ab.ooov += H.ab.ooov
    hbar.ab.oovo += H.ab.oovo
    hbar.ab.vovv += H.ab.vovv
    hbar.ab.ovvv += H.ab.ovvv
    if flag_RHF:
        hbar.bb.ooov = hbar.aa.ooov
        hbar.bb.vovv = hbar.aa.vovv
    else:
        hbar.bb.ooov += H.bb.ooov
        hbar.bb.vovv += H.bb.vovv

    # CCS-like transformed intermediates for CC3
    X = get_cc3_intermediates(T, H)

    # Update all T1 and T2 clusters together by computing T3 on-the-fly once
    # it would be nice if the T1-transformed X intermediates here were simply the
    # elements of the CCS-transformed hbar computed before T2 builds.
    T.a, T.b, T.aa, T.ab, T.bb, dT.a, dT.b, dT.aa, dT.ab, dT.bb = cc3_loops.cc3_loops.update_t(
        T.a, T.b, T.aa, T.ab, T.bb,
        dT.a, dT.b, dT.aa, dT.ab, dT.bb,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        hbar.a.ov, hbar.b.ov,
        H.aa.oovv, H.ab.oovv, H.bb.oovv,
        hbar.aa.ooov, hbar.aa.vovv,
        hbar.ab.ooov, hbar.ab.oovo, hbar.ab.vovv, hbar.ab.ovvv,
        hbar.bb.ooov, hbar.bb.vovv,
        X["aa"]["vooo"], X["aa"]["vvov"],
        X["ab"]["vooo"], X["ab"]["ovoo"], X["ab"]["vvov"], X["ab"]["vvvo"],
        X["bb"]["vooo"], X["bb"]["vvov"],
        shift)
    return T, dT

#@profile
def build_t1a(T, dT, H, X):
    """
    Update t1a amplitudes by calculating the projection <ia|(H_N e^(T1+T2))_C|0>.
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
    dT.a += H.a.vo
    return dT

#@profile
def build_t1b(T, dT, H, X):
    """
    Update t1b amplitudes by calculating the projection <i~a~|(H_N e^(T1+T2))_C|0>.
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
    dT.b += H.b.vo
    return dT

#@profile
def build_t2a(T, dT, H, H0):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2))_C|0>.
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

    dT.aa = -0.5 * np.einsum("amij,bm->abij", I2A_vooo, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("abie,ej->abij", H.aa.vvov, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", H.a.vv, T.aa, optimize=True)
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", H.a.oo, T.aa, optimize=True)
    dT.aa += np.einsum("amie,ebmj->abij", I2A_voov, T.aa, optimize=True)
    dT.aa += np.einsum("amie,bejm->abij", I2B_voov, T.ab, optimize=True)
    dT.aa += 0.25 * np.einsum("abef,efij->abij", H0.aa.vvvv, tau, optimize=True)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", I2A_oooo, T.aa, optimize=True)
    dT.aa += 0.25 * H0.aa.vvoo
    return dT

#@profile
def build_t2b(T, dT, H, H0):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2))_C|0>.
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
    dT.ab += H0.ab.vvoo
    return dT

#@profile
def build_t2c(T, dT, H, H0):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2))_C|0>.
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
    dT.bb += 0.25 * H0.bb.vvoo
    return dT
