'''
Coupled-Cluster Method with Singles and Doubles (CCSD)
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.lib.core import cc_loops2

def update(T, dT, H, X, shift, flag_RHF):

    # pre-CCS intermediates
    X = get_pre_ccs_intermediates(X, T, H, flag_RHF)

    # update T1
    T, dT = update_t1a(T, dT, X, H, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, X, H, shift)

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
    return T, dT


def update_t1a(T, dT, X, H, shift):
    """
    Update t1a amplitudes by calculating the projection <ia|(H_N e^(T1+T2))_C|0>.
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
    T.a, dT.a = cc_loops2.update_t1a(
        T.a, dT.a + H.a.vo, H.a.oo, H.a.vv, shift
    )
    return T, dT


# @profile
def update_t1b(T, dT, X, H, shift):
    """
    Update t1b amplitudes by calculating the projection <i~a~|(H_N e^(T1+T2))_C|0>.
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
    T.b, dT.b = cc_loops2.update_t1b(
        T.b, dT.b + H.b.vo, H.b.oo, H.b.vv, shift
    )
    return T, dT


# @profile
def update_t2a(T, dT, H, H0, shift):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2))_C|0>.
    """
    nua, noa = T.a.shape
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
    dT.aa += 0.125 * ccpy_einsum("mnij,abmn->abij", I2A_oooo, T.aa)
    dT.aa += 0.25 * ccpy_einsum("abef,efij->abij", H0.aa.vvvv, tau)

    T.aa, dT.aa = cc_loops2.update_t2a(
        T.aa, dT.aa + 0.25 * H0.aa.vvoo, H0.a.oo, H0.a.vv, shift
    )
    return T, dT


# @profile
def update_t2b(T, dT, H, H0, shift):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2))_C|0>.
    """
    nua, nub, noa, nob = T.ab.shape
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

    T.ab, dT.ab = cc_loops2.update_t2b(
        T.ab, dT.ab + H0.ab.vvoo, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv, shift
    )
    return T, dT


# @profile
def update_t2c(T, dT, H, H0, shift):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2))_C|0>.
    """
    nub, nob = T.b.shape
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
    dT.bb += 0.125 * ccpy_einsum("mnij,abmn->abij", I2C_oooo, T.bb)
    dT.bb += 0.25 * ccpy_einsum("abef,efij->abij", H0.bb.vvvv, tau)

    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H0.bb.vvoo, H0.b.oo, H0.b.vv, shift
    )
    return T, dT
