'''
Approximate Coupled-Cluster Method with Singles and Doubles (CC2)
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.lib.core import cc_loops2

def update(T, dT, H, X, shift, flag_RHF):

    # pre-CCS intermediates
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

    # Add parts needed to make vvvv terms work out
    X.aa.vvov += 0.5 * ccpy_einsum("abef,ei->abif", H.aa.vvvv, T.a)
    X.ab.vvov += ccpy_einsum("abef,ei->abif", H.ab.vvvv, T.a)
    X.bb.vvov += 0.5 * ccpy_einsum("abef,ei->abif", H.bb.vvvv, T.b)

    # update T2
    T, dT = update_t2a(T, dT, X, H, shift)
    T, dT = update_t2b(T, dT, X, H, shift)
    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T, dT = update_t2c(T, dT, X, H, shift)

    return T, dT

def update_t1a(T, dT, H, X, shift):
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

def update_t1b(T, dT, H, X, shift):
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

def update_t2a(T, dT, H, H0, shift):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2))_C|0>.
    """
    dT.aa = -0.5 * ccpy_einsum("amij,bm->abij", H.aa.vooo, T.a)
    dT.aa += 0.5 * ccpy_einsum("abie,ej->abij", H.aa.vvov, T.a)
    # Need iterative Fock terms in this scheme
    dT.aa -= 0.5 * ccpy_einsum("mi,abmj->abij", H0.a.oo, T.aa)
    dT.aa += 0.5 * ccpy_einsum("ae,ebij->abij", H0.a.vv, T.aa)
    T.aa, dT.aa = cc_loops2.update_t2a(
        T.aa, dT.aa + 0.25 * H0.aa.vvoo, H0.a.oo, H0.a.vv, shift
    )
    return T, dT

def update_t2b(T, dT, H, H0, shift):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2))_C|0>.
    """
    dT.ab = -ccpy_einsum("mbij,am->abij", H.ab.ovoo, T.a)
    dT.ab -= ccpy_einsum("amij,bm->abij", H.ab.vooo, T.b)
    dT.ab += ccpy_einsum("abej,ei->abij", H.ab.vvvo, T.a)
    dT.ab += ccpy_einsum("abie,ej->abij", H.ab.vvov, T.b)
    # Need iterative Fock terms in this scheme
    dT.ab -= ccpy_einsum("mi,abmj->abij", H0.a.oo, T.ab)
    dT.ab -= ccpy_einsum("mj,abim->abij", H0.b.oo, T.ab)
    dT.ab += ccpy_einsum("ae,ebij->abij", H0.a.vv, T.ab)
    dT.ab += ccpy_einsum("be,aeij->abij", H0.b.vv, T.ab)
    T.ab, dT.ab = cc_loops2.update_t2b(
        T.ab, dT.ab + H0.ab.vvoo, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv, shift
    )
    return T, dT

def update_t2c(T, dT, H, H0, shift):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2))_C|0>.
    """
    dT.bb = -0.5 * ccpy_einsum("amij,bm->abij", H.bb.vooo, T.b)
    dT.bb += 0.5 * ccpy_einsum("abie,ej->abij", H.bb.vvov, T.b)
    # Need iterative Fock terms in this scheme
    dT.bb -= 0.5 * ccpy_einsum("mi,abmj->abij", H0.b.oo, T.bb)
    dT.bb += 0.5 * ccpy_einsum("ae,ebij->abij", H0.b.vv, T.bb)
    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H0.bb.vvoo, H0.b.oo, H0.b.vv, shift
    )
    return T, dT
