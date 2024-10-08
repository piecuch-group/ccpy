"""
Module with functions that help perform the approximate coupled-cluster (CC) method
with singles and doubles, where the T2 operator is correct through 1st order of 
perturbation theory, abbreviated as CC2.
"""
import numpy as np

from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.lib.core import cc_loops2

def update(T, dT, H, X, shift, flag_RHF, system):

    # pre-CCS intermediates
    X = get_pre_ccs_intermediates(X, T, H, system, flag_RHF)

    # update T1
    T, dT = update_t1a(T, dT, H, X, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        T, dT = update_t1b(T, dT, H, X, shift)

    # CCS intermediates
    X = get_ccs_intermediates_opt(X, T, H, system, flag_RHF)

    # Add parts needed to make vvvv terms work out
    X.aa.vvov += 0.5 * np.einsum("abef,ei->abif", H.aa.vvvv, T.a, optimize=True)
    X.ab.vvov += np.einsum("abef,ei->abif", H.ab.vvvv, T.a, optimize=True)
    X.bb.vvov += 0.5 * np.einsum("abef,ei->abif", H.bb.vvvv, T.b, optimize=True)

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
    T.a, dT.a = cc_loops2.update_t1a(
        T.a, dT.a + H.a.vo, H.a.oo, H.a.vv, shift
    )
    return T, dT

def update_t1b(T, dT, H, X, shift):
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
    T.b, dT.b = cc_loops2.update_t1b(
        T.b, dT.b + H.b.vo, H.b.oo, H.b.vv, shift
    )
    return T, dT

def update_t2a(T, dT, H, H0, shift):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2))_C|0>.
    """
    dT.aa = -0.5 * np.einsum("amij,bm->abij", H.aa.vooo, T.a, optimize=True)
    dT.aa += 0.5 * np.einsum("abie,ej->abij", H.aa.vvov, T.a, optimize=True)
    # Need iterative Fock terms in this scheme
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", H0.a.oo, T.aa, optimize=True)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", H0.a.vv, T.aa, optimize=True)
    T.aa, dT.aa = cc_loops2.update_t2a(
        T.aa, dT.aa + 0.25 * H0.aa.vvoo, H0.a.oo, H0.a.vv, shift
    )
    return T, dT

def update_t2b(T, dT, H, H0, shift):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2))_C|0>.
    """
    dT.ab = -np.einsum("mbij,am->abij", H.ab.ovoo, T.a, optimize=True)
    dT.ab -= np.einsum("amij,bm->abij", H.ab.vooo, T.b, optimize=True)
    dT.ab += np.einsum("abej,ei->abij", H.ab.vvvo, T.a, optimize=True)
    dT.ab += np.einsum("abie,ej->abij", H.ab.vvov, T.b, optimize=True)
    # Need iterative Fock terms in this scheme
    dT.ab -= np.einsum("mi,abmj->abij", H0.a.oo, T.ab, optimize=True)
    dT.ab -= np.einsum("mj,abim->abij", H0.b.oo, T.ab, optimize=True)
    dT.ab += np.einsum("ae,ebij->abij", H0.a.vv, T.ab, optimize=True)
    dT.ab += np.einsum("be,aeij->abij", H0.b.vv, T.ab, optimize=True)
    T.ab, dT.ab = cc_loops2.update_t2b(
        T.ab, dT.ab + H0.ab.vvoo, H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv, shift
    )
    return T, dT

def update_t2c(T, dT, H, H0, shift):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2))_C|0>.
    """
    dT.bb = -0.5 * np.einsum("amij,bm->abij", H.bb.vooo, T.b, optimize=True)
    dT.bb += 0.5 * np.einsum("abie,ej->abij", H.bb.vvov, T.b, optimize=True)
    # Need iterative Fock terms in this scheme
    dT.bb -= 0.5 * np.einsum("mi,abmj->abij", H0.b.oo, T.bb, optimize=True)
    dT.bb += 0.5 * np.einsum("ae,ebij->abij", H0.b.vv, T.bb, optimize=True)
    T.bb, dT.bb = cc_loops2.update_t2c(
        T.bb, dT.bb + 0.25 * H0.bb.vvoo, H0.b.oo, H0.b.vv, shift
    )
    return T, dT
