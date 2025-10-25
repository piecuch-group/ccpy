import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lrcc.lrccsd_intermediates import get_lrccsd_intermediates
from ccpy.lib.core import cc_loops2

def update(T1, dT, T, W, H, X, shift, flag_RHF, system):

    X = get_lrccsd_intermediates(X, H, T1, system)

    # update T1
    T1, dT = update_t1a(T1, dT, T, W, H, shift)
    if flag_RHF:
        T1.b = T1.a.copy()
        dT.b = dT.a.copy()
    else:
        T1, dT = update_t1b(T1, dT, T, W, H, shift)

    # update T2
    T1, dT = update_t2a(T1, dT, T, W, X, H, shift)
    T1, dT = update_t2b(T1, dT, T, W, X, H, shift)
    if flag_RHF:
        T1.bb = T1.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        T1, dT = update_t2c(T1, dT, T, W, X, H, shift)

    return T1, dT


def update_t1a(T1, dT, T, W, H, shift):
    """
    Update t1a amplitudes by calculating the projection <ia|(H_N e^(T1+T2))_C|0>.
    """
    # < ia | (HBar*T1)_C | 0 >
    dT.a = -ccpy_einsum("mi,am->ai", H.a.oo, T1.a)
    dT.a += ccpy_einsum("ae,ei->ai", H.a.vv, T1.a)
    dT.a += ccpy_einsum("amie,em->ai", H.aa.voov, T1.a)
    dT.a += ccpy_einsum("amie,em->ai", H.ab.voov, T1.b)
    dT.a -= 0.5 * ccpy_einsum("mnif,afmn->ai", H.aa.ooov, T1.aa)
    dT.a -= ccpy_einsum("mnif,afmn->ai", H.ab.ooov, T1.ab)
    dT.a += 0.5 * ccpy_einsum("anef,efin->ai", H.aa.vovv, T1.aa)
    dT.a += ccpy_einsum("anef,efin->ai", H.ab.vovv, T1.ab)
    dT.a += ccpy_einsum("me,aeim->ai", H.a.ov, T1.aa)
    dT.a += ccpy_einsum("me,aeim->ai", H.b.ov, T1.ab)
    # < ia | WBar | 0 >
    I1A_oo = W.a.oo + ccpy_einsum("me,ei->mi", W.a.ov, T.a)
    dT.a -= ccpy_einsum("mi,am->ai", I1A_oo, T.a)
    dT.a += ccpy_einsum("ae,ei->ai", W.a.vv, T.a)
    dT.a += ccpy_einsum("me,aeim->ai", W.a.ov, T.aa)
    dT.a += ccpy_einsum("me,aeim->ai", W.b.ov, T.ab)
    T1.a, dT.a = cc_loops2.update_t1a(
        T1.a, dT.a + W.a.vo, H.a.oo, H.a.vv, shift
    )
    return T1, dT


# @profile
def update_t1b(T1, dT, T, W, H, shift):
    """
    Update t1b amplitudes by calculating the projection <i~a~|(H_N e^(T1+T2))_C|0>.
    """
    # < i~a~ | (HBar*T1)_C | 0 >
    dT.b = -ccpy_einsum("mi,am->ai", H.b.oo, T1.b)
    dT.b += ccpy_einsum("ae,ei->ai", H.b.vv, T1.b)
    dT.b += ccpy_einsum("maei,em->ai", H.ab.ovvo, T1.a)
    dT.b += ccpy_einsum("amie,em->ai", H.bb.voov, T1.b)
    dT.b -= ccpy_einsum("nmfi,fanm->ai", H.ab.oovo, T1.ab)
    dT.b -= 0.5 * ccpy_einsum("mnif,afmn->ai", H.bb.ooov, T1.bb)
    dT.b += ccpy_einsum("nafe,feni->ai", H.ab.ovvv, T1.ab)
    dT.b += 0.5 * ccpy_einsum("anef,efin->ai", H.bb.vovv, T1.bb)
    dT.b += ccpy_einsum("me,eami->ai", H.a.ov, T1.ab)
    dT.b += ccpy_einsum("me,aeim->ai", H.b.ov, T1.bb)
    # < i~a~ | WBar | 0 >
    I1B_oo = W.b.oo + ccpy_einsum("me,ei->mi", W.b.ov, T.b)
    dT.b -= ccpy_einsum("mi,am->ai", I1B_oo, T.b)
    dT.b += ccpy_einsum("ae,ei->ai", W.b.vv, T.b)
    dT.b += ccpy_einsum("me,aeim->ai", W.b.ov, T.bb)
    dT.b += ccpy_einsum("me,eami->ai", W.a.ov, T.ab)
    T1.b, dT.b = cc_loops2.update_t1b(
        T1.b, dT.b + W.b.vo, H.b.oo, H.b.vv, shift
    )
    return T1, dT


# @profile
def update_t2a(T1, dT, T, W, X, H, shift):
    """
    Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2))_C|0>.
    """
    # < ijab | (HBar*T1)_C | 0 >
    dT.aa = -0.5 * ccpy_einsum("mi,abmj->abij", H.a.oo, T1.aa)  # A(ij)
    dT.aa += 0.5 * ccpy_einsum("ae,ebij->abij", H.a.vv, T1.aa)  # A(ab)
    dT.aa += 0.125 * ccpy_einsum("mnij,abmn->abij", H.aa.oooo, T1.aa)
    dT.aa += 0.125 * ccpy_einsum("abef,efij->abij", H.aa.vvvv, T1.aa)
    dT.aa += ccpy_einsum("amie,ebmj->abij", H.aa.voov, T1.aa)  # A(ij)A(ab)
    dT.aa += ccpy_einsum("amie,bejm->abij", H.ab.voov, T1.ab)  # A(ij)A(ab)
    dT.aa -= 0.5 * ccpy_einsum("bmji,am->abij", H.aa.vooo, T1.a)  # A(ab)
    dT.aa += 0.5 * ccpy_einsum("baje,ei->abij", H.aa.vvov, T1.a)  # A(ij)
    dT.aa += 0.5 * ccpy_einsum("be,aeij->abij", X.a.vv, T.aa)  # A(ab)
    dT.aa -= 0.5 * ccpy_einsum("mj,abim->abij", X.a.oo, T.aa)  # A(ij)
    # < ijab | WBar | 0 >
    I1A_oo = W.a.oo + ccpy_einsum("me,ei->mi", W.a.ov, T.a)
    I1A_vv = W.a.vv - ccpy_einsum("me,am->ae", W.a.ov, T.a)
    dT.aa -= 0.5 * ccpy_einsum("mi,abmj->abij", I1A_oo, T.aa)
    dT.aa += 0.5 * ccpy_einsum("ae,ebij->abij", I1A_vv, T.aa)

    T1.aa, dT.aa = cc_loops2.update_t2a(
        T1.aa, dT.aa, H.a.oo, H.a.vv, shift
    )
    return T1, dT


# @profile
def update_t2b(T1, dT, T, W, X, H, shift):
    """
    Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2))_C|0>.
    """
    # < ij~ab~ | (HBar*T1)_C | 0 >
    dT.ab = ccpy_einsum("ae,ebij->abij", H.a.vv, T1.ab)
    dT.ab += ccpy_einsum("be,aeij->abij", H.b.vv, T1.ab)
    dT.ab -= ccpy_einsum("mi,abmj->abij", H.a.oo, T1.ab)
    dT.ab -= ccpy_einsum("mj,abim->abij", H.b.oo, T1.ab)
    dT.ab += ccpy_einsum("mnij,abmn->abij", H.ab.oooo, T1.ab)
    dT.ab += ccpy_einsum("abef,efij->abij", H.ab.vvvv, T1.ab)
    dT.ab += ccpy_einsum("amie,ebmj->abij", H.aa.voov, T1.ab)
    dT.ab += ccpy_einsum("amie,ebmj->abij", H.ab.voov, T1.bb)
    dT.ab += ccpy_einsum("mbej,aeim->abij", H.ab.ovvo, T1.aa)
    dT.ab += ccpy_einsum("bmje,aeim->abij", H.bb.voov, T1.ab)
    dT.ab -= ccpy_einsum("mbie,aemj->abij", H.ab.ovov, T1.ab)
    dT.ab -= ccpy_einsum("amej,ebim->abij", H.ab.vovo, T1.ab)
    dT.ab += ccpy_einsum("abej,ei->abij", H.ab.vvvo, T1.a)
    dT.ab += ccpy_einsum("abie,ej->abij", H.ab.vvov, T1.b)
    dT.ab -= ccpy_einsum("mbij,am->abij", H.ab.ovoo, T1.a)
    dT.ab -= ccpy_einsum("amij,bm->abij", H.ab.vooo, T1.b)
    dT.ab += ccpy_einsum("ae,ebij->abij", X.a.vv, T.ab)
    dT.ab -= ccpy_einsum("mi,abmj->abij", X.a.oo, T.ab)
    dT.ab += ccpy_einsum("be,aeij->abij", X.b.vv, T.ab)
    dT.ab -= ccpy_einsum("mj,abim->abij", X.b.oo, T.ab)
    # < ij~ab~ | WBar | 0 >
    I1A_oo = W.a.oo + ccpy_einsum("me,ei->mi", W.a.ov, T.a)
    I1A_vv = W.a.vv - ccpy_einsum("me,am->ae", W.a.ov, T.a)
    I1B_oo = W.a.oo + ccpy_einsum("me,ei->mi", W.b.ov, T.b)
    I1B_vv = W.a.vv - ccpy_einsum("me,am->ae", W.b.ov, T.b)
    dT.ab -= ccpy_einsum("mi,abmj->abij", I1A_oo, T.ab)
    dT.ab += ccpy_einsum("ae,ebij->abij", I1A_vv, T.ab)
    dT.ab -= ccpy_einsum("mj,abim->abij", I1B_oo, T.ab)
    dT.ab += ccpy_einsum("be,aeij->abij", I1B_vv, T.ab)

    T1.ab, dT.ab = cc_loops2.update_t2b(
        T1.ab, dT.ab, H.a.oo, H.a.vv, H.b.oo, H.b.vv, shift
    )
    return T1, dT


# @profile
def update_t2c(T1, dT, T, W, X, H, shift):
    """
    Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2))_C|0>.
    """
    # < i~j~a~b~ | (HBar*T1)_C | 0 >
    dT.bb = -0.5 * ccpy_einsum("mi,abmj->abij", H.b.oo, T1.bb)  # A(ij)
    dT.bb += 0.5 * ccpy_einsum("ae,ebij->abij", H.b.vv, T1.bb)  # A(ab)
    dT.bb += 0.125 * ccpy_einsum("mnij,abmn->abij", H.bb.oooo, T1.bb)
    dT.bb += 0.125 * ccpy_einsum("abef,efij->abij", H.bb.vvvv, T1.bb)
    dT.bb += ccpy_einsum("amie,ebmj->abij", H.bb.voov, T1.bb)  # A(ij)A(ab)
    dT.bb += ccpy_einsum("maei,ebmj->abij", H.ab.ovvo, T1.ab)  # A(ij)A(ab)
    dT.bb -= 0.5 * ccpy_einsum("bmji,am->abij", H.bb.vooo, T1.b)  # A(ab)
    dT.bb += 0.5 * ccpy_einsum("baje,ei->abij", H.bb.vvov, T1.b)  # A(ij)
    dT.bb += 0.5 * ccpy_einsum("be,aeij->abij", X.b.vv, T.bb)  # A(ab)
    dT.bb -= 0.5 * ccpy_einsum("mj,abim->abij", X.b.oo, T.bb)  # A(ij)
    # < i~j~a~b~ | WBar | 0 >
    I1B_oo = W.b.oo + ccpy_einsum("me,ei->mi", W.b.ov, T.b)
    I1B_vv = W.b.vv - ccpy_einsum("me,am->ae", W.b.ov, T.b)
    dT.bb -= 0.5 * ccpy_einsum("mi,abmj->abij", I1B_oo, T.bb)
    dT.bb += 0.5 * ccpy_einsum("ae,ebij->abij", I1B_vv, T.bb)

    T1.bb, dT.bb = cc_loops2.update_t2c(
        T1.bb, dT.bb, H.b.oo, H.b.vv, shift
    )
    return T1, dT
