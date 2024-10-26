import numpy as np
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
    dT.a = -np.einsum("mi,am->ai", H.a.oo, T1.a, optimize=True)
    dT.a += np.einsum("ae,ei->ai", H.a.vv, T1.a, optimize=True)
    dT.a += np.einsum("amie,em->ai", H.aa.voov, T1.a, optimize=True)
    dT.a += np.einsum("amie,em->ai", H.ab.voov, T1.b, optimize=True)
    dT.a -= 0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, T1.aa, optimize=True)
    dT.a -= np.einsum("mnif,afmn->ai", H.ab.ooov, T1.ab, optimize=True)
    dT.a += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, T1.aa, optimize=True)
    dT.a += np.einsum("anef,efin->ai", H.ab.vovv, T1.ab, optimize=True)
    dT.a += np.einsum("me,aeim->ai", H.a.ov, T1.aa, optimize=True)
    dT.a += np.einsum("me,aeim->ai", H.b.ov, T1.ab, optimize=True)
    # < ia | WBar | 0 >
    I1A_oo = W.a.oo + np.einsum("me,ei->mi", W.a.ov, T.a, optimize=True)
    dT.a -= np.einsum("mi,am->ai", I1A_oo, T.a, optimize=True)
    dT.a += np.einsum("ae,ei->ai", W.a.vv, T.a, optimize=True)
    dT.a += np.einsum("me,aeim->ai", W.a.ov, T.aa, optimize=True)
    dT.a += np.einsum("me,aeim->ai", W.b.ov, T.ab, optimize=True)
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
    dT.b = -np.einsum("mi,am->ai", H.b.oo, T1.b, optimize=True)
    dT.b += np.einsum("ae,ei->ai", H.b.vv, T1.b, optimize=True)
    dT.b += np.einsum("maei,em->ai", H.ab.ovvo, T1.a, optimize=True)
    dT.b += np.einsum("amie,em->ai", H.bb.voov, T1.b, optimize=True)
    dT.b -= np.einsum("nmfi,fanm->ai", H.ab.oovo, T1.ab, optimize=True)
    dT.b -= 0.5 * np.einsum("mnif,afmn->ai", H.bb.ooov, T1.bb, optimize=True)
    dT.b += np.einsum("nafe,feni->ai", H.ab.ovvv, T1.ab, optimize=True)
    dT.b += 0.5 * np.einsum("anef,efin->ai", H.bb.vovv, T1.bb, optimize=True)
    dT.b += np.einsum("me,eami->ai", H.a.ov, T1.ab, optimize=True)
    dT.b += np.einsum("me,aeim->ai", H.b.ov, T1.bb, optimize=True)
    # < i~a~ | WBar | 0 >
    I1B_oo = W.b.oo + np.einsum("me,ei->mi", W.b.ov, T.b, optimize=True)
    dT.b -= np.einsum("mi,am->ai", I1B_oo, T.b, optimize=True)
    dT.b += np.einsum("ae,ei->ai", W.b.vv, T.b, optimize=True)
    dT.b += np.einsum("me,aeim->ai", W.b.ov, T.bb, optimize=True)
    dT.b += np.einsum("me,eami->ai", W.a.ov, T.ab, optimize=True)
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
    dT.aa = -0.5 * np.einsum("mi,abmj->abij", H.a.oo, T1.aa, optimize=True)  # A(ij)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", H.a.vv, T1.aa, optimize=True)  # A(ab)
    dT.aa += 0.125 * np.einsum("mnij,abmn->abij", H.aa.oooo, T1.aa, optimize=True)
    dT.aa += 0.125 * np.einsum("abef,efij->abij", H.aa.vvvv, T1.aa, optimize=True)
    dT.aa += np.einsum("amie,ebmj->abij", H.aa.voov, T1.aa, optimize=True)  # A(ij)A(ab)
    dT.aa += np.einsum("amie,bejm->abij", H.ab.voov, T1.ab, optimize=True)  # A(ij)A(ab)
    dT.aa -= 0.5 * np.einsum("bmji,am->abij", H.aa.vooo, T1.a, optimize=True)  # A(ab)
    dT.aa += 0.5 * np.einsum("baje,ei->abij", H.aa.vvov, T1.a, optimize=True)  # A(ij)
    dT.aa += 0.5 * np.einsum("be,aeij->abij", X.a.vv, T.aa, optimize=True)  # A(ab)
    dT.aa -= 0.5 * np.einsum("mj,abim->abij", X.a.oo, T.aa, optimize=True)  # A(ij)
    # < ijab | WBar | 0 >
    I1A_oo = W.a.oo + np.einsum("me,ei->mi", W.a.ov, T.a, optimize=True)
    I1A_vv = W.a.vv - np.einsum("me,am->ae", W.a.ov, T.a, optimize=True)
    dT.aa -= 0.5 * np.einsum("mi,abmj->abij", I1A_oo, T.aa, optimize=True)
    dT.aa += 0.5 * np.einsum("ae,ebij->abij", I1A_vv, T.aa, optimize=True)

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
    dT.ab = np.einsum("ae,ebij->abij", H.a.vv, T1.ab, optimize=True)
    dT.ab += np.einsum("be,aeij->abij", H.b.vv, T1.ab, optimize=True)
    dT.ab -= np.einsum("mi,abmj->abij", H.a.oo, T1.ab, optimize=True)
    dT.ab -= np.einsum("mj,abim->abij", H.b.oo, T1.ab, optimize=True)
    dT.ab += np.einsum("mnij,abmn->abij", H.ab.oooo, T1.ab, optimize=True)
    dT.ab += np.einsum("abef,efij->abij", H.ab.vvvv, T1.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", H.aa.voov, T1.ab, optimize=True)
    dT.ab += np.einsum("amie,ebmj->abij", H.ab.voov, T1.bb, optimize=True)
    dT.ab += np.einsum("mbej,aeim->abij", H.ab.ovvo, T1.aa, optimize=True)
    dT.ab += np.einsum("bmje,aeim->abij", H.bb.voov, T1.ab, optimize=True)
    dT.ab -= np.einsum("mbie,aemj->abij", H.ab.ovov, T1.ab, optimize=True)
    dT.ab -= np.einsum("amej,ebim->abij", H.ab.vovo, T1.ab, optimize=True)
    dT.ab += np.einsum("abej,ei->abij", H.ab.vvvo, T1.a, optimize=True)
    dT.ab += np.einsum("abie,ej->abij", H.ab.vvov, T1.b, optimize=True)
    dT.ab -= np.einsum("mbij,am->abij", H.ab.ovoo, T1.a, optimize=True)
    dT.ab -= np.einsum("amij,bm->abij", H.ab.vooo, T1.b, optimize=True)
    dT.ab += np.einsum("ae,ebij->abij", X.a.vv, T.ab, optimize=True)
    dT.ab -= np.einsum("mi,abmj->abij", X.a.oo, T.ab, optimize=True)
    dT.ab += np.einsum("be,aeij->abij", X.b.vv, T.ab, optimize=True)
    dT.ab -= np.einsum("mj,abim->abij", X.b.oo, T.ab, optimize=True)
    # < ij~ab~ | WBar | 0 >
    I1A_oo = W.a.oo + np.einsum("me,ei->mi", W.a.ov, T.a, optimize=True)
    I1A_vv = W.a.vv - np.einsum("me,am->ae", W.a.ov, T.a, optimize=True)
    I1B_oo = W.a.oo + np.einsum("me,ei->mi", W.b.ov, T.b, optimize=True)
    I1B_vv = W.a.vv - np.einsum("me,am->ae", W.b.ov, T.b, optimize=True)
    dT.ab -= np.einsum("mi,abmj->abij", I1A_oo, T.ab, optimize=True)
    dT.ab += np.einsum("ae,ebij->abij", I1A_vv, T.ab, optimize=True)
    dT.ab -= np.einsum("mj,abim->abij", I1B_oo, T.ab, optimize=True)
    dT.ab += np.einsum("be,aeij->abij", I1B_vv, T.ab, optimize=True)

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
    dT.bb = -0.5 * np.einsum("mi,abmj->abij", H.b.oo, T1.bb, optimize=True)  # A(ij)
    dT.bb += 0.5 * np.einsum("ae,ebij->abij", H.b.vv, T1.bb, optimize=True)  # A(ab)
    dT.bb += 0.125 * np.einsum("mnij,abmn->abij", H.bb.oooo, T1.bb, optimize=True)
    dT.bb += 0.125 * np.einsum("abef,efij->abij", H.bb.vvvv, T1.bb, optimize=True)
    dT.bb += np.einsum("amie,ebmj->abij", H.bb.voov, T1.bb, optimize=True)  # A(ij)A(ab)
    dT.bb += np.einsum("maei,ebmj->abij", H.ab.ovvo, T1.ab, optimize=True)  # A(ij)A(ab)
    dT.bb -= 0.5 * np.einsum("bmji,am->abij", H.bb.vooo, T1.b, optimize=True)  # A(ab)
    dT.bb += 0.5 * np.einsum("baje,ei->abij", H.bb.vvov, T1.b, optimize=True)  # A(ij)
    dT.bb += 0.5 * np.einsum("be,aeij->abij", X.b.vv, T.bb, optimize=True)  # A(ab)
    dT.bb -= 0.5 * np.einsum("mj,abim->abij", X.b.oo, T.bb, optimize=True)  # A(ij)
    # < i~j~a~b~ | WBar | 0 >
    I1B_oo = W.b.oo + np.einsum("me,ei->mi", W.b.ov, T.b, optimize=True)
    I1B_vv = W.b.vv - np.einsum("me,am->ae", W.b.ov, T.b, optimize=True)
    dT.bb -= 0.5 * np.einsum("mi,abmj->abij", I1B_oo, T.bb, optimize=True)
    dT.bb += 0.5 * np.einsum("ae,ebij->abij", I1B_vv, T.bb, optimize=True)

    T1.bb, dT.bb = cc_loops2.update_t2c(
        T1.bb, dT.bb, H.b.oo, H.b.vv, shift
    )
    return T1, dT
