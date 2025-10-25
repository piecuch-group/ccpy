import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import cc_loops2

def update(R, omega, H, RHF_symmetry, system):

    R.a, R.b, R.aa, R.ab, R.bb = cc_loops2.update_r(
        R.a,
        R.b,
        R.aa,
        R.ab,
        R.bb,
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
        0.0,
    )
    if RHF_symmetry:
        R.b = R.a.copy()
        R.bb = R.aa.copy()
    return R

def HR(dR, R, T, H, flag_RHF, system):

    # update R1
    dR.a = build_HR_1A(R, H)
    if flag_RHF:
        dR.b = dR.a.copy()
    else:
        dR.b = build_HR_1B(R, H)
    # update R2
    dR.aa = build_HR_2A(R, T, H)
    dR.ab = build_HR_2B(R, T, H)
    if flag_RHF:
        dR.bb = dR.aa.copy()
    else:
        dR.bb = build_HR_2C(R, T, H)
    return dR.flatten()

def build_HR_1A(R, H):
    # < ia | [H(2)*(R1+R2)]_C | 0 >
    X1A = -ccpy_einsum("mi,am->ai", H.a.oo, R.a)
    X1A += ccpy_einsum("ae,ei->ai", H.a.vv, R.a)
    X1A += ccpy_einsum("amie,em->ai", H.aa.voov, R.a)
    X1A += ccpy_einsum("amie,em->ai", H.ab.voov, R.b)
    X1A -= 0.5 * ccpy_einsum("mnif,afmn->ai", H.aa.ooov, R.aa)
    X1A -= ccpy_einsum("mnif,afmn->ai", H.ab.ooov, R.ab)
    X1A += 0.5 * ccpy_einsum("anef,efin->ai", H.aa.vovv, R.aa)
    X1A += ccpy_einsum("anef,efin->ai", H.ab.vovv, R.ab)
    X1A += ccpy_einsum("me,aeim->ai", H.a.ov, R.aa)
    X1A += ccpy_einsum("me,aeim->ai", H.b.ov, R.ab)
    return X1A

def build_HR_1B(R, H):
    # < i~a~ | [H(2)*(R1+R2)]_C | 0 >
    X1B = -ccpy_einsum("mi,am->ai", H.b.oo, R.b)
    X1B += ccpy_einsum("ae,ei->ai", H.b.vv, R.b)
    X1B += ccpy_einsum("maei,em->ai", H.ab.ovvo, R.a)
    X1B += ccpy_einsum("amie,em->ai", H.bb.voov, R.b)
    X1B -= ccpy_einsum("nmfi,fanm->ai", H.ab.oovo, R.ab)
    X1B -= 0.5 * ccpy_einsum("mnif,afmn->ai", H.bb.ooov, R.bb)
    X1B += ccpy_einsum("nafe,feni->ai", H.ab.ovvv, R.ab)
    X1B += 0.5 * ccpy_einsum("anef,efin->ai", H.bb.vovv, R.bb)
    X1B += ccpy_einsum("me,eami->ai", H.a.ov, R.ab)
    X1B += ccpy_einsum("me,aeim->ai", H.b.ov, R.bb)
    return X1B

def build_HR_2A(R, T, H):
    # < ijab | [H(2)*(R1+R2)]_C | 0 >
    X2A = -0.5 * ccpy_einsum("mi,abmj->abij", H.a.oo, R.aa)  # A(ij)
    X2A += 0.5 * ccpy_einsum("ae,ebij->abij", H.a.vv, R.aa)  # A(ab)
    X2A -= 0.5 * ccpy_einsum("bmji,am->abij", H.aa.vooo, R.a)  # A(ab)
    X2A += 0.5 * ccpy_einsum("baje,ei->abij", H.aa.vvov, R.a)  # A(ij)
    X2A -= np.transpose(X2A, (1, 0, 2, 3)) # antisymmetrize (ab)
    X2A -= np.transpose(X2A, (0, 1, 3, 2)) # antisymmetrize (ij)
    return X2A

def build_HR_2B(R, T, H):
    
    X2B = ccpy_einsum("ae,ebij->abij", H.a.vv, R.ab)
    X2B += ccpy_einsum("be,aeij->abij", H.b.vv, R.ab)
    X2B -= ccpy_einsum("mi,abmj->abij", H.a.oo, R.ab)
    X2B -= ccpy_einsum("mj,abim->abij", H.b.oo, R.ab)
    X2B += ccpy_einsum("abej,ei->abij", H.ab.vvvo, R.a)
    X2B += ccpy_einsum("abie,ej->abij", H.ab.vvov, R.b)
    X2B -= ccpy_einsum("mbij,am->abij", H.ab.ovoo, R.a)
    X2B -= ccpy_einsum("amij,bm->abij", H.ab.vooo, R.b)
    return X2B

def build_HR_2C(R, T, H):

    X2C = -0.5 * ccpy_einsum("mi,abmj->abij", H.b.oo, R.bb)  # A(ij)
    X2C += 0.5 * ccpy_einsum("ae,ebij->abij", H.b.vv, R.bb)  # A(ab)
    X2C -= 0.5 * ccpy_einsum("bmji,am->abij", H.bb.vooo, R.b)  # A(ab)
    X2C += 0.5 * ccpy_einsum("baje,ei->abij", H.bb.vvov, R.b)  # A(ij)
    X2C -= np.transpose(X2C, (1, 0, 2, 3)) # antisymmetrize (ab)
    X2C -= np.transpose(X2C, (0, 1, 3, 2)) # antisymmetrize (ij)
    return X2C

