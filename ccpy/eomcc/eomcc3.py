'''
Approximate Equation-of-Motion Coupled-Cluster Method with Triple Excitations (CC3)
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.eomcc.eomcc3_intermediates import get_HR1_intermediates, get_eomccsd_intermediates
from ccpy.lib.core import cc3_loops

def update(R, omega, fock, RHF_symmetry, system):
    R.a, R.b, R.aa, R.ab, R.bb = cc3_loops.update_r(
        R.a,
        R.b,
        R.aa,
        R.ab,
        R.bb,
        omega,
        fock.a.oo,
        fock.a.vv,
        fock.b.oo,
        fock.b.vv,
    )
    if RHF_symmetry:
        R.b = R.a.copy()
        R.bb = R.aa.copy()
    return R

def HR(dR, R, T, H, H1, fock, omega, flag_RHF, system):

    # Get CCS-like intermediates for R3 contractions
    HR1 = get_HR1_intermediates(H1, R, system)
    # Get H*R EOMCCSD intermediates
    X0 = get_eomccsd_intermediates(H, R, system)
    # Compute EOMCCSD parts of R1
    dR.a = build_HR_1A(R, H)
    if flag_RHF:
        dR.b = dR.a.copy()
    else:
        dR.b = build_HR_1B(R, H)
    # Compute EOMCCSD parts of R2
    dR.aa = build_HR_2A(R, T, H, X0)
    dR.ab = build_HR_2B(R, T, H, X0)
    if flag_RHF:
        dR.bb = dR.aa.copy()
    else:
        dR.bb = build_HR_2C(R, T, H, X0)
    # Compute parts of R1 and R2 equations that involve T3 and R3 on-the-fly and add to dR
    dR.a, dR.b, dR.aa, dR.ab, dR.bb = cc3_loops.build_hr(
            dR.a, dR.b, dR.aa, dR.ab, dR.bb,
            T.aa, T.ab, T.bb, R.aa, R.ab, R.bb,
            fock.a.oo, fock.a.vv, fock.b.oo, fock.b.vv,
            H.a.ov, H.b.ov,
            H.aa.oovv, H.ab.oovv, H.bb.oovv,
            H.aa.ooov, H.aa.vovv,
            H.ab.ooov, H.ab.oovo, H.ab.vovv, H.ab.ovvv,
            H.bb.ooov, H.bb.vovv,
            H1.aa.vooo, H1.aa.vvov,
            H1.ab.vooo, H1.ab.ovoo, H1.ab.vvov, H1.ab.vvvo,
            H1.bb.vooo, H1.bb.vvov,
            X0.a.ov, X0.b.ov,
            HR1.aa.vooo, HR1.aa.vvov,
            HR1.ab.vooo, HR1.ab.ovoo, HR1.ab.vvov, HR1.ab.vvvo,
            HR1.bb.vooo, HR1.bb.vvov,
            omega,
    )
    return dR.flatten()

def build_HR_1A(R, H):
    """< ia | [H(2)*(R1+R2+R3)]_C | 0 >"""
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
    """< i~a~ | [H(2)*(R1+R2+R3)]_C | 0 >"""
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

def build_HR_2A(R, T, H, X):
    """ < ijab | [H(2)*(R1+R2+R3)]_C | 0 > """
    X2A = -0.5 * ccpy_einsum("mi,abmj->abij", H.a.oo, R.aa)  # A(ij)
    X2A += 0.5 * ccpy_einsum("ae,ebij->abij", H.a.vv, R.aa)  # A(ab)
    X2A += 0.125 * ccpy_einsum("mnij,abmn->abij", H.aa.oooo, R.aa)
    X2A += 0.125 * ccpy_einsum("abef,efij->abij", H.aa.vvvv, R.aa)
    X2A += ccpy_einsum("amie,ebmj->abij", H.aa.voov, R.aa)  # A(ij)A(ab)
    X2A += ccpy_einsum("amie,bejm->abij", H.ab.voov, R.ab)  # A(ij)A(ab)
    X2A -= 0.5 * ccpy_einsum("bmji,am->abij", H.aa.vooo, R.a)  # A(ab)
    X2A += 0.5 * ccpy_einsum("baje,ei->abij", H.aa.vvov, R.a)  # A(ij)
    X2A += 0.5 * ccpy_einsum("be,aeij->abij", X.a.vv, T.aa)  # A(ab)
    X2A -= 0.5 * ccpy_einsum("mj,abim->abij", X.a.oo, T.aa)  # A(ij)
    return X2A

def build_HR_2B(R, T, H, X):
    """< ij~ab~ | [H(2)*(R1+R2+R3)]_C | 0 >"""
    X2B = ccpy_einsum("ae,ebij->abij", H.a.vv, R.ab)
    X2B += ccpy_einsum("be,aeij->abij", H.b.vv, R.ab)
    X2B -= ccpy_einsum("mi,abmj->abij", H.a.oo, R.ab)
    X2B -= ccpy_einsum("mj,abim->abij", H.b.oo, R.ab)
    X2B += ccpy_einsum("mnij,abmn->abij", H.ab.oooo, R.ab)
    X2B += ccpy_einsum("abef,efij->abij", H.ab.vvvv, R.ab)
    X2B += ccpy_einsum("amie,ebmj->abij", H.aa.voov, R.ab)
    X2B += ccpy_einsum("amie,ebmj->abij", H.ab.voov, R.bb)
    X2B += ccpy_einsum("mbej,aeim->abij", H.ab.ovvo, R.aa)
    X2B += ccpy_einsum("bmje,aeim->abij", H.bb.voov, R.ab)
    X2B -= ccpy_einsum("mbie,aemj->abij", H.ab.ovov, R.ab)
    X2B -= ccpy_einsum("amej,ebim->abij", H.ab.vovo, R.ab)
    X2B += ccpy_einsum("abej,ei->abij", H.ab.vvvo, R.a)
    X2B += ccpy_einsum("abie,ej->abij", H.ab.vvov, R.b)
    X2B -= ccpy_einsum("mbij,am->abij", H.ab.ovoo, R.a)
    X2B -= ccpy_einsum("amij,bm->abij", H.ab.vooo, R.b)
    X2B += ccpy_einsum("ae,ebij->abij", X.a.vv, T.ab)
    X2B -= ccpy_einsum("mi,abmj->abij", X.a.oo, T.ab)
    X2B += ccpy_einsum("be,aeij->abij", X.b.vv, T.ab)
    X2B -= ccpy_einsum("mj,abim->abij", X.b.oo, T.ab)
    return X2B

def build_HR_2C(R, T, H, X):
    """< i~j~a~b~ | [H(2)*(R1+R2+R3)]_C | 0 >"""
    X2C = -0.5 * ccpy_einsum("mi,abmj->abij", H.b.oo, R.bb)  # A(ij)
    X2C += 0.5 * ccpy_einsum("ae,ebij->abij", H.b.vv, R.bb)  # A(ab)
    X2C += 0.125 * ccpy_einsum("mnij,abmn->abij", H.bb.oooo, R.bb)
    X2C += 0.125 * ccpy_einsum("abef,efij->abij", H.bb.vvvv, R.bb)
    X2C += ccpy_einsum("amie,ebmj->abij", H.bb.voov, R.bb)  # A(ij)A(ab)
    X2C += ccpy_einsum("maei,ebmj->abij", H.ab.ovvo, R.ab)  # A(ij)A(ab)
    X2C -= 0.5 * ccpy_einsum("bmji,am->abij", H.bb.vooo, R.b)  # A(ab)
    X2C += 0.5 * ccpy_einsum("baje,ei->abij", H.bb.vvov, R.b)  # A(ij)
    X2C += 0.5 * ccpy_einsum("be,aeij->abij", X.b.vv, T.bb)  # A(ab)
    X2C -= 0.5 * ccpy_einsum("mj,abim->abij", X.b.oo, T.bb)  # A(ij)
    return X2C

def _compute_r3a(R, T, H1, HR1, omega, fock):
    # <ijkabc| (H(1) * R2)_C | 0 >
    X3A = 0.25 * ccpy_einsum("baje,ecik->abcijk", H1.aa.vvov, R.aa) #
    X3A -= 0.25 * ccpy_einsum("bmji,acmk->abcijk", H1.aa.vooo, R.aa)
    # <ijkabc| ((H(1)*R1)_C * T2)_C | 0 >
    X3A += 0.25 * ccpy_einsum("baje,ecik->abcijk", HR1.aa.vvov, T.aa)
    X3A -= 0.25 * ccpy_einsum("bmji,acmk->abcijk", HR1.aa.vooo, T.aa)
    return cc3_loops.compute_r3a(X3A, omega, fock.a.oo, fock.a.vv)

def _compute_r3b(R, T, H1, HR1, omega, fock):
    # < ijk~abc~ | (H(1)*R2)_C | 0 >
    X3B = 0.5 * ccpy_einsum("bcek,aeij->abcijk", H1.ab.vvvo, R.aa)
    X3B -= 0.5 * ccpy_einsum("mcjk,abim->abcijk", H1.ab.ovoo, R.aa)
    X3B += 0.5 * ccpy_einsum("baje,ecik->abcijk", H1.aa.vvov, R.ab)
    X3B -= 0.5 * ccpy_einsum("bnji,acnk->abcijk", H1.aa.vooo, R.ab)
    X3B += ccpy_einsum("bcje,aeik->abcijk", H1.ab.vvov, R.ab)
    X3B -= ccpy_einsum("bnjk,acin->abcijk", H1.ab.vooo, R.ab)
    # < ijk~abc~ | ((H(1)*R1)_C * T2)_C | 0 >
    X3B += 0.5 * ccpy_einsum("bcek,aeij->abcijk", HR1.ab.vvvo, T.aa)
    X3B -= 0.5 * ccpy_einsum("ncjk,abin->abcijk", HR1.ab.ovoo, T.aa)
    X3B += 0.5 * ccpy_einsum("baje,ecik->abcijk", HR1.aa.vvov, T.ab)
    X3B -= 0.5 * ccpy_einsum("bnji,acnk->abcijk", HR1.aa.vooo, T.ab)
    X3B += ccpy_einsum("bcje,aeik->abcijk", HR1.ab.vvov, T.ab)
    X3B -= ccpy_einsum("bnjk,acin->abcijk", HR1.ab.vooo, T.ab)
    return cc3_loops.compute_r3b(X3B, omega, fock.a.oo, fock.a.vv, fock.b.oo, fock.b.vv)

def _compute_r3c(R, T, H1, HR1, omega, fock):
    # < ij~k~ab~c~ | (H(1)*R2)_C | 0 >
    X3C = 0.5 * ccpy_einsum("cbke,aeij->cbakji", H1.ab.vvov, R.bb)
    X3C -= 0.5 * ccpy_einsum("cmkj,abim->cbakji", H1.ab.vooo, R.bb)
    X3C += 0.5 * ccpy_einsum("baje,ceki->cbakji", H1.bb.vvov, R.ab)
    X3C -= 0.5 * ccpy_einsum("bnji,cakn->cbakji", H1.bb.vooo, R.ab)
    X3C += ccpy_einsum("cbej,eaki->cbakji", H1.ab.vvvo, R.ab)
    X3C -= ccpy_einsum("nbkj,cani->cbakji", H1.ab.ovoo, R.ab)
    # < ij~k~ab~c~ | ((H(1)*R1)_C * T2)_C | 0 >
    X3C += 0.5 * ccpy_einsum("cbke,aeij->cbakji", HR1.ab.vvov, T.bb)
    X3C -= 0.5 * ccpy_einsum("cnkj,abin->cbakji", HR1.ab.vooo, T.bb)
    X3C += 0.5 * ccpy_einsum("baje,ceki->cbakji", HR1.bb.vvov, T.ab)
    X3C -= 0.5 * ccpy_einsum("bnji,cakn->cbakji", HR1.bb.vooo, T.ab)
    X3C += ccpy_einsum("cbej,eaki->cbakji", HR1.ab.vvvo, T.ab)
    X3C -= ccpy_einsum("nbkj,cani->cbakji", HR1.ab.ovoo, T.ab)
    return cc3_loops.compute_r3c(X3C, omega, fock.a.oo, fock.a.vv, fock.b.oo, fock.b.vv)

def _compute_r3d(R, T, H1, HR1, omega, fock):
    # <i~j~k~a~b~c~| (H(1) * R2)_C | 0 >
    X3D = 0.25 * ccpy_einsum("baje,ecik->abcijk", H1.bb.vvov, R.bb)
    X3D -= 0.25 * ccpy_einsum("bmji,acmk->abcijk", H1.bb.vooo, R.bb)
    # <i~j~k~a~b~c~| ((H(1)*R1)_C * T2)_C | 0 >
    X3D += 0.25 * ccpy_einsum("baje,ecik->abcijk", HR1.bb.vvov, T.bb)
    X3D -= 0.25 * ccpy_einsum("bmji,acmk->abcijk", HR1.bb.vooo, T.bb)
    return cc3_loops.compute_r3d(X3D, omega, fock.b.oo, fock.b.vv)

def _compute_t3a(T, X, fock):
    x3a = -0.25 * ccpy_einsum("amij,bcmk->abcijk", X.aa.vooo, T.aa)
    x3a += 0.25 * ccpy_einsum("abie,ecjk->abcijk", X.aa.vvov, T.aa)
    return cc3_loops.compute_t3a(x3a, fock.a.oo, fock.a.vv)

def _compute_t3b(T, X, fock):
    x3b = 0.5 * ccpy_einsum("bcek,aeij->abcijk", X.ab.vvvo, T.aa)
    x3b -= 0.5 * ccpy_einsum("mcjk,abim->abcijk", X.ab.ovoo, T.aa)
    x3b += ccpy_einsum("acie,bejk->abcijk", X.ab.vvov, T.ab)
    x3b -= ccpy_einsum("amik,bcjm->abcijk", X.ab.vooo, T.ab)
    x3b += 0.5 * ccpy_einsum("abie,ecjk->abcijk", X.aa.vvov, T.ab)
    x3b -= 0.5 * ccpy_einsum("amij,bcmk->abcijk", X.aa.vooo, T.ab)
    return cc3_loops.compute_t3b(x3b, fock.a.oo, fock.a.vv, fock.b.oo, fock.b.vv)

def _compute_t3c(T, X, fock):
    x3c = 0.5 * ccpy_einsum("abie,ecjk->abcijk", X.ab.vvov, T.bb)
    x3c -= 0.5 * ccpy_einsum("amij,bcmk->abcijk", X.ab.vooo, T.bb)
    x3c += 0.5 * ccpy_einsum("cbke,aeij->abcijk", X.bb.vvov, T.ab)
    x3c -= 0.5 * ccpy_einsum("cmkj,abim->abcijk", X.bb.vooo, T.ab)
    x3c += ccpy_einsum("abej,ecik->abcijk", X.ab.vvvo, T.ab)
    x3c -= ccpy_einsum("mbij,acmk->abcijk", X.ab.ovoo, T.ab)
    return cc3_loops.compute_t3c(x3c, fock.a.oo, fock.a.vv, fock.b.oo, fock.b.vv)

def _compute_t3d(T, X, fock):
    x3d = -0.25 * ccpy_einsum("amij,bcmk->abcijk", X.bb.vooo, T.bb)
    x3d += 0.25 * ccpy_einsum("abie,ecjk->abcijk", X.bb.vvov, T.bb)
    return cc3_loops.compute_t3d(x3d, fock.b.oo, fock.b.vv)
