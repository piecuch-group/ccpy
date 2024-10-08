"""
Module with functions to perform the excited-state CC3 calculation to obtain 
right eigenvectors and corresponding eigenvalues, representing the vertical 
excitation energies, by diagonalization of the CC3 Jacobian using the 
equation-of-motion (EOM) formalism. This approach is equivalent to the 
traditional linear response framework for excitation energies.

References:
    [1] J. Chem. Phys. 106, 1808 (1997); doi: 10.1063/1.473322 [CC3 method]
    [2] J. Chem. Phys. 103, 7429–7441 (1995); doi: 10.1063/1.470315 [CC3 response & excited states]
    [3] J. Chem. Phys. 122, 054110 (2005); doi: 10.1063/1.1835953 [CC3 for open shells]
"""
import numpy as np
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
    X1A = -np.einsum("mi,am->ai", H.a.oo, R.a, optimize=True)
    X1A += np.einsum("ae,ei->ai", H.a.vv, R.a, optimize=True)
    X1A += np.einsum("amie,em->ai", H.aa.voov, R.a, optimize=True)
    X1A += np.einsum("amie,em->ai", H.ab.voov, R.b, optimize=True)
    X1A -= 0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, R.aa, optimize=True)
    X1A -= np.einsum("mnif,afmn->ai", H.ab.ooov, R.ab, optimize=True)
    X1A += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, R.aa, optimize=True)
    X1A += np.einsum("anef,efin->ai", H.ab.vovv, R.ab, optimize=True)
    X1A += np.einsum("me,aeim->ai", H.a.ov, R.aa, optimize=True)
    X1A += np.einsum("me,aeim->ai", H.b.ov, R.ab, optimize=True)
    return X1A

def build_HR_1B(R, H):
    """< i~a~ | [H(2)*(R1+R2+R3)]_C | 0 >"""
    X1B = -np.einsum("mi,am->ai", H.b.oo, R.b, optimize=True)
    X1B += np.einsum("ae,ei->ai", H.b.vv, R.b, optimize=True)
    X1B += np.einsum("maei,em->ai", H.ab.ovvo, R.a, optimize=True)
    X1B += np.einsum("amie,em->ai", H.bb.voov, R.b, optimize=True)
    X1B -= np.einsum("nmfi,fanm->ai", H.ab.oovo, R.ab, optimize=True)
    X1B -= 0.5 * np.einsum("mnif,afmn->ai", H.bb.ooov, R.bb, optimize=True)
    X1B += np.einsum("nafe,feni->ai", H.ab.ovvv, R.ab, optimize=True)
    X1B += 0.5 * np.einsum("anef,efin->ai", H.bb.vovv, R.bb, optimize=True)
    X1B += np.einsum("me,eami->ai", H.a.ov, R.ab, optimize=True)
    X1B += np.einsum("me,aeim->ai", H.b.ov, R.bb, optimize=True)
    return X1B

def build_HR_2A(R, T, H, X):
    """ < ijab | [H(2)*(R1+R2+R3)]_C | 0 > """
    X2A = -0.5 * np.einsum("mi,abmj->abij", H.a.oo, R.aa, optimize=True)  # A(ij)
    X2A += 0.5 * np.einsum("ae,ebij->abij", H.a.vv, R.aa, optimize=True)  # A(ab)
    X2A += 0.125 * np.einsum("mnij,abmn->abij", H.aa.oooo, R.aa, optimize=True)
    X2A += 0.125 * np.einsum("abef,efij->abij", H.aa.vvvv, R.aa, optimize=True)
    X2A += np.einsum("amie,ebmj->abij", H.aa.voov, R.aa, optimize=True)  # A(ij)A(ab)
    X2A += np.einsum("amie,bejm->abij", H.ab.voov, R.ab, optimize=True)  # A(ij)A(ab)
    X2A -= 0.5 * np.einsum("bmji,am->abij", H.aa.vooo, R.a, optimize=True)  # A(ab)
    X2A += 0.5 * np.einsum("baje,ei->abij", H.aa.vvov, R.a, optimize=True)  # A(ij)
    X2A += 0.5 * np.einsum("be,aeij->abij", X.a.vv, T.aa, optimize=True)  # A(ab)
    X2A -= 0.5 * np.einsum("mj,abim->abij", X.a.oo, T.aa, optimize=True)  # A(ij)
    return X2A

def build_HR_2B(R, T, H, X):
    """< ij~ab~ | [H(2)*(R1+R2+R3)]_C | 0 >"""
    X2B = np.einsum("ae,ebij->abij", H.a.vv, R.ab, optimize=True)
    X2B += np.einsum("be,aeij->abij", H.b.vv, R.ab, optimize=True)
    X2B -= np.einsum("mi,abmj->abij", H.a.oo, R.ab, optimize=True)
    X2B -= np.einsum("mj,abim->abij", H.b.oo, R.ab, optimize=True)
    X2B += np.einsum("mnij,abmn->abij", H.ab.oooo, R.ab, optimize=True)
    X2B += np.einsum("abef,efij->abij", H.ab.vvvv, R.ab, optimize=True)
    X2B += np.einsum("amie,ebmj->abij", H.aa.voov, R.ab, optimize=True)
    X2B += np.einsum("amie,ebmj->abij", H.ab.voov, R.bb, optimize=True)
    X2B += np.einsum("mbej,aeim->abij", H.ab.ovvo, R.aa, optimize=True)
    X2B += np.einsum("bmje,aeim->abij", H.bb.voov, R.ab, optimize=True)
    X2B -= np.einsum("mbie,aemj->abij", H.ab.ovov, R.ab, optimize=True)
    X2B -= np.einsum("amej,ebim->abij", H.ab.vovo, R.ab, optimize=True)
    X2B += np.einsum("abej,ei->abij", H.ab.vvvo, R.a, optimize=True)
    X2B += np.einsum("abie,ej->abij", H.ab.vvov, R.b, optimize=True)
    X2B -= np.einsum("mbij,am->abij", H.ab.ovoo, R.a, optimize=True)
    X2B -= np.einsum("amij,bm->abij", H.ab.vooo, R.b, optimize=True)
    X2B += np.einsum("ae,ebij->abij", X.a.vv, T.ab, optimize=True)
    X2B -= np.einsum("mi,abmj->abij", X.a.oo, T.ab, optimize=True)
    X2B += np.einsum("be,aeij->abij", X.b.vv, T.ab, optimize=True)
    X2B -= np.einsum("mj,abim->abij", X.b.oo, T.ab, optimize=True)
    return X2B

def build_HR_2C(R, T, H, X):
    """< i~j~a~b~ | [H(2)*(R1+R2+R3)]_C | 0 >"""
    X2C = -0.5 * np.einsum("mi,abmj->abij", H.b.oo, R.bb, optimize=True)  # A(ij)
    X2C += 0.5 * np.einsum("ae,ebij->abij", H.b.vv, R.bb, optimize=True)  # A(ab)
    X2C += 0.125 * np.einsum("mnij,abmn->abij", H.bb.oooo, R.bb, optimize=True)
    X2C += 0.125 * np.einsum("abef,efij->abij", H.bb.vvvv, R.bb, optimize=True)
    X2C += np.einsum("amie,ebmj->abij", H.bb.voov, R.bb, optimize=True)  # A(ij)A(ab)
    X2C += np.einsum("maei,ebmj->abij", H.ab.ovvo, R.ab, optimize=True)  # A(ij)A(ab)
    X2C -= 0.5 * np.einsum("bmji,am->abij", H.bb.vooo, R.b, optimize=True)  # A(ab)
    X2C += 0.5 * np.einsum("baje,ei->abij", H.bb.vvov, R.b, optimize=True)  # A(ij)
    X2C += 0.5 * np.einsum("be,aeij->abij", X.b.vv, T.bb, optimize=True)  # A(ab)
    X2C -= 0.5 * np.einsum("mj,abim->abij", X.b.oo, T.bb, optimize=True)  # A(ij)
    return X2C

def _compute_r3a(R, T, H1, HR1, omega, fock):
    # <ijkabc| (H(1) * R2)_C | 0 >
    X3A = 0.25 * np.einsum("baje,ecik->abcijk", H1.aa.vvov, R.aa, optimize=True) #
    X3A -= 0.25 * np.einsum("bmji,acmk->abcijk", H1.aa.vooo, R.aa, optimize=True)
    # <ijkabc| ((H(1)*R1)_C * T2)_C | 0 >
    X3A += 0.25 * np.einsum("baje,ecik->abcijk", HR1.aa.vvov, T.aa, optimize=True)
    X3A -= 0.25 * np.einsum("bmji,acmk->abcijk", HR1.aa.vooo, T.aa, optimize=True)
    return cc3_loops.compute_r3a(X3A, omega, fock.a.oo, fock.a.vv)

def _compute_r3b(R, T, H1, HR1, omega, fock):
    # < ijk~abc~ | (H(1)*R2)_C | 0 >
    X3B = 0.5 * np.einsum("bcek,aeij->abcijk", H1.ab.vvvo, R.aa, optimize=True)
    X3B -= 0.5 * np.einsum("mcjk,abim->abcijk", H1.ab.ovoo, R.aa, optimize=True)
    X3B += 0.5 * np.einsum("baje,ecik->abcijk", H1.aa.vvov, R.ab, optimize=True)
    X3B -= 0.5 * np.einsum("bnji,acnk->abcijk", H1.aa.vooo, R.ab, optimize=True)
    X3B += np.einsum("bcje,aeik->abcijk", H1.ab.vvov, R.ab, optimize=True)
    X3B -= np.einsum("bnjk,acin->abcijk", H1.ab.vooo, R.ab, optimize=True)
    # < ijk~abc~ | ((H(1)*R1)_C * T2)_C | 0 >
    X3B += 0.5 * np.einsum("bcek,aeij->abcijk", HR1.ab.vvvo, T.aa, optimize=True)
    X3B -= 0.5 * np.einsum("ncjk,abin->abcijk", HR1.ab.ovoo, T.aa, optimize=True)
    X3B += 0.5 * np.einsum("baje,ecik->abcijk", HR1.aa.vvov, T.ab, optimize=True)
    X3B -= 0.5 * np.einsum("bnji,acnk->abcijk", HR1.aa.vooo, T.ab, optimize=True)
    X3B += np.einsum("bcje,aeik->abcijk", HR1.ab.vvov, T.ab, optimize=True)
    X3B -= np.einsum("bnjk,acin->abcijk", HR1.ab.vooo, T.ab, optimize=True)
    return cc3_loops.compute_r3b(X3B, omega, fock.a.oo, fock.a.vv, fock.b.oo, fock.b.vv)

def _compute_r3c(R, T, H1, HR1, omega, fock):
    # < ij~k~ab~c~ | (H(1)*R2)_C | 0 >
    X3C = 0.5 * np.einsum("cbke,aeij->cbakji", H1.ab.vvov, R.bb, optimize=True)
    X3C -= 0.5 * np.einsum("cmkj,abim->cbakji", H1.ab.vooo, R.bb, optimize=True)
    X3C += 0.5 * np.einsum("baje,ceki->cbakji", H1.bb.vvov, R.ab, optimize=True)
    X3C -= 0.5 * np.einsum("bnji,cakn->cbakji", H1.bb.vooo, R.ab, optimize=True)
    X3C += np.einsum("cbej,eaki->cbakji", H1.ab.vvvo, R.ab, optimize=True)
    X3C -= np.einsum("nbkj,cani->cbakji", H1.ab.ovoo, R.ab, optimize=True)
    # < ij~k~ab~c~ | ((H(1)*R1)_C * T2)_C | 0 >
    X3C += 0.5 * np.einsum("cbke,aeij->cbakji", HR1.ab.vvov, T.bb, optimize=True)
    X3C -= 0.5 * np.einsum("cnkj,abin->cbakji", HR1.ab.vooo, T.bb, optimize=True)
    X3C += 0.5 * np.einsum("baje,ceki->cbakji", HR1.bb.vvov, T.ab, optimize=True)
    X3C -= 0.5 * np.einsum("bnji,cakn->cbakji", HR1.bb.vooo, T.ab, optimize=True)
    X3C += np.einsum("cbej,eaki->cbakji", HR1.ab.vvvo, T.ab, optimize=True)
    X3C -= np.einsum("nbkj,cani->cbakji", HR1.ab.ovoo, T.ab, optimize=True)
    return cc3_loops.compute_r3c(X3C, omega, fock.a.oo, fock.a.vv, fock.b.oo, fock.b.vv)

def _compute_r3d(R, T, H1, HR1, omega, fock):
    # <i~j~k~a~b~c~| (H(1) * R2)_C | 0 >
    X3D = 0.25 * np.einsum("baje,ecik->abcijk", H1.bb.vvov, R.bb, optimize=True)
    X3D -= 0.25 * np.einsum("bmji,acmk->abcijk", H1.bb.vooo, R.bb, optimize=True)
    # <i~j~k~a~b~c~| ((H(1)*R1)_C * T2)_C | 0 >
    X3D += 0.25 * np.einsum("baje,ecik->abcijk", HR1.bb.vvov, T.bb, optimize=True)
    X3D -= 0.25 * np.einsum("bmji,acmk->abcijk", HR1.bb.vooo, T.bb, optimize=True)
    return cc3_loops.compute_r3d(X3D, omega, fock.b.oo, fock.b.vv)

def _compute_t3a(T, X, fock):
    x3a = -0.25 * np.einsum("amij,bcmk->abcijk", X.aa.vooo, T.aa, optimize=True)
    x3a += 0.25 * np.einsum("abie,ecjk->abcijk", X.aa.vvov, T.aa, optimize=True)
    return cc3_loops.compute_t3a(x3a, fock.a.oo, fock.a.vv)

def _compute_t3b(T, X, fock):
    x3b = 0.5 * np.einsum("bcek,aeij->abcijk", X.ab.vvvo, T.aa, optimize=True)
    x3b -= 0.5 * np.einsum("mcjk,abim->abcijk", X.ab.ovoo, T.aa, optimize=True)
    x3b += np.einsum("acie,bejk->abcijk", X.ab.vvov, T.ab, optimize=True)
    x3b -= np.einsum("amik,bcjm->abcijk", X.ab.vooo, T.ab, optimize=True)
    x3b += 0.5 * np.einsum("abie,ecjk->abcijk", X.aa.vvov, T.ab, optimize=True)
    x3b -= 0.5 * np.einsum("amij,bcmk->abcijk", X.aa.vooo, T.ab, optimize=True)
    return cc3_loops.compute_t3b(x3b, fock.a.oo, fock.a.vv, fock.b.oo, fock.b.vv)

def _compute_t3c(T, X, fock):
    x3c = 0.5 * np.einsum("abie,ecjk->abcijk", X.ab.vvov, T.bb, optimize=True)
    x3c -= 0.5 * np.einsum("amij,bcmk->abcijk", X.ab.vooo, T.bb, optimize=True)
    x3c += 0.5 * np.einsum("cbke,aeij->abcijk", X.bb.vvov, T.ab, optimize=True)
    x3c -= 0.5 * np.einsum("cmkj,abim->abcijk", X.bb.vooo, T.ab, optimize=True)
    x3c += np.einsum("abej,ecik->abcijk", X.ab.vvvo, T.ab, optimize=True)
    x3c -= np.einsum("mbij,acmk->abcijk", X.ab.ovoo, T.ab, optimize=True)
    return cc3_loops.compute_t3c(x3c, fock.a.oo, fock.a.vv, fock.b.oo, fock.b.vv)

def _compute_t3d(T, X, fock):
    x3d = -0.25 * np.einsum("amij,bcmk->abcijk", X.bb.vooo, T.bb, optimize=True)
    x3d += 0.25 * np.einsum("abie,ecjk->abcijk", X.bb.vvov, T.bb, optimize=True)
    return cc3_loops.compute_t3d(x3d, fock.b.oo, fock.b.vv)
