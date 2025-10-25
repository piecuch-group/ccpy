'''
Equation-of-Motion Coupled-Cluster Method with Singles, Doubles, and Triples (EOMCCSDT)
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import cc_loops2
from ccpy.eomcc.eomccsdt_intermediates import get_eomccsd_intermediates, get_eomccsdt_intermediates, add_R3_terms

def update(R, omega, H, RHF_symmetry, system):
    R.a, R.b, R.aa, R.ab, R.bb, R.aaa, R.aab, R.abb, R.bbb = cc_loops2.update_r_ccsdt(
        R.a,
        R.b,
        R.aa,
        R.ab,
        R.bb,
        R.aaa,
        R.aab,
        R.abb,
        R.bbb,
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
        R.abb = R.aab.transpose((2, 1, 0, 5, 4, 3))
        R.bbb = R.aaa.copy()

    return R

def HR(dR, R, T, H, flag_RHF, system):

    dR.a = build_HR_1A(R, T, H)
    if flag_RHF:
        dR.b = dR.a.copy()
    else:
        dR.b = build_HR_1B(R, T, H)

    # Get H*R EOMCCSD intermediates
    X0 = get_eomccsd_intermediates(H, R, system)
    dR.aa = build_HR_2A(R, T, H, X0)
    dR.ab = build_HR_2B(R, T, H, X0)
    if flag_RHF:
        dR.bb = dR.aa.copy()
    else:
        dR.bb = build_HR_2C(R, T, H, X0)

    # Add on terms needed to make EOMCCSDT intermediates
    X = get_eomccsdt_intermediates(H, R, T, X0, system)
    X = add_R3_terms(X, H, R)
    dR.aaa = build_HR_3A(R, T, H, X)
    dR.aab = build_HR_3B(R, T, H, X)
    if flag_RHF:
        dR.abb = np.transpose(dR.aab, (2, 1, 0, 5, 4, 3))
        dR.bbb = dR.aaa.copy()
    else:
        dR.abb = build_HR_3C(R, T, H, X)
        dR.bbb = build_HR_3D(R, T, H, X)

    return dR.flatten()

def build_HR_1A(R, T, H):
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
    # Parts contracted with R3
    X1A += 0.25 * ccpy_einsum("mnef,aefimn->ai", H.aa.oovv, R.aaa)
    X1A += ccpy_einsum("mnef,aefimn->ai", H.ab.oovv, R.aab)
    X1A += 0.25 * ccpy_einsum("mnef,aefimn->ai", H.bb.oovv, R.abb)
    return X1A

def build_HR_1B(R, T, H):
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
    # Parts contracted with R3
    X1B += 0.25 * ccpy_einsum("mnef,efamni->ai", H.aa.oovv, R.aab)
    X1B += ccpy_einsum("mnef,efamni->ai", H.ab.oovv, R.abb)
    X1B += 0.25 * ccpy_einsum("mnef,aefimn->ai", H.bb.oovv, R.bbb)
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
    # Parts contracted with T3
    X2A += 0.25 * ccpy_einsum("me,abeijm->abij", X.a.ov, T.aaa)
    X2A += 0.25 * ccpy_einsum("me,abeijm->abij", X.b.ov, T.aab)
    # Parts contracted with R3
    X2A += 0.25 * ccpy_einsum("me,abeijm->abij", H.a.ov, R.aaa)
    X2A += 0.25 * ccpy_einsum("me,abeijm->abij", H.b.ov, R.aab)
    X2A -= 0.25 * ccpy_einsum("mnjf,abfimn->abij", H.aa.ooov, R.aaa)
    X2A -= 0.5 * ccpy_einsum("mnjf,abfimn->abij", H.ab.ooov, R.aab)
    X2A += 0.25 * ccpy_einsum("bnef,aefijn->abij", H.aa.vovv, R.aaa)
    X2A += 0.5 * ccpy_einsum("bnef,aefijn->abij", H.ab.vovv, R.aab)
    X2A -= np.transpose(X2A, (1, 0, 2, 3))  # antisymmetrize (ab)
    X2A -= np.transpose(X2A, (0, 1, 3, 2))  # antisymmetrize (ij)
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
    # Parts contracted with T3
    X2B += ccpy_einsum("me,aebimj->abij", X.a.ov, T.aab)
    X2B += ccpy_einsum("me,aebimj->abij", X.b.ov, T.abb)
    # Parts contracted with R3
    X2B += ccpy_einsum("me,aebimj->abij", H.a.ov, R.aab)
    X2B += ccpy_einsum("me,aebimj->abij", H.b.ov, R.abb)
    X2B -= ccpy_einsum("nmfj,afbinm->abij", H.ab.oovo, R.aab)
    X2B -= 0.5 * ccpy_einsum("mnjf,abfimn->abij", H.bb.ooov, R.abb)
    X2B -= 0.5 * ccpy_einsum("mnif,afbmnj->abij", H.aa.ooov, R.aab)
    X2B -= ccpy_einsum("mnif,abfmjn->abij", H.ab.ooov, R.abb)
    X2B += ccpy_einsum("nbfe,afeinj->abij", H.ab.ovvv, R.aab)
    X2B += 0.5 * ccpy_einsum("bnef,aefijn->abij", H.bb.vovv, R.abb)
    X2B += 0.5 * ccpy_einsum("anef,efbinj->abij", H.aa.vovv, R.aab)
    X2B += ccpy_einsum("anef,efbinj->abij", H.ab.vovv, R.abb)
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
    # Parts contracted with T3
    X2C += 0.25 * ccpy_einsum("me,eabmij->abij", X.a.ov, T.abb)
    X2C += 0.25 * ccpy_einsum("me,abeijm->abij", X.b.ov, T.bbb)
    # Parts contracted with R3
    X2C += 0.25 * ccpy_einsum("me,eabmij->abij", H.a.ov, R.abb)
    X2C += 0.25 * ccpy_einsum("me,abeijm->abij", H.b.ov, R.bbb)
    X2C -= 0.25 * ccpy_einsum("mnjf,abfimn->abij", H.bb.ooov, R.bbb)
    X2C -= 0.5 * ccpy_einsum("nmfj,fabnim->abij", H.ab.oovo, R.abb)
    X2C += 0.25 * ccpy_einsum("bnef,aefijn->abij", H.bb.vovv, R.bbb)
    X2C += 0.5 * ccpy_einsum("nbfe,faenij->abij", H.ab.ovvv, R.abb)
    X2C -= np.transpose(X2C, (1, 0, 2, 3))  # antisymmetrize (ab)
    X2C -= np.transpose(X2C, (0, 1, 3, 2))  # antisymmetrize (ij)
    return X2C

def build_HR_3A(R, T, H, X):
    # <ijkabc| [H(R1+R2+R3)]_C | 0 >
    X3A = 0.25 * ccpy_einsum("baje,ecik->abcijk", X.aa.vvov, T.aa)
    X3A += 0.25 * ccpy_einsum("baje,ecik->abcijk", H.aa.vvov, R.aa)
    X3A -= 0.25 * ccpy_einsum("bmji,acmk->abcijk", X.aa.vooo, T.aa)
    X3A -= 0.25 * ccpy_einsum("bmji,acmk->abcijk", H.aa.vooo, R.aa)
    # additional terms with T3 in <ijkabc|[ H(R1+R2)]_C | 0>
    X3A += (1.0 / 12.0) * ccpy_einsum("be,aecijk->abcijk", X.a.vv, T.aaa)
    X3A -= (1.0 / 12.0) * ccpy_einsum("mj,abcimk->abcijk", X.a.oo, T.aaa)
    X3A += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", X.aa.oooo, T.aaa)
    X3A += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", X.aa.vvvv, T.aaa)
    X3A += 0.25 * ccpy_einsum("bmje,aecimk->abcijk", X.aa.voov, T.aaa)
    X3A += 0.25 * ccpy_einsum("bmje,aceikm->abcijk", X.ab.voov, T.aab)
    # < ijkabc | (HR3)_C | 0 >
    X3A -= (1.0 / 12.0) * ccpy_einsum("mj,abcimk->abcijk", H.a.oo, R.aaa)
    X3A += (1.0 / 12.0) * ccpy_einsum("be,aecijk->abcijk", H.a.vv, R.aaa)
    X3A += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", H.aa.oooo, R.aaa)
    X3A += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", H.aa.vvvv, R.aaa)
    X3A += 0.25 * ccpy_einsum("amie,ebcmjk->abcijk", H.aa.voov, R.aaa)
    X3A += 0.25 * ccpy_einsum("amie,bcejkm->abcijk", H.ab.voov, R.aab)
    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3A -= np.transpose(X3A, (0, 1, 2, 3, 5, 4))
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3, 5)) + np.transpose(X3A, (0, 1, 2, 5, 4, 3))
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4, 5))
    X3A -= np.transpose(X3A, (1, 0, 2, 3, 4, 5)) + np.transpose(X3A, (2, 1, 0, 3, 4, 5))
    return X3A

def build_HR_3B(R, T, H, X):
    # < ijk~abc~ | [ H(R1+R2+R3) ]_C | 0 >
    # Intermediate 1: X2B(bcek)*Y2A(aeij) -> Z3B(abcijk)
    X3B = 0.5 * ccpy_einsum("bcek,aeij->abcijk", X.ab.vvvo, T.aa)
    X3B += 0.5 * ccpy_einsum("bcek,aeij->abcijk", H.ab.vvvo, R.aa)
    # Intermediate 2: X2B(ncjk)*Y2A(abin) -> Z3B(abcijk)
    X3B -= 0.5 * ccpy_einsum("ncjk,abin->abcijk", X.ab.ovoo, T.aa)
    X3B -= 0.5 * ccpy_einsum("mcjk,abim->abcijk", H.ab.ovoo, R.aa)
    # Intermediate 3: X2A(baje)*Y2B(ecik) -> Z3B(abcijk)
    X3B += 0.5 * ccpy_einsum("baje,ecik->abcijk", X.aa.vvov, T.ab)
    X3B += 0.5 * ccpy_einsum("baje,ecik->abcijk", H.aa.vvov, R.ab)
    # Intermediate 4: X2A(bnji)*Y2B(acnk) -> Z3B(abcijk)
    X3B -= 0.5 * ccpy_einsum("bnji,acnk->abcijk", X.aa.vooo, T.ab)
    X3B -= 0.5 * ccpy_einsum("bnji,acnk->abcijk", H.aa.vooo, R.ab)
    # Intermediate 5: X2B(bcje)*Y2B(aeik) -> Z3B(abcijk)
    X3B += ccpy_einsum("bcje,aeik->abcijk", X.ab.vvov, T.ab)
    X3B += ccpy_einsum("bcje,aeik->abcijk", H.ab.vvov, R.ab)
    # Intermediate 6: X2B(bnjk)*Y2B(acin) -> Z3B(abcijk)
    X3B -= ccpy_einsum("bnjk,acin->abcijk", X.ab.vooo, T.ab)
    X3B -= ccpy_einsum("bnjk,acin->abcijk", H.ab.vooo, R.ab)
    # additional terms with T3 (these contractions mirror the form of
    # the ones with R3 later on)
    X3B += 0.5 * ccpy_einsum("be,aecijk->abcijk", X.a.vv, T.aab)
    X3B += 0.25 * ccpy_einsum("ce,abeijk->abcijk", X.b.vv, T.aab)
    X3B -= 0.5 * ccpy_einsum("mj,abcimk->abcijk", X.a.oo, T.aab)
    X3B -= 0.25 * ccpy_einsum("mk,abcijm->abcijk", X.b.oo, T.aab)
    X3B += 0.5 * ccpy_einsum("nmjk,abcinm->abcijk", X.ab.oooo, T.aab)
    X3B += 0.125 * ccpy_einsum("mnij,abcmnk->abcijk", X.aa.oooo, T.aab)
    X3B += 0.5 * ccpy_einsum("bcfe,afeijk->abcijk", X.ab.vvvv, T.aab)
    X3B += 0.125 * ccpy_einsum("abef,efcijk->abcijk", X.aa.vvvv, T.aab)
    X3B += 0.25 * ccpy_einsum("ncfk,abfijn->abcijk", X.ab.ovvo, T.aaa)
    X3B += 0.25 * ccpy_einsum("cnkf,abfijn->abcijk", X.bb.voov, T.aab)
    X3B -= 0.5 * ccpy_einsum("bmfk,afcijm->abcijk", X.ab.vovo, T.aab)
    X3B -= 0.5 * ccpy_einsum("ncje,abeink->abcijk", X.ab.ovov, T.aab)
    X3B += ccpy_einsum("bmje,aecimk->abcijk", X.aa.voov, T.aab)
    X3B += ccpy_einsum("bmje,aecimk->abcijk", X.ab.voov, T.abb)
    # < ijk~abc~ | (HR3)_C | 0 >
    X3B -= 0.5 * ccpy_einsum("mj,abcimk->abcijk", H.a.oo, R.aab)
    X3B -= 0.25 * ccpy_einsum("mk,abcijm->abcijk", H.b.oo, R.aab)
    X3B += 0.5 * ccpy_einsum("be,aecijk->abcijk", H.a.vv, R.aab)
    X3B += 0.25 * ccpy_einsum("ce,abeijk->abcijk", H.b.vv, R.aab)
    X3B += 0.125 * ccpy_einsum("mnij,abcmnk->abcijk", H.aa.oooo, R.aab)
    X3B += 0.5 * ccpy_einsum("mnjk,abcimn->abcijk", H.ab.oooo, R.aab)
    X3B += 0.125 * ccpy_einsum("abef,efcijk->abcijk", H.aa.vvvv, R.aab)
    X3B += 0.5 * ccpy_einsum("bcef,aefijk->abcijk", H.ab.vvvv, R.aab)
    X3B += ccpy_einsum("amie,ebcmjk->abcijk", H.aa.voov, R.aab)
    X3B += ccpy_einsum("amie,becjmk->abcijk", H.ab.voov, R.abb)
    X3B += 0.25 * ccpy_einsum("mcek,abeijm->abcijk", H.ab.ovvo, R.aaa)
    X3B += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.bb.voov, R.aab)
    X3B -= 0.5 * ccpy_einsum("bmek,aecijm->abcijk", H.ab.vovo, R.aab)
    X3B -= 0.5 * ccpy_einsum("mcje,abeimk->abcijk", H.ab.ovov, R.aab)
    X3B -= np.transpose(X3B, (1, 0, 2, 3, 4, 5))
    X3B -= np.transpose(X3B, (0, 1, 2, 4, 3, 5))
    return X3B

def build_HR_3C(R, T, H, X):
    # < ij~k~ab~c~ | [ H(R1+R2+R3) ]_C | 0 >
    # Intermediate 1: X2B(cbke)*Y2C(aeij) -> Z3C(cbakji)
    X3C = 0.5 * ccpy_einsum("cbke,aeij->cbakji", X.ab.vvov, T.bb)
    X3C += 0.5 * ccpy_einsum("cbke,aeij->cbakji", H.ab.vvov, R.bb)
    # Intermediate 2: X2B(cnkj)*Y2C(abin) -> Z3C(cbakji)
    X3C -= 0.5 * ccpy_einsum("cnkj,abin->cbakji", X.ab.vooo, T.bb)
    X3C -= 0.5 * ccpy_einsum("cmkj,abim->cbakji", H.ab.vooo, R.bb)
    # Intermediate 3: X2C(baje)*Y2B(ceki) -> Z3C(cbakji)
    X3C += 0.5 * ccpy_einsum("baje,ceki->cbakji", X.bb.vvov, T.ab)
    X3C += 0.5 * ccpy_einsum("baje,ceki->cbakji", H.bb.vvov, R.ab)
    # Intermediate 4: X2C(bnji)*Y2B(cakn) -> Z3C(cbakji)
    X3C -= 0.5 * ccpy_einsum("bnji,cakn->cbakji", X.bb.vooo, T.ab)
    X3C -= 0.5 * ccpy_einsum("bnji,cakn->cbakji", H.bb.vooo, R.ab)
    # Intermediate 5: X2B(cbej)*Y2B(eaki) -> Z3C(cbakji)
    X3C += ccpy_einsum("cbej,eaki->cbakji", X.ab.vvvo, T.ab)
    X3C += ccpy_einsum("cbej,eaki->cbakji", H.ab.vvvo, R.ab)
    # Intermediate 6: X2B(nbkj)*Y2B(cani) -> Z3C(cbakji)
    X3C -= ccpy_einsum("nbkj,cani->cbakji", X.ab.ovoo, T.ab)
    X3C -= ccpy_einsum("nbkj,cani->cbakji", H.ab.ovoo, R.ab)
    # additional terms with T3
    X3C += 0.5 * ccpy_einsum("be,ceakji->cbakji", X.b.vv, T.abb)
    X3C += 0.25 * ccpy_einsum("ce,ebakji->cbakji", X.a.vv, T.abb)
    X3C -= 0.5 * ccpy_einsum("mj,cbakmi->cbakji", X.b.oo, T.abb)
    X3C -= 0.25 * ccpy_einsum("mk,cbamji->cbakji", X.a.oo, T.abb)
    X3C += 0.5 * ccpy_einsum("mnkj,cbamni->cbakji", X.ab.oooo, T.abb)
    X3C += 0.125 * ccpy_einsum("mnij,cbaknm->cbakji", X.bb.oooo, T.abb)
    X3C += 0.5 * ccpy_einsum("cbef,efakji->cbakji", X.ab.vvvv, T.abb)
    X3C += 0.125 * ccpy_einsum("abef,cfekji->cbakji", X.bb.vvvv, T.abb)
    X3C += 0.25 * ccpy_einsum("cnkf,abfijn->cbakji", X.ab.voov, T.bbb)
    X3C += 0.25 * ccpy_einsum("cnkf,fbanji->cbakji", X.aa.voov, T.abb)
    X3C -= 0.5 * ccpy_einsum("mbkf,cfamji->cbakji", X.ab.ovov, T.abb)
    X3C -= 0.5 * ccpy_einsum("cnej,ebakni->cbakji", X.ab.vovo, T.abb)
    X3C += ccpy_einsum("bmje,ceakmi->cbakji", X.bb.voov, T.abb)
    X3C += ccpy_einsum("mbej,ceakmi->cbakji", X.ab.ovvo, T.aab)
    # < ijk~abc~ | (HR3)_C | 0 >
    X3C -= 0.5 * ccpy_einsum("mj,cbakmi->cbakji", H.b.oo, R.abb)
    X3C -= 0.25 * ccpy_einsum("mk,cbamji->cbakji", H.a.oo, R.abb)
    X3C += 0.5 * ccpy_einsum("be,ceakji->cbakji", H.b.vv, R.abb)
    X3C += 0.25 * ccpy_einsum("ce,ebakji->cbakji", H.a.vv, R.abb)
    X3C += 0.125 * ccpy_einsum("mnij,cbaknm->cbakji", H.bb.oooo, R.abb)
    X3C += 0.5 * ccpy_einsum("nmkj,cbanmi->cbakji", H.ab.oooo, R.abb)
    X3C += 0.125 * ccpy_einsum("abef,cfekji->cbakji", H.bb.vvvv, R.abb)
    X3C += 0.5 * ccpy_einsum("cbfe,feakji->cbakji", H.ab.vvvv, R.abb)
    X3C += ccpy_einsum("amie,cbekjm->cbakji", H.bb.voov, R.abb)
    X3C += ccpy_einsum("maei,cebkmj->cbakji", H.ab.ovvo, R.aab)
    X3C += 0.25 * ccpy_einsum("cmke,ebamji->cbakji", H.ab.voov, R.bbb)
    X3C += 0.25 * ccpy_einsum("cmke,ebamji->cbakji", H.aa.voov, R.abb)
    X3C -= 0.5 * ccpy_einsum("mbke,ceamji->cbakji", H.ab.ovov, R.abb)
    X3C -= 0.5 * ccpy_einsum("cmej,ebakmi->cbakji", H.ab.vovo, R.abb)
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4, 5))
    X3C -= np.transpose(X3C, (0, 1, 2, 3, 5, 4))
    return X3C

def build_HR_3D(R, T, H, X):
    # <i~j~k~a~b~c~| [H(R1+R2+R3)]_C | 0 >
    X3D = 0.25 * ccpy_einsum("baje,ecik->abcijk", X.bb.vvov, T.bb)
    X3D += 0.25 * ccpy_einsum("baje,ecik->abcijk", H.bb.vvov, R.bb)
    X3D -= 0.25 * ccpy_einsum("bmji,acmk->abcijk", X.bb.vooo, T.bb)
    X3D -= 0.25 * ccpy_einsum("bmji,acmk->abcijk", H.bb.vooo, R.bb)
    # additional terms with T3 in <ijkabc|[ H(R1+R2)]_C | 0>
    X3D += (1.0 / 12.0) * ccpy_einsum("be,aecijk->abcijk", X.b.vv, T.bbb)
    X3D -= (1.0 / 12.0) * ccpy_einsum("mj,abcimk->abcijk", X.b.oo, T.bbb)
    X3D += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", X.bb.oooo, T.bbb)
    X3D += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", X.bb.vvvv, T.bbb)
    X3D += 0.25 * ccpy_einsum("bmje,aecimk->abcijk", X.bb.voov, T.bbb)
    X3D += 0.25 * ccpy_einsum("mbej,ecamki->abcijk", X.ab.ovvo, T.abb)
    # < i~j~k~a~b~c~ | (HR3)_C | 0 >
    X3D -= (1.0 / 12.0) * ccpy_einsum("mj,abcimk->abcijk", H.b.oo, R.bbb)
    X3D += (1.0 / 12.0) * ccpy_einsum("be,aecijk->abcijk", H.b.vv, R.bbb)
    X3D += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", H.bb.oooo, R.bbb)
    X3D += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", H.bb.vvvv, R.bbb)
    X3D += 0.25 * ccpy_einsum("amie,ebcmjk->abcijk", H.bb.voov, R.bbb)
    X3D += 0.25 * ccpy_einsum("maei,ecbmkj->abcijk", H.ab.ovvo, R.abb)
    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3D -= np.transpose(X3D, (0, 1, 2, 3, 5, 4))
    X3D -= np.transpose(X3D, (0, 1, 2, 4, 3, 5)) + np.transpose(X3D, (0, 1, 2, 5, 4, 3))
    X3D -= np.transpose(X3D, (0, 2, 1, 3, 4, 5))
    X3D -= np.transpose(X3D, (1, 0, 2, 3, 4, 5)) + np.transpose(X3D, (2, 1, 0, 3, 4, 5))
    return X3D
