"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for excited states using
the equation-of-motion (EOM) CC with singles, doubles, and triples (EOMCCSDT)."""
import numpy as np
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
    # Parts contracted with R3
    X1A += 0.25 * np.einsum("mnef,aefimn->ai", H.aa.oovv, R.aaa, optimize=True)
    X1A += np.einsum("mnef,aefimn->ai", H.ab.oovv, R.aab, optimize=True)
    X1A += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, R.abb, optimize=True)
    return X1A

def build_HR_1B(R, T, H):
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
    # Parts contracted with R3
    X1B += 0.25 * np.einsum("mnef,efamni->ai", H.aa.oovv, R.aab, optimize=True)
    X1B += np.einsum("mnef,efamni->ai", H.ab.oovv, R.abb, optimize=True)
    X1B += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, R.bbb, optimize=True)
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
    # Parts contracted with T3
    X2A += 0.25 * np.einsum("me,abeijm->abij", X.a.ov, T.aaa, optimize=True)
    X2A += 0.25 * np.einsum("me,abeijm->abij", X.b.ov, T.aab, optimize=True)
    # Parts contracted with R3
    X2A += 0.25 * np.einsum("me,abeijm->abij", H.a.ov, R.aaa, optimize=True)
    X2A += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, R.aab, optimize=True)
    X2A -= 0.25 * np.einsum("mnjf,abfimn->abij", H.aa.ooov, R.aaa, optimize=True)
    X2A -= 0.5 * np.einsum("mnjf,abfimn->abij", H.ab.ooov, R.aab, optimize=True)
    X2A += 0.25 * np.einsum("bnef,aefijn->abij", H.aa.vovv, R.aaa, optimize=True)
    X2A += 0.5 * np.einsum("bnef,aefijn->abij", H.ab.vovv, R.aab, optimize=True)
    X2A -= np.transpose(X2A, (1, 0, 2, 3))  # antisymmetrize (ab)
    X2A -= np.transpose(X2A, (0, 1, 3, 2))  # antisymmetrize (ij)
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
    # Parts contracted with T3
    X2B += np.einsum("me,aebimj->abij", X.a.ov, T.aab, optimize=True)
    X2B += np.einsum("me,aebimj->abij", X.b.ov, T.abb, optimize=True)
    # Parts contracted with R3
    X2B += np.einsum("me,aebimj->abij", H.a.ov, R.aab, optimize=True)
    X2B += np.einsum("me,aebimj->abij", H.b.ov, R.abb, optimize=True)
    X2B -= np.einsum("nmfj,afbinm->abij", H.ab.oovo, R.aab, optimize=True)
    X2B -= 0.5 * np.einsum("mnjf,abfimn->abij", H.bb.ooov, R.abb, optimize=True)
    X2B -= 0.5 * np.einsum("mnif,afbmnj->abij", H.aa.ooov, R.aab, optimize=True)
    X2B -= np.einsum("mnif,abfmjn->abij", H.ab.ooov, R.abb, optimize=True)
    X2B += np.einsum("nbfe,afeinj->abij", H.ab.ovvv, R.aab, optimize=True)
    X2B += 0.5 * np.einsum("bnef,aefijn->abij", H.bb.vovv, R.abb, optimize=True)
    X2B += 0.5 * np.einsum("anef,efbinj->abij", H.aa.vovv, R.aab, optimize=True)
    X2B += np.einsum("anef,efbinj->abij", H.ab.vovv, R.abb, optimize=True)
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
    # Parts contracted with T3
    X2C += 0.25 * np.einsum("me,eabmij->abij", X.a.ov, T.abb, optimize=True)
    X2C += 0.25 * np.einsum("me,abeijm->abij", X.b.ov, T.bbb, optimize=True)
    # Parts contracted with R3
    X2C += 0.25 * np.einsum("me,eabmij->abij", H.a.ov, R.abb, optimize=True)
    X2C += 0.25 * np.einsum("me,abeijm->abij", H.b.ov, R.bbb, optimize=True)
    X2C -= 0.25 * np.einsum("mnjf,abfimn->abij", H.bb.ooov, R.bbb, optimize=True)
    X2C -= 0.5 * np.einsum("nmfj,fabnim->abij", H.ab.oovo, R.abb, optimize=True)
    X2C += 0.25 * np.einsum("bnef,aefijn->abij", H.bb.vovv, R.bbb, optimize=True)
    X2C += 0.5 * np.einsum("nbfe,faenij->abij", H.ab.ovvv, R.abb, optimize=True)
    X2C -= np.transpose(X2C, (1, 0, 2, 3))  # antisymmetrize (ab)
    X2C -= np.transpose(X2C, (0, 1, 3, 2))  # antisymmetrize (ij)
    return X2C

def build_HR_3A(R, T, H, X):
    # <ijkabc| [H(R1+R2+R3)]_C | 0 >
    X3A = 0.25 * np.einsum("baje,ecik->abcijk", X.aa.vvov, T.aa, optimize=True)
    X3A += 0.25 * np.einsum("baje,ecik->abcijk", H.aa.vvov, R.aa, optimize=True)
    X3A -= 0.25 * np.einsum("bmji,acmk->abcijk", X.aa.vooo, T.aa, optimize=True)
    X3A -= 0.25 * np.einsum("bmji,acmk->abcijk", H.aa.vooo, R.aa, optimize=True)
    # additional terms with T3 in <ijkabc|[ H(R1+R2)]_C | 0>
    X3A += (1.0 / 12.0) * np.einsum("be,aecijk->abcijk", X.a.vv, T.aaa, optimize=True)
    X3A -= (1.0 / 12.0) * np.einsum("mj,abcimk->abcijk", X.a.oo, T.aaa, optimize=True)
    X3A += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", X.aa.oooo, T.aaa, optimize=True)
    X3A += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", X.aa.vvvv, T.aaa, optimize=True)
    X3A += 0.25 * np.einsum("bmje,aecimk->abcijk", X.aa.voov, T.aaa, optimize=True)
    X3A += 0.25 * np.einsum("bmje,aceikm->abcijk", X.ab.voov, T.aab, optimize=True)
    # < ijkabc | (HR3)_C | 0 >
    X3A -= (1.0 / 12.0) * np.einsum("mj,abcimk->abcijk", H.a.oo, R.aaa, optimize=True)
    X3A += (1.0 / 12.0) * np.einsum("be,aecijk->abcijk", H.a.vv, R.aaa, optimize=True)
    X3A += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, R.aaa, optimize=True)
    X3A += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, R.aaa, optimize=True)
    X3A += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.aa.voov, R.aaa, optimize=True)
    X3A += 0.25 * np.einsum("amie,bcejkm->abcijk", H.ab.voov, R.aab, optimize=True)
    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3A -= np.transpose(X3A, (0, 1, 2, 3, 5, 4))
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3, 5)) + np.transpose(X3A, (0, 1, 2, 5, 4, 3))
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4, 5))
    X3A -= np.transpose(X3A, (1, 0, 2, 3, 4, 5)) + np.transpose(X3A, (2, 1, 0, 3, 4, 5))
    return X3A

def build_HR_3B(R, T, H, X):
    # < ijk~abc~ | [ H(R1+R2+R3) ]_C | 0 >
    # Intermediate 1: X2B(bcek)*Y2A(aeij) -> Z3B(abcijk)
    X3B = 0.5 * np.einsum("bcek,aeij->abcijk", X.ab.vvvo, T.aa, optimize=True)
    X3B += 0.5 * np.einsum("bcek,aeij->abcijk", H.ab.vvvo, R.aa, optimize=True)
    # Intermediate 2: X2B(ncjk)*Y2A(abin) -> Z3B(abcijk)
    X3B -= 0.5 * np.einsum("ncjk,abin->abcijk", X.ab.ovoo, T.aa, optimize=True)
    X3B -= 0.5 * np.einsum("mcjk,abim->abcijk", H.ab.ovoo, R.aa, optimize=True)
    # Intermediate 3: X2A(baje)*Y2B(ecik) -> Z3B(abcijk)
    X3B += 0.5 * np.einsum("baje,ecik->abcijk", X.aa.vvov, T.ab, optimize=True)
    X3B += 0.5 * np.einsum("baje,ecik->abcijk", H.aa.vvov, R.ab, optimize=True)
    # Intermediate 4: X2A(bnji)*Y2B(acnk) -> Z3B(abcijk)
    X3B -= 0.5 * np.einsum("bnji,acnk->abcijk", X.aa.vooo, T.ab, optimize=True)
    X3B -= 0.5 * np.einsum("bnji,acnk->abcijk", H.aa.vooo, R.ab, optimize=True)
    # Intermediate 5: X2B(bcje)*Y2B(aeik) -> Z3B(abcijk)
    X3B += np.einsum("bcje,aeik->abcijk", X.ab.vvov, T.ab, optimize=True)
    X3B += np.einsum("bcje,aeik->abcijk", H.ab.vvov, R.ab, optimize=True)
    # Intermediate 6: X2B(bnjk)*Y2B(acin) -> Z3B(abcijk)
    X3B -= np.einsum("bnjk,acin->abcijk", X.ab.vooo, T.ab, optimize=True)
    X3B -= np.einsum("bnjk,acin->abcijk", H.ab.vooo, R.ab, optimize=True)
    # additional terms with T3 (these contractions mirror the form of
    # the ones with R3 later on)
    X3B += 0.5 * np.einsum("be,aecijk->abcijk", X.a.vv, T.aab, optimize=True)
    X3B += 0.25 * np.einsum("ce,abeijk->abcijk", X.b.vv, T.aab, optimize=True)
    X3B -= 0.5 * np.einsum("mj,abcimk->abcijk", X.a.oo, T.aab, optimize=True)
    X3B -= 0.25 * np.einsum("mk,abcijm->abcijk", X.b.oo, T.aab, optimize=True)
    X3B += 0.5 * np.einsum("nmjk,abcinm->abcijk", X.ab.oooo, T.aab, optimize=True)
    X3B += 0.125 * np.einsum("mnij,abcmnk->abcijk", X.aa.oooo, T.aab, optimize=True)
    X3B += 0.5 * np.einsum("bcfe,afeijk->abcijk", X.ab.vvvv, T.aab, optimize=True)
    X3B += 0.125 * np.einsum("abef,efcijk->abcijk", X.aa.vvvv, T.aab, optimize=True)
    X3B += 0.25 * np.einsum("ncfk,abfijn->abcijk", X.ab.ovvo, T.aaa, optimize=True)
    X3B += 0.25 * np.einsum("cnkf,abfijn->abcijk", X.bb.voov, T.aab, optimize=True)
    X3B -= 0.5 * np.einsum("bmfk,afcijm->abcijk", X.ab.vovo, T.aab, optimize=True)
    X3B -= 0.5 * np.einsum("ncje,abeink->abcijk", X.ab.ovov, T.aab, optimize=True)
    X3B += np.einsum("bmje,aecimk->abcijk", X.aa.voov, T.aab, optimize=True)
    X3B += np.einsum("bmje,aecimk->abcijk", X.ab.voov, T.abb, optimize=True)
    # < ijk~abc~ | (HR3)_C | 0 >
    X3B -= 0.5 * np.einsum("mj,abcimk->abcijk", H.a.oo, R.aab, optimize=True)
    X3B -= 0.25 * np.einsum("mk,abcijm->abcijk", H.b.oo, R.aab, optimize=True)
    X3B += 0.5 * np.einsum("be,aecijk->abcijk", H.a.vv, R.aab, optimize=True)
    X3B += 0.25 * np.einsum("ce,abeijk->abcijk", H.b.vv, R.aab, optimize=True)
    X3B += 0.125 * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, R.aab, optimize=True)
    X3B += 0.5 * np.einsum("mnjk,abcimn->abcijk", H.ab.oooo, R.aab, optimize=True)
    X3B += 0.125 * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, R.aab, optimize=True)
    X3B += 0.5 * np.einsum("bcef,aefijk->abcijk", H.ab.vvvv, R.aab, optimize=True)
    X3B += np.einsum("amie,ebcmjk->abcijk", H.aa.voov, R.aab, optimize=True)
    X3B += np.einsum("amie,becjmk->abcijk", H.ab.voov, R.abb, optimize=True)
    X3B += 0.25 * np.einsum("mcek,abeijm->abcijk", H.ab.ovvo, R.aaa, optimize=True)
    X3B += 0.25 * np.einsum("cmke,abeijm->abcijk", H.bb.voov, R.aab, optimize=True)
    X3B -= 0.5 * np.einsum("bmek,aecijm->abcijk", H.ab.vovo, R.aab, optimize=True)
    X3B -= 0.5 * np.einsum("mcje,abeimk->abcijk", H.ab.ovov, R.aab, optimize=True)
    X3B -= np.transpose(X3B, (1, 0, 2, 3, 4, 5))
    X3B -= np.transpose(X3B, (0, 1, 2, 4, 3, 5))
    return X3B

def build_HR_3C(R, T, H, X):
    # < ij~k~ab~c~ | [ H(R1+R2+R3) ]_C | 0 >
    # Intermediate 1: X2B(cbke)*Y2C(aeij) -> Z3C(cbakji)
    X3C = 0.5 * np.einsum("cbke,aeij->cbakji", X.ab.vvov, T.bb, optimize=True)
    X3C += 0.5 * np.einsum("cbke,aeij->cbakji", H.ab.vvov, R.bb, optimize=True)
    # Intermediate 2: X2B(cnkj)*Y2C(abin) -> Z3C(cbakji)
    X3C -= 0.5 * np.einsum("cnkj,abin->cbakji", X.ab.vooo, T.bb, optimize=True)
    X3C -= 0.5 * np.einsum("cmkj,abim->cbakji", H.ab.vooo, R.bb, optimize=True)
    # Intermediate 3: X2C(baje)*Y2B(ceki) -> Z3C(cbakji)
    X3C += 0.5 * np.einsum("baje,ceki->cbakji", X.bb.vvov, T.ab, optimize=True)
    X3C += 0.5 * np.einsum("baje,ceki->cbakji", H.bb.vvov, R.ab, optimize=True)
    # Intermediate 4: X2C(bnji)*Y2B(cakn) -> Z3C(cbakji)
    X3C -= 0.5 * np.einsum("bnji,cakn->cbakji", X.bb.vooo, T.ab, optimize=True)
    X3C -= 0.5 * np.einsum("bnji,cakn->cbakji", H.bb.vooo, R.ab, optimize=True)
    # Intermediate 5: X2B(cbej)*Y2B(eaki) -> Z3C(cbakji)
    X3C += np.einsum("cbej,eaki->cbakji", X.ab.vvvo, T.ab, optimize=True)
    X3C += np.einsum("cbej,eaki->cbakji", H.ab.vvvo, R.ab, optimize=True)
    # Intermediate 6: X2B(nbkj)*Y2B(cani) -> Z3C(cbakji)
    X3C -= np.einsum("nbkj,cani->cbakji", X.ab.ovoo, T.ab, optimize=True)
    X3C -= np.einsum("nbkj,cani->cbakji", H.ab.ovoo, R.ab, optimize=True)
    # additional terms with T3
    X3C += 0.5 * np.einsum("be,ceakji->cbakji", X.b.vv, T.abb, optimize=True)
    X3C += 0.25 * np.einsum("ce,ebakji->cbakji", X.a.vv, T.abb, optimize=True)
    X3C -= 0.5 * np.einsum("mj,cbakmi->cbakji", X.b.oo, T.abb, optimize=True)
    X3C -= 0.25 * np.einsum("mk,cbamji->cbakji", X.a.oo, T.abb, optimize=True)
    X3C += 0.5 * np.einsum("mnkj,cbamni->cbakji", X.ab.oooo, T.abb, optimize=True)
    X3C += 0.125 * np.einsum("mnij,cbaknm->cbakji", X.bb.oooo, T.abb, optimize=True)
    X3C += 0.5 * np.einsum("cbef,efakji->cbakji", X.ab.vvvv, T.abb, optimize=True)
    X3C += 0.125 * np.einsum("abef,cfekji->cbakji", X.bb.vvvv, T.abb, optimize=True)
    X3C += 0.25 * np.einsum("cnkf,abfijn->cbakji", X.ab.voov, T.bbb, optimize=True)
    X3C += 0.25 * np.einsum("cnkf,fbanji->cbakji", X.aa.voov, T.abb, optimize=True)
    X3C -= 0.5 * np.einsum("mbkf,cfamji->cbakji", X.ab.ovov, T.abb, optimize=True)
    X3C -= 0.5 * np.einsum("cnej,ebakni->cbakji", X.ab.vovo, T.abb, optimize=True)
    X3C += np.einsum("bmje,ceakmi->cbakji", X.bb.voov, T.abb, optimize=True)
    X3C += np.einsum("mbej,ceakmi->cbakji", X.ab.ovvo, T.aab, optimize=True)
    # < ijk~abc~ | (HR3)_C | 0 >
    X3C -= 0.5 * np.einsum("mj,cbakmi->cbakji", H.b.oo, R.abb, optimize=True)
    X3C -= 0.25 * np.einsum("mk,cbamji->cbakji", H.a.oo, R.abb, optimize=True)
    X3C += 0.5 * np.einsum("be,ceakji->cbakji", H.b.vv, R.abb, optimize=True)
    X3C += 0.25 * np.einsum("ce,ebakji->cbakji", H.a.vv, R.abb, optimize=True)
    X3C += 0.125 * np.einsum("mnij,cbaknm->cbakji", H.bb.oooo, R.abb, optimize=True)
    X3C += 0.5 * np.einsum("nmkj,cbanmi->cbakji", H.ab.oooo, R.abb, optimize=True)
    X3C += 0.125 * np.einsum("abef,cfekji->cbakji", H.bb.vvvv, R.abb, optimize=True)
    X3C += 0.5 * np.einsum("cbfe,feakji->cbakji", H.ab.vvvv, R.abb, optimize=True)
    X3C += np.einsum("amie,cbekjm->cbakji", H.bb.voov, R.abb, optimize=True)
    X3C += np.einsum("maei,cebkmj->cbakji", H.ab.ovvo, R.aab, optimize=True)
    X3C += 0.25 * np.einsum("cmke,ebamji->cbakji", H.ab.voov, R.bbb, optimize=True)
    X3C += 0.25 * np.einsum("cmke,ebamji->cbakji", H.aa.voov, R.abb, optimize=True)
    X3C -= 0.5 * np.einsum("mbke,ceamji->cbakji", H.ab.ovov, R.abb, optimize=True)
    X3C -= 0.5 * np.einsum("cmej,ebakmi->cbakji", H.ab.vovo, R.abb, optimize=True)
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4, 5))
    X3C -= np.transpose(X3C, (0, 1, 2, 3, 5, 4))
    return X3C

def build_HR_3D(R, T, H, X):
    # <i~j~k~a~b~c~| [H(R1+R2+R3)]_C | 0 >
    X3D = 0.25 * np.einsum("baje,ecik->abcijk", X.bb.vvov, T.bb, optimize=True)
    X3D += 0.25 * np.einsum("baje,ecik->abcijk", H.bb.vvov, R.bb, optimize=True)
    X3D -= 0.25 * np.einsum("bmji,acmk->abcijk", X.bb.vooo, T.bb, optimize=True)
    X3D -= 0.25 * np.einsum("bmji,acmk->abcijk", H.bb.vooo, R.bb, optimize=True)
    # additional terms with T3 in <ijkabc|[ H(R1+R2)]_C | 0>
    X3D += (1.0 / 12.0) * np.einsum("be,aecijk->abcijk", X.b.vv, T.bbb, optimize=True)
    X3D -= (1.0 / 12.0) * np.einsum("mj,abcimk->abcijk", X.b.oo, T.bbb, optimize=True)
    X3D += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", X.bb.oooo, T.bbb, optimize=True)
    X3D += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", X.bb.vvvv, T.bbb, optimize=True)
    X3D += 0.25 * np.einsum("bmje,aecimk->abcijk", X.bb.voov, T.bbb, optimize=True)
    X3D += 0.25 * np.einsum("mbej,ecamki->abcijk", X.ab.ovvo, T.abb, optimize=True)
    # < i~j~k~a~b~c~ | (HR3)_C | 0 >
    X3D -= (1.0 / 12.0) * np.einsum("mj,abcimk->abcijk", H.b.oo, R.bbb, optimize=True)
    X3D += (1.0 / 12.0) * np.einsum("be,aecijk->abcijk", H.b.vv, R.bbb, optimize=True)
    X3D += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.bb.oooo, R.bbb, optimize=True)
    X3D += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.bb.vvvv, R.bbb, optimize=True)
    X3D += 0.25 * np.einsum("amie,ebcmjk->abcijk", H.bb.voov, R.bbb, optimize=True)
    X3D += 0.25 * np.einsum("maei,ecbmkj->abcijk", H.ab.ovvo, R.abb, optimize=True)
    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3D -= np.transpose(X3D, (0, 1, 2, 3, 5, 4))
    X3D -= np.transpose(X3D, (0, 1, 2, 4, 3, 5)) + np.transpose(X3D, (0, 1, 2, 5, 4, 3))
    X3D -= np.transpose(X3D, (0, 2, 1, 3, 4, 5))
    X3D -= np.transpose(X3D, (1, 0, 2, 3, 4, 5)) + np.transpose(X3D, (2, 1, 0, 3, 4, 5))
    return X3D
