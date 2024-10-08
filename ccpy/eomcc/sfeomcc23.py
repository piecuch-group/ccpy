"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for excited states using
the spin-flip equation-of-motion (EOM) CC with singles and doubles
diagonalized in the space of singles, doubles, and triples [SF-EOMCC(2,3)]."""

import numpy as np
from ccpy.eomcc.sfeomcc23_intermediates import get_sfeomcc23_intermediates
from ccpy.lib.core import cc_loops2

def update(R, omega, H, RHF_symmetry, system):

    R.b, R.ab, R.bb, R.aab, R.abb, R.bbb = cc_loops2.update_r_sfccsdt(
        R.b,
        R.ab,
        R.bb,
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
    return R

def HR(dR, R, T, H, flag_RHF, system):

    # Intermediates used in terms with 3-body HBar
    # x(m~j)
    x_oo = (
            np.einsum("nmfe,fenj->mj", H.ab.oovv, R.ab, optimize=True)
            + 0.5 * np.einsum("mnef,fenj->mj", H.bb.oovv, R.bb, optimize=True)
            - np.einsum("mnje,em->nj", H.ab.ooov, R.b, optimize=True)
    )
    # x(b~e)
    x_vv = (
            -0.5 * np.einsum("mnef,fbnm->be", H.aa.oovv, R.ab, optimize=True)
            - np.einsum("mnef,fbnm->be", H.ab.oovv, R.bb, optimize=True)
            - np.einsum("mbfe,em->bf", H.ab.ovvv, R.b, optimize=True)
    )
    # all other X intermediates
    X = get_sfeomcc23_intermediates(H, R, T, system)

    # update R1
    dR.b = build_HR_1B(R, T, H)
    # update R2
    dR.ab = build_HR_2B(R, T, H, x_oo, x_vv)
    dR.bb = build_HR_2C(R, T, H, x_oo, x_vv)
    # update R3
    dR.aab = build_HR_3B(R, T, H, X)
    dR.abb = build_HR_3C(R, T, H, X)
    dR.bbb = build_HR_3D(R, T, H, X)

    return dR.flatten()

def build_HR_1B(R, T, H):
    # < a~i | (H(2) * R1)_C | 0 >
    x1b = np.einsum("ae,ei->ai", H.b.vv, R.b, optimize=True)
    x1b -= np.einsum("mi,am->ai", H.a.oo, R.b, optimize=True)
    x1b -= np.einsum("maie,em->ai", H.ab.ovov, R.b, optimize=True)
    # <a~i | (H(2) * R2)_C | 0 >
    x1b += np.einsum("me,eami->ai", H.a.ov, R.ab, optimize=True)
    x1b += np.einsum("me,eami->ai", H.b.ov, R.bb, optimize=True)
    x1b -= 0.5 * np.einsum("mnif,fanm->ai", H.aa.ooov, R.ab, optimize=True)
    x1b -= np.einsum("mnif,fanm->ai", H.ab.ooov, R.bb, optimize=True)
    x1b += 0.5 * np.einsum("anef,feni->ai", H.bb.vovv, R.bb, optimize=True)
    x1b += np.einsum("nafe,feni->ai", H.ab.ovvv, R.ab, optimize=True)
    # <a~i | (H(2) * R3)_C | 0 >
    x1b += 0.25 * np.einsum("mnef,efamni->ai", H.aa.oovv, R.aab, optimize=True)
    x1b += np.einsum("mnef,efamni->ai", H.ab.oovv, R.abb, optimize=True)
    x1b += 0.25 * np.einsum("mnef,efamni->ai", H.bb.oovv, R.bbb, optimize=True)
    return x1b

def build_HR_2B(R, T, H, x_oo, x_vv):
    # < ab~ij | (H(2) * R1 + R2)_C | 0 >
    x2b = np.einsum("abie,ej->abij", H.ab.vvov, R.b, optimize=True)
    x2b -= 0.5 * np.einsum("amij,bm->abij", H.aa.vooo, R.b, optimize=True)
    x2b -= np.einsum("mi,abmj->abij", H.a.oo, R.ab, optimize=True)
    x2b += 0.5 * np.einsum("be,aeij->abij", H.b.vv, R.ab, optimize=True)
    x2b += 0.5 * np.einsum("ae,ebij->abij", H.a.vv, R.ab, optimize=True)
    x2b += 0.25 * np.einsum("mnij,abmn->abij", H.aa.oooo, R.ab, optimize=True)
    x2b += 0.5 * np.einsum("abef,efij->abij", H.ab.vvvv, R.ab, optimize=True)
    x2b += np.einsum("amie,ebmj->abij", H.aa.voov, R.ab, optimize=True)
    x2b += np.einsum("amie,ebmj->abij", H.ab.voov, R.bb, optimize=True)
    x2b -= np.einsum("mbie,aemj->abij", H.ab.ovov, R.ab, optimize=True)
    x2b -= np.einsum("mj,abim->abij", x_oo, T.ab, optimize=True)
    x2b += 0.5 * np.einsum("be,aeij->abij", x_vv, T.aa, optimize=True)
    # < ab~ij | (H(2) * R3)_C | 0 >
    x2b += 0.5 * np.einsum("me,aebimj->abij", H.a.ov, R.aab, optimize=True)
    x2b += 0.5 * np.einsum("me,aebimj->abij", H.b.ov, R.abb, optimize=True)
    x2b -= 0.5 * np.einsum("mnif,afbmnj->abij", H.aa.ooov, R.aab, optimize=True)
    x2b -= np.einsum("mnif,afbmnj->abij", H.ab.ooov, R.abb, optimize=True)
    x2b += 0.25 * np.einsum("anef,efbinj->abij", H.aa.vovv, R.aab, optimize=True)
    x2b += 0.5 * np.einsum("anef,efbinj->abij", H.ab.vovv, R.abb, optimize=True)
    x2b += 0.5 * np.einsum("nbfe,afeinj->abij", H.ab.ovvv, R.aab, optimize=True)
    x2b += 0.25 * np.einsum("bnef,afeinj->abij", H.bb.vovv, R.abb, optimize=True)
    # antisymmetrize (ij)
    x2b -= np.transpose(x2b, (0, 1, 3, 2))
    return x2b

def build_HR_2C(R, T, H, x_oo, x_vv):
    # < a~b~i~j | (H(2) * R1 + R2)_C | 0 >
    x2c = 0.5 * np.einsum("abie,ej->abij", H.bb.vvov, R.b, optimize=True)
    x2c -= np.einsum("maji,bm->abij", H.ab.ovoo, R.b, optimize=True)
    x2c -= 0.5 * np.einsum("mj,abim->abij", H.a.oo, R.bb, optimize=True)
    x2c -= 0.5 * np.einsum("mi,abmj->abij", H.b.oo, R.bb, optimize=True)
    x2c += np.einsum("ae,ebij->abij", H.b.vv, R.bb, optimize=True)
    x2c += 0.25 * np.einsum("abef,efij->abij", H.bb.vvvv, R.bb, optimize=True)
    x2c += 0.5 * np.einsum("nmji,abmn->abij", H.ab.oooo, R.bb, optimize=True)
    x2c += np.einsum("amie,ebmj->abij", H.bb.voov, R.bb, optimize=True)
    x2c += np.einsum("maei,ebmj->abij", H.ab.ovvo, R.ab, optimize=True)
    x2c -= np.einsum("maje,ebim->abij", H.ab.ovov, R.bb, optimize=True)
    x2c -= 0.5 * np.einsum("mj,abim->abij", x_oo, T.bb, optimize=True)
    x2c += np.einsum("be,eaji->abij", x_vv, T.ab, optimize=True)
    # < ab~ij | (H(2) * R3)_C | 0 >
    x2c += 0.5 * np.einsum("me,eabmij->abij", H.a.ov, R.abb, optimize=True)
    x2c += 0.5 * np.einsum("me,eabmij->abij", H.b.ov, R.bbb, optimize=True)
    x2c -= 0.5 * np.einsum("nmfi,fabnmj->abij", H.ab.oovo, R.abb, optimize=True)
    x2c -= 0.25 * np.einsum("mnif,fabnmj->abij", H.bb.ooov, R.bbb, optimize=True)
    x2c -= 0.25 * np.einsum("mnjf,fabnim->abij", H.aa.ooov, R.abb, optimize=True)
    x2c -= 0.5 * np.einsum("mnjf,fabnim->abij", H.ab.ooov, R.bbb, optimize=True)
    x2c += np.einsum("nafe,febnij->abij", H.ab.ovvv, R.abb, optimize=True)
    x2c += 0.5 * np.einsum("anef,febnij->abij", H.bb.vovv, R.bbb, optimize=True)
    # antisymmetrize (ab)
    x2c -= np.transpose(x2c, (1, 0, 2, 3))
    return x2c

def build_HR_3B(R, T, H, X):
    # < abc~ijk | (H(2)*(R1 + R2))_C | 0 >
    x3b = -(6.0 / 12.0) * np.einsum("amij,bcmk->abcijk", H.aa.vooo, R.ab, optimize=True)
    x3b += (3.0 / 12.0) * np.einsum("abie,ecjk->abcijk", H.aa.vvov, R.ab, optimize=True)
    x3b += (6.0 / 12.0) * np.einsum("acie,bejk->abcijk", H.ab.vvov, R.ab, optimize=True)
    #
    x3b -= (3.0 / 12.0) * np.einsum("mcjk,abim->abcijk", X["ab"]["ovoo"], T.aa, optimize=True)
    x3b += (6.0 / 12.0) * np.einsum("bcek,aeij->abcijk", X["ab"]["vvvo"], T.aa, optimize=True)
    x3b -= (6.0 / 12.0) * np.einsum("amik,bcjm->abcijk", X["ab"]["vooo"], T.ab, optimize=True)
    # < abc~ijk | (H(2) * R3)_C | 0 >
    x3b -= (3.0 / 12.0) * np.einsum("mj,abcimk->abcijk", H.a.oo, R.aab, optimize=True)
    x3b += (2.0 / 12.0) * np.einsum("be,aecijk->abcijk", H.a.vv, R.aab, optimize=True)
    x3b += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.b.vv, R.aab, optimize=True)
    x3b += (3.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, R.aab, optimize=True)
    x3b += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, R.aab, optimize=True)
    x3b += (2.0 / 12.0) * np.einsum("bcef,aefijk->abcijk", H.ab.vvvv, R.aab, optimize=True)
    x3b += (6.0 / 12.0) * np.einsum("amie,becjmk->abcijk", H.aa.voov, R.aab, optimize=True)
    x3b += (6.0 / 12.0) * np.einsum("amie,becjmk->abcijk", H.ab.voov, R.abb, optimize=True)
    x3b -= (2.0 / 12.0) * np.einsum("mcie,abemjk->abcijk", H.ab.ovov, R.aab, optimize=True)
    # antisymmetrize (ab)(ijk)
    x3b -= np.transpose(x3b, (1, 0, 2, 3, 4, 5)) # (ab)
    x3b -= np.transpose(x3b, (0, 1, 2, 3, 5, 4)) # (jk)
    x3b -= np.transpose(x3b, (0, 1, 2, 4, 3, 5)) + np.transpose(x3b, (0, 1, 2, 5, 4, 3)) # (i/jk)
    return x3b

def build_HR_3C(R, T, H, X):
    # < ab~c~ij~k | (H(2)*(R1 + R2))_C | 0 >
    x3c = -np.einsum("mbij,acmk->abcijk", H.ab.ovoo, R.ab, optimize=True)
    x3c -= 0.25 * np.einsum("amik,bcjm->abcijk", H.aa.vooo, R.bb, optimize=True)
    x3c -= 0.5 * np.einsum("amij,bcmk->abcijk", H.ab.vooo, R.bb, optimize=True)
    x3c += np.einsum("abie,ecjk->abcijk", H.ab.vvov, R.bb, optimize=True)
    x3c += 0.5 * np.einsum("abej,ecik->abcijk", H.ab.vvvo, R.ab, optimize=True)
    x3c += 0.25 * np.einsum("bcje,aeik->abcijk", H.bb.vvov, R.ab, optimize=True)
    #
    x3c -= 0.5 * np.einsum("mcik,abmj->abcijk", X["ab"]["ovoo"], T.ab, optimize=True)
    x3c -= np.einsum("mcjk,abim->abcijk", X["bb"]["ovoo"], T.ab, optimize=True)
    x3c -= 0.25 * np.einsum("amik,bcjm->abcijk", X["ab"]["vooo"], T.bb, optimize=True)
    x3c += np.einsum("acek,ebij->abcijk", X["ab"]["vvvo"], T.ab, optimize=True)
    x3c += 0.5 * np.einsum("bcek,aeij->abcijk", X["bb"]["vvvo"], T.ab, optimize=True)
    x3c += 0.25 * np.einsum("bcje,aeik->abcijk", X["bb"]["vvov"], T.aa, optimize=True)
    # < ab~c~ij~k | (H(2) * R3)_C | 0 >
    x3c -= 0.5 * np.einsum("mi,abcmjk->abcijk", H.a.oo, R.abb, optimize=True)
    x3c -= 0.25 * np.einsum("mj,abcimk->abcijk", H.b.oo, R.abb, optimize=True)
    x3c += 0.25 * np.einsum("ae,ebcijk->abcijk", H.a.vv, R.abb, optimize=True)
    x3c += 0.5 * np.einsum("be,aecijk->abcijk", H.b.vv, R.abb, optimize=True)
    x3c += 0.125 * np.einsum("mnik,abcmjn->abcijk", H.aa.oooo, R.abb, optimize=True)
    x3c += 0.5 * np.einsum("mnij,abcmnk->abcijk", H.ab.oooo, R.abb, optimize=True)
    x3c += 0.5 * np.einsum("acef,ebfijk->abcijk", H.ab.vvvv, R.abb, optimize=True)
    x3c += 0.125 * np.einsum("bcef,aefijk->abcijk", H.bb.vvvv, R.abb, optimize=True)
    x3c += 0.5 * np.einsum("amie,ebcmjk->abcijk", H.aa.voov, R.abb, optimize=True)
    x3c += 0.5 * np.einsum("amie,ebcmjk->abcijk", H.ab.voov, R.bbb, optimize=True)
    x3c += 0.5 * np.einsum("mbej,aecimk->abcijk", H.ab.ovvo, R.aab, optimize=True)
    x3c += 0.5 * np.einsum("bmje,aecimk->abcijk", H.bb.voov, R.abb, optimize=True)
    x3c -= 0.25 * np.einsum("amej,ebcimk->abcijk", H.ab.vovo, R.abb, optimize=True)
    x3c -= np.einsum("mbie,aecmjk->abcijk", H.ab.ovov, R.abb, optimize=True)
    # antisymmetrize (bc)(ik)
    x3c -= np.transpose(x3c, (0, 2, 1, 3, 4, 5)) # (bc)
    x3c -= np.transpose(x3c, (0, 1, 2, 5, 4, 3)) # (ik)
    return x3c

def build_HR_3D(R, T, H, X):
    # < a~b~c~i~j~k | (H(2)*(R1 + R2))_C | 0 >
    x3d = -(3.0 / 12.0) * np.einsum("amij,bcmk->abcijk", H.bb.vooo, R.bb, optimize=True)
    x3d -= (6.0 / 12.0) * np.einsum("maki,bcjm->abcijk", H.ab.ovoo, R.bb, optimize=True)
    x3d += (6.0 / 12.0) * np.einsum("abie,ecjk->abcijk", H.bb.vvov, R.bb, optimize=True)
    #
    x3d -= (6.0 / 12.0) * np.einsum("mcjk,abim->abcijk", X["bb"]["ovoo"], T.bb, optimize=True)
    x3d += (6.0 / 12.0) * np.einsum("bcje,eaki->abcijk", X["bb"]["vvov"], T.ab, optimize=True)
    x3d += (3.0 / 12.0) * np.einsum("bcek,aeij->abcijk", X["bb"]["vvvo"], T.bb, optimize=True)
    # < a~b~c~i~j~k | (H(2) * R3)_C | 0 >
    x3d -= (2.0 / 12.0) * np.einsum("mi,abcmjk->abcijk", H.b.oo, R.bbb, optimize=True)
    x3d -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.a.oo, R.bbb, optimize=True)
    x3d += (3.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.b.vv, R.bbb, optimize=True)
    x3d += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.bb.oooo, R.bbb, optimize=True)
    x3d += (2.0 / 12.0) * np.einsum("mnkj,abcinm->abcijk", H.ab.oooo, R.bbb, optimize=True)
    x3d += (3.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.bb.vvvv, R.bbb, optimize=True)
    x3d += (6.0 / 12.0) * np.einsum("maei,ebcmjk->abcijk", H.ab.ovvo, R.abb, optimize=True)
    x3d += (6.0 / 12.0) * np.einsum("amie,ebcmjk->abcijk", H.bb.voov, R.bbb, optimize=True)
    x3d -= (3.0 / 12.0) * np.einsum("mbke,aecijm->abcijk", H.ab.ovov, R.bbb, optimize=True)
    # antisymmetrize (abc)(ij)
    x3d -= np.transpose(x3d, (0, 1, 2, 4, 3, 5)) # (ij)
    x3d -= np.transpose(x3d, (0, 2, 1, 3, 4, 5)) # (bc)
    x3d -= np.transpose(x3d, (1, 0, 2, 3, 4, 5)) + np.transpose(x3d, (2, 1, 0, 3, 4, 5)) # (a/bc)
    return x3d