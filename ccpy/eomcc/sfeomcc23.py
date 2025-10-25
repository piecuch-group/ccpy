'''
Spin-Flip Equation-of-Motion Coupled-Cluster Method with Singles and Doubles
in the Reference System and Triple Excitations in the Diagonalization [SF-EOMCC(2,3)]
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
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
            ccpy_einsum("nmfe,fenj->mj", H.ab.oovv, R.ab)
            + 0.5 * ccpy_einsum("mnef,fenj->mj", H.bb.oovv, R.bb)
            - ccpy_einsum("mnje,em->nj", H.ab.ooov, R.b)
    )
    # x(b~e)
    x_vv = (
            -0.5 * ccpy_einsum("mnef,fbnm->be", H.aa.oovv, R.ab)
            - ccpy_einsum("mnef,fbnm->be", H.ab.oovv, R.bb)
            - ccpy_einsum("mbfe,em->bf", H.ab.ovvv, R.b)
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
    x1b = ccpy_einsum("ae,ei->ai", H.b.vv, R.b)
    x1b -= ccpy_einsum("mi,am->ai", H.a.oo, R.b)
    x1b -= ccpy_einsum("maie,em->ai", H.ab.ovov, R.b)
    # <a~i | (H(2) * R2)_C | 0 >
    x1b += ccpy_einsum("me,eami->ai", H.a.ov, R.ab)
    x1b += ccpy_einsum("me,eami->ai", H.b.ov, R.bb)
    x1b -= 0.5 * ccpy_einsum("mnif,fanm->ai", H.aa.ooov, R.ab)
    x1b -= ccpy_einsum("mnif,fanm->ai", H.ab.ooov, R.bb)
    x1b += 0.5 * ccpy_einsum("anef,feni->ai", H.bb.vovv, R.bb)
    x1b += ccpy_einsum("nafe,feni->ai", H.ab.ovvv, R.ab)
    # <a~i | (H(2) * R3)_C | 0 >
    x1b += 0.25 * ccpy_einsum("mnef,efamni->ai", H.aa.oovv, R.aab)
    x1b += ccpy_einsum("mnef,efamni->ai", H.ab.oovv, R.abb)
    x1b += 0.25 * ccpy_einsum("mnef,efamni->ai", H.bb.oovv, R.bbb)
    return x1b

def build_HR_2B(R, T, H, x_oo, x_vv):
    # < ab~ij | (H(2) * R1 + R2)_C | 0 >
    x2b = ccpy_einsum("abie,ej->abij", H.ab.vvov, R.b)
    x2b -= 0.5 * ccpy_einsum("amij,bm->abij", H.aa.vooo, R.b)
    x2b -= ccpy_einsum("mi,abmj->abij", H.a.oo, R.ab)
    x2b += 0.5 * ccpy_einsum("be,aeij->abij", H.b.vv, R.ab)
    x2b += 0.5 * ccpy_einsum("ae,ebij->abij", H.a.vv, R.ab)
    x2b += 0.25 * ccpy_einsum("mnij,abmn->abij", H.aa.oooo, R.ab)
    x2b += 0.5 * ccpy_einsum("abef,efij->abij", H.ab.vvvv, R.ab)
    x2b += ccpy_einsum("amie,ebmj->abij", H.aa.voov, R.ab)
    x2b += ccpy_einsum("amie,ebmj->abij", H.ab.voov, R.bb)
    x2b -= ccpy_einsum("mbie,aemj->abij", H.ab.ovov, R.ab)
    x2b -= ccpy_einsum("mj,abim->abij", x_oo, T.ab)
    x2b += 0.5 * ccpy_einsum("be,aeij->abij", x_vv, T.aa)
    # < ab~ij | (H(2) * R3)_C | 0 >
    x2b += 0.5 * ccpy_einsum("me,aebimj->abij", H.a.ov, R.aab)
    x2b += 0.5 * ccpy_einsum("me,aebimj->abij", H.b.ov, R.abb)
    x2b -= 0.5 * ccpy_einsum("mnif,afbmnj->abij", H.aa.ooov, R.aab)
    x2b -= ccpy_einsum("mnif,afbmnj->abij", H.ab.ooov, R.abb)
    x2b += 0.25 * ccpy_einsum("anef,efbinj->abij", H.aa.vovv, R.aab)
    x2b += 0.5 * ccpy_einsum("anef,efbinj->abij", H.ab.vovv, R.abb)
    x2b += 0.5 * ccpy_einsum("nbfe,afeinj->abij", H.ab.ovvv, R.aab)
    x2b += 0.25 * ccpy_einsum("bnef,afeinj->abij", H.bb.vovv, R.abb)
    # antisymmetrize (ij)
    x2b -= np.transpose(x2b, (0, 1, 3, 2))
    return x2b

def build_HR_2C(R, T, H, x_oo, x_vv):
    # < a~b~i~j | (H(2) * R1 + R2)_C | 0 >
    x2c = 0.5 * ccpy_einsum("abie,ej->abij", H.bb.vvov, R.b)
    x2c -= ccpy_einsum("maji,bm->abij", H.ab.ovoo, R.b)
    x2c -= 0.5 * ccpy_einsum("mj,abim->abij", H.a.oo, R.bb)
    x2c -= 0.5 * ccpy_einsum("mi,abmj->abij", H.b.oo, R.bb)
    x2c += ccpy_einsum("ae,ebij->abij", H.b.vv, R.bb)
    x2c += 0.25 * ccpy_einsum("abef,efij->abij", H.bb.vvvv, R.bb)
    x2c += 0.5 * ccpy_einsum("nmji,abmn->abij", H.ab.oooo, R.bb)
    x2c += ccpy_einsum("amie,ebmj->abij", H.bb.voov, R.bb)
    x2c += ccpy_einsum("maei,ebmj->abij", H.ab.ovvo, R.ab)
    x2c -= ccpy_einsum("maje,ebim->abij", H.ab.ovov, R.bb)
    x2c -= 0.5 * ccpy_einsum("mj,abim->abij", x_oo, T.bb)
    x2c += ccpy_einsum("be,eaji->abij", x_vv, T.ab)
    # < ab~ij | (H(2) * R3)_C | 0 >
    x2c += 0.5 * ccpy_einsum("me,eabmij->abij", H.a.ov, R.abb)
    x2c += 0.5 * ccpy_einsum("me,eabmij->abij", H.b.ov, R.bbb)
    x2c -= 0.5 * ccpy_einsum("nmfi,fabnmj->abij", H.ab.oovo, R.abb)
    x2c -= 0.25 * ccpy_einsum("mnif,fabnmj->abij", H.bb.ooov, R.bbb)
    x2c -= 0.25 * ccpy_einsum("mnjf,fabnim->abij", H.aa.ooov, R.abb)
    x2c -= 0.5 * ccpy_einsum("mnjf,fabnim->abij", H.ab.ooov, R.bbb)
    x2c += ccpy_einsum("nafe,febnij->abij", H.ab.ovvv, R.abb)
    x2c += 0.5 * ccpy_einsum("anef,febnij->abij", H.bb.vovv, R.bbb)
    # antisymmetrize (ab)
    x2c -= np.transpose(x2c, (1, 0, 2, 3))
    return x2c

def build_HR_3B(R, T, H, X):
    # < abc~ijk | (H(2)*(R1 + R2))_C | 0 >
    x3b = -(6.0 / 12.0) * ccpy_einsum("amij,bcmk->abcijk", H.aa.vooo, R.ab)
    x3b += (3.0 / 12.0) * ccpy_einsum("abie,ecjk->abcijk", H.aa.vvov, R.ab)
    x3b += (6.0 / 12.0) * ccpy_einsum("acie,bejk->abcijk", H.ab.vvov, R.ab)
    #
    x3b -= (3.0 / 12.0) * ccpy_einsum("mcjk,abim->abcijk", X["ab"]["ovoo"], T.aa)
    x3b += (6.0 / 12.0) * ccpy_einsum("bcek,aeij->abcijk", X["ab"]["vvvo"], T.aa)
    x3b -= (6.0 / 12.0) * ccpy_einsum("amik,bcjm->abcijk", X["ab"]["vooo"], T.ab)
    # < abc~ijk | (H(2) * R3)_C | 0 >
    x3b -= (3.0 / 12.0) * ccpy_einsum("mj,abcimk->abcijk", H.a.oo, R.aab)
    x3b += (2.0 / 12.0) * ccpy_einsum("be,aecijk->abcijk", H.a.vv, R.aab)
    x3b += (1.0 / 12.0) * ccpy_einsum("ce,abeijk->abcijk", H.b.vv, R.aab)
    x3b += (3.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", H.aa.oooo, R.aab)
    x3b += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", H.aa.vvvv, R.aab)
    x3b += (2.0 / 12.0) * ccpy_einsum("bcef,aefijk->abcijk", H.ab.vvvv, R.aab)
    x3b += (6.0 / 12.0) * ccpy_einsum("amie,becjmk->abcijk", H.aa.voov, R.aab)
    x3b += (6.0 / 12.0) * ccpy_einsum("amie,becjmk->abcijk", H.ab.voov, R.abb)
    x3b -= (2.0 / 12.0) * ccpy_einsum("mcie,abemjk->abcijk", H.ab.ovov, R.aab)
    # antisymmetrize (ab)(ijk)
    x3b -= np.transpose(x3b, (1, 0, 2, 3, 4, 5)) # (ab)
    x3b -= np.transpose(x3b, (0, 1, 2, 3, 5, 4)) # (jk)
    x3b -= np.transpose(x3b, (0, 1, 2, 4, 3, 5)) + np.transpose(x3b, (0, 1, 2, 5, 4, 3)) # (i/jk)
    return x3b

def build_HR_3C(R, T, H, X):
    # < ab~c~ij~k | (H(2)*(R1 + R2))_C | 0 >
    x3c = -ccpy_einsum("mbij,acmk->abcijk", H.ab.ovoo, R.ab)
    x3c -= 0.25 * ccpy_einsum("amik,bcjm->abcijk", H.aa.vooo, R.bb)
    x3c -= 0.5 * ccpy_einsum("amij,bcmk->abcijk", H.ab.vooo, R.bb)
    x3c += ccpy_einsum("abie,ecjk->abcijk", H.ab.vvov, R.bb)
    x3c += 0.5 * ccpy_einsum("abej,ecik->abcijk", H.ab.vvvo, R.ab)
    x3c += 0.25 * ccpy_einsum("bcje,aeik->abcijk", H.bb.vvov, R.ab)
    #
    x3c -= 0.5 * ccpy_einsum("mcik,abmj->abcijk", X["ab"]["ovoo"], T.ab)
    x3c -= ccpy_einsum("mcjk,abim->abcijk", X["bb"]["ovoo"], T.ab)
    x3c -= 0.25 * ccpy_einsum("amik,bcjm->abcijk", X["ab"]["vooo"], T.bb)
    x3c += ccpy_einsum("acek,ebij->abcijk", X["ab"]["vvvo"], T.ab)
    x3c += 0.5 * ccpy_einsum("bcek,aeij->abcijk", X["bb"]["vvvo"], T.ab)
    x3c += 0.25 * ccpy_einsum("bcje,aeik->abcijk", X["bb"]["vvov"], T.aa)
    # < ab~c~ij~k | (H(2) * R3)_C | 0 >
    x3c -= 0.5 * ccpy_einsum("mi,abcmjk->abcijk", H.a.oo, R.abb)
    x3c -= 0.25 * ccpy_einsum("mj,abcimk->abcijk", H.b.oo, R.abb)
    x3c += 0.25 * ccpy_einsum("ae,ebcijk->abcijk", H.a.vv, R.abb)
    x3c += 0.5 * ccpy_einsum("be,aecijk->abcijk", H.b.vv, R.abb)
    x3c += 0.125 * ccpy_einsum("mnik,abcmjn->abcijk", H.aa.oooo, R.abb)
    x3c += 0.5 * ccpy_einsum("mnij,abcmnk->abcijk", H.ab.oooo, R.abb)
    x3c += 0.5 * ccpy_einsum("acef,ebfijk->abcijk", H.ab.vvvv, R.abb)
    x3c += 0.125 * ccpy_einsum("bcef,aefijk->abcijk", H.bb.vvvv, R.abb)
    x3c += 0.5 * ccpy_einsum("amie,ebcmjk->abcijk", H.aa.voov, R.abb)
    x3c += 0.5 * ccpy_einsum("amie,ebcmjk->abcijk", H.ab.voov, R.bbb)
    x3c += 0.5 * ccpy_einsum("mbej,aecimk->abcijk", H.ab.ovvo, R.aab)
    x3c += 0.5 * ccpy_einsum("bmje,aecimk->abcijk", H.bb.voov, R.abb)
    x3c -= 0.25 * ccpy_einsum("amej,ebcimk->abcijk", H.ab.vovo, R.abb)
    x3c -= ccpy_einsum("mbie,aecmjk->abcijk", H.ab.ovov, R.abb)
    # antisymmetrize (bc)(ik)
    x3c -= np.transpose(x3c, (0, 2, 1, 3, 4, 5)) # (bc)
    x3c -= np.transpose(x3c, (0, 1, 2, 5, 4, 3)) # (ik)
    return x3c

def build_HR_3D(R, T, H, X):
    # < a~b~c~i~j~k | (H(2)*(R1 + R2))_C | 0 >
    x3d = -(3.0 / 12.0) * ccpy_einsum("amij,bcmk->abcijk", H.bb.vooo, R.bb)
    x3d -= (6.0 / 12.0) * ccpy_einsum("maki,bcjm->abcijk", H.ab.ovoo, R.bb)
    x3d += (6.0 / 12.0) * ccpy_einsum("abie,ecjk->abcijk", H.bb.vvov, R.bb)
    #
    x3d -= (6.0 / 12.0) * ccpy_einsum("mcjk,abim->abcijk", X["bb"]["ovoo"], T.bb)
    x3d += (6.0 / 12.0) * ccpy_einsum("bcje,eaki->abcijk", X["bb"]["vvov"], T.ab)
    x3d += (3.0 / 12.0) * ccpy_einsum("bcek,aeij->abcijk", X["bb"]["vvvo"], T.bb)
    # < a~b~c~i~j~k | (H(2) * R3)_C | 0 >
    x3d -= (2.0 / 12.0) * ccpy_einsum("mi,abcmjk->abcijk", H.b.oo, R.bbb)
    x3d -= (1.0 / 12.0) * ccpy_einsum("mk,abcijm->abcijk", H.a.oo, R.bbb)
    x3d += (3.0 / 12.0) * ccpy_einsum("ce,abeijk->abcijk", H.b.vv, R.bbb)
    x3d += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", H.bb.oooo, R.bbb)
    x3d += (2.0 / 12.0) * ccpy_einsum("mnkj,abcinm->abcijk", H.ab.oooo, R.bbb)
    x3d += (3.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", H.bb.vvvv, R.bbb)
    x3d += (6.0 / 12.0) * ccpy_einsum("maei,ebcmjk->abcijk", H.ab.ovvo, R.abb)
    x3d += (6.0 / 12.0) * ccpy_einsum("amie,ebcmjk->abcijk", H.bb.voov, R.bbb)
    x3d -= (3.0 / 12.0) * ccpy_einsum("mbke,aecijm->abcijk", H.ab.ovov, R.bbb)
    # antisymmetrize (abc)(ij)
    x3d -= np.transpose(x3d, (0, 1, 2, 4, 3, 5)) # (ij)
    x3d -= np.transpose(x3d, (0, 2, 1, 3, 4, 5)) # (bc)
    x3d -= np.transpose(x3d, (1, 0, 2, 3, 4, 5)) + np.transpose(x3d, (2, 1, 0, 3, 4, 5)) # (a/bc)
    return x3d