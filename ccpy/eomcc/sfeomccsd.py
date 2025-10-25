'''
Spin-Flip Equation-of-Motion Coupled-Cluster Method with Singles and Doubles [SF-EOMCCSD]
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import cc_loops2

def update(R, omega, H, RHF_symmetry, system):

    R.b, R.ab, R.bb = cc_loops2.update_r_sfccsd(
        R.b,
        R.ab,
        R.bb,
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
        0.0,
    )
    return R


def HR(dR, R, T, H, flag_RHF, system):

    # update R1
    dR.b = build_HR_1B(R, T, H)

    # Intermediates used in terms with 3-body HBar
    x_oo = (
            ccpy_einsum("nmfe,fenj->mj", H.ab.oovv, R.ab)
            + 0.5 * ccpy_einsum("mnef,fenj->mj", H.bb.oovv, R.bb)
            - ccpy_einsum("mnje,em->nj", H.ab.ooov, R.b)
    )
    x_vv = (
            -0.5 * ccpy_einsum("mnef,fbnm->be", H.aa.oovv, R.ab)
            - ccpy_einsum("mnef,fbnm->be", H.ab.oovv, R.bb)
            - ccpy_einsum("mbfe,em->bf", H.ab.ovvv, R.b)
    )

    # update R2
    dR.ab = build_HR_2B(R, T, H, x_oo, x_vv)
    dR.bb = build_HR_2C(R, T, H, x_oo, x_vv)

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
    # antisymmetrize (ab)
    x2c -= np.transpose(x2c, (1, 0, 2, 3))
    return x2c