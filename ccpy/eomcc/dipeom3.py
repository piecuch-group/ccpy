'''
Double Ionization Potential Equation-of-Motion Coupled-Cluster Method
with 2h and 3h-1p Excitations on top of CCSD [DIP-EOMCCSD(3h-1p)]
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import cc_loops2

def update(R, omega, H, RHF_symmetry, system):
    R.ab, R.aba, R.abb = cc_loops2.update_r_3h1p(
        R.ab,
        R.aba,
        R.abb,
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
        0.0,
    )
    if RHF_symmetry:
        R.abb = np.transpose(R.aba, (1, 0, 2, 3))
    return R

def HR(dR, R, T, H, flag_RHF, system):

    # update R2
    dR.ab = build_HR_2B(R, T, H)
    # update R3
    dR.aba = build_HR_3B(R, T, H)
    if flag_RHF:
        dR.abb = np.transpose(dR.aba, (1, 0, 2, 3))
    else:
        dR.abb = build_HR_3C(R, T, H)

    return dR.flatten()

def build_HR_2B(R, T, H):
    x2b = -ccpy_einsum("mi,mj->ij", H.a.oo, R.ab)
    x2b -= ccpy_einsum("mj,im->ij", H.b.oo, R.ab)
    x2b += ccpy_einsum("mnij,mn->ij", H.ab.oooo, R.ab)
    x2b += ccpy_einsum("me,ijem->ij", H.a.ov, R.aba)
    x2b += ccpy_einsum("me,ijem->ij", H.b.ov, R.abb)
    x2b -= ccpy_einsum("nmfj,imfn->ij", H.ab.oovo, R.aba)
    x2b -= 0.5 * ccpy_einsum("mnjf,imfn->ij", H.bb.ooov, R.abb)
    x2b -= 0.5 * ccpy_einsum("mnif,mjfn->ij", H.aa.ooov, R.aba)
    x2b -= ccpy_einsum("mnif,mjfn->ij", H.ab.ooov, R.abb)
    return x2b

def build_HR_3B(R, T, H):
    x3b = -0.5 * ccpy_einsum("cmki,mj->ijck", H.aa.vooo, R.ab)
    x3b -= ccpy_einsum("cmkj,im->ijck", H.ab.vooo, R.ab)
    x3b -= ccpy_einsum("mk,ijcm->ijck", H.a.oo, R.aba)
    x3b -= 0.5 * ccpy_einsum("mj,imck->ijck", H.b.oo, R.aba)
    x3b += 0.5 * ccpy_einsum("ce,ijek->ijck", H.a.vv, R.aba)
    x3b += ccpy_einsum("cmke,ijem->ijck", H.aa.voov, R.aba)
    x3b += ccpy_einsum("cmke,ijem->ijck", H.ab.voov, R.abb)
    x3b += ccpy_einsum("mnij,mnck->ijck", H.ab.oooo, R.aba)
    x3b += 0.25 * ccpy_einsum("mnik,mjcn->ijck", H.aa.oooo, R.aba)
    x3b -= 0.5 * ccpy_einsum("cmej,imek->ijck", H.ab.vovo, R.aba)

    x_ov = (
            ccpy_einsum("mnie,mn->ie", H.ab.ooov, R.ab)
            - ccpy_einsum("nmfe,imfn->ie", H.ab.oovv, R.aba)
            - 0.5 * ccpy_einsum("nmfe,imfn->ie", H.bb.oovv, R.abb)
    )
    x_vo = (
            ccpy_einsum("mnej,mn->ej", H.ab.oovo, R.ab)
            - 0.5 * ccpy_einsum("mnef,mjfn->ej", H.aa.oovv, R.aba)
            - ccpy_einsum("mnef,mjfn->ej", H.ab.oovv, R.abb)
    )

    x3b += ccpy_einsum("ie,cekj->ijck", x_ov, T.ab)
    x3b += 0.5 * ccpy_einsum("ej,ecik->ijck", x_vo, T.aa)

    # antisymmetrize A(ik)
    x3b -= np.transpose(x3b, (3, 1, 2, 0))
    return x3b

def build_HR_3C(R, T, H):
    x3c = -ccpy_einsum("mcik,mj->ijck", H.ab.ovoo, R.ab)
    x3c -= 0.5 * ccpy_einsum("cmkj,im->ijck", H.bb.vooo, R.ab)
    x3c -= 0.5 * ccpy_einsum("mi,mjck->ijck", H.a.oo, R.abb)
    x3c -= ccpy_einsum("mj,imck->ijck", H.b.oo, R.abb)
    x3c += 0.5 * ccpy_einsum("ce,ijek->ijck", H.b.vv, R.abb)
    x3c += ccpy_einsum("mnij,mnck->ijck", H.ab.oooo, R.abb)
    x3c += 0.25 * ccpy_einsum("mnjk,imcn->ijck", H.bb.oooo, R.abb)
    x3c += ccpy_einsum("mcek,ijem->ijck", H.ab.ovvo, R.aba)
    x3c += ccpy_einsum("cmke,ijem->ijck", H.bb.voov, R.abb)
    x3c -= 0.5 * ccpy_einsum("mcie,mjek->ijck", H.ab.ovov, R.abb)

    x_ov = (
            ccpy_einsum("mnie,mn->ie", H.ab.ooov, R.ab)
            - ccpy_einsum("nmfe,imfn->ie", H.ab.oovv, R.aba)
            - 0.5 * ccpy_einsum("nmfe,imfn->ie", H.bb.oovv, R.abb)
    )
    x_vo = (
            ccpy_einsum("mnej,mn->ej", H.ab.oovo, R.ab)
            - 0.5 * ccpy_einsum("mnef,mjfn->ej", H.aa.oovv, R.aba)
            - ccpy_einsum("mnef,mjfn->ej", H.ab.oovv, R.abb)
    )

    x3c += ccpy_einsum("ej,ecik->ijck", x_vo, T.ab)
    x3c += 0.5 * ccpy_einsum("ie,ecjk->ijck", x_ov, T.bb)

    # antisymmetrize A(j~k~)
    x3c -= ccpy_einsum("ijck->ikcj", x3c)
    return x3c
