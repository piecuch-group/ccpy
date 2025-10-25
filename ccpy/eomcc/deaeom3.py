'''
Double Electron Attachment Equation-of-Motion Coupled-Cluster
Method with 2p and 3p-1h Excitations on top of CCSD [DEA-EOMCCSD(3p-1h)]
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import cc_loops2

def update(R, omega, H, RHF_symmetry, system):
    R.ab, R.aba, R.abb = cc_loops2.update_r_3p1h(
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
    x2b = ccpy_einsum("ae,eb->ab", H.a.vv, R.ab)
    x2b += ccpy_einsum("be,ae->ab", H.b.vv, R.ab)
    x2b += ccpy_einsum("abef,ef->ab", H.ab.vvvv, R.ab)
    x2b += ccpy_einsum("me,abem->ab", H.a.ov, R.aba)
    x2b += ccpy_einsum("me,abem->ab", H.b.ov, R.abb)
    x2b += ccpy_einsum("nbfe,aefn->ab", H.ab.ovvv, R.aba)
    x2b += 0.5 * ccpy_einsum("anef,ebfn->ab", H.aa.vovv, R.aba)
    x2b += 0.5 * ccpy_einsum("bnef,aefn->ab", H.bb.vovv, R.abb)
    x2b += ccpy_einsum("anef,ebfn->ab", H.ab.vovv, R.abb)
    return x2b

def build_HR_3B(R, T, H):
    # (1)
    x3b = 0.5 * ccpy_einsum("cake,eb->abck", H.aa.vvov, R.ab)
    # (2)
    x3b += ccpy_einsum("cbke,ae->abck", H.ab.vvov, R.ab)
    # (4)
    x3b += ccpy_einsum("ae,ebck->abck", H.a.vv, R.aba)
    # (5)
    x3b += 0.5 * ccpy_einsum("be,aeck->abck", H.b.vv, R.aba)
    # (6)
    x3b += ccpy_einsum("abef,efck->abck", H.ab.vvvv, R.aba)
    # (7)
    x3b += 0.25 * ccpy_einsum("acef,ebfk->abck", H.aa.vvvv, R.aba)
    # (8)
    x3b += ccpy_einsum("cmke,abem->abck", H.aa.voov, R.aba)
    # (9)
    x3b += ccpy_einsum("cmke,abem->abck", H.ab.voov, R.abb)
    # (10)
    x3b -= 0.5 * ccpy_einsum("mbke,aecm->abck", H.ab.ovov, R.aba)
    # (3) + (11)
    x_ov = (
            ccpy_einsum("mbef,ef->mb", H.ab.ovvv, R.ab)
            + 0.5 * ccpy_einsum("mnef,ebfn->mb", H.aa.oovv, R.aba)
            + ccpy_einsum("mnef,ebfn->mb", H.ab.oovv, R.abb)
    )
    x3b -= 0.5 * ccpy_einsum("mb,acmk->abck", x_ov, T.aa)
    # (3) + (12)
    x_vo = (
            ccpy_einsum("amef,ef->am", H.ab.vovv, R.ab)
            + 0.5 * ccpy_einsum("nmfe,aefn->am", H.bb.oovv, R.abb)
            + ccpy_einsum("nmfe,aefn->am", H.ab.oovv, R.aba)
    )
    x3b -= ccpy_einsum("am,cbkm->abck", x_vo, T.ab)
    # (13)
    x3b -= 0.5 * ccpy_einsum("mk,abcm->abck", H.a.oo, R.aba)
    # antisymmetrize A(ac)
    x3b -= ccpy_einsum("abck->cbak", x3b)
    return x3b

def build_HR_3C(R, T, H):
    # (1)
    x3c = ccpy_einsum("acek,eb->abck", H.ab.vvvo, R.ab)
    # (2)
    x3c += 0.5 * ccpy_einsum("cbke,ae->abck", H.bb.vvov, R.ab)
    # (4)
    x3c += 0.5 * ccpy_einsum("ae,ebck->abck", H.a.vv, R.abb)
    # (5)
    x3c += ccpy_einsum("be,aeck->abck", H.b.vv, R.abb)
    # (6)
    x3c += ccpy_einsum("abef,efck->abck", H.ab.vvvv, R.abb)
    # (7)
    x3c += 0.25 * ccpy_einsum("bcef,aefk->abck", H.bb.vvvv, R.abb)
    # (8)
    x3c += ccpy_einsum("mcek,abem->abck", H.ab.ovvo, R.aba)
    # (9)
    x3c += ccpy_einsum("cmke,abem->abck", H.bb.voov, R.abb)
    # (10)
    x3c -= 0.5 * ccpy_einsum("amek,ebcm->abck", H.ab.vovo, R.abb)
    # (3) + (11)
    x_ov = (
            ccpy_einsum("mbef,ef->mb", H.ab.ovvv, R.ab)
            + 0.5 * ccpy_einsum("mnef,ebfn->mb", H.aa.oovv, R.aba)
            + ccpy_einsum("mnef,ebfn->mb", H.ab.oovv, R.abb)
    )
    x3c -= ccpy_einsum("mb,acmk->abck", x_ov, T.ab)
    # (3) + (12)
    x_vo = (
            ccpy_einsum("amef,ef->am", H.ab.vovv, R.ab)
            + 0.5 * ccpy_einsum("nmfe,aefn->am", H.bb.oovv, R.abb)
            + ccpy_einsum("nmfe,aefn->am", H.ab.oovv, R.aba)
    )
    x3c -= 0.5 * ccpy_einsum("am,bcmk->abck", x_vo, T.bb)
    # (13)
    x3c -= 0.5 * ccpy_einsum("mk,abcm->abck", H.b.oo, R.abb)
    # antisymmetrize A(b~c~)
    x3c -= ccpy_einsum("abck->acbk", x3c)
    return x3c

