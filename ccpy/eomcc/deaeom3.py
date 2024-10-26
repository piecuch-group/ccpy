"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for doubly-attached states
using the DEA-EOMCC approach with up to 3p-1h excitations"""
import numpy as np
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
    x2b = np.einsum("ae,eb->ab", H.a.vv, R.ab, optimize=True)
    x2b += np.einsum("be,ae->ab", H.b.vv, R.ab, optimize=True)
    x2b += np.einsum("abef,ef->ab", H.ab.vvvv, R.ab, optimize=True)
    x2b += np.einsum("me,abem->ab", H.a.ov, R.aba, optimize=True)
    x2b += np.einsum("me,abem->ab", H.b.ov, R.abb, optimize=True)
    x2b += np.einsum("nbfe,aefn->ab", H.ab.ovvv, R.aba, optimize=True)
    x2b += 0.5 * np.einsum("anef,ebfn->ab", H.aa.vovv, R.aba, optimize=True)
    x2b += 0.5 * np.einsum("bnef,aefn->ab", H.bb.vovv, R.abb, optimize=True)
    x2b += np.einsum("anef,ebfn->ab", H.ab.vovv, R.abb, optimize=True)
    return x2b

def build_HR_3B(R, T, H):
    # (1)
    x3b = 0.5 * np.einsum("cake,eb->abck", H.aa.vvov, R.ab, optimize=True)
    # (2)
    x3b += np.einsum("cbke,ae->abck", H.ab.vvov, R.ab, optimize=True)
    # (4)
    x3b += np.einsum("ae,ebck->abck", H.a.vv, R.aba, optimize=True)
    # (5)
    x3b += 0.5 * np.einsum("be,aeck->abck", H.b.vv, R.aba, optimize=True)
    # (6)
    x3b += np.einsum("abef,efck->abck", H.ab.vvvv, R.aba, optimize=True)
    # (7)
    x3b += 0.25 * np.einsum("acef,ebfk->abck", H.aa.vvvv, R.aba, optimize=True)
    # (8)
    x3b += np.einsum("cmke,abem->abck", H.aa.voov, R.aba, optimize=True)
    # (9)
    x3b += np.einsum("cmke,abem->abck", H.ab.voov, R.abb, optimize=True)
    # (10)
    x3b -= 0.5 * np.einsum("mbke,aecm->abck", H.ab.ovov, R.aba, optimize=True)
    # (3) + (11)
    x_ov = (
            np.einsum("mbef,ef->mb", H.ab.ovvv, R.ab, optimize=True)
            + 0.5 * np.einsum("mnef,ebfn->mb", H.aa.oovv, R.aba, optimize=True)
            + np.einsum("mnef,ebfn->mb", H.ab.oovv, R.abb, optimize=True)
    )
    x3b -= 0.5 * np.einsum("mb,acmk->abck", x_ov, T.aa, optimize=True)
    # (3) + (12)
    x_vo = (
            np.einsum("amef,ef->am", H.ab.vovv, R.ab, optimize=True)
            + 0.5 * np.einsum("nmfe,aefn->am", H.bb.oovv, R.abb, optimize=True)
            + np.einsum("nmfe,aefn->am", H.ab.oovv, R.aba, optimize=True)
    )
    x3b -= np.einsum("am,cbkm->abck", x_vo, T.ab, optimize=True)
    # (13)
    x3b -= 0.5 * np.einsum("mk,abcm->abck", H.a.oo, R.aba, optimize=True)
    # antisymmetrize A(ac)
    x3b -= np.einsum("abck->cbak", x3b, optimize=True)
    return x3b

def build_HR_3C(R, T, H):
    # (1)
    x3c = np.einsum("acek,eb->abck", H.ab.vvvo, R.ab, optimize=True)
    # (2)
    x3c += 0.5 * np.einsum("cbke,ae->abck", H.bb.vvov, R.ab, optimize=True)
    # (4)
    x3c += 0.5 * np.einsum("ae,ebck->abck", H.a.vv, R.abb, optimize=True)
    # (5)
    x3c += np.einsum("be,aeck->abck", H.b.vv, R.abb, optimize=True)
    # (6)
    x3c += np.einsum("abef,efck->abck", H.ab.vvvv, R.abb, optimize=True)
    # (7)
    x3c += 0.25 * np.einsum("bcef,aefk->abck", H.bb.vvvv, R.abb, optimize=True)
    # (8)
    x3c += np.einsum("mcek,abem->abck", H.ab.ovvo, R.aba, optimize=True)
    # (9)
    x3c += np.einsum("cmke,abem->abck", H.bb.voov, R.abb, optimize=True)
    # (10)
    x3c -= 0.5 * np.einsum("amek,ebcm->abck", H.ab.vovo, R.abb, optimize=True)
    # (3) + (11)
    x_ov = (
            np.einsum("mbef,ef->mb", H.ab.ovvv, R.ab, optimize=True)
            + 0.5 * np.einsum("mnef,ebfn->mb", H.aa.oovv, R.aba, optimize=True)
            + np.einsum("mnef,ebfn->mb", H.ab.oovv, R.abb, optimize=True)
    )
    x3c -= np.einsum("mb,acmk->abck", x_ov, T.ab, optimize=True)
    # (3) + (12)
    x_vo = (
            np.einsum("amef,ef->am", H.ab.vovv, R.ab, optimize=True)
            + 0.5 * np.einsum("nmfe,aefn->am", H.bb.oovv, R.abb, optimize=True)
            + np.einsum("nmfe,aefn->am", H.ab.oovv, R.aba, optimize=True)
    )
    x3c -= 0.5 * np.einsum("am,bcmk->abck", x_vo, T.bb, optimize=True)
    # (13)
    x3c -= 0.5 * np.einsum("mk,abcm->abck", H.b.oo, R.abb, optimize=True)
    # antisymmetrize A(b~c~)
    x3c -= np.einsum("abck->acbk", x3c, optimize=True)
    return x3c

