"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for doubly-attached states
using the DIP-EOMCC approach with up to 3h-1p excitations"""
import numpy as np
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
    x2b = -np.einsum("mi,mj->ij", H.a.oo, R.ab, optimize=True)
    x2b -= np.einsum("mj,im->ij", H.b.oo, R.ab, optimize=True)
    x2b += np.einsum("mnij,mn->ij", H.ab.oooo, R.ab, optimize=True)
    x2b += np.einsum("me,ijem->ij", H.a.ov, R.aba, optimize=True)
    x2b += np.einsum("me,ijem->ij", H.b.ov, R.abb, optimize=True)
    x2b -= np.einsum("nmfj,imfn->ij", H.ab.oovo, R.aba, optimize=True)
    x2b -= 0.5 * np.einsum("mnjf,imfn->ij", H.bb.ooov, R.abb, optimize=True)
    x2b -= 0.5 * np.einsum("mnif,mjfn->ij", H.aa.ooov, R.aba, optimize=True)
    x2b -= np.einsum("mnif,mjfn->ij", H.ab.ooov, R.abb, optimize=True)
    return x2b

def build_HR_3B(R, T, H):
    x3b = -0.5 * np.einsum("cmki,mj->ijck", H.aa.vooo, R.ab, optimize=True)
    x3b -= np.einsum("cmkj,im->ijck", H.ab.vooo, R.ab, optimize=True)
    x3b -= np.einsum("mk,ijcm->ijck", H.a.oo, R.aba, optimize=True)
    x3b -= 0.5 * np.einsum("mj,imck->ijck", H.b.oo, R.aba, optimize=True)
    x3b += 0.5 * np.einsum("ce,ijek->ijck", H.a.vv, R.aba, optimize=True)
    x3b += np.einsum("cmke,ijem->ijck", H.aa.voov, R.aba, optimize=True)
    x3b += np.einsum("cmke,ijem->ijck", H.ab.voov, R.abb, optimize=True)
    x3b += np.einsum("mnij,mnck->ijck", H.ab.oooo, R.aba, optimize=True)
    x3b += 0.25 * np.einsum("mnik,mjcn->ijck", H.aa.oooo, R.aba, optimize=True)
    x3b -= 0.5 * np.einsum("cmej,imek->ijck", H.ab.vovo, R.aba, optimize=True)

    x_ov = (
            np.einsum("mnie,mn->ie", H.ab.ooov, R.ab, optimize=True)
            - np.einsum("nmfe,imfn->ie", H.ab.oovv, R.aba, optimize=True)
            - 0.5 * np.einsum("nmfe,imfn->ie", H.bb.oovv, R.abb, optimize=True)
    )
    x_vo = (
            np.einsum("mnej,mn->ej", H.ab.oovo, R.ab, optimize=True)
            - 0.5 * np.einsum("mnef,mjfn->ej", H.aa.oovv, R.aba, optimize=True)
            - np.einsum("mnef,mjfn->ej", H.ab.oovv, R.abb, optimize=True)
    )

    x3b += np.einsum("ie,cekj->ijck", x_ov, T.ab, optimize=True)
    x3b += 0.5 * np.einsum("ej,ecik->ijck", x_vo, T.aa, optimize=True)

    # antisymmetrize A(ik)
    x3b -= np.transpose(x3b, (3, 1, 2, 0))
    return x3b

def build_HR_3C(R, T, H):
    x3c = -np.einsum("mcik,mj->ijck", H.ab.ovoo, R.ab, optimize=True)
    x3c -= 0.5 * np.einsum("cmkj,im->ijck", H.bb.vooo, R.ab, optimize=True)
    x3c -= 0.5 * np.einsum("mi,mjck->ijck", H.a.oo, R.abb, optimize=True)
    x3c -= np.einsum("mj,imck->ijck", H.b.oo, R.abb, optimize=True)
    x3c += 0.5 * np.einsum("ce,ijek->ijck", H.b.vv, R.abb, optimize=True)
    x3c += np.einsum("mnij,mnck->ijck", H.ab.oooo, R.abb, optimize=True)
    x3c += 0.25 * np.einsum("mnjk,imcn->ijck", H.bb.oooo, R.abb, optimize=True)
    x3c += np.einsum("mcek,ijem->ijck", H.ab.ovvo, R.aba, optimize=True)
    x3c += np.einsum("cmke,ijem->ijck", H.bb.voov, R.abb, optimize=True)
    x3c -= 0.5 * np.einsum("mcie,mjek->ijck", H.ab.ovov, R.abb, optimize=True)

    x_ov = (
            np.einsum("mnie,mn->ie", H.ab.ooov, R.ab, optimize=True)
            - np.einsum("nmfe,imfn->ie", H.ab.oovv, R.aba, optimize=True)
            - 0.5 * np.einsum("nmfe,imfn->ie", H.bb.oovv, R.abb, optimize=True)
    )
    x_vo = (
            np.einsum("mnej,mn->ej", H.ab.oovo, R.ab, optimize=True)
            - 0.5 * np.einsum("mnef,mjfn->ej", H.aa.oovv, R.aba, optimize=True)
            - np.einsum("mnef,mjfn->ej", H.ab.oovv, R.abb, optimize=True)
    )

    x3c += np.einsum("ej,ecik->ijck", x_vo, T.ab, optimize=True)
    x3c += 0.5 * np.einsum("ie,ecjk->ijck", x_ov, T.bb, optimize=True)

    # antisymmetrize A(j~k~)
    x3c -= np.einsum("ijck->ikcj", x3c, optimize=True)
    return x3c
