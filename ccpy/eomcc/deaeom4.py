"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for doubly-attached states
using the DEA-EOMCC approach with up to 4p-2h excitations"""
import numpy as np
from ccpy.eomcc.deaeom4_intermediates import get_deaeom4_intermediates
from ccpy.utilities.updates import cc_loops2

def update(R, omega, H, system):
    R.ab, R.aba, R.abb, R.abaa, R.abab, R.abbb = cc_loops2.cc_loops2.update_r_4p2h(
        R.ab,
        R.aba,
        R.abb,
        R.abaa,
        R.abab,
        R.abbb,
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
        0.0,
    )
    return R

def HR(dR, R, T, H, flag_RHF, system):

    X = get_deaeom4_intermediates(H, R, T, system)
    # update R2
    dR.ab = build_HR_2B(R, T, H)
    # update R3
    dR.aba = build_HR_3B(R, T, H)
    dR.abb = build_HR_3C(R, T, H)
    # update R4
    dR.abaa = build_HR_4B(R, T, H, X)
    dR.abab = build_HR_4C(R, T, H, X)
    dR.abbb = build_HR_4D(R, T, H, X)

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
    # additional R(4p-2h) terms
    x2b += 0.25 * np.einsum("mnef,abefmn->ab", H.aa.oovv, R.abaa, optimize=True)
    x2b += np.einsum("mnef,abefmn->ab", H.ab.oovv, R.abab, optimize=True)
    x2b += 0.25 * np.einsum("mnef,abefmn->ab", H.bb.oovv, R.abbb, optimize=True)

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
    # additional R(4p-2h) terms
    x3b += 0.5 * np.einsum("me,abcekm->abck", H.a.ov, R.abaa, optimize=True)
    x3b += 0.5 * np.einsum("me,abcekm->abck", H.b.ov, R.abab, optimize=True)
    x3b -= 0.25 * np.einsum("mnkf,abcfmn->abck", H.aa.ooov, R.abaa, optimize=True)
    x3b -= 0.5 * np.einsum("mnkf,abcfmn->abck", H.ab.ooov, R.abab, optimize=True)
    x3b += 0.5 * np.einsum("cnef,abefkn->abck", H.aa.vovv, R.abaa, optimize=True)
    x3b += np.einsum("cnef,abefkn->abck", H.ab.vovv, R.abab, optimize=True)
    x3b += 0.5 * np.einsum("nbfe,aecfkn->abck", H.ab.ovvv, R.abaa, optimize=True)
    x3b += 0.25 * np.einsum("bnef,aecfkn->abck", H.bb.vovv, R.abab, optimize=True)
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
    # additional R(4p-2h) terms
    x3c += 0.5 * np.einsum("me,abeckm->abck", H.a.ov, R.abab, optimize=True)
    x3c += 0.5 * np.einsum("me,abeckm->abck", H.b.ov, R.abbb, optimize=True)
    x3c -= 0.5 * np.einsum("nmfk,abfcnm->abck", H.ab.oovo, R.abab, optimize=True)
    x3c -= 0.25 * np.einsum("mnkf,abcfmn->abck", H.bb.ooov, R.abbb, optimize=True)
    x3c += np.einsum("ncfe,abfenk->abck", H.ab.ovvv, R.abab, optimize=True)
    x3c += 0.5 * np.einsum("cnef,abefkn->abck", H.bb.vovv, R.abbb, optimize=True)
    x3c += 0.25 * np.einsum("anef,ebfcnk->abck", H.aa.vovv, R.abab, optimize=True)
    x3c += 0.5 * np.einsum("anef,ebfcnk->abck", H.ab.vovv, R.abbb, optimize=True)
    # antisymmetrize A(b~c~)
    x3c -= np.einsum("abck->acbk", x3c, optimize=True)
    return x3c

def build_HR_4B(R, T, H, X):
    pass

def build_HR_4C(R, T, H, X):
    pass

def build_HR_4D(R, T, H, X):
    pass