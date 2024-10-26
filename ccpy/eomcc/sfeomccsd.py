"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for excited states using
the spin-flip equation-of-motion (EOM) CC with singles and doubles (SF-EOMCCSD)."""
import numpy as np
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
            np.einsum("nmfe,fenj->mj", H.ab.oovv, R.ab, optimize=True)
            + 0.5 * np.einsum("mnef,fenj->mj", H.bb.oovv, R.bb, optimize=True)
            - np.einsum("mnje,em->nj", H.ab.ooov, R.b, optimize=True)
    )
    x_vv = (
            -0.5 * np.einsum("mnef,fbnm->be", H.aa.oovv, R.ab, optimize=True)
            - np.einsum("mnef,fbnm->be", H.ab.oovv, R.bb, optimize=True)
            - np.einsum("mbfe,em->bf", H.ab.ovvv, R.b, optimize=True)
    )

    # update R2
    dR.ab = build_HR_2B(R, T, H, x_oo, x_vv)
    dR.bb = build_HR_2C(R, T, H, x_oo, x_vv)

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
    # antisymmetrize (ab)
    x2c -= np.transpose(x2c, (1, 0, 2, 3))
    return x2c