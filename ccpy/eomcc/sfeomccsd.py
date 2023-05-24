"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for excited states using
the spin-flip equation-of-motion (EOM) CC with singles and doubles (SF-EOMCCSD)."""
import numpy as np
from ccpy.utilities.updates import cc_loops2

def update(R, omega, H, system):

    R.b, R.ab, R.bb = cc_loops2.cc_loops2.update_r_sfccsd(
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
    dR.a = build_HR_1B(R, T, H)

    # update R2
    dR.ab = build_HR_2B(R, T, H)
    dR.bb = build_HR_2C(R, T, H)

    return dR.flatten()

def build_HR_1B(R, T, H):

    # < a~i | (H(2) * R1)_C | 0 >
    x1a = np.einsum("ae,ei->ai", H.b.vv, R.b, optimize=True)
    x1a -= np.einsum("mi,am->ai", H.a.oo, R.b, optimize=True)
    x1a -= np.einsum("maie,em->ai", H.ab.ovov, R.b, optimize=True)
    # <a~i | (H(2) * R2)_C | 0 >
    x1a += np.einsum("me,eami->ai", H.a.ov, R.ab, optimize=True)
    x1a += np.einsum("me,eami->ai", H.b.ov, R.bb, optimize=True)
    x1a += np.einsum("mnif,fanm->ai", H.ab.ooov, R.bb, optimize=True)
    x1a -= np.einsum("anfe,feni->ai", H.bb.vovv, R.bb, optimize=True)
    return x1a

def build_HR_2B(R, T, H):
    pass

def build_HR_2C(R, T, H):
    pass
