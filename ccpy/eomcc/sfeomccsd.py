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
    pass

def build_HR_2B(R, T, H):
    pass

def build_HR_2C(R, T, H):
    pass
