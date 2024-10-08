"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for doubly-attached states
using the DEA-EOMCC approach with up to 2p excitations"""
import numpy as np
from ccpy.lib.core import cc_loops2

def update(R, omega, H, RHF_symmetry, system):
    R.ab = cc_loops2.update_r_2p(
        R.ab,
        omega,
        H.a.vv,
        H.b.vv,
        0.0,
    )
    return R

def HR(dR, R, T, H, flag_RHF, system):

    # update R2
    dR.ab = build_HR_2B(R, T, H)
    return dR.flatten()

def build_HR_2B(R, T, H):
    x2b = np.einsum("ae,eb->ab", H.a.vv, R.ab, optimize=True)
    x2b += np.einsum("be,ae->ab", H.b.vv, R.ab, optimize=True)
    x2b += np.einsum("abef,ef->ab", H.ab.vvvv, R.ab, optimize=True)
    return x2b

