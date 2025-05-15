'''
Triple Electron Attachment Equation-of-Motion Coupled-Cluster
Method with 3p Excitations on top of CCSD [TEA-EOMCCSD(3p)]
'''

import numpy as np
from ccpy.lib.core import cc_loops2

def update(R, omega, H, RHF_symmetry, system):
    R.aab = cc_loops2.update_r_3p(
        R.aab,
        omega,
        H.a.vv,
        H.b.vv,
        0.0,
    )
    return R

def HR(dR, R, T, H, flag_RHF, system):

    # update R2
    dR.aab = build_HR_3B(R, T, H)
    return dR.flatten()

def build_HR_3B(R, T, H):
    x3b = np.einsum("ae,ebc->abc", H.a.vv, R.aab, optimize=True)
    x3b += 0.5 * np.einsum("ce,abe->abc", H.b.vv, R.aab, optimize=True)
    x3b += 0.25 * np.einsum("abef,efc->abc", H.aa.vvvv, R.aab, optimize=True)
    x3b += np.einsum("bcef,aef->abc", H.ab.vvvv, R.aab, optimize=True)
    # antisymmetrize A(ab)
    x3b -= np.transpose(x3b, (1, 0, 2))
    return x3b

