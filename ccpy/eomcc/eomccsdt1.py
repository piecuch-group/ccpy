"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for excited states using
the equation-of-motion (EOM) CC with singles, doubles, and triples (EOMCCSDT)."""
import numpy as np
from ccpy.utilities.updates import cc_active_loops

from ccpy.hbar.eomccsdt_intermediates import get_eomccsd_intermediates
from ccpy.eomcc.eomccsdt1_updates.intermediates import add_HR3_intermediates

from ccpy.eomcc.eomccsdt1_updates import *

# def update(R, omega, H):
#     R.a, R.b, R.aa, R.ab, R.bb, R.aaa, R.aab, R.abb, R.bbb = cc_loops.cc_loops.update_r_ccsdt(
#         R.a,
#         R.b,
#         R.aa,
#         R.ab,
#         R.bb,
#         R.aaa,
#         R.aab,
#         R.abb,
#         R.bbb,
#         omega,
#         H.a.oo,
#         H.a.vv,
#         H.b.oo,
#         H.b.vv,
#         0.0,
#     )
#     return R


def HR(dR, R, T, H, flag_RHF, system):

    HR = get_eomccsd_intermediates(H, R, T, system)
    HR = add_HR3_intermediates(HR, H, R, system)

    dR = r1a_update.build(dR, R, H, system)
    if flag_RHF:
         dR.b = dR.a.copy()
    else:
         dR = r1b_update.build(dR, R, H, system)

    dR = r2a_update.build(dR, R, T, H, HR, system)
    dR = r2b_update.build(dR, R, T, H, HR, system)
    if flag_RHF:
         dR.bb = dR.aa.copy()
    else:
         dR = r2c_update.build(dR, R, T, H, HR, system)


    # aaa updates
    dR = r3a_111111.build(dR, R, T, H, HR, system)
    dR = r3a_111011.build(dR, R, T, H, HR, system)
    dR = r3a_110111.build(dR, R, T, H, HR, system)
    dR = r3a_110011.build(dR, R, T, H, HR, system)
    dR = r3a_100111.build(dR, R, T, H, HR, system)
    dR = r3a_111001.build(dR, R, T, H, HR, system)
    dR = r3a_100011.build(dR, R, T, H, HR, system)
    dR = r3a_110001.build(dR, R, T, H, HR, system)
    dR = r3a_100001.build(dR, R, T, H, HR, system)
    # aab updates
    dR = r3b_111111.build(dR, R, T, H, HR, system)
    dR = r3b_111110.build(dR, R, T, H, HR, system)
    dR = r3b_110111.build(dR, R, T, H, HR, system)
    dR = r3b_101111.build(dR, R, T, H, HR, system)
    dR = r3b_111011.build(dR, R, T, H, HR, system)
    dR = r3b_111001.build(dR, R, T, H, HR, system)
    dR = r3b_111010.build(dR, R, T, H, HR, system)
    dR = r3b_001111.build(dR, R, T, H, HR, system)
    dR = r3b_100111.build(dR, R, T, H, HR, system)
    dR = r3b_110011.build(dR, R, T, H, HR, system)
    dR = r3b_101110.build(dR, R, T, H, HR, system)
    dR = r3b_101011.build(dR, R, T, H, HR, system)
    dR = r3b_110110.build(dR, R, T, H, HR, system)
    dR = r3b_101001.build(dR, R, T, H, HR, system)
    dR = r3b_101010.build(dR, R, T, H, HR, system)
    dR = r3b_110001.build(dR, R, T, H, HR, system)
    dR = r3b_110010.build(dR, R, T, H, HR, system)
    dR = r3b_001011.build(dR, R, T, H, HR, system)
    dR = r3b_001110.build(dR, R, T, H, HR, system)
    dR = r3b_100011.build(dR, R, T, H, HR, system)
    dR = r3b_100110.build(dR, R, T, H, HR, system)
    dR = r3b_001001.build(dR, R, T, H, HR, system)
    dR = r3b_001010.build(dR, R, T, H, HR, system)
    dR = r3b_100001.build(dR, R, T, H, HR, system)
    dR = r3b_100010.build(dR, R, T, H, HR, system)
    # abb updates
    dR = r3c_111111.build(dR, R, T, H, HR, system)
    dR = r3c_111101.build(dR, R, T, H, HR, system)
    dR = r3c_111011.build(dR, R, T, H, HR, system)
    dR = r3c_110111.build(dR, R, T, H, HR, system)
    dR = r3c_011111.build(dR, R, T, H, HR, system)
    dR = r3c_111100.build(dR, R, T, H, HR, system)
    dR = r3c_111001.build(dR, R, T, H, HR, system)
    dR = r3c_010111.build(dR, R, T, H, HR, system)
    dR = r3c_100111.build(dR, R, T, H, HR, system)
    dR = r3c_110011.build(dR, R, T, H, HR, system)
    dR = r3c_011101.build(dR, R, T, H, HR, system)
    dR = r3c_011011.build(dR, R, T, H, HR, system)
    dR = r3c_110101.build(dR, R, T, H, HR, system)
    dR = r3c_110100.build(dR, R, T, H, HR, system)
    dR = r3c_110001.build(dR, R, T, H, HR, system)
    dR = r3c_011100.build(dR, R, T, H, HR, system)
    dR = r3c_011001.build(dR, R, T, H, HR, system)
    dR = r3c_100011.build(dR, R, T, H, HR, system)
    dR = r3c_100101.build(dR, R, T, H, HR, system)
    dR = r3c_010011.build(dR, R, T, H, HR, system)
    dR = r3c_010101.build(dR, R, T, H, HR, system)
    dR = r3c_010100.build(dR, R, T, H, HR, system)
    dR = r3c_010001.build(dR, R, T, H, HR, system)
    dR = r3c_100100.build(dR, R, T, H, HR, system)
    dR = r3c_100001.build(dR, R, T, H, HR, system)
    # bbb updates
    dR = r3d_111111.build(dR, R, T, H, HR, system)
    dR = r3d_111011.build(dR, R, T, H, HR, system)
    dR = r3d_110111.build(dR, R, T, H, HR, system)
    dR = r3d_110011.build(dR, R, T, H, HR, system)
    dR = r3d_100111.build(dR, R, T, H, HR, system)
    dR = r3d_111001.build(dR, R, T, H, HR, system)
    dR = r3d_100011.build(dR, R, T, H, HR, system)
    dR = r3d_110001.build(dR, R, T, H, HR, system)
    dR = r3d_100001.build(dR, R, T, H, HR, system)

    return dR