"""Module with functions that perform the CC with singles, doubles,
and triples (CCSDT) calculation for a molecular system."""

import numpy as np
import time

from ccpy.hbar.hbar_ccs import get_ccs_intermediates
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.cc.ccsdt1_updates import *

def update(T, dT, H, shift, flag_RHF, system):

    # update T1 at CCSD level
    dT = update_t1a.build_ccsd(T, dT, H)
    # Add on T3 parts
    dT = update_t1a.build_11(T, dT, H, system)
    dT = update_t1a.build_10(T, dT, H, system)
    dT = update_t1a.build_01(T, dT, H, system)
    dT = update_t1a.build_00(T, dT, H, system)
    # update loop
    T, dT = update_t1a.update(T, dT, H, shift)
    if flag_RHF:
        # TODO: Maybe copy isn't needed. Reference should suffice
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        dT = update_t1b.build_ccsd(T, dT, H)
        # Add on T3 parts
        dT = update_t1b.build_11(T, dT, H, system)
        dT = update_t1b.build_10(T, dT, H, system)
        dT = update_t1b.build_01(T, dT, H, system)
        dT = update_t1b.build_00(T, dT, H, system)
        # update loop
        T, dT = update_t1b.update(T, dT, H, shift)

    # CCS intermediates
    hbar = get_ccs_intermediates(T, H)

    ####### T2 updates #######
    # t2a update
    x2 = update_t2a.build_ccsd(T, hbar)   # base CCSD part (separately antisymmetrized)
    # Add on T3 parts
    dT = update_t2a.build_1111(T, dT, hbar, system)
    dT = update_t2a.build_1101(T, dT, hbar, system)
    dT = update_t2a.build_1011(T, dT, hbar, system)
    dT = update_t2a.build_1100(T, dT, hbar, system)
    dT = update_t2a.build_0011(T, dT, hbar, system)
    dT = update_t2a.build_1001(T, dT, hbar, system)
    dT = update_t2a.build_1000(T, dT, hbar, system)
    dT = update_t2a.build_0001(T, dT, hbar, system)
    dT = update_t2a.build_0000(T, dT, hbar, system)
    dT.aa += x2
    # update loop
    T, dT = update_t2a.update(T, dT, H, shift)

    dT = update_t2b.build_ccsd(T, dT, hbar)   # base CCSD part
    # Add on T3 parts
    dT = update_t2b.build_1111(T, dT, hbar, system)
    dT = update_t2b.build_1101(T, dT, hbar, system)
    dT = update_t2b.build_1110(T, dT, hbar, system)
    dT = update_t2b.build_1011(T, dT, hbar, system)
    dT = update_t2b.build_0111(T, dT, hbar, system)
    dT = update_t2b.build_1100(T, dT, hbar, system)
    dT = update_t2b.build_0011(T, dT, hbar, system)
    dT = update_t2b.build_1001(T, dT, hbar, system)
    dT = update_t2b.build_0101(T, dT, hbar, system)
    dT = update_t2b.build_1010(T, dT, hbar, system)
    dT = update_t2b.build_0110(T, dT, hbar, system)
    dT = update_t2b.build_1000(T, dT, hbar, system)
    dT = update_t2b.build_0100(T, dT, hbar, system)
    dT = update_t2b.build_0001(T, dT, hbar, system)
    dT = update_t2b.build_0010(T, dT, hbar, system)
    dT = update_t2b.build_0000(T, dT, hbar, system)
    # update loop
    T, dT = update_t2b.update(T, dT, H, shift)

    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
        x2 = update_t2c.build_ccsd(T, hbar)   # base CCSD part (separately antisymmetrized)
        # Add on T3 parts
        dT = update_t2c.build_1111(T, dT, hbar, system)
        dT = update_t2c.build_1101(T, dT, hbar, system)
        dT = update_t2c.build_1011(T, dT, hbar, system)
        dT = update_t2c.build_1100(T, dT, hbar, system)
        dT = update_t2c.build_0011(T, dT, hbar, system)
        dT = update_t2c.build_1001(T, dT, hbar, system)
        dT = update_t2c.build_1000(T, dT, hbar, system)
        dT = update_t2c.build_0001(T, dT, hbar, system)
        dT = update_t2c.build_0000(T, dT, hbar, system)
        dT.bb += x2
        # update loop
        T, dT = update_t2c.update(T, dT, H, shift)

    # CCSD intermediates
    hbar = get_ccsd_intermediates(T, H)
    # add on (V * T3)_C intermediates
    hbar = intermediates.build_VT3_intermediates(T, hbar, system)

    ####### T3 updates #######
    # update t3a
    dT = t3a_111111.build(T, dT, hbar, H, shift, system)
    dT = t3a_110111.build(T, dT, hbar, H, shift, system)
    dT = t3a_111011.build(T, dT, hbar, H, shift, system)
    dT = t3a_110011.build(T, dT, hbar, H, shift, system)
    dT = t3a_100111.build(T, dT, hbar, H, shift, system)
    dT = t3a_111001.build(T, dT, hbar, H, shift, system)
    dT = t3a_100011.build(T, dT, hbar, H, shift, system)
    dT = t3a_110001.build(T, dT, hbar, H, shift, system)
    dT = t3a_100001.build(T, dT, hbar, H, shift, system)
    # update t3b
    dT = t3b_111111.build(T, dT, hbar, H, shift, system)
    dT = t3b_111110.build(T, dT, hbar, H, shift, system)
    dT = t3b_111011.build(T, dT, hbar, H, shift, system)
    dT = t3b_110111.build(T, dT, hbar, H, shift, system)
    dT = t3b_101111.build(T, dT, hbar, H, shift, system)
    dT = t3b_111001.build(T, dT, hbar, H, shift, system)
    dT = t3b_111010.build(T, dT, hbar, H, shift, system)
    dT = t3b_001111.build(T, dT, hbar, H, shift, system)
    dT = t3b_100111.build(T, dT, hbar, H, shift, system)
    dT = t3b_110011.build(T, dT, hbar, H, shift, system)
    dT = t3b_101110.build(T, dT, hbar, H, shift, system)
    dT = t3b_101011.build(T, dT, hbar, H, shift, system)
    dT = t3b_110110.build(T, dT, hbar, H, shift, system)
    dT = t3b_101001.build(T, dT, hbar, H, shift, system)
    dT = t3b_101010.build(T, dT, hbar, H, shift, system)
    dT = t3b_110001.build(T, dT, hbar, H, shift, system)
    dT = t3b_110010.build(T, dT, hbar, H, shift, system)
    dT = t3b_001011.build(T, dT, hbar, H, shift, system)
    dT = t3b_001110.build(T, dT, hbar, H, shift, system)
    dT = t3b_100011.build(T, dT, hbar, H, shift, system)
    dT = t3b_100110.build(T, dT, hbar, H, shift, system)
    dT = t3b_001001.build(T, dT, hbar, H, shift, system)
    dT = t3b_001010.build(T, dT, hbar, H, shift, system)
    dT = t3b_100001.build(T, dT, hbar, H, shift, system)
    dT = t3b_100010.build(T, dT, hbar, H, shift, system)
    # update t3c
    dT = t3c_111111.build(T, dT, hbar, H, shift, system)
    dT = t3c_111101.build(T, dT, hbar, H, shift, system)
    dT = t3c_111011.build(T, dT, hbar, H, shift, system)
    dT = t3c_110111.build(T, dT, hbar, H, shift, system)
    dT = t3c_011111.build(T, dT, hbar, H, shift, system)
    dT = t3c_111100.build(T, dT, hbar, H, shift, system)
    dT = t3c_111001.build(T, dT, hbar, H, shift, system)
    dT = t3c_010111.build(T, dT, hbar, H, shift, system)
    dT = t3c_100111.build(T, dT, hbar, H, shift, system)
    dT = t3c_110011.build(T, dT, hbar, H, shift, system)
    dT = t3c_011101.build(T, dT, hbar, H, shift, system)
    dT = t3c_011011.build(T, dT, hbar, H, shift, system)
    dT = t3c_110101.build(T, dT, hbar, H, shift, system)
    dT = t3c_110100.build(T, dT, hbar, H, shift, system)
    dT = t3c_110001.build(T, dT, hbar, H, shift, system)
    dT = t3c_011100.build(T, dT, hbar, H, shift, system)
    dT = t3c_011001.build(T, dT, hbar, H, shift, system)
    dT = t3c_100011.build(T, dT, hbar, H, shift, system)
    dT = t3c_100101.build(T, dT, hbar, H, shift, system)
    dT = t3c_010011.build(T, dT, hbar, H, shift, system)
    dT = t3c_010101.build(T, dT, hbar, H, shift, system)
    dT = t3c_010100.build(T, dT, hbar, H, shift, system)
    dT = t3c_010001.build(T, dT, hbar, H, shift, system)
    dT = t3c_100100.build(T, dT, hbar, H, shift, system)
    dT = t3c_100001.build(T, dT, hbar, H, shift, system)
    # update t3d
    dT = t3d_111111.build(T, dT, hbar, H, shift, system)
    dT = t3d_110111.build(T, dT, hbar, H, shift, system)
    dT = t3d_111011.build(T, dT, hbar, H, shift, system)
    dT = t3d_110011.build(T, dT, hbar, H, shift, system)
    dT = t3d_100111.build(T, dT, hbar, H, shift, system)
    dT = t3d_111001.build(T, dT, hbar, H, shift, system)
    dT = t3d_100011.build(T, dT, hbar, H, shift, system)
    dT = t3d_110001.build(T, dT, hbar, H, shift, system)
    dT = t3d_100001.build(T, dT, hbar, H, shift, system)

    # perform all T3 update loops

    return T, dT
