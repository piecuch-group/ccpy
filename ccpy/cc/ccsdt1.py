"""Module with functions that perform the CC with singles, doubles,
and active-space triples (CCSDt) calculation for a molecular system."""

from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.cc.ccsdt1_updates import *

def update(T, dT, H, hbar, shift, flag_RHF, system):

    hbar = get_pre_ccs_intermediates(hbar, T, H, system, flag_RHF)

    ####### T1 updates #######
    # t1a update
    dT = update_t1a.build_ccsd(T, dT, H, hbar)   # base CCSD part
    # Add on T3 parts
    dT = update_t1a.build_11(T, dT, H, system)
    dT = update_t1a.build_10(T, dT, H, system)
    dT = update_t1a.build_01(T, dT, H, system)
    dT = update_t1a.build_00(T, dT, H, system)
    # update loop
    T, dT = update_t1a.update(T, dT, H, shift)
    if flag_RHF:
        T.b = T.a.copy()
        dT.b = dT.a.copy()
    else:
        dT = update_t1b.build_ccsd(T, dT, H, hbar)
        # Add on T3 parts
        dT = update_t1b.build_11(T, dT, H, system)
        dT = update_t1b.build_10(T, dT, H, system)
        dT = update_t1b.build_01(T, dT, H, system)
        dT = update_t1b.build_00(T, dT, H, system)
        # update loop
        T, dT = update_t1b.update(T, dT, H, shift)

    # CCS intermediates
    hbar = get_ccs_intermediates_opt(hbar, T, H, system, flag_RHF)

    # CCSD base t2 updates
    x2a = update_t2a.build_ccsd(T, hbar, H)   # base CCSD part (separately antisymmetrized)
    x2b = update_t2b.build_ccsd(T, hbar, H)   # base CCSD part
    if flag_RHF:
        x2c = x2a.copy()
    else:
        x2c = update_t2c.build_ccsd(T, hbar, H)  # base CCSD part (separately antisymmetrized)

    # adjust intermediates to CCSDT case
    hbar.aa.ooov += H.aa.ooov
    hbar.aa.vovv += H.aa.vovv
    hbar.ab.ooov += H.ab.ooov
    hbar.ab.oovo += H.ab.oovo
    hbar.ab.vovv += H.ab.vovv
    hbar.ab.ovvv += H.ab.ovvv
    if flag_RHF:
        hbar.bb.ooov = hbar.aa.ooov
        hbar.bb.vovv = hbar.aa.vovv
    else:
        hbar.bb.ooov += H.bb.ooov
        hbar.bb.vovv += H.bb.vovv

    ####### T2 updates #######
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
    dT.aa += x2a
    # update loop
    T, dT = update_t2a.update(T, dT, H, shift)

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
    dT.ab += x2b
    # update loop
    T, dT = update_t2b.update(T, dT, H, shift)

    if flag_RHF:
        T.bb = T.aa.copy()
        dT.bb = dT.aa.copy()
    else:
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
        dT.bb += x2c
        # update loop
        T, dT = update_t2c.update(T, dT, H, shift)

    # remove these parts intermediates before entering T3 updates
    hbar.aa.ooov -= H.aa.ooov
    hbar.aa.vovv -= H.aa.vovv
    hbar.ab.ooov -= H.ab.ooov
    hbar.ab.oovo -= H.ab.oovo
    hbar.ab.vovv -= H.ab.vovv
    hbar.ab.ovvv -= H.ab.ovvv
    if flag_RHF:
        hbar.bb.ooov = hbar.aa.ooov
        hbar.bb.vovv = hbar.aa.vovv
    else:
        hbar.bb.ooov -= H.bb.ooov
        hbar.bb.vovv -= H.bb.vovv

    # CCSD intermediates
    #[TODO]: Should accept CCS HBar as input and build only terms with T2 in it
    hbar = get_ccsd_intermediates(T, hbar, H, flag_RHF)
    # add on (V * T3)_C intermediates
    hbar.aa.oovv = H.aa.oovv.copy()
    hbar.ab.oovv = H.ab.oovv.copy()
    hbar.bb.oovv = H.bb.oovv.copy()
    hbar = intermediates.build_VT3_intermediates(T, hbar, system)

    # ####### T3 updates #######
    # update t3a
    dT = t3a_111111.build(T, dT, hbar, system)
    dT = t3a_110111.build(T, dT, hbar, system)
    dT = t3a_111011.build(T, dT, hbar, system)
    dT = t3a_110011.build(T, dT, hbar, system)
    dT = t3a_100111.build(T, dT, hbar, system)
    dT = t3a_111001.build(T, dT, hbar, system)
    dT = t3a_100011.build(T, dT, hbar, system)
    dT = t3a_110001.build(T, dT, hbar, system)
    dT = t3a_100001.build(T, dT, hbar, system)
    # update t3b
    dT = t3b_111111.build(T, dT, hbar, system)
    dT = t3b_111110.build(T, dT, hbar, system)
    dT = t3b_111011.build(T, dT, hbar, system)
    dT = t3b_110111.build(T, dT, hbar, system)
    dT = t3b_101111.build(T, dT, hbar, system)
    dT = t3b_111001.build(T, dT, hbar, system)
    dT = t3b_111010.build(T, dT, hbar, system)
    dT = t3b_001111.build(T, dT, hbar, system)
    dT = t3b_100111.build(T, dT, hbar, system)
    dT = t3b_110011.build(T, dT, hbar, system)
    dT = t3b_101110.build(T, dT, hbar, system)
    dT = t3b_101011.build(T, dT, hbar, system)
    dT = t3b_110110.build(T, dT, hbar, system)
    dT = t3b_101001.build(T, dT, hbar, system)
    dT = t3b_101010.build(T, dT, hbar, system)
    dT = t3b_110001.build(T, dT, hbar, system)
    dT = t3b_110010.build(T, dT, hbar, system)
    dT = t3b_001011.build(T, dT, hbar, system)
    dT = t3b_001110.build(T, dT, hbar, system)
    dT = t3b_100011.build(T, dT, hbar, system)
    dT = t3b_100110.build(T, dT, hbar, system)
    dT = t3b_001001.build(T, dT, hbar, system)
    dT = t3b_001010.build(T, dT, hbar, system)
    dT = t3b_100001.build(T, dT, hbar, system)
    dT = t3b_100010.build(T, dT, hbar, system)
    # update t3c
    dT = t3c_111111.build(T, dT, hbar, system)
    dT = t3c_111101.build(T, dT, hbar, system)
    dT = t3c_111011.build(T, dT, hbar, system)
    dT = t3c_110111.build(T, dT, hbar, system)
    dT = t3c_011111.build(T, dT, hbar, system)
    dT = t3c_111100.build(T, dT, hbar, system)
    dT = t3c_111001.build(T, dT, hbar, system)
    dT = t3c_010111.build(T, dT, hbar, system)
    dT = t3c_100111.build(T, dT, hbar, system)
    dT = t3c_110011.build(T, dT, hbar, system)
    dT = t3c_011101.build(T, dT, hbar, system)
    dT = t3c_011011.build(T, dT, hbar, system)
    dT = t3c_110101.build(T, dT, hbar, system)
    dT = t3c_110100.build(T, dT, hbar, system)
    dT = t3c_110001.build(T, dT, hbar, system)
    dT = t3c_011100.build(T, dT, hbar, system)
    dT = t3c_011001.build(T, dT, hbar, system)
    dT = t3c_100011.build(T, dT, hbar, system)
    dT = t3c_100101.build(T, dT, hbar, system)
    dT = t3c_010011.build(T, dT, hbar, system)
    dT = t3c_010101.build(T, dT, hbar, system)
    dT = t3c_010100.build(T, dT, hbar, system)
    dT = t3c_010001.build(T, dT, hbar, system)
    dT = t3c_100100.build(T, dT, hbar, system)
    dT = t3c_100001.build(T, dT, hbar, system)
    # update t3d
    dT = t3d_111111.build(T, dT, hbar, system)
    dT = t3d_110111.build(T, dT, hbar, system)
    dT = t3d_111011.build(T, dT, hbar, system)
    dT = t3d_110011.build(T, dT, hbar, system)
    dT = t3d_100111.build(T, dT, hbar, system)
    dT = t3d_111001.build(T, dT, hbar, system)
    dT = t3d_100011.build(T, dT, hbar, system)
    dT = t3d_110001.build(T, dT, hbar, system)
    dT = t3d_100001.build(T, dT, hbar, system)

    # [TODO]: Separate the update loops by presence of inactive particle/hole alpha/beta indices to allow CCSDt
    # to work with full active spaces, reproducing CCSDT.

    # # perform all T3 update loops
    #
    # # no inactive
    # T, dT = t3a_111111.update(T, dT, H, shift, system)
    # T, dT = t3b_111111.update(T, dT, H, shift, system)
    # T, dT = t3c_111111.update(T, dT, H, shift, system)
    # T, dT = t3d_111111.update(T, dT, H, shift, system)
    #
    # # only alpha hole inactive
    # T, dT = t3a_111011.update(T, dT, H, shift, system)
    # T, dT = t3a_111001.update(T, dT, H, shift, system)
    # T, dT = t3b_111011.update(T, dT, H, shift, system)
    # T, dT = t3b_111001.update(T, dT, H, shift, system)
    # T, dT = t3c_111011.update(T, dT, H, shift, system)
    #
    # # only beta hole inactive
    # T, dT = t3b_111110.update(T, dT, H, shift, system)
    # T, dT = t3c_111101.update(T, dT, H, shift, system)
    # T, dT = t3c_111100.update(T, dT, H, shift, system)
    # T, dT = t3d_111011.update(T, dT, H, shift, system)
    # T, dT = t3d_111001.update(T, dT, H, shift, system)
    #
    # # only beta particle inactive
    # T, dT = t3b_110111.update(T, dT, H, shift, system)
    # T, dT = t3c_110111.update(T, dT, H, shift, system)
    # T, dT = t3c_100111.update(T, dT, H, shift, system)
    # T, dT = t3d_110111.update(T, dT, H, shift, system)
    # T, dT = t3d_100111.update(T, dT, H, shift, system)
    #
    # # only alpha particle inactive
    # T, dT = t3a_110111.update(T, dT, H, shift, system)
    # T, dT = t3a_100111.update(T, dT, H, shift, system)
    # T, dT = t3b_101111.update(T, dT, H, shift, system)
    # T, dT = t3b_001111.update(T, dT, H, shift, system)
    # T, dT = t3c_011111.update(T, dT, H, shift, system)
    #
    # # only alpha particle and hole inactive
    # T, dT = t3a_110011.update(T, dT, H, shift, system)
    # T, dT = t3a_100011.update(T, dT, H, shift, system)
    # T, dT = t3a_110001.update(T, dT, H, shift, system)
    # T, dT = t3a_100001.update(T, dT, H, shift, system)
    # T, dT = t3b_101011.update(T, dT, H, shift, system)
    # T, dT = t3b_001001.update(T, dT, H, shift, system)
    # T, dT = t3b_001011.update(T, dT, H, shift, system)
    # T, dT = t3b_101001.update(T, dT, H, shift, system)
    #
    # # only beta particle and hole inactive
    # T, dT = t3b_110110.update(T, dT, H, shift, system)
    # T, dT = t3d_110011.update(T, dT, H, shift, system)
    # T, dT = t3d_100011.update(T, dT, H, shift, system)
    # T, dT = t3d_110001.update(T, dT, H, shift, system)
    # T, dT = t3d_100001.update(T, dT, H, shift, system)
    # T, dT = t3c_110101.update(T, dT, H, shift, system)
    # T, dT = t3c_110100.update(T, dT, H, shift, system)
    # T, dT = t3c_100101.update(T, dT, H, shift, system)
    # T, dT = t3c_100100.update(T, dT, H, shift, system)
    #
    # # only alpha hole and beta hole
    # T, dT = t3b_111010.update(T, dT, H, shift, system)
    # T, dT = t3c_111001.update(T, dT, H, shift, system)
    #
    # # only alpha particle and beta particle
    # T, dT = t3b_100111.update(T, dT, H, shift, system)
    # T, dT = t3c_010111.update(T, dT, H, shift, system)
    #
    # # only alpha hole, beta particle
    # T, dT = t3b_110011.update(T, dT, H, shift, system)
    # T, dT = t3b_110001.update(T, dT, H, shift, system)
    # T, dT = t3c_110011.update(T, dT, H, shift, system)
    # T, dT = t3c_100011.update(T, dT, H, shift, system)
    #
    # # only alpha particle, beta hole
    # T, dT = t3b_101110.update(T, dT, H, shift, system)
    # T, dT = t3b_100110.update(T, dT, H, shift, system)
    # T, dT = t3c_011101.update(T, dT, H, shift, system)
    # T, dT = t3c_011100.update(T, dT, H, shift, system)
    #
    # # only alpha hole, beta hole, alpha particle
    # T, dT = t3b_101010.update(T, dT, H, shift, system)
    # T, dT = t3b_001010.update(T, dT, H, shift, system)
    # T, dT = t3c_011001.update(T, dT, H, shift, system)
    #
    # # only alpha hole, beta hole, beta particle
    # T, dT = t3b_110010.update(T, dT, H, shift, system)
    # T, dT = t3c_110001.update(T, dT, H, shift, system)
    # T, dT = t3c_100001.update(T, dT, H, shift, system)
    #
    # # only alpha hole, alpha particle, beta particle
    #
    #
    #
    # T, dT = t3b_100010.update(T, dT, H, shift, system)
    # T, dT = t3b_001110.update(T, dT, H, shift, system)
    # T, dT = t3b_100011.update(T, dT, H, shift, system)
    # T, dT = t3b_100001.update(T, dT, H, shift, system)
    #
    # # update t3c
    # T, dT = t3c_011011.update(T, dT, H, shift, system)
    # T, dT = t3c_010011.update(T, dT, H, shift, system)
    # T, dT = t3c_010101.update(T, dT, H, shift, system)
    # T, dT = t3c_010100.update(T, dT, H, shift, system)
    # T, dT = t3c_010001.update(T, dT, H, shift, system)


    # perform all T3 update loops
    # update t3a
    T, dT = t3a_111111.update(T, dT, H, shift, system) # h(a),p(a) //
    T, dT = t3a_110111.update(T, dT, H, shift, system) # h(a),p(a) // p(a)
    T, dT = t3a_111011.update(T, dT, H, shift, system) # h(a),p(a) // h(a)
    T, dT = t3a_110011.update(T, dT, H, shift, system) # h(
    T, dT = t3a_100111.update(T, dT, H, shift, system)
    T, dT = t3a_111001.update(T, dT, H, shift, system)
    T, dT = t3a_100011.update(T, dT, H, shift, system)
    T, dT = t3a_110001.update(T, dT, H, shift, system)
    T, dT = t3a_100001.update(T, dT, H, shift, system)
    # update t3b
    T, dT = t3b_111111.update(T, dT, H, shift, system)
    T, dT = t3b_111110.update(T, dT, H, shift, system)
    T, dT = t3b_111011.update(T, dT, H, shift, system)
    T, dT = t3b_110111.update(T, dT, H, shift, system)
    T, dT = t3b_101111.update(T, dT, H, shift, system)
    T, dT = t3b_111001.update(T, dT, H, shift, system)
    T, dT = t3b_111010.update(T, dT, H, shift, system)
    T, dT = t3b_001111.update(T, dT, H, shift, system)
    T, dT = t3b_100111.update(T, dT, H, shift, system)
    T, dT = t3b_110011.update(T, dT, H, shift, system)
    T, dT = t3b_101110.update(T, dT, H, shift, system)
    T, dT = t3b_101011.update(T, dT, H, shift, system)
    T, dT = t3b_110110.update(T, dT, H, shift, system)
    T, dT = t3b_101001.update(T, dT, H, shift, system)
    T, dT = t3b_101010.update(T, dT, H, shift, system)
    T, dT = t3b_110001.update(T, dT, H, shift, system)
    T, dT = t3b_110010.update(T, dT, H, shift, system)
    T, dT = t3b_001011.update(T, dT, H, shift, system)
    T, dT = t3b_001110.update(T, dT, H, shift, system)
    T, dT = t3b_100011.update(T, dT, H, shift, system)
    T, dT = t3b_100110.update(T, dT, H, shift, system)
    T, dT = t3b_001001.update(T, dT, H, shift, system)
    T, dT = t3b_001010.update(T, dT, H, shift, system)
    T, dT = t3b_100001.update(T, dT, H, shift, system)
    T, dT = t3b_100010.update(T, dT, H, shift, system)
    # update t3c
    T, dT = t3c_111111.update(T, dT, H, shift, system)
    T, dT = t3c_111101.update(T, dT, H, shift, system)
    T, dT = t3c_111011.update(T, dT, H, shift, system)
    T, dT = t3c_110111.update(T, dT, H, shift, system)
    T, dT = t3c_011111.update(T, dT, H, shift, system)
    T, dT = t3c_111100.update(T, dT, H, shift, system)
    T, dT = t3c_111001.update(T, dT, H, shift, system)
    T, dT = t3c_010111.update(T, dT, H, shift, system)
    T, dT = t3c_100111.update(T, dT, H, shift, system)
    T, dT = t3c_110011.update(T, dT, H, shift, system)
    T, dT = t3c_011101.update(T, dT, H, shift, system)
    T, dT = t3c_011011.update(T, dT, H, shift, system)
    T, dT = t3c_110101.update(T, dT, H, shift, system)
    T, dT = t3c_110100.update(T, dT, H, shift, system)
    T, dT = t3c_110001.update(T, dT, H, shift, system)
    T, dT = t3c_011100.update(T, dT, H, shift, system)
    T, dT = t3c_011001.update(T, dT, H, shift, system)
    T, dT = t3c_100011.update(T, dT, H, shift, system)
    T, dT = t3c_100101.update(T, dT, H, shift, system)
    T, dT = t3c_010011.update(T, dT, H, shift, system)
    T, dT = t3c_010101.update(T, dT, H, shift, system)
    T, dT = t3c_010100.update(T, dT, H, shift, system)
    T, dT = t3c_010001.update(T, dT, H, shift, system)
    T, dT = t3c_100100.update(T, dT, H, shift, system)
    T, dT = t3c_100001.update(T, dT, H, shift, system)
    # update t3d
    T, dT = t3d_111111.update(T, dT, H, shift, system)
    T, dT = t3d_110111.update(T, dT, H, shift, system)
    T, dT = t3d_111011.update(T, dT, H, shift, system)
    T, dT = t3d_110011.update(T, dT, H, shift, system)
    T, dT = t3d_100111.update(T, dT, H, shift, system)
    T, dT = t3d_111001.update(T, dT, H, shift, system)
    T, dT = t3d_100011.update(T, dT, H, shift, system)
    T, dT = t3d_110001.update(T, dT, H, shift, system)
    T, dT = t3d_100001.update(T, dT, H, shift, system)

    return T, dT
