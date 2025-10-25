"""Functions to calculate the ground-state CC(t;3) triples correction to CCSDt."""
import time
from ccpy.utilities.linear_algebra import ccpy_einsum
import numpy as np

from ccpy.constants.constants import hartreetoeV
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.lib.core import ccp3_opt_loops, ccp3_adaptive_loops, ccp3_full_correction
from ccpy.left.left_cc_intermediates import build_left_ccsdt_p_intermediates
from ccpy.eomcc.eomccsdt_intermediates import get_eomccsd_intermediates, get_eomccsdt_intermediates, add_R3_p_terms
from ccpy.eomcc.eomccsdt import build_HR_3A, build_HR_3B, build_HR_3C, build_HR_3D
from ccpy.utilities.utilities import unravel_triples_amplitudes

def calc_ccp3_2ba(T, L, t3_excitations, corr_energy, H, H0, system, use_RHF=False):
    """
    Calculate the ground-state CC(P;3) correction to the CC(P) energy in a
    memory-optimized manner that avoids the usage of the large logical P space
    array. Input T and L are in the same order and match with t3_excitations.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    #### aaa correction ####
    # calculate intermediates
    I2A_vvov = H.aa.vvov + ccpy_einsum("me,abim->abie", H.a.ov, T.aa)
    # perform correction in-loop
    dA_aaa, dB_aaa, dC_aaa, dD_aaa = ccp3_opt_loops.ccp3a_2ba(
            t3_excitations["aaa"].T,
            T.aa, L.a, L.aa,
            H.aa.vooo, I2A_vvov, H.aa.oovv, H.a.ov,
            H.aa.vovv, H.aa.ooov, H0.a.oo, H0.a.vv,
            H.a.oo, H.a.vv, H.aa.voov, H.aa.oooo,
            H.aa.vvvv,
            d3aaa_o, d3aaa_v,
    )
    #### aab correction ####
    # calculate intermediates
    I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ecjk->mcjk", H.a.ov, T.ab)
    I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeik->amik", H.b.ov, T.ab)
    I2A_vooo = H.aa.vooo - ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
    dA_aab, dB_aab, dC_aab, dD_aab = ccp3_opt_loops.ccp3b_2ba(
            t3_excitations["aab"].T,
            T.aa, T.ab, L.a, L.b, L.aa, L.ab,
            I2B_ovoo, I2B_vooo, I2A_vooo,
            H.ab.vvvo, H.ab.vvov, H.aa.vvov,
            H.ab.vovv, H.ab.ovvv, H.aa.vovv,
            H.ab.ooov, H.ab.oovo, H.aa.ooov,
            H.a.ov, H.b.ov, H.aa.oovv, H.ab.oovv,
            H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
            H.aa.voov, H.aa.oooo, H.aa.vvvv, H.ab.ovov,
            H.ab.vovo, H.ab.oooo, H.ab.vvvv, H.bb.voov,
            d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
    )
    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
        correction_B = 2.0 * dB_aaa + 2.0 * dB_aab
        correction_C = 2.0 * dC_aaa + 2.0 * dC_aab
        correction_D = 2.0 * dD_aaa + 2.0 * dD_aab
    else:
        I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
        I2C_vooo = H.bb.vooo - ccpy_einsum("me,cekj->cmkj", H.b.ov, T.bb)
        I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ebij->mbij", H.a.ov, T.ab)

        dA_abb, dB_abb, dC_abb, dD_abb = ccp3_opt_loops.ccp3c_2ba(
                t3_excitations["abb"].T,
                T.ab, T.bb, L.a, L.b, L.ab, L.bb,
                I2B_vooo, I2C_vooo, I2B_ovoo,
                H.ab.vvov, H.bb.vvov, H.ab.vvvo, H.ab.ovvv,
                H.ab.vovv, H.bb.vovv, H.ab.oovo, H.ab.ooov,
                H.bb.ooov,
                H.a.ov, H.b.ov,
                H.ab.oovv, H.bb.oovv,
                H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                H.aa.voov, H.ab.ovov, H.ab.vovo, H.ab.oooo,
                H.ab.vvvv, H.bb.voov, H.bb.oooo, H.bb.vvvv,
                d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
        )

        I2C_vvov = H.bb.vvov + ccpy_einsum("me,abim->abie", H.b.ov, T.bb)
        dA_bbb, dB_bbb, dC_bbb, dD_bbb = ccp3_opt_loops.ccp3d_2ba(
                t3_excitations["bbb"].T,
                T.bb, L.b, L.bb,
                H.bb.vooo, I2C_vvov, H.bb.oovv, H.b.ov,
                H.bb.vovv, H.bb.ooov, H0.b.oo, H0.b.vv,
                H.b.oo, H.b.vv, H.bb.voov, H.bb.oooo, H.bb.vvvv,
                d3bbb_o, d3bbb_v,
        )

        correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb
        correction_B = dB_aaa + dB_aab + dB_abb + dB_bbb
        correction_C = dC_aaa + dC_aab + dC_abb + dC_bbb
        correction_D = dD_aaa + dD_aab + dD_abb + dD_bbb

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    energy_A = corr_energy + correction_A
    energy_B = corr_energy + correction_B
    energy_C = corr_energy + correction_C
    energy_D = corr_energy + correction_D

    total_energy_A = system.reference_energy + energy_A
    total_energy_B = system.reference_energy + energy_B
    total_energy_C = system.reference_energy + energy_C
    total_energy_D = system.reference_energy + energy_D

    print('   CC(P;3) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   CC(P) = {:>10.10f}".format(system.reference_energy + corr_energy))
    print(
        "   CC(P;3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CC(P;3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CC(P;3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CC(P;3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )

    Eccp3 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    deltap3 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}

    return Eccp3, deltap3

def calc_ccp3_2ba_with_selection(T, L, t3_excitations, corr_energy, H, H0, system, num_add, use_RHF=False, min_thresh=0.0, buffer_factor=2):
    """
    Calculate the ground-state CC(P;3) correction to the CC(P) energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    # initialize empty moments vector and triples list
    num_add = int(num_add)
    moments = np.zeros(buffer_factor * num_add)
    triples_list = np.zeros((buffer_factor * num_add, 6), dtype=np.int32)

    #### aaa correction ####
    # calculate intermediates
    I2A_vvov = H.aa.vvov + ccpy_einsum("me,abim->abie", H.a.ov, T.aa)
    # perform correction in-loop
    nfill = 1
    dA_aaa, dB_aaa, dC_aaa, dD_aaa, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3a_2ba_with_selection_opt(
        moments,
        triples_list,
        nfill,
        t3_excitations['aaa'].T,
        T.aa, L.a, L.aa,
        H.aa.vooo, I2A_vvov, H0.aa.oovv, H.a.ov,
        H.aa.vovv, H.aa.ooov, H0.a.oo, H0.a.vv,
        H.a.oo, H.a.vv, H.aa.voov, H.aa.oooo,
        H.aa.vvvv,
        d3aaa_o, d3aaa_v,
        num_add, min_thresh, buffer_factor,
        )

    #### aab correction ####
    # calculate intermediates
    I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ecjk->mcjk", H.a.ov, T.ab)
    I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeik->amik", H.b.ov, T.ab)
    I2A_vooo = H.aa.vooo - ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
    dA_aab, dB_aab, dC_aab, dD_aab, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3b_2ba_with_selection_opt(
        moments,
        triples_list,
        nfill,
        t3_excitations['aab'].T,
        T.aa, T.ab, L.a, L.b, L.aa, L.ab,
        I2B_ovoo, I2B_vooo, I2A_vooo,
        H.ab.vvvo, H.ab.vvov, H.aa.vvov,
        H.ab.vovv, H.ab.ovvv, H.aa.vovv,
        H.ab.ooov, H.ab.oovo, H.aa.ooov,
        H.a.ov, H.b.ov, H0.aa.oovv, H0.ab.oovv,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H.aa.voov, H.aa.oooo, H.aa.vvvv, H.ab.ovov,
        H.ab.vovo, H.ab.oooo, H.ab.vvvv, H.bb.voov,
        d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
        num_add, min_thresh, buffer_factor,
    )

    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
        correction_B = 2.0 * dB_aaa + 2.0 * dB_aab
        correction_C = 2.0 * dC_aaa + 2.0 * dC_aab
        correction_D = 2.0 * dD_aaa + 2.0 * dD_aab
    else:
        #### abb correction ####
        I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
        I2C_vooo = H.bb.vooo - ccpy_einsum("me,cekj->cmkj", H.b.ov, T.bb)
        I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ebij->mbij", H.a.ov, T.ab)
        dA_abb, dB_abb, dC_abb, dD_abb, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3c_2ba_with_selection_opt(
            moments,
            triples_list,
            nfill,
            t3_excitations['abb'].T,
            T.ab, T.bb, L.a, L.b, L.ab, L.bb,
            I2B_vooo, I2C_vooo, I2B_ovoo,
            H.ab.vvov, H.bb.vvov, H.ab.vvvo, H.ab.ovvv,
            H.ab.vovv, H.bb.vovv, H.ab.oovo, H.ab.ooov,
            H.bb.ooov,
            H.a.ov, H.b.ov,
            H0.ab.oovv, H0.bb.oovv,
            H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
            H.aa.voov, H.ab.ovov, H.ab.vovo, H.ab.oooo,
            H.ab.vvvv, H.bb.voov, H.bb.oooo, H.bb.vvvv,
            d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
            num_add, min_thresh, buffer_factor,
        )
        #### bbb correction ####
        I2C_vvov = H.bb.vvov + ccpy_einsum("me,abim->abie", H.b.ov, T.bb)
        dA_bbb, dB_bbb, dC_bbb, dD_bbb, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3d_2ba_with_selection_opt(
            moments,
            triples_list,
            nfill,
            t3_excitations['bbb'].T,
            T.bb, L.b, L.bb,
            H.bb.vooo, I2C_vvov, H0.bb.oovv, H.b.ov,
            H.bb.vovv, H.bb.ooov, H0.b.oo, H0.b.vv,
            H.b.oo, H.b.vv, H.bb.voov, H.bb.oooo, H.bb.vvvv,
            d3bbb_o, d3bbb_v,
            num_add, min_thresh, buffer_factor,
        )

        correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb
        correction_B = dB_aaa + dB_aab + dB_abb + dB_bbb
        correction_C = dC_aaa + dC_aab + dC_abb + dC_bbb
        correction_D = dD_aaa + dD_aab + dD_abb + dD_bbb

    # Important: perform a final sort of the excitations and moments, returning the first num_add elements only
    idx = np.argsort(np.abs(moments))
    triples_list = triples_list[idx[::-1], :]
    triples_list = triples_list[:num_add, :]

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    # print the results
    energy_A = corr_energy + correction_A
    energy_B = corr_energy + correction_B
    energy_C = corr_energy + correction_C
    energy_D = corr_energy + correction_D

    total_energy_A = system.reference_energy + energy_A
    total_energy_B = system.reference_energy + energy_B
    total_energy_C = system.reference_energy + energy_C
    total_energy_D = system.reference_energy + energy_D

    print('   CC(P;3) Calculation Summary')
    print('   ---------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   CC(P) = {:>10.10f}".format(system.reference_energy + corr_energy))
    print(
        "   CC(P;3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CC(P;3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CC(P;3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CC(P;3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )
    print(
        "   Selected moments account for {:>5.2f}% of the total CC(P;3)_D correction\n".format(
            sum(moments) / correction_D * 100
        )
    )

    Eccp3 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    deltap3 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}

    return Eccp3["D"], triples_list

def calc_ccp3_full(T, L, t3_excitations, corr_energy, H, H0, system, use_RHF=False):
    """
    Calculate the ground-state CC(P;3) correction to the CC(P) energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    # get L(P)*T(P) intermediates
    # determine whether l3 updates and l3*t3 intermediates should be done. Stupid compatibility with
    # empty sections of t3_excitations or l3_excitations. L3 ordering matches T3 at this point.
    do_l3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    do_t3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    if np.array_equal(t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aaa"] = False
        do_l3["aaa"] = False
    if np.array_equal(t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aab"] = False
        do_l3["aab"] = False
    if np.array_equal(t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["abb"] = False
        do_l3["abb"] = False
    if np.array_equal(t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["bbb"] = False
        do_l3["bbb"] = False
    X = build_left_ccsdt_p_intermediates(L, t3_excitations, T, t3_excitations, system, do_t3, do_l3, RHF_symmetry=use_RHF)

    # unravel triples vector into t3(abcijk) and l3(abcijk)
    T_unravel = unravel_triples_amplitudes(T, t3_excitations, system, do_t3)
    L_unravel = unravel_triples_amplitudes(L, t3_excitations, system, do_l3)

    #### aaa correction ####
    M3A = build_M3A_full(T_unravel, H)
    L3A = build_L3A_full(L_unravel, H, X)
    dA_aaa, dB_aaa, dC_aaa, dD_aaa = ccp3_opt_loops.ccp3_opt_loops.ccp3a_full(
        M3A, L3A, t3_excitations["aaa"].T,
        H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
        H.aa.voov, H.aa.oooo, H.aa.vvvv,
        d3aaa_o, d3aaa_v,
    )
    #### aab correction ####
    M3B = build_M3B_full(T_unravel, H)
    L3B = build_L3B_full(L_unravel, H, X)
    dA_aab, dB_aab, dC_aab, dD_aab = ccp3_opt_loops.ccp3_opt_loops.ccp3b_full(
        M3B, L3B, t3_excitations["aab"].T,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H.aa.voov, H.aa.oooo, H.aa.vvvv,
        H.ab.ovov, H.ab.vovo,
        H.ab.oooo, H.ab.vvvv,
        H.bb.voov,
        d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
    )
    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
        correction_B = 2.0 * dB_aaa + 2.0 * dB_aab
        correction_C = 2.0 * dC_aaa + 2.0 * dC_aab
        correction_D = 2.0 * dD_aaa + 2.0 * dD_aab
    else:
        #### abb correction ####
        M3C = build_M3C_full(T_unravel, H)
        L3C = build_L3C_full(L_unravel, H, X)
        dA_abb, dB_abb, dC_abb, dD_abb = ccp3_opt_loops.ccp3_opt_loops.ccp3c_full(
            M3C, L3C, t3_excitations["abb"].T,
            H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
            H.aa.voov,
            H.ab.ovov, H.ab.vovo,
            H.ab.oooo, H.ab.vvvv,
            H.bb.voov, H.bb.oooo, H.bb.vvvv,
            d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
        )
        #### bbb correction ####
        M3D = build_M3D_full(T_unravel, H)
        L3D = build_L3D_full(L_unravel, H, X)
        dA_bbb, dB_bbb, dC_bbb, dD_bbb = ccp3_opt_loops.ccp3_opt_loops.ccp3d_full(
            M3D, L3D, t3_excitations["bbb"].T,
            H0.b.oo, H0.b.vv, H.b.oo, H.b.vv,
            H.bb.voov, H.bb.oooo, H.bb.vvvv,
            d3bbb_o, d3bbb_v,
        )
        correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb
        correction_B = dB_aaa + dB_aab + dB_abb + dB_bbb
        correction_C = dC_aaa + dC_aab + dC_abb + dC_bbb
        correction_D = dD_aaa + dD_aab + dD_abb + dD_bbb

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    energy_A = corr_energy + correction_A
    energy_B = corr_energy + correction_B
    energy_C = corr_energy + correction_C
    energy_D = corr_energy + correction_D

    total_energy_A = system.reference_energy + energy_A
    total_energy_B = system.reference_energy + energy_B
    total_energy_C = system.reference_energy + energy_C
    total_energy_D = system.reference_energy + energy_D

    print('   CC(P;3) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   CC(P) = {:>10.10f}".format(system.reference_energy + corr_energy))
    print(
        "   CC(P;3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CC(P;3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CC(P;3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CC(P;3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )

    Eccp3 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    deltap3 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}

    return Eccp3, deltap3

def calc_ccp3_full_opt(T, L, t3_excitations, corr_energy, H, H0, system, use_RHF=False):
    """
    Calculate the ground-state CC(P;3) correction to the CC(P) energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    # get L(P)*T(P) intermediates
    # determine whether l3 updates and l3*t3 intermediates should be done. Stupid compatibility with
    # empty sections of t3_excitations or l3_excitations. L3 ordering matches T3 at this point.
    do_l3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    do_t3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    if np.array_equal(t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aaa"] = False
        do_l3["aaa"] = False
    if np.array_equal(t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aab"] = False
        do_l3["aab"] = False
    if np.array_equal(t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["abb"] = False
        do_l3["abb"] = False
    if np.array_equal(t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["bbb"] = False
        do_l3["bbb"] = False

    # Create intermediates
    X = build_left_ccsdt_p_intermediates(L, t3_excitations, T, t3_excitations, system, do_t3, do_l3, RHF_symmetry=use_RHF)
    I2A_vvov = H.aa.vvov + ccpy_einsum("me,abim->abie", H.a.ov, T.aa)
    I2A_vooo = H.aa.vooo - ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
    I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ecjk->mcjk", H.a.ov, T.ab)
    I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeik->amik", H.b.ov, T.ab)
    I2C_vooo = H.bb.vooo - ccpy_einsum("me,cekj->cmkj", H.b.ov, T.bb)

    #### aaa correction ####
    dA_aaa = 0.0
    dB_aaa = 0.0
    dC_aaa = 0.0
    dD_aaa = 0.0
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(j + 1, system.noccupied_alpha):
                M3A = ccp3_full_correction.ccp3_full_correction.build_moments3a_ijk(
                    i + 1, j + 1, k + 1,
                    T.aaa, t3_excitations["aaa"],
                    T.aab, t3_excitations["aab"],
                    T.aa,
                    H.a.oo, H.a.vv.T,
                    H.aa.oovv, I2A_vvov.transpose(3, 0, 1, 2), H.aa.vooo.transpose(1, 0, 2, 3),
                    H.aa.oooo, H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvvv.transpose(3, 2, 1, 0),
                    H.ab.oovv, H.ab.voov.transpose(1, 3, 0, 2),
                )
                L3A = ccp3_full_correction.ccp3_full_correction.build_leftamps3a_ijk(
                    i + 1, j + 1, k + 1,
                    L.a, L.aa,
                    L.aaa, t3_excitations["aaa"],
                    L.aab, t3_excitations["aab"],
                    H.a.ov, H.a.oo, H.a.vv,
                    H.aa.oooo, H.aa.ooov, H.aa.oovv,
                    H.aa.voov, H.aa.vovv, H.aa.vvvv,
                    H.ab.ovvo,
                    X.aa.ooov, X.aa.vovv,
                )
                dA_aaa, dB_aaa, dC_aaa, dD_aaa = ccp3_full_correction.ccp3_full_correction.ccp3a_ijk(
                    dA_aaa, dB_aaa, dC_aaa, dD_aaa,
                    i + 1, j + 1, k + 1, 0.0,
                    M3A, L3A, t3_excitations["aaa"],
                    H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
                    H.aa.voov, H.aa.oooo, H.aa.vvvv,
                    d3aaa_o, d3aaa_v,
                )
    #### aab correction ####
    dA_aab = 0.0
    dB_aab = 0.0
    dC_aab = 0.0
    dD_aab = 0.0
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(system.noccupied_beta):
                M3B = ccp3_full_correction.ccp3_full_correction.build_moments3b_ijk(
                    i + 1, j + 1, k + 1,
                    T.aaa, t3_excitations["aaa"],
                    T.aab, t3_excitations["aab"],
                    T.abb, t3_excitations["abb"],
                    T.aa, T.ab,
                    H.a.oo, H.a.vv.T, H.b.oo, H.b.vv.T,
                    H.aa.oovv, H.aa.vvov.transpose(3, 0, 1, 2), I2A_vooo.transpose(1, 0, 2, 3), H.aa.oooo, H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvvv.transpose(3, 2, 1, 0),
                    H.ab.oovv, H.ab.vvov.transpose(3, 0, 1, 2), H.ab.vvvo.transpose(2, 0, 1, 3), I2B_vooo.transpose(1, 0, 2, 3), I2B_ovoo,
                    H.ab.oooo, H.ab.voov.transpose(1, 3, 0, 2), H.ab.vovo.transpose(1, 2, 0, 3), H.ab.ovov.transpose(0, 3, 1, 2), H.ab.ovvo.transpose(0, 2, 1, 3), H.ab.vvvv.transpose(3, 2, 1, 0),
                    H.bb.oovv, H.bb.voov.transpose(1, 3, 0, 2),
                )
                L3B = ccp3_full_correction.ccp3_full_correction.build_leftamps3b_ijk(
                    i + 1, j + 1, k + 1,
                    L.a, L.b, L.aa, L.ab,
                    L.aaa, t3_excitations["aaa"],
                    L.aab, t3_excitations["aab"],
                    L.abb, t3_excitations["abb"],
                    H.a.ov, H.a.oo, H.a.vv,
                    H.b.ov, H.b.oo, H.b.vv,
                    H.aa.oooo, H.aa.ooov, H.aa.oovv,
                    H.aa.voov, H.aa.vovv, H.aa.vvvv,
                    H.ab.oooo, H.ab.ooov, H.ab.oovo,
                    H.ab.oovv,
                    H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
                    H.ab.vovv, H.ab.ovvv, H.ab.vvvv,
                    H.bb.voov,
                    X.aa.ooov, X.aa.vovv,
                    X.ab.ooov, X.ab.oovo, X.ab.vovv, X.ab.ovvv
                )
                dA_aab, dB_aab, dC_aab, dD_aab = ccp3_full_correction.ccp3_full_correction.ccp3b_ijk(
                    dA_aab, dB_aab, dC_aab, dD_aab,
                    i + 1, j + 1, k + 1, 0.0,
                    M3B, L3B, t3_excitations["aab"],
                    H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                    H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                    H.aa.voov, H.aa.oooo, H.aa.vvvv,
                    H.ab.ovov, H.ab.vovo,
                    H.ab.oooo, H.ab.vvvv,
                    H.bb.voov,
                    d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
                )
    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
        correction_B = 2.0 * dB_aaa + 2.0 * dB_aab
        correction_C = 2.0 * dC_aaa + 2.0 * dC_aab
        correction_D = 2.0 * dD_aaa + 2.0 * dD_aab
    else:
        #### abb correction ####
        dA_abb = 0.0
        dB_abb = 0.0
        dC_abb = 0.0
        dD_abb = 0.0
        for i in range(system.noccupied_alpha):
            for j in range(system.noccupied_beta):
                for k in range(j + 1, system.noccupied_beta):
                    M3C = ccp3_full_correction.ccp3_full_correction.build_moments3c_ijk(
                        i + 1, j + 1, k + 1,
                        T.aab, t3_excitations["aab"],
                        T.abb, t3_excitations["abb"],
                        T.bbb, t3_excitations["bbb"],
                        T.ab, T.bb,
                        H.a.oo, H.a.vv.T, H.b.oo, H.b.vv.T,
                        H.aa.oovv, H.aa.voov.transpose(1, 3, 0, 2),
                        H.ab.oovv, I2B_vooo.transpose(1, 0, 2, 3), I2B_ovoo, H.ab.vvov.transpose(3, 0, 1, 2), H.ab.vvvo.transpose(2, 0, 1, 3), H.ab.oooo,
                        H.ab.voov.transpose(1, 3, 0, 2), H.ab.vovo.transpose(1, 2, 0, 3), H.ab.ovov.transpose(0, 3, 1, 2), H.ab.ovvo.transpose(0, 2, 1, 3), H.ab.vvvv.transpose(2, 3, 0, 1),
                        H.bb.oovv, I2C_vooo.transpose(1, 0, 2, 3), H.bb.vvov.transpose(3, 0, 1, 2), H.bb.oooo, H.bb.voov.transpose(1, 3, 0, 2), H.bb.vvvv.transpose(3, 2, 1, 0),
                    )
                    L3C = ccp3_full_correction.ccp3_full_correction.build_leftamps3c_ijk(
                        i + 1, j + 1, k + 1,
                        L.a, L.b, L.ab, L.bb,
                        L.aab, t3_excitations["aab"],
                        L.abb, t3_excitations["abb"],
                        L.bbb, t3_excitations["bbb"],
                        H.a.ov, H.a.oo, H.a.vv,
                        H.b.ov, H.b.oo, H.b.vv,
                        H.aa.voov,
                        H.ab.oooo, H.ab.ooov, H.ab.oovo,
                        H.ab.oovv,
                        H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
                        H.ab.vovv, H.ab.ovvv, H.ab.vvvv,
                        H.bb.oooo, H.bb.ooov, H.bb.oovv,
                        H.bb.voov, H.bb.vovv, H.bb.vvvv,
                        X.ab.ooov, X.ab.oovo, X.ab.vovv, X.ab.ovvv,
                        X.bb.ooov, X.bb.vovv,
                    )
                    dA_abb, dB_abb, dC_abb, dD_abb = ccp3_full_correction.ccp3_full_correction.ccp3c_ijk(
                        dA_abb, dB_abb, dC_abb, dD_abb,
                        i + 1, j + 1, k + 1, 0.0,
                        M3C, L3C, t3_excitations["abb"],
                        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                        H.aa.voov,
                        H.ab.ovov, H.ab.vovo,
                        H.ab.oooo, H.ab.vvvv,
                        H.bb.voov, H.bb.oooo, H.bb.vvvv,
                        d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
                    )
        #### bbb correction ####
        dA_bbb = 0.0
        dB_bbb = 0.0
        dC_bbb = 0.0
        dD_bbb = 0.0
        for i in range(system.noccupied_beta):
            for j in range(i + 1, system.noccupied_beta):
                for k in range(j + 1, system.noccupied_beta):
                    M3D = ccp3_full_correction.ccp3_full_correction.build_moments3d_ijk(
                        i + 1, j + 1, k + 1,
                        T.abb, t3_excitations["abb"],
                        T.bbb, t3_excitations["bbb"],
                        T.bb,
                        H.b.oo, H.b.vv.T,
                        H.bb.oovv, H.bb.vvov.transpose(3, 0, 1, 2), I2C_vooo.transpose(1, 0, 2, 3),
                        H.bb.oooo, H.bb.voov.transpose(1, 3, 0, 2), H.bb.vvvv.transpose(3, 2, 1, 0),
                        H.ab.oovv, H.ab.ovvo.transpose(0, 2, 1, 3),
                    )
                    L3D = ccp3_full_correction.ccp3_full_correction.build_leftamps3d_ijk(
                        i + 1, j + 1, k + 1,
                        L.b, L.bb,
                        L.abb, t3_excitations["abb"],
                        L.bbb, t3_excitations["bbb"],
                        H.b.ov, H.b.oo, H.b.vv,
                        H.ab.voov,
                        H.bb.oooo, H.bb.ooov, H.bb.oovv,
                        H.bb.voov, H.bb.vovv, H.bb.vvvv,
                        X.bb.ooov, X.bb.vovv,
                    )
                    dA_bbb, dB_bbb, dC_bbb, dD_bbb = ccp3_full_correction.ccp3_full_correction.ccp3d_ijk(
                        dA_bbb, dB_bbb, dC_bbb, dD_bbb,
                        i + 1, j + 1, k + 1, 0.0,
                        M3D, L3D, t3_excitations["bbb"],
                        H0.b.oo, H0.b.vv, H.b.oo, H.b.vv,
                        H.bb.voov, H.bb.oooo, H.bb.vvvv,
                        d3bbb_o, d3bbb_v,
                    )
        correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb
        correction_B = dB_aaa + dB_aab + dB_abb + dB_bbb
        correction_C = dC_aaa + dC_aab + dC_abb + dC_bbb
        correction_D = dD_aaa + dD_aab + dD_abb + dD_bbb

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    energy_A = corr_energy + correction_A
    energy_B = corr_energy + correction_B
    energy_C = corr_energy + correction_C
    energy_D = corr_energy + correction_D

    total_energy_A = system.reference_energy + energy_A
    total_energy_B = system.reference_energy + energy_B
    total_energy_C = system.reference_energy + energy_C
    total_energy_D = system.reference_energy + energy_D

    print('   CC(P;3) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   CC(P) = {:>10.10f}".format(system.reference_energy + corr_energy))
    print(
        "   CC(P;3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CC(P;3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CC(P;3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CC(P;3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )

    Eccp3 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    deltap3 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}

    return Eccp3, deltap3

def calc_ccp3_full_opt_with_selection(T, L, t3_excitations, corr_energy, H, H0, system, num_add, use_RHF=False, min_thresh=0.0, buffer_factor=2):
    """
    Calculate the ground-state CC(P;3) correction to the CC(P) energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # initialize empty moments vector and triples list
    num_add = int(num_add)
    moments = np.zeros(buffer_factor * num_add)
    triples_list = np.zeros((buffer_factor * num_add, 6), dtype=np.int32)

    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    # get L(P)*T(P) intermediates
    # determine whether l3 updates and l3*t3 intermediates should be done. Stupid compatibility with
    # empty sections of t3_excitations or l3_excitations. L3 ordering matches T3 at this point.
    do_l3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    do_t3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    if np.array_equal(t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aaa"] = False
        do_l3["aaa"] = False
    if np.array_equal(t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aab"] = False
        do_l3["aab"] = False
    if np.array_equal(t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["abb"] = False
        do_l3["abb"] = False
    if np.array_equal(t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["bbb"] = False
        do_l3["bbb"] = False

    # Create intermediates
    X = build_left_ccsdt_p_intermediates(L, t3_excitations, T, t3_excitations, system, do_t3, do_l3, RHF_symmetry=use_RHF)
    I2A_vvov = H.aa.vvov + ccpy_einsum("me,abim->abie", H.a.ov, T.aa)
    I2A_vooo = H.aa.vooo - ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
    I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ecjk->mcjk", H.a.ov, T.ab)
    I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeik->amik", H.b.ov, T.ab)
    I2C_vooo = H.bb.vooo - ccpy_einsum("me,cekj->cmkj", H.b.ov, T.bb)

    nfill = 1
    #### aaa correction ####
    dA_aaa = 0.0
    dB_aaa = 0.0
    dC_aaa = 0.0
    dD_aaa = 0.0
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(j + 1, system.noccupied_alpha):
                M3A = ccp3_full_correction.ccp3_full_correction.build_moments3a_ijk(
                    i + 1, j + 1, k + 1,
                    T.aaa, t3_excitations["aaa"],
                    T.aab, t3_excitations["aab"],
                    T.aa,
                    H.a.oo, H.a.vv.T,
                    H.aa.oovv, I2A_vvov.transpose(3, 0, 1, 2), H.aa.vooo.transpose(1, 0, 2, 3),
                    H.aa.oooo, H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvvv.transpose(3, 2, 1, 0),
                    H.ab.oovv, H.ab.voov.transpose(1, 3, 0, 2),
                )
                L3A = ccp3_full_correction.ccp3_full_correction.build_leftamps3a_ijk(
                    i + 1, j + 1, k + 1,
                    L.a, L.aa,
                    L.aaa, t3_excitations["aaa"],
                    L.aab, t3_excitations["aab"],
                    H.a.ov, H.a.oo, H.a.vv,
                    H.aa.oooo, H.aa.ooov, H.aa.oovv,
                    H.aa.voov, H.aa.vovv, H.aa.vvvv,
                    H.ab.ovvo,
                    X.aa.ooov, X.aa.vovv,
                )
                dA_aaa, dB_aaa, dC_aaa, dD_aaa = ccp3_full_correction.ccp3_full_correction.ccp3a_ijk(
                    dA_aaa, dB_aaa, dC_aaa, dD_aaa,
                    i + 1, j + 1, k + 1, 0.0,
                    M3A, L3A, t3_excitations["aaa"],
                    H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
                    H.aa.voov, H.aa.oooo, H.aa.vvvv,
                    d3aaa_o, d3aaa_v,
                )
                dA_aaa, dB_aaa, dC_aaa, dD_aaa, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3_adaptive_loops.ccp3a_full_with_selection_opt(
                    moments,
                    triples_list,
                    nfill,
                    i + 1, j + 1, k + 1, 0.0,
                    M3A, L3A,
                    t3_excitations['aaa'].T,
                    H0.a.oo, H0.a.vv,
                    H.a.oo, H.a.vv, H.aa.voov, H.aa.oooo,
                    H.aa.vvvv,
                    d3aaa_o, d3aaa_v,
                    num_add, min_thresh, buffer_factor,
                )
    #### aab correction ####
    dA_aab = 0.0
    dB_aab = 0.0
    dC_aab = 0.0
    dD_aab = 0.0
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(system.noccupied_beta):
                M3B = ccp3_full_correction.ccp3_full_correction.build_moments3b_ijk(
                    i + 1, j + 1, k + 1,
                    T.aaa, t3_excitations["aaa"],
                    T.aab, t3_excitations["aab"],
                    T.abb, t3_excitations["abb"],
                    T.aa, T.ab,
                    H.a.oo, H.a.vv.T, H.b.oo, H.b.vv.T,
                    H.aa.oovv, H.aa.vvov.transpose(3, 0, 1, 2), I2A_vooo.transpose(1, 0, 2, 3), H.aa.oooo, H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvvv.transpose(3, 2, 1, 0),
                    H.ab.oovv, H.ab.vvov.transpose(3, 0, 1, 2), H.ab.vvvo.transpose(2, 0, 1, 3), I2B_vooo.transpose(1, 0, 2, 3), I2B_ovoo,
                    H.ab.oooo, H.ab.voov.transpose(1, 3, 0, 2), H.ab.vovo.transpose(1, 2, 0, 3), H.ab.ovov.transpose(0, 3, 1, 2), H.ab.ovvo.transpose(0, 2, 1, 3), H.ab.vvvv.transpose(3, 2, 1, 0),
                    H.bb.oovv, H.bb.voov.transpose(1, 3, 0, 2),
                )
                L3B = ccp3_full_correction.ccp3_full_correction.build_leftamps3b_ijk(
                    i + 1, j + 1, k + 1,
                    L.a, L.b, L.aa, L.ab,
                    L.aaa, t3_excitations["aaa"],
                    L.aab, t3_excitations["aab"],
                    L.abb, t3_excitations["abb"],
                    H.a.ov, H.a.oo, H.a.vv,
                    H.b.ov, H.b.oo, H.b.vv,
                    H.aa.oooo, H.aa.ooov, H.aa.oovv,
                    H.aa.voov, H.aa.vovv, H.aa.vvvv,
                    H.ab.oooo, H.ab.ooov, H.ab.oovo,
                    H.ab.oovv,
                    H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
                    H.ab.vovv, H.ab.ovvv, H.ab.vvvv,
                    H.bb.voov,
                    X.aa.ooov, X.aa.vovv,
                    X.ab.ooov, X.ab.oovo, X.ab.vovv, X.ab.ovvv
                )
                dA_aab, dB_aab, dC_aab, dD_aab = ccp3_full_correction.ccp3_full_correction.ccp3b_ijk(
                    dA_aab, dB_aab, dC_aab, dD_aab,
                    i + 1, j + 1, k + 1, 0.0,
                    M3B, L3B, t3_excitations["aab"],
                    H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                    H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                    H.aa.voov, H.aa.oooo, H.aa.vvvv,
                    H.ab.ovov, H.ab.vovo,
                    H.ab.oooo, H.ab.vvvv,
                    H.bb.voov,
                    d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
                )
    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
        correction_B = 2.0 * dB_aaa + 2.0 * dB_aab
        correction_C = 2.0 * dC_aaa + 2.0 * dC_aab
        correction_D = 2.0 * dD_aaa + 2.0 * dD_aab
    else:
        #### abb correction ####
        dA_abb = 0.0
        dB_abb = 0.0
        dC_abb = 0.0
        dD_abb = 0.0
        for i in range(system.noccupied_alpha):
            for j in range(system.noccupied_beta):
                for k in range(j + 1, system.noccupied_beta):
                    M3C = ccp3_full_correction.ccp3_full_correction.build_moments3c_ijk(
                        i + 1, j + 1, k + 1,
                        T.aab, t3_excitations["aab"],
                        T.abb, t3_excitations["abb"],
                        T.bbb, t3_excitations["bbb"],
                        T.ab, T.bb,
                        H.a.oo, H.a.vv.T, H.b.oo, H.b.vv.T,
                        H.aa.oovv, H.aa.voov.transpose(1, 3, 0, 2),
                        H.ab.oovv, I2B_vooo.transpose(1, 0, 2, 3), I2B_ovoo, H.ab.vvov.transpose(3, 0, 1, 2), H.ab.vvvo.transpose(2, 0, 1, 3), H.ab.oooo,
                        H.ab.voov.transpose(1, 3, 0, 2), H.ab.vovo.transpose(1, 2, 0, 3), H.ab.ovov.transpose(0, 3, 1, 2), H.ab.ovvo.transpose(0, 2, 1, 3), H.ab.vvvv.transpose(2, 3, 0, 1),
                        H.bb.oovv, I2C_vooo.transpose(1, 0, 2, 3), H.bb.vvov.transpose(3, 0, 1, 2), H.bb.oooo, H.bb.voov.transpose(1, 3, 0, 2), H.bb.vvvv.transpose(3, 2, 1, 0),
                    )
                    L3C = ccp3_full_correction.ccp3_full_correction.build_leftamps3c_ijk(
                        i + 1, j + 1, k + 1,
                        L.a, L.b, L.ab, L.bb,
                        L.aab, t3_excitations["aab"],
                        L.abb, t3_excitations["abb"],
                        L.bbb, t3_excitations["bbb"],
                        H.a.ov, H.a.oo, H.a.vv,
                        H.b.ov, H.b.oo, H.b.vv,
                        H.aa.voov,
                        H.ab.oooo, H.ab.ooov, H.ab.oovo,
                        H.ab.oovv,
                        H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
                        H.ab.vovv, H.ab.ovvv, H.ab.vvvv,
                        H.bb.oooo, H.bb.ooov, H.bb.oovv,
                        H.bb.voov, H.bb.vovv, H.bb.vvvv,
                        X.ab.ooov, X.ab.oovo, X.ab.vovv, X.ab.ovvv,
                        X.bb.ooov, X.bb.vovv,
                    )
                    dA_abb, dB_abb, dC_abb, dD_abb = ccp3_full_correction.ccp3_full_correction.ccp3c_ijk(
                        dA_abb, dB_abb, dC_abb, dD_abb,
                        i + 1, j + 1, k + 1, 0.0,
                        M3C, L3C, t3_excitations["abb"],
                        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                        H.aa.voov,
                        H.ab.ovov, H.ab.vovo,
                        H.ab.oooo, H.ab.vvvv,
                        H.bb.voov, H.bb.oooo, H.bb.vvvv,
                        d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
                    )
        #### bbb correction ####
        dA_bbb = 0.0
        dB_bbb = 0.0
        dC_bbb = 0.0
        dD_bbb = 0.0
        for i in range(system.noccupied_beta):
            for j in range(i + 1, system.noccupied_beta):
                for k in range(j + 1, system.noccupied_beta):
                    M3D = ccp3_full_correction.ccp3_full_correction.build_moments3d_ijk(
                        i + 1, j + 1, k + 1,
                        T.abb, t3_excitations["abb"],
                        T.bbb, t3_excitations["bbb"],
                        T.bb,
                        H.b.oo, H.b.vv.T,
                        H.bb.oovv, H.bb.vvov.transpose(3, 0, 1, 2), I2C_vooo.transpose(1, 0, 2, 3),
                        H.bb.oooo, H.bb.voov.transpose(1, 3, 0, 2), H.bb.vvvv.transpose(3, 2, 1, 0),
                        H.ab.oovv, H.ab.ovvo.transpose(0, 2, 1, 3),
                    )
                    L3D = ccp3_full_correction.ccp3_full_correction.build_leftamps3d_ijk(
                        i + 1, j + 1, k + 1,
                        L.b, L.bb,
                        L.abb, t3_excitations["abb"],
                        L.bbb, t3_excitations["bbb"],
                        H.b.ov, H.b.oo, H.b.vv,
                        H.ab.voov,
                        H.bb.oooo, H.bb.ooov, H.bb.oovv,
                        H.bb.voov, H.bb.vovv, H.bb.vvvv,
                        X.bb.ooov, X.bb.vovv,
                    )
                    dA_bbb, dB_bbb, dC_bbb, dD_bbb = ccp3_full_correction.ccp3_full_correction.ccp3d_ijk(
                        dA_bbb, dB_bbb, dC_bbb, dD_bbb,
                        i + 1, j + 1, k + 1, 0.0,
                        M3D, L3D, t3_excitations["bbb"],
                        H0.b.oo, H0.b.vv, H.b.oo, H.b.vv,
                        H.bb.voov, H.bb.oooo, H.bb.vvvv,
                        d3bbb_o, d3bbb_v,
                    )
        correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb
        correction_B = dB_aaa + dB_aab + dB_abb + dB_bbb
        correction_C = dC_aaa + dC_aab + dC_abb + dC_bbb
        correction_D = dD_aaa + dD_aab + dD_abb + dD_bbb

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    energy_A = corr_energy + correction_A
    energy_B = corr_energy + correction_B
    energy_C = corr_energy + correction_C
    energy_D = corr_energy + correction_D

    total_energy_A = system.reference_energy + energy_A
    total_energy_B = system.reference_energy + energy_B
    total_energy_C = system.reference_energy + energy_C
    total_energy_D = system.reference_energy + energy_D

    print('   CC(P;3) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   CC(P) = {:>10.10f}".format(system.reference_energy + corr_energy))
    print(
        "   CC(P;3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CC(P;3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CC(P;3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CC(P;3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )

    Eccp3 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    deltap3 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}

    return Eccp3, deltap3

def calc_ccp3_full_with_selection(T, L, t3_excitations, corr_energy, H, H0, system, num_add, use_RHF=False, min_thresh=0.0, buffer_factor=2):
    """
    Calculate the ground-state CC(P;3) correction to the CC(P) energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    # initialize empty moments vector and triples list
    num_add = int(num_add)
    moments = np.zeros(buffer_factor * num_add)
    triples_list = np.zeros((buffer_factor * num_add, 6), dtype=np.int32)

    # get L(P)*T(P) intermediates
    # determine whether l3 updates and l3*t3 intermediates should be done. Stupid compatibility with
    # empty sections of t3_excitations or l3_excitations. L3 ordering matches T3 at this point.
    do_l3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    do_t3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    if np.array_equal(t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aaa"] = False
        do_l3["aaa"] = False
    if np.array_equal(t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aab"] = False
        do_l3["aab"] = False
    if np.array_equal(t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["abb"] = False
        do_l3["abb"] = False
    if np.array_equal(t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["bbb"] = False
        do_l3["bbb"] = False
    X = build_left_ccsdt_p_intermediates(L, t3_excitations, T, t3_excitations, system, do_t3, do_l3, RHF_symmetry=use_RHF)

    # unravel triples vector into t3(abcijk) and l3(abcijk)
    T_unravel = unravel_triples_amplitudes(T, t3_excitations, system, do_t3)
    L_unravel = unravel_triples_amplitudes(L, t3_excitations, system, do_l3)

    nfill = 1
    #### aaa correction ####
    M3A = build_M3A_full(T_unravel, H)
    L3A = build_L3A_full(L_unravel, H, X)
    dA_aaa, dB_aaa, dC_aaa, dD_aaa, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3_adaptive_loops.ccp3a_full_with_selection_opt(
        moments,
        triples_list,
        nfill,
        M3A, L3A,
        t3_excitations['aaa'].T,
        H0.a.oo, H0.a.vv,
        H.a.oo, H.a.vv, H.aa.voov, H.aa.oooo,
        H.aa.vvvv,
        d3aaa_o, d3aaa_v,
        num_add, min_thresh, buffer_factor,
        )

    #### aab correction ####
    M3B = build_M3B_full(T_unravel, H)
    L3B = build_L3B_full(L_unravel, H, X)
    dA_aab, dB_aab, dC_aab, dD_aab, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3_adaptive_loops.ccp3b_full_with_selection_opt(
        moments,
        triples_list,
        nfill,
        M3B, L3B,
        t3_excitations['aab'].T,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H.aa.voov, H.aa.oooo, H.aa.vvvv, H.ab.ovov,
        H.ab.vovo, H.ab.oooo, H.ab.vvvv, H.bb.voov,
        d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
        num_add, min_thresh, buffer_factor,
    )

    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
        correction_B = 2.0 * dB_aaa + 2.0 * dB_aab
        correction_C = 2.0 * dC_aaa + 2.0 * dC_aab
        correction_D = 2.0 * dD_aaa + 2.0 * dD_aab
    else:
        #### abb correction ####
        M3C = build_M3C_full(T_unravel, H)
        L3C = build_L3C_full(L_unravel, H, X)
        dA_abb, dB_abb, dC_abb, dD_abb, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3_adaptive_loops.ccp3c_full_with_selection_opt(
            moments,
            triples_list,
            nfill,
            M3C, L3C,
            t3_excitations['abb'].T,
            H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
            H.aa.voov, H.ab.ovov, H.ab.vovo, H.ab.oooo,
            H.ab.vvvv, H.bb.voov, H.bb.oooo, H.bb.vvvv,
            d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
            num_add, min_thresh, buffer_factor,
        )
        #### bbb correction ####
        M3D = build_M3D_full(T_unravel, H)
        L3D = build_L3D_full(L_unravel, H, X)
        dA_bbb, dB_bbb, dC_bbb, dD_bbb, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3_adaptive_loops.ccp3d_full_with_selection_opt(
            moments,
            triples_list,
            nfill,
            M3D, L3D,
            t3_excitations['bbb'].T,
            H0.b.oo, H0.b.vv,
            H.b.oo, H.b.vv, H.bb.voov, H.bb.oooo, H.bb.vvvv,
            d3bbb_o, d3bbb_v,
            num_add, min_thresh, buffer_factor,
        )

        correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb
        correction_B = dB_aaa + dB_aab + dB_abb + dB_bbb
        correction_C = dC_aaa + dC_aab + dC_abb + dC_bbb
        correction_D = dD_aaa + dD_aab + dD_abb + dD_bbb

    # Important: perform a final sort of the excitations and moments, returning the first num_add elements only
    idx = np.argsort(np.abs(moments))
    triples_list = triples_list[idx[::-1], :]
    triples_list = triples_list[:num_add, :]

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    # print the results
    energy_A = corr_energy + correction_A
    energy_B = corr_energy + correction_B
    energy_C = corr_energy + correction_C
    energy_D = corr_energy + correction_D

    total_energy_A = system.reference_energy + energy_A
    total_energy_B = system.reference_energy + energy_B
    total_energy_C = system.reference_energy + energy_C
    total_energy_D = system.reference_energy + energy_D

    print('   CC(P;3) Calculation Summary')
    print('   ---------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   CC(P) = {:>10.10f}".format(system.reference_energy + corr_energy))
    print(
        "   CC(P;3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CC(P;3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CC(P;3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CC(P;3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )
    print(
        "   Selected moments account for {:>5.2f}% of the total CC(P;3)_D correction\n".format(
            sum(moments) / correction_D * 100
        )
    )

    Eccp3 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    deltap3 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}

    return Eccp3["D"], triples_list

def calc_eomccp3_full(T, R, L, t3_excitations, r3_excitations, r0, omega, corr_energy, H, H0, system, use_RHF=False):
    """
    Calculate the excited-state CC(t;3) correction to the EOMCCSDt energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    # get L(P)*T(P) intermediates
    # determine whether l3 updates and l3*t3 intermediates should be done. Stupid compatibility with
    # empty sections of t3_excitations or l3_excitations. L3 ordering matches T3 at this point.
    do_l3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    do_t3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    if np.array_equal(t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aaa"] = False
    if np.array_equal(t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aab"] = False
    if np.array_equal(t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["abb"] = False
    if np.array_equal(t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["bbb"] = False
    if np.array_equal(r3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["aaa"] = False
    if np.array_equal(r3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["aab"] = False
    if np.array_equal(r3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["abb"] = False
    if np.array_equal(r3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["bbb"] = False
    X1 = build_left_ccsdt_p_intermediates(L, r3_excitations, T, t3_excitations, system, do_t3, do_l3, RHF_symmetry=use_RHF)

    # Form H(2)*(R1 + R2) intermediates
    Xtemp = get_eomccsd_intermediates(H, R, system)
    X2 = get_eomccsdt_intermediates(H, R, T, Xtemp, system)
    X2 = add_R3_p_terms(X2, H, R, r3_excitations)

    # unravel triples vector into t3(abcijk), r3(abcijk), and l3(abcijk)
    T_unravel = unravel_triples_amplitudes(T, t3_excitations, system, do_t3)
    R_unravel = unravel_triples_amplitudes(R, r3_excitations, system, do_l3)
    L_unravel = unravel_triples_amplitudes(L, r3_excitations, system, do_l3)

    #### aaa correction ####
    # Moments and left vector
    M3A = build_M3A_full(T_unravel, H)
    L3A = build_L3A_full(L_unravel, H, X1)
    EOM3A = build_HR_3A(R_unravel, T_unravel, H, X2)
    # perform correction in-loop
    dA_aaa, dB_aaa, dC_aaa, dD_aaa, ddA_aaa, ddB_aaa, ddC_aaa, ddD_aaa = ccp3_opt_loops.ccp3_opt_loops.eomccp3a_full(
         EOM3A, M3A, L3A, r3_excitations["aaa"].T,
         omega, r0,
         H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
         H.aa.voov, H.aa.oooo, H.aa.vvvv,
         d3aaa_o, d3aaa_v,
    )
    #### aab correction ####
    # moments and left vector
    M3B = build_M3B_full(T_unravel, H)
    L3B = build_L3B_full(L_unravel, H, X1)
    EOM3B = build_HR_3B(R_unravel, T_unravel, H, X2)
    # perform correction in-loop
    dA_aab, dB_aab, dC_aab, dD_aab, ddA_aab, ddB_aab, ddC_aab, ddD_aab = ccp3_opt_loops.ccp3_opt_loops.eomccp3b_full(
         EOM3B, M3B, L3B, r3_excitations["aab"].T,
         omega, r0,
         H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv, H.a.oo, H.a.vv, H.b.oo, H.b.vv,
         H.aa.voov, H.aa.oooo, H.aa.vvvv, H.ab.ovov, H.ab.vovo, H.ab.oooo, H.ab.vvvv, H.bb.voov,
         d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
    )
    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
        correction_B = 2.0 * dB_aaa + 2.0 * dB_aab
        correction_C = 2.0 * dC_aaa + 2.0 * dC_aab
        correction_D = 2.0 * dD_aaa + 2.0 * dD_aab

        dcorrection_A = 2.0 * ddA_aaa + 2.0 * ddA_aab
        dcorrection_B = 2.0 * ddB_aaa + 2.0 * ddB_aab
        dcorrection_C = 2.0 * ddC_aaa + 2.0 * ddC_aab
        dcorrection_D = 2.0 * ddD_aaa + 2.0 * ddD_aab
    else:
        #### abb correction ####
        # moments and left vector
        M3C = build_M3C_full(T_unravel, H)
        L3C = build_L3C_full(L_unravel, H, X1)
        EOM3C = build_HR_3C(R_unravel, T_unravel, H, X2)
        dA_abb, dB_abb, dC_abb, dD_abb, ddA_abb, ddB_abb, ddC_abb, ddD_abb = ccp3_opt_loops.ccp3_opt_loops.eomccp3c_full(
              EOM3C, M3C, L3C, r3_excitations["abb"].T,
              omega, r0,
              H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv, H.a.oo, H.a.vv, H.b.oo, H.b.vv,
              H.aa.voov, H.ab.ovov, H.ab.vovo, H.ab.oooo, H.ab.vvvv, H.bb.voov, H.bb.oooo, H.bb.vvvv,
              d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
        )
        #### bbb correction ####
        # moments and left vector
        M3D = build_M3D_full(T_unravel, H)
        L3D = build_L3D_full(L_unravel, H, X1)
        EOM3D = build_HR_3D(R_unravel, T_unravel, H, X2)
        dA_bbb, dB_bbb, dC_bbb, dD_bbb, ddA_bbb, ddB_bbb, ddC_bbb, ddD_bbb = ccp3_opt_loops.ccp3_opt_loops.eomccp3d_full(
              EOM3D, M3D, L3D, r3_excitations["bbb"].T,
              omega, r0,
              H0.b.oo, H0.b.vv, H.b.oo, H.b.vv,
              H.bb.voov, H.bb.oooo, H.bb.vvvv,
              d3bbb_o, d3bbb_v,
        )

        correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb
        correction_B = dB_aaa + dB_aab + dB_abb + dB_bbb
        correction_C = dC_aaa + dC_aab + dC_abb + dC_bbb
        correction_D = dD_aaa + dD_aab + dD_abb + dD_bbb

        dcorrection_A = ddA_aaa + ddA_aab + ddA_abb + ddA_bbb
        dcorrection_B = ddB_aaa + ddB_aab + ddB_abb + ddB_bbb
        dcorrection_C = ddC_aaa + ddC_aab + ddC_abb + ddC_bbb
        dcorrection_D = ddD_aaa + ddD_aab + ddD_abb + ddD_bbb

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    energy_A = corr_energy + omega + correction_A
    energy_B = corr_energy + omega + correction_B
    energy_C = corr_energy + omega + correction_C
    energy_D = corr_energy + omega + correction_D

    total_energy_A = system.reference_energy + energy_A
    total_energy_B = system.reference_energy + energy_B
    total_energy_C = system.reference_energy + energy_C
    total_energy_D = system.reference_energy + energy_D

    delta_vee_A = omega + dcorrection_A
    delta_vee_B = omega + dcorrection_B
    delta_vee_C = omega + dcorrection_C
    delta_vee_D = omega + dcorrection_D

    delta_vee_eV_A = hartreetoeV * delta_vee_A
    delta_vee_eV_B = hartreetoeV * delta_vee_B
    delta_vee_eV_C = hartreetoeV * delta_vee_C
    delta_vee_eV_D = hartreetoeV * delta_vee_D

    print('   EOMCC(P;3) Calculation Summary')
    print('   ------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   EOMCC(P) = {:>10.10f}    ω = {:>10.10f}     VEE = {:>10.5f} eV".format(
        system.reference_energy + corr_energy + omega, omega, hartreetoeV * omega))
    print(
        "   EOMCC(P;3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   EOMCC(P;3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   EOMCC(P;3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   EOMCC(P;3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )
    # print(
    #     "   δ-EOMCC(t;3)_A = {:>10.10f}     δ_A = {:>10.10f}     VEE = {:>10.5f} eV".format(
    #         delta_vee_A, dcorrection_A, delta_vee_eV_A
    #     )
    # )
    # print(
    #     "   δ-EOMCC(t;3)_B = {:>10.10f}     δ_B = {:>10.10f}     VEE = {:>10.5f} eV".format(
    #         delta_vee_B, dcorrection_B, delta_vee_eV_B
    #     )
    # )
    # print(
    #     "   δ-EOMCC(t;3)_C = {:>10.10f}     δ_C = {:>10.10f}     VEE = {:>10.5f} eV".format(
    #         delta_vee_C, dcorrection_C, delta_vee_eV_C
    #     )
    # )
    # print(
    #     "   δ-EOMCC(t;3)_D = {:>10.10f}     δ_D = {:>10.10f}     VEE = {:>10.5f} eV\n".format(
    #         delta_vee_D, dcorrection_D, delta_vee_eV_D
    #     )
    # )

    Ecrcc23 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    delta23 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}
    ddelta23 = {"A": dcorrection_A, "B": dcorrection_B, "C": dcorrection_C, "D": dcorrection_D}

    return Ecrcc23, delta23, ddelta23

def calc_eomccp3_full_opt(T, R, L, t3_excitations, r3_excitations, r0, omega, corr_energy, H, H0, system, use_RHF=False):
    """
    Calculate the excited-state EOMCC(P;3) correction to the EOMCC(P) energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    # get L(P)*T(P) intermediates
    # determine whether l3 updates and l3*t3 intermediates should be done. Stupid compatibility with
    # empty sections of t3_excitations or l3_excitations. L3 ordering matches T3 at this point.
    do_l3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    do_t3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    if np.array_equal(t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aaa"] = False
    if np.array_equal(t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aab"] = False
    if np.array_equal(t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["abb"] = False
    if np.array_equal(t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["bbb"] = False
    if np.array_equal(r3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["aaa"] = False
    if np.array_equal(r3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["aab"] = False
    if np.array_equal(r3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["abb"] = False
    if np.array_equal(r3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["bbb"] = False

    # Create intermediates
    X = build_left_ccsdt_p_intermediates(L, r3_excitations, T, t3_excitations, system, do_t3, do_l3, RHF_symmetry=use_RHF)
    XR = get_eomccsd_intermediates(H, R, system)
    XR = get_eomccsdt_intermediates(H, R, T, XR, system)
    XR = add_R3_p_terms(XR, H, R, r3_excitations)

    I2A_vvov = H.aa.vvov + ccpy_einsum("me,abim->abie", H.a.ov, T.aa)
    I2A_vooo = H.aa.vooo - ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
    I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ecjk->mcjk", H.a.ov, T.ab)
    I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeik->amik", H.b.ov, T.ab)
    I2C_vooo = H.bb.vooo - ccpy_einsum("me,cekj->cmkj", H.b.ov, T.bb)

    #### aaa correction ####
    dA_aaa = 0.0
    dB_aaa = 0.0
    dC_aaa = 0.0
    dD_aaa = 0.0
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(j + 1, system.noccupied_alpha):
                EOMM3A = ccp3_full_correction.ccp3_full_correction.build_eom_moments3a_ijk(
                    i + 1, j + 1, k + 1,
                    R.aa,
                    R.aaa, r3_excitations["aaa"],
                    R.aab, r3_excitations["aab"],
                    T.aa,
                    T.aaa, t3_excitations["aaa"],
                    T.aab, t3_excitations["aab"],
                    H.a.oo, H.a.vv,
                    H.aa.oooo, H.aa.vooo, H.aa.oovv,
                    H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvov, H.aa.vvvv.transpose(2, 3, 0, 1),
                    H.ab.voov,
                    XR.a.oo, XR.a.vv,
                    XR.aa.oooo, XR.aa.vooo, XR.aa.oovv,
                    XR.aa.voov, XR.aa.vvov, XR.aa.vvvv.transpose(2, 3, 0, 1),
                    XR.ab.voov,
                )
                M3A = ccp3_full_correction.ccp3_full_correction.build_moments3a_ijk(
                    i + 1, j + 1, k + 1,
                    T.aaa, t3_excitations["aaa"],
                    T.aab, t3_excitations["aab"],
                    T.aa,
                    H.a.oo, H.a.vv.T,
                    H.aa.oovv, I2A_vvov.transpose(3, 0, 1, 2), H.aa.vooo.transpose(1, 0, 2, 3),
                    H.aa.oooo, H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvvv.transpose(3, 2, 1, 0),
                    H.ab.oovv, H.ab.voov.transpose(1, 3, 0, 2),
                )
                L3A = ccp3_full_correction.ccp3_full_correction.build_leftamps3a_ijk(
                    i + 1, j + 1, k + 1,
                    L.a, L.aa,
                    L.aaa, r3_excitations["aaa"],
                    L.aab, r3_excitations["aab"],
                    H.a.ov, H.a.oo, H.a.vv,
                    H.aa.oooo, H.aa.ooov, H.aa.oovv,
                    H.aa.voov, H.aa.vovv, H.aa.vvvv,
                    H.ab.ovvo,
                    X.aa.ooov, X.aa.vovv,
                )
                dA_aaa, dB_aaa, dC_aaa, dD_aaa = ccp3_full_correction.ccp3_full_correction.ccp3a_ijk(
                    dA_aaa, dB_aaa, dC_aaa, dD_aaa,
                    i + 1, j + 1, k + 1, omega,
                    r0 * M3A + EOMM3A, L3A, r3_excitations["aaa"],
                    H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
                    H.aa.voov, H.aa.oooo, H.aa.vvvv,
                    d3aaa_o, d3aaa_v,
                )
    #### aab correction ####
    dA_aab = 0.0
    dB_aab = 0.0
    dC_aab = 0.0
    dD_aab = 0.0
    #EOMM3B = build_HR_3B(R_unravel, T_unravel, H, XR)
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(system.noccupied_beta):
                EOMM3B = ccp3_full_correction.ccp3_full_correction.build_eom_moments3b_ijk(
                    i + 1, j + 1, k + 1,
                    R.aa, R.ab,
                    R.aaa, r3_excitations["aaa"],
                    R.aab, r3_excitations["aab"],
                    R.abb, r3_excitations["abb"],
                    T.aa, T.ab,
                    T.aaa, t3_excitations["aaa"],
                    T.aab, t3_excitations["aab"],
                    T.abb, t3_excitations["abb"],
                    H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                    H.aa.oooo, H.aa.vooo, H.aa.oovv,
                    H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvov.transpose(3, 0, 1, 2), H.aa.vvvv.transpose(2, 3, 0, 1),
                    H.ab.oooo, H.ab.vooo, H.ab.ovoo,
                    H.ab.oovv, H.ab.voov.transpose(1, 3, 0, 2), H.ab.vovo.transpose(1, 2, 0, 3),
                    H.ab.ovov.transpose(0, 3, 1, 2), H.ab.ovvo.transpose(0, 2, 1, 3), H.ab.vvov.transpose(3, 0, 1, 2),
                    H.ab.vvvo.transpose(2, 0, 1, 3), H.ab.vvvv.transpose(3, 2, 1, 0),
                    H.bb.oovv, H.bb.voov.transpose(1, 3, 0, 2),
                    XR.a.oo, XR.a.vv, XR.b.oo, XR.b.vv,
                    XR.aa.oooo, XR.aa.vooo, XR.aa.oovv,
                    XR.aa.voov.transpose(1, 3, 0, 2), XR.aa.vvov.transpose(3, 0, 1, 2), XR.aa.vvvv.transpose(2, 3, 0, 1),
                    XR.ab.oooo, XR.ab.vooo, XR.ab.ovoo,
                    XR.ab.oovv, XR.ab.voov.transpose(1, 3, 0, 2), XR.ab.vovo.transpose(1, 2, 0, 3),
                    XR.ab.ovov.transpose(0, 3, 1, 2), XR.ab.ovvo.transpose(0, 2, 1, 3), XR.ab.vvov.transpose(3, 0, 1, 2),
                    XR.ab.vvvo.transpose(2, 0, 1, 3), XR.ab.vvvv.transpose(3, 2, 1, 0),
                    XR.bb.oovv, XR.bb.voov.transpose(1, 3, 0, 2),
                )
                M3B = ccp3_full_correction.ccp3_full_correction.build_moments3b_ijk(
                    i + 1, j + 1, k + 1,
                    T.aaa, t3_excitations["aaa"],
                    T.aab, t3_excitations["aab"],
                    T.abb, t3_excitations["abb"],
                    T.aa, T.ab,
                    H.a.oo, H.a.vv.T, H.b.oo, H.b.vv.T,
                    H.aa.oovv, H.aa.vvov.transpose(3, 0, 1, 2), I2A_vooo.transpose(1, 0, 2, 3), H.aa.oooo, H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvvv.transpose(3, 2, 1, 0),
                    H.ab.oovv, H.ab.vvov.transpose(3, 0, 1, 2), H.ab.vvvo.transpose(2, 0, 1, 3), I2B_vooo.transpose(1, 0, 2, 3), I2B_ovoo,
                    H.ab.oooo, H.ab.voov.transpose(1, 3, 0, 2), H.ab.vovo.transpose(1, 2, 0, 3), H.ab.ovov.transpose(0, 3, 1, 2), H.ab.ovvo.transpose(0, 2, 1, 3), H.ab.vvvv.transpose(3, 2, 1, 0),
                    H.bb.oovv, H.bb.voov.transpose(1, 3, 0, 2),
                )
                L3B = ccp3_full_correction.ccp3_full_correction.build_leftamps3b_ijk(
                    i + 1, j + 1, k + 1,
                    L.a, L.b, L.aa, L.ab,
                    L.aaa, r3_excitations["aaa"],
                    L.aab, r3_excitations["aab"],
                    L.abb, r3_excitations["abb"],
                    H.a.ov, H.a.oo, H.a.vv,
                    H.b.ov, H.b.oo, H.b.vv,
                    H.aa.oooo, H.aa.ooov, H.aa.oovv,
                    H.aa.voov, H.aa.vovv, H.aa.vvvv,
                    H.ab.oooo, H.ab.ooov, H.ab.oovo,
                    H.ab.oovv,
                    H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
                    H.ab.vovv, H.ab.ovvv, H.ab.vvvv,
                    H.bb.voov,
                    X.aa.ooov, X.aa.vovv,
                    X.ab.ooov, X.ab.oovo, X.ab.vovv, X.ab.ovvv
                )
                dA_aab, dB_aab, dC_aab, dD_aab = ccp3_full_correction.ccp3_full_correction.ccp3b_ijk(
                    dA_aab, dB_aab, dC_aab, dD_aab,
                    i + 1, j + 1, k + 1, omega,
                    r0 * M3B + EOMM3B, L3B, r3_excitations["aab"],
                    H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                    H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                    H.aa.voov, H.aa.oooo, H.aa.vvvv,
                    H.ab.ovov, H.ab.vovo,
                    H.ab.oooo, H.ab.vvvv,
                    H.bb.voov,
                    d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
                )
    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
        correction_B = 2.0 * dB_aaa + 2.0 * dB_aab
        correction_C = 2.0 * dC_aaa + 2.0 * dC_aab
        correction_D = 2.0 * dD_aaa + 2.0 * dD_aab
    else:
        #### abb correction ####
        dA_abb = 0.0
        dB_abb = 0.0
        dC_abb = 0.0
        dD_abb = 0.0
        #EOMM3C = build_HR_3C(R_unravel, T_unravel, H, XR)
        for i in range(system.noccupied_alpha):
            for j in range(system.noccupied_beta):
                for k in range(j + 1, system.noccupied_beta):
                    EOMM3C = ccp3_full_correction.ccp3_full_correction.build_eom_moments3c_ijk(
                        i + 1, j + 1, k + 1,
                        R.ab, R.bb,
                        R.aab, r3_excitations["aab"],
                        R.abb, r3_excitations["abb"],
                        R.bbb, r3_excitations["bbb"],
                        T.ab, T.bb,
                        T.aab, t3_excitations["aab"],
                        T.abb, t3_excitations["abb"],
                        T.bbb, t3_excitations["bbb"],
                        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                        H.aa.oovv, H.aa.voov,
                        H.ab.oooo, H.ab.vooo, H.ab.ovoo,
                        H.ab.oovv, H.ab.voov, H.ab.vovo,
                        H.ab.ovov, H.ab.ovvo, H.ab.vvov,
                        H.ab.vvvo, H.ab.vvvv.transpose(2, 3, 0, 1),
                        H.bb.oooo, H.bb.vooo, H.bb.oovv,
                        H.bb.voov, H.bb.vvov, H.bb.vvvv.transpose(2, 3, 0, 1),
                        XR.a.oo, XR.a.vv, XR.b.oo, XR.b.vv,
                        XR.aa.oovv, XR.aa.voov,
                        XR.ab.oooo, XR.ab.vooo, XR.ab.ovoo,
                        XR.ab.oovv, XR.ab.voov, XR.ab.vovo,
                        XR.ab.ovov, XR.ab.ovvo, XR.ab.vvov,
                        XR.ab.vvvo, XR.ab.vvvv.transpose(2, 3, 0, 1),
                        XR.bb.oooo, XR.bb.vooo, XR.bb.oovv,
                        XR.bb.voov, XR.bb.vvov, XR.bb.vvvv.transpose(2, 3, 0, 1),
                    )
                    M3C = ccp3_full_correction.ccp3_full_correction.build_moments3c_ijk(
                        i + 1, j + 1, k + 1,
                        T.aab, t3_excitations["aab"],
                        T.abb, t3_excitations["abb"],
                        T.bbb, t3_excitations["bbb"],
                        T.ab, T.bb,
                        H.a.oo, H.a.vv.T, H.b.oo, H.b.vv.T,
                        H.aa.oovv, H.aa.voov.transpose(1, 3, 0, 2),
                        H.ab.oovv, I2B_vooo.transpose(1, 0, 2, 3), I2B_ovoo, H.ab.vvov.transpose(3, 0, 1, 2), H.ab.vvvo.transpose(2, 0, 1, 3), H.ab.oooo,
                        H.ab.voov.transpose(1, 3, 0, 2), H.ab.vovo.transpose(1, 2, 0, 3), H.ab.ovov.transpose(0, 3, 1, 2), H.ab.ovvo.transpose(0, 2, 1, 3), H.ab.vvvv.transpose(2, 3, 0, 1),
                        H.bb.oovv, I2C_vooo.transpose(1, 0, 2, 3), H.bb.vvov.transpose(3, 0, 1, 2), H.bb.oooo, H.bb.voov.transpose(1, 3, 0, 2), H.bb.vvvv.transpose(3, 2, 1, 0),
                    )
                    L3C = ccp3_full_correction.ccp3_full_correction.build_leftamps3c_ijk(
                        i + 1, j + 1, k + 1,
                        L.a, L.b, L.ab, L.bb,
                        L.aab, r3_excitations["aab"],
                        L.abb, r3_excitations["abb"],
                        L.bbb, r3_excitations["bbb"],
                        H.a.ov, H.a.oo, H.a.vv,
                        H.b.ov, H.b.oo, H.b.vv,
                        H.aa.voov,
                        H.ab.oooo, H.ab.ooov, H.ab.oovo,
                        H.ab.oovv,
                        H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
                        H.ab.vovv, H.ab.ovvv, H.ab.vvvv,
                        H.bb.oooo, H.bb.ooov, H.bb.oovv,
                        H.bb.voov, H.bb.vovv, H.bb.vvvv,
                        X.ab.ooov, X.ab.oovo, X.ab.vovv, X.ab.ovvv,
                        X.bb.ooov, X.bb.vovv,
                    )
                    dA_abb, dB_abb, dC_abb, dD_abb = ccp3_full_correction.ccp3_full_correction.ccp3c_ijk(
                        dA_abb, dB_abb, dC_abb, dD_abb,
                        i + 1, j + 1, k + 1, omega,
                        r0 * M3C + EOMM3C, L3C, r3_excitations["abb"],
                        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                        H.aa.voov,
                        H.ab.ovov, H.ab.vovo,
                        H.ab.oooo, H.ab.vvvv,
                        H.bb.voov, H.bb.oooo, H.bb.vvvv,
                        d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
                    )
        #### bbb correction ####
        dA_bbb = 0.0
        dB_bbb = 0.0
        dC_bbb = 0.0
        dD_bbb = 0.0
        #EOMM3D = build_HR_3D(R_unravel, T_unravel, H, XR)
        for i in range(system.noccupied_beta):
            for j in range(i + 1, system.noccupied_beta):
                for k in range(j + 1, system.noccupied_beta):
                    EOMM3D = ccp3_full_correction.ccp3_full_correction.build_eom_moments3d_ijk(
                        i + 1, j + 1, k + 1,
                        R.bb,
                        R.abb, r3_excitations["abb"],
                        R.bbb, r3_excitations["bbb"],
                        T.bb,
                        T.abb, t3_excitations["abb"],
                        T.bbb, t3_excitations["bbb"],
                        H.b.oo, H.b.vv,
                        H.bb.oooo, H.bb.vooo, H.bb.oovv,
                        H.bb.voov, H.bb.vvov, H.bb.vvvv.transpose(2, 3, 0, 1),
                        H.ab.ovvo,
                        XR.b.oo, XR.b.vv,
                        XR.bb.oooo, XR.bb.vooo, XR.bb.oovv,
                        XR.bb.voov, XR.bb.vvov, XR.bb.vvvv.transpose(2, 3, 0, 1),
                        XR.ab.ovvo,
                    )
                    M3D = ccp3_full_correction.ccp3_full_correction.build_moments3d_ijk(
                        i + 1, j + 1, k + 1,
                        T.abb, t3_excitations["abb"],
                        T.bbb, t3_excitations["bbb"],
                        T.bb,
                        H.b.oo, H.b.vv.T,
                        H.bb.oovv, H.bb.vvov.transpose(3, 0, 1, 2), I2C_vooo.transpose(1, 0, 2, 3),
                        H.bb.oooo, H.bb.voov.transpose(1, 3, 0, 2), H.bb.vvvv.transpose(3, 2, 1, 0),
                        H.ab.oovv, H.ab.ovvo.transpose(0, 2, 1, 3),
                    )
                    L3D = ccp3_full_correction.ccp3_full_correction.build_leftamps3d_ijk(
                        i + 1, j + 1, k + 1,
                        L.b, L.bb,
                        L.abb, r3_excitations["abb"],
                        L.bbb, r3_excitations["bbb"],
                        H.b.ov, H.b.oo, H.b.vv,
                        H.ab.voov,
                        H.bb.oooo, H.bb.ooov, H.bb.oovv,
                        H.bb.voov, H.bb.vovv, H.bb.vvvv,
                        X.bb.ooov, X.bb.vovv,
                    )
                    dA_bbb, dB_bbb, dC_bbb, dD_bbb = ccp3_full_correction.ccp3_full_correction.ccp3d_ijk(
                        dA_bbb, dB_bbb, dC_bbb, dD_bbb,
                        i + 1, j + 1, k + 1, omega,
                        r0 * M3D + EOMM3D, L3D, r3_excitations["bbb"],
                        H0.b.oo, H0.b.vv, H.b.oo, H.b.vv,
                        H.bb.voov, H.bb.oooo, H.bb.vvvv,
                        d3bbb_o, d3bbb_v,
                    )
        correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb
        correction_B = dB_aaa + dB_aab + dB_abb + dB_bbb
        correction_C = dC_aaa + dC_aab + dC_abb + dC_bbb
        correction_D = dD_aaa + dD_aab + dD_abb + dD_bbb

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    energy_A = corr_energy + omega + correction_A
    energy_B = corr_energy + omega + correction_B
    energy_C = corr_energy + omega + correction_C
    energy_D = corr_energy + omega + correction_D

    total_energy_A = system.reference_energy + energy_A
    total_energy_B = system.reference_energy + energy_B
    total_energy_C = system.reference_energy + energy_C
    total_energy_D = system.reference_energy + energy_D

    print('   EOMCC(P;3) Calculation Summary')
    print('   ------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   EOMCC(P) = {:>10.10f}    ω = {:>10.10f}     VEE = {:>10.5f} eV".format(
        system.reference_energy + corr_energy + omega, omega, hartreetoeV * omega))
    print(
        "   EOMCC(P;3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   EOMCC(P;3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   EOMCC(P;3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   EOMCC(P;3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )
    Eccp3 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    deltap3 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}
    return Eccp3, deltap3

def calc_eomccp3_full_with_selection(T, R, L, t3_excitations, r3_excitations, r0, omega, corr_energy, H, H0, system, num_add, use_RHF=False, min_thresh=0.0, buffer_factor=2):
    """
    Calculate the ground-state CC(P;3) correction to the CC(P) energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    # initialize empty moments vector and triples list
    num_add = int(num_add)
    moments = np.zeros(buffer_factor * num_add)
    triples_list = np.zeros((buffer_factor * num_add, 6), dtype=np.int32)

    # get L(P)*T(P) intermediates
    # determine whether l3 updates and l3*t3 intermediates should be done. Stupid compatibility with
    # empty sections of t3_excitations or l3_excitations. L3 ordering matches T3 at this point.
    do_l3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    do_t3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    if np.array_equal(t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aaa"] = False
    if np.array_equal(t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["aab"] = False
    if np.array_equal(t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["abb"] = False
    if np.array_equal(t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_t3["bbb"] = False
    if np.array_equal(r3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["aaa"] = False
    if np.array_equal(r3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["aab"] = False
    if np.array_equal(r3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["abb"] = False
    if np.array_equal(r3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_l3["bbb"] = False
    X1 = build_left_ccsdt_p_intermediates(L, r3_excitations, T, t3_excitations, system, do_t3, do_l3, RHF_symmetry=use_RHF)

    # Form H(2)*(R1 + R2) intermediates
    Xtemp = get_eomccsd_intermediates(H, R, system)
    X2 = get_eomccsdt_intermediates(H, R, T, Xtemp, system)
    X2 = add_R3_p_terms(X2, H, R, r3_excitations)

    # unravel triples vector into t3(abcijk), r3(abcijk), and l3(abcijk)
    T_unravel = unravel_triples_amplitudes(T, t3_excitations, system, do_t3)
    R_unravel = unravel_triples_amplitudes(R, r3_excitations, system, do_l3)
    L_unravel = unravel_triples_amplitudes(L, r3_excitations, system, do_l3)

    nfill = 1
    #### aaa correction ####
    M3A = build_M3A_full(T_unravel, H)
    L3A = build_L3A_full(L_unravel, H, X1)
    EOM3A = build_HR_3A(R_unravel, T_unravel, H, X2)
    dA_aaa, dB_aaa, dC_aaa, dD_aaa, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3_adaptive_loops.eomccp3a_full_with_selection_opt(
        moments,
        triples_list,
        nfill,
        EOM3A, M3A, L3A,
        r3_excitations['aaa'].T,
        omega, r0,
        H0.a.oo, H0.a.vv,
        H.a.oo, H.a.vv, H.aa.voov, H.aa.oooo,
        H.aa.vvvv,
        d3aaa_o, d3aaa_v,
        num_add, min_thresh, buffer_factor,
        )

    #### aab correction ####
    M3B = build_M3B_full(T_unravel, H)
    L3B = build_L3B_full(L_unravel, H, X1)
    EOM3B = build_HR_3B(R_unravel, T_unravel, H, X2)
    dA_aab, dB_aab, dC_aab, dD_aab, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3_adaptive_loops.eomccp3b_full_with_selection_opt(
        moments,
        triples_list,
        nfill,
        EOM3B, M3B, L3B,
        r3_excitations['aab'].T,
        omega, r0,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H.aa.voov, H.aa.oooo, H.aa.vvvv, H.ab.ovov,
        H.ab.vovo, H.ab.oooo, H.ab.vvvv, H.bb.voov,
        d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
        num_add, min_thresh, buffer_factor,
    )

    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
        correction_B = 2.0 * dB_aaa + 2.0 * dB_aab
        correction_C = 2.0 * dC_aaa + 2.0 * dC_aab
        correction_D = 2.0 * dD_aaa + 2.0 * dD_aab
    else:
        #### abb correction ####
        M3C = build_M3C_full(T_unravel, H)
        L3C = build_L3C_full(L_unravel, H, X1)
        EOM3C = build_HR_3C(R_unravel, T_unravel, H, X2)
        dA_abb, dB_abb, dC_abb, dD_abb, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3_adaptive_loops.eomccp3c_full_with_selection_opt(
            moments,
            triples_list,
            nfill,
            EOM3C, M3C, L3C,
            r3_excitations['abb'].T,
            omega, r0,
            H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
            H.aa.voov, H.ab.ovov, H.ab.vovo, H.ab.oooo,
            H.ab.vvvv, H.bb.voov, H.bb.oooo, H.bb.vvvv,
            d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
            num_add, min_thresh, buffer_factor,
        )
        #### bbb correction ####
        M3D = build_M3D_full(T_unravel, H)
        L3D = build_L3D_full(L_unravel, H, X1)
        EOM3D = build_HR_3D(R_unravel, T_unravel, H, X2)
        dA_bbb, dB_bbb, dC_bbb, dD_bbb, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3_adaptive_loops.eomccp3d_full_with_selection_opt(
            moments,
            triples_list,
            nfill,
            EOM3D, M3D, L3D,
            r3_excitations['bbb'].T,
            omega, r0,
            H0.b.oo, H0.b.vv,
            H.b.oo, H.b.vv, H.bb.voov, H.bb.oooo, H.bb.vvvv,
            d3bbb_o, d3bbb_v,
            num_add, min_thresh, buffer_factor,
        )

        correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb
        correction_B = dB_aaa + dB_aab + dB_abb + dB_bbb
        correction_C = dC_aaa + dC_aab + dC_abb + dC_bbb
        correction_D = dD_aaa + dD_aab + dD_abb + dD_bbb

    # Important: perform a final sort of the excitations and moments, returning the first num_add elements only
    idx = np.argsort(np.abs(moments))
    triples_list = triples_list[idx[::-1], :]
    triples_list = triples_list[:num_add, :]

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    # print the results
    energy_A = corr_energy + omega + correction_A
    energy_B = corr_energy + omega + correction_B
    energy_C = corr_energy + omega + correction_C
    energy_D = corr_energy + omega + correction_D

    total_energy_A = system.reference_energy + energy_A
    total_energy_B = system.reference_energy + energy_B
    total_energy_C = system.reference_energy + energy_C
    total_energy_D = system.reference_energy + energy_D

    print('   EOMCC(P;3) Calculation Summary')
    print('   ------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   EOMCC(P) = {:>10.10f}    ω = {:>10.10f}     VEE = {:>10.5f} eV".format(
        system.reference_energy + corr_energy + omega, omega, hartreetoeV * omega))
    print(
        "   EOMCC(P;3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   EOMCC(P;3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   EOMCC(P;3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   EOMCC(P;3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )
    print(
        "   Selected moments account for {:>5.2f}% of the total EOMCC(P;3)_D correction\n".format(
            sum(moments) / correction_D * 100
        )
    )

    Eccp3 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    deltap3 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}

    return Eccp3["D"], triples_list

def build_M3A_full(T, H):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3))_C|0>.
    """
    # <ijkabc | H(2) | 0 > + (VT3)_C intermediates
    # Recall that we are using HBar CCSDT, so the vooo and vvov parts have T3 in it already!
    I2A_vvov = H.aa.vvov + ccpy_einsum("me,abim->abie", H.a.ov, T.aa)
    # MM(2,3)A
    M3A = -0.25 * ccpy_einsum("amij,bcmk->abcijk", H.aa.vooo, T.aa)
    M3A += 0.25 * ccpy_einsum("abie,ecjk->abcijk", I2A_vvov, T.aa)
    # (HBar*T3)_C
    M3A -= (1.0 / 12.0) * ccpy_einsum("mk,abcijm->abcijk", H.a.oo, T.aaa)
    M3A += (1.0 / 12.0) * ccpy_einsum("ce,abeijk->abcijk", H.a.vv, T.aaa)
    M3A += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa)
    M3A += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa)
    M3A += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa)
    M3A += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab)
    # antisymmetrize
    M3A -= np.transpose(M3A, (0, 1, 2, 3, 5, 4)) # (jk)
    M3A -= np.transpose(M3A, (0, 1, 2, 4, 3, 5)) + np.transpose(M3A, (0, 1, 2, 5, 4, 3)) # (i/jk)
    M3A -= np.transpose(M3A, (0, 2, 1, 3, 4, 5)) # (bc)
    M3A -= np.transpose(M3A, (2, 1, 0, 3, 4, 5)) + np.transpose(M3A, (1, 0, 2, 3, 4, 5)) # (a/bc)
    return M3A

def build_M3B_full(T, H):
    """
    Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N e^(T1+T2+T3))_C|0>.
    """
    # <ijk~abc~ | H(2) | 0 > + (VT3)_C intermediates
    # Recall that we are using HBar CCSDT, so the vooo and vvov parts have T3 in it already!
    I2A_vooo = H.aa.vooo - ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
    I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ecjk->mcjk", H.a.ov, T.ab)
    I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeik->amik", H.b.ov, T.ab)
    # MM(2,3)B
    M3B = 0.5 * ccpy_einsum("bcek,aeij->abcijk", H.ab.vvvo, T.aa)
    M3B -= 0.5 * ccpy_einsum("mcjk,abim->abcijk", I2B_ovoo, T.aa)
    M3B += ccpy_einsum("acie,bejk->abcijk", H.ab.vvov, T.ab)
    M3B -= ccpy_einsum("amik,bcjm->abcijk", I2B_vooo, T.ab)
    M3B += 0.5 * ccpy_einsum("abie,ecjk->abcijk", H.aa.vvov, T.ab)
    M3B -= 0.5 * ccpy_einsum("amij,bcmk->abcijk", I2A_vooo, T.ab)
    # (HBar*T3)_C
    M3B -= 0.5 * ccpy_einsum("mi,abcmjk->abcijk", H.a.oo, T.aab)
    M3B -= 0.25 * ccpy_einsum("mk,abcijm->abcijk", H.b.oo, T.aab)
    M3B += 0.5 * ccpy_einsum("ae,ebcijk->abcijk", H.a.vv, T.aab)
    M3B += 0.25 * ccpy_einsum("ce,abeijk->abcijk", H.b.vv, T.aab)
    M3B += 0.125 * ccpy_einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aab)
    M3B += 0.5 * ccpy_einsum("mnjk,abcimn->abcijk", H.ab.oooo, T.aab)
    M3B += 0.125 * ccpy_einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aab)
    M3B += 0.5 * ccpy_einsum("bcef,aefijk->abcijk", H.ab.vvvv, T.aab)
    M3B += ccpy_einsum("amie,ebcmjk->abcijk", H.aa.voov, T.aab)
    M3B += ccpy_einsum("amie,becjmk->abcijk", H.ab.voov, T.abb)
    M3B += 0.25 * ccpy_einsum("mcek,abeijm->abcijk", H.ab.ovvo, T.aaa)
    M3B += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.bb.voov, T.aab)
    M3B -= 0.5 * ccpy_einsum("amek,ebcijm->abcijk", H.ab.vovo, T.aab)
    M3B -= 0.5 * ccpy_einsum("mcie,abemjk->abcijk", H.ab.ovov, T.aab)
    # antisymmetrize
    M3B -= np.transpose(M3B, (1, 0, 2, 3, 4, 5))
    M3B -= np.transpose(M3B, (0, 1, 2, 4, 3, 5))
    return M3B

def build_M3C_full(T, H):
    """
    Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N e^(T1+T2+T3))_C|0>.
    """
    # <ij~k~ab~c~ | H(2) | 0 > + (VT3)_C intermediates
    # Recall that we are using HBar CCSDT, so the vooo and vvov parts have T3 in it already!
    I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ebij->mbij", H.a.ov, T.ab)
    I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
    I2C_vooo = H.bb.vooo - ccpy_einsum("me,cekj->cmkj", H.b.ov, T.bb)
    # MM(2,3)C
    M3C = 0.5 * ccpy_einsum("abie,ecjk->abcijk", H.ab.vvov, T.bb)
    M3C -= 0.5 * ccpy_einsum("amij,bcmk->abcijk", I2B_vooo, T.bb)
    M3C += 0.5 * ccpy_einsum("cbke,aeij->abcijk", H.bb.vvov, T.ab)
    M3C -= 0.5 * ccpy_einsum("cmkj,abim->abcijk", I2C_vooo, T.ab)
    M3C += ccpy_einsum("abej,ecik->abcijk", H.ab.vvvo, T.ab)
    M3C -= ccpy_einsum("mbij,acmk->abcijk", I2B_ovoo, T.ab)
    # (HBar*T3)_C
    M3C -= 0.25 * ccpy_einsum("mi,abcmjk->abcijk", H.a.oo, T.abb)
    M3C -= 0.5 * ccpy_einsum("mj,abcimk->abcijk", H.b.oo, T.abb)
    M3C += 0.25 * ccpy_einsum("ae,ebcijk->abcijk", H.a.vv, T.abb)
    M3C += 0.5 * ccpy_einsum("be,aecijk->abcijk", H.b.vv, T.abb)
    M3C += 0.125 * ccpy_einsum("mnjk,abcimn->abcijk", H.bb.oooo, T.abb)
    M3C += 0.5 * ccpy_einsum("mnij,abcmnk->abcijk", H.ab.oooo, T.abb)
    M3C += 0.125 * ccpy_einsum("bcef,aefijk->abcijk", H.bb.vvvv, T.abb)
    M3C += 0.5 * ccpy_einsum("abef,efcijk->abcijk", H.ab.vvvv, T.abb)
    M3C += 0.25 * ccpy_einsum("amie,ebcmjk->abcijk", H.aa.voov, T.abb)
    M3C += 0.25 * ccpy_einsum("amie,ebcmjk->abcijk", H.ab.voov, T.bbb)
    M3C += ccpy_einsum("mbej,aecimk->abcijk", H.ab.ovvo, T.aab)
    M3C += ccpy_einsum("bmje,aecimk->abcijk", H.bb.voov, T.abb)
    M3C -= 0.5 * ccpy_einsum("mbie,aecmjk->abcijk", H.ab.ovov, T.abb)
    M3C -= 0.5 * ccpy_einsum("amej,ebcimk->abcijk", H.ab.vovo, T.abb)
    # antisymmetrize
    M3C -= np.transpose(M3C, (0, 2, 1, 3, 4, 5))
    M3C -= np.transpose(M3C, (0, 1, 2, 3, 5, 4))
    return M3C

def build_M3D_full(T, H):
    """
    Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N e^(T1+T2+T3))_C|0>.
    """
    #  <i~j~k~a~b~c~ | H(2) | 0 > + (VT3)_C intermediates
    # Recall that we are using HBar CCSDT, so the vooo and vvov parts have T3 in it already!
    I2C_vooo = H.bb.vooo - ccpy_einsum("me,aeij->amij", H.b.ov, T.bb)
    # MM(2,3)D
    M3D = -0.25 * ccpy_einsum("amij,bcmk->abcijk", I2C_vooo, T.bb)
    M3D += 0.25 * ccpy_einsum("abie,ecjk->abcijk", H.bb.vvov, T.bb)
    # (HBar*T3)_C
    M3D -= (1.0 / 12.0) * ccpy_einsum("mk,abcijm->abcijk", H.b.oo, T.bbb)
    M3D += (1.0 / 12.0) * ccpy_einsum("ce,abeijk->abcijk", H.b.vv, T.bbb)
    M3D += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", H.bb.oooo, T.bbb)
    M3D += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", H.bb.vvvv, T.bbb)
    M3D += 0.25 * ccpy_einsum("maei,ebcmjk->abcijk", H.ab.ovvo, T.abb)
    M3D += 0.25 * ccpy_einsum("amie,ebcmjk->abcijk", H.bb.voov, T.bbb)
    # antisymmetrize
    M3D -= np.transpose(M3D, (0, 1, 2, 3, 5, 4)) # (jk)
    M3D -= np.transpose(M3D, (0, 1, 2, 4, 3, 5)) + np.transpose(M3D, (0, 1, 2, 5, 4, 3)) # (i/jk)
    M3D -= np.transpose(M3D, (0, 2, 1, 3, 4, 5)) # (bc)
    M3D -= np.transpose(M3D, (2, 1, 0, 3, 4, 5)) + np.transpose(M3D, (1, 0, 2, 3, 4, 5)) # (a/bc)
    return M3D

def build_L3A_full(L, H, X):
    # < 0 | L1 * H(2) | ijkabc >
    L3A = (9.0 / 36.0) * ccpy_einsum("ai,jkbc->abcijk", L.a, H.aa.oovv)
    # < 0 | L2 * H(2) | ijkabc >
    L3A += (9.0 / 36.0) * ccpy_einsum("bcjk,ia->abcijk", L.aa, H.a.ov)
    L3A += (9.0 / 36.0) * ccpy_einsum("ebij,ekac->abcijk", L.aa, H.aa.vovv)
    L3A -= (9.0 / 36.0) * ccpy_einsum("abmj,ikmc->abcijk", L.aa, H.aa.ooov)
    # < 0 | L3 * H(2) | ijkabc >
    L3A += (3.0 / 36.0) * ccpy_einsum("ea,ebcijk->abcijk", H.a.vv, L.aaa)
    L3A -= (3.0 / 36.0) * ccpy_einsum("im,abcmjk->abcijk", H.a.oo, L.aaa)
    L3A += (9.0 / 36.0) * ccpy_einsum("eima,ebcmjk->abcijk", H.aa.voov, L.aaa)
    L3A += (9.0 / 36.0) * ccpy_einsum("ieam,bcejkm->abcijk", H.ab.ovvo, L.aab)
    L3A += (3.0 / 72.0) * ccpy_einsum("ijmn,abcmnk->abcijk", H.aa.oooo, L.aaa)
    L3A += (3.0 / 72.0) * ccpy_einsum("efab,efcijk->abcijk", H.aa.vvvv, L.aaa)
    L3A += (9.0 / 36.0) * ccpy_einsum("ijeb,ekac->abcijk", H.aa.oovv, X.aa.vovv)
    L3A -= (9.0 / 36.0) * ccpy_einsum("mjab,ikmc->abcijk", H.aa.oovv, X.aa.ooov)
    # antisymmetrize
    L3A -= np.transpose(L3A, (0, 1, 2, 3, 5, 4)) # (jk)
    L3A -= np.transpose(L3A, (0, 1, 2, 4, 3, 5)) + np.transpose(L3A, (0, 1, 2, 5, 4, 3)) # (i/jk)
    L3A -= np.transpose(L3A, (0, 2, 1, 3, 4, 5)) # (bc)
    L3A -= np.transpose(L3A, (2, 1, 0, 3, 4, 5)) + np.transpose(L3A, (1, 0, 2, 3, 4, 5)) # (a/bc)
    return L3A

def build_L3B_full(L, H, X):
    # < 0 | L1 * H(2) | ijk~abc~ >
    L3B = ccpy_einsum("ai,jkbc->abcijk", L.a, H.ab.oovv)
    L3B += 0.25 * ccpy_einsum("ck,ijab->abcijk", L.b, H.aa.oovv)
    # < 0 | L2 * H(2) | ijk~abc~ >
    L3B += ccpy_einsum("bcjk,ia->abcijk", L.ab, H.a.ov)
    L3B += 0.25 * ccpy_einsum("abij,kc->abcijk", L.aa, H.b.ov)
    L3B += 0.5 * ccpy_einsum("ekbc,aeij->abcijk", H.ab.vovv, L.aa)
    L3B -= 0.5 * ccpy_einsum("jkmc,abim->abcijk", H.ab.ooov, L.aa)
    L3B += ccpy_einsum("ieac,bejk->abcijk", H.ab.ovvv, L.ab)
    L3B -= ccpy_einsum("ikam,bcjm->abcijk", H.ab.oovo, L.ab)
    L3B += 0.5 * ccpy_einsum("eiba,ecjk->abcijk", H.aa.vovv, L.ab)
    L3B -= 0.5 * ccpy_einsum("jima,bcmk->abcijk", H.aa.ooov, L.ab)
    L3B += 0.5 * ccpy_einsum("ekbc,ijae->abcijk", X.ab.vovv, H.aa.oovv)
    L3B -= 0.5 * ccpy_einsum("jkmc,imab->abcijk", X.ab.ooov, H.aa.oovv)
    L3B += ccpy_einsum("ieac,jkbe->abcijk", X.ab.ovvv, H.ab.oovv)
    L3B -= ccpy_einsum("ikam,jmbc->abcijk", X.ab.oovo, H.ab.oovv)
    L3B += 0.5 * ccpy_einsum("eiba,jkec->abcijk", X.aa.vovv, H.ab.oovv)
    L3B -= 0.5 * ccpy_einsum("jima,mkbc->abcijk", X.aa.ooov, H.ab.oovv)
    # < 0 | L3 * H(2) | ijk~abc~ >
    L3B -= 0.5 * ccpy_einsum("im,abcmjk->abcijk", H.a.oo, L.aab)
    L3B -= 0.25 * ccpy_einsum("km,abcijm->abcijk", H.b.oo, L.aab)
    L3B += 0.5 * ccpy_einsum("ea,ebcijk->abcijk", H.a.vv, L.aab)
    L3B += 0.25 * ccpy_einsum("ec,abeijk->abcijk", H.b.vv, L.aab)
    L3B += 0.125 * ccpy_einsum("ijmn,abcmnk->abcijk", H.aa.oooo, L.aab)
    L3B += 0.5 * ccpy_einsum("jkmn,abcimn->abcijk", H.ab.oooo, L.aab)
    L3B += 0.125 * ccpy_einsum("efab,efcijk->abcijk", H.aa.vvvv, L.aab)
    L3B += 0.5 * ccpy_einsum("efbc,aefijk->abcijk", H.ab.vvvv, L.aab)
    L3B += ccpy_einsum("eima,ebcmjk->abcijk", H.aa.voov, L.aab)
    L3B += ccpy_einsum("ieam,becjmk->abcijk", H.ab.ovvo, L.abb)
    L3B += 0.25 * ccpy_einsum("ekmc,abeijm->abcijk", H.ab.voov, L.aaa)
    L3B += 0.25 * ccpy_einsum("ekmc,abeijm->abcijk", H.bb.voov, L.aab)
    L3B -= 0.5 * ccpy_einsum("ekam,ebcijm->abcijk", H.ab.vovo, L.aab)
    L3B -= 0.5 * ccpy_einsum("iemc,abemjk->abcijk", H.ab.ovov, L.aab)
    # antisymmetrize
    L3B -= np.transpose(L3B, (1, 0, 2, 3, 4, 5))
    L3B -= np.transpose(L3B, (0, 1, 2, 4, 3, 5))
    return L3B

def build_L3C_full(L, H, X):
    # < 0 | L1 * H(2) | ijk~abc~ >
    L3C = ccpy_einsum("ai,kjcb->cbakji", L.b, H.ab.oovv)
    L3C += 0.25 * ccpy_einsum("ck,ijab->cbakji", L.a, H.bb.oovv)
    # < 0 | L2 * H(2) | ijk~abc~ >
    L3C += ccpy_einsum("cbkj,ia->cbakji", L.ab, H.b.ov)
    L3C += 0.25 * ccpy_einsum("abij,kc->cbakji", L.bb, H.a.ov)
    L3C += 0.5 * ccpy_einsum("kecb,aeij->cbakji", H.ab.ovvv, L.bb)
    L3C -= 0.5 * ccpy_einsum("kjcm,abim->cbakji", H.ab.oovo, L.bb)
    L3C += ccpy_einsum("eica,ebkj->cbakji", H.ab.vovv, L.ab)
    L3C -= ccpy_einsum("kima,cbmj->cbakji", H.ab.ooov, L.ab)
    L3C += 0.5 * ccpy_einsum("eiba,cekj->cbakji", H.bb.vovv, L.ab)
    L3C -= 0.5 * ccpy_einsum("jima,cbkm->cbakji", H.bb.ooov, L.ab)
    L3C += 0.5 * ccpy_einsum("kecb,ijae->cbakji", X.ab.ovvv, H.bb.oovv)
    L3C -= 0.5 * ccpy_einsum("kjcm,imab->cbakji", X.ab.oovo, H.bb.oovv)
    L3C += ccpy_einsum("eica,kjeb->cbakji", X.ab.vovv, H.ab.oovv)
    L3C -= ccpy_einsum("kima,mjcb->cbakji", X.ab.ooov, H.ab.oovv)
    L3C += 0.5 * ccpy_einsum("eiba,kjce->cbakji", X.bb.vovv, H.ab.oovv)
    L3C -= 0.5 * ccpy_einsum("jima,kmcb->cbakji", X.bb.ooov, H.ab.oovv)
    # < 0 | L3 * H(2) | ijk~abc~ >
    L3C -= 0.5 * ccpy_einsum("im,cbakjm->cbakji", H.b.oo, L.abb)
    L3C -= 0.25 * ccpy_einsum("km,cbamji->cbakji", H.a.oo, L.abb)
    L3C += 0.5 * ccpy_einsum("ea,cbekji->cbakji", H.b.vv, L.abb)
    L3C += 0.25 * ccpy_einsum("ec,ebakji->cbakji", H.a.vv, L.abb)
    L3C += 0.125 * ccpy_einsum("ijmn,cbaknm->cbakji", H.bb.oooo, L.abb)
    L3C += 0.5 * ccpy_einsum("kjnm,cbanmi->cbakji", H.ab.oooo, L.abb)
    L3C += 0.125 * ccpy_einsum("efab,cfekji->cbakji", H.bb.vvvv, L.abb)
    L3C += 0.5 * ccpy_einsum("fecb,feakji->cbakji", H.ab.vvvv, L.abb)
    L3C += ccpy_einsum("eima,cbekjm->cbakji", H.bb.voov, L.abb)
    L3C += ccpy_einsum("eima,cebkmj->cbakji", H.ab.voov, L.aab)
    L3C += 0.25 * ccpy_einsum("kecm,abeijm->cbakji", H.ab.ovvo, L.bbb)
    L3C += 0.25 * ccpy_einsum("ekmc,ebamji->cbakji", H.aa.voov, L.abb)
    L3C -= 0.5 * ccpy_einsum("kema,cbemji->cbakji", H.ab.ovov, L.abb)
    L3C -= 0.5 * ccpy_einsum("eicm,ebakjm->cbakji", H.ab.vovo, L.abb)
    # antisymmetrize
    L3C -= np.transpose(L3C, (0, 2, 1, 3, 4, 5))
    L3C -= np.transpose(L3C, (0, 1, 2, 3, 5, 4))
    return L3C

def build_L3D_full(L, H, X):
    # < 0 | L1 * H(2) | ijkabc >
    L3D = (9.0 / 36.0) * ccpy_einsum("ai,jkbc->abcijk", L.b, H.bb.oovv)
    # < 0 | L2 * H(2) | ijkabc >
    L3D += (9.0 / 36.0) * ccpy_einsum("bcjk,ia->abcijk", L.bb, H.b.ov)
    L3D += (9.0 / 36.0) * ccpy_einsum("ebij,ekac->abcijk", L.bb, H.bb.vovv)
    L3D -= (9.0 / 36.0) * ccpy_einsum("abmj,ikmc->abcijk", L.bb, H.bb.ooov)
    # < 0 | L3 * H(2) | ijkabc >
    L3D += (3.0 / 36.0) * ccpy_einsum("ea,ebcijk->abcijk", H.b.vv, L.bbb)
    L3D -= (3.0 / 36.0) * ccpy_einsum("im,abcmjk->abcijk", H.b.oo, L.bbb)
    L3D += (9.0 / 36.0) * ccpy_einsum("eima,ebcmjk->abcijk", H.bb.voov, L.bbb)
    L3D += (9.0 / 36.0) * ccpy_einsum("eima,ecbmkj->abcijk", H.ab.voov, L.abb)
    L3D += (3.0 / 72.0) * ccpy_einsum("ijmn,abcmnk->abcijk", H.bb.oooo, L.bbb)
    L3D += (3.0 / 72.0) * ccpy_einsum("efab,efcijk->abcijk", H.bb.vvvv, L.bbb)
    L3D += (9.0 / 36.0) * ccpy_einsum("ijeb,ekac->abcijk", H.bb.oovv, X.bb.vovv)
    L3D -= (9.0 / 36.0) * ccpy_einsum("mjab,ikmc->abcijk", H.bb.oovv, X.bb.ooov)
    # antisymmetrize
    L3D -= np.transpose(L3D, (0, 1, 2, 3, 5, 4)) # (jk)
    L3D -= np.transpose(L3D, (0, 1, 2, 4, 3, 5)) + np.transpose(L3D, (0, 1, 2, 5, 4, 3)) # (i/jk)
    L3D -= np.transpose(L3D, (0, 2, 1, 3, 4, 5)) # (bc)
    L3D -= np.transpose(L3D, (2, 1, 0, 3, 4, 5)) + np.transpose(L3D, (1, 0, 2, 3, 4, 5)) # (a/bc)
    return L3D


# def calc_ccpert3(T, corr_energy, H, system, pspace, use_RHF=False):
#     """
#     Calculate the ground-state CC(P;Q)_(T) correction to the CC(P) energy associated
#     with the P space `pspace`.
#     """
#     t_start = time.perf_counter()
#
#     #### aaa correction ####
#     dA_aaa = ccsdpt_loops.ccsdpta_p(
#         pspace["aaa"],
#         T.a, T.aa,
#         H.aa.vooo, H.aa.vvov, H.aa.oovv, H.a.ov,
#         H.aa.vovv, H.aa.ooov,
#         H.a.oo, H.a.vv,
#         system.noccupied_alpha, system.nunoccupied_alpha,
#     )
#     #### aab correction ####
#     dA_aab = ccsdpt_loops.ccsdptb_p(
#         pspace["aab"],
#         T.a, T.b, T.aa, T.ab,
#         H.ab.ovoo, H.ab.vooo, H.aa.vooo,
#         H.ab.vvvo, H.ab.vvov, H.aa.vvov,
#         H.ab.vovv, H.ab.ovvv, H.aa.vovv,
#         H.ab.ooov, H.ab.oovo, H.aa.ooov,
#         H.a.ov, H.b.ov, H.aa.oovv, H.ab.oovv,
#         H.a.oo, H.a.vv, H.b.oo, H.b.vv,
#         system.noccupied_alpha, system.nunoccupied_alpha,
#         system.noccupied_beta, system.nunoccupied_beta,
#     )
#     # If using RHF, can terminate now
#     if use_RHF:
#         # Sum corrections for each spin case, using that aaa=bbb and aab=abb for RHF symmetry
#         correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
#     else:
#         ### abb correction ###
#         dA_abb = ccsdpt_loops.ccsdptc_p(
#             pspace["abb"],
#             T.a, T.b, T.ab, T.bb,
#             H.ab.vooo, H.bb.vooo, H.ab.ovoo,
#             H.ab.vvov, H.bb.vvov, H.ab.vvvo, H.ab.ovvv,
#             H.ab.vovv, H.bb.vovv, H.ab.oovo, H.ab.ooov,
#             H.bb.ooov,
#             H.a.ov, H.b.ov,
#             H.ab.oovv, H.bb.oovv,
#             H.a.oo, H.a.vv, H.b.oo, H.b.vv,
#             system.noccupied_alpha, system.nunoccupied_alpha,
#             system.noccupied_beta, system.nunoccupied_beta,
#         )
#         ### bbb correction ###
#         dA_bbb = ccsdpt_loops.ccsdptd_p(
#             pspace["bbb"],
#             T.b, T.bb,
#             H.bb.vooo, H.bb.vvov, H.bb.oovv, H.b.ov,
#             H.bb.vovv, H.bb.ooov,
#             H.b.oo, H.b.vv,
#             system.noccupied_beta, system.nunoccupied_beta,
#         )
#         # Sum the corrections from each spincase
#         correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb
#
#     t_end = time.perf_counter()
#     minutes, seconds = divmod(t_end - t_start, 60)
#
#     # print the results
#     energy_A = corr_energy + correction_A
#     total_energy_A = system.reference_energy + energy_A
#     Eccp3 = {"A" : total_energy_A}
#     deltap3 = {"A" : correction_A}
#
#     print('   CC(P;3)_(T) Calculation Summary')
#     print('   -------------------------------------')
#     print("   Completed in  ({:0.2f}m  {:0.2f}s)".format(minutes, seconds))
#     print("   CC(P) = {:>10.10f}".format(system.reference_energy + corr_energy))
#     print(
#         "   CC(P;3)_(T) = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
#             total_energy_A, energy_A, correction_A
#         )
#     )
#
#     return Eccp3, deltap3


# def calc_ccpert3_with_selection(T, corr_energy, H, system, pspace, num_add, use_RHF=False):
#     """
#     Calculate the ground-state CCSD(T) correction to the CC(P) energy.
#     """
#     t_start = time.perf_counter()
#
#     # initialize empty moments vector and triples list
#     num_add = int(num_add)
#     moments = np.zeros(num_add)
#     triples_list = np.zeros((num_add, 6), dtype=np.int8)
#
#     #### aaa correction ####
#     # calculate intermediates
#     #I2A_vvov = H.aa.vvov + ccpy_einsum("me,abim->abie", H.a.ov, T.aa)
#     # perform correction in-loop
#     dA_aaa, moments, triples_list = ccsdpt_loops.ccsdpta_p_with_selection(
#         moments,
#         triples_list,
#         pspace["aaa"],
#         T.a, T.aa,
#         H.aa.vooo, H.aa.vvov, H.aa.oovv, H.a.ov,
#         H.aa.vovv, H.aa.ooov,
#         H.a.oo, H.a.vv,
#         system.noccupied_alpha, system.nunoccupied_alpha, num_add,
#     )
#     #### aab correction ####
#     # calculate intermediates
#     #I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ecjk->mcjk", H.a.ov, T.ab)
#     #I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeik->amik", H.b.ov, T.ab)
#     #I2A_vooo = H.aa.vooo - ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
#     dA_aab, moments, triples_list = ccsdpt_loops.ccsdptb_p_with_selection(
#         moments,
#         triples_list,
#         pspace["aab"],
#         T.a, T.b, T.aa, T.ab,
#         H.ab.ovoo, H.ab.vooo, H.aa.vooo,
#         H.ab.vvvo, H.ab.vvov, H.aa.vvov,
#         H.ab.vovv, H.ab.ovvv, H.aa.vovv,
#         H.ab.ooov, H.ab.oovo, H.aa.ooov,
#         H.a.ov, H.b.ov, H.aa.oovv, H.ab.oovv,
#         H.a.oo, H.a.vv, H.b.oo, H.b.vv,
#         system.noccupied_alpha, system.nunoccupied_alpha,
#         system.noccupied_beta, system.nunoccupied_beta,
#         num_add,
#     )
#
#     if use_RHF:
#         correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
#     else:
#         #### abb correction ####
#         # calculate intermediates
#         #I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
#         #I2C_vooo = H.bb.vooo - ccpy_einsum("me,cekj->cmkj", H.b.ov, T.bb)
#         #I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ebij->mbij", H.a.ov, T.ab)
#         dA_abb, moments, triples_list = ccsdpt_loops.ccsdptc_p_with_selection(
#             moments,
#             triples_list,
#             pspace["abb"],
#             T.a, T.b, T.ab, T.bb,
#             H.ab.vooo, H.bb.vooo, H.ab.ovoo,
#             H.ab.vvov, H.bb.vvov, H.ab.vvvo, H.ab.ovvv,
#             H.ab.vovv, H.bb.vovv, H.ab.oovo, H.ab.ooov,
#             H.bb.ooov,
#             H.a.ov, H.b.ov,
#             H.ab.oovv, H.bb.oovv,
#             H.a.oo, H.a.vv, H.b.oo, H.b.vv,
#             system.noccupied_alpha, system.nunoccupied_alpha,
#             system.noccupied_beta, system.nunoccupied_beta,
#             num_add,
#         )
#
#         #### bbb correction ####
#         # calculate intermediates
#         #I2C_vvov = H.bb.vvov + ccpy_einsum("me,abim->abie", H.b.ov, T.bb)
#         dA_bbb, moments, triples_list = ccsdpt_loops.ccsdptd_p_with_selection(
#             moments,
#             triples_list,
#             pspace["bbb"],
#             T.b, T.bb,
#             H.bb.vooo, H.bb.vvov, H.bb.oovv, H.b.ov,
#             H.bb.vovv, H.bb.ooov,
#             H.b.oo, H.b.vv,
#             system.noccupied_beta, system.nunoccupied_beta, num_add,
#         )
#
#         correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb
#
#     t_end = time.perf_counter()
#     minutes, seconds = divmod(t_end - t_start, 60)
#
#     # print the results
#     energy_A = corr_energy + correction_A
#     total_energy_A = system.reference_energy + energy_A
#
#     print('   CC(P;3)_(T) Calculation Summary')
#     print('   -------------------------------------')
#     print("   Completed in  ({:0.2f}m  {:0.2f}s)".format(minutes, seconds))
#     print("   CC(P) = {:>10.10f}".format(system.reference_energy + corr_energy))
#     print(
#         "   CC(P;3)_(T) = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
#             total_energy_A, energy_A, correction_A
#         )
#     )
#     print(
#         "   Selected moments account for {:>5.2f}% of the total CC(P;3)_(T) correction\n".format(
#             sum(moments) / correction_A * 100
#         )
#     )
#
#     return total_energy_A, triples_list

# def calc_ccpert3_with_moments(T, corr_energy, H, system, pspace, use_RHF=False):
#     """
#     Calculate the ground-state CCSD(T) correction to the CC(P) energy using full moment arrays in memory.
#     """
#
#     moments = {"aaa": [], "aab": [], "abb": [], "bbb": []}
#
#     t_start = time.perf_counter()
#
#     #### aaa correction ####
#     # calculate intermediates
#     #I2A_vvov = H.aa.vvov + ccpy_einsum("me,abim->abie", H.a.ov, T.aa)
#     # perform correction in-loop
#     dA_aaa, moments["aaa"] = ccsdpt_loops.ccsdpta_p_full_moment(
#         pspace["aaa"],
#         T.a, T.aa,
#         H.aa.vooo, H.aa.vvov, H.aa.oovv, H.a.ov,
#         H.aa.vovv, H.aa.ooov,
#         H.a.oo, H.a.vv,
#         system.noccupied_alpha, system.nunoccupied_alpha
#     )
#
#     #### aab correction ####
#     # calculate intermediates
#     #I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ecjk->mcjk", H.a.ov, T.ab)
#     #I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeik->amik", H.b.ov, T.ab)
#     #I2A_vooo = H.aa.vooo - ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
#     dA_aab, moments["aab"] = ccsdpt_loops.ccsdptb_p_full_moment(
#         pspace["aab"],
#         T.a, T.b, T.aa, T.ab,
#         H.ab.ovoo, H.ab.vooo, H.aa.vooo,
#         H.ab.vvvo, H.ab.vvov, H.aa.vvov,
#         H.ab.vovv, H.ab.ovvv, H.aa.vovv,
#         H.ab.ooov, H.ab.oovo, H.aa.ooov,
#         H.a.ov, H.b.ov, H.aa.oovv, H.ab.oovv,
#         H.a.oo, H.a.vv, H.b.oo, H.b.vv,
#         system.noccupied_alpha, system.nunoccupied_alpha,
#         system.noccupied_beta, system.nunoccupied_beta,
#     )
#
#     if use_RHF:
#         correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
#     else:
#         #### abb correction ####
#         # calculate intermediates
#         #I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
#         #I2C_vooo = H.bb.vooo - ccpy_einsum("me,cekj->cmkj", H.b.ov, T.bb)
#         #I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ebij->mbij", H.a.ov, T.ab)
#         dA_abb, moments["abb"] = ccsdpt_loops.ccsdptc_p_full_moment(
#             pspace["abb"],
#             T.a, T.b, T.ab, T.bb,
#             H.ab.vooo, H.bb.vooo, H.ab.ovoo,
#             H.ab.vvov, H.bb.vvov, H.ab.vvvo, H.ab.ovvv,
#             H.ab.vovv, H.bb.vovv, H.ab.oovo, H.ab.ooov,
#             H.bb.ooov,
#             H.a.ov, H.b.ov,
#             H.ab.oovv, H.bb.oovv,
#             H.a.oo, H.a.vv, H.b.oo, H.b.vv,
#             system.noccupied_alpha, system.nunoccupied_alpha,
#             system.noccupied_beta, system.nunoccupied_beta,
#         )
#
#         #### bbb correction ####
#         # calculate intermediates
#         #I2C_vvov = H.bb.vvov + ccpy_einsum("me,abim->abie", H.b.ov, T.bb)
#         dA_bbb, moments["bbb"] = ccsdpt_loops.ccsdptd_p_full_moment(
#             pspace["bbb"],
#             T.b, T.bb,
#             H.bb.vooo, H.bb.vvov, H.bb.oovv, H.b.ov,
#             H.bb.vovv, H.bb.ooov,
#             H.b.oo, H.b.vv,
#             system.noccupied_beta, system.nunoccupied_beta
#         )
#
#         correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb
#
#     t_end = time.perf_counter()
#     minutes, seconds = divmod(t_end - t_start, 60)
#
#     # print the results
#     energy_A = corr_energy + correction_A
#     total_energy_A = system.reference_energy + energy_A
#
#     print('   CC(P;3)_(T) Calculation Summary')
#     print('   -------------------------------------')
#     print("   Completed in  ({:0.2f}m  {:0.2f}s)".format(minutes, seconds))
#     print("   CC(P) = {:>10.10f}".format(system.reference_energy + corr_energy))
#     print(
#         "   CC(P;3)_(T) = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
#             total_energy_A, energy_A, correction_A
#         )
#     )
#     return total_energy_A, moments
