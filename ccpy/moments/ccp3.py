"""Functions to calculate the ground-state CC(t;3) triples correction to CCSDt."""
import time
import numpy as np

from ccpy.constants.constants import hartreetoeV
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.lib.core import ccp3_opt_loops, ccp3_adaptive_loops, ccp3_full_correction, ccp3_full_correction_high_mem
from ccpy.left.left_cc_intermediates import build_left_ccsdt_p_intermediates
from ccpy.eomcc.eomccsdt_intermediates import get_eomccsd_intermediates, get_eomccsdt_intermediates, add_R3_p_terms
from ccpy.utilities.utilities import get_memory_usage
from ccpy.utilities.printing import get_timestamp

# [TODO]: EOMCC(P;3) full correction using high-memory Q-space vector
# [TODO]: EOMCC(P;3)_2BA selection algorithm analogous to calc_ccp3_2ba_with_selection

def calc_ccp3_2ba(T, L, t3_excitations, corr_energy, H, H0, system, use_RHF=False, target_irrep=None):
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

    # form the diagonal part of the h(vvvv) elements
    nua, nub, noa, nob = T.ab.shape
    h_aa_vvvv = np.zeros((nua, nua))
    for a in range(nua):
        for b in range(a, nua):
            h_aa_vvvv[a, b] = H.aa.vvvv[a, b, a, b]
            h_aa_vvvv[b, a] = h_aa_vvvv[a, b]
    h_ab_vvvv = np.zeros((nua, nub))
    for a in range(nua):
        for b in range(nub):
            h_ab_vvvv[a, b] = H.ab.vvvv[a, b, a, b]
    h_bb_vvvv = np.zeros((nub, nub))
    for a in range(nub):
        for b in range(a, nub):
            h_bb_vvvv[a, b] = H.bb.vvvv[a, b, a, b]
            h_bb_vvvv[b, a] = h_bb_vvvv[a, b]

    #### aaa correction ####
    # calculate intermediates
    I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
    # perform correction in-loop
    dA_aaa, dB_aaa, dC_aaa, dD_aaa = ccp3_opt_loops.ccp3a_2ba(
            t3_excitations["aaa"].T,
            T.aa, L.a, L.aa,
            H.aa.vooo, I2A_vvov, H.aa.oovv, H.a.ov,
            H.aa.vovv, H.aa.ooov, H0.a.oo, H0.a.vv,
            H.a.oo, H.a.vv, H.aa.voov, H.aa.oooo,
            h_aa_vvvv,
            d3aaa_o, d3aaa_v,
    )
    #### aab correction ####
    # calculate intermediates
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
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
            H.aa.voov, H.aa.oooo, h_aa_vvvv, H.ab.ovov,
            H.ab.vovo, H.ab.oooo, h_ab_vvvv, H.bb.voov,
            d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
    )
    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
        correction_B = 2.0 * dB_aaa + 2.0 * dB_aab
        correction_C = 2.0 * dC_aaa + 2.0 * dC_aab
        correction_D = 2.0 * dD_aaa + 2.0 * dD_aab
    else:
        I2B_vooo = H.ab.vooo - np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
        I2C_vooo = H.bb.vooo - np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)
        I2B_ovoo = H.ab.ovoo - np.einsum("me,ebij->mbij", H.a.ov, T.ab, optimize=True)

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
                h_ab_vvvv, H.bb.voov, H.bb.oooo, h_bb_vvvv,
                d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
        )

        I2C_vvov = H.bb.vvov + np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
        dA_bbb, dB_bbb, dC_bbb, dD_bbb = ccp3_opt_loops.ccp3d_2ba(
                t3_excitations["bbb"].T,
                T.bb, L.b, L.bb,
                H.bb.vooo, I2C_vvov, H.bb.oovv, H.b.ov,
                H.bb.vovv, H.bb.ooov, H0.b.oo, H0.b.vv,
                H.b.oo, H.b.vv, H.bb.voov, H.bb.oooo, h_bb_vvvv,
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

def calc_ccp3_2ba_with_selection(T, L, t3_excitations, corr_energy, H, H0, system, num_add, use_RHF=False, min_thresh=0.0, buffer_factor=2, target_irrep=None):
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

    # form the diagonal part of the h(vvvv) elements
    nua, nub, noa, nob = T.ab.shape
    h_aa_vvvv = np.zeros((nua, nua))
    for a in range(nua):
        for b in range(a, nua):
            h_aa_vvvv[a, b] = H.aa.vvvv[a, b, a, b]
            h_aa_vvvv[b, a] = h_aa_vvvv[a, b]
    h_ab_vvvv = np.zeros((nua, nub))
    for a in range(nua):
        for b in range(nub):
            h_ab_vvvv[a, b] = H.ab.vvvv[a, b, a, b]
    h_bb_vvvv = np.zeros((nub, nub))
    for a in range(nub):
        for b in range(a, nub):
            h_bb_vvvv[a, b] = H.bb.vvvv[a, b, a, b]
            h_bb_vvvv[b, a] = h_bb_vvvv[a, b]

    # initialize empty moments vector and triples list
    num_add = int(num_add)
    moments = np.zeros(buffer_factor * num_add)
    triples_list = np.zeros((buffer_factor * num_add, 6), dtype=np.int32)

    #### aaa correction ####
    # calculate intermediates
    I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
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
        h_aa_vvvv,
        d3aaa_o, d3aaa_v,
        num_add, min_thresh, buffer_factor,
        )

    #### aab correction ####
    # calculate intermediates
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
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
        H.aa.voov, H.aa.oooo, h_aa_vvvv, H.ab.ovov,
        H.ab.vovo, H.ab.oooo, h_ab_vvvv, H.bb.voov,
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
        I2B_vooo = H.ab.vooo - np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
        I2C_vooo = H.bb.vooo - np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)
        I2B_ovoo = H.ab.ovoo - np.einsum("me,ebij->mbij", H.a.ov, T.ab, optimize=True)
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
            h_ab_vvvv, H.bb.voov, H.bb.oooo, h_bb_vvvv,
            d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
            num_add, min_thresh, buffer_factor,
        )
        #### bbb correction ####
        I2C_vvov = H.bb.vvov + np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
        dA_bbb, dB_bbb, dC_bbb, dD_bbb, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3d_2ba_with_selection_opt(
            moments,
            triples_list,
            nfill,
            t3_excitations['bbb'].T,
            T.bb, L.b, L.bb,
            H.bb.vooo, I2C_vvov, H0.bb.oovv, H.b.ov,
            H.bb.vovv, H.bb.ooov, H0.b.oo, H0.b.vv,
            H.b.oo, H.b.vv, H.bb.voov, H.bb.oooo, h_bb_vvvv,
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

def calc_ccp3(T, L, t3_excitations, corr_energy, H, H0, system, use_RHF=False, target_irrep=None):
    """
    Calculate the ground-state CC(P;3) correction to the CC(P) energy.
    """
    print('   CC(P;3) Calculation')
    print('   ------------------------------')
    print('   Calculation started on', get_timestamp())
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # # get reference and target symmetry information
    # sym_ref = system.point_group_irrep_to_number[system.reference_symmetry]
    # if target_irrep is None:
    #     sym_target = -1
    # else:
    #     sym_target = system.point_group_irrep_to_number[target_irrep]
    # # get numerical array of orbital symmetry labels
    # orbsym = np.zeros(len(system.orbital_symmetries), dtype=np.int32)
    # for i, irrep in enumerate(system.orbital_symmetries):
    #     orbsym[i] = system.point_group_irrep_to_number[irrep]

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
    I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2C_vooo = H.bb.vooo - np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)

    #### aaa correction ####
    tic = time.perf_counter()
    print("   Performing aaa correction... ", end="")
    dA_aaa = 0.0
    dB_aaa = 0.0
    dC_aaa = 0.0
    dD_aaa = 0.0
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(j + 1, system.noccupied_alpha):
                # sym_ijk = sym_ref ^ orbsym[i]
                # sym_ijk = sym_ijk ^ orbsym[j]
                # sym_ijk = sym_ijk ^ orbsym[k]
                M3A = ccp3_full_correction.build_moments3a_ijk(
                    i + 1, j + 1, k + 1,
                    T.aaa, t3_excitations["aaa"],
                    T.aab, t3_excitations["aab"],
                    T.aa,
                    H.a.oo, H.a.vv.T,
                    H.aa.oovv, I2A_vvov.transpose(3, 0, 1, 2), H.aa.vooo.transpose(1, 0, 2, 3),
                    H.aa.oooo, H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvvv.transpose(3, 2, 1, 0),
                    H.ab.oovv, H.ab.voov.transpose(1, 3, 0, 2),
                    #orbsym, sym_ijk, sym_target,
                )
                L3A = ccp3_full_correction.build_leftamps3a_ijk(
                    i + 1, j + 1, k + 1,
                    L.a, L.aa,
                    L.aaa, t3_excitations["aaa"],
                    L.aab, t3_excitations["aab"],
                    H.a.ov, H.a.oo, H.a.vv,
                    H.aa.oooo, H.aa.ooov, H.aa.oovv,
                    H.aa.voov, H.aa.vovv, H.aa.vvvv,
                    H.ab.ovvo,
                    X.aa.ooov, X.aa.vovv,
                    #orbsym, sym_ijk, sym_target,
                )
                dA_aaa, dB_aaa, dC_aaa, dD_aaa = ccp3_full_correction.ccp3a_ijk(
                    dA_aaa, dB_aaa, dC_aaa, dD_aaa,
                    i + 1, j + 1, k + 1, 0.0,
                    M3A, L3A, t3_excitations["aaa"],
                    H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
                    H.aa.voov, H.aa.oooo, H.aa.vvvv,
                    d3aaa_o, d3aaa_v,
                    #orbsym, sym_ijk, sym_target,
                )
    toc = time.perf_counter()
    print(f"completed in {toc - tic}s.  Memory Usage = {round(get_memory_usage(), 2)} MB")
    #### aab correction ####
    tic = time.perf_counter()
    print("   Performing aab correction... ", end="")
    dA_aab = 0.0
    dB_aab = 0.0
    dC_aab = 0.0
    dD_aab = 0.0
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(system.noccupied_beta):

                # sym_ijk = sym_ref ^ orbsym[i]
                # sym_ijk = sym_ijk ^ orbsym[j]
                # sym_ijk = sym_ijk ^ orbsym[k]
                M3B = ccp3_full_correction.build_moments3b_ijk(
                    i + 1, j + 1, k + 1,
                    T.aaa, t3_excitations["aaa"],
                    T.aab, t3_excitations["aab"],
                    T.abb, t3_excitations["abb"],
                    T.aa, T.ab,
                    H.a.oo, H.a.vv.T, H.b.oo, H.b.vv.T,
                    H.aa.oovv, H.aa.vvov.transpose(3, 0, 1, 2), I2A_vooo.transpose(1, 0, 2, 3), H.aa.oooo,
                    H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvvv.transpose(3, 2, 1, 0),
                    H.ab.oovv, H.ab.vvov.transpose(3, 0, 1, 2), H.ab.vvvo.transpose(2, 0, 1, 3),
                    I2B_vooo.transpose(1, 0, 2, 3), I2B_ovoo,
                    H.ab.oooo, H.ab.voov.transpose(1, 3, 0, 2), H.ab.vovo.transpose(1, 2, 0, 3),
                    H.ab.ovov.transpose(0, 3, 1, 2), H.ab.ovvo.transpose(0, 2, 1, 3), H.ab.vvvv.transpose(3, 2, 1, 0),
                    H.bb.oovv, H.bb.voov.transpose(1, 3, 0, 2),
                    #orbsym, sym_ijk, sym_target,
                )
                L3B = ccp3_full_correction.build_leftamps3b_ijk(
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
                    X.ab.ooov, X.ab.oovo, X.ab.vovv, X.ab.ovvv,
                    #orbsym, sym_ijk, sym_target,
                )
                dA_aab, dB_aab, dC_aab, dD_aab = ccp3_full_correction.ccp3b_ijk(
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
    toc = time.perf_counter()
    print(f"completed in {toc - tic}s.  Memory Usage = {round(get_memory_usage(), 2)} MB")
    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
        correction_B = 2.0 * dB_aaa + 2.0 * dB_aab
        correction_C = 2.0 * dC_aaa + 2.0 * dC_aab
        correction_D = 2.0 * dD_aaa + 2.0 * dD_aab
    else:
        #### abb correction ####
        tic = time.perf_counter()
        print("   Performing abb correction... ", end="")
        dA_abb = 0.0
        dB_abb = 0.0
        dC_abb = 0.0
        dD_abb = 0.0
        for i in range(system.noccupied_alpha):
            for j in range(system.noccupied_beta):
                for k in range(j + 1, system.noccupied_beta):
                    # sym_ijk = sym_ref ^ orbsym[i]
                    # sym_ijk = sym_ijk ^ orbsym[j]
                    # sym_ijk = sym_ijk ^ orbsym[k]
                    M3C = ccp3_full_correction.build_moments3c_ijk(
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
                        #orbsym, sym_ijk, sym_target,
                    )
                    L3C = ccp3_full_correction.build_leftamps3c_ijk(
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
                        #orbsym, sym_ijk, sym_target,
                    )
                    dA_abb, dB_abb, dC_abb, dD_abb = ccp3_full_correction.ccp3c_ijk(
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
        toc = time.perf_counter()
        print(f"completed in {toc - tic}s.  Memory Usage = {round(get_memory_usage(), 2)} MB")
        #### bbb correction ####
        tic = time.perf_counter()
        print("   Performing bbb correction... ", end="")
        dA_bbb = 0.0
        dB_bbb = 0.0
        dC_bbb = 0.0
        dD_bbb = 0.0
        for i in range(system.noccupied_beta):
            for j in range(i + 1, system.noccupied_beta):
                for k in range(j + 1, system.noccupied_beta):
                    # sym_ijk = sym_ref ^ orbsym[i]
                    # sym_ijk = sym_ijk ^ orbsym[j]
                    # sym_ijk = sym_ijk ^ orbsym[k]
                    M3D = ccp3_full_correction.build_moments3d_ijk(
                        i + 1, j + 1, k + 1,
                        T.abb, t3_excitations["abb"],
                        T.bbb, t3_excitations["bbb"],
                        T.bb,
                        H.b.oo, H.b.vv.T,
                        H.bb.oovv, H.bb.vvov.transpose(3, 0, 1, 2), I2C_vooo.transpose(1, 0, 2, 3),
                        H.bb.oooo, H.bb.voov.transpose(1, 3, 0, 2), H.bb.vvvv.transpose(3, 2, 1, 0),
                        H.ab.oovv, H.ab.ovvo.transpose(0, 2, 1, 3),
                        #orbsym, sym_ijk, sym_target,
                    )
                    L3D = ccp3_full_correction.build_leftamps3d_ijk(
                        i + 1, j + 1, k + 1,
                        L.b, L.bb,
                        L.abb, t3_excitations["abb"],
                        L.bbb, t3_excitations["bbb"],
                        H.b.ov, H.b.oo, H.b.vv,
                        H.ab.voov,
                        H.bb.oooo, H.bb.ooov, H.bb.oovv,
                        H.bb.voov, H.bb.vovv, H.bb.vvvv,
                        X.bb.ooov, X.bb.vovv,
                        #orbsym, sym_ijk, sym_target,
                    )
                    dA_bbb, dB_bbb, dC_bbb, dD_bbb = ccp3_full_correction.ccp3d_ijk(
                        dA_bbb, dB_bbb, dC_bbb, dD_bbb,
                        i + 1, j + 1, k + 1, 0.0,
                        M3D, L3D, t3_excitations["bbb"],
                        H0.b.oo, H0.b.vv, H.b.oo, H.b.vv,
                        H.bb.voov, H.bb.oooo, H.bb.vvvv,
                        d3bbb_o, d3bbb_v,
                    )
        toc = time.perf_counter()
        print(f"completed in {toc - tic}s.  Memory Usage = {round(get_memory_usage(), 2)} MB")
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
    print('   Calculation ended on', get_timestamp())
    return Eccp3, deltap3

def calc_ccp3_high_memory(T, L, t3_excitations, corr_energy, H, H0, system, use_RHF=False, target_irrep=None):
    """
    Calculate the ground-state CC(P;3) correction to the CC(P) energy.
    """
    from ccpy.utilities.qspace import get_triples_qspace, get_active_triples_qspace

    print('   CC(P;3) Calculation (High-Memory Version)')
    print('   -----------------------------------------')
    print('   Calculation started on', get_timestamp(), "\n")
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # get reference and target symmetry information
    sym_ref = system.point_group_irrep_to_number[system.reference_symmetry]
    if target_irrep is None:
        sym_target = -1
    else:
        sym_target = system.point_group_irrep_to_number[target_irrep]

    # get numerical array of orbital symmetry labels
    orbsym = np.zeros(len(system.orbital_symmetries), dtype=np.int32)
    for i, irrep in enumerate(system.orbital_symmetries):
        orbsym[i] = system.point_group_irrep_to_number[irrep]

    # obtain the Q space containing all triples not in the P space
    qspace = get_triples_qspace(system, t3_excitations, target_irrep=target_irrep)
    #qspace = get_active_triples_qspace(system, num_active=1, target_irrep=target_irrep)
    do_correction = {"aaa": True, "aab": True, "abb": True, "bbb": True}
    if np.array_equal(qspace["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_correction["aaa"] = False
    if np.array_equal(qspace["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_correction["aab"] = False
    if np.array_equal(qspace["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_correction["abb"] = False
    if np.array_equal(qspace["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
        do_correction["bbb"] = False

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
    I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2C_vooo = H.bb.vooo - np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)

    #### aaa correction ####
    tic = time.perf_counter()
    print("   Performing aaa correction... ", end="")
    dA_aaa = 0.0
    dB_aaa = 0.0
    dC_aaa = 0.0
    dD_aaa = 0.0
    if do_correction["aaa"]:
        M3A = ccp3_full_correction_high_mem.build_moments3a(
            qspace["aaa"],
            T.aaa, t3_excitations["aaa"],
            T.aab, t3_excitations["aab"],
            T.aa,
            H.a.oo, H.a.vv.T,
            H.aa.oovv, I2A_vvov.transpose(3, 0, 1, 2), H.aa.vooo.transpose(1, 0, 2, 3),
            H.aa.oooo, H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvvv.transpose(3, 2, 1, 0),
            H.ab.oovv, H.ab.voov.transpose(1, 3, 0, 2),
        )
        L3A = ccp3_full_correction_high_mem.build_leftamps3a(
            qspace["aaa"],
            L.a, L.aa,
            L.aaa, t3_excitations["aaa"],
            L.aab, t3_excitations["aab"],
            H.a.ov, H.a.oo, H.a.vv,
            H.aa.oooo, H.aa.ooov, H.aa.oovv,
            H.aa.voov, H.aa.vovv, H.aa.vvvv,
            H.ab.ovvo,
            X.aa.ooov, X.aa.vovv,
        )
        dA_aaa, dB_aaa, dC_aaa, dD_aaa = ccp3_full_correction_high_mem.ccp3a(
            qspace["aaa"], 0.0,
            M3A, L3A, t3_excitations["aaa"],
            H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
            H.aa.voov, H.aa.oooo, H.aa.vvvv,
            d3aaa_o, d3aaa_v,
        )
    else:
        print("no aaa triples in Q space", end=" ")
    toc = time.perf_counter()
    print(f"completed in {toc - tic}s.  Memory Usage = {round(get_memory_usage(), 2)} MB")

    #### aab correction ####
    tic = time.perf_counter()
    print("   Performing aab correction... ", end="")
    dA_aab = 0.0
    dB_aab = 0.0
    dC_aab = 0.0
    dD_aab = 0.0
    if do_correction["aab"]:
        M3B = ccp3_full_correction_high_mem.build_moments3b(
            qspace["aab"],
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
        L3B = ccp3_full_correction_high_mem.build_leftamps3b(
            qspace["aab"],
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
            X.ab.ooov, X.ab.oovo, X.ab.vovv, X.ab.ovvv,
        )
        dA_aab, dB_aab, dC_aab, dD_aab = ccp3_full_correction_high_mem.ccp3b(
            qspace["aab"], 0.0,
            M3B, L3B, t3_excitations["aab"],
            H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
            H.aa.voov, H.aa.oooo, H.aa.vvvv,
            H.ab.ovov, H.ab.vovo,
            H.ab.oooo, H.ab.vvvv,
            H.bb.voov,
            d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
        )
    else:
        print("no aab triples in Q space", end=" ")
    toc = time.perf_counter()
    print(f"completed in {toc - tic}s.  Memory Usage = {round(get_memory_usage(), 2)} MB")
    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
        correction_B = 2.0 * dB_aaa + 2.0 * dB_aab
        correction_C = 2.0 * dC_aaa + 2.0 * dC_aab
        correction_D = 2.0 * dD_aaa + 2.0 * dD_aab
    else:
        #### abb correction ####
        tic = time.perf_counter()
        print("   Performing abb correction... ", end="")
        dA_abb = 0.0
        dB_abb = 0.0
        dC_abb = 0.0
        dD_abb = 0.0
        if do_correction["abb"]:
            M3C = ccp3_full_correction_high_mem.build_moments3c(
                qspace["abb"],
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
            L3C = ccp3_full_correction_high_mem.build_leftamps3c(
                qspace["abb"],
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
            dA_abb, dB_abb, dC_abb, dD_abb = ccp3_full_correction_high_mem.ccp3c(
                qspace["abb"], 0.0,
                M3C, L3C, t3_excitations["abb"],
                H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                H.aa.voov,
                H.ab.ovov, H.ab.vovo,
                H.ab.oooo, H.ab.vvvv,
                H.bb.voov, H.bb.oooo, H.bb.vvvv,
                d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
            )
        else:
            print("no abb triples in Q space", end=" ")
        toc = time.perf_counter()
        print(f"completed in {toc - tic}s.  Memory Usage = {round(get_memory_usage(), 2)} MB")

        #### bbb correction ####
        tic = time.perf_counter()
        print("   Performing bbb correction... ", end="")
        dA_bbb = 0.0
        dB_bbb = 0.0
        dC_bbb = 0.0
        dD_bbb = 0.0
        if do_correction["bbb"]:
            M3D = ccp3_full_correction_high_mem.build_moments3d(
                qspace["bbb"],
                T.abb, t3_excitations["abb"],
                T.bbb, t3_excitations["bbb"],
                T.bb,
                H.b.oo, H.b.vv.T,
                H.bb.oovv, H.bb.vvov.transpose(3, 0, 1, 2), I2C_vooo.transpose(1, 0, 2, 3),
                H.bb.oooo, H.bb.voov.transpose(1, 3, 0, 2), H.bb.vvvv.transpose(3, 2, 1, 0),
                H.ab.oovv, H.ab.ovvo.transpose(0, 2, 1, 3),
            )
            L3D = ccp3_full_correction_high_mem.build_leftamps3d(
                qspace["bbb"],
                L.b, L.bb,
                L.abb, t3_excitations["abb"],
                L.bbb, t3_excitations["bbb"],
                H.b.ov, H.b.oo, H.b.vv,
                H.ab.voov,
                H.bb.oooo, H.bb.ooov, H.bb.oovv,
                H.bb.voov, H.bb.vovv, H.bb.vvvv,
                X.bb.ooov, X.bb.vovv,
            )
            dA_bbb, dB_bbb, dC_bbb, dD_bbb = ccp3_full_correction_high_mem.ccp3d(
                qspace["bbb"], 0.0,
                M3D, L3D, t3_excitations["bbb"],
                H0.b.oo, H0.b.vv, H.b.oo, H.b.vv,
                H.bb.voov, H.bb.oooo, H.bb.vvvv,
                d3bbb_o, d3bbb_v,
            )
        else:
            print("no bbb triples in Q space", end=" ")
        toc = time.perf_counter()
        print(f"completed in {toc - tic}s.  Memory Usage = {round(get_memory_usage(), 2)} MB")
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
    print('   Calculation ended on', get_timestamp())
    return Eccp3, deltap3

def calc_ccp3_with_selection(T, L, t3_excitations, corr_energy, H, H0, system, num_add, use_RHF=False, min_thresh=0.0, buffer_factor=2, target_irrep=None):
    """
    Calculate the ground-state CC(P;3) correction to the CC(P) energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # # get reference and target symmetry information
    # sym_ref = system.point_group_irrep_to_number[system.reference_symmetry]
    # if target_irrep is None:
    #     sym_target = -1
    # else:
    #     sym_target = system.point_group_irrep_to_number[target_irrep]
    # # get numerical array of orbital symmetry labels
    # orbsym = np.zeros(len(system.orbital_symmetries), dtype=np.int32)
    # for i, irrep in enumerate(system.orbital_symmetries):
    #     orbsym[i] = system.point_group_irrep_to_number[irrep]

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
    I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2C_vooo = H.bb.vooo - np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)

    nfill = 1
    #### aaa correction ####
    dA_aaa = 0.0
    dB_aaa = 0.0
    dC_aaa = 0.0
    dD_aaa = 0.0
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(j + 1, system.noccupied_alpha):
                # sym_ijk = sym_ref ^ orbsym[i]
                # sym_ijk = sym_ijk ^ orbsym[j]
                # sym_ijk = sym_ijk ^ orbsym[k]
                M3A = ccp3_full_correction.build_moments3a_ijk(
                    i + 1, j + 1, k + 1,
                    T.aaa, t3_excitations["aaa"],
                    T.aab, t3_excitations["aab"],
                    T.aa,
                    H.a.oo, H.a.vv.T,
                    H.aa.oovv, I2A_vvov.transpose(3, 0, 1, 2), H.aa.vooo.transpose(1, 0, 2, 3),
                    H.aa.oooo, H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvvv.transpose(3, 2, 1, 0),
                    H.ab.oovv, H.ab.voov.transpose(1, 3, 0, 2),
                )
                L3A = ccp3_full_correction.build_leftamps3a_ijk(
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
                dA_aaa, dB_aaa, dC_aaa, dD_aaa, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3a_ijk_with_selection_opt(
                    dA_aaa, dB_aaa, dC_aaa, dD_aaa,
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
                # sym_ijk = sym_ref ^ orbsym[i]
                # sym_ijk = sym_ijk ^ orbsym[j]
                # sym_ijk = sym_ijk ^ orbsym[k]
                M3B = ccp3_full_correction.build_moments3b_ijk(
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
                L3B = ccp3_full_correction.build_leftamps3b_ijk(
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
                dA_aab, dB_aab, dC_aab, dD_aab, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3b_ijk_with_selection_opt(
                    dA_aab, dB_aab, dC_aab, dD_aab,
                    moments,
                    triples_list,
                    nfill,
                    i + 1, j + 1, k + 1, 0.0,
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
        dA_abb = 0.0
        dB_abb = 0.0
        dC_abb = 0.0
        dD_abb = 0.0
        for i in range(system.noccupied_alpha):
            for j in range(system.noccupied_beta):
                for k in range(j + 1, system.noccupied_beta):
                    # sym_ijk = sym_ref ^ orbsym[i]
                    # sym_ijk = sym_ijk ^ orbsym[j]
                    # sym_ijk = sym_ijk ^ orbsym[k]
                    M3C = ccp3_full_correction.build_moments3c_ijk(
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
                    L3C = ccp3_full_correction.build_leftamps3c_ijk(
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
                    dA_abb, dB_abb, dC_abb, dD_abb, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3c_ijk_with_selection_opt(
                        dA_abb, dB_abb, dC_abb, dD_abb,
                        moments,
                        triples_list,
                        nfill,
                        i + 1, j + 1, k + 1, 0.0,
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
        dA_bbb = 0.0
        dB_bbb = 0.0
        dC_bbb = 0.0
        dD_bbb = 0.0
        for i in range(system.noccupied_beta):
            for j in range(i + 1, system.noccupied_beta):
                for k in range(j + 1, system.noccupied_beta):
                    # sym_ijk = sym_ref ^ orbsym[i]
                    # sym_ijk = sym_ijk ^ orbsym[j]
                    # sym_ijk = sym_ijk ^ orbsym[k]
                    M3D = ccp3_full_correction.build_moments3d_ijk(
                        i + 1, j + 1, k + 1,
                        T.abb, t3_excitations["abb"],
                        T.bbb, t3_excitations["bbb"],
                        T.bb,
                        H.b.oo, H.b.vv.T,
                        H.bb.oovv, H.bb.vvov.transpose(3, 0, 1, 2), I2C_vooo.transpose(1, 0, 2, 3),
                        H.bb.oooo, H.bb.voov.transpose(1, 3, 0, 2), H.bb.vvvv.transpose(3, 2, 1, 0),
                        H.ab.oovv, H.ab.ovvo.transpose(0, 2, 1, 3),
                    )
                    L3D = ccp3_full_correction.build_leftamps3d_ijk(
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
                    dA_bbb, dB_bbb, dC_bbb, dD_bbb, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3d_ijk_with_selection_opt(
                        dA_bbb, dB_bbb, dC_bbb, dD_bbb,
                        moments,
                        triples_list,
                        nfill,
                        i + 1, j + 1, k + 1, 0.0,
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
    print(
        "   Selected moments account for {:>5.2f}% of the total CC(P;3)_D correction\n".format(
            sum(moments) / correction_D * 100
        )
    )
    Eccp3 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    deltap3 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}
    return Eccp3["D"], triples_list

def calc_eomccp3(T, R, L, t3_excitations, r3_excitations, r0, omega, corr_energy, H, H0, system, use_RHF=False, target_irrep=None):
    """
    Calculate the excited-state EOMCC(P;3) correction to the EOMCC(P) energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # # get reference and target symmetry information
    # sym_ref = system.point_group_irrep_to_number[system.reference_symmetry]
    # if target_irrep is None:
    #     sym_target = -1
    # else:
    #     sym_target = system.point_group_irrep_to_number[target_irrep]
    # # get numerical array of orbital symmetry labels
    # orbsym = np.zeros(len(system.orbital_symmetries), dtype=np.int32)
    # for i, irrep in enumerate(system.orbital_symmetries):
    #     orbsym[i] = system.point_group_irrep_to_number[irrep]

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

    I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2C_vooo = H.bb.vooo - np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)

    #### aaa correction ####
    dA_aaa = 0.0
    dB_aaa = 0.0
    dC_aaa = 0.0
    dD_aaa = 0.0
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(j + 1, system.noccupied_alpha):
                # sym_ijk = sym_ref ^ orbsym[i]
                # sym_ijk = sym_ijk ^ orbsym[j]
                # sym_ijk = sym_ijk ^ orbsym[k]
                EOMM3A = ccp3_full_correction.build_eom_moments3a_ijk(
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
                M3A = ccp3_full_correction.build_moments3a_ijk(
                    i + 1, j + 1, k + 1,
                    T.aaa, t3_excitations["aaa"],
                    T.aab, t3_excitations["aab"],
                    T.aa,
                    H.a.oo, H.a.vv.T,
                    H.aa.oovv, I2A_vvov.transpose(3, 0, 1, 2), H.aa.vooo.transpose(1, 0, 2, 3),
                    H.aa.oooo, H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvvv.transpose(3, 2, 1, 0),
                    H.ab.oovv, H.ab.voov.transpose(1, 3, 0, 2),
                )
                L3A = ccp3_full_correction.build_leftamps3a_ijk(
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
                dA_aaa, dB_aaa, dC_aaa, dD_aaa = ccp3_full_correction.ccp3a_ijk(
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
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(system.noccupied_beta):
                # sym_ijk = sym_ref ^ orbsym[i]
                # sym_ijk = sym_ijk ^ orbsym[j]
                # sym_ijk = sym_ijk ^ orbsym[k]
                EOMM3B = ccp3_full_correction.build_eom_moments3b_ijk(
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
                M3B = ccp3_full_correction.build_moments3b_ijk(
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
                L3B = ccp3_full_correction.build_leftamps3b_ijk(
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
                dA_aab, dB_aab, dC_aab, dD_aab = ccp3_full_correction.ccp3b_ijk(
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
        for i in range(system.noccupied_alpha):
            for j in range(system.noccupied_beta):
                for k in range(j + 1, system.noccupied_beta):
                    # sym_ijk = sym_ref ^ orbsym[i]
                    # sym_ijk = sym_ijk ^ orbsym[j]
                    # sym_ijk = sym_ijk ^ orbsym[k]
                    EOMM3C = ccp3_full_correction.build_eom_moments3c_ijk(
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
                    M3C = ccp3_full_correction.build_moments3c_ijk(
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
                    L3C = ccp3_full_correction.build_leftamps3c_ijk(
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
                    dA_abb, dB_abb, dC_abb, dD_abb = ccp3_full_correction.ccp3c_ijk(
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
        for i in range(system.noccupied_beta):
            for j in range(i + 1, system.noccupied_beta):
                for k in range(j + 1, system.noccupied_beta):
                    # sym_ijk = sym_ref ^ orbsym[i]
                    # sym_ijk = sym_ijk ^ orbsym[j]
                    # sym_ijk = sym_ijk ^ orbsym[k]
                    EOMM3D = ccp3_full_correction.build_eom_moments3d_ijk(
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
                    M3D = ccp3_full_correction.build_moments3d_ijk(
                        i + 1, j + 1, k + 1,
                        T.abb, t3_excitations["abb"],
                        T.bbb, t3_excitations["bbb"],
                        T.bb,
                        H.b.oo, H.b.vv.T,
                        H.bb.oovv, H.bb.vvov.transpose(3, 0, 1, 2), I2C_vooo.transpose(1, 0, 2, 3),
                        H.bb.oooo, H.bb.voov.transpose(1, 3, 0, 2), H.bb.vvvv.transpose(3, 2, 1, 0),
                        H.ab.oovv, H.ab.ovvo.transpose(0, 2, 1, 3),
                    )
                    L3D = ccp3_full_correction.build_leftamps3d_ijk(
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
                    dA_bbb, dB_bbb, dC_bbb, dD_bbb = ccp3_full_correction.ccp3d_ijk(
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

# def calc_eomccp3_high_memory(T, R, L, t3_excitations, r3_excitations, r0, omega, corr_energy, H, H0, system, use_RHF=False, target_irrep=None):
#     """
#     Calculate the excited-state EOMCC(P;3) correction to the EOMCC(P) energy.
#     """
#     t_start = time.perf_counter()
#     t_cpu_start = time.process_time()
#
#     # get the Hbar 3-body diagonal
#     d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
#     d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
#     d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
#     d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)
#
#     # get L(P)*T(P) intermediates
#     # determine whether l3 updates and l3*t3 intermediates should be done. Stupid compatibility with
#     # empty sections of t3_excitations or l3_excitations. L3 ordering matches T3 at this point.
#     do_l3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
#     do_t3 = {"aaa": True, "aab": True, "abb": True, "bbb": True}
#     if np.array_equal(t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
#         do_t3["aaa"] = False
#     if np.array_equal(t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
#         do_t3["aab"] = False
#     if np.array_equal(t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
#         do_t3["abb"] = False
#     if np.array_equal(t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
#         do_t3["bbb"] = False
#     if np.array_equal(r3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
#         do_l3["aaa"] = False
#     if np.array_equal(r3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
#         do_l3["aab"] = False
#     if np.array_equal(r3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
#         do_l3["abb"] = False
#     if np.array_equal(r3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])):
#         do_l3["bbb"] = False
#
#     # Create intermediates
#     X = build_left_ccsdt_p_intermediates(L, r3_excitations, T, t3_excitations, system, do_t3, do_l3, RHF_symmetry=use_RHF)
#     XR = get_eomccsd_intermediates(H, R, system)
#     XR = get_eomccsdt_intermediates(H, R, T, XR, system)
#     XR = add_R3_p_terms(XR, H, R, r3_excitations)
#
#     I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
#     I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
#     I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
#     I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
#     I2C_vooo = H.bb.vooo - np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)
#
#     #### aaa correction ####
#     dA_aaa = 0.0
#     dB_aaa = 0.0
#     dC_aaa = 0.0
#     dD_aaa = 0.0
#     EOMM3A = ccp3_full_correction.ccp3_full_correction.build_eom_moments3a_ijk(
#         qspace["aaa"],
#         R.aa,
#         R.aaa, r3_excitations["aaa"],
#         R.aab, r3_excitations["aab"],
#         T.aa,
#         T.aaa, t3_excitations["aaa"],
#         T.aab, t3_excitations["aab"],
#         H.a.oo, H.a.vv,
#         H.aa.oooo, H.aa.vooo, H.aa.oovv,
#         H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvov, H.aa.vvvv.transpose(2, 3, 0, 1),
#         H.ab.voov,
#         XR.a.oo, XR.a.vv,
#         XR.aa.oooo, XR.aa.vooo, XR.aa.oovv,
#         XR.aa.voov, XR.aa.vvov, XR.aa.vvvv.transpose(2, 3, 0, 1),
#         XR.ab.voov,
#     )
#     M3A = ccp3_full_correction.ccp3_full_correction.build_moments3a_ijk(
#         i + 1, j + 1, k + 1,
#         T.aaa, t3_excitations["aaa"],
#         T.aab, t3_excitations["aab"],
#         T.aa,
#         H.a.oo, H.a.vv.T,
#         H.aa.oovv, I2A_vvov.transpose(3, 0, 1, 2), H.aa.vooo.transpose(1, 0, 2, 3),
#         H.aa.oooo, H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvvv.transpose(3, 2, 1, 0),
#         H.ab.oovv, H.ab.voov.transpose(1, 3, 0, 2),
#     )
#     L3A = ccp3_full_correction.ccp3_full_correction.build_leftamps3a_ijk(
#         i + 1, j + 1, k + 1,
#         L.a, L.aa,
#         L.aaa, r3_excitations["aaa"],
#         L.aab, r3_excitations["aab"],
#         H.a.ov, H.a.oo, H.a.vv,
#         H.aa.oooo, H.aa.ooov, H.aa.oovv,
#         H.aa.voov, H.aa.vovv, H.aa.vvvv,
#         H.ab.ovvo,
#         X.aa.ooov, X.aa.vovv,
#     )
#     dA_aaa, dB_aaa, dC_aaa, dD_aaa = ccp3_full_correction.ccp3_full_correction.ccp3a_ijk(
#         dA_aaa, dB_aaa, dC_aaa, dD_aaa,
#         i + 1, j + 1, k + 1, omega,
#         r0 * M3A + EOMM3A, L3A, r3_excitations["aaa"],
#         H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
#         H.aa.voov, H.aa.oooo, H.aa.vvvv,
#         d3aaa_o, d3aaa_v,
#     )
#     #### aab correction ####
#     dA_aab = 0.0
#     dB_aab = 0.0
#     dC_aab = 0.0
#     dD_aab = 0.0
#     for i in range(system.noccupied_alpha):
#         for j in range(i + 1, system.noccupied_alpha):
#             for k in range(system.noccupied_beta):
#                 # sym_ijk = sym_ref ^ orbsym[i]
#                 # sym_ijk = sym_ijk ^ orbsym[j]
#                 # sym_ijk = sym_ijk ^ orbsym[k]
#                 EOMM3B = ccp3_full_correction.ccp3_full_correction.build_eom_moments3b_ijk(
#                     i + 1, j + 1, k + 1,
#                     R.aa, R.ab,
#                     R.aaa, r3_excitations["aaa"],
#                     R.aab, r3_excitations["aab"],
#                     R.abb, r3_excitations["abb"],
#                     T.aa, T.ab,
#                     T.aaa, t3_excitations["aaa"],
#                     T.aab, t3_excitations["aab"],
#                     T.abb, t3_excitations["abb"],
#                     H.a.oo, H.a.vv, H.b.oo, H.b.vv,
#                     H.aa.oooo, H.aa.vooo, H.aa.oovv,
#                     H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvov.transpose(3, 0, 1, 2), H.aa.vvvv.transpose(2, 3, 0, 1),
#                     H.ab.oooo, H.ab.vooo, H.ab.ovoo,
#                     H.ab.oovv, H.ab.voov.transpose(1, 3, 0, 2), H.ab.vovo.transpose(1, 2, 0, 3),
#                     H.ab.ovov.transpose(0, 3, 1, 2), H.ab.ovvo.transpose(0, 2, 1, 3), H.ab.vvov.transpose(3, 0, 1, 2),
#                     H.ab.vvvo.transpose(2, 0, 1, 3), H.ab.vvvv.transpose(3, 2, 1, 0),
#                     H.bb.oovv, H.bb.voov.transpose(1, 3, 0, 2),
#                     XR.a.oo, XR.a.vv, XR.b.oo, XR.b.vv,
#                     XR.aa.oooo, XR.aa.vooo, XR.aa.oovv,
#                     XR.aa.voov.transpose(1, 3, 0, 2), XR.aa.vvov.transpose(3, 0, 1, 2), XR.aa.vvvv.transpose(2, 3, 0, 1),
#                     XR.ab.oooo, XR.ab.vooo, XR.ab.ovoo,
#                     XR.ab.oovv, XR.ab.voov.transpose(1, 3, 0, 2), XR.ab.vovo.transpose(1, 2, 0, 3),
#                     XR.ab.ovov.transpose(0, 3, 1, 2), XR.ab.ovvo.transpose(0, 2, 1, 3), XR.ab.vvov.transpose(3, 0, 1, 2),
#                     XR.ab.vvvo.transpose(2, 0, 1, 3), XR.ab.vvvv.transpose(3, 2, 1, 0),
#                     XR.bb.oovv, XR.bb.voov.transpose(1, 3, 0, 2),
#                 )
#                 M3B = ccp3_full_correction.ccp3_full_correction.build_moments3b_ijk(
#                     i + 1, j + 1, k + 1,
#                     T.aaa, t3_excitations["aaa"],
#                     T.aab, t3_excitations["aab"],
#                     T.abb, t3_excitations["abb"],
#                     T.aa, T.ab,
#                     H.a.oo, H.a.vv.T, H.b.oo, H.b.vv.T,
#                     H.aa.oovv, H.aa.vvov.transpose(3, 0, 1, 2), I2A_vooo.transpose(1, 0, 2, 3), H.aa.oooo, H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvvv.transpose(3, 2, 1, 0),
#                     H.ab.oovv, H.ab.vvov.transpose(3, 0, 1, 2), H.ab.vvvo.transpose(2, 0, 1, 3), I2B_vooo.transpose(1, 0, 2, 3), I2B_ovoo,
#                     H.ab.oooo, H.ab.voov.transpose(1, 3, 0, 2), H.ab.vovo.transpose(1, 2, 0, 3), H.ab.ovov.transpose(0, 3, 1, 2), H.ab.ovvo.transpose(0, 2, 1, 3), H.ab.vvvv.transpose(3, 2, 1, 0),
#                     H.bb.oovv, H.bb.voov.transpose(1, 3, 0, 2),
#                 )
#                 L3B = ccp3_full_correction.ccp3_full_correction.build_leftamps3b_ijk(
#                     i + 1, j + 1, k + 1,
#                     L.a, L.b, L.aa, L.ab,
#                     L.aaa, r3_excitations["aaa"],
#                     L.aab, r3_excitations["aab"],
#                     L.abb, r3_excitations["abb"],
#                     H.a.ov, H.a.oo, H.a.vv,
#                     H.b.ov, H.b.oo, H.b.vv,
#                     H.aa.oooo, H.aa.ooov, H.aa.oovv,
#                     H.aa.voov, H.aa.vovv, H.aa.vvvv,
#                     H.ab.oooo, H.ab.ooov, H.ab.oovo,
#                     H.ab.oovv,
#                     H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
#                     H.ab.vovv, H.ab.ovvv, H.ab.vvvv,
#                     H.bb.voov,
#                     X.aa.ooov, X.aa.vovv,
#                     X.ab.ooov, X.ab.oovo, X.ab.vovv, X.ab.ovvv
#                 )
#                 dA_aab, dB_aab, dC_aab, dD_aab = ccp3_full_correction.ccp3_full_correction.ccp3b_ijk(
#                     dA_aab, dB_aab, dC_aab, dD_aab,
#                     i + 1, j + 1, k + 1, omega,
#                     r0 * M3B + EOMM3B, L3B, r3_excitations["aab"],
#                     H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
#                     H.a.oo, H.a.vv, H.b.oo, H.b.vv,
#                     H.aa.voov, H.aa.oooo, H.aa.vvvv,
#                     H.ab.ovov, H.ab.vovo,
#                     H.ab.oooo, H.ab.vvvv,
#                     H.bb.voov,
#                     d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
#                 )
#     if use_RHF:
#         correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
#         correction_B = 2.0 * dB_aaa + 2.0 * dB_aab
#         correction_C = 2.0 * dC_aaa + 2.0 * dC_aab
#         correction_D = 2.0 * dD_aaa + 2.0 * dD_aab
#     else:
#         #### abb correction ####
#         dA_abb = 0.0
#         dB_abb = 0.0
#         dC_abb = 0.0
#         dD_abb = 0.0
#         for i in range(system.noccupied_alpha):
#             for j in range(system.noccupied_beta):
#                 for k in range(j + 1, system.noccupied_beta):
#                     # sym_ijk = sym_ref ^ orbsym[i]
#                     # sym_ijk = sym_ijk ^ orbsym[j]
#                     # sym_ijk = sym_ijk ^ orbsym[k]
#                     EOMM3C = ccp3_full_correction.ccp3_full_correction.build_eom_moments3c_ijk(
#                         i + 1, j + 1, k + 1,
#                         R.ab, R.bb,
#                         R.aab, r3_excitations["aab"],
#                         R.abb, r3_excitations["abb"],
#                         R.bbb, r3_excitations["bbb"],
#                         T.ab, T.bb,
#                         T.aab, t3_excitations["aab"],
#                         T.abb, t3_excitations["abb"],
#                         T.bbb, t3_excitations["bbb"],
#                         H.a.oo, H.a.vv, H.b.oo, H.b.vv,
#                         H.aa.oovv, H.aa.voov,
#                         H.ab.oooo, H.ab.vooo, H.ab.ovoo,
#                         H.ab.oovv, H.ab.voov, H.ab.vovo,
#                         H.ab.ovov, H.ab.ovvo, H.ab.vvov,
#                         H.ab.vvvo, H.ab.vvvv.transpose(2, 3, 0, 1),
#                         H.bb.oooo, H.bb.vooo, H.bb.oovv,
#                         H.bb.voov, H.bb.vvov, H.bb.vvvv.transpose(2, 3, 0, 1),
#                         XR.a.oo, XR.a.vv, XR.b.oo, XR.b.vv,
#                         XR.aa.oovv, XR.aa.voov,
#                         XR.ab.oooo, XR.ab.vooo, XR.ab.ovoo,
#                         XR.ab.oovv, XR.ab.voov, XR.ab.vovo,
#                         XR.ab.ovov, XR.ab.ovvo, XR.ab.vvov,
#                         XR.ab.vvvo, XR.ab.vvvv.transpose(2, 3, 0, 1),
#                         XR.bb.oooo, XR.bb.vooo, XR.bb.oovv,
#                         XR.bb.voov, XR.bb.vvov, XR.bb.vvvv.transpose(2, 3, 0, 1),
#                     )
#                     M3C = ccp3_full_correction.ccp3_full_correction.build_moments3c_ijk(
#                         i + 1, j + 1, k + 1,
#                         T.aab, t3_excitations["aab"],
#                         T.abb, t3_excitations["abb"],
#                         T.bbb, t3_excitations["bbb"],
#                         T.ab, T.bb,
#                         H.a.oo, H.a.vv.T, H.b.oo, H.b.vv.T,
#                         H.aa.oovv, H.aa.voov.transpose(1, 3, 0, 2),
#                         H.ab.oovv, I2B_vooo.transpose(1, 0, 2, 3), I2B_ovoo, H.ab.vvov.transpose(3, 0, 1, 2), H.ab.vvvo.transpose(2, 0, 1, 3), H.ab.oooo,
#                         H.ab.voov.transpose(1, 3, 0, 2), H.ab.vovo.transpose(1, 2, 0, 3), H.ab.ovov.transpose(0, 3, 1, 2), H.ab.ovvo.transpose(0, 2, 1, 3), H.ab.vvvv.transpose(2, 3, 0, 1),
#                         H.bb.oovv, I2C_vooo.transpose(1, 0, 2, 3), H.bb.vvov.transpose(3, 0, 1, 2), H.bb.oooo, H.bb.voov.transpose(1, 3, 0, 2), H.bb.vvvv.transpose(3, 2, 1, 0),
#                     )
#                     L3C = ccp3_full_correction.ccp3_full_correction.build_leftamps3c_ijk(
#                         i + 1, j + 1, k + 1,
#                         L.a, L.b, L.ab, L.bb,
#                         L.aab, r3_excitations["aab"],
#                         L.abb, r3_excitations["abb"],
#                         L.bbb, r3_excitations["bbb"],
#                         H.a.ov, H.a.oo, H.a.vv,
#                         H.b.ov, H.b.oo, H.b.vv,
#                         H.aa.voov,
#                         H.ab.oooo, H.ab.ooov, H.ab.oovo,
#                         H.ab.oovv,
#                         H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
#                         H.ab.vovv, H.ab.ovvv, H.ab.vvvv,
#                         H.bb.oooo, H.bb.ooov, H.bb.oovv,
#                         H.bb.voov, H.bb.vovv, H.bb.vvvv,
#                         X.ab.ooov, X.ab.oovo, X.ab.vovv, X.ab.ovvv,
#                         X.bb.ooov, X.bb.vovv,
#                     )
#                     dA_abb, dB_abb, dC_abb, dD_abb = ccp3_full_correction.ccp3_full_correction.ccp3c_ijk(
#                         dA_abb, dB_abb, dC_abb, dD_abb,
#                         i + 1, j + 1, k + 1, omega,
#                         r0 * M3C + EOMM3C, L3C, r3_excitations["abb"],
#                         H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
#                         H.a.oo, H.a.vv, H.b.oo, H.b.vv,
#                         H.aa.voov,
#                         H.ab.ovov, H.ab.vovo,
#                         H.ab.oooo, H.ab.vvvv,
#                         H.bb.voov, H.bb.oooo, H.bb.vvvv,
#                         d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
#                     )
#         #### bbb correction ####
#         dA_bbb = 0.0
#         dB_bbb = 0.0
#         dC_bbb = 0.0
#         dD_bbb = 0.0
#         for i in range(system.noccupied_beta):
#             for j in range(i + 1, system.noccupied_beta):
#                 for k in range(j + 1, system.noccupied_beta):
#                     # sym_ijk = sym_ref ^ orbsym[i]
#                     # sym_ijk = sym_ijk ^ orbsym[j]
#                     # sym_ijk = sym_ijk ^ orbsym[k]
#                     EOMM3D = ccp3_full_correction.ccp3_full_correction.build_eom_moments3d_ijk(
#                         i + 1, j + 1, k + 1,
#                         R.bb,
#                         R.abb, r3_excitations["abb"],
#                         R.bbb, r3_excitations["bbb"],
#                         T.bb,
#                         T.abb, t3_excitations["abb"],
#                         T.bbb, t3_excitations["bbb"],
#                         H.b.oo, H.b.vv,
#                         H.bb.oooo, H.bb.vooo, H.bb.oovv,
#                         H.bb.voov, H.bb.vvov, H.bb.vvvv.transpose(2, 3, 0, 1),
#                         H.ab.ovvo,
#                         XR.b.oo, XR.b.vv,
#                         XR.bb.oooo, XR.bb.vooo, XR.bb.oovv,
#                         XR.bb.voov, XR.bb.vvov, XR.bb.vvvv.transpose(2, 3, 0, 1),
#                         XR.ab.ovvo,
#                     )
#                     M3D = ccp3_full_correction.ccp3_full_correction.build_moments3d_ijk(
#                         i + 1, j + 1, k + 1,
#                         T.abb, t3_excitations["abb"],
#                         T.bbb, t3_excitations["bbb"],
#                         T.bb,
#                         H.b.oo, H.b.vv.T,
#                         H.bb.oovv, H.bb.vvov.transpose(3, 0, 1, 2), I2C_vooo.transpose(1, 0, 2, 3),
#                         H.bb.oooo, H.bb.voov.transpose(1, 3, 0, 2), H.bb.vvvv.transpose(3, 2, 1, 0),
#                         H.ab.oovv, H.ab.ovvo.transpose(0, 2, 1, 3),
#                     )
#                     L3D = ccp3_full_correction.ccp3_full_correction.build_leftamps3d_ijk(
#                         i + 1, j + 1, k + 1,
#                         L.b, L.bb,
#                         L.abb, r3_excitations["abb"],
#                         L.bbb, r3_excitations["bbb"],
#                         H.b.ov, H.b.oo, H.b.vv,
#                         H.ab.voov,
#                         H.bb.oooo, H.bb.ooov, H.bb.oovv,
#                         H.bb.voov, H.bb.vovv, H.bb.vvvv,
#                         X.bb.ooov, X.bb.vovv,
#                     )
#                     dA_bbb, dB_bbb, dC_bbb, dD_bbb = ccp3_full_correction.ccp3_full_correction.ccp3d_ijk(
#                         dA_bbb, dB_bbb, dC_bbb, dD_bbb,
#                         i + 1, j + 1, k + 1, omega,
#                         r0 * M3D + EOMM3D, L3D, r3_excitations["bbb"],
#                         H0.b.oo, H0.b.vv, H.b.oo, H.b.vv,
#                         H.bb.voov, H.bb.oooo, H.bb.vvvv,
#                         d3bbb_o, d3bbb_v,
#                     )
#         correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb
#         correction_B = dB_aaa + dB_aab + dB_abb + dB_bbb
#         correction_C = dC_aaa + dC_aab + dC_abb + dC_bbb
#         correction_D = dD_aaa + dD_aab + dD_abb + dD_bbb
#
#     t_end = time.perf_counter()
#     t_cpu_end = time.process_time()
#     minutes, seconds = divmod(t_end - t_start, 60)
#
#     energy_A = corr_energy + omega + correction_A
#     energy_B = corr_energy + omega + correction_B
#     energy_C = corr_energy + omega + correction_C
#     energy_D = corr_energy + omega + correction_D
#
#     total_energy_A = system.reference_energy + energy_A
#     total_energy_B = system.reference_energy + energy_B
#     total_energy_C = system.reference_energy + energy_C
#     total_energy_D = system.reference_energy + energy_D
#
#     print('   EOMCC(P;3) Calculation Summary')
#     print('   ------------------------------')
#     print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
#     print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
#     print("   EOMCC(P) = {:>10.10f}    ω = {:>10.10f}     VEE = {:>10.5f} eV".format(
#         system.reference_energy + corr_energy + omega, omega, hartreetoeV * omega))
#     print(
#         "   EOMCC(P;3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
#             total_energy_A, energy_A, correction_A
#         )
#     )
#     print(
#         "   EOMCC(P;3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
#             total_energy_B, energy_B, correction_B
#         )
#     )
#     print(
#         "   EOMCC(P;3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
#             total_energy_C, energy_C, correction_C
#         )
#     )
#     print(
#         "   EOMCC(P;3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
#             total_energy_D, energy_D, correction_D
#         )
#     )
#     Eccp3 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
#     deltap3 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}
#     return Eccp3, deltap3

def calc_eomccp3_with_selection(T, R, L, t3_excitations, r3_excitations, r0, omega, corr_energy, H, H0, system, num_add, use_RHF=False, min_thresh=0.0, buffer_factor=2, target_irrep=None):
    """
    Calculate the excited-state EOMCC(P;3) correction to the EOMCC(P) energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # # get reference and target symmetry information
    # sym_ref = system.point_group_irrep_to_number[system.reference_symmetry]
    # if target_irrep is None:
    #     sym_target = -1
    # else:
    #     sym_target = system.point_group_irrep_to_number[target_irrep]
    # # get numerical array of orbital symmetry labels
    # orbsym = np.zeros(len(system.orbital_symmetries), dtype=np.int32)
    # for i, irrep in enumerate(system.orbital_symmetries):
    #     orbsym[i] = system.point_group_irrep_to_number[irrep]

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

    I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2C_vooo = H.bb.vooo - np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)

    nfill = 1
    #### aaa correction ####
    dA_aaa = 0.0
    dB_aaa = 0.0
    dC_aaa = 0.0
    dD_aaa = 0.0
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(j + 1, system.noccupied_alpha):
                # sym_ijk = sym_ref ^ orbsym[i]
                # sym_ijk = sym_ijk ^ orbsym[j]
                # sym_ijk = sym_ijk ^ orbsym[k]
                EOMM3A = ccp3_full_correction.build_eom_moments3a_ijk(
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
                M3A = ccp3_full_correction.build_moments3a_ijk(
                    i + 1, j + 1, k + 1,
                    T.aaa, t3_excitations["aaa"],
                    T.aab, t3_excitations["aab"],
                    T.aa,
                    H.a.oo, H.a.vv.T,
                    H.aa.oovv, I2A_vvov.transpose(3, 0, 1, 2), H.aa.vooo.transpose(1, 0, 2, 3),
                    H.aa.oooo, H.aa.voov.transpose(1, 3, 0, 2), H.aa.vvvv.transpose(3, 2, 1, 0),
                    H.ab.oovv, H.ab.voov.transpose(1, 3, 0, 2),
                )
                L3A = ccp3_full_correction.build_leftamps3a_ijk(
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
                dA_aaa, dB_aaa, dC_aaa, dD_aaa, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3a_ijk_with_selection_opt(
                    dA_aaa, dB_aaa, dC_aaa, dD_aaa,
                    moments,
                    triples_list,
                    nfill,
                    i + 1, j + 1, k + 1, omega,
                    r0 * M3A + EOMM3A, L3A,
                    r3_excitations['aaa'].T,
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
                # sym_ijk = sym_ref ^ orbsym[i]
                # sym_ijk = sym_ijk ^ orbsym[j]
                # sym_ijk = sym_ijk ^ orbsym[k]
                EOMM3B = ccp3_full_correction.build_eom_moments3b_ijk(
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
                M3B = ccp3_full_correction.build_moments3b_ijk(
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
                L3B = ccp3_full_correction.build_leftamps3b_ijk(
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
                dA_aab, dB_aab, dC_aab, dD_aab, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3b_ijk_with_selection_opt(
                    dA_aab, dB_aab, dC_aab, dD_aab,
                    moments,
                    triples_list,
                    nfill,
                    i + 1, j + 1, k + 1, omega,
                    r0 * M3B + EOMM3B, L3B,
                    r3_excitations['aab'].T,
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
        dA_abb = 0.0
        dB_abb = 0.0
        dC_abb = 0.0
        dD_abb = 0.0
        for i in range(system.noccupied_alpha):
            for j in range(system.noccupied_beta):
                for k in range(j + 1, system.noccupied_beta):
                    # sym_ijk = sym_ref ^ orbsym[i]
                    # sym_ijk = sym_ijk ^ orbsym[j]
                    # sym_ijk = sym_ijk ^ orbsym[k]
                    EOMM3C = ccp3_full_correction.build_eom_moments3c_ijk(
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
                    M3C = ccp3_full_correction.build_moments3c_ijk(
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
                    L3C = ccp3_full_correction.build_leftamps3c_ijk(
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
                    dA_abb, dB_abb, dC_abb, dD_abb, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3c_ijk_with_selection_opt(
                        dA_abb, dB_abb, dC_abb, dD_abb,
                        moments,
                        triples_list,
                        nfill,
                        i + 1, j + 1, k + 1, omega,
                        r0 * M3C + EOMM3C, L3C,
                        r3_excitations['abb'].T,
                        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                        H.aa.voov, H.ab.ovov, H.ab.vovo, H.ab.oooo,
                        H.ab.vvvv, H.bb.voov, H.bb.oooo, H.bb.vvvv,
                        d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
                        num_add, min_thresh, buffer_factor,
                    )
        #### bbb correction ####
        dA_bbb = 0.0
        dB_bbb = 0.0
        dC_bbb = 0.0
        dD_bbb = 0.0
        for i in range(system.noccupied_beta):
            for j in range(i + 1, system.noccupied_beta):
                for k in range(j + 1, system.noccupied_beta):
                    # sym_ijk = sym_ref ^ orbsym[i]
                    # sym_ijk = sym_ijk ^ orbsym[j]
                    # sym_ijk = sym_ijk ^ orbsym[k]
                    EOMM3D = ccp3_full_correction.build_eom_moments3d_ijk(
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
                    M3D = ccp3_full_correction.build_moments3d_ijk(
                        i + 1, j + 1, k + 1,
                        T.abb, t3_excitations["abb"],
                        T.bbb, t3_excitations["bbb"],
                        T.bb,
                        H.b.oo, H.b.vv.T,
                        H.bb.oovv, H.bb.vvov.transpose(3, 0, 1, 2), I2C_vooo.transpose(1, 0, 2, 3),
                        H.bb.oooo, H.bb.voov.transpose(1, 3, 0, 2), H.bb.vvvv.transpose(3, 2, 1, 0),
                        H.ab.oovv, H.ab.ovvo.transpose(0, 2, 1, 3),
                    )
                    L3D = ccp3_full_correction.build_leftamps3d_ijk(
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
                    dA_bbb, dB_bbb, dC_bbb, dD_bbb, moments, triples_list, nfill = ccp3_adaptive_loops.ccp3d_ijk_with_selection_opt(
                        dA_bbb, dB_bbb, dC_bbb, dD_bbb,
                        moments,
                        triples_list,
                        nfill,
                        i + 1, j + 1, k + 1, omega,
                        r0 * M3D + EOMM3D, L3D,
                        r3_excitations['bbb'].T,
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
