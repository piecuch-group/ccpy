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

    h_aa_vvvv, h_ab_vvvv, h_bb_vvvv = get_vvvv_diagonal(H, T)

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

    h_aa_vvvv, h_ab_vvvv, h_bb_vvvv = get_vvvv_diagonal(H, T)

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

def get_vvvv_diagonal(H, T):
    from ccpy.cholesky.cholesky_builders import (build_2index_batch_vvvv_aa,
                                                 build_2index_batch_vvvv_ab,
                                                 build_2index_batch_vvvv_bb)

    # form the diagonal part of the h(vvvv) elements
    nua, nub, noa, nob = T.ab.shape
    # <ab|ab> = <x|aa><x|bb>
    h_aa_vvvv = np.zeros((nua, nua))
    for a in range(nua):
        for b in range(a + 1, nua):
            # batch_ints = np.einsum("xe,xf->ef", H.chol.a.vv[:, a, :], H.chol.a.vv[:, b, :])
            # batch_ints -= batch_ints.T
            batch_ints = build_2index_batch_vvvv_aa(a, b, H)
            h_aa_vvvv[a, b] = batch_ints[a, b]
            h_aa_vvvv[b, a] = batch_ints[a, b]
    h_bb_vvvv = np.zeros((nub, nub))
    for a in range(nub):
        for b in range(a + 1, nub):
            # batch_ints = np.einsum("xe,xf->ef", H.chol.b.vv[:, a, :], H.chol.b.vv[:, b, :])
            # batch_ints -= batch_ints.T
            batch_ints = build_2index_batch_vvvv_bb(a, b, H)
            h_bb_vvvv[a, b] = batch_ints[a, b]
            h_bb_vvvv[b, a] = batch_ints[a, b]
    h_ab_vvvv = np.zeros((nua, nub))
    for a in range(nua):
        for b in range(nub):
            # batch_ints = np.einsum("xe,xf->ef", H.chol.a.vv[:, a, :], H.chol.b.vv[:, b, :])
            batch_ints = build_2index_batch_vvvv_ab(a, b, H)
            h_ab_vvvv[a, b] = batch_ints[a, b]

    for a in range(nua):
        for b in range(a + 1, nua):
            for m in range(noa):
                # h_aa_vvvv[a, b] -= h_aa_vovv[a, m, a, b] * T.a[b, m]
                for n in range(m + 1, noa):
                    h_aa_vvvv[a, b] += H.aa.oovv[m, n, a, b] * T.aa[a, b, m, n]
            h_aa_vvvv[b, a] = h_aa_vvvv[a, b]
    #
    for a in range(nub):
        for b in range(a + 1, nub):
            for m in range(nob):
                for n in range(m + 1, nob):
                    h_bb_vvvv[a, b] += H.bb.oovv[m, n, a, b] * T.bb[a, b, m, n]
            h_bb_vvvv[b, a] = h_bb_vvvv[a, b]
    #
    for a in range(nua):
        for b in range(nub):
            for m in range(noa):
                for n in range(nob):
                    h_ab_vvvv[a, b] += H.ab.oovv[m, n, a, b] * T.ab[a, b, m, n]

    return h_aa_vvvv, h_ab_vvvv, h_bb_vvvv

