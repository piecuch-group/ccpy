"""Functions to calculate the ground-state CR-CC(2,4)-like Q space quadruples correction to CC(P)."""
import time

from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.lib.core import crcc24_loops
from ccpy.moments.crcc24 import calc_crcc24


def calc_ccp4(T, L, corr_energy, H, H0, system, t4_excitations, use_RHF=False):
    """
    Calculate the ground-state CR-CC(2,4) correction to the CCSD energy.
    """

    # first compute the entire CR-CC(2,4) quadruples correction for all quadruples
    _, delta24_full = calc_crcc24(T, L, corr_energy, H, H0, system, use_RHF=use_RHF)

    # Now, compute the correction for determinants in the P space (which should be a short list hopefully)
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    #### aaaa correction ####
    dA_aaaa, dB_aaaa, dC_aaaa, dD_aaaa = crcc24_loops.crcc24a_p(
        t4_excitations["aaaa"],
        T.aa,
        L.aa,
        H0.a.oo,
        H0.a.vv,
        H.a.oo,
        H.a.vv,
        H.aa.voov,
        H.aa.oooo,
        H.aa.vvvv,
        H.aa.oovv,
        d3aaa_o,
        d3aaa_v,
    )

    #### aaab correction ####
    dA_aaab, dB_aaab, dC_aaab, dD_aaab = crcc24_loops.crcc24b_p(
        t4_excitations["aaab"],
        T.aa,
        T.ab,
        L.aa,
        L.ab,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
        H.aa.voov,
        H.aa.oooo,
        H.aa.vvvv,
        H.aa.oovv,
        H.ab.voov,
        H.ab.ovov,
        H.ab.vovo,
        H.ab.ovvo,
        H.ab.oooo,
        H.ab.vvvv,
        H.ab.oovv,
        H.bb.voov,
        d3aaa_o,
        d3aaa_v,
        d3aab_o,
        d3aab_v,
        d3abb_o,
        d3abb_v,
    )

    #### aabb correction ####
    dA_aabb, dB_aabb, dC_aabb, dD_aabb = crcc24_loops.crcc24c_p(
        t4_excitations["aabb"],
        T.aa,
        T.ab,
        T.bb,
        L.aa,
        L.ab,
        L.bb,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
        H.aa.voov,
        H.aa.oooo,
        H.aa.vvvv,
        H.aa.oovv,
        H.ab.voov,
        H.ab.ovov,
        H.ab.vovo,
        H.ab.ovvo,
        H.ab.oooo,
        H.ab.vvvv,
        H.ab.oovv,
        H.bb.voov,
        H.bb.oooo,
        H.bb.vvvv,
        H.bb.oovv,
        d3aaa_o,
        d3aaa_v,
        d3aab_o,
        d3aab_v,
        d3abb_o,
        d3abb_v,
        d3bbb_o,
        d3bbb_v,
    )

    if use_RHF:
       correction_A = 2.0 * dA_aaaa + 2.0 * dA_aaab + dA_aabb
       correction_B = 2.0 * dB_aaaa + 2.0 * dB_aaab + dB_aabb
       correction_C = 2.0 * dC_aaaa + 2.0 * dC_aaab + dC_aabb
       correction_D = 2.0 * dD_aaaa + 2.0 * dD_aaab + dD_aabb

    else:
        #### abbb correction ####
        dA_abbb, dB_abbb, dC_abbb, dD_abbb = crcc24_loops.crcc24d_p(
                t4_excitations["abbb"],
                T.ab,
                T.bb,
                L.ab,
                L.bb,
                H0.a.oo,
                H0.a.vv,
                H0.b.oo,
                H0.b.vv,
                H.a.oo,
                H.a.vv,
                H.b.oo,
                H.b.vv,
                H.aa.voov,
                H.ab.voov,
                H.ab.ovov,
                H.ab.vovo,
                H.ab.ovvo,
                H.ab.oooo,
                H.ab.vvvv,
                H.ab.oovv,
                H.bb.voov,
                H.bb.oooo,
                H.bb.vvvv,
                H.bb.oovv,
                d3aab_o,
                d3aab_v,
                d3bbb_o,
                d3bbb_v,
                d3abb_o,
                d3abb_v,
        )
        #### bbbb correction ####
        dA_bbbb, dB_bbbb, dC_bbbb, dD_bbbb = crcc24_loops.crcc24e_p(
                t4_excitations["bbbb"],
                T.bb,
                L.bb,
                H0.b.oo,
                H0.b.vv,
                H.b.oo,
                H.b.vv,
                H.bb.voov,
                H.bb.oooo,
                H.bb.vvvv,
                H.bb.oovv,
                d3bbb_o,
                d3bbb_v,
        )

        correction_A = dA_aaaa + dA_aaab + dA_aabb + dA_abbb + dA_bbbb
        correction_B = dB_aaaa + dB_aaab + dB_aabb + dB_abbb + dB_bbbb
        correction_C = dC_aaaa + dC_aaab + dC_aabb + dC_abbb + dC_bbbb
        correction_D = dD_aaaa + dD_aaab + dD_aabb + dD_abbb + dD_bbbb


    # Compute the CC(P;4) energy as the difference between the correction over all quadruples
    # minus the correction over quadruples in the P space
    correction_A = delta24_full["A"] - correction_A
    correction_B = delta24_full["B"] - correction_B
    correction_C = delta24_full["C"] - correction_C
    correction_D = delta24_full["D"] - correction_D

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

    print('   CC(P;4) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   CC(P) = {:>10.10f}".format(system.reference_energy + corr_energy))
    print(
        "   CC(P;4)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CC(P;4)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CC(P;4)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CC(P;4)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )

    Eccp4 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    deltap4 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}

    return Eccp4, deltap4
