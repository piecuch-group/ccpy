"""Functions to calculate the ground-state CR-CC(2,3) triples correction to CCSD."""
import time
import numpy as np

from ccpy.drivers.cc_energy import get_cc_energy
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.utilities.updates import crcc24_loops

def calc_crcc24(T, L, H, H0, system, use_RHF=False):
    """
    Calculate the ground-state CR-CC(2,4) correction to the CCSD energy.
    """
    t_start = time.time()
    
    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    #### aaaa correction ####
    dA_AAAA, dB_AAAA, dC_AAAA, dD_AAAA = crcc24_loops.crcc24_loops.crcc24a(
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
        d3aaa_v
    )

    #### aaab correction ####
    dA_AAAB, dB_AAAB, dC_AAAB, dD_AAAB = crcc24_loops.crcc24_loops.crcc24b(
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
        H.ab.ovov,
        H.ab.vovo,
        H.ab.oooo,
        H.ab.vvvv,
        H.ab.oovv,
        H.bb.voov,
        d3aaa_o,
        d3aaa_o,
        d3aab_o,
        d3aab_v,
        d3abb_o,
        d3abb_v,
    )
    #### aaab correction ####
    dA_AABB, dB_AABB, dC_AABB, dD_AABB = crcc24_loops.crcc24_loops.crcc24c(
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
        H.ab.ovov,
        H.ab.vovo,
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

    t_end = time.time()
    minutes, seconds = divmod(t_end - t_start, 60)

    # print the results
    cc_energy = get_cc_energy(T, H0)

    energy_A = cc_energy + correction_A
    energy_B = cc_energy + correction_B
    energy_C = cc_energy + correction_C
    energy_D = cc_energy + correction_D

    total_energy_A = system.reference_energy + energy_A
    total_energy_B = system.reference_energy + energy_B
    total_energy_C = system.reference_energy + energy_C
    total_energy_D = system.reference_energy + energy_D

    print('   CR-CC(2,4) Calculation Summary')
    print('   -------------------------------------')
    print("   Completed in  ({:0.2f}m  {:0.2f}s)\n".format(minutes, seconds))
    print("   CCSD = {:>10.10f}".format(system.reference_energy + cc_energy))
    print(
        "   CR-CC(2,4)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CR-CC(2,4)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CR-CC(2,4)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CR-CC(2,4)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )

    Ecrcc24 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    delta24 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}
    
    return Ecrcc24, delta24

