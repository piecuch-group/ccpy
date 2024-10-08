"""Functions to calculate the ground-state CR-CC(2,3) triples correction to CCSD."""
import time

import numpy as np
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.lib.core import crcc_loops, ccsdpt_loops

def calc_crcc23(T, L, corr_energy, H, H0, system, use_RHF):
    """
    Calculate the ground-state CR-CC(2,3) correction to the CCSD energy.
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
    dA_aaa, dB_aaa, dC_aaa, dD_aaa = crcc_loops.crcc23a_opt(
                                        T.aa, L.a, L.aa,
                                        H.aa.vooo, I2A_vvov, H.aa.oovv, H.a.ov,
                                        H.aa.vovv, H.aa.ooov, H0.a.oo, H0.a.vv,
                                        H.a.oo, H.a.vv, H.aa.voov, H.aa.oooo,
                                        h_aa_vvvv,
                                        d3aaa_o, d3aaa_v,
                                        system.noccupied_alpha, system.nunoccupied_alpha,
    )

    #### aab correction ####
    # calculate intermediates
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    dA_aab, dB_aab, dC_aab, dD_aab = crcc_loops.crcc23b_opt(
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
                                        system.noccupied_alpha, system.nunoccupied_alpha,
                                        system.noccupied_beta, system.nunoccupied_beta,
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
        dA_abb, dB_abb, dC_abb, dD_abb = crcc_loops.crcc23c_opt(
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
                                        system.noccupied_alpha, system.nunoccupied_alpha,
                                        system.noccupied_beta, system.nunoccupied_beta,
        )

        I2C_vvov = H.bb.vvov + np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
        dA_bbb, dB_bbb, dC_bbb, dD_bbb = crcc_loops.crcc23d_opt(
                                        T.bb, L.b, L.bb,
                                        H.bb.vooo, I2C_vvov, H.bb.oovv, H.b.ov,
                                        H.bb.vovv, H.bb.ooov, H0.b.oo, H0.b.vv,
                                        H.b.oo, H.b.vv, H.bb.voov, H.bb.oooo, h_bb_vvvv,
                                        d3bbb_o, d3bbb_v,
                                        system.noccupied_beta, system.nunoccupied_beta,
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

    print('   CR-CC(2,3) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   CCSD = {:>10.10f}".format(system.reference_energy + corr_energy))
    print(
        "   CR-CC(2,3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CR-CC(2,3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CR-CC(2,3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CR-CC(2,3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )

    Ecrcc23 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    delta23 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}
    
    return Ecrcc23, delta23


def calc_ccsdpt(T, corr_energy, H, system, use_RHF=False):
    """
    Calculate the ground-state CCSD(T) correction to the CCSD energy.
    """
    # This formulation exploits the correspondence between the CR-CC(2,3)_A
    # method and CCSD(T). The CR-CC(2,3)_A code can be converted to CCSD(T)
    # by the following approximations [see Włoch et al., J. Phys. Chem. A 2007, 111, 11359]:
    # Using delta(2,3)_A = m(abcijk) * l(ijkabc)_A, approximate
    # m(abcijk) = < ijkabc | H(2) | 0 > => < ijkabc | (V_N * T2)_C | 0 >
    # l(ijkabc)_A = < 0 | (L1 + L2) * H(2) | 0 >/D_MP(abcijk) => < 0 | (T1* + T2*) * V_N | ijkabc >/D_MP(abcijk)
    # Note that all F_N terms present in CR-CC(2,3) (e.g., originating from < 0 | (L2 * H1)_DC | ijkabc >
    # are ignored. This is important to get the correct results when using orbitals that do not
    # obey Brillouin's Thorem, e.g., open-shell ROHF orbitals using the conventional GAMESS
    # canonicalziation scheme.

    t_start = time.perf_counter()
    t_cpu_start = time.process_time()
    #### aaa correction ####
    # perform correction in-loop
    dA_aaa = ccsdpt_loops.ccsdpta_opt(
        T.a, T.aa,
        H.aa.vooo, H.aa.vvov, H.aa.oovv, H.a.ov,
        H.aa.vovv, H.aa.ooov,
        H.a.oo, H.a.vv,
        system.noccupied_alpha, system.nunoccupied_alpha,
    )
    #### aab correction ####
    dA_aab = ccsdpt_loops.ccsdptb_opt(
        T.a, T.b, T.aa, T.ab,
        H.ab.ovoo, H.ab.vooo, H.aa.vooo,
        H.ab.vvvo, H.ab.vvov, H.aa.vvov,
        H.ab.vovv, H.ab.ovvv, H.aa.vovv,
        H.ab.ooov, H.ab.oovo, H.aa.ooov,
        H.a.ov, H.b.ov, H.aa.oovv, H.ab.oovv,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        system.noccupied_alpha, system.nunoccupied_alpha,
        system.noccupied_beta, system.nunoccupied_beta,
    )
    # Can stop here if RHF symmetry is used
    if use_RHF:
        # Sum corrections from each spincase, using RHF symmetry to set aaa=bbb and aab=abb
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
    else:
        #### abb correction ####
        dA_abb = ccsdpt_loops.ccsdptc_opt(
            T.a, T.b, T.ab, T.bb,
            H.ab.vooo, H.bb.vooo, H.ab.ovoo,
            H.ab.vvov, H.bb.vvov, H.ab.vvvo, H.ab.ovvv,
            H.ab.vovv, H.bb.vovv, H.ab.oovo, H.ab.ooov,
            H.bb.ooov,
            H.a.ov, H.b.ov,
            H.ab.oovv, H.bb.oovv,
            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
            system.noccupied_alpha, system.nunoccupied_alpha,
            system.noccupied_beta, system.nunoccupied_beta,
        )
        #### bbb correction ####
        dA_bbb = ccsdpt_loops.ccsdptd_opt(
            T.b, T.bb,
            H.bb.vooo, H.bb.vvov, H.bb.oovv, H.b.ov,
            H.bb.vovv, H.bb.ooov,
            H.b.oo, H.b.vv,
            system.noccupied_beta, system.nunoccupied_beta,
        )
        # Sum corrections from each spincase
        correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    # print the results
    energy_A = corr_energy + correction_A
    total_energy_A = system.reference_energy + energy_A
    Eccsdpt = {"A" : total_energy_A}
    deltapt = {"A" : correction_A}

    print('\n   CCSD(T) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   CCSD = {:>10.10f}".format(system.reference_energy + corr_energy))
    print("   CCSD(T) = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(total_energy_A, energy_A, correction_A)
    )

    return Eccsdpt, deltapt
