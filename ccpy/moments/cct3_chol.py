"""Functions to calculate the ground-state CC(t;3) triples correction to CCSDt."""
import time
import numpy as np

from ccpy.constants.constants import hartreetoeV
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.lib.core import cct3_loops

def calc_cct3(T, L, corr_energy, H, H0, system, use_RHF=False, num_active=1):
    """
    Calculate the ground-state CC(t;3) correction to the CCSDt energy.
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
    dA_aaa, dB_aaa, dC_aaa, dD_aaa = cct3_loops.crcc23a_opt(
        T.aa, L.a, L.aa,
        H.aa.vooo, I2A_vvov, H.aa.oovv, H.a.ov,
        H.aa.vovv, H.aa.ooov, H0.a.oo, H0.a.vv,
        H.a.oo, H.a.vv, H.aa.voov, H.aa.oooo,
        h_aa_vvvv,
        d3aaa_o, d3aaa_v,
        num_active,
        system.num_act_occupied_alpha, system.num_act_unoccupied_alpha,
        system.noccupied_alpha, system.nunoccupied_alpha,
    )
    #### aab correction ####
    # calculate intermediates
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    dA_aab, dB_aab, dC_aab, dD_aab = cct3_loops.cct3_loops.crcc23b_opt(
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
        num_active,
        system.num_act_occupied_alpha, system.num_act_unoccupied_alpha,
        system.num_act_occupied_beta, system.num_act_unoccupied_beta,
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
        dA_abb, dB_abb, dC_abb, dD_abb = cct3_loops.crcc23c_opt(
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
            num_active,
            system.num_act_occupied_alpha, system.num_act_unoccupied_alpha,
            system.num_act_occupied_beta, system.num_act_unoccupied_beta,
            system.noccupied_alpha, system.nunoccupied_alpha,
            system.noccupied_beta, system.nunoccupied_beta,
        )

        I2C_vvov = H.bb.vvov + np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
        dA_bbb, dB_bbb, dC_bbb, dD_bbb = cct3_loops.crcc23d_opt(
            T.bb, L.b, L.bb,
            H.bb.vooo, I2C_vvov, H.bb.oovv, H.b.ov,
            H.bb.vovv, H.bb.ooov, H0.b.oo, H0.b.vv,
            H.b.oo, H.b.vv, H.bb.voov, H.bb.oooo, h_bb_vvvv,
            d3bbb_o, d3bbb_v,
            num_active,
            system.num_act_occupied_beta, system.num_act_unoccupied_beta,
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

    print('   CC(t;3) Calculation Summary')
    print('   ---------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   CCSDt = {:>10.10f}".format(system.reference_energy + corr_energy))
    print(
        "   CC(t;3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CC(t;3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CC(t;3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CC(t;3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )

    Ecct3 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    delta23 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}

    return Ecct3, delta23

def calc_eomcct3(T, R, L, r0, omega, corr_energy, H, H0, system, use_RHF=False, num_active=1):
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

    #### aaa correction ####
    # calculate intermediates
    I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
    chi2A_vvvo, chi2A_ovoo = calc_eomm23a_intermediates(T, R, H)
    # perform correction in-loop
    dA_aaa, dB_aaa, dC_aaa, dD_aaa, ddA_aaa, ddB_aaa, ddC_aaa, ddD_aaa = cct3_loops.creomcc23a_opt(omega, r0, T.aa, R.aa, L.a, L.aa,
                                                                          H.aa.vooo, I2A_vvov, H.aa.vvov,
                                                                          chi2A_vvvo, chi2A_ovoo,
                                                                          H.aa.oovv, H.a.ov, H.aa.vovv, H.aa.ooov,
                                                                          H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
                                                                          H.aa.voov, H.aa.oooo, H.aa.vvvv,
                                                                          d3aaa_o, d3aaa_v,
                                                                          num_active,
                                                                          system.num_act_occupied_alpha,
                                                                          system.num_act_unoccupied_alpha,
                                                                          system.noccupied_alpha,
                                                                          system.nunoccupied_alpha)
    #### aab correction ####
    # calculate intermediates
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    (
        chi2B_vvvo,
        chi2B_ovoo,
        chi2A_vvvo,
        chi2A_vooo,
        chi2B_vvov,
        chi2B_vooo,
    ) = calc_eomcc23b_intermediates(T, R, H)
    # perform correction in-loop
    dA_aab, dB_aab, dC_aab, dD_aab, ddA_aab, ddB_aab, ddC_aab, ddD_aab = cct3_loops.creomcc23b_opt(omega, r0, T.aa, T.ab, R.aa,
                                                                          R.ab, L.a, L.b, L.aa, L.ab,
                                                                          I2B_ovoo, I2B_vooo, I2A_vooo, H.ab.vvvo,
                                                                          H.ab.vvov,
                                                                          H.aa.vvov, H.ab.vovv, H.ab.ovvv,
                                                                          H.aa.vovv,
                                                                          H.ab.ooov, H.ab.oovo, H.aa.ooov,
                                                                          chi2B_vvvo, chi2B_ovoo, chi2A_vvvo,
                                                                          chi2A_vooo,
                                                                          chi2B_vvov, chi2B_vooo, H.ab.ovoo,
                                                                          H.aa.vooo, H.ab.vooo,
                                                                          H.a.ov, H.b.ov, H.aa.oovv, H.ab.oovv,
                                                                          H0.a.oo, H0.a.vv,
                                                                          H0.b.oo, H0.b.vv, H.a.oo, H.a.vv, H.b.oo,
                                                                          H.b.vv,
                                                                          H.aa.voov, H.aa.oooo, H.aa.vvvv,
                                                                          H.ab.ovov, H.ab.vovo,
                                                                          H.ab.oooo, H.ab.vvvv, H.bb.voov,
                                                                          d3aaa_o, d3aaa_v, d3aab_o, d3aab_v,
                                                                          d3abb_o, d3abb_v,
                                                                          num_active,
                                                                          system.num_act_occupied_alpha,
                                                                          system.num_act_unoccupied_alpha,
                                                                          system.num_act_occupied_beta,
                                                                          system.num_act_unoccupied_beta,
                                                                          system.noccupied_alpha,
                                                                          system.nunoccupied_alpha,
                                                                          system.noccupied_beta,
                                                                          system.nunoccupied_beta)
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
        # calculate intermediates
        I2B_vooo = H.ab.vooo - np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
        I2C_vooo = H.bb.vooo - np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)
        I2B_ovoo = H.ab.ovoo - np.einsum("me,ebij->mbij", H.a.ov, T.ab, optimize=True)
        (
            chi2B_vvov,
            chi2B_vooo,
            chi2C_vvvo,
            chi2C_vooo,
            chi2B_vvvo,
            chi2B_ovoo,
        ) = calc_eomcc23c_intermediates(T, R, H)
        dA_abb, dB_abb, dC_abb, dD_abb, ddA_abb, ddB_abb, ddC_abb, ddD_abb = cct3_loops.creomcc23c_opt(omega, r0, T.ab, T.bb, R.ab,
                                                                              R.bb, L.a, L.b, L.ab,
                                                                              L.bb,
                                                                              I2B_vooo, I2C_vooo, I2B_ovoo,
                                                                              H.ab.vvov, H.bb.vvov,
                                                                              H.ab.vvvo, H.ab.ovvv, H.ab.vovv,
                                                                              H.bb.vovv, H.ab.oovo, H.ab.ooov,
                                                                              H.bb.ooov, chi2B_vvov, chi2B_vooo,
                                                                              chi2C_vvvo, chi2C_vooo,
                                                                              chi2B_vvvo, chi2B_ovoo, H.ab.vooo,
                                                                              H.bb.vooo, H.ab.ovoo,
                                                                              H.a.ov, H.b.ov, H.ab.oovv, H.bb.oovv,
                                                                              H0.a.oo, H0.a.vv,
                                                                              H0.b.oo, H0.b.vv, H.a.oo, H.a.vv,
                                                                              H.b.oo, H.b.vv, H.aa.voov,
                                                                              H.ab.ovov, H.ab.vovo, H.ab.oooo,
                                                                              H.ab.vvvv, H.bb.voov,
                                                                              H.bb.oooo, H.bb.vvvv,
                                                                              d3aab_o, d3aab_v, d3abb_o, d3abb_v,
                                                                              d3bbb_o, d3bbb_v,
                                                                              num_active,
                                                                              system.num_act_occupied_alpha,
                                                                              system.num_act_unoccupied_alpha,
                                                                              system.num_act_occupied_beta,
                                                                              system.num_act_unoccupied_beta,
                                                                              system.noccupied_alpha,
                                                                              system.nunoccupied_alpha,
                                                                              system.noccupied_beta,
                                                                              system.nunoccupied_beta)
        #### bbb correction ####
        # calculate intermediates
        I2C_vvov = H.bb.vvov + np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
        chi2C_vvvo, chi2C_ovoo = calc_eomm23d_intermediates(T, R, H)
        dA_bbb, dB_bbb, dC_bbb, dD_bbb, ddA_bbb, ddB_bbb, ddC_bbb, ddD_bbb = cct3_loops.creomcc23d_opt(omega, r0, T.bb, R.bb,
                                                                              L.b, L.bb,
                                                                              H.bb.vooo, I2C_vvov, H.bb.vvov,
                                                                              chi2C_vvvo, chi2C_ovoo, H.bb.oovv,
                                                                              H.b.ov, H.bb.vovv, H.bb.ooov, H0.b.oo,
                                                                              H0.b.vv, H.b.oo, H.b.vv, H.bb.voov,
                                                                              H.bb.oooo, H.bb.vvvv,
                                                                              d3bbb_o, d3bbb_v,
                                                                              num_active,
                                                                              system.num_act_occupied_beta,
                                                                              system.num_act_unoccupied_beta,
                                                                              system.noccupied_beta,
                                                                              system.nunoccupied_beta)
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

    print('   EOMCC(t;3) Calculation Summary')
    print('   ------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   EOMCCSDt = {:>10.10f}    ω = {:>10.10f}     VEE = {:>10.5f} eV".format(
        system.reference_energy + corr_energy + omega, omega, hartreetoeV * omega))
    print(
        "   EOMCC(t;3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   EOMCC(t;3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   EOMCC(t;3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   EOMCC(t;3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )
    print(
        "   δ-EOMCC(t;3)_A = {:>10.10f}     δ_A = {:>10.10f}     VEE = {:>10.5f} eV".format(
            delta_vee_A, dcorrection_A, delta_vee_eV_A
        )
    )
    print(
        "   δ-EOMCC(t;3)_B = {:>10.10f}     δ_B = {:>10.10f}     VEE = {:>10.5f} eV".format(
            delta_vee_B, dcorrection_B, delta_vee_eV_B
        )
    )
    print(
        "   δ-EOMCC(t;3)_C = {:>10.10f}     δ_C = {:>10.10f}     VEE = {:>10.5f} eV".format(
            delta_vee_C, dcorrection_C, delta_vee_eV_C
        )
    )
    print(
        "   δ-EOMCC(t;3)_D = {:>10.10f}     δ_D = {:>10.10f}     VEE = {:>10.5f} eV\n".format(
            delta_vee_D, dcorrection_D, delta_vee_eV_D
        )
    )

    Ecrcc23 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    delta23 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}
    ddelta23 = {"A": dcorrection_A, "B": dcorrection_B, "C": dcorrection_C, "D": dcorrection_D}

    return Ecrcc23, delta23, ddelta23

def calc_eomm23a_intermediates(T, R, H):
    Q1 = np.einsum("mnef,fn->me", H.aa.oovv, R.a, optimize=True)
    Q1 += np.einsum("mnef,fn->me", H.ab.oovv, R.b, optimize=True)

    I1 = np.einsum("amje,bm->abej", H.aa.voov, R.a, optimize=True)
    I1 += np.einsum("amfe,bejm->abfj", H.aa.vovv, R.aa, optimize=True)
    I1 += np.einsum("amfe,bejm->abfj", H.ab.vovv, R.ab, optimize=True)
    I1 -= np.transpose(I1, (1, 0, 2, 3))
    I2 = np.einsum("abfe,ej->abfj", H.aa.vvvv, R.a, optimize=True)
    I2 += 0.5 * np.einsum("nmje,abmn->abej", H.aa.ooov, R.aa, optimize=True)
    I2 -= np.einsum("me,abmj->abej", Q1, T.aa, optimize=True)
    chi2A_vvvo = I1 + I2

    I1 = -np.einsum("bmie,ej->mbij", H.aa.voov, R.a, optimize=True)
    I1 += np.einsum("nmie,bejm->nbij", H.aa.ooov, R.aa, optimize=True)
    I1 += np.einsum("nmie,bejm->nbij", H.ab.ooov, R.ab, optimize=True)
    I1 -= np.transpose(I1, (0, 1, 3, 2))
    I2 = -1.0 * np.einsum("nmij,bm->nbij", H.aa.oooo, R.a, optimize=True)
    I2 += 0.5 * np.einsum("bmfe,efij->mbij", H.aa.vovv, R.aa, optimize=True)
    chi2A_ovoo = I1 + I2

    return chi2A_vvvo, chi2A_ovoo


def calc_eomcc23b_intermediates(T, R, H):
    Q1 = (
            np.einsum("mnef,fn->me", H.aa.oovv, R.a, optimize=True)
            + np.einsum("mnef,fn->me", H.ab.oovv, R.b, optimize=True)
    )
    Q2 = (
            np.einsum("nmfe,fn->me", H.ab.oovv, R.a, optimize=True)
            + np.einsum("nmfe,fn->me", H.bb.oovv, R.b, optimize=True)
    )
    # Intermediate 1: X2B(bcek)*Y2A(aeij) -> Z3B(abcijk)
    Int1 = -1.0 * np.einsum("mcek,bm->bcek", H.ab.ovvo, R.a, optimize=True)
    Int1 -= np.einsum("bmek,cm->bcek", H.ab.vovo, R.b, optimize=True)
    Int1 += np.einsum("bcfe,ek->bcfk", H.ab.vvvv, R.b, optimize=True)
    Int1 += np.einsum("mnek,bcmn->bcek", H.ab.oovo, R.ab, optimize=True)
    Int1 += np.einsum("bmfe,ecmk->bcfk", H.aa.vovv, R.ab, optimize=True)
    Int1 += np.einsum("bmfe,ecmk->bcfk", H.ab.vovv, R.bb, optimize=True)
    Int1 -= np.einsum("mcfe,bemk->bcfk", H.ab.ovvv, R.ab, optimize=True)
    # Intermediate 2: X2B(ncjk)*Y2A(abin) -> Z3B(abcijk)
    Int2 = -1.0 * np.einsum("nmjk,cm->ncjk", H.ab.oooo, R.b, optimize=True)
    Int2 += np.einsum("mcje,ek->mcjk", H.ab.ovov, R.b, optimize=True)
    Int2 += np.einsum("mcek,ej->mcjk", H.ab.ovvo, R.a, optimize=True)
    Int2 += np.einsum("mcef,efjk->mcjk", H.ab.ovvv, R.ab, optimize=True)
    Int2 += np.einsum("nmje,ecmk->ncjk", H.aa.ooov, R.ab, optimize=True)
    Int2 += np.einsum("nmje,ecmk->ncjk", H.ab.ooov, R.bb, optimize=True)
    Int2 -= np.einsum("nmek,ecjm->ncjk", H.ab.oovo, R.ab, optimize=True)
    # Intermediate 3: X2A(abej)*Y2B(ecik) -> Z3B(abcijk)
    Int3 = np.einsum("amje,bm->abej", H.aa.voov, R.a,
                     optimize=True)  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int3 += 0.5 * np.einsum("abfe,ej->abfj", H.aa.vvvv, R.a, optimize=True)  # (*) added factor 1/2 to compensate A(ab)
    Int3 += 0.25 * np.einsum("nmje,abmn->abej", H.aa.ooov, R.aa,
                             optimize=True)  # (*) added factor 1/2 to compensate A(ab)
    Int3 += np.einsum("amfe,bejm->abfj", H.aa.vovv, R.aa, optimize=True)
    Int3 += np.einsum("amfe,bejm->abfj", H.ab.vovv, R.ab, optimize=True)
    Int3 -= 0.5 * np.einsum("me,abmj->abej", Q1, T.aa, optimize=True)  # (*) added factor 1/2 to compensate A(ab)
    Int3 -= np.transpose(Int3, (1, 0, 2, 3))
    # Intermediate 4: X2A(bnji)*Y2B(acnk) -> Z3B(abcijk)
    Int4 = -0.5 * np.einsum("nmij,bm->bnji", H.aa.oooo, R.a, optimize=True)  # (*) added factor 1/2 to compenate A(ij)
    Int4 -= np.einsum("bmie,ej->bmji", H.aa.voov, R.a,
                      optimize=True)  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int4 += 0.25 * np.einsum("bmfe,efij->bmji", H.aa.vovv, R.aa,
                             optimize=True)  # (*) added factor 1/2 to compensate A(ij)
    Int4 += np.einsum("nmie,bejm->bnji", H.aa.ooov, R.aa, optimize=True)
    Int4 += np.einsum("nmie,bejm->bnji", H.ab.ooov, R.ab, optimize=True)
    Int4 += 0.5 * np.einsum("me,ebij->bmji", Q1, T.aa, optimize=True)  # (*) added factor 1/2 to compensate A(ij)
    Int4 -= np.transpose(Int4, (0, 1, 3, 2))
    # Intermediate 5: X2B(bcje)*Y2B(aeik) -> Z3B(abcijk)
    Int5 = -1.0 * np.einsum("mcje,bm->bcje", H.ab.ovov, R.a, optimize=True)
    Int5 -= np.einsum("bmje,cm->bcje", H.ab.voov, R.b, optimize=True)
    Int5 += np.einsum("bcef,ej->bcjf", H.ab.vvvv, R.a, optimize=True)
    Int5 += np.einsum("mnjf,bcmn->bcjf", H.ab.ooov, R.ab, optimize=True)
    Int5 += np.einsum("mcef,bejm->bcjf", H.ab.ovvv, R.aa, optimize=True)
    Int5 += np.einsum("cmfe,bejm->bcjf", H.bb.vovv, R.ab, optimize=True)
    Int5 -= np.einsum("bmef,ecjm->bcjf", H.ab.vovv, R.ab, optimize=True)
    # Intermediate 6: X2B(bnjk)*Y2B(acin) -> Z3B(abcijk)
    Int6 = -1.0 * np.einsum("mnjk,bm->bnjk", H.ab.oooo, R.a, optimize=True)
    Int6 += np.einsum("bmje,ek->bmjk", H.ab.voov, R.b, optimize=True)
    Int6 += np.einsum("bmek,ej->bmjk", H.ab.vovo, R.a, optimize=True)
    Int6 += np.einsum("bnef,efjk->bnjk", H.ab.vovv, R.ab, optimize=True)
    Int6 += np.einsum("mnek,bejm->bnjk", H.ab.oovo, R.aa, optimize=True)
    Int6 += np.einsum("nmke,bejm->bnjk", H.bb.ooov, R.ab, optimize=True)
    Int6 -= np.einsum("nmje,benk->bmjk", H.ab.ooov, R.ab, optimize=True)
    Int6 += np.einsum("me,bejk->bmjk", Q2, T.ab, optimize=True)

    return Int1, Int2, Int3, Int4, Int5, Int6


def calc_eomcc23c_intermediates(T, R, H):
    Q1 = (
            np.einsum("mnef,fn->me", H.bb.oovv, R.b, optimize=True)
            + np.einsum("nmfe,fn->me", H.ab.oovv, R.a, optimize=True)
    )
    Q2 = (
            np.einsum("mnef,fn->me", H.ab.oovv, R.b, optimize=True)
            + np.einsum("nmfe,fn->me", H.aa.oovv, R.a, optimize=True)
    )
    # Intermediate 1: X2B(cbke)*Y2C(aeij) -> Z3C(cbakji)
    Int1 = -1.0 * np.einsum("cmke,bm->cbke", H.ab.voov, R.b, optimize=True)
    Int1 -= np.einsum("mbke,cm->cbke", H.ab.ovov, R.a, optimize=True)
    Int1 += np.einsum("cbef,ek->cbkf", H.ab.vvvv, R.a, optimize=True)
    Int1 += np.einsum("nmke,cbnm->cbke", H.ab.ooov, R.ab, optimize=True)
    Int1 += np.einsum("bmfe,cekm->cbkf", H.bb.vovv, R.ab, optimize=True)
    Int1 += np.einsum("mbef,ecmk->cbkf", H.ab.ovvv, R.aa, optimize=True)
    Int1 -= np.einsum("cmef,ebkm->cbkf", H.ab.vovv, R.ab, optimize=True)
    # Intermediate 2: X2B(cnkj)*Y2C(abin) -> Z3C(cbakji)
    Int2 = -1.0 * np.einsum("mnkj,cm->cnkj", H.ab.oooo, R.a, optimize=True)
    Int2 += np.einsum("cmej,ek->cmkj", H.ab.vovo, R.a, optimize=True)
    Int2 += np.einsum("cmke,ej->cmkj", H.ab.voov, R.b, optimize=True)
    Int2 += np.einsum("cmfe,fekj->cmkj", H.ab.vovv, R.ab, optimize=True)
    Int2 += np.einsum("nmje,cekm->cnkj", H.bb.ooov, R.ab, optimize=True)
    Int2 += np.einsum("mnej,ecmk->cnkj", H.ab.oovo, R.aa, optimize=True)
    Int2 -= np.einsum("mnke,cemj->cnkj", H.ab.ooov, R.ab, optimize=True)
    # Intermediate 3: X2C(abej)*Y2B(ceki) -> Z3C(cbakji)
    Int3 = np.einsum("amje,bm->abej", H.bb.voov, R.b,
                     optimize=True)  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int3 += 0.5 * np.einsum("abfe,ej->abfj", H.bb.vvvv, R.b, optimize=True)  # (*) added factor 1/2 to compensate A(ab)
    Int3 += 0.25 * np.einsum("nmje,abmn->abej", H.bb.ooov, R.bb,
                             optimize=True)  # (*) added factor 1/2 to compensate A(ab)
    Int3 += np.einsum("amfe,bejm->abfj", H.bb.vovv, R.bb, optimize=True)
    Int3 += np.einsum("maef,ebmj->abfj", H.ab.ovvv, R.ab, optimize=True)
    Int3 -= 0.5 * np.einsum("me,abmj->abej", Q1, T.bb, optimize=True)  # (*) added factor 1/2 to compensate A(ab)
    Int3 -= np.transpose(Int3, (1, 0, 2, 3))
    # Intermediate 4: X2C(bnji)*Y2B(cakn) -> Z3C(cbakji)
    Int4 = -0.5 * np.einsum("nmij,bm->bnji", H.bb.oooo, R.b, optimize=True)  # (*) added factor 1/2 to compenate A(ij)
    Int4 -= np.einsum("bmie,ej->bmji", H.bb.voov, R.b,
                      optimize=True)  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int4 += 0.25 * np.einsum("bmfe,efij->bmji", H.bb.vovv, R.bb,
                             optimize=True)  # (*) added factor 1/2 to compensate A(ij)
    Int4 += np.einsum("nmie,bejm->bnji", H.bb.ooov, R.bb, optimize=True)
    Int4 += np.einsum("mnei,ebmj->bnji", H.ab.oovo, R.ab, optimize=True)
    Int4 += 0.5 * np.einsum("me,ebij->bmji", Q1, T.bb, optimize=True)  # (*) added factor 1/2 to compensate A(ij)
    Int4 -= np.transpose(Int4, (0, 1, 3, 2))
    # Intermediate 5: X2B(cbej)*Y2B(eaki) -> Z3C(cbakji)
    Int5 = -1.0 * np.einsum("cmej,bm->cbej", H.ab.vovo, R.b, optimize=True)
    Int5 -= np.einsum("mbej,cm->cbej", H.ab.ovvo, R.a, optimize=True)
    Int5 += np.einsum("cbfe,ej->cbfj", H.ab.vvvv, R.b, optimize=True)
    Int5 += np.einsum("nmfj,cbnm->cbfj", H.ab.oovo, R.ab, optimize=True)
    Int5 += np.einsum("cmfe,bejm->cbfj", H.ab.vovv, R.bb, optimize=True)
    Int5 += np.einsum("cmfe,ebmj->cbfj", H.aa.vovv, R.ab, optimize=True)
    Int5 -= np.einsum("mbfe,cemj->cbfj", H.ab.ovvv, R.ab, optimize=True)
    # Intermediate 6: X2B(nbkj)*Y2B(cani) -> Z3C(cbakji)
    Int6 = -1.0 * np.einsum("nmkj,bm->nbkj", H.ab.oooo, R.b, optimize=True)
    Int6 += np.einsum("mbej,ek->mbkj", H.ab.ovvo, R.a, optimize=True)
    Int6 += np.einsum("mbke,ej->mbkj", H.ab.ovov, R.b, optimize=True)
    Int6 += np.einsum("nbfe,fekj->nbkj", H.ab.ovvv, R.ab, optimize=True)
    Int6 += np.einsum("nmke,bejm->nbkj", H.ab.ooov, R.bb, optimize=True)
    Int6 += np.einsum("nmke,ebmj->nbkj", H.aa.ooov, R.ab, optimize=True)
    Int6 -= np.einsum("mnej,ebkn->mbkj", H.ab.oovo, R.ab, optimize=True)
    Int6 += np.einsum("me,ebkj->mbkj", Q2, T.ab, optimize=True)

    return Int1, Int2, Int3, Int4, Int5, Int6


def calc_eomm23d_intermediates(T, R, H):
    Q1 = np.einsum("mnef,fn->me", H.bb.oovv, R.b, optimize=True)
    Q1 += np.einsum("nmfe,fn->me", H.ab.oovv, R.a, optimize=True)

    I1 = np.einsum("amje,bm->abej", H.bb.voov, R.b, optimize=True)
    I1 += np.einsum("amfe,bejm->abfj", H.bb.vovv, R.bb, optimize=True)
    I1 += np.einsum("maef,ebmj->abfj", H.ab.ovvv, R.ab, optimize=True)
    I1 -= np.transpose(I1, (1, 0, 2, 3))
    I2 = np.einsum("abfe,ej->abfj", H.bb.vvvv, R.b, optimize=True)
    I2 += 0.5 * np.einsum("nmje,abmn->abej", H.bb.ooov, R.bb, optimize=True)
    I2 -= np.einsum("me,abmj->abej", Q1, T.bb, optimize=True)
    chi2C_vvvo = I1 + I2

    I1 = -np.einsum("bmie,ej->mbij", H.bb.voov, R.b, optimize=True)
    I1 += np.einsum("nmie,bejm->nbij", H.bb.ooov, R.bb, optimize=True)
    I1 += np.einsum("mnei,ebmj->nbij", H.ab.oovo, R.ab, optimize=True)
    I1 -= np.transpose(I1, (0, 1, 3, 2))
    I2 = -1.0 * np.einsum("nmij,bm->nbij", H.bb.oooo, R.b, optimize=True)
    I2 += 0.5 * np.einsum("bmfe,efij->mbij", H.bb.vovv, R.bb, optimize=True)
    chi2C_ovoo = I1 + I2

    return chi2C_vvvo, chi2C_ovoo

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