"""Functions to calculate the EA-EOMCC(P;Q) 3p-2h correction to EA-EOMCC(P)."""
import time
import numpy as np

from ccpy.constants.constants import hartreetoeV
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.lib.core import eaccp3_correction
from ccpy.eomcc.eaeom3_intermediates import get_eaeom3_p_intermediates
from ccpy.left.left_eaeom_intermediates import get_lefteaeom3_p_intermediates


def calc_eaccp3(T, R, L, r3_excitations, omega, corr_energy, H, H0, system, use_RHF=False):
    """
    Calculate the excited-state EA-EOMCC(P;Q) 3p-2h correction to the EA-EOMCC(P) energy.
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
    do_l3 = {"aaa": True, "aab": True, "abb": True}
    if np.array_equal(r3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1.])):
        do_l3["aaa"] = False
    if np.array_equal(r3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1.])):
        do_l3["aab"] = False
    if np.array_equal(r3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1.])):
        do_l3["abb"] = False

    # Get intermediates
    X_right = get_eaeom3_p_intermediates(H, R, r3_excitations)
    X_left = get_lefteaeom3_p_intermediates(L, r3_excitations, T, do_l3, system)

    #### aaa correction ####
    dA_aaa = 0.0
    dB_aaa = 0.0
    dC_aaa = 0.0
    dD_aaa = 0.0
    for j in range(system.noccupied_alpha):
        for k in range(j + 1, system.noccupied_alpha):
            M3A = eaccp3_correction.build_moments3a_jk(
                j + 1, k + 1,
                R.aa,
                R.aaa, r3_excitations["aaa"],
                R.aab, r3_excitations["aab"],
                T.aa,
                H.a.oo, H.a.vv,
                H.aa.vvvv.transpose(2, 3, 0, 1), H.aa.oooo, H.aa.voov, H.aa.vooo, H.aa.vvov,
                H.ab.voov,
                X_right["aa"]["voo"], X_right["aa"]["vvv"],
            )
            L3A = eaccp3_correction.build_leftamps3a_jk(
                j + 1, k + 1,
                L.a, L.aa,
                L.aaa, r3_excitations["aaa"],
                L.aab, r3_excitations["aab"],
                H.a.ov, H.a.oo, H.a.vv,
                H.aa.vvvv, H.aa.oooo, H.aa.voov.transpose(3, 2, 1, 0), H.aa.ooov, H.aa.vovv, H.aa.oovv,
                H.ab.ovvo.transpose(2, 3, 0, 1),
                X_left["aa"]["ovo"], X_left["aa"]["vvv"],
            )
            dA_aaa, dB_aaa, dC_aaa, dD_aaa = eaccp3_correction.ccp3a_jk(
                dA_aaa, dB_aaa, dC_aaa, dD_aaa,
                j + 1, k + 1, omega,
                M3A, L3A, r3_excitations["aaa"],
                H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
                H.aa.voov, H.aa.oooo, H.aa.vvvv,
                d3aaa_o, d3aaa_v,
            )
    #### aab correction ####
    dA_aab = 0.0
    dB_aab = 0.0
    dC_aab = 0.0
    dD_aab = 0.0
    for j in range(system.noccupied_alpha):
        for k in range(system.noccupied_beta):
            M3B = eaccp3_correction.build_moments3b_jk(
                j + 1, k + 1,
                R.aa, R.ab,
                R.aaa, r3_excitations["aaa"],
                R.aab, r3_excitations["aab"],
                R.abb, r3_excitations["abb"],
                T.aa, T.ab,
                H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                H.aa.vvvv.transpose(2, 3, 0, 1), H.aa.voov, H.aa.vvov,
                H.ab.vvvv.transpose(2, 3, 0, 1), H.ab.oooo, H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
                H.ab.vooo, H.ab.ovoo, H.ab.vvov, H.ab.vvvo,
                H.bb.voov,
                X_right["aa"]["voo"], X_right["aa"]["vvv"],
                X_right["ab"]["voo"], X_right["ab"]["ovo"], X_right["ab"]["vvv"],
            )
            L3B = eaccp3_correction.build_leftamps3b_jk(
                j + 1, k + 1,
                L.a, L.aa, L.ab,
                L.aaa, r3_excitations["aaa"],
                L.aab, r3_excitations["aab"],
                L.abb, r3_excitations["abb"],
                H.a.ov, H.b.ov, H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                H.aa.vvvv, H.aa.voov.transpose(3, 2, 1, 0), H.aa.vovv, H.aa.oovv,
                H.ab.vvvv, H.ab.oooo, H.ab.ovvo.transpose(2, 3, 0, 1), H.ab.vovo.transpose(2, 3, 0, 1),
                H.ab.ovov.transpose(2, 3, 0, 1), H.ab.voov.transpose(2, 3, 0, 1), H.ab.oovv,
                H.ab.ooov, H.ab.oovo, H.ab.vovv, H.ab.ovvv,
                H.bb.voov.transpose(3, 2, 1, 0),
                X_left["aa"]["ovo"], X_left["aa"]["vvv"],
                X_left["ab"]["voo"], X_left["ab"]["ovo"], X_left["ab"]["vvv"],
            )
            dA_aab, dB_aab, dC_aab, dD_aab = eaccp3_correction.ccp3b_jk(
                dA_aab, dB_aab, dC_aab, dD_aab,
                j + 1, k + 1, omega,
                M3B, L3B, r3_excitations["aab"],
                H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                H.aa.voov, H.aa.oooo, H.aa.vvvv,
                H.ab.ovov, H.ab.vovo,
                H.ab.oooo, H.ab.vvvv,
                H.bb.voov,
                d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
            )
    #### abb correction ####
    dA_abb = 0.0
    dB_abb = 0.0
    dC_abb = 0.0
    dD_abb = 0.0
    for j in range(system.noccupied_beta):
        for k in range(j + 1, system.noccupied_beta):
            M3C = eaccp3_correction.build_moments3c_jk(
                j + 1, k + 1,
                R.ab,
                R.aab, r3_excitations["aab"],
                R.abb, r3_excitations["abb"],
                T.ab, T.bb,
                H.a.vv, H.b.oo, H.b.vv,
                H.ab.vvvv.transpose(2, 3, 0, 1), H.ab.vovo, H.ab.ovvo, H.ab.vvvo,
                H.bb.vvvv.transpose(2, 3, 0, 1), H.bb.oooo, H.bb.voov, H.bb.vooo, H.bb.vvov,
                X_right["ab"]["voo"], X_right["ab"]["ovo"], X_right["ab"]["vvv"],
            )
            L3C = eaccp3_correction.build_leftamps3c_jk(
                j + 1, k + 1,
                L.a, L.ab,
                L.aab, r3_excitations["aab"],
                L.abb, r3_excitations["abb"],
                H.b.ov, H.a.vv, H.b.oo, H.b.vv,
                H.ab.vvvv, H.ab.vovo.transpose(2, 3, 0, 1), H.ab.voov.transpose(2, 3, 0, 1), H.ab.vovv, H.ab.oovv,
                H.bb.vvvv, H.bb.oooo, H.bb.voov.transpose(3, 2, 1, 0), H.bb.ooov, H.bb.vovv, H.bb.oovv,
                X_left["ab"]["voo"], X_left["ab"]["ovo"], X_left["ab"]["vvv"],
            )
            dA_abb, dB_abb, dC_abb, dD_abb = eaccp3_correction.ccp3c_jk(
                dA_abb, dB_abb, dC_abb, dD_abb,
                j + 1, k + 1, omega,
                M3C, L3C, r3_excitations["abb"],
                H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                H.aa.voov,
                H.ab.ovov, H.ab.vovo,
                H.ab.oooo, H.ab.vvvv,
                H.bb.voov, H.bb.oooo, H.bb.vvvv,
                d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
            )

    correction_A = dA_aaa + dA_aab + dA_abb
    correction_B = dB_aaa + dB_aab + dB_abb
    correction_C = dC_aaa + dC_aab + dC_abb
    correction_D = dD_aaa + dD_aab + dD_abb

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

    print('   EA-EOMCC(P;Q) Calculation Summary')
    print('   -------------------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   EA-EOMCC(P) = {:>10.10f}    ω = {:>10.10f}     VEE = {:>10.5f} eV".format(
        system.reference_energy + corr_energy + omega, omega, hartreetoeV * omega))
    print(
        "   EA-EOMCC(P;Q)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   EA-EOMCC(P;Q)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   EA-EOMCC(P;Q)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   EA-EOMCC(P;Q)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}".format(
            total_energy_D, energy_D, correction_D
        )
    )
    print("")
    Eccp3 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    deltap3 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}
    return Eccp3, deltap3
