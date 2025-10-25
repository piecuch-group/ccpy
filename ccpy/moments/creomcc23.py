import time
from ccpy.utilities.linear_algebra import ccpy_einsum
import numpy as np

from ccpy.constants.constants import hartreetoeV
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.lib.core import crcc_loops

def calc_creomcc23(T, R, L, r0, omega, corr_energy, H, H0, system, use_RHF=False):
    """
    Calculate the ground-state CR-EOMCC(2,3) correction to the EOMCCSD energy.
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

    # get intermediates
    X = get_eomcc23_intermediates(H, R, T, system)

    #### aaa correction ####
    # calculate intermediates
    I2A_vvov = H.aa.vvov + ccpy_einsum("me,abim->abie", H.a.ov, T.aa)
    # perform correction in-loop
    dA_aaa, dB_aaa, dC_aaa, dD_aaa, ddA_aaa, ddB_aaa, ddC_aaa, ddD_aaa = crcc_loops.creomcc23a_opt(
                                                               omega, r0, T.aa, R.aa, L.a, L.aa,
                                                               H.aa.vooo, I2A_vvov, H.aa.vvov, X.aa.vvov.transpose(1, 0, 3, 2),
                                                               X.aa.vooo.transpose(1, 0, 3, 2), H.aa.oovv, H.a.ov, H.aa.vovv,
                                                               H.aa.ooov, H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
                                                               H.aa.voov, H.aa.oooo, h_aa_vvvv,
                                                               d3aaa_o, d3aaa_v,
                                                               system.noccupied_alpha, system.nunoccupied_alpha)
    #### aab correction ####
    # calculate intermediates
    I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ecjk->mcjk", H.a.ov, T.ab)
    I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeik->amik", H.b.ov, T.ab)
    I2A_vooo = H.aa.vooo - ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
    # perform correction in-loop
    dA_aab, dB_aab, dC_aab, dD_aab, ddA_aab, ddB_aab, ddC_aab, ddD_aab = crcc_loops.creomcc23b_opt(
                                                               omega, r0, T.aa, T.ab, R.aa, R.ab, L.a, L.b, L.aa, L.ab,
                                                               I2B_ovoo, I2B_vooo, I2A_vooo, H.ab.vvvo, H.ab.vvov,
                                                               H.aa.vvov, H.ab.vovv, H.ab.ovvv, H.aa.vovv,
                                                               H.ab.ooov, H.ab.oovo, H.aa.ooov,
                                                               X.ab.vvvo, X.ab.ovoo, X.aa.vvov.transpose(1, 0, 3, 2), X.aa.vooo,
                                                               X.ab.vvov, X.ab.vooo, H.ab.ovoo, H.aa.vooo, H.ab.vooo,
                                                               H.a.ov, H.b.ov, H.aa.oovv, H.ab.oovv, H0.a.oo, H0.a.vv,
                                                               H0.b.oo, H0.b.vv, H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                               H.aa.voov, H.aa.oooo, h_aa_vvvv, H.ab.ovov, H.ab.vovo,
                                                               H.ab.oooo, h_ab_vvvv, H.bb.voov,
                                                               d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
                                                               system.noccupied_alpha, system.nunoccupied_alpha,
                                                               system.noccupied_beta, system.nunoccupied_beta)
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
        I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeij->amij", H.b.ov, T.ab)
        I2C_vooo = H.bb.vooo - ccpy_einsum("me,cekj->cmkj", H.b.ov, T.bb)
        I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ebij->mbij", H.a.ov, T.ab)
        dA_abb, dB_abb, dC_abb, dD_abb, ddA_abb, ddB_abb, ddC_abb, ddD_abb = crcc_loops.creomcc23c_opt(
                                                                              omega, r0, T.ab, T.bb, R.ab, R.bb, L.a, L.b, L.ab, L.bb,
                                                                              I2B_vooo, I2C_vooo, I2B_ovoo, H.ab.vvov, H.bb.vvov,
                                                                              H.ab.vvvo, H.ab.ovvv, H.ab.vovv, H.bb.vovv, H.ab.oovo, H.ab.ooov,
                                                                              H.bb.ooov, X.ab.vvov, X.ab.vooo, X.bb.vvov.transpose(1, 0, 3, 2), X.bb.vooo,
                                                                              X.ab.vvvo, X.ab.ovoo, H.ab.vooo, H.bb.vooo, H.ab.ovoo,
                                                                              H.a.ov, H.b.ov, H.ab.oovv, H.bb.oovv, H0.a.oo, H0.a.vv,
                                                                              H0.b.oo, H0.b.vv, H.a.oo, H.a.vv, H.b.oo, H.b.vv, H.aa.voov,
                                                                              H.ab.ovov, H.ab.vovo, H.ab.oooo, h_ab_vvvv, H.bb.voov,
                                                                              H.bb.oooo, h_bb_vvvv,
                                                                              d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
                                                                              system.noccupied_alpha, system.nunoccupied_alpha,
                                                                              system.noccupied_beta, system.nunoccupied_beta)
        #### bbb correction ####
        # calculate intermediates
        I2C_vvov = H.bb.vvov + ccpy_einsum("me,abim->abie", H.b.ov, T.bb)
        dA_bbb, dB_bbb, dC_bbb, dD_bbb, ddA_bbb, ddB_bbb, ddC_bbb, ddD_bbb = crcc_loops.creomcc23d_opt(
                                                                              omega, r0, T.bb, R.bb, L.b, L.bb,
                                                                              H.bb.vooo, I2C_vvov, H.bb.vvov,
                                                                              X.bb.vvov.transpose(1, 0, 3, 2), X.bb.vooo.transpose(1, 0, 3, 2), H.bb.oovv,
                                                                              H.b.ov, H.bb.vovv, H.bb.ooov, H0.b.oo,
                                                                              H0.b.vv, H.b.oo, H.b.vv, H.bb.voov,
                                                                              H.bb.oooo, h_bb_vvvv,
                                                                              d3bbb_o, d3bbb_v,
                                                                              system.noccupied_beta, system.nunoccupied_beta)
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

    print('   CR-EOMCC(2,3) / δ-CR-EOMCC(2,3) Calculation Summary')
    print('   -------------------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   EOMCCSD = {:>10.10f}    ω = {:>10.10f}     VEE = {:>10.5f} eV".format(system.reference_energy + corr_energy + omega, omega, hartreetoeV * omega))
    print(
        "   CR-EOMCC(2,3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CR-EOMCC(2,3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CR-EOMCC(2,3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CR-EOMCC(2,3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}".format(
            total_energy_D, energy_D, correction_D
        )
    )
    print(
        "   δ-CR-EOMCC(2,3)_A = {:>10.10f}     δ_A = {:>10.10f}     VEE = {:>10.5f} eV".format(
            delta_vee_A, dcorrection_A, delta_vee_eV_A
        )
    )
    print(
        "   δ-CR-EOMCC(2,3)_B = {:>10.10f}     δ_B = {:>10.10f}     VEE = {:>10.5f} eV".format(
            delta_vee_B, dcorrection_B, delta_vee_eV_B
        )
    )
    print(
        "   δ-CR-EOMCC(2,3)_C = {:>10.10f}     δ_C = {:>10.10f}     VEE = {:>10.5f} eV".format(
            delta_vee_C, dcorrection_C, delta_vee_eV_C
        )
    )
    print(
        "   δ-CR-EOMCC(2,3)_D = {:>10.10f}     δ_D = {:>10.10f}     VEE = {:>10.5f} eV\n".format(
            delta_vee_D, dcorrection_D, delta_vee_eV_D
        )
    )

    Ecrcc23 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    delta23 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}
    ddelta23 = {"A": dcorrection_A, "B": dcorrection_B, "C": dcorrection_C, "D": dcorrection_D}

    return Ecrcc23, delta23, ddelta23

def get_eomcc23_intermediates(H, R, T, system):
    """Calculate the CCSD-like intermediates for CCSDT. This routine
    should only calculate terms with T2 and any remaining terms outside of the CCS intermediate
    routine."""
    from ccpy.models.integrals import Integral

    # Create new 2-body integral object
    X = Integral.from_empty(system, 2, data_type=H.a.oo.dtype, use_none=True)

    X.a.ov = (
            ccpy_einsum("mnef,fn->me", H.aa.oovv, R.a)
            + ccpy_einsum("mnef,fn->me", H.ab.oovv, R.b)
    )

    X.b.ov = (
            ccpy_einsum("nmfe,fn->me", H.ab.oovv, R.a)
            + ccpy_einsum("nmfe,fn->me", H.bb.oovv, R.b)
    )

    X.aa.vvov = (
            ccpy_einsum("amje,bm->baje", H.aa.voov, R.a)
            + ccpy_einsum("amfe,bejm->bajf", H.aa.vovv, R.aa)
            + ccpy_einsum("amfe,bejm->bajf", H.ab.vovv, R.ab)
            + 0.5 * ccpy_einsum("abfe,ej->bajf", H.aa.vvvv, R.a)
            + 0.25 * ccpy_einsum("nmje,abmn->baje", H.aa.ooov, R.aa)
            - 0.5 * ccpy_einsum("me,abmj->baje", X.a.ov, T.aa)  # counterterm, similar to CR-CC(2,3)
    )
    X.aa.vvov -= np.transpose(X.aa.vvov, (1, 0, 2, 3))

    X.aa.vooo = (
            -ccpy_einsum("bmie,ej->bmji", H.aa.voov, R.a)
            + ccpy_einsum("nmie,bejm->bnji", H.aa.ooov, R.aa)
            + ccpy_einsum("nmie,bejm->bnji", H.ab.ooov, R.ab)
            - 0.5 * ccpy_einsum("nmij,bm->bnji", H.aa.oooo, R.a)
            + 0.25 * ccpy_einsum("bmfe,efij->bmji", H.aa.vovv, R.aa)
    )
    X.aa.vooo -= np.transpose(X.aa.vooo, (0, 1, 3, 2))

    X.ab.vvvo = (
            - ccpy_einsum("mcek,bm->bcek", H.ab.ovvo, R.a)
            - ccpy_einsum("bmek,cm->bcek", H.ab.vovo, R.b)
            + ccpy_einsum("bcfe,ek->bcfk", H.ab.vvvv, R.b)
            + ccpy_einsum("mnek,bcmn->bcek", H.ab.oovo, R.ab)
            + ccpy_einsum("bmfe,ecmk->bcfk", H.aa.vovv, R.ab)
            + ccpy_einsum("bmfe,ecmk->bcfk", H.ab.vovv, R.bb)
            - ccpy_einsum("mcfe,bemk->bcfk", H.ab.ovvv, R.ab)
            - ccpy_einsum("me,bcmk->bcek", X.a.ov, T.ab)  # counterterm, similar to CR-CC(2,3)
    )

    X.ab.ovoo = (
            - ccpy_einsum("nmjk,cm->ncjk", H.ab.oooo, R.b)
            + ccpy_einsum("mcje,ek->mcjk", H.ab.ovov, R.b)
            + ccpy_einsum("mcek,ej->mcjk", H.ab.ovvo, R.a)
            + ccpy_einsum("mcef,efjk->mcjk", H.ab.ovvv, R.ab)
            + ccpy_einsum("nmje,ecmk->ncjk", H.aa.ooov, R.ab)
            + ccpy_einsum("nmje,ecmk->ncjk", H.ab.ooov, R.bb)
            - ccpy_einsum("nmek,ecjm->ncjk", H.ab.oovo, R.ab)
    )

    X.ab.vvov = (
            - ccpy_einsum("mcje,bm->bcje", H.ab.ovov, R.a)
            - ccpy_einsum("bmje,cm->bcje", H.ab.voov, R.b)
            + ccpy_einsum("bcef,ej->bcjf", H.ab.vvvv, R.a)
            + ccpy_einsum("mnjf,bcmn->bcjf", H.ab.ooov, R.ab)
            + ccpy_einsum("mcef,bejm->bcjf", H.ab.ovvv, R.aa)
            + ccpy_einsum("cmfe,bejm->bcjf", H.bb.vovv, R.ab)
            - ccpy_einsum("bmef,ecjm->bcjf", H.ab.vovv, R.ab)
            - ccpy_einsum("me,bcjm->bcje", X.b.ov, T.ab)  # counterterm, similar to CR-CC(2,3)
    )

    X.ab.vooo = (
            - ccpy_einsum("mnjk,bm->bnjk", H.ab.oooo, R.a)
            + ccpy_einsum("bmje,ek->bmjk", H.ab.voov, R.b)
            + ccpy_einsum("bmek,ej->bmjk", H.ab.vovo, R.a)
            + ccpy_einsum("bnef,efjk->bnjk", H.ab.vovv, R.ab)
            + ccpy_einsum("mnek,bejm->bnjk", H.ab.oovo, R.aa)
            + ccpy_einsum("nmke,bejm->bnjk", H.bb.ooov, R.ab)
            - ccpy_einsum("nmje,benk->bmjk", H.ab.ooov, R.ab)
    )

    X.bb.vvov = (
            ccpy_einsum("amje,bm->baje", H.bb.voov, R.b)
            + 0.5 * ccpy_einsum("abfe,ej->bajf", H.bb.vvvv, R.b)
            + 0.25 * ccpy_einsum("nmje,abmn->baje", H.bb.ooov, R.bb)
            + ccpy_einsum("amfe,bejm->bajf", H.bb.vovv, R.bb)
            + ccpy_einsum("maef,ebmj->bajf", H.ab.ovvv, R.ab)
            - 0.5 * ccpy_einsum("me,abmj->baje", X.b.ov, T.bb)  # counterterm, similar to CR-CC(2,3)
    )
    X.bb.vvov -= np.transpose(X.bb.vvov, (1, 0, 2, 3))

    X.bb.vooo = (
            -0.5 * ccpy_einsum("nmij,bm->bnji", H.bb.oooo, R.b)
            - ccpy_einsum("bmie,ej->bmji", H.bb.voov, R.b)
            + 0.25 * ccpy_einsum("bmfe,efij->bmji", H.bb.vovv, R.bb)
            + ccpy_einsum("nmie,bejm->bnji", H.bb.ooov, R.bb)
            + ccpy_einsum("mnei,ebmj->bnji", H.ab.oovo, R.ab)
    )
    X.bb.vooo -= np.transpose(X.bb.vooo, (0, 1, 3, 2))
    return X