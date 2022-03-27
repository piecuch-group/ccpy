import time

import numpy as np
from ccpy.drivers.cc_energy import get_cc_energy
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.utilities.updates import crcc_loops


def calc_creomcc23(T, R, L, r0, omega, H, H0, system, use_RHF=False):
    """
    Calculate the ground-state CR-EOMCC(2,3) correction to the EOMCCSD energy.
    """
    t_start = time.time()

    nroot = len(R)
    assert(len(omega) == nroot)
    assert(len(r0) == nroot)

    correction_A = [0.0] * nroot
    correction_B = [0.0] * nroot
    correction_C = [0.0] * nroot
    correction_D = [0.0] * nroot

    # if list of L vectors including ground state is passed in, take just the excited states
    if len(L) == nroot + 1:
        L = L[1:]

    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    for n in range(nroot):

        #### aaa correction ####
        # calculate intermediates
        I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
        chi2A_vvvo, chi2A_ovoo = calc_eomm23a_intermediates(T, R[n], H)
        # perform correction in-loop
        dA_aaa, dB_aaa, dC_aaa, dD_aaa = crcc_loops.crcc_loops.creomcc23a_opt(omega[n], r0[n], T.aa, R[n].aa, L[n].a, L[n].aa,
                                                                   H.aa.vooo, I2A_vvov, H.aa.vvov, chi2A_vvvo,
                                                                   chi2A_ovoo, H0.aa.oovv, H.a.ov, H.aa.vovv,
                                                                   H.aa.ooov, H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
                                                                   H.aa.voov, H.aa.oooo, H.aa.vvvv,
                                                                   d3aaa_o, d3aaa_v,
                                                                   system.noccupied_alpha, system.nunoccupied_alpha)
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
        ) = calc_eomcc23b_intermediates(T, R[n], H)
        # perform correction in-loop
        dA_aab, dB_aab, dC_aab, dD_aab = crcc_loops.crcc_loops.creomcc23b_opt(omega[n], r0[n], T.aa, T.ab, R[n].aa, R[n].ab, L[n].a, L[n].b, L[n].aa, L[n].ab,
                                                                   I2B_ovoo, I2B_vooo, I2A_vooo, H.ab.vvvo, H.ab.vvov,
                                                                   H.aa.vvov, H.ab.vovv, H.ab.ovvv, H.aa.vovv,
                                                                   H.ab.ooov, H.ab.oovo, H.aa.ooov,
                                                                   chi2B_vvvo, chi2B_ovoo, chi2A_vvvo, chi2A_vooo,
                                                                   chi2B_vvov, chi2B_vooo, H.ab.ovoo, H.aa.vooo, H.ab.vooo,
                                                                   H.a.ov, H.b.ov, H0.aa.oovv, H0.ab.oovv, H0.a.oo, H0.a.vv,
                                                                   H0.b.oo, H0.b.vv, H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                                   H.aa.voov, H.aa.oooo, H.aa.vvvv, H.ab.ovov, H.ab.vovo,
                                                                   H.ab.oooo, H.ab.vvvv, H.bb.voov,
                                                                   d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
                                                                   system.noccupied_alpha, system.nunoccupied_alpha,
                                                                   system.noccupied_beta, system.nunoccupied_beta)
        if use_RHF:
            correction_A[n] = 2.0 * dA_aaa + 2.0 * dA_aab
            correction_B[n] = 2.0 * dB_aaa + 2.0 * dB_aab
            correction_C[n] = 2.0 * dC_aaa + 2.0 * dC_aab
            correction_D[n] = 2.0 * dD_aaa + 2.0 * dD_aab
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
            ) = calc_eomcc23c_intermediates(T, R[n], H)
            dA_abb, dB_abb, dC_abb, dD_abb = crcc_loops.crcc_loops.creomcc23c_opt(omega[n], r0[n], T.ab, T.bb, R[n].ab, R[n].bb, L[n].a, L[n].b, L[n].ab, L[n].bb,
                                                                                  I2B_vooo, I2C_vooo, I2B_ovoo, H.ab.vvov, H.bb.vvov,
                                                                                  H.ab.vvvo, H.ab.ovvv, H.ab.vovv, H.bb.vovv, H.ab.oovo, H.ab.ooov,
                                                                                  H.bb.ooov, chi2B_vvov, chi2B_vooo, chi2C_vvvo, chi2C_vooo,
                                                                                  chi2B_vvvo, chi2B_ovoo, H.ab.vooo, H.bb.vooo, H.ab.ovoo,
                                                                                  H.a.ov, H.b.ov, H0.ab.oovv, H.bb.oovv, H0.a.oo, H0.a.vv,
                                                                                  H0.b.oo, H0.b.vv, H.a.oo, H.a.vv, H.b.oo, H.b.vv, H.aa.voov,
                                                                                  H.ab.ovov, H.ab.vovo, H.ab.oooo, H.ab.vvvv, H.bb.voov,
                                                                                  H.bb.oooo, H.bb.vvvv,
                                                                                  d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
                                                                                  system.noccupied_alpha, system.nunoccupied_alpha,
                                                                                  system.noccupied_beta, system.nunoccupied_beta)
            #### bbb correction ####
            # calculate intermediates
            I2C_vvov = H.bb.vvov + np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
            chi2C_vvvo, chi2C_ovoo = calc_eomm23d_intermediates(T, R[n], H)
            dA_bbb, dB_bbb, dC_bbb, dD_bbb = crcc_loops.crcc_loops.creomcc23d_opt(omega[n], r0[n], T.bb, R[n].bb, L[n].b, L[n].bb,
                                                                                  H.bb.vooo, I2C_vvov, H.bb.vvov,
                                                                                  chi2C_vvvo, chi2C_ovoo, H0.bb.oovv,
                                                                                  H.b.ov, H.bb.vovv, H.bb.ooov, H0.b.oo,
                                                                                  H0.b.vv, H.b.oo, H.b.vv, H.bb.voov,
                                                                                  H.bb.oooo, H.bb.vvvv,
                                                                                  d3bbb_o, d3bbb_v,
                                                                                  system.noccupied_beta, system.nunoccupied_beta)
            correction_A[n] = dA_aaa + dA_aab + dA_abb + dA_bbb
            correction_B[n] = dB_aaa + dB_aab + dB_abb + dB_bbb
            correction_C[n] = dC_aaa + dC_aab + dC_abb + dC_bbb
            correction_D[n] = dD_aaa + dD_aab + dD_abb + dD_bbb

    t_end = time.time()
    minutes, seconds = divmod(t_end - t_start, 60)

    # print the results
    cc_energy = get_cc_energy(T, H0)

    energy_A = [cc_energy + x + y for x, y in zip(omega, correction_A)]
    energy_B = [cc_energy + x + y for x, y in zip(omega, correction_B)]
    energy_C = [cc_energy + x + y for x, y in zip(omega, correction_C)]
    energy_D = [cc_energy + x + y for x, y in zip(omega, correction_D)]

    total_energy_A = [system.reference_energy + x for x in energy_A]
    total_energy_B = [system.reference_energy + x for x in energy_B]
    total_energy_C = [system.reference_energy + x for x in energy_C]
    total_energy_D = [system.reference_energy + x for x in energy_D]

    print('   CR-EOMCC(2,3) Calculation Summary')
    print('   -------------------------------------')
    print("   Completed in  ({:0.2f}m  {:0.2f}s)\n".format(minutes, seconds))
    for n in range(nroot):
        print("   State", n + 1)
        print("   ---------")
        print("   EOMCCSD = {:>10.10f}".format(system.reference_energy + cc_energy + omega[n]))
        print(
            "   CR-EOMCC(2,3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
                total_energy_A[n], energy_A[n], correction_A[n]
            )
        )
        print(
            "   CR-EOMCC(2,3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
                total_energy_B[n], energy_B[n], correction_B[n]
            )
        )
        print(
            "   CR-EOMCC(2,3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
                total_energy_C[n], energy_C[n], correction_C[n]
            )
        )
        print(
            "   CR-EOMCC(2,3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
                total_energy_D[n], energy_D[n], correction_D[n]
            )
        )

    Ecrcc23 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    delta23 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}

    return Ecrcc23, delta23


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
    Int3 = np.einsum("amje,bm->abej", H.aa.voov, R.a, optimize=True)  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int3 += 0.5 * np.einsum("abfe,ej->abfj", H.aa.vvvv, R.a, optimize=True)  # (*) added factor 1/2 to compensate A(ab)
    Int3 += 0.25 * np.einsum("nmje,abmn->abej", H.aa.ooov, R.aa, optimize=True)  # (*) added factor 1/2 to compensate A(ab)
    Int3 += np.einsum("amfe,bejm->abfj", H.aa.vovv, R.aa, optimize=True)
    Int3 += np.einsum("amfe,bejm->abfj", H.ab.vovv, R.ab, optimize=True)
    Int3 -= 0.5 * np.einsum("me,abmj->abej", Q1, T.aa, optimize=True)  # (*) added factor 1/2 to compensate A(ab)
    Int3 -= np.transpose(Int3, (1, 0, 2, 3))
    # Intermediate 4: X2A(bnji)*Y2B(acnk) -> Z3B(abcijk)
    Int4 = -0.5 * np.einsum("nmij,bm->bnji", H.aa.oooo, R.a, optimize=True)  # (*) added factor 1/2 to compenate A(ij)
    Int4 -= np.einsum("bmie,ej->bmji", H.aa.voov, R.a, optimize=True)  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int4 += 0.25 * np.einsum("bmfe,efij->bmji", H.aa.vovv, R.aa, optimize=True)  # (*) added factor 1/2 to compensate A(ij)
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
    Int3 = np.einsum("amje,bm->abej", H.bb.voov, R.b, optimize=True)  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int3 += 0.5 * np.einsum("abfe,ej->abfj", H.bb.vvvv, R.b, optimize=True)  # (*) added factor 1/2 to compensate A(ab)
    Int3 += 0.25 * np.einsum("nmje,abmn->abej", H.bb.ooov, R.bb, optimize=True)  # (*) added factor 1/2 to compensate A(ab)
    Int3 += np.einsum("amfe,bejm->abfj", H.bb.vovv, R.bb, optimize=True)
    Int3 += np.einsum("maef,ebmj->abfj", H.ab.ovvv, R.ab, optimize=True)
    Int3 -= 0.5 * np.einsum("me,abmj->abej", Q1, T.bb, optimize=True)  # (*) added factor 1/2 to compensate A(ab)
    Int3 -= np.transpose(Int3, (1, 0, 2, 3))
    # Intermediate 4: X2C(bnji)*Y2B(cakn) -> Z3C(cbakji)
    Int4 = -0.5 * np.einsum("nmij,bm->bnji", H.bb.oooo, R.b, optimize=True)  # (*) added factor 1/2 to compenate A(ij)
    Int4 -= np.einsum("bmie,ej->bmji", H.bb.voov, R.b, optimize=True)  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int4 += 0.25 * np.einsum("bmfe,efij->bmji", H.bb.vovv, R.bb, optimize=True)  # (*) added factor 1/2 to compensate A(ij)
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