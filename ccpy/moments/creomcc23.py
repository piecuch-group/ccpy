import numpy as np

#     for iroot in range(nroot):
#         print("Performing correction for root {}".format(iroot + 1))
#         t_start = time.time()
#
#         # correction containers
#         deltaA = 0.0
#         deltaB = 0.0
#         deltaC = 0.0
#         deltaD = 0.0
#
#         chi2A_vvvo, chi2A_ovoo = calc_eomm23a_intermediates(cc_t, H2A, H2B, iroot)
#         I2A_vvov = H2A["vvov"] + np.einsum(
#             "me,abim->abie", H1A["ov"], cc_t["t2a"], optimize=True
#         )
#         dA_AAA, dB_AAA, dC_AAA, dD_AAA = crcc_loops.creomcc23a_opt(
#             omega[iroot],
#             cc_t["r0"][iroot],
#             cc_t["t2a"],
#             cc_t["r2a"][iroot],
#             cc_t["l1a"][iroot + 1],
#             cc_t["l2a"][iroot + 1],
#             H2A["vooo"],
#             I2A_vvov,
#             H2A["vvov"],
#             chi2A_vvvo,
#             chi2A_ovoo,
#             ints["vA"]["oovv"],
#             H1A["ov"],
#             H2A["vovv"],
#             H2A["ooov"],
#             ints["fA"]["oo"],
#             ints["fA"]["vv"],
#             H1A["oo"],
#             H1A["vv"],
#             H2A["voov"],
#             H2A["oooo"],
#             H2A["vvvv"],
#             D3A["O"],
#             D3A["V"],
#             sys["Nocc_a"],
#             sys["Nunocc_a"],
#         )
#
#         I2B_ovoo = H2B["ovoo"] - np.einsum(
#             "me,ecjk->mcjk", H1A["ov"], cc_t["t2b"], optimize=True
#         )
#         I2B_vooo = H2B["vooo"] - np.einsum(
#             "me,aeik->amik", H1B["ov"], cc_t["t2b"], optimize=True
#         )
#         I2A_vooo = H2A["vooo"] - np.einsum(
#             "me,aeij->amij", H1A["ov"], cc_t["t2a"], optimize=True
#         )
#         (
#             chi2B_vvvo,
#             chi2B_ovoo,
#             chi2A_vvvo,
#             chi2A_vooo,
#             chi2B_vvov,
#             chi2B_vooo,
#         ) = calc_eomcc23b_intermediates(cc_t, H2A, H2B, H2C, iroot)
#         dA_AAB, dB_AAB, dC_AAB, dD_AAB = crcc_loops.creomcc23b_opt(
#             omega[iroot],
#             cc_t["r0"][iroot],
#             cc_t["t2a"],
#             cc_t["t2b"],
#             cc_t["r2a"][iroot],
#             cc_t["r2b"][iroot],
#             cc_t["l1a"][iroot + 1],
#             cc_t["l1b"][iroot + 1],
#             cc_t["l2a"][iroot + 1],
#             cc_t["l2b"][iroot + 1],
#             I2B_ovoo,
#             I2B_vooo,
#             I2A_vooo,
#             H2B["vvvo"],
#             H2B["vvov"],
#             H2A["vvov"],
#             H2B["vovv"],
#             H2B["ovvv"],
#             H2A["vovv"],
#             H2B["ooov"],
#             H2B["oovo"],
#             H2A["ooov"],
#             chi2B_vvvo,
#             chi2B_ovoo,
#             chi2A_vvvo,
#             chi2A_vooo,
#             chi2B_vvov,
#             chi2B_vooo,
#             H2B["ovoo"],
#             H2A["vooo"],
#             H2B["vooo"],
#             H1A["ov"],
#             H1B["ov"],
#             ints["vA"]["oovv"],
#             ints["vB"]["oovv"],
#             ints["fA"]["oo"],
#             ints["fA"]["vv"],
#             ints["fB"]["oo"],
#             ints["fB"]["vv"],
#             H1A["oo"],
#             H1A["vv"],
#             H1B["oo"],
#             H1B["vv"],
#             H2A["voov"],
#             H2A["oooo"],
#             H2A["vvvv"],
#             H2B["ovov"],
#             H2B["vovo"],
#             H2B["oooo"],
#             H2B["vvvv"],
#             H2C["voov"],
#             D3A["O"],
#             D3A["V"],
#             D3B["O"],
#             D3B["V"],
#             D3C["O"],
#             D3C["V"],
#             sys["Nocc_a"],
#             sys["Nunocc_a"],
#             sys["Nocc_b"],
#             sys["Nunocc_b"],
#         )
#
#         if flag_RHF:
#             deltaA = 2.0 * dA_AAA + 2.0 * dA_AAB
#             deltaB = 2.0 * dB_AAA + 2.0 * dB_AAB
#             deltaC = 2.0 * dC_AAA + 2.0 * dC_AAB
#             deltaD = 2.0 * dD_AAA + 2.0 * dD_AAB
#         else:
#
#             I2B_vooo = H2B["vooo"] - np.einsum(
#                 "me,aeij->amij", H1B["ov"], cc_t["t2b"], optimize=True
#             )
#             I2C_vooo = H2C["vooo"] - np.einsum(
#                 "me,cekj->cmkj", H1B["ov"], cc_t["t2c"], optimize=True
#             )
#             I2B_ovoo = H2B["ovoo"] - np.einsum(
#                 "me,ebij->mbij", H1A["ov"], cc_t["t2b"], optimize=True
#             )
#             (
#                 chi2B_vvov,
#                 chi2B_vooo,
#                 chi2C_vvvo,
#                 chi2C_vooo,
#                 chi2B_vvvo,
#                 chi2B_ovoo,
#             ) = calc_eomcc23c_intermediates(cc_t, H2A, H2B, H2C, iroot)
#             dA_ABB, dB_ABB, dC_ABB, dD_ABB = crcc_loops.creomcc23c_opt(
#                 omega[iroot],
#                 cc_t["r0"][iroot],
#                 cc_t["t2b"],
#                 cc_t["t2c"],
#                 cc_t["r2b"][iroot],
#                 cc_t["r2c"][iroot],
#                 cc_t["l1a"][iroot + 1],
#                 cc_t["l1b"][iroot + 1],
#                 cc_t["l2b"][iroot + 1],
#                 cc_t["l2c"][iroot + 1],
#                 I2B_vooo,
#                 I2C_vooo,
#                 I2B_ovoo,
#                 H2B["vvov"],
#                 H2C["vvov"],
#                 H2B["vvvo"],
#                 H2B["ovvv"],
#                 H2B["vovv"],
#                 H2C["vovv"],
#                 H2B["oovo"],
#                 H2B["ooov"],
#                 H2C["ooov"],
#                 chi2B_vvov,
#                 chi2B_vooo,
#                 chi2C_vvvo,
#                 chi2C_vooo,
#                 chi2B_vvvo,
#                 chi2B_ovoo,
#                 H2B["vooo"],
#                 H2C["vooo"],
#                 H2B["ovoo"],
#                 H1A["ov"],
#                 H1B["ov"],
#                 ints["vB"]["oovv"],
#                 ints["vC"]["oovv"],
#                 ints["fA"]["oo"],
#                 ints["fA"]["vv"],
#                 ints["fB"]["oo"],
#                 ints["fB"]["vv"],
#                 H1A["oo"],
#                 H1A["vv"],
#                 H1B["oo"],
#                 H1B["vv"],
#                 H2A["voov"],
#                 H2B["ovov"],
#                 H2B["vovo"],
#                 H2B["oooo"],
#                 H2B["vvvv"],
#                 H2C["voov"],
#                 H2C["oooo"],
#                 H2C["vvvv"],
#                 D3B["O"],
#                 D3B["V"],
#                 D3C["O"],
#                 D3C["V"],
#                 D3D["O"],
#                 D3D["V"],
#                 sys["Nocc_a"],
#                 sys["Nunocc_a"],
#                 sys["Nocc_b"],
#                 sys["Nunocc_b"],
#             )
#
#             I2C_vvov = H2C["vvov"] + np.einsum(
#                 "me,abim->abie", H1B["ov"], cc_t["t2c"], optimize=True
#             )
#             chi2C_vvvo, chi2C_ovoo = calc_eomm23d_intermediates(cc_t, H2B, H2C, iroot)
#             dA_BBB, dB_BBB, dC_BBB, dD_BBB = crcc_loops.creomcc23d_opt(
#                 omega[iroot],
#                 cc_t["r0"][iroot],
#                 cc_t["t2c"],
#                 cc_t["r2c"][iroot],
#                 cc_t["l1b"][iroot + 1],
#                 cc_t["l2c"][iroot + 1],
#                 H2C["vooo"],
#                 I2C_vvov,
#                 H2C["vvov"],
#                 chi2C_vvvo,
#                 chi2C_ovoo,
#                 ints["vC"]["oovv"],
#                 H1B["ov"],
#                 H2C["vovv"],
#                 H2C["ooov"],
#                 ints["fB"]["oo"],
#                 ints["fB"]["vv"],
#                 H1B["oo"],
#                 H1B["vv"],
#                 H2C["voov"],
#                 H2C["oooo"],
#                 H2C["vvvv"],
#                 D3D["O"],
#                 D3D["V"],
#                 sys["Nocc_b"],
#                 sys["Nunocc_b"],
#             )
#
#             deltaA = dA_AAA + dA_AAB + dA_ABB + dA_BBB
#             deltaB = dB_AAA + dB_AAB + dB_ABB + dB_BBB
#             deltaC = dC_AAA + dC_AAB + dC_ABB + dC_BBB
#             deltaD = dD_AAA + dD_AAB + dD_ABB + dD_BBB
#
#         EcorrA = Ecorr + omega[iroot] + deltaA
#         EcorrB = Ecorr + omega[iroot] + deltaB
#         EcorrC = Ecorr + omega[iroot] + deltaC
#         EcorrD = Ecorr + omega[iroot] + deltaD
#
#         E23A = ints["Escf"] + EcorrA
#         VEE_A = (E23A - Ecrcc23[0]["A"]) * 27.211396641308
#         E23B = ints["Escf"] + EcorrB
#         VEE_B = (E23B - Ecrcc23[0]["B"]) * 27.211396641308
#         E23C = ints["Escf"] + EcorrC
#         VEE_C = (E23C - Ecrcc23[0]["C"]) * 27.211396641308
#         E23D = ints["Escf"] + EcorrD
#         VEE_D = (E23D - Ecrcc23[0]["D"]) * 27.211396641308
#
#         print("EOMCCSD = {} Eh".format(ints["Escf"] + Ecorr + omega[iroot]))
#         print(
#             "CR-CC(2,3)_A = {} Eh     Ecorr_A = {} Eh     Delta_A = {} Eh     VEE = {} eV".format(
#                 E23A, EcorrA, deltaA, VEE_A
#             )
#         )
#         print(
#             "CR-CC(2,3)_B = {} Eh     Ecorr_B = {} Eh     Delta_B = {} Eh     VEE = {} eV".format(
#                 E23B, EcorrB, deltaB, VEE_B
#             )
#         )
#         print(
#             "CR-CC(2,3)_C = {} Eh     Ecorr_C = {} Eh     Delta_C = {} Eh     VEE = {} eV".format(
#                 E23C, EcorrC, deltaC, VEE_C
#             )
#         )
#         print(
#             "CR-CC(2,3)_D = {} Eh     Ecorr_D = {} Eh     Delta_D = {} Eh     VEE = {} eV".format(
#                 E23D, EcorrD, deltaD, VEE_D
#             )
#         )
#
#         Ecrcc23[iroot + 1] = {"A": E23A, "B": E23B, "C": E23C, "D": E23D}
#         delta23[iroot + 1] = {"A": deltaA, "B": deltaB, "C": deltaC, "D": deltaD}
#
#         t_end = time.time()
#         minutes, seconds = divmod(t_end - t_start, 60)
#         print("finished in ({:0.2f}m  {:0.2f}s)".format(minutes, seconds))
#
#     return Ecrcc23, delta23

def calc_eomm23a_intermediates(cc_t, H2A, H2B, iroot):

    Q1 = np.einsum("mnef,fn->me", H2A["oovv"], cc_t["r1a"][iroot], optimize=True)
    Q1 += np.einsum("mnef,fn->me", H2B["oovv"], cc_t["r1b"][iroot], optimize=True)
    I1 = np.einsum("amje,bm->abej", H2A["voov"], cc_t["r1a"][iroot], optimize=True)
    I1 += np.einsum("amfe,bejm->abfj", H2A["vovv"], cc_t["r2a"][iroot], optimize=True)
    I1 += np.einsum("amfe,bejm->abfj", H2B["vovv"], cc_t["r2b"][iroot], optimize=True)
    I1 -= np.transpose(I1, (1, 0, 2, 3))
    I2 = np.einsum("abfe,ej->abfj", H2A["vvvv"], cc_t["r1a"][iroot], optimize=True)
    I2 += 0.5 * np.einsum(
        "nmje,abmn->abej", H2A["ooov"], cc_t["r2a"][iroot], optimize=True
    )
    I2 -= np.einsum("me,abmj->abej", Q1, cc_t["t2a"], optimize=True)
    chi2A_vvvo = I1 + I2
    I1 = -np.einsum("bmie,ej->mbij", H2A["voov"], cc_t["r1a"][iroot], optimize=True)
    I1 += np.einsum("nmie,bejm->nbij", H2A["ooov"], cc_t["r2a"][iroot], optimize=True)
    I1 += np.einsum("nmie,bejm->nbij", H2B["ooov"], cc_t["r2b"][iroot], optimize=True)
    I1 -= np.transpose(I1, (0, 1, 3, 2))
    I2 = -1.0 * np.einsum(
        "nmij,bm->nbij", H2A["oooo"], cc_t["r1a"][iroot], optimize=True
    )
    I2 += 0.5 * np.einsum(
        "bmfe,efij->mbij", H2A["vovv"], cc_t["r2a"][iroot], optimize=True
    )
    chi2A_ovoo = I1 + I2

    return chi2A_vvvo, chi2A_ovoo


def calc_eomcc23b_intermediates(cc_t, H2A, H2B, H2C, iroot):

    Q1 = np.einsum(
        "mnef,fn->me", H2A["oovv"], cc_t["r1a"][iroot], optimize=True
    ) + np.einsum("mnef,fn->me", H2B["oovv"], cc_t["r1b"][iroot], optimize=True)
    Q2 = np.einsum(
        "nmfe,fn->me", H2B["oovv"], cc_t["r1a"][iroot], optimize=True
    ) + np.einsum("nmfe,fn->me", H2C["oovv"], cc_t["r1b"][iroot], optimize=True)
    # Intermediate 1: X2B(bcek)*Y2A(aeij) -> Z3B(abcijk)
    Int1 = -1.0 * np.einsum(
        "mcek,bm->bcek", H2B["ovvo"], cc_t["r1a"][iroot], optimize=True
    )
    Int1 -= np.einsum("bmek,cm->bcek", H2B["vovo"], cc_t["r1b"][iroot], optimize=True)
    Int1 += np.einsum("bcfe,ek->bcfk", H2B["vvvv"], cc_t["r1b"][iroot], optimize=True)
    Int1 += np.einsum("mnek,bcmn->bcek", H2B["oovo"], cc_t["r2b"][iroot], optimize=True)
    Int1 += np.einsum("bmfe,ecmk->bcfk", H2A["vovv"], cc_t["r2b"][iroot], optimize=True)
    Int1 += np.einsum("bmfe,ecmk->bcfk", H2B["vovv"], cc_t["r2c"][iroot], optimize=True)
    Int1 -= np.einsum("mcfe,bemk->bcfk", H2B["ovvv"], cc_t["r2b"][iroot], optimize=True)
    # Intermediate 2: X2B(ncjk)*Y2A(abin) -> Z3B(abcijk)
    Int2 = -1.0 * np.einsum(
        "nmjk,cm->ncjk", H2B["oooo"], cc_t["r1b"][iroot], optimize=True
    )
    Int2 += np.einsum("mcje,ek->mcjk", H2B["ovov"], cc_t["r1b"][iroot], optimize=True)
    Int2 += np.einsum("mcek,ej->mcjk", H2B["ovvo"], cc_t["r1a"][iroot], optimize=True)
    Int2 += np.einsum("mcef,efjk->mcjk", H2B["ovvv"], cc_t["r2b"][iroot], optimize=True)
    Int2 += np.einsum("nmje,ecmk->ncjk", H2A["ooov"], cc_t["r2b"][iroot], optimize=True)
    Int2 += np.einsum("nmje,ecmk->ncjk", H2B["ooov"], cc_t["r2c"][iroot], optimize=True)
    Int2 -= np.einsum("nmek,ecjm->ncjk", H2B["oovo"], cc_t["r2b"][iroot], optimize=True)
    # Intermediate 3: X2A(abej)*Y2B(ecik) -> Z3B(abcijk)
    Int3 = np.einsum(
        "amje,bm->abej", H2A["voov"], cc_t["r1a"][iroot], optimize=True
    )  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int3 += 0.5 * np.einsum(
        "abfe,ej->abfj", H2A["vvvv"], cc_t["r1a"][iroot], optimize=True
    )  # (*) added factor 1/2 to compensate A(ab)
    Int3 += 0.25 * np.einsum(
        "nmje,abmn->abej", H2A["ooov"], cc_t["r2a"][iroot], optimize=True
    )  # (*) added factor 1/2 to compensate A(ab)
    Int3 += np.einsum("amfe,bejm->abfj", H2A["vovv"], cc_t["r2a"][iroot], optimize=True)
    Int3 += np.einsum("amfe,bejm->abfj", H2B["vovv"], cc_t["r2b"][iroot], optimize=True)
    Int3 -= 0.5 * np.einsum(
        "me,abmj->abej", Q1, cc_t["t2a"], optimize=True
    )  # (*) added factor 1/2 to compensate A(ab)
    Int3 -= np.transpose(Int3, (1, 0, 2, 3))
    # Intermediate 4: X2A(bnji)*Y2B(acnk) -> Z3B(abcijk)
    Int4 = -0.5 * np.einsum(
        "nmij,bm->bnji", H2A["oooo"], cc_t["r1a"][iroot], optimize=True
    )  # (*) added factor 1/2 to compenate A(ij)
    Int4 -= np.einsum(
        "bmie,ej->bmji", H2A["voov"], cc_t["r1a"][iroot], optimize=True
    )  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int4 += 0.25 * np.einsum(
        "bmfe,efij->bmji", H2A["vovv"], cc_t["r2a"][iroot], optimize=True
    )  # (*) added factor 1/2 to compensate A(ij)
    Int4 += np.einsum("nmie,bejm->bnji", H2A["ooov"], cc_t["r2a"][iroot], optimize=True)
    Int4 += np.einsum("nmie,bejm->bnji", H2B["ooov"], cc_t["r2b"][iroot], optimize=True)
    Int4 += 0.5 * np.einsum(
        "me,ebij->bmji", Q1, cc_t["t2a"], optimize=True
    )  # (*) added factor 1/2 to compensate A(ij)
    Int4 -= np.transpose(Int4, (0, 1, 3, 2))
    # Intermediate 5: X2B(bcje)*Y2B(aeik) -> Z3B(abcijk)
    Int5 = -1.0 * np.einsum(
        "mcje,bm->bcje", H2B["ovov"], cc_t["r1a"][iroot], optimize=True
    )
    Int5 -= np.einsum("bmje,cm->bcje", H2B["voov"], cc_t["r1b"][iroot], optimize=True)
    Int5 += np.einsum("bcef,ej->bcjf", H2B["vvvv"], cc_t["r1a"][iroot], optimize=True)
    Int5 += np.einsum("mnjf,bcmn->bcjf", H2B["ooov"], cc_t["r2b"][iroot], optimize=True)
    Int5 += np.einsum("mcef,bejm->bcjf", H2B["ovvv"], cc_t["r2a"][iroot], optimize=True)
    Int5 += np.einsum("cmfe,bejm->bcjf", H2C["vovv"], cc_t["r2b"][iroot], optimize=True)
    Int5 -= np.einsum("bmef,ecjm->bcjf", H2B["vovv"], cc_t["r2b"][iroot], optimize=True)
    # Intermediate 6: X2B(bnjk)*Y2B(acin) -> Z3B(abcijk)
    Int6 = -1.0 * np.einsum(
        "mnjk,bm->bnjk", H2B["oooo"], cc_t["r1a"][iroot], optimize=True
    )
    Int6 += np.einsum("bmje,ek->bmjk", H2B["voov"], cc_t["r1b"][iroot], optimize=True)
    Int6 += np.einsum("bmek,ej->bmjk", H2B["vovo"], cc_t["r1a"][iroot], optimize=True)
    Int6 += np.einsum("bnef,efjk->bnjk", H2B["vovv"], cc_t["r2b"][iroot], optimize=True)
    Int6 += np.einsum("mnek,bejm->bnjk", H2B["oovo"], cc_t["r2a"][iroot], optimize=True)
    Int6 += np.einsum("nmke,bejm->bnjk", H2C["ooov"], cc_t["r2b"][iroot], optimize=True)
    Int6 -= np.einsum("nmje,benk->bmjk", H2B["ooov"], cc_t["r2b"][iroot], optimize=True)
    Int6 += np.einsum("me,bejk->bmjk", Q2, cc_t["t2b"], optimize=True)

    return Int1, Int2, Int3, Int4, Int5, Int6


def calc_eomcc23c_intermediates(cc_t, H2A, H2B, H2C, iroot):
    Q1 = np.einsum(
        "mnef,fn->me", H2C["oovv"], cc_t["r1b"][iroot], optimize=True
    ) + np.einsum("nmfe,fn->me", H2B["oovv"], cc_t["r1a"][iroot], optimize=True)
    Q2 = np.einsum(
        "mnef,fn->me", H2B["oovv"], cc_t["r1b"][iroot], optimize=True
    ) + np.einsum("nmfe,fn->me", H2A["oovv"], cc_t["r1a"][iroot], optimize=True)
    # Intermediate 1: X2B(cbke)*Y2C(aeij) -> Z3C(cbakji)
    Int1 = -1.0 * np.einsum(
        "cmke,bm->cbke", H2B["voov"], cc_t["r1b"][iroot], optimize=True
    )
    Int1 -= np.einsum("mbke,cm->cbke", H2B["ovov"], cc_t["r1a"][iroot], optimize=True)
    Int1 += np.einsum("cbef,ek->cbkf", H2B["vvvv"], cc_t["r1a"][iroot], optimize=True)
    Int1 += np.einsum("nmke,cbnm->cbke", H2B["ooov"], cc_t["r2b"][iroot], optimize=True)
    Int1 += np.einsum("bmfe,cekm->cbkf", H2C["vovv"], cc_t["r2b"][iroot], optimize=True)
    Int1 += np.einsum("mbef,ecmk->cbkf", H2B["ovvv"], cc_t["r2a"][iroot], optimize=True)
    Int1 -= np.einsum("cmef,ebkm->cbkf", H2B["vovv"], cc_t["r2b"][iroot], optimize=True)
    # Intermediate 2: X2B(cnkj)*Y2C(abin) -> Z3C(cbakji)
    Int2 = -1.0 * np.einsum(
        "mnkj,cm->cnkj", H2B["oooo"], cc_t["r1a"][iroot], optimize=True
    )
    Int2 += np.einsum("cmej,ek->cmkj", H2B["vovo"], cc_t["r1a"][iroot], optimize=True)
    Int2 += np.einsum("cmke,ej->cmkj", H2B["voov"], cc_t["r1b"][iroot], optimize=True)
    Int2 += np.einsum("cmfe,fekj->cmkj", H2B["vovv"], cc_t["r2b"][iroot], optimize=True)
    Int2 += np.einsum("nmje,cekm->cnkj", H2C["ooov"], cc_t["r2b"][iroot], optimize=True)
    Int2 += np.einsum("mnej,ecmk->cnkj", H2B["oovo"], cc_t["r2a"][iroot], optimize=True)
    Int2 -= np.einsum("mnke,cemj->cnkj", H2B["ooov"], cc_t["r2b"][iroot], optimize=True)
    # Intermediate 3: X2C(abej)*Y2B(ceki) -> Z3C(cbakji)
    Int3 = np.einsum(
        "amje,bm->abej", H2C["voov"], cc_t["r1b"][iroot], optimize=True
    )  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int3 += 0.5 * np.einsum(
        "abfe,ej->abfj", H2C["vvvv"], cc_t["r1b"][iroot], optimize=True
    )  # (*) added factor 1/2 to compensate A(ab)
    Int3 += 0.25 * np.einsum(
        "nmje,abmn->abej", H2C["ooov"], cc_t["r2c"][iroot], optimize=True
    )  # (*) added factor 1/2 to compensate A(ab)
    Int3 += np.einsum("amfe,bejm->abfj", H2C["vovv"], cc_t["r2c"][iroot], optimize=True)
    Int3 += np.einsum("maef,ebmj->abfj", H2B["ovvv"], cc_t["r2b"][iroot], optimize=True)
    Int3 -= 0.5 * np.einsum(
        "me,abmj->abej", Q1, cc_t["t2c"], optimize=True
    )  # (*) added factor 1/2 to compensate A(ab)
    Int3 -= np.transpose(Int3, (1, 0, 2, 3))
    # Intermediate 4: X2C(bnji)*Y2B(cakn) -> Z3C(cbakji)
    Int4 = -0.5 * np.einsum(
        "nmij,bm->bnji", H2C["oooo"], cc_t["r1b"][iroot], optimize=True
    )  # (*) added factor 1/2 to compenate A(ij)
    Int4 -= np.einsum(
        "bmie,ej->bmji", H2C["voov"], cc_t["r1b"][iroot], optimize=True
    )  # (*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int4 += 0.25 * np.einsum(
        "bmfe,efij->bmji", H2C["vovv"], cc_t["r2c"][iroot], optimize=True
    )  # (*) added factor 1/2 to compensate A(ij)
    Int4 += np.einsum("nmie,bejm->bnji", H2C["ooov"], cc_t["r2c"][iroot], optimize=True)
    Int4 += np.einsum("mnei,ebmj->bnji", H2B["oovo"], cc_t["r2b"][iroot], optimize=True)
    Int4 += 0.5 * np.einsum(
        "me,ebij->bmji", Q1, cc_t["t2c"], optimize=True
    )  # (*) added factor 1/2 to compensate A(ij)
    Int4 -= np.transpose(Int4, (0, 1, 3, 2))
    # Intermediate 5: X2B(cbej)*Y2B(eaki) -> Z3C(cbakji)
    Int5 = -1.0 * np.einsum(
        "cmej,bm->cbej", H2B["vovo"], cc_t["r1b"][iroot], optimize=True
    )
    Int5 -= np.einsum("mbej,cm->cbej", H2B["ovvo"], cc_t["r1a"][iroot], optimize=True)
    Int5 += np.einsum("cbfe,ej->cbfj", H2B["vvvv"], cc_t["r1b"][iroot], optimize=True)
    Int5 += np.einsum("nmfj,cbnm->cbfj", H2B["oovo"], cc_t["r2b"][iroot], optimize=True)
    Int5 += np.einsum("cmfe,bejm->cbfj", H2B["vovv"], cc_t["r2c"][iroot], optimize=True)
    Int5 += np.einsum("cmfe,ebmj->cbfj", H2A["vovv"], cc_t["r2b"][iroot], optimize=True)
    Int5 -= np.einsum("mbfe,cemj->cbfj", H2B["ovvv"], cc_t["r2b"][iroot], optimize=True)
    # Intermediate 6: X2B(nbkj)*Y2B(cani) -> Z3C(cbakji)
    Int6 = -1.0 * np.einsum(
        "nmkj,bm->nbkj", H2B["oooo"], cc_t["r1b"][iroot], optimize=True
    )
    Int6 += np.einsum("mbej,ek->mbkj", H2B["ovvo"], cc_t["r1a"][iroot], optimize=True)
    Int6 += np.einsum("mbke,ej->mbkj", H2B["ovov"], cc_t["r1b"][iroot], optimize=True)
    Int6 += np.einsum("nbfe,fekj->nbkj", H2B["ovvv"], cc_t["r2b"][iroot], optimize=True)
    Int6 += np.einsum("nmke,bejm->nbkj", H2B["ooov"], cc_t["r2c"][iroot], optimize=True)
    Int6 += np.einsum("nmke,ebmj->nbkj", H2A["ooov"], cc_t["r2b"][iroot], optimize=True)
    Int6 -= np.einsum("mnej,ebkn->mbkj", H2B["oovo"], cc_t["r2b"][iroot], optimize=True)
    Int6 += np.einsum("me,ebkj->mbkj", Q2, cc_t["t2b"], optimize=True)

    return Int1, Int2, Int3, Int4, Int5, Int6


def calc_eomm23d_intermediates(cc_t, H2B, H2C, iroot):

    Q1 = np.einsum("mnef,fn->me", H2C["oovv"], cc_t["r1b"][iroot], optimize=True)
    Q1 += np.einsum("nmfe,fn->me", H2B["oovv"], cc_t["r1a"][iroot], optimize=True)
    I1 = np.einsum("amje,bm->abej", H2C["voov"], cc_t["r1b"][iroot], optimize=True)
    I1 += np.einsum("amfe,bejm->abfj", H2C["vovv"], cc_t["r2c"][iroot], optimize=True)
    I1 += np.einsum("maef,ebmj->abfj", H2B["ovvv"], cc_t["r2b"][iroot], optimize=True)
    I1 -= np.transpose(I1, (1, 0, 2, 3))
    I2 = np.einsum("abfe,ej->abfj", H2C["vvvv"], cc_t["r1b"][iroot], optimize=True)
    I2 += 0.5 * np.einsum(
        "nmje,abmn->abej", H2C["ooov"], cc_t["r2c"][iroot], optimize=True
    )
    I2 -= np.einsum("me,abmj->abej", Q1, cc_t["t2c"], optimize=True)
    chi2C_vvvo = I1 + I2
    I1 = -np.einsum("bmie,ej->mbij", H2C["voov"], cc_t["r1b"][iroot], optimize=True)
    I1 += np.einsum("nmie,bejm->nbij", H2C["ooov"], cc_t["r2c"][iroot], optimize=True)
    I1 += np.einsum("mnei,ebmj->nbij", H2B["oovo"], cc_t["r2b"][iroot], optimize=True)
    I1 -= np.transpose(I1, (0, 1, 3, 2))
    I2 = -1.0 * np.einsum(
        "nmij,bm->nbij", H2C["oooo"], cc_t["r1b"][iroot], optimize=True
    )
    I2 += 0.5 * np.einsum(
        "bmfe,efij->mbij", H2C["vovv"], cc_t["r2c"][iroot], optimize=True
    )
    chi2C_ovoo = I1 + I2

    return chi2C_vvvo, chi2C_ovoo