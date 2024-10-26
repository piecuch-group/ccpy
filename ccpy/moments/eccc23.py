"""Functions to calculate the internal and external triples corrections to the ec-CC-II computations."""
import time

import numpy as np
from ccpy.energy.cc_energy import get_cc_energy
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.lib.core import ccp3_loops


def calc_ccp3(T, L, H, H0, system, pspace, use_RHF=False):
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

    # get LT intermediates
    #X = build_left_ccsdt_intermediates(L, T, system)

    #### aaa correction ####
    MM23A = build_M3A(T, H)
    L3A = build_L3A(L, H, flag_2ba=True)
    dA_aaa, dB_aaa, dC_aaa, dD_aaa = ccp3_loops.crcc23a_p_full(
        pspace[0]["aaa"],
        MM23A, L3A, 0.0,
        H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
        H.aa.voov, H.aa.oooo, H.aa.vvvv,
        d3aaa_o, d3aaa_v,
        system.noccupied_alpha, system.nunoccupied_alpha,
    )

    #### aab correction ####
    MM23B = build_M3B(T, H)
    L3B = build_L3B(L, H, flag_2ba=True)
    dA_aab, dB_aab, dC_aab, dD_aab = ccp3_loops.crcc23b_p_full(
        pspace[0]["aab"],
        MM23B, L3B, 0.0,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H.aa.voov, H.aa.oooo, H.aa.vvvv,
        H.ab.ovov, H.ab.vovo,
        H.ab.oooo, H.ab.vvvv,
        H.bb.voov,
        d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
        system.noccupied_alpha, system.nunoccupied_alpha,
        system.noccupied_beta, system.nunoccupied_beta,
    )

    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
        correction_B = 2.0 * dB_aaa + 2.0 * dB_aab
        correction_C = 2.0 * dC_aaa + 2.0 * dC_aab
        correction_D = 2.0 * dD_aaa + 2.0 * dD_aab

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
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

    print('   CC(P;3) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   CC(P) = {:>10.10f}".format(system.reference_energy + cc_energy))
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

def build_M3A(T, H):
    """
    Update t3a amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3))_C|0>.
    """

    # <ijkabc | H(2) | 0 > + (VT3)_C intermediates
    # Recall that we are using HBar CCSDT, so the vooo and vvov parts have T3 in it already!
    I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)

    # MM(2,3)A
    MM23A = -0.25 * np.einsum("amij,bcmk->abcijk", H.aa.vooo, T.aa, optimize=True)
    MM23A += 0.25 * np.einsum("abie,ecjk->abcijk", I2A_vvov, T.aa, optimize=True)
    # (HBar*T3)_C
    MM23A -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.a.oo, T.aaa, optimize=True)
    MM23A += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.a.vv, T.aaa, optimize=True)
    MM23A += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa, optimize=True)
    MM23A += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa, optimize=True)
    MM23A += 0.25 * np.einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa, optimize=True)
    MM23A += 0.25 * np.einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab, optimize=True)

    MM23A -= np.transpose(MM23A, (0, 1, 2, 3, 5, 4)) # (jk)
    MM23A -= np.transpose(MM23A, (0, 1, 2, 4, 3, 5)) + np.transpose(MM23A, (0, 1, 2, 5, 4, 3)) # (i/jk)
    MM23A -= np.transpose(MM23A, (0, 2, 1, 3, 4, 5)) # (bc)
    MM23A -= np.transpose(MM23A, (2, 1, 0, 3, 4, 5)) + np.transpose(MM23A, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return MM23A


# @profile
def build_M3B(T, H):
    """
    Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N e^(T1+T2+T3))_C|0>.
    """
    # <ijk~abc~ | H(2) | 0 > + (VT3)_C intermediates
    # Recall that we are using HBar CCSDT, so the vooo and vvov parts have T3 in it already!
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)

    # MM(2,3)B
    MM23B = 0.5 * np.einsum("bcek,aeij->abcijk", H.ab.vvvo, T.aa, optimize=True)
    MM23B -= 0.5 * np.einsum("mcjk,abim->abcijk", I2B_ovoo, T.aa, optimize=True)
    MM23B += np.einsum("acie,bejk->abcijk", H.ab.vvov, T.ab, optimize=True)
    MM23B -= np.einsum("amik,bcjm->abcijk", I2B_vooo, T.ab, optimize=True)
    MM23B += 0.5 * np.einsum("abie,ecjk->abcijk", H.aa.vvov, T.ab, optimize=True)
    MM23B -= 0.5 * np.einsum("amij,bcmk->abcijk", I2A_vooo, T.ab, optimize=True)
    # (HBar*T3)_C
    MM23B -= 0.5 * np.einsum("mi,abcmjk->abcijk", H.a.oo, T.aab, optimize=True)
    MM23B -= 0.25 * np.einsum("mk,abcijm->abcijk", H.b.oo, T.aab, optimize=True)
    MM23B += 0.5 * np.einsum("ae,ebcijk->abcijk", H.a.vv, T.aab, optimize=True)
    MM23B += 0.25 * np.einsum("ce,abeijk->abcijk", H.b.vv, T.aab, optimize=True)
    MM23B += 0.125 * np.einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aab, optimize=True)
    MM23B += 0.5 * np.einsum("mnjk,abcimn->abcijk", H.ab.oooo, T.aab, optimize=True)
    MM23B += 0.125 * np.einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aab, optimize=True)
    MM23B += 0.5 * np.einsum("bcef,aefijk->abcijk", H.ab.vvvv, T.aab, optimize=True)
    MM23B += np.einsum("amie,ebcmjk->abcijk", H.aa.voov, T.aab, optimize=True)
    MM23B += np.einsum("amie,becjmk->abcijk", H.ab.voov, T.abb, optimize=True)
    MM23B += 0.25 * np.einsum("mcek,abeijm->abcijk", H.ab.ovvo, T.aaa, optimize=True)
    MM23B += 0.25 * np.einsum("cmke,abeijm->abcijk", H.bb.voov, T.aab, optimize=True)
    MM23B -= 0.5 * np.einsum("amek,ebcijm->abcijk", H.ab.vovo, T.aab, optimize=True)
    MM23B -= 0.5 * np.einsum("mcie,abemjk->abcijk", H.ab.ovov, T.aab, optimize=True)

    MM23B -= np.transpose(MM23B, (1, 0, 2, 3, 4, 5))
    MM23B -= np.transpose(MM23B, (0, 1, 2, 4, 3, 5))

    return MM23B


def build_L3A(L, H, X=None, flag_2ba=True):

    # < 0 | L1 * H(2) | ijkabc >
    L3A = (9.0 / 36.0) * np.einsum("ai,jkbc->abcijk", L.a, H.aa.oovv, optimize=True)

    # < 0 | L2 * H(2) | ijkabc >
    L3A += (9.0 / 36.0) * np.einsum("bcjk,ia->abcijk", L.aa, H.a.ov, optimize=True)

    L3A += (9.0 / 36.0) * np.einsum("ebij,ekac->abcijk", L.aa, H.aa.vovv, optimize=True)
    L3A -= (9.0 / 36.0) * np.einsum("abmj,ikmc->abcijk", L.aa, H.aa.ooov, optimize=True)

    if not flag_2ba:
        # < 0 | L3 * H(2) | ijkabc >
        L3A += (3.0 / 36.0) * np.einsum("ea,ebcijk->abcijk", H.a.vv, L.aaa, optimize=True)
        L3A -= (3.0 / 36.0) * np.einsum("im,abcmjk->abcijk", H.a.oo, L.aaa, optimize=True)
        L3A += (9.0 / 36.0) * np.einsum("eima,ebcmjk->abcijk", H.aa.voov, L.aaa, optimize=True)
        L3A += (9.0 / 36.0) * np.einsum("ieam,bcejkm->abcijk", H.ab.ovvo, L.aab, optimize=True)
        L3A += (3.0 / 72.0) * np.einsum("ijmn,abcmnk->abcijk", H.aa.oooo, L.aaa, optimize=True)
        L3A += (3.0 / 72.0) * np.einsum("efab,efcijk->abcijk", H.aa.vvvv, L.aaa, optimize=True)

        L3A += (9.0 / 36.0) * np.einsum("ijeb,ekac->abcijk", H.aa.oovv, X.aa.vovv, optimize=True)
        L3A -= (9.0 / 36.0) * np.einsum("mjab,ikmc->abcijk", H.aa.oovv, X.aa.ooov, optimize=True)

    L3A -= np.transpose(L3A, (0, 1, 2, 3, 5, 4)) # (jk)
    L3A -= np.transpose(L3A, (0, 1, 2, 4, 3, 5)) + np.transpose(L3A, (0, 1, 2, 5, 4, 3)) # (i/jk)
    L3A -= np.transpose(L3A, (0, 2, 1, 3, 4, 5)) # (bc)
    L3A -= np.transpose(L3A, (2, 1, 0, 3, 4, 5)) + np.transpose(L3A, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return L3A


def build_L3B(L, H, X=None, flag_2ba=True):

    # < 0 | L1 * H(2) | ijk~abc~ >
    L3B = np.einsum("ai,jkbc->abcijk", L.a, H.ab.oovv, optimize=True)
    L3B += 0.25 * np.einsum("ck,ijab->abcijk", L.b, H.aa.oovv, optimize=True)

    # < 0 | L2 * H(2) | ijk~abc~ >
    L3B += np.einsum("bcjk,ia->abcijk", L.ab, H.a.ov, optimize=True)
    L3B += 0.25 * np.einsum("abij,kc->abcijk", L.aa, H.b.ov, optimize=True)

    L3B += 0.5 * np.einsum("ekbc,aeij->abcijk", H.ab.vovv, L.aa, optimize=True)
    L3B -= 0.5 * np.einsum("jkmc,abim->abcijk", H.ab.ooov, L.aa, optimize=True)
    L3B += np.einsum("ieac,bejk->abcijk", H.ab.ovvv, L.ab, optimize=True)
    L3B -= np.einsum("ikam,bcjm->abcijk", H.ab.oovo, L.ab, optimize=True)
    L3B += 0.5 * np.einsum("eiba,ecjk->abcijk", H.aa.vovv, L.ab, optimize=True)
    L3B -= 0.5 * np.einsum("jima,bcmk->abcijk", H.aa.ooov, L.ab, optimize=True)

    L3B += 0.5 * np.einsum("ekbc,ijae->abcijk", X.ab.vovv, H.aa.oovv, optimize=True)
    L3B -= 0.5 * np.einsum("jkmc,imab->abcijk", X.ab.ooov, H.aa.oovv, optimize=True)
    L3B += np.einsum("ieac,jkbe->abcijk", X.ab.ovvv, H.ab.oovv, optimize=True)
    L3B -= np.einsum("ikam,jmbc->abcijk", X.ab.oovo, H.ab.oovv, optimize=True)
    L3B += 0.5 * np.einsum("eiba,jkec->abcijk", X.aa.vovv, H.ab.oovv, optimize=True)
    L3B -= 0.5 * np.einsum("jima,mkbc->abcijk", X.aa.ooov, H.ab.oovv, optimize=True)

    if not flag_2ba:
        # < 0 | L3 * H(2) | ijk~abc~ >
        L3B -= 0.5 * np.einsum("im,abcmjk->abcijk", H.a.oo, L.aab, optimize=True)
        L3B -= 0.25 * np.einsum("km,abcijm->abcijk", H.b.oo, L.aab, optimize=True)
        L3B += 0.5 * np.einsum("ea,ebcijk->abcijk", H.a.vv, L.aab, optimize=True)
        L3B += 0.25 * np.einsum("ec,abeijk->abcijk", H.b.vv, L.aab, optimize=True)
        L3B += 0.125 * np.einsum("ijmn,abcmnk->abcijk", H.aa.oooo, L.aab, optimize=True)
        L3B += 0.5 * np.einsum("jkmn,abcimn->abcijk", H.ab.oooo, L.aab, optimize=True)
        L3B += 0.125 * np.einsum("efab,efcijk->abcijk", H.aa.vvvv, L.aab, optimize=True)
        L3B += 0.5 * np.einsum("efbc,aefijk->abcijk", H.ab.vvvv, L.aab, optimize=True)
        L3B += np.einsum("eima,ebcmjk->abcijk", H.aa.voov, L.aab, optimize=True)
        L3B += np.einsum("ieam,becjmk->abcijk", H.ab.ovvo, L.abb, optimize=True)
        L3B += 0.25 * np.einsum("ekmc,abeijm->abcijk", H.ab.voov, L.aaa, optimize=True)
        L3B += 0.25 * np.einsum("ekmc,abeijm->abcijk", H.bb.voov, L.aab, optimize=True)
        L3B -= 0.5 * np.einsum("ekam,ebcijm->abcijk", H.ab.vovo, L.aab, optimize=True)
        L3B -= 0.5 * np.einsum("iemc,abemjk->abcijk", H.ab.ovov, L.aab, optimize=True)

    L3B -= np.transpose(L3B, (1, 0, 2, 3, 4, 5))
    L3B -= np.transpose(L3B, (0, 1, 2, 4, 3, 5))

    return L3B