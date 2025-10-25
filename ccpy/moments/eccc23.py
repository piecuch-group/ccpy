"""Functions to calculate the internal and external triples corrections to the ec-CC-II computations."""
import time
from ccpy.utilities.linear_algebra import ccpy_einsum

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
    I2A_vvov = H.aa.vvov + ccpy_einsum("me,abim->abie", H.a.ov, T.aa)

    # MM(2,3)A
    MM23A = -0.25 * ccpy_einsum("amij,bcmk->abcijk", H.aa.vooo, T.aa)
    MM23A += 0.25 * ccpy_einsum("abie,ecjk->abcijk", I2A_vvov, T.aa)
    # (HBar*T3)_C
    MM23A -= (1.0 / 12.0) * ccpy_einsum("mk,abcijm->abcijk", H.a.oo, T.aaa)
    MM23A += (1.0 / 12.0) * ccpy_einsum("ce,abeijk->abcijk", H.a.vv, T.aaa)
    MM23A += (1.0 / 24.0) * ccpy_einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aaa)
    MM23A += (1.0 / 24.0) * ccpy_einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aaa)
    MM23A += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.aa.voov, T.aaa)
    MM23A += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.ab.voov, T.aab)

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
    I2A_vooo = H.aa.vooo - ccpy_einsum("me,aeij->amij", H.a.ov, T.aa)
    I2B_ovoo = H.ab.ovoo - ccpy_einsum("me,ecjk->mcjk", H.a.ov, T.ab)
    I2B_vooo = H.ab.vooo - ccpy_einsum("me,aeik->amik", H.b.ov, T.ab)

    # MM(2,3)B
    MM23B = 0.5 * ccpy_einsum("bcek,aeij->abcijk", H.ab.vvvo, T.aa)
    MM23B -= 0.5 * ccpy_einsum("mcjk,abim->abcijk", I2B_ovoo, T.aa)
    MM23B += ccpy_einsum("acie,bejk->abcijk", H.ab.vvov, T.ab)
    MM23B -= ccpy_einsum("amik,bcjm->abcijk", I2B_vooo, T.ab)
    MM23B += 0.5 * ccpy_einsum("abie,ecjk->abcijk", H.aa.vvov, T.ab)
    MM23B -= 0.5 * ccpy_einsum("amij,bcmk->abcijk", I2A_vooo, T.ab)
    # (HBar*T3)_C
    MM23B -= 0.5 * ccpy_einsum("mi,abcmjk->abcijk", H.a.oo, T.aab)
    MM23B -= 0.25 * ccpy_einsum("mk,abcijm->abcijk", H.b.oo, T.aab)
    MM23B += 0.5 * ccpy_einsum("ae,ebcijk->abcijk", H.a.vv, T.aab)
    MM23B += 0.25 * ccpy_einsum("ce,abeijk->abcijk", H.b.vv, T.aab)
    MM23B += 0.125 * ccpy_einsum("mnij,abcmnk->abcijk", H.aa.oooo, T.aab)
    MM23B += 0.5 * ccpy_einsum("mnjk,abcimn->abcijk", H.ab.oooo, T.aab)
    MM23B += 0.125 * ccpy_einsum("abef,efcijk->abcijk", H.aa.vvvv, T.aab)
    MM23B += 0.5 * ccpy_einsum("bcef,aefijk->abcijk", H.ab.vvvv, T.aab)
    MM23B += ccpy_einsum("amie,ebcmjk->abcijk", H.aa.voov, T.aab)
    MM23B += ccpy_einsum("amie,becjmk->abcijk", H.ab.voov, T.abb)
    MM23B += 0.25 * ccpy_einsum("mcek,abeijm->abcijk", H.ab.ovvo, T.aaa)
    MM23B += 0.25 * ccpy_einsum("cmke,abeijm->abcijk", H.bb.voov, T.aab)
    MM23B -= 0.5 * ccpy_einsum("amek,ebcijm->abcijk", H.ab.vovo, T.aab)
    MM23B -= 0.5 * ccpy_einsum("mcie,abemjk->abcijk", H.ab.ovov, T.aab)

    MM23B -= np.transpose(MM23B, (1, 0, 2, 3, 4, 5))
    MM23B -= np.transpose(MM23B, (0, 1, 2, 4, 3, 5))

    return MM23B


def build_L3A(L, H, X=None, flag_2ba=True):

    # < 0 | L1 * H(2) | ijkabc >
    L3A = (9.0 / 36.0) * ccpy_einsum("ai,jkbc->abcijk", L.a, H.aa.oovv)

    # < 0 | L2 * H(2) | ijkabc >
    L3A += (9.0 / 36.0) * ccpy_einsum("bcjk,ia->abcijk", L.aa, H.a.ov)

    L3A += (9.0 / 36.0) * ccpy_einsum("ebij,ekac->abcijk", L.aa, H.aa.vovv)
    L3A -= (9.0 / 36.0) * ccpy_einsum("abmj,ikmc->abcijk", L.aa, H.aa.ooov)

    if not flag_2ba:
        # < 0 | L3 * H(2) | ijkabc >
        L3A += (3.0 / 36.0) * ccpy_einsum("ea,ebcijk->abcijk", H.a.vv, L.aaa)
        L3A -= (3.0 / 36.0) * ccpy_einsum("im,abcmjk->abcijk", H.a.oo, L.aaa)
        L3A += (9.0 / 36.0) * ccpy_einsum("eima,ebcmjk->abcijk", H.aa.voov, L.aaa)
        L3A += (9.0 / 36.0) * ccpy_einsum("ieam,bcejkm->abcijk", H.ab.ovvo, L.aab)
        L3A += (3.0 / 72.0) * ccpy_einsum("ijmn,abcmnk->abcijk", H.aa.oooo, L.aaa)
        L3A += (3.0 / 72.0) * ccpy_einsum("efab,efcijk->abcijk", H.aa.vvvv, L.aaa)

        L3A += (9.0 / 36.0) * ccpy_einsum("ijeb,ekac->abcijk", H.aa.oovv, X.aa.vovv)
        L3A -= (9.0 / 36.0) * ccpy_einsum("mjab,ikmc->abcijk", H.aa.oovv, X.aa.ooov)

    L3A -= np.transpose(L3A, (0, 1, 2, 3, 5, 4)) # (jk)
    L3A -= np.transpose(L3A, (0, 1, 2, 4, 3, 5)) + np.transpose(L3A, (0, 1, 2, 5, 4, 3)) # (i/jk)
    L3A -= np.transpose(L3A, (0, 2, 1, 3, 4, 5)) # (bc)
    L3A -= np.transpose(L3A, (2, 1, 0, 3, 4, 5)) + np.transpose(L3A, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return L3A


def build_L3B(L, H, X=None, flag_2ba=True):

    # < 0 | L1 * H(2) | ijk~abc~ >
    L3B = ccpy_einsum("ai,jkbc->abcijk", L.a, H.ab.oovv)
    L3B += 0.25 * ccpy_einsum("ck,ijab->abcijk", L.b, H.aa.oovv)

    # < 0 | L2 * H(2) | ijk~abc~ >
    L3B += ccpy_einsum("bcjk,ia->abcijk", L.ab, H.a.ov)
    L3B += 0.25 * ccpy_einsum("abij,kc->abcijk", L.aa, H.b.ov)

    L3B += 0.5 * ccpy_einsum("ekbc,aeij->abcijk", H.ab.vovv, L.aa)
    L3B -= 0.5 * ccpy_einsum("jkmc,abim->abcijk", H.ab.ooov, L.aa)
    L3B += ccpy_einsum("ieac,bejk->abcijk", H.ab.ovvv, L.ab)
    L3B -= ccpy_einsum("ikam,bcjm->abcijk", H.ab.oovo, L.ab)
    L3B += 0.5 * ccpy_einsum("eiba,ecjk->abcijk", H.aa.vovv, L.ab)
    L3B -= 0.5 * ccpy_einsum("jima,bcmk->abcijk", H.aa.ooov, L.ab)

    L3B += 0.5 * ccpy_einsum("ekbc,ijae->abcijk", X.ab.vovv, H.aa.oovv)
    L3B -= 0.5 * ccpy_einsum("jkmc,imab->abcijk", X.ab.ooov, H.aa.oovv)
    L3B += ccpy_einsum("ieac,jkbe->abcijk", X.ab.ovvv, H.ab.oovv)
    L3B -= ccpy_einsum("ikam,jmbc->abcijk", X.ab.oovo, H.ab.oovv)
    L3B += 0.5 * ccpy_einsum("eiba,jkec->abcijk", X.aa.vovv, H.ab.oovv)
    L3B -= 0.5 * ccpy_einsum("jima,mkbc->abcijk", X.aa.ooov, H.ab.oovv)

    if not flag_2ba:
        # < 0 | L3 * H(2) | ijk~abc~ >
        L3B -= 0.5 * ccpy_einsum("im,abcmjk->abcijk", H.a.oo, L.aab)
        L3B -= 0.25 * ccpy_einsum("km,abcijm->abcijk", H.b.oo, L.aab)
        L3B += 0.5 * ccpy_einsum("ea,ebcijk->abcijk", H.a.vv, L.aab)
        L3B += 0.25 * ccpy_einsum("ec,abeijk->abcijk", H.b.vv, L.aab)
        L3B += 0.125 * ccpy_einsum("ijmn,abcmnk->abcijk", H.aa.oooo, L.aab)
        L3B += 0.5 * ccpy_einsum("jkmn,abcimn->abcijk", H.ab.oooo, L.aab)
        L3B += 0.125 * ccpy_einsum("efab,efcijk->abcijk", H.aa.vvvv, L.aab)
        L3B += 0.5 * ccpy_einsum("efbc,aefijk->abcijk", H.ab.vvvv, L.aab)
        L3B += ccpy_einsum("eima,ebcmjk->abcijk", H.aa.voov, L.aab)
        L3B += ccpy_einsum("ieam,becjmk->abcijk", H.ab.ovvo, L.abb)
        L3B += 0.25 * ccpy_einsum("ekmc,abeijm->abcijk", H.ab.voov, L.aaa)
        L3B += 0.25 * ccpy_einsum("ekmc,abeijm->abcijk", H.bb.voov, L.aab)
        L3B -= 0.5 * ccpy_einsum("ekam,ebcijm->abcijk", H.ab.vovo, L.aab)
        L3B -= 0.5 * ccpy_einsum("iemc,abemjk->abcijk", H.ab.ovov, L.aab)

    L3B -= np.transpose(L3B, (1, 0, 2, 3, 4, 5))
    L3B -= np.transpose(L3B, (0, 1, 2, 4, 3, 5))

    return L3B