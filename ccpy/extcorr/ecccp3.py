"""Functions to calculate the ground-state CC(t;3) triples correction to CCSDt."""
import time

import numpy as np
from ccpy.energy.cc_energy import get_cc_energy
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.lib.core import ecccp3_loops


def calc_ecccp3(T, L, C, H, H0, system, pspace, use_RHF=False):
    """
    Calculate the ground-state CC(P;3) correction to the CC(P) energy for internal T3 amplitudes.
    """
    t_start = time.perf_counter()

    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    #### aaa correction ####
    M3A = build_M3A_ccsdt(T, H, H0)
    L3A = build_L3A_2ba(L, H)
    dA_aaa_int, dB_aaa_int, dC_aaa_int, dD_aaa_int, dA_aaa_ext, dB_aaa_ext, dC_aaa_ext, dD_aaa_ext =\
    ecccp3_loops.ecccp3a(
        pspace[0]["aaa"],
        M3A, L3A, C.aaa, 0.0,
        H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
        H.aa.voov, H.aa.oooo, H.aa.vvvv,
        d3aaa_o, d3aaa_v,
        system.noccupied_alpha, system.nunoccupied_alpha,
    )

    #### aab correction ####
    M3B = build_M3B_ccsdt(T, H, H0)
    L3B = build_L3B_2ba(L, H)
    dA_aab_int, dB_aab_int, dC_aab_int, dD_aab_int, dA_aab_ext, dB_aab_ext, dC_aab_ext, dD_aab_ext =\
    ecccp3_loops.ecccp3b(
        pspace[0]["aab"],
        M3B, L3B, C.aab, 0.0,
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
        correction_A_int = 2.0 * dA_aaa_int + 2.0 * dA_aab_int
        correction_B_int = 2.0 * dB_aaa_int + 2.0 * dB_aab_int
        correction_C_int = 2.0 * dC_aaa_int + 2.0 * dC_aab_int
        correction_D_int = 2.0 * dD_aaa_int + 2.0 * dD_aab_int

        correction_A_ext = 2.0 * dA_aaa_ext + 2.0 * dA_aab_ext
        correction_B_ext = 2.0 * dB_aaa_ext + 2.0 * dB_aab_ext
        correction_C_ext = 2.0 * dC_aaa_ext + 2.0 * dC_aab_ext
        correction_D_ext = 2.0 * dD_aaa_ext + 2.0 * dD_aab_ext

    else:

        #### abb correction ####
        M3C = build_M3C_ccsdt(T, H, H0)
        L3C = build_L3C_2ba(L, H)
        dA_abb_int, dB_abb_int, dC_abb_int, dD_abb_int, dA_abb_ext, dB_abb_ext, dC_abb_ext, dD_abb_ext =\
        ecccp3_loops.ecccp3c(
            pspace[0]["abb"],
            M3C, L3C, C.abb, 0.0,
            H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
            H.aa.voov,
            H.ab.ovov, H.ab.vovo,
            H.ab.oooo, H.ab.vvvv,
            H.bb.voov, H.bb.oooo, H.bb.vvvv,
            d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v,
            system.noccupied_alpha, system.nunoccupied_alpha,
            system.noccupied_beta, system.nunoccupied_beta,
        )

        #### bbb correction ####
        M3D = build_M3D_ccsdt(T, H, H0)
        L3D = build_L3D_2ba(L, H)
        dA_bbb_int, dB_bbb_int, dC_bbb_int, dD_bbb_int, dA_bbb_ext, dB_bbb_ext, dC_bbb_ext, dD_bbb_ext =\
        ecccp3_loops.ecccp3d(
            pspace[0]["bbb"],
            M3D, L3D, C.bbb, 0.0,
            H0.b.oo, H0.b.vv, H.b.oo, H.b.vv,
            H.bb.voov, H.bb.oooo, H.bb.vvvv,
            d3bbb_o, d3bbb_v,
            system.noccupied_beta, system.nunoccupied_beta,
        )

        correction_A_int = dA_aaa_int + dA_aab_int + dA_abb_int + dA_bbb_int
        correction_B_int = dB_aaa_int + dB_aab_int + dB_abb_int + dB_bbb_int
        correction_C_int = dC_aaa_int + dC_aab_int + dC_abb_int + dC_bbb_int
        correction_D_int = dD_aaa_int + dD_aab_int + dD_abb_int + dD_bbb_int

        correction_A_ext = dA_aaa_ext + dA_aab_ext + dA_abb_ext + dA_bbb_ext
        correction_B_ext = dB_aaa_ext + dB_aab_ext + dB_abb_ext + dB_bbb_ext
        correction_C_ext = dC_aaa_ext + dC_aab_ext + dC_abb_ext + dC_bbb_ext
        correction_D_ext = dD_aaa_ext + dD_aab_ext + dD_abb_ext + dD_bbb_ext

    t_end = time.perf_counter()
    minutes, seconds = divmod(t_end - t_start, 60)

    # print the results
    cc_energy = get_cc_energy(T, H0)

    energy_A_int = cc_energy + correction_A_int
    energy_B_int = cc_energy + correction_B_int
    energy_C_int = cc_energy + correction_C_int
    energy_D_int = cc_energy + correction_D_int

    total_energy_A_int = system.reference_energy + energy_A_int
    total_energy_B_int = system.reference_energy + energy_B_int
    total_energy_C_int = system.reference_energy + energy_C_int
    total_energy_D_int = system.reference_energy + energy_D_int

    print('   Internal CC(P;3) Calculation Summary')
    print('   -------------------------------------')
    print("   Completed in  ({:0.2f}m  {:0.2f}s)\n".format(minutes, seconds))
    print("   CC(P) = {:>10.10f}".format(system.reference_energy + cc_energy))
    print(
        "   CC(P;3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A_int, energy_A_int, correction_A_int
        )
    )
    print(
        "   CC(P;3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B_int, energy_B_int, correction_B_int
        )
    )
    print(
        "   CC(P;3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C_int, energy_C_int, correction_C_int
        )
    )
    print(
        "   CC(P;3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D_int, energy_D_int, correction_D_int
        )
    )

    energy_A_ext = cc_energy + correction_A_ext
    energy_B_ext = cc_energy + correction_B_ext
    energy_C_ext = cc_energy + correction_C_ext
    energy_D_ext = cc_energy + correction_D_ext

    total_energy_A_ext = system.reference_energy + energy_A_ext
    total_energy_B_ext = system.reference_energy + energy_B_ext
    total_energy_C_ext = system.reference_energy + energy_C_ext
    total_energy_D_ext = system.reference_energy + energy_D_ext

    print('   External CC(P;3) Calculation Summary')
    print('   -------------------------------------')
    print("   Completed in  ({:0.2f}m  {:0.2f}s)\n".format(minutes, seconds))
    print("   CC(P) = {:>10.10f}".format(system.reference_energy + cc_energy))
    print(
        "   CC(P;3)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A_ext, energy_A_ext, correction_A_ext
        )
    )
    print(
        "   CC(P;3)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B_ext, energy_B_ext, correction_B_ext
        )
    )
    print(
        "   CC(P;3)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C_ext, energy_C_ext, correction_C_ext
        )
    )
    print(
        "   CC(P;3)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D_ext, energy_D_ext, correction_D_ext
        )
    )


    Eccp3 = {"A": system.reference_energy + cc_energy + correction_A_ext + correction_A_int,
             "B": system.reference_energy + cc_energy + correction_B_ext + correction_B_int,
             "C": system.reference_energy + cc_energy + correction_C_ext + correction_C_int,
             "D": system.reference_energy + cc_energy + correction_D_ext + correction_D_int}

    deltap3 = {"A": correction_A_ext + correction_A_int,
               "B": correction_B_ext + correction_B_int,
               "C": correction_C_ext + correction_C_int,
               "D": correction_D_ext + correction_D_int}

    return Eccp3, deltap3

def build_M3A_ccsdt(T, H, H0):
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


def build_M3B_ccsdt(T, H, H0):
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


def build_M3C_ccsdt(T, H, H0):
    """
    Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N e^(T1+T2+T3))_C|0>.
    """
    # <ijk~abc~ | H(2) | 0 > + (VT3)_C intermediates
    # Recall that we are using HBar CCSDT, so the vooo and vvov parts have T3 in it already!
    I2C_vooo = H.bb.vooo - np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,cekj->cmkj", H.b.ov, T.ab, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,eaki->maki", H.a.ov, T.ab, optimize=True)

    # MM(2,3)C
    MM23C = 0.5 * np.einsum("cbke,aeij->cbakji", H.ab.vvov, T.bb, optimize=True)
    MM23C -= 0.5 * np.einsum("cmkj,abim->cbakji", I2B_vooo, T.bb, optimize=True)
    MM23C += np.einsum("caei,ebkj->cbakji", H.ab.vvvo, T.ab, optimize=True)
    MM23C -= np.einsum("maki,cbmj->cbakji", I2B_ovoo, T.ab, optimize=True)
    MM23C += 0.5 * np.einsum("abie,cekj->cbakji", H.bb.vvov, T.ab, optimize=True)
    MM23C -= 0.5 * np.einsum("amij,cbkm->cbakji", I2C_vooo, T.ab, optimize=True)
    # (HBar*T3)_C
    MM23C -= 0.5 * np.einsum("mi,cbakjm->cbakji", H.b.oo, T.abb, optimize=True)
    MM23C -= 0.25 * np.einsum("mk,cbamji->cbakji", H.a.oo, T.abb, optimize=True)
    MM23C += 0.5 * np.einsum("ae,cbekji->cbakji", H.b.vv, T.abb, optimize=True)
    MM23C += 0.25 * np.einsum("ce,ebakji->cbakji", H.a.vv, T.abb, optimize=True)
    MM23C += 0.125 * np.einsum("mnij,cbaknm->cbakji", H.bb.oooo, T.abb, optimize=True)
    MM23C += 0.5 * np.einsum("nmkj,cbanmi->cbakji", H.ab.oooo, T.abb, optimize=True)
    MM23C += 0.125 * np.einsum("abef,cfekji->cbakji", H.bb.vvvv, T.abb, optimize=True)
    MM23C += 0.5 * np.einsum("cbfe,feakji->cbakji", H.ab.vvvv, T.abb, optimize=True)
    MM23C += np.einsum("amie,cbekjm->cbakji", H.bb.voov, T.abb, optimize=True)
    MM23C += np.einsum("maei,cebkmj->cbakji", H.ab.ovvo, T.aab, optimize=True)
    MM23C += 0.25 * np.einsum("cmke,abeijm->cbakji", H.ab.voov, T.bbb, optimize=True)
    MM23C += 0.25 * np.einsum("cmke,ebamji->cbakji", H.aa.voov, T.abb, optimize=True)
    MM23C -= 0.5 * np.einsum("make,cbemji->cbakji", H.ab.ovov, T.abb, optimize=True)
    MM23C -= 0.5 * np.einsum("cmei,ebakjm->cbakji", H.ab.vovo, T.abb, optimize=True)

    MM23C -= np.transpose(MM23C, (0, 2, 1, 3, 4, 5))
    MM23C -= np.transpose(MM23C, (0, 1, 2, 3, 5, 4))

    return MM23C


def build_M3D_ccsdt(T, H, H0):
    """
    Update t3d amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3))_C|0>.
    """

    # <ijkabc | H(2) | 0 > + (VT3)_C intermediates
    # Recall that we are using HBar CCSDT, so the vooo and vvov parts have T3 in it already!
    I2C_vvov = H.bb.vvov + np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)

    # MM(2,3)D
    MM23D = -0.25 * np.einsum("amij,bcmk->abcijk", H.bb.vooo, T.bb, optimize=True)
    MM23D += 0.25 * np.einsum("abie,ecjk->abcijk", I2C_vvov, T.bb, optimize=True)
    # (HBar*T3)_C
    MM23D -= (1.0 / 12.0) * np.einsum("mk,abcijm->abcijk", H.b.oo, T.bbb, optimize=True)
    MM23D += (1.0 / 12.0) * np.einsum("ce,abeijk->abcijk", H.b.vv, T.bbb, optimize=True)
    MM23D += (1.0 / 24.0) * np.einsum("mnij,abcmnk->abcijk", H.bb.oooo, T.bbb, optimize=True)
    MM23D += (1.0 / 24.0) * np.einsum("abef,efcijk->abcijk", H.bb.vvvv, T.bbb, optimize=True)
    MM23D += 0.25 * np.einsum("cmke,abeijm->abcijk", H.bb.voov, T.bbb, optimize=True)
    MM23D += 0.25 * np.einsum("mcek,ebamji->abcijk", H.ab.ovvo, T.abb, optimize=True)

    MM23D -= np.transpose(MM23D, (0, 1, 2, 3, 5, 4)) # (jk)
    MM23D -= np.transpose(MM23D, (0, 1, 2, 4, 3, 5)) + np.transpose(MM23D, (0, 1, 2, 5, 4, 3)) # (i/jk)
    MM23D -= np.transpose(MM23D, (0, 2, 1, 3, 4, 5)) # (bc)
    MM23D -= np.transpose(MM23D, (2, 1, 0, 3, 4, 5)) + np.transpose(MM23D, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return MM23D


def build_L3A_2ba(L, H):

    # < 0 | L1 * H(2) | ijkabc >
    L3A = (9.0 / 36.0) * np.einsum("ai,jkbc->abcijk", L.a, H.aa.oovv, optimize=True)

    # < 0 | L2 * H(2) | ijkabc >
    L3A += (9.0 / 36.0) * np.einsum("bcjk,ia->abcijk", L.aa, H.a.ov, optimize=True)

    L3A += (9.0 / 36.0) * np.einsum("ebij,ekac->abcijk", L.aa, H.aa.vovv, optimize=True)
    L3A -= (9.0 / 36.0) * np.einsum("abmj,ikmc->abcijk", L.aa, H.aa.ooov, optimize=True)

    L3A -= np.transpose(L3A, (0, 1, 2, 3, 5, 4)) # (jk)
    L3A -= np.transpose(L3A, (0, 1, 2, 4, 3, 5)) + np.transpose(L3A, (0, 1, 2, 5, 4, 3)) # (i/jk)
    L3A -= np.transpose(L3A, (0, 2, 1, 3, 4, 5)) # (bc)
    L3A -= np.transpose(L3A, (2, 1, 0, 3, 4, 5)) + np.transpose(L3A, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return L3A


def build_L3B_2ba(L, H):

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

    L3B -= np.transpose(L3B, (1, 0, 2, 3, 4, 5))
    L3B -= np.transpose(L3B, (0, 1, 2, 4, 3, 5))

    return L3B


def build_L3C_2ba(L, H):

    # < 0 | L1 * H(2) | ijk~abc~ >
    L3C = np.einsum("ai,kjcb->cbakji", L.b, H.ab.oovv, optimize=True)
    L3C += 0.25 * np.einsum("ck,ijab->cbakji", L.a, H.bb.oovv, optimize=True)

    # < 0 | L2 * H(2) | ijk~abc~ >
    L3C += np.einsum("cbkj,ia->cbakji", L.ab, H.b.ov, optimize=True)
    L3C += 0.25 * np.einsum("abij,kc->cbakji", L.bb, H.a.ov, optimize=True)

    L3C += 0.5 * np.einsum("kecb,aeij->cbakji", H.ab.ovvv, L.bb, optimize=True)
    L3C -= 0.5 * np.einsum("kjcm,abim->cbakji", H.ab.oovo, L.bb, optimize=True)
    L3C += np.einsum("eica,ebkj->cbakji", H.ab.vovv, L.ab, optimize=True)
    L3C -= np.einsum("kima,cbmj->cbakji", H.ab.ooov, L.ab, optimize=True)
    L3C += 0.5 * np.einsum("eiba,cekj->cbakji", H.bb.vovv, L.ab, optimize=True)
    L3C -= 0.5 * np.einsum("jima,cbkm->cbakji", H.bb.ooov, L.ab, optimize=True)

    L3C -= np.transpose(L3C, (0, 2, 1, 3, 4, 5))
    L3C -= np.transpose(L3C, (0, 1, 2, 3, 5, 4))

    return L3C


def build_L3D_2ba(L, H):

    # < 0 | L1 * H(2) | ijkabc >
    L3D = (9.0 / 36.0) * np.einsum("ai,jkbc->abcijk", L.b, H.bb.oovv, optimize=True)

    # < 0 | L2 * H(2) | ijkabc >
    L3D += (9.0 / 36.0) * np.einsum("bcjk,ia->abcijk", L.bb, H.b.ov, optimize=True)

    L3D += (9.0 / 36.0) * np.einsum("ebij,ekac->abcijk", L.bb, H.bb.vovv, optimize=True)
    L3D -= (9.0 / 36.0) * np.einsum("abmj,ikmc->abcijk", L.bb, H.bb.ooov, optimize=True)

    L3D -= np.transpose(L3D, (0, 1, 2, 3, 5, 4)) # (jk)
    L3D -= np.transpose(L3D, (0, 1, 2, 4, 3, 5)) + np.transpose(L3D, (0, 1, 2, 5, 4, 3)) # (i/jk)
    L3D -= np.transpose(L3D, (0, 2, 1, 3, 4, 5)) # (bc)
    L3D -= np.transpose(L3D, (2, 1, 0, 3, 4, 5)) + np.transpose(L3D, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return L3D
