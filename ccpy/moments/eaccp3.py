"""Functions to calculate the EA-EOMCC(P;Q) 3p-2h correction to EA-EOMCC(P)."""
import time
import numpy as np

from ccpy.constants.constants import hartreetoeV
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.utilities.updates import eaccp3_loops
from ccpy.eomcc.eaeom3_intermediates import get_eaeom3_p_intermediates
from ccpy.left.left_eaeom_intermediates import get_lefteaeom3_p_intermediates
from ccpy.eomcc.eaeom3 import build_HR_3A, build_HR_3B, build_HR_3C
from ccpy.utilities.utilities import unravel_3p2h_amplitudes

def calc_eaccp3_full(T, R, L, r3_excitations, omega, corr_energy, H, H0, system, use_RHF=False):
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

    # unravel triples vector into r3(abcijk) and l3(abcijk)
    R_unravel = unravel_3p2h_amplitudes(R, r3_excitations, system, do_l3)
    L_unravel = unravel_3p2h_amplitudes(L, r3_excitations, system, do_l3)

    #### aaa correction ####
    # Moments and left vector
    M3A = build_HR_3A(R_unravel, T, X_right, H)
    L3A = build_LH_3A(L_unravel, H, X_left)
    # perform correction in-loop
    dA_aaa, dB_aaa, dC_aaa, dD_aaa = eaccp3_loops.eaccp3_loops.eaccp3a_full(
       M3A, L3A, r3_excitations["aaa"],
       omega,
       H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
       H.aa.vvvv, H.aa.oooo, H.aa.voov,
       d3aaa_o, d3aaa_v,
    )
    #### aab correction ####
    # moments and left vector
    M3B = build_HR_3B(R_unravel, T, X_right, H)
    L3B = build_LH_3B(L_unravel, H, X_left)
    # perform correction in-loop
    dA_aab, dB_aab, dC_aab, dD_aab = eaccp3_loops.eaccp3_loops.eaccp3b_full(
       M3B, L3B, r3_excitations["aab"],
       omega,
       H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
       H.a.oo, H.a.vv, H.b.oo, H.b.vv,
       H.aa.vvvv, H.aa.voov,
       H.ab.vvvv, H.ab.oooo, H.ab.ovov, H.ab.vovo,
       H.bb.voov,
       d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v,
    )
    #### abb correction ####
    # moments and left vector
    M3C = build_HR_3C(R_unravel, T, X_right, H)
    L3C = build_LH_3C(L_unravel, H, X_left)
    dA_abb, dB_abb, dC_abb, dD_abb = eaccp3_loops.eaccp3_loops.eaccp3c_full(
        M3C, L3C, r3_excitations["abb"],
        omega,
        H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
        H.a.oo, H.a.vv, H.b.oo, H.b.vv,
        H.ab.vvvv, H.ab.ovov, H.ab.vovo,
        H.bb.vvvv, H.bb.oooo, H.bb.voov,
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
    print("   EA-EOMCC(P) = {:>10.10f}    ω = {:>10.10f}     VEE = {:>10.5f} eV".format(system.reference_energy + corr_energy + omega, omega, hartreetoeV * omega))
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

    Ecrcc23 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    delta23 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}

    return Ecrcc23, delta23

def build_LH_3A(L, H, X):
    """Calculate the projection < 0 | (L1p+L2p1h+L3p2h)*(H_N e^(T1+T2))_C | jkabc >."""
    # moment-like terms < 0 | (L1p+L2p1h)*(H_N e^(T1+T2))_C | jkabc >
    X3A = (3.0 / 12.0) * np.einsum("a,jkbc->abcjk", L.a, H.aa.oovv, optimize=True)
    X3A += (6.0 / 12.0) * np.einsum("abj,kc->abcjk", L.aa, H.a.ov, optimize=True)
    X3A -= (3.0 / 12.0) * np.einsum("abm,jkmc->abcjk", L.aa, H.aa.ooov, optimize=True)
    X3A += (6.0 / 12.0) * np.einsum("eck,ejab->abcjk", L.aa, H.aa.vovv, optimize=True)
    # <0|L3p2h*(H_N e^(T1+T2))_C | jkabc>
    X3A -= (2.0 / 12.0) * np.einsum("jm,abcmk->abcjk", H.a.oo, L.aaa, optimize=True)
    X3A += (3.0 / 12.0) * np.einsum("eb,aecjk->abcjk", H.a.vv, L.aaa, optimize=True)
    X3A += (1.0 / 24.0) * np.einsum("jkmn,abcmn->abcjk", H.aa.oooo, L.aaa, optimize=True)
    X3A += (3.0 / 24.0) * np.einsum("efbc,aefjk->abcjk", H.aa.vvvv, L.aaa, optimize=True)
    X3A += (6.0 / 12.0) * np.einsum("ejmb,acekm->abcjk", H.aa.voov, L.aaa, optimize=True)
    X3A += (6.0 / 12.0) * np.einsum("jebm,acekm->abcjk", H.ab.ovvo, L.aab, optimize=True)
    # three-body Hbar terms (verified against explicit 3-body hbars)
    X3A -= (6.0 / 12.0) * np.einsum("mck,mjab->abcjk", X["aa"]["ovo"], H.aa.oovv, optimize=True)
    X3A += (3.0 / 12.0) * np.einsum("aeb,jkec->abcjk", X["aa"]["vvv"], H.aa.oovv, optimize=True)
    X3A -= np.transpose(X3A, (1, 0, 2, 3, 4)) + np.transpose(X3A, (2, 1, 0, 3, 4)) # antisymmetrize A(a/bc)
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3A

def build_LH_3B(L, H, X):
    """Calculate the projection < 0 | (L1p+L2p1h+L3p2h)(H_N e^(T1+T2))_C | jk~abc~ >."""
    # moment-like terms < 0 | (L1p+L2p1h)*(H_N e^(T1+T2))_C | jk~abc~ >
    X3B = np.einsum("a,jkbc->abcjk", L.a, H.ab.oovv, optimize=True)
    X3B += (1.0 / 2.0) * np.einsum("abj,kc->abcjk", L.aa, H.b.ov, optimize=True)
    X3B += np.einsum("ack,jb->abcjk", L.ab, H.a.ov, optimize=True)
    X3B -= (1.0 / 2.0) * np.einsum("abm,jkmc->abcjk", L.aa, H.ab.ooov, optimize=True)
    X3B -= np.einsum("acm,jkbm->abcjk", L.ab, H.ab.oovo, optimize=True)
    X3B += np.einsum("aej,ekbc->abcjk", L.aa, H.ab.vovv, optimize=True)
    X3B += np.einsum("aek,jebc->abcjk", L.ab, H.ab.ovvv, optimize=True)
    X3B += (1.0 / 2.0) * np.einsum("eck,ejab->abcjk", L.ab, H.aa.vovv, optimize=True)
    # < 0 | L3p2h*(H_N e^(T1+T2))_C | jk~abc~ >
    X3B -= (1.0 / 2.0) * np.einsum("jm,abcmk->abcjk", H.a.oo, L.aab, optimize=True) # (1)
    X3B -= (1.0 / 2.0) * np.einsum("km,abcjm->abcjk", H.b.oo, L.aab, optimize=True) # (2)
    X3B += np.einsum("ea,ebcjk->abcjk", H.a.vv, L.aab, optimize=True) # (3)
    X3B += (1.0 / 2.0) * np.einsum("ec,abejk->abcjk", H.b.vv, L.aab, optimize=True) # (4)
    X3B += (1.0 / 2.0) * np.einsum("jkmn,abcmn->abcjk", H.ab.oooo, L.aab, optimize=True) # (5)
    X3B += (1.0 / 4.0) * np.einsum("efab,efcjk->abcjk", H.aa.vvvv, L.aab, optimize=True) # (6)
    X3B += np.einsum("efbc,aefjk->abcjk", H.ab.vvvv, L.aab, optimize=True) # (7)
    X3B += np.einsum("ejmb,aecmk->abcjk", H.aa.voov, L.aab, optimize=True) # (8) !
    X3B += np.einsum("jebm,aecmk->abcjk", H.ab.ovvo, L.abb, optimize=True) # (9)
    X3B += (1.0 / 2.0) * np.einsum("ekmc,abejm->abcjk", H.ab.voov, L.aaa, optimize=True) # (10)
    X3B += (1.0 / 2.0) * np.einsum("ekmc,abejm->abcjk", H.bb.voov, L.aab, optimize=True) # (11)
    X3B -= (1.0 / 2.0) * np.einsum("jemc,abemk->abcjk", H.ab.ovov, L.aab, optimize=True) # (12)
    X3B -= np.einsum("ekbm,aecjm->abcjk", H.ab.vovo, L.aab, optimize=True) # (13)
    # three-body Hbar terms
    X3B -= np.einsum("akm,jmbc->abcjk", X["ab"]["voo"], H.ab.oovv, optimize=True) # (1)
    X3B -= (1.0 / 2.0) * np.einsum("mck,mjab->abcjk", X["ab"]["ovo"], H.aa.oovv, optimize=True) # (2)
    X3B -= np.einsum("mbj,mkac->abcjk", X["aa"]["ovo"], H.ab.oovv, optimize=True) # (3)
    X3B += (1.0 / 2.0) * np.einsum("aeb,jkec->abcjk", X["aa"]["vvv"], H.ab.oovv, optimize=True) # (4)
    X3B += np.einsum("aec,jkbe->abcjk", X["ab"]["vvv"], H.ab.oovv, optimize=True) # (5)
    X3B -= np.transpose(X3B, (1, 0, 2, 3, 4)) # antisymmetrize A(ab)
    return X3B

def build_LH_3C(L, H, X):
    """Calculate the projection < 0 | (L1p+L2p1h+L3p2h)(H_N e^(T1+T2))_C | j~k~ab~c~ >."""
    # moment-like terms < 0 | (L1p+L2p1h)*(H_N e^(T1+T2))_C | j~k~ab~c~ >
    X3C = (1.0 / 4.0) * np.einsum("a,jkbc->abcjk", L.a, H.bb.oovv, optimize=True)
    X3C += np.einsum("abj,kc->abcjk", L.ab, H.b.ov, optimize=True)
    X3C -= (2.0 / 4.0) * np.einsum("abm,jkmc->abcjk", L.ab, H.bb.ooov, optimize=True)
    X3C += (2.0 / 4.0) * np.einsum("aej,ekbc->abcjk", L.ab, H.bb.vovv, optimize=True)
    X3C += np.einsum("eck,ejab->abcjk", L.ab, H.ab.vovv, optimize=True) # !
    # < 0 | L3p2h*(H_N e^(T1+T2))_C | j!k~ab!c~ >
    X3C -= (2.0 / 4.0) * np.einsum("jm,abcmk->abcjk", H.b.oo, L.abb, optimize=True)
    X3C += (2.0 / 4.0) * np.einsum("eb,aecjk->abcjk", H.b.vv, L.abb, optimize=True)
    X3C += (1.0 / 4.0) * np.einsum("ea,ebcjk->abcjk", H.a.vv, L.abb, optimize=True)
    X3C += (1.0 / 8.0) * np.einsum("jkmn,abcmn->abcjk", H.bb.oooo, L.abb, optimize=True)
    X3C += (2.0 / 4.0) * np.einsum("efab,efcjk->abcjk", H.ab.vvvv, L.abb, optimize=True)
    X3C += (1.0 / 8.0) * np.einsum("efbc,aefjk->abcjk", H.bb.vvvv, L.abb, optimize=True)
    X3C += np.einsum("ejmb,aecmk->abcjk", H.ab.voov, L.aab, optimize=True)
    X3C += np.einsum("ejmb,aecmk->abcjk", H.bb.voov, L.abb, optimize=True)
    X3C -= (2.0 / 4.0) * np.einsum("ejam,ebcmk->abcjk", H.ab.vovo, L.abb, optimize=True)
    # three-body Hbar terms
    X3C -= np.einsum("mck,mjab->abcjk", X["ab"]["ovo"], H.ab.oovv, optimize=True)
    X3C -= (2.0 / 4.0) * np.einsum("ajm,mkbc->abcjk", X["ab"]["voo"], H.bb.oovv, optimize=True)
    X3C += (2.0 / 4.0) * np.einsum("aeb,jkec->abcjk", X["ab"]["vvv"], H.bb.oovv, optimize=True)
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3C -= np.transpose(X3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3C
