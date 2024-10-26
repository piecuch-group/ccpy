import time
import numpy as np

from ccpy.constants.constants import hartreetoeV
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.lib.core import creacc_loops


def calc_creacc23(T, R, L, omega, corr_energy, H, H0, system, use_RHF=False):
    """
    Calculate the correction for 3p-2h correlations to the EA-EOMCCSD(2p-1h) energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    # compute H*R intermediates
    X = get_eaeom23_intermediates(H, R)

    #### aaa correction ####
    # calculate intermediates
    M3A = build_HR_3A(R, T, X, H)
    L3A = build_LH_3A(L, T, H)
    # perform correction in-loop
    dA_aaa, dB_aaa, dC_aaa, dD_aaa = creacc_loops.crcc23a(M3A, L3A, omega,
                                                                       H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
                                                                       H.aa.vvvv, H.aa.oooo, H.aa.voov,
                                                                       d3aaa_o, d3aaa_v)
    #### aab correction ####
    # calculate intermediates
    M3B = build_HR_3B(R, T, X, H)
    L3B = build_LH_3B(L, T, H)
    # perform correction in-loop
    dA_aab, dB_aab, dC_aab, dD_aab = creacc_loops.crcc23b(M3B, L3B, omega,
                                                                       H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                                                                       H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                                       H.aa.vvvv, H.aa.voov,
                                                                       H.ab.vvvv, H.ab.oooo, H.ab.ovov, H.ab.vovo,
                                                                       H.bb.voov,
                                                                       d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v)
    #### abb correction ####
    # calculate intermediates
    M3C = build_HR_3C(R, T, X, H) 
    L3C = build_LH_3C(L, T, H)
    # perform correction in-loop
    dA_abb, dB_abb, dC_abb, dD_abb = creacc_loops.crcc23c(M3C, L3C, omega,
                                                                       H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                                                                       H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                                       H.ab.vvvv, H.ab.ovov, H.ab.vovo,
                                                                       H.bb.vvvv, H.bb.oooo, H.bb.voov,
                                                                       d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v)
    # Add up individual contributions to form total 3h-2p correction
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

    print('   CR-EA-EOMCCSD(2p-1h,3p-2h) Calculation Summary')
    print('   -------------------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   EA-EOMCCSD(2p-1h) = {:>10.10f}    ω = {:>10.10f}     VEE = {:>10.5f} eV".format(system.reference_energy + corr_energy + omega, omega, hartreetoeV * omega))
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
    print("")

    Ecrcc23 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    delta23 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}

    return Ecrcc23, delta23

def get_eaeom23_intermediates(H, R):

    # These intermediates will be 3-index quantities, which are not
    # set up in the models at the moment. We will just use a dictionary
    # as a workaround for now.

    X = {"aa": {}, "ab": {}}

    # x2a(amj)
    X["aa"]["voo"] = (
                    +np.einsum("mnjf,afn->amj", H.aa.ooov, R.aa, optimize=True)
                    +np.einsum("mnjf,afn->amj", H.ab.ooov, R.ab, optimize=True)
                    +0.5*np.einsum("amef,efj->amj", H.aa.vovv, R.aa, optimize=True)
                    -np.einsum("amje,e->amj", H.aa.voov, R.a, optimize=True) # CAREFUL: this is a minus sign
    )
    # x2b(mb~j~)
    X["ab"]["ovo"] = (
                    -np.einsum("mnej,ebn->mbj", H.ab.oovo, R.ab, optimize=True)
                    +np.einsum("mbef,efj->mbj", H.ab.ovvv, R.ab, optimize=True)
                    +np.einsum("mbfj,f->mbj", H.ab.ovvo, R.a, optimize=True)
    )
    # x2b(am~j~)
    X["ab"]["voo"] = (
                    +np.einsum("nmfj,afn->amj", H.ab.oovo, R.aa, optimize=True)
                    +np.einsum("mnjf,afn->amj", H.bb.ooov, R.ab, optimize=True)
                    +np.einsum("amef,efj->amj", H.ab.vovv, R.ab, optimize=True)
                    +np.einsum("amej,e->amj", H.ab.vovo, R.a, optimize=True)
    )

    # x2a(abe)
    X["aa"]["vvv"] = (
                    +np.einsum("bnef,afn->abe", H.aa.vovv, R.aa, optimize=True)
                    +np.einsum("bnef,afn->abe", H.ab.vovv, R.ab, optimize=True)
                    +0.5*np.einsum("abfe,f->abe", H.aa.vvvv, R.a, optimize=True)
    )
    X["aa"]["vvv"] -= np.transpose(X["aa"]["vvv"], (1, 0, 2))
    # x2b(ab~e~)
    X["ab"]["vvv"] = (
                    +np.einsum("nbfe,afn->abe", H.ab.ovvv, R.aa, optimize=True)
                    +np.einsum("bnef,afn->abe", H.bb.vovv, R.ab, optimize=True)
                    -np.einsum("amfe,fbm->abe", H.ab.vovv, R.ab, optimize=True)
                    +np.einsum("abfe,f->abe", H.ab.vvvv, R.a, optimize=True)
    )
    return X

def build_HR_3A(R, T, X, H):
    """Calculate the projection <jkabc|[ (H_N e^(T1+T2))_C*(R1p+R2p1h) ]_C|0>."""
    # moment-like terms
    X3A = -(3.0 / 12.0) * np.einsum("cmkj,abm->abcjk", H.aa.vooo, R.aa, optimize=True)     # (7)
    X3A += (6.0 / 12.0) * np.einsum("cbke,aej->abcjk", H.aa.vvov, R.aa, optimize=True)     # (8)
    # 3-body Hbar terms factorized using intermediates
    X3A -= (6.0 / 12.0) * np.einsum("amj,bcmk->abcjk", X["aa"]["voo"], T.aa, optimize=True) # (9)
    X3A += (3.0 / 12.0) * np.einsum("abe,ecjk->abcjk", X["aa"]["vvv"], T.aa, optimize=True) # (10)
    X3A -= np.transpose(X3A, (1, 0, 2, 3, 4)) + np.transpose(X3A, (2, 1, 0, 3, 4)) # antisymmetrize A(a/bc)
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3A

def build_HR_3B(R, T, X, H):
    """Calculate the projection <jk~abc~|[ (H_N e^(T1+T2))_C*(R1p+R2p1h) ]_C|0>."""
    # moment-like terms
    X3B = -(1.0 / 2.0) * np.einsum("mcjk,abm->abcjk", H.ab.ovoo, R.aa, optimize=True) # (14)
    X3B -= np.einsum("bmjk,acm->abcjk", H.ab.vooo, R.ab, optimize=True) # (15)
    X3B += np.einsum("bcek,aej->abcjk", H.ab.vvvo, R.aa, optimize=True) # (16)
    X3B += np.einsum("bcje,aek->abcjk", H.ab.vvov, R.ab, optimize=True) # (17)
    X3B += (1.0 / 2.0) * np.einsum("baje,eck->abcjk", H.aa.vvov, R.ab, optimize=True) # (23)
    # 3-body Hbar terms factorized using intermediates
    X3B -= (1.0 / 2.0) * np.einsum("mck,abmj->abcjk", X["ab"]["ovo"], T.aa, optimize=True) # (18)
    X3B -= np.einsum("amj,bcmk->abcjk", X["aa"]["voo"], T.ab, optimize=True) # (19)
    X3B -= np.einsum("amk,bcjm->abcjk", X["ab"]["voo"], T.ab, optimize=True) # (20)
    X3B += (1.0 / 2.0) * np.einsum("abe,ecjk->abcjk", X["aa"]["vvv"], T.ab, optimize=True) # (21)
    X3B += np.einsum("ace,bejk->abcjk", X["ab"]["vvv"], T.ab, optimize=True) # (22)
    X3B -= np.transpose(X3B, (1, 0, 2, 3, 4)) # antisymmetrize A(ab)
    return X3B

def build_HR_3C(R, T, X, H):
    """Calculate the projection <j~k~ab~c~|[ (H_N e^(T1+T2))_C*(R1p+R2p1h) ]_C|0>."""
    # moment-like terms
    X3C = -(2.0 / 4.0) * np.einsum("cmkj,abm->abcjk", H.bb.vooo, R.ab, optimize=True) # (10)
    X3C += (2.0 / 4.0) * np.einsum("cbke,aej->abcjk", H.bb.vvov, R.ab, optimize=True) # (11)
    X3C += np.einsum("acek,ebj->abcjk", H.ab.vvvo, R.ab, optimize=True) # (12)
    # 3-body Hbar terms factorized using intermediates
    X3C -= (2.0 / 4.0) * np.einsum("amj,bcmk->abcjk", X["ab"]["voo"], T.bb, optimize=True) # (13)
    X3C -= np.einsum("mck,abmj->abcjk", X["ab"]["ovo"], T.ab, optimize=True) # (14)
    X3C += (2.0 / 4.0) * np.einsum("abe,ecjk->abcjk", X["ab"]["vvv"], T.bb, optimize=True) # (15)
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3C -= np.transpose(X3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3C

def build_LH_3A(L, T, H):
    """Calculate the projection <0| (L1p+L2p1h)(H_N e^(T1+T2))_C |jkabc>."""
    X3A = (3.0 / 12.0) * np.einsum("a,jkbc->abcjk", L.a, H.aa.oovv, optimize=True)
    X3A += (6.0 / 12.0) * np.einsum("abj,kc->abcjk", L.aa, H.a.ov, optimize=True)
    X3A -= (3.0 / 12.0) * np.einsum("abm,jkmc->abcjk", L.aa, H.aa.ooov, optimize=True)
    X3A += (6.0 / 12.0) * np.einsum("eck,ejab->abcjk", L.aa, H.aa.vovv, optimize=True)
    X3A -= np.transpose(X3A, (1, 0, 2, 3, 4)) + np.transpose(X3A, (2, 1, 0, 3, 4)) # antisymmetrize A(a/bc)
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3A

def build_LH_3B(L, T, H):
    """Calculate the projection <0| (L1p+L2p1h)(H_N e^(T1+T2))_C |jk~abc~>."""
    X3B = np.einsum("a,jkbc->abcjk", L.a, H.ab.oovv, optimize=True)
    X3B += (1.0 / 2.0) * np.einsum("abj,kc->abcjk", L.aa, H.b.ov, optimize=True)
    X3B += np.einsum("ack,jb->abcjk", L.ab, H.a.ov, optimize=True)
    X3B -= (1.0 / 2.0) * np.einsum("abm,jkmc->abcjk", L.aa, H.ab.ooov, optimize=True)
    X3B -= np.einsum("acm,jkbm->abcjk", L.ab, H.ab.oovo, optimize=True)
    X3B += np.einsum("aej,ekbc->abcjk", L.aa, H.ab.vovv, optimize=True)
    X3B += np.einsum("aek,jebc->abcjk", L.ab, H.ab.ovvv, optimize=True)
    X3B += (1.0 / 2.0) * np.einsum("eck,ejab->abcjk", L.ab, H.aa.vovv, optimize=True)
    X3B -= np.transpose(X3B, (1, 0, 2, 3, 4)) # antisymmetrize A(ab)
    return X3B

def build_LH_3C(L, T, H):
    """Calculate the projection <0| (L1p+L2p1h)(H_N e^(T1+T2))_C |j~k~ab~c~>."""
    X3C = (1.0 / 4.0) * np.einsum("a,jkbc->abcjk", L.a, H.bb.oovv, optimize=True)
    X3C += np.einsum("abj,kc->abcjk", L.ab, H.b.ov, optimize=True)
    X3C -= (2.0 / 4.0) * np.einsum("abm,jkmc->abcjk", L.ab, H.bb.ooov, optimize=True)
    X3C += (2.0 / 4.0) * np.einsum("aej,ekbc->abcjk", L.ab, H.bb.vovv, optimize=True)
    X3C += np.einsum("eck,ejab->abcjk", L.ab, H.ab.vovv, optimize=True)
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3C -= np.transpose(X3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3C
