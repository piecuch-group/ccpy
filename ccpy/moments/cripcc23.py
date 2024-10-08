import time
import numpy as np

from ccpy.constants.constants import hartreetoeV
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.lib.core import cripcc_loops


def calc_cripcc23(T, R, L, omega, corr_energy, H, H0, system, use_RHF=False):
    """
    Calculate the correction for 3h-2p correlations to the IP-EOMCCSD(2h-1p) energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    # compute H*R intermediates
    X = get_ipeom23_intermediates(H, R)

    #### aaa correction ####
    # calculate intermediates
    M3A = build_HR_3A(R, T, X, H)
    L3A = build_LH_3A(L, T, H) 
    # perform correction in-loop
    dA_aaa, dB_aaa, dC_aaa, dD_aaa = cripcc_loops.crcc23a(M3A, L3A, omega,
                                                                       H0.a.oo, H0.a.vv, H.a.oo, H.a.vv,
                                                                       H.aa.vvvv, H.aa.oooo, H.aa.voov,
                                                                       d3aaa_o, d3aaa_v)
    #### aab correction ####
    # calculate intermediates
    M3B = build_HR_3B(R, T, X, H)
    L3B = build_LH_3B(L, T, H)
    # perform correction in-loop
    dA_aab, dB_aab, dC_aab, dD_aab = cripcc_loops.crcc23b(M3B, L3B, omega,
                                                                       H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                                                                       H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                                       H.aa.oooo, H.aa.voov,
                                                                       H.ab.vvvv, H.ab.oooo, H.ab.ovov, H.ab.vovo,
                                                                       H.bb.voov,
                                                                       d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v)
    #### abb correction ####
    # calculate intermediates
    M3C = build_HR_3C(R, T, X, H) 
    L3C = build_LH_3C(L, T, H)
    # perform correction in-loop
    dA_abb, dB_abb, dC_abb, dD_abb = cripcc_loops.crcc23c(M3C, L3C, omega,
                                                                       H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv,
                                                                       H.a.oo, H.a.vv, H.b.oo, H.b.vv,
                                                                       H.ab.oooo, H.ab.ovov, H.ab.vovo,
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

    print('   CR-IP-EOMCCSD(2h-1p,3h-2p) Calculation Summary')
    print('   -------------------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   IP-EOMCCSD(2h-1p) = {:>10.10f}    ω = {:>10.10f}     VEE = {:>10.5f} eV".format(system.reference_energy + corr_energy + omega, omega, hartreetoeV * omega))
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

def get_ipeom23_intermediates(H, R):

    # These intermediates will be 3-index quantities, which are not
    # set up in the models at the moment. We will just use a dictionary
    # as a workaround for now.

    X = {"aa" : {}, "ab" : {}}

    # x2a(ibe)
    X["aa"]["ovv"] = (
            +np.einsum("bnef,ifn->ibe", H.aa.vovv, R.aa, optimize=True)
            +np.einsum("bnef,ifn->ibe", H.ab.vovv, R.ab, optimize=True)
            +0.5 * np.einsum("nmie,nbm->ibe", H.aa.ooov, R.aa, optimize=True)
            +np.einsum("bmie,m->ibe", H.aa.voov, R.a, optimize=True)
    )
    # x2b(eb~j~)
    X["ab"]["vvo"] = (
            -np.einsum("mbef,mfj->ebj", H.ab.ovvv, R.ab, optimize=True)
            +np.einsum("mnej,mbn->ebj", H.ab.oovo, R.ab, optimize=True)
            -np.einsum("mbej,m->ebj", H.ab.ovvo, R.a, optimize=True)
    )
    # x2b(ib~e~)
    X["ab"]["ovv"] = (
            +np.einsum("nbfe,ifn->ibe", H.ab.ovvv, R.aa, optimize=True)
            +np.einsum("bnef,ifn->ibe", H.bb.vovv, R.ab, optimize=True)
            +np.einsum("nmie,nbm->ibe", H.ab.ooov, R.ab, optimize=True)
            -np.einsum("mbie,m->ibe", H.ab.ovov, R.a, optimize=True)
    )

    # x2a(imj)
    X["aa"]["ooo"] = (
            +np.einsum("mnjf,ifn->imj", H.aa.ooov, R.aa, optimize=True)
            +np.einsum("mnjf,ifn->imj", H.ab.ooov, R.ab, optimize=True)
            -0.5 * np.einsum("mnji,n->imj", H.aa.oooo, R.a, optimize=True)
    )
    X["aa"]["ooo"] -= np.transpose(X["aa"]["ooo"], (2, 1, 0))
    # x2b(im~j~)
    X["ab"]["ooo"] = (
            +np.einsum("nmfj,ifn->imj", H.ab.oovo, R.aa, optimize=True)
            +np.einsum("mnjf,ifn->imj", H.bb.ooov, R.ab, optimize=True)
            -np.einsum("nmie,nej->imj", H.ab.ooov, R.ab, optimize=True)
            -np.einsum("nmij,n->imj", H.ab.oooo, R.a, optimize=True)
    )
    return X

def build_HR_3A(R, T, X, H):
    """Calculate the projection <ijkbc|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    # moment-like terms
    X3A = -(6.0 / 12.0) * np.einsum("cmkj,ibm->ibcjk", H.aa.vooo, R.aa, optimize=True)
    X3A += (3.0 / 12.0) * np.einsum("cbke,iej->ibcjk", H.aa.vvov, R.aa, optimize=True)
    # 3-body Hbar terms factorized using intermediates
    X3A -= (3.0 / 12.0) * np.einsum("imj,bcmk->ibcjk", X["aa"]["ooo"], T.aa, optimize=True)
    X3A += (6.0 / 12.0) * np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovv"], T.aa, optimize=True)
    X3A -= np.transpose(X3A, (3, 1, 2, 0, 4)) + np.transpose(X3A, (4, 1, 2, 3, 0)) # antisymmetrize A(i/jk)
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    return X3A

def build_HR_3B(R, T, X, H):
    """Calculate the projection <ijk~bc~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    # moment-like terms
    X3B = -np.einsum("mcjk,ibm->ibcjk", H.ab.ovoo, R.aa, optimize=True) # (14)
    X3B -= 0.5 * np.einsum("bmji,mck->ibcjk", H.aa.vooo, R.ab, optimize=True) # (15)
    X3B -= np.einsum("bmjk,icm->ibcjk", H.ab.vooo, R.ab, optimize=True) # (16)
    X3B += np.einsum("bcje,iek->ibcjk", H.ab.vvov, R.ab, optimize=True) # (17)
    X3B += 0.5 * np.einsum("bcek,iej->ibcjk", H.ab.vvvo, R.aa, optimize=True) # (18)
    # 3-body Hbar terms factorized using intermediates
    X3B += 0.5 * np.einsum("eck,ebij->ibcjk", X["ab"]["vvo"], T.aa, optimize=True) # (19)
    X3B -= 0.5 * np.einsum("imj,bcmk->ibcjk", X["aa"]["ooo"], T.ab, optimize=True) # (20)
    X3B -= np.einsum("imk,bcjm->ibcjk", X["ab"]["ooo"], T.ab, optimize=True) # (21)
    X3B += np.einsum("ice,bejk->ibcjk", X["ab"]["ovv"], T.ab, optimize=True) # (22)
    X3B += np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovv"], T.ab, optimize=True) # (23)
    X3B -= np.transpose(X3B, (3, 1, 2, 0, 4)) # antisymmetrize (ij)
    return X3B

def build_HR_3C(R, T, X, H):
    """Calculate the projection <ij~k~b~c~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    # moment-like terms
    X3C = -np.einsum("mcik,mbj->ibcjk", H.ab.ovoo, R.ab, optimize=True) # (10)
    X3C -= (2.0 / 4.0) * np.einsum("cmkj,ibm->ibcjk", H.bb.vooo, R.ab, optimize=True) # (11)
    X3C += (2.0 / 4.0) * np.einsum("cbke,iej->ibcjk", H.bb.vvov, R.ab, optimize=True) # (12)
    # 3-body Hbar terms factorized using intermediates
    X3C -= (2.0 / 4.0) * np.einsum("imj,bcmk->ibcjk", X["ab"]["ooo"], T.bb, optimize=True) # (13)
    X3C += (2.0 / 4.0) * np.einsum("ibe,ecjk->ibcjk", X["ab"]["ovv"], T.bb, optimize=True) # (14)
    X3C += np.einsum("ebj,ecik->ibcjk", X["ab"]["vvo"], T.ab, optimize=True) # (15)
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3C -= np.transpose(X3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3C

def build_LH_3A(L, T, H):
    """Calculate the projection <0| (L1h+L2h1p)(H_N e^(T1+T2))_C |ijkbc>."""
    # moment-like terms
    X3A = (3.0 / 12.0) * np.einsum("i,jkbc->ibcjk", L.a, H.aa.oovv, optimize=True)
    X3A += (6.0 / 12.0) * np.einsum("ibj,kc->ibcjk", L.aa, H.a.ov, optimize=True)
    X3A += (3.0 / 12.0) * np.einsum("iej,ekbc->ibcjk", L.aa, H.aa.vovv, optimize=True)
    X3A -= (6.0 / 12.0) * np.einsum("mck,ijmb->ibcjk", L.aa, H.aa.ooov, optimize=True)
    X3A -= np.transpose(X3A, (3, 1, 2, 0, 4)) + np.transpose(X3A, (4, 1, 2, 3, 0)) # antisymmetrize A(i/jk)
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    return X3A

def build_LH_3B(L, T, H):
    """Calculate the projection <0| (L1h+L2h1p)(H_N e^(T1+T2))_C |ijk~bc~>."""
    # moment-like terms
    X3B = np.einsum("i,jkbc->ibcjk", L.a, H.ab.oovv, optimize=True)
    X3B += (1.0 / 2.0) * np.einsum("ibj,kc->ibcjk", L.aa, H.b.ov, optimize=True)
    X3B += np.einsum("ick,jb->ibcjk", L.ab, H.a.ov, optimize=True)
    X3B += (1.0 / 2.0) * np.einsum("iej,ekbc->ibcjk", L.aa, H.ab.vovv, optimize=True)
    X3B += np.einsum("iek,jebc->ibcjk", L.ab, H.ab.ovvv, optimize=True)
    X3B -= np.einsum("mbj,ikmc->ibcjk", L.aa, H.ab.ooov, optimize=True)
    X3B -= (1.0 / 2.0) * np.einsum("mck,ijmb->ibcjk", L.ab, H.aa.ooov, optimize=True)
    X3B -= np.einsum("icm,jkbm->ibcjk", L.ab, H.ab.oovo, optimize=True)
    X3B -= np.transpose(X3B, (3, 1, 2, 0, 4)) # antisymmetrize (ij)
    return X3B

def build_LH_3C(L, T, H):
    """Calculate the projection <0| (L1h+L2h1p)(H_N e^(T1+T2))_C |ij~k~b~c~>."""
    # moment-like terms
    X3C = (1.0 / 4.0) * np.einsum("i,jkbc->ibcjk", L.a, H.bb.oovv, optimize=True)
    X3C += np.einsum("ibj,kc->ibcjk", L.ab, H.b.ov, optimize=True)
    X3C += (2.0 / 4.0) * np.einsum("iej,ekbc->ibcjk", L.ab, H.bb.vovv, optimize=True)
    X3C -= np.einsum("mck,ijmb->ibcjk", L.ab, H.ab.ooov, optimize=True)
    X3C -= (2.0 / 4.0) * np.einsum("ibm,jkmc->ibcjk", L.ab, H.bb.ooov, optimize=True)
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3C -= np.transpose(X3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3C
