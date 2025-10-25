import time
from ccpy.utilities.linear_algebra import ccpy_einsum
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
            +ccpy_einsum("bnef,ifn->ibe", H.aa.vovv, R.aa)
            +ccpy_einsum("bnef,ifn->ibe", H.ab.vovv, R.ab)
            +0.5 * ccpy_einsum("nmie,nbm->ibe", H.aa.ooov, R.aa)
            +ccpy_einsum("bmie,m->ibe", H.aa.voov, R.a)
    )
    # x2b(eb~j~)
    X["ab"]["vvo"] = (
            -ccpy_einsum("mbef,mfj->ebj", H.ab.ovvv, R.ab)
            +ccpy_einsum("mnej,mbn->ebj", H.ab.oovo, R.ab)
            -ccpy_einsum("mbej,m->ebj", H.ab.ovvo, R.a)
    )
    # x2b(ib~e~)
    X["ab"]["ovv"] = (
            +ccpy_einsum("nbfe,ifn->ibe", H.ab.ovvv, R.aa)
            +ccpy_einsum("bnef,ifn->ibe", H.bb.vovv, R.ab)
            +ccpy_einsum("nmie,nbm->ibe", H.ab.ooov, R.ab)
            -ccpy_einsum("mbie,m->ibe", H.ab.ovov, R.a)
    )

    # x2a(imj)
    X["aa"]["ooo"] = (
            +ccpy_einsum("mnjf,ifn->imj", H.aa.ooov, R.aa)
            +ccpy_einsum("mnjf,ifn->imj", H.ab.ooov, R.ab)
            -0.5 * ccpy_einsum("mnji,n->imj", H.aa.oooo, R.a)
    )
    X["aa"]["ooo"] -= np.transpose(X["aa"]["ooo"], (2, 1, 0))
    # x2b(im~j~)
    X["ab"]["ooo"] = (
            +ccpy_einsum("nmfj,ifn->imj", H.ab.oovo, R.aa)
            +ccpy_einsum("mnjf,ifn->imj", H.bb.ooov, R.ab)
            -ccpy_einsum("nmie,nej->imj", H.ab.ooov, R.ab)
            -ccpy_einsum("nmij,n->imj", H.ab.oooo, R.a)
    )
    return X

def build_HR_3A(R, T, X, H):
    """Calculate the projection <ijkbc|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    # moment-like terms
    X3A = -(6.0 / 12.0) * ccpy_einsum("cmkj,ibm->ibcjk", H.aa.vooo, R.aa)
    X3A += (3.0 / 12.0) * ccpy_einsum("cbke,iej->ibcjk", H.aa.vvov, R.aa)
    # 3-body Hbar terms factorized using intermediates
    X3A -= (3.0 / 12.0) * ccpy_einsum("imj,bcmk->ibcjk", X["aa"]["ooo"], T.aa)
    X3A += (6.0 / 12.0) * ccpy_einsum("ibe,ecjk->ibcjk", X["aa"]["ovv"], T.aa)
    X3A -= np.transpose(X3A, (3, 1, 2, 0, 4)) + np.transpose(X3A, (4, 1, 2, 3, 0)) # antisymmetrize A(i/jk)
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    return X3A

def build_HR_3B(R, T, X, H):
    """Calculate the projection <ijk~bc~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    # moment-like terms
    X3B = -ccpy_einsum("mcjk,ibm->ibcjk", H.ab.ovoo, R.aa) # (14)
    X3B -= 0.5 * ccpy_einsum("bmji,mck->ibcjk", H.aa.vooo, R.ab) # (15)
    X3B -= ccpy_einsum("bmjk,icm->ibcjk", H.ab.vooo, R.ab) # (16)
    X3B += ccpy_einsum("bcje,iek->ibcjk", H.ab.vvov, R.ab) # (17)
    X3B += 0.5 * ccpy_einsum("bcek,iej->ibcjk", H.ab.vvvo, R.aa) # (18)
    # 3-body Hbar terms factorized using intermediates
    X3B += 0.5 * ccpy_einsum("eck,ebij->ibcjk", X["ab"]["vvo"], T.aa) # (19)
    X3B -= 0.5 * ccpy_einsum("imj,bcmk->ibcjk", X["aa"]["ooo"], T.ab) # (20)
    X3B -= ccpy_einsum("imk,bcjm->ibcjk", X["ab"]["ooo"], T.ab) # (21)
    X3B += ccpy_einsum("ice,bejk->ibcjk", X["ab"]["ovv"], T.ab) # (22)
    X3B += ccpy_einsum("ibe,ecjk->ibcjk", X["aa"]["ovv"], T.ab) # (23)
    X3B -= np.transpose(X3B, (3, 1, 2, 0, 4)) # antisymmetrize (ij)
    return X3B

def build_HR_3C(R, T, X, H):
    """Calculate the projection <ij~k~b~c~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    # moment-like terms
    X3C = -ccpy_einsum("mcik,mbj->ibcjk", H.ab.ovoo, R.ab) # (10)
    X3C -= (2.0 / 4.0) * ccpy_einsum("cmkj,ibm->ibcjk", H.bb.vooo, R.ab) # (11)
    X3C += (2.0 / 4.0) * ccpy_einsum("cbke,iej->ibcjk", H.bb.vvov, R.ab) # (12)
    # 3-body Hbar terms factorized using intermediates
    X3C -= (2.0 / 4.0) * ccpy_einsum("imj,bcmk->ibcjk", X["ab"]["ooo"], T.bb) # (13)
    X3C += (2.0 / 4.0) * ccpy_einsum("ibe,ecjk->ibcjk", X["ab"]["ovv"], T.bb) # (14)
    X3C += ccpy_einsum("ebj,ecik->ibcjk", X["ab"]["vvo"], T.ab) # (15)
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3C -= np.transpose(X3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3C

def build_LH_3A(L, T, H):
    """Calculate the projection <0| (L1h+L2h1p)(H_N e^(T1+T2))_C |ijkbc>."""
    # moment-like terms
    X3A = (3.0 / 12.0) * ccpy_einsum("i,jkbc->ibcjk", L.a, H.aa.oovv)
    X3A += (6.0 / 12.0) * ccpy_einsum("ibj,kc->ibcjk", L.aa, H.a.ov)
    X3A += (3.0 / 12.0) * ccpy_einsum("iej,ekbc->ibcjk", L.aa, H.aa.vovv)
    X3A -= (6.0 / 12.0) * ccpy_einsum("mck,ijmb->ibcjk", L.aa, H.aa.ooov)
    X3A -= np.transpose(X3A, (3, 1, 2, 0, 4)) + np.transpose(X3A, (4, 1, 2, 3, 0)) # antisymmetrize A(i/jk)
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    return X3A

def build_LH_3B(L, T, H):
    """Calculate the projection <0| (L1h+L2h1p)(H_N e^(T1+T2))_C |ijk~bc~>."""
    # moment-like terms
    X3B = ccpy_einsum("i,jkbc->ibcjk", L.a, H.ab.oovv)
    X3B += (1.0 / 2.0) * ccpy_einsum("ibj,kc->ibcjk", L.aa, H.b.ov)
    X3B += ccpy_einsum("ick,jb->ibcjk", L.ab, H.a.ov)
    X3B += (1.0 / 2.0) * ccpy_einsum("iej,ekbc->ibcjk", L.aa, H.ab.vovv)
    X3B += ccpy_einsum("iek,jebc->ibcjk", L.ab, H.ab.ovvv)
    X3B -= ccpy_einsum("mbj,ikmc->ibcjk", L.aa, H.ab.ooov)
    X3B -= (1.0 / 2.0) * ccpy_einsum("mck,ijmb->ibcjk", L.ab, H.aa.ooov)
    X3B -= ccpy_einsum("icm,jkbm->ibcjk", L.ab, H.ab.oovo)
    X3B -= np.transpose(X3B, (3, 1, 2, 0, 4)) # antisymmetrize (ij)
    return X3B

def build_LH_3C(L, T, H):
    """Calculate the projection <0| (L1h+L2h1p)(H_N e^(T1+T2))_C |ij~k~b~c~>."""
    # moment-like terms
    X3C = (1.0 / 4.0) * ccpy_einsum("i,jkbc->ibcjk", L.a, H.bb.oovv)
    X3C += ccpy_einsum("ibj,kc->ibcjk", L.ab, H.b.ov)
    X3C += (2.0 / 4.0) * ccpy_einsum("iej,ekbc->ibcjk", L.ab, H.bb.vovv)
    X3C -= ccpy_einsum("mck,ijmb->ibcjk", L.ab, H.ab.ooov)
    X3C -= (2.0 / 4.0) * ccpy_einsum("ibm,jkmc->ibcjk", L.ab, H.bb.ooov)
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3C -= np.transpose(X3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3C
