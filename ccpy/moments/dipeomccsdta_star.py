"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for doubly-attached states
using the DIP-EOMCC approach with up to 4h-2p excitations"""
import time
import numpy as np
from ccpy.constants.constants import hartreetoeV

def calc_dipeomccsdta_star(T, R, L, omega, corr_energy, H, H0, system, use_RHF=False):
    """
    Calculate the ground-state DIP-EOMCCSDT(4h-2p)T(a)* correction to the DIP-EOMCCSD(3h-1p) energy.
    The specific formula used to formula the correction is
    delta(4h-2p)T(a)* = < 0 | L3*M3 | 0 >,
    where
    L3 = [R3]*,
    R3 = M3 / (omega -  D_MP),
    and
    M3 = < ijklcd | (V_N*R2)_C + (V_N*T2*R1)_C | 0 >
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    n = np.newaxis
    eps_o_a = np.diagonal(H0.a.oo); eps_v_a = np.diagonal(H0.a.vv);
    eps_o_b = np.diagonal(H0.b.oo); eps_v_b = np.diagonal(H0.b.vv);
    e4_abaa = (eps_o_a[:, n, n, n, n, n] + eps_o_b[n, :, n, n, n, n]
              -eps_v_a[n, n, :, n, n, n] - eps_v_a[n, n, n, :, n, n]
              +eps_o_a[n, n, n, n, :, n] + eps_o_a[n, n, n, n, n, :])
    e4_abab = (eps_o_a[:, n, n, n, n, n] + eps_o_b[n, :, n, n, n, n]
              -eps_v_a[n, n, :, n, n, n] - eps_v_b[n, n, n, :, n, n]
              +eps_o_a[n, n, n, n, :, n] + eps_o_b[n, n, n, n, n, :])

    X = get_dipeom4_intermediates(H0, R)

    # update R4
    M4B = build_HR_4B(R, T, H0, X)
    L4B = M4B/(omega + e4_abaa)
    dA_abaa = (1.0 / 12.0) * np.einsum("ijcdkl,ijcdkl->", L4B, M4B, optimize=True)

    M4C = build_HR_4C(R, T, H0, X)
    L4C = M4C/(omega + e4_abab)
    dA_abab = (1.0 / 4.0) * np.einsum("ijcdkl,ijcdkl->", L4C, M4C, optimize=True)

    if use_RHF:
        correction_A = 2.0 * dA_abaa + dA_abab
    else:
        e4_abbb = (eps_o_a[:, n, n, n, n, n] + eps_o_b[n, :, n, n, n, n]
                  -eps_v_b[n, n, :, n, n, n] - eps_v_b[n, n, n, :, n, n]
                  +eps_o_b[n, n, n, n, :, n] + eps_o_b[n, n, n, n, n, :])

        M4D = build_HR_4D(R, T, H0, X)
        L4D = M4D/(omega + e4_abbb)
        dA_abbb = (1.0 / 12.0) * np.einsum("ijcdkl,ijcdkl->", L4D, M4D, optimize=True)

        correction_A = dA_abaa + dA_abab + dA_abbb

    # divide by norm of R vector as a mimic to <L|R> = 1
    rnorm = np.einsum("ij,ij->", R.ab, R.ab, optimize=True)
    rnorm += 0.5 * np.einsum("ijck,ijck->", R.aba, R.aba, optimize=True)
    rnorm += 0.5 * np.einsum("ijck,ijck->", R.abb, R.abb, optimize=True)
    correction_A /= rnorm

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    energy_A = corr_energy + omega + correction_A
    total_energy_A = system.reference_energy + energy_A

    print("")
    print('   DIP-EOMCC(4h-2p)T(a)* Calculation Summary')
    print('   -------------------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds")
    print("   DIP-EOMCC(3h-1p)T(a) = {:>10.10f}    ω = {:>10.10f}     VEE = {:>10.5f} eV".format(system.reference_energy + corr_energy + omega, omega, hartreetoeV * omega))
    print(
        "   DIP-EOMCC(4h-2p)T(a)* = {:>10.10f}     δ_A = {:>10.10f}     VEE = {:>10.5f} eV".format(
            total_energy_A, correction_A, (omega + correction_A) * hartreetoeV
        )
    )
    Ecrcc23 = {"A": total_energy_A, "B": 0.0, "C": 0.0, "D": 0.0}
    delta23 = {"A": correction_A, "B": 0.0, "C": 0.0, "D": 0.0}

    return Ecrcc23, delta23

def build_HR_4B(R, T, H0, X):
    ### Moment-like terms < ij~klcd | (H(2)[R(2h) + R(3h-1p)])_C | 0 > ###
    x4b = -(6.0 / 12.0) * np.einsum("cmkl,ijdm->ijcdkl", H0.aa.vooo, R.aba, optimize=True)
    x4b -= (6.0 / 12.0) * np.einsum("cmkj,imdl->ijcdkl", H0.ab.vooo, R.aba, optimize=True)
    x4b += (3.0 / 12.0) * np.einsum("cdke,ijel->ijcdkl", H0.aa.vvov, R.aba, optimize=True)
    x4b += (6.0 / 12.0) * np.einsum("ijde,cekl->ijcdkl", X["aba"]["oovv"], T.aa, optimize=True)
    x4b -= (3.0 / 12.0) * np.einsum("ijml,cdkm->ijcdkl", X["aba"]["oooo"], T.aa, optimize=True)
    x4b += (6.0 / 12.0) * np.einsum("ieck,delj->ijcdkl", X["aba"]["ovvo"], T.ab, optimize=True)
    # antisymmetrize A(ikl)A(cd)
    x4b -= np.transpose(x4b, (0, 1, 3, 2, 4, 5)) # A(cd)
    x4b -= np.transpose(x4b, (0, 1, 2, 3, 5, 4)) # A(kl)
    x4b -= np.transpose(x4b, (4, 1, 2, 3, 0, 5)) + np.transpose(x4b, (5, 1, 2, 3, 4, 0)) # A(i/kl)
    return x4b

def build_HR_4C(R, T, H0, X):
    ### Moment-like terms < ij~kl~cd~ | (H(2)[R(2h) + R(3h-1p)])_C | 0 > ###
    x4c = -np.einsum("mdkl,ijcm->ijcdkl", H0.ab.ovoo, R.aba, optimize=True)
    x4c -= np.einsum("cmkl,ijdm->ijcdkl", H0.ab.vooo, R.abb, optimize=True)
    x4c += (2.0 / 4.0) * np.einsum("cdel,ijek->ijcdkl", H0.ab.vvvo, R.aba, optimize=True)
    x4c += (2.0 / 4.0) * np.einsum("cdke,ijel->ijcdkl", H0.ab.vvov, R.abb, optimize=True)
    x4c -= (1.0 / 4.0) * np.einsum("cmki,mjdl->ijcdkl", H0.aa.vooo, R.abb, optimize=True)
    x4c -= (1.0 / 4.0) * np.einsum("dmlj,imck->ijcdkl", H0.bb.vooo, R.aba, optimize=True)
    x4c -= (2.0 / 4.0) * np.einsum("ijml,cdkm->ijcdkl", X["abb"]["oooo"], T.ab, optimize=True)
    x4c -= (2.0 / 4.0) * np.einsum("ijmk,cdml->ijcdkl", X["aba"]["oooo"], T.ab, optimize=True)
    x4c += np.einsum("ijce,edkl->ijcdkl", X["aba"]["oovv"], T.ab, optimize=True)
    x4c += np.einsum("ijde,cekl->ijcdkl", X["abb"]["oovv"], T.ab, optimize=True)
    x4c += (1.0 / 4.0) * np.einsum("ieck,edjl->ijcdkl", X["aba"]["ovvo"], T.bb, optimize=True)
    x4c += (1.0 / 4.0) * np.einsum("ejdl,ecik->ijcdkl", X["abb"]["vovo"], T.aa, optimize=True)
    # antisymmetrize A(ik)A(jl)
    x4c -= np.transpose(x4c, (4, 1, 2, 3, 0, 5)) # A(ik)
    x4c -= np.transpose(x4c, (0, 5, 2, 3, 4, 1)) # A(jl)
    return x4c

def build_HR_4D(R, T, H0, X):
    ### Moment-like terms < ij~k~l~c~d~ | (H(2)[R(2h) + R(3h-1p)])_C | 0 > ###
    x4d = -(6.0 / 12.0) * np.einsum("cmkl,ijdm->ijcdkl", H0.bb.vooo, R.abb, optimize=True)
    x4d -= (6.0 / 12.0) * np.einsum("mcik,mjdl->ijcdkl", H0.ab.ovoo, R.abb, optimize=True)
    x4d += (3.0 / 12.0) * np.einsum("cdke,ijel->ijcdkl", H0.bb.vvov, R.abb, optimize=True)
    x4d -= (3.0 / 12.0) * np.einsum("ijml,cdkm->ijcdkl", X["abb"]["oooo"], T.bb, optimize=True)
    x4d += (6.0 / 12.0) * np.einsum("ijde,cekl->ijcdkl", X["abb"]["oovv"], T.bb, optimize=True)
    x4d += (6.0 / 12.0) * np.einsum("ejck,edil->ijcdkl", X["abb"]["vovo"], T.ab, optimize=True)
    # antisymmetrize A(jkl)A(cd)
    x4d -= np.transpose(x4d, (0, 1, 3, 2, 4, 5)) # A(cd)
    x4d -= np.transpose(x4d, (0, 1, 2, 3, 5, 4)) # A(kl)
    x4d -= np.transpose(x4d, (0, 4, 2, 3, 1, 5)) + np.transpose(x4d, (0, 5, 2, 3, 4, 1)) # A(j/kl)
    return x4d

def get_dipeom4_intermediates(H, R):

    # Create dictionary to store intermediates, which have spincases that resemble those of the DIP R operator itself
    X = {"ab": {"vo": np.array([0.0]), "ov": np.array([0.0]), "vv": np.array([0.0])},
         "aba": {"oooo": np.array([0.0]), "oovv": np.array([0.0]), "ovvo": np.array([0.0])},
         "abb": {"oooo": np.array([0.0]), "oovv": np.array([0.0]), "vovo": np.array([0.0])}}

    ### two-body intermediates ###
    # x(ij~ce) [1]
    X["aba"]["oovv"] = (
            # np.einsum("cnef,ijfn->ijce", H.aa.vovv, R.aba, optimize=True)
            # + np.einsum("cnef,ijfn->ijce", H.ab.vovv, R.abb, optimize=True)
            + np.einsum("cmie,mj->ijce", H.aa.voov, R.ab, optimize=True) # flip sign, h2a(vovo) -> -h2a(voov)
            # + np.einsum("mnej,incm->ijce", H.ab.oovo, R.aba, optimize=True)
            # + 0.5 * np.einsum("mnie,mjcn->ijce", H.aa.ooov, R.aba, optimize=True)
            - np.einsum("cmej,im->ijce", H.ab.vovo, R.ab, optimize=True)
    )

    # x(ij~mk) [2]
    X["aba"]["oooo"] = (
            # np.einsum("mnkf,ijfn->ijmk", H.aa.ooov, R.aba, optimize=True)
            # + np.einsum("mnkf,ijfn->ijmk", H.ab.ooov, R.abb, optimize=True)
            # - 0.5 * np.einsum("mnej,inek->ijmk", H.ab.oovo, R.aba, optimize=True)
            - 0.5 * np.einsum("nmik,nj->ijmk", H.aa.oooo, R.ab, optimize=True)
            - np.einsum("mnkj,in->ijmk", H.ab.oooo, R.ab, optimize=True)
    )
    # antisymmetrize A(ik)
    X["aba"]["oooo"] -= np.transpose(X["aba"]["oooo"], (3, 1, 2, 0))

    # x(ieck) [3]
    X["aba"]["ovvo"] = (
            # np.einsum("nmie,nmck->ieck", H.ab.ooov, R.aba, optimize=True)
            # - 0.5 * np.einsum("cmfe,imfk->ieck", H.ab.vovv, R.aba, optimize=True)
            - np.einsum("cmke,im->ieck", H.ab.voov, R.ab, optimize=True)
    )
    # antisymmetrize A(ik)
    X["aba"]["ovvo"] -= np.transpose(X["aba"]["ovvo"], (3, 1, 2, 0))

    # x(ijde) [4]
    X["abb"]["oovv"] = (
            # np.einsum("ndfe,ijfn->ijde", H.ab.ovvv, R.aba, optimize=True)
            # + np.einsum("dnef,ijfn->ijde", H.bb.vovv, R.abb, optimize=True)
            + np.einsum("dmje,im->ijde", H.bb.voov, R.ab, optimize=True) # flip sign, h2c(vovo) -> -h2c(voov)
            # + 0.5 * np.einsum("mnje,imdn->ijde", H.bb.ooov, R.abb, optimize=True)
            # + np.einsum("nmie,njdm->ijde", H.ab.ooov, R.abb, optimize=True)
            - np.einsum("mdie,mj->ijde", H.ab.ovov, R.ab, optimize=True)
    )

    # x(ij~m~k~) [5]
    X["abb"]["oooo"] = (
            # np.einsum("nmfk,ijfn->ijmk", H.ab.oovo, R.aba, optimize=True)
            # + np.einsum("mnkf,ijfn->ijmk", H.bb.ooov, R.abb, optimize=True)
            - np.einsum("nmik,nj->ijmk", H.ab.oooo, R.ab, optimize=True)
            # - 0.5 * np.einsum("nmie,njek->ijmk", H.ab.ooov, R.abb, optimize=True)
            - 0.5 * np.einsum("nmjk,in->ijmk", H.bb.oooo, R.ab, optimize=True)
    )
    # antisymmetrize A(jk)
    X["abb"]["oooo"] -= np.transpose(X["abb"]["oooo"], (0, 3, 2, 1))

    # x(ejdl) [6]
    X["abb"]["vovo"] = (
            # - 0.5 * np.einsum("mdef,mjfl->ejdl", H.ab.ovvv, R.abb, optimize=True)
            # + np.einsum("mnej,mndl->ejdl", H.ab.oovo, R.abb, optimize=True)
            - np.einsum("mdel,mj->ejdl", H.ab.ovvo, R.ab, optimize=True)
    )
    # antisymmetrize A(jl)
    X["abb"]["vovo"] -= np.transpose(X["abb"]["vovo"], (0, 3, 2, 1))
    return X
