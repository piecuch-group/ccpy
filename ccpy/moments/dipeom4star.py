"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for doubly-attached states
using the DIP-EOMCC approach with up to 4h-2p excitations"""
import time
import numpy as np
from ccpy.constants.constants import hartreetoeV

def calc_dipeom4star(T, R, L, omega, corr_energy, H, H0, system, use_RHF=False):
    """
    Calculate the ground-state CR-EOMCC(2,3) correction to the EOMCCSD energy.
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

    X = get_dipeom4_intermediates(H, T, R)

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

def get_dipeom4_intermediates(H, T, R):

    # Create dictionary to store intermediates, which have spincases that resemble those of the DIP R operator itself
    X = {"ab": {"vo": np.array([0.0]), "ov": np.array([0.0]), "vv": np.array([0.0])},
         "aba": {"oooo": np.array([0.0]), "oovv": np.array([0.0]), "ovvo": np.array([0.0])},
         "abb": {"oooo": np.array([0.0]), "oovv": np.array([0.0]), "vovo": np.array([0.0])}}

    Q1 = -np.einsum("mnfe,an->amef", H.aa.oovv, T.a, optimize=True)
    I2A_vovv = H.aa.vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H.aa.oovv, T.a, optimize=True)
    I2A_ooov = H.aa.ooov + 0.5 * Q1

    Q1 = -np.einsum("nmef,an->amef", H.ab.oovv, T.a, optimize=True)
    I2B_vovv = H.ab.vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H.ab.oovv, T.a, optimize=True)
    I2B_ooov = H.ab.ooov + 0.5 * Q1

    Q1 = -np.einsum("mnef,an->maef", H.ab.oovv, T.b, optimize=True)
    I2B_ovvv = H.ab.ovvv + 0.5 * Q1

    Q1 = np.einsum("nmef,fi->nmei", H.ab.oovv, T.b, optimize=True)
    I2B_oovo = H.ab.oovo + 0.5 * Q1

    Q1 = -np.einsum("nmef,an->amef", H.bb.oovv, T.b, optimize=True)
    I2C_vovv = H.bb.vovv + 0.5 * Q1

    Q1 = np.einsum("mnfe,fi->mnie", H.bb.oovv, T.b, optimize=True)
    I2C_ooov = H.bb.ooov + 0.5 * Q1

    Q1 = +np.einsum("nmje,ei->mnij", I2A_ooov, T.a, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    I_aa_oooo = H.aa.oooo - Q1

    I_ab_oooo = H.ab.oooo - (
            np.einsum("mnej,ei->mnij", I2B_oovo, T.a, optimize=True)
            + np.einsum("mnie,ej->mnij", I2B_ooov, T.b, optimize=True)
    )

    Q1 = +np.einsum("nmje,ei->mnij", I2C_ooov, T.b, optimize=True)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    I_bb_oooo = H.bb.oooo - Q1

    I_aa_voov = H.aa.voov - (
            np.einsum("amfe,fi->amie", I2A_vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", I2A_ooov, T.a, optimize=True)
    )

    I_ab_voov = H.ab.voov - (
            np.einsum("amfe,fi->amie", I2B_vovv, T.a, optimize=True)
            - np.einsum("nmie,an->amie", I2B_ooov, T.a, optimize=True)
    )

    I_ab_ovvo = H.ab.ovvo - (
            np.einsum("maef,fi->maei", I2B_ovvv, T.b, optimize=True)
            - np.einsum("mnei,an->maei", I2B_oovo, T.b, optimize=True)
    )

    I_ab_ovov = H.ab.ovov - (
            np.einsum("mafe,fi->maie", I2B_ovvv, T.a, optimize=True)
            - np.einsum("mnie,an->maie", I2B_ooov, T.b, optimize=True)
    )

    I_ab_vovo = H.ab.vovo - (
            - np.einsum("nmei,an->amei", I2B_oovo, T.a, optimize=True)
            + np.einsum("amef,fi->amei", I2B_vovv, T.b, optimize=True)
    )

    I_bb_voov = H.bb.voov - (
            np.einsum("amfe,fi->amie", I2C_vovv, T.b, optimize=True)
            - np.einsum("nmie,an->amie", I2C_ooov, T.b, optimize=True)
    )

    ### two-body intermediates ###
    # x(ij~ce) [1]
    X["aba"]["oovv"] = (
            # np.einsum("cnef,ijfn->ijce", H.aa.vovv, R.aba, optimize=True)
            # + np.einsum("cnef,ijfn->ijce", H.ab.vovv, R.abb, optimize=True)
            + np.einsum("cmie,mj->ijce", I_aa_voov, R.ab, optimize=True) # flip sign, h2a(vovo) -> -h2a(voov)
            # + np.einsum("mnej,incm->ijce", H.ab.oovo, R.aba, optimize=True)
            # + 0.5 * np.einsum("mnie,mjcn->ijce", H.aa.ooov, R.aba, optimize=True)
            - np.einsum("cmej,im->ijce", I_ab_vovo, R.ab, optimize=True)
    )

    # x(ij~mk) [2]
    X["aba"]["oooo"] = (
            # np.einsum("mnkf,ijfn->ijmk", H.aa.ooov, R.aba, optimize=True)
            # + np.einsum("mnkf,ijfn->ijmk", H.ab.ooov, R.abb, optimize=True)
            # - 0.5 * np.einsum("mnej,inek->ijmk", H.ab.oovo, R.aba, optimize=True)
            - 0.5 * np.einsum("nmik,nj->ijmk", I_aa_oooo, R.ab, optimize=True)
            - np.einsum("mnkj,in->ijmk", I_ab_oooo, R.ab, optimize=True)
    )
    # antisymmetrize A(ik)
    X["aba"]["oooo"] -= np.transpose(X["aba"]["oooo"], (3, 1, 2, 0))

    # x(ieck) [3]
    X["aba"]["ovvo"] = (
            # np.einsum("nmie,nmck->ieck", H.ab.ooov, R.aba, optimize=True)
            # - 0.5 * np.einsum("cmfe,imfk->ieck", H.ab.vovv, R.aba, optimize=True)
            - np.einsum("cmke,im->ieck", I_ab_voov, R.ab, optimize=True)
    )
    # antisymmetrize A(ik)
    X["aba"]["ovvo"] -= np.transpose(X["aba"]["ovvo"], (3, 1, 2, 0))

    # x(ijde) [4]
    X["abb"]["oovv"] = (
            # np.einsum("ndfe,ijfn->ijde", H.ab.ovvv, R.aba, optimize=True)
            # + np.einsum("dnef,ijfn->ijde", H.bb.vovv, R.abb, optimize=True)
            + np.einsum("dmje,im->ijde", I_bb_voov, R.ab, optimize=True) # flip sign, h2c(vovo) -> -h2c(voov)
            # + 0.5 * np.einsum("mnje,imdn->ijde", H.bb.ooov, R.abb, optimize=True)
            # + np.einsum("nmie,njdm->ijde", H.ab.ooov, R.abb, optimize=True)
            - np.einsum("mdie,mj->ijde", I_ab_ovov, R.ab, optimize=True)
    )

    # x(ij~m~k~) [5]
    X["abb"]["oooo"] = (
            # np.einsum("nmfk,ijfn->ijmk", H.ab.oovo, R.aba, optimize=True)
            # + np.einsum("mnkf,ijfn->ijmk", H.bb.ooov, R.abb, optimize=True)
            - np.einsum("nmik,nj->ijmk", I_ab_oooo, R.ab, optimize=True)
            # - 0.5 * np.einsum("nmie,njek->ijmk", H.ab.ooov, R.abb, optimize=True)
            - 0.5 * np.einsum("nmjk,in->ijmk", I_bb_oooo, R.ab, optimize=True)
    )
    # antisymmetrize A(jk)
    X["abb"]["oooo"] -= np.transpose(X["abb"]["oooo"], (0, 3, 2, 1))

    # x(ejdl) [6]
    X["abb"]["vovo"] = (
            # - 0.5 * np.einsum("mdef,mjfl->ejdl", H.ab.ovvv, R.abb, optimize=True)
            # + np.einsum("mnej,mndl->ejdl", H.ab.oovo, R.abb, optimize=True)
            - np.einsum("mdel,mj->ejdl", I_ab_ovvo, R.ab, optimize=True)
    )
    # antisymmetrize A(jl)
    X["abb"]["vovo"] -= np.transpose(X["abb"]["vovo"], (0, 3, 2, 1))
    return X
