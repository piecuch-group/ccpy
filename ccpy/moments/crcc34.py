"""Functions to calculate the ground-state CR-CC(3,4) quadruples correction to CCSDT."""
import time

import numpy as np
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal

def calc_crcc34(T, L, corr_energy, H, H0, system, use_RHF):
    """
    Calculate the ground-state CR-CC(3,4) correction to the CCSDT energy.
    """
    t_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # build MP denoms in full
    oa = np.diagonal(H0.a.oo)
    ob = np.diagonal(H0.b.oo)
    va = np.diagonal(H0.a.vv)
    vb = np.diagonal(H0.b.vv)
    n = np.newaxis
    e4_aaaa = 1.0 / (- va[:, n, n, n, n, n, n, n] - va[n, :, n, n, n, n, n, n] - va[n, n, :, n, n, n, n, n] - va[n, n, n, :, n, n, n, n]
                        + oa[n, n, n, n, :, n, n, n] + oa[n, n, n, n, n, :, n, n] + oa[n, n, n, n, n, n, :, n] + oa[n, n, n, n, n, n, n, :])
    e4_aaab = 1.0 / (- va[:, n, n, n, n, n, n, n] - va[n, :, n, n, n, n, n, n] - va[n, n, :, n, n, n, n, n] - vb[n, n, n, :, n, n, n, n]
                        + oa[n, n, n, n, :, n, n, n] + oa[n, n, n, n, n, :, n, n] + oa[n, n, n, n, n, n, :, n] + ob[n, n, n, n, n, n, n, :])
    e4_aabb = 1.0 / (- va[:, n, n, n, n, n, n, n] - va[n, :, n, n, n, n, n, n] - vb[n, n, :, n, n, n, n, n] - vb[n, n, n, :, n, n, n, n]
                        + oa[n, n, n, n, :, n, n, n] + oa[n, n, n, n, n, :, n, n] + ob[n, n, n, n, n, n, :, n] + ob[n, n, n, n, n, n, n, :])
    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    # form the diagonal part of the h(vvvv) elements
    nua, nub, noa, nob = T.ab.shape
    h_aa_vvvv = np.zeros((nua, nua))
    for a in range(nua):
        for b in range(a, nua):
            h_aa_vvvv[a, b] = H.aa.vvvv[a, b, a, b]
            h_aa_vvvv[b, a] = h_aa_vvvv[a, b]
    h_ab_vvvv = np.zeros((nua, nub))
    for a in range(nua):
        for b in range(nub):
            h_ab_vvvv[a, b] = H.ab.vvvv[a, b, a, b]
    h_bb_vvvv = np.zeros((nub, nub))
    for a in range(nub):
        for b in range(a, nub):
            h_bb_vvvv[a, b] = H.bb.vvvv[a, b, a, b]
            h_bb_vvvv[b, a] = h_bb_vvvv[a, b]

    #### aaaa correction ####
    M4_aaaa = build_m4a(H, T)
    L4_aaaa = build_l4a(H, L)
    L4_aaaa *= e4_aaaa
    dA_aaaa = (1.0 / 576.0) * np.sum(M4_aaaa * L4_aaaa)
    #### aaab correction ####
    M4_aaab = build_m4b(H, T)
    L4_aaab = build_l4b(H, L)
    L4_aaab *= e4_aaab
    dA_aaab = (1.0 / 36.0) * np.sum(M4_aaab * L4_aaab)
    #### aabb correction ####
    M4_aabb = build_m4c(H, T)
    L4_aabb = build_l4c(H, L)
    L4_aabb *= e4_aabb
    dA_aabb = (1.0 / 16.0) * np.sum(M4_aabb * L4_aabb)
    if use_RHF:
        correction_A = 2.0 * dA_aaaa + 2.0 * dA_aaab + dA_aabb
        correction_B = 0.0
        correction_C = 0.0
        correction_D = 0.0
    else:
        pass

    t_end = time.perf_counter()
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(t_end - t_start, 60)

    energy_A = corr_energy + correction_A
    energy_B = corr_energy + correction_B
    energy_C = corr_energy + correction_C
    energy_D = corr_energy + correction_D

    total_energy_A = system.reference_energy + energy_A
    total_energy_B = system.reference_energy + energy_B
    total_energy_C = system.reference_energy + energy_C
    total_energy_D = system.reference_energy + energy_D

    print('   CR-CC(3,4) Calculation Summary')
    print('   -------------------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("   CCSDT = {:>10.10f}".format(system.reference_energy + corr_energy))
    print(
        "   CR-CC(3,4)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CR-CC(3,4)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CR-CC(3,4)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CR-CC(3,4)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )

    Ecrcc34 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    delta34 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}
    
    return Ecrcc34, delta34

def build_m4a(H, T):
    # <ijklabcd | H(2) | 0 >
    m4a = -(144.0 / 576.0) * np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.aa, T.aa, optimize=True)  # (jl/i/k)(bc/a/d) = 12 * 12 = 144
    m4a += (36.0 / 576.0) * np.einsum("mnij,adml,bcnk->abcdijkl", H.aa.oooo, T.aa, T.aa, optimize=True)  # (ij/kl)(bc/ad) = 6 * 6 = 36
    m4a += (36.0 / 576.0) * np.einsum("abef,fcjk,edil->abcdijkl", H.aa.vvvv, T.aa, T.aa, optimize=True)  # (jk/il)(ab/cd) = 6 * 6 = 36

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    m4a += (24.0 / 576.0) * np.einsum("cdke,abeijl->abcdijkl", H.aa.vvov, T.aaa, optimize=True)  # (cd/ab)(k/ijl) = 6 * 4 = 24
    m4a -= (24.0 / 576.0) * np.einsum("cmkl,abdijm->abcdijkl", H.aa.vooo, T.aaa, optimize=True)  # (c/abd)(kl/ij) = 6 * 4 = 24

    I3A_vooooo = np.einsum("nmle,bejk->bmnjkl", H.aa.ooov, T.aa, optimize=True)
    I3A_vooooo -= np.transpose(I3A_vooooo, (0, 1, 2, 5, 4, 3)) + np.transpose(I3A_vooooo, (0, 1, 2, 3, 5, 4))
    I3A_vooooo += 0.5 * np.einsum("mnef,befjkl->bmnjkl", H.aa.oovv, T.aaa, optimize=True)
    m4a += 0.5 * (16.0 / 576.0) * np.einsum("bmnjkl,acdimn->abcdijkl", I3A_vooooo, T.aaa, optimize=True)  # (b/acd)(i/jkl) = 4 * 4 = 16

    I3A_vvvovv = -np.einsum("dmfe,bcjm->bcdjef", H.aa.vovv, T.aa, optimize=True)
    I3A_vvvovv -= np.transpose(I3A_vvvovv, (2, 1, 0, 3, 4, 5)) + np.transpose(I3A_vvvovv, (0, 2, 1, 3, 4, 5))
    m4a += 0.5 * (16.0 / 576.0) * np.einsum("bcdjef,aefikl->abcdijkl", I3A_vvvovv, T.aaa, optimize=True)  # (a/bcd)(j/ikl) = 4 * 4 = 16

    I3A_vvooov = (
            -0.5 * np.einsum("nmke,cdnl->cdmkle", H.aa.ooov, T.aa, optimize=True)
            + 0.5 * np.einsum("cmfe,fdkl->cdmkle", H.aa.vovv, T.aa, optimize=True)
            + 0.125 * np.einsum("mnef,cdfkln->cdmkle", H.aa.oovv, T.aaa, optimize=True)  # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
            + 0.25 * np.einsum("mnef,cdfkln->cdmkle", H.ab.oovv, T.aab, optimize=True)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    m4a += (36.0 / 576.0) * np.einsum("cdmkle,abeijm->abcdijkl", I3A_vvooov, T.aaa, optimize=True)  # (cd/ab)(kl/ij) = 6 * 6 = 36

    I3B_vvooov = (
            -0.5 * np.einsum("nmke,cdnl->cdmkle", H.ab.ooov, T.aa, optimize=True)
            + 0.5 * np.einsum("cmfe,fdkl->cdmkle", H.ab.vovv, T.aa, optimize=True)
            + 0.125 * np.einsum("mnef,cdfkln->cdmkle", H.bb.oovv, T.aab, optimize=True)
    # (ij/kl)(c/ab), compensate by factor of 1/2 !!!
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    m4a += (36.0 / 576.0) * np.einsum("cdmkle,abeijm->abcdijkl", I3B_vvooov, T.aab, optimize=True)  # (cd/ab)(kl/ij) = 6 * 6 = 36
    # antisymmetrize
    m4a -= np.transpose(m4a, (0, 1, 2, 3, 4, 6, 5, 7)) # (jk)
    m4a -= np.transpose(m4a, (0, 1, 2, 3, 4, 7, 6, 5)) + np.transpose(m4a, (0, 1, 2, 3, 4, 5, 7, 6)) # (l/jk)
    m4a -= np.transpose(m4a, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(m4a, (0, 1, 2, 3, 6, 5, 4, 7)) + np.transpose(m4a, (0, 1, 2, 3, 7, 5, 6, 4)) # (i/jkl)
    m4a -= np.transpose(m4a, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    m4a -= np.transpose(m4a, (0, 3, 2, 1, 4, 5, 6, 7)) + np.transpose(m4a, (0, 1, 3, 2, 4, 5, 6, 7)) # (d/bc)
    m4a -= np.transpose(m4a, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(m4a, (2, 1, 0, 3, 4, 5, 6, 7)) + np.transpose(m4a, (3, 1, 2, 0, 4, 5, 6, 7)) # (a/bcd)
    return m4a

def build_m4b(H, T):
    # <ijklabcd | H(2) | 0 >
    m4b = -(9.0 / 36.0) * np.einsum("mdel,abim,ecjk->abcdijkl", H.ab.ovvo, T.aa, T.aa, optimize=True)    # (i/jk)(c/ab) = 9
    m4b += (9.0 / 36.0) * np.einsum("mnij,bcnk,adml->abcdijkl", H.aa.oooo, T.aa, T.ab, optimize=True)    # (k/ij)(a/bc) = 9
    m4b -= (18.0 / 36.0) * np.einsum("mdjf,abim,cfkl->abcdijkl", H.ab.ovov, T.aa, T.ab, optimize=True)   # (ijk)(c/ab) = (i/jk)(c/ab)(jk) = 18
    m4b -= np.einsum("amie,bejl,cdkm->abcdijkl", H.ab.voov, T.ab, T.ab, optimize=True)                   # (ijk)(abc) = (i/jk)(a/bc)(jk)(bc) = 36
    m4b += (18.0 / 36.0) * np.einsum("mnjl,bcmk,adin->abcdijkl", H.ab.oooo, T.aa, T.ab, optimize=True)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    m4b -= (18.0 / 36.0) * np.einsum("bmel,ecjk,adim->abcdijkl", H.ab.vovo, T.aa, T.ab, optimize=True)   # (i/jk)(abc) = (i/jk)(a/bc)(bc) = 18
    m4b -= (18.0 / 36.0) * np.einsum("amie,ecjk,bdml->abcdijkl", H.aa.voov, T.aa, T.ab, optimize=True)   # (i/kj)(abc) = (i/kj)(a/bc)(bc) = 18
    m4b += (9.0 / 36.0) * np.einsum("abef,fcjk,edil->abcdijkl", H.aa.vvvv, T.aa, T.ab, optimize=True)    # (i/jk)(c/ab) = (i/jk)(c/ab) = 9
    m4b -= (18.0 / 36.0) * np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.aa, T.ab, optimize=True)   # (ijk)(a/bc) = (i/jk)(a/bc)(jk) = 18
    m4b += (18.0 / 36.0) * np.einsum("adef,ebij,cfkl->abcdijkl", H.ab.vvvv, T.aa, T.ab, optimize=True)   # (k/ij)(abc) = (k/ij)(a/bc)(bc) = 18

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    m4b -= (1.0 / 12.0) * np.einsum("mdkl,abcijm->abcdijkl", H.ab.ovoo, T.aaa, optimize=True)  # (k/ij) = 3
    m4b -= (9.0 / 36.0) * np.einsum("amik,bcdjml->abcdijkl", H.aa.vooo, T.aab, optimize=True)  # (j/ik)(a/bc) = 9
    m4b -= (9.0 / 36.0) * np.einsum("amil,bcdjkm->abcdijkl", H.ab.vooo, T.aab, optimize=True)  # (a/bc)(i/jk) = 9

    m4b += (1.0 / 12.0) * np.einsum("cdel,abeijk->abcdijkl", H.ab.vvvo, T.aaa, optimize=True)  # (c/ab) = 3
    m4b += (9.0 / 36.0) * np.einsum("acie,bedjkl->abcdijkl", H.aa.vvov, T.aab, optimize=True)  # (b/ac)(i/jk) = 9
    m4b += (9.0 / 36.0) * np.einsum("adie,bcejkl->abcdijkl", H.ab.vvov, T.aab, optimize=True)  # (a/bc)(i/jk) = 9

    I3B_oovooo = (
                    np.einsum("mnie,edjl->mndijl", H.aa.ooov, T.ab, optimize=True)
                   +0.25 * np.einsum("mnef,efdijl->mndijl", H.aa.oovv, T.aab, optimize=True)
    )
    I3B_oovooo -= np.transpose(I3B_oovooo, (0, 1, 2, 4, 3, 5))
    m4b += (1.0 / 12.0) * 0.5 * np.einsum("mndijl,abcmnk->abcdijkl", I3B_oovooo, T.aaa, optimize=True)  # (k/ij) = 3

    I3A_vooooo = np.einsum("mnie,delj->dmnlij", H.aa.ooov, T.aa, optimize=True)
    I3A_vooooo -= np.transpose(I3A_vooooo, (0, 1, 2, 4, 3, 5)) + np.transpose(I3A_vooooo, (0, 1, 2, 5, 4, 3))
    I3A_vooooo += 0.5 * np.einsum("mnef,efdijl->dmnlij", H.aa.oovv, T.aaa, optimize=True)
    m4b += (1.0 / 12.0) * 0.5 * np.einsum("cmnkij,abdmnl->abcdijkl", I3A_vooooo, T.aab, optimize=True)  # (c/ab) = 3

    I3B_vooooo = (
                    0.5 * np.einsum("mnel,aeik->amnikl", H.ab.oovo, T.aa, optimize=True)
                  + np.einsum("mnke,aeil->amnikl", H.ab.ooov, T.ab, optimize=True)
                  + 0.5 * np.einsum("mnef,aefikl->amnikl", H.ab.oovv, T.aab, optimize=True)
    )
    I3B_vooooo -= np.transpose(I3B_vooooo, (0, 1, 2, 4, 3, 5))
    m4b += (9.0 / 36.0) * np.einsum("amnikl,bcdjmn->abcdijkl", I3B_vooooo, T.aab, optimize=True)  # (a/bc)(j/ik) = 9

    I3B_vvvvvo = -np.einsum("amef,bdml->abdefl", H.aa.vovv, T.ab, optimize=True)
    I3B_vvvvvo -= np.transpose(I3B_vvvvvo, (1, 0, 2, 3, 4, 5))
    m4b += (1.0 / 12.0) * 0.5 * np.einsum("abdefl,efcijk->abcdijkl", I3B_vvvvvo, T.aaa, optimize=True)  # (c/ab) = 3

    I3A_vvvvvo = -np.einsum("amef,bcmk->abcefk", H.aa.vovv, T.aa, optimize=True)
    I3A_vvvvvo -= np.transpose(I3A_vvvvvo, (1, 0, 2, 3, 4, 5)) + np.transpose(I3A_vvvvvo, (2, 1, 0, 3, 4, 5))
    m4b += (1.0 / 12.0) * 0.5 * np.einsum("abcefk,efdijl->abcdijkl", I3A_vvvvvo, T.aab, optimize=True)  # (k/ij) = 3

    I3B_vvvovv = (
                    -0.5 * np.einsum("mdef,acim->acdief", H.ab.ovvv, T.aa, optimize=True)
                    - np.einsum("cmef,adim->acdief", H.ab.vovv, T.ab, optimize=True)
    )
    I3B_vvvovv -= np.transpose(I3B_vvvovv, (1, 0, 2, 3, 4, 5))
    m4b += (9.0 / 36.0) * np.einsum("acdief,befjkl->abcdijkl", I3B_vvvovv, T.aab, optimize=True)  # (b/ac)(i/jk) = 9

    I3B_vovovo = (
                    -np.einsum("nmie,adnl->amdiel", H.aa.ooov, T.ab, optimize=True)
                    +np.einsum("amfe,fdil->amdiel", H.aa.vovv, T.ab, optimize=True)
                    -np.einsum("mnel,adin->amdiel", H.ab.oovo, T.ab, optimize=True)
                    +np.einsum("mdef,afil->amdiel", H.ab.ovvv, T.ab, optimize=True)
                    +np.einsum("mnef,afdinl->amdiel", H.aa.oovv, T.aab, optimize=True)
                    +np.einsum("mnef,afdinl->amdiel", H.ab.oovv, T.abb, optimize=True)
    )
    m4b += (9.0 / 36.0) * np.einsum("amdiel,bcejkm->abcdijkl", I3B_vovovo, T.aaa, optimize=True)  # (a/bc)(i/jk) = 9

    I3A_vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.aa.ooov, T.aa, optimize=True)
                +0.5 * np.einsum("bmfe,afij->abmije", H.aa.vovv, T.aa, optimize=True)
                +0.25 * np.einsum("mnef,abfijn->abmije", H.ab.oovv, T.aab, optimize=True)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    m4b += (9.0 / 36.0) * np.einsum("abmije,cedkml->abcdijkl", I3A_vvooov, T.aab, optimize=True)  # (c/ab)(k/ij) = 9

    I3B_vvoovo = (
                -0.5 * np.einsum("nmel,acin->acmiel", H.ab.oovo, T.aa, optimize=True)
                + np.einsum("cmef,afil->acmiel", H.ab.vovv, T.ab, optimize=True)
                - 0.5 * np.einsum("nmef,acfinl->acmiel", H.ab.oovv, T.aab, optimize=True)
    )
    I3B_vvoovo -= np.transpose(I3B_vvoovo, (1, 0, 2, 3, 4, 5))
    m4b -= (9.0 / 36.0) * np.einsum("acmiel,ebdkjm->abcdijkl", I3B_vvoovo, T.aab, optimize=True)  # (b/ac)(i/jk) = 9

    I3B_vovoov = (
                0.5 * np.einsum("mdfe,afik->amdike", H.ab.ovvv, T.aa, optimize=True)
                -np.einsum("mnke,adin->amdike", H.ab.ooov, T.ab, optimize=True)
    )
    I3B_vovoov -= np.transpose(I3B_vovoov, (0, 1, 2, 4, 3, 5))
    m4b -= (9.0 / 36.0) * np.einsum("amdike,bcejml->abcdijkl", I3B_vovoov, T.aab, optimize=True)  # (a/bc)(j/ik) = 9

    I3C_vvooov = (
                -np.einsum("nmie,adnl->admile", H.ab.ooov, T.ab, optimize=True)
                -np.einsum("nmle,adin->admile", H.bb.ooov, T.ab, optimize=True)
                +np.einsum("amfe,fdil->admile", H.ab.vovv, T.ab, optimize=True)
                +np.einsum("dmfe,afil->admile", H.bb.vovv, T.ab, optimize=True)
                +np.einsum("mnef,afdinl->admile", H.bb.oovv, T.abb, optimize=True)  # added 5/2/22
    )
    m4b += (9.0 / 36.0) * np.einsum("admile,bcejkm->abcdijkl", I3C_vvooov, T.aab, optimize=True)  # (a/bc)(i/jk) = 9

    I3B_vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.ab.ooov, T.aa, optimize=True)
                +0.5 * np.einsum("bmfe,afij->abmije", H.ab.vovv, T.aa, optimize=True)
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    m4b += (9.0 / 36.0) * np.einsum("abmije,cdeklm->abcdijkl", I3B_vvooov, T.abb, optimize=True)  # (c/ab)(k/ij) = 9
    # antisymmetrize
    m4b -= np.transpose(m4b, (0, 1, 2, 3, 4, 6, 5, 7))  # (jk)
    m4b -= np.transpose(m4b, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(m4b, (0, 1, 2, 3, 6, 5, 4, 7)) # (i/jk)
    m4b -= np.transpose(m4b, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    m4b -= np.transpose(m4b, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(m4b, (2, 1, 0, 3, 4, 5, 6, 7)) # (a/bc)
    return m4b

def build_m4c(H, T):
    # <ijklabcd | H(2) | 0 >
    m4c = -np.einsum("cmke,adim,bejl->abcdijkl", H.bb.voov, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    m4c -= np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    m4c -= 0.5 * np.einsum("mcek,aeij,bdml->abcdijkl", H.ab.ovvo, T.aa, T.ab, optimize=True)    # (kl)(ab)(cd) = 8
    m4c -= 0.5 * np.einsum("amie,bdjm,cekl->abcdijkl", H.ab.voov, T.ab, T.bb, optimize=True)    # (ij)(ab)(cd) = 8
    m4c -= 0.5 * np.einsum("mcek,abim,edjl->abcdijkl", H.ab.ovvo, T.aa, T.ab, optimize=True)    # (ij)(kl)(cd) = 8
    m4c -= 0.5 * np.einsum("amie,cdkm,bejl->abcdijkl", H.ab.voov, T.bb, T.ab, optimize=True)    # (ij)(kl)(ab) = 8
    m4c -= np.einsum("bmel,adim,ecjk->abcdijkl", H.ab.vovo, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    m4c -= np.einsum("mdje,bcmk,aeil->abcdijkl", H.ab.ovov, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    m4c -= 0.25 * np.einsum("mdje,abim,cekl->abcdijkl", H.ab.ovov, T.aa, T.bb, optimize=True)   # (ij)(cd) = 4
    m4c -= 0.25 * np.einsum("bmel,cdkm,aeij->abcdijkl", H.ab.vovo, T.bb, T.aa, optimize=True)   # (kl)(ab) = 4
    m4c += 0.25 * np.einsum("mnij,acmk,bdnl->abcdijkl", H.aa.oooo, T.ab, T.ab, optimize=True)   # (kl)(ab) = 4 !!! (tricky asym)
    m4c += 0.25 * np.einsum("abef,ecik,fdjl->abcdijkl", H.aa.vvvv, T.ab, T.ab, optimize=True)   # (ij)(kl) = 4 !!! (tricky asym)
    m4c += 0.25 * np.einsum("mnik,abmj,cdnl->abcdijkl", H.ab.oooo, T.aa, T.bb, optimize=True)   # (ij)(kl) = 4
    m4c += 0.25 * np.einsum("acef,ebij,fdkl->abcdijkl", H.ab.vvvv, T.aa, T.bb, optimize=True)   # (ab)(cd) = 4
    m4c += np.einsum("mnik,adml,bcjn->abcdijkl", H.ab.oooo, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    m4c += np.einsum("acef,edil,bfjk->abcdijkl", H.ab.vvvv, T.ab, T.ab, optimize=True)          # (ij)(kl)(ab)(cd) = 16
    m4c += 0.25 * np.einsum("mnkl,adin,bcjm->abcdijkl", H.bb.oooo, T.ab, T.ab, optimize=True)   # (ij)(cd) = 4 !!! (tricky asym)
    m4c += 0.25 * np.einsum("cdef,afil,bejk->abcdijkl", H.bb.vvvv, T.ab, T.ab, optimize=True)   # (ij)(kl) = 4 !!! (tricky asym)

    # <ijklabcd | (H(2)*T3)_C + 1/2*(H(2)*T3^2)_C | 0 >
    m4c -= (8.0 / 16.0) * np.einsum("mdil,abcmjk->abcdijkl", H.ab.ovoo, T.aab, optimize=True)  # [1]  (ij)(kl)(cd) = 8
    m4c -= (2.0 / 16.0) * np.einsum("bmji,acdmkl->abcdijkl", H.aa.vooo, T.abb, optimize=True)  # [2]  (ab) = 2
    m4c -= (2.0 / 16.0) * np.einsum("cmkl,abdijm->abcdijkl", H.bb.vooo, T.aab, optimize=True)  # [3]  (cd) = 2
    m4c -= (8.0 / 16.0) * np.einsum("amil,bcdjkm->abcdijkl", H.ab.vooo, T.abb, optimize=True)  # [4]  (ij)(ab)(kl) = 8
    m4c += (8.0 / 16.0) * np.einsum("adel,becjik->abcdijkl", H.ab.vvvo, T.aab, optimize=True)  # [5]  (ab)(kl)(cd) = 8
    m4c += (2.0 / 16.0) * np.einsum("baje,ecdikl->abcdijkl", H.aa.vvov, T.abb, optimize=True)  # [6]  (ij) = 2
    m4c += (8.0 / 16.0) * np.einsum("adie,bcejkl->abcdijkl", H.ab.vvov, T.abb, optimize=True)  # [7]  (ij)(ab)(cd) = 8
    m4c += (2.0 / 16.0) * np.einsum("cdke,abeijl->abcdijkl", H.bb.vvov, T.aab, optimize=True)  # [8]  (kl) = 2

    I3B_oovooo = (
                np.einsum("mnif,fdjl->mndijl", H.aa.ooov, T.ab, optimize=True)
               + 0.25 * np.einsum("mnef,efdijl->mndijl", H.aa.oovv, T.aab, optimize=True)
    )
    I3B_oovooo -= np.transpose(I3B_oovooo, (0, 1, 2, 4, 3, 5))
    m4c += (4.0 / 16.0) * 0.5 * np.einsum("mndijl,abcmnk->abcdijkl", I3B_oovooo, T.aab, optimize=True)  # [9]  (kl)(cd) = 4

    I3B_ovoooo = (
                np.einsum("mnif,bfjl->mbnijl", H.ab.ooov, T.ab, optimize=True)
                + 0.5 * np.einsum("mnfl,bfji->mbnijl", H.ab.oovo, T.aa, optimize=True)
                + 0.5 * np.einsum("mnef,befjil->mbnijl", H.ab.oovv, T.aab, optimize=True)
    )
    I3B_ovoooo -= np.transpose(I3B_ovoooo, (0, 1, 2, 4, 3, 5))
    m4c += (4.0 / 16.0) * np.einsum("mbnijl,acdmkn->abcdijkl", I3B_ovoooo, T.abb, optimize=True)  # [10]  (kl)(ab) = 4

    I3C_vooooo = (
                np.einsum("nmlf,afik->amnikl", H.bb.ooov, T.ab, optimize=True)
                + 0.25 * np.einsum("mnef,aefikl->amnikl", H.bb.oovv, T.abb, optimize=True)
    )
    I3C_vooooo -= np.transpose(I3C_vooooo, (0, 1, 2, 3, 5, 4))
    m4c += (4.0 / 16.0) * 0.5 * np.einsum("amnikl,bcdjmn->abcdijkl", I3C_vooooo, T.abb, optimize=True)  # [11]  (ij)(ab) = 4

    I3C_oovooo = (
                0.5 * np.einsum("mnif,cfkl->mncilk", H.ab.ooov, T.bb, optimize=True)
                + np.einsum("mnfl,fcik->mncilk", H.ab.oovo, T.ab, optimize=True)
                + 0.5 * np.einsum("mnef,efcilk->mncilk", H.ab.oovv, T.abb, optimize=True)
    )
    I3C_oovooo -= np.transpose(I3C_oovooo, (0, 1, 2, 3, 5, 4))
    m4c += (4.0 / 16.0) * np.einsum("mncilk,abdmjn->abcdijkl", I3C_oovooo, T.aab, optimize=True)  # [12]  (ij)(cd) = 4

    I3B_vvvvvo = -np.einsum("bmfe,acmk->abcefk", H.aa.vovv, T.ab, optimize=True)
    I3B_vvvvvo -= np.transpose(I3B_vvvvvo, (1, 0, 2, 3, 4, 5))
    m4c += (4.0 / 16.0) * 0.5 * np.einsum("abcefk,efdijl->abcdijkl", I3B_vvvvvo, T.aab, optimize=True)  # [13]  (kl)(cd) = 4

    I3C_vvvvov = (
                -np.einsum("mdef,acmk->acdekf", H.ab.ovvv, T.ab, optimize=True)
                - 0.5 * np.einsum("amef,cdkm->acdekf", H.ab.vovv, T.bb, optimize=True)
    )
    I3C_vvvvov -= np.transpose(I3C_vvvvov, (0, 2, 1, 3, 4, 5))
    m4c += (4.0 / 16.0) * np.einsum("acdekf,ebfijl->abcdijkl", I3C_vvvvov, T.aab, optimize=True)  # [14]  (kl)(ab) = 4

    I3B_vvvvov = (
                -0.5 * np.einsum("mdef,abmj->abdejf", H.ab.ovvv, T.aa, optimize=True)
                -np.einsum("amef,bdjm->abdejf", H.ab.vovv, T.ab, optimize=True)
    )
    I3B_vvvvov -= np.transpose(I3B_vvvvov, (1, 0, 2, 3, 4, 5))
    m4c += (4.0 / 16.0) * np.einsum("abdejf,efcilk->abcdijkl", I3B_vvvvov, T.abb, optimize=True)  # [15]  (ij)(cd) = 4

    I3C_vvvovv = -np.einsum("cmef,adim->acdief", H.bb.vovv, T.ab, optimize=True)
    I3C_vvvovv -= np.transpose(I3C_vvvovv, (0, 2, 1, 3, 4, 5))
    m4c += (4.0 / 16.0) * 0.5 * np.einsum("acdief,befjkl->abcdijkl", I3C_vvvovv, T.abb, optimize=True)  # [16]  (ij)(ab) = 4

    I3A_vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.aa.ooov, T.aa, optimize=True)
                +0.5 * np.einsum("bmfe,afij->abmije", H.aa.vovv, T.aa, optimize=True)
                +0.25 * np.einsum("mnef,abfijn->abmije", H.aa.oovv, T.aaa, optimize=True)
                +0.25 * np.einsum("mnef,abfijn->abmije", H.ab.oovv, T.aab, optimize=True)
    )
    I3A_vvooov -= np.transpose(I3A_vvooov, (1, 0, 2, 3, 4, 5))
    I3A_vvooov -= np.transpose(I3A_vvooov, (0, 1, 2, 4, 3, 5))
    m4c += (1.0 / 16.0) * np.einsum("abmije,ecdmkl->abcdijkl", I3A_vvooov, T.abb, optimize=True)  # [17]  (1) = 1

    I3B_vvooov = (
                -0.5 * np.einsum("nmje,abin->abmije", H.ab.ooov, T.aa, optimize=True)
                +0.5 * np.einsum("bmfe,afij->abmije", H.ab.vovv, T.aa, optimize=True)
                +0.25 * np.einsum("nmfe,abfijn->abmije", H.ab.oovv, T.aaa, optimize=True)
                +0.25 * np.einsum("nmfe,abfijn->abmije", H.bb.oovv, T.aab, optimize=True)
    )
    I3B_vvooov -= np.transpose(I3B_vvooov, (1, 0, 2, 3, 4, 5))
    I3B_vvooov -= np.transpose(I3B_vvooov, (0, 1, 2, 4, 3, 5))
    m4c += (1.0 / 16.0) * np.einsum("abmije,ecdmkl->abcdijkl", I3B_vvooov, T.bbb, optimize=True)  # [18]  (1) = 1

    I3C_ovvvoo = (
                -0.5 * np.einsum("mnek,cdnl->mcdekl", H.ab.oovo, T.bb, optimize=True)
                +0.5 * np.einsum("mcef,fdkl->mcdekl", H.ab.ovvv, T.bb, optimize=True)
    )
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 2, 1, 3, 4, 5))
    I3C_ovvvoo -= np.transpose(I3C_ovvvoo, (0, 1, 2, 3, 5, 4))
    m4c += (1.0 / 16.0) * np.einsum("mcdekl,abeijm->abcdijkl", I3C_ovvvoo, T.aaa, optimize=True)  # [19]  (1) = 1

    I3D_vvooov = (
                -0.5 * np.einsum("nmke,cdnl->cdmkle", H.bb.ooov, T.bb, optimize=True)
                +0.5 * np.einsum("cmfe,fdkl->cdmkle", H.bb.vovv, T.bb, optimize=True)
    )
    I3D_vvooov -= np.transpose(I3D_vvooov, (1, 0, 2, 3, 4, 5))
    I3D_vvooov -= np.transpose(I3D_vvooov, (0, 1, 2, 4, 3, 5))
    m4c += (1.0 / 16.0) * np.einsum("cdmkle,abeijm->abcdijkl", I3D_vvooov, T.aab, optimize=True)  # [20]  (1) = 1

    I3B_vovovo = (
                -np.einsum("mnel,adin->amdiel", H.ab.oovo, T.ab, optimize=True)
                +np.einsum("mdef,afil->amdiel", H.ab.ovvv, T.ab, optimize=True)
                +0.5 * np.einsum("mnef,afdinl->amdiel", H.aa.oovv, T.aab, optimize=True) # !!! factor 1/2 to compensate asym
                +np.einsum("mnef,afdinl->amdiel", H.ab.oovv, T.abb, optimize=True)
                -np.einsum("nmie,adnl->amdiel", H.aa.ooov, T.ab, optimize=True)
                +np.einsum("amfe,fdil->amdiel", H.aa.vovv, T.ab, optimize=True)
    )
    m4c += np.einsum("amdiel,becjmk->abcdijkl", I3B_vovovo, T.aab, optimize=True)  # [21]  (ij)(kl)(ab)(cd) = 16

    I3C_vovovo = (
                -np.einsum("nmie,adnl->amdiel", H.ab.ooov, T.ab, optimize=True)
                +np.einsum("amfe,fdil->amdiel", H.ab.vovv, T.ab, optimize=True)
                -np.einsum("nmle,adin->amdiel", H.bb.ooov, T.ab, optimize=True)
                +np.einsum("dmfe,afil->amdiel", H.bb.vovv, T.ab, optimize=True)
                +0.5 * np.einsum("mnef,afdinl->amdiel", H.bb.oovv, T.abb, optimize=True) # !!! factor 1/2 to compensate asym
    )
    m4c += np.einsum("amdiel,becjmk->abcdijkl", I3C_vovovo, T.abb, optimize=True)  # [22]  (ij)(kl)(ab)(cd) = 16

    I3B_vovoov = (
                -np.einsum("mnie,bdjn->bmdjie", H.ab.ooov, T.ab, optimize=True)
                +0.5 * np.einsum("mdfe,bfji->bmdjie", H.ab.ovvv, T.aa, optimize=True)
                -0.5 * np.einsum("mnfe,bfdjin->bmdjie", H.ab.oovv, T.aab, optimize=True)
    )
    I3B_vovoov -= np.transpose(I3B_vovoov, (0, 1, 2, 4, 3, 5))
    m4c -= (4.0 / 16.0) * np.einsum("bmdjie,aecmlk->abcdijkl", I3B_vovoov, T.abb, optimize=True)  # [23]  (ab)(cd) = 4

    I3C_ovvoov = (
                -0.5 * np.einsum("mnie,cdkn->mcdike", H.ab.ooov, T.bb, optimize=True)
                +np.einsum("mdfe,fcik->mcdike", H.ab.ovvv, T.ab, optimize=True)
                -0.5 * np.einsum("mnfe,fcdikn->mcdike", H.ab.oovv, T.abb, optimize=True)
    )
    I3C_ovvoov -= np.transpose(I3C_ovvoov, (0, 2, 1, 3, 4, 5))
    m4c -= (4.0 / 16.0) * np.einsum("mcdike,abemjl->abcdijkl", I3C_ovvoov, T.aab, optimize=True)  # [24]  (ij)(kl) = 4

    I3B_vvovoo = (
                -0.5 * np.einsum("nmel,abnj->abmejl", H.ab.oovo, T.aa, optimize=True)
                +np.einsum("amef,bfjl->abmejl", H.ab.vovv, T.ab, optimize=True)
    )
    I3B_vvovoo -= np.transpose(I3B_vvovoo, (1, 0, 2, 3, 4, 5))
    m4c -= (4.0 / 16.0) * np.einsum("abmejl,ecdikm->abcdijkl", I3B_vvovoo, T.abb, optimize=True)  # [25]  (ij)(kl) = 4

    I3C_vovvoo = (
                -np.einsum("nmel,acnk->amcelk", H.ab.oovo, T.ab, optimize=True)
                +0.5 * np.einsum("amef,fclk->amcelk", H.ab.vovv, T.bb, optimize=True)
    )
    I3C_vovvoo -= np.transpose(I3C_vovvoo, (0, 1, 2, 3, 5, 4))
    m4c -= (4.0 / 16.0) * np.einsum("amcelk,bedjim->abcdijkl", I3C_vovvoo, T.aab, optimize=True)  # [26]  (ab)(cd) = 4
    # antisymmetrize
    m4c -= np.transpose(m4c, (1, 0, 2, 3, 4, 5, 6, 7)) # (ab)
    m4c -= np.transpose(m4c, (0, 1, 3, 2, 4, 5, 6, 7)) # (cd)
    m4c -= np.transpose(m4c, (0, 1, 2, 3, 5, 4, 6, 7)) # (ij)
    m4c -= np.transpose(m4c, (0, 1, 2, 3, 4, 5, 7, 6)) # (kl)
    return m4c

def build_l4a(H, L):
    # < 0 | (L2 + L3) H(3) | ijklabcd >
    l4a = (24.0 / 576.0) * np.einsum("dkec,abeijl->abcdijkl", H.aa.vovv, L.aaa, optimize=True) # (cd/ab)(k/ijl) = 6 * 4 = 24
    l4a -= (24.0 / 576.0) * np.einsum("lkmc,abdijm->abcdijkl", H.aa.ooov, L.aaa, optimize=True) # (c/abd)(kl/ij) = 6 * 4 = 24
    # Disconnected terms
    l4a += (36.0 / 576.0) * np.einsum("klcd,abij->abcdijkl", H.aa.oovv, L.aa, optimize=True) # (ij/kl)(ab/cd) = 6 * 6 = 36
    l4a += (16.0 / 576.0) * np.einsum("ia,bcdjkl->abcdijkl", H.a.ov, L.aaa, optimize=True) # (i/jkl)(a/bcd) = 4 * 4 = 16
    # antisymmetrize
    l4a -= np.transpose(l4a, (0, 1, 2, 3, 4, 6, 5, 7)) # (jk)
    l4a -= np.transpose(l4a, (0, 1, 2, 3, 4, 7, 6, 5)) + np.transpose(l4a, (0, 1, 2, 3, 4, 5, 7, 6)) # (l/jk)
    l4a -= np.transpose(l4a, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(l4a, (0, 1, 2, 3, 6, 5, 4, 7)) + np.transpose(l4a, (0, 1, 2, 3, 7, 5, 6, 4)) # (i/jkl)
    l4a -= np.transpose(l4a, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    l4a -= np.transpose(l4a, (0, 3, 2, 1, 4, 5, 6, 7)) + np.transpose(l4a, (0, 1, 3, 2, 4, 5, 6, 7)) # (d/bc)
    l4a -= np.transpose(l4a, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(l4a, (2, 1, 0, 3, 4, 5, 6, 7)) + np.transpose(l4a, (3, 1, 2, 0, 4, 5, 6, 7)) # (a/bcd)
    return l4a

def build_l4b(H, L):
    # < 0 | (L2 + L3) H(3) | ijkl~abcd~ >
    l4b = -(1.0 / 12.0) * np.einsum("klmd,abcijm->abcdijkl", H.ab.ooov, L.aaa, optimize=True)  # (k/ij) = 3
    l4b -= (9.0 / 36.0) * np.einsum("kima,bcdjml->abcdijkl", H.aa.ooov, L.aab, optimize=True)  # (j/ik)(a/bc) = 9
    l4b -= (9.0 / 36.0) * np.einsum("ilam,bcdjkm->abcdijkl", H.ab.oovo, L.aab, optimize=True)  # (a/bc)(i/jk) = 9
    l4b += (1.0 / 12.0) * np.einsum("elcd,abeijk->abcdijkl", H.ab.vovv, L.aaa, optimize=True)  # (c/ab) = 3
    l4b += (9.0 / 36.0) * np.einsum("eica,bedjkl->abcdijkl", H.aa.vovv, L.aab, optimize=True)  # (b/ac)(i/jk) = 9
    l4b += (9.0 / 36.0) * np.einsum("iead,bcejkl->abcdijkl", H.ab.ovvv, L.aab, optimize=True)  # (a/bc)(i/jk) = 9
    # Disconnected terms
    l4b += (9.0 / 36.0) * np.einsum("ia,bcdjkl->abcdijkl", H.a.ov, L.aab, optimize=True) # (i/jk)(a/bc) = 9
    l4b += (1.0 / 36.0) * np.einsum("ld,abcijk->abcdijkl", H.b.ov, L.aaa, optimize=True)
    l4b += (9.0 / 36.0) * np.einsum("ijab,cdkl->abcdijkl", H.aa.oovv, L.ab, optimize=True) # (c/ab)(k/ij) = 9
    l4b += (9.0 / 36.0) * np.einsum("klcd,abij->abcdijkl", H.ab.oovv, L.aa, optimize=True) # (c/ab)(k/ij) = 9
    # antisymmetrize
    l4b -= np.transpose(l4b, (0, 1, 2, 3, 4, 6, 5, 7))  # (jk)
    l4b -= np.transpose(l4b, (0, 1, 2, 3, 5, 4, 6, 7)) + np.transpose(l4b, (0, 1, 2, 3, 6, 5, 4, 7)) # (i/jk)
    l4b -= np.transpose(l4b, (0, 2, 1, 3, 4, 5, 6, 7)) # (bc)
    l4b -= np.transpose(l4b, (1, 0, 2, 3, 4, 5, 6, 7)) + np.transpose(l4b, (2, 1, 0, 3, 4, 5, 6, 7)) # (a/bc)
    return l4b

def build_l4c(H, L):
    # < 0 | (L2 + L3) H(3) | ijk~l~abc~d~ >
    l4c = -(8.0 / 16.0) * np.einsum("ilmd,abcmjk->abcdijkl", H.ab.ooov, L.aab, optimize=True)  # [1]  (ij)(kl)(cd) = 8
    l4c -= (2.0 / 16.0) * np.einsum("ijmb,acdmkl->abcdijkl", H.aa.ooov, L.abb, optimize=True)  # [2]  (ab) = 2
    l4c -= (2.0 / 16.0) * np.einsum("lkmc,abdijm->abcdijkl", H.bb.ooov, L.aab, optimize=True)  # [3]  (cd) = 2
    l4c -= (8.0 / 16.0) * np.einsum("ilam,bcdjkm->abcdijkl", H.ab.oovo, L.abb, optimize=True)  # [4]  (ij)(ab)(kl) = 8
    l4c += (8.0 / 16.0) * np.einsum("elad,becjik->abcdijkl", H.ab.vovv, L.aab, optimize=True)  # [5]  (ab)(kl)(cd) = 8
    l4c += (2.0 / 16.0) * np.einsum("ejab,ecdikl->abcdijkl", H.aa.vovv, L.abb, optimize=True)  # [6]  (ij) = 2
    l4c += (8.0 / 16.0) * np.einsum("iead,bcejkl->abcdijkl", H.ab.ovvv, L.abb, optimize=True)  # [7]  (ij)(ab)(cd) = 8
    l4c += (2.0 / 16.0) * np.einsum("ekdc,abeijl->abcdijkl", H.bb.vovv, L.aab, optimize=True)  # [8]  (kl) = 2
    # Disconnected terms
    l4c += (4.0 / 16.0) * np.einsum("ia,bcdjkl->abcdijkl", H.a.ov, L.abb, optimize=True)
    l4c += (4.0 / 16.0) * np.einsum("kc,abdijl->abcdijkl", H.b.ov, L.aab, optimize=True)
    l4c += (1.0 / 16.0) * np.einsum("ijab,cdkl->abcdijkl", H.aa.oovv, L.bb, optimize=True)
    l4c += np.einsum("ikac,bdjl->abcdijkl", H.ab.oovv, L.ab, optimize=True)
    l4c += (1.0 / 16.0) * np.einsum("klcd,abij->abcdijkl", H.bb.oovv, L.aa, optimize=True) 
    # antisymmetrize
    l4c -= np.transpose(l4c, (1, 0, 2, 3, 4, 5, 6, 7)) # (ab)
    l4c -= np.transpose(l4c, (0, 1, 3, 2, 4, 5, 6, 7)) # (cd)
    l4c -= np.transpose(l4c, (0, 1, 2, 3, 5, 4, 6, 7)) # (ij)
    l4c -= np.transpose(l4c, (0, 1, 2, 3, 4, 5, 7, 6)) # (kl)
    return l4c
