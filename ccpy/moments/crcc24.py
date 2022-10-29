"""Functions to calculate the ground-state CR-CC(2,3) triples correction to CCSD."""
import time
import numpy as np

from ccpy.drivers.cc_energy import get_cc_energy
from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.utilities.updates import crcc24_loops

def permute(x, perm_list):
    str1 = ["a", "b", "c", "d", "i", "j", "k", "l"]
    str2 = "".join([str1[x - 1] for x in perm_list])
    str1 = "".join(s for s in str1)
    contr = str1 + "->" + str2
    return np.einsum(contr, x, optimize=True)

def i_jk(x):
    return (
        x
        - permute(x, [1, 2, 3, 4, 6, 5, 7, 8])
        - permute(x, [1, 2, 3, 4, 7, 6, 5, 8])
    )

def c_ab(x):
    return (
        x
        - permute(x, [3, 2, 1, 4, 5, 6, 7, 8])
        - permute(x, [1, 3, 2, 4, 5, 6, 7, 8])
    )

def a_bc(x):
    return (
        x
        - permute(x, [2, 1, 3, 4, 5, 6, 7, 8])
        - permute(x, [3, 2, 1, 4, 5, 6, 7, 8])
    )

def k_ij(x):
    return (
        x
        - permute(x, [1, 2, 3, 4, 7, 6, 5, 8])
        - permute(x, [1, 2, 3, 4, 5, 7, 6, 8])
    )

def i_kj(x):
    return (
        x
        - permute(x, [1, 2, 3, 4, 7, 6, 5, 8])
        - permute(x, [1, 2, 3, 4, 6, 5, 7, 8])
    )

def jk(x):
    return x - permute(x, [1, 2, 3, 4, 5, 7, 6, 8])

def bc(x):
    return x - permute(x, [1, 3, 2, 4, 5, 6, 7, 8])

def ij(x):
    return x - permute(x, [1, 2, 3, 4, 6, 5, 7, 8])

def kl(x):
    return x - permute(x, [1, 2, 3, 4, 5, 6, 8, 7])

def ab(x):
    return x - permute(x, [2, 1, 3, 4, 5, 6, 7, 8])

def cd(x):
    return x - permute(x, [1, 2, 4, 3, 5, 6, 7, 8])

def calc_crcc24(T, L, H, H0, system, use_RHF=True):
    """
    Calculate the ground-state CR-CC(2,4) correction to the CCSD energy.
    """
    t_start = time.time()
    
    # get the Hbar 3-body diagonal
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)

    #### aaaa correction ####
    # # (jl/i/k)(bc/a/d)
    # D1 = -np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.aa, T.aa, optimize=True)
    # # (jl/ik)(ik) - THE ORDER MATTERS HERE???
    # D1 += (
    #         - permute(D1, [1, 2, 3, 4, 6, 5, 7, 8])
    #         - permute(D1, [1, 2, 3, 4, 5, 7, 6, 8])
    #         - permute(D1, [1, 2, 3, 4, 8, 6, 7, 5])
    #         - permute(D1, [1, 2, 3, 4, 5, 6, 8, 7])
    #         + permute(D1, [1, 2, 3, 4, 6, 5, 8, 7])
    # )
    # D1 += -permute(D1, [1, 2, 3, 4, 7, 6, 5, 8])
    # # A(bc/a/d) = A(bc/ad)A(ad)
    # D1 += (
    #         -permute(D1, [2, 1, 3, 4, 5, 6, 7, 8])
    #         - permute(D1, [3, 2, 1, 4, 5, 6, 7, 8])
    #         - permute(D1, [1, 4, 3, 2, 5, 6, 7, 8])
    #         - permute(D1, [1, 2, 4, 3, 5, 6, 7, 8])
    #         + permute(D1, [2, 1, 4, 3, 5, 6, 7, 8])
    # )
    # D1 += -permute(D1, [4, 2, 3, 1, 5, 6, 7, 8])
    #
    # # (ij/kl)(bc/ad)
    # D2 = np.einsum("mnij,adml,bcnk->abcdijkl", H.aa.oooo, T.aa, T.aa, optimize=True)
    # # (ij/kl)
    # D2 += (
    #     -permute(D2, [1, 2, 3, 4, 7, 6, 5, 8])
    #     - permute(D2, [1, 2, 3, 4, 8, 6, 7, 5])
    #     - permute(D2, [1, 2, 3, 4, 5, 7, 6, 8])
    #     - permute(D2, [1, 2, 3, 4, 5, 8, 7, 6])
    #     + permute(D2, [1, 2, 3, 4, 7, 8, 5, 6])
    # )
    # # (bc/ad)
    # D2 += (
    #     -permute(D2, [2, 1, 3, 4, 5, 6, 7, 8])
    #     - permute(D2, [3, 2, 1, 4, 5, 6, 7, 8])
    #     - permute(D2, [1, 4, 3, 2, 5, 6, 7, 8])
    #     - permute(D2, [1, 2, 4, 3, 5, 6, 7, 8])
    #     + permute(D2, [2, 1, 4, 3, 5, 6, 7, 8])
    # )
    # # (jk/il)(ab/cd)
    # D3 = np.einsum("abef,fcjk,edil->abcdijkl", H.aa.vvvv, T.aa, T.aa, optimize=True)
    # # (jk/il)
    # D3 += (
    #     -permute(D3, [1, 2, 3, 4, 6, 5, 7, 8])
    #     - permute(D3, [1, 2, 3, 4, 7, 6, 5, 8])
    #     - permute(D3, [1, 2, 3, 4, 5, 6, 8, 7])
    #     - permute(D3, [1, 2, 3, 4, 5, 8, 7, 6])
    #     + permute(D3, [1, 2, 3, 4, 6, 5, 8, 7])
    # )
    # # (ab/cd)
    # D3 += (
    #     -permute(D3, [3, 2, 1, 4, 5, 6, 7, 8])
    #     - permute(D3, [4, 2, 3, 1, 5, 6, 7, 8])
    #     - permute(D3, [1, 3, 2, 4, 5, 6, 7, 8])
    #     - permute(D3, [1, 4, 3, 2, 5, 6, 7, 8])
    #     + permute(D3, [3, 4, 1, 2, 5, 6, 7, 8])
    # )
    #
    # L4A = np.einsum("ijab,cdkl->abcdijkl", H0.aa.oovv, L.aa, optimize=True)
    # L4A += (
    #     -permute(L4A, [1, 2, 3, 4, 7, 6, 5, 8])
    #     - permute(L4A, [1, 2, 3, 4, 8, 6, 7, 5])
    #     - permute(L4A, [1, 2, 3, 4, 5, 7, 6, 8])
    #     - permute(L4A, [1, 2, 3, 4, 5, 8, 7, 6])
    #     + permute(L4A, [1, 2, 3, 4, 7, 8, 5, 6])
    # )  # A(ij/kl)
    # L4A += (
    #     -permute(L4A, [3, 2, 1, 4, 5, 6, 7, 8])
    #     - permute(L4A, [4, 2, 3, 1, 5, 6, 7, 8])
    #     - permute(L4A, [1, 3, 2, 4, 5, 6, 7, 8])
    #     - permute(L4A, [1, 4, 3, 2, 5, 6, 7, 8])
    #     + permute(L4A, [3, 4, 1, 2, 5, 6, 7, 8])
    # )  # A(ab/cd)
    dA_aaaa, dB_aaaa, dC_aaaa, dD_aaaa = crcc24_loops.crcc24_loops.crcc24a(
        T.aa,
        L.aa,
        H0.a.oo,
        H0.a.vv,
        H.a.oo,
        H.a.vv,
        H.aa.voov,
        H.aa.oooo,
        H.aa.vvvv,
        H.aa.oovv,
        d3aaa_o,
        d3aaa_v,
    )

    #### aaab correction ####
    # # (i/jk)(c/ab)
    # D1 = -np.einsum("mdel,abim,ecjk->abcdijkl", H.ab.ovvo, T.aa, T.aa, optimize=True)
    # D1 = i_jk(c_ab(D1))
    # # (k/ij)(a/bc)
    # D2 = +np.einsum("mnij,bcnk,adml->abcdijkl", H.aa.oooo, T.aa, T.ab, optimize=True)
    # D2 = k_ij(a_bc(D2))
    # # (ijk)(c/ab) = (i/jk)(c/ab)(jk)
    # D3 = -np.einsum("mdjf,abim,cfkl->abcdijkl", H.ab.ovov, T.aa, T.ab, optimize=True)
    # D3 = i_jk(c_ab(jk(D3)))
    # # (ijk)(abc) = (i/jk)(a/bc)(jk)(bc)
    # D4 = -np.einsum("amie,bejl,cdkm->abcdijkl", H.ab.voov, T.ab, T.ab, optimize=True)
    # D4 = i_jk(a_bc(jk(bc(D4))))
    # # (ijk)(a/bc) = (i/jk)(a/bc)(jk)
    # D5 = +np.einsum("mnjl,bcmk,adin->abcdijkl", H.ab.oooo, T.aa, T.ab, optimize=True)
    # D5 = i_jk(a_bc(jk(D5)))
    # # (i/jk)(abc) = (i/jk)(a/bc)(bc)
    # D6 = -np.einsum("bmel,ecjk,adim->abcdijkl", H.ab.vovo, T.aa, T.ab, optimize=True)
    # D6 = i_jk(a_bc(bc(D6)))
    # # (i/kj)(abc) = (i/kj)(a/bc)(bc)
    # D7 = -np.einsum("amie,ecjk,bdml->abcdijkl", H.aa.voov, T.aa, T.ab, optimize=True)
    # D7 = i_kj(a_bc(bc(D7)))
    # # (i/jk)(c/ab) = (i/jk)(c/ab)
    # D8 = +np.einsum("abef,fcjk,edil->abcdijkl", H.aa.vvvv, T.aa, T.ab, optimize=True)
    # D8 = i_jk(c_ab(D8))
    # # (ijk)(a/bc) = (i/jk)(a/bc)(jk)
    # D9 = -np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.aa, T.ab, optimize=True)
    # D9 = i_jk(a_bc(jk(D9)))
    # # (k/ij)(abc) = (k/ij)(a/bc)(bc)
    # D10 = +np.einsum("adef,ebij,cfkl->abcdijkl", H.ab.vvvv, T.aa, T.ab, optimize=True)
    # D10 = k_ij(a_bc(bc(D10)))
    # 
    # L4B = (
    #       np.einsum("ijab,cdkl->abcdijkl", H0.aa.oovv, L.ab, optimize=True)
    #     + np.einsum("klcd,abij->abcdijkl", H0.ab.oovv, L.aa, optimize=True)
    # )
    # L4B += -permute(L4B, [1, 2, 3, 4, 7, 6, 5, 8]) - permute(
    #     L4B, [1, 2, 3, 4, 5, 7, 6, 8]
    # )  # A(k/ij)
    # L4B += -permute(L4B, [3, 2, 1, 4, 5, 6, 7, 8]) - permute(
    #     L4B, [1, 3, 2, 4, 5, 6, 7, 8]
    # )  # A(c/ab)
    dA_aaab, dB_aaab, dC_aaab, dD_aaab = crcc24_loops.crcc24_loops.crcc24b(
        T.aa,
        T.ab,
        L.aa,
        L.ab,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
        H.aa.voov,
        H.aa.oooo,
        H.aa.vvvv,
        H.aa.oovv,
        H.ab.voov,
        H.ab.ovov,
        H.ab.vovo,
        H.ab.ovvo,
        H.ab.oooo,
        H.ab.vvvv,
        H.ab.oovv,
        H.bb.voov,
        d3aaa_o,
        d3aaa_v,
        d3aab_o,
        d3aab_v,
        d3abb_o,
        d3abb_v,
    )
    #### aabb correction ####
    # # 1 - (ij)(kl)(ab)(cd)
    # D1 = -np.einsum("cmke,adim,bejl->abcdijkl", H.bb.voov, T.ab, T.ab, optimize=True)
    # D1 = ij(kl(ab(cd(D1))))
    # # 2 - (ij)(kl)(ab)(cd)
    # D2 = -np.einsum("amie,bcmk,edjl->abcdijkl", H.aa.voov, T.ab, T.ab, optimize=True)
    # D2 = ij(kl(ab(cd(D2))))
    # # 3 - (kl)(ab)(cd)
    # D3 = -np.einsum("mcek,aeij,bdml->abcdijkl", H.ab.ovvo, T.aa, T.ab, optimize=True)
    # D3 = kl(ab(cd(D3)))
    # # 4 - (ij)(ab)(cd)
    # D4 = -np.einsum("amie,bdjm,cekl->abcdijkl", H.ab.voov, T.ab, T.bb, optimize=True)
    # D4 = ij(ab(cd(D4)))
    # # 5 - (ij)(kl)(cd)
    # D5 = -np.einsum("mcek,abim,edjl->abcdijkl", H.ab.ovvo, T.aa, T.ab, optimize=True)
    # D5 = ij(kl(cd(D5)))
    # # 6 - (ij)(kl)(ab)
    # D6 = -np.einsum("amie,cdkm,bejl->abcdijkl", H.ab.voov, T.bb, T.ab, optimize=True)
    # D6 = ij(kl(ab(D6)))
    # # 7 - (ij)(kl)(ab)(cd)
    # D7 = -np.einsum("bmel,adim,ecjk->abcdijkl", H.ab.vovo, T.ab, T.ab, optimize=True)
    # D7 = ij(kl(ab(cd(D7))))
    # # 8 - (ij)(kl)(ab)(cd)
    # D8 = -np.einsum("mdje,bcmk,aeil->abcdijkl", H.ab.ovov, T.ab, T.ab, optimize=True)
    # D8 = ij(kl(ab(cd(D8))))
    # # 9 - (ij)(cd)
    # D9 = -np.einsum("mdje,abim,cekl->abcdijkl", H.ab.ovov, T.aa, T.bb, optimize=True)
    # D9 = ij(cd(D9))
    # # 10 - (kl)(ab)
    # D10 = -np.einsum("bmel,cdkm,aeij->abcdijkl", H.ab.vovo, T.bb, T.aa, optimize=True)
    # D10 = kl(ab(D10))
    # # 11 - (kl)(ab) !!!
    # D11 = np.einsum("mnij,acmk,bdnl->abcdijkl", H.aa.oooo, T.ab, T.ab, optimize=True)
    # D11 = kl(ab(D11))
    # # 12 - (ij)(kl) !!!
    # D12 = np.einsum("abef,ecik,fdjl->abcdijkl", H.aa.vvvv, T.ab, T.ab, optimize=True)
    # D12 = ij(kl(D12))
    # # 13 - (ij)(kl)
    # D13 = np.einsum("mnik,abmj,cdnl->abcdijkl", H.ab.oooo, T.aa, T.bb, optimize=True)
    # D13 = ij(kl(D13))
    # # 14 - (ab)(cd)
    # D14 = np.einsum("acef,ebij,fdkl->abcdijkl", H.ab.vvvv, T.aa, T.bb, optimize=True)
    # D14 = ab(cd(D14))
    # # 15 - (ij)(kl)(ab)(cd)
    # D15 = np.einsum("mnik,adml,bcjn->abcdijkl", H.ab.oooo, T.ab, T.ab, optimize=True)
    # D15 = ij(kl(ab(cd(D15))))
    # # 16 - (ij)(kl)(ab)(cd)
    # D16 = np.einsum("acef,edil,bfjk->abcdijkl", H.ab.vvvv, T.ab, T.ab, optimize=True)
    # D16 = ij(kl(ab(cd(D16))))
    # # 17 - (ij)(cd) !!!
    # D17 = np.einsum("mnkl,adin,bcjm->abcdijkl", H.bb.oooo, T.ab, T.ab, optimize=True)
    # D17 = ij(cd(D17))
    # # 18 - (ij)(ab) !!!
    # D18 = np.einsum("cdef,afil,bejk->abcdijkl", H.bb.vvvv, T.ab, T.ab, optimize=True)
    # D18 = ij(ab(D18))
    #
    # L4C = np.einsum("ijab,cdkl->abcdijkl", H0.aa.oovv, L.bb, optimize=True)
    # L4C += np.einsum("abij,klcd->abcdijkl", L.aa, H0.bb.oovv, optimize=True)
    # D1 = np.einsum("bcjk,ilad->abcdijkl", L.ab, H0.ab.oovv, optimize=True)
    # D1 += -permute(D1, [1, 2, 3, 4, 6, 5, 7, 8])  # A(ij)
    # D1 += -permute(D1, [1, 2, 3, 4, 5, 6, 8, 7])  # A(kl)
    # D1 += -permute(D1, [2, 1, 3, 4, 5, 6, 7, 8])  # A(ab)
    # D1 += -permute(D1, [1, 2, 4, 3, 5, 6, 7, 8])  # A(cd)
    # L4C += D1

    dA_aabb, dB_aabb, dC_aabb, dD_aabb = crcc24_loops.crcc24_loops.crcc24c(
        T.aa,
        T.ab,
        T.bb,
        L.aa,
        L.ab,
        L.bb,
        H0.a.oo,
        H0.a.vv,
        H0.b.oo,
        H0.b.vv,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
        H.aa.voov,
        H.aa.oooo,
        H.aa.vvvv,
        H.aa.oovv,
        H.ab.voov,
        H.ab.ovov,
        H.ab.vovo,
        H.ab.ovvo,
        H.ab.oooo,
        H.ab.vvvv,
        H.ab.oovv,
        H.bb.voov,
        H.bb.oooo,
        H.bb.vvvv,
        H.bb.oovv,
        d3aaa_o,
        d3aaa_v,
        d3aab_o,
        d3aab_v,
        d3abb_o,
        d3abb_v,
        d3bbb_o,
        d3bbb_v,
    )

    
    if use_RHF:
        correction_A = 2.0 * dA_aaaa + 2.0 * dA_aaab + dA_aabb
        correction_B = 2.0 * dB_aaaa + 2.0 * dB_aaab + dB_aabb
        correction_C = 2.0 * dC_aaaa + 2.0 * dC_aaab + dC_aabb
        correction_D = 2.0 * dD_aaaa + 2.0 * dD_aaab + dD_aabb

    t_end = time.time()
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

    print('   CR-CC(2,4) Calculation Summary')
    print('   -------------------------------------')
    print("   Completed in  ({:0.2f}m  {:0.2f}s)\n".format(minutes, seconds))
    print("   CCSD = {:>10.10f}".format(system.reference_energy + cc_energy))
    print(
        "   CR-CC(2,4)_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}".format(
            total_energy_A, energy_A, correction_A
        )
    )
    print(
        "   CR-CC(2,4)_B = {:>10.10f}     ΔE_B = {:>10.10f}     δ_B = {:>10.10f}".format(
            total_energy_B, energy_B, correction_B
        )
    )
    print(
        "   CR-CC(2,4)_C = {:>10.10f}     ΔE_C = {:>10.10f}     δ_C = {:>10.10f}".format(
            total_energy_C, energy_C, correction_C
        )
    )
    print(
        "   CR-CC(2,4)_D = {:>10.10f}     ΔE_D = {:>10.10f}     δ_D = {:>10.10f}\n".format(
            total_energy_D, energy_D, correction_D
        )
    )

    Ecrcc24 = {"A": total_energy_A, "B": total_energy_B, "C": total_energy_C, "D": total_energy_D}
    delta24 = {"A": correction_A, "B": correction_B, "C": correction_C, "D": correction_D}
    
    return Ecrcc24, delta24

