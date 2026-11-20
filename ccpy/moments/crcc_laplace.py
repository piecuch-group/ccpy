"""Laplace-transformed CR-CC(2,3)_A and CR-CC(2,4)_A corrections to CCSD, and their
two-body CC(P;Q) analogs restricted to the triples/quadruples outside a P space.

The A variant uses Moller-Plesset denominators D = sum(e_occ) - sum(e_vir) = -Delta, and
1/Delta ~ sum_g w_g exp(-t_g Delta) factorizes into one exponential per index. For each
spin case the correction

    delta = sum_{restricted} m l / D = sum_{all indices} x * l / D,   m = A_v A_o x,

where A_v, A_o antisymmetrize the same-spin virtual and occupied labels, x is a sum of
unantisymmetrized moment diagrams, and l is the fully antisymmetric left vector. Every
x-diagram times every l-diagram is a closed network of 4-index tensors in which each index
occurs twice, so it contracts pairwise with at most six distinct indices per step:
O(N^6) per quadrature point and O(N^4) memory.

The diagrams are those of crcc23.py / crcc24.py. Each x-diagram carries its coefficient
in those codes times (number of occupied permutations listed) / |G_o|, where G_o is the
group of same-spin occupied permutations; the virtual antisymmetrizer applied in the
Fortran kernels is the A_v above. Each l-diagram is written once and expanded over
A_v A_o here.

A P space (t3_excitations / t4_excitations) is removed by evaluating l_K m_K for each
K in P and subtracting it with the same quadrature, so the quadrature error stays on Q.
"""
import functools
import itertools
import time

import numpy as np
from scipy.optimize import least_squares


def calc_crcc23_laplace(T, L, corr_energy, H, H0, system, use_RHF=False, npts=10, t3_excitations=None):
    """
    Calculate the Laplace-transformed CR-CC(2,3)_A correction to the CCSD energy, or with
    t3_excitations, the two-body CC(P;3)_A correction over the triples outside P.
    """
    t_start = time.perf_counter()

    t, w, qerr = _quadrature(H0, 3, npts)

    dA_aaa = laplace_spincase(*triples_aaa(T, L, H), H0, t, w, _pspace(t3_excitations, "aaa"))
    dA_aab = laplace_spincase(*triples_aab(T, L, H), H0, t, w, _pspace(t3_excitations, "aab"))
    if use_RHF:
        correction_A = 2.0 * dA_aaa + 2.0 * dA_aab
    else:
        dA_abb = laplace_spincase(*triples_abb(T, L, H), H0, t, w, _pspace(t3_excitations, "abb"))
        dA_bbb = laplace_spincase(*triples_bbb(T, L, H), H0, t, w, _pspace(t3_excitations, "bbb"))
        correction_A = dA_aaa + dA_aab + dA_abb + dA_bbb

    return _summary("CR-CC(2,3)" if t3_excitations is None else "CC(P;3)",
                    corr_energy, correction_A, system, npts, qerr, t_start)


def calc_crcc24_laplace(T, L, corr_energy, H, H0, system, use_RHF=False, npts=10, t4_excitations=None):
    """
    Calculate the Laplace-transformed CR-CC(2,4)_A correction to the CCSD energy, or with
    t4_excitations, the two-body CC(P;4)_A correction over the quadruples outside P.
    """
    t_start = time.perf_counter()

    t, w, qerr = _quadrature(H0, 4, npts)

    dA_aaaa = laplace_spincase(*quadruples_aaaa(T, L, H), H0, t, w, _pspace(t4_excitations, "aaaa"))
    dA_aaab = laplace_spincase(*quadruples_aaab(T, L, H), H0, t, w, _pspace(t4_excitations, "aaab"))
    dA_aabb = laplace_spincase(*quadruples_aabb(T, L, H), H0, t, w, _pspace(t4_excitations, "aabb"))
    if use_RHF:
        correction_A = 2.0 * dA_aaaa + 2.0 * dA_aaab + dA_aabb
    else:
        dA_abbb = laplace_spincase(*quadruples_abbb(T, L, H), H0, t, w, _pspace(t4_excitations, "abbb"))
        dA_bbbb = laplace_spincase(*quadruples_bbbb(T, L, H), H0, t, w, _pspace(t4_excitations, "bbbb"))
        correction_A = dA_aaaa + dA_aaab + dA_aabb + dA_abbb + dA_bbbb

    return _summary("CR-CC(2,4)" if t4_excitations is None else "CC(P;4)",
                    corr_energy, correction_A, system, npts, qerr, t_start)

# ---------------------------------------------------------------------------
# Triples spin cases: (spincase, virtual labels, occupied labels, tensors, x, l)
# Lowercase abcd/ijkl are external; uppercase E F M N are summed in x, e f m n in l.
# ---------------------------------------------------------------------------

def triples_aaa(T, L, H):
    I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
    tensors = {"t2a": T.aa, "l1a": L.a, "l2a": L.aa, "h1a_ov": H.a.ov,
               "h2a_vooo": H.aa.vooo, "i2a_vvov": I2A_vvov, "h2a_vovv": H.aa.vovv,
               "h2a_ooov": H.aa.ooov, "h2a_oovv": H.aa.oovv}
    x = [
        # Diagram 1: -A(k/ij)A(a/bc) h2a(amij) * t2a(bcmk)
        (-0.25, [("h2a_vooo", "aMij"), ("t2a", "bcMk")]),
        # Diagram 2: A(i/jk)A(c/ab) i2a(abie) * t2a(ecjk)
        (0.25, [("i2a_vvov", "abiE"), ("t2a", "Ecjk")]),
    ]
    l = [
        # Diagram 1: A(i/jk)A(c/ab) h2a(eiba) * l2a(ecjk)
        (0.25, [("h2a_vovv", "eiba"), ("l2a", "ecjk")]),
        # Diagram 2: -A(k/ij)A(a/bc) h2a(jima) * l2a(bcmk)
        (-0.25, [("h2a_ooov", "jima"), ("l2a", "bcmk")]),
        # Diagram 3: A(k/ij)A(c/ab) l1a(ck) * h2a(ijab)
        (0.25, [("l1a", "ck"), ("h2a_oovv", "ijab")]),
        # Diagram 4: A(k/ij)A(c/ab) h1a(kc) * l2a(abij)
        (0.25, [("h1a_ov", "kc"), ("l2a", "abij")]),
    ]
    return "aaa", "abc", "ijk", tensors, x, l


def triples_aab(T, L, H):
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
    I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
    I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    tensors = {"t2a": T.aa, "t2b": T.ab, "l1a": L.a, "l1b": L.b, "l2a": L.aa, "l2b": L.ab,
               "h1a_ov": H.a.ov, "h1b_ov": H.b.ov, "h2a_oovv": H.aa.oovv, "h2b_oovv": H.ab.oovv,
               "i2b_ovoo": I2B_ovoo, "i2b_vooo": I2B_vooo, "i2a_vooo": I2A_vooo,
               "h2b_vvvo": H.ab.vvvo, "h2b_vvov": H.ab.vvov, "h2a_vvov": H.aa.vvov,
               "h2b_vovv": H.ab.vovv, "h2b_ovvv": H.ab.ovvv, "h2a_vovv": H.aa.vovv,
               "h2b_ooov": H.ab.ooov, "h2b_oovo": H.ab.oovo, "h2a_ooov": H.aa.ooov}
    x = [
        # Diagram 1: A(ab) h2b(bcek) * t2a(aeij)
        (0.5, [("t2a", "aEij"), ("h2b_vvvo", "bcEk")]),
        # Diagram 2: -A(ij) i2b(mcjk) * t2a(abim)
        (0.5, [("t2a", "abMi"), ("i2b_ovoo", "Mcjk")]),
        # Diagram 3: A(ab)A(ij) h2b(bcje) * t2b(aeik)
        (1.0, [("t2b", "aEik"), ("h2b_vvov", "bcjE")]),
        # Diagram 4: -A(ab)A(ij) i2b(amik) * t2b(bcjm)
        (-1.0, [("i2b_vooo", "aMik"), ("t2b", "bcjM")]),
        # Diagram 5: A(ij) h2a(abie) * t2b(ecjk)
        (0.5, [("h2a_vvov", "abiE"), ("t2b", "Ecjk")]),
        # Diagram 6: -A(ab) i2a(amij) * t2b(bcmk)
        (-0.5, [("i2a_vooo", "aMij"), ("t2b", "bcMk")]),
    ]
    l = [
        # Diagram 1: A(ab) h2b(ekbc) * l2a(aeij)
        (0.5, [("l2a", "aeij"), ("h2b_vovv", "ekbc")]),
        # Diagram 2: A(ij) h2a(eiba) * l2b(ecjk)
        (0.5, [("h2a_vovv", "eiba"), ("l2b", "ecjk")]),
        # Diagram 3: A(ij)A(ab) h2b(jebc) * l2b(aeik)
        (1.0, [("l2b", "aeik"), ("h2b_ovvv", "jebc")]),
        # Diagram 4: A(ij) h2b(jkmc) * l2a(abmi)
        (0.5, [("l2a", "abmi"), ("h2b_ooov", "jkmc")]),
        # Diagram 5: -A(ab) h2a(jima) * l2b(bcmk)
        (-0.5, [("h2a_ooov", "jima"), ("l2b", "bcmk")]),
        # Diagram 6: -A(ij)A(ab) h2b(ikam) * l2b(bcjm)
        (-1.0, [("h2b_oovo", "ikam"), ("l2b", "bcjm")]),
        # Diagram 7: A(ij)A(ab) l1a(ai) * h2b(jkbc)
        (1.0, [("l1a", "ai"), ("h2b_oovv", "jkbc")]),
        # Diagram 8: l1b(ck) * h2a(ijab)
        (0.25, [("l1b", "ck"), ("h2a_oovv", "ijab")]),
        # Diagram 9: A(ij)A(ab) h1a(ia) * l2b(bcjk)
        (1.0, [("h1a_ov", "ia"), ("l2b", "bcjk")]),
        # Diagram 10: h1b(kc) * l2a(abij)
        (0.25, [("h1b_ov", "kc"), ("l2a", "abij")]),
    ]
    return "aab", "abc", "ijk", tensors, x, l


def triples_abb(T, L, H):
    I2B_vooo = H.ab.vooo - np.einsum("me,aeij->amij", H.b.ov, T.ab, optimize=True)
    I2C_vooo = H.bb.vooo - np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)
    I2B_ovoo = H.ab.ovoo - np.einsum("me,ebij->mbij", H.a.ov, T.ab, optimize=True)
    tensors = {"t2b": T.ab, "t2c": T.bb, "l1a": L.a, "l1b": L.b, "l2b": L.ab, "l2c": L.bb,
               "h1a_ov": H.a.ov, "h1b_ov": H.b.ov, "h2b_oovv": H.ab.oovv, "h2c_oovv": H.bb.oovv,
               "i2b_vooo": I2B_vooo, "i2c_vooo": I2C_vooo, "i2b_ovoo": I2B_ovoo,
               "h2b_vvov": H.ab.vvov, "h2c_vvov": H.bb.vvov, "h2b_vvvo": H.ab.vvvo,
               "h2b_ovvv": H.ab.ovvv, "h2b_vovv": H.ab.vovv, "h2c_vovv": H.bb.vovv,
               "h2b_oovo": H.ab.oovo, "h2b_ooov": H.ab.ooov, "h2c_ooov": H.bb.ooov}
    x = [
        # Diagram 1: A(bc) h2b(abie) * t2c(ecjk)
        (0.5, [("h2b_vvov", "abiE"), ("t2c", "Ecjk")]),
        # Diagram 2: -A(jk) i2b(amij) * t2c(bcmk)
        (-0.5, [("i2b_vooo", "aMij"), ("t2c", "bcMk")]),
        # Diagram 3: A(jk) h2c(cbke) * t2b(aeij)
        (0.5, [("t2b", "aEij"), ("h2c_vvov", "cbkE")]),
        # Diagram 4: -A(bc) i2c(cmkj) * t2b(abim)
        (-0.5, [("t2b", "abiM"), ("i2c_vooo", "cMkj")]),
        # Diagram 5: A(jk)A(bc) h2b(abej) * t2b(ecik)
        (1.0, [("h2b_vvvo", "abEj"), ("t2b", "Ecik")]),
        # Diagram 6: -A(jk)A(bc) i2b(mcik) * t2b(abmj)
        (-1.0, [("t2b", "abMj"), ("i2b_ovoo", "Mcik")]),
    ]
    l = [
        # Diagram 1: A(bc) h2b(ieab) * l2c(ecjk)
        (0.5, [("h2b_ovvv", "ieab"), ("l2c", "ecjk")]),
        # Diagram 2: A(jk) h2c(ekbc) * l2b(aeij)
        (0.5, [("l2b", "aeij"), ("h2c_vovv", "ekbc")]),
        # Diagram 3: A(jk)A(bc) h2b(ejab) * l2b(ecik)
        (1.0, [("h2b_vovv", "ejab"), ("l2b", "ecik")]),
        # Diagram 4: -A(jk) h2b(ijam) * l2c(bcmk)
        (-0.5, [("h2b_oovo", "ijam"), ("l2c", "bcmk")]),
        # Diagram 5: -A(bc) h2c(jkmc) * l2b(abim)
        (-0.5, [("l2b", "abim"), ("h2c_ooov", "jkmc")]),
        # Diagram 6: -A(jk)A(bc) h2b(ikmc) * l2b(abmj)
        (-1.0, [("l2b", "abmj"), ("h2b_ooov", "ikmc")]),
        # Diagram 7: A(jk)A(bc) l1b(ck) * h2b(ijab)
        (1.0, [("l1b", "ck"), ("h2b_oovv", "ijab")]),
        # Diagram 8: l1a(ai) * h2c(jkbc)
        (0.25, [("l1a", "ai"), ("h2c_oovv", "jkbc")]),
        # Diagram 9: A(jk)A(bc) h1b(kc) * l2b(abij)
        (1.0, [("h1b_ov", "kc"), ("l2b", "abij")]),
        # Diagram 10: h1a(ia) * l2c(bcjk)
        (0.25, [("h1a_ov", "ia"), ("l2c", "bcjk")]),
    ]
    return "abb", "abc", "ijk", tensors, x, l


def triples_bbb(T, L, H):
    I2C_vvov = H.bb.vvov + np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
    tensors = {"t2c": T.bb, "l1b": L.b, "l2c": L.bb, "h1b_ov": H.b.ov,
               "h2c_vooo": H.bb.vooo, "i2c_vvov": I2C_vvov, "h2c_vovv": H.bb.vovv,
               "h2c_ooov": H.bb.ooov, "h2c_oovv": H.bb.oovv}
    x = [
        # Diagram 1: -A(k/ij)A(a/bc) h2c(amij) * t2c(bcmk)
        (-0.25, [("h2c_vooo", "aMij"), ("t2c", "bcMk")]),
        # Diagram 2: A(i/jk)A(c/ab) i2c(abie) * t2c(ecjk)
        (0.25, [("i2c_vvov", "abiE"), ("t2c", "Ecjk")]),
    ]
    l = [
        # Diagram 1: A(i/jk)A(c/ab) h2c(eiba) * l2c(ecjk)
        (0.25, [("h2c_vovv", "eiba"), ("l2c", "ecjk")]),
        # Diagram 2: -A(k/ij)A(a/bc) h2c(jima) * l2c(bcmk)
        (-0.25, [("h2c_ooov", "jima"), ("l2c", "bcmk")]),
        # Diagram 3: A(k/ij)A(c/ab) l1b(ck) * h2c(ijab)
        (0.25, [("l1b", "ck"), ("h2c_oovv", "ijab")]),
        # Diagram 4: A(k/ij)A(c/ab) h1b(kc) * l2c(abij)
        (0.25, [("h1b_ov", "kc"), ("l2c", "abij")]),
    ]
    return "bbb", "abc", "ijk", tensors, x, l

# ---------------------------------------------------------------------------
# Quadruples spin cases
# ---------------------------------------------------------------------------

def quadruples_aaaa(T, L, H):
    tensors = {"t2a": T.aa, "l2a": L.aa, "h2a_voov": H.aa.voov, "h2a_oooo": H.aa.oooo,
               "h2a_vvvv": H.aa.vvvv, "h2a_oovv": H.aa.oovv}
    x = [
        # Diagram 1: -A(jl/i/k)A(bc/a/d) h2a(amie) * t2a(bcmk) * t2a(edjl)
        (-0.25, [("h2a_voov", "aMiE"), ("t2a", "bcMk"), ("t2a", "Edjl")]),
        # Diagram 2: A(ij/kl)A(bc/ad) h2a(mnij) * t2a(adml) * t2a(bcnk)
        (1.0 / 16.0, [("h2a_oooo", "MNij"), ("t2a", "adMl"), ("t2a", "bcNk")]),
        # Diagram 3: A(jk/il)A(ab/cd) h2a(abef) * t2a(fcjk) * t2a(edil)
        (1.0 / 16.0, [("h2a_vvvv", "abEF"), ("t2a", "Fcjk"), ("t2a", "Edil")]),
    ]
    l = [
        # Diagram 1: A(ij/kl)A(ab/cd) h2a(klcd) * l2a(abij)
        (1.0 / 16.0, [("l2a", "abij"), ("h2a_oovv", "klcd")]),
    ]
    return "aaaa", "abcd", "ijkl", tensors, x, l


def quadruples_aaab(T, L, H):
    tensors = {"t2a": T.aa, "t2b": T.ab, "l2a": L.aa, "l2b": L.ab,
               "h2a_voov": H.aa.voov, "h2a_oooo": H.aa.oooo, "h2a_vvvv": H.aa.vvvv, "h2a_oovv": H.aa.oovv,
               "h2b_voov": H.ab.voov, "h2b_ovov": H.ab.ovov, "h2b_vovo": H.ab.vovo, "h2b_ovvo": H.ab.ovvo,
               "h2b_oooo": H.ab.oooo, "h2b_vvvv": H.ab.vvvv, "h2b_oovv": H.ab.oovv}
    x = [
        # Diagram 1: -A(i/jk)A(c/ab) h2b(mdel) * t2a(abim) * t2a(ecjk)
        (-0.25, [("h2b_ovvo", "MdEl"), ("t2a", "abiM"), ("t2a", "Ecjk")]),
        # Diagram 2: A(k/ij)A(a/bc) h2a(mnij) * t2a(bcnk) * t2b(adml)
        (0.25, [("h2a_oooo", "MNij"), ("t2a", "bcNk"), ("t2b", "adMl")]),
        # Diagram 3: -A(ijk)A(c/ab) h2b(mdjf) * t2a(abim) * t2b(cfkl)
        (-0.5, [("h2b_ovov", "MdjF"), ("t2a", "abiM"), ("t2b", "cFkl")]),
        # Diagram 4: -A(ijk)A(abc) h2b(amie) * t2b(bejl) * t2b(cdkm)
        (-1.0, [("h2b_voov", "aMiE"), ("t2b", "bEjl"), ("t2b", "cdkM")]),
        # Diagram 5: A(ijk)A(a/bc) h2b(mnjl) * t2a(bcmk) * t2b(adin)
        (0.5, [("h2b_oooo", "MNjl"), ("t2a", "bcMk"), ("t2b", "adiN")]),
        # Diagram 6: -A(i/jk)A(abc) h2b(bmel) * t2a(ecjk) * t2b(adim)
        (-0.5, [("h2b_vovo", "bMEl"), ("t2a", "Ecjk"), ("t2b", "adiM")]),
        # Diagram 7: -A(i/jk)A(abc) h2a(amie) * t2a(ecjk) * t2b(bdml)
        (-0.5, [("h2a_voov", "aMiE"), ("t2a", "Ecjk"), ("t2b", "bdMl")]),
        # Diagram 8: A(i/jk)A(c/ab) h2a(abef) * t2a(fcjk) * t2b(edil)
        (0.25, [("h2a_vvvv", "abEF"), ("t2a", "Fcjk"), ("t2b", "Edil")]),
        # Diagram 9: -A(ijk)A(a/bc) h2a(amie) * t2a(bcmk) * t2b(edjl)
        (-0.5, [("h2a_voov", "aMiE"), ("t2a", "bcMk"), ("t2b", "Edjl")]),
        # Diagram 10: A(k/ij)A(abc) h2b(adef) * t2a(ebij) * t2b(cfkl)
        (0.5, [("h2b_vvvv", "adEF"), ("t2a", "Ebij"), ("t2b", "cFkl")]),
    ]
    l = [
        # Diagram 1: A(k/ij)A(c/ab) h2a(ijab) * l2b(cdkl)
        (0.25, [("h2a_oovv", "ijab"), ("l2b", "cdkl")]),
        # Diagram 2: A(k/ij)A(c/ab) h2b(klcd) * l2a(abij)
        (0.25, [("h2b_oovv", "klcd"), ("l2a", "abij")]),
    ]
    return "aaab", "abcd", "ijkl", tensors, x, l


def quadruples_aabb(T, L, H):
    tensors = {"t2a": T.aa, "t2b": T.ab, "t2c": T.bb, "l2a": L.aa, "l2b": L.ab, "l2c": L.bb,
               "h2a_voov": H.aa.voov, "h2a_oooo": H.aa.oooo, "h2a_vvvv": H.aa.vvvv, "h2a_oovv": H.aa.oovv,
               "h2b_voov": H.ab.voov, "h2b_ovov": H.ab.ovov, "h2b_vovo": H.ab.vovo, "h2b_ovvo": H.ab.ovvo,
               "h2b_oooo": H.ab.oooo, "h2b_vvvv": H.ab.vvvv, "h2b_oovv": H.ab.oovv,
               "h2c_voov": H.bb.voov, "h2c_oooo": H.bb.oooo, "h2c_vvvv": H.bb.vvvv, "h2c_oovv": H.bb.oovv}
    x = [
        # Diagram 1: -A(ij)A(kl)A(ab)A(cd) h2c(cmke) * t2b(adim) * t2b(bejl)
        (-1.0, [("h2c_voov", "cMkE"), ("t2b", "adiM"), ("t2b", "bEjl")]),
        # Diagram 2: -A(ij)A(kl)A(ab)A(cd) h2a(amie) * t2b(bcmk) * t2b(edjl)
        (-1.0, [("h2a_voov", "aMiE"), ("t2b", "bcMk"), ("t2b", "Edjl")]),
        # Diagram 3: -A(kl)A(ab)A(cd) h2b(mcek) * t2a(aeij) * t2b(bdml)
        (-0.5, [("h2b_ovvo", "McEk"), ("t2a", "aEij"), ("t2b", "bdMl")]),
        # Diagram 4: -A(ij)A(ab)A(cd) h2b(amie) * t2b(bdjm) * t2c(cekl)
        (-0.5, [("h2b_voov", "aMiE"), ("t2b", "bdjM"), ("t2c", "cEkl")]),
        # Diagram 5: -A(ij)A(kl)A(cd) h2b(mcek) * t2a(abim) * t2b(edjl)
        (-0.5, [("h2b_ovvo", "McEk"), ("t2a", "abiM"), ("t2b", "Edjl")]),
        # Diagram 6: -A(ij)A(kl)A(ab) h2b(amie) * t2c(cdkm) * t2b(bejl)
        (-0.5, [("h2b_voov", "aMiE"), ("t2c", "cdkM"), ("t2b", "bEjl")]),
        # Diagram 7: -A(ij)A(kl)A(ab)A(cd) h2b(bmel) * t2b(adim) * t2b(ecjk)
        (-1.0, [("h2b_vovo", "bMEl"), ("t2b", "adiM"), ("t2b", "Ecjk")]),
        # Diagram 8: -A(ij)A(kl)A(ab)A(cd) h2b(mdje) * t2b(bcmk) * t2b(aeil)
        (-1.0, [("h2b_ovov", "MdjE"), ("t2b", "bcMk"), ("t2b", "aEil")]),
        # Diagram 9: -A(ij)A(cd) h2b(mdje) * t2a(abim) * t2c(cekl)
        (-0.25, [("h2b_ovov", "MdjE"), ("t2a", "abiM"), ("t2c", "cEkl")]),
        # Diagram 10: -A(kl)A(ab) h2b(bmel) * t2a(aeij) * t2c(cdkm)
        (-0.25, [("h2b_vovo", "bMEl"), ("t2a", "aEij"), ("t2c", "cdkM")]),
        # Diagram 11: A(kl)A(ab) h2a(mnij) * t2b(acmk) * t2b(bdnl)
        (0.25, [("h2a_oooo", "MNij"), ("t2b", "acMk"), ("t2b", "bdNl")]),
        # Diagram 12: A(ij)A(kl) h2a(abef) * t2b(ecik) * t2b(fdjl)
        (0.25, [("h2a_vvvv", "abEF"), ("t2b", "Ecik"), ("t2b", "Fdjl")]),
        # Diagram 13: A(ij)A(kl) h2b(mnik) * t2a(abmj) * t2c(cdnl)
        (0.25, [("h2b_oooo", "MNik"), ("t2a", "abMj"), ("t2c", "cdNl")]),
        # Diagram 14: A(ab)A(cd) h2b(acef) * t2a(ebij) * t2c(fdkl)
        (0.25, [("h2b_vvvv", "acEF"), ("t2a", "Ebij"), ("t2c", "Fdkl")]),
        # Diagram 15: A(ij)A(kl)A(ab)A(cd) h2b(mnik) * t2b(adml) * t2b(bcjn)
        (1.0, [("h2b_oooo", "MNik"), ("t2b", "adMl"), ("t2b", "bcjN")]),
        # Diagram 16: A(ij)A(kl)A(ab)A(cd) h2b(acef) * t2b(edil) * t2b(bfjk)
        (1.0, [("h2b_vvvv", "acEF"), ("t2b", "Edil"), ("t2b", "bFjk")]),
        # Diagram 17: A(ij)A(cd) h2c(mnkl) * t2b(adin) * t2b(bcjm)
        (0.25, [("h2c_oooo", "MNkl"), ("t2b", "adiN"), ("t2b", "bcjM")]),
        # Diagram 18: A(ij)A(kl) h2c(cdef) * t2b(afil) * t2b(bejk)
        (0.25, [("h2c_vvvv", "cdEF"), ("t2b", "aFil"), ("t2b", "bEjk")]),
    ]
    l = [
        # Diagram 1: h2a(ijab) * l2c(cdkl)
        (1.0 / 16.0, [("h2a_oovv", "ijab"), ("l2c", "cdkl")]),
        # Diagram 2: h2c(klcd) * l2a(abij)
        (1.0 / 16.0, [("h2c_oovv", "klcd"), ("l2a", "abij")]),
        # Diagram 3: A(ij)A(kl)A(ab)A(cd) h2b(ikac) * l2b(bdjl)
        (1.0, [("h2b_oovv", "ikac"), ("l2b", "bdjl")]),
    ]
    return "aabb", "abcd", "ijkl", tensors, x, l


def quadruples_abbb(T, L, H):
    # labels follow moments_abbb_ijkl: d, l are alpha; a, b, c and i, j, k are beta
    tensors = {"t2b": T.ab, "t2c": T.bb, "l2b": L.ab, "l2c": L.bb,
               "h2b_voov": H.ab.voov, "h2b_ovov": H.ab.ovov, "h2b_vovo": H.ab.vovo, "h2b_ovvo": H.ab.ovvo,
               "h2b_oooo": H.ab.oooo, "h2b_vvvv": H.ab.vvvv, "h2b_oovv": H.ab.oovv,
               "h2c_voov": H.bb.voov, "h2c_oooo": H.bb.oooo, "h2c_vvvv": H.bb.vvvv, "h2c_oovv": H.bb.oovv}
    x = [
        # Diagram 1: -A(i/jk)A(c/ab) h2b(dmle) * t2c(abim) * t2c(ecjk)
        (-0.25, [("h2b_voov", "dMlE"), ("t2c", "abiM"), ("t2c", "Ecjk")]),
        # Diagram 2: A(k/ij)A(a/bc) h2c(mnij) * t2c(bcnk) * t2b(dalm)
        (0.25, [("h2c_oooo", "MNij"), ("t2c", "bcNk"), ("t2b", "dalM")]),
        # Diagram 3: -A(ijk)A(c/ab) h2b(dmfj) * t2c(abim) * t2b(fclk)
        (-0.5, [("h2b_vovo", "dMFj"), ("t2c", "abiM"), ("t2b", "Fclk")]),
        # Diagram 4: -A(ijk)A(abc) h2b(maei) * t2b(eblj) * t2b(dcmk)
        (-1.0, [("h2b_ovvo", "MaEi"), ("t2b", "Eblj"), ("t2b", "dcMk")]),
        # Diagram 5: A(ijk)A(a/bc) h2b(nmlj) * t2c(bcmk) * t2b(dani)
        (0.5, [("h2b_oooo", "NMlj"), ("t2c", "bcMk"), ("t2b", "daNi")]),
        # Diagram 6: -A(i/jk)A(abc) h2b(mble) * t2c(ecjk) * t2b(dami)
        (-0.5, [("h2b_ovov", "MblE"), ("t2c", "Ecjk"), ("t2b", "daMi")]),
        # Diagram 7: -A(i/jk)A(abc) h2c(amie) * t2c(ecjk) * t2b(dblm)
        (-0.5, [("h2c_voov", "aMiE"), ("t2c", "Ecjk"), ("t2b", "dblM")]),
        # Diagram 8: A(i/jk)A(c/ab) h2c(abef) * t2c(fcjk) * t2b(deli)
        (0.25, [("h2c_vvvv", "abEF"), ("t2c", "Fcjk"), ("t2b", "dEli")]),
        # Diagram 9: -A(ijk)A(a/bc) h2c(amie) * t2c(bcmk) * t2b(delj)
        (-0.5, [("h2c_voov", "aMiE"), ("t2c", "bcMk"), ("t2b", "dElj")]),
        # Diagram 10: A(k/ij)A(abc) h2b(dafe) * t2c(ebij) * t2b(fclk)
        (0.5, [("h2b_vvvv", "daFE"), ("t2c", "Ebij"), ("t2b", "Fclk")]),
    ]
    l = [
        # Diagram 1: A(k/ij)A(c/ab) h2c(ijab) * l2b(dclk)
        (0.25, [("h2c_oovv", "ijab"), ("l2b", "dclk")]),
        # Diagram 2: A(k/ij)A(c/ab) h2b(lkdc) * l2c(abij)
        (0.25, [("h2b_oovv", "lkdc"), ("l2c", "abij")]),
    ]
    return "abbb", "dabc", "lijk", tensors, x, l


def quadruples_bbbb(T, L, H):
    tensors = {"t2c": T.bb, "l2c": L.bb, "h2c_voov": H.bb.voov, "h2c_oooo": H.bb.oooo,
               "h2c_vvvv": H.bb.vvvv, "h2c_oovv": H.bb.oovv}
    x = [
        # Diagram 1: -A(jl/i/k)A(bc/a/d) h2c(amie) * t2c(bcmk) * t2c(edjl)
        (-0.25, [("h2c_voov", "aMiE"), ("t2c", "bcMk"), ("t2c", "Edjl")]),
        # Diagram 2: A(ij/kl)A(bc/ad) h2c(mnij) * t2c(adml) * t2c(bcnk)
        (1.0 / 16.0, [("h2c_oooo", "MNij"), ("t2c", "adMl"), ("t2c", "bcNk")]),
        # Diagram 3: A(jk/il)A(ab/cd) h2c(abef) * t2c(fcjk) * t2c(edil)
        (1.0 / 16.0, [("h2c_vvvv", "abEF"), ("t2c", "Fcjk"), ("t2c", "Edil")]),
    ]
    l = [
        # Diagram 1: A(ij/kl)A(ab/cd) h2c(klcd) * l2c(abij)
        (1.0 / 16.0, [("l2c", "abij"), ("h2c_oovv", "klcd")]),
    ]
    return "bbbb", "abcd", "ijkl", tensors, x, l

# ---------------------------------------------------------------------------
# Laplace contraction of one spin case
# ---------------------------------------------------------------------------

def laplace_spincase(spincase, vir, occ, tensors, x, l, H0, t, w, pspace=None):
    """sum_{all indices} x * l / D for one spin case, minus the P-space tuples if given."""
    spin = dict(zip(vir + occ, spincase + spincase))
    e_occ = {"a": np.diagonal(H0.a.oo), "b": np.diagonal(H0.b.oo)}
    e_vir = {"a": np.diagonal(H0.a.vv), "b": np.diagonal(H0.b.vv)}
    l_full = antisymmetrize(l, spincase, vir, occ)
    size = {c: tensors[name].shape[k] for _, f in x + l_full for name, lab in f for k, c in enumerate(lab)}

    delta = 0.0
    for tg, wg in zip(t, w):
        # exp(-t Delta) = prod_vir exp(-t e_a) * prod_occ exp(t e_i), absorbed into the x factors
        # (each external label occurs exactly once in an x diagram)
        weight = {c: np.exp(-tg * e_vir[spin[c]]) for c in vir} | {c: np.exp(tg * e_occ[spin[c]]) for c in occ}
        for cx, xf in x:
            x_arrays = [_scale(tensors[name], lab, weight) for name, lab in xf]
            for cl, lf in l_full:
                arrays = x_arrays + [tensors[name] for name, _ in lf]
                labels = [lab for _, lab in xf + lf]
                delta -= wg * cx * cl * contract_network(arrays, labels, size)

    if pspace is not None and len(pspace) > 0:
        # l_K m_K for K in P, weighted with the same quadrature (D_K = -Delta_K)
        m_P = tuple_values(antisymmetrize(x, spincase, vir, occ), tensors, pspace, vir + occ)
        l_P = tuple_values(l_full, tensors, pspace, vir + occ)
        n = len(vir)
        Delta = sum(e_vir[spin[c]][pspace[:, k]] for k, c in enumerate(vir)) \
              - sum(e_occ[spin[c]][pspace[:, n + k]] for k, c in enumerate(occ))
        delta += np.sum(m_P * l_P * (np.exp(-np.outer(Delta, t)) @ w))
    return delta


def _scale(a, lab, weight):
    for k, c in enumerate(lab):
        if c in weight:
            shape = [1] * a.ndim
            shape[k] = -1
            a = a * weight[c].reshape(shape)
    return a


def _pspace(excitations, spincase):
    """0-based P-space rows of one spin case (ccpy marks an empty spin case with a row of ones)."""
    if excitations is None:
        return None
    x = np.asarray(excitations[spincase], dtype=np.int64)
    if x.size == 0 or np.all(x[0] == 1):
        return np.zeros((0, x.shape[-1]), dtype=np.int64)
    return x - 1


def _pattern(name):
    """Orbital-space pattern of a tensor name (t2a -> vvoo, h2b_vovv -> vovv, ...)."""
    if "_" in name:
        return name.split("_")[1]
    return {"t2": "vvoo", "l2": "vvoo", "l1": "vo"}[name[:2]]


def antisymmetrize(terms, spincase, vir, occ):
    """Expand A_v A_o (same-spin permutations) over each diagram; merge diagrams that differ by
    the antisymmetry of same-spin two-body tensors."""
    groups = [[c for c, s in zip(labels, spincase) if s == sp] for labels in (vir, occ) for sp in "ab"]
    out = {}
    for coef, factors in terms:
        for perms in itertools.product(*(itertools.permutations(g) for g in groups)):
            sign = 1
            relabel = {}
            for g, p in zip(groups, perms):
                relabel.update(zip(g, p))
                sign *= _parity([g.index(c) for c in p])
            key = []
            for name, lab in factors:
                lab = [relabel.get(c, c) for c in lab]
                pattern = _pattern(name)
                if len(lab) == 4 and name[2] in "ac":
                    for p, q in ((0, 1), (2, 3)):
                        if pattern[p] == pattern[q] and lab[p] > lab[q]:
                            lab[p], lab[q] = lab[q], lab[p]
                            sign = -sign
                key.append((name, "".join(lab)))
            key = tuple(key)
            out[key] = out.get(key, 0.0) + coef * sign
    return [(c, list(k)) for k, c in out.items() if abs(c) > 1.0e-14]


def _parity(perm):
    perm, sign = list(perm), 1
    for i in range(len(perm)):
        while perm[i] != i:
            j = perm[i]
            perm[i], perm[j] = perm[j], perm[i]
            sign = -sign
    return sign


def tuple_values(terms, tensors, tuples, external, chunk=4096):
    """sum_terms coef * prod(factors) at each external tuple (row of 0-based indices ordered as
    `external`), summing the internal labels."""
    col = {c: k for k, c in enumerate(external)}
    out = np.zeros(len(tuples))
    for start in range(0, len(tuples), chunk):
        tup = tuples[start:start + chunk]
        for coef, factors in terms:
            ops, subs = [], []
            for name, lab in factors:
                ext = [k for k, c in enumerate(lab) if c in col]
                rest = [k for k in range(len(lab)) if k not in ext]
                a = tensors[name].transpose(ext + rest)
                ops.append(a[tuple(tup[:, col[lab[k]]] for k in ext)])
                subs.append("p" + "".join(lab[k] for k in rest))
            out[start:start + chunk] += coef * np.einsum(",".join(subs) + "->p", *ops, optimize=True)
    return out

# ---------------------------------------------------------------------------
# Closed-network contraction with an explicit order search
# ---------------------------------------------------------------------------

_PATHS = {}


def _best_path(labels, size):
    """Pairwise order minimizing (max distinct indices per step, flops); each index occurs twice."""
    key = (tuple(labels), tuple(sorted(size.items())))
    if key in _PATHS:
        return _PATHS[key]

    def search(labs):
        if len(labs) == 1:
            return (0, 0), []
        best = None
        for p, q in itertools.combinations(range(len(labs)), 2):
            u = set(labs[p]) | set(labs[q])
            out = "".join(c for c in labs[p] + labs[q] if (c in labs[p]) != (c in labs[q]))
            rest = [lab for k, lab in enumerate(labs) if k not in (p, q)] + [out]
            (mx, fl), path = search(rest)
            flops = int(np.prod([size[c] for c in u]))
            cand = ((max(mx, len(u)), fl + flops), [(p, q, out, "".join(sorted(u)))] + path)
            if best is None or cand[0] < best[0]:
                best = cand
        return best

    _PATHS[key] = search(list(labels))
    return _PATHS[key]


def contract_network(arrays, labels, size):
    _, path = _best_path(labels, size)
    arrays, labels = list(arrays), list(labels)
    for p, q, out, _ in path:
        r = np.einsum(f"{labels[p]},{labels[q]}->{out}", arrays[p], arrays[q], optimize=True)
        arrays = [a for k, a in enumerate(arrays) if k not in (p, q)] + [r]
        labels = [lab for k, lab in enumerate(labels) if k not in (p, q)] + [out]
    return float(arrays[0])


def network_scaling(spincase, vir, occ, tensors, x, l, no=20, nv=200):
    """Worst step (distinct indices) over all networks of a spin case, its o^p v^q forms,
    and the number of networks, for model dimensions no, nv."""
    l_full = antisymmetrize(l, spincase, vir, occ)
    worst, forms = 0, set()
    for _, xf in x:
        for _, lf in l_full:
            space = {c: _pattern(name)[k] for name, lab in xf + lf for k, c in enumerate(lab)}
            size = {c: (nv if s == "v" else no) for c, s in space.items()}
            (mx, _), path = _best_path([lab for _, lab in xf + lf], size)
            if mx > worst:
                worst, forms = mx, set()
            for *_, u in path:
                if len(u) == worst:
                    forms.add(f"o^{sum(space[c] == 'o' for c in u)} v^{sum(space[c] == 'v' for c in u)}")
    return worst, sorted(forms), len(x) * len(l_full)

# ---------------------------------------------------------------------------
# Laplace quadrature for 1/x on [xmin, xmax]
# ---------------------------------------------------------------------------

def _quadrature(H0, n, npts):
    """Minimax quadrature covering every n-fold MP denominator of both spins."""
    e_occ = np.concatenate([np.diagonal(H0.a.oo), np.diagonal(H0.b.oo)])
    e_vir = np.concatenate([np.diagonal(H0.a.vv), np.diagonal(H0.b.vv)])
    xmin, xmax = n * (e_vir.min() - e_occ.max()), n * (e_vir.max() - e_occ.min())
    t, w = minimax_quadrature(npts, xmin, xmax)
    return t, w, quadrature_error(t, w, xmin, xmax)


def laplace_quadrature(npts, xmin, xmax):
    """Trapezoid rule in s = ln t for 1/x = int exp(s - x e^s) ds.

    The aliasing error ~exp(-pi^2/h) and the two truncation tails are balanced
    for the requested number of points. Used as the starting guess for the minimax fit.
    """
    def nodes(eps):
        h = np.pi ** 2 / np.log(1.0 / eps)
        smin = np.log(eps / xmax)
        smax = np.log(np.log(1.0 / eps) / xmin)
        return h, smin, int(np.ceil((smax - smin) / h)) + 1

    # largest accuracy target eps that fits in npts points (monotone in eps)
    lo, hi = 1.0e-16, 0.5
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        if nodes(mid)[2] > npts:
            lo = mid
        else:
            hi = mid
    h, smin, _ = nodes(hi)
    s = smin + h * np.arange(npts)
    return np.exp(s), h * np.exp(s)


@functools.lru_cache(maxsize=None)
def _minimax_unit(npts, R):
    """Exponential sum minimizing max |y*sum_g w_g exp(-t_g y) - 1| on [1, R].

    Lawson iteration (reweighted least squares) from the trapezoid nodes; parameters
    are fit as logs so t, w stay positive.
    """
    y = np.geomspace(1.0, R, 60 * npts)

    def model(p):
        t, w = np.exp(p[:npts]), np.exp(p[npts:])
        E = np.exp(-np.outer(y, t))
        return y * (E @ w) - 1.0, np.hstack([-(y ** 2)[:, None] * E * (w * t), y[:, None] * E * w])

    p = np.log(np.concatenate(laplace_quadrature(npts, 1.0, R)))
    wt = np.full(y.size, 1.0 / y.size)
    best = (np.inf, p)
    for _ in range(40):
        s = np.sqrt(wt)
        p = least_squares(lambda q: s * model(q)[0], p, jac=lambda q: s[:, None] * model(q)[1],
                          method="trf", bounds=(-40.0, 10.0), x_scale="jac",
                          ftol=1.0e-15, xtol=1.0e-15, gtol=1.0e-15).x
        r = np.abs(model(p)[0])
        if r.max() < best[0]:
            best = (r.max(), p)
        wt = wt * r
        wt /= wt.sum()
    p = best[1]
    return np.exp(p[:npts]), np.exp(p[npts:])


def minimax_quadrature(npts, xmin, xmax):
    """Minimax exponential sum for 1/x on [xmin, xmax] (scale-invariant fit on [1, R])."""
    # round R up on a sqrt(2) grid so fits are reused; a fit on a wider interval stays valid
    R = 2.0 ** (np.ceil(2.0 * np.log2(xmax / xmin)) / 2.0)
    t, w = _minimax_unit(npts, float(R))
    return t / xmin, w / xmin


def quadrature_error(t, w, xmin, xmax):
    x = np.geomspace(xmin, xmax, 2000)
    approx = np.exp(-np.outer(x, t)) @ w
    return np.max(np.abs(approx * x - 1.0))


def _summary(name, corr_energy, correction_A, system, npts, qerr, t_start):
    minutes, seconds = divmod(time.perf_counter() - t_start, 60)
    energy_A = corr_energy + correction_A
    total_energy_A = system.reference_energy + energy_A
    print(f"   Laplace {name}_A Calculation Summary")
    print("   -------------------------------------")
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Laplace points = {npts}, max rel. quadrature error = {qerr:.2e}")
    print("   {}_A = {:>10.10f}     ΔE_A = {:>10.10f}     δ_A = {:>10.10f}\n".format(
        name, total_energy_A, energy_A, correction_A))
    return {"A": total_energy_A}, {"A": correction_A}
