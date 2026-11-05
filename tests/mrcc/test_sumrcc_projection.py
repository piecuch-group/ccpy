"""P(q->p)_K = <Phi_K(p)|exp(T(q))|Phi_q> from sumrcc_loops against a brute-force exp(T(q))
on the determinant space, for complete and general model spaces. The brute force imposes the
C conditions itself: each rank >= 3 internal amplitude cancels <Phi_r|exp(T(q))|Phi_q>."""
from itertools import combinations

import numpy as np
import pytest

from ccpy.gradients.ccpq.determinants import DeterminantSpace
from ccpy.models.operators import ClusterOperator
from ccpy.models.system import System
from ccpy.mrcc.sumrcc import get_cas_determinants, get_model_space, get_projection, get_internal_amplitudes
from ccpy.mrcc.sumrccsd import apply_masks, get_core_virtual

KEYS = ("a", "b", "aa", "ab", "bb")


def random_cluster_operator(system, ms, q, rng, scale=0.3):
    """Random T(q), antisymmetric t2a/t2c, internal t1/t2 from the C conditions."""
    T = ClusterOperator(system, 2)
    for key in KEYS:
        x = scale * rng.standard_normal(getattr(T, key).shape)
        if key in ("aa", "bb"):
            x -= x.transpose(1, 0, 2, 3)
            x -= x.transpose(0, 1, 3, 2)
        setattr(T, key, x)
    apply_masks(T, ClusterOperator(system, 2), ms.internal[q], get_core_virtual(system, ms), None)
    return T


def excitation_strings(orbs, norb, nfrozen=0):
    """Operator strings of every singles/doubles block entry in ccpy index convention."""
    occ_a, unocc_a, occ_b, unocc_b = [x + nfrozen for x in orbs]
    out = {key: [] for key in KEYS}
    for spin, o, u, off in (("a", occ_a, unocc_a, 0), ("b", occ_b, unocc_b, norb)):
        for i, I in enumerate(o):
            for a, A in enumerate(u):
                out[spin].append(((a, i), [("+", A + off), ("-", I + off)]))
    for key, o1, o2, u1, u2, f1, f2 in (("aa", occ_a, occ_a, unocc_a, unocc_a, 0, 0),
                                        ("ab", occ_a, occ_b, unocc_a, unocc_b, 0, norb),
                                        ("bb", occ_b, occ_b, unocc_b, unocc_b, norb, norb)):
        for i, I in enumerate(o1):
            for j, J in enumerate(o2):
                for a, A in enumerate(u1):
                    for b, B in enumerate(u2):
                        if key != "ab" and (i == j or a == b):
                            continue
                        ops = [("+", A + f1), ("+", B + f2), ("-", J + f2), ("-", I + f1)]
                        out[key].append(((a, b, i, j), ops))
    return out


def determinant(space, orbs, nfrozen=0):
    occ_a, _, occ_b, _ = [x + nfrozen for x in orbs]
    frozen = (1 << space.nfrozen) - 1
    d = frozen | (frozen << space.norb) | sum(1 << int(g) for g in occ_a) | sum(1 << (space.norb + int(g)) for g in occ_b)
    idx, found = space.lookup(np.array([d], dtype=np.uint64))
    assert found[0]
    return idx[0]


def exponential(T, v):
    psi, term = v.copy(), v.copy()
    for k in range(1, 30):
        term = T @ term / k
        psi += term
    return psi


def cc_wavefunction(space, ms, q, T, nfrozen=0, return_t=False):
    """exp(T(q))|Phi_q> over the determinant space, with the rank >= 3 internal amplitudes of
    q set by the C conditions (in order of rank). Checks <Phi_r|exp(T(q))|Phi_q> = 0. With
    return_t, also the full T(q) as a sparse matrix."""
    norb = space.norb
    ops, amps = [], []
    for key, entries in excitation_strings(ms.references[q], norb, nfrozen).items():
        amp = getattr(T, key)
        for idx, op in entries:
            if key in ("aa", "bb") and not (idx[0] < idx[1] and idx[2] < idx[3]):
                continue
            ops.append(op)
            amps.append(amp[idx])
    Tm = space.transitions_of(ops).matrix(np.array(amps), space.ndet)
    ref = np.zeros(space.ndet)
    ref[determinant(space, ms.references[q], nfrozen)] = 1.0
    int_h, int_p, int_k = ms.higher[q]
    so = lambda s: (s % (norb - nfrozen)) + nfrozen + (s // (norb - nfrozen)) * norb
    # amplitudes of one rank do not enter each other's C conditions: one exponential per rank
    for k in sorted(set(int_k) - {0}):
        psi = exponential(Tm, ref)
        for e in np.flatnonzero(int_k == k):
            # canonical string a+_P1..a+_Pk a_Hk..a_H1 on the (frozen-shifted) bits
            op = [("+", so(s)) for s in int_p[:k, e]] + [("-", so(s)) for s in reversed(int_h[:k, e])]
            E = space.transitions_of([op]).matrix(np.array([1.0]), space.ndet)
            c = (E @ ref) @ psi  # = sign * <Phi_r|psi>
            Tm = Tm - c * E  # amplitude -c * sign for the canonical string; sign^2 = 1
    psi = exponential(Tm, ref)
    for r, other in enumerate(ms.references):
        if r != q:
            assert abs(psi[determinant(space, other, nfrozen)]) < 1e-12, "C conditions violated"
    return (psi, Tm) if return_t else psi


def brute_force_projection(space, ms, p, q, T):
    psi = cc_wavefunction(space, ms, q, T)
    src = np.array([determinant(space, ms.references[p])])
    P = {key: np.zeros(getattr(T, key).shape) for key in KEYS}
    for key, entries in excitation_strings(ms.references[p], space.norb).items():
        for idx, op in entries:
            _, d, sign = space.apply(op, src=src)
            if len(d):
                k, found = space.lookup(d)
                P[key][idx] = sign[0] * psi[k[0]]
    return P


def check(system, determinants, pairs, seed):
    rng = np.random.default_rng(seed)
    ms = get_model_space(system, determinants)
    space = DeterminantSpace(system.norbitals, system.noccupied_alpha, system.noccupied_beta)
    nonzero = 0
    for p, q in pairs:
        T = random_cluster_operator(system, ms, q, rng)
        ref = brute_force_projection(space, ms, p, q, T)
        out = get_projection(T, get_internal_amplitudes(T, ms, q, system), ms, p, q, system)
        for key, x in zip(KEYS, out):
            assert np.allclose(x, ref[key], atol=1e-11), (p, q, key, np.abs(x - ref[key]).max())
            nonzero += np.count_nonzero(np.abs(ref[key]) > 1e-8)
    assert nonzero > 0
    return ms


@pytest.mark.parametrize("ncore, nact, nel, nvirt", [(2, 2, 2, 2), (2, 4, 4, 2), (2, 6, 6, 2)])
def test_complete_model_space(ncore, nact, nel, nvirt):
    system = System(2 * ncore + nel, ncore + nact + nvirt, 1, 0)
    dets = get_cas_determinants(system, nel, nact)
    rng = np.random.default_rng(nact)
    pairs = [(0, 0), (len(dets) - 1, 0)] + [(int(p), 0) for p in rng.choice(len(dets), 3)]
    ms = check(system, dets, pairs, nact)
    # in a complete model space the C conditions give vanishing internal amplitudes
    T = random_cluster_operator(system, ms, 0, rng)
    assert not np.any(get_internal_amplitudes(T, ms, 0, system))
    assert not np.any(T.a[ms.internal[0]["a"]]) and not np.any(T.ab[ms.internal[0]["ab"]])


def test_two_references_quadruply_excited():
    """{core^2 (12)^2, core^2 (34)^2}: a rank-4 internal excitation fixed by the C conditions,
    built from external active singles and doubles."""
    system = System(8, 8, 1, 0)
    dets = [[1, 2, 3, 4, 1, 2, 3, 4], [1, 2, 5, 6, 1, 2, 5, 6]]
    ms = check(system, dets, [(0, 1), (1, 0), (0, 0)], 3)
    assert ms.higher[0][2][0] == 4 and not ms.internal[0]["ab"].any()


@pytest.mark.parametrize("nremove", [1, 5])
def test_incomplete_cas(nremove):
    """CAS(4,4) with determinants removed: internal doubles -A t1 t1 and rank 3-4 amplitudes."""
    system = System(8, 8, 1, 0)
    cas = get_cas_determinants(system, 4, 4)
    rng = np.random.default_rng(nremove)
    keep = sorted(set(range(len(cas))) - set(rng.choice(range(1, len(cas)), nremove, replace=False).tolist()))
    dets = [cas[k] for k in keep]
    pairs = [(0, len(dets) - 1), (len(dets) - 1, 0), (3, 7), (7, 3), (5, 5)]
    check(system, dets, pairs, nremove)


def test_sparse_cas66():
    """A dozen CAS(6,6) determinants including the rank-6 pair."""
    system = System(10, 10, 1, 0)
    cas = get_cas_determinants(system, 6, 6)
    rng = np.random.default_rng(0)
    keep = [0, len(cas) - 1] + rng.choice(range(1, len(cas) - 1), 10, replace=False).tolist()
    dets = [cas[k] for k in keep]
    check(system, dets, [(1, 0), (0, 1), (2, 3), (4, 0)], 11)
