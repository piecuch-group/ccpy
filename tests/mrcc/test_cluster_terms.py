"""cluster_terms lists exactly the partition products the Fortran recursion sums."""
import numpy as np
import pytest

from ccpy.lib.core import sumrcc_loops
from ccpy.mrcc.sumrcc import cluster_terms, format_cluster_terms

NORB, NOCC = 6, 3  # spatial orbitals 0-2 occupied in Phi_q (both spins), 3-5 empty


def amplitudes(rng):
    no, nu = NOCC, NORB - NOCC
    t1a, t1b = rng.standard_normal((nu, no)), rng.standard_normal((nu, no))
    t2a, t2b, t2c = (rng.standard_normal((nu, nu, no, no)) for _ in range(3))
    for t2 in (t2a, t2c):
        t2 -= t2.transpose(1, 0, 2, 3)
        t2 -= t2.transpose(0, 1, 3, 2)
    return t1a, t1b, t2a, t2b, t2c


def evaluate(terms, t, internal_amp):
    t1a, t1b, t2a, t2b, t2c = t
    lo = lambda s: s % NORB
    lu = lambda s: s % NORB - NOCC
    total = 0.0
    for sign, blocks in terms:
        value = float(sign)
        for h, p in blocks:
            if len(h) == 1:
                value *= (t1a if h[0] < NORB else t1b)[lu(p[0]), lo(h[0])]
            elif len(h) == 2:
                spins = (h[0] >= NORB) + (h[1] >= NORB)
                value *= (t2a, t2b, t2c)[spins][lu(p[0]), lu(p[1]), lo(h[0]), lo(h[1])]
            else:
                value *= internal_amp
        total += value
    return total


@pytest.mark.parametrize("holes, particles", [
    ((0,), (3,)),
    ((0, 7), (4, 10)),
    ((0, 1, 6), (3, 4, 9)),          # aab triple, also the internal block
    ((0, 1, 6, 7), (3, 5, 9, 11)),
    ((0, 1, 2, 6, 7), (3, 4, 5, 9, 10)),
])
def test_cluster_terms_match_fortran(holes, particles):
    rng = np.random.default_rng(len(holes))
    t = amplitudes(rng)
    internal = [((0, 1, 6), (3, 4, 9))]
    internal_amp = 0.37
    terms = cluster_terms(holes, particles, NORB, internal)
    nso = 2 * NORB
    lo = np.zeros(nso, np.int32)
    lu = np.zeros(nso, np.int32)
    for s in range(nso):
        g = s % NORB
        if g < NOCC:
            lo[s] = g + 1
        else:
            lu[s] = g - NOCC + 1
    int_h = np.asfortranarray(np.array([[0], [1], [6]], np.int32))
    int_p = np.asfortranarray(np.array([[3], [4], [9]], np.int32))
    fortran = sumrcc_loops.cluster_coefficient(np.array(holes, np.int32), np.array(particles, np.int32), *t,
                                               int_h, int_p, np.array([3], np.int32), np.array([internal_amp]),
                                               lo, lu, np.zeros(nso, np.int32), 0, norb=NORB)
    assert abs(evaluate(terms, t, internal_amp) - fortran) < 1e-12
    assert format_cluster_terms(terms, NORB).count("\n") == len(terms) - 1
