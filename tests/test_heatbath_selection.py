"""Heat-bath selection of triples and quadruples: exhausted (eps -> 0) it is the exact top-k of the
CC(P;Q)_A contributions outside a P space, and those contributions add up to the Laplace correction."""
import itertools

import numpy as np
import pytest
from pyscf import gto, scf

from ccpy import Driver
from ccpy.moments import crcc_laplace as lap
from ccpy.moments.heatbath_selection import QUADRUPLES, TRIPLES, _flip_columns, _le, _to_spinorbital, select_heatbath


@pytest.fixture(scope="module")
def water():
    mol = gto.M(atom="O 0 0 -0.018; H 0 3.030526 -2.117796; H 0 -3.030526 -2.117796", basis="6-31g",
                unit="Bohr", symmetry="C2v", verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    d = Driver.from_pyscf(mf, nfrozen=1)
    d.run_cc(method="ccsd")
    d.run_hbar(method="ccsd")
    d.run_leftcc(method="left_ccsd")
    return d


def _all_tuples(spincase, system):
    n = {"v": {"a": system.nunoccupied_alpha, "b": system.nunoccupied_beta},
         "o": {"a": system.noccupied_alpha, "b": system.noccupied_beta}}
    groups = [[a + b for a, b in itertools.product(
        *(itertools.combinations(range(n[s][sp]), spincase.count(sp)) for sp in "ab"))] for s in "vo"]
    return np.array([v + o for v, o in itertools.product(*groups)], dtype=np.int64)


def _random_pspace(spincases, system, frac, rng):
    out = {}
    for sc in spincases:
        x = _all_tuples(sc, system) + 1
        x = x[rng.random(len(x)) < frac]
        out[sc] = x if len(x) else np.ones((1, 2 * len(sc)), dtype=np.int64)
    return out


def _brute_force(d, p3, p4, use_RHF):
    """Spinorbital row -> CC(P;Q)_A contribution of every excitation outside P."""
    e_occ = {"a": np.diagonal(d.fock.a.oo), "b": np.diagonal(d.fock.b.oo)}
    e_vir = {"a": np.diagonal(d.fock.a.vv), "b": np.diagonal(d.fock.b.vv)}
    out, total = {}, 0.0
    for spincases, pspace in ((TRIPLES, p3), (QUADRUPLES, p4)):
        for s, f in spincases.items():
            if use_RHF and s.count("a") < 2:
                continue
            spincase, vir, occ, tensors, x, l = f(d.T, d.L[0], d.hamiltonian)
            c = _all_tuples(s, d.system)
            p = set(map(tuple, lap._pspace(pspace, s)))
            c = c[[tuple(r) not in p for r in c]]
            m = lap.tuple_values(lap.antisymmetrize(x, s, vir, occ), tensors, c, vir + occ)
            lk = lap.tuple_values(lap.antisymmetrize(l, s, vir, occ), tensors, c, vir + occ)
            r = len(s)
            delta = sum(e_vir[s[k]][c[:, k]] for k in range(r)) - sum(e_occ[s[k]][c[:, r + k]] for k in range(r))
            v = -lk * m / delta
            weight = 1.0 if not use_RHF else (1.0 if s == "aabb" else 2.0)
            total += weight * v.sum()
            if use_RHF and s == "aabb":
                keep = _le(c, c[:, _flip_columns(r)])
                c, v = c[keep], v[keep]
            out.update(zip(map(tuple, _to_spinorbital(c, s)), v))
    return out, total


@pytest.mark.parametrize("use_RHF", [True, False])
def test_exhausted_heatbath_is_exact_topk(water, use_RHF):
    d = water
    rng = np.random.default_rng(3)
    p3 = _random_pspace(TRIPLES, d.system, 0.2, rng)
    p4 = _random_pspace(QUADRUPLES, d.system, 0.02, rng)
    exact, total = _brute_force(d, p3, p4, use_RHF)

    # the contributions add up to the Laplace CC(P;3)_A + CC(P;4)_A corrections
    args = (d.T, d.L[0], d.correlation_energy, d.hamiltonian, d.fock, d.system)
    d3 = lap.calc_crcc23_laplace(*args, use_RHF=use_RHF, npts=10, t3_excitations=p3)[1]["A"]
    d4 = lap.calc_crcc24_laplace(*args, use_RHF=use_RHF, npts=10, t4_excitations=p4)[1]["A"]
    assert abs(total - d3 - d4) < 1.0e-8

    num_add = 200
    s3, s4 = select_heatbath(d.T, d.L[0], d.hamiltonian, d.fock, d.system, p3, num_add, t4_excitations=p4,
                             use_RHF=use_RHF, buffer_factor=10**9)
    selected = list(map(tuple, s3)) + list(map(tuple, s4))
    assert len(selected) == len(set(selected)) == num_add
    assert all(k in exact for k in selected)  # outside P, canonical
    ranked = sorted(np.abs(list(exact.values())), reverse=True)
    cut = ranked[num_add - 1]
    assert np.isclose(min(abs(exact[k]) for k in selected), cut, rtol=0, atol=1.0e-14)
    above = {k for k, v in exact.items() if abs(v) > cut + 1.0e-14}
    assert above <= set(selected)
    assert any(len(k) == 8 for k in selected) and any(len(k) == 6 for k in selected)
