"""Laplace CR-CC(2,3)_A and CR-CC(2,4)_A converge to the canonical corrections, restrict to a CC(P;Q)
Q space, and contract in O(N^6) in every spin case."""
import itertools

import numpy as np
import pytest
from pyscf import gto, scf

from ccpy import Driver
from ccpy.moments import crcc_laplace
from ccpy.moments.ccp3 import calc_ccp3_2ba
from ccpy.moments.ccp34 import calc_ccp4_with_selection
from ccpy.moments.crcc_laplace import calc_crcc23_laplace, calc_crcc24_laplace


def _driver(atom, spin, method, basis="6-31g"):
    mol = gto.M(atom=atom, basis=basis, unit="Bohr", spin=spin, symmetry="C2v", verbose=0)
    mf = method(mol)
    mf.kernel()
    d = Driver.from_pyscf(mf, nfrozen=0)
    d.run_cc(method="ccsd")
    d.run_hbar(method="ccsd")
    d.run_leftcc(method="left_ccsd")
    return d


def _random_pspace(spincases, system, frac, rng):
    """Random P space in ccpy's 1-based row format (alpha indices first); frac=1 keeps every tuple."""
    n = {"v": {"a": system.nunoccupied_alpha, "b": system.nunoccupied_beta},
         "o": {"a": system.noccupied_alpha, "b": system.noccupied_beta}}
    out = {}
    for sc in spincases:
        groups = [[a + b for a, b in itertools.product(
            *(itertools.combinations(range(1, n[s][sp] + 1), sc.count(sp)) for sp in "ab"))] for s in "vo"]
        x = np.array([v + o for v, o in itertools.product(*groups)], dtype=np.int64)
        x = x[rng.random(len(x)) < frac]
        out[sc] = x if len(x) else np.ones((1, 2 * len(sc)), dtype=np.int64)
    return out


@pytest.mark.parametrize("atom, spin, method", [
    ("O 0 0 -0.018; H 0 3.030526 -2.117796; H 0 -3.030526 -2.117796", 0, scf.RHF),  # H2O, 2Re
    ("O 0 0 0; H 0 0 1.8324", 1, scf.ROHF),                                           # OH radical
])
def test_laplace_matches_canonical(atom, spin, method):
    d = _driver(atom, spin, method)
    d.run_ccp3(method="crcc23")
    d.run_ccp4(method="crcc24")
    args = (d.T, d.L[0], d.correlation_energy, d.hamiltonian, d.fock, d.system)
    rhf = [False, True] if spin == 0 else [False]

    for use_RHF in rhf:
        for npts, tol in ((6, 1.0e-5), (10, 1.0e-8)):
            _, d3 = calc_crcc23_laplace(*args, use_RHF=use_RHF, npts=npts)
            _, d4 = calc_crcc24_laplace(*args, use_RHF=use_RHF, npts=npts)
            assert abs(d3["A"] - d.deltap3[0]["A"]) < tol
            assert abs(d4["A"] - d.deltap4[0]["A"]) < tol

    # CC(P;Q): Q = complement of a P space, against ccpy's masked two-body corrections
    rng = np.random.default_rng(7)
    for frac in (0.3, 1.0):
        p3 = _random_pspace(("aaa", "aab", "abb", "bbb"), d.system, frac, rng)
        p4 = _random_pspace(("aaaa", "aaab", "aabb", "abbb", "bbbb"), d.system, frac, rng)
        _, ref3 = calc_ccp3_2ba(d.T, d.L[0], p3, d.correlation_energy, d.hamiltonian, d.fock, d.system)
        ref4 = calc_ccp4_with_selection(d.T, d.L[0], p4, d.fock, d.hamiltonian, d.system)[0][0]
        _, d3 = calc_crcc23_laplace(*args, npts=10, t3_excitations=p3)
        _, d4 = calc_crcc24_laplace(*args, npts=10, t4_excitations=p4)
        assert abs(d3["A"] - ref3["A"]) < 1.0e-8
        assert abs(d4["A"] - ref4) < 1.0e-8


def test_driver_methods():
    d = _driver("O 0 0 -0.018; H 0 3.030526 -2.117796; H 0 -3.030526 -2.117796", 0, scf.RHF)
    d.run_ccp3(method="crcc23")
    d.run_ccp4(method="crcc24")
    ref3, ref4 = d.deltap3[0]["A"], d.deltap4[0]["A"]
    d.run_ccp3(method="crcc23_laplace")
    d.run_ccp4(method="crcc24_laplace")
    assert abs(d.deltap3[0]["A"] - ref3) < 1.0e-8
    assert abs(d.deltap4[0]["A"] - ref4) < 1.0e-8

    rng = np.random.default_rng(3)
    p3 = _random_pspace(("aaa", "aab", "abb", "bbb"), d.system, 0.3, rng)
    p4 = _random_pspace(("aaaa", "aaab", "aabb", "abbb", "bbbb"), d.system, 0.3, rng)
    d.options["RHF_symmetry"] = False  # random P spaces are not alpha/beta symmetric
    d.run_ccp3(method="ccp3", state_index=0, t3_excitations=p3)
    ref3 = d.deltap3[0]["A"]
    ref4 = calc_ccp4_with_selection(d.T, d.L[0], p4, d.fock, d.hamiltonian, d.system)[0][0]
    d.run_ccp3(method="ccp3_laplace", t3_excitations=p3)
    d.run_ccp4(method="ccp4_laplace", t4_excitations=p4)
    assert abs(d.deltap3[0]["A"] - ref3) < 1.0e-8
    assert abs(d.deltap4[0]["A"] - ref4) < 1.0e-8


def test_networks_are_n6():
    d = _driver("O 0 0 0; H 0 0 1.8324", 1, scf.ROHF, basis="sto-3g")
    for spincase in (crcc_laplace.triples_aaa, crcc_laplace.triples_aab, crcc_laplace.triples_abb,
                     crcc_laplace.triples_bbb, crcc_laplace.quadruples_aaaa, crcc_laplace.quadruples_aaab,
                     crcc_laplace.quadruples_aabb, crcc_laplace.quadruples_abbb, crcc_laplace.quadruples_bbbb):
        worst, *_ = crcc_laplace.network_scaling(*spincase(d.T, d.L[0], d.hamiltonian))
        assert worst == 6
