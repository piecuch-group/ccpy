"""Semi-stochastic CR-CC(2,3), CC(t;3), and CC(P;3) (ccpy.moments.stochastic_triples): run to
completion they reproduce the deterministic corrections; stopped early their statistical
errors are meaningful."""
import copy

import numpy as np
import pytest
from conftest import AUX, OPTS, converged_scf

from ccpy import CCP3, CCSDT_P, CCT3, CRCC23, AdaptiveCCPQ, Molecule, RHF, Reference, get_active_triples_pspace

EXHAUSTIVE = {"target_error": 0.0, "seed": 0, "nbuckets": 20, "batch_size": 8}


@pytest.fixture(scope="module")
def f2():
    mol = Molecule("F 0.0 0.0 -1.8; F 0.0 0.0 1.8", "6-31g", unit="bohr", auxiliary_basis=AUX, verbose=False)
    return converged_scf(RHF, mol)


@pytest.fixture(scope="module")
def f2_ccsdt_p(f2):
    ref = Reference(f2, nfrozen=1)
    ref.system.set_active_space(nact_occupied=3, nact_unoccupied=1)
    cc = CCSDT_P(f2, get_active_triples_pspace(ref.system), nfrozen=1, **OPTS)
    cc.energy()
    return cc


def assert_same_deltas(stochastic, deterministic):
    assert stochastic.stochastic_info["exact"]
    for v in "ABCD":
        assert np.isclose(stochastic.deltas[v], deterministic.deltas[v], atol=1.0e-12, rtol=0.0)
        assert stochastic.errors[v] == 0.0


@pytest.mark.parametrize("scf, rhf_symmetry", [("water_rhf", True), ("water_rhf", False), ("oh_rohf", None),
                                               ("oh_uhf", None)])
def test_crcc23_exhaustive(scf, rhf_symmetry, request):
    mf = request.getfixturevalue(scf)
    options = dict(OPTS) if rhf_symmetry is None else dict(OPTS, RHF_symmetry=rhf_symmetry)
    det = CRCC23(mf, nfrozen=1, **options)
    det.energy()
    sto = CRCC23(det.left, stochastic=EXHAUSTIVE)
    assert np.isclose(sto.energy(), det.energy(), atol=1.0e-12)
    assert_same_deltas(sto, det)


def test_cct3_and_ccp3_exhaustive(f2_ccsdt_p):
    cc = f2_ccsdt_p
    for det, sto in ((CCT3(cc, nacto=3, nactu=1), CCT3(cc, nacto=3, nactu=1, stochastic=EXHAUSTIVE)),
                     (CCP3(cc), CCP3(cc, stochastic=EXHAUSTIVE))):
        det.energy()
        sto.energy()
        assert_same_deltas(sto, det)


def test_crcc23_error_bars(water_rhf):
    det = CRCC23(water_rhf, nfrozen=1, **OPTS)
    det.energy()
    exact = det.deltas
    sto = CRCC23(det.left, stochastic={"target_error": 2.0e-5, "seed": 3, "nbuckets": 20, "batch_size": 2})
    sto.energy()
    info = sto.stochastic_info
    assert not info["exact"] and info["fraction"] < 1.0
    assert sto.errors["D"] < 2.0e-5
    for v in "ABCD":
        assert abs(sto.deltas[v] - exact[v]) < 5.0 * sto.errors[v]


def test_stochastic_options():
    with pytest.raises(ValueError):
        CCP3(None, two_body_approx=False, stochastic=True)


def test_adaptive_ccpq_stochastic_final_correction(f2):
    percentage = [0.0, 1.0]
    det = AdaptiveCCPQ(f2, nfrozen=1, percentage=percentage, **OPTS)
    sto = AdaptiveCCPQ(f2, nfrozen=1, percentage=percentage, stochastic=EXHAUSTIVE, **OPTS)
    assert np.isclose(sto.energy(), det.energy(), atol=1.0e-10)
    assert sto.ccp3.stochastic_info["exact"]


def test_adaptive_ccpq_stochastic_stopping(f2):
    # the selection is deterministic, so both runs build the same P spaces; the stochastic one
    # ends at the first macroiteration whose stochastic energy passes the stopping test
    det = AdaptiveCCPQ(f2, nfrozen=1, energy_tolerance=1.0e-4, **OPTS)
    sto = AdaptiveCCPQ(f2, nfrozen=1, energy_tolerance=1.0e-4, stochastic={"seed": 1, "nbuckets": 20}, **OPTS)
    det.energy()
    e = sto.energy()
    n = len(sto.ccpq_energies)
    assert det.adaptive_converged and sto.adaptive_converged
    assert np.allclose(sto.ccp_energies, det.ccp_energies[:n], atol=1.0e-9)
    assert np.allclose(sto.ccpq_energies[:-1], det.ccpq_energies[:n - 1], atol=1.0e-9)
    error = sto.ccpq_errors[-1]
    assert 0.0 < error <= sto.stochastic["target_error"]
    assert abs(e - det.ccpq_energies[n - 1]) < 5.0 * error
    assert abs(e - sto.ccpq_energies[-2]) + 1.96 * error < 1.0e-4


def test_adaptive_ccpq_stopping_test(f2):
    adapt = AdaptiveCCPQ(f2, nfrozen=1, energy_tolerance=1.0e-4, stochastic=True)
    assert np.isclose(adapt.stochastic["target_error"], 1.0e-4 / (2.0 * np.sqrt(2.0) * 1.96))
    adapt.ccpq_energies, adapt.ccpq_errors = [-1.0], [3.0e-5]
    # |dE| + 1.96 sqrt(3e-5^2 + 4e-5^2) = 1e-5 + 9.8e-5
    assert not adapt._converged(-1.0 + 1.0e-5, 4.0e-5)
    assert adapt._converged(-1.0 + 1.0e-6, 4.0e-5)
    with pytest.raises(ValueError):
        AdaptiveCCPQ(f2, nfrozen=1, energy_tolerance=1.0e-4, stochastic={"target_error": 4.0e-5})
    with pytest.raises(ValueError):
        AdaptiveCCPQ(f2, nfrozen=1, stochastic={"variant": "A"})
