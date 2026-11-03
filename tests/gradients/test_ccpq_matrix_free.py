"""Independent dense versus FCIpy-backed matrix-free oracle checks.

The optional connector is built with devtools/build_fcipy_oracle.py. Set
CCPY_FCIPY_EXTENSION to run these tests; ordinary dense installations skip them.
"""
import os
import numpy as np
import pytest

from ccpy.gradients.ccpq import DeterminantBackend, MatrixFreeBackend, CCPQGradient, spaces
from ccpy.gradients.ccpq.determinants import DeterminantSpace, spin_orbital_integrals, hamiltonian_matrix
from ccpy.gradients.ccpq.densities import transition_densities, spin_trace
from ccpy.gradients.ccpq.fcipy_adapter import FCIpyOperator
from ccpy.gradients.ccpq.matrix_free import davidson

requires_fcipy = pytest.mark.skipif(not os.environ.get("CCPY_FCIPY_EXTENSION"), reason="Optional FCIpy connector not configured")


def integrals(n, seed=51):
    rng = np.random.default_rng(seed)
    h = np.diag(np.linspace(-1.3, .75, n))
    a = rng.normal(size=(n, n)) * .01
    h += (a + a.T) / 2
    v = rng.normal(size=(5, n, n)) * .13
    v = (v + v.transpose(0, 2, 1)) / 2
    return h, np.einsum('kpq,krs->pqrs', v, v)


@requires_fcipy
@pytest.mark.parametrize("nfrozen", [0, 1])
@pytest.mark.parametrize("sparse_cache", [False, True])
def test_slater_actions_and_transition_densities(nfrozen, sparse_cache):
    s = DeterminantSpace(4, 2, 2, nfrozen)
    h, eri = integrals(4)
    H = FCIpyOperator(s, h, eri, sparse_cache=sparse_cache)
    hs, vs = spin_orbital_integrals(h, eri)
    dense = hamiltonian_matrix(s, hs, vs)
    rng = np.random.default_rng(68)
    left, right = rng.normal(size=(2, s.ndet))
    np.testing.assert_allclose(H(right), dense @ right, atol=1e-12)
    np.testing.assert_allclose(H(right + 1j * left), dense @ (right + 1j * left), atol=1e-12)
    actual = H.transition_density(left, right)
    expected = spin_trace(transition_densities(np.outer(left, right), s, s.one_body_transitions(), s.two_body_transitions()), s.norb)
    np.testing.assert_allclose(actual.D, expected.D, atol=1e-12)
    np.testing.assert_allclose(actual.G, expected.G, atol=1e-12)


@requires_fcipy
@pytest.mark.parametrize("denominator", ["full", "mp"])
@pytest.mark.parametrize("state", [0, 1, 2])
@pytest.mark.parametrize("nonclosed", [False, True])
def test_dense_comparison(denominator, state, nonclosed):
    s = DeterminantSpace(3, 1, 1)
    h, eri = integrals(3, seed=73 if nonclosed else 51)
    P = np.r_[spaces.ranks(s, (1,))[:1], spaces.ranks(s, (2,))] if nonclosed else spaces.ranks(s, (1,))
    Q = spaces.complement(s, P)
    args = (3, 1, 1, 0, P, Q, denominator)
    expected = DeterminantBackend(*args, state=state).run(h, eri)
    backend = MatrixFreeBackend(*args, state=state)
    actual = backend.run(h, eri)
    assert actual.energy_base == pytest.approx(expected.energy_base, abs=2e-10)
    assert actual.delta == pytest.approx(expected.delta, abs=2e-10)
    for a, b in ((actual.rho_base, expected.rho_base), (actual.rho_corr, expected.rho_corr)):
        np.testing.assert_allclose(a.D, b.D, atol=2e-9)
        np.testing.assert_allclose(a.G, b.G, atol=2e-9)
    e, d = backend.energies(h, eri)
    assert e == pytest.approx(actual.energy_base, abs=2e-10)
    assert d == pytest.approx(actual.delta, abs=2e-10)


@requires_fcipy
@pytest.mark.parametrize("state", [0, 1, 2])
def test_full_p_empty_q(state):
    h, eri = integrals(3)
    args = (3, 1, 1, 0, (1, 2), None, "full")
    expected = DeterminantBackend(*args, state=state).run(h, eri)
    actual = MatrixFreeBackend(*args, state=state).run(h, eri)
    assert actual.delta == 0.
    assert np.max(abs(actual.rho_corr.D)) == 0.
    np.testing.assert_allclose(actual.rho_base.D, expected.rho_base.D, atol=1e-9)


@requires_fcipy
@pytest.mark.parametrize("denominator", ["full", "mp"])
@pytest.mark.parametrize("state", [0, 1])
def test_molecular_selected_triples(denominator, state):
    from ccpy import Molecule
    mol = Molecule("H 0 0 0; H .1 0 1.5; H 0 .2 3.1; H .2 .1 4.8", "sto-3g", verbose=False)
    def selected(s):
        return spaces.union(spaces.ranks(s, (1, 2)), spaces.ranks(s, (3,))[::3])
    calc = CCPQGradient(mol, selected, (3, 4), denominator, state=state, backend="matrix_free")
    expected = CCPQGradient(mol, selected, (3, 4), denominator, state=state).gradient_components()
    for key, actual in calc.gradient_components().items():
        np.testing.assert_allclose(actual, expected[key], atol=2e-9)
    for key, result in calc.check_gradient(components=[(1, 2)]).items():
        assert result['max_error'] < 2e-7, (key, result)


def test_iterative_nonsymmetric_davidson():
    rng = np.random.default_rng(25)
    A = np.diag(np.arange(50, dtype=float)) + .025 * rng.normal(size=(50, 50))
    actual, right, _ = davidson(lambda x: A @ x, np.diag(A), nroots=2)
    values, left, _ = __import__('scipy').linalg.eig(A, left=True, right=True)
    expected = np.sort(values.real)[:2]
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    assert np.max(abs(A @ right - right * actual)) < 1e-9
    ev, l, _ = davidson(lambda x: A.T @ x, np.diag(A), target=actual[1])
    assert ev[0] == pytest.approx(actual[1], abs=1e-9)


@requires_fcipy
def test_backend_does_not_call_dense_determinant_routines(monkeypatch):
    from ccpy.gradients.ccpq import determinants, amplitudes
    def forbidden(*args, **kwargs):
        raise AssertionError('Dense determinant routine called')
    monkeypatch.setattr(determinants, 'hamiltonian_matrix', forbidden)
    monkeypatch.setattr(amplitudes, 'similarity_transform', forbidden)
    monkeypatch.setattr(determinants.Transitions, 'contract', forbidden)
    h, eri = integrals(3)
    stage = MatrixFreeBackend(3, 1, 1, 0, (1,), (2,), 'full', state=1).run(h, eri)
    assert stage.checks['constraint'] < 1e-9


@requires_fcipy
@pytest.mark.parametrize('state', [0, 1])
def test_frozen_core_molecular_gradient(state):
    from ccpy import Molecule
    mol = Molecule('Li 0 0 0; H .1 .2 1.6', 'sto-3g', verbose=False)
    calc = CCPQGradient(mol, (1,), (2,), 'mp', nfrozen=1, state=state, backend='matrix_free')
    for key, result in calc.check_gradient(components=[(1, 2)]).items():
        assert result['max_error'] < 2e-8, (key, result)


def test_davidson_includes_sector_missing_from_low_diagonal_guesses():
    """A low eigenstate can have large determinant diagonals in another block."""
    rng = np.random.default_rng(194)
    A = np.zeros((30, 30))
    A[:28, :28] = np.diag(np.linspace(1., 4., 28)) + .02*rng.normal(size=(28, 28))
    A[28:, 28:] = [[10., -9.8], [-9.8, 10.]]
    # A few lowest diagonal unit guesses have exactly zero overlap with the
    # 0.2-energy hidden state. Full-support seeds must discover that sector.
    values, vectors, _ = davidson(lambda x: A @ x, np.diag(A), nroots=2)
    expected = np.sort(np.linalg.eigvals(A).real)[:2]
    np.testing.assert_allclose(values, expected, atol=1e-9)
    assert np.max(abs(A @ vectors - vectors*values)) < 1e-9


@requires_fcipy
@pytest.mark.parametrize('state', [1, 2])
def test_guarded_reference_following_matches_ordered_root(state):
    h, eri = integrals(3, seed=51)
    args = (3, 1, 1, 0, (1,), (2,), 'full')
    backend = MatrixFreeBackend(*args, state=state, sparse_cache=True)
    reference = backend.run(h, eri)
    guard = backend.follow_reference(reference.data)
    assert guard['radius'] > 0
    perturbation = np.array([[0., .2, -.1], [.2, -.3, .15], [-.1, .15, .1]]) * 1e-5
    followed = backend.run(h+perturbation, eri)
    ordered = MatrixFreeBackend(*args, state=state, sparse_cache=True).run(h+perturbation, eri)
    assert followed.energy_base == pytest.approx(ordered.energy_base, abs=2e-10)
    assert followed.delta == pytest.approx(ordered.delta, abs=2e-10)
    for a, b in [(followed.rho_base, ordered.rho_base), (followed.rho_corr, ordered.rho_corr)]:
        np.testing.assert_allclose(a.D, b.D, atol=2e-9)
        np.testing.assert_allclose(a.G, b.G, atol=2e-9)
    shifted = backend.energies(h+10*np.eye(3), eri)
    assert shifted[0] == pytest.approx(reference.energy_base+20, abs=2e-10)
    assert shifted[1] == pytest.approx(reference.delta, abs=2e-10)
    with pytest.raises(RuntimeError, match='guarded reference'):
        backend.energies(h+10*np.diag([0.,1.,1.]), eri)
    from copy import copy
    degenerate = copy(reference.data)
    degenerate.eom_values = reference.data.eom_values.copy()
    degenerate.eom_values[1 if state == 1 else 0] = reference.energy_base
    with pytest.raises(ValueError, match='degenerate reference'):
        backend.follow_reference(degenerate)
    backend.clear_reference()
