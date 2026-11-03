"""Complete SD P spaces and complete complementary T/TQ correction spaces."""
import os
import numpy as np
import pytest
from ccpy import Molecule
from ccpy.gradients.ccpq import CCPQGradient, DeterminantBackend, MatrixFreeBackend, spaces
from ccpy.gradients.ccpq.determinants import DeterminantSpace, spin_orbital_integrals, hamiltonian_matrix, fock_diagonal
from ccpy.gradients.ccpq.amplitudes import similarity_transform
from ccpy.gradients.ccpq.densities import transition_densities, spin_trace
from ccpy.gradients.ccpq.fcipy_adapter import FCIpyOperator
from ccpy.gradients.ccpq.matrix_free import MatrixFreeState

requires_fcipy = pytest.mark.skipif(not os.environ.get('CCPY_FCIPY_EXTENSION'), reason='Optional FCIpy connector not configured')


def selections(s, quadruples=False):
    # Every single/double, then CAS-style high-rank selections.
    occupied = list(range(max(s.nfrozen, s.nalpha-2), s.nalpha))
    virtual = list(range(s.nalpha, min(s.norb, s.nalpha+3)))
    selected = spaces.active_excitations(s, (3,), occupied, virtual)
    P = spaces.union(spaces.ranks(s, (1, 2)), selected)
    if quadruples:
        P = spaces.union(P, spaces.active_excitations(s, (4,), occupied, virtual)[::2])
    Q = spaces.complement(s, P, (3, 4) if quadruples else (3,))
    assert np.all(np.isin(spaces.ranks(s, (1, 2)), P))
    np.testing.assert_array_equal(spaces.union(P, Q), spaces.ranks(s, (1, 2, 3, 4) if quadruples else (1, 2, 3)))
    assert not np.intersect1d(P, Q).size
    return P, Q


def test_h6_full_basis_selection_sizes():
    s = DeterminantSpace(12, 3, 3)
    for quad, counts, nq in [(False, [54, 945, 36, 0], 5964), (True, [54, 945, 36, 5], 22159)]:
        P, Q = selections(s, quad)
        assert [np.count_nonzero(s.rank[P] == r) for r in range(1, 5)] == counts
        assert len(Q) == nq


@requires_fcipy
def test_sparse_diagonal_algebra_against_independent_dense_transform():
    s = DeterminantSpace(4, 2, 2)
    rng = np.random.default_rng(71)
    h = np.diag([-1.6, -.8, .3, .8])
    L = rng.normal(size=(5, 4, 4)) * .1; L = (L + L.transpose(0, 2, 1))/2
    eri = np.einsum('kpq,krs->pqrs', L, L)
    H = FCIpyOperator(s, h, eri, sparse_cache=True)
    hs, vs = spin_orbital_integrals(h, eri)
    dense = hamiltonian_matrix(s, hs, vs)
    np.testing.assert_allclose(H.sparse.toarray(), dense, atol=1e-12)
    P = spaces.ranks(s, (1, 2, 3, 4))
    calc = MatrixFreeState(s, P, [], H, fock_diagonal(hs, vs, s.ref_occ), 0, 'full')
    t = rng.normal(size=len(P)) * .05
    calc.set_t(t)
    U, V, B = similarity_transform(dense, calc.T.toarray(), 4)
    np.testing.assert_allclose(calc.hbar_diagonal(), np.diag(B), atol=1e-12)
    weights = rng.normal(size=s.ndet)
    actual = calc.diagonal_density(weights)
    expected = spin_trace(transition_densities(U.T @ np.diag(weights) @ V.T, s,
                           s.one_body_transitions(), s.two_body_transitions()), s.norb)
    np.testing.assert_allclose(actual.D, expected.D, atol=1e-12)
    np.testing.assert_allclose(actual.G, expected.G, atol=1e-12)
    actual = calc.diagonal_t_gradient(weights)
    # Complex-step each amplitude, including all triples and the quadruple.
    for i in range(len(t)):
        delta = np.zeros(len(t), dtype=complex); delta[i] = 1j*1e-25
        _, _, trial = similarity_transform(dense, calc.exc.matrix(t+delta, s.ndet).toarray(), 4)
        assert actual[i] == pytest.approx(np.imag(weights @ np.diag(trial))/1e-25, abs=1e-12)
    left, right = rng.normal(size=(2, s.ndet))
    actual = H.transition_density(left, right)
    expected = spin_trace(transition_densities(np.outer(left, right), s,
                           s.one_body_transitions(), s.two_body_transitions()), s.norb)
    np.testing.assert_allclose(actual.D, expected.D, atol=1e-12)
    np.testing.assert_allclose(actual.G, expected.G, atol=1e-12)


@requires_fcipy
@pytest.mark.parametrize('quadruples', [False, True])
@pytest.mark.parametrize('state', [0, 1, 2])
@pytest.mark.parametrize('denominator', ['mp', 'full'])
def test_complete_sd_complement_molecular_gradient(quadruples, state, denominator):
    # H4/6-31G has a complete 784-determinant space, including T and Q.
    mol = Molecule('H 0 0 0; H .1 .03 1.0; H -.03 .07 2.1; H .06 -.04 3.0', '6-31g', verbose=False)
    s = DeterminantSpace(mol.nmo, 2, 2)
    P, Q = selections(s, quadruples)
    calc = CCPQGradient(mol, P, Q, denominator, state=state, backend='matrix_free', sparse_cache=True)
    checks = calc.check_gradient(components=[(1, 2), (2, 0)])
    assert calc.stages['amplitudes'].checks['constraint'] < 1e-9
    assert calc.stages['amplitudes'].checks['left_root_error'] < 1e-9
    assert len(Q) > 8
    for key, value in checks.items():
        assert value['max_error'] < 2e-8, (key, value)
    assert abs(calc.stages['amplitudes'].delta) > 1e-9


@requires_fcipy
def test_second_root_quadruples_against_dense_projected_eigensystem():
    """Pin the left-root mismatch exposed by full SD plus selected quadruples."""
    from scipy.linalg import eig
    from ccpy.gradients.ccpq.scf import run_rhf
    mol = Molecule('H 0 0 0; H .1 .03 1.0; H -.03 .07 2.1; H .06 -.04 3.0', '6-31g', verbose=False)
    h, eri = run_rhf(mol).mo_integrals()
    s = DeterminantSpace(mol.nmo, 2, 2)
    P, Q = selections(s, True)
    backend = MatrixFreeBackend(mol.nmo, 2, 2, 0, P, Q, 'full', state=2, sparse_cache=True)
    data = backend.run(h, eri, densities=False)
    H = FCIpyOperator(s, h, eri, sparse_cache=True)
    hs, vs = spin_orbital_integrals(h, eri)
    calc = MatrixFreeState(s, P, Q, H, fock_diagonal(hs, vs, s.ref_occ), 2, 'full', excitations=backend._exc)
    calc.set_t(data.t)
    full = np.r_[0, P]
    # Only the small projected EOM matrix is built for this independent check.
    projected = np.column_stack([calc.bar(calc.embed([1.], [k]))[full] for k in full])
    values, left, _ = eig(projected, left=True, right=True)
    ground = np.argmin(abs(values - data.E0))
    order = sorted([i for i in range(len(values)) if i != ground], key=lambda i: values[i].real)
    i = order[1]
    assert abs(values[i].imag) < 1e-10
    assert data.E_base == pytest.approx(values[i].real, abs=1e-9)
    expected_left = left[:, i].real
    expected_left /= expected_left @ data.r
    np.testing.assert_allclose(data.l, expected_left, atol=1e-8)
    assert data.checks['left_root_error'] < 1e-9
    assert data.checks['constraint'] < 1e-9


@pytest.mark.parametrize('nvirtual,expected_t,expected_q', [(6, 1660, 2745), (7, 2716, 5439)])
def test_substantial_h6_high_rank_selections(nvirtual, expected_t, expected_q):
    s = DeterminantSpace(12, 3, 3)
    selected = spaces.active_excitations(s, (3, 4), range(3), range(3, 3+nvirtual))
    P = spaces.union(spaces.ranks(s, (1, 2)), selected)
    Q = spaces.complement(s, P, (3, 4))
    assert np.sum(s.rank[P] == 3) == expected_t
    assert np.sum(s.rank[P] == 4) == expected_q
    assert np.sum(s.rank[Q] == 3) == 6000-expected_t
    assert np.sum(s.rank[Q] == 4) == 16200-expected_q
    np.testing.assert_array_equal(spaces.union(P, Q), spaces.ranks(s, (1, 2, 3, 4)))
    assert not np.intersect1d(P, Q).size


@requires_fcipy
@pytest.mark.parametrize('nvirtual', [4, 5])
@pytest.mark.parametrize('state', [0, 1, 2])
@pytest.mark.parametrize('denominator', ['mp', 'full'])
def test_large_fraction_tq_nuclear_gradients(nvirtual, state, denominator):
    mol = Molecule('H 0 0 0; H .1 .03 1.0; H -.03 .07 2.1; H .06 -.04 3.0', '6-31g', verbose=False)
    s = DeterminantSpace(mol.nmo, 2, 2)
    P = spaces.union(spaces.ranks(s, (1, 2)),
        spaces.active_excitations(s, (3, 4), range(2), range(2, 2+nvirtual)))
    Q = spaces.complement(s, P, (3, 4))
    assert np.sum(s.rank[P] == 3)/np.sum(s.rank == 3) > .25
    assert np.sum(s.rank[P] == 4)/np.sum(s.rank == 4) >= .16
    calc = CCPQGradient(mol, P, Q, denominator, state=state, backend='matrix_free', sparse_cache=True)
    for key, value in calc.check_gradient(components=[(1, 2), (2, 0)]).items():
        assert value['max_error'] < 2e-8, (key, value)


@requires_fcipy
def test_phase_alignment_preserves_selected_cc_eom_energies():
    from dataclasses import replace
    from ccpy.gradients.ccpq.scf import run_rhf, align_orbital_phases
    mol = Molecule('H 0 0 0; H .1 .03 1.0; H -.03 .07 2.1; H .06 -.04 3.0', 'sto-3g', verbose=False)
    reference = run_rhf(mol)
    mf = replace(reference, C=reference.C*np.array([-1., 1., -1., 1.])[None, :])
    raw = mf.mo_integrals()
    flips = align_orbital_phases(reference, mf)
    np.testing.assert_array_equal(flips, [-1., 1., -1., 1.])
    np.testing.assert_allclose(mf.C, reference.C, atol=1e-14)
    aligned = mf.mo_integrals()
    s = DeterminantSpace(4, 2, 2)
    P = spaces.union(spaces.ranks(s, (1, 2)), spaces.ranks(s, (3,))[::2])
    Q = spaces.complement(s, P, (3, 4))
    for state in (0, 1, 2):
        a = MatrixFreeBackend(4, 2, 2, 0, P, Q, 'full', state, sparse_cache=True).energies(*raw)
        b = MatrixFreeBackend(4, 2, 2, 0, P, Q, 'full', state, sparse_cache=True).energies(*aligned)
        np.testing.assert_allclose(a, b, atol=2e-10)
