import numpy as np
import pytest
from ccpy.gradients.ccpq import DeterminantBackend, StateOracle
from ccpy.gradients.ccpq.determinants import DeterminantSpace


@pytest.mark.parametrize("p,q", [([0], []), ([1, 1], []), ([-1], []), ([9], []),
                                  ([1.2], []), ([[1]], []), ([1], [1]), ([1], [2, 2])])
def test_invalid_spaces(p, q):
    with pytest.raises(ValueError):
        DeterminantBackend(3, 1, 1, 0, p, q, "full")
    with pytest.raises(ValueError):
        StateOracle(DeterminantSpace(3, 1, 1), p, q)


@pytest.mark.parametrize("state", [-1, 2, 0.5])
def test_invalid_state(state):
    with pytest.raises((ValueError, TypeError)):
        DeterminantBackend(3, 1, 1, 0, [1], [], "mp", state=state)


def test_reference_only():
    backend = DeterminantBackend(2, 1, 1, 0, [], [], "full")
    stage = backend.run(np.diag([-1., 1.]), np.zeros((2,) * 4))
    assert stage.energy_base == -2
    assert stage.delta == 0


def test_space_arrays_are_copied():
    p = np.array([1, 2]); q = np.array([3])
    backend = DeterminantBackend(3, 1, 1, 0, p, q, "full")
    p[:] = 0; q[:] = 0
    assert np.array_equal(backend.P, [1, 2])
    assert np.array_equal(backend.Q, [3])


def test_mp_directional_response():
    from ccpy.gradients.ccpq.determinants import spin_orbital_integrals, hamiltonian_matrix, fock_diagonal
    from ccpy.gradients.ccpq import spaces
    s = DeterminantSpace(3, 1, 1)
    rng = np.random.default_rng(61)
    h = np.diag([-1.2, .1, .55])
    factors = rng.normal(size=(4, 3, 3)) * .1
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    eri = np.einsum('kpq,krs->pqrs', factors, factors)
    hs, vs = spin_orbital_integrals(h, eri)
    H = hamiltonian_matrix(s, hs, vs)
    f = fock_diagonal(hs, vs, s.ref_occ)
    p = spaces.ranks(s, (1,)); q = spaces.complement(s, p)
    oracle = StateOracle(s, p, q, state=1, denominator="mp")
    result = oracle.solve(H, f)
    df = rng.normal(size=f.shape)
    eps = 1e-5
    fd = (oracle.solve(H, f + eps * df).E - oracle.solve(H, f - eps * df).E) / (2 * eps)
    assert oracle.directional_gradient(result, np.zeros_like(H), df) == pytest.approx(fd, abs=1e-8)
    with pytest.raises(ValueError, match="dfock_diag"):
        oracle.directional_gradient(result, np.zeros_like(H))
