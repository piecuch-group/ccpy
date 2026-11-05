"""SU-MRCC effective Hamiltonian Heff_pq = <Phi_p|H exp(T(q))|Phi_q> against the
determinant space: the model-space CI matrix at T = 0, and a brute-force exp(T(q)) (with the
C conditions imposed independently) otherwise, for complete and general model spaces."""
import numpy as np
import pytest

from ccpy.gradients.ccpq.determinants import DeterminantSpace, spin_orbital_integrals, hamiltonian_matrix
from ccpy.models.operators import ClusterOperator
from ccpy.models.system import System
from ccpy.mrcc.sumrcc import get_cas_determinants, get_model_space, get_reference_hamiltonian, \
    get_effective_hamiltonian
from test_sumrcc_projection import random_cluster_operator, determinant, cc_wavefunction

NFROZEN = 1
E_NUC = 0.7


def random_integrals(norb, rng):
    h = rng.standard_normal((norb, norb))
    h = 0.5 * (h + h.T)
    # (pq|rs) from random symmetric DF vectors: 8-fold symmetric and positive semidefinite, as
    # the DF reference Hamiltonians require
    B = 0.2 * rng.standard_normal((2 * norb, norb, norb))
    B = 0.5 * (B + B.transpose(0, 2, 1))
    return h, np.einsum("xpq,xrs->pqrs", B, B)


def setup(ncore, nact, nel, nvirt, seed, remove=0):
    """CAS(nel, nact) above ncore correlated core and NFROZEN frozen orbitals, with `remove`
    randomly chosen determinants (never the first) dropped."""
    rng = np.random.default_rng(seed)
    norb = NFROZEN + ncore + nact + nvirt
    nelectrons = 2 * (NFROZEN + ncore) + nel
    h, eri = random_integrals(norb, rng)
    system = System(nelectrons, norb, 1, NFROZEN, nuclear_repulsion=E_NUC)
    dets = get_cas_determinants(system, nel, nact)
    drop = set(rng.choice(range(1, len(dets)), remove, replace=False).tolist()) if remove else set()
    ms = get_model_space(system, [d for k, d in enumerate(dets) if k not in drop])
    H, e_ref = zip(*[get_reference_hamiltonian(h, eri.transpose(0, 2, 1, 3), system, ref) for ref in ms.references])
    space = DeterminantSpace(norb, nelectrons // 2, nelectrons // 2, NFROZEN)
    Hmat = hamiltonian_matrix(space, *spin_orbital_integrals(h, eri)) + E_NUC * np.eye(space.ndet)
    return rng, system, ms, H, e_ref, space, Hmat


@pytest.mark.parametrize("ncore, nact, nel, nvirt, remove", [(1, 2, 2, 2, 0), (1, 4, 4, 2, 0), (1, 6, 6, 1, 0),
                                                             (1, 4, 4, 2, 9)])
def test_heff_at_zero_t_is_model_space_ci(ncore, nact, nel, nvirt, remove):
    rng, system, ms, H, e_ref, space, Hmat = setup(ncore, nact, nel, nvirt, 0, remove)
    T = [ClusterOperator(system, 2) for _ in ms.references]
    heff = get_effective_hamiltonian(T, H, e_ref, ms, system)
    idx = [determinant(space, ref, NFROZEN) for ref in ms.references]
    assert np.allclose(heff, Hmat[np.ix_(idx, idx)], atol=1e-12)


@pytest.mark.parametrize("ncore, nact, nel, nvirt, remove", [(1, 2, 2, 2, 0), (1, 4, 4, 2, 0), (1, 2, 2, 2, 1),
                                                             (1, 4, 4, 2, 9), (1, 4, 4, 2, 30)])
def test_heff_matches_determinant_space(ncore, nact, nel, nvirt, remove):
    rng, system, ms, H, e_ref, space, Hmat = setup(ncore, nact, nel, nvirt, 1, remove)
    T = [random_cluster_operator(system, ms, q, rng) for q in range(len(ms.references))]
    heff = get_effective_hamiltonian(T, H, e_ref, ms, system)
    idx = [determinant(space, ref, NFROZEN) for ref in ms.references]
    ref = np.array([Hmat[idx] @ cc_wavefunction(space, ms, q, T[q], NFROZEN) for q in range(len(T))]).T
    assert np.allclose(heff, ref, atol=1e-10), np.abs(heff - ref).max()
