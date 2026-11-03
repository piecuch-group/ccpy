"""Molecular regressions for higher-rank enlarged-model STEOM."""

import numpy as np
import pytest
from conftest import AUX, converged_scf

from ccpy import CCSD, Molecule, RHF, STEOMCC, STEOMCCSDT
from ccpy.eomcc.steomccsdt import SDConfigurations, active_triples


@pytest.fixture(scope="module")
def h2_cc():
    mol = Molecule("H 0 0 0; H 0 0 .74", "6-31g", auxiliary_basis=AUX, verbose=False)
    return CCSD(
        converged_scf(RHF, mol),
        amp_convergence=1.0e-9,
        energy_convergence=1.0e-10,
        maximum_iterations=100,
    ).solve()


@pytest.fixture(scope="module")
def lih_cc():
    mol = Molecule("Li 0 0 0; H 0 0 1.6", "sto-3g", auxiliary_basis=AUX, verbose=False)
    return CCSD(
        converged_scf(RHF, mol),
        amp_convergence=1.0e-8,
        energy_convergence=1.0e-10,
        maximum_iterations=100,
    ).solve()


def reference_eomccsd(method):
    from ccpy.eomcc.eomccsd import HR
    from ccpy.models.operators import ClusterOperator

    cfg = SDConfigurations(method.system, method.multiplicity)
    columns = []
    for column in cfg.basis.T.toarray():
        R = cfg.to_operator(column)
        dR = ClusterOperator(method.system, order=2)
        HR(dR, R, method.cc.T, method.hbar.H, False, method.system)
        columns.append(cfg.basis.T @ cfg.from_operator(dR))
    return np.sort(np.linalg.eigvals(np.array(columns).T).real)


@pytest.mark.parametrize("multiplicity", [1, 3])
def test_two_electron_full_neutral_limit(h2_cc, multiplicity):
    # For two electrons the complete neutral excitation space is SD. All
    # induced triples vanish by Pauli, so G must be similar to EE-EOMCCSD.
    method = STEOMCC(
        h2_cc, level="sdt", nroots=2, multiplicity=multiplicity, nacto=1, nactu=1
    ).solve()
    assert isinstance(method, STEOMCCSDT)
    assert method.converged
    np.testing.assert_allclose(method.omega, reference_eomccsd(method)[:2], atol=2.0e-8)
    assert len(method.ip.R) > h2_cc.system.noccupied_alpha
    assert len(method.ea.R) > h2_cc.system.nunoccupied_alpha
    assert method.action.maximum_induced_triples == 0
    assert np.max(method.residual_norms) < 1.0e-9
    version = method.version
    method.energy()
    assert method.version == version


def test_dense_and_full_pspace_auxiliaries(lih_cc):
    no, nv = lih_cc.system.noccupied_alpha, lih_cc.system.nunoccupied_alpha
    # Retain a small satellite model while making only the triples P spaces complete.
    ip = active_triples(no, nv, range(no), range(nv), "ip")
    ea = active_triples(no, nv, range(no), range(nv), "ea")
    old_ip = {s: v.copy() for s, v in ip.items()}
    old_t = lih_cc.T.flatten().copy()
    a = STEOMCCSDT(
        lih_cc, nroots=1, nacto=1, nactu=1, ip_pspace=ip, ea_pspace=ea
    ).solve()
    b = STEOMCCSDT(lih_cc, nroots=1, nacto=1, nactu=1, auxiliary_solver="dense").solve()
    assert a.converged and b.converged
    np.testing.assert_allclose(a.ip.omega, b.ip.omega, atol=2.0e-7)
    np.testing.assert_allclose(a.ea.omega, b.ea.omega, atol=2.0e-7)
    np.testing.assert_allclose(a.matrix, b.matrix, atol=2.0e-7)
    np.testing.assert_allclose(a.omega, b.omega, atol=2.0e-7)
    assert a.action.maximum_induced_triples > 0
    # At least one auxiliary is predominantly a satellite, and is retained.
    from ccpy.eomcc.steomccsdt import auxiliary_entries

    r1_weights = []
    for result, sector in [(a.ip, "ip"), (a.ea, "ea")]:
        for k, R in enumerate(result.R):
            exc = result.excitations[k]
            total = sum(
                abs(v) ** 2
                for rank in (1, 2, 3)
                for _, v in auxiliary_entries(R, no, nv, sector, rank, exc)
            )
            r1_weights.append(np.dot(R.a, R.a) / total)
    assert min(r1_weights) < 0.5
    for s in ip:
        np.testing.assert_array_equal(ip[s], old_ip[s])
    np.testing.assert_array_equal(lih_cc.T.flatten(), old_t)
    # Final eigenvectors must solve the unprojected Ms=0 equation as well.
    B = a.action.configurations.basis
    for energy, vector in zip(a.omega, a.working_vectors.T):
        np.testing.assert_allclose(
            a.action.raw_matvec(B @ vector), energy * (B @ vector), atol=1.0e-7
        )


def test_matrix_free_and_doubly_excited_roots(h2_cc):
    dense = STEOMCCSDT(h2_cc, nroots=2, nacto=1, nactu=1).solve()
    iterative = STEOMCCSDT(h2_cc, nroots=2, nacto=1, nactu=1, dense_limit=0).solve()
    assert iterative.matrix is None and iterative.converged
    np.testing.assert_allclose(iterative.omega, dense.omega, atol=2.0e-8)
    # Ask for more states than a CIS-sized problem can provide.
    extended = STEOMCCSDT(h2_cc, nroots=5, nacto=1, nactu=1, auxiliary_order=2).solve()
    assert (
        extended.nroots > h2_cc.system.noccupied_alpha * h2_cc.system.nunoccupied_alpha
    )
    np.testing.assert_allclose(
        extended.omega, reference_eomccsd(extended)[:5], atol=2.0e-8
    )
    assert min(extended.singles_weights) < 0.5


def test_invalid_extended_configuration(h2_cc):
    for kwargs in [
        dict(auxiliary_order=4),
        dict(auxiliary_solver="other"),
        dict(minimum_active=0),
        dict(dense_limit=-1),
        dict(auxiliary_order=2, ip_pspace={}),
    ]:
        with pytest.raises(ValueError):
            STEOMCCSDT(h2_cc, **kwargs)
    method = STEOMCCSDT(h2_cc, nacto=1, nactu=1, maximum_iterations=1)
    with pytest.raises(RuntimeError, match="auxiliaries did not converge"):
        method.solve()
    assert not method.converged


def test_automatic_space_and_geometry_reconstruction(h2_cc):
    method = STEOMCC(h2_cc, level="sdt", nroots=2).solve()
    assert method.automatic and method.converged
    np.testing.assert_allclose(method.omega, reference_eomccsd(method)[:2], atol=2.0e-8)
    replica = method.at_geometry(method.molecule)
    assert isinstance(replica, STEOMCCSDT) and replica.nroots == 2
    assert replica.auxiliary_order == 3 and replica.auxiliary_solver == "pspace"


def test_empty_auxiliary_triples_spaces(lih_cc):
    empty = STEOMCCSDT(lih_cc, nroots=1, nacto=1, nactu=1, minimum_active=3).solve()
    lower = STEOMCCSDT(lih_cc, nroots=1, nacto=1, nactu=1, auxiliary_order=2).solve()
    np.testing.assert_allclose(empty.ip.omega, lower.ip.omega, atol=2.0e-8)
    np.testing.assert_allclose(empty.ea.omega, lower.ea.omega, atol=2.0e-8)
    np.testing.assert_allclose(empty.matrix, lower.matrix, atol=2.0e-7)
    for result in (empty.ip, empty.ea):
        assert all(np.all(rows == 1) for rows in result.excitations[0].values())


@pytest.mark.parametrize(
    "multiplicity,method,root", [(1, "cisd", 6), (3, "cisd", 4), (1, "cis", 2)]
)
def test_targeted_neutral_guess_dense_and_iterative(h2_cc, multiplicity, method, root):
    settings = dict(method=method, nacto=1, nactu=3, roots=[root])
    dense = (
        STEOMCCSDT(h2_cc, multiplicity=multiplicity, nacto=1, nactu=1)
        .initial_guess(**settings)
        .solve()
    )
    iterative = (
        STEOMCCSDT(h2_cc, multiplicity=multiplicity, nacto=1, nactu=1, dense_limit=0)
        .initial_guess(**settings)
        .solve()
    )
    assert dense.converged and iterative.converged and iterative.matrix is None
    np.testing.assert_allclose(iterative.omega, dense.omega, atol=2.0e-8)
    cutoff = 0.9 if method == "cisd" else 0.5
    assert dense.guess_overlaps[0] > cutoff and iterative.guess_overlaps[0] > cutoff
    assert dense.omega[0] > reference_eomccsd(dense)[0] + 0.1
    if method == "cisd":
        assert dense.guess_singles_weights[0] < 0.5
        assert dense.singles_weights[0] < 0.5
    # Check the actual assignment in the physical SD metric, not C coordinates.
    values, vectors = np.linalg.eig(dense.matrix)
    eom = dense.metric @ (dense.action.configurations.basis @ vectors)
    eom /= np.linalg.norm(eom, axis=0)
    best = np.argmax(abs(dense._guess_targets.T @ eom)[0] ** 2)
    np.testing.assert_allclose(dense.omega[0], values[best].real, atol=2.0e-8)


def test_cisd_target_order_automatic_selection_and_cache(h2_cc):
    method = (
        STEOMCCSDT(h2_cc, nacto=1, nactu=1)
        .initial_guess("cisd", nacto=1, nactu=3, roots=[6, 0])
        .solve()
    )
    assert method.omega[0] > method.omega[1]
    assert np.all(method.guess_overlaps > 0.9)
    replica = method.at_geometry(method.molecule)
    assert replica.guess_settings == method.guess_settings
    assert replica.nroots == 2
    version = method.version
    method.initial_guess("cisd", nacto=1, nactu=3, roots=[0])
    assert not method.is_current()
    method.energy()
    assert method.version == version + 1
    automatic = (
        STEOMCCSDT(h2_cc).initial_guess("cisd", nacto=1, nactu=3, roots=[6]).solve()
    )
    np.testing.assert_allclose(
        automatic.omega[0], reference_eomccsd(automatic)[6], atol=2.0e-8
    )
    assert (
        automatic.guess_singles_weights[0] < 0.05 and automatic.guess_overlaps[0] > 0.9
    )


def test_invalid_neutral_guesses(h2_cc):
    for kwargs in [
        dict(method="other"),
        dict(method="cisd"),
        dict(roots=[-1]),
        dict(roots=[0, 0]),
        dict(roots_per_irrep={"A": 0}),
        dict(roots=[2], roots_per_irrep={"A": 1}),
    ]:
        with pytest.raises(ValueError):
            STEOMCCSDT(h2_cc).initial_guess(**kwargs)
    unavailable = STEOMCCSDT(h2_cc, nacto=1, nactu=1).initial_guess("cis", roots=[20])
    with pytest.raises(ValueError, match="unavailable"):
        unavailable.solve()
