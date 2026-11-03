"""STEOM orchestration, orbital covariance, and failures on a small DF molecule."""

import numpy as np
import pytest
from conftest import AUX, OPTS, converged_scf

from ccpy import CCSD, HBarCCSDLegacy, Molecule, RHF, STEOMCC
from ccpy.eomcc.steomccsd import build_g, rotate_hbar, singles_matrix


@pytest.fixture(scope="module")
def lih_cc():
    mol = Molecule("Li 0 0 0; H 0 0 1.6", "sto-3g", auxiliary_basis=AUX, verbose=False)
    return CCSD(converged_scf(RHF, mol), maximum_iterations=100, **OPTS).solve()


@pytest.mark.parametrize("multiplicity", [1, 3])
def test_manual_and_rotated_full_active_space(lih_cc, multiplicity):
    no, nv = lih_cc.system.noccupied_alpha, lih_cc.system.nunoccupied_alpha
    method = STEOMCC(
        lih_cc,
        nroots=2,
        multiplicity=multiplicity,
        nacto=no,
        nactu=nv,
        ip_threshold=0.0,
        ea_threshold=0.0,
    ).solve()
    assert method.converged and all(method.root_converged)
    np.testing.assert_allclose(
        method.energy() - lih_cc.energy(), method.omega, atol=1.0e-12
    )
    np.testing.assert_allclose(method.active_weights, 1.0, atol=1.0e-12)
    assert method.matrix.shape == (no * nv, no * nv)
    from ccpy.eomcc import ipeom2, eaeom2
    from ccpy.models.operators import FockOperator

    for result, module, particles, holes in [
        (method.ip, ipeom2, 1, 2),
        (method.ea, eaeom2, 2, 1),
    ]:
        for R, omega in zip(result.R, result.omega):
            dR = FockOperator(method.system, particles, holes)
            residual = (
                module.HR(dR, R, lih_cc.T, method.hbar.H, False, method.system)
                - omega * R.flatten()
            )
            assert np.linalg.norm(residual) < 1.0e-9

    # A random rotation of both complete model spaces must leave the spectrum
    # unchanged, including the quadratic terms. No auxiliary reoptimization is
    # necessary when the complete model/eigenvector spaces are rotated together.
    rng = np.random.default_rng(60)
    uo = np.linalg.qr(rng.normal(size=(no, no)))[0]
    uv = np.linalg.qr(rng.normal(size=(nv, nv)))[0]
    H = method.hbar.H
    old_h = H.ab.voov.copy()
    old_t = lih_cc.T.flatten().copy()
    rotated, _ = rotate_hbar(H, lih_cc.T, uo, uv)
    sip = np.einsum(
        "mijb,mM,iI,jJ,bB->MIJB", method.S_ip, uo, uo, uo, uv, optimize=True
    )
    sea = np.einsum(
        "eabj,eE,aA,bB,jJ->EABJ", method.S_ea, uv, uv, uv, uo, optimize=True
    )
    G = build_g(rotated, sip, sea, np.arange(no), np.arange(nv))
    basis = np.kron(uv, uo)
    np.testing.assert_allclose(
        singles_matrix(G, multiplicity), basis.T @ method.matrix @ basis, atol=1.0e-11
    )
    np.testing.assert_array_equal(H.ab.voov, old_h)
    np.testing.assert_array_equal(lih_cc.T.flatten(), old_t)
    version = method.version
    method.energy()
    assert method.version == version


@pytest.mark.parametrize("multiplicity", [1, 3])
def test_automatic_selection(lih_cc, multiplicity):
    hbar = lih_cc.get_hbar(HBarCCSDLegacy)
    old_h, old_t = hbar.H.ab.voov.copy(), lih_cc.T.flatten().copy()
    method = STEOMCC(lih_cc, nroots=2, multiplicity=multiplicity).solve()
    assert method.converged
    assert np.all(
        method.ip.singles_character[method.ip.retained_roots] >= method.ip_threshold
    )
    assert np.all(
        method.ea.singles_character[method.ea.retained_roots] >= method.ea_threshold
    )
    assert all(
        0 < len(x) <= n
        for x, n in zip(
            (method.active_occupied, method.active_virtual),
            (lih_cc.system.noccupied_alpha, lih_cc.system.nunoccupied_alpha),
        )
    )
    for R in method.R:
        assert np.isclose(np.linalg.norm(R.flatten()), 1.0)
        np.testing.assert_allclose(R.b, (1 if multiplicity == 1 else -1) * R.a)
    np.testing.assert_array_equal(hbar.H.ab.voov, old_h)
    np.testing.assert_array_equal(lih_cc.T.flatten(), old_t)
    assert np.max(method.residual_norms) < 1.0e-10


def test_manual_label_order_and_cache_invalidation(lih_cc):
    a = STEOMCC(lih_cc, nroots=2, nacto=1, nactu=2).solve()
    b = STEOMCC(lih_cc, nroots=2, active_occupied=[1], active_virtual=[1, 0]).solve()
    np.testing.assert_allclose(a.omega, b.omega, atol=1.0e-10)
    # A new upstream solution invalidates the downstream cache.
    lih_cc.solve()
    assert not a.is_current()
    version = a.version
    a.energy()
    assert a.version == version + 1 and a.is_current()


@pytest.mark.parametrize(
    "settings",
    [
        dict(nroots=0),
        dict(multiplicity=2),
        dict(nacto=1),
        dict(active_occupied=[1]),
        dict(active_occupied=[1], active_virtual=[0, 0]),
        dict(nacto=1, nactu=1, active_virtual=[0]),
        dict(ip_threshold=1.1),
        dict(occupied_threshold=0),
        dict(condition_limit=0.5),
    ],
)
def test_invalid_settings(lih_cc, settings):
    with pytest.raises(ValueError):
        STEOMCC(lih_cc, **settings)


def test_unconverged_auxiliary_is_not_used(lih_cc):
    method = STEOMCC(lih_cc, nacto=1, nactu=1, maximum_iterations=1)
    with pytest.raises(RuntimeError, match="auxiliary calculation did not converge"):
        method.solve()
    assert not method.converged
    assert method.ip.root_converged == [False]


def test_empty_filtered_sector_is_not_used(lih_cc):
    method = STEOMCC(lih_cc, nacto=1, nactu=1, ip_threshold=1.0)
    with pytest.raises(ValueError, match="No IP roots"):
        method.solve()
    assert not method.converged


@pytest.mark.parametrize("guess_method", ["cis", "cisd"])
def test_configured_neutral_guess_for_singles_steom(lih_cc, guess_method):
    method = (
        STEOMCC(lih_cc, nacto=1, nactu=2)
        .initial_guess(guess_method, nacto=1, nactu=2, roots=[1])
        .solve()
    )
    R = method.guess_vectors[0]
    spatial = (R.a + R.b).reshape(-1)
    spatial /= np.linalg.norm(spatial)
    energy, vectors = np.linalg.eig(method.matrix)
    best = np.argmax(abs(spatial @ vectors) ** 2)
    np.testing.assert_allclose(method.omega[0], energy[best].real, atol=1.0e-10)
    assert method.converged and method.guess_overlaps[0] > 0.5
