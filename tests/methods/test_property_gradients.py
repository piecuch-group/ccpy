"""First-order one-electron properties from the analytic-gradient densities: dE/dlambda for
h -> h + sum_k lambda_k V_k (gradient(perturbation=V)) against finite fields, with V the
electronic dipole and one quadrupole component (libint2 emultipole2), for CCSD and
CR-CC(2,3)_A (RHF and UHF, frozen core) and EOMCCSD and delta-CR-EOMCC(2,3)_A (two roots).

Each finite-field point re-solves the SCF in the perturbed core Hamiltonian, so the orbital
response (Z-vector) is part of what is checked.
"""
import contextlib
import io

import numpy as np
import pytest
from conftest import AUX, OPTS, converged_scf

from ccpy import CCSD, CREOMCC23, CRCC23, EOMCCSD, Molecule, RHF, UHF
from ccpy.gradients.derivative_integrals import multipole_integrals
from ccpy.gradients.finite_difference import finite_difference

TIGHT = dict(OPTS, energy_convergence=1.0e-13, amp_convergence=1.0e-12)
WATER = "O 0.0 0.05 0.1173; H 0.0 0.7572 -0.4692; H 0.1 -0.7572 -0.4692"
NH3 = "N 0.0 0.0 0.12; H 0.0 0.94 -0.27; H 0.81 -0.47 -0.29; H -0.80 -0.45 -0.25"
ORIGIN = (0.1, -0.2, 0.3)


def _quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


def _operators(mol):
    """x, y, z and xy about ORIGIN."""
    labels, V = multipole_integrals(mol, order=2, origin=ORIGIN)
    return V[[labels.index(c) for c in ("x", "y", "z", "xy")]]


def _in_field(mol, V, lam):
    """The molecule with core Hamiltonian h + sum_k lam_k V_k."""
    h = mol.ints_hcore() + np.einsum("k,kmn->mn", lam, V)
    mol.ints_hcore = lambda: h
    return mol


# ---------------------------------------------------------------------- ground state
def _ground(kind, method, lam=None):
    if kind == "rhf":
        mol, scf = Molecule(WATER, "6-31g", auxiliary_basis=AUX, verbose=False), RHF
    else:
        mol, scf = Molecule(WATER, "6-31g", auxiliary_basis=AUX, charge=1, multiplicity=2, verbose=False), UHF
    V = _operators(mol)
    if lam is not None:
        mol = _in_field(mol, V, lam)
    mf = converged_scf(scf, mol)
    m = CCSD(mf, nfrozen=1, **TIGHT) if method == "ccsd" else CRCC23(mf, variant="A", nfrozen=1, **TIGHT)
    return m, V


@pytest.mark.parametrize("method", ["ccsd", "crcc23"])
@pytest.mark.parametrize("kind", ["rhf", "uhf"])
def test_ground_state_property_vs_finite_field(kind, method):
    m, V = _ground(kind, method)
    dE = _quiet(m.gradient, perturbation=V)
    fd = finite_difference(lambda lam: _quiet(_ground(kind, method, lam)[0].energy), np.zeros(len(V)),
                           step=1.0e-3, npoints=4)
    assert np.allclose(dE, fd, atol=1.0e-9)


def test_multipole_moments_add_nuclear_part():
    m, _ = _ground("rhf", "ccsd")
    mol = m.molecule
    labels, V = multipole_integrals(mol, order=2, origin=ORIGIN)
    dE = _quiet(m.gradient, perturbation=V)
    mu = _quiet(m.multipole_moments, order=2, origin=ORIGIN)
    Z = np.array([a[0] for a in mol.atoms], dtype=float)
    R = np.array([a[1] for a in mol.atoms], dtype=float) - np.array(ORIGIN)
    assert list(mu) == list(labels)
    assert np.isclose(mu["y"], Z @ R[:, 1] - dE[labels.index("y")], atol=1.0e-12)
    assert np.isclose(mu["xz"], Z @ (R[:, 0] * R[:, 2]) - dE[labels.index("xz")], atol=1.0e-12)
    # the dipole of the neutral molecule does not depend on the origin
    mu0 = _quiet(m.multipole_moments, order=1)
    assert np.allclose([mu0[c] for c in "xyz"], [mu[c] for c in "xyz"], atol=1.0e-10)


# ---------------------------------------------------------------------- excited states
def _excited(lam=None):
    mol = Molecule(NH3, "sto-3g", auxiliary_basis=AUX, verbose=False)
    V = _operators(mol)
    if lam is not None:
        mol = _in_field(mol, V, lam)
    eom = EOMCCSD(converged_scf(RHF, mol), nfrozen=1, **TIGHT)
    eom.initial_guess(method="cisd", multiplicity=1, nacto=2, nactu=2, roots_per_irrep={"A": 2})
    m = CREOMCC23(eom, variant="A")
    _quiet(m.ensure_solved)
    return eom, m, V


@pytest.fixture(scope="module")
def nh3_field():
    eom, m, V = _excited()

    def energies(lam):
        e, mm, _ = _excited(lam)
        return np.array([[e.energy()[k], mm.energy()[k]] for k in range(2)])

    return eom, m, V, finite_difference(energies, np.zeros(len(V)), step=1.0e-3, npoints=4)  # (k, root, method)


@pytest.mark.parametrize("root", [0, 1])
def test_excited_state_property_vs_finite_field(nh3_field, root):
    eom, m, V, fd = nh3_field
    assert np.allclose(_quiet(eom.gradient, root=root, perturbation=V), fd[:, root, 0], atol=1.0e-9)
    assert np.allclose(_quiet(m.gradient, root=root, perturbation=V), fd[:, root, 1], atol=1.0e-9)
