"""IR intensities from the finite-difference Hessian of analytic gradients: the atomic polar
tensor P = d mu / dx is the central difference of the orbital-relaxed dipoles that each
displaced gradient calculation returns (gradient_and_dipole, fd_hessian(dipole=True)).

P is checked against energies alone, P_{a, Ab} = Z_A delta_ab - d^2 E / d lambda_a d x_Ab by
mixed central differences (finite fields, SCF re-solved, at each displaced geometry; the same
step in x as fd_hessian, so only the field differences add error), and against the
translational sum rule sum_A P_{a, Ab} = q delta_ab, which the O(h^2) differences in x
(h = 2.5e-3 bohr) satisfy to ~1e-6; a wrong nuclear term or sign would miss by O(1). The
intensities are checked against the CO2 selection rules.

(Finite fields on the analytic nuclear gradient would need the derivative integrals of the
dipole operator, lambda D . d<m|r|n>/dx, which the gradient code does not contract.)
"""
import contextlib
import io

import numpy as np
import pytest
from conftest import AUX, OPTS, converged_scf
from test_property_gradients import _in_field

from ccpy import CCSD, CRCC23, Molecule, RHF, UHF
from ccpy.gradients.derivative_integrals import multipole_integrals
from ccpy.gradients.finite_difference import finite_difference
from ccpy.gradients.hessian import vibrational_analysis

TIGHT = dict(OPTS, energy_convergence=1.0e-13, amp_convergence=1.0e-12)
WATER = "O 0.0 0.05 0.1173; H 0.0 0.7572 -0.4692; H 0.1 -0.7572 -0.4692"
CO2 = "C 0.0 0.0 0.0; O 0.0 0.0 1.20; O 0.0 0.0 -1.20"


def _quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


def _method(kind, mol):
    if kind == "ccsd":
        return CCSD(converged_scf(RHF, mol), nfrozen=1, **TIGHT)
    return CRCC23(converged_scf(UHF, mol), variant="A", nfrozen=1, **TIGHT)


def _molecule(kind, geometry=WATER):
    charge, multiplicity = (0, 1) if kind == "ccsd" else (1, 2)
    return Molecule(geometry, "sto-3g", auxiliary_basis=AUX, charge=charge, multiplicity=multiplicity,
                    symmetry=False, verbose=False)


@pytest.fixture(scope="module", params=["ccsd", "crcc23"])
def water(request):
    kind = request.param
    mol = _molecule(kind)
    template = _method(kind, mol)
    res = _quiet(vibrational_analysis, mol, gradient=lambda m: template.at_geometry(m).gradient_and_dipole(),
                 dipole=True)
    return kind, mol, res


def test_gradient_and_dipole_match_separate_calculations(water):
    kind, mol, _ = water
    m = _method(kind, mol)
    grad, mu = _quiet(m.gradient_and_dipole)
    assert np.allclose(grad, _quiet(m.gradient), atol=1.0e-12)
    assert np.allclose(mu, list(_quiet(m.multipole_moments).values()), atol=1.0e-12)


def test_atomic_polar_tensor_sum_rule(water):
    _, mol, res = water
    P = res["atomic_polar_tensor"].reshape(3, mol.natoms, 3)
    assert np.allclose(P.sum(axis=1), mol.charge * np.eye(3), atol=1.0e-5)


def test_atomic_polar_tensor_vs_energies(water):
    kind, mol, res = water

    def dE_dlambda(X):
        V = multipole_integrals(mol.with_geometry(X, unit="bohr"), 1)[1]
        return finite_difference(
            lambda lam: _quiet(_method(kind, _in_field(mol.with_geometry(X, unit="bohr"), V, lam)).energy),
            np.zeros(3), step=1.0e-3, npoints=4)

    X0 = np.array([a[1] for a in mol.atoms], dtype=float)
    d2E = finite_difference(dE_dlambda, X0, step=2.5e-3, npoints=2)  # (natoms, 3, field)
    Z = np.array([a[0] for a in mol.atoms], dtype=float)
    P_E = np.einsum("A,ab->aAb", Z, np.eye(3)) - np.moveaxis(d2E, -1, 0)
    assert np.allclose(res["atomic_polar_tensor"], P_E.reshape(3, -1), atol=1.0e-7)
    assert np.all(res["ir_intensities"] > 0.0)


def test_co2_selection_rules():
    mol = _molecule("ccsd", CO2)
    template = _method("ccsd", mol)
    res = _quiet(vibrational_analysis, mol, gradient=lambda m: template.at_geometry(m).gradient_and_dipole(),
                 dipole=True)
    w, ir, modes = res["frequencies"], res["ir_intensities"], res["normal_modes"]
    assert len(w) == 4
    # symmetric stretch (sigma_g+: carbon at rest) is IR inactive; bends and asymmetric stretch are not
    symmetric = np.argmin(np.linalg.norm(modes[:, 0], axis=1))
    assert np.linalg.norm(modes[symmetric, 0]) < 1.0e-2
    assert ir[symmetric] < 1.0e-3
    assert np.all(np.delete(ir, symmetric) > 1.0)
    bends = np.argsort(w)[:2]
    assert np.isclose(ir[bends[0]], ir[bends[1]], rtol=1.0e-4)
