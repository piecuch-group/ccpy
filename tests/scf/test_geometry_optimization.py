"""Minimum and transition-state searches through pysisyphus (ccpy.optimize)."""
import numpy as np

from ccpy import Molecule, RHF
from ccpy.optimize import fd_energy_and_gradient, optimize_geometry, optimize_transition_state


def _hf_energy(mol):
    return RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10).kernel()


def _hf_energy_and_gradient(mol):
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    return mf.kernel(), mf.gradient()


def test_minimum_analytic_vs_finite_difference_gradient():
    mol = Molecule("O 0 0 0.12; H 0 0.80 -0.47; H 0 -0.78 -0.45", "cc-pvdz",
                   auxiliary_basis="cc-pvtz-jkfit", verbose=False)
    analytic = optimize_geometry(mol, _hf_energy_and_gradient, verbose=False)
    numerical = optimize_geometry(mol, fd_energy_and_gradient(_hf_energy), verbose=False)
    assert analytic.converged and numerical.converged
    assert np.isclose(analytic.energy, numerical.energy, atol=1.0e-08)
    assert np.abs(_hf_energy_and_gradient(analytic.molecule)[1]).max() < 5.0e-4


def test_nh3_inversion_transition_state():
    nh3 = Molecule("N 0 0 0.10; H 0.95 0 0; H -0.475 0.8227 0; H -0.475 -0.8227 0", "cc-pvdz",
                   auxiliary_basis="cc-pvtz-jkfit", verbose=False)
    ts = optimize_transition_state(nh3, _hf_energy_and_gradient, verbose=False)
    assert ts.converged
    # exactly one imaginary mode beyond the (near-zero) translations and rotations
    assert np.count_nonzero(ts.hessian_eigenvalues < -1.0e-3) == 1
    # the inversion saddle point is planar
    X = np.array([xyz for _, xyz in ts.molecule.atoms])
    normal = np.cross(X[2] - X[1], X[3] - X[1])
    assert abs(np.dot(X[0] - X[1], normal / np.linalg.norm(normal))) < 1.0e-2
    minimum = optimize_geometry(nh3, _hf_energy_and_gradient, verbose=False)
    assert ts.energy > minimum.energy
