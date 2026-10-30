"""CR-CC(2,3)_A geometry optimization with the analytic gradient (ccpy.optimize)."""
import numpy as np

from ccpy import Driver, Molecule, RHF
from ccpy.optimize import fd_energy_and_gradient, optimize_geometry


def _crcc23A_energy(mol):
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=1)
    driver.options.update({"energy_convergence": 1.0e-12, "amp_convergence": 1.0e-10})
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")
    energy = driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["A"]
    return energy


def _crcc23A_energy_and_gradient(mol):
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=1)
    driver.options.update({"energy_convergence": 1.0e-12, "amp_convergence": 1.0e-10})
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")
    energy = driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["A"]
    driver.run_gradient(method="crcc23a")
    gradient = driver.gradient
    return (energy, gradient)


def test_crcc23A_optimize():
    mol = Molecule("O 0 0 0.12; H 0 0.80 -0.47; H 0 -0.78 -0.45", "6-31g",
                   auxiliary_basis="def2-universal-jkfit", verbose=False)
    analytic = optimize_geometry(mol, _crcc23A_energy_and_gradient, thresh="gau_tight", verbose=False)
    assert analytic.converged
    X = np.array([xyz for _, xyz in analytic.molecule.atoms])


def test_crcc23A_minimum_analytic_vs_finite_difference_gradient():
    mol = Molecule("O 0 0 0.12; H 0 0.80 -0.47; H 0 -0.78 -0.45", "6-31g",
                   auxiliary_basis="def2-universal-jkfit", verbose=False)
    analytic = optimize_geometry(mol, _crcc23A_energy_and_gradient, thresh="gau_tight", verbose=False)
    numerical = optimize_geometry(mol, fd_energy_and_gradient(_crcc23A_energy), thresh="gau_tight", verbose=False)
    assert analytic.converged and numerical.converged
    assert np.isclose(analytic.energy, numerical.energy, atol=1.0e-09)
    X = np.array([xyz for _, xyz in analytic.molecule.atoms])
    Y = np.array([xyz for _, xyz in numerical.molecule.atoms])
    # same bond lengths and angle (the frames may differ by a rigid motion)
    d = lambda Z: [np.linalg.norm(Z[1] - Z[0]), np.linalg.norm(Z[2] - Z[0]), np.linalg.norm(Z[2] - Z[1])]
    assert np.allclose(d(X), d(Y), atol=1.0e-4)

if __name__ == "__main__":
    test_crcc23A_optimize()
