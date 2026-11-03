"""gradient() / energy_and_gradient() of ccpy.methods: analytic gradients against the
legacy Driver, the finite-difference fallback, and geometry optimization of a method."""
import numpy as np
import pytest
from conftest import AUX, OPTS, converged_scf, fresh_driver, total

from ccpy import CCSD, CRCC23, EOMCCSD, MP2, Molecule, RHF, UHF

TIGHT = dict(OPTS, energy_convergence=1.0e-12, amp_convergence=1.0e-10)
DISTORTED_WATER = "O 0.0 0.05 0.1173; H 0.0 0.7572 -0.4692; H 0.1 -0.7572 -0.4692"


@pytest.fixture(scope="module")
def distorted_water():
    return converged_scf(RHF, Molecule(DISTORTED_WATER, "6-31g", auxiliary_basis=AUX, verbose=False))


@pytest.fixture(scope="module")
def reference_driver(distorted_water):
    driver = fresh_driver(distorted_water)
    driver.options.update(energy_convergence=1.0e-12, amp_convergence=1.0e-10)
    driver.run_cc("ccsd")
    driver.run_hbar("ccsd")
    driver.run_leftcc("left_ccsd")
    driver.run_ccp3("crcc23")
    return driver


def test_ccsd_analytic_gradient(distorted_water, reference_driver):
    cc = CCSD(distorted_water, nfrozen=1, **TIGHT)
    assert cc.has_analytic_gradient()
    energy, gradient = cc.energy_and_gradient()
    assert np.isclose(energy, total(reference_driver), atol=1.0e-10)
    assert np.allclose(gradient, reference_driver.run_gradient("ccsd"), atol=1.0e-10)
    assert np.allclose(gradient.sum(axis=0), 0.0, atol=1.0e-8)
    # left-CCSD reports the gradient of its CC
    assert np.allclose(cc.left.gradient(), gradient, atol=1.0e-12)


def test_crcc23A_analytic_gradient(distorted_water, reference_driver):
    cr = CRCC23(distorted_water, variant="A", nfrozen=1, **TIGHT)
    energy, gradient = cr.energy_and_gradient()
    assert np.isclose(energy, total(reference_driver) + reference_driver.deltap3[0]["A"], atol=1.0e-10)
    assert np.allclose(gradient, reference_driver.run_gradient("crcc23a"), atol=1.0e-10)
    assert not CRCC23(cr.cc, variant="D").has_analytic_gradient()


def test_uhf_analytic_gradients():
    # the spin-integrated densities and orbital response serve UHF references unchanged
    mol = Molecule("C 0.0 0.02 0.1; H 0.0 0.98 -0.35; H 0.05 -0.95 -0.40", "6-31g", auxiliary_basis=AUX,
                   multiplicity=3, verbose=False)
    mf = converged_scf(UHF, mol)
    for method in (CCSD(mf, nfrozen=1, **TIGHT), CRCC23(mf, variant="A", nfrozen=1, **TIGHT)):
        assert method.has_analytic_gradient()
        assert np.allclose(method.gradient(), method.fd_gradient(), atol=1.0e-7)


def test_finite_difference_fallback(distorted_water):
    cc = CCSD(distorted_water, nfrozen=1, **TIGHT)
    assert np.allclose(cc.fd_gradient(), cc.gradient(), atol=1.0e-8)
    mp2 = MP2(distorted_water, nfrozen=1)
    assert not mp2.has_analytic_gradient()
    assert mp2.gradient().shape == (3, 3)


def test_eom_root_gradient_by_finite_differences(distorted_water):
    eom = EOMCCSD(distorted_water, nfrozen=1, **OPTS)
    eom.initial_guess(method="cisd", multiplicity=1, nacto=2, nactu=2, roots_per_irrep={"A": 1})
    gradient = eom.gradient(root=0)
    assert gradient.shape == (3, 3)
    assert np.allclose(gradient.sum(axis=0), 0.0, atol=1.0e-6)


def test_optimize_geometry_with_a_method():
    pytest.importorskip("pysisyphus")
    from ccpy.optimize import optimize_geometry

    mol = Molecule("O 0 0 0.12; H 0 0.80 -0.47; H 0 -0.78 -0.45", "6-31g", auxiliary_basis=AUX, verbose=False)
    cc = CCSD(RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10), nfrozen=1, **TIGHT)
    result = optimize_geometry(cc, thresh="gau", verbose=False)
    assert result.converged
    final = CCSD(RHF(result.molecule, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10), nfrozen=1, **TIGHT)
    assert np.isclose(final.energy(), result.energy, atol=1.0e-9)
    assert np.abs(final.gradient()).max() < 1.0e-3
