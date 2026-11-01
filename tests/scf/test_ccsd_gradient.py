"""Analytic CCSD and CR-CC(2,3)_A gradients (factorized; density fitting or Cholesky) from the
native RHF and UHF."""
import numpy as np
import pytest

from ccpy import Driver
from ccpy.integrals import Molecule
from ccpy.gradients import ccsd_gradient, crcc23A_gradient
from ccpy.scf import RHF, UHF

H2O = "O 0.0 0.05 0.1173; H 0.0 0.7572 -0.4692; H 0.1 -0.7572 -0.4692"


def _ccsd(mol, nfrozen, gradient=False):
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=nfrozen)
    driver.options["energy_convergence"] = 1.0e-12
    driver.options["amp_convergence"] = 1.0e-10
    driver.run_cc(method="ccsd")
    energy = driver.system.reference_energy + driver.correlation_energy
    if not gradient:
        return energy
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    return energy, driver.run_gradient("ccsd")


@pytest.mark.parametrize("integrals", ["df", "cholesky"])
@pytest.mark.parametrize("nfrozen", [0, 1])
def test_ccsd_gradient_vs_finite_difference(integrals, nfrozen):
    if integrals == "df":
        mol = Molecule(H2O, "cc-pvdz", auxiliary_basis="cc-pvtz-rifit")
    else:
        # the Cholesky gradient is exact for fixed pivots: displaced geometries reuse them
        mol = Molecule(H2O, "cc-pvdz", cholesky_tei=True, cholesky_tol=1.0e-8)
        mol.cholesky_pivots = mol.df.pivots
    _, grad = _ccsd(mol, nfrozen, gradient=True)
    assert np.allclose(grad.sum(axis=0), 0.0, atol=1.0e-08)  # translational invariance

    X = np.array([atom[1] for atom in mol.atoms])
    h = 1.0e-4
    for atom, x in ((0, 1), (1, 2)):
        energies = []
        for step in (-h, h):
            Y = X.copy()
            Y[atom, x] += step
            energies.append(_ccsd(mol.with_geometry(Y), nfrozen))
        assert np.isclose(grad[atom, x], (energies[1] - energies[0]) / (2 * h), atol=1.0e-07)


def _uhf_ccsd_crcc23(mol, nfrozen, gradient=False):
    mf = UHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=nfrozen)
    driver.options["energy_convergence"] = 1.0e-12
    driver.options["amp_convergence"] = 1.0e-10
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")
    e_ccsd = driver.system.reference_energy + driver.correlation_energy
    energies = np.array([e_ccsd, e_ccsd + driver.deltap3[0]["A"]])
    if not gradient:
        return energies
    return energies, ccsd_gradient(driver, verbose=False), crcc23A_gradient(driver, verbose=False)


@pytest.mark.parametrize("nfrozen", [0, 1])
def test_uhf_gradients_vs_finite_difference(nfrozen):
    # triplet CH2: alpha and beta orbitals, densities and orbital responses all differ
    mol = Molecule("C 0.0 0.02 0.1; H 0.0 0.98 -0.35; H 0.05 -0.95 -0.40", "cc-pvdz",
                   auxiliary_basis="cc-pvtz-rifit", multiplicity=3)
    _, g_ccsd, g_crcc = _uhf_ccsd_crcc23(mol, nfrozen, gradient=True)
    for grad in (g_ccsd, g_crcc):
        assert np.allclose(grad.sum(axis=0), 0.0, atol=1.0e-08)

    X = np.array([atom[1] for atom in mol.atoms])
    h = 1.0e-4
    for atom, x in ((0, 2), (2, 0)):
        energies = []
        for step in (-h, h):
            Y = X.copy()
            Y[atom, x] += step
            energies.append(_uhf_ccsd_crcc23(mol.with_geometry(Y), nfrozen))
        fd = (energies[1] - energies[0]) / (2 * h)
        assert np.isclose(g_ccsd[atom, x], fd[0], atol=1.0e-08)
        assert np.isclose(g_crcc[atom, x], fd[1], atol=1.0e-08)


def test_ccsd_gradient_with_symmetry():
    # C2v water: the gradient is unchanged by symmetry blocking of the SCF and has no
    # component perpendicular to the molecular plane
    geometry = "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692"
    grads = []
    for symmetry in (False, True):
        mol = Molecule(geometry, "cc-pvdz", auxiliary_basis="cc-pvtz-rifit", symmetry=symmetry)
        _, grad = _ccsd(mol, 1, gradient=True)
        grads.append(np.linalg.norm(grad, axis=1))
    assert np.allclose(grads[0], grads[1], atol=1.0e-08)


def test_ccsd_gradient_against_pyscf():
    # tight Cholesky vectors reproduce the exact integrals, hence PySCF's conventional gradient
    pyscf = pytest.importorskip("pyscf")
    from pyscf import cc, gto, scf

    pmol = gto.M(atom=H2O, basis="cc-pvdz", verbose=0)
    pmf = scf.RHF(pmol)
    pmf.conv_tol = 1.0e-12
    pmf.kernel()
    pcc = cc.CCSD(pmf, frozen=1)
    pcc.conv_tol = 1.0e-12
    pcc.conv_tol_normt = 1.0e-10
    pcc.kernel()
    ref = pcc.nuc_grad_method().kernel()

    mol = Molecule(H2O, "cc-pvdz", cholesky_tei=True, cholesky_tol=1.0e-11)
    energy, grad = _ccsd(mol, 1, gradient=True)
    assert np.isclose(energy, pcc.e_tot, atol=1.0e-09)
    assert np.allclose(grad, ref, atol=1.0e-06)
