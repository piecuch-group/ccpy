"""ccpy.gradients.finite_difference against analytic gradients."""
import numpy as np

from ccpy import Driver
from ccpy.gradients import fd_gradient, finite_difference
from ccpy.integrals import Molecule
from ccpy.scf import RHF

H2O = "O 0.0 0.05 0.1173; H 0.0 0.7572 -0.4692; H 0.1 -0.7572 -0.4692"


def test_finite_difference_polynomial():
    # the 4-point stencil is exact for polynomials up to degree 4
    f = lambda x: x[0] ** 4 - 3.0 * x[0] * x[1] ** 2
    d = finite_difference(f, np.array([0.7, -1.3]), step=0.1)
    assert np.allclose(d, [4 * 0.7**3 - 3 * 1.3**2, -6 * 0.7 * -1.3], atol=1.0e-10)


def test_fd_gradient_rhf_and_ccsd():
    mol = Molecule(H2O, "cc-pvdz", auxiliary_basis="def2-universal-jkfit")

    def rhf_energy(m):
        return RHF(m, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10).kernel()

    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    assert np.allclose(fd_gradient(rhf_energy, mol, verbose=False), mf.gradient(), atol=1.0e-08)

    def ccsd_energy(m, gradient=False):
        mf = RHF(m, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
        mf.kernel()
        driver = Driver.from_scf(mf, nfrozen=1)
        driver.options["energy_convergence"] = 1.0e-12
        driver.options["amp_convergence"] = 1.0e-10
        driver.run_cc(method="ccsd")
        if gradient:
            driver.run_hbar(method="ccsd")
            driver.run_leftcc(method="left_ccsd")
            return driver.run_gradient()
        return driver.system.reference_energy + driver.correlation_energy

    grad_fd = fd_gradient(ccsd_energy, mol, atoms=[1], verbose=False)
    assert np.allclose(grad_fd[1], ccsd_energy(mol, gradient=True)[1], atol=1.0e-08)
