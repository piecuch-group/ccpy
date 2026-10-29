"""CCSD(T) computation for the stretched F2 molecule at at
interatomic distance of R = 2Re, where Re = 2.66816 bohr,
described using the cc-pVTZ basis set.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver
from ccpy import Molecule, ROHF

def _run_and_check(driver):
    driver.run_cc(method="ccsd_chol")
    driver.run_hbar(method="ccsd_chol")
    driver.run_leftcc(method="left_ccsd_chol")
    driver.run_ccp3(method="crcc23_chol")

    # Check that CCSD total energy is correct
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -199.0007818156, rtol=1.0e-07)
    # Check that CR-CC(2,3)_D total energy is correct
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["D"], -199.0485115081, rtol=1.0e-07)


def test_crcc23_f2():
    geometry = [["F", (0.0, 0.0, -2.66816)],
                ["F", (0.0, 0.0, 2.66816)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvdz",
        charge=0,
        spin=2,
        symmetry="D2H",
        cart=False,
        unit="Bohr",
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2, use_cholesky=True, cholesky_tol=1.0e-07)
    _run_and_check(driver)


def test_crcc23_f2_native():
    # native ccpy SCF + tight Cholesky integrals
    mol = Molecule("F 0.0 0.0 -2.66816; F 0.0 0.0 2.66816", "cc-pvdz", unit="bohr", charge=0, multiplicity=3,
                   symmetry=True, cartesian=False, cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = ROHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=2, use_cholesky=True)
    _run_and_check(driver)


if __name__ == "__main__":
    test_crcc23_f2()
    test_crcc23_f2_native()
