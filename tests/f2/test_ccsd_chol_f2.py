"""CCSD computation for the stretched F2 molecule at at
interatomic distance of R = 2Re, where Re = 2.66816 bohr,
described using the cc-pVTZ basis set.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver
from ccpy import Molecule, RHF

def _run_and_check(driver):
    driver.system.print_info()
    driver.options["RHF_symmetry"] = False
    driver.run_cc(method="ccsd_chol")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -198.48327030, atol=1.0e-07)
    # Check CCSD energy
    assert np.allclose(driver.correlation_energy, -0.69225474, atol=1.0e-07)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -199.17552504,
        atol=1.0e-07
    )


def test_ccsd_f2():
    geometry = [["F", (0.0, 0.0, -2.66816)],
                ["F", (0.0, 0.0, 2.66816)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvtz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2, use_cholesky=True, cholesky_tol=1.0e-07)
    _run_and_check(driver)


def test_ccsd_f2_native():
    # native ccpy SCF + tight Cholesky integrals
    mol = Molecule("F 0.0 0.0 -2.66816; F 0.0 0.0 2.66816", "cc-pvtz", unit="bohr", charge=0, multiplicity=1,
                   symmetry=True, cartesian=False, cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=2, use_cholesky=True)
    _run_and_check(driver)


if __name__ == "__main__":
    test_ccsd_f2()
    test_ccsd_f2_native()
