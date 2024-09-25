"""CCSD(T) computation for the stretched F2 molecule at at
interatomic distance of R = 2Re, where Re = 2.66816 bohr,
described using the cc-pVTZ basis set.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver

def test_crcc23_f2():
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

    driver = Driver.from_pyscf(mf, nfrozen=2, use_cholesky=True, cholesky_tol=1.0e-07, cmax=10)
    driver.run_cc(method="ccsd_chol")
    driver.run_hbar(method="ccsd_chol")
    driver.run_leftcc(method="left_ccsd_chol")
    driver.run_ccp3(method="crcc23_chol")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -198.48327030, rtol=1.0e-07)
    # Check CCSD energy
    assert np.allclose(driver.correlation_energy, -0.69225474, rtol=1.0e-07)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -199.17552504,
        rtol=1.0e-07
    )
    # Check CR-CC(2,3)D energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["D"], -199.2340905242, rtol=1.0e-07)

if __name__ == "__main__":
    test_crcc23_f2()
