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
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2, use_cholesky=True, cholesky_tol=1.0e-014)
    driver.options["RHF_symmetry"] = False
    driver.run_cc(method="ccsd_chol")
    driver.run_hbar(method="ccsd_chol")
    driver.run_leftcc(method="left_ccsd_chol")
    driver.run_ccp3(method="crcc23")

    print("Cholesky CR-CC(2,3)_A energy = ", driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["A"])
    print("CR-CC(2,3)_A energy shuld equal", -199.0518443133)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["A"], -199.0518443133, rtol=1.0e-012)

if __name__ == "__main__":
    test_crcc23_f2()
