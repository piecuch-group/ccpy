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
    mf = scf.UHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2, uhf=True)
    driver.system.print_info()
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")

    # Check that CCSD total energy is correct
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -199.0125625694, atol=1.0e-07)
    # Check that CR-CC(2,3)_D total energy is correct
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["D"], -199.0563392841, atol=1.0e-07)

if __name__ == "__main__":
    test_crcc23_f2()
