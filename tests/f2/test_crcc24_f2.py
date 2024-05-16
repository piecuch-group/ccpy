"""CR-CC(2,4) computation for the stretched F2 molecule at
interatomic separation R = 2.0Re, where Re = 2.66816 bohr,
described using the cc-pVDZ basis set with Cartesian
components used for the d orbitals.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""

import numpy as np
from pyscf import gto, scf
from ccpy.drivers.driver import Driver

def test_crcc24_f2():
    geometry = [["F", (0.0, 0.0, -2.66816)], ["F", (0.0, 0.0, 2.66816)]]
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

    driver = Driver.from_pyscf(mf, nfrozen=2)
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")
    driver.run_ccp4(method="crcc24")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -198.420096282673, atol=1.0e-07)
    # Check CCSD energy
    assert np.allclose(driver.correlation_energy, -0.59246629, atol=1.0e-07)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -199.01256257, atol=1.0e-07)
    # Check CR-CC(2,3)_D energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["D"], -199.0563392841, atol=1.0e-07)
    # Check CR-CC(2,4)_AA energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["A"] + driver.deltap4[0]["A"], -199.05642649630002, atol=1.0e-07)
    # Check CR-CC(2,4)_DA energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["D"] + driver.deltap4[0]["A"], -199.0609214773, atol=1.0e-07)
    # Check CR-CC(2,4)_DD energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["D"] + driver.deltap4[0]["D"], -199.06133761, atol=1.0e-07)

if __name__ == "__main__":
    test_crcc24_f2()
