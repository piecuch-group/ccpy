"""CR-CC(2,4) calculation for the symmetrically stretched
H2O molecule with R(OH) = 2Re, where Re = 1.84345 bohr, described
using the Dunning DZ basis set.
Reference: Mol. Phys, 115, 2860 (2017)."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver

def test_crcc24_h2o():
    # 2 Re
    geometry = [["O", (0.0, 0.0, -0.0180)],
                ["H", (0.0, 3.030526, -2.117796)],
                ["H", (0.0, -3.030526, -2.117796)]]
    mol = gto.M(
        atom=geometry,
        basis="dz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")
    driver.run_ccp4(method="crcc24")

    # Check CCSD energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -75.8959141002, atol=1.0e-07)
    # Check CR-CC(2,3)_A energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["A"], -75.9047403064, atol=1.0e-07)
    # Check CR-CC(2,3)_D energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["D"], -75.9076522384)
    # Check CR-CC(2,4)_AA energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["A"] + driver.deltap4[0]["A"], -75.90617829971033)
    # Check CR-CC(2,4)_DA energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["D"] + driver.deltap4[0]["A"], -75.90909023171034)
    # Check CR-CC(2,4)_DD energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["D"] + driver.deltap4[0]["D"], -75.9094507817)

if __name__ == "__main__":
    test_crcc24_h2o()
