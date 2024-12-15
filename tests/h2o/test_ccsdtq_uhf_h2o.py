"""Closed-shell CCSDTQ calculation for the symmetrically stretched
H2O molecule with R(OH) = 2Re, where Re = 1.84345 bohr, described
using the Dunning DZ basis set.
Reference: Mol. Phys, 115, 2860 (2017)."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver

def test_ccsdtq_h2o():
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
    mf = scf.UHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0, uhf=True)
    driver.system.print_info()

    # Important: Set RHF flag to false manually for UHF references
    driver.options["RHF_symmetry"] = False
    driver.run_cc(method="ccsdtq")

    # Check CCSDTQ correlation energy
    assert np.allclose(driver.correlation_energy, -0.30995754, atol=1.0e-07)
    # Check CCSDTQ total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -75.90513822, atol=1.0e-07)

if __name__ == "__main__":
    test_ccsdtq_h2o()
