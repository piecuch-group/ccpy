"""MPn calculations (n = 2,3,4) for the symmetrically stretched
H2O molecule with R(OH) = 2Re, where Re = 1.84345 bohr, described
using the spherical cc-pVDZ basis set.
Reference: J. Chem. Phys. 104, 8007 (1996)."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver

def test_mpn_h2o():
    # 2 Re
    geometry = [["O", (0.0, 0.0, -0.0180)],
                ["H", (0.0, 3.030526, -2.117796)],
                ["H", (0.0, -3.030526, -2.117796)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    # Check reference energy
    assert np.allclose(
        driver.system.reference_energy, -75.587711,
        atol=1.0e-07
    )
    driver.run_mbpt(method="mp2")
    # Check MP2 total energy
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -75.896935, atol=1.0e-07
    )
    driver.run_mbpt(method="mp3")
    # Check MP3 total energy
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -75.882569, atol=1.0e-07
    )
    # [TODO]: MP4 METHOD IS NOT WORKING YET
    driver.run_mbpt(method="mp4")
    # Check MP4 total energy
    #assert np.allclose(
    #    driver.system.reference_energy + driver.correlation_energy, -75.935619
    #)

if __name__ == "__main__":
    test_mpn_h2o()
