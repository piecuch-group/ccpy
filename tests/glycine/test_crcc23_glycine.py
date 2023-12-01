"""CR-CC(2,3) computation for the glycine molecule.
(This example was taken from the CC tests in GAMESS)."""

import numpy as np
from pyscf import gto, scf
from ccpy.drivers.driver import Driver

def test_crcc23_glycine():
    geometry = [
        ["O", (-2.877091949897, -1.507375565672, -0.398996049903)],
        ["C", (-0.999392972049, -0.222326510867, 0.093940021615)],
        ["C", (1.633098051399, -1.126399113321, 0.723677865007)],
        ["O", (-1.316707936360, 2.330484008081, 0.195537896270)],
        ["N", (3.588772131647, 0.190046035276, -0.635572324857)],
        ["H", (1.738434758147, -3.192291478262, 0.201142047999)],
        ["H", (1.805107822402, -0.972547254301, 2.850386782716)],
        ["H", (3.367427816470, 2.065392438845, -0.521139962778)],
        ["H", (5.288732713155, -0.301105855518, 0.028508872837)],
        ["H", (-3.050135067115, 2.755707159769, -0.234244183166)],
    ]

    mol = gto.M(
        atom=geometry,
        basis="6-31g**",
        charge=0,
        spin=0,
        symmetry="C1",
        unit="Bohr",
        cart=True,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=5)
    driver.system.print_info()

    driver.options["RHF_symmetry"] = True
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -282.8324072998, atol=1.0e-07)
    # Check CCSD energy
    assert np.allclose(driver.correlation_energy, -0.8319770162, atol=1.0e-07)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -283.6643843160,
        atol=1.0e-07
    )
    # Check CR-CC(2,3)_A energy
    assert np.allclose(
        driver.correlation_energy + driver.deltap3[0]["A"], -0.8546012006, atol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["A"],
        -283.6870085004,
        atol=1.0e-07
    )
    # Check CR-CC(2,3)_D energy
    assert np.allclose(
        driver.correlation_energy + driver.deltap3[0]["D"], -0.8574520862, atol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["D"],
        -283.6898593860,
        atol=1.0e-07
    )

if __name__ == "__main__":
    test_crcc23_glycine()
