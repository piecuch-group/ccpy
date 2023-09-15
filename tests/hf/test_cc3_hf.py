""" CC3 computation for the stretched HF molecule at interatomic
separation R = 1.6 angstrom."""

import numpy as np
from pyscf import scf, gto
from ccpy.drivers.driver import Driver

def test_cc3_hf():
    geometry = [["H", (0.0, 0.0, -0.8)], ["F", (0.0, 0.0, 0.8)]]
    mol = gto.M(
        atom=geometry,
        basis="6-31g",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=True,
        unit="Angstrom",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)

    driver.options["RHF_symmetry"] = False
    driver.run_cc(method="cc3")
    driver.run_hbar(method="cc3")
    driver.run_guess(method="cis", nroot=5, multiplicity=1)
    driver.run_eomcc(method="eomcc3", state_index=[1])

    # Check CC3 correlation energy
    assert np.allclose(driver.correlation_energy, -0.178932834091, atol=1.0e-07)
    # Check CC3 total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -100.00884041, atol=1.0e-07)
    # Check EOMCC3 vertical excitation energy
    assert np.allclose(driver.vertical_excitation_energy[1], 0.10083870, atol=1.0e-07)
    # Check EOMCC3 total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[1], -99.90800171, atol=1.0e-07) 

if __name__ == "__main__":
    test_cc3_hf()
