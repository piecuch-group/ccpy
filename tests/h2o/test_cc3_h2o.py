""" CC3 computation for the stretched HF molecule at interatomic
separation R = 1.6 angstrom."""

import numpy as np
from pyscf import scf, gto
from ccpy.drivers.driver import Driver

def test_cc3_h2o():
    geometry = [["O", (0.0, 0.0, -0.0180)],
                ["H", (0.0, 3.030526, -2.117796)],
                ["H", (0.0, -3.030526, -2.117796)]]
    mol = gto.M(
        atom=geometry,
        basis="6-31g",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=True,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)

    driver.options["RHF_symmetry"] = False
    driver.run_cc(method="cc3")

    driver.run_hbar(method="cc3")
    driver.run_guess(method="cis", roots_per_irrep={"A1": 0, "B1": 1, "B2": 0, "A2": 0}, multiplicity=1)
    driver.run_eomcc(method="eomcc3", state_index=[1])

    # Check the CC3 correlation energy
    assert np.allclose(driver.correlation_energy, -0.3047279747) 
    # Check the CC3 total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -75.87813718)
    # Check the CC3 vertical excitation energy of 1st singlet excited state
    assert np.allclose(driver.vertical_excitation_energy[1], 0.0282254899) 
    # Check the CC3 total energy of 1st singlet excited state
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[1], -75.84991169) 


if __name__ == "__main__":
    test_cc3_h2o()
