""" CC3 computation for the singlet ground state of the
CH2 molecule using a customized aug-cc-pVDZ basis set that
takes the Dunning cc-pVDZ basis set and adds a diffuse s
function to both the C and H atoms.
Reference: Chem. Phys. Lett. 244, 75 (1995)
"""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_cc3_ch2():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/ch2/ch2-avdz-koch.log",
        fcidump=TEST_DATA_DIR + "/ch2/ch2-avdz-koch.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()

    driver.run_cc(method="cc3")

    # Check the CC3 total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -39.024868, atol=1.0e-07)

    driver.run_hbar(method="cc3")
    driver.run_guess(method="cisd", roots_per_irrep={"A1": 3, "B1": 3, "B2": 2, "A2": 1}, multiplicity=1, nact_occupied=2, nact_unoccupied=4)
    #driver.options["RHF_symmetry"] = False
    driver.run_eomcc(method="eomcc3", state_index=[1, 2, 3, 4, 5, 6, 7, 8, 9])

    expected_ref_energy = -38.88142579
    expected_corr_energy = -0.14344228
    expected_vee = [0.0,
                    0.18837284, #
                    0.23924155, #
                    0.31138281, #
                    0.06571101, #
                    0.06571101, # 0.34831055, this was changed when orthonormalized B0 matrix was removed
                    0.06571101, #
                    0.28371563, #
                    0.31357745, #
                    0.21528220] #

    for i in range(10):
        computed_total_energy = driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[i]
        expected_total_energy = expected_ref_energy + expected_corr_energy + expected_vee[i]
        assert np.allclose(computed_total_energy, expected_total_energy, atol=1.0e-06)

if __name__ == "__main__":
    test_cc3_ch2()
