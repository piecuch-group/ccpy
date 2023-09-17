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
    driver.run_guess(method="cisd", nroot=10, multiplicity=3, nact_occupied=2, nact_unoccupied=4)
    driver.options["RHF_symmetry"] = False
    driver.run_eomcc(method="eomcc3", state_index=[i + 1 for i in range(10)])

if __name__ == "__main__":
    test_cc3_ch2()
