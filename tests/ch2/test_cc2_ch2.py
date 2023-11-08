""" CC2 computation for the singlet ground state of the
CH2 molecule using a customized aug-cc-pVDZ basis set that
takes the Dunning cc-pVDZ basis set and adds a diffuse s
function to both the C and H atoms.
Reference: Chem. Phys. Lett. 244, 75 (1995)
"""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_cc2_ch2():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/ch2/ch2-avdz-koch.log",
        fcidump=TEST_DATA_DIR + "/ch2/ch2-avdz-koch.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()

    driver.options["RHF_symmetry"] = False
    driver.run_cc(method="cc2")
    #driver.run_hbar(method="cc2")
    #driver.run_guess(method="cisd", roots_per_irrep={"A1": 3, "B1": 3, "B2": 2, "A2": 1}, multiplicity=1, nact_occupied=2, nact_unoccupied=4)
    #driver.run_eomcc(method="eomcc2", state_index=[1, 2, 3, 4, 5, 6, 7, 8, 9])

    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -38.99447317, atol=1.0e-08) 

if __name__ == "__main__":
    test_cc2_ch2()
