"""EA-EOMCCSD(3p-2h) computation used to describe the spectrum of the
open-shell CH molecule by attaching an electron to closed-shell CH+."""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eaeom3_chplus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="eacis", multiplicity=2, nroot=10, debug=False)
    driver.run_eaeomcc(method="eaeom3", state_index=[0,1,2,3,4,5])

if __name__ == "__main__":
    test_eaeom3_chplus()
