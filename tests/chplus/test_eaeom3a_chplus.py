"""EA-EOMCCSD(3p-2h) computation used to describe the spectrum of the
open-shell CH molecule by attaching an electron to closed-shell CH+."""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver
from ccpy.utilities.pspace import get_active_3p2h_pspace

from ccpy.models.operators import FockOperator
from ccpy.utilities.utilities import convert_excitations_c_to_f

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eaeom3a_chplus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")

    driver.system.set_active_space(nact_unoccupied=23, nact_occupied=0)
    r3_excitations = get_active_3p2h_pspace(driver.system)

    driver.run_guess(method="eacis", multiplicity=2, roots_per_irrep={"A1": 6, "B1": 0, "B2": 0, "A2": 0}, debug=False, use_symmetry=False)
    driver.run_eaeomccp(method="eaeom3_p", state_index=0, r3_excitations=r3_excitations)

if __name__ == "__main__":
    test_eaeom3a_chplus()
