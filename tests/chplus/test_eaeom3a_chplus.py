"""EA-EOMCCSD(3p-2h){Nu} computation used to describe the spectrum of the
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

    driver.system.set_active_space(nact_unoccupied=3, nact_occupied=0)
    r3_excitations = get_active_3p2h_pspace(driver.system, num_active=1)

    driver.run_guess(method="eacisd", multiplicity=-1, nact_occupied=3, nact_unoccupied=8,
                     roots_per_irrep={"A1": 2, "B1": 2, "B2": 2, "A2": 2})

    for i in [0, 1, 2, 3, 4, 5, 6, 7]:
        driver.run_eaeomccp(method="eaeom3_p", state_index=i, r3_excitations=r3_excitations)

    expected_vee = [-0.26354074, -0.22337561, -0.37674145, -0.08128328, -0.37674145, -0.08128328, -0.35640329, -0.24898403]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i], vee)

if __name__ == "__main__":
    test_eaeom3a_chplus()
