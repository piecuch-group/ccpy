"""EA-EOMCCSD(2h-1p) computation used to describe the spectrum of the
open-shell CH molecule by attaching an electron to closed-shell CH+."""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver
from ccpy.utilities.pspace import get_active_3p2h_pspace

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eaeom2_chplus():

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="eacisd", multiplicity=-1, nact_occupied=3, nact_unoccupied=8,
                     roots_per_irrep={"A1": 2, "B1": 2, "B2": 2, "A2": 2})
    driver.system.set_active_space(0, 0)
    r3_excitations = get_active_3p2h_pspace(driver.system, 1, target_irrep=None)
    for istate in [0, 1, 2, 3, 4, 5, 6, 7]:
        driver.run_eaeomccp(method="eaeom3_p", state_index=istate, r3_excitations=r3_excitations)
        driver.run_lefteaeomccp(method="left_eaeom3_p", state_index=istate, r3_excitations=r3_excitations)
        driver.run_eaccp3(method="creacc23", state_index=[istate])
    #
    # Check the results
    #
    expected_vee = [-0.19558721, -0.16826126, -0.37794266, -0.08109635, -0.37794266, -0.08109635, -0.29285023, -0.19558721]
    for i, vee in enumerate(expected_vee):
       assert np.allclose(driver.vertical_excitation_energy[i], vee)

if __name__ == "__main__":
    test_eaeom2_chplus()
