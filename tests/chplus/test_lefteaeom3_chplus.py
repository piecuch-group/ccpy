"""EA-EOMCCSD(3p-2h) computation used to describe the spectrum of the
open-shell CH molecule by attaching an electron to closed-shell CH+."""

import pytest
from pathlib import Path
import numpy as np
from ccpy import Driver

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

@pytest.mark.short
def test_eaeom3_chplus():
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
    driver.run_eaeomcc(method="eaeom3", state_index=[0, 1, 2, 3, 4, 5, 6, 7])
    driver.run_lefteaeomcc(method="left_eaeom3", state_index=[0, 1, 2, 3, 4, 5, 6, 7])

    #
    # Check the results
    #
    expected_vee = [-0.26411607, -0.22435764, -0.37815570, -0.08130247, -0.37815570, -0.08130247, -0.35696964]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i], vee)

if __name__ == "__main__":
    test_eaeom3_chplus()
