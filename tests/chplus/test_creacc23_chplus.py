"""EA-EOMCCSD(2h-1p) computation used to describe the spectrum of the
open-shell CH molecule by attaching an electron to closed-shell CH+."""

import pytest
from pathlib import Path
import numpy as np
from ccpy import Driver

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

@pytest.mark.short
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
    driver.run_eaeomcc(method="eaeom2", state_index=[0, 1, 2, 3, 4, 5, 6, 7])
    driver.run_lefteaeomcc(method="left_eaeom2", state_index=[0, 1, 2, 3, 4, 5, 6, 7])
    driver.run_eaccp3(method="creacc23", state_index=[0, 1, 2, 3, 4, 5, 6, 7])

    #
    # Check the results
    #
    expected_vee = [-0.19558721, -0.16826126, -0.37794266, -0.08109635, -0.37794266, -0.08109635, -0.29285023, -0.19558721]
    expected_crcc23 = [-38.2862219438, -38.2467552810, -38.3950222424, -38.0988834026, -38.3950222424, -38.0988834026, -38.3741059608, -38.2881834536]
    for i, (vee, veep3) in enumerate(zip(expected_vee, expected_crcc23)):
        assert np.allclose(driver.vertical_excitation_energy[i], vee, atol=1.0e-07, rtol=1.0e-07)
        assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[i] + driver.deltap3[i]["D"], veep3, atol=1.0e-07, rtol=1.0e-07)

if __name__ == "__main__":
    test_eaeom2_chplus()
