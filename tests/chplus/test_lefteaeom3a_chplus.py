"""Active-space EA-EOMCCSD(3p-2h){Nu} (also known as EA-EOMCCSDt) calculation to
describe the vertical excitation spectrum of the open-shell CH molecule by attaching
an electron to CH+ closed shell."""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver
from ccpy.utilities.pspace import get_active_3p2h_pspace

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eaeom3a_chplus():
    # Obtain the Driver from GAMESS logfile and FCIDUMP
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()

    # Perform CCSD for the closed-shell core
    driver.run_cc(method="ccsd")
    # Obtain the CCSD-level similarity-transformed Hamiltonian
    driver.run_hbar(method="ccsd")

    # Set the number of active unoccupied orbitals used to define the r_{Abc}^{jk} operator
    driver.system.set_active_space(nact_unoccupied=3, nact_occupied=0)
    # Obtain the active-space 3p2h list
    r3_excitations = get_active_3p2h_pspace(driver.system, num_active=1)

    # Perform guess vectors by diagonalizaing within the 1p + active 2p-1h space
    driver.run_guess(method="eacisd", multiplicity=-1, nact_occupied=3, nact_unoccupied=8,
                     roots_per_irrep={"A1": 2, "B1": 2, "B2": 2, "A2": 2})
    # Loop over all guess vectors and perform the EA-EOMCSDt calculation
    for i in [0, 1, 2, 3, 4, 5, 6]:
        driver.run_eaeomccp(method="eaeom3_p", state_index=i, r3_excitations=r3_excitations)
        driver.run_lefteaeomccp(method="left_eaeom3_p", state_index=i, r3_excitations=r3_excitations)
        driver.run_eaccp3(method="eaccp3", state_index=i, r3_excitations=r3_excitations)

    #
    # Check the results
    #
    expected_vee = [-0.26354074, -0.22337561, -0.37674145, -0.08128328, -0.37674145, -0.08128328, -0.35640329]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i], vee)

if __name__ == "__main__":
    test_eaeom3a_chplus()
