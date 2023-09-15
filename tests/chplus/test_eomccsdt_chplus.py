""" EOMCCSDT computation for the CH+ molecule at R = Re, where
Re = 2.13713 bohr described using the Olsen basis set."""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eomccsdt_chplus():

    # Corresponds to 1 -> 1 Pi, 3 -> 3 Sigma, and 4 -> 2 Pi
    selected_states = [0, 1, 3, 4]

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()
    driver.options["maximum_iterations"] = 600
    driver.options["davidson_max_subspace_size"] = 50
    # Run CCSDT calculation
    driver.run_cc(method="ccsdt")
    # Run CCSDT HBar
    driver.run_hbar(method="ccsdt")
    # Run EOMCCSDT
    driver.run_guess(method="cis", multiplicity=1, nroot=10)
    driver.run_eomcc(method="eomccsdt", state_index=selected_states[1:])
    # Run left CCSDT for ground and excited states
    driver.options["energy_shift"] = 0.3
    driver.run_leftcc(method="left_ccsdt", state_index=selected_states)

    expected_ref_energy = -37.90276818
    expected_cc_energy = -38.01951563
    expected_corr_energy = -0.11674744
    expected_vee = [0.0, 0.11859434, 0.49705838, 0.52137274]
    expected_total_energy = [expected_cc_energy + omega for omega in expected_vee]

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, expected_ref_energy)
    for n in range(len(expected_vee)):
        if n == 0:
            # Check CCSDT energy
            assert np.allclose(driver.correlation_energy, expected_corr_energy)
            assert np.allclose(
                driver.system.reference_energy + driver.correlation_energy, expected_cc_energy
            )
        else:
            # Check EOMCCSDT energy
            assert np.allclose(driver.vertical_excitation_energy[selected_states[n]], expected_vee[n])
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.vertical_excitation_energy[selected_states[n]],
                expected_total_energy[n],
            )

if __name__ == "__main__":
    test_eomccsdt_chplus()
