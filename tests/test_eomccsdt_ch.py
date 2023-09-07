""" EOMCCSDT computation for the CH+ molecule at R = Re, where
Re = 2.13713 bohr described using the Olsen basis set."""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver

TEST_DATA_DIR = str(Path(__file__).parent.absolute() / "data")

def test_eomccsdt_ch():

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/ch/ch.log",
        fcidump=TEST_DATA_DIR + "/ch/ch.FCIDUMP",
        nfrozen=1,
    )
    driver.system.print_info()
    driver.options["maximum_iterations"] = 100
    driver.options["davidson_max_subspace_size"] = 50
    driver.run_cc(method="ccsdt")
    driver.run_hbar(method="ccsdt")
    driver.run_guess(method="cis", multiplicity=2, nroot=10)
    driver.run_eomcc(method="eomccsdt", state_index=[1, 2, 3])
    # Run left CCSDT for ground and excited states
    driver.options["energy_shift"] = 0.0
    driver.run_leftcc(method="left_ccsdt", state_index=[0, 1, 2, 3])

    expected_ref_energy = -38.27132475
    expected_cc_energy = -38.38774854
    expected_corr_energy = -0.11642379
    expected_vee = [0.0, 0.00001342, 0.12216165, 0.11075965]
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
            assert np.allclose(driver.vertical_excitation_energy[n], expected_vee[n])
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.vertical_excitation_energy[n],
                expected_total_energy[n],
            )

if __name__ == "__main__":
    test_eomccsdt_ch()
