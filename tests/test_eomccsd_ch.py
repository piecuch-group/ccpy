"""EOMCCSD computation for the open-shell CH molcule."""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver

TEST_DATA_DIR = str(Path(__file__).parent.absolute() / "data")

def test_eomccsd_ch():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/ch/ch.log",
        fcidump=TEST_DATA_DIR + "/ch/ch.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="cis", multiplicity=2, nroot=10)
    driver.run_eomcc(method="eomccsd", state_index=[1, 2, 3])

    expected_vee = [0.0, 0.00016954, 0.11798458, 0.12122133]
    expected_total_energy = [-38.38631169 + omega for omega in expected_vee]

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -38.27132475)
    for n in range(4):
        if n == 0:
            # Check CCSD energy
            assert np.allclose(driver.correlation_energy, -0.11498694)
            assert np.allclose(
                driver.system.reference_energy + driver.correlation_energy, -38.38631169
            )
        else:
            # Check EOMCCSD energy
            assert np.allclose(driver.vertical_excitation_energy[n], expected_vee[n])
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.vertical_excitation_energy[n],
                expected_total_energy[n],
            )
if __name__ == "__main__":
    test_eomccsd_ch()
