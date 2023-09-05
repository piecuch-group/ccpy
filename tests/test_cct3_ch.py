"""CC(t;3) computation on open-shell CH molecule."""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver

TEST_DATA_DIR = str(Path(__file__).parent.absolute() / "data")

def test_cct3_ch():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/ch/ch.log",
        fcidump=TEST_DATA_DIR + "/ch/ch.FCIDUMP",
        nfrozen=1,
    )
    driver.system.set_active_space(nact_occupied=1, nact_unoccupied=2)
    driver.system.print_info()

    driver.run_cc(method="ccsdt1")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="cct3")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -38.2713247488)
    # Check CCSDt energy
    assert np.allclose(driver.correlation_energy, -0.11469532)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -38.38602007
    )
    # Check CC(t;3)_A energy
    assert np.allclose(
        driver.correlation_energy + driver.deltapq[0]["A"], -0.1160181452
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltapq[0]["A"],
        -38.3873428940,
    )
    # Check CC(t;3)_D energy
    assert np.allclose(
        driver.correlation_energy + driver.deltapq[0]["D"], -0.1162820915
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltapq[0]["D"],
        -38.3876068402,
    )

if __name__ == "__main__":
    test_cct3_ch()