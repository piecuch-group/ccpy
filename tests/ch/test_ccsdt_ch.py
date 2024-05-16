"""CCSDT computation on open-shell CH molecule."""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_ccsdt_ch():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/ch/ch.log",
        fcidump=TEST_DATA_DIR + "/ch/ch.FCIDUMP",
        nfrozen=1,
    )
    driver.system.print_info()

    driver.options["RHF_symmetry"] = False
    driver.run_cc(method="ccsdt")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -38.2713247488, atol=1.0e-07)
    # Check CCSDT energy
    assert np.allclose(driver.correlation_energy, -0.1164237849, atol=1.0e-07)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -38.3877485336, atol=1.0e-07
    )

if __name__ == "__main__":
    test_ccsdt_ch()
