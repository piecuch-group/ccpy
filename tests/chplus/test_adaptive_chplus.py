from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver
from ccpy.drivers.adaptive import AdaptEOMDriver, AdaptEOMDriverSS

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_adaptive_chplus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()
    driver.options["maximum_iterations"] = 500

    adaptdriver = AdaptEOMDriver(driver, state_index=[2, 3, 4], roots_per_irrep={"A1": 4}, multiplicity=1,
                                   nacto=3, nactu=3,
                                   percentage=[0.0, 1.0, 2.0])
    adaptdriver.options["energy_tolerance"] = 1.0e-09
    adaptdriver.run()

if __name__ == "__main__":
    test_adaptive_chplus()
