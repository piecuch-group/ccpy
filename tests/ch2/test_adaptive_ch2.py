from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver
from ccpy.drivers.adaptive import AdaptEOMDriverSS

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_adaptive_ch2():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/ch2/ch2-avdz-koch.log",
        fcidump=TEST_DATA_DIR + "/ch2/ch2-avdz-koch.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()

    adaptdriver = AdaptEOMDriverSS(driver, state_index=1, roots_per_irrep={"B1": 1}, multiplicity=1,
                                   nacto=3, nactu=3,
                                   percentage=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    adaptdriver.options["energy_tolerance"] = 1.0e-09

    adaptdriver.run()

if __name__ == "__main__":
    test_adaptive_ch2()
