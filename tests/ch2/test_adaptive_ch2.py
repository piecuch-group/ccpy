from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver
from ccpy.drivers.adaptive import AdaptEOMDriver

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_adaptive_ch2():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/ch2/ch2-avdz-koch.log",
        fcidump=TEST_DATA_DIR + "/ch2/ch2-avdz-koch.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()

    state_index=1
    roots_per_irrep = {"B1": 1}
    multiplicity = 1
    adaptdriver = AdaptEOMDriver(driver, state_index, roots_per_irrep, multiplicity, 
                                 nacto=3, nactu=3, 
                                 percentage=[0.0, 1.0])

    adaptdriver.run()

if __name__ == "__main__":
    test_adaptive_ch2()
