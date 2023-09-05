""" EOMCCSDT computation for the CH+ molecule at R = Re, where
Re = 2.13713 bohr described using the Olsen basis set."""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver

TEST_DATA_DIR = str(Path(__file__).parent.absolute() / "data")

def test_eomccsdt_chplus():

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()
    driver.options["maximum_iterations"] = 100 # 4 Sigma state requires ~661 iterations in left-CCSD
    driver.options["davidson_max_subspace_size"] = 50
    driver.run_cc(method="ccsdt")
    driver.run_hbar(method="ccsdt")
    driver.run_guess(method="cis", multiplicity=1, nroot=10)
    driver.run_eomcc(method="eomccsdt", state_index=[1])
    driver.run_leftcc(method="left_ccsdt", state_index=[1])

if __name__ == "__main__":
    test_eomccsdt_chplus()