""" CC(t;3) computation for the open-shell (HFH)- molecule in the triplet
ground state using the D2H-symmetric H-F distance of 2.0 angstrom described
using the 6-31g(1d,1p) basis set. Active space consists of (2,2).
Reference: J. Chem. Theory Comput. 8, 4968 (2012)
"""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver

TEST_DATA_DIR = str(Path(__file__).parent.absolute() / "data")

def test_cct3_hfhminus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/hfhminus-triplet/hfhminus-triplet.log",
        fcidump=TEST_DATA_DIR + "/hfhminus-triplet/hfhminus-triplet.FCIDUMP",
        nfrozen=1,
    )
    driver.system.set_active_space(nact_unoccupied=2, nact_occupied=2)
    driver.system.print_info()

    driver.run_cc(method="ccsdt1")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="cct3", state_index=[0])

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -100.3591573557)
    # Check CCSDt energy
    assert np.allclose(driver.correlation_energy, -0.19275952)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -100.5519168770
    )
    # Check CC(t;3)_A energy
    assert np.allclose(
        driver.correlation_energy + driver.deltapq[0]["A"], -0.1936570574
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltapq[0]["A"],
        -100.5528144130,
    )
    # Check CC(t;3)_D energy
    assert np.allclose(
        driver.correlation_energy + driver.deltapq[0]["D"], -0.1938411916
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltapq[0]["D"],
        -100.5529985472,
    )

if __name__ == "__main__":
    test_cct3_hfhminus()
