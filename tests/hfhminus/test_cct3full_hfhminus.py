""" CC(t;3) computation for the open-shell (HFH)- molecule in the triplet
ground state using the D2H-symmetric H-F distance of 2.0 angstrom described
using the 6-31g(1d,1p) basis set. Active space consists of (2,2).
Reference: J. Chem. Theory Comput. 8, 4968 (2012)
"""

from pathlib import Path
import numpy as np
from ccpy import Driver, get_active_triples_pspace

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_cct3_hfhminus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/hfhminus-triplet/hfhminus-triplet.log",
        fcidump=TEST_DATA_DIR + "/hfhminus-triplet/hfhminus-triplet.FCIDUMP",
        nfrozen=1,
    )
    driver.system.print_info()
    driver.system.set_active_space(nact_unoccupied=2, nact_occupied=2)
    t3_excitations = get_active_triples_pspace(driver.system, target_irrep="B1U")

    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_leftccp(method="left_ccsdt_p", t3_excitations=t3_excitations)
    driver.run_ccp3(method="ccp3", state_index=0, two_body_approx=False, t3_excitations=t3_excitations)

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -100.3591573557, atol=1.0e-07)
    # Check CCSDt energy
    assert np.allclose(driver.correlation_energy, -0.19275952, atol=1.0e-07)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -100.5519168770,
        atol=1.0e-07
    )
    # Check CC(t;3)_A energy
    assert np.allclose(
        driver.correlation_energy + driver.deltap3[0]["A"], -0.1936283410, atol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["A"],
        -100.5527856967,
        atol=1.0e-07
    )
    # Check CC(t;3)_D energy
    assert np.allclose(
        driver.correlation_energy + driver.deltap3[0]["D"], -0.1938060944, atol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["D"],
        -100.5529634501,
        atol=1.0e-07
    )

if __name__ == "__main__":
    test_cct3_hfhminus()
