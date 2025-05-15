"""CC(t;3) computation on open-shell CH molecule."""

import pytest
from pathlib import Path
import numpy as np
from ccpy import Driver, get_active_triples_pspace

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

@pytest.mark.short
def test_cct3_ch():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/ch/ch.log",
        fcidump=TEST_DATA_DIR + "/ch/ch.FCIDUMP",
        nfrozen=1,
    )
    driver.system.set_active_space(nact_occupied=1, nact_unoccupied=2)
    driver.system.print_info()

    t3_excitations = get_active_triples_pspace(driver.system, target_irrep="B2")
    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="cct3")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -38.2713247488, atol=1.0e-07)
    # Check CCSDt energy
    assert np.allclose(driver.correlation_energy, -0.11469532, atol=1.0e-07)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -38.38602007, atol=1.0e-07
    )
    # Check CC(t;3)_A energy
    assert np.allclose(
        driver.correlation_energy + driver.deltap3[0]["A"], -0.1160181452, atol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["A"],
        -38.3873428940,
        atol=1.0e-07
    )
    # Check CC(t;3)_D energy
    assert np.allclose(
        driver.correlation_energy + driver.deltap3[0]["D"], -0.1162820915, atol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["D"],
        -38.3876068402,
        atol=1.0e-07
    )

if __name__ == "__main__":
    test_cct3_ch()
