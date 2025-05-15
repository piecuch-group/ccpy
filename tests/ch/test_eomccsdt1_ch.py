"""EOMCCSDt computation for the open-shell CH molcule."""

import pytest
from pathlib import Path
import numpy as np
from ccpy import Driver, get_active_triples_pspace

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

@pytest.mark.short
def test_eomccsdt1_ch():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/ch/ch.log",
        fcidump=TEST_DATA_DIR + "/ch/ch.FCIDUMP",
        nfrozen=1,
    )
    driver.system.print_info()
    driver.system.set_active_space(nact_occupied=1, nact_unoccupied=1)

    t3_excitations = get_active_triples_pspace(driver.system, target_irrep="B2")

    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_guess(method="cis", multiplicity=2, roots_per_irrep={"A1": 1, "B1": 1, "B2": 0, "A2": 1})

    r3_excitations = get_active_triples_pspace(driver.system, target_irrep=None)
    for istate in [1, 2, 3]:
        driver.run_eomccp(method="eomccsdt_p", state_index=istate, r3_excitations=r3_excitations, t3_excitations=t3_excitations)

    expected_vee = [0.0, 0.11287040, 0.00015539, 0.12326569]
    expected_total_energy = [-38.38596742 + omega for omega in expected_vee]

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -38.2713247488, atol=1.0e-07)
    for n in range(4):
        if n == 0:
            # Check CCSDt energy
            assert np.allclose(driver.correlation_energy, -0.11464267, atol=1.0e-07)
            assert np.allclose(
                driver.system.reference_energy + driver.correlation_energy, -38.38596742, atol=1.0e-07
            )
        else:
            # Check EOMCCSDt energy
            assert np.allclose(driver.vertical_excitation_energy[n], expected_vee[n], atol=1.0e-07)
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.vertical_excitation_energy[n],
                expected_total_energy[n], atol=1.0e-07
            )

if __name__ == "__main__":
    test_eomccsdt1_ch()
