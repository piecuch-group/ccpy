"""EOMCCSDt computation for the CH+ molecule at R = Re, where
Re = 2.13713 bohr described using the Olsen basis set. The active
space consists of the highest-energy occupied orbital (3 sigma)
and the three lowest-energy unoccupied orbitals (1 pi_x = 1 pi,
1 pi_y = 2 pi, and 4 sigma).
Reference: J. Chem. Phys. 115, 643 (2001)."""

import pytest
from pathlib import Path
import numpy as np
from ccpy import Driver

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

@pytest.mark.short
def test_eomccsdt1_chplus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()
    driver.system.set_active_space(nact_occupied=1, nact_unoccupied=3)

    driver.run_cc(method="ccsdt1")
    driver.run_hbar(method="ccsdt1")
    driver.run_guess(method="cisd", multiplicity=1, roots_per_irrep={"A1": 4, "B1": 2, "B2": 0, "A2": 2},  nact_occupied=3, nact_unoccupied=7)

    driver.options["davidson_max_subspace_size"] = 50
    driver.run_eomcc(method="eomccsdt1", state_index=[2, 3, 4, 5, 6, 7, 8])

    expected_vee = [0.0, 0.0, 0.31753748, 0.49704224, 0.63315928,  # sigma states
                    0.11879449, 0.52261185,  # pi states
                    0.25800905, 0.61916891]   # delta states
    expected_total_energy = [-38.01904114 + omega for omega in expected_vee]

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -37.9027681837, atol=1.0e-07)
    for n in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        if n == 0:
            # Check CCSD energy
            assert np.allclose(driver.correlation_energy, -0.11627295, atol=1.0e-07)
            assert np.allclose(
                driver.system.reference_energy + driver.correlation_energy, -38.01904114, atol=1.0e-07
            )
        else:
            # Check EOMCCSDt energy
            assert np.allclose(driver.vertical_excitation_energy[n], expected_vee[n], atol=1.0e-07)
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.vertical_excitation_energy[n],
                expected_total_energy[n],
                atol=1.0e-07
            )

if __name__ == "__main__":
    test_eomccsdt1_chplus()
