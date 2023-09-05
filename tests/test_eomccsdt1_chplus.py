"""EOMCCSDt computation for the CH+ molecule at R = Re, where
Re = 2.13713 bohr described using the Olsen basis set. The active
space consists of the highest-energy occupied orbital (3 sigma)
and the three lowest-energy unoccupied orbitals (1 pi_x = 1 pi,
1 pi_y = 2 pi, and 4 sigma).
Reference: J. Chem. Phys. 115, 643 (2001)."""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver

TEST_DATA_DIR = str(Path(__file__).parent.absolute() / "data")

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
    driver.run_guess(method="cis", multiplicity=1, nroot=10)
    driver.run_eomcc(method="eomccsdt1", state_index=[1, 2, 3, 4, 5])

    expected_vee = [0.0, 0.11879449, 0.11879449, 0.49704224, 0.52261182, 0.52261184]
    expected_total_energy = [-38.01904114 + omega for omega in expected_vee]

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -37.9027681837)
    for n in range(6):
        if n == 0:
            # Check CCSD energy
            assert np.allclose(driver.correlation_energy, -0.11627295)
            assert np.allclose(
                driver.system.reference_energy + driver.correlation_energy, -38.01904114
            )
        else:
            # Check EOMCCSDt energy
            assert np.allclose(driver.vertical_excitation_energy[n], expected_vee[n])
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.vertical_excitation_energy[n],
                expected_total_energy[n],
            )

if __name__ == "__main__":
    test_eomccsdt1_chplus()