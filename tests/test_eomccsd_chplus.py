""" EOMCCSD computation for the CH+ molecule at R = Re, where
Re = 2.13713 bohr described using the Olsen basis set.
Reference: Chem. Phys. Lett. 154, 380 (1989) [original Olsen paper with basis set]
           Mol. Phys. 118, e1817592 (2020) [CC(P;Q) results]"""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver

TEST_DATA_DIR = str(Path(__file__).parent.absolute() / "data")

def test_eomccsd_chplus():
    selected_states = [0, 1, 2, 3, 4, 5, 9] # Pick guess vectors for states (I knew this beforehand)

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()
    driver.options["maximum_iterations"] = 1000 # 4 Sigma state requires ~661 iterations in left-CCSD
    driver.options["davidson_max_subspace_size"] = 50
    driver.options["RHF_symmetry"] = False
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="cis", multiplicity=1, nroot=10)
    driver.run_eomcc(method="eomccsd", state_index=selected_states[1:])

    expected_vee = [
        0.0,
        0.11982887,
        0.11982887,
        0.49906873,
        0.53118318,
        0.53118318,
        0.0,
        0.0,
        0.0,
        0.63633490,
    ]
    expected_total_energy = [
        -38.0176701653,
        -37.8978412944,
        -37.8978412944,
        -37.5186014361,
        -37.4864869901,
        -37.4864869901,
        0.0,
        0.0,
        0.0,
        -37.3813352611,
    ]

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -37.9027681837)
    for n in selected_states:
        if n == 0:
            # Check CCSD energy
            assert np.allclose(driver.correlation_energy, -0.11490198)
            assert np.allclose(
                driver.system.reference_energy + driver.correlation_energy, -38.01767017
            )
        else:
            # Check EOMCCSD energy
            assert np.allclose(driver.vertical_excitation_energy[n], expected_vee[n])
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.vertical_excitation_energy[n],
                expected_total_energy[n],
            )

if __name__ == "__main__":
    test_eomccsd_chplus()
