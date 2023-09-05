""" CR-EOMCC(2,3) computation for the CH+ molecule at R = Re, where
Re = 2.13713 bohr described using the Olsen basis set.
Excited-state EOMCCSD, CR-EOMCC(2,3), and delta-CR-EOMCC(2,3)
calculations performed for 5 excited states initiated using the
CIS initial guess.
The state orderings obtained using CIS are as follows:
1 and 2 -> 1 Pi
3 -> 3 Sigma
4 and 5 -> 2 Pi
9 -> 4 Sigma.
The 2 Sigma and both Delta states are dominated by two-electron
excitations are are thus invisible to the CIS initial guess.
Reference: Chem. Phys. Lett. 154, 380 (1989) [original Olsen paper with basis set]
           Mol. Phys. 118, e1817592 (2020) [CC(P;Q) results]"""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver

TEST_DATA_DIR = str(Path(__file__).parent.absolute() / "data")

def test_creom23_chplus():
    selected_states = [0, 1, 2, 3, 4, 5, 9] # Pick guess vectors for states (I knew this beforehand)

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()
    driver.options["maximum_iterations"] = 1000 # 4 Sigma state requires ~661 iterations in left-CCSD
    driver.options["davidson_max_subspace_size"] = 50
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="cis", multiplicity=1, nroot=10)
    driver.run_eomcc(method="eomccsd", state_index=selected_states[1:])
    driver.options[
        "energy_shift"
    ] = 0.8  # set energy shift to help converge left-EOMCCSD
    driver.options["diis_size"] = 12
    driver.run_leftcc(method="left_ccsd", state_index=selected_states)
    driver.run_ccp3(method="crcc23", state_index=selected_states)

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
    expected_deltapq = {
        "A": [
            -0.0013798405,
            -0.0016296078,
            -0.0016296078,
            -0.0021697718,
            -0.0045706983,
            -0.0045706983,
            0.0,
            0.0,
            0.0,
            -0.0032097085,
        ],
        "D": [
            -0.0017825588,
            -0.0022877876,
            -0.0022877876,
            -0.0030686698,
            -0.0088507112,
            -0.0088507112,
            0.0,
            0.0,
            0.0,
            -0.0045827171,
        ],
    }
    expected_ddeltapq = {
        "A": [
            0.0,
            -0.0016296078,
            -0.0016296078,
            -0.0022291593,
            -0.0045706983,
            -0.0045706983,
            0.0,
            0.0,
            0.0,
            -0.0033071442,
        ],
        "D": [
            0.0,
            -0.0022877876,
            -0.0022877876,
            -0.0031525794,
            -0.0088507112,
            -0.0088507112,
            0.0,
            0.0,
            0.0,
            -0.0047158142,
        ],
    }

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -37.9027681837)
    for n in selected_states:
        if n == 0:
            # Check CCSD energy
            assert np.allclose(driver.correlation_energy, -0.11490198)
            assert np.allclose(
                driver.system.reference_energy + driver.correlation_energy, -38.01767017
            )
            # Check CR-CC(2,3)_A energy
            assert np.allclose(
                driver.correlation_energy + driver.deltapq[0]["A"], -0.1162818221
            )
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.deltapq[0]["A"],
                -38.0190500058,
            )
            # Check CR-CC(2,3)_D energy
            assert np.allclose(
                driver.correlation_energy + driver.deltapq[0]["D"], -0.1166845404
            )
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.deltapq[0]["D"],
                -38.0194527241,
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
            # Check CR-CC(2,3)_A energy
            assert np.allclose(
                driver.vertical_excitation_energy[n] + driver.deltapq[n]["A"],
                expected_vee[n] + expected_deltapq["A"][n],
            )
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.vertical_excitation_energy[n]
                + driver.deltapq[n]["A"],
                -38.01767017 + expected_vee[n] + expected_deltapq["A"][n],
            )
            assert np.allclose(
                driver.vertical_excitation_energy[n] + driver.ddeltapq[n]["A"],
                expected_vee[n] + expected_ddeltapq["A"][n],
            )

            # Check CR-CC(2,3)_D energy
            assert np.allclose(
                driver.vertical_excitation_energy[n] + driver.deltapq[n]["D"],
                expected_vee[n] + expected_deltapq["D"][n],
            )
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.vertical_excitation_energy[n]
                + driver.deltapq[n]["D"],
                -38.01767017 + expected_vee[n] + expected_deltapq["D"][n],
            )
            assert np.allclose(
                driver.vertical_excitation_energy[n] + driver.ddeltapq[n]["D"],
                expected_vee[n] + expected_ddeltapq["D"][n],
            )

if __name__ == "__main__":
    test_creom23_chplus()