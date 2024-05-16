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

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_creom23_chplus():

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
    driver.run_guess(method="cis", multiplicity=1, roots_per_irrep={"A1": 3, "B1": 2})
    driver.run_eomcc(method="eomccsd", state_index=[1, 3, 4, 5])
    driver.options[
        "energy_shift"
    ] = 0.8  # set energy shift to help converge left-EOMCCSD
    driver.options["diis_size"] = 12
    driver.run_leftcc(method="left_ccsd", state_index=[0, 1, 3, 4, 5])
    driver.run_ccp3(method="crcc23", state_index=[0, 1, 3, 4, 5])

    expected_vee = [
        0.0,
        0.49906873,
        0.0,
        0.63633490,
        0.11982887,
        0.53118318,
    ]
    expected_total_energy = [
        -38.0176701653,
        -37.5186014361,
         0.0,
        -37.3813352611,
        -37.8978412944,
        -37.4864869901,
    ]
    expected_deltap3 = {
        "A": [
            -0.0013798405,
            -0.0021697718,
             0.0,
            -0.0032097085,
            -0.0016296078,
            -0.0045706983,
        ],
        "D": [
            -0.0017825588,
            -0.0030686698,
             0.0,
            -0.0045827171,
            -0.0022877876,
            -0.0088507112,
        ],
    }
    expected_ddeltap3 = {
        "A": [
             0.0,
            -0.0022291593,
             0.0,
            -0.0033071442,
            -0.0016296078,
            -0.0045706983,
        ],
        "D": [
            0.0,
            -0.0031525794,
             0.0,
            -0.0047158142,
            -0.0022877876,
            -0.0088507112,
        ],
    }

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -37.9027681837, atol=1.0e-07)
    for n in [0, 1, 3, 4, 5]:
        if n == 0:
            # Check CCSD energy
            assert np.allclose(driver.correlation_energy, -0.11490198, atol=1.0e-07)
            assert np.allclose(
                driver.system.reference_energy + driver.correlation_energy, -38.01767017,
                atol=1.0e-07
            )
            # Check CR-CC(2,3)_A energy
            assert np.allclose(
                driver.correlation_energy + driver.deltap3[0]["A"], -0.1162818221, atol=1.0e-07
            )
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.deltap3[0]["A"],
                -38.0190500058,
                atol=1.0e-07
            )
            # Check CR-CC(2,3)_D energy
            assert np.allclose(
                driver.correlation_energy + driver.deltap3[0]["D"], -0.1166845404, atol=1.0e-07
            )
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.deltap3[0]["D"],
                -38.0194527241,
                atol=1.0e-07
            )
        else:
            # Check EOMCCSD energy
            assert np.allclose(driver.vertical_excitation_energy[n], expected_vee[n], atol=1.0e-07)
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.vertical_excitation_energy[n],
                expected_total_energy[n],
                atol=1.0e-07
            )
            # Check CR-CC(2,3)_A energy
            assert np.allclose(
                driver.vertical_excitation_energy[n] + driver.deltap3[n]["A"],
                expected_vee[n] + expected_deltap3["A"][n],
                atol=1.0e-07
            )
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.vertical_excitation_energy[n]
                + driver.deltap3[n]["A"],
                -38.01767017 + expected_vee[n] + expected_deltap3["A"][n],
                atol=1.0e-07
            )
            assert np.allclose(
                driver.vertical_excitation_energy[n] + driver.ddeltap3[n]["A"],
                expected_vee[n] + expected_ddeltap3["A"][n],
                atol=1.0e-07
            )
            # Check CR-CC(2,3)_D energy
            assert np.allclose(
                driver.vertical_excitation_energy[n] + driver.deltap3[n]["D"],
                expected_vee[n] + expected_deltap3["D"][n],
                atol=1.0e-07
            )
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.vertical_excitation_energy[n]
                + driver.deltap3[n]["D"],
                -38.01767017 + expected_vee[n] + expected_deltap3["D"][n],
                atol=1.0e-07
            )
            assert np.allclose(
                driver.vertical_excitation_energy[n] + driver.ddeltap3[n]["D"],
                expected_vee[n] + expected_ddeltap3["D"][n],
                atol=1.0e-07
            )

if __name__ == "__main__":
    test_creom23_chplus()
