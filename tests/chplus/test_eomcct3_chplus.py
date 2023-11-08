from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver
from ccpy.utilities.pspace import get_active_pspace

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eomcct3_chplus():
    nacto = 2
    nactu = 2

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()
    # Set the active space
    driver.system.set_active_space(nact_occupied=nacto, nact_unoccupied=nactu)
    # Get the P space lists corresponding to this active space
    t3_excitations, _ = get_active_pspace(driver.system)
    r3_excitations, _ = get_active_pspace(driver.system)
    # Run CC(P)
    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_leftccp(method="left_ccsdt_p", t3_excitations=t3_excitations)
    # Initial guess
    driver.run_guess(method="cisd", roots_per_irrep={"B1": 3}, nact_occupied=4, nact_unoccupied=4, multiplicity=1)
    # Run EOMCC(P)
    for state_index in [1, 2]:
        driver.run_eomccp(method="eomccsdt_p", state_index=state_index, t3_excitations=t3_excitations, r3_excitations=r3_excitations)
        driver.run_lefteomccp(method="left_ccsdt_p", state_index=state_index, t3_excitations=t3_excitations, r3_excitations=r3_excitations)
    # Perform CC(t;3) corrections using T(P), R(P), and L(P) within 2-body approximation
    driver.run_ccp3(method="cct3", state_index=[0, 1, 2])

    expected_vee = [
        0.0,
        0.1188518907,
        0.5228422119,
    ]
    expected_total_energy = [
        -38.0190286716,
        -37.9001767809,
        -37.4961864597,
    ]
    expected_deltap3 = {
        "A": [
            -0.0003749894,
            -0.0004363457,
            -0.0012636823,
        ],
        "D": [
            -0.0004506047,
            -0.0005285731,
            -0.0015876530,
        ],
    }

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -37.9027681837, atol=1.0e-09)
    for n in [0, 1, 2]:
        if n == 0:
            # Check CCSDt energy
            assert np.allclose(driver.correlation_energy, -0.11626049, atol=1.0e-09)
            assert np.allclose(driver.system.reference_energy + driver.correlation_energy, expected_total_energy[n], atol=1.0e-09)
            # Check CC(t;3)_A energy
            assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[n]["A"], 
                expected_total_energy[n] + expected_deltap3["A"][n],
                atol=1.0e-09
            )
            # Check CC(t;3)_D energy
            assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[n]["D"],
                expected_total_energy[n] + expected_deltap3["D"][n],
                atol=1.0e-09
            )
        else:
            # Check EOMCCSDt energy
            assert np.allclose(driver.vertical_excitation_energy[n], expected_vee[n], atol=1.0e-09)
            assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[n],
                expected_total_energy[n],
                atol=1.0e-09
            )
            # Check EOMCC(t;3)_A energy
            assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[n] + driver.deltap3[n]["A"],
                expected_total_energy[n] + expected_deltap3["A"][n],
                atol=1.0e-09
            )
            # Check EOMCC(t;3)_D energy
            assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[n] + driver.deltap3[n]["D"],
                expected_total_energy[n] + expected_deltap3["D"][n],
                atol=1.0e-09
            )
            

if __name__ == "__main__":
    test_eomcct3_chplus()
