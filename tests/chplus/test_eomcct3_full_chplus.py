import pytest
from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver
from ccpy.utilities.pspace import get_active_triples_pspace

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

@pytest.mark.short
def test_eomcct3_full_chplus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()
    driver.options["RHF_symmetry"] = True # left CC(P) for 4 Sigma^+ needs (4,4) guess space
    driver.options["maximum_iterations"] = 500
    # Set the active space
    driver.system.set_active_space(nact_occupied=1, nact_unoccupied=3)
    # Get the P space lists corresponding to this active space
    t3_excitations = get_active_triples_pspace(driver.system, target_irrep=driver.system.reference_symmetry)
    # Run CC(P)
    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_leftccp(method="left_ccsdt_p", t3_excitations=t3_excitations)
    driver.run_ccp3(method="ccp3", state_index=0, t3_excitations=t3_excitations, two_body_approx=False)

    # Initial guess
    driver.run_guess(method="cisd", multiplicity=1, roots_per_irrep={"A1": 4, "B1": 2, "B2": 0, "A2": 2},  nact_occupied=4, nact_unoccupied=4)
    roots = [2, 3, 4, 5, 6, 7, 8]
    irreps = ["A1", "A1", "A1", "B1", "B1", "A2", "A2"]
    # Run EOMCC(P)
    for state_index, irrep in zip(roots, irreps):
        r3_excitations = get_active_triples_pspace(driver.system, target_irrep=irrep)
        driver.run_eomccp(method="eomccsdt_p", state_index=state_index, t3_excitations=t3_excitations, r3_excitations=r3_excitations)
        driver.run_lefteomccp(method="left_ccsdt_p", state_index=state_index, t3_excitations=t3_excitations, r3_excitations=r3_excitations)
        # Perform the full CC(t;3) corrections using T(P), R(P), and L(P) 
        driver.run_ccp3(method="ccp3", state_index=state_index, t3_excitations=t3_excitations, r3_excitations=r3_excitations, two_body_approx=False)

    energy_ccsdt = [-38.019516, -37.702621, -37.522457, -37.386872, # sigma states
                    -37.900921, -37.498143, # pi states
                    -37.762113, -37.402308] # delta states

    energy_cct3 = [(driver.system.reference_energy 
                    + driver.correlation_energy 
                    + driver.vertical_excitation_energy[i]
                    + driver.deltap3[i]["D"]) for i in [0] + roots]

    expected_vee = [
        0.0,
        0.0,
        0.3175374823,
        0.4970422450,
        0.6331592771,
        0.1187944925,
        0.5226118511,
        0.2580090482,
        0.6191689138,
    ]
    expected_total_energy = [
        -38.0190411381,
         0.0,
        -37.7015036557,
        -37.5219988931,
        -37.3858818610,
        -37.9002466456,
        -37.4964292870,
        -37.7610320899,
        -37.3998722243,
    ]
    expected_deltap3 = {
        "A": [-0.0003880272,
               0.0,
              -0.0007857473,
              -0.0002949310,
              -0.0006538567,
              -0.0004737807,
              -0.0011990110,
              -0.0007820846,
              -0.0018324677],
        "D": [-0.0004634576,
               0.0,
              -0.0009703178,
              -0.0003616511,
              -0.0008238382,
              -0.0005775981 ,           
              -0.0015096300,
              -0.0009763455,
              -0.0022024492]}

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -37.9027681837, atol=1.0e-09)
    for n in [0] + roots:
        if n == 0:
            # Check CCSDt energy
            assert np.allclose(driver.correlation_energy, -0.11627295, atol=1.0e-09)
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

    for i, (ecct3, eccsdt) in enumerate(zip(energy_cct3, energy_ccsdt)):
        print(f"Error in root {i} = {np.round((ecct3 - eccsdt) * 1000, 4)} mEh")

if __name__ == "__main__":
    test_eomcct3_full_chplus()
