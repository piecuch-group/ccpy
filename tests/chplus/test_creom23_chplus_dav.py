import pytest
import numpy as np
from pathlib import Path
from ccpy import Driver

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

@pytest.mark.short
def test_creom23_chplus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
        ndelete=0,
    )
    driver.system.print_info()
    # Run ground-state CC calculation
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd") 
    driver.run_leftcc(method="left_ccsd", state_index=[0])
    # Run excited-state EOMCC calculation
    driver.run_guess(method="cisd", multiplicity=1, roots_per_irrep={"A1": 4, "B1": 2, "B2": 0, "A2": 2},  nact_occupied=3, nact_unoccupied=7)
    driver.run_eomcc(method="eomccsd", state_index=[2, 3, 4, 5, 6, 7, 8])
    driver.run_lefteomcc(method="left_ccsd", state_index=[2, 3, 4, 5, 6, 7, 8])
    # Compute CR-CC(2,3) ground- and excited-state corrections
    driver.run_ccp3(method="crcc23", state_index=[0, 2, 3, 4, 5, 6, 7, 8])

    # Sigma states (including ground state)
    sigma = [None for x in range(4)]
    sigma[0] = (driver.system.reference_energy + driver.correlation_energy,
                driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["D"])
    sigma[1] = (driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[2],
                driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[2] + driver.deltap3[2]["D"])
    sigma[2] = (driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[3],
                driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[3] + driver.deltap3[3]["D"])
    sigma[3] = (driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[4],
                driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[4] + driver.deltap3[4]["D"])
    # Pi states
    pi = [None for x in range(2)]
    pi[0] = (driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[5],
             driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[5] + driver.deltap3[5]["D"])
    pi[1] = (driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[6],
             driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[6] + driver.deltap3[6]["D"])
    # Delta states
    delta = [None for x in range(2)]
    delta[0] = (driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[7],
                driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[7] + driver.deltap3[7]["D"])
    delta[1] = (driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[8],
                driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[8] + driver.deltap3[8]["D"])

    sigma_ccsdt = [-38.019516, -37.702621, -37.522457, -37.386872]
    for i in range(4):
        ccsd, crcc23 = sigma[i]
        ccsdt = sigma_ccsdt[i]
        print(f"{i + 1}Sigma+: CCSD = {np.round((ccsd - ccsdt)*1000, 4)}  CR-CC(2,3) = {np.round((crcc23 - ccsdt)*1000, 4)}")
    pi_ccsdt = [-37.900921, -37.498143]
    for i in range(2):
        ccsd, crcc23 = pi[i]
        ccsdt = pi_ccsdt[i]
        print(f"{i + 1}Pi: CCSD = {np.round((ccsd - ccsdt)*1000, 4)}  CR-CC(2,3) = {np.round((crcc23 - ccsdt)*1000, 4)}")
    delta_ccsdt= [-37.762113, -37.402308]
    for i in range(2):
        ccsd, crcc23 = delta[i]
        ccsdt = delta_ccsdt[i]
        print(f"{i + 1}Delta: CCSD = {np.round((ccsd - ccsdt)*1000, 4)}  CR-CC(2,3) = {np.round((crcc23 - ccsdt)*1000, 4)}")

if __name__ == "__main__":
    test_creom23_chplus()
