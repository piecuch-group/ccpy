import numpy as np
from pathlib import Path
from ccpy import Driver
from ccpy.constants.constants import hartreetoeV

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eomccsdtastar_chplus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
        ndelete=0,
    )
    driver.system.print_info()
    # Run ground-state CC calculation
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsdta") 
    # Run excited-state EOMCC calculation
    driver.run_guess(method="cisd", multiplicity=1, roots_per_irrep={"A1": 4, "B1": 2, "B2": 0, "A2": 2},  nact_occupied=3, nact_unoccupied=7)
    driver.run_eomcc(method="eomccsd", state_index=[2, 3, 4, 5, 6, 7, 8])
    driver.run_lefteomcc(method="left_ccsd", state_index=[2, 3, 4, 5, 6, 7, 8])
    # Compute EOMCCSDT(a)* excited-state corrections
    driver.run_ccp3(method="eomccsdta_star", state_index=[0, 2, 3, 4, 5, 6, 7, 8])

    #
    # Check the results
    #
    assert np.allclose(driver.correlation_energy, -0.11587601076883182)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -38.018644194459256)
    #
    expected_vee_eomccsd = [0.3354923755, 0.4997656842, 0.6371275358, 0.1205517602, 0.5321109625, 0.2907349366, 0.6509314995]
    expected_delta_star = [-0.0132113073, -0.0025074237, -0.0036117876, -0.0017407286, -0.0047324606, -0.0235589830, -0.0230871240]
    for i, istate in enumerate([2, 3, 4, 5, 6, 7, 8]):
        assert np.allclose(driver.vertical_excitation_energy[istate], expected_vee_eomccsd[i])
        assert np.allclose(driver.deltap3[istate]["A"], expected_delta_star[i])

    #
    # Compare EOMCCSDT(a)* and EOMCCSDT vertical excitation energies
    #
    print("")
    e0_ccsdt = -38.019516
    exc_ccsdt = [-37.702621, -37.522457, -37.386872, -37.900921, -37.498143, -37.762113, -37.402308]
    for i, istate in enumerate([2, 3, 4, 5, 6, 7, 8]):
        vee_ccsdt = (exc_ccsdt[i] - e0_ccsdt) * hartreetoeV
        vee_star = (driver.vertical_excitation_energy[istate] + driver.deltap3[istate]["A"]) * hartreetoeV
        print(f"  Root {i + 1}: EOMCCSDT(a)* = {np.round(vee_star, 4)} eV   EOMCCSDT = {np.round(vee_ccsdt, 4)} eV")


if __name__ == "__main__":
    test_eomccsdtastar_chplus()
