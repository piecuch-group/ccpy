import pytest
from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver
from ccpy.utilities.pspace import get_active_triples_pspace
from ccpy import Molecule, ROHF

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

# Same system as chplus.log: CH+ at R = 2.13713 bohr, Olsen basis (cartesian d)
CHPLUS_GEOMETRY = "C 0.0 0.0 0.0; H 0.0 0.0 2.13713"
CHPLUS_BASIS = str(Path(__file__).parents[1].absolute() / "data" / "chplus" / "olsen_chplus.gbs")


def _run_and_check(driver):
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
    # Initial guess
    driver.run_guess(method="cisd", multiplicity=1, roots_per_irrep={"A1": 4, "B1": 2, "B2": 0, "A2": 2},  nact_occupied=4, nact_unoccupied=4)
    roots = [2, 3, 4, 5, 6, 7, 8]
    irreps = ["A1", "A1", "A1", "B1", "B1", "A2", "A2"]
    # Run EOMCC(P)
    for state_index, irrep in zip(roots, irreps):
        r3_excitations = get_active_triples_pspace(driver.system, target_irrep=irrep)
        driver.run_eomccp(method="eomccsdt_p", state_index=state_index, t3_excitations=t3_excitations, r3_excitations=r3_excitations)
        driver.run_lefteomccp(method="left_ccsdt_p", state_index=state_index, t3_excitations=t3_excitations, r3_excitations=r3_excitations)
    # Perform CC(t;3) corrections using T(P), R(P), and L(P) within 2-body approximation
    driver.run_ccp3(method="cct3", state_index=[0] + roots, t3_excitations=t3_excitations, r3_excitations=r3_excitations)

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
        "A": [
            -0.0003735932,
             0.0,
            -0.0004470967,
            -0.0002854260,
            -0.0004937632,
            -0.0003868002,
            -0.0011116516,
            -0.0005036481,
            -0.0012962381,
        ],
        "D": [
            -0.0004458108,
             0.0,
            -0.0005448709,
            -0.0003492252,
            -0.0006162290,
            -0.0004714467,
            -0.0013967501,
            -0.0006256142,
            -0.0015070426,
        ],
    }

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


@pytest.mark.short
def test_eomcct3_chplus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    _run_and_check(driver)


@pytest.mark.short
def test_eomcct3_chplus_native():
    # native ccpy ROHF (GAMESS SCFTYP=ROHF, ISPHER=-1, UNITS=BOHR) + tight Cholesky integrals
    mol = Molecule(CHPLUS_GEOMETRY, CHPLUS_BASIS, unit="bohr", charge=1, multiplicity=1,
                   symmetry=True, cartesian=True, cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = ROHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    _run_and_check(driver)


if __name__ == "__main__":
    test_eomcct3_chplus()
    test_eomcct3_chplus_native()
