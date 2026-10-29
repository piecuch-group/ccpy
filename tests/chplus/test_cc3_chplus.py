from pathlib import Path
import numpy as np
from ccpy import Driver, Molecule, ROHF

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

# Same system as chplus.log: CH+ at R = 2.13713 bohr, Olsen basis (cartesian d)
CHPLUS_GEOMETRY = "C 0.0 0.0 0.0; H 0.0 0.0 2.13713"
CHPLUS_BASIS = str(Path(__file__).parents[1].absolute() / "data" / "chplus" / "olsen_chplus.gbs")


def _run_and_check(driver):
    driver.system.print_info()
    driver.options["RHF_symmetry"] = True # left CC(P) for 4 Sigma^+ needs (4,4) guess space
    driver.options["maximum_iterations"] = 500

    # Run CC3
    driver.run_cc(method="cc3")
    driver.run_hbar(method="cc3")

    # Initial guess
    driver.run_guess(method="cisd", multiplicity=1, roots_per_irrep={"A1": 4, "B1": 2, "B2": 0, "A2": 2},  nact_occupied=3, nact_unoccupied=11)
    # Run excited-state CC3
    roots = [2, 3, 4, 5, 6, 7, 8]
    irreps = ["A1", "A1", "A1", "A1", "B1", "B1", "A2", "A2"]
    driver.run_eomcc(method="eomcc3", state_index=roots)

    #
    # Check the results
    #
    energy_ccsdt = [-38.019516, -37.702621, -37.522457, -37.386872, # sigma states
                    -37.900921, -37.498143, # pi states
                    -37.762113, -37.402308] # delta states

    energy_cc3 = [(driver.system.reference_energy
                   + driver.correlation_energy
                   + driver.vertical_excitation_energy[i]) for i in [0] + roots]

    for i, (ecc3, eccsdt) in enumerate(zip(energy_cc3, energy_ccsdt)):
        print(f"Error in root {i}, ({irreps[i]}) = {np.round((ecc3 - eccsdt) * 1000, 4)} mEh")


def test_cc3_chplus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    _run_and_check(driver)


def test_cc3_chplus_native():
    # native ccpy ROHF (GAMESS SCFTYP=ROHF, ISPHER=-1, UNITS=BOHR) + tight Cholesky integrals
    mol = Molecule(CHPLUS_GEOMETRY, CHPLUS_BASIS, unit="bohr", charge=1, multiplicity=1,
                   symmetry=True, cartesian=True, cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = ROHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    _run_and_check(driver)


if __name__ == "__main__":
    test_cc3_chplus()
    test_cc3_chplus_native()
