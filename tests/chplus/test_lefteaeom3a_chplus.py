"""Active-space EA-EOMCCSD(3p-2h){Nu} (also known as EA-EOMCCSDt) calculation to
describe the vertical excitation spectrum of the open-shell CH molecule by attaching
an electron to CH+ closed shell."""

import pytest
from pathlib import Path
import numpy as np
from ccpy import Driver, get_active_3p2h_pspace, Molecule, ROHF

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

# Same system as chplus.log: CH+ at R = 2.13713 bohr, Olsen basis (cartesian d)
CHPLUS_GEOMETRY = "C 0.0 0.0 0.0; H 0.0 0.0 2.13713"
CHPLUS_BASIS = str(Path(__file__).parents[1].absolute() / "data" / "chplus" / "olsen_chplus.gbs")


def _run_and_check(driver):
    driver.system.print_info()

    # Perform CCSD for the closed-shell core
    driver.run_cc(method="ccsd")
    # Obtain the CCSD-level similarity-transformed Hamiltonian
    driver.run_hbar(method="ccsd")

    # Set the number of active unoccupied orbitals used to define the r_{Abc}^{jk} operator
    driver.system.set_active_space(nact_unoccupied=3, nact_occupied=0)

    # Perform guess vectors by diagonalizaing within the 1p + active 2p-1h space
    driver.run_guess(method="eacisd", multiplicity=-1, nact_occupied=3, nact_unoccupied=8,
                     roots_per_irrep={"A1": 2, "B1": 2, "B2": 2, "A2": 2})
    # Loop over all guess vectors and perform the EA-EOMCSDt calculation
    syms = ["A1", "A1", "B1", "B1", "B2", "B2", "A2", "A2"]
    for i in [0, 1, 2, 3, 4, 5, 6]:
        # Obtain the active-space 3p2h list
        r3_excitations = get_active_3p2h_pspace(driver.system, num_active=1, target_irrep=syms[i])

        driver.run_eaeomccp(method="eaeom3_p", state_index=i, r3_excitations=r3_excitations)
        driver.run_lefteaeomccp(method="left_eaeom3_p", state_index=i, r3_excitations=r3_excitations)
        driver.run_eaccp3(method="eaccp3", state_index=i, r3_excitations=r3_excitations)

    #
    # Check the results
    #
    expected_vee = [-0.26354074, -0.22337561, -0.37674145, -0.08128328, -0.37674145, -0.08128328, -0.35640329]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i], vee)


@pytest.mark.short
def test_eaeom3a_chplus():
    # Obtain the Driver from GAMESS logfile and FCIDUMP
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    _run_and_check(driver)


@pytest.mark.short
def test_eaeom3a_chplus_native():
    # native ccpy ROHF (GAMESS SCFTYP=ROHF, ISPHER=-1, UNITS=BOHR) + tight Cholesky integrals
    mol = Molecule(CHPLUS_GEOMETRY, CHPLUS_BASIS, unit="bohr", charge=1, multiplicity=1,
                   symmetry=True, cartesian=True, cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = ROHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    _run_and_check(driver)


if __name__ == "__main__":
    test_eaeom3a_chplus()
    test_eaeom3a_chplus_native()
