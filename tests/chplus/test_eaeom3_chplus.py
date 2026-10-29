"""EA-EOMCCSD(3p-2h) computation used to describe the spectrum of the
open-shell CH molecule by attaching an electron to closed-shell CH+."""

import pytest
from pathlib import Path
import numpy as np
from ccpy import Driver, Molecule, ROHF

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

# Same system as chplus.log: CH+ at R = 2.13713 bohr, Olsen basis (cartesian d)
CHPLUS_GEOMETRY = "C 0.0 0.0 0.0; H 0.0 0.0 2.13713"
CHPLUS_BASIS = str(Path(__file__).parents[1].absolute() / "data" / "chplus" / "olsen_chplus.gbs")


def _run_and_check(driver):
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="eacisd", multiplicity=-1, nact_occupied=3, nact_unoccupied=8,
                     roots_per_irrep={"A1": 2, "B1": 2, "B2": 2, "A2": 2})
    driver.run_eaeomcc(method="eaeom3", state_index=[0, 1, 2, 3, 4, 5, 6])

    #
    # Check the results
    #
    expected_vee = [-0.26411607, -0.22435764, -0.37815570, -0.08130247, -0.37815570, -0.08130247, -0.35696964]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i], vee)


@pytest.mark.short
def test_eaeom3_chplus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    _run_and_check(driver)


@pytest.mark.short
def test_eaeom3_chplus_native():
    # native ccpy ROHF (GAMESS SCFTYP=ROHF, ISPHER=-1, UNITS=BOHR) + tight Cholesky integrals
    mol = Molecule(CHPLUS_GEOMETRY, CHPLUS_BASIS, unit="bohr", charge=1, multiplicity=1,
                   symmetry=True, cartesian=True, cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = ROHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    _run_and_check(driver)


if __name__ == "__main__":
    test_eaeom3_chplus()
    test_eaeom3_chplus_native()
