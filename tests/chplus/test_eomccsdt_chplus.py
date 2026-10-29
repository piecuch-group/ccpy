import pytest
import numpy as np
from pathlib import Path
from ccpy import Driver, Molecule, ROHF

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

# Same system as chplus.log: CH+ at R = 2.13713 bohr, Olsen basis (cartesian d)
CHPLUS_GEOMETRY = "C 0.0 0.0 0.0; H 0.0 0.0 2.13713"
CHPLUS_BASIS = str(Path(__file__).parent.absolute() / "olsen_chplus.gbs")


def _run_and_check(driver):
    driver.system.print_info()
    driver.run_cc(method="ccsdt")
    driver.run_hbar(method="ccsdt")
    driver.run_guess(method="cisd", multiplicity=1, roots_per_irrep={"A1": 4, "B1": 2, "B2": 0, "A2": 2},  nact_occupied=3, nact_unoccupied=7)

    driver.options["davidson_max_subspace_size"] = 50
    driver.run_eomcc(method="eomccsdt", state_index=[2, 3, 4, 5, 6, 7, 8])

    expected_ref_energy = -37.90276818
    expected_cc_energy = -38.01951563
    expected_corr_energy = -0.11674744
    expected_vee = [0.0, 0.0, 0.31689457, 0.49705838, 0.63264386, 0.11859434, 0.52137274, 0.25740252, 0.61720742]
    expected_total_energy = [expected_cc_energy + omega for omega in expected_vee]

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, expected_ref_energy, atol=1.0e-07)
    for n in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        if n == 0:
            # Check CCSDT energy
            assert np.allclose(driver.correlation_energy, expected_corr_energy, atol=1.0e-07)
            assert np.allclose(
                driver.system.reference_energy + driver.correlation_energy, expected_cc_energy,
                atol=1.0e-07
            )
        else:
            # Check EOMCCSDT energy
            assert np.allclose(driver.vertical_excitation_energy[n], expected_vee[n], atol=1.0e-07)
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.vertical_excitation_energy[n],
                expected_total_energy[n],
                atol=1.0e-07
            )


@pytest.mark.short
def test_eomccsdt_chplus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
        ndelete=0,
    )
    _run_and_check(driver)


@pytest.mark.short
def test_eomccsdt_chplus_native():
    # native ccpy ROHF (GAMESS SCFTYP=ROHF, ISPHER=-1, UNITS=BOHR) + tight Cholesky integrals
    mol = Molecule(CHPLUS_GEOMETRY, CHPLUS_BASIS, unit="bohr", charge=1, multiplicity=1,
                   symmetry=True, cartesian=True, cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = ROHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0, ndelete=0)
    _run_and_check(driver)


if __name__ == "__main__":
    test_eomccsdt_chplus()
    test_eomccsdt_chplus_native()
