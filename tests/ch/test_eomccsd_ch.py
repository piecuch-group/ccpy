"""EOMCCSD computation for the open-shell CH molcule."""

import pytest
from pathlib import Path
import numpy as np
from ccpy import Driver
from ccpy import Molecule, ROHF

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def _run_and_check(driver):
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="cis", multiplicity=2, roots_per_irrep={"A1": 1, "B1": 1, "B2": 0, "A2": 1})
    driver.run_eomcc(method="eomccsd", state_index=[1, 2, 3])

    expected_vee = [0.0, 0.12122133, 0.00016954, 0.11798458]
    expected_total_energy = [-38.38631169 + omega for omega in expected_vee]

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -38.27132475, atol=1.0e-07)
    for n in range(4):
        if n == 0:
            # Check CCSD energy
            assert np.allclose(driver.correlation_energy, -0.11498694, atol=1.0e-07)
            assert np.allclose(
                driver.system.reference_energy + driver.correlation_energy, -38.38631169, atol=1.0e-07
            )
        else:
            # Check EOMCCSD energy
            assert np.allclose(driver.vertical_excitation_energy[n], expected_vee[n], atol=1.0e-07)
            assert np.allclose(
                driver.system.reference_energy
                + driver.correlation_energy
                + driver.vertical_excitation_energy[n],
                expected_total_energy[n], atol=1.0e-07
            )

@pytest.mark.short
def test_eomccsd_ch():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/ch/ch.log",
        fcidump=TEST_DATA_DIR + "/ch/ch.FCIDUMP",
        nfrozen=0,
    )
    _run_and_check(driver)

@pytest.mark.short
def test_eomccsd_ch_native():
    # native ccpy SCF + tight Cholesky AO integrals (no PySCF/GAMESS input)
    mol = Molecule("C 0.0 0.0 0.0; H 0.0 0.0 2.1160902167", "aug-cc-pvdz", unit="bohr",
                   charge=0, multiplicity=2, symmetry=True, cartesian=False,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    # GAMESS singly occupies the B2 component of the degenerate pi pair
    mf = ROHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10, target_symmetry="B2")
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    _run_and_check(driver)

if __name__ == "__main__":
    test_eomccsd_ch()
    test_eomccsd_ch_native()
