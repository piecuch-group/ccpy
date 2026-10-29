"""CC(t;3) computation on open-shell CH molecule."""

import pytest
from pathlib import Path
import numpy as np
from ccpy import Driver, get_active_triples_pspace
from ccpy import Molecule, ROHF

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def _run_and_check(driver):
    driver.system.set_active_space(nact_occupied=1, nact_unoccupied=2)
    driver.system.print_info()

    t3_excitations = get_active_triples_pspace(driver.system, target_irrep="B2")
    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="cct3")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -38.2713247488, atol=1.0e-07)
    # Check CCSDt energy
    assert np.allclose(driver.correlation_energy, -0.11469532, atol=1.0e-07)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -38.38602007, atol=1.0e-07
    )
    # Check CC(t;3)_A energy
    assert np.allclose(
        driver.correlation_energy + driver.deltap3[0]["A"], -0.1160181452, atol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["A"],
        -38.3873428940,
        atol=1.0e-07
    )
    # Check CC(t;3)_D energy
    assert np.allclose(
        driver.correlation_energy + driver.deltap3[0]["D"], -0.1162820915, atol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["D"],
        -38.3876068402,
        atol=1.0e-07
    )

@pytest.mark.short
def test_cct3_ch():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/ch/ch.log",
        fcidump=TEST_DATA_DIR + "/ch/ch.FCIDUMP",
        nfrozen=1,
    )
    _run_and_check(driver)

@pytest.mark.short
def test_cct3_ch_native():
    # native ccpy SCF + tight Cholesky AO integrals (no PySCF/GAMESS input)
    mol = Molecule("C 0.0 0.0 0.0; H 0.0 0.0 2.1160902167", "aug-cc-pvdz", unit="bohr",
                   charge=0, multiplicity=2, symmetry=True, cartesian=False,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    # GAMESS singly occupies the B2 component of the degenerate pi pair
    mf = ROHF(mol, canonicalization="gamess", verbose=0, e_tol=1.0e-12, d_tol=1.0e-10, target_symmetry="B2")
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=1)
    _run_and_check(driver)

if __name__ == "__main__":
    test_cct3_ch()
    test_cct3_ch_native()
