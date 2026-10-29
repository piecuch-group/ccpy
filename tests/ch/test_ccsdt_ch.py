"""CCSDT computation on open-shell CH molecule."""

import pytest
from pathlib import Path
import numpy as np
from ccpy import Driver
from ccpy import Molecule, ROHF

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def _run_and_check(driver):
    driver.system.print_info()

    driver.options["RHF_symmetry"] = False
    driver.run_cc(method="ccsdt")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -38.2713247488, atol=1.0e-07)
    # Check CCSDT energy
    assert np.allclose(driver.correlation_energy, -0.1164237849, atol=1.0e-07)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -38.3877485336, atol=1.0e-07
    )

@pytest.mark.short
def test_ccsdt_ch():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/ch/ch.log",
        fcidump=TEST_DATA_DIR + "/ch/ch.FCIDUMP",
        nfrozen=1,
    )
    _run_and_check(driver)

@pytest.mark.short
def test_ccsdt_ch_native():
    # native ccpy SCF + tight Cholesky AO integrals (no PySCF/GAMESS input)
    mol = Molecule("C 0.0 0.0 0.0; H 0.0 0.0 2.1160902167", "aug-cc-pvdz", unit="bohr",
                   charge=0, multiplicity=2, symmetry=True, cartesian=False,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    # GAMESS singly occupies the B2 component of the degenerate pi pair
    mf = ROHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10, target_symmetry="B2")
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=1)
    _run_and_check(driver)

if __name__ == "__main__":
    test_ccsdt_ch()
    test_ccsdt_ch_native()
