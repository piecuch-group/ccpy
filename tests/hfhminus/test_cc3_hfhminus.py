""" CC3 computation for the open-shell (HFH)- molecule in the triplet
ground state using the D2H-symmetric H-F distance of 2.0 angstrom described
using the 6-31g(1d,1p) basis set. Active space consists of (2,2).
Reference: J. Chem. Theory Comput. 8, 4968 (2012)
"""

import pytest
from pathlib import Path
import numpy as np
from ccpy import Driver
from ccpy import Molecule, ROHF

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def _run_and_check(driver):
    driver.system.print_info()

    driver.run_cc(method="cc3")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -100.3591573557, atol=1.0e-07)
    # Check CC3 energy
    assert np.allclose(driver.correlation_energy, -0.19371334, atol=1.0e-07)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -100.55287069, atol=1.0e-07)

def test_cc3_hfhminus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/hfhminus-triplet/hfhminus-triplet.log",
        fcidump=TEST_DATA_DIR + "/hfhminus-triplet/hfhminus-triplet.FCIDUMP",
        nfrozen=1,
    )
    _run_and_check(driver)

def test_cc3_hfhminus_native():
    # native ccpy SCF + tight Cholesky AO integrals (no PySCF/GAMESS input)
    mol = Molecule("F 0.0 0.0 0.0; H 0.0 0.0 -3.7794519754; H 0.0 0.0 3.7794519754", "6-31g**",
                   unit="bohr", charge=-1, multiplicity=3, symmetry=True, cartesian=False,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = ROHF(mol, canonicalization="gamess", verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=1)
    _run_and_check(driver)

if __name__ == "__main__":
    test_cc3_hfhminus()
    test_cc3_hfhminus_native()
