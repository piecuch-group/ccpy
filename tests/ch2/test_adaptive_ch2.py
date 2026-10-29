from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver
from ccpy.drivers.adaptive import AdaptEOMDriverSS
from ccpy import Molecule, RHF

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def _run_and_check(driver):
    driver.system.print_info()

    adaptdriver = AdaptEOMDriverSS(driver, state_index=1, roots_per_irrep={"B1": 1}, multiplicity=1,
                                   nacto=3, nactu=3,
                                   percentage=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    adaptdriver.options["energy_tolerance"] = 1.0e-09

    adaptdriver.run()

def test_adaptive_ch2():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/ch2/ch2-avdz-koch.log",
        fcidump=TEST_DATA_DIR + "/ch2/ch2-avdz-koch.FCIDUMP",
        nfrozen=0,
    )
    _run_and_check(driver)

def test_adaptive_ch2_native():
    # native ccpy SCF + tight Cholesky AO integrals (no PySCF/GAMESS input)
    mol = Molecule("C 0.0 0.0 0.0; H 0.0 -1.644403 1.32213; H 0.0 1.644403 1.32213",
                   TEST_DATA_DIR + "/ch2/ch2-avdz-koch.gbs", unit="bohr",
                   charge=0, multiplicity=1, symmetry=True, cartesian=False,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    _run_and_check(driver)

if __name__ == "__main__":
    test_adaptive_ch2()
    test_adaptive_ch2_native()
