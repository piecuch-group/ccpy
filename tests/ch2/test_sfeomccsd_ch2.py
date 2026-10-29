import numpy as np
from pyscf import gto, scf
from ccpy.drivers.driver import Driver
from ccpy import Molecule, ROHF

def _run_and_check(driver):
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="sfcis", multiplicity=1, roots_per_irrep={"A1": 2}, nact_occupied=0, nact_unoccupied=0)
    driver.run_sfeomcc(method="sfeomccsd", state_index=[0, 1])

def test_sfeomccsd_ch2():

    mol = gto.M(atom=[["C", (0.0, 0.0, 0.0)],
                      ["H", (0.0, 1.644403, -1.32213)],
                      ["H", (0.0, -1.644403, -1.32213)]],
                basis="cc-pVDZ",
                charge=0,
                symmetry="C2V",
                cart=True,
                spin=2,
                unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    _run_and_check(driver)

def test_sfeomccsd_ch2_native():
    # native ccpy SCF + tight Cholesky AO integrals (no PySCF/GAMESS input)
    mol = Molecule("C 0.0 0.0 0.0; H 0.0 1.644403 -1.32213; H 0.0 -1.644403 -1.32213", "cc-pvdz",
                   unit="bohr", charge=0, multiplicity=3, symmetry=True, cartesian=True,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = ROHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    _run_and_check(driver)

if __name__ == "__main__":
    test_sfeomccsd_ch2()
    test_sfeomccsd_ch2_native()
