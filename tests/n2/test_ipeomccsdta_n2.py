'''
IP-EOMCCSD(T)(a) for N2 in ANO basis set
D. A. Matthews and J. F. Stanton, J. Chem. Phys 145, 124102 (2016)
'''
from pathlib import Path
import numpy as np
from pyscf import gto, scf
from ccpy import Driver
from pyscf.gto.basis import parse_gaussian
from ccpy import Molecule, RHF

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def _run_and_check(driver):
    state_index = list(range(7))
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsdta")

    driver.run_guess(method="ipcisd", nact_occupied=0, nact_unoccupied=0,
                     multiplicity=2, roots_per_irrep={"AG": 7}, use_symmetry=False)

    driver.options["amp_convergence"] = 1.0e-05
    driver.run_ipeomcc(method="ipeomccsdta", state_index=state_index)

    for istate in state_index:
        print(f"\nState {istate}")
        print("-----")
        print(f"IP Energy = {(driver.vertical_excitation_energy[istate])* 27.2114} eV")


def test_ipeomccsdta_n2():

    ANO0 = {
        'N': parse_gaussian.load(TEST_DATA_DIR + '/n2/ano0.gbs', 'N'),
    }

    geom = [["N",  (0.0,  0.0,  0.0)],
            ["N",  (0.0,  0.0,  1.094)]]


    mol = gto.M(atom=geom,
                basis=ANO0,
                charge=0,
                symmetry="D2H",
                cart=False,
                spin=0,
                unit="Angstrom")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    _run_and_check(driver)


def test_ipeomccsdta_n2_native():
    # native ccpy SCF + tight Cholesky integrals
    mol = Molecule("N 0.0 0.0 0.0; N 0.0 0.0 1.094", TEST_DATA_DIR + "/n2/ano0.gbs", unit="angstrom", charge=0, multiplicity=1,
                   symmetry=True, cartesian=False, cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    _run_and_check(driver)


if __name__ == "__main__":
    test_ipeomccsdta_n2()
    test_ipeomccsdta_n2_native()
