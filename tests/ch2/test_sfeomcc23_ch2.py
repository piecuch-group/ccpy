import numpy as np
from pyscf import gto, scf
from ccpy.drivers.driver import Driver

def test_sfeomcc23_ch2():

    mol = gto.M(atom=[["C", (0.0, 0.0, 0.0)],
                      ["H", (0.0, 1.644403, -1.32213)],
                      ["H", (0.0, -1.644403, -1.32213)]],
                basis="6-31g",
                charge=0,
                symmetry="C2V",
                cart=True,
                spin=2,
                unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="sfcis", multiplicity=1, roots_per_irrep={"A1": 2}, nact_occupied=0, nact_unoccupied=0)
    driver.run_sfeomcc(method="sfeomcc23", state_index=[0, 1])

if __name__ == "__main__":
    test_sfeomcc23_ch2()