import numpy as np
from pyscf import gto, scf
from ccpy import Driver

def test_dipeom4_cl2():

    basis = '6-31g'
    nfrozen = 10

    geom = [["Cl", (0.0, 0.0, 0.0)],
            ["Cl", (0.0, 0.0, 1.9870)]]

    mol = gto.M(atom=geom, basis=basis, spin=0, symmetry="D2H", unit="Angstrom")
    mf = scf.RHF(mol).x2c()
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=nfrozen)
    driver.system.print_info()
    driver.fock = driver.hamiltonian

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsdta")
    driver.run_guess(method="dipcis", multiplicity=-1, roots_per_irrep={"B1G": 10}, nact_occupied=7, use_symmetry=False)
    driver.run_dipeomcc(method="dipeom3", state_index=[1])
    driver.run_dipccp4(method="dipeom4star", state_index=[1])

if __name__ == "__main__":
    test_dipeom4_cl2()
