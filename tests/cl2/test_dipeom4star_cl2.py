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
    driver.run_guess(method="dipcis", multiplicity=-1, roots_per_irrep={"AG": 10}, nact_occupied=driver.system.noccupied_alpha, use_symmetry=False)
    # picks up: 0     -> triplet 3\Sigma_{g}^−
    #           1 & 2 -> singlet \Delta_{g} (2x degenerate) 
    #           3     -> singlet \Sigma_{g}^+
    #           4     -> singlet \Sigma_{u}^-
    driver.run_dipeomcc(method="dipeom3", state_index=[0, 1, 3, 4])
    driver.run_dipccp4(method="dipeom4star", state_index=[0, 1, 3, 4])

if __name__ == "__main__":
    test_dipeom4_cl2()
