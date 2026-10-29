"""
Reference: J. Chem. Phys. 123, 134113 (2005)
"""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver, get_active_triples_pspace, get_active_3h2p_pspace
from ccpy import Molecule, RHF

def _run_and_check(driver):
    driver.system.print_info()

    # Set the active space used to define the t_{Abc}^{ijK} operator
    # No = 2 includes the 1 pi_x and 1 pi_y orbitals in the active space
    driver.system.set_active_space(nact_unoccupied=2, nact_occupied=2)
    # Obtain the active-space 3h2p list
    t3_excitations = get_active_triples_pspace(driver.system, num_active=1)
    # Perform CCSDt for the closed-shell core
    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    # Obtain the CCSD-level similarity-transformed Hamiltonian
    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)

    # Set the number of active occupied orbitals used to define the r_{bc}^{ijK} operator
    # No = 2 includes the 1 pi_x and 1 pi_y orbitals in the active space
    driver.system.set_active_space(nact_unoccupied=0, nact_occupied=2)
    # Obtain the active-space 3h2p list
    r3_excitations = get_active_3h2p_pspace(driver.system, num_active=1)

    # Perform guess vectors by diagonalizaing within the 1h + active 2h-1p space
    driver.run_guess(method="ipcisd", multiplicity=-1, nact_occupied=4, nact_unoccupied=6,
                     roots_per_irrep={"B1": 2, "B2": 0, "A1": 4, "A2": 2})
    # Loop over all guess vectors and perform the IP-EOMCSDt calculation
    for istate in [0, 1, 2, 3, 4, 5, 6, 7]:
        driver.run_ipeomccp(method="ipeomccsdt_p", state_index=istate, r3_excitations=r3_excitations, t3_excitations=t3_excitations)

def test_ipeomt3a_ohminus():
    mol = gto.M(atom='''O  0.0  0.0  -0.96966/2
                        H  0.0  0.0   0.96966/2''',
                basis="6-31g**",
                charge=-1,
                spin=0,
                cart=False,
                symmetry="C2V",
                unit="Angstrom")
    mf = scf.RHF(mol)
    mf.kernel()
    driver = Driver.from_pyscf(mf, nfrozen=1)
    _run_and_check(driver)

def test_ipeomt3a_ohminus_native():
    # native ccpy SCF + tight Cholesky AO integrals (no PySCF/GAMESS input)
    mol = Molecule("O 0.0 0.0 -0.48483; H 0.0 0.0 0.48483", "6-31g**", unit="angstrom",
                   charge=-1, multiplicity=1, symmetry=True, cartesian=False,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=1)
    _run_and_check(driver)

if __name__ == "__main__":
    test_ipeomt3a_ohminus()
    test_ipeomt3a_ohminus_native()
