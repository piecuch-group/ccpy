"""
Active-space IP-EOMCCSD(3h-2p){No} (also known as IP-EOMCCSDt) calculation to
obtain the vertical excitation spectrum of the open-shell OH radical, as
by described by the 6-31G** basis set, by removing one electron from the
(OH)- closed shell.
Reference: J. Chem. Phys. 123, 134113 (2005)
"""

import numpy as np
from pyscf import scf, gto
from ccpy.drivers.driver import Driver
from ccpy.utilities.pspace import get_active_3h2p_pspace

def test_ipeom3a_ohminus():
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
    driver.system.print_info()

    # Perform CCSD for the closed-shell core
    driver.run_cc(method="ccsd")
    # Obtain the CCSD-level similarity-transformed Hamiltonian
    driver.run_hbar(method="ccsd")

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
        driver.run_ipeomccp(method="ipeom3_p", state_index=istate, r3_excitations=r3_excitations)
        driver.run_leftipeomccp(method="left_ipeom3_p", state_index=istate, r3_excitations=r3_excitations)
        driver.run_ipccp3(method="ipccp3", state_index=istate, r3_excitations=r3_excitations)

    #
    # Check the results
    #
    expected_vee = [-0.01583013, 0.41171370, 0.14362613, 0.38859946, 0.43476381, 0.67650621, 0.29502520, 0.33377185]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i], vee)

if __name__ == "__main__":
    test_ipeom3a_ohminus()
