"""Active-space IP-EOMCCSD(3h-2p){No} (also known as IP-EOMCCSDt) calculation to
describe the vertical excitation spectrum of the open-shell OH radical by removing
one electron from the (OH)- closed shell."""

import numpy as np
from pyscf import scf, gto
from ccpy.drivers.driver import Driver
from ccpy.utilities.pspace import get_active_3h2p_pspace

def test_ipeom3a_ohminus():
    mol = gto.M(atom='''O  0.0  0.0  -0.8
                        H  0.0  0.0   0.8''',
                basis="cc-pvdz",
                charge=-1,
                spin=0,
                cart=False,
                symmetry="C2V")
    mf = scf.RHF(mol)
    mf.kernel()
    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()

    # Perform CCSD for the closed-shell core
    driver.run_cc(method="ccsd")
    # Obtain the CCSD-level similarity-transformed Hamiltonian
    driver.run_hbar(method="ccsd")

    # Set the number of active occupied orbitals used to define the r_{bc}^{ijK} operator
    driver.system.set_active_space(nact_unoccupied=0, nact_occupied=3)
    # Obtain the active-space 3h2p list
    r3_excitations = get_active_3h2p_pspace(driver.system, num_active=1)

    # Perform guess vectors by diagonalizaing within the 1h space (no restriction on spatial symmetry is used)
    driver.run_guess(method="ipcis", multiplicity=2,
                     roots_per_irrep={"A1": 5, "B1": 0, "B2": 0, "A2": 0}, use_symmetry=False)
    # Loop over all guess vectors and perform the IP-EOMCSDt calculation
    for istate in [0, 1, 2, 3, 4]:
        driver.run_ipeomccp(method="ipeom3_p", state_index=istate, r3_excitations=r3_excitations)

    #
    # Check the results
    #
    expected_vee = [-0.00814417, -0.00814417, 0.11305131, 0.62231423, 19.46549731]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i], vee)

if __name__ == "__main__":
    test_ipeom3a_ohminus()
