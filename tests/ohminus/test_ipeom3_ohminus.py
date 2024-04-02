"""
IP-EOMCCSD(3h-2p) calculation to obtain the vertical excitation spectrum
of the open-shell OH radical, as by described by the 6-31G** basis set, by
removing one electron from the (OH)- closed shell.
Reference: J. Chem. Phys. 123, 134113 (2005)
"""

import numpy as np
from pyscf import scf, gto
from ccpy.drivers.driver import Driver

def test_ipeom3_ohminus():
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

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    # Perform guess vectors by diagonalizaing within the 1h + active 2h-1p space
    driver.run_guess(method="ipcisd", multiplicity=-1, nact_occupied=4, nact_unoccupied=6,
                     roots_per_irrep={"B1": 2, "B2": 0, "A1": 4, "A2": 2})
    driver.run_ipeomcc(method="ipeom3", state_index=[0, 1, 2, 3, 4, 5, 6, 7])

    #
    # Check the results
    #
    expected_vee = [-0.01598430, 0.40930363, 0.14381607, 0.38859076, 0.43203093, 0.66870088, 0.29502178, 0.33376960]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i], vee)
        en = driver.vertical_excitation_energy[i] - driver.vertical_excitation_energy[0]
        print(f"Root {i} = {en*27.2114} eV")

if __name__ == "__main__":
    test_ipeom3_ohminus()
