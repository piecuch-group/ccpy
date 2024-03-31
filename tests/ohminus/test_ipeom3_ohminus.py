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
                symmetry="C2V")
    mf = scf.RHF(mol)
    mf.kernel()
    driver = Driver.from_pyscf(mf, nfrozen=1)
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="ipcis", multiplicity=2, roots_per_irrep={"A1": 4, "B1": 0, "B2": 0, "A2": 0}, use_symmetry=False)
    driver.run_ipeomcc(method="ipeom3", state_index=[0, 1, 2, 3])

    #
    # Check the results
    #
    expected_vee = [-0.01598430, -0.01598430, 0.14381607, 0.66870089]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i], vee)

if __name__ == "__main__":
    test_ipeom3_ohminus()
