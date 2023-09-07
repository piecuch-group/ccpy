"""IP-EOMCCSD(2h-1p) computation to describe the spectrum of the open-shell
OH molecule described by ionizing an electron from the closed-shell (OH)- ion."""

import numpy as np
from pyscf import scf, gto
from ccpy.drivers.driver import Driver

def test_ipeom2_ohminus():
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

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="ipcis", multiplicity=2, nroot=5, debug=False)
    driver.run_ipeomcc(method="ipeom2", state_index=[0])

    driver.run_leftipeomcc(method="left_ipeom2", state_index=[0])

if __name__ == "__main__":
    test_ipeom2_ohminus()
