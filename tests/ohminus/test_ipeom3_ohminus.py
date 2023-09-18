"""IP-EOMCCSD(3h-2p) computation to describe the spectrum of the open-shell
OH molecule described by ionizing an electron from the closed-shell (OH)- ion."""

import numpy as np
from pyscf import scf, gto
from ccpy.drivers.driver import Driver

def test_ipeom3_ohminus():
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
    driver.run_guess(method="ipcis", multiplicity=2, roots_per_irrep={"A1": 5, "B1": 0, "B2": 0, "A2": 0}, debug=False, use_symmetry=False)
    driver.run_ipeomcc(method="ipeom3", state_index=[0, 1, 2, 3, 4])

if __name__ == "__main__":
    test_ipeom3_ohminus()