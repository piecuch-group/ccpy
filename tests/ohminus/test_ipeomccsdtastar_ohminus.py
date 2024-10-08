"""
IP-EOMCCSDT(a)* calculation to obtain the vertical excitation spectrum of
the open-shell OH radical, as by described by the 6-31G basis set, by removing
one electron from the (OH)- closed shell.
Reference: J. Chem. Phys. 123, 134113 (2005)
"""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver

def test_ipeomccsdtastar_ohminus():
    mol = gto.M(atom='''O  0.0  0.0  -0.8
                        H  0.0  0.0   0.8''',
                basis="6-31g",
                charge=-1,
                spin=0,
                cart=False,
                symmetry="C2V",
                unit="Angstrom")
    mf = scf.RHF(mol)
    mf.kernel()
    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsdta")
    driver.run_guess(method="ipcisd", multiplicity=-1, nact_occupied=4, nact_unoccupied=6,
                     roots_per_irrep={"B1": 2, "B2": 0, "A1": 4, "A2": 2})
    driver.run_ipeomcc(method="ipeom2", state_index=[0, 1, 2, 3, 4, 5, 6, 7])
    driver.run_leftipeomcc(method="left_ipeom2", state_index=[0, 1, 2, 3, 4, 5, 6, 7])
    driver.run_ipccp3(method="ipeomccsdta_star", state_index=[0, 1, 2, 3, 4, 5, 6, 7])

    #
    # Check the results
    #
    expected_vee = [-0.003347942028767371,
                    0.10492491595761046,
                    0.12214269570376851,
                    0.3465144798386487,
                    0.2437123658196561,
                    0.30641350125651096,
                    0.1501940563414676,
                    0.20678796226069587,
    ]
    for i, vee in enumerate(expected_vee):
       assert np.allclose(driver.vertical_excitation_energy[i] + driver.deltap3[i]["A"], vee)

if __name__ == "__main__":
    test_ipeomccsdtastar_ohminus()
