"""
IP-EOMCCSD(2h-1p) calculation to obtain the vertical excitation spectrum of
the open-shell OH radical, as by described by the 6-31G** basis set, by removing
one electron from the (OH)- closed shell.
Reference: J. Chem. Phys. 123, 134113 (2005)
"""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver

def test_cripcc23_ohminus():
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
    driver.run_guess(method="ipcisd", multiplicity=-1, nact_occupied=4, nact_unoccupied=6,
                     roots_per_irrep={"B1": 2, "B2": 0, "A1": 4, "A2": 2})
    driver.run_ipeomcc(method="ipeom2", state_index=[0, 1, 2, 3, 4, 5, 6, 7])
    driver.run_leftipeomcc(method="left_ipeom2", state_index=[0, 1, 2, 3, 4, 5, 6, 7])
    driver.run_ipccp3(method="cripcc23", state_index=[0, 1, 2, 3, 4, 5, 6, 7])

    #
    # Check the results
    #
    expected_vee = [-0.02049758, 0.56909275, 0.14122875, 0.60699750, 0.61645987, 0.70904961, 0.51783683, 0.58412700]
    expected_crcc23 = [-75.5437471773, -75.1404066669, -75.3840902578, -75.1776050392, -75.0853359523, -74.8867671659, -75.2687622370, -75.2408992393]
    for i, (vee, veep3) in enumerate(zip(expected_vee, expected_crcc23)):
        assert np.allclose(driver.vertical_excitation_energy[i], vee, atol=1.0e-07, rtol=1.0e-07)
        assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[i] + driver.deltap3[i]["D"], veep3, atol=1.0e-07, rtol=1.0e-07)


if __name__ == "__main__":
    test_cripcc23_ohminus()
