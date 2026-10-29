"""
IP-EOMCCSD(2h-1p) calculation to obtain the vertical excitation spectrum of
the open-shell OH radical, as by described by the 6-31G** basis set, by removing
one electron from the (OH)- closed shell.
Reference: J. Chem. Phys. 123, 134113 (2005)
"""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver
from ccpy import Molecule, RHF

def _run_and_check(driver):
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="ipcisd", multiplicity=-1, nact_occupied=4, nact_unoccupied=6,
                     roots_per_irrep={"B1": 2, "B2": 0, "A1": 4, "A2": 2})
    driver.run_ipeomcc(method="ipeom2", state_index=[0, 1, 2, 3, 4, 5, 6, 7])
    driver.run_leftipeomcc(method="left_ipeom2", state_index=[0, 1, 2, 3, 4, 5, 6, 7])

    #
    # Check the results
    #
    expected_vee = [-0.02049758, 0.56909275, 0.14122875, 0.60699750, 0.61645987, 0.70904961, 0.51783683, 0.58412700]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i], vee)
        en = driver.vertical_excitation_energy[i] - driver.vertical_excitation_energy[0]
        print(f"root {i} = {en*27.2114} eV")

def test_ipeom2_ohminus():
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

def test_ipeom2_ohminus_native():
    # native ccpy SCF + tight Cholesky AO integrals (no PySCF/GAMESS input)
    mol = Molecule("O 0.0 0.0 -0.48483; H 0.0 0.0 0.48483", "6-31g**", unit="angstrom",
                   charge=-1, multiplicity=1, symmetry=True, cartesian=False,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=1)
    _run_and_check(driver)

if __name__ == "__main__":
    test_ipeom2_ohminus()
    test_ipeom2_ohminus_native()
