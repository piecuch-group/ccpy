import numpy as np
from pyscf import gto, scf
from ccpy import Driver
from ccpy import Molecule, RHF

def _run_and_check(driver):

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="dipcis", multiplicity=-1, nact_occupied=driver.system.noccupied_alpha, roots_per_irrep={"AG": 5}, use_symmetry=False)

    driver.options["davidson_out_of_core"] = True
    driver.run_dipeomcc(method="dipeom4", state_index=[0, 1])

    #
    # Check the results
    #
    expected_vee = [-0.133359091568, -0.133336529768]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i], vee, atol=1.0e-07)


def test_dipeom4_f2():

    mol = gto.M(atom=[["F", (0.0, 0.0, -2.66816)],
                      ["F", (0.0, 0.0,  2.66816)]],
                basis="6-31g",
                charge=-2,
                symmetry="D2H",
                cart=True,
                spin=0,
                unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2)
    _run_and_check(driver)


def test_dipeom4_f2_native():
    # native ccpy SCF + tight Cholesky integrals
    mol = Molecule("F 0.0 0.0 -2.66816; F 0.0 0.0 2.66816", "6-31g", unit="bohr", charge=-2, multiplicity=1,
                   symmetry=True, cartesian=True, cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=2)
    _run_and_check(driver)


if __name__ == "__main__":
    test_dipeom4_f2()
    test_dipeom4_f2_native()
