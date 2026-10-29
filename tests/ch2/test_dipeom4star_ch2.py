import numpy as np
from pyscf import gto, scf
from ccpy.drivers.driver import Driver
from ccpy import Molecule, RHF

def _run_and_check(driver):
    driver.options["RHF_symmetry"] = False

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsdta")
    driver.run_guess(method="dipcis", multiplicity=-1, nact_occupied=driver.system.noccupied_alpha, roots_per_irrep={"A1": 6}, use_symmetry=False)
    driver.run_dipeomcc(method="dipeom3", state_index=[0, 1])
    driver.run_dipccp4(method="dipeomccsdta_star", state_index=[0, 1])

    #
    # Check the results
    #
    expected_vee = [-0.456636415364, -0.440094626855]
    expected_correction = [-0.021716102147627467, -0.022139900346410033]
    #
    # Check the results
    #
    assert np.allclose(driver.correlation_energy, -0.10559024413233081)
    for i in range(len(expected_vee)):
        assert np.allclose(driver.vertical_excitation_energy[i], expected_vee[i])
        assert np.allclose(driver.deltap4[i]["A"], expected_correction[i])

def test_dipeom4star_ch2():

    mol = gto.M(atom=[["C", (0.0, 0.0, 0.0)],
                      ["H", (0.0, 1.644403, -1.32213)],
                      ["H", (0.0, -1.644403, -1.32213)]],
                basis="6-31g",
                charge=-2,
                symmetry="C2V",
                cart=True,
                spin=0,
                unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    _run_and_check(driver)

def test_dipeom4star_ch2_native():
    # native ccpy SCF + tight Cholesky AO integrals (no PySCF/GAMESS input)
    mol = Molecule("C 0.0 0.0 0.0; H 0.0 1.644403 -1.32213; H 0.0 -1.644403 -1.32213", "6-31g",
                   unit="bohr", charge=-2, multiplicity=1, symmetry=True, cartesian=True,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    _run_and_check(driver)

if __name__ == "__main__":
    test_dipeom4star_ch2()
    test_dipeom4star_ch2_native()
