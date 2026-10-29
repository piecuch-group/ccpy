""" CC3 computation for the stretched HF molecule at interatomic
separation R = 1.6 angstrom."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver, Molecule, RHF

def _run_and_check(driver):
    driver.options["RHF_symmetry"] = False
    driver.run_cc(method="cc3")

    driver.run_hbar(method="cc3")
    driver.run_guess(method="cis", roots_per_irrep={"A1": 0, "B1": 1, "B2": 0, "A2": 0}, multiplicity=1, use_symmetry=False)
    driver.run_eomcc(method="eomcc3", state_index=[1])

    # Check the CC3 correlation energy
    assert np.allclose(driver.correlation_energy, -0.3047279747, atol=1.0e-07) 
    # Check the CC3 total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -75.87813718, atol=1.0e-07)
    # Check the CC3 vertical excitation energy of 1st singlet excited state without symmetry control
    assert np.allclose(driver.vertical_excitation_energy[1], 0.0282254899, atol=1.0e-07) 
    # Check the CC3 total energy of 1st singlet excited state
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[1], -75.84991169, atol=1.0e-07) 

def test_cc3_h2o():
    geometry = [["O", (0.0, 0.0, -0.0180)],
                ["H", (0.0, 3.030526, -2.117796)],
                ["H", (0.0, -3.030526, -2.117796)]]
    mol = gto.M(
        atom=geometry,
        basis="6-31g",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=True,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    _run_and_check(driver)

def test_cc3_h2o_native():
    # native ccpy SCF + tight Cholesky integrals
    mol = Molecule("O 0.0 0.0 -0.0180; H 0.0 3.030526 -2.117796; H 0.0 -3.030526 -2.117796",
                   "6-31g", unit="bohr", charge=0, multiplicity=1, symmetry=True, cartesian=True,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    _run_and_check(driver)


if __name__ == "__main__":
    test_cc3_h2o()
    test_cc3_h2o_native()
