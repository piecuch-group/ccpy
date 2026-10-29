""" CC3 computation for the stretched HF molecule at interatomic
separation R = 1.6 angstrom."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver, Molecule, RHF

def _run_and_check(driver):
    driver.options["RHF_symmetry"] = False
    driver.run_cc(method="cc3")
    driver.run_hbar(method="cc3")
    driver.run_guess(method="cis", roots_per_irrep={"A1": 0, "B1": 1, "B2": 0, "A2": 0}, multiplicity=1)
    driver.run_eomcc(method="eomcc3", state_index=[1])

    # Check CC3 correlation energy
    assert np.allclose(driver.correlation_energy, -0.178932834091, atol=1.0e-07)
    # Check CC3 total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -100.00884041, atol=1.0e-07)
    # Check EOMCC3 vertical excitation energy
    assert np.allclose(driver.vertical_excitation_energy[1], 0.10083870, atol=1.0e-07)
    # Check EOMCC3 total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[1], -99.90800171, atol=1.0e-07)

def test_cc3_hf():
    geometry = [["H", (0.0, 0.0, -0.8)], ["F", (0.0, 0.0, 0.8)]]
    mol = gto.M(
        atom=geometry,
        basis="6-31g",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=True,
        unit="Angstrom",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    _run_and_check(driver)

def test_cc3_hf_native():
    # native ccpy SCF + tight Cholesky integrals
    mol = Molecule("H 0.0 0.0 -0.8; F 0.0 0.0 0.8", "6-31g", unit="angstrom", charge=0, multiplicity=1,
                   symmetry=True, cartesian=True, cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    _run_and_check(driver)

if __name__ == "__main__":
    test_cc3_hf()
    test_cc3_hf_native()
