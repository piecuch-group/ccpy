"""The spin-orbital CR-CC(2,3) evaluator (M_K, N_K, D_K per triple) used for analytic
gradients reproduces ccpy's Fortran delta(2,3) A-D."""
import numpy as np
import pytest

from ccpy import Driver, Molecule, RHF, ROHF, UHF
from crcc23_triples import crcc23_correction

CH2 = "C 0 0 0.1; H 0 0.98 -0.6; H 0 -0.98 -0.6"


@pytest.mark.parametrize("scf, geometry, basis, multiplicity, nfrozen", [
    (RHF, "O 0 0 0.12; H 0 0.76 -0.47; H 0 -0.76 -0.47", "6-31g", 1, 1),
    (UHF, CH2, "6-31g", 3, 1),
    (ROHF, CH2, "6-31g", 3, 1),
    (ROHF, "O 0 0 0; H 0 0 0.97", "6-31g**", 2, 0),
])
def test_spin_orbital_crcc23_matches_fortran(scf, geometry, basis, multiplicity, nfrozen):
    mol = Molecule(geometry, basis, multiplicity=multiplicity, cholesky_tei=True,
                   cholesky_tol=1.0e-12, verbose=False)
    mf = scf(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=nfrozen)
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")
    delta = crcc23_correction(driver)
    for variant in "ABCD":
        assert np.isclose(delta[variant], driver.deltap3[0][variant], atol=1.0e-12, rtol=0.0)
