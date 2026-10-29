"""Closed-shell CCSDTQ calculation for the symmetrically stretched
H2O molecule with R(OH) = 2Re, where Re = 1.84345 bohr, described
using the Dunning DZ basis set.
Reference: Mol. Phys, 115, 2860 (2017)."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver, Molecule, RHF

def _run_and_check(driver):
    driver.system.print_info()

    driver.run_cc(method="ccsdt")
    driver.run_hbar(method="ccsdt")
    driver.run_leftcc(method="left_ccsdt")
    driver.run_ccp4(method="crcc34")

    #
    # Check the results
    #
    # Check that CCSDT total energy is correct
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -75.9530698378, atol=1.0e-07)
    # Check that CR-CC(3,4)_A = CCSDT(2)_Q total energy is correct
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap4[0]["A"], -75.9521955199, atol=1.0e-07)

def test_crcc34_h2o():
    # 2 Re
    geometry = [["O", (0.0, 0.0, -0.0180)],
                ["H", (0.0, 3.030526, -2.117796)],
                ["H", (0.0, -3.030526, -2.117796)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    _run_and_check(driver)

def test_crcc34_h2o_native():
    # native ccpy SCF + tight Cholesky integrals
    mol = Molecule("O 0.0 0.0 -0.0180; H 0.0 3.030526 -2.117796; H 0.0 -3.030526 -2.117796",
                   "cc-pvdz", unit="bohr", charge=0, multiplicity=1, symmetry=True, cartesian=False,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    _run_and_check(driver)


if __name__ == "__main__":
    test_crcc34_h2o()
    test_crcc34_h2o_native()
