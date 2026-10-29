""" LR-CCSD computation for the stretched HF molecule at interatomic
separation R = 1.6 angstrom."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver, Molecule, RHF
from ccpy.interfaces.pyscf_tools import get_multipole_integral
from ccpy.interfaces import scf_tools

def _run_and_check(driver, mu, mu_ref):
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")

    mu_cc = np.zeros(3)
    for i in range(3):
        driver.run_lrcc(method="lrccsd", prop=mu[i])
        mu_cc[i] = driver.correlation_property
    mu_cc += mu_ref

    print("")
    print("   Reference Dipole Moment")
    print(f"  x: {mu_ref[0]}") 
    print(f"  y: {mu_ref[1]}")
    print(f"  z: {mu_ref[2]}")
    print("   LR-CCSD Dipole Moment")
    print(f"  x: {mu_cc[0]}") 
    print(f"  y: {mu_cc[1]}")
    print(f"  z: {mu_cc[2]}")

def test_lrccsd_hf():
    geom = [['H', (0.0, 0.0, -1.7330)],
            ['F', (0.0, 0.0,  1.7330)]]
    mol = gto.M(
        atom=geom,
        basis="cc-pvtz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=True,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=1)
    mu, mu_ref = get_multipole_integral(1, mol, mf, driver.system)
    _run_and_check(driver, mu, mu_ref)

def test_lrccsd_hf_native():
    # native ccpy SCF + tight Cholesky integrals; dipole integrals from libint2
    # (the native Molecule may be translated/reoriented to its principal axes, so the
    # printed electronic dipole components need not match the PySCF variant)
    mol = Molecule("H 0.0 0.0 -1.7330; F 0.0 0.0 1.7330", "cc-pvtz", unit="bohr", charge=0, multiplicity=1,
                   symmetry=True, cartesian=True, cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=1)
    mu, mu_ref = scf_tools.get_multipole_integral(1, mf, driver.system)
    _run_and_check(driver, mu, mu_ref)

if __name__ == "__main__":
    test_lrccsd_hf()
    test_lrccsd_hf_native()
