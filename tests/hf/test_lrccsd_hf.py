""" LR-CCSD computation for the stretched HF molecule at interatomic
separation R = 1.6 angstrom."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver
from ccpy.interfaces.pyscf_tools import get_multipole_integral

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

if __name__ == "__main__":
    test_lrccsd_hf()
