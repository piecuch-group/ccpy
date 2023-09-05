"""Adaptive CC(P;Q) aimed at converging CCSDT computation:
F2 / cc-pVDZ at R = 2.0Re, where Re = 2.66816 bohr using RHF.
Cartesian orbitals are used for the d orbitals in the cc-pVTZ basis.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""

import numpy as np
from pyscf import scf, gto
from ccpy.drivers.driver import Driver, AdaptDriver

def test_adaptive_f2():
    geometry = [["F", (0.0, 0.0, -2.66816)], ["F", (0.0, 0.0, 2.66816)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    percentages = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    driver = Driver.from_pyscf(mf, nfrozen=2)
    driver.system.print_info()
    #driver.options["RHF_symmetry"] = False
    adaptdriver = AdaptDriver(
        driver,
        percentages,
        full_storage=False,
        perturbative=False,
        pspace_analysis=False,
        two_body_left=False,
    )
    adaptdriver.run()

if __name__ == "__main__":
    test_adaptive_f2()