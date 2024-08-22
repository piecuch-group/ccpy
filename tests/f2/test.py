""" CC(t;3) computation for the stretched F2 molecule at interatomic
separation R = 2.0Re, where Re = 2.66816 bohr. The cc-pVDZ basis set
is used with Cartesian components for the d orbitals.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver, get_active_triples_pspace
from ccpy.utilities.utilities import unravel_triples_amplitudes

def test_cct3_f2():
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

    driver = Driver.from_pyscf(mf, nfrozen=2)
    driver.system.print_info()
    driver.system.set_active_space(nact_occupied=driver.system.noccupied_alpha, nact_unoccupied=driver.system.nunoccupied_alpha)
    t3_excitations = get_active_triples_pspace(driver.system, target_irrep="AG")
    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    T_new = unravel_triples_amplitudes(driver.T, t3_excitations, driver.system, {"aaa": True, "aab": True, "abb": True, "bbb": True})  

    driver = Driver.from_pyscf(mf, nfrozen=2)
    driver.system.print_info()
    driver.run_cc(method="ccsdt")
    T_old = driver.T


    no = driver.system.noccupied_alpha
    nu = driver.system.nunoccupied_alpha

    for a in range(nu):
        for b in range(a + 1, nu):
            for c in range(b + 1, nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        for k in range(j + 1, no):
                            if abs(T_new.aaa[a, b, c, i, j, k] - T_old.aaa[a, b, c, i, j, k]) > 1.0e-09:
                                print(a, b, c, i, j, k, "expected:", T_old.aaa[a, b, c, i, j, k], "Got:", T_new.aaa[a, b, c, i, j, k], "error:", T_old.aaa[a, b, c, i, j, k] - T_new.aaa[a, b, c, i, j, k])

    for a in range(nu):
        for b in range(a + 1, nu):
            for c in range(nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        for k in range(no):
                            if abs(T_new.aab[a, b, c, i, j, k] - T_old.aab[a, b, c, i, j, k]) > 1.0e-09:
                                print(a, b, c, i, j, k, "expected:", T_old.aab[a, b, c, i, j, k], "Got:", T_new.aab[a, b, c, i, j, k], "error:", T_old.aab[a, b, c, i, j, k] - T_new.aab[a, b, c, i, j, k])




if __name__ == "__main__":
    test_cct3_f2()
