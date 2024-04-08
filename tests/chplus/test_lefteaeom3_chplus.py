"""EA-EOMCCSD(3p-2h) computation used to describe the spectrum of the
open-shell CH molecule by attaching an electron to closed-shell CH+."""

import numpy as np
from pyscf import gto, scf
from ccpy.drivers.driver import Driver

def test_eaeom3_chplus():
    mol = gto.M(atom=[['C', (0.0, 0.0, 2.1773/2)], 
                      ['H', (0.0, 0.0, -2.1773/2)]],
                basis="6-31g",
                charge=1,
                unit="Bohr",
                symmetry="C2V")
    mf = scf.RHF(mol)
    mf.kernel()
    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    # driver.run_guess(method="eacisd", multiplicity=-1, nact_occupied=3, nact_unoccupied=8,
    #                  roots_per_irrep={"A1": 2, "B1": 2, "B2": 2, "A2": 2})
    # driver.run_eaeomcc(method="eaeom3", state_index=[0, 1, 2, 3, 4, 5, 6])
    driver.run_guess(method="eacisd", multiplicity=-1, nact_occupied=0, nact_unoccupied=0,
                     roots_per_irrep={"B1": 2, "A1": 4})
    driver.run_eaeomcc(method="eaeom3", state_index=[0])
    #driver.run_lefteaeomcc(method="left_eaeom3", state_index=[0])

    #L = driver.L[0]
    #print("norm of l3a = ", np.linalg.norm(L.aaa.flatten()))
    #
    # Check the results
    #
    # expected_vee = [-0.26411607, -0.22435764, -0.37815570, -0.08130247, -0.37815570, -0.08130247, -0.35696964]
    # for i, vee in enumerate(expected_vee):
    #     assert np.allclose(driver.vertical_excitation_energy[i], vee)

if __name__ == "__main__":
    test_eaeom3_chplus()
