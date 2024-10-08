"""EA-EOMCCSDT(a)* computation used to describe the spectrum of the
open-shell CH molecule by attaching an electron to closed-shell CH+."""

import numpy as np
from ccpy import Driver
from pyscf import gto, scf

def test_eaeomccsdtastar_chplus():

    # Define molecule geometry and basis set
    basis = '6-31g'
    geom = [["C", (0.0, 0.0, 0.0)],
            ["H", (0.0, 0.0, 2.13713)]]

    mol = gto.M(atom=geom, basis=basis, spin=0, symmetry="C2V", charge=1, unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsdta")
    driver.run_guess(method="eacisd", multiplicity=-1, nact_occupied=3, nact_unoccupied=8,
                     roots_per_irrep={"A1": 2, "B1": 2, "B2": 0, "A2": 2})
    driver.run_eaeomcc(method="eaeom2", state_index=[0, 1, 2, 3, 4, 5])
    driver.run_lefteaeomcc(method="left_eaeom2", state_index=[0, 1, 2, 3, 4, 5])
    driver.run_eaccp3(method="eaeomccsdta_star", state_index=[0, 1, 2, 3, 4, 5])

    #
    # Check the results
    #
    expected_vee = [-0.2328008158291373,
                    -0.1993847903983961,
                    -0.3641487642700113,
                    -0.04101898766682737,
                    -0.34036045427281636,
                    -0.23280077881783232]

    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i] + driver.deltap3[i]["A"], vee, atol=1.0e-06)

if __name__ == "__main__":
    test_eaeomccsdtastar_chplus()
