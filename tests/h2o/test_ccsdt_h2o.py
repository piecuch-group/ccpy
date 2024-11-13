"""Closed-shell CCSDT calculation for the symmetrically stretched
H2O molecule with R(OH) = 2Re, where Re = 1.84345 bohr, described
using the Dunning DZ basis set.
Reference: Mol. Phys, 115, 2860 (2017)."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver

def test_ccsdt_h2o():

    mol = gto.M(
        atom=[["O", (0.0, 0.0, -0.0180)],
              ["H", (0.0, 3.030526, -2.117796)],
              ["H", (0.0, -3.030526, -2.117796)]],
        basis="dz",
        symmetry="C2v",
        charge=0,
        spin=0,
        cart=False,
        unit="Bohr",
        verbose=5,
    )
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-14
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0, uhf=True)
    driver.system.print_info()

    # Important: Set RHF flag to false manually for UHF references
    driver.options["RHF_symmetry"] = False
    driver.options["energy_convergence"] = 1.0e-09
    driver.options["amp_convergence"] = 1.0e-08
    driver.options["maximum_iterations"] = 80

    driver.run_cc(method="ccsdt")
    driver.run_hbar(method="ccsdt")
    driver.run_guess(method="cisd", roots_per_irrep={"B1": 2, "A1": 2}, multiplicity=-1, nact_occupied=3, nact_unoccupied=3)
    driver.run_eomcc(method="eomccsdt", state_index=[1, 2, 3, 4])

    #
    # Check the results
    #
    expected_vee = [0.02728992, 0.03205095, 0.03301541, 0.14402798]
    assert np.allclose(driver.correlation_energy, -0.31227673, rtol=1.0e-07, atol=1.0e-07)
    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i + 1], expected_vee[i], rtol=1.0e-07, atol=1.0e-7)

if __name__ == "__main__":
    test_ccsdt_h2o()
