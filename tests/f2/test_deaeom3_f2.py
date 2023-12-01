"""DEA-EOMCCSD(3p-1h) computation used to describe the spectrum of F2
by attaching two electrons to the doubly ionized F2^(2+) cation."""

import numpy as np
from pyscf import gto, scf
from ccpy.drivers.driver import Driver


def test_deaeom2_f2():

    geometry = [["F", (0.0, 0.0, -2.66816)], ["F", (0.0, 0.0, 2.66816)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvdz",
        charge=2,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2)
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="deacis", nact_unoccupied=4, roots_per_irrep={"Ag": 5}, multiplicity=1)
    driver.run_deaeomcc(method="deaeom3", state_index=[0, 1, 2])

    expected_vee = [-1.42489070,
                    -1.39495074,
                    -1.39407860]

    expected_total_energy = [-199.02088039,
                             -198.99094042,
                             -198.99006829]

    # Check that the CCSD energy of F2^{2+} is correct
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -197.59598969, atol=1.0e-07)
    for i in [0, 1, 2]:
        # Check that DEA-EOMCC EA energy is correct 
        assert np.allclose(driver.vertical_excitation_energy[i], expected_vee[i], atol=1.0e-07)
        # Check that total state energy is correct
        assert np.allclose(driver.vertical_excitation_energy[i]
                          +driver.system.reference_energy + driver.correlation_energy, expected_total_energy[i], atol=1.0e-07)

if __name__ == "__main__":
    test_deaeom2_f2()
