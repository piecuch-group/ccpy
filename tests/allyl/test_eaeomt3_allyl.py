import numpy as np
from pyscf import gto, scf

from ccpy.drivers.driver import Driver
from ccpy.utilities.pspace import get_active_3p2h_pspace

def test_eaeom3a_allyl():

    mol = gto.M(
            atom='''C 0.00000000 0.00000000 0.83050732
                    C 0.00000000 2.30981224 -0.38722841
                    C 0.00000000 -2.30981224 -0.38722841
                    H 0.00000000 0.00000000 2.87547067
                    H 0.00000000 4.06036949 0.65560561
                    H 0.00000000 -4.06036949 0.65560561
                    H 0.00000000 2.41059890 -2.42703281
                    H 0.00000000 -2.41059890 -2.42703281''',
            basis="6-31g",
            spin=0,
            charge=1,
            symmetry="C2V",
            unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=3)
    driver.system.print_info()
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")

    driver.system.set_active_space(nact_occupied=0, nact_unoccupied=2)

    irreps = ["A2", "B1", "A1"]
    nroots = [1, 1, 1]
    driver.run_guess(method="eacisd", nact_occupied=5, nact_unoccupied=8, multiplicity=-1,
                     roots_per_irrep=dict(zip(irreps, nroots)))

    for i, sym in enumerate(irreps):
        r3_excitations = get_active_3p2h_pspace(driver.system, target_irrep=sym)
        driver.run_eaeomccp(method="eaeom3_p", state_index=i, r3_excitations=r3_excitations)
        driver.run_lefteaeomccp(method="left_eaeom3_p", state_index=i, r3_excitations=r3_excitations)
        driver.run_eaccp3(method="eaccp3", state_index=i, r3_excitations=r3_excitations)

    #
    # Check the results
    #
    expected_vee = [-0.28050484, -0.14827956, 0.00067634]
    expected_deltaD = [-0.0019126694, -0.0009361803, -0.0014588215]
    for i, (vee, vee_d) in enumerate(zip(expected_vee, expected_deltaD)):
        assert np.allclose(driver.vertical_excitation_energy[i], vee)
        assert np.allclose(driver.deltap3[i]["D"], vee_d)

if __name__ == "__main__":

    test_eaeom3a_allyl()
