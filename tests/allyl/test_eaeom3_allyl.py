import numpy as np
from pyscf import gto, scf
from ccpy.drivers.driver import Driver

def test_eaeom3_allyl():

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
    driver.run_guess(method="eacisd", nact_occupied=5, nact_unoccupied=8, multiplicity=-1,
                     roots_per_irrep={"A2": 1, "B1": 1, "A1": 1})
    driver.run_eaeomcc(method="eaeom3", state_index=[0, 1, 2])

    #
    # Check the results
    #
    expected_vee = [-0.28237593, -0.14922026, -0.00087619]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i], vee)

if __name__ == "__main__":
    test_eaeom3_allyl()
