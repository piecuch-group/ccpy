import numpy as np
from pyscf import gto, scf
from ccpy.drivers.driver import Driver

def test_dipeom4_f2():

    mol = gto.M(atom=[["F", (0.0, 0.0, -2.66816)],
                      ["F", (0.0, 0.0,  2.66816)]],
                basis="6-31g",
                charge=-2,
                symmetry="D2H",
                cart=True,
                spin=0,
                unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2)

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="dipcis", multiplicity=-1, nact_occupied=driver.system.noccupied_alpha, roots_per_irrep={"AG": 5})

    driver.options["davidson_out_of_core"] = True
    driver.run_dipeomcc(method="dipeom4", state_index=[0, 1])

    #
    # Check the results
    #
    expected_vee = [-0.133359091568, -0.133336529768]
    for i, vee in enumerate(expected_vee):
        assert np.allclose(driver.vertical_excitation_energy[i], vee, atol=1.0e-07)
    

if __name__ == "__main__":
    test_dipeom4_f2()
