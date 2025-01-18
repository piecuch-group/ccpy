import numpy as np
from pyscf import gto, scf
from ccpy import Driver, get_active_4p2h_pspace

def test_deaeom4_p_ch2():

    mol = gto.M(atom=[["C", (0.0, 0.0, 0.0)],
                      ["H", (0.0, 1.644403, -1.32213)],
                      ["H", (0.0, -1.644403, -1.32213)]],
                basis="6-31g",
                charge=2,
                symmetry="C2V",
                cart=True,
                spin=0,
                unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.set_active_space(nact_occupied=0, nact_unoccupied=driver.system.nunoccupied_alpha)
    driver.system.print_info()

    driver.options["davidson_out_of_core"] = True
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="deacis", multiplicity=-1, nact_unoccupied=5, roots_per_irrep={"A1": 6}, use_symmetry=False)

    r3_excitations = get_active_4p2h_pspace(driver.system, target_irrep=None)
    for i in [0, 1, 2, 3, 4, 5]:
        driver.run_deaeomccp(method="deaeom4_p", state_index=i, r3_excitations=r3_excitations)

    expected_vee = [-1.20632891, -1.22803218, -1.14348083, -1.04281182, -0.91190576, -0.88424966]
    for i in range(len(expected_vee)):
        assert np.allclose(expected_vee[i], driver.vertical_excitation_energy[i], atol=1.0e-07)

if __name__ == "__main__":
    test_deaeom4_p_ch2()


