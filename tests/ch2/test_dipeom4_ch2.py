import numpy as np
from pyscf import gto, scf
from ccpy.drivers.driver import Driver

def test_dipeom4_ch2():

    mol = gto.M(atom=[["C", (0.0, 0.0, 0.0)],
                      ["H", (0.0, 1.644403, -1.32213)],
                      ["H", (0.0, -1.644403, -1.32213)]],
                basis="6-31g",
                charge=-2,
                symmetry="C2V",
                cart=True,
                spin=0,
                unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.options["RHF_symmetry"] = False

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_guess(method="dipcis", multiplicity=-1, nact_occupied=driver.system.noccupied_alpha, roots_per_irrep={"A1": 6})
    driver.run_dipeomcc(method="dipeom4", state_index=[0, 1, 2, 3, 4, 5])

    expected_vee = [-0.47006858, -0.44903598, -0.38539205, -0.28546072, -0.25449691, -0.22891625]
    for i in range(len(expected_vee)):
       assert np.allclose(expected_vee[i], driver.vertical_excitation_energy[i], atol=1.0e-07)

if __name__ == "__main__":
    test_dipeom4_ch2()