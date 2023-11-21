import numpy as np
from pyscf import gto, scf
from ccpy.drivers.driver import Driver

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
driver.options["davidson_out_of_core"] = True
driver.run_cc(method="ccsd")
driver.run_hbar(method="ccsd")
driver.run_guess(method="deacis", multiplicity=-1, nact_unoccupied=5, roots_per_irrep={"A1": 6})
driver.run_deaeomcc(method="deaeom4", state_index=[0, 1, 2, 3, 4, 5])

expected_vee = [-1.20632891, -1.22803218, -1.14348083, -1.04281182, -0.91190576, -0.88424966]
for i in range(len(expected_vee)):
    assert np.allclose(expected_vee[i], driver.vertical_excitation_energy[i], atol=1.0e-08)



