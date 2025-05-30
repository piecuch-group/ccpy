import numpy as np
from pyscf import gto, scf
from ccpy.drivers.driver import Driver

def test_dipeomccsdta_cl2():

    nfrozen = 10

    geom = [["Cl", (0.0, 0.0, 0.0)],
            ["Cl", (0.0, 0.0, 1.9870)]]

    mol = gto.M(atom=geom, basis="cc-pvdz", symmetry="D2H", unit="Angstrom", cart=False, charge=0)
    mf = scf.RHF(mol)
    mf = mf.x2c() # use SFX2C-1e treatment of scalar relativity
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=nfrozen)
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsdta")

    driver.run_guess(method="dipcis", multiplicity=-1, nact_occupied=driver.system.noccupied_alpha, roots_per_irrep={"A1": 10}, use_symmetry=False) 
    driver.run_dipeomcc(method="dipeomccsdta", state_index=[0, 1, 3, 4])

    #
    # Check the results
    #
    assert np.allclose(driver.vertical_excitation_energy[0], 1.11710195, atol=1.0e-07, rtol=1.0e-07)  # X ^{3}Sigma_g-
    assert np.allclose(driver.vertical_excitation_energy[1], 1.13764771, atol=1.0e-07, rtol=1.0e-07)  # a ^{1}Delta_g
    assert np.allclose(driver.vertical_excitation_energy[3], 1.15045610, atol=1.0e-07, rtol=1.0e-07)  # b ^{1}Sigma_g+
    assert np.allclose(driver.vertical_excitation_energy[4], 1.18895140, atol=1.0e-07, rtol=1.0e-07)  # c ^{1}Sigma_u-

if __name__ == "__main__":
    test_dipeomccsdta_cl2()
