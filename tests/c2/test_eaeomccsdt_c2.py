'''
EA-EOMCCSDT for C2 in cc-pVDZ basis set
M. Musial and R. J. Bartlett, J. Chem. Phys 119, 1901 (2003)
'''
import numpy as np
from pyscf import gto, scf

def test_eaeomccsdt_c2():
    from ccpy import Driver, get_active_triples_pspace
    from ccpy.utilities.utilities import convert_t3_from_pspace

    geom = [["C",  (0.0,  0.0,  0.0)],
            ["C",  (0.0,  0.0,  1.243)]]

    state_index = [0]

    mol = gto.M(atom=geom,
                basis="cc-pvdz",
                charge=0,
                symmetry="D2H",
                cart=False,
                spin=0,
                unit="Angstrom")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2)
    driver.system.print_info()
    #
    driver.system.set_active_space(nact_occupied=driver.system.noccupied_alpha, nact_unoccupied=driver.system.nunoccupied_alpha)
    t3_excitations = get_active_triples_pspace(driver.system, target_irrep="AG")
    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
    #
    convert_t3_from_pspace(driver, t3_excitations)
    #
    driver.run_guess(method="eacisd", nact_occupied=0, nact_unoccupied=0,
                     multiplicity=-1, roots_per_irrep={"AG": 7}, use_symmetry=False)
    driver.run_eaeomcc(method="eaeomccsdt", state_index=state_index)

    ea_energy = np.zeros(len(state_index))
    for i, istate in enumerate(state_index):
        print(f"\nState {istate}")
        print("-----")
        ea_energy[i] = (driver.vertical_excitation_energy[istate])* 27.2114 
        print(f"EA Energy = {ea_energy[i]} eV")

    expected_ea = [-2.30]
    for w_calc, w_exp in zip(ea_energy, expected_ea):
        assert np.allclose(w_calc, w_exp, atol=1.0e-02)

if __name__ == "__main__":

    test_eaeomccsdt_c2()

