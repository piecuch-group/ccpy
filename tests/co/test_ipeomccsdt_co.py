'''
IP-EOMCCSDT for CO in cc-pVDZ basis set
M. Musial, S. A. Kucharski, and R. J. Bartlett, J. Chem. Phys 118, 1128 (2003)
'''
import numpy as np
from pyscf import gto, scf

def test_ipeomccsdt_co():
    from ccpy import Driver, get_active_triples_pspace
    from ccpy.utilities.utilities import convert_t3_from_pspace

    geom = [["C",  (0.0,  0.0,  0.0)],
            ["O",  (0.0,  0.0,  1.128323)]]

    state_index = [0, 1, 3]

    mol = gto.M(atom=geom,
                basis="cc-pvdz",
                charge=0,
                symmetry="C2V",
                cart=False,
                spin=0,
                unit="Angstrom")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()
    #
    driver.system.set_active_space(nact_occupied=driver.system.noccupied_alpha, nact_unoccupied=driver.system.nunoccupied_alpha)
    t3_excitations = get_active_triples_pspace(driver.system, target_irrep="A1")
    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
    #
    convert_t3_from_pspace(driver, t3_excitations)
    #
    driver.run_guess(method="ipcisd", nact_occupied=0, nact_unoccupied=0,
                     multiplicity=-1, roots_per_irrep={"A1": 5}, use_symmetry=False)
    driver.run_ipeomcc(method="ipeomccsdt", state_index=state_index)

    ip_energy = np.zeros(len(state_index))
    for i, istate in enumerate(state_index):
        print(f"\nState {istate}")
        print("-----")
        ip_energy[i] = (driver.vertical_excitation_energy[istate])* 27.2114 
        print(f"IP Energy = {ip_energy[i]} eV")

    expected_ip = [13.58, 16.71, 19.33]
    for w_calc, w_exp in zip(ip_energy, expected_ip):
        assert np.allclose(w_calc, w_exp, atol=1.0e-02)

if __name__ == "__main__":

    test_ipeomccsdt_co()

