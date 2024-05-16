"""ACC(t;3) [= ACCSDt + CC(t;3)-like correction] computation for 
the symmetrically stretched H6 ring, as described using the cc-pVTZ
basis set. Different weighting schemes for the (T2**2) diagrams in
the double projection of CCSDt are used, corresponding to the ACCSDt(1,3), 
ACCSDt(1,(3+4)/2), ACCSDt(1,3 x no/(no+nu) + 4 x nu/(no+nu)), and 
ACCSDt(1,4) approaches. Each of these ACP calculations is corrected for 
the missing T3 correlations using the CR(t;3)-like formulas, resulting in 
the associated ACC(t;3) methods.
Reference: Mol. Phys. 120, e2057365 (2022)."""

import numpy as np
from pyscf import gto, scf
from ccpy.drivers.driver import Driver
from ccpy.utilities.pspace import get_active_triples_pspace

def h6_geometry(r):
    theta = [2.0*np.pi/6 * n for n in range(6)]
    coords = np.zeros((6, 3))
    for i in range(6):
        coords[i, 0] = r * np.cos(theta[i])
        coords[i, 1] = r * np.sin(theta[i])
        coords[i, 2] = 0.0
    return coords

def test_acct3_h6():

    r_HH = 1.0
    coords = h6_geometry(r_HH)
    mol = gto.M(
            atom=f'''H {coords[0,0]} {coords[0,1]} {coords[0,2]}
                     H {coords[1,0]} {coords[1,1]} {coords[1,2]}
                     H {coords[2,0]} {coords[2,1]} {coords[2,2]}
                     H {coords[3,0]} {coords[3,1]} {coords[3,2]}
                     H {coords[4,0]} {coords[4,1]} {coords[4,2]}
                     H {coords[5,0]} {coords[5,1]} {coords[5,2]}''',
            basis="cc-pvtz",
            spin=0,
            unit="Angstrom",
            symmetry="D2H",
            cart=False)
    mf = scf.RHF(mol)
    mf.kernel()

    #  ACCSDt(1,3)
    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()
    driver.system.set_active_space(nact_occupied=3, nact_unoccupied=3)
    t3_excitations = get_active_triples_pspace(driver.system, target_irrep=driver.system.reference_symmetry)
    driver.run_ccp(method="accsdt_p", acparray=[1., 0., 1., 0., 0.], t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="ccp3", t3_excitations=t3_excitations, state_index=0)
    # check results
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -3.41382557)

    # ACCSDt(1,(3+4)/2) = DCSD
    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()
    driver.system.set_active_space(nact_occupied=3, nact_unoccupied=3)
    t3_excitations = get_active_triples_pspace(driver.system, target_irrep=driver.system.reference_symmetry)
    driver.run_ccp(method="accsdt_p", acparray=[1., 0., 0.5, 0.5, 0.], t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="ccp3", t3_excitations=t3_excitations, state_index=0)
    # check results
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -3.41108922)

    # ACCSDt(1, 3 x no/norb + 4 x nu/norb)
    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()
    driver.system.set_active_space(nact_occupied=3, nact_unoccupied=3)
    t3_excitations = get_active_triples_pspace(driver.system, target_irrep=driver.system.reference_symmetry)
    d3 = driver.system.noccupied_alpha/driver.system.norbitals
    d4 = driver.system.nunoccupied_alpha/driver.system.norbitals
    driver.run_ccp(method="accsdt_p", acparray=[1., 0., d3, d4, 0.], t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="ccp3", t3_excitations=t3_excitations, state_index=0)
    # check results
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -3.40872915)

    # ACCSDt(1,4)
    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()
    driver.system.set_active_space(nact_occupied=3, nact_unoccupied=3)
    t3_excitations = get_active_triples_pspace(driver.system, target_irrep=driver.system.reference_symmetry)
    driver.run_ccp(method="accsdt_p", acparray=[1., 0., 0., 1., 0.], t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="ccp3", t3_excitations=t3_excitations, state_index=0)
    # check results
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -3.40855414)

if __name__ == "__main__":

    test_acct3_h6()
