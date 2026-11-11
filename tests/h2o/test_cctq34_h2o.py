""" CC(t,q;3,4) computation for the stretched H2O molecule at interatomic
separation R = 2.0Re, where Re = 2.66816 bohr. The cc-pVDZ basis set
is used with Cartesian components for the d orbitals.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""

import numpy as np
from ccpy import Driver, get_active_triples_pspace, get_active_triples_quadruples_pspace
from ccpy import Molecule, RHF
from pyscf import gto, scf


def test_cctq34_h2o_dz():
    # native ccpy SCF + tight Cholesky integrals
    mol = Molecule('''O 0.0 0.0 -0.0180
                      H 0.0 3.030526 -2.117796
                      H 0.0 -3.030526 -2.117796''',
                   basis="dz",
                   unit="bohr",
                   charge=0,
                   multiplicity=1,
                   symmetry=True,
                   cartesian=False,
                   cholesky_tei=True,
                   cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    driver.system.print_info()
    driver.system.set_active_space(nact_occupied=2, nact_unoccupied=2)
    t3_excitations, t4_excitations = get_active_triples_quadruples_pspace(driver.system, target_irrep="A1")

    driver.run_ccp(method="ccsdtq_p", t3_excitations=t3_excitations, t4_excitations=t4_excitations)
    # driver.run_hbar(method="ccsd")
    # driver.run_leftcc(method="left_ccsd")
    # driver.run_ccp3(method="cct3")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -75.59518068, atol=1.0e-07)
    # Check CCSDtq energy
    assert np.allclose(driver.correlation_energy, -0.30872980, atol=1.0e-07)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -75.90391049,
        atol=1.0e-07
    )

def test_cctq34_h2o_ccpvdz():
    # native ccpy SCF + tight Cholesky integrals
    # geometry = '''
    #     O 0.0 0.0 -0.0180
    #     H 0.0 3.030526 -2.117796
    #     H 0.0 -3.030526 -2.117796
    # '''
    geometry = '''
        O 0.0 0.0 -0.0135
        H 0.0 2.2728945 -1.588347
        H 0.0 -2.2728945 -1.588347
    '''
    mol = Molecule(geometry,
                   basis="cc-pvdz",
                   unit="bohr",
                   charge=0,
                   multiplicity=1,
                   symmetry=True,
                   cartesian=False,
                   cholesky_tei=True,
                   cholesky_tol=1.0e-14)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    driver.system.print_info()
    # active space: occupied - 1b2, 3a1, unoccupied - 4a1, 2b2
    driver.system.set_active_space(active_occupied=[3, 5], active_unoccupied=[6, 7])
    t3_excitations, t4_excitations = get_active_triples_quadruples_pspace(driver.system, target_irrep="A1")

    driver.run_ccp(method="ccsdtq_p", t3_excitations=t3_excitations, t4_excitations=t4_excitations)
    # driver.run_hbar(method="ccsd")
    # driver.run_leftcc(method="left_ccsd")
    # driver.run_ccp3(method="ccp3", state_index=0, t3_excitations=t3_excitations, target_irrep="A1")


if __name__ == "__main__":
    test_cctq34_h2o_ccpvdz()
