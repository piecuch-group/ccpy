""" 
CR-EOMCC(2,3) for the asymmetrically stretched water molecule along 
the H2O -> OH + H bond-breaking PEC at the point R(OH) = 3.6 bohr, taken
from X. Li and J. Paldus, J. Chem. Phys. 133, 024102 (2010). The 
basis set used is the "TZ" basis set, corresponding to cc-pVTZ with
all polarization functions removed, resulting in a small basis that
comprises of 3 s orbitals on H and 3 s and 3 p on O.
[c.f., Comput. Theor. Chem. 1040 (2014)].
"""
from pathlib import Path
import numpy as np
from pyscf import gto, scf
from pyscf.gto.basis import parse_gaussian
from ccpy import Driver

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_creom23_a1_singlets_h2o():

    WATER = '''O  -0.2015391293       -0.1082616784        0.0000000000
               H   3.3968787793       -0.0015445063        0.0000000000
               H  -0.1983073653        1.7197354648        0.0000000000'''
    paldusTZ = {
                "H": parse_gaussian.load(TEST_DATA_DIR + '/h2o/paldusTZ.gbs', 'H'),
                "O": parse_gaussian.load(TEST_DATA_DIR + '/h2o/paldusTZ.gbs', 'O')
    }
    mol = gto.M(atom=WATER, basis=paldusTZ, spin=0, symmetry="Cs", charge=0, unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=1, use_cholesky=True, cholesky_tol=1.0e-09)
    driver.system.print_info()
    driver.run_cc(method="ccsd_chol")
    driver.run_hbar(method="ccsd_chol")
    driver.run_leftcc(method="left_ccsd_chol")
    
    driver.run_guess(method="cisd_chol", nact_occupied=3, nact_unoccupied=2, roots_per_irrep={"A'": 3}, multiplicity=1)
    driver.run_eomcc(method="eomccsd_chol", state_index=[1, 2, 3])
    driver.run_lefteomcc(method="left_ccsd_chol", state_index=[1, 2, 3])
    driver.run_ccp3(method="crcc23_chol", state_index=[0, 1, 2, 3])

    # Check the CCSD total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -76.0299191191, rtol=1.0e-08)
    # Check the CR-CC(2,3)_D total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["D"], -76.0385721524, rtol=1.0e-08)

    # Check the EOMCCSD total energies
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[1], -75.7051577894, rtol=1.0e-08)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[2], -75.8177593641, rtol=1.0e-08)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[3], -75.6371348770, rtol=1.0e-08)
    # Check the CR-EOMCC(2,3)_D total energies
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[1] + driver.deltap3[1]["D"], -75.7354406458, rtol=1.0e-08)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[2] + driver.deltap3[2]["D"], -75.8365241661, rtol=1.0e-08)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[3] + driver.deltap3[3]["D"], -75.6555306790, rtol=1.0e-08)

def test_creom23_a2_singlets_h2o():

    WATER = '''O  -0.2015391293       -0.1082616784        0.0000000000
               H   3.3968787793       -0.0015445063        0.0000000000
               H  -0.1983073653        1.7197354648        0.0000000000'''
    paldusTZ = {
                "H": parse_gaussian.load(TEST_DATA_DIR + '/h2o/paldusTZ.gbs', 'H'),
                "O": parse_gaussian.load(TEST_DATA_DIR + '/h2o/paldusTZ.gbs', 'O')
    }
    mol = gto.M(atom=WATER, basis=paldusTZ, spin=0, symmetry="Cs", charge=0, unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=1, use_cholesky=True, cholesky_tol=1.0e-09)
    driver.system.print_info()
    driver.run_cc(method="ccsd_chol")
    driver.run_hbar(method="ccsd_chol")

    driver.run_guess(method="cisd_chol", nact_occupied=3, nact_unoccupied=2, roots_per_irrep={"A\"": 2}, multiplicity=1)
    driver.run_eomcc(method="eomccsd_chol", state_index=[1, 2])
    driver.run_lefteomcc(method="left_ccsd_chol", state_index=[1, 2])
    driver.run_ccp3(method="crcc23_chol", state_index=[1, 2])

    # Check the EOMCCSD total energies
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[1], -75.9741899000, rtol=1.0e-08)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[2], -75.7311853758, rtol=1.0e-08)
    # Check the CR-EOMCC(2,3)_D total energies
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[1] + driver.deltap3[1]["D"], -75.9892626216, rtol=1.0e-08)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[2] + driver.deltap3[2]["D"], -75.7317203543, rtol=1.0e-08)

def test_creom23_a1_triplets_h2o():

    WATER = '''O  -0.2015391293       -0.1082616784        0.0000000000
               H   3.3968787793       -0.0015445063        0.0000000000
               H  -0.1983073653        1.7197354648        0.0000000000'''
    paldusTZ = {
                "H": parse_gaussian.load(TEST_DATA_DIR + '/h2o/paldusTZ.gbs', 'H'),
                "O": parse_gaussian.load(TEST_DATA_DIR + '/h2o/paldusTZ.gbs', 'O')
    }
    mol = gto.M(atom=WATER, basis=paldusTZ, spin=0, symmetry="Cs", charge=0, unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=1, use_cholesky=True, cholesky_tol=1.0e-09)
    driver.system.print_info()
    driver.run_cc(method="ccsd_chol")
    driver.run_hbar(method="ccsd_chol")

    driver.run_guess(method="cisd_chol", nact_occupied=3, nact_unoccupied=2, roots_per_irrep={"A'": 3}, multiplicity=3)
    driver.run_eomcc(method="eomccsd_chol", state_index=[1, 2, 3])
    driver.run_lefteomcc(method="left_ccsd_chol", state_index=[1, 2, 3])
    driver.run_ccp3(method="crcc23_chol", state_index=[1, 2, 3])

    # Check the EOMCCSD total energies
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[1], -75.9806957586, rtol=1.0e-08)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[2], -75.8249571441, rtol=1.0e-08)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[3], -75.6428074690, rtol=1.0e-08)
    # Check the CR-EOMCC(2,3)_D total energies
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[1] + driver.deltap3[1]["D"], -75.9839755205, rtol=1.0e-08)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[2] + driver.deltap3[2]["D"], -75.8406855050, rtol=1.0e-08)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[3] + driver.deltap3[3]["D"], -75.6517974898, rtol=1.0e-08)

def test_creom23_a2_triplets_h2o():

    WATER = '''O  -0.2015391293       -0.1082616784        0.0000000000
               H   3.3968787793       -0.0015445063        0.0000000000
               H  -0.1983073653        1.7197354648        0.0000000000'''
    paldusTZ = {
                "H": parse_gaussian.load(TEST_DATA_DIR + '/h2o/paldusTZ.gbs', 'H'),
                "O": parse_gaussian.load(TEST_DATA_DIR + '/h2o/paldusTZ.gbs', 'O')
    }
    mol = gto.M(atom=WATER, basis=paldusTZ, spin=0, symmetry="Cs", charge=0, unit="Bohr")
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=1, use_cholesky=True, cholesky_tol=1.0e-09)
    driver.run_cc(method="ccsd_chol")
    driver.run_hbar(method="ccsd_chol")
    driver.run_leftcc(method="left_ccsd_chol")

    driver.run_guess(method="cisd_chol", nact_occupied=3, nact_unoccupied=2, roots_per_irrep={"A\"": 3}, multiplicity=3)
    driver.run_eomcc(method="eomccsd_chol", state_index=[1, 2, 3])
    driver.run_lefteomcc(method="left_ccsd_chol", state_index=[1, 2, 3])
    driver.run_ccp3(method="crcc23_chol", state_index=[1, 2, 3])

    # Check the EOMCCSD total energies
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[1], -75.9819074697, rtol=1.0e-07)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[2], -75.7539048494, rtol=1.0e-07)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[3], -75.6629328071, rtol=1.0e-07)
    # Check the CR-EOMCC(2,3)_D total energies
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[1] + driver.deltap3[1]["D"], -75.9958999644, rtol=1.0e-08)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[2] + driver.deltap3[2]["D"], -75.7590284455, rtol=1.0e-08)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[3] + driver.deltap3[3]["D"], -75.7171758472, rtol=1.0e-08)

if __name__ == "__main__":
    test_creom23_a1_singlets_h2o()
    test_creom23_a2_singlets_h2o()
    test_creom23_a1_triplets_h2o()
    test_creom23_a2_triplets_h2o()
