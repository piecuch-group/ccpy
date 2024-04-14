""" 
CR-EOMCC(2,3) for the asymmetrically stretched water molecule along 
the H2O -> OH + H bond-breaking PEC corresponding to R = 3.6, taken 
from X. Li and J. Paldus, J. Chem. Phys. 133, 024102 (2010). The 
basis set used is the "TZ" basis set, corresponding to cc-pVTZ with
all polarization functions removed, resulting in a small basis that
comprises of 3 s orbitals on H and 3 s and 3 p on O.
[c.f., Comput. Theor. Chem. 1040 (2014)].
"""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver
from ccpy.utilities.pspace import get_active_pspace

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_cct3_a1_singlets_h2o():

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/h2o/h2o-tz-3.6.log",
        fcidump=TEST_DATA_DIR + "/h2o/h2o-tz-3.6.FCIDUMP",
        nfrozen=1,
    )
    driver.system.set_active_space(nact_occupied=3, nact_unoccupied=2)
    t3_excitations = get_active_pspace(driver.system, target_irrep=driver.system.reference_symmetry)
    r3_excitations = get_active_pspace(driver.system, target_irrep="A'")

    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_leftccp(method="left_ccsdt_p", t3_excitations=t3_excitations)
    driver.run_ccp3(method="ccp3", state_index=0, t3_excitations=t3_excitations, two_body_approx=False)
       
    driver.run_guess(method="cisd", nact_occupied=3, nact_unoccupied=2, multiplicity=1, roots_per_irrep={"A'": 3})
    for istate in [1, 2, 3]:
        driver.run_eomccp(method="eomccsdt_p", state_index=istate, t3_excitations=t3_excitations, r3_excitations=r3_excitations)
        driver.run_lefteomccp(method="left_ccsdt_p", state_index=istate, t3_excitations=t3_excitations, r3_excitations=r3_excitations)
        driver.run_ccp3(method="ccp3", state_index=istate, two_body_approx=False, t3_excitations=t3_excitations, r3_excitations=r3_excitations)

    #
    # Check the results
    #
    expected_ccp_energy = [-76.0359934146, -75.7277686400, -75.8331899091, -75.6588407413]
    expected_ccpqD_energy = [-76.0379948778, -75.7297010389, -75.8342799467, -75.6599390773]
    for i, (w_ccp, w_ccpq) in enumerate(zip(expected_ccp_energy, expected_ccpqD_energy)):
        ccp_energy = driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[i]
        ccpq_energy = ccp_energy + driver.deltap3[i]["D"]
        assert np.allclose(ccp_energy, w_ccp)
        assert np.allclose(ccpq_energy, w_ccpq)

def test_cct3_a2_singlets_h2o():

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/h2o/h2o-tz-3.6.log",
        fcidump=TEST_DATA_DIR + "/h2o/h2o-tz-3.6.FCIDUMP",
        nfrozen=1,
    )
    driver.system.set_active_space(nact_occupied=3, nact_unoccupied=2)
    t3_excitations = get_active_pspace(driver.system, target_irrep=driver.system.reference_symmetry)
    r3_excitations = get_active_pspace(driver.system, target_irrep="A\"")

    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_leftccp(method="left_ccsdt_p", t3_excitations=t3_excitations)
    driver.run_ccp3(method="ccp3", state_index=0, t3_excitations=t3_excitations, two_body_approx=False)
        
    driver.run_guess(method="cisd", nact_occupied=3, nact_unoccupied=2, multiplicity=1, roots_per_irrep={"A\"": 2})
    for istate in [1, 2]:
        driver.run_eomccp(method="eomccsdt_p", state_index=istate, t3_excitations=t3_excitations, r3_excitations=r3_excitations)
        driver.run_lefteomccp(method="left_ccsdt_p", state_index=istate, t3_excitations=t3_excitations, r3_excitations=r3_excitations)
        driver.run_ccp3(method="ccp3", state_index=istate, two_body_approx=False, t3_excitations=t3_excitations, r3_excitations=r3_excitations)

    #
    # Check the results
    #
    expected_ccp_energy = [-76.0359934146, -75.9858578137, -75.7320697878]
    expected_ccpqD_energy = [-76.0379948778, -75.9869154384, -75.7328857518]
    for i, (w_ccp, w_ccpq) in enumerate(zip(expected_ccp_energy, expected_ccpqD_energy)):
        ccp_energy = driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[i]
        ccpq_energy = ccp_energy + driver.deltap3[i]["D"]
        assert np.allclose(ccp_energy, w_ccp)
        assert np.allclose(ccpq_energy, w_ccpq)

def test_cct3_a1_triplets_h2o():

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/h2o/h2o-tz-3.6.log",
        fcidump=TEST_DATA_DIR + "/h2o/h2o-tz-3.6.FCIDUMP",
        nfrozen=1,
    )
    driver.system.set_active_space(nact_occupied=3, nact_unoccupied=2)
    t3_excitations = get_active_pspace(driver.system, target_irrep=driver.system.reference_symmetry)
    r3_excitations = get_active_pspace(driver.system, target_irrep="A'")

    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_leftccp(method="left_ccsdt_p", t3_excitations=t3_excitations)
    driver.run_ccp3(method="ccp3", state_index=0, t3_excitations=t3_excitations, two_body_approx=False)

    driver.run_guess(method="cisd", nact_occupied=3, nact_unoccupied=2, roots_per_irrep={"A'": 3}, multiplicity=3)
    for istate in [1, 2, 3]:
        driver.run_eomccp(method="eomccsdt_p", state_index=istate, t3_excitations=t3_excitations, r3_excitations=r3_excitations)
        driver.run_lefteomccp(method="left_ccsdt_p", state_index=istate, t3_excitations=t3_excitations, r3_excitations=r3_excitations)
        driver.run_ccp3(method="ccp3", state_index=istate, two_body_approx=False, t3_excitations=t3_excitations, r3_excitations=r3_excitations)

    #
    # Check the results
    #
    expected_ccp_energy = [-76.0359934146, -75.9834458125, -75.8379695982, -75.6527176487]
    expected_ccpqD_energy = [-76.0379948778, -75.9840907354, -75.8390257543, -75.6533246392]
    for i, (w_ccp, w_ccpq) in enumerate(zip(expected_ccp_energy, expected_ccpqD_energy)):
        ccp_energy = driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[i]
        ccpq_energy = ccp_energy + driver.deltap3[i]["D"]
        assert np.allclose(ccp_energy, w_ccp)
        assert np.allclose(ccpq_energy, w_ccpq)

def test_cct3_a2_triplets_h2o():

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/h2o/h2o-tz-3.6.log",
        fcidump=TEST_DATA_DIR + "/h2o/h2o-tz-3.6.FCIDUMP",
        nfrozen=1,
    )
    driver.system.set_active_space(nact_occupied=3, nact_unoccupied=2)
    t3_excitations = get_active_pspace(driver.system, target_irrep=driver.system.reference_symmetry)
    r3_excitations = get_active_pspace(driver.system, target_irrep="A\"")

    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_leftccp(method="left_ccsdt_p", t3_excitations=t3_excitations)
    driver.run_ccp3(method="ccp3", state_index=0, t3_excitations=t3_excitations, two_body_approx=False)
        
    driver.run_guess(method="cisd", nact_occupied=3, nact_unoccupied=2, roots_per_irrep={"A\"": 3}, multiplicity=3)
    for istate in [1, 2, 3]:
        driver.run_eomccp(method="eomccsdt_p", state_index=istate, t3_excitations=t3_excitations, r3_excitations=r3_excitations)
        driver.run_lefteomccp(method="left_ccsdt_p", state_index=istate, t3_excitations=t3_excitations, r3_excitations=r3_excitations)
        driver.run_ccp3(method="ccp3", state_index=istate, two_body_approx=False, t3_excitations=t3_excitations, r3_excitations=r3_excitations)

    #
    # Check the results
    #
    expected_ccp_energy = [-76.0359934146, -75.9934037000, -75.7643630726, -75.7148109748]
    expected_ccpqD_energy = [-76.0379948778, -75.9944354623, -75.7652874068, -75.7162086420]
    for i, (w_ccp, w_ccpq) in enumerate(zip(expected_ccp_energy, expected_ccpqD_energy)):
        ccp_energy = driver.system.reference_energy + driver.correlation_energy + driver.vertical_excitation_energy[i]
        ccpq_energy = ccp_energy + driver.deltap3[i]["D"]
        assert np.allclose(ccp_energy, w_ccp)
        assert np.allclose(ccpq_energy, w_ccpq)

if __name__ == "__main__":

    # X {1}^A' (ground), 1 {1}^A', 2 {1}^A', 3 {1}^A'
    test_cct3_a1_singlets_h2o()
    # 1 {1}^A", 2 {1}^A"
    test_cct3_a2_singlets_h2o()
    # 1 {3}^A', 2 {3}^A', 3 {3}^A'
    test_cct3_a1_triplets_h2o()
    # 1 {3}^A", 2 {3}^A", 3 {3}^A"
    test_cct3_a2_triplets_h2o()
