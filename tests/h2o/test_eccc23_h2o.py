from pathlib import Path
import numpy as np
from ccpy import Driver, get_triples_pspace_from_cipsi, get_quadruples_pspace_from_cipsi

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_eccc24_h2o():

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/h2o/h2o-Re.log",
        onebody=TEST_DATA_DIR + "/h2o/onebody.inp",
        twobody=TEST_DATA_DIR + "/h2o/twobody.inp",
        nfrozen=0,
    )
    driver.system.print_info()

    civecs = TEST_DATA_DIR + "/h2o/civecs-10k.dat"
    print("Reading triples excitations from CIPSI")
    t3_excitations, _ = get_triples_pspace_from_cipsi(civecs, driver.system)
    print("Reading quadruples excitations from CIPSI")
    t4_excitations, _ = get_quadruples_pspace_from_cipsi(civecs, driver.system)

    driver.run_eccc(method="eccc2", ci_vectors_file=civecs)
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="ccp3", state_index=0, t3_excitations=t3_excitations)
    driver.run_ccp4(method="ccp4", state_index=0, t4_excitations=t4_excitations)

    # Check ec-CC-II correlation energy
    assert np.allclose(driver.correlation_energy, -0.21593428, atol=1.0e-07)
    # Check ec-CC-II total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -76.23997278, atol=1.0e-07)
    # Check ec-CC-II(3)_A correction energy
    assert np.allclose(driver.deltap3[0]["A"], -0.0015382093, atol=1.0e-07)
    # Check ec-CC-II(3)_A total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["A"], -76.2415109937, atol=1.0e-07)
    # Check ec-CC-II(3)_D correction energy
    assert np.allclose(driver.deltap3[0]["D"], -0.0018237078, atol=1.0e-07)
    # Check ec-CC-II(3)_D total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["D"], -76.2417964922, atol=1.0e-07)

    # Check ec-CC-II(4)_A correction energy
    assert np.allclose(driver.deltap4[0]["A"], -0.0002633118, atol=1.0e-07)
    # Check ec-CC-II(4)_D correction energy
    assert np.allclose(driver.deltap4[0]["D"], -0.0003150049, atol=1.0e-07)

    # Check ec-CC-II(3,4)_AA total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy
                       + driver.deltap3[0]["A"] + driver.deltap4[0]["A"], -76.2417742997, atol=1.0e-07)
    # Check ec-CC-II(3,4)_AD total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy
                       + driver.deltap3[0]["A"] + driver.deltap4[0]["D"], -76.2418259928, atol=1.0e-07)
    # Check ec-CC-II(3,4)_DA total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy
                       + driver.deltap3[0]["D"] + driver.deltap4[0]["A"], -76.2420597982, atol=1.0e-07)
    # Check ec-CC-II(3,4)_DD total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy
                       + driver.deltap3[0]["D"] + driver.deltap4[0]["D"], -76.24211149130001, atol=1.0e-07)

if __name__ == "__main__":
    test_eccc24_h2o()
