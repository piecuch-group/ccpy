from pathlib import Path
import numpy as np
from ccpy import Driver, get_triples_pspace_from_cipsi

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_cipsi_ccpq_h2o():

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/h2o/h2o-Re.log",
        onebody=TEST_DATA_DIR + "/h2o/onebody.inp",
        twobody=TEST_DATA_DIR + "/h2o/twobody.inp",
        nfrozen=0,
    )

    civecs = TEST_DATA_DIR + "/h2o/civecs-10k.dat"
    t3_excitations, _ = get_triples_pspace_from_cipsi(civecs, driver.system)

    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_leftccp(method="left_ccsdt_p", t3_excitations=t3_excitations)
    driver.run_ccp3(method="ccp3", state_index=0, t3_excitations=t3_excitations)

    # Check CC(P) total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -76.2396274886, atol=1.0e-07)
    # Check CC(P;Q) total energy
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap3[0]["D"], -76.2414725396, atol=1.0e-07)


if __name__ == "__main__":

    test_cipsi_ccpq_h2o()