from pathlib import Path
import numpy as np
from ccpy import Driver, Molecule, RHF, get_triples_pspace_from_cipsi

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def _run_and_check(driver):

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

def test_cipsi_ccpq_h2o():

    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/h2o/h2o-Re.log",
        onebody=TEST_DATA_DIR + "/h2o/onebody.inp",
        twobody=TEST_DATA_DIR + "/h2o/twobody.inp",
        nfrozen=0,
    )
    _run_and_check(driver)

def test_cipsi_ccpq_h2o_native():
    # native ccpy SCF + tight Cholesky integrals; geometry (bohr), basis (GBASIS=CCD) and
    # ISPHER=1 (spherical) taken from h2o-Re.log
    mol = Molecule("H 0.0 -1.515263 -1.058898; H 0.0 1.515263 -1.058898; O 0.0 0.0 -0.0090",
                   "cc-pvdz", unit="bohr", charge=0, multiplicity=1, symmetry=True, cartesian=False,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    _run_and_check(driver)


if __name__ == "__main__":

    test_cipsi_ccpq_h2o()
    test_cipsi_ccpq_h2o_native()
