from pathlib import Path
import numpy as np
from ccpy import Driver, AdaptEOMDriverSS

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_adaptive_chplus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()
    driver.options["maximum_iterations"] = 500

    adaptdriver = AdaptEOMDriverSS(driver, state_index=2, roots_per_irrep={"B1": 2}, multiplicity=1,
                                   nacto=3, nactu=3,
                                   percentage=[0.0, 1.0, 2.0])
    adaptdriver.options["energy_tolerance"] = 1.0e-09
    adaptdriver.run()

    #
    # Check the results
    #
    expected_ccp_energy = np.array([[-38.0176701646, -38.0188907303, -38.0191328777],
                                    [-37.4864870005, -37.4950227271, -37.4964737979]]) # nstates x niter

    expected_ccpq_energy = np.array([[-38.0194527234, -38.0194961897, -38.0195020270],
                                     [-37.4953377145, -37.4976835560, -37.4979710033]]) # nstates x niter

    for imacro in range(expected_ccpq_energy.shape[1]):
        assert np.allclose(expected_ccp_energy[0, imacro], adaptdriver.ccp_energy[imacro], atol=1.0e-07)
        assert np.allclose(expected_ccpq_energy[0, imacro], adaptdriver.ccpq_energy[imacro], atol=1.0e-07)
        assert np.allclose(expected_ccp_energy[1, imacro], adaptdriver.eomccp_energy[imacro], atol=1.0e-07)
        assert np.allclose(expected_ccpq_energy[1, imacro], adaptdriver.eomccpq_energy[imacro], atol=1.0e-07)

if __name__ == "__main__":
    test_adaptive_chplus()
