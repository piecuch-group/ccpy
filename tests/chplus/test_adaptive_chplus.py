import pytest
from pathlib import Path
import numpy as np
from ccpy import Driver, AdaptEOMDriver

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

@pytest.mark.short
def test_adaptive_chplus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/chplus/chplus.log",
        fcidump=TEST_DATA_DIR + "/chplus/chplus.FCIDUMP",
        nfrozen=0,
    )
    driver.system.print_info()
    driver.options["maximum_iterations"] = 500

    adaptdriver = AdaptEOMDriver(driver, state_index=[2, 3, 4], roots_per_irrep={"A1": 4}, multiplicity=1,
                                 nacto=3, nactu=3,
                                 percentage=[0.0, 1.0, 2.0],
                                 unify=False)
    adaptdriver.options["energy_tolerance"] = 1.0e-09
    adaptdriver.run()

    #
    # Check the results
    #
    expected_ccp_energy = np.array([[-38.0176701646, -38.0188907303, -38.0191328777],
                                    [-37.6829267747, -37.6912855965, -37.6943482722],
                                    [-37.5186014338, -37.5201713506, -37.5215116130],
                                    [-37.3813352631, -37.3850813316, -37.3857913733]]) # nstates x niter

    expected_ccpq_energy = np.array([[-38.0194527234, -38.0194961897, -38.0195020270],
                                     [-37.7012481973, -37.7018945048, -37.7021483576],
                                     [-37.5216701032, -37.5222483557, -37.5223309649],
                                     [-37.3859179808, -37.3868390816, -37.3867279384]]) # nstates x niter

    for istate, (e_ccp, e_ccpq) in enumerate(zip(expected_ccp_energy, expected_ccpq_energy)):
        for imacro in range(len(e_ccp)):
            assert np.allclose(e_ccp[imacro], adaptdriver.ccp_energy[istate, imacro], atol=1.0e-07)
            assert np.allclose(e_ccpq[imacro], adaptdriver.ccpq_energy[istate, imacro], atol=1.0e-07)


if __name__ == "__main__":
    test_adaptive_chplus()
