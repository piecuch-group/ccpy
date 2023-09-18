""" CC3 computation for the open-shell (HFH)- molecule in the triplet
ground state using the D2H-symmetric H-F distance of 2.0 angstrom described
using the 6-31g(1d,1p) basis set. Active space consists of (2,2).
Reference: J. Chem. Theory Comput. 8, 4968 (2012)
"""

from pathlib import Path
import numpy as np
from ccpy.drivers.driver import Driver

TEST_DATA_DIR = str(Path(__file__).parents[1].absolute() / "data")

def test_cc3_hfhminus():
    driver = Driver.from_gamess(
        logfile=TEST_DATA_DIR + "/hfhminus-triplet/hfhminus-triplet.log",
        fcidump=TEST_DATA_DIR + "/hfhminus-triplet/hfhminus-triplet.FCIDUMP",
        nfrozen=1,
    )
    driver.system.print_info()

    driver.run_cc(method="cc3")

    for i in range(driver.system.noccupied_alpha):
        for j in range(i, driver.system.noccupied_alpha):
            if abs(driver.hamiltonian.a.oo[i, j]) > 1.0e-06:
                print(f"f_a_oo({i + 1, j + 1}) = {driver.hamiltonian.a.oo[i, j]}")
    for i in range(driver.system.noccupied_beta):
        for j in range(i, driver.system.noccupied_beta):
            if abs(driver.hamiltonian.b.oo[i, j]) > 1.0e-06:
                print(f"f_b_oo({i + 1, j + 1}) = {driver.hamiltonian.b.oo[i, j]}")
    for i in range(driver.system.nunoccupied_alpha):
        for j in range(i, driver.system.nunoccupied_alpha):
            if abs(driver.hamiltonian.a.vv[i, j]) > 1.0e-06:
                print(f"f_a_vv({i + 1, j + 1}) = {driver.hamiltonian.a.vv[i, j]}")
    for i in range(driver.system.nunoccupied_beta):
        for j in range(i, driver.system.nunoccupied_beta):
            if abs(driver.hamiltonian.b.vv[i, j]) > 1.0e-06:
                print(f"f_b_vv({i + 1, j + 1}) = {driver.hamiltonian.b.vv[i, j]}")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -100.3591573557)
    # Check CC3 energy
    assert np.allclose(driver.correlation_energy, -0.19371334)
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -100.55287069)

if __name__ == "__main__":
    test_cc3_hfhminus()
