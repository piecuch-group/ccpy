""" CC(t;3) computation for the stretched F2 molecule at interatomic
separation R = 2.0Re, where Re = 2.66816 bohr. The cc-pVDZ basis set
is used with Cartesian components for the d orbitals.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""

import numpy as np
from pyscf import scf, gto
from ccpy.drivers.driver import Driver

def test_cct3_f2():
    geometry = [["F", (0.0, 0.0, -2.66816)], ["F", (0.0, 0.0, 2.66816)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2)
    driver.system.set_active_space(nact_occupied=5, nact_unoccupied=1)
    driver.run_cc(method="ccsdt1")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="cct3", state_index=[0])

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -198.4200962814)
    # Check CCSDt energy
    assert np.allclose(driver.correlation_energy, -0.6363154135)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -199.0564116949
    )
    # Check CC(t;3)_A energy
    assert np.allclose(
        driver.correlation_energy + driver.deltapq[0]["A"], -0.6376818524
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltapq[0]["A"],
        -199.0577781338,
    )
    # Check CC(t;3)_D energy
    assert np.allclose(
        driver.correlation_energy + driver.deltapq[0]["D"], -0.6378384699
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltapq[0]["D"],
        -199.0579347513,
    )

    if __name__ == "__main__":
        test_cct3_f2()