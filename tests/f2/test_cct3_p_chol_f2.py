""" CC(t;3) computation for the stretched F2 molecule at interatomic
separation R = 2.0Re, where Re = 2.66816 bohr. The cc-pVDZ basis set
is used with Cartesian components for the d orbitals.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver, get_active_triples_pspace

def test_cct3_f2():
    geometry = [["F", (0.0, 0.0, -2.66816)], ["F", (0.0, 0.0, 2.66816)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2, use_cholesky=True, cholesky_tol=1.0e-07)
    driver.system.print_info()
    driver.system.set_active_space(nact_occupied=5, nact_unoccupied=1)
    t3_excitations = get_active_triples_pspace(driver.system, target_irrep="AG")

    driver.run_ccp(method="ccsdt_p_chol", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsd_chol")
    driver.run_leftcc(method="left_ccsd_chol")
    driver.run_ccp3(method="cct3_chol")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -198.41983912, rtol=1.0e-07)
    # Check CCSDt energy
    assert np.allclose(driver.correlation_energy, -0.63244508, rtol=1.0e-07)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -199.05228420,
        rtol=1.0e-07
    )
    # Check CC(t;3)_A energy
    assert np.allclose(
        driver.correlation_energy + driver.deltap3[0]["A"], -0.6337332279, rtol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["A"],
        -199.0535722189,
        rtol=1.0e-07
    )
    # Check CC(t;3)_D energy
    assert np.allclose(
        driver.correlation_energy + driver.deltap3[0]["D"], -0.6338844036, rtol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["D"],
        -199.0537233946,
        rtol=1.0e-07
    )

if __name__ == "__main__":
    test_cct3_f2()
