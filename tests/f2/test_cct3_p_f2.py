""" CC(t;3) computation for the stretched F2 molecule at interatomic
separation R = 2.0Re, where Re = 2.66816 bohr. The cc-pVDZ basis set
is used with Cartesian components for the d orbitals.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver, get_active_triples_pspace
from ccpy import Molecule, RHF

def _run_and_check(driver):
    driver.system.print_info()
    driver.system.set_active_space(nact_occupied=5, nact_unoccupied=1)
    t3_excitations = get_active_triples_pspace(driver.system, target_irrep="AG")

    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="cct3")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -198.4200962814, atol=1.0e-07)
    # Check CCSDt energy
    assert np.allclose(driver.correlation_energy, -0.6363154135, atol=1.0e-07)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -199.0564116949,
        atol=1.0e-07
    )
    # Check CC(t;3)_A energy
    assert np.allclose(
        driver.correlation_energy + driver.deltap3[0]["A"], -0.6376818524, atol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["A"],
        -199.0577781338,
        atol=1.0e-07
    )
    # Check CC(t;3)_D energy
    assert np.allclose(
        driver.correlation_energy + driver.deltap3[0]["D"], -0.6378384699, atol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["D"],
        -199.0579347513,
        atol=1.0e-07
    )


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
    _run_and_check(driver)


def test_cct3_f2_native():
    # native ccpy SCF + tight Cholesky integrals
    mol = Molecule("F 0.0 0.0 -2.66816; F 0.0 0.0 2.66816", "cc-pvdz", unit="bohr", charge=0, multiplicity=1,
                   symmetry=True, cartesian=True, cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=2)
    _run_and_check(driver)


if __name__ == "__main__":
    test_cct3_f2()
    test_cct3_f2_native()
