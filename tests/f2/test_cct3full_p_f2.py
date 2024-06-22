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
        basis="cc-pvtz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2)
    driver.system.print_info()
    driver.system.set_active_space(nact_occupied=5, nact_unoccupied=1)
    t3_excitations = get_active_triples_pspace(driver.system, target_irrep="AG")

    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
    driver.run_leftccp(method="left_ccsdt_p", t3_excitations=t3_excitations)

    driver.options["RHF_symmetry"] = False
    driver.run_ccp3(method="ccp3", t3_excitations=t3_excitations, two_body_approx=False, state_index=0, target_irrep="AG")

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
        driver.correlation_energy + driver.deltap3[0]["A"], -0.6379324807, atol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["A"],
        -199.0580287634,
        atol=1.0e-07
    )
    # Check CC(t;3)_D energy
    assert np.allclose(
        driver.correlation_energy + driver.deltap3[0]["D"], -0.6381143252, atol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["D"],
        -199.0582106079,
        atol=1.0e-07
    )

if __name__ == "__main__":
    test_cct3_f2()
