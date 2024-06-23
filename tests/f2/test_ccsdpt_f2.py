"""CCSD(T) computation for the stretched F2 molecule at at
interatomic distance of R = 2Re, where Re = 2.66816 bohr,
described using the cc-pVTZ basis set.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver

def test_ccsdpt_f2():
    geometry = [["F", (0.0, 0.0, -2.66816)],
                ["F", (0.0, 0.0, 2.66816)]]
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

    driver = Driver.from_pyscf(mf, nfrozen=2, use_cholesky=True, cholesky_tol=1.0e-09)
    driver.system.print_info()
    driver.options["RHF_symmetry"] = False
    driver.run_cc(method="ccsd")
    driver.run_ccp3(method="ccsd(t)")

    # Check reference energy
    assert np.allclose(driver.system.reference_energy, -198.48327030, atol=1.0e-07)
    # Check CCSD energy
    assert np.allclose(driver.correlation_energy, -0.69225474, atol=1.0e-07)
    assert np.allclose(
        driver.system.reference_energy + driver.correlation_energy, -199.17552504,
        atol=1.0e-07
    )
    # Check CCSD(T) energy
    assert np.allclose(
        driver.correlation_energy + driver.deltap3[0]["A"], -0.7814280834,
        atol=1.0e-07
    )
    assert np.allclose(
        driver.system.reference_energy
        + driver.correlation_energy
        + driver.deltap3[0]["A"],
        -199.2646983796,
        atol=1.0e-07
    )

if __name__ == "__main__":
    test_ccsdpt_f2()
