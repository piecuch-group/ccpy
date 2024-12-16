"""CCSD(T) computation for the stretched F2 molecule at at
interatomic distance of R = 2Re, where Re = 2.66816 bohr,
described using the cc-pVTZ basis set.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""

import numpy as np
from pyscf import scf, gto
from ccpy import Driver

def test_crcc34_f2():
    geometry = [["F", (0.0, 0.0, -2.66816)],
                ["F", (0.0, 0.0, 2.66816)]]
    mol = gto.M(
        atom=geometry,
        basis="6-31g",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=2)
    driver.run_cc(method="ccsdt")
    driver.run_hbar(method="ccsdt")
    driver.run_leftcc(method="left_ccsdt")
    driver.run_ccp4(method="crcc34")

    #
    # Check the results
    #
    # Check that CCSDT total energy is correct
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -198.8922018713, atol=1.0e-07)
    # Check that CR-CC(3,4)_A = CCSDT(2)_Q total energy is correct
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy + driver.deltap4[0]["A"], -198.8933446977, atol=1.0e-07)

if __name__ == "__main__":
    test_crcc34_f2()
