"""Adaptive CC(P;Q) aimed at converging CCSDT computation:
F2 / cc-pVDZ at R = 2.0Re, where Re = 2.66816 bohr using RHF.
Cartesian orbitals are used for the d orbitals in the cc-pVTZ basis.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""
import numpy as np
from pyscf import scf, gto
from ccpy import Driver, AdaptDriver

def test_adaptive_f2():
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

    percentages = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    driver = Driver.from_pyscf(mf, nfrozen=2, use_cholesky=True, cholesky_tol=1.0e-07)
    driver.system.print_info()
    driver.options["RHF_symmetry"] = False
    adaptdriver = AdaptDriver(driver, percentage=percentages)
    adaptdriver.options["energy_tolerance"] = 1.0e-08
    adaptdriver.options["two_body_approx"] = True
    adaptdriver.run()

    # expected results using left-CCSD 2BA and no RHF symmetry (i.e., aaa/bbb and aab/abb spaces unbalanced slightly)
    expected_ccp = [
        -199.0086195664,
        -199.0527329286,
        -199.0530779791,
        -199.0532073849,
        -199.0533070429,
        -199.0533798878,
        -199.0534507830,
        -199.0535185251,
        -199.0535755404,
        -199.0536243499,
        -199.0536676910,
    ]

    expected_ccpq = [
        -199.0521040880,
        -199.0537746730,
        -199.0538819064,
        -199.0538670594,
        -199.0538628056,
        -199.0538564991,
        -199.0538697374,
        -199.0538861417,
        -199.0538948473,
        -199.0539047901,
        -199.0539150156,
    ]

    for imacro in range(len(percentages)):
            assert np.allclose(expected_ccp[imacro], adaptdriver.ccp_energy[imacro], rtol=1.0e-07)
            assert np.allclose(expected_ccpq[imacro], adaptdriver.ccpq_energy[imacro], rtol=1.0e-07)

if __name__ == "__main__":
    test_adaptive_f2()
