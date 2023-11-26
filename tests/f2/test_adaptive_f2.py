"""Adaptive CC(P;Q) aimed at converging CCSDT computation:
F2 / cc-pVDZ at R = 2.0Re, where Re = 2.66816 bohr using RHF.
Cartesian orbitals are used for the d orbitals in the cc-pVTZ basis.
Reference: Chem. Phys. Lett. 344, 165 (2001)."""
import numpy as np
from pyscf import scf, gto
from ccpy.drivers.driver import Driver
from ccpy.drivers.adaptive import AdaptDriver

def test_adaptive_f2():
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

    percentages = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    driver = Driver.from_pyscf(mf, nfrozen=2)
    driver.system.print_info()
    driver.options["RHF_symmetry"] = False
    adaptdriver = AdaptDriver(driver, percentage=percentages)
    adaptdriver.options["energy_tolerance"] = 1.0e-08
    adaptdriver.run()

    # expected results using left-CCSD 2BA and no RHF symmetry (i.e., aaa/bbb and aab/abb spaces unbalanced slightly)
    expected_ccp = [
        -199.0125625783,
        -199.0568996806,
        -199.0572252039,
        -199.0573744420,
        -199.0574860760,
        -199.0575741030,
        -199.0576471614,
        -199.0577178858,
        -199.0577747214,
        -199.0578204962,
        -199.0578586940,
    ]

    expected_ccpq = [
        -199.0563392932,
        -199.0580212546,
        -199.0581020430,
        -199.0580922836,
        -199.0580801695,
        -199.0580881229,
        -199.0580913749,
        -199.0580995675,
        -199.0581037187,
        -199.0581058065,
        -199.0581113016,
    ]

    for imacro in range(len(percentages)):
        assert np.allclose(expected_ccp[imacro], adaptdriver.ccp_energy[imacro], atol=1.0e-07)
        assert np.allclose(expected_ccpq[imacro], adaptdriver.ccpq_energy[imacro], atol=1.0e-07)

if __name__ == "__main__":
    test_adaptive_f2()
