import numpy as np
from pyscf import scf, gto
from ccpy import Driver
from ccpy import Molecule, RHF

def _run_and_check(driver):
    driver.run_cc(method="ccsdt")

    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -199.05400663, rtol=1.0e-07)


def test_ccsdt_f2():
    geometry = [["F", (0.0, 0.0, -2.66816)],
                ["F", (0.0, 0.0, 2.66816)]]
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

    driver = Driver.from_pyscf(mf, nfrozen=2)
    _run_and_check(driver)


def test_ccsdt_f2_native():
    # native ccpy SCF + tight Cholesky integrals
    mol = Molecule("F 0.0 0.0 -2.66816; F 0.0 0.0 2.66816", "cc-pvdz", unit="bohr", charge=0, multiplicity=1,
                   symmetry=True, cartesian=False, cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=2)
    _run_and_check(driver)


if __name__ == "__main__":
    test_ccsdt_f2()
    test_ccsdt_f2_native()
