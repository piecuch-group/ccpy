import numpy as np
from pyscf import scf, gto
from ccpy import Driver

def test_ccsdt_f2():
    geometry = [["F", (0.0, 0.0, -2.66816)],
                ["F", (0.0, 0.0, 2.66816)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.UHF(mol)
    mf.kernel()
    driver = Driver.from_pyscf(mf, nfrozen=2, uhf=True)
    driver.system.print_info()

    driver.options["RHF_symmetry"] = False
    driver.run_cc(method="ccsdt")

    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -199.058201, rtol=1.0e-07, atol=1.0e-07)

if __name__ == "__main__":
    test_ccsdt_f2()
