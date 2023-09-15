""" CC3 computation for the stretched HF molecule at interatomic
separation R = 1.6 angstrom."""

import numpy as np
from pyscf import scf, gto
from ccpy.drivers.driver import Driver

def test_cc3_hf():
    geometry = [["H", (0.0, 0.0, -0.8)], ["F", (0.0, 0.0, 0.8)]]
    mol = gto.M(
        atom=geometry,
        basis="6-31g",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=True,
        unit="Angstrom",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)

    driver.options["RHF_symmetry"] = False
    driver.run_cc(method="cc3")

    assert np.allclose(driver.correlation_energy, -0.178932834091, atol=1.0e-07)

if __name__ == "__main__":
    test_cc3_hf()
