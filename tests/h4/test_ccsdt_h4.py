import numpy as np
from pyscf import gto, scf
from ccpy import Driver

def test_ccsdt_h4():

        Re = 1.0
        geom = [['H', (-Re, -Re, 0.000)],
                ['H', (-Re,  Re, 0.000)],
                ['H', (Re, -Re, 0.000)],
                ['H', (Re,  Re, 0.000)]]

        mol = gto.M(atom=geom, basis="dz", spin=0, symmetry="D2H", unit="Bohr")
        mf = scf.RHF(mol)
        mf.kernel()

        driver = Driver.from_pyscf(mf, nfrozen=0)
        driver.system.print_info()

        driver.run_cc(method="ccsdt")

        #
        # Check the results
        #
        assert np.allclose(driver.correlation_energy, -0.06350719, atol=1.0e-07)

if __name__ == "__main__":
        test_ccsdt_h4()
