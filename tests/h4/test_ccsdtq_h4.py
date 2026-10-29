import numpy as np
from pyscf import gto, scf
from ccpy import Driver, Molecule, RHF

def _run_and_check(driver):
        driver.system.print_info()

        driver.run_cc(method="ccsdtq")

        #
        # Check the results
        #
        assert np.allclose(driver.correlation_energy, -0.064295914558, atol=1.0e-07)

def test_ccsdtq_h4():

        Re = 1.0
        geom = [['H', (-Re, -Re, 0.000)],
                ['H', (-Re,  Re, 0.000)],
                ['H', (Re, -Re, 0.000)],
                ['H', (Re,  Re, 0.000)]]

        mol = gto.M(atom=geom, basis="dz", spin=0, symmetry="D2H", unit="Bohr")
        mf = scf.RHF(mol)
        mf.kernel()

        driver = Driver.from_pyscf(mf, nfrozen=0)
        _run_and_check(driver)

def test_ccsdtq_h4_native():
        # native ccpy SCF + tight Cholesky integrals
        mol = Molecule("H -1.0 -1.0 0.0; H -1.0 1.0 0.0; H 1.0 -1.0 0.0; H 1.0 1.0 0.0", "dz", unit="bohr",
                       charge=0, multiplicity=1, symmetry=True, cartesian=False,
                       cholesky_tei=True, cholesky_tol=1.0e-12)
        mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
        mf.kernel()
        driver = Driver.from_scf(mf, nfrozen=0)
        _run_and_check(driver)

if __name__ == "__main__":
        test_ccsdtq_h4()
        test_ccsdtq_h4_native()
