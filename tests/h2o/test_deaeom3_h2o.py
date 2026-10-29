
import numpy as np
from pyscf import scf, gto
from ccpy import Driver, Molecule, RHF

def _run_and_check(driver):
    driver.system.print_info()

    driver.run_cc(method="ccsd")
    assert np.allclose(-74.7347458944, driver.correlation_energy + driver.system.reference_energy, atol=1.0e-07)

    driver.run_hbar(method="ccsd")
    driver.run_guess(method="deacis", roots_per_irrep={"A1": 5, "B1": 0, "B2": 0, "A2": 0}, multiplicity=1, nact_unoccupied=4, use_symmetry=False)
    driver.run_deaeomcc(method="deaeom3", state_index=[0, 1, 2, 3, 4])

    expected_ea_energy = [-1.0587336648, -1.1435694430, -1.1147055248, -0.7502349476, -0.7505287522]

    for n in [0, 1, 2, 3, 4]:
        assert np.allclose(driver.vertical_excitation_energy[n], expected_ea_energy[n], atol=1.0e-07)

def test_deaeom3_h2o():
    geometry = [["O", (0.0, 0.0, -0.0180)],
                ["H", (0.0, 3.030526, -2.117796)],
                ["H", (0.0, -3.030526, -2.117796)]]
    mol = gto.M(
        atom=geometry,
        basis="cc-pvdz",
        charge=2,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=0)
    _run_and_check(driver)

def test_deaeom3_h2o_native():
    # native ccpy SCF + tight Cholesky integrals
    mol = Molecule("O 0.0 0.0 -0.0180; H 0.0 3.030526 -2.117796; H 0.0 -3.030526 -2.117796",
                   "cc-pvdz", unit="bohr", charge=2, multiplicity=1, symmetry=True, cartesian=False,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=0)
    _run_and_check(driver)

if __name__ == "__main__":
    test_deaeom3_h2o()
    test_deaeom3_h2o_native()
