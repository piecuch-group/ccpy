"""ACC(2,3) [= ACCSD + CR-CC(2,3)-like correction] computation for 
the symmetrically stretched H6 ring, as described using the cc-pVTZ
basis set. Different weighting schemes for the (T2**2) diagrams
in CCSD are used, corresponding to the ACCSD(1,3), ACCSD(1,(3+4)/2) = DCSD,
ACCSD(1,3 x no/(no+nu) + 4 x nu/(no+nu)), and ACCSD(1,4) approaches.
Each of these ACP calculations is corrected for T3 correlations
using the CR-CC(2,3)-like formulas, resulting in the associated
ACC(2,3) methods.
Reference: Mol. Phys. 120, e2057365 (2022)."""

import numpy as np
from pyscf import gto, scf
from ccpy import Driver, Molecule, RHF

def h6_geometry(r):
    theta = [2.0*np.pi/6 * n for n in range(6)]
    coords = np.zeros((6, 3))
    for i in range(6):
        coords[i, 0] = r * np.cos(theta[i])
        coords[i, 1] = r * np.sin(theta[i])
        coords[i, 2] = 0.0
    return coords

def _run_and_check(make_driver):
    # make_driver() returns a fresh Driver built from the same SCF
    #  ACCSD(1,3)
    driver = make_driver()
    driver.system.print_info()
    driver.run_cc(method="accsd", acparray=[1., 0., 1., 0., 0.])
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")
    # Check results
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -3.41073401)

    # ACCSD(1,(3+4)/2) = DCSD
    driver = make_driver()
    driver.system.print_info()
    driver.run_cc(method="accsd", acparray=[1., 0., 0.5, 0.5, 0.])
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")
    # check results
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -3.40816226)

    # ACCSD(1, 3 x no/norb + 4 x nu/norb)
    driver = make_driver()
    driver.system.print_info()
    d3 = driver.system.noccupied_alpha/driver.system.norbitals
    d4 = driver.system.nunoccupied_alpha/driver.system.norbitals
    driver.run_cc(method="accsd", acparray=[1., 0., d3, d4, 0.])
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")
    # check results
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -3.40593689)

    # ACCSD(1,4)
    driver = make_driver()
    driver.system.print_info()
    driver.run_cc(method="accsd", acparray=[1., 0., 0., 1., 0.])
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")
    # check results
    assert np.allclose(driver.system.reference_energy + driver.correlation_energy, -3.40577161)

def test_acc23_h6():

    r_HH = 1.0
    coords = h6_geometry(r_HH)
    mol = gto.M(
            atom=f'''H {coords[0,0]} {coords[0,1]} {coords[0,2]}
                     H {coords[1,0]} {coords[1,1]} {coords[1,2]}
                     H {coords[2,0]} {coords[2,1]} {coords[2,2]}
                     H {coords[3,0]} {coords[3,1]} {coords[3,2]}
                     H {coords[4,0]} {coords[4,1]} {coords[4,2]}
                     H {coords[5,0]} {coords[5,1]} {coords[5,2]}''',
            basis="cc-pvtz",
            spin=0,
            unit="Angstrom",
            symmetry="D2H",
            cart=False)
    mf = scf.RHF(mol)
    mf.kernel()

    _run_and_check(lambda: Driver.from_pyscf(mf, nfrozen=0))

def test_acc23_h6_native():
    # native ccpy SCF + tight Cholesky integrals
    r_HH = 1.0
    coords = h6_geometry(r_HH)
    mol = Molecule("; ".join(f"H {x:.12f} {y:.12f} {z:.12f}" for x, y, z in coords), "cc-pvtz", unit="angstrom",
                   charge=0, multiplicity=1, symmetry=True, cartesian=False,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    _run_and_check(lambda: Driver.from_scf(mf, nfrozen=0))

if __name__ == "__main__":

    test_acc23_h6()
    test_acc23_h6_native()
