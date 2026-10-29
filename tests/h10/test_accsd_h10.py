import numpy as np
from pyscf import gto, scf
from ccpy import Driver, Molecule, RHF

def h10_geometry(rHH):
    theta = [2.0*np.pi/10 * n for n in range(10)]
    print(np.rad2deg(theta))
    coords = np.zeros((10, 3))

    dtheta = abs(theta[1] - theta[0])
    R = rHH/(2.0 * np.sin(dtheta/2.0))
    for i in range(10):
        coords[i, 0] = R * np.cos(theta[i])
        coords[i, 1] = R * np.sin(theta[i])
        coords[i, 2] = 0.0
    return coords

def _pyscf_driver():

    r = 1.0
    coords = h10_geometry(r)
    mol = gto.M(
            atom=f'''H {coords[0,0]} {coords[0,1]} {coords[0,2]}
                     H {coords[1,0]} {coords[1,1]} {coords[1,2]}
                     H {coords[2,0]} {coords[2,1]} {coords[2,2]}
                     H {coords[3,0]} {coords[3,1]} {coords[3,2]}
                     H {coords[4,0]} {coords[4,1]} {coords[4,2]}
                     H {coords[5,0]} {coords[5,1]} {coords[5,2]}
                     H {coords[6,0]} {coords[6,1]} {coords[6,2]}
                     H {coords[7,0]} {coords[7,1]} {coords[7,2]}
                     H {coords[8,0]} {coords[8,1]} {coords[8,2]}
                     H {coords[9,0]} {coords[9,1]} {coords[9,2]}''',
            basis="cc-pvdz",
            spin=0,
            unit="Angstrom",
            symmetry="D2H",
            cart=False)
    mf = scf.RHF(mol)
    mf.kernel()

    return Driver.from_pyscf(mf, nfrozen=0)

def _native_driver():
    # native ccpy SCF + tight Cholesky integrals
    r = 1.0
    coords = h10_geometry(r)
    mol = Molecule("; ".join(f"H {x:.12f} {y:.12f} {z:.12f}" for x, y, z in coords), "cc-pvdz", unit="angstrom",
                   charge=0, multiplicity=1, symmetry=True, cartesian=False,
                   cholesky_tei=True, cholesky_tol=1.0e-12)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    return Driver.from_scf(mf, nfrozen=0)

def _run_ccsd(driver):
    driver.system.print_info()
    driver.run_cc(method="ccsd")

def _run_accsd13(driver):
    driver.system.print_info()
    driver.run_cc(method="accsd", acparray=[1., 0., 1., 0., 0.])

def _run_accsd134(driver):
    driver.system.print_info()
    driver.run_cc(method="accsd", acparray=[1., 0., 0.5, 0.5, 0.])

def _run_accsd134scaled(driver):
    driver.system.print_info()

    d3 = driver.system.noccupied_alpha/driver.system.norbitals
    d4 = driver.system.nunoccupied_alpha/driver.system.norbitals
    driver.run_cc(method="accsd", acparray=[1., 0., d3, d4, 0.])

def _run_accsd14(driver):
    driver.system.print_info()
    driver.run_cc(method="accsd", acparray=[1., 0., 0., 1., 0.])

def test_ccsd_h10():
    _run_ccsd(_pyscf_driver())

def test_ccsd_h10_native():
    _run_ccsd(_native_driver())

def test_accsd13_h10():
    _run_accsd13(_pyscf_driver())

def test_accsd13_h10_native():
    _run_accsd13(_native_driver())

def test_accsd134_h10():
    _run_accsd134(_pyscf_driver())

def test_accsd134_h10_native():
    _run_accsd134(_native_driver())

def test_accsd134scaled_h10():
    _run_accsd134scaled(_pyscf_driver())

def test_accsd134scaled_h10_native():
    _run_accsd134scaled(_native_driver())

def test_accsd14_h10():
    _run_accsd14(_pyscf_driver())

def test_accsd14_h10_native():
    _run_accsd14(_native_driver())

if __name__ == "__main__":


    test_accsd13_h10()
    test_accsd13_h10_native()
    test_accsd134_h10()
    test_accsd134_h10_native()
    test_accsd134scaled_h10()
    test_accsd134scaled_h10_native()
    test_accsd14_h10()
    test_accsd14_h10_native()
