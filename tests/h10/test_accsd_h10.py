import numpy as np
from pyscf import gto, scf
from ccpy import Driver

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

def test_ccsd_h10():

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

    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()
    driver.run_cc(method="ccsd")

def test_accsd13_h10():

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

    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()
    driver.run_cc(method="accsd", acparray=[1., 0., 1., 0., 0.])


def test_accsd134_h10():

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

    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()
    driver.run_cc(method="accsd", acparray=[1., 0., 0.5, 0.5, 0.])

def test_accsd134scaled_h10():

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

    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()

    d3 = driver.system.noccupied_alpha/driver.system.norbitals
    d4 = driver.system.nunoccupied_alpha/driver.system.norbitals
    driver.run_cc(method="accsd", acparray=[1., 0., d3, d4, 0.])


def test_accsd14_h10():

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

    driver = Driver.from_pyscf(mf, nfrozen=0)
    driver.system.print_info()
    driver.run_cc(method="accsd", acparray=[1., 0., 0., 1., 0.])


if __name__ == "__main__":


    test_accsd13_h10()
    test_accsd134_h10()
    test_accsd134scaled_h10()
    test_accsd14_h10()
