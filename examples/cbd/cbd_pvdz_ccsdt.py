from pyscf import gto, scf
from ccpy.drivers.driver import Driver

if __name__ == "__main__":

    mol = gto.Mole()

    mol.build(
        atom="""C   0.68350000  0.78650000  0.00000000
                C  -0.68350000  0.78650000  0.00000000
                C   0.68350000 -0.78650000  0.00000000
                C  -0.68350000 -0.78650000  0.00000000
                H   1.45771544  1.55801763  0.00000000
                H   1.45771544 -1.55801763  0.00000000
                H  -1.45771544  1.55801763  0.00000000
                H  -1.45771544 -1.55801763  0.00000000""",
        basis="ccpvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=False,
        unit="Angstrom",
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    driver = Driver.from_pyscf(mf, nfrozen=4)
    driver.system.set_active_space(nact_occupied=1, nact_unoccupied=1)
    driver.run_cc(method="ccsdt1")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="cct3")


