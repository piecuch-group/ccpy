import numpy as np

def main():

    from pyscf import gto, scf

    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
    from ccpy.drivers.driver import Driver

    mol = gto.Mole()

    Re = 2.66816 # a.u.

    mol.build(
        atom=[['F', (0, 0, -0.5 * 2 * Re)], ['F', (0, 0, 0.5 * 2 * Re)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    mycc = Driver.from_pyscf(mf, nfrozen=2)
    mycc.run_cc(method="ccsd")
    mycc.run_hbar(method="ccsd")
    mycc.run_leftcc(method="left_ccsd")
    #mycc.run_eomcc(method="eomccsd", state_index=[1, 2, 3])
    mycc.run_ccp3(method="crcc23", state_index=[0])


if __name__ == "__main__":

    main()


