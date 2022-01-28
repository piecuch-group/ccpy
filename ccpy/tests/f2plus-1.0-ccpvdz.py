

if __name__ == '__main__':

    from ccpy.interfaces.pyscf_tools import loadFromPyscfMolecular
    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation
    from ccpy.drivers.calc_driver_main import calc_driver_main

    # Testing from PySCF
    mol = gto.Mole()
    mol.build(
        atom='''F 0.0 0.0 -1.33408
                F 0.0 0.0  1.33408''',
        basis='ccpvdz',
        charge=1,
        spin=1,
        symmetry='D2H',
        cart=True,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    nfrozen = 2
    system, H = loadFromPyscfMolecular(mf, nfrozen, dumpIntegrals=False)

    print(system)

    calculation = Calculation('CCSD')
    T, cc_energy = calc_driver_main(calculation, system, H)

