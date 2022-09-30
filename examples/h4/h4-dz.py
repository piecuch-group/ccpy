import argparse


def main(args):

    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals, get_dipole_integrals
    from ccpy.drivers.driver import cc_driver

    geom = [['H', (1.9021130326, 1.6180339887, 0.0)], 
            ['H', (0.000, 1.000, 0.000)], 
            ['H', (0.000, -1.000, 0.000)], 
            ['H', (1.9021130326, -1.6180339887, 0.0)]]

    mol = gto.Mole()

    mol.build(
        atom=geom,
        basis={'H': gto.basis.parse('''
                H    S
                      33.6444000  0.025373997879         
                      5.0579600   0.189682884145         
                      1.1468000   0.852930228705         
                H    S
                      0.3211440   0.606885215651         
                      0.1013090   0.449020011580         
                H    P
                      0.9300000   1.000000000000''')},
        charge=0,
        spin=0,
        symmetry="C1",
        cart=False,
        unit="Bohr"
    )

    mf = scf.RHF(mol)
    mf.kernel()

    dip_ints = get_dipole_integrals(mol, mf)

    for i, comp in enumerate(['x', 'y', 'z']):
        print("Dipole", comp, "integrals")
        print(dip_ints[i, :, :])

    system, H = load_pyscf_integrals(mf, nfrozen=0)
    system.print_info()

    calculation = Calculation(
        calculation_type=args.method,
        convergence_tolerance=1.0e-08,
        diis_size=6,
        maximum_iterations=80,
        RHF_symmetry=True
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run all-electron H2O calculation for geometries from Bartlett & Balkova.")
    parser.add_argument("method", type=str, help="CC method to run (default is 'ccsd').", default="ccsd")
    #.add_argument("-basis", type=str, help="Basis set (default is DZ).", default="dz")

    args = parser.parse_args()

    main(args)



