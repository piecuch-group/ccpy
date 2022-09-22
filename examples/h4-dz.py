import argparse


def main(args):

    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
    from ccpy.drivers.driver import cc_driver

    geom = [['H', (1.9021130326, 1.6180339887, 0.0)], 
            ['H', (0.000, 1.000, 0.000)], 
            ['H', (0.000, -1.000, 0.000)], 
            ['H', (1.9021130326, -1.6180339887, 0.0)]]

    if args.method == 'ccsd':
        order = 2
    if args.method == 'ccsdt':
        order = 3
    if args.method == 'ccsdtq':
        order = 4

    mol = gto.Mole()

    mol.build(
        atom=geom,
        basis=args.basis,
        charge=0,
        spin=0,
        symmetry="C1",
        cart=False,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, nfrozen=0)
    system.print_info()

    calculation = Calculation(
        order=order,
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
    parser.add_argument("-basis", type=str, help="Basis set (default is DZ).", default="dz")

    args = parser.parse_args()

    main(args)



