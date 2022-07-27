import argparse


def main(args):

    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
    from ccpy.drivers.driver import cc_driver

    if args.re== 1:
        geom = [['H', (0, 1.515263, -1.058898)], ['H', (0, -1.515263, -1.058898)], ['O', (0.0, 0.0, -0.0090)]]
    elif args.re == 2:
        geom = [['H', (0, 3.030526, -2.117796)], ['H', (0, -3.030526, -2.117796)], ['O', (0.0, 0.0, -0.0180)]]
    elif args.re == 3:
        geom = [['H', (0, 4.545789, -3.176694)], ['H', (0, -4.545789, -3.176694)], ['O', (0.0, 0.0, -0.0270)]]

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
        symmetry="C2V",
        cart=False,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, nfrozen=1, normal_ordered=False, sorted=False)
    system.print_info()

    print(H.aa)
    print(H.aa.shape)

    # calculation = Calculation(
    #     order=order,
    #     calculation_type=args.method,
    #     convergence_tolerance=1.0e-08,
    #     diis_size=6,
    #     maximum_iterations=80,
    #     RHF_symmetry=True
    # )
    #
    # T, total_energy, is_converged = cc_driver(calculation, system, H)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run all-electron H2O calculation for geometries from Bartlett & Balkova.")
    parser.add_argument("method", type=str, help="CC method to run (default is 'ccsd').", default="ccsd")
    parser.add_argument("-re", type=float, help="Geometry defined in terms of equlibrium value of R_OH (default is 1).", default=1.0)
    parser.add_argument("-basis", type=str, help="Basis set (default is DZ).", default="dz")

    args = parser.parse_args()

    main(args)



