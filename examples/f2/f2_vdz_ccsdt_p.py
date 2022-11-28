import argparse

def main(args):

    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
    from ccpy.drivers.driver import cc_driver

    from ccpy.utilities.pspace import get_full_pspace

    mol = gto.Mole()

    Re = 2.66816 # a.u.

    use_cartesian = False

    # for consistency with our papers on F2, cc-pVDZ basis uses Cartesian basis functions
    if args.basis == 'ccpvdz':
        use_cartesian = True

    mol.build(
        atom=[['F', (0, 0, -0.5 * args.re * Re)], ['F', (0, 0, 0.5 * args.re * Re)]],
        basis=args.basis,
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=use_cartesian,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, nfrozen=2)
    system.print_info()

    calculation = Calculation(
        order=3,
        calculation_type="ccsdt_p_v2",
        convergence_tolerance=1.0e-08,
        diis_size=6,
    )

    pspace = get_full_pspace(system, 3)

    T, total_energy, is_converged = cc_driver(calculation, system, H, pspace=pspace)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run F2 calculation at certain separation in units of Re (Re = 2.66816 a.u.).")
    parser.add_argument("-re", type=float, help="Separation in units of Re (default is 2).", default=2.0)
    parser.add_argument("-basis", type=str, help="Basis set (default is ccpvdz).", default="ccpvdz")

    args = parser.parse_args()

    main(args)



