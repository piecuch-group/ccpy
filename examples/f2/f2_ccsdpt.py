import argparse

def main(args):

    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
    from ccpy.drivers.driver import cc_driver, lcc_driver

    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

    from ccpy.moments.crcc23 import calc_crcc23, calc_ccsdpt

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
    system.set_active_space(nact_occupied=5, nact_unoccupied=9)
    system.print_info()

    calculation = Calculation(
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08,
        diis_size=6,
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)

    _, _ = calc_ccsdpt(T, H, system, use_RHF=False)

    calculation = Calculation(
        order=2,
        calculation_type="left_ccsd",
        convergence_tolerance=1.0e-08
    )

    Hbar = build_hbar_ccsd(T, H)

    L, total_energy, is_converged = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None)

    _, _ = calc_crcc23(T, L, Hbar, H, system, use_RHF=False)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run F2 calculation at certain separation in units of Re (Re = 2.66816 a.u.).")
    parser.add_argument("-re", type=float, help="Separation in units of Re (default is 2).", default=2.0)
    parser.add_argument("-basis", type=str, help="Basis set (default is ccpvdz).", default="ccpvdz")

    args = parser.parse_args()

    main(args)

