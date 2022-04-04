import argparse

def main(args):

    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
    from ccpy.drivers.driver import cc_driver, lcc_driver

    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

    from ccpy.moments.cct3 import calc_cct3

    mol = gto.Mole()

    Re = 2.66816 # a.u.

    mol.build(
        atom=[['F', (0, 0, -0.5 * args.re * Re)], ['F', (0, 0, 0.5 * args.re * Re)]],
        basis=args.basis,
        charge=0,
        spin=0,
        symmetry="D2H",
        cart=True,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, nfrozen=2)
    #system.set_active_space(nact_occupied=5, nact_unoccupied=9)
    system.print_info()

    if args.method == 'ccsd':
        order = 2
    elif args.method == 'ccsdt' or args.method == 'ccsdt1':
        order = 3
    else:
        print('Undefined method order!')

    calculation = Calculation(
        order=order,
        calculation_type=args.method,
        convergence_tolerance=1.0e-08,
        diis_size=6,
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)

    # calculation = Calculation(
    #     order=2,
    #     calculation_type="left_ccsd",
    #     convergence_tolerance=1.0e-08
    # )
    #
    # Hbar = build_hbar_ccsd(T, H)
    #
    # L, total_energy, is_converged = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None)
    #
    # Ecct3, delta23 = calc_cct3(T, L, Hbar, H, system, use_RHF=False)
    #


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run F2 calculation at certain separation in units of Re (Re = 2.66816 a.u.).")
    parser.add_argument("-method", type=str, help="CC method to run (default is 'ccsd').", default="ccsd")
    parser.add_argument("-re", type=float, help="Separation in units of Re (default is 2).", default=2.0)
    parser.add_argument("-basis", type=str, help="Basis set (default is ccpvdz).", default="ccpvdz")

    args = parser.parse_args()

    main(args)



