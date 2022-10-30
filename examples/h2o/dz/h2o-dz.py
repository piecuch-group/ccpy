import argparse

def main(args):

    from pyscf import gto, scf

    from ccpy.models.calculation import Calculation
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals
    from ccpy.drivers.driver import cc_driver, lcc_driver
    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd
    from ccpy.moments.crcc24 import calc_crcc24
    from ccpy.moments.crcc23 import calc_crcc23

    if args.re== 1:
        geom = [['H', (0, 1.515263, -1.058898)], ['H', (0, -1.515263, -1.058898)], ['O', (0.0, 0.0, -0.0090)]]
    elif args.re == 2:
        geom = [['H', (0, 3.030526, -2.117796)], ['H', (0, -3.030526, -2.117796)], ['O', (0.0, 0.0, -0.0180)]]
    elif args.re == 3:
        geom = [['H', (0, 4.545789, -3.176694)], ['H', (0, -4.545789, -3.176694)], ['O', (0.0, 0.0, -0.0270)]]

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

    system, H = load_pyscf_integrals(mf, nfrozen=0)
    system.print_info()

    calculation = Calculation(
        calculation_type=args.method,
        convergence_tolerance=1.0e-08,
        RHF_symmetry=True
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)

    Hbar = build_hbar_ccsd(T, H)

    calculation = Calculation(
        calculation_type="left_ccsd",
        RHF_symmetry=True,
    )
    L, _, _ = lcc_driver(calculation, system, T, Hbar)

    _, delta23 = calc_crcc23(T, L, Hbar, H, system)
    _, delta24 = calc_crcc24(T, L, Hbar, H, system)

    Ecrcc24_AA = total_energy + delta23['A'] + delta24['A']
    Ecrcc24_DA = total_energy + delta23['D'] + delta24['A']

    print("CR-CC(2,4)_AA = ", Ecrcc24_AA)
    print("CR-CC(2,4)_DA = ", Ecrcc24_DA)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run all-electron H2O calculation for geometries from Bartlett & Balkova.")
    parser.add_argument("method", type=str, help="CC method to run (default is 'ccsd').", default="ccsd")
    parser.add_argument("-re", type=float, help="Geometry defined in terms of equlibrium value of R_OH (default is 1).", default=1.0)
    parser.add_argument("-basis", type=str, help="Basis set (default is DZ).", default="dz")

    args = parser.parse_args()

    main(args)



