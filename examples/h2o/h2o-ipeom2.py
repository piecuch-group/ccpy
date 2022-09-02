def main():
    from ccpy.models.calculation import Calculation
    from ccpy.drivers.driver import cc_driver, lcc_driver, eomcc_driver
    from ccpy.interfaces.pyscf_tools import load_pyscf_integrals

    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd
    from ccpy.eomcc.initial_guess import get_initial_guess

    from pyscf import gto, scf

    geom = [['H', (0, 1.515263, -1.058898)], ['H', (0, -1.515263, -1.058898)], ['O', (0.0, 0.0, -0.0090)]]
    #geom = [['H', (0, 3.030526, -2.117796)], ['H', (0, -3.030526, -2.117796)], ['O', (0.0, 0.0, -0.0180)]]
    #geom = [['H', (0, 4.545789, -3.176694)], ['H', (0, -4.545789, -3.176694)], ['O', (0.0, 0.0, -0.0270)]]

    mol = gto.Mole()

    mol.build(
        atom=geom,
        basis="ccpvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=False,
        unit='Bohr',
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    system, H = load_pyscf_integrals(mf, nfrozen=1)
    system.print_info()

    calculation = Calculation(calculation_type="ccsd", convergence_tolerance=1.0e-08)

    T, total_energy, is_converged = cc_driver(calculation, system, H)
    Hbar = build_hbar_ccsd(T, H)

    calculation = Calculation(calculation_type="ipeom2", multiplicity=1, RHF_symmetry=True, convergence_tolerance=1.0e-08)

    R, omega = get_initial_guess(calculation, system, Hbar, 5, noact=0, nuact=0, guess_order=1)

    R, omega, _, _ = eomcc_driver(calculation, system, Hbar, T, R, omega)

    for i in range(len(R)):

        calculation = Calculation(
            calculation_type="left_ipeom2",
            RHF_symmetry=True,
            convergence_tolerance=1.0e-08,
        )

        L, _, _ = lcc_driver(calculation, system, T, Hbar, omega=omega[i], R=R[i])


if __name__ == "__main__":
    main()