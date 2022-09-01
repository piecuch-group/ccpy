def main():
    from ccpy.models.calculation import Calculation
    from ccpy.drivers.driver import cc_driver, eomcc_driver
    from ccpy.interfaces.gamess_tools import load_from_gamess

    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd
    from ccpy.eomcc.initial_guess import get_initial_guess

    system, H = load_from_gamess(
        "ch2.log",
        "onebody.inp",
        "twobody.inp",
        nfrozen=0,
    )

    system.print_info()

    calculation = Calculation(
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08,
        diis_size=6,
        energy_shift=0.0,
        maximum_iterations=500,
        low_memory=False
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)

    Hbar = build_hbar_ccsd(T, H)

    calculation = Calculation(
        calculation_type="ipeom2",
        maximum_iterations=60,
        convergence_tolerance=1.0e-08,
        multiplicity=1,
        RHF_symmetry=False,
        low_memory=False,
    )

    R, omega = get_initial_guess(calculation, system, Hbar, 5, noact=0, nuact=0, guess_order=1)

    R, omega, _, _ = eomcc_driver(calculation, system, Hbar, T, R, omega)


if __name__ == "__main__":
    main()
