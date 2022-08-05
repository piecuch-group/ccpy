

def main():
    from ccpy.models.calculation import Calculation
    from ccpy.drivers.driver import cc_driver, lcc_driver
    from ccpy.interfaces.gamess_tools import load_from_gamess

    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd
    from ccpy.moments.crcc23 import calc_crcc23

    system, H = load_from_gamess(
        "CH2.log",
        "onebody.inp",
        "twobody.inp",
        nfrozen=0,
    )

    system.print_info()

    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08,
        diis_size=6,
        energy_shift=0.0,
        maximum_iterations=500,
        low_memory=False
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)


    calculation = Calculation(
        order=2,
        calculation_type="left_ccsd",
        convergence_tolerance=1.0e-08,
        maximum_iterations=500,
        energy_shift=0.0,
        low_memory=False
    )

    Hbar = build_hbar_ccsd(T, H)

    L, total_energy, is_converged = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None)

    Ecrcc23, delta23 = calc_crcc23(T, L, Hbar, H, system)


if __name__ == "__main__":
    main()

