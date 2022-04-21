
def main():


    from ccpy.models.calculation import Calculation
    from ccpy.interfaces.gamess_tools import load_from_gamess
    from ccpy.drivers.driver import cc_driver, lcc_driver

    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

    from ccpy.moments.crcc23 import calc_crcc23


    system, H = load_from_gamess(
                "/scratch/gururang/cis7hq/cis7hq.log",
                "/scratch/gururang/cis7hq/onebody.inp",
                "/scratch/gururang/cis7hq/twobody.inp",
                nfrozen=11)
    system.print_info()

    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-07,
        diis_size=6,
        RHF_symmetry=True
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)

    calculation = Calculation(
        order=2,
        calculation_type="left_ccsd",
        convergence_tolerance=1.0e-07,
        diis_size=6,
        RHF_symmetry=True
    )

    Hbar = build_hbar_ccsd(T, H)

    L, total_energy, is_converged = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None)

    Ecrcc23, delta23 = calc_crcc23(T, L, Hbar, H, system, use_RHF=True)



if __name__ == "__main__":

    main()



