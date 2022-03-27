
from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver, lcc_driver, eomcc_driver

from ccpy.hbar.hbar_ccsdt import build_hbar_ccsdt
from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

from ccpy.moments.crcc23 import calc_crcc23
from ccpy.moments.creomcc23 import calc_creomcc23

from ccpy.eomcc.initial_guess import get_initial_guess

if __name__ == "__main__":


    num_roots = 3

    L = [None] * (num_roots + 1)
    Ecrcc23 = [None] * (num_roots + 1)
    delta23 = [None] * (num_roots + 1)


    system, H = load_from_gamess(
            "chplus_re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )

    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08,
        RHF_symmetry=False,
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)

    Hbar = build_hbar_ccsd(T, H)

    calculation = Calculation(
        order=2,
        calculation_type="left_ccsd",
        convergence_tolerance=1.0e-08,
        maximum_iterations=200,
    )

    L[0], total_energy, is_converged = lcc_driver(calculation, system, T, Hbar)
    Ecrcc23[0], delta23[0] = calc_crcc23(T, L[0], Hbar, H, system, use_RHF=False)

    calculation = Calculation(
        order=2,
        calculation_type="eomccsd",
        maximum_iterations=60,
        convergence_tolerance=1.0e-08,
        multiplicity=1,
        RHF_symmetry=False,
        low_memory=False,
    )

    R, omega = get_initial_guess(calculation, system, Hbar, num_roots, noact=0, nuact=0, guess_order=1)

    R, omega, r0, _ = eomcc_driver(calculation, system, Hbar, T, R, omega)

    for i in range(len(R)):

        calculation = Calculation(
            order=2,
            calculation_type="left_ccsd",
            convergence_tolerance=1.0e-08,
            maximum_iterations=200,
        )

        L[i + 1], _, _ = lcc_driver(calculation, system, T, Hbar, omega=omega[i], R=R[i])
        Ecrcc23[i + 1], delta23[i + 1] = calc_creomcc23(T, R[i], L[i + 1], r0[i], omega[i], Hbar, H, system)

    #Ecrcc23, delta23 = calc_crcc23(T, L, Hbar, H, system, use_RHF=False)
