
from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver, lcc_driver, eomcc_driver

from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

from ccpy.eomcc.initial_guess import get_initial_guess

if __name__ == "__main__":
    system, H = load_from_gamess(
            "chplus_re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )

    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)

    Hbar = build_hbar_ccsd(T, H)

    calculation = Calculation(
        order=2,
        calculation_type="eomccsd",
        convergence_tolerance=1.0e-08,
        multiplicity=1,
    )

    R, _ = get_initial_guess(calculation, system, Hbar, 10, noact=0, nuact=0, guess_order=1)

    R, omega, is_converged = eomcc_driver(calculation, system, Hbar, T, R)