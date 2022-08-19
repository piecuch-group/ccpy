
from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver, lcc_driver

from ccpy.hbar.hbar_ccsdt import build_hbar_ccsdt

if __name__ == "__main__":

    system, H = load_from_gamess(
            "chplus_re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )

    calculation = Calculation(
        order=3,
        calculation_type="ccsdt",
        convergence_tolerance=1.0e-08,
        RHF_symmetry=False,
    )

    T, total_energy, _ = cc_driver(calculation, system, H)

    Hbar = build_hbar_ccsdt(T, H)

    calculation = Calculation(
        order=3,
        calculation_type="left_ccsdt",
        convergence_tolerance=1.0e-08,
        maximum_iterations=200,
        RHF_symmetry=True,
    )

    L, _, _ = lcc_driver(calculation, system, T, Hbar)
