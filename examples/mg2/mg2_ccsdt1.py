from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver


if __name__ == "__main__":


    system, H = load_from_gamess(
                "/scratch/gururang/mg2_test/Mg2_AWCQZ_PF_CCSDt_3.9.log",
                "/scratch/gururang/mg2_test/onebody.inp",
                "/scratch/gururang/mg2_test/twobody.inp",
                nfrozen=2)

    system.print_info()
    system.set_active_space(nact_occupied=2, nact_unoccupied=6)

    calculation = Calculation(
        order=3,
        active_orders=[3],
        num_active=[1],
        calculation_type="ccsdt1",
        convergence_tolerance=1.0e-08,
        maximum_iterations=2,
        diis_size=6,
        low_memory=False,
        )

    T, total_energy, is_converged = cc_driver(calculation, system, H)
