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

    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08,
        maximum_iterations=80,
        diis_size=6,
        RHF_symmetry=False,
        )

    T, total_energy, is_converged = cc_driver(calculation, system, H)
