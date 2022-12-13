import numpy as np

from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver
from ccpy.utilities.pspace import get_pspace_from_cipsi

if __name__ == "__main__":


    system, H = load_from_gamess(
            "h2o-Re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )
    system.print_info()

    civecs = "civecs.dat"

    print("   Using P space file: ", civecs)
    pspace, excitations, excitation_count = get_pspace_from_cipsi(civecs, system, nexcit=3)
    print("   P space composition:")
    print("   ----------------------")
    for n in range(len(pspace)):
        print("      Excitation rank", n + 3)
        num_excits = 0
        for spincase, num in excitation_count[n].items():
            print("      Number of {} = {}".format(spincase, num))
            num_excits += num
        print("      Total number of rank {} = {}".format(n + 3, num_excits))

    calculation = Calculation(
        calculation_type="ccsdt_p_linear",
        convergence_tolerance=1.0e-08,
        diis_size=6,
        energy_shift=0.0,
        maximum_iterations=50,
        RHF_symmetry=False,
    )

    T, total_energy_p, converged = cc_driver(calculation, system, H, t3_excitations=excitations[0])

    calculation = Calculation(
        calculation_type="ccsdt_p_slow",
        convergence_tolerance=1.0e-08,
        diis_size=6,
        energy_shift=0.0,
        maximum_iterations=50,
        RHF_symmetry=False,
    )

    T, total_energy, converged = cc_driver(calculation, system, H, pspace=pspace)

    print("Energy from slow CC(P) = ", total_energy)
    print("Energy from linear speedup CC(P) = ", total_energy_p)


