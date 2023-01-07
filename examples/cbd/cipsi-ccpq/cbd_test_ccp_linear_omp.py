import argparse
import numpy as np
import os

os.environ["OMP_NUM_THREADS"]="1"

def main(args):

    from ccpy.models.calculation import Calculation
    from ccpy.drivers.driver import cc_driver
    from ccpy.interfaces.gamess_tools import load_from_gamess

    from ccpy.utilities.pspace import get_pspace_from_cipsi


    system, H = load_from_gamess(
            "/scratch/gururang/test_ccpq_2ba/cbd-R/rectangle-d2h.log",
            "/scratch/gururang/test_ccpq_2ba/cbd-R/onebody.inp",
            "/scratch/gururang/test_ccpq_2ba/cbd-R/twobody.inp",
            nfrozen=4,
    )

    system.print_info()

    print("   Using P space file: ", args.civecs)
    pspace, excitations, excitation_count = get_pspace_from_cipsi(args.civecs, system, nexcit=3)
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
            calculation_type="ccsdt_p_linear_omp",
            convergence_tolerance=1.0e-08,
            diis_size=6,
            energy_shift=0.0,
            maximum_iterations=500,
            RHF_symmetry=True,
    )   

    T, total_energy, is_converged = cc_driver(calculation, system, H, t3_excitations=excitations[0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the CIPSI-driven CC(P;Q) calculation using the full CC(P;Q) triples correction for a specific CI vector file.")
    parser.add_argument("-civecs", type=str, help="Path to processed CI vector file containing list of all determinants in spinorbital occupation notation.")
    args = parser.parse_args()
    main(args)



