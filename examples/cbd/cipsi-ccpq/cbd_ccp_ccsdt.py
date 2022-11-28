import numpy as np

def main():

    from ccpy.models.calculation import Calculation
    from ccpy.drivers.driver import cc_driver, lcc_driver
    from ccpy.interfaces.gamess_tools import load_from_gamess
    from ccpy.utilities.pspace import get_full_pspace, get_empty_pspace


    system, H = load_from_gamess(
            "/scratch/gururang/test_ccpq_2ba/cbd-R/rectangle-d2h.log",
            "/scratch/gururang/test_ccpq_2ba/cbd-R/onebody.inp",
            "/scratch/gururang/test_ccpq_2ba/cbd-R/twobody.inp",
            nfrozen=4,
    )

    system.print_info()

    print("Using empty P space to create CCSD")
    #pspace = get_full_pspace(system, 3)
    pspace = get_empty_pspace(system, 3)

    calculation = Calculation(
            calculation_type="ccsdt_p",
            convergence_tolerance=1.0e-08,
            diis_size=6,
            energy_shift=0.0,
            maximum_iterations=500,
            RHF_symmetry=False,
            low_memory=False 
    )   

    T, total_energy, is_converged = cc_driver(calculation, system, H, pspace=pspace)


if __name__ == "__main__":

    main()



