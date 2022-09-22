import argparse

def main(args):

    from ccpy.models.calculation import Calculation
    from ccpy.drivers.driver import cc_driver, lcc_driver
    from ccpy.interfaces.gamess_tools import load_from_gamess

    from ccpy.utilities.pspace import get_pspace_from_cipsi

    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd
    from ccpy.moments.ccp3 import calc_ccp3

    system, H = load_from_gamess(
            "f2-2Re-pvtz/F2-2.0-VTZ-D2h.log",
            "f2-2Re-pvtz/onebody.inp",
            "f2-2Re-pvtz/twobody.inp",
            nfrozen=2,
    )

    system.print_info()

    print("   Using P space file: ", args.civecs)
    pspace, excitation_count = get_pspace_from_cipsi(args.civecs, system, nexcit=3)
    print("   P space composition:")
    print("   ----------------------")
    for n in range(len(pspace)):
        print("      Excitation rank", n + 3)
        num_excits = 0
        for spincase, num in pspace[n].items():
            print("      Number of {} = {}".format(spincase, num))
            num_excits += num
        print("      Total number of rank {} = {}".format(n + 3, num_excits))

    calculation = Calculation(
            order=3,
            calculation_type="ccsdt_p",
            convergence_tolerance=1.0e-08,
            diis_size=6,
            energy_shift=0.0,
            maximum_iterations=500,
            low_memory=False 
    )   

    T, total_energy, is_converged = cc_driver(calculation, system, H, pspace=pspace)

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
    
    Eccp3, deltap3 = calc_ccp3(T, L, Hbar, H, system, pspace, use_RHF=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the CIPSI-driven CC(P;Q) calculation using the 2BA for a specific CI vector file.")
    parser.add_argument("-civecs", type=str, help="Path to processed CI vector file containing list of all determinants in spinorbital occupation notation.")
    args = parser.parse_args()
    main(args)



