import argparse


def main(args):
    from ccpy.models.calculation import Calculation
    from ccpy.drivers.driver import cc_driver, lcc_driver
    from ccpy.interfaces.gamess_tools import load_from_gamess

    from ccpy.utilities.pspace import get_pspace_from_cipsi

    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd
    from ccpy.moments.ccp3 import calc_ccp3

    system, H = load_from_gamess(
        "CH2.log",
        "onebody.inp",
        "twobody.inp",
        nfrozen=0,
    )

    system.print_info()

    print("   Using P space file: ", args.civecs)
    pspace, excitation_count = get_pspace_from_cipsi(args.civecs, system, nexcit=3)
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

    # Eccp = [-38.9802624745,
    #         -38.9806474054, # 1000 dets
    #         -38.9809970624,
    #         -38.9810531086,
    #         -38.9810592029]

    #Hbar = build_hbar_ccsd(T, H)

    #L, total_energy, is_converged = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None)

    #Eccp3, deltap3 = calc_ccp3(T, L, Hbar, H, system, pspace, use_RHF=False)

    #Eccsd = -38.9802624745
    #Ecrcc23 = -38.9810572210
    #Eccsdt = -38.98105947

    #Eccp = total_energy
    #Eccpq = Eccp3['D']

    # print("CCSDT = ", Eccsdt)
    # print("Error in CCSD = ", (Eccsd - Eccsdt) * 1000, "mEh")
    # print("Error in CR-CC(2,3) = ", (Ecrcc23 - Eccsdt) * 1000, "mEh")
    # print("Error in CC(P) = ", (Eccp - Eccsdt) * 1000, "mEh")
    # print("Error in CC(P;Q) = ", (Eccpq - Eccsdt) * 1000, "mEh")
    # print("Number of triples = ", num_excits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the CIPSI-driven CC(P;Q) calculation using the 2BA for a specific CI vector file.")
    parser.add_argument("-civecs", type=str,
                        help="Path to processed CI vector file containing list of all determinants in spinorbital occupation notation.")
    args = parser.parse_args()
    main(args)



