import argparse
import numpy as np

def main(args):

    from ccpy.models.calculation import Calculation
    from ccpy.drivers.driver import cc_driver, lcc_driver
    from ccpy.interfaces.gamess_tools import load_from_gamess

    from ccpy.utilities.pspace import get_pspace_from_cipsi
    from ccpy.utilities.utilities import read_amplitudes_from_jun

    from ccpy.hbar.hbar_ccsdt import build_hbar_ccsdt
    from ccpy.moments.ccp3 import calc_ccp3_full

    system, H = load_from_gamess(
            "/scratch/gururang/test_ccpq_2ba/f2-2Re/f2-2Re.log",
            "/scratch/gururang/test_ccpq_2ba/f2-2Re/onebody.inp",
            "/scratch/gururang/test_ccpq_2ba/f2-2Re/twobody.inp",
            nfrozen=2,
    )

    system.print_info()

    #T_jun = read_amplitudes_from_jun("/home2/gururang/ccpy/examples/f2/cipsi-ccpq/f2-2Re/ndet_5000/f2.ccsdt10",
    #                                 system,
    #                                 3,
    #                                 amp_type="T")

    #L_jun = read_amplitudes_from_jun("/home2/gururang/ccpy/examples/f2/cipsi-ccpq/f2-2Re/ndet_5000/L-CCSDt",
    #                                 system,
    #                                 3,
    #                                 amp_type="L")

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
            calculation_type="ccsdt_p_slow",
            convergence_tolerance=1.0e-08,
            diis_size=6,
            energy_shift=0.0,
            maximum_iterations=500,
            RHF_symmetry=True,
            low_memory=False 
    )   

    T, total_energy, is_converged = cc_driver(calculation, system, H, pspace=pspace)

    #print("Error in T1A = ", np.linalg.norm(T_jun.a.flatten() - T.a.flatten()))
    #print("Error in T1B = ", np.linalg.norm(T_jun.b.flatten() - T.b.flatten()))
    #print("Error in T2A = ", np.linalg.norm(T_jun.aa.flatten() - T.aa.flatten()))
    #print("Error in T2B = ", np.linalg.norm(T_jun.ab.flatten() - T.ab.flatten()))
    #print("Error in T2C = ", np.linalg.norm(T_jun.bb.flatten() - T.bb.flatten()))
    #print("Error in T3A = ", np.linalg.norm(T_jun.aaa.flatten() - T.aaa.flatten()))
    #print("Error in T3B = ", np.linalg.norm(T_jun.aab.flatten() - T.aab.flatten()))
    #print("Error in T3C = ", np.linalg.norm(T_jun.abb.flatten() - T.abb.flatten()))
    #print("Error in T3D = ", np.linalg.norm(T_jun.bbb.flatten() - T.bbb.flatten()))

    calculation = Calculation(
            calculation_type="left_ccsdt_p",
            convergence_tolerance=1.0e-08,
            maximum_iterations=500,
            energy_shift=0.0,
            RHF_symmetry=True,
            low_memory=False 
    )
    
    Hbar = build_hbar_ccsdt(T, H)
    
    L, total_energy, is_converged = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None, pspace=pspace)

    #print("Error in L1A = ", np.linalg.norm(L_jun.a.flatten() - L.a.flatten()))
    #print("Error in L1B = ", np.linalg.norm(L_jun.b.flatten() - L.b.flatten()))
    #print("Error in L2A = ", np.linalg.norm(L_jun.aa.flatten() - L.aa.flatten()))
    #print("Error in L2B = ", np.linalg.norm(L_jun.ab.flatten() - L.ab.flatten()))
    #print("Error in L2C = ", np.linalg.norm(L_jun.bb.flatten() - L.bb.flatten()))
    #print("Error in L3A = ", np.linalg.norm(L_jun.aaa.flatten() - L.aaa.flatten()))
    #print("Error in L3B = ", np.linalg.norm(L_jun.aab.flatten() - L.aab.flatten()))
    #print("Error in L3C = ", np.linalg.norm(L_jun.abb.flatten() - L.abb.flatten()))
    #print("Error in L3D = ", np.linalg.norm(L_jun.bbb.flatten() - L.bbb.flatten()))
    
    Eccp3, deltap3 = calc_ccp3_full(T, L, Hbar, H, system, pspace, use_RHF=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the CIPSI-driven CC(P;Q) calculation using the full CC(P;Q) triples correction for a specific CI vector file.")
    parser.add_argument("-civecs", type=str, help="Path to processed CI vector file containing list of all determinants in spinorbital occupation notation.")
    args = parser.parse_args()
    main(args)



