import numpy as np

def main():


    from ccpy.models.calculation import Calculation
    from ccpy.interfaces.gamess_tools import load_from_gamess
    from ccpy.drivers.driver import cc_driver, lcc_driver

    from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

    from ccpy.moments.crcc23 import calc_crcc23

    from ccpy.density.rdm1 import calc_rdm1
    from ccpy.density.ccsd_no import convert_to_ccsd_no

    system, H = load_from_gamess(
                "/scratch/gururang/cis7hq/cis7hq.log",
                "/scratch/gururang/cis7hq/onebody.inp",
                "/scratch/gururang/cis7hq/twobody.inp",
                nfrozen=11)
    system.print_info()

    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-07,
        diis_size=6,
        RHF_symmetry=True
    )

    T, total_energy, is_converged = cc_driver(calculation, system, H)

    np.save("/scratch/gururang/cis7hq/t_a.npy", T.a)
    np.save("/scratch/gururang/cis7hq/t_b.npy", T.b)
    np.save("/scratch/gururang/cis7hq/t_aa.npy", T.aa)
    np.save("/scratch/gururang/cis7hq/t_ab.npy", T.ab)
    np.save("/scratch/gururang/cis7hq/t_bb.npy", T.bb)

    calculation = Calculation(
        order=2,
        calculation_type="left_ccsd",
        convergence_tolerance=1.0e-07,
        diis_size=6,
        RHF_symmetry=True
    )

    Hbar = build_hbar_ccsd(T, H)

    L, total_energy, is_converged = lcc_driver(calculation, system, T, Hbar, omega=0.0, L=None, R=None)

    np.save("/scratch/gururang/cis7hq/l_a.npy", L.a)
    np.save("/scratch/gururang/cis7hq/l_b.npy", L.b)
    np.save("/scratch/gururang/cis7hq/l_aa.npy", L.aa)
    np.save("/scratch/gururang/cis7hq/l_ab.npy", L.ab)
    np.save("/scratch/gururang/cis7hq/l_bb.npy", L.bb)

    rdm1 = calc_rdm1(T, L, system)
    H, system = convert_to_ccsd_no(rdm1, H, system)

    # Calculate CR-CC(2,3) ground-state energy correction
    #Ecrcc23, delta23 = calc_crcc23(T, L, Hbar, H, system, use_RHF=True)



if __name__ == "__main__":

    main()



