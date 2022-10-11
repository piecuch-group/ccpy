from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import eccc_driver, lcc_driver
from ccpy.hbar.hbar_ccsd import build_hbar_ccsd
from ccpy.moments.ccp3 import calc_ccp3
from ccpy.utilities.pspace import get_pspace_from_cipsi

if __name__ == "__main__":


    system, H = load_from_gamess(
            "h2o-Re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )
    system.print_info()

    civecs = "ndet_50000/civecs.dat"

    calculation = Calculation(calculation_type="eccc2_slow")
    T, total_energy, converged = eccc_driver(calculation, system, H, external_wavefunction=civecs)

    Hbar = build_hbar_ccsd(T, H)

    calculation = Calculation(calculation_type="left_ccsd")
    L, _, converged = lcc_driver(calculation, system, T, Hbar)

    pspace, excitation_count = get_pspace_from_cipsi(civecs, system, nexcit=3, ordered_index=False)
    Ecrcc23, _ = calc_ccp3(T, L, Hbar, H, system, pspace)

