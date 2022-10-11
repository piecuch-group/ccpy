from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import eccc_driver, lcc_driver
from ccpy.drivers.cc_energy import get_ci_energy, get_cc_energy
from ccpy.hbar.hbar_ccsd import build_hbar_ccsd
from ccpy.moments.ccp3 import calc_ccp3
from ccpy.utilities.external_correction import parse_ci_wavefunction, cluster_analysis
from ccpy.utilities.pspace import get_pspace_from_cipsi

if __name__ == "__main__":


    system, H = load_from_gamess(
            "h2o-Re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )
    system.print_info()

    civecs = "ndet_5000/civecs.dat"
    C, C4_excits, C4_amps, excitation_count = parse_ci_wavefunction(civecs, system)

    print("Excitation Content")
    print("-------------------")
    print("Number of singles = ", excitation_count['a'] + excitation_count['b'])
    print("Number of doubles = ", excitation_count['aa'] + excitation_count['ab'] + excitation_count['bb'])
    print("Number of triples = ", excitation_count['aaa'] + excitation_count['aab'] + excitation_count['abb'] + excitation_count['bbb'])
    print("Number of quadruples = ", excitation_count['aaaa'] + excitation_count['aaab'] + excitation_count['aabb'] + excitation_count['abbb'] + excitation_count['bbbb'])
    print("")

    Ecorr_c = get_ci_energy(C, H)
    print("External correction energy = ", Ecorr_c)

    T_ext = cluster_analysis(C, C4_excits, C4_amps, system)

    Ecorr_t = get_cc_energy(T_ext, H)
    print("T vector energy = ", Ecorr_t)

    calculation = Calculation(calculation_type="eccc2_slow")
    T, total_energy, converged = eccc_driver(calculation, system, H, T_ext)

    Hbar = build_hbar_ccsd(T, H)

    calculation = Calculation(calculation_type="left_ccsd")
    L, _, converged = lcc_driver(calculation, system, T, Hbar)

    pspace, excitation_count = get_pspace_from_cipsi(civecs, system, nexcit=3, ordered_index=False)
    Ecrcc23, _ = calc_ccp3(T, L, Hbar, H, system, pspace)

