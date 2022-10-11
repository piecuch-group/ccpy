import numpy as np

from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver
from ccpy.drivers.cc_energy import get_ci_energy, get_cc_energy

from ccpy.utilities.external_correction import parse_ci_wavefunction, cluster_analysis

if __name__ == "__main__":


    system, H = load_from_gamess(
            "h2o-Re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )
    system.print_info()

    C, _, _, excitation_count = parse_ci_wavefunction("ndet_50000/civecs.dat", system, full_quadruples=True)

    print("Excitation Content")
    print("-------------------")
    print("Number of singles = ", excitation_count['a'] + excitation_count['b'])
    print("Number of doubles = ", excitation_count['aa'] + excitation_count['ab'] + excitation_count['bb'])
    print("Number of triples = ", excitation_count['aaa'] + excitation_count['aab'] + excitation_count['abb'] + excitation_count['bbb'])
    print("Number of quadruples = ", excitation_count['aaaa'] + excitation_count['aaab'] + excitation_count['aabb'] + excitation_count['abbb'] + excitation_count['bbbb'])
    print("")

    Ecorr_c = get_ci_energy(C, H)

    print("External correction energy = ", Ecorr_c)

    T = cluster_analysis(C, system)

    Ecorr_t = get_cc_energy(T, H)
    print("T vector energy = ", Ecorr_t)

    assert(abs(Ecorr_t - Ecorr_c) < 1.0e-07)