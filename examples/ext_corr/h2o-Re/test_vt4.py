import numpy as np

from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.utilities.external_correction import parse_ci_wavefunction, cluster_analyze_ci

if __name__ == "__main__":


    system, H = load_from_gamess(
            "h2o-Re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )
    system.print_info()

    civecs = "ndet_50000/civecs.dat"

    # Parse the CI wave function to get C1 - C3 and list of C4
    print("   Reading the CI vector file at", civecs)
    C, C4_excitations, C4_amplitudes, excitation_count = parse_ci_wavefunction(civecs, system)

    # Print the excitation content
    print("")
    print("   Excitation Content")
    print("   -------------------")
    print("   Number of singles = ", excitation_count['a'] + excitation_count['b'])
    print("   Number of doubles = ", excitation_count['aa'] + excitation_count['ab'] + excitation_count['bb'])
    print("   Number of triples = ", excitation_count['aaa'] + excitation_count['aab'] + excitation_count['abb'] + excitation_count['bbb'])
    print("   Number of quadruples = ", excitation_count['aaaa'] + excitation_count['aaab'] + excitation_count['aabb'] + excitation_count['abbb'] + excitation_count['bbbb'])
    print("")

    # Perform cluster analysis
    print("   Computing T from CI amplitudes")
    T_ext = cluster_analyze_ci(C, C4_excitations, C4_amplitudes, system, 4)

    x2_aa_exact = 0.25 * 0.25 * np.einsum("mnef,abefijmn->abij", H.bb.oovv, T_ext.aabb, optimize=True)
    x2_aa_exact += 0.25 * np.einsum("mnef,abefijmn->abij", H.ab.oovv, T_ext.aaab, optimize=True)

    x2_aa_exact -= np.transpose(x2_aa_exact, (1, 0, 2, 3))
    x2_aa_exact -= np.transpose(x2_aa_exact, (0, 1, 3, 2))

    x2 = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha))
    for idet in range(len(C4_amplitudes["aaab"])):
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aaab"][idet]]

        t_amp = T_ext.aaab[a, b, e, f, i, j, m, n]

        # I don't know why, but this sign structure works... H.ab.oovv[i, n, e, f] -> + and H.ab.oovv[m, n, a, f] -> + FOR SOME REASON???
        hmatel = (
                     H.ab.oovv[m, n, e, f] + H.ab.oovv[i, n, e, f] - H.ab.oovv[j, n, e, f]
                    +H.ab.oovv[m, n, a, f] + H.ab.oovv[i, n, a, f] - H.ab.oovv[j, n, a, f]
                    -H.ab.oovv[m, n, b, f] - H.ab.oovv[i, n, b, f] + H.ab.oovv[j, n, b, f]
        )
        x2[a, b, i, j] += hmatel * t_amp
        x2[b, a, i, j] = -1.0 * x2[a, b, i, j]
        x2[a, b, j, i] = -1.0 * x2[a, b, i, j]
        x2[b, a, j, i] = x2[a, b, i, j]

    for idet in range(len(C4_amplitudes["aabb"])):
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aabb"][idet]]

        t_amp = T_ext.aabb[a, b, e, f, i, j, m, n]

        hmatel = H.bb.oovv[m, n, e, f]

        x2[a, b, i, j] += hmatel * t_amp
        x2[b, a, i, j] = -1.0 * x2[a, b, i, j]
        x2[a, b, j, i] = -1.0 * x2[a, b, i, j]
        x2[b, a, j, i] = x2[a, b, i, j]

    # for idet in range(len(C4_amplitudes["aaaa"])):
    #
    #     a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aaaa"][idet]]
    #
    #     hmatel = (
    #              H.aa.oovv[m, n, e, f] - H.aa.oovv[i, n, e, f] - H.aa.oovv[m, i, e, f] - H.aa.oovv[j, n, e, f] - H.aa.oovv[m, j, e, f] + H.aa.oovv[i, j, e, f] #  (1)
    #             -H.aa.oovv[m, n, a, f] + H.aa.oovv[i, n, a, f] + H.aa.oovv[m, i, a, f] + H.aa.oovv[j, n, a, f] + H.aa.oovv[m, j, a, f] - H.aa.oovv[i, j, a, f] # -(ae)
    #             -H.aa.oovv[m, n, e, a] + H.aa.oovv[i, n, e, a] + H.aa.oovv[m, i, e, a] + H.aa.oovv[j, n, e, a] + H.aa.oovv[m, j, e, a] - H.aa.oovv[i, j, e, a] # -(af)
    #             -H.aa.oovv[m, n, b, f] + H.aa.oovv[i, n, b, f] + H.aa.oovv[m, i, b, f] + H.aa.oovv[j, n, b, f] + H.aa.oovv[m, j, b, f] - H.aa.oovv[i, j, b, f] # -(be)
    #             -H.aa.oovv[m, n, e, b] + H.aa.oovv[i, n, e, b] + H.aa.oovv[m, i, e, b] + H.aa.oovv[j, n, e, b] + H.aa.oovv[m, j, e, b] - H.aa.oovv[i, j, e, b] # -(bf)
    #             +H.aa.oovv[m, n, a, b] - H.aa.oovv[i, n, a, b] - H.aa.oovv[m, i, a, b] - H.aa.oovv[j, n, a, b] - H.aa.oovv[m, j, a, b] + H.aa.oovv[i, j, a, b] # +(ae)(bf)
    #     )
    #     x2[a, b, i, j] += hmatel * T_ext.aaaa[a, b, e, f, i, j, m, n]

    error = 0.0
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    error += x2_aa_exact[a, b, i, j] - x2[a, b, i, j]
    print("Error = ", error)
