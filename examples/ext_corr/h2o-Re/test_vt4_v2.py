import numpy as np
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.utilities.external_correction import parse_ci_wavefunction, cluster_analyze_ci
from itertools import permutations

def contract_vt4_exact(H, T_ext):

    # < ijab | [V_N, T4] | 0 >
    x2_aa_exact = np.einsum("mnef,abefijmn->abij", H.ab.oovv, T_ext.aaab, optimize=True)

    return x2_aa_exact

def contract_vt4_matel(C4_excitations, C4_amplitudes, H, T_ext, system):

    x2a = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha))
    # Loop over aaab determinants
    for idet in range(len(C4_amplitudes["aaab"])):

        print("quadruple", idet, "/", len(C4_amplitudes["aaab"]))
        c, d, e, f, k, l, m, n = [x - 1 for x in C4_excitations["aaab"][idet]]

        t_amp = T_ext.aaab[c, d, e, f, k, l, m, n]

        for a in range(system.nunoccupied_alpha):
            for b in range(a + 1, system.nunoccupied_alpha):
                for i in range(system.noccupied_alpha):
                    for j in range(i + 1, system.noccupied_alpha):

                        # x2a(abij) <- A(ij)A(ab)A(m/kl)A(e/cd) delta(i,k)*delta(j,l)*delta(a,c)*delta(b,d)*v_ab(m,n,e,f)
                        hmatel = (
                                  (i == k) * (j == l) * (a == c) * (b == d) * H.ab.oovv[m, n, e, f]
                                - (i == k) * (j == l) * (a == e) * (b == d) * H.ab.oovv[m, n, c, f]
                                - (i == k) * (j == l) * (a == c) * (b == e) * H.ab.oovv[m, n, d, f]
                                - (i == m) * (j == l) * (a == c) * (b == d) * H.ab.oovv[k, n, e, f]
                                + (i == m) * (j == l) * (a == e) * (b == d) * H.ab.oovv[k, n, c, f]
                                + (i == m) * (j == l) * (a == c) * (b == e) * H.ab.oovv[k, n, d, f]
                                - (i == k) * (j == m) * (a == c) * (b == d) * H.ab.oovv[l, n, e, f]
                                + (i == k) * (j == m) * (a == e) * (b == d) * H.ab.oovv[l, n, c, f]
                                + (i == k) * (j == m) * (a == c) * (b == e) * H.ab.oovv[l, n, d, f]
                                - (i == k) * (j == l) * (b == c) * (a == d) * H.ab.oovv[m, n, e, f]
                                + (i == k) * (j == l) * (b == e) * (a == d) * H.ab.oovv[m, n, c, f]
                                + (i == k) * (j == l) * (b == c) * (a == e) * H.ab.oovv[m, n, d, f]
                                + (i == m) * (j == l) * (b == c) * (a == d) * H.ab.oovv[k, n, e, f]
                                - (i == m) * (j == l) * (b == e) * (a == d) * H.ab.oovv[k, n, c, f]
                                - (i == m) * (j == l) * (b == c) * (a == e) * H.ab.oovv[k, n, d, f]
                                + (i == k) * (j == m) * (b == c) * (a == d) * H.ab.oovv[l, n, e, f]
                                - (i == k) * (j == m) * (b == e) * (a == d) * H.ab.oovv[l, n, c, f]
                                - (i == k) * (j == m) * (b == c) * (a == e) * H.ab.oovv[l, n, d, f]
                                - (j == k) * (i == l) * (a == c) * (b == d) * H.ab.oovv[m, n, e, f]
                                + (j == k) * (i == l) * (a == e) * (b == d) * H.ab.oovv[m, n, c, f]
                                + (j == k) * (i == l) * (a == c) * (b == e) * H.ab.oovv[m, n, d, f]
                                + (j == m) * (i == l) * (a == c) * (b == d) * H.ab.oovv[k, n, e, f]
                                - (j == m) * (i == l) * (a == e) * (b == d) * H.ab.oovv[k, n, c, f]
                                - (j == m) * (i == l) * (a == c) * (b == e) * H.ab.oovv[k, n, d, f]
                                + (j == k) * (i == m) * (a == c) * (b == d) * H.ab.oovv[l, n, e, f]
                                - (j == k) * (i == m) * (a == e) * (b == d) * H.ab.oovv[l, n, c, f]
                                - (j == k) * (i == m) * (a == c) * (b == e) * H.ab.oovv[l, n, d, f]
                                + (j == k) * (i == l) * (b == c) * (a == d) * H.ab.oovv[m, n, e, f]
                                - (j == k) * (i == l) * (b == e) * (a == d) * H.ab.oovv[m, n, c, f]
                                - (j == k) * (i == l) * (b == c) * (a == e) * H.ab.oovv[m, n, d, f]
                                - (j == m) * (i == l) * (b == c) * (a == d) * H.ab.oovv[k, n, e, f]
                                + (j == m) * (i == l) * (b == e) * (a == d) * H.ab.oovv[k, n, c, f]
                                + (j == m) * (i == l) * (b == c) * (a == e) * H.ab.oovv[k, n, d, f]
                                - (j == k) * (i == m) * (b == c) * (a == d) * H.ab.oovv[l, n, e, f]
                                + (j == k) * (i == m) * (b == e) * (a == d) * H.ab.oovv[l, n, c, f]
                                + (j == k) * (i == m) * (b == c) * (a == e) * H.ab.oovv[l, n, d, f]
                        )

                        x2a[a, b, i, j] += hmatel * t_amp

    return x2a

# def contract_vt4_opt(C4_excitations, C4_amplitudes, H, T_ext, system):
#
#     x2a = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha))
#     # Loop over aaab determinants
#     for idet in range(len(C4_amplitudes["aaab"])):
#
#         # x2a(abij) <- A(m/ij)A(e/ab) v_ab(mnef) * t_aaab(abefijmn)
#
#         # (abefijmn) -> (abij)(efmn)
#         a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aaab"][idet]]
#         t_amp = T_ext.aaab[a, b, e, f, i, j, m, n]
#
#         # (1)
#         hmatel = (
#                   H.ab.oovv[m, n, e, f] - H.ab.oovv[i, n, e, f] - H.ab.oovv[j, n, e, f]
#                 - H.ab.oovv[m, n, a, f] + H.ab.oovv[i, n, a, f] + H.ab.oovv[j, n, a, f]
#                 - H.ab.oovv[m, n, b, f] + H.ab.oovv[i, n, b, f] + H.ab.oovv[j, n, b, f]
#         )
#         x2a[a, b, i, j] += hmatel * t_amp
#
#         # (afmn)
#         hmatel = (
#                   H.ab.oovv[m, n, a, f] - H.ab.oovv[i, n, a, f] - H.ab.oovv[j, n, a, f]
#                 - H.ab.oovv[m, n, e, f] + H.ab.oovv[i, n, e, f] + H.ab.oovv[j, n, e, f]
#                 - H.ab.oovv[m, n, b, f] + H.ab.oovv[i, n, b, f] + H.ab.oovv[j, n, b, f]
#         )
#         x2a[b, e, i, j] -= hmatel * t_amp
#
#     return x2a

if __name__ == "__main__":


    system, H = load_from_gamess(
            "h2o-Re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )
    system.print_info()

    civecs = "ndet_10000/civecs.dat"

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

    # Get the expected result for the contraction, computed using full T_ext
    x2_aa_exact = contract_vt4_exact(H, T_ext)

    # Get the on-the-fly contraction result (although for now, we are still using full T4)
    x2_aa = contract_vt4_matel(C4_excitations, C4_amplitudes, H, T_ext, system)

    error = np.zeros_like(x2_aa)
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    error[a, b, i, j] = x2_aa_exact[a, b, i, j] - x2_aa[a, b, i, j]
                    if abs(error[a, b, i, j]) > 1.0e-012:
                        print(a + 1, b + 1, i + 1, j + 1, "Expected = ", x2_aa_exact[a, b, i, j], "Got = ", x2_aa[a, b, i, j],
                              "Error = ", error[a, b, i, j])
    print(np.linalg.norm(error.flatten()))