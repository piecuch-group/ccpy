import numpy as np
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.utilities.external_correction import parse_ci_wavefunction, cluster_analyze_ci
from itertools import permutations

def contract_vt4_exact(H, T_ext):

    # < ijab | [V_N, T4] | 0 >
    #x2_aa_exact = 0.25 * np.einsum("mnef,abefijmn->abij", H.bb.oovv, T_ext.aabb, optimize=True)
    x2_aa_exact = np.einsum("mnef,abefijmn->abij", H.ab.oovv, T_ext.aaab, optimize=True)
    #x2_aa_exact = 0.25 * np.einsum("mnef,abefijmn->abij", H.aa.oovv, T_ext.aaaa, optimize=True)
    #x2_aa_exact -= np.transpose(x2_aa_exact, (1, 0, 2, 3))
    #x2_aa_exact -= np.transpose(x2_aa_exact, (0, 1, 3, 2))

    # < ij~ab~ | [V_N, T4] | 0 >
    x2_ab_exact = 0.25 * np.einsum("mnef,aefbimnj->abij", H.aa.oovv, T_ext.aaab, optimize=True)
    x2_ab_exact += 0.25 * np.einsum("mnef,aefbimnj->abij", H.bb.oovv, T_ext.abbb, optimize=True)
    x2_ab_exact += np.einsum("mnef,aefbimnj->abij", H.ab.oovv, T_ext.aabb, optimize=True)

    # < i~j~a~b~ | [V_N, T4] | 0 >
    x2_bb_exact = 0.25 * 0.25 * np.einsum("mnef,efabmnij->abij", H.aa.oovv, T_ext.aabb, optimize=True)
    x2_bb_exact += 0.25 * np.einsum("mnef,efabmnij->abij", H.ab.oovv, T_ext.abbb, optimize=True)
    x2_bb_exact += 0.25 * 0.25 * np.einsum("mnef,abefijmn->abij", H.bb.oovv, T_ext.bbbb, optimize=True)
    x2_bb_exact -= np.transpose(x2_bb_exact, (1, 0, 2, 3))
    x2_bb_exact -= np.transpose(x2_bb_exact, (0, 1, 3, 2))

    return x2_aa_exact, x2_ab_exact, x2_bb_exact

def contract_vt4_opt(C4_excitations, C4_amplitudes, H, T_ext, system):

    x2a = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha))
    x2b = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta))
    x2c = np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta))

    # Loop over aaaa determinants
    for idet in range(len(C4_amplitudes["aaaa"])):

        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aaaa"][idet]]
        t_amp = T_ext.aaaa[a, b, e, f, i, j, m, n]

        # x2a(abij) <- A(ij/mn)A(ab/ef) v_aa(mnef) * t_aaaa(abefijmn)
        hmatel = (
                H.aa.oovv[m, n, e, f] - H.aa.oovv[i, n, e, f] - H.aa.oovv[m, i, e, f] - H.aa.oovv[j, n, e, f] -
                H.aa.oovv[m, j, e, f] + H.aa.oovv[i, j, e, f]  # (1)
                - H.aa.oovv[m, n, a, f] + H.aa.oovv[i, n, a, f] + H.aa.oovv[m, i, a, f] + H.aa.oovv[j, n, a, f] +
                H.aa.oovv[m, j, a, f] - H.aa.oovv[i, j, a, f]  # -(ae)
                - H.aa.oovv[m, n, e, a] + H.aa.oovv[i, n, e, a] + H.aa.oovv[m, i, e, a] + H.aa.oovv[j, n, e, a] +
                H.aa.oovv[m, j, e, a] - H.aa.oovv[i, j, e, a]  # -(af)
                - H.aa.oovv[m, n, b, f] + H.aa.oovv[i, n, b, f] + H.aa.oovv[m, i, b, f] + H.aa.oovv[j, n, b, f] +
                H.aa.oovv[m, j, b, f] - H.aa.oovv[i, j, b, f]  # -(be)
                - H.aa.oovv[m, n, e, b] + H.aa.oovv[i, n, e, b] + H.aa.oovv[m, i, e, b] + H.aa.oovv[j, n, e, b] +
                H.aa.oovv[m, j, e, b] - H.aa.oovv[i, j, e, b]  # -(bf)
                + H.aa.oovv[m, n, a, b] - H.aa.oovv[i, n, a, b] - H.aa.oovv[m, i, a, b] - H.aa.oovv[j, n, a, b] -
                H.aa.oovv[m, j, a, b] + H.aa.oovv[i, j, a, b]  # +(ae)(bf)
        )

        #x2a[a, b, i, j] += hmatel * t_amp
        # x2a[b, a, i, j] = -1.0 * x2a[a, b, i, j]
        # x2a[a, b, j, i] = -1.0 * x2a[a, b, i, j]
        # x2a[b, a, j, i] = x2a[a, b, i, j]

    # Loop over aaab determinants
    for idet in range(len(C4_amplitudes["aaab"])):
        # x2a(abij) <- A(m/ij)A(e/ab) v_ab(mnef) * t_aaab(abefijmn)
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aaab"][idet]]
        t_amp = T_ext.aaab[a, b, e, f, i, j, m, n]

        hmatel = (
                  H.ab.oovv[m, n, e, f] + H.ab.oovv[i, n, e, f] - H.ab.oovv[j, n, e, f]
                + H.ab.oovv[m, n, a, f] + H.ab.oovv[i, n, a, f] - H.ab.oovv[j, n, a, f]
                - H.ab.oovv[m, n, b, f] - H.ab.oovv[i, n, b, f] + H.ab.oovv[j, n, b, f]
        )
        x2a[a, b, i, j] += hmatel * t_amp
        # x2a[b, a, i, j] = -1.0 * x2a[a, b, i, j]
        # x2a[a, b, j, i] = -1.0 * x2a[a, b, i, j]
        # x2a[b, a, j, i] = x2a[a, b, i, j]

        # x2b(abij) < - A(i/mn)A(a/ef) v_aa(mnef) * t_aaab(aefbimnj)
        a, e, f, b, i, m, n, j = [x - 1 for x in C4_excitations["aaab"][idet]]
        t_amp = T_ext.aaab[a, e, f, b, i, m, n, j]

        hmatel = (
                H.aa.oovv[m, n, e, f] - H.aa.oovv[i, n, e, f] - H.aa.oovv[m, i, e, f]
                - H.aa.oovv[m, n, a, f] + H.aa.oovv[i, n, a, f] + H.aa.oovv[m, i, a, f]
                - H.aa.oovv[m, n, e, a] + H.aa.oovv[i, n, e, a] + H.aa.oovv[m, i, e, a]
        )
        x2b[a, b, i, j] += hmatel * t_amp

    # Loop over aabb determinants
    for idet in range(len(C4_amplitudes["aabb"])):
        # x2a(abij) <- v_bb(mnef) * t_aabb(abefijmn)
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aabb"][idet]]
        t_amp = T_ext.aabb[a, b, e, f, i, j, m, n]

        hmatel = H.bb.oovv[m, n, e, f]

        # x2a[a, b, i, j] += hmatel * t_amp
        # x2a[b, a, i, j] = -1.0 * x2a[a, b, i, j]
        # x2a[a, b, j, i] = -1.0 * x2a[a, b, i, j]
        # x2a[b, a, j, i] = x2a[a, b, i, j]

        # x2b(abij) <- A(jn)A(im)A(ae)A(bf) v_ab(mnef) * t_aabb(aefbimnj)
        a, e, f, b, i, m, n, j = [x - 1 for x in C4_excitations["aabb"][idet]]
        t_amp = T_ext.aabb[a, e, f, b, i, m, n, j]

        hmatel = (
                H.ab.oovv[m, n, e, f]  # (1)
                - H.ab.oovv[i, n, e, f]  # -(im)
                - H.ab.oovv[m, j, e, f]  # -(jn)
                - H.ab.oovv[m, n, a, f]  # -(ae)
                - H.ab.oovv[m, n, e, b]  # -(bf)
                + H.ab.oovv[i, n, a, f]  # +(im)(ae)
                + H.ab.oovv[m, j, a, f]  # +(jn)(ae)
                + H.ab.oovv[i, n, e, b]  # +(im)(bf)
                + H.ab.oovv[m, j, e, b]  # +(jn)(bf)
                + H.ab.oovv[m, n, a, b]  # +(ae)(bf)
                + H.ab.oovv[i, j, e, f]  # +(im)(jn)
                - H.ab.oovv[i, j, a, f]  # -(im)(jn)(ae)
                - H.ab.oovv[i, j, e, b]  # -(im)(jn)(bf)
                - H.ab.oovv[i, n, a, b]  # -(im)(ae)(bf)
                - H.ab.oovv[m, j, a, b]  # -(jn)(ae)(bf)
                + H.ab.oovv[i, j, a, b]  # +(im)(jn)(ae)(bf)
        )

        x2b[a, b, i, j] += hmatel * t_amp

        # x2c(abij) <- v_aa(mnef) * t_aabb(efabmnij)
        e, f, a, b, m, n, i, j = [x - 1 for x in C4_excitations["aabb"][idet]]
        t_amp = T_ext.aabb[e, f, a, b, m, n, i, j]

        hmatel = H.aa.oovv[m, n, e, f]

        x2c[a, b, i, j] += hmatel * t_amp
        x2c[a, b, j, i] = -1.0 * x2c[a, b, i, j]
        x2c[b, a, i, j] = -1.0 * x2c[a, b, i, j]
        x2c[b, a, j, i] = x2c[a, b, i, j]

    # Loop over abbb determinants
    for idet in range(len(C4_amplitudes["abbb"])):
        # x2b(abij) < - A(j/mn)A(b/ef) v_bb(mnef) * t_abbb(aefbimnj)
        a, e, f, b, i, m, n, j = [x - 1 for x in C4_excitations["abbb"][idet]]
        t_amp = T_ext.abbb[a, e, f, b, i, m, n, j]

        hmatel = (
                H.bb.oovv[m, n, e, f] - H.bb.oovv[j, n, e, f] - H.bb.oovv[m, j, e, f]
                - H.bb.oovv[m, n, b, f] + H.bb.oovv[j, n, b, f] + H.bb.oovv[m, j, b, f]
                - H.bb.oovv[m, n, e, b] + H.bb.oovv[j, n, e, b] + H.bb.oovv[m, j, e, b]
        )
        x2b[a, b, i, j] += hmatel * t_amp

        # x2c(abij) <- A(n/ij)A(f/ab) v_ab(mnef) * t_abbb(efabmnij)
        e, a, f, b, m, i, n, j = [x - 1 for x in C4_excitations["abbb"][idet]]
        t_amp = T_ext.abbb[e, a, f, b, m, i, n, j]

        hmatel = (
                H.ab.oovv[m, n, e, f] - H.ab.oovv[m, i, e, f] - H.ab.oovv[m, j, e, f]
                - H.ab.oovv[m, n, e, a] + H.ab.oovv[m, i, e, a] + H.ab.oovv[m, j, e, a]
                - H.ab.oovv[m, n, e, b] + H.ab.oovv[m, i, e, b] + H.ab.oovv[m, j, e, b]
        )

        x2c[a, b, i, j] += hmatel * t_amp
        x2c[a, b, j, i] = -1.0 * x2c[a, b, i, j]
        x2c[b, a, i, j] = -1.0 * x2c[a, b, i, j]
        x2c[b, a, j, i] = x2c[a, b, i, j]

    # Loop over bbbb determinants
    for idet in range(len(C4_amplitudes["bbbb"])):
        a, e, f, b, i, m, n, j = [x - 1 for x in C4_excitations["bbbb"][idet]]
        t_amp = T_ext.bbbb[a, e, f, b, i, m, n, j]

        # x2c(abij) <- A(ij/mn)A(ab/ef) v_bb(mnef) * t_bbbb(abefijmn)
        hmatel = (
                H.bb.oovv[m, n, e, f] - H.bb.oovv[i, n, e, f] - H.bb.oovv[m, i, e, f] - H.bb.oovv[j, n, e, f] -
                H.bb.oovv[m, j, e, f] + H.bb.oovv[i, j, e, f]  # (1)
                - H.bb.oovv[m, n, a, f] + H.bb.oovv[i, n, a, f] + H.bb.oovv[m, i, a, f] + H.bb.oovv[j, n, a, f] +
                H.bb.oovv[m, j, a, f] - H.bb.oovv[i, j, a, f]  # -(ae)
                - H.bb.oovv[m, n, e, a] + H.bb.oovv[i, n, e, a] + H.bb.oovv[m, i, e, a] + H.bb.oovv[j, n, e, a] +
                H.bb.oovv[m, j, e, a] - H.bb.oovv[i, j, e, a]  # -(af)
                - H.bb.oovv[m, n, b, f] + H.bb.oovv[i, n, b, f] + H.bb.oovv[m, i, b, f] + H.bb.oovv[j, n, b, f] +
                H.bb.oovv[m, j, b, f] - H.bb.oovv[i, j, b, f]  # -(be)
                - H.bb.oovv[m, n, e, b] + H.bb.oovv[i, n, e, b] + H.bb.oovv[m, i, e, b] + H.bb.oovv[j, n, e, b] +
                H.bb.oovv[m, j, e, b] - H.bb.oovv[i, j, e, b]  # -(bf)
                + H.bb.oovv[m, n, a, b] - H.bb.oovv[i, n, a, b] - H.bb.oovv[m, i, a, b] - H.bb.oovv[j, n, a, b] -
                H.bb.oovv[m, j, a, b] + H.bb.oovv[i, j, a, b]  # +(ae)(bf)
        )
        x2c[a, b, i, j] += hmatel * t_amp
        x2c[b, a, i, j] = -1.0 * x2c[a, b, i, j]
        x2c[a, b, j, i] = -1.0 * x2c[a, b, i, j]
        x2c[b, a, j, i] = x2c[a, b, i, j]

    return x2a, x2b, x2c

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
    x2_aa_exact, x2_ab_exact, x2_bb_exact = contract_vt4_exact(H, T_ext)

    # Get the on-the-fly contraction result (although for now, we are still using full T4)
    x2_aa, x2_ab, x2_bb = contract_vt4_opt(C4_excitations, C4_amplitudes, H, T_ext, system)

    #print("Error in aa = ", np.linalg.norm((x2_aa_exact - x2_aa).flatten()))
    #print("Error in ab = ", np.linalg.norm(x2_ab_exact.flatten() - x2_ab.flatten()))
    error = np.zeros_like(x2_aa)
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    error[a, b, i, j] = x2_aa_exact[a, b, i, j] - x2_aa[a, b, i, j]
                    if abs(error[a, b, i, j]) > 1.0e-012:
                        print(a + 1, b + 1, i + 1, j + 1, "Expected = ", x2_aa_exact[a, b, i, j], "Got = ", x2_aa[a, b, i, j],
                              "Error = ", error[a, b, i, j])
    # print(error)
    # print("Error in aa = ", np.linalg.norm(error.flatten()))
    #
    # error = 0.0
    # for a in range(system.nunoccupied_alpha):
    #     for b in range(system.nunoccupied_beta):
    #         for i in range(system.noccupied_alpha):
    #             for j in range(system.noccupied_beta):
    #                 error += x2_ab_exact[a, b, i, j] - x2_ab[a, b, i, j]
    # print("Error in ab = ", error)
    #
    # error = 0.0
    # for a in range(system.nunoccupied_beta):
    #     for b in range(system.nunoccupied_beta):
    #         for i in range(system.noccupied_beta):
    #             for j in range(system.noccupied_beta):
    #                 error += x2_bb_exact[a, b, i, j] - x2_bb[a, b, i, j]
    # print("Error in bb = ", error)

    # Former magic sign trick:..
    # hmatel = (
    #         H.ab.oovv[m, n, e, f] + H.ab.oovv[i, n, e, f] - H.ab.oovv[j, n, e, f]
    #         + H.ab.oovv[m, n, a, f] + H.ab.oovv[i, n, a, f] - H.ab.oovv[j, n, a, f]
    #         - H.ab.oovv[m, n, b, f] - H.ab.oovv[i, n, b, f] + H.ab.oovv[j, n, b, f]
    # )