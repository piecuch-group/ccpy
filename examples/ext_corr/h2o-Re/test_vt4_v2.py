import numpy as np
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.utilities.external_correction import parse_ci_wavefunction, cluster_analyze_ci
from itertools import permutations

def contract_vt4_exact(H, T_ext):

    # < ijab | [V_N, T4] | 0 >
    x2_aa_exact = np.einsum("mnef,abefijmn->abij", H.ab.oovv, T_ext.aaab, optimize=True)
    x2_aa_exact += 0.25 * np.einsum("mnef,abefijmn->abij", H.bb.oovv, T_ext.aabb, optimize=True)
    #x2_aa_exact += 0.25 * np.einsum("mnef,abefijmn->abij", H.aa.oovv, T_ext.aaaa, optimize=True)

    # <ij~ab~ | [V_N, T4] | 0 >
    x2_ab_exact = 0.25 * np.einsum("mnef,aefbimnj->abij", H.aa.oovv, T_ext.aaab, optimize=True)
    x2_ab_exact += np.einsum("mnef,aefbimnj->abij", H.ab.oovv, T_ext.aabb, optimize=True)
    x2_ab_exact += 0.25 * np.einsum("mnef,aefbimnj->abij", H.bb.oovv, T_ext.abbb, optimize=True)

    return x2_aa_exact, x2_ab_exact

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

def contract_vt4_opt(C4_excitations, C4_amplitudes, H, T_ext, system):

    x2a = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha))
    x2b = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta))
    x2c = np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta))


    # Loop over aaaa determinants
    for idet in range(len(C4_amplitudes["aaaa"])):

        # x2a(abij) <- A(ij/mn)A(ab/ef) v_aa(mnef) * t_aaaa(abefijmn)
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aaaa"][idet]]
        t_amp = T_ext.aaaa[a, b, e, f, i, j, m, n]
        #  (1)
        #  (im)
        #  (in)
        #  (jm)
        #  (jn)
        #  (im)(jn)
        #  (ae)
        #  (af)
        #  (be)
        #  (bf)
        #  (ae)(bf)
        #  (im)

    # Loop over aaab determinants
    for idet in range(len(C4_amplitudes["aaab"])):

        # x2a(abij) <- A(ij)A(ab)A(m/kl)A(e/cd) delta(i,k)*delta(j,l)*delta(a,c)*delta(b,d)*v_ab(m,n,e,f)
        #           <- A(m/ij)A(e/ab) v_ab(mnef) * t_aaab(abefijmn)
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aaab"][idet]]
        t_amp = T_ext.aaab[a, b, e, f, i, j, m, n]

        x2a[a, b, i, j] += H.ab.oovv[m, n, e, f] * t_amp # (1)
        x2a[b, a, i, j] = -x2a[a, b, i, j]
        x2a[a, b, j, i] = -x2a[a, b, i, j]
        x2a[b, a, j, i] = x2a[a, b, i, j]

        x2a[e, b, i, j] -= H.ab.oovv[m, n, a, f] * t_amp # (ae)
        x2a[e, b, j, i] = -x2a[e, b, i, j]
        x2a[b, e, i, j] = -x2a[e, b, i, j]
        x2a[b, e, j, i] = x2a[e, b, i, j]

        x2a[a, e, i, j] -= H.ab.oovv[m, n, b, f] * t_amp # (be)
        x2a[a, e, j, i] = -x2a[a, e, i, j]
        x2a[e, a, i, j] = -x2a[a, e, i, j]
        x2a[e, a, j, i] = x2a[a, e, i, j]

        x2a[a, b, m, j] -= H.ab.oovv[i, n, e, f] * t_amp # (mi)
        x2a[a, b, j, m] = -x2a[a, b, m, j]
        x2a[b, a, m, j] = -x2a[a, b, m, j]
        x2a[b, a, j, m] = x2a[a, b, m, j]

        x2a[e, b, m, j] += H.ab.oovv[i, n, a, f] * t_amp # (mi)(ae)
        x2a[e, b, j, m] = -x2a[e, b, m, j]
        x2a[b, e, m, j] = -x2a[e, b, m, j]
        x2a[b, e, j, m] = x2a[e, b, m, j]

        x2a[a, e, m, j] += H.ab.oovv[i, n, b, f] * t_amp # (mi)(be)
        x2a[a, e, j, m] = -x2a[a, e, m, j]
        x2a[e, a, m, j] = -x2a[a, e, m, j]
        x2a[e, a, j, m] = x2a[a, e, m, j]

        x2a[a, b, i, m] -= H.ab.oovv[j, n, e, f] * t_amp # (mj)
        x2a[a, b, m, i] = -x2a[a, b, i, m]
        x2a[b, a, i, m] = -x2a[a, b, i, m]
        x2a[b, a, m, i] = x2a[a, b, i, m]

        x2a[e, b, i, m] += H.ab.oovv[j, n, a, f] * t_amp # (ae)(mj)
        x2a[e, b, m, i] = -x2a[e, b, i, m]
        x2a[b, e, i, m] = -x2a[e, b, i, m]
        x2a[b, e, m, i] = x2a[e, b, i, m]

        x2a[a, e, i, m] += H.ab.oovv[j, n, b, f] * t_amp # (be)(mj)
        x2a[a, e, m, i] = -x2a[a, e, i, m]
        x2a[e, a, i, m] = -x2a[a, e, i, m]
        x2a[e, a, m, i] = x2a[a, e, i, m]

        # x2b(abij) < - A(i/mn)A(a/ef) v_aa(mnef) * t_aaab(aefbimnj)
        a, e, f, b, i, m, n, j = [x - 1 for x in C4_excitations["aaab"][idet]]
        t_amp = T_ext.aaab[a, e, f, b, i, m, n, j]

        x2b[a, b, i, j] += H.aa.oovv[m, n, e, f] * t_amp  # (1)
        x2b[a, b, m, j] -= H.aa.oovv[i, n, e, f] * t_amp  # (im)
        x2b[a, b, n, j] -= H.aa.oovv[m, i, e, f] * t_amp  # (in)
        x2b[e, b, i, j] -= H.aa.oovv[m, n, a, f] * t_amp  # (ae)
        x2b[f, b, i, j] -= H.aa.oovv[m, n, e, a] * t_amp  # (af)
        x2b[e, b, m, j] += H.aa.oovv[i, n, a, f] * t_amp  # (ae)(im)
        x2b[e, b, n, j] += H.aa.oovv[m, i, a, f] * t_amp  # (ae)(in)
        x2b[f, b, m, j] += H.aa.oovv[i, n, e, a] * t_amp  # (af)(im)
        x2b[f, b, n, j] += H.aa.oovv[m, i, e, a] * t_amp  # (af)(in)

    # Loop over aabb determinants
    for idet in range(len(C4_amplitudes["aabb"])):

        # x2a(abij) <- v_bb(mnef) * t_aabb(abefijmn)
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aabb"][idet]]
        t_amp = T_ext.aabb[a, b, e, f, i, j, m, n]

        x2a[a, b, i, j] += H.bb.oovv[m, n, e, f] * t_amp
        x2a[b, a, i, j] = -x2a[a, b, i, j]
        x2a[a, b, j, i] = -x2a[a, b, i, j]
        x2a[b, a, j, i] = x2a[a, b, i, j]

        # x2b(abij) <- A(ae)A(bf)A(im)(jn) v_ab(mnef) * t_aabb(aefbimnj)
        a, e, f, b, i, m, n, j = [x - 1 for x in C4_excitations["aabb"][idet]]
        t_amp = T_ext.aabb[a, e, f, b, i, m, n, j]

        x2b[a, b, i, j] += H.ab.oovv[m, n, e, f] * t_amp  # (1)
        x2b[e, b, i, j] -= H.ab.oovv[m, n, a, f] * t_amp  # (ae)
        x2b[a, f, i, j] -= H.ab.oovv[m, n, e, b] * t_amp  #  (bf)
        x2b[a, b, m, j] -= H.ab.oovv[i, n, e, f] * t_amp  #  (im)
        x2b[a, b, i, n] -= H.ab.oovv[m, j, e, f] * t_amp  #  (jn)
        x2b[e, f, i, j] += H.ab.oovv[m, n, a, b] * t_amp  #  (ae)(bf)
        x2b[e, b, m, j] += H.ab.oovv[i, n, a, f] * t_amp  #  (ae)(im)
        x2b[e, b, i, n] += H.ab.oovv[m, j, a, f] * t_amp  #  (ae)(jn)
        x2b[a, f, m, j] += H.ab.oovv[i, n, e, b] * t_amp  #  (bf)(im)
        x2b[a, f, i, n] += H.ab.oovv[m, j, e, b] * t_amp  #  (bf)(jn)
        x2b[a, b, m, n] += H.ab.oovv[i, j, e, f] * t_amp  #  (im)(jn)
        x2b[e, f, m, j] -= H.ab.oovv[i, n, a, b] * t_amp  #  (ae)(bf)(im)
        x2b[e, f, i, n] -= H.ab.oovv[m, j, a, b] * t_amp  #  (ae)(bf)(jn)
        x2b[e, b, m, n] -= H.ab.oovv[i, j, a, f] * t_amp  #  (ae)(im)(jn)
        x2b[a, f, m, n] -= H.ab.oovv[i, j, e, b] * t_amp  #  (bf)(im)(jn)
        x2b[e, f, m, n] += H.ab.oovv[i, j, a, b] * t_amp  #  (ae)(bf)(im)(jn)

    # Loop over abbb determinants
    for idet in range(len(C4_amplitudes["abbb"])):

        # x2b(abij) <- A(j/mn)A(b/ef) v_bb(mnef) * t_abbb(aefbimnj)
        a, e, f, b, i, m, n, j = [x - 1 for x in C4_excitations["abbb"][idet]]
        t_amp = T_ext.abbb[a, e, f, b, i, m, n, j]

        x2b[a, b, i, j] += H.bb.oovv[m, n, e, f] * t_amp  #  (1)
        x2b[a, b, i, m] -= H.bb.oovv[j, n, e, f] * t_amp  #  (jm)
        x2b[a, b, i, n] -= H.bb.oovv[m, j, e, f] * t_amp  #  (jn)
        x2b[a, e, i, j] -= H.bb.oovv[m, n, b, f] * t_amp  #  (be)
        x2b[a, f, i, j] -= H.bb.oovv[m, n, e, b] * t_amp  #  (bf)
        x2b[a, e, i, m] += H.bb.oovv[j, n, b, f] * t_amp  #  (jm)(be)
        x2b[a, e, i, n] += H.bb.oovv[m, j, b, f] * t_amp  #  (jn)(be)
        x2b[a, f, i, m] += H.bb.oovv[j, n, e, b] * t_amp  #  (jm)(bf)
        x2b[a, f, i, n] += H.bb.oovv[m, j, e, b] * t_amp  #  (jn)(bf)


    return x2a, x2b

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

    # Get the expected result for the contraction, computed using full T_ext
    x2_aa_exact, x2_ab_exact = contract_vt4_exact(H, T_ext)

    # Get the on-the-fly contraction result (although for now, we are still using full T4)
    x2_aa, x2_ab = contract_vt4_opt(C4_excitations, C4_amplitudes, H, T_ext, system)


    print("")
    print("Error in x2a = ", np.linalg.norm(x2_aa.flatten() - x2_aa_exact.flatten()))
    print("Error in x2b = ", np.linalg.norm(x2_ab.flatten() - x2_ab_exact.flatten()))