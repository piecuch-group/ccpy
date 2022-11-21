import numpy as np
import time

from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.extcorr.external_correction import parse_ci_wavefunction, cluster_analyze_ci

def contract_vt4_exact(H, T_ext):

    x3a = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha))
    x3b = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta))
    x3c = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta, system.noccupied_beta))
    x3d = np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta, system.noccupied_beta))

    x3a = (3.0 / 36.0) * 0.5 * np.einsum("bnef,aecfijkn->abcijk", H.aa.vovv, T_ext.aaaa, optimize=True)

    x3a -= np.transpose(x3a, (0, 1, 2, 3, 5, 4)) # (jk)
    x3a -= np.transpose(x3a, (0, 1, 2, 4, 3, 5)) + np.transpose(x3a, (0, 1, 2, 5, 4, 3)) # (i/jk)
    x3a -= np.transpose(x3a, (0, 2, 1, 3, 4, 5)) # (bc)
    x3a -= np.transpose(x3a, (2, 1, 0, 3, 4, 5)) + np.transpose(x3a, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return x3a, x3b, x3c, x3d

def contract_vt4_fly(H, T4_excitations, T4_amplitudes, system):

    x3a = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha))
    x3b = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta))
    x3c = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta, system.noccupied_beta))
    x3d = np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta, system.noccupied_beta))

    # Loop over aaaa determinants
    for idet in range(len(T4_amplitudes["aaaa"])):

        # Get the particular aaaa T4 amplitude
        t_amp = T4_amplitudes["aaaa"][idet]

        # x3a(abcijk) <- A(ijk)A(ac)A(b/ac)[ A(ac/ef)A(n/ijk) h_aa(bnef) * t_aaaa(aecfijkn) ]
        #             = A(ijk)A(abc)[ A(ac/ef)A(n/ijk) h_aa(bnef) * t_aaaa(aecfijkn) ]
        a, e, c, f, i, j, k, n = [x - 1 for x in T4_excitations["aaaa"][idet]]

        # (1)
        x3a[a, :, c, i, j, k] += H.aa.vovv[:, n, e, f] * t_amp # (1)
        x3a[e, :, c, i, j, k] -= H.aa.vovv[:, n, a, f] * t_amp # (ae)
        x3a[f, :, c, i, j, k] -= H.aa.vovv[:, n, e, a] * t_amp # (af)
        x3a[a, :, e, i, j, k] -= H.aa.vovv[:, n, c, f] * t_amp # (ce)
        x3a[a, :, f, i, j, k] -= H.aa.vovv[:, n, e, c] * t_amp # (cf)
        x3a[e, :, f, i, j, k] += H.aa.vovv[:, n, a, c] * t_amp # (ae)(cf)

        # (in)
        x3a[a, :, c, n, j, k] -= H.aa.vovv[:, i, e, f] * t_amp # (1)
        x3a[e, :, c, n, j, k] += H.aa.vovv[:, i, a, f] * t_amp # (ae)
        x3a[f, :, c, n, j, k] += H.aa.vovv[:, i, e, a] * t_amp # (af)
        x3a[a, :, e, n, j, k] += H.aa.vovv[:, i, c, f] * t_amp # (ce)
        x3a[a, :, f, n, j, k] += H.aa.vovv[:, i, e, c] * t_amp # (cf)
        x3a[e, :, f, n, j, k] -= H.aa.vovv[:, i, a, c] * t_amp # (ae)(cf)

        # (jn)
        x3a[a, :, c, i, n, k] -= H.aa.vovv[:, j, e, f] * t_amp # (1)
        x3a[e, :, c, i, n, k] += H.aa.vovv[:, j, a, f] * t_amp # (ae)
        x3a[f, :, c, i, n, k] += H.aa.vovv[:, j, e, a] * t_amp # (af)
        x3a[a, :, e, i, n, k] += H.aa.vovv[:, j, c, f] * t_amp # (ce)
        x3a[a, :, f, i, n, k] += H.aa.vovv[:, j, e, c] * t_amp # (cf)
        x3a[e, :, f, i, n, k] -= H.aa.vovv[:, j, a, c] * t_amp # (ae)(cf)

        # (kn)
        x3a[a, :, c, i, j, n] -= H.aa.vovv[:, k, e, f] * t_amp # (1)
        x3a[e, :, c, i, j, n] += H.aa.vovv[:, k, a, f] * t_amp # (ae)
        x3a[f, :, c, i, j, n] += H.aa.vovv[:, k, e, a] * t_amp # (af)
        x3a[a, :, e, i, j, n] += H.aa.vovv[:, k, c, f] * t_amp # (ce)
        x3a[a, :, f, i, j, n] += H.aa.vovv[:, k, e, c] * t_amp # (cf)
        x3a[e, :, f, i, j, n] -= H.aa.vovv[:, k, a, c] * t_amp # (ae)(cf)


    # Loop over aaab determinants
    #for idet in range(len(C4_amplitudes["aaab"])):
    #
    #    # Get the particular aaab T4 amplitude
    #    t_amp = T4_amplitudes["aaab"][idet]


    # Loop over aabb determinants
    #for idet in range(len(C4_amplitudes["aabb"])):
    #
    #    # Get the particular aabb T4 amplitude
    #    t_amp = T4_amplitudes["aabb"][idet]


    # Loop over abbb determinants
    #for idet in range(len(C4_amplitudes["abbb"])):
    #
    #    # Get the particular abbb T4 amplitude
    #    t_amp = T4_amplitudes["abbb"][idet]


    # Loop over bbbb determinants
    #for idet in range(len(C4_amplitudes["bbbb"])):
    #
    #    # Get the particular bbbb T4 amplitude
    #    t_amp = T4_amplitudes["bbbb"][idet]


    x3a -= np.transpose(x3a, (0, 1, 2, 3, 5, 4)) # (jk)
    x3a -= np.transpose(x3a, (0, 1, 2, 4, 3, 5)) + np.transpose(x3a, (0, 1, 2, 5, 4, 3)) # (i/jk)
    x3a -= np.transpose(x3a, (0, 2, 1, 3, 4, 5)) # (bc)
    x3a -= np.transpose(x3a, (2, 1, 0, 3, 4, 5)) + np.transpose(x3a, (1, 0, 2, 3, 4, 5)) # (a/bc)

    return x3a, x3b, x3c, x3d

if __name__ == "__main__":

    #ccpy_root = "/home2/gururang/ccpy"
    ccpy_root = "/Users/harellab/Documents/ccpy"

    system, H = load_from_gamess(
            ccpy_root + "/examples/ext_corr/h2o-Re/h2o-Re.log",
            ccpy_root + "/examples/ext_corr/h2o-Re/onebody.inp",
            ccpy_root + "/examples/ext_corr/h2o-Re/twobody.inp",
            nfrozen=0,
    )
    system.print_info()

    civecs = ccpy_root + "/examples/ext_corr/h2o-Re/ndet_100000/civecs.dat"

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
    T_ext, T4_amplitudes = cluster_analyze_ci(C, C4_excitations, C4_amplitudes, system, 4)

    # Get the expected result for the contraction, computed using full T_ext
    print("   Exact H*T4 contraction")
    x3_aaa_exact, x3_aab_exact, x3_abb_exact, x3_bbb_exact = contract_vt4_exact(H, T_ext)

    # Get the on-the-fly contraction result
    print("   On-the-fly H*T4 contraction")
    t1 = time.time()
    x3_aaa, x3_aab, x3_abb, x3_bbb = contract_vt4_fly(H, C4_excitations, T4_amplitudes, system)
    print("   Completed in ", time.time() - t1, "seconds")

    print("")
    print("Error in x3a = ", np.linalg.norm(x3_aaa.flatten() - x3_aaa_exact.flatten()))
    print("Error in x3b = ", np.linalg.norm(x3_aab.flatten() - x3_aab_exact.flatten()))
    print("Error in x3c = ", np.linalg.norm(x3_abb.flatten() - x3_abb_exact.flatten()))
    print("Error in x3d = ", np.linalg.norm(x3_bbb.flatten() - x3_bbb_exact.flatten()))

    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for c in range(b + 1, system.nunoccupied_alpha):
                for i in range(system.noccupied_alpha):
                    for j in range(i + 1, system.noccupied_alpha):
                        for k in range(j + 1, system.noccupied_alpha):

                            error = x3_aaa[a, b, c, i, j, k] - x3_aaa_exact[a, b, c, i, j, k]

                            if abs(error) > 1.0e-012:
                                print("Expected", x3_aaa[a, b, c, i, j, k], "Got", x3_aaa_exact[a, b, c, i, j, k])


