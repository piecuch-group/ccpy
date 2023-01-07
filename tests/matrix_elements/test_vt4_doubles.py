import numpy as np
import time

from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.utilities.external_correction import parse_ci_wavefunction, cluster_analyze_ci
#from ccpy.utilities.updates import clusteranalysis

def contract_vt4_exact(H, T_ext):

    # < ijab | [V_N, T4] | 0 >
    x2_aa_exact = np.einsum("mnef,abefijmn->abij", H.ab.oovv, T_ext.aaab, optimize=True)
    x2_aa_exact += 0.25 * np.einsum("mnef,abefijmn->abij", H.bb.oovv, T_ext.aabb, optimize=True)
    x2_aa_exact += 0.25 * np.einsum("mnef,abefijmn->abij", H.aa.oovv, T_ext.aaaa, optimize=True)

    # <ij~ab~ | [V_N, T4] | 0 >
    x2_ab_exact = 0.25 * np.einsum("mnef,aefbimnj->abij", H.aa.oovv, T_ext.aaab, optimize=True)
    x2_ab_exact += np.einsum("mnef,aefbimnj->abij", H.ab.oovv, T_ext.aabb, optimize=True)
    x2_ab_exact += 0.25 * np.einsum("mnef,aefbimnj->abij", H.bb.oovv, T_ext.abbb, optimize=True)

    # <i~j~a~b~ | [V_N, T4] | 0 >
    x2_bb_exact = 0.25 * np.einsum("mnef,efabmnij->abij", H.bb.oovv, T_ext.aabb, optimize=True)
    x2_bb_exact += np.einsum("mnef,efabmnij->abij", H.ab.oovv, T_ext.abbb, optimize=True)
    x2_bb_exact += 0.25 * np.einsum("mnef,abefijmn->abij", H.bb.oovv, T_ext.bbbb, optimize=True)

    return x2_aa_exact, x2_ab_exact, x2_bb_exact

def contract_vt4_opt(C4_excitations, C4_amplitudes, C, H, T_ext, T4_amplitudes, system):

    x2a = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha))
    x2b = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta))
    x2c = np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta))

    # Loop over aaaa determinants
    for idet in range(len(C4_amplitudes["aaaa"])):

        # Get the particular aaaa T4 amplitude
        t_amp = T4_amplitudes["aaaa"][idet]

        # x2a(abij) <- A(ij/mn)A(ab/ef) v_aa(mnef) * t_aaaa(abefijmn)
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aaaa"][idet]]
        #t_amp = T_ext.aaaa[a, b, e, f, i, j, m, n]
        #t_amp = clusteranalysis.clusteranalysis.get_t4aaaa_amp(C.a, C.aa, C.aaa, C4_amplitudes["aaaa"][idet],
        #                                                       a+1, b+1, e+1, f+1, i+1, j+1, m+1, n+1)

        x2a[a, b, i, j] += H.aa.oovv[m, n, e, f] * t_amp  #  (1)
        x2a[b, a, i, j] = -x2a[a, b, i, j]
        x2a[a, b, j, i] = -x2a[a, b, i, j]
        x2a[b, a, j, i] = x2a[a, b, i, j]

        x2a[a, b, m, j] -= H.aa.oovv[i, n, e, f] * t_amp  #  (im)
        x2a[b, a, m, j] = -x2a[a, b, m, j]
        x2a[a, b, j, m] = -x2a[a, b, m, j]
        x2a[b, a, j, m] = x2a[a, b, m, j]

        x2a[a, b, n, j] -= H.aa.oovv[m, i, e, f] * t_amp  #  (in)
        x2a[b, a, n, j] = -x2a[a, b, n, j]
        x2a[a, b, j, n] = -x2a[a, b, n, j]
        x2a[b, a, j, n] = x2a[a, b, n, j]

        x2a[a, b, i, m] -= H.aa.oovv[j, n, e, f] * t_amp  #  (jm)
        x2a[b, a, i, m] = -x2a[a, b, i, m]
        x2a[a, b, m, i] = -x2a[a, b, i, m]
        x2a[b, a, m, i] = x2a[a, b, i, m]

        x2a[a, b, i, n] -= H.aa.oovv[m, j, e, f] * t_amp  #  (jn)
        x2a[b, a, i, n] = -x2a[a, b, i, n]
        x2a[a, b, n, i] = -x2a[a, b, i, n]
        x2a[b, a, n, i] = x2a[a, b, i, n]

        x2a[a, b, m, n] += H.aa.oovv[i, j, e, f] * t_amp  #  (im)(jn)
        x2a[b, a, m, n] = -x2a[a, b, m, n]
        x2a[a, b, n, m] = -x2a[a, b, m, n]
        x2a[b, a, n, m] = x2a[a, b, m, n]

        x2a[e, b, i, j] -= H.aa.oovv[m, n, a, f] * t_amp  #  (ae)
        x2a[b, e, i, j] = -x2a[e, b, i, j]
        x2a[e, b, j, i] = -x2a[e, b, i, j]
        x2a[b, e, j, i] = x2a[e, b, i, j]

        x2a[e, b, m, j] += H.aa.oovv[i, n, a, f] * t_amp  #  (im)(ae)
        x2a[b, e, m, j] = -x2a[e, b, m, j]
        x2a[e, b, j, m] = -x2a[e, b, m, j]
        x2a[b, e, j, m] = x2a[e, b, m, j]

        x2a[e, b, n, j] += H.aa.oovv[m, i, a, f] * t_amp  #  (in)(ae)
        x2a[b, e, n, j] = -x2a[e, b, n, j]
        x2a[e, b, j, n] = -x2a[e, b, n, j]
        x2a[b, e, j, n] = x2a[e, b, n, j]

        x2a[e, b, i, m] += H.aa.oovv[j, n, a, f] * t_amp  #  (jm)(ae)
        x2a[b, e, i, m] = -x2a[e, b, i, m]
        x2a[e, b, m, i] = -x2a[e, b, i, m]
        x2a[b, e, m, i] = x2a[e, b, i, m]

        x2a[e, b, i, n] += H.aa.oovv[m, j, a, f] * t_amp  #  (jn)(ae)
        x2a[b, e, i, n] = -x2a[e, b, i, n]
        x2a[e, b, n, i] = -x2a[e, b, i, n]
        x2a[b, e, n, i] = x2a[e, b, i, n]

        x2a[e, b, m, n] -= H.aa.oovv[i, j, a, f] * t_amp  #  (im)(jn)(ae)
        x2a[b, e, m, n] = -x2a[e, b, m, n]
        x2a[e, b, n, m] = -x2a[e, b, m, n]
        x2a[b, e, n, m] = x2a[e, b, m, n]

        x2a[f, b, i, j] -= H.aa.oovv[m, n, e, a] * t_amp  #  (af)
        x2a[b, f, i, j] = -x2a[f, b, i, j]
        x2a[f, b, j, i] = -x2a[f, b, i, j]
        x2a[b, f, j, i] = x2a[f, b, i, j]

        x2a[f, b, m, j] += H.aa.oovv[i, n, e, a] * t_amp  #  (im)(af)
        x2a[b, f, m, j] = -x2a[f, b, m, j]
        x2a[f, b, j, m] = -x2a[f, b, m, j]
        x2a[b, f, j, m] = x2a[f, b, m, j]

        x2a[f, b, n, j] += H.aa.oovv[m, i, e, a] * t_amp  #  (in)(af)
        x2a[b, f, n, j] = -x2a[f, b, n, j]
        x2a[f, b, j, n] = -x2a[f, b, n, j]
        x2a[b, f, j, n] = x2a[f, b, n, j]

        x2a[f, b, i, m] += H.aa.oovv[j, n, e, a] * t_amp  #  (jm)(af)
        x2a[b, f, i, m] = -x2a[f, b, i, m]
        x2a[f, b, m, i] = -x2a[f, b, i, m]
        x2a[b, f, m, i] = x2a[f, b, i, m]

        x2a[f, b, i, n] += H.aa.oovv[m, j, e, a] * t_amp  #  (jn)(af)
        x2a[b, f, i, n] = -x2a[f, b, i, n]
        x2a[f, b, n, i] = -x2a[f, b, i, n]
        x2a[b, f, n, i] = x2a[f, b, i, n]

        x2a[f, b, m, n] -= H.aa.oovv[i, j, e, a] * t_amp  #  (im)(jn)(af)
        x2a[b, f, m, n] = -x2a[f, b, m, n]
        x2a[f, b, n, m] = -x2a[f, b, m, n]
        x2a[b, f, n, m] = x2a[f, b, m, n]

        x2a[a, e, i, j] -= H.aa.oovv[m, n, b, f] * t_amp  #  (be)
        x2a[e, a, i, j] = -x2a[a, e, i, j]
        x2a[a, e, j, i] = -x2a[a, e, i, j]
        x2a[e, a, j, i] = x2a[a, e, i, j]

        x2a[a, e, m, j] += H.aa.oovv[i, n, b, f] * t_amp  #  (im)(be)
        x2a[e, a, m, j] = -x2a[a, e, m, j]
        x2a[a, e, j, m] = -x2a[a, e, m, j]
        x2a[e, a, j, m] = x2a[a, e, m, j]

        x2a[a, e, n, j] += H.aa.oovv[m, i, b, f] * t_amp  #  (in)(be)
        x2a[e, a, n, j] = -x2a[a, e, n, j]
        x2a[a, e, j, n] = -x2a[a, e, n, j]
        x2a[e, a, j, n] = x2a[a, e, n, j]

        x2a[a, e, i, m] += H.aa.oovv[j, n, b, f] * t_amp  #  (jm)(be)
        x2a[e, a, i, m] = -x2a[a, e, i, m]
        x2a[a, e, m, i] = -x2a[a, e, i, m]
        x2a[e, a, m, i] = x2a[a, e, i, m]

        x2a[a, e, i, n] += H.aa.oovv[m, j, b, f] * t_amp  #  (jn)(be)
        x2a[e, a, i, n] = -x2a[a, e, i, n]
        x2a[a, e, n, i] = -x2a[a, e, i, n]
        x2a[e, a, n, i] = x2a[a, e, i, n]

        x2a[a, e, m, n] -= H.aa.oovv[i, j, b, f] * t_amp  #  (im)(jn)(be)
        x2a[e, a, m, n] = -x2a[a, e, m, n]
        x2a[a, e, n, m] = -x2a[a, e, m, n]
        x2a[e, a, n, m] = x2a[a, e, m, n]

        x2a[a, f, i, j] -= H.aa.oovv[m, n, e, b] * t_amp  #  (bf)
        x2a[f, a, i, j] = -x2a[a, f, i, j]
        x2a[a, f, j, i] = -x2a[a, f, i, j]
        x2a[f, a, j, i] = x2a[a, f, i, j]

        x2a[a, f, m, j] += H.aa.oovv[i, n, e, b] * t_amp  #  (im)(bf)
        x2a[f, a, m, j] = -x2a[a, f, m, j]
        x2a[a, f, j, m] = -x2a[a, f, m, j]
        x2a[f, a, j, m] = x2a[a, f, m, j]

        x2a[a, f, n, j] += H.aa.oovv[m, i, e, b] * t_amp  #  (in)(bf)
        x2a[f, a, n, j] = -x2a[a, f, n, j]
        x2a[a, f, j, n] = -x2a[a, f, n, j]
        x2a[f, a, j, n] = x2a[a, f, n, j]

        x2a[a, f, i, m] += H.aa.oovv[j, n, e, b] * t_amp  #  (jm)(bf)
        x2a[f, a, i, m] = -x2a[a, f, i, m]
        x2a[a, f, m, i] = -x2a[a, f, i, m]
        x2a[f, a, m, i] = x2a[a, f, i, m]

        x2a[a, f, i, n] += H.aa.oovv[m, j, e, b] * t_amp  #  (jn)(bf)
        x2a[f, a, i, n] = -x2a[a, f, i, n]
        x2a[a, f, n, i] = -x2a[a, f, i, n]
        x2a[f, a, n, i] = x2a[a, f, i, n]

        x2a[a, f, m, n] -= H.aa.oovv[i, j, e, b] * t_amp  #  (im)(jn)(bf)
        x2a[f, a, m, n] = -x2a[a, f, m, n]
        x2a[a, f, n, m] = -x2a[a, f, m, n]
        x2a[f, a, n, m] = x2a[a, f, m, n]

        x2a[e, f, i, j] += H.aa.oovv[m, n, a, b] * t_amp  #  (ae)(bf)
        x2a[f, e, i, j] = -x2a[e, f, i, j]
        x2a[e, f, j, i] = -x2a[e, f, i, j]
        x2a[f, e, j, i] = x2a[e, f, i, j]

        x2a[e, f, m, j] -= H.aa.oovv[i, n, a, b] * t_amp  #  (im)(ae)(bf)
        x2a[f, e, m, j] = -x2a[e, f, m, j]
        x2a[e, f, j, m] = -x2a[e, f, m, j]
        x2a[f, e, j, m] = x2a[e, f, m, j]

        x2a[e, f, n, j] -= H.aa.oovv[m, i, a, b] * t_amp  #  (in)(ae)(bf)
        x2a[f, e, n, j] = -x2a[e, f, n, j]
        x2a[e, f, j, n] = -x2a[e, f, n, j]
        x2a[f, e, j, n] = x2a[e, f, n, j]

        x2a[e, f, i, m] -= H.aa.oovv[j, n, a, b] * t_amp  #  (jm)(ae)(bf)
        x2a[f, e, i, m] = -x2a[e, f, i, m]
        x2a[e, f, m, i] = -x2a[e, f, i, m]
        x2a[f, e, m, i] = x2a[e, f, i, m]

        x2a[e, f, i, n] -= H.aa.oovv[m, j, a, b] * t_amp  #  (jn)(ae)(bf)
        x2a[f, e, i, n] = -x2a[e, f, i, n]
        x2a[e, f, n, i] = -x2a[e, f, i, n]
        x2a[f, e, n, i] = x2a[e, f, i, n]

        x2a[e, f, m, n] += H.aa.oovv[i, j, a, b] * t_amp  #  (im)(jn)(ae)(bf)
        x2a[f, e, m, n] = -x2a[e, f, m, n]
        x2a[e, f, n, m] = -x2a[e, f, m, n]
        x2a[f, e, n, m] = x2a[e, f, m, n]

    # Loop over aaab determinants
    for idet in range(len(C4_amplitudes["aaab"])):

        # Get the particular aaab T4 amplitude
        t_amp = T4_amplitudes["aaab"][idet]

        # x2a(abij) <- A(m/ij)A(e/ab) v_ab(mnef) * t_aaab(abefijmn)
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aaab"][idet]]
        #t_amp = T_ext.aaab[a, b, e, f, i, j, m, n]
        #t_amp = clusteranalysis.clusteranalysis.get_t4aaab_amp(C.a, C.b, C.aa, C.ab, C.aaa, C.aab, C4_amplitudes["aaab"][idet],
        #                                                       a+1, b+1, e+1, f+1, i+1, j+1, m+1, n+1)

        x2a[a, b, i, j] += H.ab.oovv[m, n, e, f] * t_amp  # (1)
        x2a[b, a, i, j] = -x2a[a, b, i, j]
        x2a[a, b, j, i] = -x2a[a, b, i, j]
        x2a[b, a, j, i] = x2a[a, b, i, j]

        x2a[e, b, i, j] -= H.ab.oovv[m, n, a, f] * t_amp  # (ae)
        x2a[e, b, j, i] = -x2a[e, b, i, j]
        x2a[b, e, i, j] = -x2a[e, b, i, j]
        x2a[b, e, j, i] = x2a[e, b, i, j]

        x2a[a, e, i, j] -= H.ab.oovv[m, n, b, f] * t_amp  # (be)
        x2a[a, e, j, i] = -x2a[a, e, i, j]
        x2a[e, a, i, j] = -x2a[a, e, i, j]
        x2a[e, a, j, i] = x2a[a, e, i, j]

        x2a[a, b, m, j] -= H.ab.oovv[i, n, e, f] * t_amp  # (mi)
        x2a[a, b, j, m] = -x2a[a, b, m, j]
        x2a[b, a, m, j] = -x2a[a, b, m, j]
        x2a[b, a, j, m] = x2a[a, b, m, j]

        x2a[e, b, m, j] += H.ab.oovv[i, n, a, f] * t_amp  # (mi)(ae)
        x2a[e, b, j, m] = -x2a[e, b, m, j]
        x2a[b, e, m, j] = -x2a[e, b, m, j]
        x2a[b, e, j, m] = x2a[e, b, m, j]

        x2a[a, e, m, j] += H.ab.oovv[i, n, b, f] * t_amp  # (mi)(be)
        x2a[a, e, j, m] = -x2a[a, e, m, j]
        x2a[e, a, m, j] = -x2a[a, e, m, j]
        x2a[e, a, j, m] = x2a[a, e, m, j]

        x2a[a, b, i, m] -= H.ab.oovv[j, n, e, f] * t_amp  # (mj)
        x2a[a, b, m, i] = -x2a[a, b, i, m]
        x2a[b, a, i, m] = -x2a[a, b, i, m]
        x2a[b, a, m, i] = x2a[a, b, i, m]

        x2a[e, b, i, m] += H.ab.oovv[j, n, a, f] * t_amp  # (ae)(mj)
        x2a[e, b, m, i] = -x2a[e, b, i, m]
        x2a[b, e, i, m] = -x2a[e, b, i, m]
        x2a[b, e, m, i] = x2a[e, b, i, m]

        x2a[a, e, i, m] += H.ab.oovv[j, n, b, f] * t_amp  # (be)(mj)
        x2a[a, e, m, i] = -x2a[a, e, i, m]
        x2a[e, a, i, m] = -x2a[a, e, i, m]
        x2a[e, a, m, i] = x2a[a, e, i, m]

        # x2b(abij) < - A(i/mn)A(a/ef) v_aa(mnef) * t_aaab(aefbimnj)
        a, e, f, b, i, m, n, j = [x - 1 for x in C4_excitations["aaab"][idet]]
        #t_amp = T_ext.aaab[a, e, f, b, i, m, n, j]
        #t_amp = clusteranalysis.clusteranalysis.get_t4aaab_amp(C.a, C.b, C.aa, C.ab, C.aaa, C.aab, C4_amplitudes["aaab"][idet],
        #                                                       a+1, e+1, f+1, b+1, i+1, m+1, n+1, j+1)

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

        # Get the particular aabb T4 amplitude
        t_amp = T4_amplitudes["aabb"][idet]

        # x2a(abij) <- v_bb(mnef) * t_aabb(abefijmn)
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aabb"][idet]]
        #t_amp = T_ext.aabb[a, b, e, f, i, j, m, n]
        #t_amp = clusteranalysis.clusteranalysis.get_t4aabb_amp(C.a, C.b, C.aa, C.ab, C.bb, C.aaa, C.aab, C.abb, C4_amplitudes["aabb"][idet],
        #                                                       a+1, b+1, e+1, f+1, i+1, j+1, m+1, n+1)

        x2a[a, b, i, j] += H.bb.oovv[m, n, e, f] * t_amp
        x2a[b, a, i, j] = -x2a[a, b, i, j]
        x2a[a, b, j, i] = -x2a[a, b, i, j]
        x2a[b, a, j, i] = x2a[a, b, i, j]

        # x2b(abij) <- A(ae)A(bf)A(im)(jn) v_ab(mnef) * t_aabb(aefbimnj)
        a, e, f, b, i, m, n, j = [x - 1 for x in C4_excitations["aabb"][idet]]
        #t_amp = T_ext.aabb[a, e, f, b, i, m, n, j]
        #t_amp = clusteranalysis.clusteranalysis.get_t4aabb_amp(C.a, C.b, C.aa, C.ab, C.bb, C.aaa, C.aab, C.abb, C4_amplitudes["aabb"][idet],
        #                                                       a+1, e+1, f+1, b+1, i+1, m+1, n+1, j+1)

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

        # x2c(abij) <- v_aa(mnef) * t_aabb(efabmnij)
        e, f, a, b, m, n, i, j = [x - 1 for x in C4_excitations["aabb"][idet]]
        #t_amp = T_ext.aabb[e, f, a, b, m, n, i, j]
        #t_amp = clusteranalysis.clusteranalysis.get_t4aabb_amp(C.a, C.b, C.aa, C.ab, C.bb, C.aaa, C.aab, C.abb, C4_amplitudes["aabb"][idet],
        #                                                       e+1, f+1, a+1, b+1, m+1, n+1, i+1, j+1)

        x2c[a, b, i, j] += H.aa.oovv[m, n, e, f] * t_amp
        x2c[b, a, i, j] = -x2c[a, b, i, j]
        x2c[a, b, j, i] = -x2c[a, b, i, j]
        x2c[b, a, j, i] = x2c[a, b, i, j]

    # Loop over abbb determinants
    for idet in range(len(C4_amplitudes["abbb"])):

        # Get the particular abbb T4 amplitude
        t_amp = T4_amplitudes["abbb"][idet]

        # x2b(abij) <- A(j/mn)A(b/ef) v_bb(mnef) * t_abbb(aefbimnj)
        a, e, f, b, i, m, n, j = [x - 1 for x in C4_excitations["abbb"][idet]]
        #t_amp = T_ext.abbb[a, e, f, b, i, m, n, j]
        #t_amp = clusteranalysis.clusteranalysis.get_t4abbb_amp(C.a, C.b, C.ab, C.bb, C.abb, C.bbb, C4_amplitudes["abbb"][idet],
        #                                                       a+1, e+1, f+1, b+1, i+1, m+1, n+1, j+1)

        x2b[a, b, i, j] += H.bb.oovv[m, n, e, f] * t_amp  #  (1)
        x2b[a, b, i, m] -= H.bb.oovv[j, n, e, f] * t_amp  #  (jm)
        x2b[a, b, i, n] -= H.bb.oovv[m, j, e, f] * t_amp  #  (jn)
        x2b[a, e, i, j] -= H.bb.oovv[m, n, b, f] * t_amp  #  (be)
        x2b[a, f, i, j] -= H.bb.oovv[m, n, e, b] * t_amp  #  (bf)
        x2b[a, e, i, m] += H.bb.oovv[j, n, b, f] * t_amp  #  (jm)(be)
        x2b[a, e, i, n] += H.bb.oovv[m, j, b, f] * t_amp  #  (jn)(be)
        x2b[a, f, i, m] += H.bb.oovv[j, n, e, b] * t_amp  #  (jm)(bf)
        x2b[a, f, i, n] += H.bb.oovv[m, j, e, b] * t_amp  #  (jn)(bf)

        # x2c(abij) <- A(n/ij)A(f/ab) v_ab(mnef) * t_abbb(efabmnij)
        e, f, a, b, m, n, i, j = [x - 1 for x in C4_excitations["abbb"][idet]]
        #t_amp = T_ext.abbb[e, f, a, b, m, n, i, j]
        #t_amp = clusteranalysis.clusteranalysis.get_t4abbb_amp(C.a, C.b, C.ab, C.bb, C.abb, C.bbb, C4_amplitudes["abbb"][idet],
        #                                                       e+1, f+1, a+1, b+1, m+1, n+1, i+1, j+1)

        x2c[a, b, i, j] += H.ab.oovv[m, n, e, f] * t_amp  #  (1)
        x2c[b, a, i, j] = -x2c[a, b, i, j]
        x2c[a, b, j, i] = -x2c[a, b, i, j]
        x2c[b, a, j, i] = x2c[a, b, i, j]

        x2c[a, b, n, j] -= H.ab.oovv[m, i, e, f] * t_amp  #  (in)
        x2c[b, a, n, j] = -x2c[a, b, n, j]
        x2c[a, b, j, n] = -x2c[a, b, n, j]
        x2c[b, a, j, n] = x2c[a, b, n, j]

        x2c[a, b, i, n] -= H.ab.oovv[m, j, e, f] * t_amp  #  (jn)
        x2c[b, a, i, n] = -x2c[a, b, i, n]
        x2c[a, b, n, i] = -x2c[a, b, i, n]
        x2c[b, a, n, i] = x2c[a, b, i, n]

        x2c[f, b, i, j] -= H.ab.oovv[m, n, e, a] * t_amp  #  (af)
        x2c[b, f, i, j] = -x2c[f, b, i, j]
        x2c[f, b, j, i] = -x2c[f, b, i, j]
        x2c[b, f, j, i] = x2c[f, b, i, j]

        x2c[a, f, i, j] -= H.ab.oovv[m, n, e, b] * t_amp  #  (bf)
        x2c[f, a, i, j] = -x2c[a, f, i, j]
        x2c[a, f, j, i] = -x2c[a, f, i, j]
        x2c[f, a, j, i] = x2c[a, f, i, j]

        x2c[f, b, n, j] += H.ab.oovv[m, i, e, a] * t_amp  #  (in)(af)
        x2c[b, f, n, j] = -x2c[f, b, n, j]
        x2c[f, b, j, n] = -x2c[f, b, n, j]
        x2c[b, f, j, n] = x2c[f, b, n, j]

        x2c[a, f, n, j] += H.ab.oovv[m, i, e, b] * t_amp  #  (in)(bf)
        x2c[f, a, n, j] = -x2c[a, f, n, j]
        x2c[a, f, j, n] = -x2c[a, f, n, j]
        x2c[f, a, j, n] = x2c[a, f, n, j]

        x2c[f, b, i, n] += H.ab.oovv[m, j, e, a] * t_amp  #  (jn)(af)
        x2c[b, f, i, n] = -x2c[f, b, i, n]
        x2c[f, b, n, i] = -x2c[f, b, i, n]
        x2c[b, f, n, i] = x2c[f, b, i, n]

        x2c[a, f, i, n] += H.ab.oovv[m, j, e, b] * t_amp  #  (jn)(bf)
        x2c[f, a, i, n] = -x2c[a, f, i, n]
        x2c[a, f, n, i] = -x2c[a, f, i, n]
        x2c[f, a, n, i] = x2c[a, f, i, n]

    # Loop over bbbb determinants
    for idet in range(len(C4_amplitudes["bbbb"])):

        # Get the particular bbbb T4 amplitude
        t_amp = T4_amplitudes["bbbb"][idet]

        # x2c(abij) <- A(ij/mn)A(ab/ef) v_bb(mnef) * t_bbbb(abefijmn)
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["bbbb"][idet]]
        #t_amp = T_ext.bbbb[a, b, e, f, i, j, m, n]
        #t_amp = clusteranalysis.clusteranalysis.get_t4bbbb_amp(C.b, C.bb, C.bbb, C4_amplitudes["bbbb"][idet],
        #                                                       a+1, b+1, e+1, f+1, i+1, j+1, m+1, n+1)

        x2c[a, b, i, j] += H.bb.oovv[m, n, e, f] * t_amp  #  (1)
        x2c[b, a, i, j] = -x2c[a, b, i, j]
        x2c[a, b, j, i] = -x2c[a, b, i, j]
        x2c[b, a, j, i] = x2c[a, b, i, j]

        x2c[a, b, m, j] -= H.bb.oovv[i, n, e, f] * t_amp  #  (im)
        x2c[b, a, m, j] = -x2c[a, b, m, j]
        x2c[a, b, j, m] = -x2c[a, b, m, j]
        x2c[b, a, j, m] = x2c[a, b, m, j]

        x2c[a, b, n, j] -= H.bb.oovv[m, i, e, f] * t_amp  #  (in)
        x2c[b, a, n, j] = -x2c[a, b, n, j]
        x2c[a, b, j, n] = -x2c[a, b, n, j]
        x2c[b, a, j, n] = x2c[a, b, n, j]

        x2c[a, b, i, m] -= H.bb.oovv[j, n, e, f] * t_amp  #  (jm)
        x2c[b, a, i, m] = -x2c[a, b, i, m]
        x2c[a, b, m, i] = -x2c[a, b, i, m]
        x2c[b, a, m, i] = x2c[a, b, i, m]

        x2c[a, b, i, n] -= H.bb.oovv[m, j, e, f] * t_amp  #  (jn)
        x2c[b, a, i, n] = -x2c[a, b, i, n]
        x2c[a, b, n, i] = -x2c[a, b, i, n]
        x2c[b, a, n, i] = x2c[a, b, i, n]

        x2c[a, b, m, n] += H.bb.oovv[i, j, e, f] * t_amp  #  (im)(jn)
        x2c[b, a, m, n] = -x2c[a, b, m, n]
        x2c[a, b, n, m] = -x2c[a, b, m, n]
        x2c[b, a, n, m] = x2c[a, b, m, n]

        x2c[e, b, i, j] -= H.bb.oovv[m, n, a, f] * t_amp  #  (ae)
        x2c[b, e, i, j] = -x2c[e, b, i, j]
        x2c[e, b, j, i] = -x2c[e, b, i, j]
        x2c[b, e, j, i] = x2c[e, b, i, j]

        x2c[e, b, m, j] += H.bb.oovv[i, n, a, f] * t_amp  #  (im)(ae)
        x2c[b, e, m, j] = -x2c[e, b, m, j]
        x2c[e, b, j, m] = -x2c[e, b, m, j]
        x2c[b, e, j, m] = x2c[e, b, m, j]

        x2c[e, b, n, j] += H.bb.oovv[m, i, a, f] * t_amp  #  (in)(ae)
        x2c[b, e, n, j] = -x2c[e, b, n, j]
        x2c[e, b, j, n] = -x2c[e, b, n, j]
        x2c[b, e, j, n] = x2c[e, b, n, j]

        x2c[e, b, i, m] += H.bb.oovv[j, n, a, f] * t_amp  #  (jm)(ae)
        x2c[b, e, i, m] = -x2c[e, b, i, m]
        x2c[e, b, m, i] = -x2c[e, b, i, m]
        x2c[b, e, m, i] = x2c[e, b, i, m]

        x2c[e, b, i, n] += H.bb.oovv[m, j, a, f] * t_amp  #  (jn)(ae)
        x2c[b, e, i, n] = -x2c[e, b, i, n]
        x2c[e, b, n, i] = -x2c[e, b, i, n]
        x2c[b, e, n, i] = x2c[e, b, i, n]

        x2c[e, b, m, n] -= H.bb.oovv[i, j, a, f] * t_amp  #  (im)(jn)(ae)
        x2c[b, e, m, n] = -x2c[e, b, m, n]
        x2c[e, b, n, m] = -x2c[e, b, m, n]
        x2c[b, e, n, m] = x2c[e, b, m, n]

        x2c[f, b, i, j] -= H.bb.oovv[m, n, e, a] * t_amp  #  (af)
        x2c[b, f, i, j] = -x2c[f, b, i, j]
        x2c[f, b, j, i] = -x2c[f, b, i, j]
        x2c[b, f, j, i] = x2c[f, b, i, j]

        x2c[f, b, m, j] += H.bb.oovv[i, n, e, a] * t_amp  #  (im)(af)
        x2c[b, f, m, j] = -x2c[f, b, m, j]
        x2c[f, b, j, m] = -x2c[f, b, m, j]
        x2c[b, f, j, m] = x2c[f, b, m, j]

        x2c[f, b, n, j] += H.bb.oovv[m, i, e, a] * t_amp  #  (in)(af)
        x2c[b, f, n, j] = -x2c[f, b, n, j]
        x2c[f, b, j, n] = -x2c[f, b, n, j]
        x2c[b, f, j, n] = x2c[f, b, n, j]

        x2c[f, b, i, m] += H.bb.oovv[j, n, e, a] * t_amp  #  (jm)(af)
        x2c[b, f, i, m] = -x2c[f, b, i, m]
        x2c[f, b, m, i] = -x2c[f, b, i, m]
        x2c[b, f, m, i] = x2c[f, b, i, m]

        x2c[f, b, i, n] += H.bb.oovv[m, j, e, a] * t_amp  #  (jn)(af)
        x2c[b, f, i, n] = -x2c[f, b, i, n]
        x2c[f, b, n, i] = -x2c[f, b, i, n]
        x2c[b, f, n, i] = x2c[f, b, i, n]

        x2c[f, b, m, n] -= H.bb.oovv[i, j, e, a] * t_amp  #  (im)(jn)(af)
        x2c[b, f, m, n] = -x2c[f, b, m, n]
        x2c[f, b, n, m] = -x2c[f, b, m, n]
        x2c[b, f, n, m] = x2c[f, b, m, n]

        x2c[a, e, i, j] -= H.bb.oovv[m, n, b, f] * t_amp  #  (be)
        x2c[e, a, i, j] = -x2c[a, e, i, j]
        x2c[a, e, j, i] = -x2c[a, e, i, j]
        x2c[e, a, j, i] = x2c[a, e, i, j]

        x2c[a, e, m, j] += H.bb.oovv[i, n, b, f] * t_amp  #  (im)(be)
        x2c[e, a, m, j] = -x2c[a, e, m, j]
        x2c[a, e, j, m] = -x2c[a, e, m, j]
        x2c[e, a, j, m] = x2c[a, e, m, j]

        x2c[a, e, n, j] += H.bb.oovv[m, i, b, f] * t_amp  #  (in)(be)
        x2c[e, a, n, j] = -x2c[a, e, n, j]
        x2c[a, e, j, n] = -x2c[a, e, n, j]
        x2c[e, a, j, n] = x2c[a, e, n, j]

        x2c[a, e, i, m] += H.bb.oovv[j, n, b, f] * t_amp  #  (jm)(be)
        x2c[e, a, i, m] = -x2c[a, e, i, m]
        x2c[a, e, m, i] = -x2c[a, e, i, m]
        x2c[e, a, m, i] = x2c[a, e, i, m]

        x2c[a, e, i, n] += H.bb.oovv[m, j, b, f] * t_amp  #  (jn)(be)
        x2c[e, a, i, n] = -x2c[a, e, i, n]
        x2c[a, e, n, i] = -x2c[a, e, i, n]
        x2c[e, a, n, i] = x2c[a, e, i, n]

        x2c[a, e, m, n] -= H.bb.oovv[i, j, b, f] * t_amp  #  (im)(jn)(be)
        x2c[e, a, m, n] = -x2c[a, e, m, n]
        x2c[a, e, n, m] = -x2c[a, e, m, n]
        x2c[e, a, n, m] = x2c[a, e, m, n]

        x2c[a, f, i, j] -= H.bb.oovv[m, n, e, b] * t_amp  #  (bf)
        x2c[f, a, i, j] = -x2c[a, f, i, j]
        x2c[a, f, j, i] = -x2c[a, f, i, j]
        x2c[f, a, j, i] = x2c[a, f, i, j]

        x2c[a, f, m, j] += H.bb.oovv[i, n, e, b] * t_amp  #  (im)(bf)
        x2c[f, a, m, j] = -x2c[a, f, m, j]
        x2c[a, f, j, m] = -x2c[a, f, m, j]
        x2c[f, a, j, m] = x2c[a, f, m, j]

        x2c[a, f, n, j] += H.bb.oovv[m, i, e, b] * t_amp  #  (in)(bf)
        x2c[f, a, n, j] = -x2c[a, f, n, j]
        x2c[a, f, j, n] = -x2c[a, f, n, j]
        x2c[f, a, j, n] = x2c[a, f, n, j]

        x2c[a, f, i, m] += H.bb.oovv[j, n, e, b] * t_amp  #  (jm)(bf)
        x2c[f, a, i, m] = -x2c[a, f, i, m]
        x2c[a, f, m, i] = -x2c[a, f, i, m]
        x2c[f, a, m, i] = x2c[a, f, i, m]

        x2c[a, f, i, n] += H.bb.oovv[m, j, e, b] * t_amp  #  (jn)(bf)
        x2c[f, a, i, n] = -x2c[a, f, i, n]
        x2c[a, f, n, i] = -x2c[a, f, i, n]
        x2c[f, a, n, i] = x2c[a, f, i, n]

        x2c[a, f, m, n] -= H.bb.oovv[i, j, e, b] * t_amp  #  (im)(jn)(bf)
        x2c[f, a, m, n] = -x2c[a, f, m, n]
        x2c[a, f, n, m] = -x2c[a, f, m, n]
        x2c[f, a, n, m] = x2c[a, f, m, n]

        x2c[e, f, i, j] += H.bb.oovv[m, n, a, b] * t_amp  #  (ae)(bf)
        x2c[f, e, i, j] = -x2c[e, f, i, j]
        x2c[e, f, j, i] = -x2c[e, f, i, j]
        x2c[f, e, j, i] = x2c[e, f, i, j]

        x2c[e, f, m, j] -= H.bb.oovv[i, n, a, b] * t_amp  #  (im)(ae)(bf)
        x2c[f, e, m, j] = -x2c[e, f, m, j]
        x2c[e, f, j, m] = -x2c[e, f, m, j]
        x2c[f, e, j, m] = x2c[e, f, m, j]

        x2c[e, f, n, j] -= H.bb.oovv[m, i, a, b] * t_amp  #  (in)(ae)(bf)
        x2c[f, e, n, j] = -x2c[e, f, n, j]
        x2c[e, f, j, n] = -x2c[e, f, n, j]
        x2c[f, e, j, n] = x2c[e, f, n, j]

        x2c[e, f, i, m] -= H.bb.oovv[j, n, a, b] * t_amp  #  (jm)(ae)(bf)
        x2c[f, e, i, m] = -x2c[e, f, i, m]
        x2c[e, f, m, i] = -x2c[e, f, i, m]
        x2c[f, e, m, i] = x2c[e, f, i, m]

        x2c[e, f, i, n] -= H.bb.oovv[m, j, a, b] * t_amp  #  (jn)(ae)(bf)
        x2c[f, e, i, n] = -x2c[e, f, i, n]
        x2c[e, f, n, i] = -x2c[e, f, i, n]
        x2c[f, e, n, i] = x2c[e, f, i, n]

        x2c[e, f, m, n] += H.bb.oovv[i, j, a, b] * t_amp  #  (im)(jn)(ae)(bf)
        x2c[f, e, m, n] = -x2c[e, f, m, n]
        x2c[e, f, n, m] = -x2c[e, f, m, n]
        x2c[f, e, n, m] = x2c[e, f, m, n]

    return x2a, x2b, x2c

if __name__ == "__main__":

    ccpy_root = "/Users/karthik/Documents/Python/ccpy"

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
    print("   Exact V*T4 contraction")
    x2_aa_exact, x2_ab_exact, x2_bb_exact = contract_vt4_exact(H, T_ext)

    # Get the on-the-fly contraction result
    print("   On-the-fly V*T4 contraction")
    t1 = time.time()
    x2_aa, x2_ab, x2_bb = contract_vt4_opt(C4_excitations, C4_amplitudes, C, H, T_ext, T4_amplitudes, system)
    print("   Completed in ", time.time() - t1, "seconds")

    print("")
    print("Error in x2a = ", np.linalg.norm(x2_aa.flatten() - x2_aa_exact.flatten()))
    print("Error in x2b = ", np.linalg.norm(x2_ab.flatten() - x2_ab_exact.flatten()))
    print("Error in x2c = ", np.linalg.norm(x2_bb.flatten() - x2_bb_exact.flatten()))