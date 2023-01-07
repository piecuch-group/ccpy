import numpy as np

def contract_ht4_triples(C4_excitations, H, T4_amplitudes, system):

    x3a = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha))
    x3b = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta))
    x3c = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta, system.noccupied_beta))
    x3d = np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta, system.noccupied_beta))

    n4aaaa = C4_excitations["aaaa"].shape[0]
    n4aaab = C4_excitations["aaab"].shape[0]
    n4aabb = C4_excitations["aabb"].shape[0]
    n4abbb = C4_excitations["abbb"].shape[0]
    n4bbbb = C4_excitations["bbbb"].shape[0]

    # Loop over aaaa determinants
    for idet in range(n4aaaa):

        # Get the particular aaaa T4 amplitude
        t_amp = T4_amplitudes["aaaa"][idet]

        # x3a(abcijk) <- A(ijk)A(b/ac)[ A(ac/ef)A(n/ijk) h_aa(bnef) * t_aaaa(aecfijkn) ]
        a, e, c, f, i, j, k, n = [x - 1 for x in C4_excitations["aaaa"][idet]]

        x3a[a, :, c, i, j, k] += H.aa.vovv[:, n, e, f] * t_amp # (1)

    
    return x3a, x3b, x3c, x3d


def contract_vt4_doubles(C4_excitations, H, T4_amplitudes, system):

    x2a = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha))
    x2b = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta))
    x2c = np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta))

    n4aaaa = C4_excitations["aaaa"].shape[0]
    n4aaab = C4_excitations["aaab"].shape[0]
    n4aabb = C4_excitations["aabb"].shape[0]
    n4abbb = C4_excitations["abbb"].shape[0]
    n4bbbb = C4_excitations["bbbb"].shape[0]

    # Loop over aaaa determinants
    for idet in range(n4aaaa):

        # Get the particular aaaa T4 amplitude
        t_amp = T4_amplitudes["aaaa"][idet]

        # x2a(abij) <- A(ij/mn)A(ab/ef) v_aa(mnef) * t_aaaa(abefijmn)
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aaaa"][idet]]

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
    for idet in range(n4aaab):

        # Get the particular aaab T4 amplitude
        t_amp = T4_amplitudes["aaab"][idet]

        # x2a(abij) <- A(m/ij)A(e/ab) v_ab(mnef) * t_aaab(abefijmn)
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aaab"][idet]]

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
    for idet in range(n4aabb):

        # Get the particular aabb T4 amplitude
        t_amp = T4_amplitudes["aabb"][idet]

        # x2a(abij) <- v_bb(mnef) * t_aabb(abefijmn)
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["aabb"][idet]]

        x2a[a, b, i, j] += H.bb.oovv[m, n, e, f] * t_amp
        x2a[b, a, i, j] = -x2a[a, b, i, j]
        x2a[a, b, j, i] = -x2a[a, b, i, j]
        x2a[b, a, j, i] = x2a[a, b, i, j]

        # x2b(abij) <- A(ae)A(bf)A(im)(jn) v_ab(mnef) * t_aabb(aefbimnj)
        a, e, f, b, i, m, n, j = [x - 1 for x in C4_excitations["aabb"][idet]]

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

        x2c[a, b, i, j] += H.aa.oovv[m, n, e, f] * t_amp
        x2c[b, a, i, j] = -x2c[a, b, i, j]
        x2c[a, b, j, i] = -x2c[a, b, i, j]
        x2c[b, a, j, i] = x2c[a, b, i, j]

    # Loop over abbb determinants
    for idet in range(n4abbb):

        # Get the particular abbb T4 amplitude
        t_amp = T4_amplitudes["abbb"][idet]

        # x2b(abij) <- A(j/mn)A(b/ef) v_bb(mnef) * t_abbb(aefbimnj)
        a, e, f, b, i, m, n, j = [x - 1 for x in C4_excitations["abbb"][idet]]

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
    for idet in range(n4bbbb):

        # Get the particular bbbb T4 amplitude
        t_amp = T4_amplitudes["bbbb"][idet]

        # x2c(abij) <- A(ij/mn)A(ab/ef) v_bb(mnef) * t_bbbb(abefijmn)
        a, b, e, f, i, j, m, n = [x - 1 for x in C4_excitations["bbbb"][idet]]

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
