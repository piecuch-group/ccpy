import numpy as np

from ccpy.utilities.determinants import get_excits_from, get_excits_to, get_spincase, get_excit_rank, calculate_permutation_parity, spatial_orb_idx
from ccpy.models.operators import ClusterOperator
from ccpy.models.integrals import Integral
from ccpy.utilities.updates import clusteranalysis
from ccpy.drivers.cc_energy import get_cc_energy, get_ci_energy
import time

def cluster_analysis(wavefunction_file, hamiltonian, system):

    print("   Performing cluster analysis")
    print("   ------------------------------")
    t1 = time.time()

    # Create the VT4 storage object
    VT_ext = Integral.from_empty(system, 2, data_type=hamiltonian.a.oo.dtype, use_none=True)

    # Parse the CI wave function to get C1 - C3 and list of C4
    print("   Reading the CI vector file at", wavefunction_file)
    C, C4_excitations, C4_amplitudes, excitation_count = parse_ci_wavefunction(wavefunction_file, system)

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
    T_ext, T4_amplitudes = cluster_analyze_ci(C, C4_excitations, C4_amplitudes, system)

    print("")
    print("   External CI wave function energy = ", get_ci_energy(C, hamiltonian))
    print("   CC correlation energy = ", get_cc_energy(T_ext, hamiltonian))

    print("   Performing < ia | [V_N, T3(ext)] | 0 > contraction")
    VT_ext.a.vo = 0.25 * np.einsum("mnef,aefimn->ai", hamiltonian.aa.oovv, T_ext.aaa, optimize=True)
    VT_ext.a.vo += np.einsum("mnef,aefimn->ai", hamiltonian.ab.oovv, T_ext.aab, optimize=True)
    VT_ext.a.vo += 0.25 * np.einsum("mnef,aefimn->ai", hamiltonian.bb.oovv, T_ext.abb, optimize=True)

    VT_ext.b.vo = 0.25 * np.einsum("mnef,aefimn->ai", hamiltonian.bb.oovv, T_ext.bbb, optimize=True)
    VT_ext.b.vo += 0.25 * np.einsum("mnef,efamni->ai", hamiltonian.aa.oovv, T_ext.aab, optimize=True)
    VT_ext.b.vo += np.einsum("mnef,efamni->ai", hamiltonian.ab.oovv, T_ext.abb, optimize=True)

    print("   Performing < ijab | [V_N, T4(ext)] | 0 > contraction")
    VT_ext.aa.vvoo, VT_ext.ab.vvoo, VT_ext.bb.vvoo = contract_vt4_opt(C4_excitations, C4_amplitudes, C, hamiltonian, T4_amplitudes, system)
    VT_ext.aa.vvoo *= 0.25
    VT_ext.bb.vvoo *= 0.25

    elapsed_time = time.time() - t1
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"({minutes:.1f}m {seconds:.1f}s)"
    print("   --------------------")
    print("   Completed in", time_str)
    print("")

    return T_ext, VT_ext

def cluster_analyze_ci(C, C4_excitations, C4_amplitudes, system, order=3):

    T = ClusterOperator(system, order)

    T.a = C.a.copy()
    T.b = C.b.copy()

    T.aa, T.ab, T.bb = clusteranalysis.clusteranalysis.cluster_analysis_t2(C.a, C.b,
                                                                           C.aa, C.ab, C.bb)

    T.aaa, T.aab, T.abb, T.bbb = clusteranalysis.clusteranalysis.cluster_analysis_t3(C.a, C.b,
                                                                                     C.aa, C.ab, C.bb,
                                                                                     C.aaa, C.aab, C.abb, C.bbb)

    t4_aaaa_amps, t4_aaab_amps, t4_aabb_amps, t4_abbb_amps, t4_bbbb_amps = clusteranalysis.clusteranalysis.cluster_analysis_t4(C.a, C.b,
                                                                                                     C.aa, C.ab, C.bb,
                                                                                                     C.aaa, C.aab, C.abb, C.bbb,
                                                                                                     C4_excitations['aaaa'],
                                                                                                     C4_excitations['aaab'],
                                                                                                     C4_excitations['aabb'],
                                                                                                     C4_excitations['abbb'],
                                                                                                     C4_excitations['bbbb'],
                                                                                                     C4_amplitudes['aaaa'],
                                                                                                     C4_amplitudes['aaab'],
                                                                                                     C4_amplitudes['aabb'],
                                                                                                     C4_amplitudes['abbb'],
                                                                                                     C4_amplitudes['bbbb'])

    T4_amplitudes = {"aaaa" : t4_aaaa_amps, "aaab" : t4_aaab_amps, "aabb" : t4_aabb_amps, "abbb" : t4_abbb_amps, "bbbb" : t4_bbbb_amps}

    if T.order > 3:
        T.aaaa, T.aaab, T.aabb, T.abbb, T.bbbb = clusteranalysis.clusteranalysis.cluster_analysis_t4_full(C.a, C.b,
                                                                                                     C.aa, C.ab, C.bb,
                                                                                                     C.aaa, C.aab, C.abb, C.bbb,
                                                                                                     C4_excitations['aaaa'],
                                                                                                     C4_excitations['aaab'],
                                                                                                     C4_excitations['aabb'],
                                                                                                     C4_excitations['abbb'],
                                                                                                     C4_excitations['bbbb'],
                                                                                                     C4_amplitudes['aaaa'],
                                                                                                     C4_amplitudes['aaab'],
                                                                                                     C4_amplitudes['aabb'],
                                                                                                     C4_amplitudes['abbb'],
                                                                                                     C4_amplitudes['bbbb'])

    return T, T4_amplitudes

def parse_ci_wavefunction(ci_file, system):

    # container to count excitations in the wave function
    excitation_count = {'a' : 0, 'b' : 0,
                        'aa' : 0, 'ab' : 0, 'bb' : 0,
                        'aaa' : 0, 'aab' : 0, 'abb' : 0, 'bbb' : 0,
                        'aaaa' : 0, 'aaab' : 0, 'aabb' : 0, 'abbb' : 0, 'bbbb' : 0}

    # C is stored in full through triples; quadruples stored as a list of what's there
    C = ClusterOperator(system, 3)
    C4_excits = {'aaaa' : [], 'aaab' : [], 'aabb' : [], 'abbb' : [], 'bbbb' : []}
    C4_amps   = {'aaaa' : [], 'aaab' : [], 'aabb' : [], 'abbb' : [], 'bbbb' : []}

    HF = sorted(
        [2 * i - 1 for i in range(1, system.noccupied_alpha + 1)]
        + [2 * i for i in range(1, system.noccupied_beta + 1)]
    )

    occupied_lower_bound = 1
    occupied_upper_bound = system.noccupied_alpha
    unoccupied_lower_bound = 1
    unoccupied_upper_bound = system.nunoccupied_beta

    orb_table = {'a': system.noccupied_alpha, 'b': system.noccupied_beta}

    with open(ci_file) as f:

        for line in f.readlines():

            det = list(map(int, line.split()[2:]))
            coeff = float(line.split()[1])

            # get and save CI coefficient of HF to write everything in intermediate normalization
            if det == HF:
                c_hf = coeff
                continue

            excit_rank = get_excit_rank(det, HF)

            if excit_rank > 4:
                continue

            spinorb_occ = get_excits_from(det, HF)
            spinorb_unocc = get_excits_to(det, HF)
            spincase = get_spincase(spinorb_occ, spinorb_unocc)

            unocc_shift = [orb_table[x] for x in spincase]

            # if determinants were printed right, these should be sorted
            spinorb_occ_alpha = [i for i in spinorb_occ if i % 2 == 1]
            spinorb_occ_alpha.sort()
            spinorb_occ_beta = [i for i in spinorb_occ if i % 2 == 0]
            spinorb_occ_beta.sort()
            spinorb_unocc_alpha = [i for i in spinorb_unocc if i % 2 == 1]
            spinorb_unocc_alpha.sort()
            spinorb_unocc_beta = [i for i in spinorb_unocc if i % 2 == 0]
            spinorb_unocc_beta.sort()

            # WARNING: You will need to compute phase factor associated with QMC determinant ordering (abab...)
            # alpha -> odd, beta -> even; e.g., 1, 2, 3, 4,...
            # and Slater determinant excitation N[a+b+c+d+lkji] (holes are reversed!)
            excited_det_spinorb = spinorb_unocc_alpha + spinorb_unocc_beta + list(reversed(spinorb_occ_alpha)) + list(reversed(spinorb_occ_beta))
            i_perm = np.argsort(excited_det_spinorb)

            perm_parity = calculate_permutation_parity(i_perm)


            idx_occ = [spatial_orb_idx(x) for x in spinorb_occ_alpha] + [spatial_orb_idx(x) for x in spinorb_occ_beta]
            idx_unocc = [spatial_orb_idx(x) for x in spinorb_unocc_alpha] + [spatial_orb_idx(x) for x in spinorb_unocc_beta]

            idx_unocc = [x - shift for x, shift in zip(idx_unocc, unocc_shift)]


            if any([i > occupied_upper_bound for i in idx_occ]) or any([i < occupied_lower_bound for i in idx_occ]):
                print("Occupied orbitals out of range!")
                print(idx_occ)
                break
            if any([i > unoccupied_upper_bound for i in idx_unocc]) or any([i < unoccupied_lower_bound for i in idx_unocc]):
                print("Unoccupied orbitals out of range!")
                print(idx_unocc)
                break

            excitation = idx_unocc + idx_occ  # determinant expressed as excitation out of HF
            coefficient = coeff / c_hf        # intermediate normalization
            coefficient *= perm_parity        # permutation phase

            excitation_count[spincase] += 1   # increment excitation count for this spincase

            if excit_rank == 4:
                C4_excits[spincase].append(excitation)
                C4_amps[spincase].append(coefficient)
            else:
                C = insert_ci_amplitude(C, [x - 1 for x in excitation], coefficient, spincase)

        for key in C4_excits.keys():
            C4_excits[key] = np.asarray(C4_excits[key])
            C4_amps[key] = np.asarray(C4_amps[key])
            # fix in case C4 for a certain spincase is empty
            if len(C4_excits[key].shape) < 2:
                C4_excits[key] = np.zeros((0, 8))
                C4_amps[key] = np.zeros(shape=(1,))

    # Put in the sign fix... not sure why this is, but it has to do with the ordering of excited_det_spinorb
    C.b *= -1.0
    C.aab *= -1.0
    C.bbb *= -1.0

    return C, C4_excits, C4_amps, excitation_count


def insert_ci_amplitude(C, excitation, coefficient, spincase):

    # Technically, based on how clusteranalysis.f90 is carried out, C4 does not
    # need to be antisymmetrized. You just need the coefficient of the unique
    # determinant.

    if spincase == 'a':
        a, i = excitation
        C.a[a, i] = coefficient

    elif spincase == 'b':
        a, i = excitation
        C.b[a, i] = coefficient 

    elif spincase == 'aa':
        a, b, i, j = excitation
        C.aa[a, b, i, j] = coefficient
        C.aa[b, a, i, j] = -1.0 * coefficient
        C.aa[a, b, j, i] = -1.0 * coefficient
        C.aa[b, a, j, i] = coefficient

    elif spincase == 'ab':
        a, b, i, j = excitation
        C.ab[a, b, i, j] = coefficient

    elif spincase == 'bb':
        a, b, i, j = excitation
        C.bb[a, b, i, j] = coefficient
        C.bb[b, a, i, j] = -1.0 * coefficient
        C.bb[a, b, j, i] = -1.0 * coefficient
        C.bb[b, a, j, i] = coefficient

    elif spincase == 'aaa':
        a, b, c, i, j, k = excitation
        C.aaa[a, b, c, i, j, k] = coefficient
        C.aaa[a, c, b, i, j, k] = -1.0 * coefficient
        C.aaa[b, a, c, i, j, k] = -1.0 * coefficient
        C.aaa[b, c, a, i, j, k] = coefficient
        C.aaa[c, a, b, i, j, k] = coefficient
        C.aaa[c, b, a, i, j, k] = -1.0 * coefficient
        C.aaa[a, b, c, i, k, j] = -1.0 * coefficient
        C.aaa[a, c, b, i, k, j] = coefficient
        C.aaa[b, a, c, i, k, j] = coefficient
        C.aaa[b, c, a, i, k, j] = -1.0 * coefficient
        C.aaa[c, a, b, i, k, j] = -1.0 * coefficient
        C.aaa[c, b, a, i, k, j] = coefficient
        C.aaa[a, b, c, j, k, i] = coefficient
        C.aaa[a, c, b, j, k, i] = -1.0 * coefficient
        C.aaa[b, a, c, j, k, i] = -1.0 * coefficient
        C.aaa[b, c, a, j, k, i] = coefficient
        C.aaa[c, a, b, j, k, i] = coefficient
        C.aaa[c, b, a, j, k, i] = -1.0 * coefficient
        C.aaa[a, b, c, j, i, k] = -1.0 * coefficient
        C.aaa[a, c, b, j, i, k] = coefficient
        C.aaa[b, a, c, j, i, k] = coefficient
        C.aaa[b, c, a, j, i, k] = -1.0 * coefficient
        C.aaa[c, a, b, j, i, k] = -1.0 * coefficient
        C.aaa[c, b, a, j, i, k] = coefficient
        C.aaa[a, b, c, k, i, j] = coefficient
        C.aaa[a, c, b, k, i, j] = -1.0 * coefficient
        C.aaa[b, a, c, k, i, j] = -1.0 * coefficient
        C.aaa[b, c, a, k, i, j] = coefficient
        C.aaa[c, a, b, k, i, j] = coefficient
        C.aaa[c, b, a, k, i, j] = -1.0 * coefficient
        C.aaa[a, b, c, k, j, i] = -1.0 * coefficient
        C.aaa[a, c, b, k, j, i] = coefficient
        C.aaa[b, a, c, k, j, i] = coefficient
        C.aaa[b, c, a, k, j, i] = -1.0 * coefficient
        C.aaa[c, a, b, k, j, i] = -1.0 * coefficient
        C.aaa[c, b, a, k, j, i] = coefficient

    elif spincase == 'aab':
        a, b, c, i, j, k = excitation
        C.aab[a, b, c, i, j, k] = coefficient
        C.aab[b, a, c, i, j, k] = -1.0 * coefficient
        C.aab[a, b, c, j, i, k] = -1.0 * coefficient
        C.aab[b, a, c, j, i, k] = coefficient

    elif spincase == 'abb':
        a, b, c, i, j, k = excitation
        C.abb[a, b, c, i, j, k] = coefficient
        C.abb[a, c, b, i, j, k] = -1.0 * coefficient
        C.abb[a, b, c, i, k, j] = -1.0 * coefficient
        C.abb[a, c, b, i, k, j] = coefficient

    elif spincase == 'bbb':
        a, b, c, i, j, k = excitation
        C.bbb[a, b, c, i, j, k] = coefficient
        C.bbb[a, c, b, i, j, k] = -1.0 * coefficient
        C.bbb[b, a, c, i, j, k] = -1.0 * coefficient
        C.bbb[b, c, a, i, j, k] = coefficient
        C.bbb[c, a, b, i, j, k] = coefficient
        C.bbb[c, b, a, i, j, k] = -1.0 * coefficient
        C.bbb[a, b, c, i, k, j] = -1.0 * coefficient
        C.bbb[a, c, b, i, k, j] = coefficient
        C.bbb[b, a, c, i, k, j] = coefficient
        C.bbb[b, c, a, i, k, j] = -1.0 * coefficient
        C.bbb[c, a, b, i, k, j] = -1.0 * coefficient
        C.bbb[c, b, a, i, k, j] = coefficient
        C.bbb[a, b, c, j, k, i] = coefficient
        C.bbb[a, c, b, j, k, i] = -1.0 * coefficient
        C.bbb[b, a, c, j, k, i] = -1.0 * coefficient
        C.bbb[b, c, a, j, k, i] = coefficient
        C.bbb[c, a, b, j, k, i] = coefficient
        C.bbb[c, b, a, j, k, i] = -1.0 * coefficient
        C.bbb[a, b, c, j, i, k] = -1.0 * coefficient
        C.bbb[a, c, b, j, i, k] = coefficient
        C.bbb[b, a, c, j, i, k] = coefficient
        C.bbb[b, c, a, j, i, k] = -1.0 * coefficient
        C.bbb[c, a, b, j, i, k] = -1.0 * coefficient
        C.bbb[c, b, a, j, i, k] = coefficient
        C.bbb[a, b, c, k, i, j] = coefficient
        C.bbb[a, c, b, k, i, j] = -1.0 * coefficient
        C.bbb[b, a, c, k, i, j] = -1.0 * coefficient
        C.bbb[b, c, a, k, i, j] = coefficient
        C.bbb[c, a, b, k, i, j] = coefficient
        C.bbb[c, b, a, k, i, j] = -1.0 * coefficient
        C.bbb[a, b, c, k, j, i] = -1.0 * coefficient
        C.bbb[a, c, b, k, j, i] = coefficient
        C.bbb[b, a, c, k, j, i] = coefficient
        C.bbb[b, c, a, k, j, i] = -1.0 * coefficient
        C.bbb[c, a, b, k, j, i] = -1.0 * coefficient
        C.bbb[c, b, a, k, j, i] = coefficient

    elif spincase == 'aaaa':
        a, b, c, d, i, j, k, l = excitation
        C.aaaa[a, b, c, d, i, j, k, l] = coefficient

    elif spincase == 'aaab':
        a, b, c, d, i, j, k, l = excitation
        C.aaab[a, b, c, d, i, j, k, l] = coefficient

    elif spincase == 'aabb':
        a, b, c, d, i, j, k, l = excitation
        C.aabb[a, b, c, d, i, j, k, l] = coefficient

    elif spincase == 'abbb':
        a, b, c, d, i, j, k, l = excitation
        C.abbb[a, b, c, d, i, j, k, l] = coefficient

    elif spincase == 'bbbb':
        a, b, c, d, i, j, k, l = excitation
        C.bbbb[a, b, c, d, i, j, k, l] = coefficient

    return C

def contract_vt4_opt(C4_excitations, C4_amplitudes, C, H, T4_amplitudes, system):

    x2a = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha))
    x2b = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta))
    x2c = np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta))

    # Loop over aaaa determinants
    for idet in range(len(C4_amplitudes["aaaa"])):

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
    for idet in range(len(C4_amplitudes["aaab"])):

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
    for idet in range(len(C4_amplitudes["aabb"])):

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
    for idet in range(len(C4_amplitudes["abbb"])):

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
    for idet in range(len(C4_amplitudes["bbbb"])):

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