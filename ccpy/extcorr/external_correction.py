import numpy as np

from ccpy.utilities.determinants import get_excits_from, get_excits_to, get_spincase, get_excit_rank, calculate_permutation_parity, spatial_orb_idx
from ccpy.extcorr.contractions_t3 import contract_vt3_singles
from ccpy.extcorr.contractions_t4 import contract_vt4_doubles
from ccpy.models.operators import ClusterOperator
from ccpy.models.integrals import Integral
from ccpy.lib.core import clusteranalysis
from ccpy.energy.cc_energy import get_cc_energy, get_ci_energy
import time

def cluster_analysis(wavefunction_file, hamiltonian, system):

    print("   Performing cluster analysis")
    print("   ------------------------------")
    t1 = time.perf_counter()
    t_cpu_start = time.process_time()

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
    VT_ext.a.vo, VT_ext.b.vo = contract_vt3_singles(hamiltonian, T_ext, system)

    print("   Performing < ijab | [V_N, T4(ext)] | 0 > contraction")
    VT_ext.aa.vvoo, VT_ext.ab.vvoo, VT_ext.bb.vvoo = contract_vt4_doubles(C4_excitations, hamiltonian, T4_amplitudes, system)

    elapsed_time = time.perf_counter() - t1
    t_cpu_end = time.process_time()
    minutes, seconds = divmod(elapsed_time, 60)
    print("   --------------------")
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Total CPU time: {t_cpu_end - t_cpu_start} seconds\n")
    print("")

    return T_ext, VT_ext

def cluster_analyze_ci(C, C4_excitations, C4_amplitudes, system, order=3):

    T = ClusterOperator(system, order)

    T.a = C.a.copy()
    T.b = C.b.copy()

    T.aa, T.ab, T.bb = clusteranalysis.cluster_analysis_t2(C.a, C.b,
                                                                           C.aa, C.ab, C.bb)

    T.aaa, T.aab, T.abb, T.bbb = clusteranalysis.cluster_analysis_t3(C.a, C.b,
                                                                                     C.aa, C.ab, C.bb,
                                                                                     C.aaa, C.aab, C.abb, C.bbb)

    t4_aaaa_amps, t4_aaab_amps, t4_aabb_amps, t4_abbb_amps, t4_bbbb_amps = clusteranalysis.cluster_analysis_t4(C.a, C.b,
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
        T.aaaa, T.aaab, T.aabb, T.abbb, T.bbbb = clusteranalysis.cluster_analysis_t4_full(C.a, C.b,
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
    excitation_count = {'a': 0, 'b': 0,
                        'aa': 0, 'ab': 0, 'bb': 0,
                        'aaa': 0, 'aab': 0, 'abb': 0, 'bbb': 0,
                        'aaaa': 0, 'aaab': 0, 'aabb': 0, 'abbb': 0, 'bbbb': 0}

    # C is stored in full through triples; quadruples stored as a list of what's there
    C = ClusterOperator(system, 3)
    C4_excits = {'aaaa': [], 'aaab': [], 'aabb': [], 'abbb': [], 'bbbb': []}
    C4_amps = {'aaaa': [], 'aaab': [], 'aabb': [], 'abbb': [], 'bbbb': []}

    HF = sorted(
        [2 * i - 1 for i in range(1, system.noccupied_alpha + 1)]
        + [2 * i for i in range(1, system.noccupied_beta + 1)]
    )

    occupied_lower_bound = 1
    occupied_upper_bound = system.noccupied_alpha
    unoccupied_lower_bound = 1
    unoccupied_upper_bound = system.nunoccupied_beta

    orb_table = {'a': system.noccupied_alpha, 'b': system.noccupied_beta}

    # Find the HF coefficient
    with open(ci_file) as f:

        for line in f.readlines():
            det = list(map(int, line.split()[2:]))
            coeff = float(line.split()[1])

            # get and save the CI coefficient of HF
            if det == HF:
                c_hf = coeff
                print("   Found Reference State =", det, "c_ref = ", c_hf)
                break

    with open(ci_file) as f:

        for line in f.readlines():

            det = list(map(int, line.split()[2:]))
            coeff = float(line.split()[1])

            # get and save CI coefficient of HF to write everything in intermediate normalization
            if det == HF:
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

