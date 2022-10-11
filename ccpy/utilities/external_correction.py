import numpy as np
from ccpy.models.operators import ClusterOperator
from ccpy.utilities.updates import clusteranalysis
from itertools import permutations

#print(clusteranalysis.clusteranalysis.__doc__)

def calculate_permutation_parity(lst):
    """Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd."""
    parity = 1.0
    for i in range(0, len(lst) - 1):
        if lst[i] != i:
            parity *= -1.0
            mn = min(range(i, len(lst)), key=lst.__getitem__)
            lst[i], lst[mn] = lst[mn], lst[i]
    return parity

def get_excit_rank(D, D0):
    return len(set(D) - set(D0))


def get_excits_from(D, D0):
    return list(set(D0) - set(D))


def get_excits_to(D, D0):
    return list(set(D) - set(D0))


def spatial_orb_idx(x):
    if x % 2 == 1:
        return int((x + 1) / 2)
    else:
        return int(x / 2)


def get_spincase(excits_from, excits_to):
    assert (len(excits_from) == len(excits_to))

    num_alpha_occ = sum([i % 2 for i in excits_from])
    num_alpha_unocc = sum([i % 2 for i in excits_to])

    assert (num_alpha_occ == num_alpha_unocc)

    spincase = 'a' * num_alpha_occ + 'b' * (len(excits_from) - num_alpha_occ)

    return spincase

def cluster_analysis(C, C4_excitations, C4_amplitudes, system):

    T = ClusterOperator(system, 4)

    print("Performing Cluster Analysis")
    print("---------------------------")
    print("T1... ", end="")
    T.a = C.a.copy()
    T.b = C.b.copy()
    print("done")

    print("T2... ", end="")
    T.aa, T.ab, T.bb = clusteranalysis.clusteranalysis.cluster_analysis_t2(C.a, C.b,
                                                                           C.aa, C.ab, C.bb)
    print("done")

    print("T3... ", end="")
    T.aaa, T.aab, T.abb, T.bbb = clusteranalysis.clusteranalysis.cluster_analysis_t3(C.a, C.b,
                                                                                     C.aa, C.ab, C.bb,
                                                                                     C.aaa, C.aab, C.abb, C.bbb)
    print("done")

    if T.order > 3:
        print("T4... ", end="")
        T.aaaa, T.aaab, T.aabb, T.abbb, T.bbbb = clusteranalysis.clusteranalysis.cluster_analysis_t4(C.a, C.b,
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
        print("done")

    return T

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

    print("Parsing external CI file up through quadruples")
    with open(ci_file) as f:

        for line in f.readlines():

            det = list(map(int, line.split()[2:]))
            coeff = float(line.split()[1])

            # get and save CI coefficient of HF to write everything in intermediate normalization
            if det == HF:
                c_hf = coeff
                print("Hartree-Fock coefficient = ", c_hf)
                print("")
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
            # and Slater determinant excitation {a+b+c+d+lkji} (holes are reversed!)
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

        # Put in the sign fix... not sure why this is but it has to do with the ordering of excited_det_spinorb
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
