import time
import numpy as np
import math
from itertools import permutations

from ccpy.utilities.determinants import get_excits_from, get_excits_to, get_spincase, spatial_orb_idx, get_excit_rank


def get_empty_pspace(system, nexcit, use_bool=False):
    if nexcit == 3:
        if use_bool:
            pspace = {"aaa": np.full((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha,
                                      system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha),
                                     fill_value=False, dtype=bool),
                      "aab": np.full((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta,
                                      system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta),
                                     fill_value=False, dtype=bool),
                      "abb": np.full((system.nunoccupied_alpha, system.nunoccupied_beta, system.nunoccupied_beta,
                                      system.noccupied_alpha, system.noccupied_beta, system.noccupied_beta),
                                     fill_value=False, dtype=bool),
                      "bbb": np.full((system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta,
                                      system.noccupied_beta, system.noccupied_beta, system.noccupied_beta),
                                     fill_value=False, dtype=bool)}

        else:
            pspace = {'aaa': np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha,
                                       system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha)),
                      'aab': np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta,
                                       system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta)),
                      'abb': np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.nunoccupied_beta,
                                       system.noccupied_alpha, system.noccupied_beta, system.noccupied_beta)),
                      'bbb': np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta,
                                       system.noccupied_beta, system.noccupied_beta, system.noccupied_beta))}
    return pspace


def get_full_pspace(system, nexcit, use_bool=False):
    if nexcit == 3:
        if use_bool:
            pspace = {"aaa": np.full((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha,
                                      system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha),
                                     fill_value=True, dtype=bool),
                      "aab": np.full((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta,
                                      system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta),
                                     fill_value=True, dtype=bool),
                      "abb": np.full((system.nunoccupied_alpha, system.nunoccupied_beta, system.nunoccupied_beta,
                                      system.noccupied_alpha, system.noccupied_beta, system.noccupied_beta),
                                     fill_value=True, dtype=bool),
                      "bbb": np.full((system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta,
                                      system.noccupied_beta, system.noccupied_beta, system.noccupied_beta),
                                     fill_value=True, dtype=bool)}

        else:
            pspace = {'aaa': np.ones((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha,
                                      system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha)),
                      'aab': np.ones((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta,
                                      system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta)),
                      'abb': np.ones((system.nunoccupied_alpha, system.nunoccupied_beta, system.nunoccupied_beta,
                                      system.noccupied_alpha, system.noccupied_beta, system.noccupied_beta)),
                      'bbb': np.ones((system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta,
                                      system.noccupied_beta, system.noccupied_beta, system.noccupied_beta))}
    return pspace

def get_triples_pspace_from_cipsi(pspace_file, system):

    HF = sorted(
        [2 * i - 1 for i in range(1, system.noccupied_alpha + 1)]
        + [2 * i for i in range(1, system.noccupied_beta + 1)]
    )

    occupied_lower_bound = 1
    occupied_upper_bound = system.noccupied_alpha
    unoccupied_lower_bound = 1
    unoccupied_upper_bound = system.nunoccupied_beta

    orb_table = {'a': system.noccupied_alpha, 'b': system.noccupied_beta}
    h_sym = len(system.point_group_irrep_to_number)

    excitation_count_by_symmetry = [{'aaa': 0, 'aab': 0, 'abb': 0, 'bbb': 0} for i in range(h_sym)]
    excitations = {'aaa': [], 'aab': [], 'abb': [], 'bbb': []}

    with open(pspace_file) as f:

        for line in f.readlines():

            det = list(map(int, line.split()[2:]))

            excit_rank = get_excit_rank(det, HF)

            if excit_rank != 3: continue

            spinorb_occ = get_excits_from(det, HF)
            spinorb_unocc = get_excits_to(det, HF)
            spincase = get_spincase(spinorb_occ, spinorb_unocc)

            unocc_shift = [orb_table[x] for x in spincase]

            spinorb_occ_alpha = [i for i in spinorb_occ if i % 2 == 1]
            spinorb_occ_alpha.sort()
            spinorb_occ_beta = [i for i in spinorb_occ if i % 2 == 0]
            spinorb_occ_beta.sort()
            spinorb_unocc_alpha = [i for i in spinorb_unocc if i % 2 == 1]
            spinorb_unocc_alpha.sort()
            spinorb_unocc_beta = [i for i in spinorb_unocc if i % 2 == 0]
            spinorb_unocc_beta.sort()

            idx_occ = [spatial_orb_idx(x) for x in spinorb_occ_alpha] + [spatial_orb_idx(x) for x in spinorb_occ_beta]
            idx_unocc = [spatial_orb_idx(x) for x in spinorb_unocc_alpha] + [spatial_orb_idx(x) for x in
                                                                             spinorb_unocc_beta]
            idx_unocc = [x - shift for x, shift in zip(idx_unocc, unocc_shift)]

            if any([i > occupied_upper_bound for i in idx_occ]) or any([i < occupied_lower_bound for i in idx_occ]):
                print("Occupied orbitals out of range!")
                print(idx_occ)
                break
            if any([i > unoccupied_upper_bound for i in idx_unocc]) or any(
                    [i < unoccupied_lower_bound for i in idx_unocc]):
                print("Unoccupied orbitals out of range!")
                print(idx_unocc)
                break

            # Get the symmetry irrep of the triple excitation
            excitations[spincase].append([idx_unocc[0], idx_unocc[1], idx_unocc[2], idx_occ[0], idx_occ[1], idx_occ[2]])

            if spincase == 'aaa':
                sym = system.point_group_irrep_to_number[system.reference_symmetry]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[0] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[1] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[2] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[0] - 1 + system.noccupied_alpha]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[1] - 1 + system.noccupied_alpha]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[2] - 1 + system.noccupied_alpha]]
                excitation_count_by_symmetry[sym][spincase] += 1
                #for perms_unocc in permutations((idx_unocc[0], idx_unocc[1], idx_unocc[2])):
                #    for perms_occ in permutations((idx_occ[0], idx_occ[1], idx_occ[2])):
                #        a, b, c = perms_unocc
                #        i, j, k = perms_occ
                #        pspace[spincase][a - 1, b - 1, c - 1, i - 1, j - 1, k - 1] = 1

            if spincase == 'aab':
                sym = system.point_group_irrep_to_number[system.reference_symmetry]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[0] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[1] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[2] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[0] - 1 + system.noccupied_alpha]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[1] - 1 + system.noccupied_alpha]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[2] - 1 + system.noccupied_beta]]
                excitation_count_by_symmetry[sym][spincase] += 1
                #for perms_unocc in permutations((idx_unocc[0], idx_unocc[1])):
                #    for perms_occ in permutations((idx_occ[0], idx_occ[1])):
                #        a, b = perms_unocc
                #        i, j = perms_occ
                #        pspace[spincase][a - 1, b - 1, idx_unocc[2] - 1, i - 1, j - 1, idx_occ[2] - 1] = 1

            if spincase == 'abb':
                sym = system.point_group_irrep_to_number[system.reference_symmetry]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[0] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[1] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[2] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[0] - 1 + system.noccupied_alpha]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[1] - 1 + system.noccupied_beta]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[2] - 1 + system.noccupied_beta]]
                excitation_count_by_symmetry[sym][spincase] += 1
                #for perms_unocc in permutations((idx_unocc[1], idx_unocc[2])):
                #    for perms_occ in permutations((idx_occ[1], idx_occ[2])):
                #        b, c = perms_unocc
                #        j, k = perms_occ
                #        pspace[spincase][idx_unocc[0] - 1, b - 1, c - 1, idx_occ[0] - 1, j - 1, k - 1] = 1

            if spincase == 'bbb':
                sym = system.point_group_irrep_to_number[system.reference_symmetry]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[0] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[1] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[2] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[0] - 1 + system.noccupied_beta]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[1] - 1 + system.noccupied_beta]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[2] - 1 + system.noccupied_beta]]
                excitation_count_by_symmetry[sym][spincase] += 1
                #for perms_unocc in permutations((idx_unocc[0], idx_unocc[1], idx_unocc[2])):
                #    for perms_occ in permutations((idx_occ[0], idx_occ[1], idx_occ[2])):
                #        a, b, c = perms_unocc
                #        i, j, k = perms_occ
                #        pspace[spincase][a - 1, b - 1, c - 1, i - 1, j - 1, k - 1] = 1

    for isym, excitation_count_irrep in enumerate(excitation_count_by_symmetry):
        tot_excitation_count_irrep = excitation_count_irrep['aaa'] + excitation_count_irrep['aab'] + \
                                     excitation_count_irrep['abb'] + excitation_count_irrep['bbb']
        print("   Symmetry", system.point_group_number_to_irrep[isym], "-", "Total number of triples in P space = ",
              tot_excitation_count_irrep)
        print("      Number of aaa = ", excitation_count_irrep['aaa'])
        print("      Number of aab = ", excitation_count_irrep['aab'])
        print("      Number of abb = ", excitation_count_irrep['abb'])
        print("      Number of bbb = ", excitation_count_irrep['bbb'])

    # convert excitation arrays to Numpy arrays
    for spincase in ["aaa", "aab", "abb", "bbb"]:
        excitations[spincase] = np.asarray(excitations[spincase])
        if len(excitations[spincase].shape) < 2:
            excitations[spincase] = np.ones((1, 6))

    return excitations, excitation_count_by_symmetry


def get_quadruples_pspace_from_cipsi(pspace_file, system):
    HF = sorted(
        [2 * i - 1 for i in range(1, system.noccupied_alpha + 1)]
        + [2 * i for i in range(1, system.noccupied_beta + 1)]
    )

    occupied_lower_bound = 1
    occupied_upper_bound = system.noccupied_alpha
    unoccupied_lower_bound = 1
    unoccupied_upper_bound = system.nunoccupied_beta

    orb_table = {'a': system.noccupied_alpha, 'b': system.noccupied_beta}
    h_sym = len(system.point_group_irrep_to_number)

    excitation_count_by_symmetry = [{'aaaa': 0, 'aaab': 0, 'aabb': 0, 'abbb': 0, 'bbbb': 0} for i in range(h_sym)]
    excitations = {'aaaa': [], 'aaab': [], 'aabb': [], 'abbb': [], 'bbbb': []}

    with open(pspace_file) as f:

        for line in f.readlines():

            det = list(map(int, line.split()[2:]))

            excit_rank = get_excit_rank(det, HF)

            if excit_rank != 4: continue

            spinorb_occ = get_excits_from(det, HF)
            spinorb_unocc = get_excits_to(det, HF)
            spincase = get_spincase(spinorb_occ, spinorb_unocc)

            unocc_shift = [orb_table[x] for x in spincase]

            spinorb_occ_alpha = [i for i in spinorb_occ if i % 2 == 1]
            spinorb_occ_alpha.sort()
            spinorb_occ_beta = [i for i in spinorb_occ if i % 2 == 0]
            spinorb_occ_beta.sort()
            spinorb_unocc_alpha = [i for i in spinorb_unocc if i % 2 == 1]
            spinorb_unocc_alpha.sort()
            spinorb_unocc_beta = [i for i in spinorb_unocc if i % 2 == 0]
            spinorb_unocc_beta.sort()

            idx_occ = [spatial_orb_idx(x) for x in spinorb_occ_alpha] + [spatial_orb_idx(x) for x in spinorb_occ_beta]
            idx_unocc = [spatial_orb_idx(x) for x in spinorb_unocc_alpha] + [spatial_orb_idx(x) for x in
                                                                             spinorb_unocc_beta]
            idx_unocc = [x - shift for x, shift in zip(idx_unocc, unocc_shift)]

            if any([i > occupied_upper_bound for i in idx_occ]) or any([i < occupied_lower_bound for i in idx_occ]):
                print("Occupied orbitals out of range!")
                print(idx_occ)
                break
            if any([i > unoccupied_upper_bound for i in idx_unocc]) or any(
                    [i < unoccupied_lower_bound for i in idx_unocc]):
                print("Unoccupied orbitals out of range!")
                print(idx_unocc)
                break

            # Get the symmetry irrep of the triple excitation
            excitations[spincase].append([idx_unocc[0], idx_unocc[1], idx_unocc[2], idx_unocc[3],
                                          idx_occ[0], idx_occ[1], idx_occ[2], idx_occ[3]])

            if spincase == 'aaaa':
                sym = system.point_group_irrep_to_number[system.reference_symmetry]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[0] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[1] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[2] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[3] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[0] - 1 + system.noccupied_alpha]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[1] - 1 + system.noccupied_alpha]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[2] - 1 + system.noccupied_alpha]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[3] - 1 + system.noccupied_alpha]]
                excitation_count_by_symmetry[sym][spincase] += 1

            if spincase == 'aaab':
                sym = system.point_group_irrep_to_number[system.reference_symmetry]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[0] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[1] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[2] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[3] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[0] - 1 + system.noccupied_alpha]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[1] - 1 + system.noccupied_alpha]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[2] - 1 + system.noccupied_alpha]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[3] - 1 + system.noccupied_beta]]
                excitation_count_by_symmetry[sym][spincase] += 1

            if spincase == 'aabb':
                sym = system.point_group_irrep_to_number[system.reference_symmetry]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[0] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[1] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[2] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[3] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[0] - 1 + system.noccupied_alpha]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[1] - 1 + system.noccupied_alpha]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[2] - 1 + system.noccupied_beta]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[3] - 1 + system.noccupied_beta]]
                excitation_count_by_symmetry[sym][spincase] += 1

            if spincase == 'abbb':
                sym = system.point_group_irrep_to_number[system.reference_symmetry]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[0] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[1] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[2] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[3] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[0] - 1 + system.noccupied_alpha]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[1] - 1 + system.noccupied_beta]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[2] - 1 + system.noccupied_beta]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[3] - 1 + system.noccupied_beta]]
                excitation_count_by_symmetry[sym][spincase] += 1

            if spincase == 'bbbb':
                sym = system.point_group_irrep_to_number[system.reference_symmetry]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[0] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[1] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[2] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[idx_occ[3] - 1]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[0] - 1 + system.noccupied_beta]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[1] - 1 + system.noccupied_beta]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[2] - 1 + system.noccupied_beta]]
                sym = sym ^ system.point_group_irrep_to_number[
                    system.orbital_symmetries[idx_unocc[3] - 1 + system.noccupied_beta]]
                excitation_count_by_symmetry[sym][spincase] += 1

    for isym, excitation_count_irrep in enumerate(excitation_count_by_symmetry):
        tot_excitation_count_irrep = excitation_count_irrep['aaaa'] + excitation_count_irrep['aaab'] + \
                                     excitation_count_irrep['aabb'] + excitation_count_irrep['abbb'] + \
                                     excitation_count_irrep['bbbb']
        print("   Symmetry", system.point_group_number_to_irrep[isym], "-", "Total number of quadruples in P space = ",
              tot_excitation_count_irrep)
        print("      Number of aaaa = ", excitation_count_irrep['aaaa'])
        print("      Number of aaab = ", excitation_count_irrep['aaab'])
        print("      Number of aabb = ", excitation_count_irrep['aabb'])
        print("      Number of abbb = ", excitation_count_irrep['abbb'])
        print("      Number of bbbb = ", excitation_count_irrep['bbbb'])

    # convert excitation arrays to Numpy arrays
    for spincase in ["aaaa", "aaab", "aabb", "abbb", "bbbb"]:
        excitations[spincase] = np.asarray(excitations[spincase])
        if len(excitations[spincase].shape) < 2:
            excitations[spincase] = np.ones((1, 8))

    return excitations, excitation_count_by_symmetry

def get_pspace_from_qmc(ewalkers_file, system, sym_target=None, threshold_walkers=1, nexcit=3):
    def _check_sym_excit_(i, j, k, a, b, c, sym_target):
        if sym_target is None:
            return True
        else:
            sym_excit = (
                    system.point_group_irrep_to_number[system.reference_symmetry]
                    ^ system.point_group_irrep_to_number[system.orbital_symmetries[i - 1]]
                    ^ system.point_group_irrep_to_number[system.orbital_symmetries[j - 1]]
                    ^ system.point_group_irrep_to_number[system.orbital_symmetries[k - 1]]
                    ^ system.point_group_irrep_to_number[system.orbital_symmetries[a - 1]]
                    ^ system.point_group_irrep_to_number[system.orbital_symmetries[b - 1]]
                    ^ system.point_group_irrep_to_number[system.orbital_symmetries[c - 1]]
            )
            return sym_excit == system.point_group_irrep_to_number[sym_target]

    HF = [i for i in range(1, system.nelectrons + 1)]

    nua = system.nunoccupied_alpha
    nub = system.nunoccupied_beta
    noa = system.noccupied_alpha
    nob = system.noccupied_beta

    # Currently, implementation only works with closed-shells
    assert noa == nob and nua == nub
    nu = nua
    no = noa
    # Full P space
    pspace = np.zeros((nub, nub, nub, noa, noa, noa), dtype=np.int32)
    excitations = {"aaa": [], "aab": [], "abb": [], "bbb": []}
    num_triples = 0
    with open(ewalkers_file, "r") as f:
        for line in f.readlines():
            # Stores the information per line in a list
            # fields[0] -> determinant number
            # fields[1] -> signed walker population
            # fields[2:] -> spinorbital occupiation of correlated electrons (there are N numbers here)
            fields = [int(x) for x in line.split()]
            if abs(fields[1]) >= threshold_walkers:
                det = fields[2:]
                excits_from = list(set(HF) - set(det))  # this gives me occupied i,j,k,... in the excitation
                excits_to = list(set(det) - set(HF))  # this gives unoccupied a,b,c,... in the excitation

                excitation_rank = len(excits_from)
                if excitation_rank == 3:  # only process if you find a triple
                    # check symmetry of determinant
                    #
                    num_triples += 1
                    # sort the excitation indices
                    excits_from = [i for i in sorted(excits_from)]
                    excits_to = [a for a in sorted(excits_to)]
                    # convert to spatial
                    occ = [math.ceil(x / 2) for x in excits_from]
                    unocc = [math.ceil(x / 2) for x in excits_to]

                    for p_unocc in permutations(unocc):
                        for p_occ in permutations(occ):
                            a, b, c = p_unocc
                            i, j, k = p_occ
                            if _check_sym_excit_(i, j, k, a, b, c, sym_target):
                                pspace[a - 1 - no, b - 1 - no, c - 1 - no, i - 1, j - 1, k - 1] = 1
    num_aaa = 0
    num_aab = 0
    num_abb = 0
    num_bbb = 0
    # Extract aaa triples
    for a in range(nu):
        for b in range(a + 1, nu):
            for c in range(b + 1, nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        for k in range(j + 1, no):
                            if pspace[a, b, c, i, j, k] == 1:
                                excitations["aaa"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                                num_aaa += 1
    # Extract aab triples
    for a in range(nu):
        for b in range(a + 1, nu):
            for c in range(nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        for k in range(no):
                            if pspace[a, b, c, i, j, k] == 1:
                                excitations["aab"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                                num_aab += 1
    # Extract abb triples
    for a in range(nu):
        for b in range(nu):
            for c in range(b + 1, nu):
                for i in range(no):
                    for j in range(no):
                        for k in range(j + 1, no):
                            if pspace[a, b, c, i, j, k] == 1:
                                excitations["abb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                                num_abb += 1
    # Extract bbb triples
    for a in range(nu):
        for b in range(a + 1, nu):
            for c in range(b + 1, nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        for k in range(j + 1, no):
                            if pspace[a, b, c, i, j, k] == 1:
                                excitations["bbb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                                num_bbb += 1

    # Convert the spin-integrated lists into Numpy arrays
    for spincase, array in excitations.items():
        excitations[spincase] = np.asarray(array)
        if len(excitations[spincase].shape) < 2:
            excitations[spincase] = np.ones((1, 6))
    print("")
    print("   CIQMC P Space Excitation Summary")
    print("   ================================")
    print("   Target symmetry = ", sym_target)
    print("   Found", num_triples, "triples in CIQMC list")
    print("   num_aaa  = ", num_aaa)
    print("   num_aab  = ", num_aab)
    print("   num_abb  = ", num_abb)
    print("   num_bbb  = ", num_bbb)
    print("   Total number of triples in P space = ", num_aaa + num_aab + num_abb + num_bbb)
    print("")
    return excitations, pspace


def count_excitations_in_pspace(pspace, system):
    h_sym = len(system.point_group_irrep_to_number)

    excitation_count = [{'aaa': 0, 'aab': 0, 'abb': 0, 'bbb': 0} for i in range(h_sym)]

    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for c in range(b + 1, system.nunoccupied_alpha):
                for i in range(system.noccupied_alpha):
                    for j in range(i + 1, system.noccupied_alpha):
                        for k in range(j + 1, system.noccupied_alpha):

                            sym = system.point_group_irrep_to_number[system.reference_symmetry]
                            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[i]]
                            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
                            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
                            sym = sym ^ system.point_group_irrep_to_number[
                                system.orbital_symmetries[a + system.noccupied_alpha]]
                            sym = sym ^ system.point_group_irrep_to_number[
                                system.orbital_symmetries[b + system.noccupied_alpha]]
                            sym = sym ^ system.point_group_irrep_to_number[
                                system.orbital_symmetries[c + system.noccupied_alpha]]

                            if pspace['aaa'][a, b, c, i, j, k] == 1:
                                excitation_count[sym]['aaa'] += 1

    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for c in range(system.nunoccupied_beta):
                for i in range(system.noccupied_alpha):
                    for j in range(i + 1, system.noccupied_alpha):
                        for k in range(system.noccupied_beta):

                            sym = system.point_group_irrep_to_number[system.reference_symmetry]
                            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[i]]
                            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
                            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
                            sym = sym ^ system.point_group_irrep_to_number[
                                system.orbital_symmetries[a + system.noccupied_alpha]]
                            sym = sym ^ system.point_group_irrep_to_number[
                                system.orbital_symmetries[b + system.noccupied_alpha]]
                            sym = sym ^ system.point_group_irrep_to_number[
                                system.orbital_symmetries[c + system.noccupied_beta]]

                            if pspace['aab'][a, b, c, i, j, k] == 1:
                                excitation_count[sym]['aab'] += 1

    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for c in range(b + 1, system.nunoccupied_beta):
                for i in range(system.noccupied_alpha):
                    for j in range(system.noccupied_beta):
                        for k in range(j + 1, system.noccupied_beta):

                            sym = system.point_group_irrep_to_number[system.reference_symmetry]
                            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[i]]
                            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
                            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
                            sym = sym ^ system.point_group_irrep_to_number[
                                system.orbital_symmetries[a + system.noccupied_alpha]]
                            sym = sym ^ system.point_group_irrep_to_number[
                                system.orbital_symmetries[b + system.noccupied_beta]]
                            sym = sym ^ system.point_group_irrep_to_number[
                                system.orbital_symmetries[c + system.noccupied_beta]]

                            if pspace['abb'][a, b, c, i, j, k] == 1:
                                excitation_count[sym]['abb'] += 1

    for a in range(system.nunoccupied_beta):
        for b in range(a + 1, system.nunoccupied_beta):
            for c in range(b + 1, system.nunoccupied_beta):
                for i in range(system.noccupied_beta):
                    for j in range(i + 1, system.noccupied_beta):
                        for k in range(j + 1, system.noccupied_beta):

                            sym = system.point_group_irrep_to_number[system.reference_symmetry]
                            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[i]]
                            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
                            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
                            sym = sym ^ system.point_group_irrep_to_number[
                                system.orbital_symmetries[a + system.noccupied_beta]]
                            sym = sym ^ system.point_group_irrep_to_number[
                                system.orbital_symmetries[b + system.noccupied_beta]]
                            sym = sym ^ system.point_group_irrep_to_number[
                                system.orbital_symmetries[c + system.noccupied_beta]]

                            if pspace['bbb'][a, b, c, i, j, k] == 1:
                                excitation_count[sym]['bbb'] += 1

    for isym, excitation_count_irrep in enumerate(excitation_count):
        tot_excitation_count_irrep = excitation_count_irrep['aaa'] + excitation_count_irrep['aab'] + \
                                     excitation_count_irrep['abb'] + excitation_count_irrep['bbb']
        print("   Symmetry", system.point_group_number_to_irrep[isym], "-", "Total number of triples in P space = ",
              tot_excitation_count_irrep)
        print("      Number of aaa = ", excitation_count_irrep['aaa'])
        print("      Number of aab = ", excitation_count_irrep['aab'])
        print("      Number of abb = ", excitation_count_irrep['abb'])
        print("      Number of bbb = ", excitation_count_irrep['bbb'])

    return excitation_count

def get_active_triples_pspace(system, num_active=1, target_irrep=None):
    from ccpy.utilities.active_space import active_hole, active_particle

    nacto_alpha = system.num_act_occupied_alpha
    nactu_alpha = system.num_act_unoccupied_alpha
    nacto_beta = system.num_act_occupied_beta
    nactu_beta = system.num_act_unoccupied_beta

    if target_irrep is None:
        sym_target = -1
    else:
        sym_target = system.point_group_irrep_to_number[target_irrep]

    def count_active_occ_alpha(occ):
        return sum([active_hole(i, system.noccupied_alpha, nacto_alpha) for i in occ])

    def count_active_occ_beta(occ):
        return sum([active_hole(i, system.noccupied_beta, nacto_beta) for i in occ])

    def count_active_unocc_alpha(unocc):
        return sum([active_particle(a, nactu_alpha) for a in unocc])

    def count_active_unocc_beta(unocc):
        return sum([active_particle(a, nactu_beta) for a in unocc])

    def checksym_aaa(i, j, k, a, b, c):
        if sym_target == -1: return True
        sym = system.point_group_irrep_to_number[system.reference_symmetry]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[i]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[a + system.noccupied_alpha]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[b + system.noccupied_alpha]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[c + system.noccupied_alpha]]
        if sym == sym_target:
            return True
        else:
            return False

    def checksym_aab(i, j, k, a, b, c):
        if sym_target == -1: return True
        sym = system.point_group_irrep_to_number[system.reference_symmetry]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[i]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[a + system.noccupied_alpha]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[b + system.noccupied_alpha]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[c + system.noccupied_beta]]
        if sym == sym_target:
            return True
        else:
            return False

    def checksym_abb(i, j, k, a, b, c):
        if sym_target == -1: return True
        sym = system.point_group_irrep_to_number[system.reference_symmetry]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[i]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[a + system.noccupied_alpha]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[b + system.noccupied_beta]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[c + system.noccupied_beta]]
        if sym == sym_target:
            return True
        else:
            return False

    def checksym_bbb(i, j, k, a, b, c):
        if sym_target == -1: return True
        sym = system.point_group_irrep_to_number[system.reference_symmetry]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[i]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[a + system.noccupied_beta]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[b + system.noccupied_beta]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[c + system.noccupied_beta]]
        if sym == sym_target:
            return True
        else:
            return False

    print(f"   Constructing triples list for CCSDt({'I' * num_active})-type P space")
    print("   ---------------------------------------------------")
    print(f"   Target irrep = {target_irrep} ({system.point_group})")
    print("   Number of active occupied alpha orbitals = ", system.num_act_occupied_alpha)
    print("   Number of active unoccupied alpha orbitals = ", system.num_act_unoccupied_alpha)
    print("   Number of active occupied beta orbitals = ", system.num_act_occupied_beta)
    print("   Number of active unoccupied beta orbitals = ", system.num_act_unoccupied_beta)

    tic = time.perf_counter()
    excitations = {"aaa": [], "aab": [], "abb": [], "bbb": []}
    # aaa
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(j + 1, system.noccupied_alpha):
                if count_active_occ_alpha([i, j, k]) < num_active: continue
                for a in range(system.nunoccupied_alpha):
                    for b in range(a + 1, system.nunoccupied_alpha):
                        for c in range(b + 1, system.nunoccupied_alpha):
                            if count_active_unocc_alpha([a, b, c]) >= num_active:
                                if not checksym_aaa(i, j, k, a, b, c): continue
                                excitations["aaa"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
    # aab
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(system.noccupied_beta):
                if (count_active_occ_alpha([i, j]) + count_active_occ_beta([k])) < num_active: continue
                for a in range(system.nunoccupied_alpha):
                    for b in range(a + 1, system.nunoccupied_alpha):
                        for c in range(system.nunoccupied_beta):
                            if (count_active_unocc_alpha([a, b]) + count_active_unocc_beta([c])) >= num_active:
                                if not checksym_aab(i, j, k, a, b, c): continue
                                excitations["aab"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
    # abb
    for i in range(system.noccupied_alpha):
        for j in range(system.noccupied_beta):
            for k in range(j + 1, system.noccupied_beta):
                if (count_active_occ_alpha([i]) + count_active_occ_beta([j, k])) < num_active: continue
                for a in range(system.nunoccupied_alpha):
                    for b in range(system.nunoccupied_beta):
                        for c in range(b + 1, system.nunoccupied_beta):
                            if (count_active_unocc_alpha([a]) + count_active_unocc_beta([b, c])) >= num_active:
                                if not checksym_abb(i, j, k, a, b, c): continue
                                excitations["abb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
    # bbb
    for i in range(system.noccupied_beta):
        for j in range(i + 1, system.noccupied_beta):
            for k in range(j + 1, system.noccupied_beta):
                if count_active_occ_beta([i, j, k]) < num_active: continue
                for a in range(system.nunoccupied_beta):
                    for b in range(a + 1, system.nunoccupied_beta):
                        for c in range(b + 1, system.nunoccupied_beta):
                            if count_active_unocc_beta([a, b, c]) >= num_active:
                                if not checksym_bbb(i, j, k, a, b, c): continue
                                excitations["bbb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
    # Convert the spin-integrated lists into Numpy arrays
    for spincase, array in excitations.items():
        excitations[spincase] = np.asarray(array, order="F")
        if len(excitations[spincase].shape) < 2:
            excitations[spincase] = np.ones((1, 6))
        # Print the number of triples of a given spincase 
        print(f"   Spincase {spincase} contains {excitations[spincase].shape[0]} triples")

    toc = time.perf_counter()
    minutes, seconds = divmod(toc - tic, 60)
    print(f"   Completed in {minutes:.1f}m {seconds:.1f}s\n")
    return excitations


def get_active_3p2h_pspace(system, num_active=1, target_irrep=None):
    from ccpy.utilities.active_space import active_hole, active_particle

    assert system.num_act_occupied_alpha == system.num_act_occupied_beta
    assert system.num_act_unoccupied_alpha == system.num_act_unoccupied_beta

    nact_o = system.num_act_occupied_alpha
    nact_u = system.num_act_unoccupied_alpha

    if target_irrep is None:
        sym_target = -1
    else:
        sym_target = system.point_group_irrep_to_number[target_irrep]

    def count_active_occ_alpha(occ):
        return sum([active_hole(i, system.noccupied_alpha, nact_o) for i in occ])

    def count_active_occ_beta(occ):
        return sum([active_hole(i, system.noccupied_beta, nact_o) for i in occ])

    def count_active_unocc_alpha(unocc):
        return sum([active_particle(a, nact_u) for a in unocc])

    def count_active_unocc_beta(unocc):
        return sum([active_particle(a, nact_u) for a in unocc])

    def checksym_aaa(j, k, a, b, c):
        if sym_target == -1: return True
        sym = system.point_group_irrep_to_number[system.reference_symmetry]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[a + system.noccupied_alpha]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[b + system.noccupied_alpha]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[c + system.noccupied_alpha]]
        if sym == sym_target:
            return True
        else:
            return False

    def checksym_aab(j, k, a, b, c):
        if sym_target == -1: return True
        sym = system.point_group_irrep_to_number[system.reference_symmetry]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[a + system.noccupied_alpha]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[b + system.noccupied_alpha]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[c + system.noccupied_beta]]
        if sym == sym_target:
            return True
        else:
            return False

    def checksym_abb(j, k, a, b, c):
        if sym_target == -1: return True
        sym = system.point_group_irrep_to_number[system.reference_symmetry]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[a + system.noccupied_alpha]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[b + system.noccupied_beta]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[c + system.noccupied_beta]]
        if sym == sym_target:
            return True
        else:
            return False

    print(f"   Constructing 3p-2h list for EA-EOMCCSDt({'I' * num_active})-type P space")
    print("   -------------------------------------------------------")
    print(f"   Target irrep = {target_irrep} ({system.point_group})")
    print("   Number of active occupied alpha orbitals = ", system.num_act_occupied_alpha)
    print("   Number of active unoccupied alpha orbitals = ", system.num_act_unoccupied_alpha)
    print("   Number of active occupied beta orbitals = ", system.num_act_occupied_beta)
    print("   Number of active unoccupied beta orbitals = ", system.num_act_unoccupied_beta)

    tic = time.perf_counter()
    excitations = {"aaa": [], "aab": [], "abb": []}
    # aaa
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for c in range(b + 1, system.nunoccupied_alpha):
                if count_active_unocc_alpha([a, b, c]) < num_active: continue
                for j in range(system.noccupied_alpha):
                    for k in range(j + 1, system.noccupied_alpha):
                        if not checksym_aaa(j, k, a, b, c): continue
                        excitations["aaa"].append([a + 1, b + 1, c + 1, j + 1, k + 1])
    # aab
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for c in range(system.nunoccupied_beta):
                if (count_active_unocc_alpha([a, b]) + count_active_unocc_beta([c])) < num_active: continue
                for j in range(system.noccupied_alpha):
                    for k in range(system.noccupied_beta):
                        if not checksym_aab(j, k, a, b, c): continue
                        excitations["aab"].append([a + 1, b + 1, c + 1, j + 1, k + 1])
    # abb
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for c in range(b + 1, system.nunoccupied_beta):
                if (count_active_unocc_alpha([a]) + count_active_unocc_beta([b, c])) < num_active: continue
                for j in range(system.noccupied_beta):
                    for k in range(j + 1, system.noccupied_beta):
                        if not checksym_abb(j, k, a, b, c): continue
                        excitations["abb"].append([a + 1, b + 1, c + 1, j + 1, k + 1])

    # Convert the spin-integrated lists into Numpy arrays
    for spincase, array in excitations.items():
        excitations[spincase] = np.asarray(array, order="F")
        if len(excitations[spincase].shape) < 2:
            excitations[spincase] = np.ones((1, 5))
        # Print the number of triples of a given spincase
        print(f"   Spincase {spincase} contains {excitations[spincase].shape[0]} triples")

    toc = time.perf_counter()
    minutes, seconds = divmod(toc - tic, 60)
    print(f"   Completed in {minutes:.1f}m {seconds:.1f}s\n")
    return excitations


def get_active_3h2p_pspace(system, num_active=1, target_irrep=None):
    from ccpy.utilities.active_space import active_hole, active_particle

    assert system.num_act_occupied_alpha == system.num_act_occupied_beta
    assert system.num_act_unoccupied_alpha == system.num_act_unoccupied_beta

    nact_o = system.num_act_occupied_alpha
    nact_u = system.num_act_unoccupied_alpha

    if target_irrep is None:
        sym_target = -1
    else:
        sym_target = system.point_group_irrep_to_number[target_irrep]

    def count_active_occ_alpha(occ):
        return sum([active_hole(i, system.noccupied_alpha, nact_o) for i in occ])

    def count_active_occ_beta(occ):
        return sum([active_hole(i, system.noccupied_beta, nact_o) for i in occ])

    def count_active_unocc_alpha(unocc):
        return sum([active_particle(a, nact_u) for a in unocc])

    def count_active_unocc_beta(unocc):
        return sum([active_particle(a, nact_u) for a in unocc])

    def checksym_aaa(i, j, k, b, c):
        if sym_target == -1: return True
        sym = system.point_group_irrep_to_number[system.reference_symmetry]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[i]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[b + system.noccupied_alpha]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[c + system.noccupied_alpha]]
        if sym == sym_target:
            return True
        else:
            return False

    def checksym_aab(i, j, k, b, c):
        if sym_target == -1: return True
        sym = system.point_group_irrep_to_number[system.reference_symmetry]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[i]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[b + system.noccupied_alpha]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[c + system.noccupied_beta]]
        if sym == sym_target:
            return True
        else:
            return False

    def checksym_abb(i, j, k, b, c):
        if sym_target == -1: return True
        sym = system.point_group_irrep_to_number[system.reference_symmetry]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[i]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[b + system.noccupied_beta]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[c + system.noccupied_beta]]
        if sym == sym_target:
            return True
        else:
            return False

    print(f"   Constructing 3h-2p list for IP-EOMCCSDt({'I' * num_active})-type P space")
    print("   -------------------------------------------------------")
    print(f"   Target irrep = {target_irrep} ({system.point_group})")
    print("   Number of active occupied alpha orbitals = ", system.num_act_occupied_alpha)
    print("   Number of active unoccupied alpha orbitals = ", system.num_act_unoccupied_alpha)
    print("   Number of active occupied beta orbitals = ", system.num_act_occupied_beta)
    print("   Number of active unoccupied beta orbitals = ", system.num_act_unoccupied_beta)

    tic = time.perf_counter()
    excitations = {"aaa": [], "aab": [], "abb": []}
    # aaa
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(j + 1, system.noccupied_alpha):
                if count_active_occ_alpha([i, j, k]) < num_active: continue
                for b in range(system.nunoccupied_alpha):
                    for c in range(b + 1, system.nunoccupied_alpha):
                        if not checksym_aaa(i, j, k, b, c): continue
                        excitations["aaa"].append([b + 1, c + 1, i + 1, j + 1, k + 1])
    # aab
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(system.noccupied_beta):
                if (count_active_occ_alpha([i, j]) + count_active_occ_beta([k])) < num_active: continue
                for b in range(system.nunoccupied_alpha):
                    for c in range(system.nunoccupied_beta):
                        if not checksym_aab(i, j, k, b, c): continue
                        excitations["aab"].append([b + 1, c + 1, i + 1, j + 1, k + 1])
    # abb
    for i in range(system.noccupied_alpha):
        for j in range(system.noccupied_beta):
            for k in range(j + 1, system.noccupied_beta):
                if (count_active_occ_alpha([i]) + count_active_occ_beta([j, k])) < num_active: continue
                for b in range(system.nunoccupied_beta):
                    for c in range(b + 1, system.nunoccupied_beta):
                        if not checksym_abb(i, j, k, b, c): continue
                        excitations["abb"].append([b + 1, c + 1, i + 1, j + 1, k + 1])

    # Convert the spin-integrated lists into Numpy arrays
    for spincase, array in excitations.items():
        excitations[spincase] = np.asarray(array, order="F")
        if len(excitations[spincase].shape) < 2:
            excitations[spincase] = np.ones((1, 5))
        # Print the number of triples of a given spincase
        print(f"   Spincase {spincase} contains {excitations[spincase].shape[0]} triples")

    toc = time.perf_counter()
    minutes, seconds = divmod(toc - tic, 60)
    print(f"   Completed in {minutes:.1f}m {seconds:.1f}s\n")
    return excitations

def overlap_p_spaces(excits1, excits2):
    # if excits1 is an empty list (initial step of recusive procedure), then return excits 2 as the overlapped
    if isinstance(excits1, list):
        if not excits1:
            return excits2
    # otherwise, concatenate the excitation arrays and return the unique rows
    return np.unique(np.vstack((excits1, excits2)), axis=0)

