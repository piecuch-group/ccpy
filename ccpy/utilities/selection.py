import numpy as np
from itertools import permutations
from ccpy.utilities.determinants import spatial_orb_idx

def add_spinorbital_triples_to_pspace(triples_list, t3_excitations, excitation_count_by_symmetry, system, RHF_symmetry):
    """Expand the size of the previous P space using the determinants contained in the list
    of triples (stored as a, b, c, i, j, k) in triples_list. The variable triples_list stores
    triples in spinorbital form, where all orbital indices start from 1 and odd indices
    correspond to alpha orbitals while even indices correspond to beta orbitals."""

    def _add_t3_excitations(new_t3_excitations, old_t3_excitations, num_add, spincase):
        if num_add > 0:
            if np.array_equal(old_t3_excitations[spincase][0, :], np.ones(6)):
                new_t3_excitations[spincase] = np.asarray(new_t3_excitations[spincase])
            else:
                new_t3_excitations[spincase] = np.vstack(
                    (old_t3_excitations[spincase], np.asarray(new_t3_excitations[spincase])))
        else:
            new_t3_excitations[spincase] = old_t3_excitations[spincase].copy()

        return new_t3_excitations

    def _get_excitation_symmetry(a, b, c, i, j, k, spincase):
        sym = system.point_group_irrep_to_number[system.reference_symmetry]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[i]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[j]]
        sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[k]]
        if spincase == "aaa":
            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[a + system.noccupied_alpha]]
            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[b + system.noccupied_alpha]]
            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[c + system.noccupied_alpha]]
        elif spincase == "aab":
            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[a + system.noccupied_alpha]]
            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[b + system.noccupied_alpha]]
            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[c + system.noccupied_beta]]
        elif spincase == "abb":
            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[a + system.noccupied_alpha]]
            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[b + system.noccupied_beta]]
            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[c + system.noccupied_beta]]
        elif spincase == "bbb":
            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[a + system.noccupied_beta]]
            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[b + system.noccupied_beta]]
            sym = sym ^ system.point_group_irrep_to_number[system.orbital_symmetries[c + system.noccupied_beta]]
        return sym

    new_t3_excitations = {
        "aaa": [],
        "aab": [],
        "abb": [],
        "bbb": []
    }

    num_add = triples_list.shape[0]

    n3aaa = 0
    n3aab = 0
    n3abb = 0
    n3bbb = 0
    for n in range(num_add):

        num_alpha = int(sum([x % 2 for x in triples_list[n, :]]) / 2)
        idx = [spatial_orb_idx(p) - 1 for p in triples_list[n, :]]
        a, b, c, i, j, k = idx

        # skips over triple if it is unset from selection routine, meaning that all entries are -1
        if a == -1 and b == -1 and c == -1 and i == -1 and j == -1 and k == -1: continue

        if num_alpha == 3:
            new_t3_excitations["aaa"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
            n3aaa += 1
            # find symmetry of excitation and increment excitation count
            sym = _get_excitation_symmetry(a, b, c, i, j, k, "aaa")
            excitation_count_by_symmetry[sym]["aaa"] += 1
            if RHF_symmetry:  # include the same bbb excitations if RHF symmetry is applied
                new_t3_excitations["bbb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                n3bbb += 1
                # find symmetry of excitation and increment excitation count
                sym = _get_excitation_symmetry(a, b, c, i, j, k, "bbb")
                excitation_count_by_symmetry[sym]["bbb"] += 1

        if num_alpha == 2:
            new_t3_excitations["aab"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
            n3aab += 1
            # find symmetry of excitation and increment excitation count
            sym = _get_excitation_symmetry(a, b, c, i, j, k, "aab")
            excitation_count_by_symmetry[sym]["aab"] += 1
            if RHF_symmetry:  # include the same abb excitations if RHF symmetry is applied
                new_t3_excitations["abb"].append([c + 1, a + 1, b + 1, k + 1, i + 1, j + 1])
                n3abb += 1
                # find symmetry of excitation and increment excitation count
                sym = _get_excitation_symmetry(c, a, b, k, i, j, "abb")
                excitation_count_by_symmetry[sym]["abb"] += 1

        if not RHF_symmetry:  # only consider adding abb and bbb excitations if not using RHF

            if num_alpha == 1:
                new_t3_excitations["abb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                n3abb += 1
                # find symmetry of excitation and increment excitation count
                sym = _get_excitation_symmetry(a, b, c, i, j, k, "abb")
                excitation_count_by_symmetry[sym]["abb"] += 1

            if num_alpha == 0:
                new_t3_excitations["bbb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                n3bbb += 1
                # find symmetry of excitation and increment excitation count
                sym = _get_excitation_symmetry(a, b, c, i, j, k, "bbb")
                excitation_count_by_symmetry[sym]["bbb"] += 1

    # Update the t3 excitation lists with the new content from the moment selection
    new_t3_excitations = _add_t3_excitations(new_t3_excitations, t3_excitations, n3aaa, "aaa")
    new_t3_excitations = _add_t3_excitations(new_t3_excitations, t3_excitations, n3aab, "aab")
    new_t3_excitations = _add_t3_excitations(new_t3_excitations, t3_excitations, n3abb, "abb")
    new_t3_excitations = _add_t3_excitations(new_t3_excitations, t3_excitations, n3bbb, "bbb")

    return new_t3_excitations, excitation_count_by_symmetry

def add_spinorbital_triples_to_pspace_bak(triples_list, pspace, t3_excitations, RHF_symmetry):
    """Expand the size of the previous P space using the determinants contained in the list
    of triples (stored as a, b, c, i, j, k) in triples_list. The variable triples_list stores
    triples in spinorbital form, where all orbital indices start from 1 and odd indices
    correspond to alpha orbitals while even indices corresopnd to beta orbitals."""

    def _add_t3_excitations(new_t3_excitations, old_t3_excitations, num_add, spincase):
        if num_add > 0:
            if np.array_equal(old_t3_excitations[spincase][0, :], np.ones(6)):
                new_t3_excitations[spincase] = np.asarray(new_t3_excitations[spincase])
            else:
                new_t3_excitations[spincase] = np.vstack(
                    (old_t3_excitations[spincase], np.asarray(new_t3_excitations[spincase])))
        else:
            new_t3_excitations[spincase] = old_t3_excitations[spincase].copy()

        return new_t3_excitations

    # should these be copied?
    new_pspace = {
        "aaa": pspace["aaa"],
        "aab": pspace["aab"],
        "abb": pspace["abb"],
        "bbb": pspace["bbb"],
    }
    new_t3_excitations = {
        "aaa": [],
        "aab": [],
        "abb": [],
        "bbb": []
    }

    num_add = triples_list.shape[0]

    n3aaa = 0
    n3aab = 0
    n3abb = 0
    n3bbb = 0
    for n in range(num_add):

        num_alpha = int(sum([x % 2 for x in triples_list[n, :]]) / 2)
        idx = [spatial_orb_idx(p) - 1 for p in triples_list[n, :]]
        a, b, c, i, j, k = idx

        if a == 0 and b == 0 and c == 0 and i == 0 and j == 0 and k == 0: continue

        if num_alpha == 3:
            new_t3_excitations["aaa"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
            n3aaa += 1
            for perms_unocc in permutations((a, b, c)):
                for perms_occ in permutations((i, j, k)):
                    a, b, c = perms_unocc
                    i, j, k = perms_occ
                    new_pspace['aaa'][a, b, c, i, j, k] = True

            if RHF_symmetry:  # include the same bbb excitations if RHF symmetry is applied
                new_t3_excitations["bbb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                n3bbb += 1
                for perms_unocc in permutations((a, b, c)):
                    for perms_occ in permutations((i, j, k)):
                        a, b, c = perms_unocc
                        i, j, k = perms_occ
                        new_pspace['bbb'][a, b, c, i, j, k] = True

        if num_alpha == 2:
            new_t3_excitations["aab"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
            n3aab += 1
            for perms_unocc in permutations((a, b)):
                for perms_occ in permutations((i, j)):
                    a, b = perms_unocc
                    i, j = perms_occ
                    new_pspace['aab'][a, b, c, i, j, k] = True

            if RHF_symmetry:  # include the same abb excitations if RHF symmetry is applied
                new_t3_excitations["abb"].append([c + 1, a + 1, b + 1, k + 1, i + 1, j + 1])
                n3abb += 1
                for perms_unocc in permutations((a, b)):
                    for perms_occ in permutations((i, j)):
                        a, b = perms_unocc
                        i, j = perms_occ
                        new_pspace['abb'][c, a, b, k, i, j] = True

        if not RHF_symmetry:  # only consider adding abb and bbb excitations if not using RHF

            if num_alpha == 1:
                new_t3_excitations["abb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                n3abb += 1
                for perms_unocc in permutations((b, c)):
                    for perms_occ in permutations((j, k)):
                        b, c = perms_unocc
                        j, k = perms_occ
                        new_pspace['abb'][a, b, c, i, j, k] = True

            if num_alpha == 0:
                new_t3_excitations["bbb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                n3bbb += 1
                for perms_unocc in permutations((a, b, c)):
                    for perms_occ in permutations((i, j, k)):
                        a, b, c = perms_unocc
                        i, j, k = perms_occ
                        new_pspace['bbb'][a, b, c, i, j, k] = True

    # Update the t3 excitation lists with the new content from the moment selection
    new_t3_excitations = _add_t3_excitations(new_t3_excitations, t3_excitations, n3aaa, "aaa")
    new_t3_excitations = _add_t3_excitations(new_t3_excitations, t3_excitations, n3aab, "aab")
    new_t3_excitations = _add_t3_excitations(new_t3_excitations, t3_excitations, n3abb, "abb")
    new_t3_excitations = _add_t3_excitations(new_t3_excitations, t3_excitations, n3bbb, "bbb")

    return new_pspace, new_t3_excitations


def adaptive_triples_selection_from_moments(moments, pspace, t3_excitations, num_add, system, RHF_symmetry):
    def _add_t3_excitations(new_t3_excitations, old_t3_excitations, num_add, spincase):
        if num_add > 0:
            if np.array_equal(old_t3_excitations[spincase][0, :], np.ones(6)):
                new_t3_excitations[spincase] = np.asarray(new_t3_excitations[spincase])
            else:
                new_t3_excitations[spincase] = np.vstack(
                    (old_t3_excitations[spincase], np.asarray(new_t3_excitations[spincase])))
        else:
            new_t3_excitations[spincase] = old_t3_excitations[spincase].copy()

        return new_t3_excitations

    # should these be copied?
    new_pspace = {
        "aaa": pspace["aaa"],
        "aab": pspace["aab"],
        "abb": pspace["abb"],
        "bbb": pspace["bbb"],
    }
    new_t3_excitations = {
        "aaa": [],
        "aab": [],
        "abb": [],
        "bbb": []
    }

    n3aaa = system.noccupied_alpha ** 3 * system.nunoccupied_alpha ** 3
    n3aab = system.noccupied_alpha ** 2 * system.noccupied_beta * system.nunoccupied_alpha ** 2 * system.nunoccupied_beta
    n3abb = system.noccupied_alpha * system.noccupied_beta ** 2 * system.nunoccupied_alpha * system.nunoccupied_beta ** 2
    n3bbb = system.noccupied_beta ** 3 * system.nunoccupied_beta ** 3

    # Create full moment vector and populate it with different spin cases
    if RHF_symmetry:
        mvec = np.zeros(n3aaa + n3aab)
        mvec[:n3aaa] = moments["aaa"].flatten()
        mvec[n3aaa:] = moments["aab"].flatten()
    else:
        mvec = np.zeros(n3aaa + n3aab + n3abb + n3bbb)
        mvec[:n3aaa] = moments["aaa"].flatten()
        mvec[n3aaa:n3aaa + n3aab] = moments["aab"].flatten()
        mvec[n3aaa + n3aab:n3aaa + n3aab + n3abb] = moments["abb"].flatten()
        mvec[n3aaa + n3aab + n3abb:] = moments["bbb"].flatten()

    idx = np.flip(np.argsort(abs(mvec)))  # sort the moments in descending order by absolute value

    ct = 0
    n3a = 0
    n3b = 0
    n3c = 0
    n3d = 0

    if RHF_symmetry:
        stop_fcn = lambda n3a, n3b, n3c, n3d: n3a + n3b
    else:
        stop_fcn = lambda n3a, n3b, n3c, n3d: n3a + n3b + n3c + n3d

    while stop_fcn(n3a, n3b, n3c, n3d) < num_add:
        if idx[ct] < n3aaa:
            a, b, c, i, j, k = np.unravel_index(idx[ct], moments["aaa"].shape)
            if pspace["aaa"][a, b, c, i, j, k]:
                ct += 1
                continue
            else:
                n3a += 1
                new_t3_excitations["aaa"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                for perms_unocc in permutations((a, b, c)):
                    for perms_occ in permutations((i, j, k)):
                        a, b, c = perms_unocc
                        i, j, k = perms_occ
                        new_pspace['aaa'][a, b, c, i, j, k] = True
                if RHF_symmetry:  # include the same bbb excitations if RHF symmetry is applied
                    new_t3_excitations["bbb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                    n3d += 1
                    for perms_unocc in permutations((a, b, c)):
                        for perms_occ in permutations((i, j, k)):
                            a, b, c = perms_unocc
                            i, j, k = perms_occ
                            new_pspace['bbb'][a, b, c, i, j, k] = True
        elif idx[ct] < n3aaa + n3aab:
            a, b, c, i, j, k = np.unravel_index(idx[ct] - n3aaa, moments["aab"].shape)
            if pspace["aab"][a, b, c, i, j, k]:
                ct += 1
                continue
            else:
                n3b += 1
                new_t3_excitations["aab"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                for perms_unocc in permutations((a, b)):
                    for perms_occ in permutations((i, j)):
                        a, b = perms_unocc
                        i, j = perms_occ
                        new_pspace['aab'][a, b, c, i, j, k] = True
                if RHF_symmetry:  # include the same abb excitations if RHF symmetry is applied
                    new_t3_excitations["abb"].append([c + 1, a + 1, b + 1, k + 1, i + 1, j + 1])
                    n3c += 1
                    for perms_unocc in permutations((a, b)):
                        for perms_occ in permutations((i, j)):
                            a, b = perms_unocc
                            i, j = perms_occ
                            new_pspace['abb'][c, a, b, k, i, j] = True
        elif idx[ct] < n3aaa + n3aab + n3abb and not RHF_symmetry:
            a, b, c, i, j, k = np.unravel_index(idx[ct] - n3aaa - n3aab, moments["abb"].shape)
            if pspace["abb"][a, b, c, i, j, k]:
                ct += 1
                continue
            else:
                n3c += 1
                new_t3_excitations["abb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                for perms_unocc in permutations((b, c)):
                    for perms_occ in permutations((j, k)):
                        b, c = perms_unocc
                        j, k = perms_occ
                        new_pspace['abb'][a, b, c, i, j, k] = True
        elif not RHF_symmetry:
            a, b, c, i, j, k = np.unravel_index(idx[ct] - n3aaa - n3aab - n3abb, moments["bbb"].shape)
            if pspace["bbb"][a, b, c, i, j, k]:
                ct += 1
                continue
            else:
                n3d += 1
                new_t3_excitations["bbb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
                for perms_unocc in permutations((a, b, c)):
                    for perms_occ in permutations((i, j, k)):
                        a, b, c = perms_unocc
                        i, j, k = perms_occ
                        new_pspace['bbb'][a, b, c, i, j, k] = True

    # Update the t3 excitation lists with the new content from the moment selection
    new_t3_excitations = _add_t3_excitations(new_t3_excitations, t3_excitations, n3a, "aaa")
    new_t3_excitations = _add_t3_excitations(new_t3_excitations, t3_excitations, n3b, "aab")
    new_t3_excitations = _add_t3_excitations(new_t3_excitations, t3_excitations, n3c, "abb")
    new_t3_excitations = _add_t3_excitations(new_t3_excitations, t3_excitations, n3d, "bbb")

    return new_pspace, new_t3_excitations