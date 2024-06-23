import time
import numpy as np

def get_triples_qspace(system, t3_excitations, target_irrep=None):

    if target_irrep is None:
        sym_target = -1
    else:
        sym_target = system.point_group_irrep_to_number[target_irrep]

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

    qspace = {"aaa": np.full((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha,
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

    for idet in range(t3_excitations["aaa"].shape[0]):
        if np.array_equal(t3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1., 1.])): continue
        a, b, c, i, j, k = [x - 1 for x in t3_excitations["aaa"][idet, :]]
        qspace["aaa"][a, b, c, i, j, k] = False
    for idet in range(t3_excitations["aab"].shape[0]):
        if np.array_equal(t3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1., 1.])): continue
        a, b, c, i, j, k = [x - 1 for x in t3_excitations["aab"][idet, :]]
        qspace["aab"][a, b, c, i, j, k] = False
    for idet in range(t3_excitations["abb"].shape[0]):
        if np.array_equal(t3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1., 1.])): continue
        a, b, c, i, j, k = [x - 1 for x in t3_excitations["abb"][idet, :]]
        qspace["abb"][a, b, c, i, j, k] = False
    for idet in range(t3_excitations["bbb"].shape[0]):
        if np.array_equal(t3_excitations["bbb"][0, :], np.array([1., 1., 1., 1., 1., 1.])): continue
        a, b, c, i, j, k = [x - 1 for x in t3_excitations["bbb"][idet, :]]
        qspace["bbb"][a, b, c, i, j, k] = False

    print(f"   Constructing triples list for general Q space")
    print("   ---------------------------------------------------")
    print(f"   Target irrep = {target_irrep} ({system.point_group})")

    tic = time.perf_counter()
    excitations = {"aaa": [], "aab": [], "abb": [], "bbb": []}
    # aaa
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(j + 1, system.noccupied_alpha):
                for a in range(system.nunoccupied_alpha):
                    for b in range(a + 1, system.nunoccupied_alpha):
                        for c in range(b + 1, system.nunoccupied_alpha):
                            if qspace["aaa"][a, b, c, i, j, k]:
                                if not checksym_aaa(i, j, k, a, b, c): continue
                                excitations["aaa"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
    # aab
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(system.noccupied_beta):
                for a in range(system.nunoccupied_alpha):
                    for b in range(a + 1, system.nunoccupied_alpha):
                        for c in range(system.nunoccupied_beta):
                            if qspace["aab"][a, b, c, i, j, k]:
                                if not checksym_aab(i, j, k, a, b, c): continue
                                excitations["aab"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
    # abb
    for i in range(system.noccupied_alpha):
        for j in range(system.noccupied_beta):
            for k in range(j + 1, system.noccupied_beta):
                for a in range(system.nunoccupied_alpha):
                    for b in range(system.nunoccupied_beta):
                        for c in range(b + 1, system.nunoccupied_beta):
                            if qspace["abb"][a, b, c, i, j, k]:
                                if not checksym_abb(i, j, k, a, b, c): continue
                                excitations["abb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
    # bbb
    for i in range(system.noccupied_beta):
        for j in range(i + 1, system.noccupied_beta):
            for k in range(j + 1, system.noccupied_beta):
                for a in range(system.nunoccupied_beta):
                    for b in range(a + 1, system.nunoccupied_beta):
                        for c in range(b + 1, system.nunoccupied_beta):
                            if qspace["bbb"][a, b, c, i, j, k]:
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

def get_active_triples_qspace(system, num_active=1, target_irrep=None):
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

    print(f"   Constructing triples list for CCSDt({'I' * num_active})-type Q space")
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
                for a in range(system.nunoccupied_alpha):
                    for b in range(a + 1, system.nunoccupied_alpha):
                        for c in range(b + 1, system.nunoccupied_alpha):
                            if count_active_unocc_alpha([a, b, c]) < num_active or count_active_occ_alpha([i, j, k]) < num_active:
                                if not checksym_aaa(i, j, k, a, b, c): continue
                                excitations["aaa"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
    # aab
    for i in range(system.noccupied_alpha):
        for j in range(i + 1, system.noccupied_alpha):
            for k in range(system.noccupied_beta):
                for a in range(system.nunoccupied_alpha):
                    for b in range(a + 1, system.nunoccupied_alpha):
                        for c in range(system.nunoccupied_beta):
                            if (count_active_unocc_alpha([a, b]) + count_active_unocc_beta([c])) < num_active or (count_active_occ_alpha([i, j]) + count_active_occ_beta([k])) < num_active:
                                if not checksym_aab(i, j, k, a, b, c): continue
                                excitations["aab"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
    # abb
    for i in range(system.noccupied_alpha):
        for j in range(system.noccupied_beta):
            for k in range(j + 1, system.noccupied_beta):
                for a in range(system.nunoccupied_alpha):
                    for b in range(system.nunoccupied_beta):
                        for c in range(b + 1, system.nunoccupied_beta):
                            if (count_active_unocc_alpha([a]) + count_active_unocc_beta([b, c])) < num_active or (count_active_occ_alpha([i]) + count_active_occ_beta([j, k])) < num_active:
                                if not checksym_abb(i, j, k, a, b, c): continue
                                excitations["abb"].append([a + 1, b + 1, c + 1, i + 1, j + 1, k + 1])
    # bbb
    for i in range(system.noccupied_beta):
        for j in range(i + 1, system.noccupied_beta):
            for k in range(j + 1, system.noccupied_beta):
                for a in range(system.nunoccupied_beta):
                    for b in range(a + 1, system.nunoccupied_beta):
                        for c in range(b + 1, system.nunoccupied_beta):
                            if count_active_unocc_beta([a, b, c]) < num_active or count_active_occ_beta([i, j, k]) < num_active:
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