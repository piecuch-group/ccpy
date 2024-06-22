import time
import numpy as np

def get_triples_qspace(system, target_irrep=None):

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

    print(f"   Constructing triples list for Q space")
    print("   ---------------------------------------------------")
    print(f"   Target irrep = {target_irrep} ({system.point_group})")

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