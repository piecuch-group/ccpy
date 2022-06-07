import numpy as np

def get_empty_pspace(system, nexcit):
    if nexcit == 3:
        pspace = [{'aaa' : np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha,
                                    system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha)),
                  'aab' : np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta,
                                    system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta)),
                  'abb' : np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.nunoccupied_beta,
                                    system.noccupied_alpha, system.noccupied_beta, system.noccupied_beta)),
                  'bbb' : np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta,
                                    system.noccupied_beta, system.noccupied_beta, system.noccupied_beta))}]
    if nexcit == 4:
        pspace = [ {'aaa' : np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha,
                                    system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha)),
                   'aab' : np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta,
                                    system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta)),
                   'abb' : np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.nunoccupied_beta,
                                    system.noccupied_alpha, system.noccupied_beta, system.noccupied_beta)),
                   'bbb' : np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta,
                                    system.noccupied_beta, system.noccupied_beta, system.noccupied_beta))},
                   {'aaaa' : np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha,
                                    system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha)),
                   'aaab' : np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta,
                                    system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta)),
                   'aabb' : np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta, system.nunoccupied_beta,
                                    system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta, system.noccupied_beta)),
                   'bbbb' : np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta,
                                    system.noccupied_beta, system.noccupied_beta, system.noccupied_beta, system.noccupied_beta))} ]
    return pspace

def get_full_pspace(system, nexcit):
    if nexcit == 3:
        pspace = [{'aaa' : np.ones((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha,
                                    system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha)),
                  'aab' : np.ones((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta,
                                    system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta)),
                  'abb' : np.ones((system.nunoccupied_alpha, system.nunoccupied_beta, system.nunoccupied_beta,
                                    system.noccupied_alpha, system.noccupied_beta, system.noccupied_beta)),
                  'bbb' : np.ones((system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta,
                                    system.noccupied_beta, system.noccupied_beta, system.noccupied_beta))}]
    if nexcit == 4:
        pspace = [ {'aaa' : np.ones((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha,
                                    system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha)),
                   'aab' : np.ones((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta,
                                    system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta)),
                   'abb' : np.ones((system.nunoccupied_alpha, system.nunoccupied_beta, system.nunoccupied_beta,
                                    system.noccupied_alpha, system.noccupied_beta, system.noccupied_beta)),
                   'bbb' : np.ones((system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta,
                                    system.noccupied_beta, system.noccupied_beta, system.noccupied_beta))},
                   {'aaaa' : np.ones((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha,
                                    system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha)),
                   'aaab' : np.ones((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta,
                                    system.noccupied_alpha, system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta)),
                   'aabb' : np.ones((system.nunoccupied_alpha, system.nunoccupied_alpha, system.nunoccupied_beta, system.nunoccupied_beta,
                                    system.noccupied_alpha, system.noccupied_alpha, system.noccupied_beta, system.noccupied_beta)),
                   'bbbb' : np.ones((system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta, system.nunoccupied_beta,
                                    system.noccupied_beta, system.noccupied_beta, system.noccupied_beta, system.noccupied_beta))} ]
    return pspace

def get_excit_rank(D, D0):
    return len( set(D) - set(D0) )

def get_excits_from(D, D0):
	return list( set(D0) - set(D) )

def get_excits_to(D, D0):
	return list( set(D) - set(D0) )

def spatial_orb_idx(x):
		if x%2 == 1:
			return int( (x+1)/2 )
		else:
			return int( x/2 )

def get_spincase(excits_from, excits_to):
    
    assert(len(excits_from) == len(excits_to))

    num_alpha_occ = sum([i % 2 for i in excits_from])
    num_alpha_unocc = sum([i % 2 for i in excits_to])

    assert(num_alpha_occ == num_alpha_unocc)

    spincase = 'a' * num_alpha_occ + 'b' * (len(excits_from) - num_alpha_occ)

    return spincase

def get_pspace_from_cipsi(pspace_file, system, nexcit=3):

    pspace = get_empty_pspace(system, nexcit)

    HF = list(range(1, system.nelectrons + 1))
    occupied_lower_bound = 1
    occupied_upper_bound = system.noccupied_alpha
    unoccupied_lower_bound = 1
    unoccupied_upper_bound = system.nunoccupied_beta

    orb_table = {'a' : system.noccupied_alpha, 'b' : system.noccupied_beta}

    excitation_count = [{'aaa' : 0, 'aab' : 0, 'abb' : 0, 'bbb' : 0},
                        {'aaaa' : 0, 'aaab' : 0, 'aabb' : 0, 'abbb' : 0, 'bbbb' : 0}]

    with open(pspace_file) as f:
   
        for line in f.readlines():

            det = list( map(int, line.split()[2:]) )

            excit_rank = get_excit_rank(det, HF)

            if excit_rank < 3 or excit_rank > nexcit: continue

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

            n = excit_rank - 3

            excitation_count[n][spincase] += 1

            if excit_rank == 3:
                pspace[n][spincase][idx_unocc[0]-1, idx_unocc[1]-1, idx_unocc[2]-1, idx_occ[0]-1, idx_occ[1]-1, idx_occ[2]-1] = 1
            if excit_rank == 4:
                pspace[n][spincase][idx_unocc[0]-1, idx_unocc[1]-1, idx_unocc[2]-1, idx_unocc[3]-1, idx_occ[0]-1, idx_occ[1]-1, idx_occ[2]-1, idx_occ[3]-1] = 1

    return pspace, excitation_count

def count_excitations_in_pspace(pspace, system):

    excitation_count = [{'aaa' : 0, 'aab' : 0, 'abb' : 0, 'bbb' : 0},
                        {'aaaa' : 0, 'aaab' : 0, 'aabb' : 0, 'abbb' : 0, 'bbbb' : 0}]

    for n, p in enumerate(pspace):

        if n == 0:

            for a in range(system.nunoccupied_alpha):
                for b in range(a + 1, system.nunoccupied_alpha):
                    for c in range(b + 1, system.nunoccupied_alpha):
                        for i in range(system.noccupied_alpha):
                            for j in range(i + 1, system.noccupied_alpha):
                                for k in range(j + 1, system.noccupied_alpha):
                                    if p['aaa'][a, b, c, i, j, k] == 1:
                                        excitation_count[n]['aaa'] += 1
            for a in range(system.nunoccupied_alpha):
                for b in range(a + 1, system.nunoccupied_alpha):
                    for c in range(system.nunoccupied_beta):
                        for i in range(system.noccupied_alpha):
                            for j in range(i + 1, system.noccupied_alpha):
                                for k in range(system.noccupied_beta):
                                    if p['aab'][a, b, c, i, j, k] == 1:
                                        excitation_count[n]['aab'] += 1
            for a in range(system.nunoccupied_alpha):
                for b in range(system.nunoccupied_beta):
                    for c in range(b + 1, system.nunoccupied_beta):
                        for i in range(system.noccupied_alpha):
                            for j in range(system.noccupied_beta):
                                for k in range(j + 1, system.noccupied_beta):
                                    if p['abb'][a, b, c, i, j, k] == 1:
                                        excitation_count[n]['abb'] += 1
            for a in range(system.nunoccupied_beta):
                for b in range(a + 1, system.nunoccupied_beta):
                    for c in range(b + 1, system.nunoccupied_beta):
                        for i in range(system.noccupied_beta):
                            for j in range(i + 1, system.noccupied_beta):
                                for k in range(j + 1, system.noccupied_beta):
                                    if p['bbb'][a, b, c, i, j, k] == 1:
                                        excitation_count[n]['bbb'] += 1

    return excitation_count


def add_spinorbital_triples_to_pspace(triples_list, pspace):
    """Expand the size of the previous P space using the determinants contained in the list
    of triples (stored as a, b, c, i, j, k) in triples_list. The variable triples_list stores
    triples in spinorbital form, where all orbital indices start from 1 and odd indices
    correspond to alpha orbitals while even indices corresopnd to beta orbitals."""

    # should these be copied?
    new_pspace = {
        "aaa": pspace["aaa"],
        "aab": pspace["aab"],
        "abb": pspace["abb"],
        "bbb": pspace["bbb"],
    }

    num_add = triples_list.shape[0]
    for n in range(num_add):
        num_alpha = int(sum([x % 2 for x in triples_list[n, :]]) / 2)
        idx = [spatial_orb_idx(p) - 1 for p in triples_list[n, :]]
        a, b, c, i, j, k = idx
        if num_alpha == 3:
            new_pspace['aaa'][a, b, c, i, j, k] = 1
        elif num_alpha == 2:
            new_pspace['aab'][a, b, c, i, j, k] = 1
        elif num_alpha == 1:
            new_pspace['abb'][a, b, c, i, j, k] = 1
        else:
            new_pspace['bbb'][a, b, c, i, j, k] = 1

    return new_pspace


def add_spinorbital_quadruples_to_pspace(quadruples_list, pspace):
    """Expand the size of the previous P space using the determinants contained in the list
    of quadruples (stored as a, b, c, d, i, j, k, l) in quadruples_list. The variable quadruples_list stores
    quadruples in spinorbital form, where all orbital indices start from 1 and odd indices
    correspond to alpha orbitals while even indices corresopnd to beta orbitals."""

    # should these be copied?
    new_pspace = {
        "aaaa": pspace["aaaa"],
        "aaab": pspace["aaab"],
        "aabb": pspace["aabb"],
        "abbb": pspace["abbb"],
        "bbbb": pspace["bbbb"],
    }

    num_add = quadruples_list.shape[0]
    for n in range(num_add):
        num_alpha = int(sum([x % 2 for x in quadruples_list[n, :]]) / 2)
        idx = [spatial_orb_idx(p) - 1 for p in quadruples_list[n, :]]
        a, b, c, d, i, j, k, l = idx
        if num_alpha == 4:
            new_pspace['aaaa'][a, b, c, d, i, j, k, l] = 1
        elif num_alpha == 3:
            new_pspace['aaab'][a, b, c, d, i, j, k, l] = 1
        elif num_alpha == 2:
            new_pspace['aabb'][a, b, c, d, i, j, k, l] = 1
        elif num_alpha == 1:
            new_pspace['abbb'][a, b, c, d, i, j, k, l] = 1
        else:
            new_pspace['bbbb'][a, b, c, d, i, j, k, l] = 1

    return new_pspace

