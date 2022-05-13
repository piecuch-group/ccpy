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
    spincase_idx_triples = {'aaa' : 0, 'aab' : 1, 'abb' : 2, 'bbb' : 3}

    excitation_count = [0 for i in range(nexcit-2)]

    triples_count_spincase = [0, 0, 0, 0]

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
    
            excitation_count[n] += 1
            triples_count_spincase[spincase_idx_triples[spincase]] += 1

            if excit_rank == 3:
                pspace[n][spincase][idx_unocc[0]-1, idx_unocc[1]-1, idx_unocc[2]-1, idx_occ[0]-1, idx_occ[1]-1, idx_occ[2]-1] = 1
            if excit_rank == 4:
                pspace[n][spincase][idx_unocc[0]-1, idx_unocc[1]-1, idx_unocc[2]-1, idx_unocc[3]-1, idx_occ[0]-1, idx_occ[1]-1, idx_occ[2]-1, idx_occ[3]-1] = 1

    return pspace, excitation_count, triples_count_spincase
