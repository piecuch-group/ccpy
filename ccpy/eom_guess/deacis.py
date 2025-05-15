'''
DEA-EOMCCSD(2p) Guess Routine for DEA-EOMCC
'''

import time
import numpy as np
from ccpy.eom_guess.s2matrix import spin_adapt_guess

def run_diagonalization(system, H, multiplicity, roots_per_irrep, nacto, nactu, debug=False, use_symmetry=False):

    nroots_total = 0
    for key, value in roots_per_irrep.items():
        nroots_total += value

    noa, nob, nua, nub = H.ab.oovv.shape
    ndim = nua*nub
    V = np.zeros((ndim, nroots_total))
    omega_guess = np.zeros(nroots_total)
    n_found = 0

    # print results of initial guess procedure
    print("   DEA-CIS initial guess routine")
    print("   --------------------------")
    print("   Multiplicity = ", multiplicity)
    print("   Active occupied alpha = ", min(nacto + (system.multiplicity - 1), system.noccupied_alpha))
    print("   Active occupied beta = ", min(nacto, system.noccupied_beta))
    print("   Active unoccupied alpha = ", min(nactu, system.nunoccupied_alpha))
    print("   Active unoccupied beta = ", min(nactu + (system.multiplicity - 1), system.nunoccupied_beta))

    for irrep, nroot in roots_per_irrep.items():
        if nroot == 0: continue
        if not use_symmetry: irrep = None

        # Build the indexing arrays for the given irrep
        idx_ab, ndim_irrep = get_index_arrays(nacto, nactu, system, irrep)
        t1 = time.time()
        # Compute the active-space 2p Hamiltonian
        Hmat = build_2p_hamiltonian(H, nactu, system, idx_ab)
        # Compute the S2 matrix in the same projection subspace
        S2mat = build_s2matrix_2p(system, nactu, idx_ab)
        # Project H onto the spin subspace with the specified multiplicity
        omega, V_act = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=debug)
        nroot = min(nroot, V_act.shape[1])
        kout = 0
        for i in range(len(omega)):
            if omega[i] == 0.0: continue
            V[:, n_found] = scatter(V_act[:, i], nactu, system)
            omega_guess[n_found] = omega[i]
            n_found += 1
            kout += 1
            if kout == nroot:
                break

        elapsed_time = time.time() - t1
        print("   -----------------------------------")
        print("   Target symmetry irrep = ", irrep, f"({system.point_group})")
        print("   Dimension of eigenvalue problem = ", ndim_irrep)
        print("   Elapsed time = ", np.round(elapsed_time, 2), "seconds")
        for i in range(n_found - kout, n_found):
            print("   Eigenvalue of root", i + 1, " = ", np.round(omega_guess[i], 8))
    print("")
    return omega_guess, V

def scatter(V_in, nactu, system):

    V_out = np.zeros(system.nunoccupied_alpha * system.nunoccupied_beta)

    ct = 0
    ct2 = 0
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            if a < nactu and b < nactu:
                V_out[ct] = V_in[ct2]
                ct2 += 1
            ct += 1
    return V_out

def build_2p_hamiltonian(H, nactu, system, idx_ab):

    noa = system.noccupied_alpha
    nob = system.noccupied_beta
    nua = system.nunoccupied_alpha
    nub = system.nunoccupied_beta

    # nacto_a = min(nacto + (system.multiplicity - 1), noa)
    # nacto_b = min(nacto, nob)
    nactu_a = min(nactu, nua)
    nactu_b = min(nactu + (system.multiplicity - 1), nub)

    n2b = nactu_a * nactu_b

    Hab = np.zeros((n2b, n2b))
    for a in range(nactu_a):
        for b in range(nactu_b):
            idet = idx_ab[a, b]
            if idet == 0: continue
            ind1 = abs(idet) - 1
            for c in range(nactu_a):
                for d in range(nactu_b):
                    jdet = idx_ab[c, d]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        Hab[ind1, ind2] = (
                              H.b.vv[b, d] * (a == c)
                            + H.a.vv[a, c] * (b == d)
                            + H.ab.vvvv[a, b, c, d]
                        )
    return Hab

def get_sz2(system, Ms):
    Ns = float((system.noccupied_alpha + Ms) - (system.noccupied_beta - Ms))
    sz = Ns / 2.0
    sz2 = (sz + 1.0) * sz
    return sz2

def build_s2matrix_2p(system, nactu, idx_ab):

    def pi_alpha(p):
        if p >= system.noccupied_alpha and p < system.nunoccupied_alpha + system.noccupied_alpha:
            return 1.0
        else:
            return 0.0

    noa = system.noccupied_alpha
    nob = system.noccupied_beta
    nua = system.nunoccupied_alpha
    nub = system.nunoccupied_beta

    # nacto_a = min(nacto + (system.multiplicity - 1), noa)
    # nacto_b = min(nacto, nob)
    nactu_a = min(nactu, nua)
    nactu_b = min(nactu + (system.multiplicity - 1), nub)

    n2b = nactu_a * nactu_b
    sz2 = get_sz2(system, Ms=0) # this needs to be modified potentially
    Sab = np.zeros((n2b, n2b))
    for a in range(system.noccupied_alpha, system.noccupied_alpha + nactu_a):
        for b in range(system.noccupied_beta, system.noccupied_beta + nactu_b):
            idet = idx_ab[a - noa, b - nob]
            if idet == 0: continue
            ind1 = abs(idet) - 1
            for c in range(system.noccupied_alpha, system.noccupied_alpha + nactu_a):
                for d in range(system.noccupied_beta, system.noccupied_beta + nactu_b):
                    jdet = idx_ab[c - noa, d - nob]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        Sab[ind1, ind2] += (sz2 + 1.0 * pi_alpha(a)) * (a == c) * (b == d)
                        Sab[ind1, ind2] -= (b == c) * (a == d) # why is this a minus sign, you ask... "ghost loop" rule
    return Sab

def get_index_arrays(nacto, nactu, system, target_irrep):

    noa = system.noccupied_alpha
    nua = system.nunoccupied_alpha
    nob = system.noccupied_beta
    nub = system.nunoccupied_beta

    # set active space parameters
    # nacto_a = min(nacto + (system.multiplicity - 1), noa)
    # nacto_b = min(nacto, nob)
    nactu_a = min(nactu, nua)
    nactu_b = min(nactu + (system.multiplicity - 1), nub)

    if target_irrep is None:
        sym1 = lambda a, b: True
    else:
        sym = lambda orbital_number: system.point_group_irrep_to_number[system.orbital_symmetries[orbital_number]]
        ref_sym = system.point_group_irrep_to_number[system.reference_symmetry]
        target_sym = system.point_group_irrep_to_number[target_irrep]

        sym1 = lambda a, b: sym(a + noa) ^ sym(b + nob) ^ ref_sym == target_sym

    ndim = 0
    idx_ab = np.zeros((nua, nub), dtype=np.int32)
    ct = 1
    for a in range(nactu_a):
        for b in range(nactu_b):
            if sym1(a, b):
                idx_ab[a, b] = ct
                ndim += 1
            ct += 1
    return idx_ab, ndim