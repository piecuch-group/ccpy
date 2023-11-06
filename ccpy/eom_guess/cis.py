import numpy as np
import time
from ccpy.eom_guess.s2matrix import spin_adapt_guess

def run_diagonalization(system, H, multiplicity, roots_per_irrep, nacto, nactu, use_symmetry=True, debug=False):

    nroots_total = 0
    for key, value in roots_per_irrep.items():
        nroots_total += value

    noa, nob, nua, nub = H.ab.oovv.shape
    n1 = noa * nua + nob * nub
    ndim = noa * nua + nob * nub + noa ** 2 * nua ** 2 + noa * nob * nua * nub + nob ** 2 * nub ** 2
    V = np.zeros((ndim, nroots_total))
    omega_guess = np.zeros(nroots_total)
    n_found = 0

    # print results of initial guess procedure
    print("   CIS initial guess routine")
    print("   --------------------------")
    print("   Multiplicity = ", multiplicity)

    for irrep, nroot in roots_per_irrep.items():
        if nroot == 0: continue
        if not use_symmetry: irrep = None

        # Build the indexing arrays for the given irrep
        idx_a, idx_b, ndim_irrep = get_index_arrays(system, irrep)
        t1 = time.perf_counter()
        # Compute the CISd-like Hamiltonian
        Hmat = build_cis_hamiltonian(H, idx_a, idx_b, system)
        # Compute the S2 matrix in the same projection subspace
        S2mat = build_s2matrix(system, idx_a, idx_b)
        # Project H onto the spin subspace with the specified multiplicity
        omega, V_act = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=debug)

        nroot = min(nroot, V_act.shape[1])
        kout = 0
        for i in range(len(omega)):
            if omega[i] == 0.0: continue
            V[:n1, n_found] = V_act[:, i]
            omega_guess[n_found] = omega[i]
            n_found += 1
            kout += 1
            if kout == nroot:
                break

        elapsed_time = time.perf_counter() - t1
        print("   -----------------------------------")
        print("   Target symmetry irrep = ", irrep, f"({system.point_group})")
        print("   Dimension of eigenvalue problem = ", ndim_irrep)
        print("   Elapsed time = ", np.round(elapsed_time, 2), "seconds")
        for i in range(n_found - kout, n_found):
            print("   Eigenvalue of root", i + 1, " = ", np.round(omega_guess[i], 8))

    return omega_guess, V

def build_cis_hamiltonian(H, idx_a, idx_b, system):

    noa, nob, nua, nub = H.ab.oovv.shape

    # set dimensions of CIS problem
    n1a = noa * nua
    n1b = nob * nub

    ####################################################################################
    # ALPHA SINGLES
    ####################################################################################
    a_H_a = np.zeros((n1a, n1a))
    a_H_b = np.zeros((n1a, n1b))
    for a in range(nua):
        for i in range(noa):
            idet = idx_a[a, i]
            if idet == 0: continue
            I = abs(idet) - 1
            # -h1a(mi) * r1a(am)
            for m in range(noa):
                jdet = idx_a[a, m]
                if jdet == 0: continue
                J = abs(jdet) - 1
                a_H_a[I, J] -= H.a.oo[m, i]
            # h1a(ae) * r1a(ei)
            for e in range(nua):
                jdet = idx_a[e, i]
                if jdet == 0: continue
                J = abs(jdet) - 1
                a_H_a[I, J] += H.a.vv[a, e]
            # h2a(amie) * r1a(em)
            for e in range(nua):
                for m in range(noa):
                    jdet = idx_a[e, m]
                    if jdet == 0: continue
                    J = abs(jdet) - 1
                    a_H_a[I, J] += H.aa.voov[a, m, i, e]
            # h2b(amie) * r1b(em)
            for e in range(nub):
                for m in range(nob):
                    jdet = idx_b[e, m]
                    if jdet == 0: continue
                    J = abs(jdet) - 1
                    a_H_b[I, J] += H.ab.voov[a, m, i, e]
    ####################################################################################
    # BETA SINGLES
    ####################################################################################
    b_H_a = np.zeros((n1b, n1a))
    b_H_b = np.zeros((n1b, n1b))
    for a in range(nub):
        for i in range(nob):
            idet = idx_b[a, i]
            if idet == 0: continue
            I = abs(idet) - 1
            # -h1b(mi) * r1b(am)
            for m in range(nob):
                jdet = idx_b[a, m]
                if jdet == 0: continue
                J = abs(jdet) - 1
                b_H_b[I, J] -= H.b.oo[m, i]
            # h1b(ae) * r1b(ei)
            for e in range(nub):
                jdet = idx_b[e, i]
                if jdet == 0: continue
                J = abs(jdet) - 1
                b_H_b[I, J] += H.b.vv[a, e]
            # h2c(amie) * r1b(em)
            for e in range(nub):
                for m in range(nob):
                    jdet = idx_b[e, m]
                    if jdet == 0: continue
                    J = abs(jdet) - 1
                    b_H_b[I, J] += H.bb.voov[a, m, i, e]
            # h2b(maei) * r1a(em)
            for e in range(nua):
                for m in range(noa):
                    jdet = idx_a[e, m]
                    if jdet == 0: continue
                    J = abs(jdet) - 1
                    b_H_a[I, J] += H.ab.ovvo[m, a, e, i]
    # Assemble and return full matrix
    return np.concatenate(
        (np.concatenate((a_H_a, a_H_b), axis=1),
         np.concatenate((b_H_a, b_H_b), axis=1),
         ), axis=0
    )

# def build_cis_hamiltonian(H, system, target_irrep):
#
#     n1a = system.noccupied_alpha * system.nunoccupied_alpha
#     n1b = system.noccupied_beta * system.nunoccupied_beta
#     noa, nob, nua, nub = H.ab.oovv.shape
#
#     if target_irrep is None:
#         sym1 = lambda a, i: True
#     else:
#         sym = lambda orbital_number: system.point_group_irrep_to_number[system.orbital_symmetries[orbital_number]]
#         ref_sym = system.point_group_irrep_to_number[system.reference_symmetry]
#         target_sym = system.point_group_irrep_to_number[target_irrep]
#         sym1 = lambda a, i: sym(i) ^ sym(a) ^ ref_sym == target_sym
#
#     Haa = np.zeros((n1a, n1a))
#     ct1 = 0
#     for a in range(system.nunoccupied_alpha):
#         for i in range(system.noccupied_alpha):
#             if sym1(a + noa, i):
#                 ct2 = 0
#                 for b in range(system.nunoccupied_alpha):
#                     for j in range(system.noccupied_alpha):
#                         if sym1(b + noa, j):
#                             Haa[ct1, ct2] = (
#                                   H.a.vv[a, b] * (i == j)
#                                 - H.a.oo[j, i] * (a == b)
#                                 + H.aa.voov[a, j, i, b]
#                             )
#                         ct2 += 1
#             ct1 += 1
#     Hab = np.zeros((n1a, n1b))
#     ct1 = 0
#     for a in range(system.nunoccupied_alpha):
#         for i in range(system.noccupied_alpha):
#             if sym1(a + noa, i):
#                 ct2 = 0
#                 for b in range(system.nunoccupied_beta):
#                     for j in range(system.noccupied_beta):
#                         if sym1(b + nob, j):
#                             Hab[ct1, ct2] = H.ab.voov[a, j, i, b]
#                         ct2 += 1
#             ct1 += 1
#     Hba = np.zeros((n1b, n1a))
#     ct1 = 0
#     for a in range(system.nunoccupied_beta):
#         for i in range(system.noccupied_beta):
#             if sym1(a + nob, i):
#                 ct2 = 0
#                 for b in range(system.nunoccupied_alpha):
#                     for j in range(system.noccupied_alpha):
#                         if sym1(b + noa, j):
#                             Hba[ct1, ct2] = H.ab.ovvo[j, a, b, i]
#                         ct2 += 1
#             ct1 += 1
#     Hbb = np.zeros((n1b, n1b))
#     ct1 = 0
#     for a in range(system.nunoccupied_beta):
#         for i in range(system.noccupied_beta):
#             if sym1(a + nob, i):
#                 ct2 = 0
#                 for b in range(system.nunoccupied_beta):
#                     for j in range(system.noccupied_beta):
#                         if sym1(b + nob, j):
#                             Hbb[ct1, ct2] = (
#                                 H.b.vv[a, b] * (i == j)
#                                 - H.b.oo[j, i] * (a == b)
#                                 + H.bb.voov[a, j, i, b]
#                             )
#                         ct2 += 1
#             ct1 += 1
#     return np.concatenate(
#         (np.concatenate((Haa, Hab), axis=1), np.concatenate((Hba, Hbb), axis=1)), axis=0
#     )

def get_sz2(system, Ms):
    Ns = float((system.noccupied_alpha + Ms) - (system.noccupied_beta - Ms))
    sz = Ns / 2.0
    sz2 = (sz + 1.0) * sz
    return sz2

def build_s2matrix(system, idx_a, idx_b):

    # orbital dimensions
    noa = system.noccupied_alpha
    nua = system.nunoccupied_alpha
    nob = system.noccupied_beta
    nub = system.nunoccupied_beta

    def chi_beta(p):
        if p >= 0 and p < system.noccupied_beta:
            return 1.0
        else:
            return 0.0

    def pi_alpha(p):
        if p >= system.noccupied_alpha and p < system.nunoccupied_alpha + system.noccupied_alpha:
            return 1.0
        else:
            return 0.0

    n1a = system.nunoccupied_alpha * system.noccupied_alpha
    n1b = system.nunoccupied_beta * system.noccupied_beta
    #
    nsocc = noa - nob
    sz2 = get_sz2(system, Ms=0)
    #
    a_S_a = np.zeros((n1a, n1a))
    a_S_b = np.zeros((n1a, n1b))
    for a in range(nua):
        for i in range(noa):
            idet = idx_a[a, i]
            if idet == 0: continue
            I = abs(idet) - 1
            # < ia | S2 | ia >
            a_S_a[I, I] += (sz2 + 1.0 * chi_beta(i))
            # < ia | S2 | j~b~ > = -delta(i,j~) * delta(a,b~)
            if chi_beta(i):
                jdet = idx_b[a + nsocc, i]
                if jdet != 0:
                    J = abs(jdet) - 1
                    a_S_b[I, J] -= 1.0
    #
    b_S_b = np.zeros((n1b, n1b))
    for a in range(nub):
        for i in range(nob):
            idet = idx_b[a, i]
            if idet == 0: continue
            I = abs(idet) - 1
            # < i~a~ | S2 | i~a~ >
            b_S_b[I, I] += (sz2 + 1.0 * pi_alpha(a + nob))

    S2mat = np.concatenate(
        (np.concatenate((a_S_a, a_S_b), axis=1),
         np.concatenate((a_S_b.T, b_S_b), axis=1),
         ), axis=0)
    return S2mat

def get_index_arrays(system, target_irrep):

    noa = system.noccupied_alpha
    nua = system.nunoccupied_alpha
    nob = system.noccupied_beta
    nub = system.nunoccupied_beta

    if target_irrep is None:
        sym1 = lambda a, i: True
    else:
        sym = lambda orbital_number: system.point_group_irrep_to_number[system.orbital_symmetries[orbital_number]]
        ref_sym = system.point_group_irrep_to_number[system.reference_symmetry]
        target_sym = system.point_group_irrep_to_number[target_irrep]
        sym1 = lambda a, i: sym(i) ^ sym(a) ^ ref_sym == target_sym

    ndim = 0
    idx_a = np.zeros((nua, noa), dtype=np.int32)
    ct = 1
    for a in range(nua):
        for i in range(noa):
            if sym1(a + noa, i):
                idx_a[a, i] = ct
                ndim += 1
            ct += 1
    idx_b = np.zeros((nub, nob), dtype=np.int32)
    ct = 1
    for a in range(nub):
        for i in range(nob):
            if sym1(a + nob, i):
                idx_b[a, i] = ct
                ndim += 1
            ct += 1

    return idx_a, idx_b, ndim
