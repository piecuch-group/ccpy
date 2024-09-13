import numpy as np
import time
from ccpy.eom_guess.s2matrix import spin_adapt_guess

def run_diagonalization(system, H, multiplicity, roots_per_irrep, nacto, nactu, use_symmetry=True, debug=False):

    nroots_total = 0
    for key, value in roots_per_irrep.items():
        nroots_total += value

    noa, nob, nua, nub = H.ab.oovv.shape
    ndim = noa + noa**2*nua + noa*nub*nob
    V = np.zeros((ndim, nroots_total))
    omega_guess = np.zeros(nroots_total)
    n_found = 0

    # print results of initial guess procedure
    print("   2h-1p initial guess routine")
    print("   --------------------------")
    print("   Multiplicity = ", multiplicity)
    print("   Active occupied alpha = ", min(nacto + (system.multiplicity - 1), noa))
    print("   Active occupied beta = ", min(nacto, nob))
    print("   Active unoccupied alpha = ", min(nactu, nua))
    print("   Active unoccupied beta = ", min(nactu + (system.multiplicity - 1), nub))

    for irrep, nroot in roots_per_irrep.items():
        if nroot == 0: continue
        if not use_symmetry: irrep = None

        # Build the indexing arrays for the given irrep
        idx_a, idx_aa, idx_ab, ndim_irrep = get_index_arrays(nacto, nactu, system, irrep)
        t1 = time.time()
        # Compute the active-space 2p-1h Hamiltonian
        Hmat = build_ipcisd_hamiltonian(H, nacto, nactu, idx_a, idx_aa, idx_ab, system)
        # Compute the S2 matrix in the same projection subspace
        S2mat = build_s2matrix(system, nacto, nactu, idx_a, idx_aa, idx_ab)
        # Project H onto the spin subspace with the specified multiplicity
        omega, V_act = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=True)
        nroot = min(nroot, V_act.shape[1])
        kout = 0
        for i in range(len(omega)):
            if omega[i] == 0.0: continue
            V[:, n_found] = scatter(V_act[:, i], nacto, nactu, system)
            omega_guess[n_found] = omega[i]
            n_found += 1
            kout += 1
            if kout == nroot:
                break
        # omega, V_act = np.linalg.eig(Hmat)
        # idx = np.argsort(omega)
        # omega = omega[idx]
        # V_act = V_act[:, idx]
        #
        # nroot = min(nroot, V_act.shape[1])
        # kout = 0
        # for i in range(len(omega)):
        #     if omega[i] == 0.0: continue
        #     V2, guess_mult = scatter_with_spin(V_act[:, i], nacto, nactu, system)
        #     print("I AM IN:", omega[i], "GUESS MULT:", guess_mult)
        #     if guess_mult == multiplicity or multiplicity == -1:
        #         V[:, n_found] = V2
        #         omega_guess[n_found] = omega[i]
        #         n_found += 1
        #         kout += 1
        #     if kout == nroot:
        #         break
        elapsed_time = time.time() - t1

        print("   -----------------------------------")
        print("   Target symmetry irrep = ", irrep, f"({system.point_group})")
        print("   Dimension of eigenvalue problem = ", ndim_irrep)
        print("   Elapsed time = ", np.round(elapsed_time, 2), "seconds")
        for i in range(n_found - kout, n_found):
            print("   Eigenvalue of root", i + 1, " = ", np.round(omega_guess[i], 8))
    print("")
    return omega_guess, V

def scatter(V_in, nacto, nactu, system):
    # orbital dimensions
    noa = system.noccupied_alpha
    nob = system.noccupied_beta
    nua = system.nunoccupied_alpha
    nub = system.nunoccupied_beta

    # set active space parameters
    nacto_a = min(nacto + (system.multiplicity - 1), noa)
    nacto_b = min(nacto, nob)
    nactu_a = min(nactu, nua)
    nactu_b = min(nactu + (system.multiplicity - 1), nub)

    # allocate full-length output vector
    V_a_out = np.zeros(noa)
    V_aa_out = np.zeros((noa, nua, noa))
    V_ab_out = np.zeros((noa, nub, nob))
    # Start filling in the array
    offset = 0
    for i in range(noa):
        V_a_out[i] = V_in[offset]
        offset += 1
    for i in range(noa):
        for b in range(nua):
            for j in range(i + 1, noa):
                if i >= noa - nacto_a and b < nactu_a and j >= noa - nacto_a:
                    V_aa_out[i, b, j] = V_in[offset]
                    V_aa_out[j, b, i] = -V_in[offset]
                    offset += 1
    for i in range(noa):
        for b in range(nub):
            for j in range(nob):
                if i >= noa - nacto_a and b < nactu_b and j >= nob - nacto_b:
                    V_ab_out[i, b, j] = V_in[offset]
                    offset += 1
    return np.hstack((V_a_out.flatten(), V_aa_out.flatten(), V_ab_out.flatten()))

def scatter_with_spin(V_in, nacto, nactu, system):
    # orbital dimensions
    noa = system.noccupied_alpha
    nob = system.noccupied_beta
    nua = system.nunoccupied_alpha
    nub = system.nunoccupied_beta

    # set active space parameters
    nacto_a = min(nacto + (system.multiplicity - 1), noa)
    nacto_b = min(nacto, nob)
    nactu_a = min(nactu, nua)
    nactu_b = min(nactu + (system.multiplicity - 1), nub)

    # allocate full-length output vector
    V_a_out = np.zeros(noa)
    V_aa_out = np.zeros((noa, nua, noa))
    V_ab_out = np.zeros((noa, nub, nob))
    # amplitude sums for spin assignment
    sum_aa = 0.0
    sum_ab = 0.0
    # Start filling in the array
    offset = 0
    for i in range(noa):
        V_a_out[i] = V_in[offset]
        offset += 1
    for i in range(noa):
        for b in range(nua):
            for j in range(i + 1, noa):
                if i >= noa - nacto_a and b < nactu_a and j >= noa - nacto_a:
                    V_aa_out[i, b, j] = V_in[offset]
                    V_aa_out[j, b, i] = -V_in[offset]
                    sum_aa += V_aa_out[i, b, j]
                    offset += 1
    for i in range(noa):
        for b in range(nub):
            for j in range(nob):
                if i >= noa - nacto_a and b < nactu_b and j >= nob - nacto_b:
                    V_ab_out[i, b, j] = V_in[offset]
                    if i < j:
                        sum_ab += V_ab_out[i, b, j]
                    offset += 1
    # Assign multiplicity in a cheap way
    guess_mult = -1
    if np.max(np.abs(V_a_out)) > 1.0e-08: # if there is ANY R1 amplitude, it is a doublet
        guess_mult = 2
    else: # either a doublet or quartet
        print(sum_aa, sum_ab)
        print("")
        if abs(sum_aa) - abs(sum_ab) < 1.0e-08: # r2a(i,b,j) + r2b(i,b,j) = 0 for i /= j signifies quartet
            guess_mult = 4
        # elif abs(2.0 * sum_ab) - abs(sum_aa) < 1.0e-08: # 2*r2b(i,b,j) - r2a(i,b,j) = 0 for i /= j signifies doublet
        #     guess_mult = 2
        else:
            guess_mult = 2
    return np.hstack((V_a_out.flatten(), V_aa_out.flatten(), V_ab_out.flatten())), guess_mult

def build_ipcisd_hamiltonian(H, nacto, nactu, idx_a, idx_aa, idx_ab, system):
    # orbital dimensions
    noa, nob, nua, nub = H.ab.oovv.shape
    # set active space parameters
    nacto_a = min(nacto + (system.multiplicity - 1), noa)
    nacto_b = min(nacto, nob)
    nactu_a = min(nactu, nua)
    nactu_b = min(nactu + (system.multiplicity - 1), nub)
    # set dimensions of 2h-1p problem
    n1a = noa
    n2a = int(nactu_a * nacto_a * (nacto_a - 1) / 2)
    n2b = nacto_b * nacto_a * nactu_b
    ####################################################################################
    # ALPHA 1h
    ####################################################################################
    a_H_a = np.zeros((n1a, n1a))
    a_H_aa = np.zeros((n1a, n2a))
    a_H_ab = np.zeros((n1a, n2b))
    for i in range(noa):
        idet = idx_a[i]
        if idet == 0: continue
        ind1 = abs(idet) - 1
        # -h1a(mi) * r_a(m)
        for m in range(noa):
            jdet = idx_a[m]
            if jdet != 0:
                ind2 = abs(jdet) - 1
                a_H_a[ind1, ind2] -= H.a.oo[m, i]
        # -1/2 h2a(mnif) * r_aa(mfn)
        for m in range(noa - nacto_a, noa):
            for f in range(nactu_a):
                for n in range(m + 1, noa):
                    jdet = idx_aa[m, f, n]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        a_H_aa[ind1, ind2] -= H.aa.ooov[m, n, i, f]
        # -h2b(mnif) * r_ab(mfn)
        for m in range(noa - nacto_a, noa):
            for f in range(nactu_b):
                for n in range(nob - nacto_b, nob):
                    jdet = idx_ab[m, f, n]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        a_H_ab[ind1, ind2] -= H.ab.ooov[m, n, i, f]
        # h1a(me) * r_aa(iem)
        for e in range(nactu_a):
            for m in range(noa - nacto_a, noa):
                jdet = idx_aa[i, e, m]
                if jdet != 0:
                    ind2 = abs(jdet) - 1
                    phase = np.sign(jdet)
                    a_H_aa[ind1, ind2] += H.a.ov[m, e] * phase
        # h1b(me) * r_ab(iem)
        for e in range(nactu_b):
            for m in range(nob - nacto_b, nob):
                jdet = idx_ab[i, e, m]
                if jdet != 0:
                    ind2 = abs(jdet) - 1
                    a_H_ab[ind1, ind2] += H.b.ov[m, e]
    ####################################################################################
    # ALPHA-ALPHA 2h-1p
    ####################################################################################
    aa_H_a = np.zeros((n2a, n1a))
    aa_H_aa = np.zeros((n2a, n2a))
    aa_H_ab = np.zeros((n2a, n2b))
    for i in range(noa - nacto_a, noa):
        for b in range(nactu_a):
            for j in range(i + 1, noa):
                idet = idx_aa[i, b, j]
                if idet == 0: continue
                ind1 = abs(idet) - 1
                # -h2a(bmji) * r_a(m)
                for m in range(noa):
                    jdet = idx_a[m]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        aa_H_a[ind1, ind2] -= H.aa.vooo[b, m, j, i]
                # h1a(be) * r_aa(iej)
                for e in range(nactu_a):
                    jdet = idx_aa[i, e, j]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        phase = np.sign(jdet)
                        aa_H_aa[ind1, ind2] += H.a.vv[b, e] * phase
                # 1/2 h2a(mnij) * r_aa(mbn)
                for m in range(noa - nacto_a, noa):
                    for n in range(m + 1, noa):
                        jdet = idx_aa[m, b, n]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            phase = np.sign(jdet)
                            aa_H_aa[ind1, ind2] += H.aa.oooo[m, n, i, j] * phase
                # -A(ij) h1a(mi) * r_aa(mbj)
                for m in range(noa - nacto_a, noa):
                    jdet = idx_aa[m, b, j]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        phase = np.sign(jdet)
                        aa_H_aa[ind1, ind2] -= H.a.oo[m, i] * phase
                    jdet = idx_aa[m, b, i]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        phase = np.sign(jdet)
                        aa_H_aa[ind1, ind2] += H.a.oo[m, j] * phase
                # A(ij) h2a(bmje) * r_aa(iem)
                for e in range(nactu_a):
                    for m in range(noa - nacto_a, noa):
                        jdet = idx_aa[i, e, m]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            phase = np.sign(jdet)
                            aa_H_aa[ind1, ind2] += H.aa.voov[b, m, j, e] * phase
                        jdet = idx_aa[j, e, m]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            phase = np.sign(jdet)
                            aa_H_aa[ind1, ind2] -= H.aa.voov[b, m, i, e] * phase
                # A(ij) h2b(bmje) * r_ab(iem)
                for e in range(nactu_b):
                    for m in range(nob - nacto_b, nob):
                        jdet = idx_ab[i, e, m]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            aa_H_ab[ind1, ind2] += H.ab.voov[b, m, j, e]
                        jdet = idx_ab[j, e, m]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            aa_H_ab[ind1, ind2] -= H.ab.voov[b, m, i, e]
    ####################################################################################
    # ALPHA-BETA 2h-1p
    ####################################################################################
    ab_H_a = np.zeros((n2b, n1a))
    ab_H_aa = np.zeros((n2b, n2a))
    ab_H_ab = np.zeros((n2b, n2b))
    for i in range(noa - nacto_a, noa):
        for b in range(nactu_b):
            for j in range(nob - nacto_b, nob):
                idet = idx_ab[i, b, j]
                if idet == 0: continue
                ind1 = abs(idet) - 1
                # -h2b(mbij) * r_a(m)
                for m in range(noa):
                    jdet = idx_a[m]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        ab_H_a[ind1, ind2] -= H.ab.ovoo[m, b, i, j]
                # -h1a(mi) * r_ab(mbj)
                for m in range(noa - nacto_a, noa):
                    jdet = idx_ab[m, b, j]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        ab_H_ab[ind1, ind2] -= H.a.oo[m, i]
                # -h1b(mj) * r_ab(ibm)
                for m in range(nob - nacto_b, nob):
                    jdet = idx_ab[i, b, m]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        ab_H_ab[ind1, ind2] -= H.b.oo[m, j]
                # h1b(be) * r_ab(iej)
                for e in range(nactu_b):
                    jdet = idx_ab[i, e, j]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        ab_H_ab[ind1, ind2] += H.b.vv[b, e]
                # h2b(mbej) * r_aa(iem)
                for e in range(nactu_a):
                    for m in range(noa - nacto_a, noa):
                        jdet = idx_aa[i, e, m]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            phase = np.sign(jdet)
                            ab_H_aa[ind1, ind2] += H.ab.ovvo[m, b, e, j] * phase
                # h2c(bmje) * r_ab(iem)
                for e in range(nactu_b):
                    for m in range(nob - nacto_b, nob):
                        jdet = idx_ab[i, e, m]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            ab_H_ab[ind1, ind2] += H.bb.voov[b, m, j, e]
                # -h2b(mbie) * r_ab(mej)
                for e in range(nactu_b):
                    for m in range(noa - nacto_a, noa):
                        jdet = idx_ab[m, e, j]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            ab_H_ab[ind1, ind2] -= H.ab.ovov[m, b, i, e]
                # h2b(mnij) * r_ab(mbn)
                for m in range(noa - nacto_a, noa):
                    for n in range(nob - nacto_b, nob):
                        jdet = idx_ab[m, b, n]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            ab_H_ab[ind1, ind2] += H.ab.oooo[m, n, i, j]
    # Assemble and return full matrix
    return np.concatenate(
        (np.concatenate((a_H_a, a_H_aa, a_H_ab), axis=1),
         np.concatenate((aa_H_a, aa_H_aa, aa_H_ab), axis=1),
         np.concatenate((ab_H_a, ab_H_aa, ab_H_ab), axis=1),
         ), axis=0
    )

def get_sz2(system, Ms):
    # Ns = float((system.noccupied_alpha + Ms) - (system.noccupied_beta - Ms))
    # sz = Ns / 2.0
    # sz2 = (sz + 1.0) * sz
    Ns = float((system.noccupied_alpha - 1) - (system.noccupied_beta))
    sz = Ns / 2.0
    sz2 = (sz + 1.0) * sz
    return sz2

def build_s2matrix(system, nacto, nactu, idx_a, idx_aa, idx_ab):

    # orbital dimensions
    noa = system.noccupied_alpha
    nua = system.nunoccupied_alpha
    nob = system.noccupied_beta
    nub = system.nunoccupied_beta

    # set active space parameters
    nacto_a = min(nacto + (system.multiplicity - 1), system.noccupied_alpha)
    nacto_b = min(nacto, system.noccupied_beta)
    nactu_a = min(nactu, system.nunoccupied_alpha)
    nactu_b = min(nactu + (system.multiplicity - 1), system.nunoccupied_beta)

    #
    n1a = noa
    n2a = int(noa * (noa - 1)/2 * nua)
    n2b = noa * nob * nub

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

    sz2 = get_sz2(system, Ms=0)

    # < i | S2 | j >
    a_S_a = np.zeros((n1a, n1a))
    for i in range(noa):
        idet = idx_a[i]
        if idet == 0: continue
        ind1 = abs(idet) - 1
        for j in range(noa):
            jdet = idx_a[j]
            if jdet != 0:
                ind2 = abs(jdet) - 1
                # < Ia | S2 | Ja > = Sz*(Sz + 1)
                a_S_a[ind1, ind2] = 0.75 * (i == j)

    # < i | S2 | jck >
    a_S_aa = np.zeros((n1a, n2a))

    # < i | S2 | jc~k~ >
    a_S_ab = np.zeros((n1a, n2b))

    aa_S_aa = np.zeros((n2a, n2a))
    aa_S_ab = np.zeros((n2a, n2b))
    for i in range(noa - nacto_a, noa):
        for j in range(i + 1, noa):
            for b in range(nacto_a):
                idet = idx_aa[i, b, j]
                if idet == 0: continue
                ind1 = abs(idet) - 1
                # < ibj | S2 | kdl >
                aa_S_aa[ind1, ind1] = 0.75
                # < ibj | S2 | kd~l~ >
                for k in range(noa - nacto_a, noa):
                    for l in range(nob - nacto_b, nob):
                        for d in range(nacto_b):
                            jdet = idx_ab[i, b, j]
                            if jdet != 0:
                                ind2 = abs(jdet) - 1
                                aa_S_ab[ind1, ind2] = (
                                                        (j == l) * (b == d) * (i == k)
                                                        - (i == l) * (b == d) * (j == k)
                                )

    # < ib~j~ | S2 | kd~l~ >
    ab_S_ab = np.zeros((n2b, n2b))
    for i in range(noa - nacto_a, noa):
        for j in range(nob - nacto_b, nob):
            for b in range(nacto_b):
                idet = idx_ab[i, b, j]
                if idet == 0: continue
                ind1 = abs(idet) - 1

                ab_S_ab[ind1, ind1] = 0.75

                for k in range(noa - nacto_a, noa):
                    for l in range(nob - nacto_b, nob):
                        for d in range(nacto_b):
                            jdet = idx_ab[i, b, j]
                            if jdet != 0:
                                ind2 = abs(jdet) - 1
                                ab_S_ab[ind1, ind2] = (i == l) * (j == k) * (b == d)

    S2mat = np.concatenate(
        (np.concatenate((a_S_a, a_S_aa, a_S_ab), axis=1),
         np.concatenate((a_S_aa.T, aa_S_aa, aa_S_ab), axis=1),
         np.concatenate((a_S_ab.T, aa_S_ab.T, ab_S_ab), axis=1),
         ), axis=0)
    return S2mat

def get_index_arrays(nacto, nactu, system, target_irrep):

    noa = system.noccupied_alpha
    nua = system.nunoccupied_alpha
    nob = system.noccupied_beta
    nub = system.nunoccupied_beta

    # set active space parameters
    nacto_a = min(nacto + (system.multiplicity - 1), noa)
    nacto_b = min(nacto, nob)
    nactu_a = min(nactu, nua)
    nactu_b = min(nactu + (system.multiplicity - 1), nub)

    if target_irrep is None:
        sym1 = lambda i: True
        sym2 = lambda i, b, j: True
    else:
        sym = lambda orbital_number: system.point_group_irrep_to_number[system.orbital_symmetries[orbital_number]]
        ref_sym = system.point_group_irrep_to_number[system.reference_symmetry]
        target_sym = system.point_group_irrep_to_number[target_irrep]

        sym1 = lambda i: sym(i) ^ ref_sym == target_sym
        sym2 = lambda i, b, j: sym(i) ^ sym(j) ^ sym(b) ^ ref_sym == target_sym

    ndim = 0
    idx_a = np.zeros(noa, dtype=np.int32)
    ct = 1
    for i in range(noa):
        if sym1(i):
            idx_a[i] = ct
            ndim += 1
        ct += 1
    idx_aa = np.zeros((noa, nua, noa), dtype=np.int32)
    ct = 1
    for i in range(noa - nacto_a, noa):
        for b in range(nactu_a):
            for j in range(i + 1, noa):
                if sym2(i, b + noa, j):
                    idx_aa[i, b, j] = ct
                    idx_aa[j, b, i] = -ct
                    ndim += 1
                ct += 1
    idx_ab = np.zeros((noa, nub, nob), dtype=np.int32)
    ct = 1
    for i in range(noa - nacto_a, noa):
        for b in range(nactu_b):
            for j in range(nob - nacto_b, nob):
                if sym2(i, b + nob, j):
                    idx_ab[i, b, j] = ct
                    ndim += 1
                ct += 1
    return idx_a, idx_aa, idx_ab, ndim
