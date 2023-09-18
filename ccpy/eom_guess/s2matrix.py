import numpy as np
import math

def spin_adapt_guess(S2, H, multiplicity, debug=False):

    def _get_multiplicity(s2):
        s = -0.5 + math.sqrt(0.25 + s2)
        return 2.0 * s + 1.0

    # if multiplicity = -1, then just diagonalize H and return
    if multiplicity == -1:
        omega, V = np.linalg.eig(H)
        idx = np.argsort(omega)
        return np.real(omega[idx]), np.real(V[:, idx])

    ndim = H.shape[0]

    eval_s2, V_s2 = np.linalg.eigh(S2)

    if debug:
        omega_ref, _ = np.linalg.eig(H)
        idx = np.argsort(omega_ref)
        omega_ref = omega_ref[idx]
        for i, s2 in enumerate(eval_s2):
            print("root", i + 1, "s2 = ", s2, "mult = ", _get_multiplicity(s2))
        for i in range(min(len(omega_ref), 50)):
            print("root", i + 1, "E = ", omega_ref[i])

    idx_s2 = [i for i, s2 in enumerate(eval_s2) if abs(_get_multiplicity(s2) - multiplicity) < 1.0e-07]
    n_s2_sub = len(idx_s2)

    W = np.zeros((ndim, n_s2_sub))
    for i in range(n_s2_sub):
        W[:, i] = V_s2[:, idx_s2[i]]

    # Transform into determinantal eigenbasis of S2
    G = np.einsum("Ku,Nv,Lu,Mv,LM->KN", W, W, W, W, H, optimize=True)
    # diagonalize and sort the resulting eigenvalues
    omega, V = np.linalg.eig(G)
    omega = np.real(omega)
    V = np.real(V)
    idx = np.argsort(omega)
    omega = omega[idx]
    V = V[:, idx]

    # now all the eigenvalues that do not have the correct multiplicity are going to be numerically 0
    # retain only those that are non-zero to find the spin-adapted subspace
    # Unfortunately, for SF-CIS, one root will genuinely have 1 excitation energy equal to 0, corresponding to
    # the low-spin triplet ground-state described as a spin-flip excitation out of the high-spin reference
    omega_adapt = np.zeros(n_s2_sub)
    V_adapt = np.zeros((ndim, n_s2_sub))
    n = 0
    for i in range(len(omega)):
        if abs(omega[i]) < 1.0e-09: continue
        omega_adapt[n] = omega[i]
        V_adapt[:, n] = V[:, i]
        n += 1
    return omega_adapt, V_adapt

def get_sz2(system, Ms):
    Ns = float((system.noccupied_alpha + Ms) - (system.noccupied_beta - Ms))
    sz = Ns / 2.0
    sz2 = (sz + 1.0) * sz
    return sz2

def get_sz2_ip(system):
    Ns = float((system.noccupied_alpha - 1) - (system.noccupied_beta))
    sz = Ns / 2.0
    sz2 = (sz + 1.0) * sz
    return sz2

def get_sz2_ea(system):
    Ns = float((system.noccupied_alpha + 1) - (system.noccupied_beta))
    sz = Ns / 2.0
    sz2 = (sz - 1.0) * sz # is this needed? seems wrong actually.
    return sz2

def build_s2matrix_cis(system):

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

    sz2 = get_sz2(system, Ms=0)

    Saa = np.zeros((n1a, n1a))
    ct1 = 0
    for a in range(system.noccupied_alpha, system.noccupied_alpha + system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for b in range(system.noccupied_alpha, system.noccupied_alpha + system.nunoccupied_alpha):
                for j in range(system.noccupied_alpha):
                    Saa[ct1, ct2] = (sz2 + 1.0 * chi_beta(i)) * (i == j) * (a == b)
                    ct2 += 1
            ct1 += 1
    Sab = np.zeros((n1a, n1b))
    ct1 = 0
    for a in range(system.noccupied_alpha, system.noccupied_alpha + system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for b in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
                for j in range(system.noccupied_beta):
                    Sab[ct1, ct2] = -1.0 * (i == j) * (a == b) * chi_beta(i) * pi_alpha(a)
                    ct2 += 1
            ct1 += 1
    Sba = np.zeros((n1b, n1a))
    ct1 = 0
    for a in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            ct2 = 0
            for b in range(system.noccupied_alpha, system.noccupied_alpha + system.nunoccupied_alpha):
                for j in range(system.noccupied_alpha):
                    Sba[ct1, ct2] = -1.0 * (i == j) * (a == b) * chi_beta(i) * pi_alpha(a)
                    ct2 += 1
            ct1 += 1
    Sbb = np.zeros((n1b, n1b))
    ct1 = 0
    for a in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            ct2 = 0
            for b in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
                for j in range(system.noccupied_beta):
                    Sbb[ct1, ct2] = (sz2 + 1.0 * pi_alpha(a)) * (i == j) * (a == b)
                    ct2 += 1
            ct1 += 1

    return np.concatenate(
        (np.concatenate((Saa, Sab), axis=1),
         np.concatenate((Sba, Sbb), axis=1)), axis=0
    )

def build_s2matrix_cisd(system, nacto, nactu):

    # set active space parameters
    nacto_a = min(nacto + (system.multiplicity - 1), system.noccupied_alpha)
    nacto_b = min(nacto, system.noccupied_beta)
    nactu_a = min(nactu, system.nunoccupied_alpha)
    nactu_b = min(nactu + (system.multiplicity - 1), system.nunoccupied_beta)

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
    n2a = int(nactu_a * (nactu_a - 1) / 2 * nacto_a * (nacto_a - 1) / 2)
    n2b = nacto_a * nacto_b * nactu_a * nactu_b
    n2c = int(nactu_b * (nactu_b - 1) / 2 * nacto_b * (nacto_b - 1) / 2)

    sz2 = get_sz2(system, Ms=0)
    #
    a_S_a = np.zeros((n1a, n1a))
    ct1 = 0
    for a in range(system.noccupied_alpha, system.noccupied_alpha + system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for b in range(system.noccupied_alpha, system.noccupied_alpha + system.nunoccupied_alpha):
                for j in range(system.noccupied_alpha):
                    a_S_a[ct1, ct2] = (sz2 + 1.0 * chi_beta(i)) * (i == j) * (a == b)
                    ct2 += 1
            ct1 += 1
    #print(np.linalg.norm(a_S_a.flatten()))
    #
    a_S_b = np.zeros((n1a, n1b))
    ct1 = 0
    for a in range(system.noccupied_alpha, system.noccupied_alpha + system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for b in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
                for j in range(system.noccupied_beta):
                    a_S_b[ct1, ct2] = -1.0 * (i == j) * (a == b)
                    ct2 += 1
            ct1 += 1
    b_S_a = a_S_b.T
    #print(np.linalg.norm(a_S_b.flatten()))
    #
    b_S_b = np.zeros((n1b, n1b))
    ct1 = 0
    for a in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            ct2 = 0
            for b in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
                for j in range(system.noccupied_beta):
                    b_S_b[ct1, ct2] = (sz2 + 1.0 * pi_alpha(a)) * (i == j) * (a == b)
                    ct2 += 1
            ct1 += 1
    #print(np.linalg.norm(b_S_b.flatten()))
    #
    a_S_ab = np.zeros((n1a, n2b))
    ct1 = 0
    for a in range(system.noccupied_alpha, system.noccupied_alpha + system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for B in range(system.noccupied_alpha, system.noccupied_alpha + nactu_a):
                for C in range(system.noccupied_beta, system.noccupied_beta + nactu_b):
                    for J in range(system.noccupied_alpha - nacto_a, system.noccupied_alpha):
                        for K in range(system.noccupied_beta - nacto_b, system.noccupied_beta):
                            # + d(a,B) <i|K~> <J|C~>
                            a_S_ab[ct1, ct2] += (a == B) * (i == K) * (J == C)

                            ct2 += 1
            ct1 += 1
    ab_S_a = a_S_ab.T
    #print(np.linalg.norm(a_S_ab.flatten()))
    #
    b_S_ab = np.zeros((n1b, n2b))
    ct1 = 0
    for a in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            ct2 = 0
            for B in range(system.noccupied_alpha, system.noccupied_alpha + nactu_a):
                for C in range(system.noccupied_beta, system.noccupied_beta + nactu_b):
                    for J in range(system.noccupied_alpha - nacto_a, system.noccupied_alpha):
                        for K in range(system.noccupied_beta - nacto_b, system.noccupied_beta):
                            # - d(i,K) <J|C~> <a~|B>
                            b_S_ab[ct1, ct2] -= (a == B) * (i == K) * (J == C)
                            ct2 += 1
            ct1 += 1
    ab_S_b = b_S_ab.T
    #print(np.linalg.norm(b_S_ab.flatten()))
    #
    aa_S_aa = np.zeros((n2a, n2a))
    ct1 = 0
    for A in range(system.noccupied_alpha, system.noccupied_alpha + nactu_a):
        for B in range(A + 1, system.noccupied_alpha + nactu_a):
            for I in range(system.noccupied_alpha - nacto_a, system.noccupied_alpha):
                for J in range(I + 1, system.noccupied_alpha):
                    ct2 = 0
                    for C in range(system.noccupied_alpha, system.noccupied_alpha + nactu_a):
                        for D in range(C + 1, system.noccupied_alpha + nactu_a):
                            for K in range(system.noccupied_alpha - nacto_a, system.noccupied_alpha):
                                for L in range(K + 1, system.noccupied_alpha):
                                    # +A(IJ)A(KL) d(A,C) d(J,L) d(B,D) d(I,K) chi_alpha(I)
                                    aa_S_aa[ct1, ct2] += chi_beta(I) * (I == K) * (A == C) * (J == L) * (B == D) # (1)
                                    aa_S_aa[ct1, ct2] -= chi_beta(J) * (J == K) * (A == C) * (I == L) * (B == D) # (IJ)
                                    aa_S_aa[ct1, ct2] -= chi_beta(I) * (I == L) * (A == C) * (J == K) * (B == D) # (KL)
                                    aa_S_aa[ct1, ct2] += chi_beta(J) * (J == L) * (A == C) * (I == K) * (B == D) # (IJ)(KL)

                                    aa_S_aa[ct1, ct2] += sz2 * (I == K) * (A == C) * (J == L) * (B == D)
                                    ct2 += 1
                    ct1 += 1
    #print(np.linalg.norm(aa_S_aa.flatten()))
    #
    aa_S_ab = np.zeros((n2a, n2b))
    ct1 = 0
    for A in range(system.noccupied_alpha, system.noccupied_alpha + nactu_a):
        for B in range(A + 1, system.noccupied_alpha + nactu_a):
            for I in range(system.noccupied_alpha - nacto_a, system.noccupied_alpha):
                for J in range(I + 1, system.noccupied_alpha):
                    ct2 = 0
                    for C in range(system.noccupied_alpha, system.noccupied_alpha + nactu_a):
                        for D in range(system.noccupied_beta, system.noccupied_beta + nactu_b):
                            for K in range(system.noccupied_alpha - nacto_a, system.noccupied_alpha):
                                for L in range(system.noccupied_beta - nacto_b, system.noccupied_beta):
                                    # -A(IJ)A(AB) d(I,K) d(A,C) <L~|J> <B|D~>
                                    aa_S_ab[ct1, ct2] -= (I == K) * (A == C) * (L == J) * (B == D) # (1)
                                    aa_S_ab[ct1, ct2] += (J == K) * (A == C) * (L == I) * (B == D) # (IJ)
                                    aa_S_ab[ct1, ct2] += (I == K) * (B == C) * (L == J) * (A == D) # (AB)
                                    aa_S_ab[ct1, ct2] -= (J == K) * (B == C) * (L == I) * (A == D) # (IJ)(AB)

                                    ct2 += 1
                    ct1 += 1
    ab_S_aa = aa_S_ab.T
    #print(np.linalg.norm(aa_S_ab.flatten()))
    #
    ab_S_ab = np.zeros((n2b, n2b))
    ct1 = 0
    for A in range(system.noccupied_alpha, system.noccupied_alpha + nactu_a):
        for B in range(system.noccupied_beta, system.noccupied_beta + nactu_b):
            for I in range(system.noccupied_alpha - nacto_a, system.noccupied_alpha):
                for J in range(system.noccupied_beta - nacto_b, system.noccupied_beta):
                    ct2 = 0
                    for C in range(system.noccupied_alpha, system.noccupied_alpha + nactu_a):
                        for D in range(system.noccupied_beta, system.noccupied_beta + nactu_b):
                            for K in range(system.noccupied_alpha - nacto_a, system.noccupied_alpha):
                                for L in range(system.noccupied_beta - nacto_b, system.noccupied_beta):
                                    # -d(A,C) d(B~,D~) <L~|I> <K|J~>
                                    ab_S_ab[ct1, ct2] -= (A == C) * (B == D) * (L == I) * (K == J)
                                    # -d(I,K) d(J~,L~) <A|D~> <B~|C>
                                    ab_S_ab[ct1, ct2] -= (I == K) * (J == L) * (A == D) * (B == C)
                                    # d(J~,L~) d(A,C) <K|D~> <B~|I>
                                    ab_S_ab[ct1, ct2] += (J == L) * (A == C) * (K == D) * (B == I)
                                    # +d(J,L) d(A,C) d(I,K) d(B,D) pi_alpha(B)
                                    ab_S_ab[ct1, ct2] += (J == L) * (I == K) * (A == C) * (B == D) * pi_alpha(B)
                                    # +d(I,K) d(J,L) d(B,D) d(A,C) chi_beta(I)
                                    ab_S_ab[ct1, ct2] += (A == C) * (I == K) * (J == L) * (B == D) * chi_beta(I)

                                    ab_S_ab[ct1, ct2] += sz2 * (I == K) * (J == L) * (A == C) * (B == D)
                                    ct2 += 1
                    ct1 += 1
    #print(np.linalg.norm(ab_S_ab.flatten()))
    #
    ab_S_bb = np.zeros((n2b, n2c))
    ct1 = 0
    for A in range(system.noccupied_alpha, system.noccupied_alpha + nactu_a):
        for B in range(system.noccupied_beta, system.noccupied_beta + nactu_b):
            for I in range(system.noccupied_alpha - nacto_a, system.noccupied_alpha):
                for J in range(system.noccupied_beta - nacto_b, system.noccupied_beta):
                    ct2 = 0
                    for C in range(system.noccupied_beta, system.noccupied_beta + nactu_b):
                        for D in range(C + 1, system.noccupied_beta + nactu_b):
                            for K in range(system.noccupied_beta - nacto_b, system.noccupied_beta):
                                for L in range(K + 1, system.noccupied_beta):
                                    # -A(K~L~)A(C~D~) d(J~,L~) d(B~,D~) <K~|I> <A|C~>
                                    ab_S_bb[ct1, ct2] -= (J == L) * (B == D) * (K == I) * (A == C) # (1)
                                    ab_S_bb[ct1, ct2] += (J == K) * (B == D) * (L == I) * (A == C) # (KL)
                                    ab_S_bb[ct1, ct2] += (J == L) * (B == C) * (K == I) * (A == D) # (CD)
                                    ab_S_bb[ct1, ct2] -= (J == K) * (B == C) * (L == I) * (A == D) # (KL)(CD)

                                    ct2 += 1
                    ct1 += 1
    bb_S_ab = ab_S_bb.T
    #print(np.linalg.norm(ab_S_bb.flatten()))
    #
    bb_S_bb = np.zeros((n2c, n2c))
    ct1 = 0
    for A in range(system.noccupied_beta, system.noccupied_beta + nactu_b):
        for B in range(A + 1, system.noccupied_beta + nactu_b):
            for I in range(system.noccupied_beta - nacto_b, system.noccupied_beta):
                for J in range(I + 1, system.noccupied_beta):
                    ct2 = 0
                    for C in range(system.noccupied_beta, system.noccupied_beta + nactu_b):
                        for D in range(C + 1, system.noccupied_beta + nactu_b):
                            for K in range(system.noccupied_beta - nacto_b, system.noccupied_beta):
                                for L in range(K + 1, system.noccupied_beta):
                                    # A(AB)A(CD) d(I,K) d(J,L) d(B,D) d(A,C) pi_alpha(A)
                                    bb_S_bb[ct1, ct2] += pi_alpha(A) * (I == K) * (A == C) * (J == L) * (B == D) # (1)
                                    bb_S_bb[ct1, ct2] -= pi_alpha(B) * (I == K) * (B == C) * (J == L) * (A == D) # (AB)
                                    bb_S_bb[ct1, ct2] -= pi_alpha(A) * (I == K) * (A == D) * (J == L) * (B == C) # (CD)
                                    bb_S_bb[ct1, ct2] += pi_alpha(B) * (I == K) * (B == D) * (J == L) * (A == C) # (AB)(CD)

                                    bb_S_bb[ct1, ct2] += sz2 * (I == K) * (A == C) * (J == L) * (B == D)
                                    ct2 += 1
                    ct1 += 1
    #print(np.linalg.norm(bb_S_bb.flatten()))

    # Lot of zero blocks
    a_S_aa = np.zeros((n1a, n2a))
    aa_S_a = a_S_aa.T
    b_S_aa = np.zeros((n1b, n2a))
    aa_S_b = b_S_aa.T
    b_S_bb = np.zeros((n1b, n2c))
    bb_S_b = b_S_bb.T
    a_S_bb = np.zeros((n1a, n2c))
    bb_S_a = a_S_bb.T
    aa_S_bb = np.zeros((n2a, n2c))
    bb_S_aa = aa_S_bb.T

    return np.concatenate(
        (np.concatenate((a_S_a, a_S_b, a_S_aa, a_S_ab, a_S_bb), axis=1),
         np.concatenate((b_S_a, b_S_b, b_S_aa, b_S_ab, b_S_bb), axis=1),
         np.concatenate((aa_S_a, aa_S_b, aa_S_aa, aa_S_ab, aa_S_bb), axis=1),
         np.concatenate((ab_S_a, ab_S_b, ab_S_aa, ab_S_ab, ab_S_bb), axis=1),
         np.concatenate((bb_S_a, bb_S_b, bb_S_aa, bb_S_ab, bb_S_bb), axis=1)
         ), axis=0
    )

def build_s2matrix_sfcis(system, Ms):

    n1b = system.nunoccupied_beta * system.noccupied_alpha
    sz2 = get_sz2(system, Ms)
    Sbb = np.zeros((n1b, n1b))
    ct1 = 0
    for a in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for b in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
                for j in range(system.noccupied_alpha):
                    Sbb[ct1, ct2] += sz2 * (a == b) * (i == j)
                    Sbb[ct1, ct2] += (a == i) * (b == j)
                    ct2 += 1
            ct1 += 1
    return Sbb

def build_s2matrix_1h(system):

    def chi_beta(p):
        if p >= 0 and p < system.noccupied_beta:
            return 1.0
        else:
            return 0.0

    sz2 = get_sz2_ip(system) # this needs to be modified potentially
    Sa = np.zeros((system.noccupied_alpha, system.noccupied_alpha))
    ct1 = 0
    for i in range(system.noccupied_alpha):
        ct2 = 0
        for j in range(system.noccupied_alpha):
            Sa[ct1, ct2] += sz2 * (i == j)
            Sa[ct1, ct2] += chi_beta(i) * (i == j)
            ct2 += 1
        ct1 += 1
    return Sa

def build_s2matrix_1p(system):

    def pi_alpha(p):
        if p >= system.noccupied_alpha and p < system.nunoccupied_alpha + system.noccupied_alpha:
            return 1.0
        else:
            return 0.0

    sz2 = get_sz2_ea(system) # this needs to be modified potentially
    Sa = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha))
    ct1 = 0
    for a in range(system.noccupied_alpha, system.noccupied_alpha + system.nunoccupied_alpha):
        ct2 = 0
        for b in range(system.noccupied_alpha, system.noccupied_alpha + system.nunoccupied_alpha):
            Sa[ct1, ct2] += sz2 * (a == b)
            Sa[ct1, ct2] += pi_alpha(a) * (a == b)
            ct2 += 1
        ct1 += 1
    return Sa

def build_s2matrix_2p(system, nactu):

    def pi_alpha(p):
        if p >= system.noccupied_alpha and p < system.nunoccupied_alpha + system.noccupied_alpha:
            return 1.0
        else:
            return 0.0

    n2b = nactu**2
    sz2 = get_sz2(system, Ms=0) # this needs to be modified potentially
    Sab = np.zeros((n2b, n2b))
    ct1 = 0
    for a in range(system.noccupied_alpha, system.noccupied_alpha + nactu):
        for b in range(system.noccupied_beta, system.noccupied_beta + nactu):
            ct2 = 0
            for c in range(system.noccupied_alpha, system.noccupied_alpha + nactu):
                for d in range(system.noccupied_beta, system.noccupied_beta + nactu):
                    Sab[ct1, ct2] += (sz2 + 1.0 * pi_alpha(a)) * (a == c) * (b == d)
                    Sab[ct1, ct2] -= (b == c) * (a == d) # why is this a minus sign??
                    ct2 += 1
            ct1 += 1
    return Sab
