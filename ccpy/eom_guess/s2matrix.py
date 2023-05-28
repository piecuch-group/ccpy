import numpy as np

def spin_adapt_guess(S2, H, multiplicity):

    def _get_multiplicity(s2):
        s = -0.5 + np.sqrt(0.25 + s2)
        return 2.0 * s + 1.0

    ndim = H.shape[0]

    eval_s2, V_s2 = np.linalg.eig(S2)
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

    Ns = float( (system.noccupied_alpha + Ms) - (system.noccupied_beta - Ms))
    sz = Ns / 2.0
    sz2 = (sz + 1.0) * sz

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

def build_s2matrix_cisd(system):

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
    n2a = system.nunoccupied_alpha**2 * system.noccupied_alpha**2
    n2b = system.nunoccupied_alpha * system.nunoccupied_beta * system.noccupied_alpha * system.noccupied_beta
    n2c = system.nunoccupied_beta**2 * system.noccupied_beta**2

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

    # Saaa = np.zeros((n1a, n2a))
    # Sbaa = np.zeros((n1b, n2a))
    #
    # Sbbb = np.zeros((n1b, n2c))
    # Sabb = np.zeros((n1a, n2c))
    #
    # Saab = np.zeros((n1a, n2b))
    # ct1 = 0
    # for a in range(system.noccupied_alpha, system.noccupied_alpha + system.nunoccupied_alpha):
    #     for i in range(system.noccupied_alpha):
    #         ct2 = 0
    #         for b in range(system.noccupied_alpha, system.noccupied_alpha + system.nunoccupied_alpha):
    #             for c in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
    #                 for j in range(system.noccupied_alpha):
    #                     for k in range(system.noccupied_beta):
    #                         Saab[ct1, ct2] = -1.0 * (a == b) * (i == k) * (c == j)
    # Sbab = np.zeros((n1b, n2b))
    #
    # Saaaa = np.zeros((n2a, n2a))
    # Saaab = np.zeros((n2a, n2b))
    # Saabb = np.zeros((n2a, n2c))
    #
    # Sabaa = np.zeros((n2b, n2a))
    # Sabab = np.zeros((n2b, n2b))
    # Sabbb = np.zeros((n2b, n2c))
    #
    # Sbbaa = np.zeros((n2c, n2a))
    # Sbbab = np.zeros((n2c, n2b))
    # Sbbbb = np.zeros((n2c, n2c))



    return np.concatenate(
        (np.concatenate((Saa, Sab), axis=1),
         np.concatenate((Sba, Sbb), axis=1)), axis=0
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

def build_s2matrix_2p(system):

    n2b = system.nunoccupied_beta * system.nunoccupied_alpha
    sz2 = get_sz2(system) # this needs to be modified
    Sab = np.zeros((n2b, n2b))
    ct1 = 0
    for a in range(system.noccupied_alpha, system.noccupied_alpha + system.nunoccupied_alpha):
        for b in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
            ct2 = 0
            for c in range(system.noccupied_alpha, system.noccupied_alpha + system.nunoccupied_alpha):
                for d in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
                    Sab[ct1, ct2] += sz2 * (a == c) * (b == d)
                    ct2 += 1
            ct1 += 1
    return Sab