import numpy as np

def get_sz2(system):

    Ns = float(system.noccupied_alpha - system.noccupied_beta)
    sz2 = (Ns / 2.0 + 1.0) * (Ns / 2.0)

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

    sz2 = get_sz2(system)

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

    sz2 = get_sz2(system)

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