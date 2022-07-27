import numpy as np

def get_dimensions(noa, nob, norb):

    nua = norb - noa
    nub = norb - nob

    n1a = 0
    for a in range(nua):
        for i in range(noa):
            n1a += 1
    n1b = 0
    for a in range(nub):
        for i in range(nob):
            n1b += 1
    n2a = 0
    for a in range(nua):
        for b in range(a + 1, nua):
            for i in range(noa):
                for j in range(i + 1, noa):
                    n2a += 1
    n2b = 0
    for a in range(nua):
        for b in range(nub):
            for i in range(noa):
                for j in range(nob):
                    n2b += 1
    n2c = 0
    for a in range(nub):
        for b in range(a + 1, nub):
            for i in range(nob):
                for j in range(i + 1, nob):
                    n2c += 1

    return n1a, n1b, n2a, n2b, n2c

def get_sz2(nocc_a, nocc_b):

    Ns = float(nocc_a - nocc_b)
    sz2 = (Ns / 2.0 + 1.0) * (Ns / 2.0)

    return sz2

def build_s2matrix_cis(nocc_a, nocc_b, norb):

    def chi_beta(p):
        if p >= 0 and p < norb - nocc_b:
            return 1.0
        else:
            return 0.0

    def pi_alpha(p):
        if p >= nocc_a and p < norb:
            return 1.0
        else:
            return 0.0

    n1a, n1b, n2a, n2b, n2c = get_dimensions(nocc_a, nocc_b, norb)

    sz2 = get_sz2(nocc_a, nocc_b)

    Saa = np.zeros((n1a, n1a))
    ct1 = 0
    for a in range(nocc_a, norb):
        for i in range(nocc_a):
            ct2 = 0
            for b in range(nocc_a, norb):
                for j in range(nocc_a):
                    Saa[ct1, ct2] = (sz2 + 1.0 * chi_beta(i)) * (i == j) * (a == b)
                    ct2 += 1
            ct1 += 1
    Sab = np.zeros((n1a, n1b))
    ct1 = 0
    for a in range(nocc_a, norb):
        for i in range(nocc_a):
            ct2 = 0
            for b in range(nocc_b, norb):
                for j in range(nocc_b):
                    Sab[ct1, ct2] = -1.0 * (i == j) * (a == b)
                    ct2 += 1
            ct1 += 1
    Sba = np.zeros((n1b, n1a))
    ct1 = 0
    for a in range(nocc_b, norb):
        for i in range(nocc_b):
            ct2 = 0
            for b in range(nocc_a, norb):
                for j in range(nocc_a):
                    Sba[ct1, ct2] = -1.0 * (i == j) * (a == b)
                    ct2 += 1
            ct1 += 1
    Sbb = np.zeros((n1b, n1b))
    ct1 = 0
    for a in range(nocc_b, norb):
        for i in range(nocc_b):
            ct2 = 0
            for b in range(nocc_b, norb):
                for j in range(nocc_b):
                    Sbb[ct1, ct2] = (sz2 + 1.0 * pi_alpha(a)) * (i == j) * (a == b)
                    ct2 += 1
            ct1 += 1

    return np.concatenate(
        (np.concatenate((Saa, Sab), axis=1),
         np.concatenate((Sba, Sbb), axis=1)), axis=0
    )

def build_s2matrix_cisd(nocc_a, nocc_b, norb):

    def chi_beta(p):
        if p >= 0 and p < norb - nocc_b:
            return 1.0
        else:
            return 0.0

    def pi_alpha(p):
        if p >= nocc_a and p < norb:
            return 1.0
        else:
            return 0.0

    n1a = nocc_a * (norb - nocc_a)
    n1b = nocc_b * (norb - nocc_b)
    n2a = nocc_a **2 * (norb - nocc_a)**2
    n2b = nocc_a * nocc_b * (norb - nocc_a) * (norb - nocc_b)
    n2c = nocc_b**2 * (norb - nocc_b)**2

    sz2 = get_sz2(nocc_a, nocc_b)

    Saa = np.zeros((n1a, n1a))
    ct1 = 0
    for a in range(nocc_a, norb):
        for i in range(nocc_a):
            ct2 = 0
            for b in range(nocc_a, norb):
                for j in range(nocc_a):
                    Saa[ct1, ct2] = (sz2 + 1.0 * chi_beta(i)) * (i == j) * (a == b)
                    ct2 += 1
            ct1 += 1
    Sab = np.zeros((n1a, n1b))
    ct1 = 0
    for a in range(nocc_a, norb):
        for i in range(nocc_a):
            ct2 = 0
            for b in range(nocc_b, norb):
                for j in range(nocc_b):
                    Sab[ct1, ct2] = -1.0 * (i == j) * (a == b)
                    ct2 += 1
            ct1 += 1
    Sba = np.zeros((n1b, n1a))
    ct1 = 0
    for a in range(nocc_b, norb):
        for i in range(nocc_b):
            ct2 = 0
            for b in range(nocc_a, norb):
                for j in range(nocc_a):
                    Sba[ct1, ct2] = -1.0 * (i == j) * (a == b)
                    ct2 += 1
            ct1 += 1
    Sbb = np.zeros((n1b, n1b))
    ct1 = 0
    for a in range(nocc_b, norb):
        for i in range(nocc_b):
            ct2 = 0
            for b in range(nocc_b, norb):
                for j in range(nocc_b):
                    Sbb[ct1, ct2] = (sz2 + 1.0 * pi_alpha(a)) * (i == j) * (a == b)
                    ct2 += 1
            ct1 += 1
    Saab = np.zeros((n1a, n2b))
    ct1 = 0
    for a in range(nocc_a, norb):
        for i in range(nocc_a):
            ct2 = 0
            for b in range(nocc_a, norb):
                for c in range(nocc_b, norb):
                    for j in range(nocc_a):
                        for k in range(nocc_b):
                            Saab[ct1, ct2] = (
                                            (a == b) * (i == k) * (j == c)
                                          - (i == j) * (a == c) * (k == b)
                            )
                            ct2 += 1
    Sbab = np.zeros((n1b, n2b))
    ct1 = 0
    for a in range(nocc_b, norb):
        for i in range(nocc_b):
            ct2 = 0
            for b in range(nocc_a, norb):
                for c in range(nocc_b, norb):
                    for j in range(nocc_a):
                        for k in range(nocc_b):
                            Sbab[ct1, ct2] = (
                                            (a == c) * (i == j) * (k == b)
                                          - (i == k) * (a == b) * (j == c)
                            )
                            ct2 += 1
    Saaa = np.zeros((n1a, n2a))
    Sabb = np.zeros((n1a, n2c))
    Sbaa = np.zeros((n1b, n2a))
    Sbbb = np.zeros((n1b, n2c))

    Saaaa = np.zeros((n2a, n2a))
    ct1 = 0
    for a in range(nocc_a, norb):
        for b in range(a + 1, norb):
            for i in range(nocc_a):
                for j in range(i + 1, nocc_a):
                    ct2 = 0
                    for c in range(nocc_a, norb):
                        for d in range(c + 1, norb):
                            for k in range(nocc_a):
                                for l in range(k + 1, nocc_a):
                                    Saaaa[ct1, ct2] = (sz2 + 1.0 * chi_beta(i) + 1.0 * chi_beta(j))


    return np.concatenate(
        (np.concatenate((Saa, Sab), axis=1),
         np.concatenate((Sba, Sbb), axis=1)), axis=0
    )

if __name__ == "__main__":

    multiplicity = 2

    S2 = build_s2matrix_cis(nocc_a = 5, nocc_b = 4, norb = 30)

    ndim = S2.shape[0]

    eval_s2, V_s2 = np.linalg.eig(S2)
    idx = np.argsort(eval_s2)
    eval_s2 = eval_s2[idx]
    V_s2 = V_s2[:, idx]

    Ns2 = 0
    idx_s2 = []
    print("S2 Eigenvalues:")
    for i in range(ndim):
        sval = -0.5 + np.sqrt(0.25 + eval_s2[i])
        multval = 2 * sval + 1
        print("S{} = {}  (mult = {})".format(i + 1, sval, multval))
        if abs(multval - multiplicity) < 1.0e-07:
            idx_s2.append(i)
            Ns2 += 1

    print("Dimension of spin subspace of multiplicity {} = {}".format(multiplicity, Ns2))
    Qs2 = ndim - Ns2
    W = np.zeros((ndim, Ns2))
    for i in range(Ns2):
        W[:, i] = V_s2[:, idx_s2[i]]
