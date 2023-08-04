import numpy as np
from ccpy.eom_guess.s2matrix import build_s2matrix_cisd, spin_adapt_guess

def run_diagonalization(system, H, multiplicity, nroot, nacto, nactu, debug=False):

    Hmat = build_cisd_hamiltonian(H, system, nacto, nactu)
    S2mat = build_s2matrix_cisd(system, nacto, nactu)
    omega, V_act = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=debug)

    nroot = min(nroot, V_act.shape[1])
    noa, nob, nua, nub = H.ab.oovv.shape
    ndim = noa*nua + nob*nub + noa**2*nua**2 + noa*nob*nua*nub + nob**2*nub**2
    V = np.zeros((ndim, nroot))
    for i in range(nroot):
        V[:, i] = scatter(V_act[:, i], nacto, nactu, system)

    # print results of initial guess procedure
    print("   Eigenvalues of CISd initial guess")
    print("   Multiplicity = ", multiplicity)
    print("   Dimension of eigenvalue problem = ", V_act.shape[0])
    print("   Active occupied alpha = ", min(nacto + (system.multiplicity - 1), noa))
    print("   Active occupied beta = ", min(nacto, nob))
    print("   Active unoccupied alpha = ", min(nactu, nua))
    print("   Active unoccupied beta = ", min(nactu + (system.multiplicity - 1), nub))
    print("   -----------------------------------")
    for i in range(nroot):
        print("   Eigenvalue of root", i + 1, " = ", omega[i])
    print("")

    return omega[:nroot], V

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
    V_a_out = np.zeros((nua, noa))
    V_b_out = np.zeros((nub, nob))
    V_aa_out = np.zeros((nua, nua, noa, noa))
    V_ab_out = np.zeros((nua, nub, noa, nob))
    V_bb_out = np.zeros((nub, nub, nob, nob))
    # Start filling in the array
    offset = 0
    for a in range(nua):
        for i in range(noa):
            V_a_out[a, i] = V_in[offset]
            offset += 1
    for a in range(nub):
        for i in range(nob):
            V_b_out[a, i] = V_in[offset]
            offset += 1
    for a in range(nua):
        for b in range(a + 1, nua):
            for i in range(noa):
                for j in range(i + 1, noa):
                    if a < nactu_a and b < nactu_a and i >= noa - nacto_a and j >= noa - nacto_a:
                        V_aa_out[a, b, i, j] = V_in[offset]
                        V_aa_out[b, a, i, j] = -V_in[offset]
                        V_aa_out[a, b, j, i] = -V_in[offset]
                        V_aa_out[b, a, j, i] = V_in[offset]
                        offset += 1
    for a in range(nua):
        for b in range(nub):
            for i in range(noa):
                for j in range(nob):
                    if a < nactu_a and b < nactu_b and i >= noa - nacto_a and j >= nob - nacto_b:
                        V_ab_out[a, b, i, j] = V_in[offset]
                        offset += 1
    for a in range(nub):
        for b in range(a + 1, nub):
            for i in range(nob):
                for j in range(i + 1, nob):
                    if a < nactu_b and b < nactu_b and i >= nob - nacto_b and j >= nob - nacto_b:
                        V_bb_out[a, b, i, j] = V_in[offset]
                        V_bb_out[b, a, i, j] = -V_in[offset]
                        V_bb_out[a, b, j, i] = -V_in[offset]
                        V_bb_out[b, a, j, i] = V_in[offset]
                        offset += 1
    return np.hstack((V_a_out.flatten(), V_b_out.flatten(), V_aa_out.flatten(), V_ab_out.flatten(), V_bb_out.flatten()))

def build_cisd_hamiltonian(H, system, nacto, nactu):

    noa, nob, nua, nub = H.ab.oovv.shape

    # set active space parameters
    nacto_a = min(nacto + (system.multiplicity - 1), noa)
    nacto_b = min(nacto, nob)
    nactu_a = min(nactu, nua)
    nactu_b = min(nactu + (system.multiplicity - 1), nub)

    # set dimensions of CISD problem
    n1a = noa * nua
    n1b = nob * nub
    n2a = int(nacto_a * (nacto_a - 1) / 2 * nactu_a * (nactu_a - 1) / 2)
    n2b = nacto_a * nacto_b * nactu_a * nactu_b
    n2c = int(nacto_b * (nacto_b - 1) / 2 * nactu_b * (nactu_b - 1) / 2)
    # total dimension
    ndim = n1a + n1b + n2a + n2b + n2c

    # < ia | H | jb >
    a_H_a = np.zeros((n1a, n1a))
    ct1 = 0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for b in range(system.nunoccupied_alpha):
                for j in range(system.noccupied_alpha):
                    a_H_a[ct1, ct2] = (
                          H.a.vv[a, b] * (i == j)
                        - H.a.oo[j, i] * (a == b)
                        + H.aa.voov[a, j, i, b]
                    )
                    ct2 += 1
            ct1 += 1
    #print("norm of a_H_a = ", np.linalg.norm(a_H_a.flatten()))
    # < ia | H | j~b~ >
    a_H_b = np.zeros((n1a, n1b))
    ct1 = 0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for b in range(system.nunoccupied_beta):
                for j in range(system.noccupied_beta):
                    a_H_b[ct1, ct2] = H.ab.voov[a, j, i, b]
                    ct2 += 1
            ct1 += 1
    #print("norm of a_H_b = ", np.linalg.norm(a_H_b.flatten()))
    # < i~a~ | H | jb >
    b_H_a = np.zeros((n1b, n1a))
    ct1 = 0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            ct2 = 0
            for b in range(system.nunoccupied_alpha):
                for j in range(system.noccupied_alpha):
                    b_H_a[ct1, ct2] = H.ab.ovvo[j, a, b, i]
                    ct2 += 1
            ct1 += 1
    #print("norm of b_H_a = ", np.linalg.norm(b_H_a.flatten()))
    # < i~a~ | H | j~b~ >
    b_H_b = np.zeros((n1b, n1b))
    ct1 = 0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            ct2 = 0
            for b in range(system.nunoccupied_beta):
                for j in range(system.noccupied_beta):
                    b_H_b[ct1, ct2] = (
                          H.b.vv[a, b] * (i == j)
                        - H.b.oo[j, i] * (a == b)
                        + H.bb.voov[a, j, i, b]
                    )
                    ct2 += 1
            ct1 += 1
    #print("norm of b_H_b = ", np.linalg.norm(b_H_b.flatten()))
    # < ia | H | jkbc >
    a_H_aa = np.zeros((n1a, n2a))
    idet = 0
    for a in range(nua):
        for i in range(noa):
            jdet = 0
            for b in range(nactu_a):
                for c in range(b + 1, nactu_a):
                    for j in range(noa - nacto_a, noa):
                        for k in range(j + 1, noa):
                            hmatel = (
                               (i == k) * (a == c) * H.a.ov[j, b]
                              +(i == j) * (a == b) * H.a.ov[k, c]
                              -(i == j) * (a == c) * H.a.ov[k, b]
                              -(i == k) * (a == b) * H.a.ov[j, c]
                              -(a == b) * H.aa.ooov[j, k, i, c]
                              +(a == c) * H.aa.ooov[j, k, i, b]
                              +(i == j) * H.aa.vovv[a, k, b, c]
                              -(i == k) * H.aa.vovv[a, j, b, c]
                            )
                            a_H_aa[idet, jdet] = hmatel
                            jdet += 1
            idet += 1
    #print("norm of a_H_aa = ", np.linalg.norm(a_H_aa.flatten()))
    # < ia | H | jk~bc~ >
    a_H_ab = np.zeros((n1a, n2b))
    idet = 0
    for a in range(nua):
        for i in range(noa):
            jdet = 0
            for b in range(nactu_a):
                for c in range(nactu_b):
                    for j in range(noa - nacto_a, noa):
                        for k in range(nob - nacto_b, nob):
                            hmatel = (
                                (i == j) * H.ab.vovv[a, k, b, c]
                               -(a == b) * H.ab.ooov[j, k, i, c]
                               +(i == j) * (a == b) * H.b.ov[k, c]

                            )
                            a_H_ab[idet, jdet] = hmatel
                            jdet += 1
            idet += 1
    #print("norm of a_H_ab = ", np.linalg.norm(a_H_ab.flatten()))
    # < i~a~ | H | jk~bc~ >
    b_H_ab = np.zeros((n1b, n2b))
    idet = 0
    for a in range(nub):
        for i in range(nob):
            jdet = 0
            for b in range(nactu_a):
                for c in range(nactu_b):
                    for j in range(noa - nacto_a, noa):
                        for k in range(nob - nacto_b, nob):
                            hmatel = (
                                (i == k) * H.ab.ovvv[j, a, b, c]
                               -(a == c) * H.ab.oovo[j, k, b, i]
                               +(i == k) * (a == c) * H.a.ov[j, b]

                            )
                            b_H_ab[idet, jdet] = hmatel
                            jdet += 1
            idet += 1
    #print("norm of b_H_ab = ", np.linalg.norm(b_H_ab.flatten()))
    # < i~a~ | H | j~k~b~c~ >
    b_H_bb = np.zeros((n1b, n2c))
    idet = 0
    for a in range(nub):
        for i in range(nob):
            jdet = 0
            for b in range(nactu_b):
                for c in range(b + 1, nactu_b):
                    for j in range(nob - nacto_b, nob):
                        for k in range(j + 1, nob):
                            hmatel = (
                               (i == k) * (a == c) * H.b.ov[j, b]
                              +(i == j) * (a == b) * H.b.ov[k, c]
                              -(i == j) * (a == c) * H.b.ov[k, b]
                              -(i == k) * (a == b) * H.b.ov[j, c]
                              -(a == b) * H.bb.ooov[j, k, i, c]
                              +(a == c) * H.bb.ooov[j, k, i, b]
                              +(i == j) * H.bb.vovv[a, k, b, c]
                              -(i == k) * H.bb.vovv[a, j, b, c]
                            )
                            b_H_bb[idet, jdet] = hmatel
                            jdet += 1
            idet += 1
    #print("norm of b_H_bb = ", np.linalg.norm(b_H_bb.flatten()))
    # < ijab | H | kc >
    aa_H_a = np.zeros((n2a, n1a))
    idet = 0
    for a in range(nactu_a):
        for b in range(a + 1, nactu_a):
            for i in range(noa - nacto_a, noa):
                for j in range(i + 1, noa):
                    jdet = 0
                    for c in range(nua):
                        for k in range(noa):
                            hmatel = (
                                 (j == k) * H.aa.vvov[a, b, i, c]
                                +(i == k) * H.aa.vvov[b, a, j, c]
                                -(b == c) * H.aa.vooo[a, k, i, j]
                                -(a == c) * H.aa.vooo[b, k, j, i]
                            )
                            aa_H_a[idet, jdet] = hmatel
                            jdet += 1
                    idet += 1
    #print("norm of aa_H_a = ", np.linalg.norm(aa_H_a.flatten()))
    # < ij~ab~ | H | kc >
    ab_H_a = np.zeros((n2b, n1a))
    idet = 0
    for a in range(nactu_a):
        for b in range(nactu_b):
            for i in range(noa - nacto_a, noa):
                for j in range(nob - nacto_b, nob):
                    jdet = 0
                    for c in range(nua):
                        for k in range(noa):
                            hmatel = (
                                 (i == k) * H.ab.vvvo[a, b, c, j]
                                -(a == c) * H.ab.ovoo[k, b, i, j]
                            )
                            ab_H_a[idet, jdet] = hmatel
                            jdet += 1
                    idet += 1
    #print("norm of ab_H_a = ", np.linalg.norm(ab_H_a.flatten()))
    # < ij~ab~ | H | k~c~ >
    ab_H_b = np.zeros((n2b, n1b))
    idet = 0
    for a in range(nactu_a):
        for b in range(nactu_b):
            for i in range(noa - nacto_a, noa):
                for j in range(nob - nacto_b, nob):
                    jdet = 0
                    for c in range(nub):
                        for k in range(nob):
                            hmatel = (
                                 (j == k) * H.ab.vvov[a, b, i, c]
                                -(b == c) * H.ab.vooo[a, k, i, j]
                            )
                            ab_H_b[idet, jdet] = hmatel
                            jdet += 1
                    idet += 1
    #print("norm of ab_H_b = ", np.linalg.norm(ab_H_b.flatten()))
    # < i~j~a~b~ | H | k~c~ >
    bb_H_b = np.zeros((n2c, n1b))
    idet = 0
    for a in range(nactu_b):
        for b in range(a + 1, nactu_b):
            for i in range(nob - nacto_b, nob):
                for j in range(i + 1, nob):
                    jdet = 0
                    for c in range(nub):
                        for k in range(nob):
                            hmatel = (
                                 (j == k) * H.bb.vvov[a, b, i, c]
                                +(i == k) * H.bb.vvov[b, a, j, c]
                                -(b == c) * H.bb.vooo[a, k, i, j]
                                -(a == c) * H.bb.vooo[b, k, j, i]
                            )
                            bb_H_b[idet, jdet] = hmatel
                            jdet += 1
                    idet += 1
    #print("norm of bb_H_b = ", np.linalg.norm(bb_H_b.flatten()))
    # < ijab | H | klcd >
    aa_H_aa = np.zeros((n2a, n2a))
    idet = 0
    for a in range(nactu_a):
        for b in range(a + 1, nactu_a):
            for i in range(noa - nacto_a, noa):
                for j in range(i + 1, noa):
                    jdet = 0
                    for c in range(nactu_a):
                        for d in range(c + 1, nactu_a):
                            for k in range(noa - nacto_a, noa):
                                for l in range(k + 1, noa):
                                    hmatel = (
                                        (a == c) * (b ==d) *
                                        (
                                                -(j == l) * H.a.oo[k, i]
                                                +(i == l) * H.a.oo[k, j]
                                                +(j == k) * H.a.oo[l, i]
                                                -(i == k) * H.a.oo[l, j]
                                        )
                                        + (j == l) * (i == k) *
                                        (
                                                + (b == d) * H.a.vv[a, c]
                                                - (b == c) * H.a.vv[a, d]
                                                + (a == c) * H.a.vv[b, d]
                                                - (a == d) * H.a.vv[b, c]
                                        )
                                        + (i == k) * (a == c) * H.aa.voov[b, l, j, d]
                                        - (i == k) * (a == d) * H.aa.voov[b, l, j, c]
                                        - (i == k) * (b == c) * H.aa.voov[a, l, j, d]
                                        + (i == k) * (b == d) * H.aa.voov[a, l, j, c]
                                        - (i == l) * (a == c) * H.aa.voov[b, k, j, d]
                                        + (i == l) * (a == d) * H.aa.voov[b, k, j, c]
                                        + (i == l) * (b == c) * H.aa.voov[a, k, j, d]
                                        - (i == l) * (b == d) * H.aa.voov[a, k, j, c]
                                        - (j == k) * (a == c) * H.aa.voov[b, l, i, d]
                                        + (j == k) * (a == d) * H.aa.voov[b, l, i, c]
                                        + (j == k) * (b == c) * H.aa.voov[a, l, i, d]
                                        - (j == k) * (b == d) * H.aa.voov[a, l, i, c]
                                        + (j == l) * (a == c) * H.aa.voov[b, k, i, d]
                                        - (j == l) * (a == d) * H.aa.voov[b, k, i, c]
                                        - (j == l) * (b == c) * H.aa.voov[a, k, i, d]
                                        + (j == l) * (b == d) * H.aa.voov[a, k, i, c]
                                        + (b == d) * (a == c) * H.aa.oooo[k, l, i, j]
                                        + (i == k) * (j == l) * H.aa.vvvv[a, b, c, d]
                                    )
                                    aa_H_aa[idet, jdet] = hmatel
                                    jdet += 1
                    idet += 1
    #print("norm of aa_H_aa = ", np.linalg.norm(aa_H_aa.flatten()))
    # < ijab | H | kl~cd~ >
    aa_H_ab = np.zeros((n2a, n2b))
    idet = 0
    for a in range(nactu_a):
        for b in range(a + 1, nactu_a):
            for i in range(noa - nacto_a, noa):
                for j in range(i + 1, noa):
                    jdet = 0
                    for c in range(nactu_a):
                        for d in range(nactu_b):
                            for k in range(noa - nacto_a, noa):
                                for l in range(nob - nacto_b, nob):
                                    hmatel = (
                                         (i == k) * (a == c) * H.ab.voov[b, l, j, d]
                                        -(i == k) * (b == c) * H.ab.voov[a, l, j, d]
                                        -(j == k) * (a == c) * H.ab.voov[b, l, i, d]
                                        +(j == k) * (b == c) * H.ab.voov[a, l, i, d]
                                    )
                                    aa_H_ab[idet, jdet] = hmatel
                                    jdet += 1
                    idet += 1
    #print("norm of aa_H_ab = ", np.linalg.norm(aa_H_ab.flatten()))
    # < ij~ab~ | H | klcd >
    ab_H_aa = np.zeros((n2b, n2a))
    idet = 0
    for a in range(nactu_a):
        for b in range(nactu_b):
            for i in range(noa - nacto_a, noa):
                for j in range(nob - nacto_b, nob):
                    jdet = 0
                    for c in range(nactu_a):
                        for d in range(c + 1, nactu_a):
                            for k in range(noa - nacto_a, noa):
                                for l in range(k + 1, noa):
                                    hmatel = (
                                         (i == k) * (a == c) * H.ab.ovvo[l, b, d, j]
                                        -(i == k) * (a == d) * H.ab.ovvo[l, b, c, j]
                                        -(i == l) * (a == c) * H.ab.ovvo[k, b, d, j]
                                        +(i == l) * (a == d) * H.ab.ovvo[k, b, c, j]
                                    )
                                    ab_H_aa[idet, jdet] = hmatel
                                    jdet += 1
                    idet += 1
    #print("norm of ab_H_aa = ", np.linalg.norm(ab_H_aa.flatten()))
    # < ij~ab~ | H | kl~cd~ >
    ab_H_ab = np.zeros((n2b, n2b))
    idet = 0
    for a in range(nactu_a):
        for b in range(nactu_b):
            for i in range(noa - nacto_a, noa):
                for j in range(nob - nacto_b, nob):
                    jdet = 0
                    for c in range(nactu_a):
                        for d in range(nactu_b):
                            for k in range(noa - nacto_a, noa):
                                for l in range(nob - nacto_b, nob):
                                    hmatel = (
                                         (j == l) * (b == d) * H.aa.voov[a, k, i, c]
                                        +(j == l) * (i == k) * H.ab.vvvv[a, b, c, d]
                                        +(a == c) * (i == k) * H.bb.voov[b, l, j, d]
                                        +(a == c) * (b == d) * H.ab.oooo[k, l, i, j]
                                        -(j == l) * (a == c) * H.ab.ovov[k, b, i, d]
                                        -(i == k) * (b == d) * H.ab.vovo[a, l, c, j]
                                        -(j == l) * (a == c) * (b == d) * H.a.oo[k, i]
                                        -(a == c) * (b == d) * (i == k) * H.b.oo[l, j]
                                        +(i == k) * (b == d) * (j == l) * H.a.vv[a, c]
                                        +(j == l) * (i == k) * (a == c) * H.b.vv[b, d]
                                    )
                                    ab_H_ab[idet, jdet] = hmatel
                                    jdet += 1
                    idet += 1
    #print("norm of ab_H_ab = ", np.linalg.norm(ab_H_ab.flatten()))
    # < ij~ab~ | H | k~l~c~d~ >
    ab_H_bb = np.zeros((n2b, n2c))
    idet = 0
    for a in range(nactu_a):
        for b in range(nactu_b):
            for i in range(noa - nacto_a, noa):
                for j in range(nob - nacto_b, nob):
                    jdet = 0
                    for c in range(nactu_b):
                        for d in range(c + 1, nactu_b):
                            for k in range(nob - nacto_b, nob):
                                for l in range(k + 1, nob):
                                    hmatel = (
                                         (j == k) * (b == c) * H.ab.voov[a, l, i, d]
                                        -(j == k) * (b == d) * H.ab.voov[a, l, i, c]
                                        -(j == l) * (b == c) * H.ab.voov[a, k, i, d]
                                        +(j == l) * (b == d) * H.ab.voov[a, k, i, c]
                                    )
                                    ab_H_bb[idet, jdet] = hmatel
                                    jdet += 1
                    idet += 1
    #print("norm of ab_H_bb = ", np.linalg.norm(ab_H_bb.flatten()))
    # < i~j~a~b~ | H | kl~cd~ >
    bb_H_ab = np.zeros((n2c, n2b))
    idet = 0
    for a in range(nactu_b):
        for b in range(a + 1, nactu_b):
            for i in range(nob - nacto_b, nob):
                for j in range(i + 1, nob):
                    jdet = 0
                    for c in range(nactu_a):
                        for d in range(nactu_b):
                            for k in range(noa - nacto_a, noa):
                                for l in range(nob - nacto_b, nob):
                                    hmatel = (
                                         (i == l) * (a == d) * H.ab.ovvo[k, b, c, j]
                                        -(i == l) * (b == d) * H.ab.ovvo[k, a, c, j]
                                        -(j == l) * (a == d) * H.ab.ovvo[k, b, c, i]
                                        +(j == l) * (b == d) * H.ab.ovvo[k, a, c, i]
                                    )
                                    bb_H_ab[idet, jdet] = hmatel
                                    jdet += 1
                    idet += 1
    #print("norm of bb_H_ab = ", np.linalg.norm(bb_H_ab.flatten()))
    # < i~j~a~b~ | H | k~l~c~d~ >
    bb_H_bb = np.zeros((n2c, n2c))
    idet = 0
    for a in range(nactu_b):
        for b in range(a + 1, nactu_b):
            for i in range(nob - nacto_b, nob):
                for j in range(i + 1, nob):
                    jdet = 0
                    for c in range(nactu_b):
                        for d in range(c + 1, nactu_b):
                            for k in range(nob - nacto_b, nob):
                                for l in range(k + 1, nob):
                                    hmatel = (
                                        (a == c) * (b ==d) * (
                                                -(j == l) * H.b.oo[k, i]
                                                +(i == l) * H.b.oo[k, j]
                                                +(j == k) * H.b.oo[l, i]
                                                -(i == k) * H.b.oo[l, j]
                                            )
                                        + (j == l) * (i == k) * (
                                                + (b == d) * H.b.vv[a, c]
                                                - (b == c) * H.b.vv[a, d]
                                                + (a == c) * H.b.vv[b, d]
                                                - (a == d) * H.b.vv[b, c]
                                            )
                                        + (i == k) * (a == c) * H.bb.voov[b, l, j, d]
                                        - (i == k) * (a == d) * H.bb.voov[b, l, j, c]
                                        - (i == k) * (b == c) * H.bb.voov[a, l, j, d]
                                        + (i == k) * (b == d) * H.bb.voov[a, l, j, c]
                                        - (i == l) * (a == c) * H.bb.voov[b, k, j, d]
                                        + (i == l) * (a == d) * H.bb.voov[b, k, j, c]
                                        + (i == l) * (b == c) * H.bb.voov[a, k, j, d]
                                        - (i == l) * (b == d) * H.bb.voov[a, k, j, c]
                                        - (j == k) * (a == c) * H.bb.voov[b, l, i, d]
                                        + (j == k) * (a == d) * H.bb.voov[b, l, i, c]
                                        + (j == k) * (b == c) * H.bb.voov[a, l, i, d]
                                        - (j == k) * (b == d) * H.bb.voov[a, l, i, c]
                                        + (j == l) * (a == c) * H.bb.voov[b, k, i, d]
                                        - (j == l) * (a == d) * H.bb.voov[b, k, i, c]
                                        - (j == l) * (b == c) * H.bb.voov[a, k, i, d]
                                        + (j == l) * (b == d) * H.bb.voov[a, k, i, c]
                                        + (b == d) * (a == c) * H.bb.oooo[k, l, i, j]
                                        + (i == k) * (j == l) * H.bb.vvvv[a, b, c, d]
                                    )
                                    bb_H_bb[idet, jdet] = hmatel
                                    jdet += 1
                    idet += 1
    #print("norm of bb_H_bb = ", np.linalg.norm(bb_H_bb.flatten()))
    # Zero blocks
    a_H_bb = np.zeros((n1a, n2c))
    b_H_aa = np.zeros((n1b, n2a))
    aa_H_bb = np.zeros((n2a, n2c))
    # Assemble and return full matrix
    return np.concatenate(
        (np.concatenate((a_H_a, a_H_b, a_H_aa, a_H_ab, a_H_bb), axis=1),
         np.concatenate((b_H_a, b_H_b, b_H_aa, b_H_ab, b_H_bb), axis=1),
         np.concatenate((aa_H_a, b_H_aa.T, aa_H_aa, aa_H_ab, aa_H_bb), axis=1),
         np.concatenate((ab_H_a, ab_H_b, ab_H_aa, ab_H_ab, ab_H_bb), axis=1),
         np.concatenate((a_H_bb.T, bb_H_b, aa_H_bb.T, bb_H_ab, bb_H_bb), axis=1)
         ), axis=0
    )
