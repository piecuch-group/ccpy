import numpy as np

from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver

from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

def get_index_arrays(system):

    idx1A = np.zeros((system.nunoccupied_alpha, system.noccupied_alpha), dtype=np.int8)
    n1a = 0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            idx1A[a, i] = n1a
            n1a += 1

    idx1B = np.zeros((system.nunoccupied_beta, system.noccupied_beta), dtype=np.int8)
    n1b = 0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            idx1B[a, i] = n1b
            n1b += 1

    idx2A = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha), dtype=np.int8)
    n2a = 0
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    idx2A[a, b, i, j] = n2a
                    idx2A[b, a, i, j] = -n2a
                    idx2A[a, b, j, i] = -n2a
                    idx2A[b, a, j, i] = n2a
                    n2a += 1

    idx2B = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta), dtype=np.int8)
    n2b = 0
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for i in range(system.noccupied_alpha):
                for j in range(system.noccupied_beta):
                    idx2B[a, b, i, j] = n2b
                    n2b += 1

    idx2C = -np.ones((system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta), dtype=np.int8)
    n2c = 0
    for a in range(system.nunoccupied_beta):
        for b in range(a + 1, system.nunoccupied_beta):
            for i in range(system.noccupied_beta):
                for j in range(i + 1, system.noccupied_beta):
                    idx2C[a, b, i, j] = n2c
                    idx2C[b, a, i, j] = -n2c
                    idx2C[a, b, j, i] = -n2c
                    idx2C[b, a, j, i] = n2c
                    n2c += 1

    return idx1A, idx1B, idx2A, idx2B, idx2C, n1a, n1b, n2a, n2b, n2c

if __name__ == "__main__":


    system, H0 = load_from_gamess(
            "chplus_re.log",
            "onebody.inp",
            "twobody.inp",
            nfrozen=0,
    )

    calculation = Calculation(
        order=2,
        calculation_type="ccsd",
        convergence_tolerance=1.0e-08,
        RHF_symmetry=False,
    )

    T, total_energy, _ = cc_driver(calculation, system, H0)

    H = build_hbar_ccsd(T, H0)

    idx1A, idx1B, idx2A, idx2B, idx2C, n1a, n1b, n2a, n2b, n2c = get_index_arrays(system)

    # 1A - 1A
    H1A1A = np.zeros((n1a, n1a))
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            for b in range(system.nunoccupied_alpha):
                for j in range(system.noccupied_alpha):
                    idet = idx1A[a, i]
                    jdet = idx1A[b, j]
                    H1A1A[idet, jdet] += (
                                        -(a == b) * H.a.oo[j, i]
                                        +(i == j) * H.a.vv[a, b]
                                        + H.aa.voov[a, j, i, b]
                    )

    H1A1A2 = np.zeros((n1a, n1a))
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            idet = idx1A[a, i]
            for m in range(system.noccupied_alpha):
                jdet = idx1A[a, m]
                H1A1A2[idet, jdet] -= H.a.oo[m, i]
            for e in range(system.nunoccupied_alpha):
                jdet = idx1A[e, i]
                H1A1A2[idet, jdet] += H.a.vv[a, e]
            for e in range(system.nunoccupied_alpha):
                for m in range(system.noccupied_alpha):
                    jdet = idx1A[e, m]
                    H1A1A2[idet, jdet] += H.aa.voov[a, m, i, e]

    print("Error 1A1A = ", np.linalg.norm(H1A1A.flatten() - H1A1A2.flatten()))

    # 1A - 2A
    H1A2A = np.zeros((n1a, n2a))
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            for b in range(system.nunoccupied_alpha):
                for c in range(b + 1, system.nunoccupied_alpha):
                    for j in range(system.noccupied_alpha):
                        for k in range(j + 1, system.noccupied_alpha):

                            idet = idx1A[a, i]
                            jdet = idx2A[b, c, j, k]

                            H1A2A[idet, jdet] += (
                                               +(i == j) * H.aa.vovv[a, k, b, c]
                                               -(i == k) * H.aa.vovv[a, j, b, c]
                            )

    H1A2A2 = np.zeros((n1a, n2a))
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):

            idet = idx1A[a, i]

            # 0.5 * h2A(anef) * r2a(efin)
            for e in range(system.nunoccupied_alpha):
                for f in range(system.nunoccupied_alpha):
                    for n in range(system.noccupied_alpha):
                        i1 = idx2A[e, f, i, n]
                        jdet = abs(i1)
                        H1A2A2[idet, jdet] += 0.5 * H.aa.vovv[a, n, e, f]# * float(np.sign(i1))

    print("Error 1A2A = ", np.linalg.norm(H1A2A.flatten() - H1A2A2.flatten()))

    # 1A - 2B
    H1A2B = np.zeros((n1a, n2b))
    ct1 = 0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            for b in range(system.nunoccupied_alpha):
                for c in range(system.nunoccupied_beta):
                    for j in range(system.noccupied_alpha):
                        for k in range(system.noccupied_beta):

                            idet = idx1A[a, i]
                            jdet = idx2B[b, c, j, k]
                            H1A2B[idet, jdet] += (
                                                    +(i == j) * H.ab.vovv[a, k, b, c]
                                                    -(a == b) * H.ab.ooov[j, k, i, c]
                            )

    H1A2B2 = np.zeros((n1a, n2b))
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):

            idet = idx1A[a, i]

            # h2b(anef) * r2b(efin)
            for e in range(system.nunoccupied_alpha):
                for f in range(system.nunoccupied_beta):
                    for n in range(system.noccupied_beta):
                        jdet = idx2B[e, f, i, n]
                        H1A2B2[idet, jdet] += H.ab.vovv[a, n, e, f]

            # -h2b(mnif) * r2b(afmn)
            for m in range(system.noccupied_alpha):
                for n in range(system.noccupied_beta):
                    for f in range(system.nunoccupied_beta):
                        jdet = idx2B[a, f, m, n]
                        H1A2B2[idet, jdet] -= H.ab.ooov[m, n, i, f]

    print("Error 1A2B = ", np.linalg.norm(H1A2B.flatten() - H1A2B2.flatten()))

    # # 2A - 2A
    # X2A = -0.5 * np.einsum("mi,abmj->abij", H.a.oo, T.aa, optimize=True)
    # X2A += 0.5 * np.einsum("ae,ebij->abij", H.a.vv, T.aa, optimize=True)
    # X2A += np.einsum("amie,ebmj->abij", H.aa.voov, T.aa, optimize=True)
    #
    # X2A -= np.transpose(X2A, (0, 1, 3, 2))
    # X2A -= np.transpose(X2A, (1, 0, 2, 3))
    #
    # X2A2 = np.zeros_like(X2A)
    # n2a_u = (system.noccupied_alpha**2 - system.noccupied_alpha)
    # for a in range(system.nunoccupied_alpha):
    #     for b in range(a + 1, system.nunoccupied_alpha):
    #         for i in range(system.noccupied_alpha):
    #             for j in range(i + 1, system.noccupied_alpha):
    #
    #                 idet = idx2A[a, b, i, j]
    #
    #                 # -A(ij) h1a(mi)*t2a(abmj)
    #                 for m in range(system.noccupied_alpha):
    #                     jdet = idx2A[a, b, m, j]
    #                     X2A2[a, b, i, j] -= H.a.oo[m, i] * T.aa[a, b, m, j]
    #                     jdet = idx2A[a, b, m, i]
    #                     X2A2[a, b, i, j] += H.a.oo[m, j] * T.aa[a, b, m, i]
    #                 # A(ab) h1a(ae)t2a(ebij)
    #                 for e in range(system.nunoccupied_alpha):
    #                     jdet = idx2A[e, b, i, j]
    #                     X2A2[a, b, i, j] += H.a.vv[a, e] * T.aa[e, b, i, j]
    #                     jdet = idx2A[e, a, i, j]
    #                     X2A2[a, b, i, j] -= H.a.vv[b, e] * T.aa[e, a, i, j]
    #                 # A(ab)A(ij) h2a(amie)t(ebmj)
    #                 for e in range(system.nunoccupied_alpha):
    #                     for m in range(system.noccupied_alpha):
    #                         jdet = idx2A[e, b, m, j]
    #                         X2A2[a, b, i, j] += H.aa.voov[a, m, i, e] * T.aa[e, b, m, j]
    #                         jdet = idx2A[e, a, m, j]
    #                         X2A2[a, b, i, j] -= H.aa.voov[b, m, i, e] * T.aa[e, a, m, j]
    #                         jdet = idx2A[e, b, m, i]
    #                         X2A2[a, b, i, j] -= H.aa.voov[a, m, j, e] * T.aa[e, b, m, i]
    #                         jdet = idx2A[e, a, m, i]
    #                         X2A2[a, b, i, j] += H.aa.voov[b, m, j, e] * T.aa[e, a, m, i]
    #
    #                 X2A2[a, b, j, i] = -1.0 * X2A2[a, b, i, j]
    #                 X2A2[b, a, i, j] = -1.0 * X2A2[a, b, i, j]
    #                 X2A2[b, a, j, i] = X2A2[a, b, i, j]
    #
    # print("Error = ", np.linalg.norm(X2A.flatten() - X2A2.flatten()))




