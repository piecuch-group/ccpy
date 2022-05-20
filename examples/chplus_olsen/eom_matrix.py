import numpy as np

from ccpy.models.calculation import Calculation
from ccpy.interfaces.gamess_tools import load_from_gamess
from ccpy.drivers.driver import cc_driver

from ccpy.hbar.hbar_ccsd import build_hbar_ccsd

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

    n1a = system.noccupied_alpha * system.nunoccupied_alpha
    n1b = system.noccupied_beta * system.nunoccupied_beta

    idx1A = np.zeros((system.nunoccupied_alpha, system.noccupied_alpha), dtype=np.int8)
    ct = 0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            idx1A[a, i] = ct
            ct += 1

    idx1B = np.zeros((system.nunoccupied_beta, system.noccupied_beta), dtype=np.int8)
    ct = 0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            idx1B[a, i] = ct
            ct += 1

    idx2A = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha), dtype=np.int8)
    ct = 0
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):
                    idx2A[a, b, i, j] = ct
                    ct += 1

    idx2B = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta), dtype=np.int8)
    ct = 0
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_beta):
                    idx2B[a, b, i, j] = ct
                    ct += 1

    idx2C = np.zeros((system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta), dtype=np.int8)
    ct = 0
    for a in range(system.nunoccupied_beta):
        for b in range(a + 1, system.nunoccupied_beta):
            for i in range(system.noccupied_beta):
                for j in range(i + 1, system.noccupied_beta):
                    idx2C[a, b, i, j] = ct
                    ct += 1

    # 1A - 1A
    H1A1A = np.zeros((n1a, n1a))
    ct1 = 0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for b in range(system.nunoccupied_alpha):
                for j in range(system.noccupied_alpha):
                    H1A1A[ct1, ct2] = (
                                        -(a == b) * H.a.oo[j, i]
                                        +(i == j) * H.a.vv[a, b]
                                        + H.aa.voov[a, j, i, b]
                    )
                    ct2 += 1
            ct1 += 1

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

    print("Error = ", np.linalg.norm(H1A1A.flatten() - H1A1A2.flatten()))

    # 2A - 2A
    X2A = -0.5 * np.einsum("mi,abmj->abij", H.a.oo, T.aa, optimize=True)
    X2A += 0.5 * np.einsum("ae,ebij->abij", H.a.vv, T.aa, optimize=True)
    X2A += np.einsum("amie,ebmj->abij", H.aa.voov, T.aa, optimize=True)

    X2A -= np.transpose(X2A, (0, 1, 3, 2))
    X2A -= np.transpose(X2A, (1, 0, 2, 3))

    X2A2 = np.zeros_like(X2A)
    n2a_u = (system.noccupied_alpha**2 - system.noccupied_alpha)
    for a in range(system.nunoccupied_alpha):
        for b in range(a + 1, system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(i + 1, system.noccupied_alpha):

                    idet = idx2A[a, b, i, j]

                    # -A(ij) h1a(mi)*t2a(abmj)
                    for m in range(system.noccupied_alpha):
                        jdet = idx2A[a, b, m, j]
                        X2A2[a, b, i, j] -= H.a.oo[m, i] * T.aa[a, b, m, j]
                        jdet = idx2A[a, b, m, i]
                        X2A2[a, b, i, j] += H.a.oo[m, j] * T.aa[a, b, m, i]
                    # A(ab) h1a(ae)t2a(ebij)
                    for e in range(system.nunoccupied_alpha):
                        jdet = idx2A[e, b, i, j]
                        X2A2[a, b, i, j] += H.a.vv[a, e] * T.aa[e, b, i, j]
                        jdet = idx2A[e, a, i, j]
                        X2A2[a, b, i, j] -= H.a.vv[b, e] * T.aa[e, a, i, j]
                    # A(ab)A(ij) h2a(amie)t(ebmj)
                    for e in range(system.nunoccupied_alpha):
                        for m in range(system.noccupied_alpha):
                            jdet = idx2A[e, b, m, j]
                            X2A2[a, b, i, j] += H.aa.voov[a, m, i, e] * T.aa[e, b, m, j]
                            jdet = idx2A[e, a, m, j]
                            X2A2[a, b, i, j] -= H.aa.voov[b, m, i, e] * T.aa[e, a, m, j]
                            jdet = idx2A[e, b, m, i]
                            X2A2[a, b, i, j] -= H.aa.voov[a, m, j, e] * T.aa[e, b, m, i]
                            jdet = idx2A[e, a, m, i]
                            X2A2[a, b, i, j] += H.aa.voov[b, m, j, e] * T.aa[e, a, m, i]

                    X2A2[a, b, j, i] = -1.0 * X2A2[a, b, i, j]
                    X2A2[b, a, i, j] = -1.0 * X2A2[a, b, i, j]
                    X2A2[b, a, j, i] = X2A2[a, b, i, j]

    print("Error = ", np.linalg.norm(X2A.flatten() - X2A2.flatten()))




