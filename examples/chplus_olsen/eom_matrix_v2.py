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
            n1a += 1
            idx1A[a, i] = n1a

    idx1B = np.zeros((system.nunoccupied_beta, system.noccupied_beta), dtype=np.int8)
    n1b = 0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            n1b += 1
            idx1B[a, i] = n1b

    idx2A = np.zeros((system.nunoccupied_alpha, system.nunoccupied_alpha, system.noccupied_alpha, system.noccupied_alpha), dtype=np.int8)
    n2a = 0
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_alpha):
            for i in range(system.noccupied_alpha):
                for j in range(system.noccupied_alpha):
                    n2a += 1
                    if i == j or a == b: continue
                    idx2A[a, b, i, j] = n2a


    idx2B = np.zeros((system.nunoccupied_alpha, system.nunoccupied_beta, system.noccupied_alpha, system.noccupied_beta), dtype=np.int8)
    n2b = 0
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            for i in range(system.noccupied_alpha):
                for j in range(system.noccupied_beta):
                    n2b += 1
                    idx2B[a, b, i, j] = n2b

    idx2C = -np.ones((system.nunoccupied_beta, system.nunoccupied_beta, system.noccupied_beta, system.noccupied_beta), dtype=np.int8)
    n2c = 0
    for a in range(system.nunoccupied_beta):
        for b in range(system.nunoccupied_beta):
            for i in range(system.noccupied_beta):
                for j in range(system.noccupied_beta):
                    n2c += 1
                    if i == j or a == b: continue
                    idx2C[a, b, i, j] = n2c

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
                            #H1A2A[idet - 1, jdet - 1] += 0.5 * (i == j) * H.aa.vovv[a, k, b, c]

                            H1A2A[idet - 1, jdet - 1] += (
                                               +(i == j) * H.aa.vovv[a, k, b, c]
                                               -(i == k) * H.aa.vovv[a, j, b, c]
                            )

    H1A2A2 = np.zeros((n1a, n2a))
    error = 0.0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):

            idet = idx1A[a, i]

            # 0.5 * h2A(anef) * r2a(efin)
            for e in range(system.nunoccupied_alpha):
                for f in range(system.nunoccupied_alpha):
                    for n in range(system.noccupied_alpha):
                        i1 = idx2A[e, f, i, n]
                        jdet = abs(i1)
                        if jdet == 0: continue
                        H1A2A2[idet - 1, jdet - 1] += 0.5 * H.aa.vovv[a, n, e, f]

                        error += H1A2A2[idet - 1, jdet - 1] - H1A2A[idet - 1, jdet - 1]

    print("Error 1A2A = ", error)

