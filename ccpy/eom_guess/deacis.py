import numpy as np

from ccpy.eom_guess.s2matrix import build_s2matrix_2p, spin_adapt_guess

def run_diagonalization(system, H, multiplicity, nroot):

    Hmat = build_2p_hamiltonian(H, system)
    #S2mat = build_s2matrix_2p(system)
    #omega, V = spin_adapt_guess(S2mat, Hmat, multiplicity)
    omega, V = np.linalg.eig(Hmat)
    idx = np.argsort(omega)
    omega = omega[idx]
    V = V[:, idx]
    return omega[:nroot], V[:, :nroot]

def build_2p_hamiltonian(H, system):

    n2b = system.nunoccupied_alpha * system.nunoccupied_beta

    Hab = np.zeros((n2b, n2b))
    ct1 = 0
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            ct2 = 0
            for c in range(system.nunoccupied_alpha):
                for d in range(system.nunoccupied_beta):
                    Hab[ct1, ct2] = (
                          H.b.vv[d, b] * (a == c)
                        + H.a.vv[c, a] * (b == d)
                        + H.ab.vvvv[c, d, a, b]
                    )
                    ct2 += 1
            ct1 += 1

    return Hab
