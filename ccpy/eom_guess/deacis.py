import numpy as np

from ccpy.eom_guess.s2matrix import build_s2matrix_2p, spin_adapt_guess

def run_diagonalization(system, H, multiplicity, nroot, nacto, nactu):

    Hmat = build_2p_hamiltonian(H, system, nactu)
    #S2mat = build_s2matrix_2p(system)
    #omega, V = spin_adapt_guess(S2mat, Hmat, multiplicity)
    omega, V = np.linalg.eig(Hmat)
    idx = np.argsort(omega)
    omega = omega[idx]
    V = V[:, idx]
    return omega[:nroot], V[:, :nroot]

def build_2p_hamiltonian(H, system, nactu):

    n2b = nactu ** 2

    Hab = np.zeros((n2b, n2b))
    ct1 = 0
    for a in range(nactu):
        for b in range(nactu):
            ct2 = 0
            for c in range(nactu):
                for d in range(nactu):
                    Hab[ct1, ct2] = (
                          H.b.vv[b, d] * (a == c)
                        + H.a.vv[a, c] * (b == d)
                        + H.ab.vvvv[a, b, c, d]
                    )
                    ct2 += 1
            ct1 += 1

    return Hab
