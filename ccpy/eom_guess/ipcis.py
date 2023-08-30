import numpy as np

from ccpy.eom_guess.s2matrix import build_s2matrix_1h, spin_adapt_guess

def run_diagonalization(system, H, multiplicity, nroot, nacto, nactu, debug=False):

    Hmat = build_1h_hamiltonian(H, system)
    # S2mat = build_s2matrix_1h(system)
    # omega, V = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=debug)
    omega, V = np.linalg.eig(Hmat)
    idx = np.argsort(omega)
    omega = omega[idx]
    V = V[:, idx]

    return omega[:nroot], V[:, :nroot]

def build_1h_hamiltonian(H, system):

    noa = system.noccupied_alpha
    Hmat = np.zeros((noa, noa))
    for i in range(noa):
        for j in range(noa):
            Hmat[i, j] = -H.a.oo[j, i]
    return Hmat

