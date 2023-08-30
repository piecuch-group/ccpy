import numpy as np

from ccpy.eom_guess.s2matrix import build_s2matrix_1p, spin_adapt_guess

def run_diagonalization(system, H, multiplicity, nroot, nacto, nactu, debug=False):

    Hmat = build_1p_hamiltonian(H, system)
    S2mat = build_s2matrix_1p(system)
    omega, V = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=debug)
    return omega[:nroot], V[:, :nroot]

def build_1p_hamiltonian(H, system):

    nua = system.nunoccupied_alpha
    Hmat = np.zeros((nua, nua))
    for a in range(nua):
        for b in range(nua):
            Hmat[a, b] = H.a.vv[a, b]
    return Hmat