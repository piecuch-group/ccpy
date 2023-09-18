import numpy as np

from ccpy.eom_guess.s2matrix import build_s2matrix_1h, spin_adapt_guess

def run_diagonalization(system, H, multiplicity, roots_per_irrep, nacto, nactu, debug=False, use_symmetry=False):

    nroot = 0
    for irrep, nroot_irrep in roots_per_irrep.items():
        nroot += nroot_irrep

    Hmat = build_1h_hamiltonian(H, system)
    S2mat = build_s2matrix_1h(system)
    omega, V = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=debug)
    return omega[:nroot], V[:, :nroot]

def build_1h_hamiltonian(H, system):

    noa = system.noccupied_alpha
    Hmat = np.zeros((noa, noa))
    for i in range(noa):
        for j in range(noa):
            Hmat[i, j] = -H.a.oo[j, i]
    return Hmat

