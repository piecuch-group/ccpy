import numpy as np

from ccpy.eom_guess.s2matrix import build_s2matrix_sfcis, spin_adapt_guess

def run_diagonalization(system, H, multiplicity, nroot, nacto, nactu, debug=False):

    Hmat = build_sfcis_hamiltonian(H, system)
    S2mat = build_s2matrix_sfcis(system, Ms=-1, debug=debug)
    omega, V = spin_adapt_guess(S2mat, Hmat, multiplicity)
    return omega[:nroot], V[:, :nroot]

def build_sfcis_hamiltonian(H, system):

    n1b = system.noccupied_alpha * system.nunoccupied_beta

    Hbb = np.zeros((n1b, n1b))
    ct1 = 0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for b in range(system.nunoccupied_beta):
                for j in range(system.noccupied_alpha):
                    Hbb[ct1, ct2] = (
                          H.b.vv[a, b] * (i == j)
                        - H.a.oo[j, i] * (a == b)
                        - H.ab.ovov[j, a, i, b]
                    )
                    ct2 += 1
            ct1 += 1

    return Hbb
