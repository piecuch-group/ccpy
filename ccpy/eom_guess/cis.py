import numpy as np

from ccpy.eom_guess.s2matrix import build_s2matrix_cis, spin_adapt_guess

def run_diagonalization(system, H, multiplicity, nroot, nacto, nactu, debug=False):

    Hmat = build_cis_hamiltonian(H, system)
    S2mat = build_s2matrix_cis(system)
    omega, V = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=debug)

    return omega[:nroot], V[:, :nroot]

def build_cis_hamiltonian(H, system):

    n1a = system.noccupied_alpha * system.nunoccupied_alpha
    n1b = system.noccupied_beta * system.nunoccupied_beta

    Haa = np.zeros((n1a, n1a))
    ct1 = 0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for b in range(system.nunoccupied_alpha):
                for j in range(system.noccupied_alpha):
                    Haa[ct1, ct2] = (
                          H.a.vv[a, b] * (i == j)
                        - H.a.oo[j, i] * (a == b)
                        + H.aa.voov[a, j, i, b]
                    )
                    ct2 += 1
            ct1 += 1
    Hab = np.zeros((n1a, n1b))
    ct1 = 0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for b in range(system.nunoccupied_beta):
                for j in range(system.noccupied_beta):
                    Hab[ct1, ct2] = H.ab.voov[a, j, i, b]
                    ct2 += 1
            ct1 += 1
    Hba = np.zeros((n1b, n1a))
    ct1 = 0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            ct2 = 0
            for b in range(system.nunoccupied_alpha):
                for j in range(system.noccupied_alpha):
                    Hba[ct1, ct2] = H.ab.ovvo[j, a, b, i]
                    ct2 += 1
            ct1 += 1
    Hbb = np.zeros((n1b, n1b))
    ct1 = 0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            ct2 = 0
            for b in range(system.nunoccupied_beta):
                for j in range(system.noccupied_beta):
                    Hbb[ct1, ct2] = (
                        H.b.vv[a, b] * (i == j)
                        - H.b.oo[j, i] * (a == b)
                        + H.bb.voov[a, j, i, b]
                    )
                    ct2 += 1
            ct1 += 1
    return np.concatenate(
        (np.concatenate((Haa, Hab), axis=1), np.concatenate((Hba, Hbb), axis=1)), axis=0
    )
