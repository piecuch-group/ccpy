import numpy as np

from ccpy.eomcc.s2matrix import build_s2matrix_cis

def run_diagonalization(system, H, multiplicity):

    Hmat = build_cis_hamiltonian(H, system)
    omega, V = spin_adapt_guess(system, Hmat, multiplicity)
    idx = np.argsort(omega)
    omega = omega[idx]
    V = V[:, idx]

    return omega, V

def spin_adapt_guess(system, H, multiplicity):

    def _get_multiplicity(s2):
        s = -0.5 + np.sqrt(0.25 + s2)
        return 2.0 * s + 1.0

    ndim = H.shape[0]

    S2 = build_s2matrix_cis(system)
    eval_s2, V_s2 = np.linalg.eig(S2)
    idx_s2 = [i for i, s2 in enumerate(eval_s2) if abs(_get_multiplicity(s2) - multiplicity) < 1.0e-07]

    W = np.zeros((ndim, len(idx_s2)))
    for i in range(len(idx_s2)):
        W[:, i] = V_s2[:, idx_s2[i]]

    # Transform from determinantal basis to basis of S2 eigenfunctions
    G = np.einsum("Ku,Nv,Lu,Mv,LM->KN", W, W, W, W, H, optimize=True)

    omega, V = np.linalg.eig(G)
    omega = np.real(omega)
    V = np.real(V)

    idx = np.argsort(omega)

    return omega[idx[len(idx_s2):]], V[:, idx[len(idx_s2):]]


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
