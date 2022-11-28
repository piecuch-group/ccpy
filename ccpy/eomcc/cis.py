import numpy as np

from ccpy.utilities.printing import print_amplitudes
from ccpy.eomcc.s2matrix import build_s2matrix_cis

def cis(calculation, system, H):

    ndim = (
            system.noccupied_alpha * system.nunoccupied_alpha 
            * system.noccupied_beta * system.nunoccupied_beta
    )
    H_cis = build_cis_hamiltonian(H, system)
    omega_cis, V_cis = np.linalg.eig(H_cis)
    idx = np.argsort(omega_cis)
    omega_cis = omega_cis[idx]
    V_cis = V_cis[:, idx]

    print("CIS Eigenvalues:")
    for i in range(ndim):
        print("E{} = {}".format(i + 1, omega_cis[i]))
        print_amplitudes(V_cis[:, i], system)

    S2 = build_s2matrix_cis(system)
    eval_s2, V_s2 = np.linalg.eig(S2)
    idx = np.argsort(eval_s2)
    eval_s2 = eval_s2[idx]
    V_s2 = V_s2[:, idx]

    Ns2 = 0
    idx_s2 = []
    print("S2 Eigenvalues:")
    for i in range(ndim):
        sval = -0.5 + np.sqrt(0.25 + eval_s2[i])
        multval = 2 * sval + 1
        print("S{} = {}  (mult = {})".format(i + 1, sval, multval))
        if abs(multval - calculation.multiplicity) < 1.0e-07:
            idx_s2.append(i)
            Ns2 += 1

    print("Dimension of spin subspace of multiplicity {} = {}".format(calculation.multiplicity, Ns2))
    Qs2 = ndim - Ns2
    W = np.zeros((ndim, Ns2))
    for i in range(Ns2):
        W[:, i] = V_s2[:, idx_s2[i]]

    G = np.einsum("Ku,Nv,Lu,Mv,LM->KN", W, W, W, W, H_cis, optimize=True)

    omega_cis, V_cis = np.linalg.eig(G)
    omega_cis = np.real(omega_cis)
    V_cis = np.real(V_cis)
    idx = np.argsort(omega_cis)
    omega_cis = omega_cis[idx]
    V_cis = V_cis[:, idx]
    print("Spin-adpated CIS eigenvalues:")
    for i in range(calculation.nroot):
        print("E{} = {}".format(i + 1, omega_cis[i + Qs2]))
        print_amplitudes(V_cis[:, i + Qs2], system)

    return omega_cis, V_cis


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
