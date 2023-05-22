import numpy as np

#from ccpy.eom_guess.s2matrix import build_s2matrix_sfcis

def run_diagonalization(system, H, multiplicity, nroot):

    Hmat = build_sfcis_hamiltonian(H, system)
    # print("Untransformed omega")
    # tmp_e, tmp_v = np.linalg.eig(Hmat)
    # idx = np.argsort(tmp_e)
    # tmp_e = tmp_e[idx]
    # tmp_v = tmp_v[:, idx]
    # omega = tmp_e
    # V = tmp_v
    omega, V = spin_adapt_guess(system, Hmat, multiplicity)

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

def spin_adapt_guess(system, H, multiplicity):

    def _get_multiplicity(s2):
        s = -0.5 + np.sqrt(0.25 + s2)
        return 2.0 * s + 1.0

    ndim = H.shape[0]

    S2 = build_s2matrix_sfcis(system)
    eval_s2, V_s2 = np.linalg.eig(S2)
    idx_s2 = [i for i, s2 in enumerate(eval_s2) if abs(_get_multiplicity(s2) - multiplicity) < 1.0e-07]
    n_s2_sub = len(idx_s2)

    W = np.zeros((ndim, n_s2_sub))
    for i in range(n_s2_sub):
        W[:, i] = V_s2[:, idx_s2[i]]

    # Transform into determinantal eigenbasis of S2
    G = np.einsum("Ku,Nv,Lu,Mv,LM->KN", W, W, W, W, H, optimize=True)
    # diagonalize and sort the resulting eigenvalues
    omega, V = np.linalg.eig(G)
    omega = np.real(omega)
    V = np.real(V)
    idx = np.argsort(omega)
    omega = omega[idx]
    V = V[:, idx]

    # now all the eigenvalues that do not have the correct multiplicity are going to be numerically 0
    # retain only those that are non-zero to find the spin-adapted subspace
    # Unfortunately, for SF-CIS, one root will genuinely have 1 excitation energy equal to 0, corresponding to
    # the low-spin triplet ground-state described as a spin-flip excitation out of the high-spin reference
    omega_adapt = np.zeros(n_s2_sub)
    V_adapt = np.zeros((ndim, n_s2_sub))
    n = 0
    for i in range(len(omega)):
        if abs(omega[i]) < 1.0e-09: continue
        omega_adapt[n] = omega[i]
        V_adapt[:, n] = V[:, i]
        n += 1
    return omega_adapt, V_adapt

def get_sz2(system):

    Ns = float(system.noccupied_alpha - 1 - (system.noccupied_beta + 1))
    sz = Ns / 2.0
    sz2 = (sz + 1.0) * sz

    return sz2

def build_s2matrix_sfcis(system):

    n1b = system.nunoccupied_beta * system.noccupied_alpha

    sz2 = get_sz2(system)

    Sbb = np.zeros((n1b, n1b))
    ct1 = 0
    for a in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
        for i in range(system.noccupied_alpha):
            ct2 = 0
            for b in range(system.noccupied_beta, system.noccupied_beta + system.nunoccupied_beta):
                for j in range(system.noccupied_alpha):
                    Sbb[ct1, ct2] += sz2 * (a == b) * (i == j)
                    Sbb[ct1, ct2] += (a == i) * (b == j)
                    ct2 += 1
            ct1 += 1

    return Sbb
