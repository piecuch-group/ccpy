import numpy as np

from ccpy.eom_guess.s2matrix import build_s2matrix_2p, spin_adapt_guess

def run_diagonalization(system, H, multiplicity, roots_per_irrep, nacto, nactu, debug=False, use_symmetry=False):

    nroot = 0
    for irrep, nroot_irrep in roots_per_irrep.items():
        nroot += nroot_irrep

    Hmat = build_2p_hamiltonian(H, nactu)
    S2mat = build_s2matrix_2p(system, nactu)
    omega, V_act = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=debug)
    nroot = min(nroot, V_act.shape[1])

    # scatter active-space guess into full space
    V = np.zeros((system.nunoccupied_alpha*system.nunoccupied_beta, nroot))
    for i in range(nroot):
        V[:, i] = scatter(V_act[:, i], nactu, system)

    return omega, V

def scatter(V_in, nactu, system):

    V_out = np.zeros(system.nunoccupied_alpha * system.nunoccupied_beta)

    ct = 0
    ct2 = 0
    for a in range(system.nunoccupied_alpha):
        for b in range(system.nunoccupied_beta):
            if a < nactu and b < nactu:
                V_out[ct] = V_in[ct2]
                ct2 += 1
            ct += 1
    return V_out

def build_2p_hamiltonian(H, nactu):

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
