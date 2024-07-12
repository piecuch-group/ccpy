import numpy as np

#from ccpy.eom_guess.s2matrix import build_s2matrix_2h, spin_adapt_guess

def run_diagonalization(system, H, multiplicity, roots_per_irrep, nacto, nactu, debug=False, use_symmetry=False):

    # print results of initial guess procedure
    print("   DIP-CIS initial guess routine")
    print("   --------------------------")
    print("   Multiplicity = ", multiplicity)
    #print("   Active occupied alpha = ", min(nacto + (system.multiplicity - 1), noa))
    #print("   Active occupied beta = ", min(nacto, nob))
    #print("   Active unoccupied alpha = ", min(nactu, nua))
    #print("   Active unoccupied beta = ", min(nactu + (system.multiplicity - 1), nub))

    nroot = 0
    for irrep, nroot_irrep in roots_per_irrep.items():
        nroot += nroot_irrep

    Hmat = build_2h_hamiltonian(H, nacto)
    #S2mat = build_s2matrix_2h(system, nacto)
    #omega, V_act = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=debug)
    omega, V_act = np.linalg.eig(Hmat)
    idx = np.argsort(omega)
    omega = np.real(omega[idx])
    V_act = np.real(V_act[:, idx])

    nroot = min(nroot, V_act.shape[1])

    # scatter active-space guess into full space
    V = np.zeros((system.noccupied_alpha*system.noccupied_beta, nroot))
    for i in range(nroot):
        V[:, i] = scatter(V_act[:, i], nacto, system)

    return omega, V

def scatter(V_in, nacto, system):

    V_out = np.zeros(system.noccupied_alpha * system.noccupied_beta)

    ct = 0
    ct2 = 0
    for i in range(system.noccupied_alpha):
        for j in range(system.noccupied_beta):
            if i >= system.noccupied_alpha - nacto and j >= system.noccupied_beta - nacto:
                V_out[ct] = V_in[ct2]
                ct2 += 1
            ct += 1
    return V_out

def build_2h_hamiltonian(H, nacto):

    n2b = nacto ** 2

    Hab = np.zeros((n2b, n2b))
    ct1 = 0
    for i in range(nacto):
        for j in range(nacto):
            ct2 = 0
            for k in range(nacto):
                for l in range(nacto):
                    Hab[ct1, ct2] = (
                        - H.b.oo[l, j] * (i == k)
                        - H.a.oo[k, i] * (j == l)
                        + H.ab.oooo[k, l, i, j]
                    )
                    ct2 += 1
            ct1 += 1

    return Hab
