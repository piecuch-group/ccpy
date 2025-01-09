'''
DIP-EOMCCSD(2h) Guess Routine for DIP-EOMCC
'''

import numpy as np
from ccpy.eom_guess.s2matrix import spin_adapt_guess

def run_diagonalization(system, H, multiplicity, roots_per_irrep, nacto, nactu, debug=False, use_symmetry=False):

    # print results of initial guess procedure
    print("   DIP-CIS initial guess routine")
    print("   --------------------------")
    print("   Multiplicity = ", multiplicity)
    print("   Active occupied alpha = ", min(nacto + (system.multiplicity - 1), system.noccupied_alpha))
    print("   Active occupied beta = ", min(nacto, system.noccupied_beta))
    print("   Active unoccupied alpha = ", min(nactu, system.nunoccupied_alpha))
    print("   Active unoccupied beta = ", min(nactu + (system.multiplicity - 1), system.nunoccupied_beta))

    nroot = 0
    for irrep, nroot_irrep in roots_per_irrep.items():
        nroot += nroot_irrep

    Hmat = build_2h_hamiltonian(H, system, nacto)
    S2mat = build_s2matrix_2h(system, nacto)
    omega, V_act = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=debug)

    nroot = min(nroot, V_act.shape[1])

    # scatter active-space guess into full space
    V = np.zeros((system.noccupied_alpha*system.noccupied_beta, nroot))
    for i in range(nroot):
        V[:, i] = scatter(V_act[:, i], nacto, system)
        print("   Eigenvalue of root", i + 1, " = ", np.round(omega[i], 8))
    print("")
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

def build_2h_hamiltonian(H, system, nacto):

    noa = system.noccupied_alpha
    nob = system.noccupied_beta
    nua = system.nunoccupied_alpha
    nub = system.nunoccupied_beta

    nacto_a = min(nacto + (system.multiplicity - 1), noa)
    nacto_b = min(nacto, nob)
    # nactu_a = min(nactu, nua)
    # nactu_b = min(nactu + (system.multiplicity - 1), nub)

    n2b = nacto_a * nacto_b

    Hab = np.zeros((n2b, n2b))
    ct1 = 0
    for i in range(noa - nacto_a, noa):
        for j in range(nob - nacto_b, nob):
            ct2 = 0
            for k in range(noa - nacto_a, noa):
                for l in range(nob - nacto_b, nob):
                    Hab[ct1, ct2] = (
                        - H.b.oo[l, j] * (i == k)
                        - H.a.oo[k, i] * (j == l)
                        + H.ab.oooo[k, l, i, j]
                    )
                    ct2 += 1
            ct1 += 1
    return Hab

def build_s2matrix_2h(system, nacto):

    noa = system.noccupied_alpha
    nob = system.noccupied_beta
    nua = system.nunoccupied_alpha
    nub = system.nunoccupied_beta

    nacto_a = min(nacto + (system.multiplicity - 1), noa)
    nacto_b = min(nacto, nob)
    # nactu_a = min(nactu, nua)
    # nactu_b = min(nactu + (system.multiplicity - 1), nub)

    n2b = nacto_a * nacto_b

    Sab = np.zeros((n2b, n2b))
    ct1 = 0
    for i in range(noa - nacto_a, noa):
        for j in range(nob - nacto_b, nob):
            ct2 = 0
            for k in range(noa - nacto_a, noa):
                for l in range(nob - nacto_b, nob):
                    Sab[ct1, ct2] = (
                          (i == k) * (j == l)
                        + (i == l) * (j == k)
                    )
                    ct2 += 1
            ct1 += 1
    return Sab
