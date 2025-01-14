'''
DEA-EOMCCSD(2p) Guess Routine for DEA-EOMCC
'''

import numpy as np
from ccpy.eom_guess.s2matrix import spin_adapt_guess

def run_diagonalization(system, H, multiplicity, roots_per_irrep, nacto, nactu, debug=False, use_symmetry=False):

    # print results of initial guess procedure
    print("   DEA-CIS initial guess routine")
    print("   --------------------------")
    print("   Multiplicity = ", multiplicity)
    print("   Active occupied alpha = ", min(nacto + (system.multiplicity - 1), system.noccupied_alpha))
    print("   Active occupied beta = ", min(nacto, system.noccupied_beta))
    print("   Active unoccupied alpha = ", min(nactu, system.nunoccupied_alpha))
    print("   Active unoccupied beta = ", min(nactu + (system.multiplicity - 1), system.nunoccupied_beta))

    nroot = 0
    for irrep, nroot_irrep in roots_per_irrep.items():
        nroot += nroot_irrep

    Hmat = build_2p_hamiltonian(H, system, nactu)
    S2mat = build_s2matrix_2p(system, nactu)
    omega, V_act = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=debug)

    nroot = min(nroot, V_act.shape[1])

    # scatter active-space guess into full space
    V = np.zeros((system.nunoccupied_alpha*system.nunoccupied_beta, nroot))
    for i in range(nroot):
        V[:, i] = scatter(V_act[:, i], nactu, system)
        print("   Eigenvalue of root", i + 1, " = ", np.round(omega[i], 8))
    print("")
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

def build_2p_hamiltonian(H, system, nactu):

    noa = system.noccupied_alpha
    nob = system.noccupied_beta
    nua = system.nunoccupied_alpha
    nub = system.nunoccupied_beta

    # nacto_a = min(nacto + (system.multiplicity - 1), noa)
    # nacto_b = min(nacto, nob)
    nactu_a = min(nactu, nua)
    nactu_b = min(nactu + (system.multiplicity - 1), nub)

    n2b = nactu_a * nactu_b

    Hab = np.zeros((n2b, n2b))
    ct1 = 0
    for a in range(nactu_a):
        for b in range(nactu_b):
            ct2 = 0
            for c in range(nactu_a):
                for d in range(nactu_b):
                    Hab[ct1, ct2] = (
                          H.b.vv[b, d] * (a == c)
                        + H.a.vv[a, c] * (b == d)
                        + H.ab.vvvv[a, b, c, d]
                    )
                    ct2 += 1
            ct1 += 1

    return Hab

def get_sz2(system, Ms):
    Ns = float((system.noccupied_alpha + Ms) - (system.noccupied_beta - Ms))
    sz = Ns / 2.0
    sz2 = (sz + 1.0) * sz
    return sz2

def build_s2matrix_2p(system, nactu):

    def pi_alpha(p):
        if p >= system.noccupied_alpha and p < system.nunoccupied_alpha + system.noccupied_alpha:
            return 1.0
        else:
            return 0.0

    noa = system.noccupied_alpha
    nob = system.noccupied_beta
    nua = system.nunoccupied_alpha
    nub = system.nunoccupied_beta

    # nacto_a = min(nacto + (system.multiplicity - 1), noa)
    # nacto_b = min(nacto, nob)
    nactu_a = min(nactu, nua)
    nactu_b = min(nactu + (system.multiplicity - 1), nub)

    n2b = nactu_a * nactu_b
    sz2 = get_sz2(system, Ms=0) # this needs to be modified potentially
    Sab = np.zeros((n2b, n2b))
    ct1 = 0
    for a in range(system.noccupied_alpha, system.noccupied_alpha + nactu_a):
        for b in range(system.noccupied_beta, system.noccupied_beta + nactu_b):
            ct2 = 0
            for c in range(system.noccupied_alpha, system.noccupied_alpha + nactu_a):
                for d in range(system.noccupied_beta, system.noccupied_beta + nactu_b):
                    Sab[ct1, ct2] += (sz2 + 1.0 * pi_alpha(a)) * (a == c) * (b == d)
                    Sab[ct1, ct2] -= (b == c) * (a == d) # why is this a minus sign, you ask... "ghost loop" rule
                    ct2 += 1
            ct1 += 1
    return Sab