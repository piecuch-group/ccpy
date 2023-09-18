import numpy as np
import time
from ccpy.eom_guess.s2matrix import build_s2matrix_cis, spin_adapt_guess

def run_diagonalization(system, H, multiplicity, roots_per_irrep, nacto, nactu, use_symmetry=True, debug=False):

    # if target_irrep is not None:
    #     multiplicity = -1
    #
    # Hmat = build_cis_hamiltonian(H, system, target_irrep)
    # S2mat = build_s2matrix_cis(system)
    # omega, V = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=debug)
    #
    # nroot = min(nroot, V.shape[1])
    # noa, nob, nua, nub = H.ab.oovv.shape
    # n1 = noa*nua + nob*nub
    # ndim = noa*nua + nob*nub + noa**2*nua**2 + noa*nob*nua*nub + nob**2*nub**2
    # V_guess = np.zeros((ndim, nroot))
    # omega_guess = np.zeros(nroot)
    # ct = 0
    # for i in range(len(omega)):
    #     if omega[i] == 0.0: continue
    #     V_guess[:n1, ct] = V[:, i]
    #     omega_guess[ct] = omega[i]
    #     ct += 1
    #     if ct == nroot: break
    nroots_total = 0
    for key, value in roots_per_irrep.items():
        nroots_total += value

    noa, nob, nua, nub = H.ab.oovv.shape
    n1 = noa * nua + nob * nub
    ndim = noa * nua + nob * nub + noa ** 2 * nua ** 2 + noa * nob * nua * nub + nob ** 2 * nub ** 2
    V = np.zeros((ndim, nroots_total))
    omega_guess = np.zeros(nroots_total)
    n_found = 0

    # print results of initial guess procedure
    print("   CIS initial guess routine")
    print("   --------------------------")
    print("   Multiplicity = ", multiplicity)

    t1 = time.time()
    S2mat = build_s2matrix_cis(system)
    print("   Time requried for S2 matrix =", time.time() - t1, "seconds")

    for irrep, nroot in roots_per_irrep.items():
        if nroot == 0: continue
        if not use_symmetry: irrep = None
        t1 = time.time()
        Hmat = build_cis_hamiltonian(H, system, irrep)
        t2 = time.time()
        omega, V_cis = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=debug)

        nroot = min(nroot, V_cis.shape[1])
        kout = 0
        for i in range(len(omega)):
            if omega[i] == 0.0: continue
            V[:n1, n_found] = V_cis[:, i]
            omega_guess[n_found] = omega[i]
            n_found += 1
            kout += 1
            if kout == nroot:
                break

        print("   -----------------------------------")
        print("   Target symmetry irrep = ", irrep, f"({system.point_group})")
        print("   Dimension of eigenvalue problem = ", V_cis.shape[0])
        print("   Time required for H matrix = ", t2 - t1, "seconds")
        for i in range(n_found - kout, n_found):
            print("   Eigenvalue of root", i + 1, " = ", np.round(omega_guess[i], 8))

    return omega_guess, V

def build_cis_hamiltonian(H, system, target_irrep):

    n1a = system.noccupied_alpha * system.nunoccupied_alpha
    n1b = system.noccupied_beta * system.nunoccupied_beta
    noa, nob, nua, nub = H.ab.oovv.shape

    if target_irrep is None:
        sym1 = lambda a, i: True
    else:
        sym = lambda orbital_number: system.point_group_irrep_to_number[system.orbital_symmetries[orbital_number]]
        ref_sym = system.point_group_irrep_to_number[system.reference_symmetry]
        target_sym = system.point_group_irrep_to_number[target_irrep]
        sym1 = lambda a, i: sym(i) ^ sym(a) ^ ref_sym == target_sym

    Haa = np.zeros((n1a, n1a))
    ct1 = 0
    for a in range(system.nunoccupied_alpha):
        for i in range(system.noccupied_alpha):
            if sym1(a + noa, i):
                ct2 = 0
                for b in range(system.nunoccupied_alpha):
                    for j in range(system.noccupied_alpha):
                        if sym1(b + noa, j):
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
            if sym1(a + noa, i):
                ct2 = 0
                for b in range(system.nunoccupied_beta):
                    for j in range(system.noccupied_beta):
                        if sym1(b + nob, j):
                            Hab[ct1, ct2] = H.ab.voov[a, j, i, b]
                        ct2 += 1
            ct1 += 1
    Hba = np.zeros((n1b, n1a))
    ct1 = 0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            if sym1(a + nob, i):
                ct2 = 0
                for b in range(system.nunoccupied_alpha):
                    for j in range(system.noccupied_alpha):
                        if sym1(b + noa, j):
                            Hba[ct1, ct2] = H.ab.ovvo[j, a, b, i]
                        ct2 += 1
            ct1 += 1
    Hbb = np.zeros((n1b, n1b))
    ct1 = 0
    for a in range(system.nunoccupied_beta):
        for i in range(system.noccupied_beta):
            if sym1(a + nob, i):
                ct2 = 0
                for b in range(system.nunoccupied_beta):
                    for j in range(system.noccupied_beta):
                        if sym1(b + nob, j):
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
