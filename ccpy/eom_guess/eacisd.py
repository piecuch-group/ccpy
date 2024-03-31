import numpy as np
import time
from ccpy.eom_guess.s2matrix import spin_adapt_guess

def run_diagonalization(system, H, multiplicity, roots_per_irrep, nacto, nactu, use_symmetry=True, debug=False):

    nroots_total = 0
    for key, value in roots_per_irrep.items():
        nroots_total += value

    noa, nob, nua, nub = H.ab.oovv.shape
    ndim = nua + nua**2*noa + nua*nub*nob
    V = np.zeros((ndim, nroots_total))
    omega_guess = np.zeros(nroots_total)
    n_found = 0

    # print results of initial guess procedure
    print("   2p-1h initial guess routine")
    print("   --------------------------")
    print("   Multiplicity = ", multiplicity)
    print("   Active occupied alpha = ", min(nacto + (system.multiplicity - 1), noa))
    print("   Active occupied beta = ", min(nacto, nob))
    print("   Active unoccupied alpha = ", min(nactu, nua))
    print("   Active unoccupied beta = ", min(nactu + (system.multiplicity - 1), nub))

    for irrep, nroot in roots_per_irrep.items():
        if nroot == 0: continue
        if not use_symmetry: irrep = None

        # Build the indexing arrays for the given irrep
        idx_a, idx_aa, idx_ab, ndim_irrep = get_index_arrays(nacto, nactu, system, irrep)
        t1 = time.time()
        # Compute the active-space 2p-1h Hamiltonian
        Hmat = build_eacisd_hamiltonian(H, nacto, nactu, idx_a, idx_aa, idx_ab, system)
        # Compute the S2 matrix in the same projection subspace
        S2mat = build_s2matrix(system, nacto, nactu, idx_a, idx_aa, idx_ab)
        # Project H onto the spin subspace with the specified multiplicity
        omega, V_act = spin_adapt_guess(S2mat, Hmat, multiplicity, debug=debug)

        nroot = min(nroot, V_act.shape[1])
        kout = 0
        for i in range(len(omega)):
            if omega[i] == 0.0: continue
            V[:, n_found] = scatter(V_act[:, i], nacto, nactu, system)
            omega_guess[n_found] = omega[i]
            n_found += 1
            kout += 1
            if kout == nroot:
                break

        elapsed_time = time.time() - t1
        print("   -----------------------------------")
        print("   Target symmetry irrep = ", irrep, f"({system.point_group})")
        print("   Dimension of eigenvalue problem = ", ndim_irrep)
        print("   Elapsed time = ", np.round(elapsed_time, 2), "seconds")
        for i in range(n_found - kout, n_found):
            print("   Eigenvalue of root", i + 1, " = ", np.round(omega_guess[i], 8))
    print("")
    return omega_guess, V

def scatter(V_in, nacto, nactu, system):
    # orbital dimensions
    noa = system.noccupied_alpha
    nob = system.noccupied_beta
    nua = system.nunoccupied_alpha
    nub = system.nunoccupied_beta

    # set active space parameters
    nacto_a = min(nacto + (system.multiplicity - 1), noa)
    nacto_b = min(nacto, nob)
    nactu_a = min(nactu, nua)
    nactu_b = min(nactu + (system.multiplicity - 1), nub)

    # allocate full-length output vector
    V_a_out = np.zeros(nua)
    V_aa_out = np.zeros((nua, nua, noa))
    V_ab_out = np.zeros((nua, nub, nob))
    # Start filling in the array
    offset = 0
    for a in range(nua):
        V_a_out[a] = V_in[offset]
        offset += 1
    for a in range(nua):
        for b in range(a + 1, nua):
            for j in range(noa):
                if a < nactu_a and b < nactu_a and j >= noa - nacto_a:
                    V_aa_out[a, b, j] = V_in[offset]
                    V_aa_out[b, a, j] = -V_in[offset]
                    offset += 1
    for a in range(nua):
        for b in range(nub):
            for j in range(nob):
                if a < nactu_a and b < nactu_b and j >= nob - nacto_b:
                    V_ab_out[a, b, j] = V_in[offset]
                    offset += 1
    return np.hstack((V_a_out.flatten(), V_aa_out.flatten(), V_ab_out.flatten()))

def build_eacisd_hamiltonian(H, nacto, nactu, idx_a, idx_aa, idx_ab, system):
    # orbital dimensions
    noa, nob, nua, nub = H.ab.oovv.shape
    # set active space parameters
    nacto_a = min(nacto + (system.multiplicity - 1), noa)
    nacto_b = min(nacto, nob)
    nactu_a = min(nactu, nua)
    nactu_b = min(nactu + (system.multiplicity - 1), nub)
    # set dimensions of 2p-1h problem
    n1a = nua
    n2a = int(nacto_a * nactu_a * (nactu_a - 1) / 2)
    n2b = nacto_b * nactu_a * nactu_b
    ####################################################################################
    # ALPHA 1p
    ####################################################################################
    a_H_a = np.zeros((n1a, n1a))
    a_H_aa = np.zeros((n1a, n2a))
    a_H_ab = np.zeros((n1a, n2b))
    for a in range(nua):
        idet = idx_a[a]
        if idet == 0: continue
        ind1 = abs(idet) - 1
        # h1a(ae) * r_a(e)
        for e in range(nua):
            jdet = idx_a[e]
            if jdet != 0:
                ind2 = abs(jdet) - 1
                a_H_a[ind1, ind2] += H.a.vv[a, e]
        # 1/2 h2a(anef) * r_aa(efn)
        for e in range(nactu_a):
            for f in range(e + 1, nactu_a):
                for n in range(noa - nacto_a, noa):
                    jdet = idx_aa[e, f, n]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        a_H_aa[ind1, ind2] += H.aa.vovv[a, n, e, f]
        # h2b(anef) * r_ab(efn)
        for e in range(nactu_a):
            for f in range(nactu_b):
                for n in range(nob - nacto_b, nob):
                    jdet = idx_ab[e, f, n]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        a_H_ab[ind1, ind2] += H.ab.vovv[a, n, e, f]
        # h1a(me) * r_aa(aem)
        for e in range(nactu_a):
            for m in range(noa - nacto_a, noa):
                jdet = idx_aa[a, e, m]
                if jdet != 0:
                    ind2 = abs(jdet) - 1
                    phase = np.sign(jdet)
                    a_H_aa[ind1, ind2] += H.a.ov[m, e] * phase
        # h1b(me) * r_ab(aem)
        for e in range(nactu_b):
            for m in range(nob - nacto_b, nob):
                jdet = idx_ab[a, e, m]
                if jdet != 0:
                    ind2 = abs(jdet) - 1
                    a_H_ab[ind1, ind2] += H.b.ov[m, e]
    ####################################################################################
    # ALPHA-ALPHA 2p-1h
    ####################################################################################
    aa_H_a = np.zeros((n2a, n1a))
    aa_H_aa = np.zeros((n2a, n2a))
    aa_H_ab = np.zeros((n2a, n2b))
    for a in range(nactu_a):
        for b in range(a + 1, nactu_a):
            for j in range(noa - nacto_a, noa):
                idet = idx_aa[a, b, j]
                if idet == 0: continue
                ind1 = abs(idet) - 1
                # h2a(baje) * r_a(e)
                for e in range(nua):
                    jdet = idx_a[e]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        aa_H_a[ind1, ind2] += H.aa.vvov[b, a, j, e]
                # -h1a(mj) * r_aa(abm)
                for m in range(noa - nacto_a, noa):
                    jdet = idx_aa[a, b, m]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        aa_H_aa[ind1, ind2] -= H.a.oo[m, j]
                # 1/2 h2a(abef) * r_aa(efj)
                for e in range(nactu_a):
                    for f in range(e + 1, nactu_a):
                        jdet = idx_aa[e, f, j]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            aa_H_aa[ind1, ind2] += H.aa.vvvv[a, b, e, f]
                # A(ab) h1a(ae) * r_aa(ebj)
                for e in range(nactu_a):
                    jdet = idx_aa[e, b, j]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        phase = np.sign(jdet)
                        aa_H_aa[ind1, ind2] += H.a.vv[a, e] * phase
                    jdet = idx_aa[e, a, j]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        phase = np.sign(jdet)
                        aa_H_aa[ind1, ind2] -= H.a.vv[b, e] * phase
                # A(ab) h2a(bmje) * r_aa(aem)
                for e in range(nactu_a):
                    for m in range(noa - nacto_a, noa):
                        jdet = idx_aa[a, e, m]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            phase = np.sign(jdet)
                            aa_H_aa[ind1, ind2] += H.aa.voov[b, m, j, e] * phase
                        jdet = idx_aa[b, e, m]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            phase = np.sign(jdet)
                            aa_H_aa[ind1, ind2] -= H.aa.voov[a, m, j, e] * phase
                # A(ab) h2b(bmje) * r_ab(aem)
                for e in range(nactu_b):
                    for m in range(nob - nacto_b, nob):
                        jdet = idx_ab[a, e, m]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            aa_H_ab[ind1, ind2] += H.ab.voov[b, m, j, e]
                        jdet = idx_ab[b, e, m]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            aa_H_ab[ind1, ind2] -= H.ab.voov[a, m, j, e]
    ####################################################################################
    # ALPHA-BETA 2p-1h
    ####################################################################################
    ab_H_a = np.zeros((n2b, n1a))
    ab_H_aa = np.zeros((n2b, n2a))
    ab_H_ab = np.zeros((n2b, n2b))
    for a in range(nactu_a):
        for b in range(nactu_b):
            for j in range(nob - nacto_b, nob):
                idet = idx_ab[a, b, j]
                if idet == 0: continue
                ind1 = abs(idet) - 1
                # h2b(abej) * r_a(e)
                for e in range(nua):
                    jdet = idx_a[e]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        ab_H_a[ind1, ind2] += H.ab.vvvo[a, b, e, j]
                # h1a(ae) * r_ab(ebj)
                for e in range(nactu_a):
                    jdet = idx_ab[e, b, j]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        ab_H_ab[ind1, ind2] += H.a.vv[a, e]
                # h1b(be) * r_ab(aej)
                for e in range(nactu_b):
                    jdet = idx_ab[a, e, j]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        ab_H_ab[ind1, ind2] += H.b.vv[b, e]
                # -h1b(mj) * r_ab(abm)
                for m in range(nob - nacto_b, nob):
                    jdet = idx_ab[a, b, m]
                    if jdet != 0:
                        ind2 = abs(jdet) - 1
                        ab_H_ab[ind1, ind2] -= H.b.oo[m, j]
                # h2b(mbej) * r_aa(aem)
                for e in range(nactu_a):
                    for m in range(noa - nacto_a, noa):
                        jdet = idx_aa[a, e, m]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            phase = np.sign(jdet)
                            ab_H_aa[ind1, ind2] += H.ab.ovvo[m, b, e, j] * phase
                # h2c(bmje) * r_ab(aem)
                for e in range(nactu_b):
                    for m in range(nob - nacto_b, nob):
                        jdet = idx_ab[a, e, m]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            ab_H_ab[ind1, ind2] += H.bb.voov[b, m, j, e]
                # -h2b(amej) * r_ab(ebm)
                for e in range(nactu_a):
                    for m in range(nob - nacto_b, nob):
                        jdet = idx_ab[e, b, m]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            ab_H_ab[ind1, ind2] -= H.ab.vovo[a, m, e, j]
                # h2b(abef) * r_ab(efj)
                for e in range(nactu_a):
                    for f in range(nactu_b):
                        jdet = idx_ab[e, f, j]
                        if jdet != 0:
                            ind2 = abs(jdet) - 1
                            ab_H_ab[ind1, ind2] += H.ab.vvvv[a, b, e, f]
    # Assemble and return full matrix
    return np.concatenate(
        (np.concatenate((a_H_a, a_H_aa, a_H_ab), axis=1),
         np.concatenate((aa_H_a, aa_H_aa, aa_H_ab), axis=1),
         np.concatenate((ab_H_a, ab_H_aa, ab_H_ab), axis=1),
         ), axis=0
    )

def get_sz2(system, Ms):
    Ns = float((system.noccupied_alpha + Ms) - (system.noccupied_beta - Ms))
    sz = Ns / 2.0
    sz2 = (sz + 1.0) * sz
    return sz2

def build_s2matrix(system, nacto, nactu, idx_a, idx_aa, idx_ab):

    # orbital dimensions
    noa = system.noccupied_alpha
    nua = system.nunoccupied_alpha
    nob = system.noccupied_beta
    nub = system.nunoccupied_beta

    # set active space parameters
    nacto_a = min(nacto + (system.multiplicity - 1), system.noccupied_alpha)
    nacto_b = min(nacto, system.noccupied_beta)
    nactu_a = min(nactu, system.nunoccupied_alpha)
    nactu_b = min(nactu + (system.multiplicity - 1), system.nunoccupied_beta)

    def chi_beta(p):
        if p >= 0 and p < system.noccupied_beta:
            return 1.0
        else:
            return 0.0

    def pi_alpha(p):
        if p >= system.noccupied_alpha and p < system.nunoccupied_alpha + system.noccupied_alpha:
            return 1.0
        else:
            return 0.0

    return -1

def get_index_arrays(nacto, nactu, system, target_irrep):

    noa = system.noccupied_alpha
    nua = system.nunoccupied_alpha
    nob = system.noccupied_beta
    nub = system.nunoccupied_beta

    # set active space parameters
    nacto_a = min(nacto + (system.multiplicity - 1), noa)
    nacto_b = min(nacto, nob)
    nactu_a = min(nactu, nua)
    nactu_b = min(nactu + (system.multiplicity - 1), nub)

    if target_irrep is None:
        sym1 = lambda a: True
        sym2 = lambda a, b, j: True
    else:
        sym = lambda orbital_number: system.point_group_irrep_to_number[system.orbital_symmetries[orbital_number]]
        ref_sym = system.point_group_irrep_to_number[system.reference_symmetry]
        target_sym = system.point_group_irrep_to_number[target_irrep]

        sym1 = lambda a: sym(a) ^ ref_sym == target_sym
        sym2 = lambda a, b, j: sym(a) ^ sym(j) ^ sym(b) ^ ref_sym == target_sym

    ndim = 0
    idx_a = np.zeros(nua, dtype=np.int32)
    ct = 1
    for a in range(nua):
        if sym1(a + noa):
            idx_a[a] = ct
            ndim += 1
        ct += 1
    idx_aa = np.zeros((nua, nua, noa), dtype=np.int32)
    ct = 1
    for a in range(nactu_a):
        for b in range(a + 1, nactu_a):
            for j in range(noa - nacto_a, noa):
                if sym2(a + noa, b + noa, j):
                    idx_aa[a, b, j] = ct
                    idx_aa[b, a, j] = -ct
                    ndim += 1
                ct += 1
    idx_ab = np.zeros((nua, nub, nob), dtype=np.int32)
    ct = 1
    for a in range(nactu_a):
        for b in range(nactu_b):
            for j in range(nob - nacto_b, nob):
                if sym2(a + noa, b + nob, j):
                    idx_ab[a, b, j] = ct
                    ndim += 1
                ct += 1
    return idx_a, idx_aa, idx_ab, ndim
