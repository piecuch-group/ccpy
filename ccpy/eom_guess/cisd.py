import numpy as np
import time
from ccpy.eom_guess.s2matrix import build_s2matrix_cisd, spin_adapt_guess

def run_diagonalization(system, H, multiplicity, roots_per_irrep, nacto, nactu, use_symmetry=True, debug=False):

    nroots_total = 0
    for key, value in roots_per_irrep.items():
        nroots_total += value

    noa, nob, nua, nub = H.ab.oovv.shape
    ndim = noa * nua + nob * nub + noa ** 2 * nua ** 2 + noa * nob * nua * nub + nob ** 2 * nub ** 2
    V = np.zeros((ndim, nroots_total))
    omega_guess = np.zeros(nroots_total)
    n_found = 0

    # print results of initial guess procedure
    print("   CISd initial guess routine")
    print("   --------------------------")
    print("   Multiplicity = ", multiplicity)
    print("   Active occupied alpha = ", min(nacto + (system.multiplicity - 1), noa))
    print("   Active occupied beta = ", min(nacto, nob))
    print("   Active unoccupied alpha = ", min(nactu, nua))
    print("   Active unoccupied beta = ", min(nactu + (system.multiplicity - 1), nub))

    t1 = time.time()
    S2mat = build_s2matrix_cisd(system, nacto, nactu)
    print("   Time requried for S2 matrix =", time.time() - t1, "seconds")

    for irrep, nroot in roots_per_irrep.items():
        if nroot == 0: continue
        if not use_symmetry: irrep = None
        t1 = time.time()
        Hmat = build_cisd_hamiltonian(H, system, nacto, nactu, irrep)
        t2 = time.time()
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

        print("   -----------------------------------")
        print("   Target symmetry irrep = ", irrep, f"({system.point_group})")
        print("   Dimension of eigenvalue problem = ", V_act.shape[0])
        print("   Time required for H matrix = ", t2 - t1, "seconds")
        for i in range(n_found - kout, n_found):
            print("   Eigenvalue of root", i + 1, " = ", np.round(omega_guess[i], 8))

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
    V_a_out = np.zeros((nua, noa))
    V_b_out = np.zeros((nub, nob))
    V_aa_out = np.zeros((nua, nua, noa, noa))
    V_ab_out = np.zeros((nua, nub, noa, nob))
    V_bb_out = np.zeros((nub, nub, nob, nob))
    # Start filling in the array
    offset = 0
    for a in range(nua):
        for i in range(noa):
            V_a_out[a, i] = V_in[offset]
            offset += 1
    for a in range(nub):
        for i in range(nob):
            V_b_out[a, i] = V_in[offset]
            offset += 1
    for a in range(nua):
        for b in range(a + 1, nua):
            for i in range(noa):
                for j in range(i + 1, noa):
                    if a < nactu_a and b < nactu_a and i >= noa - nacto_a and j >= noa - nacto_a:
                        V_aa_out[a, b, i, j] = V_in[offset]
                        V_aa_out[b, a, i, j] = -V_in[offset]
                        V_aa_out[a, b, j, i] = -V_in[offset]
                        V_aa_out[b, a, j, i] = V_in[offset]
                        offset += 1
    for a in range(nua):
        for b in range(nub):
            for i in range(noa):
                for j in range(nob):
                    if a < nactu_a and b < nactu_b and i >= noa - nacto_a and j >= nob - nacto_b:
                        V_ab_out[a, b, i, j] = V_in[offset]
                        offset += 1
    for a in range(nub):
        for b in range(a + 1, nub):
            for i in range(nob):
                for j in range(i + 1, nob):
                    if a < nactu_b and b < nactu_b and i >= nob - nacto_b and j >= nob - nacto_b:
                        V_bb_out[a, b, i, j] = V_in[offset]
                        V_bb_out[b, a, i, j] = -V_in[offset]
                        V_bb_out[a, b, j, i] = -V_in[offset]
                        V_bb_out[b, a, j, i] = V_in[offset]
                        offset += 1
    return np.hstack((V_a_out.flatten(), V_b_out.flatten(), V_aa_out.flatten(), V_ab_out.flatten(), V_bb_out.flatten()))

def build_cisd_hamiltonian(H, system, nacto, nactu, target_irrep):

    noa, nob, nua, nub = H.ab.oovv.shape

    # set active space parameters
    nacto_a = min(nacto + (system.multiplicity - 1), noa)
    nacto_b = min(nacto, nob)
    nactu_a = min(nactu, nua)
    nactu_b = min(nactu + (system.multiplicity - 1), nub)

    # set dimensions of CISD problem
    n1a = noa * nua
    n1b = nob * nub
    n2a = int(nacto_a * (nacto_a - 1) / 2 * nactu_a * (nactu_a - 1) / 2)
    n2b = nacto_a * nacto_b * nactu_a * nactu_b
    n2c = int(nacto_b * (nacto_b - 1) / 2 * nactu_b * (nactu_b - 1) / 2)
    # total dimension
    ndim = n1a + n1b + n2a + n2b + n2c

    idx_a, idx_b, idx_aa, idx_ab, idx_bb = get_index_arrays(nacto, nactu, system, target_irrep)

    ####################################################################################
    # ALPHA SINGLES
    ####################################################################################
    a_H_a = np.zeros((n1a, n1a))
    a_H_b = np.zeros((n1a, n1b))
    a_H_aa = np.zeros((n1a, n2a))
    a_H_ab = np.zeros((n1a, n2b))
    for a in range(nua):
        for i in range(noa):
            idet = idx_a[a, i]
            if idet == 0: continue
            I = abs(idet) - 1
            # -h1a(mi) * r1a(am)
            for m in range(noa):
                jdet = idx_a[a, m]
                if jdet == 0: continue
                J = abs(jdet) - 1
                a_H_a[I, J] -= H.a.oo[m, i]
            # h1a(ae) * r1a(ei)
            for e in range(nua):
                jdet = idx_a[e, i]
                if jdet == 0: continue
                J = abs(jdet) - 1
                a_H_a[I, J] += H.a.vv[a, e]
            # h2a(amie) * r1a(em)
            for e in range(nua):
                for m in range(noa):
                    jdet = idx_a[e, m]
                    if jdet == 0: continue
                    J = abs(jdet) - 1
                    a_H_a[I, J] += H.aa.voov[a, m, i, e]
            # h2b(amie) * r1b(em)
            for e in range(nub):
                for m in range(nob):
                    jdet = idx_b[e, m]
                    if jdet == 0: continue
                    J = abs(jdet) - 1
                    a_H_b[I, J] += H.ab.voov[a, m, i, e]
            # h1a(me) * r2a(aeim)
            for e in range(nactu_a):
                for m in range(noa - nacto_a, noa):
                    jdet = idx_aa[a, e, i, m]
                    if jdet != 0:
                        J = abs(jdet) - 1
                        phase = np.sign(jdet)
                        a_H_aa[I, J] += H.a.ov[m, e] * phase
            # -1/2 h2a(mnif) * r2a(afmn)
            for m in range(noa - nacto_a, noa):
                for n in range(m + 1, noa):
                    for f in range(nactu_a):
                        jdet = idx_aa[a, f, m, n]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            a_H_aa[I, J] -= H.aa.ooov[m, n, i, f] * phase
            # 1/2 h2a(anef) * r2a(efin)
            for e in range(nactu_a):
                for f in range(e + 1, nactu_a):
                    for n in range(noa - nacto_a, noa):
                        jdet = idx_aa[e, f, i, n]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            a_H_aa[I, J] += H.aa.vovv[a, n, e, f] * phase
            # h1b(me) * r2b(aeim)
            for e in range(nactu_b):
                for m in range(nob - nacto_b, nob):
                    jdet = idx_ab[a, e, i, m]
                    if jdet != 0:
                        J = abs(jdet) - 1
                        a_H_ab[I, J] += H.b.ov[m, e]
            # -h2b(mnif) * r2b(afmn)
            for m in range(noa - nacto_a, noa):
                for n in range(nob - nacto_b, nob):
                    for f in range(nactu_b):
                        jdet = idx_ab[a, f, m, n]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            a_H_ab[I, J] -= H.ab.ooov[m, n, i, f]
            # h2b(anef) * r2b(efin)
            for e in range(nactu_a):
                for f in range(nactu_b):
                    for n in range(nob - nacto_b, nob):
                        jdet = idx_ab[e, f, i, n]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            a_H_ab[I, J] += H.ab.vovv[a, n, e, f]
    ####################################################################################
    # BETA SINGLES
    ####################################################################################
    b_H_a = np.zeros((n1b, n1a))
    b_H_b = np.zeros((n1b, n1b))
    b_H_ab = np.zeros((n1b, n2b))
    b_H_bb = np.zeros((n1b, n2c))
    for a in range(nub):
        for i in range(nob):
            idet = idx_b[a, i]
            if idet == 0: continue
            I = abs(idet) - 1
            # -h1b(mi) * r1b(am)
            for m in range(nob):
                jdet = idx_b[a, m]
                if jdet == 0: continue
                J = abs(jdet) - 1
                b_H_b[I, J] -= H.b.oo[m, i]
            # h1b(ae) * r1b(ei)
            for e in range(nub):
                jdet = idx_b[e, i]
                if jdet == 0: continue
                J = abs(jdet) - 1
                b_H_b[I, J] += H.b.vv[a, e]
            # h2c(amie) * r1b(em)
            for e in range(nub):
                for m in range(nob):
                    jdet = idx_b[e, m]
                    if jdet == 0: continue
                    J = abs(jdet) - 1
                    b_H_b[I, J] += H.bb.voov[a, m, i, e]
            # h2b(maei) * r1a(em)
            for e in range(nua):
                for m in range(noa):
                    jdet = idx_a[e, m]
                    if jdet == 0: continue
                    J = abs(jdet) - 1
                    b_H_a[I, J] += H.ab.ovvo[m, a, e, i]
            # h1b(me) * r2c(aeim)
            for e in range(nactu_b):
                for m in range(nob - nacto_b, nob):
                    jdet = idx_bb[a, e, i, m]
                    if jdet != 0:
                        J = abs(jdet) - 1
                        phase = np.sign(jdet)
                        b_H_bb[I, J] += H.b.ov[m, e] * phase
            # 1/2 h2c(anef) * r2c(efin)
            for e in range(nactu_b):
                for f in range(e + 1, nactu_b):
                    for n in range(nob - nacto_b, nob):
                        jdet = idx_bb[e, f, i, n]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            b_H_bb[I, J] += H.bb.vovv[a, n, e, f] * phase
            # -1/2 h2c(mnif) * r2c(afmn)
            for m in range(nob - nacto_b, nob):
                for n in range(m + 1, nob):
                    for f in range(nactu_b):
                        jdet = idx_bb[a, f, m, n]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            b_H_bb[I, J] -= H.bb.ooov[m, n, i, f] * phase
            # h1a(me) * r2b(eami)
            for e in range(nactu_a):
                for m in range(noa - nacto_a, noa):
                    jdet = idx_ab[e, a, m, i]
                    if jdet != 0:
                        J = abs(jdet) - 1
                        b_H_ab[I, J] += H.a.ov[m, e]
            # -h2b(nmfi) * r2b(fanm)
            for m in range(nob - nacto_b, nob):
                for n in range(noa - nacto_a, noa):
                    for f in range(nactu_a):
                        jdet = idx_ab[f, a, n, m]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            b_H_ab[I, J] -= H.ab.oovo[n, m, f, i]
            # h2b(nafe) * r2b(feni)
            for e in range(nactu_b):
                for f in range(nactu_a):
                    for n in range(noa - nacto_a, noa):
                        jdet = idx_ab[f, e, n, i]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            b_H_ab[I, J] += H.ab.ovvv[n, a, f, e]
    ####################################################################################
    # ALPHA-ALPHA DOUBLES
    ####################################################################################
    aa_H_a = np.zeros((n2a, n1a))
    aa_H_aa = np.zeros((n2a, n2a))
    aa_H_ab = np.zeros((n2a, n2b))
    for a in range(nactu_a):
        for b in range(a + 1, nactu_a):
            for i in range(noa - nacto_a, noa):
                for j in range(i + 1, noa):
                    idet = idx_aa[a, b, i, j]
                    if idet == 0: continue
                    I = abs(idet) - 1
                    # -A(ab) h2a(amij) * r1a(bm)
                    for m in range(noa):
                        # (1)
                        jdet = idx_a[b, m]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            aa_H_a[I, J] -= H.aa.vooo[a, m, i, j]
                        # (ab)
                        jdet = idx_a[a, m]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            aa_H_a[I, J] += H.aa.vooo[b, m, i, j]
                    # A(ij) h2a(abie) * r1a(ej)
                    for e in range(nua):
                        # (1)
                        jdet = idx_a[e, i]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            aa_H_a[I, J] += H.aa.vvov[b, a, j, e]
                        # (ij)
                        jdet = idx_a[e, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            aa_H_a[I, J] -= H.aa.vvov[b, a, i, e]
                    # -A(ij) h1a(mi) * r2a(abmj)
                    for m in range(noa - nacto_a, noa):
                        # (1)
                        jdet = idx_aa[a, b, m, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            aa_H_aa[I, J] -= H.a.oo[m, i] * phase
                        # (ij)
                        jdet = idx_aa[a, b, m, i]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            aa_H_aa[I, J] += H.a.oo[m, j] * phase
                    # A(ab) h1a(ae) * r2a(ebij)
                    for e in range(nactu_a):
                        # (1)
                        jdet = idx_aa[e, b, i, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            aa_H_aa[I, J] += H.a.vv[a, e] * phase
                        # (ab)
                        jdet = idx_aa[e, a, i, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            aa_H_aa[I, J] -= H.a.vv[b, e] * phase
                    # 1/2 h2a(mnij) * r2a(abmn)
                    for m in range(noa - nacto_a, noa):
                        for n in range(m + 1, noa):
                            jdet = idx_aa[a, b, m, n]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                aa_H_aa[I, J] += H.aa.oooo[m, n, i, j] * phase
                    # 1/2 h2a(abef) * r2a(efij)
                    for e in range(nactu_a):
                        for f in range(e + 1, nactu_a):
                            jdet = idx_aa[e, f, i, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                aa_H_aa[I, J] += H.aa.vvvv[a, b, e, f] * phase
                    # A(ij)A(ab) h2a(amie) * r2a(ebmj)
                    for e in range(nactu_a):
                        for m in range(noa - nacto_a, noa):
                            # (1)
                            jdet = idx_aa[e, b, m, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                aa_H_aa[I, J] += H.aa.voov[a, m, i, e] * phase
                            # (ij)
                            jdet = idx_aa[e, b, m, i]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                aa_H_aa[I, J] -= H.aa.voov[a, m, j, e] * phase
                            # (ab)
                            jdet = idx_aa[e, a, m, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                aa_H_aa[I, J] -= H.aa.voov[b, m, i, e] * phase
                            # (ij)(ab)
                            jdet = idx_aa[e, a, m, i]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                aa_H_aa[I, J] += H.aa.voov[b, m, j, e] * phase
                    # A(ij)A(ab) h2b(bmje) * r2b(aeim)
                    for e in range(nactu_b):
                        for m in range(nob - nacto_b, nob):
                            # (1)
                            jdet = idx_ab[a, e, i, m]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                aa_H_ab[I, J] += H.ab.voov[b, m, j, e]
                            # (ab)
                            jdet = idx_ab[b, e, i, m]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                aa_H_ab[I, J] -= H.ab.voov[a, m, j, e]
                            # (ij)
                            jdet = idx_ab[a, e, j, m]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                aa_H_ab[I, J] -= H.ab.voov[b, m, i, e]
                            # (ij)(ab)
                            jdet = idx_ab[b, e, j, m]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                aa_H_ab[I, J] += H.ab.voov[a, m, i, e]
    ####################################################################################
    # ALPHA-BETA DOUBLES
    ####################################################################################
    ab_H_a = np.zeros((n2b, n1a))
    ab_H_b = np.zeros((n2b, n1b))
    ab_H_aa = np.zeros((n2b, n2a))
    ab_H_ab = np.zeros((n2b, n2b))
    ab_H_bb = np.zeros((n2b, n2c))
    for a in range(nactu_a):
        for b in range(nactu_b):
            for i in range(noa - nacto_a, noa):
                for j in range(nob - nacto_b, nob):
                    idet = idx_ab[a, b, i, j]
                    if idet == 0: continue
                    I = abs(idet) - 1
                    # -h2b(mbij) * r1a(am)
                    for m in range(noa):
                        jdet = idx_a[a, m]
                        if jdet == 0: continue
                        J = abs(jdet) - 1
                        ab_H_a[I, J] -= H.ab.ovoo[m, b, i, j]
                    # h2b(abej) * r1a(ei)
                    for e in range(nua):
                        jdet = idx_a[e, i]
                        if jdet == 0: continue
                        J = abs(jdet) - 1
                        ab_H_a[I, J] += H.ab.vvvo[a, b, e, j]
                    # -h2b(amij) * r1b(bm)
                    for m in range(nob):
                        jdet = idx_b[b, m]
                        if jdet == 0: continue
                        J = abs(jdet) - 1
                        ab_H_b[I, J] -= H.ab.vooo[a, m, i, j]
                    # h2b(abie) * r1b(ej)
                    for e in range(nub):
                        jdet = idx_b[e, j]
                        if jdet == 0: continue
                        J = abs(jdet) - 1
                        ab_H_b[I, J] += H.ab.vvov[a, b, i, e]
                    # h2b(mbej) * r2a(aeim)
                    for e in range(nactu_a):
                        for m in range(noa - nacto_a, noa):
                            jdet = idx_aa[a, e, i, m]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                ab_H_aa[I, J] += H.ab.ovvo[m, b, e, j] * phase
                    # -h1a(mi) * r2b(abmj)
                    for m in range(noa - nacto_a, noa):
                        jdet = idx_ab[a, b, m, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            ab_H_ab[I, J] -= H.a.oo[m, i]
                    # -h1b(mj) * r2b(abim)
                    for m in range(nob - nacto_b, nob):
                        jdet = idx_ab[a, b, i, m]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            ab_H_ab[I, J] -= H.b.oo[m, j]
                    # h1a(ae) * r2b(ebij)
                    for e in range(nactu_a):
                        jdet = idx_ab[e, b, i, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            ab_H_ab[I, J] += H.a.vv[a, e]
                    # h1a(be) * r2b(aeij)
                    for e in range(nactu_b):
                        jdet = idx_ab[a, e, i, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            ab_H_ab[I, J] += H.b.vv[b, e]
                    # h2b(mnij) * r2b(abmn)
                    for m in range(noa - nacto_a, noa):
                        for n in range(nob - nacto_b, nob):
                            jdet = idx_ab[a, b, m, n]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                ab_H_ab[I, J] += H.ab.oooo[m, n, i, j]
                    # h2b(abef) * r2b(efij)
                    for e in range(nactu_a):
                        for f in range(nactu_b):
                            jdet = idx_ab[e, f, i, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                ab_H_ab[I, J] += H.ab.vvvv[a, b, e, f]
                    # h2a(amie) * r2b(ebmj)
                    for e in range(nactu_a):
                        for m in range(noa - nacto_a, noa):
                            jdet = idx_ab[e, b, m, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                ab_H_ab[I, J] += H.aa.voov[a, m, i, e]
                    # h2c(bmje) * r2b(aeim)
                    for e in range(nactu_b):
                        for m in range(nob - nacto_b, nob):
                            jdet = idx_ab[a, e, i, m]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                ab_H_ab[I, J] += H.bb.voov[b, m, j, e]
                    # -h2b(amej) * r2b(ebim)
                    for e in range(nactu_a):
                        for m in range(nob - nacto_b, nob):
                            jdet = idx_ab[e, b, i, m]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                ab_H_ab[I, J] -= H.ab.vovo[a, m, e, j]
                    # -h2b(mbie) * r2b(aemj)
                    for e in range(nactu_b):
                        for m in range(noa - nacto_a, noa):
                            jdet = idx_ab[a, e, m, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                ab_H_ab[I, J] -= H.ab.ovov[m, b, i, e]
                    # h2b(amie) * r2c(bejm)
                    for e in range(nactu_b):
                        for m in range(nob - nacto_b, nob):
                            jdet = idx_bb[b, e, j, m]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                ab_H_bb[I, J] += H.ab.voov[a, m, i, e] * phase
    ####################################################################################
    # BETA-BETA DOUBLES
    ####################################################################################
    bb_H_b = np.zeros((n2c, n1b))
    bb_H_ab = np.zeros((n2c, n2b))
    bb_H_bb = np.zeros((n2c, n2c))
    for a in range(nactu_b):
        for b in range(a + 1, nactu_b):
            for i in range(nob - nacto_b, nob):
                for j in range(i + 1, nob):
                    idet = idx_bb[a, b, i, j]
                    if idet == 0: continue
                    I = abs(idet) - 1
                    # -A(ab) h2c(amij) * r1b(bm)
                    for m in range(nob):
                        # (1)
                        jdet = idx_b[b, m]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            bb_H_b[I, J] -= H.bb.vooo[a, m, i, j]
                        # (ab)
                        jdet = idx_b[a, m]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            bb_H_b[I, J] += H.bb.vooo[b, m, i, j]
                    # A(ij) h2c(abie) * r1b(ej)
                    for e in range(nub):
                        # (1)
                        jdet = idx_b[e, i]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            bb_H_b[I, J] += H.bb.vvov[b, a, j, e]
                        # (ij)
                        jdet = idx_b[e, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            bb_H_b[I, J] -= H.bb.vvov[b, a, i, e]
                    # -A(ij) h1b(mi) * r2c(abmj)
                    for m in range(nob - nacto_b, nob):
                        # (1)
                        jdet = idx_bb[a, b, m, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            bb_H_bb[I, J] -= H.b.oo[m, i] * phase
                        # (ij)
                        jdet = idx_bb[a, b, m, i]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            bb_H_bb[I, J] += H.b.oo[m, j] * phase
                    # A(ab) h1b(ae) * r2c(ebij)
                    for e in range(nactu_b):
                        # (1)
                        jdet = idx_bb[e, b, i, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            bb_H_bb[I, J] += H.b.vv[a, e] * phase
                        # (ab)
                        jdet = idx_bb[e, a, i, j]
                        if jdet != 0:
                            J = abs(jdet) - 1
                            phase = np.sign(jdet)
                            bb_H_bb[I, J] -= H.b.vv[b, e] * phase
                    # 1/2 h2c(mnij) * r2c(abmn)
                    for m in range(nob - nacto_b, nob):
                        for n in range(m + 1, nob):
                            jdet = idx_bb[a, b, m, n]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                bb_H_bb[I, J] += H.bb.oooo[m, n, i, j] * phase
                    # 1/2 h2c(abef) * r2c(efij)
                    for e in range(nactu_b):
                        for f in range(e + 1, nactu_b):
                            jdet = idx_bb[e, f, i, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                bb_H_bb[I, J] += H.bb.vvvv[a, b, e, f] * phase
                    # A(ij)A(ab) h2c(amie) * r2c(ebmj)
                    for e in range(nactu_b):
                        for m in range(nob - nacto_b, nob):
                            # (1)
                            jdet = idx_bb[e, b, m, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                bb_H_bb[I, J] += H.bb.voov[a, m, i, e] * phase
                            # (ij)
                            jdet = idx_bb[e, b, m, i]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                bb_H_bb[I, J] -= H.bb.voov[a, m, j, e] * phase
                            # (ab)
                            jdet = idx_bb[e, a, m, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                bb_H_bb[I, J] -= H.bb.voov[b, m, i, e] * phase
                            # (ij)(ab)
                            jdet = idx_bb[e, a, m, i]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                phase = np.sign(jdet)
                                bb_H_bb[I, J] += H.bb.voov[b, m, j, e] * phase
                    # A(ij)A(ab) h2b(mbej) * r2b(eami)
                    for e in range(nactu_a):
                        for m in range(noa - nacto_a, noa):
                            # (1)
                            jdet = idx_ab[e, a, m, i]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                bb_H_ab[I, J] += H.ab.ovvo[m, b, e, j]
                            # (ab)
                            jdet = idx_ab[e, b, m, i]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                bb_H_ab[I, J] -= H.ab.ovvo[m, a, e, j]
                            # (ij)
                            jdet = idx_ab[e, a, m, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                bb_H_ab[I, J] -= H.ab.ovvo[m, b, e, i]
                            # (ab)(ij)
                            jdet = idx_ab[e, b, m, j]
                            if jdet != 0:
                                J = abs(jdet) - 1
                                bb_H_ab[I, J] += H.ab.ovvo[m, a, e, i]

    # Zero blocks
    a_H_bb = np.zeros((n1a, n2c))
    b_H_aa = np.zeros((n1b, n2a))
    aa_H_bb = np.zeros((n2a, n2c))
    # Assemble and return full matrix
    return np.concatenate(
        (np.concatenate((a_H_a, a_H_b, a_H_aa, a_H_ab, a_H_bb), axis=1),
         np.concatenate((b_H_a, b_H_b, b_H_aa, b_H_ab, b_H_bb), axis=1),
         np.concatenate((aa_H_a, b_H_aa.T, aa_H_aa, aa_H_ab, aa_H_bb), axis=1),
         np.concatenate((ab_H_a, ab_H_b, ab_H_aa, ab_H_ab, ab_H_bb), axis=1),
         np.concatenate((a_H_bb.T, bb_H_b, aa_H_bb.T, bb_H_ab, bb_H_bb), axis=1)
         ), axis=0
    )

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
        sym1 = lambda a, i: True
        sym2 = lambda a, b, i, j: True
    else:
        sym = lambda orbital_number: system.point_group_irrep_to_number[system.orbital_symmetries[orbital_number]]
        ref_sym = system.point_group_irrep_to_number[system.reference_symmetry]
        target_sym = system.point_group_irrep_to_number[target_irrep]

        sym1 = lambda a, i: sym(i) ^ sym(a) ^ ref_sym == target_sym
        sym2 = lambda a, b, i, j: sym(i) ^ sym(a) ^ sym(j) ^ sym(b) ^ ref_sym == target_sym

    idx_a = np.zeros((nua, noa), dtype=np.int32)
    ct = 1
    for a in range(nua):
        for i in range(noa):
            if sym1(a + noa, i):
                idx_a[a, i] = ct
            ct += 1
    idx_b = np.zeros((nub, nob), dtype=np.int32)
    ct = 1
    for a in range(nub):
        for i in range(nob):
            if sym1(a + nob, i):
                idx_b[a, i] = ct
            ct += 1
    idx_aa = np.zeros((nua, nua, noa, noa), dtype=np.int32)
    ct = 1
    for a in range(nactu_a):
        for b in range(a + 1, nactu_a):
            for i in range(noa - nacto_a, noa):
                for j in range(i + 1, noa):
                    if sym2(a + noa, b + noa, i, j):
                        idx_aa[a, b, i, j] = ct
                        idx_aa[b, a, i, j] = -ct
                        idx_aa[a, b, j, i] = -ct
                        idx_aa[b, a, j, i] = ct
                    ct += 1
    idx_ab = np.zeros((nua, nub, noa, nob), dtype=np.int32)
    ct = 1
    for a in range(nactu_a):
        for b in range(nactu_b):
            for i in range(noa - nacto_a, noa):
                for j in range(nob - nacto_b, nob):
                    if sym2(a + noa, b + nob, i, j):
                        idx_ab[a, b, i, j] = ct
                    ct += 1
    idx_bb = np.zeros((nub, nub, nob, nob), dtype=np.int32)
    ct = 1
    for a in range(nactu_b):
        for b in range(a + 1, nactu_b):
            for i in range(nob - nacto_b, nob):
                for j in range(i + 1, nob):
                    if sym2(a + nob, b + nob, i, j):
                        idx_bb[a, b, i, j] = ct
                        idx_bb[b, a, i, j] = -ct
                        idx_bb[a, b, j, i] = -ct
                        idx_bb[b, a, j, i] = ct
                    ct += 1

    return idx_a, idx_b, idx_aa, idx_ab, idx_bb
