import numpy as np

def cis(nroot,mult,H1A,H1B,H2A,H2B,H2C,sys):

    print('\n==================================++Entering CIS Routine++=================================\n')

    num_roots_total = sum(nroot)

    n1a = sys['Nocc_a']*sys['Nunocc_a']
    n1b = sys['Nocc_b']*sys['Nunocc_b']
    ndim = n1a+n1b

    H_cis = build_cis_hamiltonian(H1A,H1B,H2A,H2B,H2C,sys)
    omega_cis, V_cis = np.linalg.eig(H_cis)
    idx = np.argsort(omega_cis)
    omega_cis = omega_cis[idx]
    V_cis = V_cis[:,idx]

    print('CIS Eigenvalues:')
    for i in range(ndim):
        print('E{} = {}'.format(i+1,omega_cis[i]))
        print_amplitudes(V_cis[:,i],sys)

    S2 = build_s2_matrix(sys)
    eval_s2, V_s2 = np.linalg.eig(S2)
    idx = np.argsort(eval_s2)
    eval_s2 = eval_s2[idx]
    V_s2 = V_s2[:,idx]
    
    Ns2 = 0
    idx_s2 = []
    print('S2 Eigenvalues:')
    for i in range(ndim):
        sval = -0.5 + np.sqrt(0.25 + eval_s2[i])
        multval = 2*sval+1
        print('S{} = {}  (mult = {})'.format(i+1,sval,multval))
        if abs(multval - mult) < 1.0e-07:
            idx_s2.append(i)
            Ns2 += 1

    print('Dimension of spin subspace of multiplicity {} = {}'.format(mult,Ns2))
    Qs2 = ndim - Ns2
    W = np.zeros((ndim,Ns2))
    for i in range(Ns2):
        W[:,i] = V_s2[:,idx_s2[i]]

    G = np.einsum('Ku,Nv,Lu,Mv,LM->KN',W,W,W,W,H_cis,optimize=True)
    #G = np.einsum('Ku,Nv,Lu,Mv,LM->KN',V_s2,V_s2,V_s2,V_s2,H_cis,optimize=True)

    omega_cis, V_cis = np.linalg.eig(G)
    omega_cis = np.real(omega_cis)
    V_cis = np.real(V_cis)
    idx = np.argsort(omega_cis)
    omega_cis = omega_cis[idx]
    V_cis = V_cis[:,idx]
    print('Spin-adpated CIS eigenvalues:')
    for i in range(num_roots_total):
        print('E{} = {}'.format(i+1,omega_cis[i+Qs2]))
        print_amplitudes(V_cis[:,i+Qs2],sys)
    #for i in range(ndim):
    #    print('E{} = {}'.format(i+1,omega_cis[i]))
    #    print_amplitudes(V_cis[:,i],sys)
    #for i in range(num_roots_total):
    #    print('E{} = {}'.format(i+1,omega_cis[idx_s2[i]]))
    #    print_amplitudes(V_cis[:,idx_s2[i]],sys)

    return omega_cis, V_cis

def build_cis_hamiltonian(H1A,H1B,H2A,H2B,H2C,sys):

    n1a = sys['Nunocc_a']*sys['Nocc_a']
    n1b = sys['Nunocc_b']*sys['Nocc_b']

    Haa = np.zeros((n1a,n1a))
    ct1 = 0
    for a in range(sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            ct2 = 0
            for b in range(sys['Nunocc_a']):
                for j in range(sys['Nocc_a']):
                    Haa[ct1,ct2] = H1A['vv'][a,b]*(i==j) - H1A['oo'][j,i]*(a==b)\
                                    +H2A['voov'][a,j,i,b]
                    ct2 += 1
            ct1 += 1
    Hab = np.zeros((n1a,n1b))
    ct1 = 0
    for a in range(sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            ct2 = 0
            for b in range(sys['Nunocc_b']):
                for j in range(sys['Nocc_b']):
                    Hab[ct1,ct2] = H2B['voov'][a,j,i,b]
                    ct2 += 1
            ct1 += 1
    Hba = np.zeros((n1b,n1a))
    ct1 = 0
    for a in range(sys['Nunocc_b']):
        for i in range(sys['Nocc_b']):
            ct2 = 0
            for b in range(sys['Nunocc_a']):
                for j in range(sys['Nocc_a']):
                    Hba[ct1,ct2] = H2B['ovvo'][j,a,b,i]
                    ct2 += 1
            ct1 += 1
    Hbb = np.zeros((n1b,n1b))
    ct1 = 0
    for a in range(sys['Nunocc_b']):
        for i in range(sys['Nocc_b']):
            ct2 = 0
            for b in range(sys['Nunocc_b']):
                for j in range(sys['Nocc_b']):
                    Hbb[ct1,ct2] = H1B['vv'][a,b]*(i==j) - H1B['oo'][j,i]*(a==b)\
                                    +H2C['voov'][a,j,i,b]
                    ct2 += 1
            ct1 += 1
    return np.concatenate( (np.concatenate((Haa,Hab),axis=1),\
                            np.concatenate((Hba,Hbb),axis=1)), axis=0)

def build_s2_matrix(sys):

    def chi_beta(p):
        if p >=0 and p < sys['Nocc_b']:
            return 1.0
        else:
            return 0.0

    def pi_alpha(p):
        if p >= sys['Nocc_a'] and p < sys['Nunocc_a']+sys['Nocc_a']:
            return 1.0
        else:
            return 0.0

    n1a = sys['Nunocc_a']*sys['Nocc_a']
    n1b = sys['Nunocc_b']*sys['Nocc_b']

    Nelec = sys['Nelec']
    Na = sys['Nocc_a']
    Nb = sys['Nocc_b']
    Ns = float(Na - Nb)

    s0 = (Ns/2.0+1.0)*(Ns/2.0)

    Saa = np.zeros((n1a,n1a))
    ct1 = 0
    for a in range(sys['Nocc_a'],sys['Nocc_a']+sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            ct2 = 0
            for b in range(sys['Nocc_a'],sys['Nocc_a']+sys['Nunocc_a']):
                for j in range(sys['Nocc_a']):
                    Saa[ct1,ct2] = (s0+1.0*chi_beta(i))*(i==j)*(a==b)
                    ct2 += 1
            ct1 += 1
    Sab = np.zeros((n1a,n1b))
    ct1 = 0
    for a in range(sys['Nocc_a'],sys['Nocc_a']+sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            ct2 = 0
            for b in range(sys['Nocc_b'],sys['Nocc_b']+sys['Nunocc_b']):
                for j in range(sys['Nocc_b']):
                    #Sab[ct1,ct2] = -1.0*(i==j)*(a==b)*chi_beta(i)*pi_alpha(a)
                    Sab[ct1,ct2] = -1.0*(i==j)*(a==b)
                    ct2 += 1
            ct1 += 1
    Sba = np.zeros((n1b,n1a))
    ct1 = 0
    for a in range(sys['Nocc_b'],sys['Nocc_b']+sys['Nunocc_b']):
        for i in range(sys['Nocc_b']):
            ct2 = 0
            for b in range(sys['Nocc_a'],sys['Nocc_a']+sys['Nunocc_a']):
                for j in range(sys['Nocc_a']):
                    #Sba[ct1,ct2] = -1.0*(i==j)*(a==b)*chi_beta(i)*pi_alpha(a)
                    Sba[ct1,ct2] = -1.0*(i==j)*(a==b)
                    ct2 += 1
            ct1 += 1
    Sbb = np.zeros((n1b,n1b))
    ct1 = 0
    for a in range(sys['Nocc_b'],sys['Nocc_b']+sys['Nunocc_b']):
        for i in range(sys['Nocc_b']):
            ct2 = 0
            for b in range(sys['Nocc_b'],sys['Nocc_b']+sys['Nunocc_b']):
                for j in range(sys['Nocc_b']):
                    Sbb[ct1,ct2] = (s0+1.0*pi_alpha(a))*(i==j)*(a==b)
                    ct2 += 1
            ct1 += 1

    return np.concatenate( (np.concatenate((Saa,Sab),axis=1),\
                            np.concatenate((Sba,Sbb),axis=1)), axis=0)

def get_smp(sys):

    def chi_beta(p):
        if p >=0 and p < sys['Nocc_b']:
            return 1.0
        else:
            return 0.0

    def pi_beta(p):
        if p >= sys['Nocc_b'] and p < sys['Nocc_b']+sys['Nunocc_b']:
            return 1.0
        else:
            return 0.0

    def chi_alpha(p):
        if p >=0 and p < sys['Nocc_a']:
            return 1.0
        else:
            return 0.0

    def pi_alpha(p):
        if p >= sys['Nocc_a'] and p < sys['Nunocc_a']+sys['Nocc_a']:
            return 1.0
        else:
            return 0.0
    
    norb = sys['Nocc_a']+sys['Nunocc_a']
    noa = sys['Nocc_a']
    nob = sys['Nocc_b']
    nua = sys['Nunocc_a']
    nub = sys['Nunocc_b']

    S1A = np.zeros((norb,norb))
    # oo
    for i in range(noa):
        for j in range(noa):
            S1A[i,j] = -1.0*(i==j)*chi_beta(i)

    # The form of the S1B operator is 
    # Pi(A) * sum_A N[X_
    S1B = np.zeros((norb,norb))
    # vv
    for a in range(nob,norb):
        for b in range(nob,norb):
            S1B[a,b] = (a==b)*pi_alpha(a)

    # we have set up S2B in one-to-one correspondence with the normal-order
    # form N[ X_pb* X_pa X_qa* X_qb ]. The convention is <p~q|s|rs~>, so
    # <beta outgoing, alpha outgoing | alpha incoming, beta incoming>
    S2B = np.zeros((norb,norb,norb,norb))
    # oooo
    for p in range(nob):
        for q in range(noa):
            for r in range(noa):
                for s in range(nob):
                    S2B[p,q,r,s] = (p==r)*(q==s)*chi_beta(p)*chi_alpha(q)*chi_alpha(r)*chi_beta(s)
    # ooov
    for p in range(nob):
        for q in range(noa):
            for r in range(noa):
                for s in range(nob,norb):
                    S2B[p,q,r,s] = (p==r)*(q==s)*chi_beta(p)*chi_alpha(q)*chi_alpha(r)*pi_beta(s)
    # oovo
    for p in range(nob):
        for q in range(noa):
            for r in range(noa,norb):
                for s in range(nob):
                    S2B[p,q,r,s] = (p==r)*(q==s)*chi_beta(p)*chi_alpha(q)*pi_alpha(r)*chi_beta(s)
    # ovoo
    for p in range(nob):
        for q in range(noa,norb):
            for r in range(noa):
                for s in range(nob):
                    S2B[p,q,r,s] = (p==r)*(q==s)*chi_beta(p)*pi_alpha(q)*chi_alpha(r)*chi_beta(s)
    # vooo
    for p in range(nob,norb):
        for q in range(noa):
            for r in range(noa):
                for s in range(nob):
                    S2B[p,q,r,s] = (p==r)*(q==s)*pi_beta(p)*chi_alpha(q)*chi_alpha(r)*chi_beta(s)
    # vvoo
    for p in range(nob,norb):
        for q in range(noa,norb):
            for r in range(noa):
                for s in range(nob):
                    S2B[p,q,r,s] = (p==r)*(q==s)*pi_beta(p)*pi_alpha(q)*chi_alpha(r)*chi_beta(s)
    # vovo
    for p in range(nob,norb):
        for q in range(noa):
            for r in range(noa,norb):
                for s in range(nob):
                    S2B[p,q,r,s] = (p==r)*(q==s)*pi_beta(p)*chi_alpha(q)*pi_alpha(r)*chi_beta(s)
    # voov
    for p in range(nob,norb):
        for q in range(noa):
            for r in range(noa):
                for s in range(nob,norb):
                    S2B[p,q,r,s] = (p==r)*(q==s)*pi_beta(p)*chi_alpha(q)*chi_alpha(r)*pi_beta(s)
    # ovvo
    for p in range(nob):
        for q in range(noa,norb):
            for r in range(noa,norb):
                for s in range(nob):
                    S2B[p,q,r,s] = (p==r)*(q==s)*chi_beta(p)*pi_alpha(q)*pi_alpha(r)*chi_beta(s)
    # ovov
    for p in range(nob):
        for q in range(noa,norb):
            for r in range(noa):
                for s in range(nob,norb):
                    S2B[p,q,r,s] = (p==r)*(q==s)*chi_beta(p)*pi_alpha(q)*chi_alpha(r)*pi_beta(s)
    # oovv
    for p in range(nob):
        for q in range(noa):
            for r in range(noa,norb):
                for s in range(nob,norb):
                    S2B[p,q,r,s] = (p==r)*(q==s)*chi_beta(p)*chi_alpha(q)*pi_alpha(r)*pi_beta(s)
    # vvvo
    for p in range(nob,norb):
        for q in range(noa,norb):
            for r in range(noa,norb):
                for s in range(nob):
                    S2B[p,q,r,s] = (p==r)*(q==s)*pi_beta(p)*pi_alpha(q)*pi_alpha(r)*chi_beta(s)
    # vvov
    for p in range(nob,norb):
        for q in range(noa,norb):
            for r in range(noa):
                for s in range(nob,norb):
                    S2B[p,q,r,s] = (p==r)*(q==s)*pi_beta(p)*pi_alpha(q)*chi_alpha(r)*pi_beta(s)
    # vovv
    for p in range(nob,norb):
        for q in range(noa):
            for r in range(noa,norb):
                for s in range(nob,norb):
                    S2B[p,q,r,s] = (p==r)*(q==s)*pi_beta(p)*chi_alpha(q)*pi_alpha(r)*pi_beta(s)
    # ovvv
    for p in range(nob):
        for q in range(noa,norb):
            for r in range(noa,norb):
                for s in range(nob,norb):
                    S2B[p,q,r,s] = (p==r)*(q==s)*chi_beta(p)*pi_alpha(q)*pi_alpha(r)*pi_beta(s)
    # vvvv
    for p in range(nob,norb):
        for q in range(noa,norb):
            for r in range(noa,norb):
                for s in range(nob,norb):
                    S2B[p,q,r,s] = (p==r)*(q==s)*pi_beta(p)*pi_alpha(q)*pi_alpha(r)*pi_beta(s)

    return S1A, S1B, S2B

def build_s2_matrix_v2(sys):

    n1a = sys['Nunocc_a']*sys['Nocc_a']
    n1b = sys['Nunocc_b']*sys['Nocc_b']
    ndim = n1a + n1b

    Nelec = sys['Nelec']
    Na = sys['Nocc_a']
    Nb = sys['Nocc_b']
    Ns = float(Na - Nb)

    s0 = (Ns/2.0+1.0)*(Ns/2.0)

    s1a, s1b, s2b = get_smp(sys)

    Saa = np.zeros((n1a,n1a))
    ct1 = 0
    for a in range(sys['Nocc_a'],sys['Nocc_a']+sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            ct2 = 0
            for b in range(sys['Nocc_a'],sys['Nocc_a']+sys['Nunocc_a']):
                for j in range(sys['Nocc_a']):
                    Saa[ct1,ct2] = -s1a[j,i]*(a==b)
                    ct2 += 1
            ct1 += 1
    Sab = np.zeros((n1a,n1b))
    ct1 = 0
    for a in range(sys['Nocc_a'],sys['Nocc_a']+sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            ct2 = 0
            for b in range(sys['Nocc_b'],sys['Nocc_b']+sys['Nunocc_b']):
                for j in range(sys['Nocc_b']):
                    Sab[ct1,ct2] = -1.0*s2b[j,a,i,b]
                    ct2 += 1
            ct1 += 1
    Sba = np.zeros((n1b,n1a))
    ct1 = 0
    for a in range(sys['Nocc_b'],sys['Nocc_b']+sys['Nunocc_b']):
        for i in range(sys['Nocc_b']):
            ct2 = 0
            for b in range(sys['Nocc_a'],sys['Nocc_a']+sys['Nunocc_a']):
                for j in range(sys['Nocc_a']):
                    Sba[ct1,ct2] = -1.0*s2b[a,j,b,i]
                    ct2 += 1
            ct1 += 1
    Sbb = np.zeros((n1b,n1b))
    ct1 = 0
    for a in range(sys['Nocc_b'],sys['Nocc_b']+sys['Nunocc_b']):
        for i in range(sys['Nocc_b']):
            ct2 = 0
            for b in range(sys['Nocc_b'],sys['Nocc_b']+sys['Nunocc_b']):
                for j in range(sys['Nocc_b']):
                    Sbb[ct1,ct2] = s1b[a,b]*(i==j)
                    ct2 += 1
            ct1 += 1

    S2 = np.concatenate( (np.concatenate((Saa,Sab),axis=1),\
                            np.concatenate((Sba,Sbb),axis=1)), axis=0)

    return S2 + np.diag(s0*np.ones(ndim))


def print_amplitudes(R1,sys,nprint=5):

    n1a = sys['Nocc_a']*sys['Nunocc_a']
    n1b = sys['Nocc_b']*sys['Nunocc_b']

    idx = np.flip(np.argsort(abs(R1)))
    print('     Largest Singly Excited Amplitudes:')
    for n in range(nprint):
        if idx[n] < n1a:
            a,i = np.unravel_index(idx[n],(sys['Nunocc_a'],sys['Nocc_a']),order='C')
            if abs(R1[idx[n]]) < 0.1: continue
            print('      [{}]     {}A  ->  {}A     {:.6f}'.format(n+1,i+sys['Nfroz']+1,a+sys['Nfroz']+sys['Nocc_a']+1,R1[idx[n]]))
        else:
            a,i = np.unravel_index(idx[n]-n1a,(sys['Nunocc_b'],sys['Nocc_b']),order='C')
            if abs(R1[idx[n]]) < 0.1: continue
            print('      [{}]     {}B  ->  {}B     {:.6f}'.format(n+1,i+sys['Nfroz']+1,a+sys['Nocc_b']+sys['Nfroz']+1,R1[idx[n]]))
    return
