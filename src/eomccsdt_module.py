import numpy as np
from HBar_module import HBar_CCSD
import cc_loops

def eomccsdt(nroot,H1A,H1B,H2A,H2B,H2C,cc_t,ints,sys,initial_guess='cis',tol=1.0e-06,maxit=80):

    print('\n==================================++Entering EOM-CCSDT Routine++=================================\n')

    if initial_guess == 'cis':
        n1a = sys['Nocc_a'] * sys['Nunocc_a']
        n1b = sys['Nocc_b'] * sys['Nunocc_b']

        Cvec, omega_cis = cis(nroot,ints,sys)
        C1A = Cvec[:n1a,:]
        C1B = Cvec[n1a:,:]

        B0 = np.zeros((n1a+n1b,nroot))
        E0 = np.zeros(nroot)

        # locate only singlet roots (for RHF reference)
        ct = 0
        for i in range(len(omega_cis)):
            chk = np.linalg.norm(C1A[:,i] - C1B[:,i])
            if abs(chk) < 1.0e-09:
                B0[:,ct] = Cvec[:,i]
                E0[ct] = omega_cis[i]
                if ct+1 == nroot:
                    break
                ct += 1
        else:
            print('Could not find {} singlet roots in CIS guess!'.format(nroot))

        print('Initial CIS energies:')
        for i in range(nroot):
                print('Root - {}     E = {:.10f}    ({:.10f})'.format(i+1,E0[i],E0[i]+ints['Escf']))
        print('')
        n_doubles = sys['Nocc_a']**2*sys['Nunocc_a']**2\
                    +sys['Nocc_a']*sys['Nocc_b']*sys['Nunocc_a']*sys['Nunocc_b']\
                    +sys['Nocc_b']**2*sys['Nunocc_b']**2
        n_triples=  sys['Nocc_a']**3*sys['Nunocc_a']**3\
                    +sys['Nocc_a']**2*sys['Nocc_b']*sys['Nunocc_a']**2*sys['Nunocc_b']\
                    +sys['Nocc_a']*sys['Nocc_b']**2*sys['Nunocc_a']*sys['Nunocc_b']**2\
                    +sys['Nocc_b']**3*sys['Nunocc_b']**3
        ZEROS_DOUBLES = np.zeros((n_doubles,nroot))
        ZEROS_TRIPLES = np.zeros((n_triples,nroot))
        B0 = np.concatenate((B0,ZEROS_DOUBLES,ZEROS_TRIPLES),axis=0)

    Rvec, omega, is_converged = davidson_solver(H1A,H1B,H2A,H2B,H2C,ints,cc_t,nroot,B0,E0,sys,maxit,tol)
    
    cc_t['r1a'] = [None]*len(omega)
    cc_t['r1b'] = [None]*len(omega)
    cc_t['r2a'] = [None]*len(omega)
    cc_t['r2b'] = [None]*len(omega)
    cc_t['r2c'] = [None]*len(omega)
    cc_t['r3a'] = [None]*len(omega)
    cc_t['r3b'] = [None]*len(omega)
    cc_t['r3c'] = [None]*len(omega)
    cc_t['r3d'] = [None]*len(omega)
    cc_t['r0'] = [None]*len(omega)

    print('Summary of EOMCCSDT:')
    Eccsdt = ints['Escf'] + calc_cc_energy(cc_t,ints)
    for i in range(len(omega)):
        r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d = unflatten_R(Rvec[:,i],sys)
        r0 = calc_r0(r1a,r1b,r2a,r2b,r2c,H1A,H1B,ints,omega[i])  
        cc_t['r1a'][i] = r1a
        cc_t['r1b'][i] = r1b 
        cc_t['r2a'][i] = r2a
        cc_t['r2b'][i] = r2b
        cc_t['r2c'][i] = r2c
        cc_t['r3a'][i] = r3a
        cc_t['r3b'][i] = r3b
        cc_t['r3c'][i] = r3c
        cc_t['r3d'][i] = r3d
        cc_t['r0'][i] = r0
        if is_converged[i]:
            tmp = 'CONVERGED'
        else:
            tmp = 'NOT CONVERGED'
        print('   Root - {}    E = {}    omega = {:.10f}    r0 = {:.10f}    [{}]'\
                        .format(i+1,omega[i]+Eccsdt,omega[i],r0,tmp))

    return cc_t, omega

def davidson_solver(H1A,H1B,H2A,H2B,H2C,ints,cc_t,nroot,B0,E0,sys,maxit,tol):

    noa = H1A['ov'].shape[0]
    nob = H1B['ov'].shape[0]
    nua = H1A['ov'].shape[1]
    nub = H1B['ov'].shape[1]

    ndim = noa*nua + nob*nub\
           +noa**2*nua**2 + noa*nob*nua*nub + nob**2*nub**2\
           +noa**3*nua**3 + noa**2*nob*nua**2*nub + noa*nob**2*nua*nub**2 + nob**3*nub**3

    Rvec = np.zeros((ndim,nroot))
    is_converged = [False] * nroot
    omega = np.zeros(nroot)
    residuals = np.zeros(nroot)

    for iroot in range(nroot):

        print('Solving for root - {}'.format(iroot+1))
        print('--------------------------------------------------------------------------------')
        B = B0[:,iroot][:,np.newaxis]

        sigma = np.zeros((ndim,maxit))
    
        omega[iroot] = E0[iroot]
        for it in range(maxit):

            omega_old = omega[iroot]

            sigma[:,it] = HR(B[:,it],cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)

            G = np.dot(B.T,sigma[:,:it+1])
            e, alpha = np.linalg.eig(G)

            # select root based on maximum overlap with initial guess
            # < b0 | V_i > = < b0 | \sum_k alpha_{ik} |b_k>
            # = \sum_k alpha_{ik} < b0 | b_k > = \sum_k alpha_{i0}
            idx = np.argsort( abs(alpha[0,:]) )
            omega[iroot] = np.real(e[idx[-1]])
            alpha = np.real(alpha[:,idx[-1]])
            Rvec[:,iroot] = np.dot(B,alpha)

            # calculate residual vector
            q = np.dot(sigma[:,:it+1],alpha) - omega[iroot]*Rvec[:,iroot]
            residuals[iroot] = np.linalg.norm(q)
            deltaE = omega[iroot] - omega_old

            print('   Iter - {}      e = {:.10f}       |r| = {:.10f}      de = {:.10f}'.\
                            format(it+1,omega[iroot],residuals[iroot],deltaE))

            if residuals[iroot] < tol and abs(deltaE) < tol:
                is_converged[iroot] = True
                break
            
            # update residual vector
            q1a,q1b,q2a,q2b,q2c,q3a,q3b,q3c,q3d = unflatten_R(q,sys)
            q1a,q1b,q2a,q2b,q2c,q3a,q3b,q3c,q3d = cc_loops.cc_loops.update_r_ccsdt(q1a,q1b,q2a,q2b,q2c,q3a,q3b,q3c,q3d,\
                            omega[iroot],
                            H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],0.0,\
                            sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'])
            q = flatten_R(q1a,q1b,q2a,q2b,q2c,q3a,q3b,q3c,q3d)
            q *= 1.0/np.linalg.norm(q)
            q = orthogonalize(q,B)
            q *= 1.0/np.linalg.norm(q)

            B = np.concatenate((B,q[:,np.newaxis]),axis=1)

        if is_converged[iroot]:
            print('Converged root {}'.format(iroot+1))
        else:
            print('Failed to converge root {}'.format(iroot+1))
        print('')

    return Rvec, omega, is_converged

def calc_r0(r1a,r1b,r2a,r2b,r2c,H1A,H1B,ints,omega):

    r0 = 0.0
    r0 += np.einsum('me,em->',H1A['ov'],r1a,optimize=True)
    r0 += np.einsum('me,em->',H1B['ov'],r1b,optimize=True)
    r0 += 0.25*np.einsum('mnef,efmn->',ints['vA']['oovv'],r2a,optimize=True)
    r0 += np.einsum('mnef,efmn->',ints['vB']['oovv'],r2b,optimize=True)
    r0 += 0.25*np.einsum('mnef,efmn->',ints['vC']['oovv'],r2c,optimize=True)

    return r0/omega

def orthogonalize(q,B):
    
    for i in range(B.shape[1]):
        b = B[:,i]/np.linalg.norm(B[:,i])
        q -= np.dot(b.T,q)*b
    return q

def flatten_R(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d):
    return np.concatenate((r1a.flatten(),r1b.flatten(),\
                           r2a.flatten(),r2b.flatten(),r2c.flatten(),\
                           r3a.flatten(),r3b.flatten(),r3c.flatten(),r3d.flatten()),axis=0)

def unflatten_R(R,sys,order='C'):

    n1a = sys['Nocc_a'] * sys['Nunocc_a']
    n1b = sys['Nocc_b'] * sys['Nunocc_b']
    n2a = sys['Nocc_a'] ** 2 * sys['Nunocc_a'] ** 2
    n2b = sys['Nocc_a'] * sys['Nocc_b'] * sys['Nunocc_a'] * sys['Nunocc_b']
    n2c = sys['Nocc_b'] ** 2 * sys['Nunocc_b'] ** 2
    n3a = sys['Nocc_a'] ** 3 * sys['Nunocc_a'] ** 3
    n3b = sys['Nocc_a']**2 * sys['Nocc_b'] * sys['Nunocc_a']**2 * sys['Nunocc_b']
    n3c = sys['Nocc_a'] * sys['Nocc_b']**2 * sys['Nunocc_a'] * sys['Nunocc_b']**2
    n3d = sys['Nocc_b'] ** 3 * sys['Nunocc_b'] ** 3

    idx_1a = slice(0,n1a)
    idx_1b = slice(n1a,n1a+n1b)
    idx_2a = slice(n1a+n1b,n1a+n1b+n2a)
    idx_2b = slice(n1a+n1b+n2a,n1a+n1b+n2a+n2b)
    idx_2c = slice(n1a+n1b+n2a+n2b,n1a+n1b+n2a+n2b+n2c)
    idx_3a = slice(n1a+n1b+n2a+n2b+n2c,n1a+n1b+n2a+n2b+n2c+n3a)
    idx_3b = slice(n1a+n1b+n2a+n2b+n2c+n3a,n1a+n1b+n2a+n2b+n2c+n3a+n3b)
    idx_3c = slice(n1a+n1b+n2a+n2b+n2c+n3a+n3b,n1a+n1b+n2a+n2b+n2c+n3a+n3b+n3c)
    idx_3d = slice(n1a+n1b+n2a+n2b+n2c+n3a+n3b+n3c,n1a+n1b+n2a+n2b+n2c+n3a+n3b+n3c+n3d)

    r1a  = np.reshape(R[idx_1a],(sys['Nunocc_a'],sys['Nocc_a']),order=order)
    r1b  = np.reshape(R[idx_1b],(sys['Nunocc_b'],sys['Nocc_b']),order=order)
    r2a  = np.reshape(R[idx_2a],(sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a']),order=order)
    r2b  = np.reshape(R[idx_2b],(sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b']),order=order)
    r2c  = np.reshape(R[idx_2c],(sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b']),order=order)
    r3a  = np.reshape(R[idx_3a],(sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_a']),order=order)
    r3b  = np.reshape(R[idx_3b],(sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_b']),order=order)
    r3c  = np.reshape(R[idx_3c],(sys['Nunocc_a'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b'],sys['Nocc_b']),order=order)
    r3d  = np.reshape(R[idx_3d],(sys['Nunocc_b'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b'],sys['Nocc_b']),order=order)

    return r1a, r1b, r2a, r2b, r2c, r3a, r3b, r3c, r3d


def HR(R,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):

    r1a, r1b, r2a, r2b, r2c, r3a, r3b, r3c, r3d = unflatten_R(R,sys)

    X1A = build_HR_1A(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X1B = build_HR_1B(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X2A = build_HR_2A(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X2B = build_HR_2B(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X2C = build_HR_2C(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X3A = build_HR_3A(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X3B = build_HR_3B(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    # closed shell symmetry
    X3C = np.transpose(X3B,(2,1,0,5,4,3))
    X3D = X3A.copy()

    return flatten_R(X1A, X1B, X2A, X2B, X2C, X3A, X3B, X3C, X3D)

def build_HR_1A(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):

    X1A = 0.0
    X1A -= np.einsum('mi,am->ai',H1A['oo'],r1a,optimize=True)
    X1A += np.einsum('ae,ei->ai',H1A['vv'],r1a,optimize=True)
    X1A += np.einsum('amie,em->ai',H2A['voov'],r1a,optimize=True)
    X1A += np.einsum('amie,em->ai',H2B['voov'],r1b,optimize=True)
    X1A -= 0.5*np.einsum('mnif,afmn->ai',H2A['ooov'],r2a,optimize=True)
    X1A -= np.einsum('mnif,afmn->ai',H2B['ooov'],r2b,optimize=True)
    X1A += 0.5*np.einsum('anef,efin->ai',H2A['vovv'],r2a,optimize=True)
    X1A += np.einsum('anef,efin->ai',H2B['vovv'],r2b,optimize=True)
    X1A += np.einsum('me,aeim->ai',H1A['ov'],r2a,optimize=True)
    X1A += np.einsum('me,aeim->ai',H1B['ov'],r2b,optimize=True)

    X1A += 0.25*np.einsum('mnef,aefimn->ai',ints['vA']['oovv'],r3a,optimize=True)
    X1A += np.einsum('mnef,aefimn->ai',ints['vB']['oovv'],r3b,optimize=True)
    X1A += 0.25*np.einsum('mnef,aefimn->ai',ints['vC']['oovv'],r3c,optimize=True)

    return X1A

def build_HR_1B(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):

    X1B = 0.0
    X1B -= np.einsum('mi,am->ai',H1B['oo'],r1b,optimize=True)
    X1B += np.einsum('ae,ei->ai',H1B['vv'],r1b,optimize=True)
    X1B += np.einsum('maei,em->ai',H2B['ovvo'],r1a,optimize=True)
    X1B += np.einsum('amie,em->ai',H2C['voov'],r1b,optimize=True)
    X1B -= np.einsum('nmfi,fanm->ai',H2B['oovo'],r2b,optimize=True)
    X1B -= 0.5*np.einsum('mnif,afmn->ai',H2C['ooov'],r2c,optimize=True)
    X1B += np.einsum('nafe,feni->ai',H2B['ovvv'],r2b,optimize=True)
    X1B += 0.5*np.einsum('anef,efin->ai',H2C['vovv'],r2c,optimize=True)
    X1B += np.einsum('me,eami->ai',H1A['ov'],r2b,optimize=True)
    X1B += np.einsum('me,aeim->ai',H1B['ov'],r2c,optimize=True)

    X1B += 0.25*np.einsum('mnef,efamni->ai',ints['vA']['oovv'],r3b,optimize=True)
    X1B += np.einsum('mnef,efamni->ai',ints['vB']['oovv'],r3c,optimize=True)
    X1B += 0.25*np.einsum('mnef,aefimn->ai',ints['vC']['oovv'],r3d,optimize=True)

    return X1B

def build_HR_2A(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):

    t2a = cc_t['t2a']
    t3a = cc_t['t3a']
    t3b = cc_t['t3b']
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']

    X2A = 0.0
    D1 = -np.einsum('mi,abmj->abij',H1A['oo'],r2a,optimize=True) # A(ij) 
    D2 = np.einsum('ae,ebij->abij',H1A['vv'],r2a,optimize=True) # A(ab)
    X2A += 0.5*np.einsum('mnij,abmn->abij',H2A['oooo'],r2a,optimize=True)
    X2A += 0.5*np.einsum('abef,efij->abij',H2A['vvvv'],r2a,optimize=True)
    D3 = np.einsum('amie,ebmj->abij',H2A['voov'],r2a,optimize=True) # A(ij)A(ab)
    D4 = np.einsum('amie,bejm->abij',H2B['voov'],r2b,optimize=True) # A(ij)A(ab)
    D5 = -np.einsum('bmji,am->abij',H2A['vooo'],r1a,optimize=True) # A(ab)
    D6 = np.einsum('baje,ei->abij',H2A['vvov'],r1a,optimize=True) # A(ij)

    Q1 = -0.5*np.einsum('mnef,bfmn->eb',vA['oovv'],r2a,optimize=True)
    D7 = np.einsum('eb,aeij->abij',Q1,t2a,optimize=True) # A(ab)
    Q2 = -np.einsum('mnef,bfmn->eb',vB['oovv'],r2b,optimize=True)
    D8 = np.einsum('eb,aeij->abij',Q2,t2a,optimize=True) # A(ab)

    Q1 = 0.5*np.einsum('mnef,efjn->mj',vA['oovv'],r2a,optimize=True)
    D9 = -np.einsum('mj,abim->abij',Q1,t2a,optimize=True) # A(ij)
    Q2 = np.einsum('mnef,efjn->mj',vB['oovv'],r2b,optimize=True)
    D10 = -np.einsum('mj,abim->abij',Q2,t2a,optimize=True) # A(ij)

    Q1 = np.einsum('amfe,em->af',H2A['vovv'],r1a,optimize=True)
    D11 = np.einsum('af,fbij->abij',Q1,t2a,optimize=True) # A(ab)
    Q2 = np.einsum('nmie,em->ni',H2A['ooov'],r1a,optimize=True)
    D12 = -np.einsum('ni,abnj->abij',Q2,t2a,optimize=True) # A(ij)

    Q1 = np.einsum('amfe,em->af',H2B['vovv'],r1b,optimize=True)
    D13 = np.einsum('af,fbij->abij',Q1,t2a,optimize=True) # A(ab)
    Q2 = np.einsum('nmie,em->ni',H2B['ooov'],r1b,optimize=True)
    D14 = -np.einsum('ni,abnj->abij',Q2,t2a,optimize=True) # A(ij)

    I1 = np.einsum('mnef,fn->me',vA['oovv'],r1a,optimize=True)\
        +np.einsum('mnef,fn->me',vB['oovv'],r1b,optimize=True)
    X2A += np.einsum('me,abeijm->abij',I1,t3a,optimize=True)

    I1 = np.einsum('nmfe,fn->me',vB['oovv'],r1a,optimize=True)\
        +np.einsum('mnef,fn->me',vC['oovv'],r1b,optimize=True)
    X2A += np.einsum('me,abeijm->abij',I1,t3b,optimize=True)

    DR3_1 = np.einsum('me,abeijm->abij',H1A['ov'],r3a,optimize=True)
    DR3_2 = np.einsum('me,abeijm->abij',H1B['ov'],r3b,optimize=True)
    DR3_3 = -0.5*np.einsum('mnjf,abfimn->abij',H2A['ooov'],r3a,optimize=True)
    DR3_4 = -1.0*np.einsum('mnjf,abfimn->abij',H2B['ooov'],r3b,optimize=True)
    DR3_5 = 0.5*np.einsum('bnef,aefijn->abij',H2A['vovv'],r3a,optimize=True)
    DR3_6 = np.einsum('bnef,aefijn->abij',H2B['vovv'],r3b,optimize=True)

    D_ij = D1 + D6 + D9 + D10 + D12 + D14 + DR3_3 + DR3_4
    D_ab = D2 + D5 + D7 + D8  + D11 + D13 + DR3_5 + DR3_6
    D_abij = D3 + D4

    D_ij -= np.einsum('abij->abji',D_ij,optimize=True)
    D_ab -= np.einsum('abij->baij',D_ab,optimize=True)
    D_abij += -np.einsum('abij->baij',D_abij,optimize=True)\
    -np.einsum('abij->abji',D_abij,optimize=True)\
    +np.einsum('abij->baji',D_abij,optimize=True)    

    X2A += D_ij + D_ab + D_abij + DR3_1 + DR3_2

    return X2A

def build_HR_2B(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):

    t2b = cc_t['t2b']
    t3b = cc_t['t3b']
    t3c = cc_t['t3c']
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']

    X2B = 0.0
    X2B += np.einsum('ae,ebij->abij',H1A['vv'],r2b,optimize=True)
    X2B += np.einsum('be,aeij->abij',H1B['vv'],r2b,optimize=True)
    X2B -= np.einsum('mi,abmj->abij',H1A['oo'],r2b,optimize=True)
    X2B -= np.einsum('mj,abim->abij',H1B['oo'],r2b,optimize=True)
    X2B += np.einsum('mnij,abmn->abij',H2B['oooo'],r2b,optimize=True)
    X2B += np.einsum('abef,efij->abij',H2B['vvvv'],r2b,optimize=True)
    X2B += np.einsum('amie,ebmj->abij',H2A['voov'],r2b,optimize=True)
    X2B += np.einsum('amie,ebmj->abij',H2B['voov'],r2c,optimize=True)
    X2B += np.einsum('mbej,aeim->abij',H2B['ovvo'],r2a,optimize=True)
    X2B += np.einsum('bmje,aeim->abij',H2C['voov'],r2b,optimize=True)
    X2B -= np.einsum('mbie,aemj->abij',H2B['ovov'],r2b,optimize=True)
    X2B -= np.einsum('amej,ebim->abij',H2B['vovo'],r2b,optimize=True)
    X2B += np.einsum('abej,ei->abij',H2B['vvvo'],r1a,optimize=True)
    X2B += np.einsum('abie,ej->abij',H2B['vvov'],r1b,optimize=True)
    X2B -= np.einsum('mbij,am->abij',H2B['ovoo'],r1a,optimize=True)
    X2B -= np.einsum('amij,bm->abij',H2B['vooo'],r1b,optimize=True)

    Q1 = -0.5*np.einsum('mnef,afmn->ae',vA['oovv'],r2a,optimize=True)
    X2B += np.einsum('ae,ebij->abij',Q1,t2b,optimize=True)
    Q2 = 0.5*np.einsum('mnef,efin->mi',vA['oovv'],r2a,optimize=True)
    X2B -= np.einsum('mi,abmj->abij',Q2,t2b,optimize=True)

    Q1 = -np.einsum('nmfe,fbnm->be',vB['oovv'],r2b,optimize=True)
    X2B += np.einsum('be,aeij->abij',Q1,t2b,optimize=True)
    Q2 = -np.einsum('mnef,afmn->ae',vB['oovv'],r2b,optimize=True)
    X2B += np.einsum('ae,ebij->abij',Q2,t2b,optimize=True)
    Q3 = np.einsum('nmfe,fenj->mj',vB['oovv'],r2b,optimize=True)
    X2B -= np.einsum('mj,abim->abij',Q3,t2b,optimize=True)
    Q4 = np.einsum('mnef,efin->mi',vB['oovv'],r2b,optimize=True)
    X2B -= np.einsum('mi,abmj->abij',Q4,t2b,optimize=True)

    Q1 = -0.5*np.einsum('mnef,bfmn->be',vC['oovv'],r2c,optimize=True)
    X2B += np.einsum('be,aeij->abij',Q1,t2b,optimize=True)
    Q2 = 0.5*np.einsum('mnef,efjn->mj',vC['oovv'],r2c,optimize=True)
    X2B -= np.einsum('mj,abim->abij',Q2,t2b,optimize=True)

    Q1 = np.einsum('mbef,em->bf',H2B['ovvv'],r1a,optimize=True)
    X2B += np.einsum('bf,afij->abij',Q1,t2b,optimize=True)
    Q2 = np.einsum('mnej,em->nj',H2B['oovo'],r1a,optimize=True)
    X2B -= np.einsum('nj,abin->abij',Q2,t2b,optimize=True)
    Q3 = np.einsum('amfe,em->af',H2A['vovv'],r1a,optimize=True)
    X2B += np.einsum('af,fbij->abij',Q3,t2b,optimize=True)
    Q4 = np.einsum('nmie,em->ni',H2A['ooov'],r1a,optimize=True)
    X2B -= np.einsum('ni,abnj->abij',Q4,t2b,optimize=True)

    Q1 = np.einsum('amfe,em->af',H2B['vovv'],r1b,optimize=True)
    X2B += np.einsum('af,fbij->abij',Q1,t2b,optimize=True)
    Q2 = np.einsum('nmie,em->ni',H2B['ooov'],r1b,optimize=True)
    X2B -= np.einsum('ni,abnj->abij',Q2,t2b,optimize=True)
    Q3 = np.einsum('bmfe,em->bf',H2C['vovv'],r1b,optimize=True)
    X2B += np.einsum('bf,afij->abij',Q3,t2b,optimize=True)
    Q4 = np.einsum('nmje,em->nj',H2C['ooov'],r1b,optimize=True)
    X2B -= np.einsum('nj,abin->abij',Q4,t2b,optimize=True)

    I1 = np.einsum('mnef,fn->me',vA['oovv'],r1a,optimize=True)\
        +np.einsum('mnef,fn->me',vB['oovv'],r1b,optimize=True)
    X2B += np.einsum('me,aebimj->abij',I1,t3b,optimize=True)

    I1 = np.einsum('nmfe,fn->me',vB['oovv'],r1a,optimize=True)\
        +np.einsum('mnef,fn->me',vC['oovv'],r1b,optimize=True)
    X2B += np.einsum('me,aebimj->abij',I1,t3c,optimize=True)

    X2B += np.einsum('me,aebimj->abij',H1A['ov'],r3b,optimize=True)
    X2B += np.einsum('me,aebimj->abij',H1B['ov'],r3c,optimize=True)
    X2B -= np.einsum('nmfj,afbinm->abij',H2B['oovo'],r3b,optimize=True)
    X2B -= 0.5*np.einsum('mnjf,abfimn->abij',H2C['ooov'],r3c,optimize=True)
    X2B -= 0.5*np.einsum('mnif,afbmnj->abij',H2A['ooov'],r3b,optimize=True)
    X2B -= np.einsum('mnif,abfmjn->abij',H2B['ooov'],r3c,optimize=True)
    X2B += np.einsum('nbfe,afeinj->abij',H2B['ovvv'],r3b,optimize=True)
    X2B += 0.5*np.einsum('bnef,aefijn->abij',H2C['vovv'],r3c,optimize=True)
    X2B += 0.5*np.einsum('anef,efbinj->abij',H2A['vovv'],r3b,optimize=True)
    X2B += np.einsum('anef,efbinj->abij',H2B['vovv'],r3c,optimize=True)

    return X2B

def build_HR_2C(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):

    t2c = cc_t['t2c']
    t3c = cc_t['t3c']
    t3d = cc_t['t3d']
    vA = ints['vA']
    vC = ints['vC']
    vB = ints['vB']

    X2C = 0.0
    D1 = -np.einsum('mi,abmj->abij',H1B['oo'],r2c,optimize=True) # A(ij) 
    D2 = np.einsum('ae,ebij->abij',H1B['vv'],r2c,optimize=True) # A(ab)
    X2C += 0.5*np.einsum('mnij,abmn->abij',H2C['oooo'],r2c,optimize=True)
    X2C += 0.5*np.einsum('abef,efij->abij',H2C['vvvv'],r2c,optimize=True)
    D3 = np.einsum('amie,ebmj->abij',H2C['voov'],r2c,optimize=True) # A(ij)A(ab)
    D4 = np.einsum('maei,ebmj->abij',H2B['ovvo'],r2b,optimize=True) # A(ij)A(ab)
    D5 = -np.einsum('bmji,am->abij',H2C['vooo'],r1b,optimize=True) # A(ab)
    D6 = np.einsum('baje,ei->abij',H2C['vvov'],r1b,optimize=True) # A(ij)

    Q1 = -0.5*np.einsum('mnef,bfmn->eb',vC['oovv'],r2c,optimize=True)
    D7 = np.einsum('eb,aeij->abij',Q1,t2c,optimize=True) # A(ab)
    Q2 = -np.einsum('nmfe,fbnm->eb',vB['oovv'],r2b,optimize=True)
    D8 = np.einsum('eb,aeij->abij',Q2,t2c,optimize=True) # A(ab)

    Q1 = 0.5*np.einsum('mnef,efjn->mj',vC['oovv'],r2c,optimize=True)
    D9 = -np.einsum('mj,abim->abij',Q1,t2c,optimize=True) # A(ij)
    Q2 = np.einsum('nmfe,fenj->mj',vB['oovv'],r2b,optimize=True)
    D10 = -np.einsum('mj,abim->abij',Q2,t2c,optimize=True) # A(ij)

    Q1 = np.einsum('amfe,em->af',H2C['vovv'],r1b,optimize=True)
    D11 = np.einsum('af,fbij->abij',Q1,t2c,optimize=True) # A(ab)
    Q2 = np.einsum('nmie,em->ni',H2C['ooov'],r1b,optimize=True)
    D12 = -np.einsum('ni,abnj->abij',Q2,t2c,optimize=True) # A(ij)

    Q1 = np.einsum('maef,em->af',H2B['ovvv'],r1a,optimize=True)
    D13 = np.einsum('af,fbij->abij',Q1,t2c,optimize=True) # A(ab)
    Q2 = np.einsum('mnei,em->ni',H2B['oovo'],r1a,optimize=True)
    D14 = -np.einsum('ni,abnj->abij',Q2,t2c,optimize=True) # A(ij)

    I1 = np.einsum('mnef,fn->me',vA['oovv'],r1a,optimize=True)\
        +np.einsum('mnef,fn->me',vB['oovv'],r1b,optimize=True)
    X2C += np.einsum('me,eabmij->abij',I1,t3c,optimize=True)

    I1 = np.einsum('nmfe,fn->me',vB['oovv'],r1a,optimize=True)\
        +np.einsum('mnef,fn->me',vC['oovv'],r1b,optimize=True)
    X2C += np.einsum('me,abeijm->abij',I1,t3d,optimize=True)

    DR3_1 = np.einsum('me,eabmij->abij',H1A['ov'],r3c,optimize=True)
    DR3_2 = np.einsum('me,abeijm->abij',H1B['ov'],r3d,optimize=True)
    DR3_3 = -0.5*np.einsum('mnjf,abfimn->abij',H2C['ooov'],r3d,optimize=True)
    DR3_4 = -1.0*np.einsum('nmfj,fabnim->abij',H2B['oovo'],r3c,optimize=True)
    DR3_5 = 0.5*np.einsum('bnef,aefijn->abij',H2C['vovv'],r3d,optimize=True)
    DR3_6 = np.einsum('nbfe,faenij->abij',H2B['ovvv'],r3c,optimize=True)

    D_ij = D1 + D6 + D9 + D10 + D12 + D14 + DR3_3 + DR3_4
    D_ab = D2 + D5 + D7 + D8  + D11 + D13 + DR3_5 + DR3_6
    D_abij = D3 + D4

    D_ij -= np.einsum('abij->abji',D_ij,optimize=True)
    D_ab -= np.einsum('abij->baij',D_ab,optimize=True)
    D_abij += -np.einsum('abij->baij',D_abij,optimize=True)\
    -np.einsum('abij->abji',D_abij,optimize=True)\
    +np.einsum('abij->baji',D_abij,optimize=True)    

    X2C += D_ij + D_ab + D_abij + DR3_1 + DR3_2

    return X2C

def build_HR_3A(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):

    t2a = cc_t['t2a']
    t3a = cc_t['t3a']
    t3b = cc_t['t3b']
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']

    X3A = 0.0
    D_ijk_cab = 0.0
    D_ijk_abc = 0.0
    D_kij_abc = 0.0
    D_kij_cab = 0.0
    D_jik_bac = 0.0
    D_bac = 0.0
    D_cab = 0.0
    D_jik = 0.0
    D_kij = 0.0

    # < ijkabc | (HR1)_C | 0 >
    I1 = -1.0*np.einsum('amie,cm->acie',H2A['voov'],r1a,optimize=True)
    I1 -= np.transpose(I1,(1,0,2,3))
    I2 = -1.0*np.einsum('nmjk,cm->ncjk',H2A['oooo'],r1a,optimize=True)
    D1 = np.einsum('abie,ecjk->abcijk',I1,t2a,optimize=True)\
        -np.einsum('ncjk,abin->abcijk',I2,t2a,optimize=True)
    D_ijk_cab += D1

    I1 = np.einsum('amie,ej->amij',H2A['voov'],r1a,optimize=True)
    I1 -= np.transpose(I1,(0,1,3,2))
    I2 = np.einsum('abfe,fi->abie',H2A['vvvv'],r1a,optimize=True)
    D2 = -1.0*np.einsum('amij,bcmk->abcijk',I1,t2a,optimize=True)\
        +np.einsum('cbke,aeij->abcijk',I2,t2a,optimize=True)
    D_kij_abc += D2

    I1 = np.einsum('mnef,fn->me',H2A['oovv'],r1a,optimize=True)\
        +np.einsum('mnef,fn->me',H2B['oovv'],r1b,optimize=True)
    I2 = np.einsum('me,ecjk->mcjk',I1,t2a,optimize=True)
    D3 = -1.0*np.einsum('mcjk,abim->abcijk',I2,t2a,optimize=True)
    D_ijk_cab += D3

    # additional terms with T3
    I1 = -1.0*np.einsum('me,bm->be',H1A['ov'],r1a,optimize=True)\
         +np.einsum('bnef,fn->be',H2A['vovv'],r1a,optimize=True)\
         +np.einsum('bnef,fn->be',H2B['vovv'],r1b,optimize=True)
    D4 = np.einsum('be,aecijk->abcijk',I1,t3a,optimize=True) # A(b/ac)
    D_bac += D4

    I1 = np.einsum('me,ej->mj',H1A['ov'],r1a,optimize=True)\
        +np.einsum('mnjf,fn->mj',H2A['ooov'],r1a,optimize=True)\
        +np.einsum('mnjf,fn->mj',H2B['ooov'],r1b,optimize=True)
    D5 = -1.0*np.einsum('mj,abcimk->abcijk',I1,t3a,optimize=True) # A(j/ik)
    D_jik += D5

    I1 = np.einsum('nmje,ei->mnij',H2A['ooov'],r1a,optimize=True)
    I1 -= np.transpose(I1,(0,1,3,2))
    D6 = 0.5*np.einsum('mnij,abcmnk->abcijk',I1,t3a,optimize=True) # A(k/ij)
    D_kij += D6

    I1 = -1.0*np.einsum('amef,bm->abef',H2A['vovv'],r1a,optimize=True)
    I1 -= np.transpose(I1,(1,0,2,3))
    D7 = 0.5*np.einsum('abef,efcijk->abcijk',I1,t3a,optimize=True) # A(c/ab)
    D_cab += D7

    I1 = -1.0*np.einsum('nmje,bn->bmje',H2A['ooov'],r1a,optimize=True)\
         +np.einsum('bmfe,fj->bmje',H2A['vovv'],r1a,optimize=True)
    D8 = np.einsum('bmje,aecimk->abcijk',I1,t3a,optimize=True) # A(j/ik)A(b/ac)
    D_jik_bac += D8

    I1 = -1.0*np.einsum('nmje,bn->bmje',H2B['ooov'],r1a,optimize=True)\
         +np.einsum('bmfe,fj->bmje',H2B['vovv'],r1a,optimize=True)
    D9 = np.einsum('bmje,aceikm->abcijk',I1,t3b,optimize=True) # A(j/ik)A(b/ac)
    D_jik_bac += D9

    # < ijkabc | (HR2)_C | 0 >
    I1 = 0.5*np.einsum('amef,efij->amij',H2A['vovv'],r2a,optimize=True)
    D1 = -1.0*np.einsum('amij,bcmk->abcijk',I1,t2a,optimize=True)
    D_kij_abc += D1

    I1 = 0.5*np.einsum('mnie,abmn->abie',H2A['ooov'],r2a,optimize=True)
    D2 = np.einsum('abie,ecjk->abcijk',I1,t2a,optimize=True)
    D_ijk_cab += D2

    I1 = np.einsum('bmfe,aeim->abif',H2A['vovv'],r2a,optimize=True)
    I1 -= np.transpose(I1,(1,0,2,3))
    I2 = np.einsum('nmje,cekm->cnkj',H2A['ooov'],r2a,optimize=True)
    I2 -= np.transpose(I2,(0,1,3,2))
    D3 = np.einsum('abif,fcjk->abcijk',I1,t2a,optimize=True)\
        -np.einsum('cnkj,abin->abcijk',I2,t2a,optimize=True)
    D_ijk_cab += D3

    I1 = np.einsum('bmfe,aeim->abif',H2B['vovv'],r2b,optimize=True)
    I1 -= np.transpose(I1,(1,0,2,3))
    I2 = np.einsum('nmje,cekm->cnkj',H2B['ooov'],r2b,optimize=True)
    I2 -= np.transpose(I2,(0,1,3,2))
    D4 = np.einsum('abif,fcjk->abcijk',I1,t2a,optimize=True)\
        -np.einsum('cnkj,abin->abcijk',I2,t2a,optimize=True)
    D_ijk_cab += D4

    D5 = -1.0*np.einsum('amij,bcmk->abcijk',H2A['vooo'],r2a,optimize=True)
    D_kij_abc += D5

    D6 = np.einsum('abie,ecjk->abcijk',H2A['vvov'],r2a,optimize=True)
    D_ijk_cab += D6

    # additional terms with T3
    I1 = 0.5*np.einsum('mnef,efij->mnij',vA['oovv'],r2a,optimize=True)
    D7 = 0.5*np.einsum('mnij,abcmnk->abcijk',I1,t3a,optimize=True)
    D_kij += D7

    I1 = 0.5*np.einsum('mnef,abmn->abef',vA['oovv'],r2a,optimize=True)
    D8 = 0.5*np.einsum('abef,efcijk->abcijk',I1,t3a,optimize=True)
    D_cab += D8

    I1 = np.einsum('mnef,fcnk->cmke',vA['oovv'],r2a,optimize=True)\
        +np.einsum('mnef,cfkn->cmke',vB['oovv'],r2b,optimize=True)
    D9 = np.einsum('cmke,abeijm->abcijk',I1,t3a,optimize=True) 
    D_kij_cab += D9

    I1 = np.einsum('nmfe,fcnk->cmke',vB['oovv'],r2a,optimize=True)\
        +np.einsum('mnef,cfkn->cmke',vC['oovv'],r2b,optimize=True)
    D10 = np.einsum('cmke,abeijm->abcijk',I1,t3b,optimize=True)
    D_kij_cab += D10

    I1 = -0.5*np.einsum('mnef,bfmn->be',vA['oovv'],r2a,optimize=True)\
         -np.einsum('mnef,bfmn->be',vB['oovv'],r2b,optimize=True)
    D11 = np.einsum('be,aecijk->abcijk',I1,t3a,optimize=True)
    D_bac += D11

    I1 = 0.5*np.einsum('mnef,efjn->mj',vA['oovv'],r2a,optimize=True)\
        +np.einsum('mnef,efjn->mj',vB['oovv'],r2b,optimize=True)
    D12 = -1.0*np.einsum('mj,abcimk->abcijk',I1,t3a,optimize=True)
    D_jik += D12

    # < ijkabc | (HR3)_C | 0 >
    D1 = -1.0*np.einsum('mj,abcimk->abcijk',H1A['oo'],r3a,optimize=True)
    D_jik += D1

    D2 = np.einsum('be,aecijk->abcijk',H1A['vv'],r3a,optimize=True)
    D_bac += D2

    D3 = 0.5*np.einsum('mnij,abcmnk->abcijk',H2A['oooo'],r3a,optimize=True)
    D_kij += D3

    D4 = 0.5*np.einsum('abef,efcijk->abcijk',H2A['vvvv'],r3a,optimize=True)
    D_cab += D4

    D5 = np.einsum('amie,ebcmjk->abcijk',H2A['voov'],r3a,optimize=True)
    D_ijk_abc += D5

    D6 = np.einsum('amie,bcejkm->abcijk',H2B['voov'],r3b,optimize=True)
    D_ijk_abc += D6

    I1 = 0.5*np.einsum('mnef,efcjnk->mcjk',vA['oovv'],r3a,optimize=True)\
        +np.einsum('mnef,ecfjkn->mcjk',vB['oovv'],r3b,optimize=True)
    D7 = -1.0*np.einsum('mcjk,abim->abcijk',I1,t2a,optimize=True)
    D_ijk_cab += D7

    I1 = -0.5*np.einsum('mnef,abfimn->abie',vA['oovv'],r3a,optimize=True)\
        -np.einsum('mnef,abfimn->abie',vB['oovv'],r3b,optimize=True)
    D8 = np.einsum('abie,ecjk->abcijk',I1,t2a,optimize=True)
    D_ijk_cab += D8

    # antisymmetrize terms and add up
    D_ijk_cab -= np.transpose(D_ijk_cab,(2,1,0,3,4,5)) + np.transpose(D_ijk_cab,(0,2,1,3,4,5))
    D_ijk_cab -= np.transpose(D_ijk_cab,(0,1,2,5,4,3)) + np.transpose(D_ijk_cab,(0,1,2,4,3,5))

    D_ijk_abc -= np.transpose(D_ijk_abc,(1,0,2,3,4,5)) + np.transpose(D_ijk_abc,(2,1,0,3,4,5))
    D_ijk_abc -= np.transpose(D_ijk_abc,(0,1,2,5,4,3)) + np.transpose(D_ijk_abc,(0,1,2,4,3,5))

    D_kij_abc -= np.transpose(D_kij_abc,(1,0,2,3,4,5)) + np.transpose(D_kij_abc,(2,1,0,3,4,5))
    D_kij_abc -= np.transpose(D_kij_abc,(0,1,2,5,4,3)) + np.transpose(D_kij_abc,(0,1,2,3,5,4))

    D_kij_cab -= np.transpose(D_kij_cab,(0,2,1,3,4,5)) + np.transpose(D_kij_cab,(2,1,0,3,4,5))
    D_kij_cab -= np.transpose(D_kij_cab,(0,1,2,5,4,3)) + np.transpose(D_kij_cab,(0,1,2,3,5,4))

    D_jik_bac -= np.transpose(D_jik_bac,(0,2,1,3,4,5)) + np.transpose(D_jik_bac,(1,0,2,3,4,5))
    D_jik_bac -= np.transpose(D_jik_bac,(0,1,2,4,3,5)) + np.transpose(D_jik_bac,(0,1,2,3,5,4))

    D_cab -= np.transpose(D_cab,(2,1,0,3,4,5)) + np.transpose(D_cab,(0,2,1,3,4,5))
    D_bac -= np.transpose(D_bac,(0,2,1,3,4,5)) + np.transpose(D_bac,(1,0,2,3,4,5))
    D_jik -= np.transpose(D_jik,(0,1,2,4,3,5)) + np.transpose(D_jik,(0,1,2,3,5,4))
    D_kij -= np.transpose(D_kij,(0,1,2,5,4,3)) + np.transpose(D_kij,(0,1,2,3,5,4))

    X3A += D_ijk_abc + D_ijk_cab + D_kij_abc + D_kij_cab + D_jik_bac + D_cab + D_bac + D_jik + D_kij

    return X3A

def build_HR_3B(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):

    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t3a = cc_t['t3a']
    t3b = cc_t['t3b']
    t3c = cc_t['t3c']
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']

    X3B = 0.0
    D_ij = 0.0
    D_ab = 0.0
    D_ij_ab = 0.0

    # < ijk~abc~ | (HR1)_C | 0 >
    I1 = np.einsum('mcie,ek->mcik',H2B['ovov'],r1b,optimize=True)
    I2 = np.einsum('acfe,ek->acfk',H2B['vvvv'],r1b,optimize=True)
    I3 = np.einsum('amie,ek->amik',H2B['voov'],r1b,optimize=True)
    A1 = -1.0*np.einsum('mcik,abmj->abcijk',I1,t2a,optimize=True)
    D_ij += A1
    A2 = np.einsum('acfk,fbij->abcijk',I2,t2a,optimize=True)
    D_ab += A2
    A3  = -1.0*np.einsum('amik,bcjm->abcijk',I3,t2b,optimize=True)
    D_ij_ab += A3

    I1 = np.einsum('mcek,ej->mcjk',H2B['ovvo'],r1a,optimize=True)
    I2 = np.einsum('bmek,ej->bmjk',H2B['vovo'],r1a,optimize=True)
    I3 = np.einsum('amie,ej->amij',H2A['voov'],r1a,optimize=True)
    I3 -= np.transpose(I3,(0,1,3,2))
    I4 = np.einsum('bcef,ej->bcjf',H2B['vvvv'],r1a,optimize=True)
    I5 = np.einsum('abfe,ej->abfj',H2A['vvvv'],r1a,optimize=True)
    A1 = -1.0*np.einsum('mcjk,abim->abcijk',I1,t2a,optimize=True)
    A2 = -1.0*np.einsum('bmjk,acim->abcijk',I2,t2b,optimize=True)
    A3 = -1.0*np.einsum('amij,bcmk->abcijk',I3,t2b,optimize=True)
    A4 = np.einsum('bcjf,afik->abcijk',I4,t2b,optimize=True)
    A5 = np.einsum('abfj,fcik->abcijk',I5,t2b,optimize=True)
    B1 = A1 + A5
    B2 = A2 + A4
    D_ij += B1
    D_ab += A3
    D_ij_ab += B2

    I1 = -1.0*np.einsum('bmek,cm->bcek',H2B['vovo'],r1b,optimize=True)
    I2 = -1.0*np.einsum('bmje,cm->bcje',H2B['voov'],r1b,optimize=True)
    I3 = -1.0*np.einsum('nmjk,cm->ncjk',H2B['oooo'],r1b,optimize=True)
    A1 = np.einsum('bcek,aeij->abcijk',I1,t2a,optimize=True)
    A2 = np.einsum('bcje,aeik->abcijk',I2,t2b,optimize=True)
    A3 = -1.0*np.einsum('ncjk,abin->abcijk',I3,t2a,optimize=True)
    D_ij += A3
    D_ab += A1
    D_ij_ab += A2

    I1 = -1.0*np.einsum('mcje,bm->bcje',H2B['ovov'],r1a,optimize=True)
    I2 = -1.0*np.einsum('mcek,bm->bcek',H2B['ovvo'],r1a,optimize=True)
    I3 = -1.0*np.einsum('mnjk,bm->bnjk',H2B['oooo'],r1a,optimize=True)
    I4 = -1.0*np.einsum('mnji,bm->bnji',H2A['oooo'],r1a,optimize=True)
    I5 = np.einsum('amje,bm->abej',H2A['voov'],r1a,optimize=True)
    I5 -= np.transpose(I5,(1,0,2,3))
    A1 = np.einsum('bcje,aeik->abcijk',I1,t2b,optimize=True)
    A2 = np.einsum('bcek,aeij->abcijk',I2,t2a,optimize=True)
    A3 = -1.0*np.einsum('bnjk,acin->abcijk',I3,t2b,optimize=True)
    A4 = -1.0*np.einsum('bnji,acnk->abcijk',I4,t2b,optimize=True)
    A5 = np.einsum('abej,ecik->abcijk',I5,t2b,optimize=True)
    B1 = A1 + A3
    B2 = A2 + A4
    D_ij_ab += B1
    D_ab += B2
    D_ij += A5
    
    I1A = np.einsum('mnef,fn->me',H2A['oovv'],r1a,optimize=True)\
         +np.einsum('mnef,fn->me',H2B['oovv'],r1b,optimize=True)
    I1B = np.einsum('nmfe,fn->me',H2C['oovv'],r1b,optimize=True)\
         +np.einsum('nmfe,fn->me',H2B['oovv'],r1a,optimize=True)
    
    I1 = np.einsum('me,aeij->amij',I1A,t2a,optimize=True)
    B1 = -1.0*np.einsum('amij,bcmk->abcijk',I1,t2b,optimize=True)
    D_ab += B1

    I1 = -1.0*np.einsum('me,abim->abie',I1A,t2a,optimize=True)
    B2 = np.einsum('abie,ecjk->abcijk',I1,t2b,optimize=True)
    D_ij += B2

    I1 = np.einsum('me,aeik->amik',I1B,t2b,optimize=True)
    B3 = -1.0*np.einsum('amik,bcjm->abcijk',I1,t2b,optimize=True)
    D_ij_ab += B3

    # additional terms with T3
    I1 = -1.0*np.einsum('me,bm->be',H1A['ov'],r1a,optimize=True)\
         +np.einsum('bnef,fn->be',H2A['vovv'],r1a,optimize=True)\
         +np.einsum('bnef,fn->be',H2B['vovv'],r1b,optimize=True)
    D1 = np.einsum('be,aecijk->abcijk',I1,t3b,optimize=True)
    D_ab += D1

    I1 = -1.0*np.einsum('me,cm->ce',H1B['ov'],r1b,optimize=True)\
         +np.einsum('ncfe,fn->ce',H2B['ovvv'],r1a,optimize=True)\
         +np.einsum('cnef,fn->ce',H2C['vovv'],r1b,optimize=True)
    D2 = np.einsum('ce,abeijk->abcijk',I1,t3b,optimize=True)
    X3B += D2

    I1 = np.einsum('me,ej->mj',H1A['ov'],r1a,optimize=True)\
        +np.einsum('mnjf,fn->mj',H2A['ooov'],r1a,optimize=True)\
        +np.einsum('mnjf,fn->mj',H2B['ooov'],r1b,optimize=True)
    D3 = -1.0*np.einsum('mj,abcimk->abcijk',I1,t3b,optimize=True)
    D_ij += D3

    I1 = np.einsum('me,ek->mk',H1B['ov'],r1b,optimize=True)\
        +np.einsum('nmfk,fn->mk',H2B['oovo'],r1a,optimize=True)\
        +np.einsum('mnkf,fn->mk',H2C['ooov'],r1b,optimize=True)
    D4 = -1.0*np.einsum('mk,abcijm->abcijk',I1,t3b,optimize=True)
    X3B += D4

    I1 = np.einsum('nmje,ek->nmjk',H2B['ooov'],r1b,optimize=True)\
        +np.einsum('nmek,ej->nmjk',H2B['oovo'],r1a,optimize=True)
    D5 = np.einsum('nmjk,abcinm->abcijk',I1,t3b,optimize=True)
    D_ij += D5

    I1 = np.einsum('mnie,ej->mnij',H2A['ooov'],r1a,optimize=True)
    I1 -= np.transpose(I1,(0,1,3,2))
    D6 = 0.5*np.einsum('mnij,abcmnk->abcijk',I1,t3b,optimize=True)
    X3B += D6

    I1 = -1.0*np.einsum('bmfe,cm->bcfe',H2B['vovv'],r1b,optimize=True)\
         -np.einsum('mcfe,bm->bcfe',H2B['ovvv'],r1a,optimize=True)
    D7 = np.einsum('bcfe,afeijk->abcijk',I1,t3b,optimize=True)
    D_ab += D7

    I1 = -1.0*np.einsum('amef,bm->abef',H2A['vovv'],r1a,optimize=True)
    I1 -= np.transpose(I1,(1,0,2,3))
    D8 = 0.5*np.einsum('abef,efcijk->abcijk',I1,t3b,optimize=True)
    X3B += D8

    I1 = -1.0*np.einsum('nmfk,cm->ncfk',H2B['oovo'],r1b,optimize=True)\
         +np.einsum('ncfe,ek->ncfk',H2B['ovvv'],r1b,optimize=True)
    D9 = np.einsum('ncfk,abfijn->abcijk',I1,t3a,optimize=True)
    X3B += D9

    I1 = -1.0*np.einsum('mnkf,cm->cnkf',H2C['ooov'],r1b,optimize=True)\
         +np.einsum('cnef,ek->cnkf',H2C['vovv'],r1b,optimize=True)
    D10 = np.einsum('cnkf,abfijn->abcijk',I1,t3b,optimize=True)
    X3B += D10

    I1 = np.einsum('bmfe,ek->bmfk',H2B['vovv'],r1b,optimize=True)\
        -np.einsum('nmfk,bn->bmfk',H2B['oovo'],r1a,optimize=True)
    D11 = -1.0*np.einsum('bmfk,afcijm->abcijk',I1,t3b,optimize=True)
    D_ab += D11

    I1 = -1.0*np.einsum('nmje,cm->ncje',H2B['ooov'],r1b,optimize=True)\
         +np.einsum('ncfe,fj->ncje',H2B['ovvv'],r1a,optimize=True)
    D12 = -1.0*np.einsum('ncje,abeink->abcijk',I1,t3b,optimize=True)
    D_ij += D12

    I1 = -1.0*np.einsum('nmje,bn->bmje',H2A['ooov'],r1a,optimize=True)\
         +np.einsum('bmfe,fj->bmje',H2A['vovv'],r1a,optimize=True)
    D13 = np.einsum('bmje,aecimk->abcijk',I1,t3b,optimize=True)
    D_ij_ab += D13

    I1 = -1.0*np.einsum('nmje,bn->bmje',H2B['ooov'],r1a,optimize=True)\
         +np.einsum('bmfe,fj->bmje',H2B['vovv'],r1a,optimize=True)
    D14 = np.einsum('bmje,aecimk->abcijk',I1,t3c,optimize=True)
    D_ij_ab += D14

    # < ijk~abc~ | (HR2)_C | 0 >
    I1 = 0.5*np.einsum('nmje,abmn->abej',H2A['ooov'],r2a,optimize=True)
    D1 = np.einsum('abej,ecik->abcijk',I1,t2b,optimize=True)
    D_ij += D1

    I1 = 0.5*np.einsum('bmef,efji->mbij',H2A['vovv'],r2a,optimize=True)
    D2 = -1.0*np.einsum('mbij,acmk->abcijk',I1,t2b,optimize=True)
    D_ab += D2

    I1 = np.einsum('mnek,acmn->acek',H2B['oovo'],r2b,optimize=True)
    I2 = np.einsum('mnie,acmn->acie',H2B['ooov'],r2b,optimize=True)
    A1 = np.einsum('acek,ebij->abcijk',I1,t2a,optimize=True)
    D_ab += A1
    A2 = np.einsum('acie,bejk->abcijk',I2,t2b,optimize=True)
    D_ij_ab += A2

    I1 = np.einsum('mcef,efik->mcik',H2B['ovvv'],r2b,optimize=True)
    I2 = np.einsum('amef,efik->amik',H2B['vovv'],r2b,optimize=True)
    A1 = -1.0*np.einsum('mcik,abmj->abcijk',I1,t2a,optimize=True)
    D_ij += A1
    A2 = -1.0*np.einsum('amik,bcjm->abcijk',I2,t2b,optimize=True)
    D_ij_ab += A2

    I1 = np.einsum('nmie,ecmk->ncik',H2A['ooov'],r2b,optimize=True)
    I2 = np.einsum('amfe,ecmk->acfk',H2A['vovv'],r2b,optimize=True)
    A1 = -1.0*np.einsum('ncik,abnj->abcijk',I1,t2a,optimize=True)
    D_ij += A1
    A2 = np.einsum('acfk,fbij->abcijk',I2,t2a,optimize=True)
    D_ab += A2

    I1 = np.einsum('nmie,ecmk->ncik',H2B['ooov'],r2c,optimize=True)
    I2 = np.einsum('amfe,ecmk->acfk',H2B['vovv'],r2c,optimize=True)
    A1 = -1.0*np.einsum('ncik,abnj->abcijk',I1,t2a,optimize=True)
    D_ij += A1
    A2 = np.einsum('acfk,fbij->abcijk',I2,t2a,optimize=True)
    D_ab += A2

    I1 = np.einsum('nmie,ebmj->nbij',H2A['ooov'],r2a,optimize=True)
    I1 -= np.transpose(I1,(0,1,3,2))
    I2 = np.einsum('mnek,bejm->bnjk',H2B['oovo'],r2a,optimize=True)
    I3 = np.einsum('amfe,bejm->abfj',H2A['vovv'],r2a,optimize=True)
    I3 -= np.transpose(I3,(1,0,2,3))
    I4 = np.einsum('mcef,bejm->bcjf',H2B['ovvv'],r2a,optimize=True)
    A1 = -1.0*np.einsum('nbij,acnk->abcijk',I1,t2b,optimize=True)
    D_ab += A1
    A2 = -1.0*np.einsum('bnjk,acin->abcijk',I2,t2b,optimize=True)
    A3 = np.einsum('abfj,fcik->abcijk',I3,t2b,optimize=True)
    D_ij += A3
    A4 = np.einsum('bcjf,afik->abcijk',I4,t2b,optimize=True)
    B1 = A2 + A4
    D_ij_ab += B1

    I1 = np.einsum('nmie,bejm->nbij',H2B['ooov'],r2b,optimize=True)
    I1 -= np.transpose(I1,(0,1,3,2))
    I2 = np.einsum('amfe,bejm->abfj',H2B['vovv'],r2b,optimize=True)
    I2 -= np.transpose(I2,(1,0,2,3))
    I3 = np.einsum('cmfe,bejm->bcjf',H2C['vovv'],r2b,optimize=True)
    I4 = np.einsum('nmke,bejm->bnjk',H2C['ooov'],r2b,optimize=True)
    A1 = -1.0*np.einsum('nbij,acnk->abcijk',I1,t2b,optimize=True)
    D_ab += A1
    A2 = np.einsum('abfj,fcik->abcijk',I2,t2b,optimize=True)
    D_ij += A2
    A3 = np.einsum('bcjf,afik->abcijk',I3,t2b,optimize=True)
    A4 = -1.0*np.einsum('bnjk,acin->abcijk',I4,t2b,optimize=True)
    B1 = A3 + A4
    D_ij_ab += B1

    I1 = -1.0*np.einsum('nmek,ecim->ncik',H2B['oovo'],r2b,optimize=True)
    I2 = -1.0*np.einsum('amef,ecim->acif',H2B['vovv'],r2b,optimize=True)
    A1 = -1.0*np.einsum('ncik,abnj->abcijk',I1,t2a,optimize=True)
    D_ij += A1
    A2 = np.einsum('acif,bfjk->abcijk',I2,t2b,optimize=True)
    D_ij_ab += A2

    I1 = -1.0*np.einsum('mnie,aemk->anik',H2B['ooov'],r2b,optimize=True)
    I2 = -1.0*np.einsum('mcfe,aemk->acfk',H2B['ovvv'],r2b,optimize=True)
    A1 = -1.0*np.einsum('anik,bcjn->abcijk',I1,t2b,optimize=True)
    D_ij_ab += A1
    A2 = np.einsum('acfk,fbij->abcijk',I2,t2a,optimize=True)
    D_ab += A2

    D11 = -1.0*np.einsum('mcjk,abim->abcijk',H2B['ovoo'],r2a,optimize=True)
    D_ij += D11

    D12 = -1.0*np.einsum('amij,bcmk->abcijk',H2A['vooo'],r2b,optimize=True)
    D_ab += D12

    D13 = -1.0*np.einsum('amik,bcjm->abcijk',H2B['vooo'],r2b,optimize=True)
    D_ij_ab += D13

    D14 = np.einsum('bcek,aeij->abcijk',H2B['vvvo'],r2a,optimize=True)
    D_ab += D14

    D15 = np.einsum('abie,ecjk->abcijk',H2A['vvov'],r2b,optimize=True)
    D_ij += D15

    D16 = np.einsum('acie,bejk->abcijk',H2B['vvov'],r2b,optimize=True)
    D_ij_ab += D16

    # additional terms with T3
    I1 = 0.5*np.einsum('mnef,efij->mnij',vA['oovv'],r2a,optimize=True)
    D1 = 0.5*np.einsum('mnij,abcmnk->abcijk',I1,t3b,optimize=True)
    X3B += D1

    I1 = np.einsum('mnef,efjk->mnjk',vB['oovv'],r2b,optimize=True)
    D2 = np.einsum('mnjk,abcimn->abcijk',I1,t3b,optimize=True)
    D_ij += D2

    I1 = 0.5*np.einsum('mnef,abmn->abef',vA['oovv'],r2a,optimize=True)
    D3 = 0.5*np.einsum('abef,efcijk->abcijk',I1,t3b,optimize=True)
    X3B += D3

    I1 = np.einsum('mnef,bcmn->bcef',vB['oovv'],r2b,optimize=True)
    D4 = np.einsum('bcef,aefijk->abcijk',I1,t3b,optimize=True)
    D_ab += D4

    I1 = np.einsum('mnef,ecmk->ncfk',vA['oovv'],r2b,optimize=True)\
        +np.einsum('nmfe,ecmk->ncfk',vB['oovv'],r2c,optimize=True)
    D5 = np.einsum('ncfk,abfijn->abcijk',I1,t3a,optimize=True)
    X3B += D5

    I1 = np.einsum('mnef,ecmk->ncfk',vB['oovv'],r2b,optimize=True)\
        +np.einsum('mnef,ecmk->ncfk',vC['oovv'],r2c,optimize=True)
    D6 = np.einsum('ncfk,abfijn->abcijk',I1,t3b,optimize=True)
    X3B += D6

    I1 = np.einsum('mnef,aeim->anif',vA['oovv'],r2a,optimize=True)\
        +np.einsum('nmfe,aeim->anif',vB['oovv'],r2b,optimize=True)
    D7 = np.einsum('anif,bfcjnk->abcijk',I1,t3b,optimize=True)
    D_ij_ab += D7

    I1 = np.einsum('mnef,aeim->anif',vB['oovv'],r2a,optimize=True)\
        +np.einsum('mnef,aeim->anif',vC['oovv'],r2b,optimize=True)
    D8 = np.einsum('anif,bfcjnk->abcijk',I1,t3c,optimize=True)
    D_ij_ab += D8

    I1 = -1.0*np.einsum('mnef,bfmk->bnek',vB['oovv'],r2b,optimize=True)
    D9 = -1.0*np.einsum('bnek,aecijn->abcijk',I1,t3b,optimize=True)
    D_ab += D9

    I1 = -1.0*np.einsum('mnef,ecjn->mcjf',vB['oovv'],r2b,optimize=True)
    D10 = -1.0*np.einsum('mcjf,abfimk->abcijk',I1,t3b,optimize=True)
    D_ij += D10

    I1 = np.einsum('nmfe,fenk->mk',vB['oovv'],r2b,optimize=True)\
        +0.5*np.einsum('mnef,efkn->mk',vC['oovv'],r2c,optimize=True)
    D11 = -1.0*np.einsum('mk,abcijm->abcijk',I1,t3b,optimize=True)
    X3B += D11

    I1 = -1.0*np.einsum('nmfe,fcnm->ce',vB['oovv'],r2b,optimize=True)\
         -0.5*np.einsum('mnef,fcnm->ce',vC['oovv'],r2c,optimize=True)
    D12 = np.einsum('ce,abeijk->abcijk',I1,t3b,optimize=True)
    X3B += D12

    I1 = 0.5*np.einsum('mnef,efjn->mj',vA['oovv'],r2a,optimize=True)\
        +np.einsum('mnef,efjn->mj',vB['oovv'],r2b,optimize=True)
    D13 = -1.0*np.einsum('mj,abcimk->abcijk',I1,t3b,optimize=True)
    D_ij += D13

    I1 = -0.5*np.einsum('mnef,bfmn->be',vA['oovv'],r2a,optimize=True)\
         -np.einsum('mnef,bfmn->be',vB['oovv'],r2b,optimize=True)
    D14 = np.einsum('be,aecijk->abcijk',I1,t3b,optimize=True)
    D_ab += D14

    # < ijk~abc~ | (HR3)_C | 0 >
    D1 = -1.0*np.einsum('mj,abcimk->abcijk',H1A['oo'],r3b,optimize=True)
    D_ij += D1

    D2 = -1.0*np.einsum('mk,abcijm->abcijk',H1B['oo'],r3b,optimize=True)
    X3B += D2

    D3 = np.einsum('be,aecijk->abcijk',H1A['vv'],r3b,optimize=True)
    D_ab += D3

    D4 = np.einsum('ce,abeijk->abcijk',H1B['vv'],r3b,optimize=True)
    X3B += D4

    D5 = 0.5*np.einsum('mnij,abcmnk->abcijk',H2A['oooo'],r3b,optimize=True)
    X3B += D5

    D6 = np.einsum('mnjk,abcimn->abcijk',H2B['oooo'],r3b,optimize=True)
    D_ij += D6

    D7 = 0.5*np.einsum('abef,efcijk->abcijk',H2A['vvvv'],r3b,optimize=True)
    X3B += D7

    D8 = np.einsum('bcef,aefijk->abcijk',H2B['vvvv'],r3b,optimize=True)
    D_ab += D8

    D9 = np.einsum('amie,ebcmjk->abcijk',H2A['voov'],r3b,optimize=True)
    D_ij_ab += D9

    D10 = np.einsum('amie,becjmk->abcijk',H2B['voov'],r3c,optimize=True)
    D_ij_ab += D10

    D11 = np.einsum('mcek,abeijm->abcijk',H2B['ovvo'],r3a,optimize=True)
    X3B += D11

    D12 = np.einsum('cmke,abeijm->abcijk',H2C['voov'],r3b,optimize=True)
    X3B += D12

    D13 = -1.0*np.einsum('bmek,aecijm->abcijk',H2B['vovo'],r3b,optimize=True)
    D_ab += D13

    D14 = -1.0*np.einsum('mcje,abeimk->abcijk',H2B['ovov'],r3b,optimize=True)
    D_ij += D14

    I1 = -0.5*np.einsum('mnef,abfimn->abie',vA['oovv'],r3a,optimize=True)\
         -np.einsum('mnef,abfimn->abie',vB['oovv'],r3b,optimize=True)
    D15 = np.einsum('abie,ecjk->abcijk',I1,t2b,optimize=True)
    D_ij += D15

    I1 = -0.5*np.einsum('mnef,bfcmnk->bcek',vA['oovv'],r3b,optimize=True)\
         -np.einsum('mnef,bfcmnk->bcek',vB['oovv'],r3c,optimize=True)
    D16 = np.einsum('bcek,aeij->abcijk',I1,t2a,optimize=True)
    D_ab += D16

    I1 = -1.0*np.einsum('nmfe,bfcjnm->bcje',vB['oovv'],r3b,optimize=True)\
         -0.5*np.einsum('mnef,bfcjnm->bcje',vC['oovv'],r3c,optimize=True)
    D17 = np.einsum('bcje,aeik->abcijk',I1,t2b,optimize=True)
    D_ij_ab += D17

    I1 = 0.5*np.einsum('mnef,aefijn->amij',vA['oovv'],r3a,optimize=True)\
        +np.einsum('mnef,aefijn->amij',vB['oovv'],r3b,optimize=True)
    D18 = -1.0*np.einsum('amij,bcmk->abcijk',I1,t2b,optimize=True)
    D_ab += D18

    I1 = 0.5*np.einsum('mnef,efcjnk->mcjk',vA['oovv'],r3b,optimize=True)\
        +np.einsum('mnef,efcjnk->mcjk',vB['oovv'],r3c,optimize=True)
    D19 = -1.0*np.einsum('mcjk,abim->abcijk',I1,t2a,optimize=True)
    D_ij += D19

    I1 = np.einsum('nmfe,bfejnk->bmjk',vB['oovv'],r3b,optimize=True)\
        +0.5*np.einsum('mnef,befjkn->bmjk',vC['oovv'],r3c,optimize=True)
    D20 = -1.0*np.einsum('bmjk,acim->abcijk',I1,t2b,optimize=True)
    D_ij_ab += D20

    D_ij -= np.transpose(D_ij,(0,1,2,4,3,5))
    D_ab -= np.transpose(D_ab,(1,0,2,3,4,5))
    D_ij_ab -= np.transpose(D_ij_ab,(0,1,2,4,3,5)) + np.transpose(D_ij_ab,(1,0,2,3,4,5)) - np.transpose(D_ij_ab,(1,0,2,4,3,5))

    X3B += D_ij + D_ab + D_ij_ab

    return X3B

def cis(nroot,ints,sys):

    fA = ints['fA']
    fB = ints['fB']
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']

    n1a = sys['Nocc_a'] * sys['Nunocc_a']
    n1b = sys['Nocc_b'] * sys['Nunocc_b']

    HAA = np.zeros((n1a,n1a))
    HAB = np.zeros((n1a,n1b))
    HBA = np.zeros((n1b,n1a))
    HBB = np.zeros((n1b,n1b))

    ct1 = 0 
    for a in range(sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            ct2 = 0 
            for b in range(sys['Nunocc_a']):
                for j in range(sys['Nocc_a']):
                    HAA[ct1,ct2] += vA['voov'][a,j,i,b]
                    HAA[ct1,ct2] += (i == j) * fA['vv'][a,b]
                    HAA[ct1,ct2] -= (a == b) * fA['oo'][j,i]
                    ct2 += 1
            ct1 += 1
    ct1 = 0
    for a in range(sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            ct2 = 0 
            for b in range(sys['Nunocc_b']):
                for j in range(sys['Nocc_b']):
                    HAB[ct1,ct2] += vB['voov'][a,j,i,b]
                    ct2 += 1
            ct1 += 1
    ct1 = 0
    for a in range(sys['Nunocc_b']):
        for i in range(sys['Nocc_b']):
            ct2 = 0 
            for b in range(sys['Nunocc_a']):
                for j in range(sys['Nocc_a']):
                    HBA[ct1,ct2] += vB['ovvo'][j,a,b,i]
                    ct2 += 1
            ct1 += 1
    ct1 = 0 
    for a in range(sys['Nunocc_b']):
        for i in range(sys['Nocc_b']):
            ct2 = 0 
            for b in range(sys['Nunocc_b']):
                for j in range(sys['Nocc_b']):
                    HBB[ct1,ct2] += vC['voov'][a,j,i,b]
                    HBB[ct1,ct2] += (i == j) * fB['vv'][a,b]
                    HBB[ct1,ct2] -= (a == b) * fB['oo'][j,i]
                    ct2 += 1
            ct1 += 1

    H = np.hstack( (np.vstack((HAA,HBA)), np.vstack((HAB,HBB))) )

    E_cis, C = np.linalg.eigh(H) 
    idx = np.argsort(E_cis)
    E_cis = E_cis[idx]
    C = C[:,idx]

    return C, E_cis

def calc_cc_energy(cc_t,ints):

    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    fA = ints['fA']
    fB = ints['fB']
    t1a = cc_t['t1a']
    t1b = cc_t['t1b']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']

    Ecorr = 0.0
    Ecorr += np.einsum('me,em->',fA['ov'],t1a,optimize=True)
    Ecorr += np.einsum('me,em->',fB['ov'],t1b,optimize=True)
    Ecorr += 0.25*np.einsum('mnef,efmn->',vA['oovv'],t2a,optimize=True)
    Ecorr += np.einsum('mnef,efmn->',vB['oovv'],t2b,optimize=True)
    Ecorr += 0.25*np.einsum('mnef,efmn->',vC['oovv'],t2c,optimize=True)
    Ecorr += 0.5*np.einsum('mnef,fn,em->',vA['oovv'],t1a,t1a,optimize=True)
    Ecorr += 0.5*np.einsum('mnef,fn,em->',vC['oovv'],t1b,t1b,optimize=True)
    Ecorr += np.einsum('mnef,em,fn->',vB['oovv'],t1a,t1b,optimize=True)

    return Ecorr

def test_updates(matfile,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):

    from scipy.io import loadmat
    #from fortran_cis import cis_hamiltonian

    print('')
    print('TEST SUBROUTINE:')
    print('Loading Matlab .mat file from {}'.format(matfile))
    print('')

    data_dict = loadmat(matfile)
    Rvec = data_dict['Rvec']

    fA = ints['fA']
    fB = ints['fB']
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']


    for j in range(Rvec.shape[1]):

        print('Testing updates on root {}'.format(j+1))
        print("----------------------------------------")

        #r1a = data_dict['r1a'+'-'+str(j+1)]
        #r1b = data_dict['r1b'+'-'+str(j+1)]
        #r2a = data_dict['r2a'+'-'+str(j+1)]
        #r2b = data_dict['r2b'+'-'+str(j+1)]
        #r2c = data_dict['r2c'+'-'+str(j+1)]
        r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d = unflatten_R(Rvec[:,j],sys,order='F')

        # test r1a update
        X1A = build_HR_1A(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        print('|X1A| = {}'.format(np.linalg.norm(X1A)))

        # test r1b update
        X1B = build_HR_1B(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        print('|X1B| = {}'.format(np.linalg.norm(X1B)))

        # test r2a update
        X2A = build_HR_2A(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        print('|X2A| = {}'.format(np.linalg.norm(X2A)))

        # test r2b update
        X2B = build_HR_2B(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        print('|X2B| = {}'.format(np.linalg.norm(X2B)))

        # test t2c update
        X2C = build_HR_2C(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        print('|X2C| = {}'.format(np.linalg.norm(X2C)))
    
        X3A = build_HR_3A(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        print('|X3A| = {}'.format(np.linalg.norm(X3A)))

        X3B = build_HR_3B(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        print('|X3B| = {}'.format(np.linalg.norm(X3B)))
    return
