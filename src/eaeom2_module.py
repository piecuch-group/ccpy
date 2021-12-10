"""Module containing functions to calculate the vertical ionization
energies and linear excitation amplitudes for excited states of an
N+1 electron system out of the CC ground state of an N electron system 
using the electron attachment (EA) equation-of-motion (EOM) CC with 
singles and doubles (EA-EOMCCSD) with up to 2p-1h excitations."""
import numpy as np
from cc_energy import calc_cc_energy
import cc_loops

def eaeom2(nroot,H1A,H1B,H2A,H2B,H2C,cc_t,ints,sys,initial_guess='cis',tol=1.0e-06,maxit=80):
    print('\n==================================++Entering EA-EOMCCSD(2p-1h) Routine++=================================\n')

    if initial_guess == 'cis':
        C_1p, E_1p = guess_1p(ints,sys)
        B0 = np.zeros((sys['Nunocc_a']+sys['Nunocc_b'],nroot))
        E0 = np.zeros(nroot)
        ct = 0
        for i in reversed(range(len(E_1p))):
            B0[:,ct] = C_1p[:,i]
            E0[ct] = -1.0 * E_1p[i]
            if ct+1 == nroot:
                break
            ct += 1
        print('Initial 1p Energies:')
        for i in range(nroot):
            print('Root - {}     E = {:.10f}'.format(i+1,E0[i]))
        print('')
        n_2p1h = sys['Nocc_a']*sys['Nunocc_a']**2\
                    +sys['Nunocc_b']*sys['Nocc_a']*sys['Nunocc_a']\
                    +sys['Nunocc_a']*sys['Nocc_b']*sys['Nunocc_b']\
                    +sys['Nocc_b']*sys['Nunocc_b']**2
        ZEROS_2p1h = np.zeros((n_2p1h,nroot))
        B0 = np.concatenate((B0,ZEROS_2p1h),axis=0)

    Rvec, omega, is_converged = davidson_solver(H1A,H1B,H2A,H2B,H2C,ints,cc_t,nroot,B0,E0,sys,maxit,tol)
    
    cc_t['r1a'] = [None]*len(omega)
    cc_t['r1b'] = [None]*len(omega)
    cc_t['r2a'] = [None]*len(omega)
    cc_t['r2b'] = [None]*len(omega)
    cc_t['r2c'] = [None]*len(omega)
    cc_t['r2d'] = [None]*len(omega)

    print('Summary of EA-EOMCCSD(2p-1h):')
    Eccsd = ints['Escf'] + calc_cc_energy(cc_t,ints)
    for i in range(len(omega)):
        r1a,r1b,r2a,r2b,r2c,r2d = unflatten_R(Rvec[:,i],sys)
        cc_t['r1a'][i] = r1a
        cc_t['r1b'][i] = r1b
        cc_t['r2a'][i] = r2a
        cc_t['r2b'][i] = r2b
        cc_t['r2c'][i] = r2c
        cc_t['r2d'][i] = r2d
        if is_converged[i]:
            tmp = 'CONVERGED'
        else:
            tmp = 'NOT CONVERGED'
        print('   Root - {}    E = {}    omega = {:.10f}    [{}]'\
                        .format(i+1,omega[i]+Eccsd,omega[i],tmp))

    return cc_t, omega

def davidson_solver(H1A,H1B,H2A,H2B,H2C,ints,cc_t,nroot,B0,E0,sys,maxit,tol):
    """Diagonalize the CCSD similarity-transformed Hamiltonian HBar using the
    non-Hermitian Davidson algorithm.

    Parameters
    ----------
    H1*, H2* : dict
        Sliced CCSD similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    cc_t : dict
        Cluster amplitudes T1, T2 of the ground-state
    nroot : int
        Number of excited-states to solve for
    B0 : ndarray(dtype=float, shape=(ndim_ccsd,nroot))
        Matrix containing the initial guess vectors for the Davidson procedure
    E0 : ndarray(dtype=float, shape=(nroot))
        Vector containing the energies corresponding to the initial guess vectors
    sys : dict
        System information dictionary
    maxit : int, optional
        Maximum number of Davidson iterations in the EOMCC procedure.
    tol : float, optional
        Convergence tolerance for the EOMCC calculation. Default is 1.0e-06.

    Returns
    -------
    Rvec : ndarray(dtype=float, shape=(ndim_2p1h,nroot))
        Matrix containing the final converged R vectors corresponding to the EA-EOMCCSD linear excitation amplitudes
    omega : ndarray(dtype=float, shape=(nroot))
        Vector of vertical excitation energies (in hartree) for each root
    is_converged : list
        List of boolean indicating whether each root converged to within the specified tolerance
    """
    noa = H1A['ov'].shape[0]
    nob = H1B['ov'].shape[0]
    nua = H1A['ov'].shape[1]
    nub = H1B['ov'].shape[1]

    ndim = nua + nub + noa*nua**2 + nub*noa*nua + nua*nob*nub + nob*nub**2

    Rvec = np.zeros((ndim,nroot))
    is_converged = [False] * nroot
    omega = np.zeros(nroot)
    residuals = np.zeros(nroot)

    # orthognormalize the initial trial space
    B0,_ = np.linalg.qr(B0)

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
            q1a,q1b,q2a,q2b,q2c,q2d = unflatten_R(q,sys)
            q1a,q1b,q2a,q2b,q2c,q2d = cc_loops.cc_loops.update_r_2p1h(q1a,q1b,q2a,q2b,q2c,q2d,omega[iroot],\
                            H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],0.0,\
                            sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'])
            q = flatten_R(q1a,q1b,q2a,q2b,q2c,q2d)
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


def orthogonalize(q,B):
    """Orthogonalize the correction vector to the vectors comprising
    the current subspace.

    Parameters
    ----------
    q : ndarray(dtype=float, shape=(ndim_ccsd))
        Preconditioned residual vector from Davidson procedure
    B : ndarray(dtype=float, shape=(ndim_ccsd,curr_size))
        Matrix of subspace vectors in Davidson procedure

    Returns
    -------
    q : ndarray(dtype=float, shape=(ndim_ccsd))
        Orthogonalized residual vector
    """
    for i in range(B.shape[1]):
        b = B[:,i]/np.linalg.norm(B[:,i])
        q -= np.dot(b.T,q)*b
    return q

def flatten_R(r1a,r1b,r2a,r2b,r2c,r2d):
    """Flatten the R vector.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua))
        Linear EOMCC excitation amplitudes R1p(a)
    r1b : ndarray(dtype=float, shape=(nub))
        Linear EOMCC excitation amplitudes R1p(b)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(aaa)
    r2b : ndarray(dtype=float, shape=(nub,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(baa)
    r2c : ndarray(dtype=float, shape=(nua,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(abb)
    r2d : ndarray(dtype=float, shape=(nub,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(bbb)

    Returns
    -------
    R : ndarray(dtype=float, shape=(ndim_2p1h))
        Flattened array of R vector for the given root
    """
    return np.concatenate((r1a.flatten(),r1b.flatten(),r2a.flatten(),r2b.flatten(),r2c.flatten(),r2d.flatten()),axis=0)

def unflatten_R(R,sys,order='C'):
    """Unflatten the R vector into many-body tensor components.

    Parameters
    ----------
    R : ndarray(dtype=float, shape=(ndim_2p1h))
        Flattened array of R vector for the given root
    sys : dict
        System information dictionary
    order : str, optional
        String of value 'C' or 'F' indicating whether row-major or column-major
        flattening should be used. Default is 'C'.

    Returns
    ----------
    r1a : ndarray(dtype=float, shape=(nua))
        Linear EOMCC excitation amplitudes R1p(a)
    r1b : ndarray(dtype=float, shape=(nub))
        Linear EOMCC excitation amplitudes R1p(b)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(aaa)
    r2b : ndarray(dtype=float, shape=(nub,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(baa)
    r2c : ndarray(dtype=float, shape=(nua,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(abb)
    r2d : ndarray(dtype=float, shape=(nub,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(bbb)
    """
    n1a = sys['Nunocc_a']
    n1b = sys['Nunocc_b']
    n2a = sys['Nocc_a'] * sys['Nunocc_a']**2 
    n2b = sys['Nunocc_b'] * sys['Nocc_a'] * sys['Nunocc_a']
    n2c = sys['Nunocc_a'] * sys['Nocc_b'] * sys['Nunocc_b']
    n2d = sys['Nunocc_b'] ** 2 * sys['Nocc_b']
    idx_1a = slice(0,n1a)
    idx_1b = slice(n1a,n1a+n1b)
    idx_2a = slice(n1a+n1b,n1a+n1b+n2a)
    idx_2b = slice(n1a+n1b+n2a,n1a+n1b+n2a+n2b)
    idx_2c = slice(n1a+n1b+n2a+n2b,n1a+n1b+n2a+n2b+n2c)
    idx_2d = slice(n1a+n1b+n2a+n2b+n2c,n1a+n1b+n2a+n2b+n2c+n2d)

    r1a  = np.reshape(R[idx_1a],sys['Nunocc_a'],order=order)
    r1b  = np.reshape(R[idx_1b],sys['Nunocc_b'],order=order)
    r2a  = np.reshape(R[idx_2a],(sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a']),order=order)
    r2b  = np.reshape(R[idx_2b],(sys['Nunocc_b'],sys['Nunocc_a'],sys['Nocc_a']),order=order)
    r2c  = np.reshape(R[idx_2c],(sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_b']),order=order)
    r2d  = np.reshape(R[idx_2d],(sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b']),order=order)

    return r1a, r1b, r2a, r2b, r2c, r2d


def HR(R,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the matrix-vector product H(CCSD)*R.

    Parameters
    ----------
    R : ndarray(dtype=float, shape=(ndim_2p1h))
        Flattened vector of R amplitudes
    cc_t : dict
        Cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced CCSD similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    -------
    HR : ndarray(dtype=float, shape=(ndim_2p1h))
        Vector containing the matrix-vector product H(CCSD)*R
    """
    r1a, r1b, r2a, r2b, r2c, r2d = unflatten_R(R,sys)

    X1A = build_HR_1A(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X1B = build_HR_1B(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X2A = build_HR_2A(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X2B = build_HR_2B(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X2C = build_HR_2C(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X2D = build_HR_2D(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)

    return flatten_R(X1A, X1B, X2A, X2B, X2C, X2D)

def build_HR_1A(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <a|[ (H_N e^(T1+T2))_C*(R1p+R2p1h) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua))
        Linear EOMCC excitation amplitudes R1p(a)
    r1b : ndarray(dtype=float, shape=(nub))
        Linear EOMCC excitation amplitudes R1p(b)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(aaa)
    r2b : ndarray(dtype=float, shape=(nub,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(baa)
    r2c : ndarray(dtype=float, shape=(nua,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(abb)
    r2d : ndarray(dtype=float, shape=(nub,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced CCSD similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    --------
    X1A : ndarray(dtype=float, shape=(nua))
        Calculated HR Projection
    """
    X1A = 0.0
    X1A += np.einsum('ae,e->a',H1A['vv'],r1a,optimize=True)
    X1A += 0.5*np.einsum('anef,efn->a',H2A['vovv'],r2a,optimize=True)
    X1A += np.einsum('anef,efn->a',H2B['vovv'],r2c,optimize=True)
    X1A += np.einsum('me,aem->a',H1A['ov'],r2a,optimize=True)
    X1A += np.einsum('me,aem->a',H1B['ov'],r2c,optimize=True)

    return X1A

def build_HR_1B(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <a~|[ (H_N e^(T1+T2))_C*(R1p+R2p1h) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua))
        Linear EOMCC excitation amplitudes R1p(a)
    r1b : ndarray(dtype=float, shape=(nub))
        Linear EOMCC excitation amplitudes R1p(b)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(aaa)
    r2b : ndarray(dtype=float, shape=(nub,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(baa)
    r2c : ndarray(dtype=float, shape=(nua,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(abb)
    r2d : ndarray(dtype=float, shape=(nub,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced CCSD similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    --------
    X1B : ndarray(dtype=float, shape=(nub))
        Calculated HR Projection
    """
    X1B = 0.0
    X1B += np.einsum('ae,e->a',H1B['vv'],r1b,optimize=True)
    X1B += np.einsum('nafe,efn->a',H2B['ovvv'],r2b,optimize=True)
    X1B += 0.5*np.einsum('anef,efn->a',H2C['vovv'],r2d,optimize=True)
    X1B += np.einsum('me,aem->a',H1A['ov'],r2b,optimize=True)
    X1B += np.einsum('me,aem->a',H1B['ov'],r2d,optimize=True)

    return X1B

def build_HR_2A(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <jab|[ (H_N e^(T1+T2))_C*(R1p+R2p1h) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua))
        Linear EOMCC excitation amplitudes R1p(a)
    r1b : ndarray(dtype=float, shape=(nub))
        Linear EOMCC excitation amplitudes R1p(b)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(aaa)
    r2b : ndarray(dtype=float, shape=(nub,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(baa)
    r2c : ndarray(dtype=float, shape=(nua,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(abb)
    r2d : ndarray(dtype=float, shape=(nub,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced CCSD similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    --------
    X2A : ndarray(dtype=float, shape=(nua,nua,noa))
        Calculated HR Projection
    """
    vA = ints['vA']
    vB = ints['vB']
    t2a = cc_t['t2a']

    X2A = 0.0
    X2A += np.einsum('baje,e->abj',H2A['vvov'],r1a,optimize=True)
    X2A += np.einsum('mj,abm->abj',H1A['oo'],r2a,optimize=True)
    X2A += 0.5*np.einsum('abef,efj->abj',H2A['vvvv'],r2a,optimize=True)
    I1 = 0.5*np.einsum('mnef,efn->m',vA['oovv'],r2a,optimize=True)\
        +np.einsum('mnef,efn->m',vB['oovv'],r2c,optimize=True)
    X2A == np.einsum('m,abmj->abj',I1,t2a,optimize=True)

    D_ab = 0.0
    D_ab -= np.einsum('ae,ebj->abj',H1A['vv'],r2a,optimize=True)
    D_ab += np.einsum('bmje,aem->abj',H2A['voov'],r2a,optimize=True)
    D_ab += np.einsum('bmje,aem->abj',H2B['voov'],r2c,optimize=True)
    D_ab -= np.transpose(D_ab,(1,0,2))

    X2A += D_ab

    return X2A

def build_HR_2B(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <ja~b|[ (H_N e^(T1+T2))_C*(R1p+R2p1h) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua))
        Linear EOMCC excitation amplitudes R1p(a)
    r1b : ndarray(dtype=float, shape=(nub))
        Linear EOMCC excitation amplitudes R1p(b)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(aaa)
    r2b : ndarray(dtype=float, shape=(nub,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(baa)
    r2c : ndarray(dtype=float, shape=(nua,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(abb)
    r2d : ndarray(dtype=float, shape=(nub,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced CCSD similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    --------
    X2B : ndarray(dtype=float, shape=(nub,nua,noa))
        Calculated HR Projection
    """
    t2b = cc_t['t2b']
    vB = ints['vB']
    vC = ints['vC']

    X2B = 0.0
    X2B -= np.einsum('bmji,m->bij',H2B['vooo'],r1b,optimize=True)
    X2B -= np.einsum('mi,bmj->bij',H1B['oo'],r2b,optimize=True)
    X2B -= np.einsum('mj,bim->bij',H1A['oo'],r2b,optimize=True)
    X2B += np.einsum('be,eij->bij',H1A['vv'],r2b,optimize=True)
    X2B += np.einsum('nmji,bmn->bij',H2B['oooo'],r2b,optimize=True)
    X2B += np.einsum('bmje,eim->bij',H2A['voov'],r2b,optimize=True)
    X2B += np.einsum('bmje,eim->bij',H2B['voov'],r2d,optimize=True)
    X2B -= np.einsum('bmei,emj->bij',H2B['vovo'],r2b,optimize=True)
    I1 = -np.einsum('nmfe,fmn->e',vB['oovv'],r2b,optimize=True)\
        -0.5*np.einsum('mnef,fmn->e',vC['oovv'],r2d,optimize=True)
    X2B += np.einsum('e,beji->bij',I1,t2b,optimize=True)

    return X2B

def build_HR_2C(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <j~ab~|[ (H_N e^(T1+T2))_C*(R1p+R2p1h) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua))
        Linear EOMCC excitation amplitudes R1p(a)
    r1b : ndarray(dtype=float, shape=(nub))
        Linear EOMCC excitation amplitudes R1p(b)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(aaa)
    r2b : ndarray(dtype=float, shape=(nub,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(baa)
    r2c : ndarray(dtype=float, shape=(nua,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(abb)
    r2d : ndarray(dtype=float, shape=(nub,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced CCSD similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    --------
    X2C : ndarray(dtype=float, shape=(nua,nub,nob))
        Calculated HR Projection
    """
    t2b = cc_t['t2b']
    vA = ints['vA']
    vB = ints['vB']

    X2C = 0.0
    X2C -= np.einsum('mbij,m->bij',H2B['ovoo'],r1a,optimize=True)
    X2C -= np.einsum('mi,bmj->bij',H1A['oo'],r2c,optimize=True)
    X2C -= np.einsum('mj,bim->bij',H1B['oo'],r2c,optimize=True)
    X2C += np.einsum('be,eij->bij',H1B['vv'],r2c,optimize=True)
    X2C += np.einsum('mnij,bmn->bij',H2B['oooo'],r2c,optimize=True)
    X2C += np.einsum('mbej,eim->bij',H2B['ovvo'],r2a,optimize=True)
    X2C += np.einsum('bmje,eim->bij',H2C['voov'],r2c,optimize=True)
    X2C -= np.einsum('mbie,emj->bij',H2B['ovov'],r2c,optimize=True)
    I1 = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],r2a,optimize=True)\
        -np.einsum('mnef,fmn->e',vB['oovv'],r2c,optimize=True)
    X2C += np.einsum('e,ebij->bij',I1,t2b,optimize=True)

    return X2C

def build_HR_2D(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <j~a~b~|[ (H_N e^(T1+T2))_C*(R1p+R2p1h) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua))
        Linear EOMCC excitation amplitudes R1p(a)
    r1b : ndarray(dtype=float, shape=(nub))
        Linear EOMCC excitation amplitudes R1p(b)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(aaa)
    r2b : ndarray(dtype=float, shape=(nub,nua,noa))
        Linear EOMCC excitation amplitudes R2p1h(baa)
    r2c : ndarray(dtype=float, shape=(nua,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(abb)
    r2d : ndarray(dtype=float, shape=(nub,nub,nob))
        Linear EOMCC excitation amplitudes R2p1h(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced CCSD similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    --------
    X2D : ndarray(dtype=float, shape=(nub,nub,nob))
        Calculated HR Projection
    """
    vB = ints['vB']
    vC = ints['vC']
    t2c = cc_t['t2c']

    X2D = 0.0
    X2D -= np.einsum('bmji,m->bij',H2C['vooo'],r1b,optimize=True)
    X2D += np.einsum('be,eij->bij',H1B['vv'],r2d,optimize=True)
    X2D += 0.5*np.einsum('mnij,bmn->bij',H2C['oooo'],r2d,optimize=True)
    I1 = -0.5*np.einsum('mnef,fmn->e',vC['oovv'],r2d,optimize=True)\
        -np.einsum('nmfe,fmn->e',vB['oovv'],r2b,optimize=True)
    X2D += np.einsum('e,ebij->bij',I1,t2c,optimize=True)

    D_ij = 0.0
    D_ij -= np.einsum('mi,bmj->bij',H1B['oo'],r2d,optimize=True)
    D_ij += np.einsum('bmje,eim->bij',H2C['voov'],r2d,optimize=True)
    D_ij += np.einsum('mbej,eim->bij',H2B['ovvo'],r2b,optimize=True)
    D_ij -= np.transpose(D_ij,(0,2,1))

    X2D += D_ij

    return X2D

def guess_1p(ints,sys):
    """Build and diagonalize the Hamiltonian in the space of 1p excitations.

    Parameters
    ----------
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    -------
    C : ndarray(dtype=float, shape=(ndim_1p,ndim_1p))
        Matrix of 1p eigenvectors
    E_1h : ndarray(dtype=float, shape=(ndim_cis))
        Vector of 1p eigenvalues
    """
    fA = ints['fA']
    fB = ints['fB']

    n1a = sys['Nunocc_a']
    n1b = sys['Nunocc_b']

    HAA = np.zeros((n1a,n1a))
    HAB = np.zeros((n1a,n1b))
    HBA = np.zeros((n1b,n1a))
    HBB = np.zeros((n1b,n1b))

    ct1 = 0
    for a in range(sys['Nunocc_a']):
        ct2 = 0
        for b in range(sys['Nunocc_a']):
            HAA[ct1,ct2] = fA['vv'][a,b]
            ct2 += 1
        ct1+=1

    ct1 = 0
    for a in range(sys['Nunocc_b']):
        ct2 = 0
        for b in range(sys['Nunocc_b']):
            HBB[ct1,ct2] = fB['vv'][a,b]
            ct2 += 1
        ct1 += 1

    H = np.hstack( (np.vstack((HAA,HBA)), np.vstack((HAB,HBB))) )

    E_1p, C = np.linalg.eigh(H) 
    idx = np.argsort(E_1p)
    E_1p = E_1p[idx]
    C = C[:,idx]

    return C, E_1p
