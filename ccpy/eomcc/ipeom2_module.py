"""Module containing functions to calculate the vertical ionization
energies and linear excitation amplitudes for excited states of an
N-1 electron system out of the CC ground state of an N electron system 
using the ionization process (IP) equation-of-motion (EOM) CC with 
singles and doubles (IP-EOMCCSD) with up to 2h-1p excitations."""
import numpy as np
from cc_energy import calc_cc_energy
from solvers import davidson_out_of_core
from functools import partial
import cc_loops

def ipeom2(nroot,H1A,H1B,H2A,H2B,H2C,cc_t,ints,sys,noact=0,nuact=0,tol=1.0e-06,maxit=80,flag_RHF=False):
    print('\n==================================++Entering IP-EOMCCSD(2h-1p) Routine++=================================\n')

    num_roots_total = sum(nroot)

    C_1h, E_1h = guess_1h(ints,sys)
    B0 = np.zeros((sys['Nocc_a']+sys['Nocc_b'],num_roots_total))
    E0 = np.zeros(num_roots_total)
    ct = 0
    for i in reversed(range(len(E_1h))):
        B0[:,ct] = C_1h[:,i]
        E0[ct] = -1.0 * E_1h[i]
        if ct+1 == num_roots_total:
            break
        ct += 1
    print('Initial 1h Energies:')
    for i in range(num_roots_total):
        print('Root - {}     E = {:.10f}'.format(i+1,E0[i]))
    print('')
    n_2h1p = sys['Nocc_a']**2*sys['Nunocc_a']\
                    +sys['Nocc_a']*sys['Nocc_b']*sys['Nunocc_a']\
                    +sys['Nocc_b']*sys['Nocc_a']*sys['Nunocc_b']\
                    +sys['Nocc_b']**2*sys['Nunocc_b']
    ZEROS_2h1p = np.zeros((n_2h1p,num_roots_total))
    B0 = np.concatenate((B0,ZEROS_2h1p),axis=0)

    # Get the HR function
    HR_func = partial(HR,cc_t=cc_t,H1A=H1A,H1B=H1B,H2A=H2A,H2B=H2B,H2C=H2C,ints=ints,sys=sys,flag_RHF=flag_RHF)
    # Get the R update function
    update_R_func = lambda r,omega : update_R(r,omega,H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],sys)
    # Diagonalize Hamiltonian using Davidson algorithm
    Rvec, omega, is_converged = davidson_out_of_core(HR_func,update_R_func,B0,E0,maxit,tol)
    
    cc_t['r1a'] = [None]*len(omega)
    cc_t['r1b'] = [None]*len(omega)
    cc_t['r2a'] = [None]*len(omega)
    cc_t['r2b'] = [None]*len(omega)
    cc_t['r2c'] = [None]*len(omega)
    cc_t['r2d'] = [None]*len(omega)

    print('Summary of IP-EOMCCSD(2h-1p):')
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
        print('   Root - {}    E = {}    omega_IP = {:.10f}    omega = {:.10f}  [{}]'\
                        .format(i+1,omega[i]+Eccsd,omega[i],omega[i]-omega[0],tmp))

    return cc_t, omega

def update_R(r,omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,sys):

    r1a,r1b,r2a,r2b,r2c,r2d = unflatten_R(r,sys)
    r1a,r1b,r2a,r2b,r2c,r2d = cc_loops.cc_loops.update_r_2h1p(r1a,r1b,r2a,r2b,r2c,r2d,omega,\
                            H1A_oo,H1A_vv,H1B_oo,H1B_vv,0.0,\
                            sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'])
    return flatten_R(r1a,r1b,r2a,r2b,r2c,r2d)

def flatten_R(r1a,r1b,r2a,r2b,r2c,r2d):
    """Flatten the R vector.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(noa))
        Linear EOMCC excitation amplitudes R1h(a)
    r1b : ndarray(dtype=float, shape=(nob))
        Linear EOMCC excitation amplitudes R1h(b)
    r2a : ndarray(dtype=float, shape=(nua,noa,noa))
        Linear EOMCC excitation amplitudes R2h1p(aaa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa))
        Linear EOMCC excitation amplitudes R2h1p(aba)
    r2c : ndarray(dtype=float, shape=(nub,noa,nob))
        Linear EOMCC excitation amplitudes R2h1p(bab)
    r2d : ndarray(dtype=float, shape=(nub,nob,nob))
        Linear EOMCC excitation amplitudes R2h1p(bbb)

    Returns
    -------
    R : ndarray(dtype=float, shape=(ndim_2h1p))
        Flattened array of R vector for the given root
    """
    return np.concatenate((r1a.flatten(),r1b.flatten(),r2a.flatten(),r2b.flatten(),r2c.flatten(),r2d.flatten()),axis=0)

def unflatten_R(R,sys,order='C'):
    """Unflatten the R vector into many-body tensor components.

    Parameters
    ----------
    R : ndarray(dtype=float, shape=(ndim_2h1p))
        Flattened array of R vector for the given root
    sys : dict
        System information dictionary
    order : str, optional
        String of value 'C' or 'F' indicating whether row-major or column-major
        flattening should be used. Default is 'C'.

    Returns
    -------
    r1a : ndarray(dtype=float, shape=(noa))
        Linear EOMCC excitation amplitudes R1h(a)
    r1b : ndarray(dtype=float, shape=(nob))
        Linear EOMCC excitation amplitudes R1h(b)
    r2a : ndarray(dtype=float, shape=(nua,noa,noa))
        Linear EOMCC excitation amplitudes R2h1p(aaa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa))
        Linear EOMCC excitation amplitudes R2h1p(aba)
    r2c : ndarray(dtype=float, shape=(nub,noa,nob))
        Linear EOMCC excitation amplitudes R2h1p(bab)
    r2d : ndarray(dtype=float, shape=(nub,nob,nob))
        Linear EOMCC excitation amplitudes R2h1p(bbb)
    """
    n1a = sys['Nocc_a']
    n1b = sys['Nocc_b']
    n2a = sys['Nocc_a'] ** 2 * sys['Nunocc_a'] 
    n2b = sys['Nocc_a'] * sys['Nocc_b'] * sys['Nunocc_a']
    n2c = sys['Nocc_a'] * sys['Nocc_b'] * sys['Nunocc_b']
    n2d = sys['Nocc_b'] ** 2 * sys['Nunocc_b']
    idx_1a = slice(0,n1a)
    idx_1b = slice(n1a,n1a+n1b)
    idx_2a = slice(n1a+n1b,n1a+n1b+n2a)
    idx_2b = slice(n1a+n1b+n2a,n1a+n1b+n2a+n2b)
    idx_2c = slice(n1a+n1b+n2a+n2b,n1a+n1b+n2a+n2b+n2c)
    idx_2d = slice(n1a+n1b+n2a+n2b+n2c,n1a+n1b+n2a+n2b+n2c+n2d)

    r1a  = np.reshape(R[idx_1a],sys['Nocc_a'],order=order)
    r1b  = np.reshape(R[idx_1b],sys['Nocc_b'],order=order)
    r2a  = np.reshape(R[idx_2a],(sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a']),order=order)
    r2b  = np.reshape(R[idx_2b],(sys['Nunocc_a'],sys['Nocc_b'],sys['Nocc_a']),order=order)
    r2c  = np.reshape(R[idx_2c],(sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b']),order=order)
    r2d  = np.reshape(R[idx_2d],(sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b']),order=order)

    return r1a, r1b, r2a, r2b, r2c, r2d


def HR(R,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF):
    """Calculate the matrix-vector product H(CCSD)*R.

    Parameters
    ----------
    R : ndarray(dtype=float, shape=(ndim_2h1p))
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
    HR : ndarray(dtype=float, shape=(ndim_2h1p))
        Vector containing the matrix-vector product H(CCSD)*R
    """
    r1a, r1b, r2a, r2b, r2c, r2d = unflatten_R(R,sys)

    if flag_RHF:
        X1A = build_HR_1A(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X2A = build_HR_2A(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X2B = build_HR_2B(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        Xout = flatten_R(X1A, X1A, X2A, X2B, X2B, X2A)
    else:
        X1A = build_HR_1A(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X1B = build_HR_1B(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X2A = build_HR_2A(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X2B = build_HR_2B(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X2C = build_HR_2C(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X2D = build_HR_2D(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        Xout = flatten_R(X1A, X1B, X2A, X2B, X2C, X2D)

    return Xout

def build_HR_1A(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <i|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(noa))
        Linear EOMCC excitation amplitudes R1h(a)
    r1b : ndarray(dtype=float, shape=(nob))
        Linear EOMCC excitation amplitudes R1h(b)
    r2a : ndarray(dtype=float, shape=(nua,noa,noa))
        Linear EOMCC excitation amplitudes R2h1p(aaa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa))
        Linear EOMCC excitation amplitudes R2h1p(aba)
    r2c : ndarray(dtype=float, shape=(nub,noa,nob))
        Linear EOMCC excitation amplitudes R2h1p(bab)
    r2d : ndarray(dtype=float, shape=(nub,nob,nob))
        Linear EOMCC excitation amplitudes R2h1p(bbb)
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
    X1A : ndarray(dtype=float, shape=(noa))
        Calculated HR Projection
    """
    X1A = 0.0
    X1A -= np.einsum('mi,m->i',H1A['oo'],r1a,optimize=True)
    X1A -= 0.5*np.einsum('mnif,fmn->i',H2A['ooov'],r2a,optimize=True)
    X1A -= np.einsum('mnif,fmn->i',H2B['ooov'],r2c,optimize=True)
    X1A += np.einsum('me,eim->i',H1A['ov'],r2a,optimize=True)
    X1A += np.einsum('me,eim->i',H1B['ov'],r2c,optimize=True)

    return X1A

def build_HR_1B(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <i~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(noa))
        Linear EOMCC excitation amplitudes R1h(a)
    r1b : ndarray(dtype=float, shape=(nob))
        Linear EOMCC excitation amplitudes R1h(b)
    r2a : ndarray(dtype=float, shape=(nua,noa,noa))
        Linear EOMCC excitation amplitudes R2h1p(aaa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa))
        Linear EOMCC excitation amplitudes R2h1p(aba)
    r2c : ndarray(dtype=float, shape=(nub,noa,nob))
        Linear EOMCC excitation amplitudes R2h1p(bab)
    r2d : ndarray(dtype=float, shape=(nub,nob,nob))
        Linear EOMCC excitation amplitudes R2h1p(bbb)
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
    X1B : ndarray(dtype=float, shape=(nob))
        Calculated HR Projection
    """
    X1B = 0.0
    X1B -= np.einsum('mi,m->i',H1B['oo'],r1b,optimize=True)
    X1B -= np.einsum('nmfi,fmn->i',H2B['oovo'],r2b,optimize=True)
    X1B -= 0.5*np.einsum('mnif,fmn->i',H2C['ooov'],r2d,optimize=True)
    X1B += np.einsum('me,eim->i',H1A['ov'],r2b,optimize=True)
    X1B += np.einsum('me,eim->i',H1B['ov'],r2d,optimize=True)

    return X1B

def build_HR_2A(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <ijb|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(noa))
        Linear EOMCC excitation amplitudes R1h(a)
    r1b : ndarray(dtype=float, shape=(nob))
        Linear EOMCC excitation amplitudes R1h(b)
    r2a : ndarray(dtype=float, shape=(nua,noa,noa))
        Linear EOMCC excitation amplitudes R2h1p(aaa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa))
        Linear EOMCC excitation amplitudes R2h1p(aba)
    r2c : ndarray(dtype=float, shape=(nub,noa,nob))
        Linear EOMCC excitation amplitudes R2h1p(bab)
    r2d : ndarray(dtype=float, shape=(nub,nob,nob))
        Linear EOMCC excitation amplitudes R2h1p(bbb)
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
    X2A : ndarray(dtype=float, shape=(nua,noa,noa))
        Calculated HR Projection
    """
    vA = ints['vA']
    vB = ints['vB']
    t2a = cc_t['t2a']

    X2A = 0.0
    X2A -= np.einsum('bmji,m->bij',H2A['vooo'],r1a,optimize=True)
    X2A += np.einsum('be,eij->bij',H1A['vv'],r2a,optimize=True)
    X2A += 0.5*np.einsum('mnij,bmn->bij',H2A['oooo'],r2a,optimize=True)
    I1 = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],r2a,optimize=True)\
        -np.einsum('mnef,fmn->e',vB['oovv'],r2c,optimize=True)
    X2A += np.einsum('e,ebij->bij',I1,t2a,optimize=True)

    D_ij = 0.0
    D_ij -= np.einsum('mi,bmj->bij',H1A['oo'],r2a,optimize=True)
    D_ij += np.einsum('bmje,eim->bij',H2A['voov'],r2a,optimize=True)
    D_ij += np.einsum('bmje,eim->bij',H2B['voov'],r2c,optimize=True)
    D_ij -= np.transpose(D_ij,(0,2,1))

    X2A += D_ij

    return X2A

def build_HR_2B(r1a,r1b,r2a,r2b,r2c,r2d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <i~jb|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(noa))
        Linear EOMCC excitation amplitudes R1h(a)
    r1b : ndarray(dtype=float, shape=(nob))
        Linear EOMCC excitation amplitudes R1h(b)
    r2a : ndarray(dtype=float, shape=(nua,noa,noa))
        Linear EOMCC excitation amplitudes R2h1p(aaa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa))
        Linear EOMCC excitation amplitudes R2h1p(aba)
    r2c : ndarray(dtype=float, shape=(nub,noa,nob))
        Linear EOMCC excitation amplitudes R2h1p(bab)
    r2d : ndarray(dtype=float, shape=(nub,nob,nob))
        Linear EOMCC excitation amplitudes R2h1p(bbb)
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
    X2B : ndarray(dtype=float, shape=(nua,nob,noa))
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
    """Calculate the projection <ij~b~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(noa))
        Linear EOMCC excitation amplitudes R1h(a)
    r1b : ndarray(dtype=float, shape=(nob))
        Linear EOMCC excitation amplitudes R1h(b)
    r2a : ndarray(dtype=float, shape=(nua,noa,noa))
        Linear EOMCC excitation amplitudes R2h1p(aaa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa))
        Linear EOMCC excitation amplitudes R2h1p(aba)
    r2c : ndarray(dtype=float, shape=(nub,noa,nob))
        Linear EOMCC excitation amplitudes R2h1p(bab)
    r2d : ndarray(dtype=float, shape=(nub,nob,nob))
        Linear EOMCC excitation amplitudes R2h1p(bbb)
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
    X2C : ndarray(dtype=float, shape=(nub,noa,nob))
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
    """Calculate the projection <i~j~b~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(noa))
        Linear EOMCC excitation amplitudes R1h(a)
    r1b : ndarray(dtype=float, shape=(nob))
        Linear EOMCC excitation amplitudes R1h(b)
    r2a : ndarray(dtype=float, shape=(nua,noa,noa))
        Linear EOMCC excitation amplitudes R2h1p(aaa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa))
        Linear EOMCC excitation amplitudes R2h1p(aba)
    r2c : ndarray(dtype=float, shape=(nub,noa,nob))
        Linear EOMCC excitation amplitudes R2h1p(bab)
    r2d : ndarray(dtype=float, shape=(nub,nob,nob))
        Linear EOMCC excitation amplitudes R2h1p(bbb)
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
    X2D : ndarray(dtype=float, shape=(nub,nob,nob))
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

def guess_1h(ints,sys):
    """Build and diagonalize the Hamiltonian in the space of 1h excitations.

    Parameters
    ----------
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    -------
    C : ndarray(dtype=float, shape=(ndim_1h,ndim_1h))
        Matrix of 1h eigenvectors
    E_1h : ndarray(dtype=float, shape=(ndim_cis))
        Vector of 1h eigenvalues
    """
    fA = ints['fA']
    fB = ints['fB']

    n1a = sys['Nocc_a']
    n1b = sys['Nocc_b']

    HAA = np.zeros((n1a,n1a))
    HAB = np.zeros((n1a,n1b))
    HBA = np.zeros((n1b,n1a))
    HBB = np.zeros((n1b,n1b))

    ct1 = 0
    for i in range(sys['Nocc_a']):
        ct2 = 0
        for j in range(sys['Nocc_a']):
            HAA[ct1,ct2] = fA['oo'][i,j]
            ct2 += 1
        ct1+=1

    ct1 = 0
    for i in range(sys['Nocc_b']):
        ct2 = 0
        for j in range(sys['Nocc_b']):
            HBB[ct1,ct2] = fB['oo'][i,j]
            ct2 += 1
        ct1 += 1

    H = np.hstack( (np.vstack((HAA,HBA)), np.vstack((HAB,HBB))) )

    E_1h, C = np.linalg.eigh(H) 
    idx = np.argsort(E_1h)
    E_1h = E_1h[idx]
    C = C[:,idx]

    return C, E_1h
