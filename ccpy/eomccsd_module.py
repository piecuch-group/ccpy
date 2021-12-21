"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for excited states using
the equation-of-motion (EOM) CC with singles and doubles (EOMCCSD)."""
import numpy as np
from cc_energy import calc_cc_energy
import cc_loops
from solvers import davidson, davidson_out_of_core
from eomcc_initialize import get_eomcc_initial_guess
from functools import partial
#print(eomcc_initial_guess.eomcc_initial_guess.eomccs_d_matrix.__doc__)

def eomccsd(nroot,H1A,H1B,H2A,H2B,H2C,cc_t,ints,sys,noact=0,nuact=0,tol=1.0e-06,maxit=80,flag_RHF=False):
    """Perform the EOMCCSD excited-state calculation.

    Parameters
    ----------
    nroot : int
        Number of excited-states to solve for in the EOMCCSD procedure
    H1*, H2* : dict
        Sliced CCSD similarity-transformed HBar integrals
    cc_t : dict
        Cluster amplitudes T1, T2 of the ground-state
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    noact : int, optional
        Number of active occupied orbitals used in EOMCCSd initial guess. 
        Default is 0, corresponding to CIS.
    nuact : int, optional
        Number of active unoccupied orbitals used in EOMCCSd initial guess. 
        Default is 0, corresponding to CIS.
    tol : float, optional
        Convergence tolerance for the EOMCC calculation. Default is 1.0e-06.
    maxit : int, optional
        Maximum number of Davidson iterations in the EOMCC procedure.

    Returns
    -------
    cc_t : dict
        Updated dictionary of cluster amplitudes with r0, R1, R2 amplitudes for each excited state.
    omega : ndarray(dtype=float, shape=(nroot))
        Vector of vertical excitation energies (in hartree) for each root
    """
    print('\n==================================++Entering EOM-CCSD Routine++=================================\n')

    n1a = sys['Nocc_a'] * sys['Nunocc_a']
    n1b = sys['Nocc_b'] * sys['Nunocc_b']
    n2a = sys['Nocc_a']**2*sys['Nunocc_a']**2
    n2b = sys['Nocc_a']*sys['Nocc_b']*sys['Nunocc_a']*sys['Nunocc_b']
    n2c = sys['Nocc_b']**2*sys['Nunocc_b']**2
    ndim = n1a+n1b+n2a+n2b+n2c

    # Obtain initial guess using the EOMCCSd method
    B0, E0 = get_eomcc_initial_guess(nroot,noact,nuact,ndim,H1A,H1B,H2A,H2B,H2C,ints,sys)
    # Get the HR function
    HR_func = partial(HR,cc_t=cc_t,H1A=H1A,H1B=H1B,H2A=H2A,H2B=H2B,H2C=H2C,ints=ints,sys=sys,flag_RHF=flag_RHF)
    # Get the R update function
    update_R_func = lambda r,omega : update_R(r,omega,H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],sys)
    # Diagonalize Hamiltonian using Davidson algorithm
    #Rvec, omega, is_converged = davidson(HR_func,update_R_func,B0,E0,maxit,80,tol)
    Rvec, omega, is_converged = davidson_out_of_core(HR_func,update_R_func,B0,E0,maxit,tol)

    cc_t['r1a'] = [None]*len(omega)
    cc_t['r1b'] = [None]*len(omega)
    cc_t['r2a'] = [None]*len(omega)
    cc_t['r2b'] = [None]*len(omega)
    cc_t['r2c'] = [None]*len(omega)
    cc_t['r0'] = [None]*len(omega)

    print('Summary of EOMCCSD:')
    Eccsd = ints['Escf'] + calc_cc_energy(cc_t,ints)
    for i in range(len(omega)):
        r1a,r1b,r2a,r2b,r2c = unflatten_R(Rvec[:,i],sys)
        r0 = calc_r0(r1a,r1b,r2a,r2b,r2c,H1A,H1B,ints,omega[i])  
        cc_t['r1a'][i] = r1a
        cc_t['r1b'][i] = r1b 
        cc_t['r2a'][i] = r2a
        cc_t['r2b'][i] = r2b
        cc_t['r2c'][i] = r2c
        cc_t['r0'][i] = r0
        if is_converged[i]:
            tmp = 'CONVERGED'
        else:
            tmp = 'NOT CONVERGED'
        print('   Root - {}    E = {}    omega = {:.10f}    r0 = {:.10f}    [{}]'\
                        .format(i+1,omega[i]+Eccsd,omega[i],r0,tmp))

    return cc_t, omega

def update_R(r,omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,sys):

    r1a,r1b,r2a,r2b,r2c = unflatten_R(r,sys)
    r1a,r1b,r2a,r2b,r2c, = cc_loops.cc_loops.update_r(r1a,r1b,r2a,r2b,r2c,omega,\
                    H1A_oo,H1A_vv,H1B_oo,H1B_vv,0.0,\
                    sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'])
    return flatten_R(r1a,r1b,r2a,r2b,r2c)

def calc_r0(r1a,r1b,r2a,r2b,r2c,H1A,H1B,ints,omega):
    """Calculate the EOMCC overlap <0|[ (H_N e^T)_C * (R1+R2) ]_C|0>.
    
    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua,noa))
        Linear EOMCC excitation amplitudes R1(aa)
    r1b : ndarray(dtype=float, shape=(nub,nob))
        Linear EOMCC excitation amplitudes R1(bb)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa,noa))
        Linear EOMCC excitation amplitudes R2(aa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa,nob))
        Linear EOMCC excitation amplitudes R2(ab)
    r2c : ndarray(dtype=float, shape=(nub,nub,nob,nob))
        Linear EOMCC excitation amplitudes R2(bb)
    H1*, H2* : dict
        Sliced CCSD similarity-transformed HBar integrals
    ints : dict
        Sliced integrals F_N and V_N that define the bare Hamiltonian H_N
    omega : float
        Vertical excitation energy (in hartree) for the given root

    Returns
    -------
    r0 : float
        Overlap of excited state with ground state
    """
    r0 = 0.0
    r0 += np.einsum('me,em->',H1A['ov'],r1a,optimize=True)
    r0 += np.einsum('me,em->',H1B['ov'],r1b,optimize=True)
    r0 += 0.25*np.einsum('mnef,efmn->',ints['vA']['oovv'],r2a,optimize=True)
    r0 += np.einsum('mnef,efmn->',ints['vB']['oovv'],r2b,optimize=True)
    r0 += 0.25*np.einsum('mnef,efmn->',ints['vC']['oovv'],r2c,optimize=True)

    return r0/omega

def flatten_R(r1a,r1b,r2a,r2b,r2c,order='C'):
    """Flatten the R vector.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua,noa))
        Linear EOMCC excitation amplitudes R1(aa)
    r1b : ndarray(dtype=float, shape=(nub,nob))
        Linear EOMCC excitation amplitudes R1(bb)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa,noa))
        Linear EOMCC excitation amplitudes R2(aa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa,nob))
        Linear EOMCC excitation amplitudes R2(ab)
    r2c : ndarray(dtype=float, shape=(nub,nub,nob,nob))
        Linear EOMCC excitation amplitudes R2(bb)

    Returns
    -------
    R : ndarray(dtype=float, shape=(ndim_ccsd))
        Flattened array of R vector for the given root
    """
    return np.concatenate((r1a.flatten(order=order),r1b.flatten(order=order),r2a.flatten(order=order),r2b.flatten(order=order),r2c.flatten(order=order)),axis=0)

def unflatten_R(R,sys,order='C'):
    """Unflatten the R vector into many-body tensor components.

    Parameters
    ----------
    R : ndarray(dtype=float, shape=(ndim_ccsd))
        Flattened array of R vector for the given root
    sys : dict
        System information dictionary
    order : str, optional
        String of value 'C' or 'F' indicating whether row-major or column-major
        flattening should be used. Default is 'C'.

    Returns
    -------
    r1a : ndarray(dtype=float, shape=(nua,noa))
        Linear EOMCC excitation amplitudes R1(aa)
    r1b : ndarray(dtype=float, shape=(nub,nob))
        Linear EOMCC excitation amplitudes R1(bb)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa,noa))
        Linear EOMCC excitation amplitudes R2(aa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa,nob))
        Linear EOMCC excitation amplitudes R2(ab)
    r2c : ndarray(dtype=float, shape=(nub,nub,nob,nob))
        Linear EOMCC excitation amplitudes R2(bb)
    """
    n1a = sys['Nocc_a'] * sys['Nunocc_a']
    n1b = sys['Nocc_b'] * sys['Nunocc_b']
    n2a = sys['Nocc_a'] ** 2 * sys['Nunocc_a'] ** 2
    n2b = sys['Nocc_a'] * sys['Nocc_b'] * sys['Nunocc_a'] * sys['Nunocc_b']
    n2c = sys['Nocc_b'] ** 2 * sys['Nunocc_b'] ** 2
    idx_1a = slice(0,n1a)
    idx_1b = slice(n1a,n1a+n1b)
    idx_2a = slice(n1a+n1b,n1a+n1b+n2a)
    idx_2b = slice(n1a+n1b+n2a,n1a+n1b+n2a+n2b)
    idx_2c = slice(n1a+n1b+n2a+n2b,n1a+n1b+n2a+n2b+n2c)

    r1a  = np.reshape(R[idx_1a],(sys['Nunocc_a'],sys['Nocc_a']),order=order)
    r1b  = np.reshape(R[idx_1b],(sys['Nunocc_b'],sys['Nocc_b']),order=order)
    r2a  = np.reshape(R[idx_2a],(sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a']),order=order)
    r2b  = np.reshape(R[idx_2b],(sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b']),order=order)
    r2c  = np.reshape(R[idx_2c],(sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b']),order=order)

    return r1a, r1b, r2a, r2b, r2c


def HR(R,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF):
    """Calculate the matrix-vector product H(CCSD)*R.

    Parameters
    ----------
    R : ndarray(dtype=float, shape=(ndim_ccsd))
        Flattened vector of R amplitudes
    cc_t : dict
        Cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced CCSD similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    flag_RHF : bool
        True/False whether to use closed-shell RHF spin symmetry

    Returns
    -------
    HR : ndarray(dtype=float, shape=(ndim_ccsd))
        Vector containing the matrix-vector product H(CCSD)*R
    """
    r1a, r1b, r2a, r2b, r2c = unflatten_R(R,sys)

    if flag_RHF:
        X1A = build_HR_1A(r1a,r1b,r2a,r2b,r2c,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X2A = build_HR_2A(r1a,r1b,r2a,r2b,r2c,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X2B = build_HR_2B(r1a,r1b,r2a,r2b,r2c,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        Xout = flatten_R(X1A,X1A,X2A,X2B,X2A)
    else:
        X1A = build_HR_1A(r1a,r1b,r2a,r2b,r2c,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X1B = build_HR_1B(r1a,r1b,r2a,r2b,r2c,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X2A = build_HR_2A(r1a,r1b,r2a,r2b,r2c,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X2B = build_HR_2B(r1a,r1b,r2a,r2b,r2c,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X2C = build_HR_2C(r1a,r1b,r2a,r2b,r2c,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        Xout = flatten_R(X1A, X1B, X2A, X2B, X2C)
    return Xout

def build_HR_1A(r1a,r1b,r2a,r2b,r2c,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <ia|[ (H_N e^(T1+T2))_C*(R1+R2) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua,noa))
        Linear EOMCC excitation amplitudes R1(aa)
    r1b : ndarray(dtype=float, shape=(nub,nob))
        Linear EOMCC excitation amplitudes R1(bb)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa,noa))
        Linear EOMCC excitation amplitudes R2(aa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa,nob))
        Linear EOMCC excitation amplitudes R2(ab)
    r2c : ndarray(dtype=float, shape=(nub,nub,nob,nob))
        Linear EOMCC excitation amplitudes R2(bb)
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
    X1A : ndarray(dtype=float, shape=(nua,noa))
        Calculated HR Projection
    """
    X1A = 0.0
    X1A -= np.einsum('mi,am->ai',H1A['oo'],r1a,optimize=True)
    X1A += np.einsum('ae,ei->ai',H1A['vv'],r1a,optimize=True)
    X1A += np.einsum('amie,em->ai',H2A['voov'],r1a,optimize=True)
    X1A += np.einsum('amie,em->ai',H2B['voov'],r1b,optimize=True)
    X1A  -= 0.5*np.einsum('mnif,afmn->ai',H2A['ooov'],r2a,optimize=True)
    X1A -= np.einsum('mnif,afmn->ai',H2B['ooov'],r2b,optimize=True)
    X1A += 0.5*np.einsum('anef,efin->ai',H2A['vovv'],r2a,optimize=True)
    X1A += np.einsum('anef,efin->ai',H2B['vovv'],r2b,optimize=True)
    X1A += np.einsum('me,aeim->ai',H1A['ov'],r2a,optimize=True)
    X1A += np.einsum('me,aeim->ai',H1B['ov'],r2b,optimize=True)

    return X1A

def build_HR_1B(r1a,r1b,r2a,r2b,r2c,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <i~a~|[ (H_N e^(T1+T2))_C*(R1+R2) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua,noa))
        Linear EOMCC excitation amplitudes R1(aa)
    r1b : ndarray(dtype=float, shape=(nub,nob))
        Linear EOMCC excitation amplitudes R1(bb)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa,noa))
        Linear EOMCC excitation amplitudes R2(aa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa,nob))
        Linear EOMCC excitation amplitudes R2(ab)
    r2c : ndarray(dtype=float, shape=(nub,nub,nob,nob))
        Linear EOMCC excitation amplitudes R2(bb)
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
    X1B : ndarray(dtype=float, shape=(nub,nob))
        Calculated HR Projection
    """
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

    return X1B

def build_HR_2A(r1a,r1b,r2a,r2b,r2c,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <ijab|[ (H_N e^(T1+T2))_C*(R1+R2) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua,noa))
        Linear EOMCC excitation amplitudes R1(aa)
    r1b : ndarray(dtype=float, shape=(nub,nob))
        Linear EOMCC excitation amplitudes R1(bb)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa,noa))
        Linear EOMCC excitation amplitudes R2(aa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa,nob))
        Linear EOMCC excitation amplitudes R2(ab)
    r2c : ndarray(dtype=float, shape=(nub,nub,nob,nob))
        Linear EOMCC excitation amplitudes R2(bb)
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
    X2A : ndarray(dtype=float, shape=(nua,nua,noa,noa))
        Calculated HR Projection
    """
    t2a = cc_t['t2a']
    vA = ints['vA']
    vB = ints['vB']

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

    D_ij = D1 + D6 + D9 + D10 + D12 + D14
    D_ab = D2 + D5 + D7 + D8  + D11 + D13
    D_abij = D3 + D4

    D_ij -= np.einsum('abij->abji',D_ij,optimize=True)
    D_ab -= np.einsum('abij->baij',D_ab,optimize=True)
    D_abij += -np.einsum('abij->baij',D_abij,optimize=True)\
    -np.einsum('abij->abji',D_abij,optimize=True)\
    +np.einsum('abij->baji',D_abij,optimize=True)    

    X2A += D_ij + D_ab + D_abij

    return X2A

def build_HR_2B(r1a,r1b,r2a,r2b,r2c,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <ij~ab~|[ (H_N e^(T1+T2))_C*(R1+R2) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua,noa))
        Linear EOMCC excitation amplitudes R1(aa)
    r1b : ndarray(dtype=float, shape=(nub,nob))
        Linear EOMCC excitation amplitudes R1(bb)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa,noa))
        Linear EOMCC excitation amplitudes R2(aa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa,nob))
        Linear EOMCC excitation amplitudes R2(ab)
    r2c : ndarray(dtype=float, shape=(nub,nub,nob,nob))
        Linear EOMCC excitation amplitudes R2(bb)
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
    X2B : ndarray(dtype=float, shape=(nua,nub,noa,nob))
        Calculated HR Projection
    """
    t2b = cc_t['t2b']
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
    return X2B

def build_HR_2C(r1a,r1b,r2a,r2b,r2c,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <i~j~a~b~|[ (H_N e^(T1+T2))_C*(R1+R2) ]_C|0>.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua,noa))
        Linear EOMCC excitation amplitudes R1(aa)
    r1b : ndarray(dtype=float, shape=(nub,nob))
        Linear EOMCC excitation amplitudes R1(bb)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa,noa))
        Linear EOMCC excitation amplitudes R2(aa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa,nob))
        Linear EOMCC excitation amplitudes R2(ab)
    r2c : ndarray(dtype=float, shape=(nub,nub,nob,nob))
        Linear EOMCC excitation amplitudes R2(bb)
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
    X2C : ndarray(dtype=float, shape=(nub,nub,nob,nob))
        Calculated HR Projection
    """
    t2c = cc_t['t2c']
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
    #D7 = 0.0
    #D8 = 0.0

    Q1 = 0.5*np.einsum('mnef,efjn->mj',vC['oovv'],r2c,optimize=True)
    D9 = -np.einsum('mj,abim->abij',Q1,t2c,optimize=True) # A(ij)
    Q2 = np.einsum('nmfe,fenj->mj',vB['oovv'],r2b,optimize=True)
    D10 = -np.einsum('mj,abim->abij',Q2,t2c,optimize=True) # A(ij)
    #D9 = 0.0
    #D10 = 0.0

    Q1 = np.einsum('amfe,em->af',H2C['vovv'],r1b,optimize=True)
    D11 = np.einsum('af,fbij->abij',Q1,t2c,optimize=True) # A(ab)
    Q2 = np.einsum('nmie,em->ni',H2C['ooov'],r1b,optimize=True)
    D12 = -np.einsum('ni,abnj->abij',Q2,t2c,optimize=True) # A(ij)
    #D11 = 0.0
    #D12 = 0.0

    Q1 = np.einsum('maef,em->af',H2B['ovvv'],r1a,optimize=True)
    D13 = np.einsum('af,fbij->abij',Q1,t2c,optimize=True) # A(ab)
    Q2 = np.einsum('mnei,em->ni',H2B['oovo'],r1a,optimize=True)
    D14 = -np.einsum('ni,abnj->abij',Q2,t2c,optimize=True) # A(ij)
    #D13 = 0.0
    #D14 = 0.0

    D_ij = D1 + D6 + D9 + D10 + D12 + D14
    D_ab = D2 + D5 + D7 + D8  + D11 + D13
    D_abij = D3 + D4

    D_ij -= np.einsum('abij->abji',D_ij,optimize=True)
    D_ab -= np.einsum('abij->baij',D_ab,optimize=True)
    D_abij += -np.einsum('abij->baij',D_abij,optimize=True)\
    -np.einsum('abij->abji',D_abij,optimize=True)\
    +np.einsum('abij->baji',D_abij,optimize=True)    

    X2C += D_ij + D_ab + D_abij

    return X2C
