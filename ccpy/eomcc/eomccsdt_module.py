"""Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for excited states using
the equation-of-motion (EOM) CC with singles, doubles, and triples (EOMCCSDT)."""
import numpy as np
from cc_energy import calc_cc_energy
import cc_loops
from solvers import davidson_out_of_core, davidson
from eomcc_initialize import get_eomcc_initial_guess
from functools import partial

def eomccsdt(nroot,H1A,H1B,H2A,H2B,H2C,cc_t,ints,sys,noact=0,nuact=0,tol=1.0e-06,maxit=80,flag_RHF=False):
    """Perform the EOMCCSDT excited-state calculation.

    Parameters
    ----------
    nroot : int
        Number of excited-states to solve for in the EOMCCSDT procedure
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    cc_t : dict
        Cluster amplitudes T1, T2, and T3 of the ground-state
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
        Updated dictionary of cluster amplitudes with r0, R1, R2, and R3 amplitudes for each excited state.
    omega : ndarray(dtype=float, shape=(nroot))
        Vector of vertical excitation energies (in hartree) for each root
    """
    print('\n==================================++Entering EOM-CCSDT Routine++=================================\n')

    n1a = sys['Nocc_a'] * sys['Nunocc_a']
    n1b = sys['Nocc_b'] * sys['Nunocc_b']
    n2a = sys['Nocc_a']**2*sys['Nunocc_a']**2
    n2b = sys['Nocc_a']*sys['Nocc_b']*sys['Nunocc_a']*sys['Nunocc_b']
    n2c = sys['Nocc_b']**2*sys['Nunocc_b']**2
    n3a = sys['Nocc_a']**3 * sys['Nunocc_a']**3
    n3b = sys['Nocc_a']**2*sys['Nunocc_a']**2*sys['Nocc_b']*sys['Nunocc_b']
    n3c = sys['Nocc_a']*sys['Nunocc_a']*sys['Nocc_b']**2*sys['Nunocc_b']**2
    n3d = sys['Nocc_b']**3 * sys['Nunocc_b']**3

    ndim = n1a+n1b+n2a+n2b+n2c+n3a+n3b+n3c+n3d

    # Obtain initial guess using the EOMCCSd method
    B0, E0 = get_eomcc_initial_guess(nroot,noact,nuact,ndim,H1A,H1B,H2A,H2B,H2C,ints,sys)
    # Get the HR function
    HR_func = partial(HR,cc_t=cc_t,H1A=H1A,H1B=H1B,H2A=H2A,H2B=H2B,H2C=H2C,ints=ints,sys=sys,flag_RHF=flag_RHF)
    # Get the R update function
    update_R_func = lambda r,omega : update_R(r,omega,H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],sys)
    # Diagonalize Hamiltonian using Davidson algorithm
    Rvec, omega, is_converged = davidson(HR_func,update_R_func,B0,E0,maxit,maxit,tol)
    #Rvec, omega, is_converged = davidson_out_of_core(HR_func,update_R_func,B0,E0,maxit,tol)
    
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

def update_R(r,omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,sys):

    r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d = unflatten_R(r,sys)
    r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d = cc_loops.cc_loops.update_r_ccsdt(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,\
                    omega,H1A_oo,H1A_vv,H1B_oo,H1B_vv,0.0,\
                    sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'])

    return flatten_R(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d)

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

def flatten_R(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d):
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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)

    Returns
    -------
    R : ndarray(dtype=float, shape=(ndim_ccsdt))
        Flattened array of R vector for the given root
    """
    return np.concatenate((r1a.flatten(),r1b.flatten(),\
                           r2a.flatten(),r2b.flatten(),r2c.flatten(),\
                           r3a.flatten(),r3b.flatten(),r3c.flatten(),r3d.flatten()),axis=0)

def unflatten_R(R,sys,order='C'):
    """Unflatten the R vector into many-body tensor components.

    Parameters
    ----------
    R : ndarray(dtype=float, shape=(ndim_ccsdt))
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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    """
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


def HR(R,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF):
    """Calculate the matrix-vector product H(CCSDT)*R.

    Parameters
    ----------
    R : ndarray(dtype=float, shape=(ndim_ccsdt))
        Flattened vector of R amplitudes
    cc_t : dict
        Cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    -------
    HR : ndarray(dtype=float, shape=(ndim_ccsdt))
        Vector containing the matrix-vector product H(CCSDT)*R
    """
    r1a, r1b, r2a, r2b, r2c, r3a, r3b, r3c, r3d = unflatten_R(R,sys)

    X1A = build_HR_1A(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X2A = build_HR_2A(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X2B = build_HR_2B(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X3A = build_HR_3A(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    X3B = build_HR_3B(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    # closed shell symmetry
    if flag_RHF:
        return flatten_R(X1A, X1A, X2A, X2B, X2A, X3A, X3B,\
                        np.transpose(X3B,(2,1,0,5,4,3)), X3A)
    else:
        X1B = build_HR_1B(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X2C = build_HR_2C(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X3C = build_HR_3C(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        X3D = build_HR_3D(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
        return flatten_R(X1A, X1B, X2A, X2B, X2C, X3A, X3B, X3C, X3D)

def build_HR_1A(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <ia|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.

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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
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
    """Calculate the projection <i~a~|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.

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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
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

    X1B += 0.25*np.einsum('mnef,efamni->ai',ints['vA']['oovv'],r3b,optimize=True)
    X1B += np.einsum('mnef,efamni->ai',ints['vB']['oovv'],r3c,optimize=True)
    X1B += 0.25*np.einsum('mnef,aefimn->ai',ints['vC']['oovv'],r3d,optimize=True)

    return X1B

def build_HR_2A(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <ijab|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.

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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
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
    """Calculate the projection <ij~ab~|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.

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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
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
    """Calculate the projection <i~j~a~b~|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.

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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
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
    """Calculate the projection <ijkabc|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.

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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    --------
    X3A : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Calculated HR Projection
    """
    t2a = cc_t['t2a']
    t3a = cc_t['t3a']
    t3b = cc_t['t3b']
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']

    X3A = 0.0
    # <ijkabc| [H(R1+R2)]_C | 0 >
    Q1 = np.einsum('mnef,fn->me',vA['oovv'],r1a,optimize=True)
    Q1 += np.einsum('mnef,fn->me',vB['oovv'],r1b,optimize=True)
    I1 = np.einsum('amje,bm->abej',H2A['voov'],r1a,optimize=True)
    I1 += np.einsum('amfe,bejm->abfj',H2A['vovv'],r2a,optimize=True)
    I1 += np.einsum('amfe,bejm->abfj',H2B['vovv'],r2b,optimize=True)
    I1 -= np.transpose(I1,(1,0,2,3))
    I2 = np.einsum('abfe,ej->abfj',H2A['vvvv'],r1a,optimize=True)
    I2 += 0.5*np.einsum('nmje,abmn->abej',H2A['ooov'],r2a,optimize=True)
    I2 -= np.einsum('me,abmj->abej',Q1,cc_t['t2a'],optimize=True)
    I3 = -0.5*np.einsum('mnef,abfimn->baei',vA['oovv'],r3a,optimize=True)\
        -np.einsum('mnef,abfimn->baei',vB['oovv'],r3b,optimize=True)
    X3A = 0.25*np.einsum('abej,ecik->abcijk',I1+I2+I3,t2a,optimize=True)
    X3A += 0.25*np.einsum('baje,ecik->abcijk',H2A['vvov'],r2a,optimize=True)

    I1 = -np.einsum('bmie,ej->mbij',H2A['voov'],r1a,optimize=True)
    I1 += np.einsum('nmie,bejm->nbij',H2A['ooov'],r2a,optimize=True)
    I1 += np.einsum('nmie,bejm->nbij',H2B['ooov'],r2b,optimize=True)
    I1 -= np.transpose(I1,(0,1,3,2))
    I2 = -1.0*np.einsum('nmij,bm->nbij',H2A['oooo'],r1a,optimize=True)
    I2 += 0.5*np.einsum('bmfe,efij->mbij',H2A['vovv'],r2a,optimize=True)
    I3 = 0.5*np.einsum('mnef,efcjnk->mcjk',vA['oovv'],r3a,optimize=True)\
        +np.einsum('mnef,ecfjkn->mcjk',vB['oovv'],r3b,optimize=True)
    X3A -= 0.25*np.einsum('mbij,acmk->abcijk',I1+I2+I3,t2a,optimize=True)
    X3A -= 0.25*np.einsum('bmji,acmk->abcijk',H2A['vooo'],r2a,optimize=True)

    # additional terms with T3 in <ijkabc|[ H(R1+R2)]_C | 0>
    I1 = -1.0*np.einsum('me,bm->be',H1A['ov'],r1a,optimize=True)\
         +np.einsum('bnef,fn->be',H2A['vovv'],r1a,optimize=True)\
         +np.einsum('bnef,fn->be',H2B['vovv'],r1b,optimize=True)
    I2 = -0.5*np.einsum('mnef,bfmn->be',vA['oovv'],r2a,optimize=True)\
         -np.einsum('mnef,bfmn->be',vB['oovv'],r2b,optimize=True)
    X3A += (1.0/12.0)*np.einsum('be,aecijk->abcijk',I1+I2,t3a,optimize=True) # A(b/ac)

    I1 = np.einsum('me,ej->mj',H1A['ov'],r1a,optimize=True)\
        +np.einsum('mnjf,fn->mj',H2A['ooov'],r1a,optimize=True)\
        +np.einsum('mnjf,fn->mj',H2B['ooov'],r1b,optimize=True)
    I2 = 0.5*np.einsum('mnef,efjn->mj',vA['oovv'],r2a,optimize=True)\
        +np.einsum('mnef,efjn->mj',vB['oovv'],r2b,optimize=True)
    X3A -= (1.0/12.0)*np.einsum('mj,abcimk->abcijk',I1+I2,t3a,optimize=True) # A(j/ik)

    I1 = np.einsum('nmje,ei->mnij',H2A['ooov'],r1a,optimize=True)
    I1 -= np.transpose(I1,(0,1,3,2))
    I2 = 0.5*np.einsum('mnef,efij->mnij',vA['oovv'],r2a,optimize=True)
    X3A += (1.0/24.0)*np.einsum('mnij,abcmnk->abcijk',I1+I2,t3a,optimize=True) # A(k/ij)

    I1 = -1.0*np.einsum('amef,bm->abef',H2A['vovv'],r1a,optimize=True)
    I1 -= np.transpose(I1,(1,0,2,3))
    I2 = 0.5*np.einsum('mnef,abmn->abef',vA['oovv'],r2a,optimize=True)
    X3A += (1.0/24.0)*np.einsum('abef,efcijk->abcijk',I1+I2,t3a,optimize=True) # A(c/ab)

    I1 = -1.0*np.einsum('nmje,bn->bmje',H2A['ooov'],r1a,optimize=True)\
         +np.einsum('bmfe,fj->bmje',H2A['vovv'],r1a,optimize=True)
    I2 = np.einsum('mnef,fcnk->cmke',vA['oovv'],r2a,optimize=True)\
        +np.einsum('mnef,cfkn->cmke',vB['oovv'],r2b,optimize=True)
    X3A += 0.25*np.einsum('bmje,aecimk->abcijk',I1+I2,t3a,optimize=True) # A(j/ik)A(b/ac)

    I1 = -1.0*np.einsum('nmje,bn->bmje',H2B['ooov'],r1a,optimize=True)\
         +np.einsum('bmfe,fj->bmje',H2B['vovv'],r1a,optimize=True)
    I2 = np.einsum('nmfe,fcnk->cmke',vB['oovv'],r2a,optimize=True)\
        +np.einsum('mnef,cfkn->cmke',vC['oovv'],r2b,optimize=True)
    X3A += 0.25*np.einsum('bmje,aceikm->abcijk',I1+I2,t3b,optimize=True) # A(j/ik)A(b/ac)

    # < ijkabc | (HR3)_C | 0 >
    X3A -= (1.0/12.0)*np.einsum('mj,abcimk->abcijk',H1A['oo'],r3a,optimize=True)
    X3A += (1.0/12.0)*np.einsum('be,aecijk->abcijk',H1A['vv'],r3a,optimize=True)
    X3A += (1.0/24.0)*np.einsum('mnij,abcmnk->abcijk',H2A['oooo'],r3a,optimize=True)
    X3A += (1.0/24.0)*np.einsum('abef,efcijk->abcijk',H2A['vvvv'],r3a,optimize=True)
    X3A += 0.25*np.einsum('amie,ebcmjk->abcijk',H2A['voov'],r3a,optimize=True)
    X3A += 0.25*np.einsum('amie,bcejkm->abcijk',H2B['voov'],r3b,optimize=True)

    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3A -= np.transpose(X3A,(0,1,2,3,5,4))
    X3A -= np.transpose(X3A,(0,1,2,4,3,5)) + np.transpose(X3A,(0,1,2,5,4,3))
    X3A -= np.transpose(X3A,(0,2,1,3,4,5))
    X3A -= np.transpose(X3A,(1,0,2,3,4,5)) + np.transpose(X3A,(2,1,0,3,4,5))

    return X3A

def build_HR_3B(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <ijk~abc~|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.

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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    --------
    X3B : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Calculated HR Projection
    """
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t3a = cc_t['t3a']
    t3b = cc_t['t3b']
    t3c = cc_t['t3c']
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']

    # < ijk~abc~ | [ H(R1+R2) ]_C | 0 >
    Q1 = np.einsum('mnef,fn->me',vA['oovv'],r1a,optimize=True)\
                        +np.einsum('mnef,fn->me',vB['oovv'],r1b,optimize=True)
    Q2 = np.einsum('nmfe,fn->me',vB['oovv'],r1a,optimize=True)\
                        +np.einsum('nmfe,fn->me',vC['oovv'],r1b,optimize=True)
    # Intermediate 1: X2B(bcek)*Y2A(aeij) -> Z3B(abcijk)
    Int1 = -1.0*np.einsum('mcek,bm->bcek',H2B['ovvo'],r1a,optimize=True)
    Int1 -= np.einsum('bmek,cm->bcek',H2B['vovo'],r1b,optimize=True)
    Int1 += np.einsum('bcfe,ek->bcfk',H2B['vvvv'],r1b,optimize=True)
    Int1 += np.einsum('mnek,bcmn->bcek',H2B['oovo'],r2b,optimize=True)
    Int1 += np.einsum('bmfe,ecmk->bcfk',H2A['vovv'],r2b,optimize=True)
    Int1 += np.einsum('bmfe,ecmk->bcfk',H2B['vovv'],r2c,optimize=True)
    Int1 -= np.einsum('mcfe,bemk->bcfk',H2B['ovvv'],r2b,optimize=True)
    I1 = -0.5*np.einsum('mnef,bfcmnk->bcek',vA['oovv'],r3b,optimize=True)\
         -np.einsum('mnef,bfcmnk->bcek',vB['oovv'],r3c,optimize=True)
    X3B = 0.5*np.einsum('bcek,aeij->abcijk',Int1+I1,t2a,optimize=True)
    X3B += 0.5*np.einsum('bcek,aeij->abcijk',H2B['vvvo'],r2a,optimize=True)
    # Intermediate 2: X2B(ncjk)*Y2A(abin) -> Z3B(abcijk)
    Int2 = -1.0*np.einsum('nmjk,cm->ncjk',H2B['oooo'],r1b,optimize=True)
    Int2 += np.einsum('mcje,ek->mcjk',H2B['ovov'],r1b,optimize=True)
    Int2 += np.einsum('mcek,ej->mcjk',H2B['ovvo'],r1a,optimize=True)
    Int2 += np.einsum('mcef,efjk->mcjk',H2B['ovvv'],r2b,optimize=True)
    Int2 += np.einsum('nmje,ecmk->ncjk',H2A['ooov'],r2b,optimize=True)
    Int2 += np.einsum('nmje,ecmk->ncjk',H2B['ooov'],r2c,optimize=True)
    Int2 -= np.einsum('nmek,ecjm->ncjk',H2B['oovo'],r2b,optimize=True)
    I1 = 0.5*np.einsum('mnef,efcjnk->mcjk',vA['oovv'],r3b,optimize=True)\
        +np.einsum('mnef,efcjnk->mcjk',vB['oovv'],r3c,optimize=True)
    X3B -= 0.5*np.einsum('ncjk,abin->abcijk',Int2+I1,t2a,optimize=True)
    X3B -= 0.5*np.einsum('mcjk,abim->abcijk',H2B['ovoo'],r2a,optimize=True)
    # Intermediate 3: X2A(abej)*Y2B(ecik) -> Z3B(abcijk)
    Int3 = np.einsum('amje,bm->abej',H2A['voov'],r1a,optimize=True) #(*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int3 += 0.5*np.einsum('abfe,ej->abfj',H2A['vvvv'],r1a,optimize=True) #(*) added factor 1/2 to compensate A(ab)
    Int3 += 0.25*np.einsum('nmje,abmn->abej',H2A['ooov'],r2a,optimize=True) #(*) added factor 1/2 to compensate A(ab)
    Int3 += np.einsum('amfe,bejm->abfj',H2A['vovv'],r2a,optimize=True)
    Int3 += np.einsum('amfe,bejm->abfj',H2B['vovv'],r2b,optimize=True)
    Int3 -= 0.5*np.einsum('me,abmj->abej',Q1,cc_t['t2a'],optimize=True) #(*) added factor 1/2 to compensate A(ab)
    Int3 -= np.transpose(Int3,(1,0,2,3))
    I1 = -0.5*np.einsum('mnef,abfmjn->abej',vA['oovv'],r3a,optimize=True)\
         -np.einsum('mnef,abfmjn->abej',vB['oovv'],r3b,optimize=True)
    X3B += 0.5*np.einsum('abej,ecik->abcijk',Int3+I1,t2b,optimize=True)
    X3B += 0.5*np.einsum('baje,ecik->abcijk',H2A['vvov'],r2b,optimize=True)
    # Intermediate 4: X2A(bnji)*Y2B(acnk) -> Z3B(abcijk)
    Int4 = -0.5*np.einsum('nmij,bm->bnji',H2A['oooo'],r1a,optimize=True) #(*) added factor 1/2 to compenate A(ij)
    Int4 -= np.einsum('bmie,ej->bmji',H2A['voov'],r1a,optimize=True) #(*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int4 += 0.25*np.einsum('bmfe,efij->bmji',H2A['vovv'],r2a,optimize=True) #(*) added factor 1/2 to compensate A(ij)
    Int4 += np.einsum('nmie,bejm->bnji',H2A['ooov'],r2a,optimize=True)
    Int4 += np.einsum('nmie,bejm->bnji',H2B['ooov'],r2b,optimize=True)
    Int4 += 0.5*np.einsum('me,ebij->bmji',Q1,cc_t['t2a'],optimize=True) # (*) added factor 1/2 to compensate A(ij)
    Int4 -= np.transpose(Int4,(0,1,3,2))
    I1 = 0.5*np.einsum('mnef,aefijn->amij',vA['oovv'],r3a,optimize=True)\
        +np.einsum('mnef,aefijn->amij',vB['oovv'],r3b,optimize=True)
    X3B -= 0.5*np.einsum('bnji,acnk->abcijk',Int4+I1,t2b,optimize=True)
    X3B -= 0.5*np.einsum('bnji,acnk->abcijk',H2A['vooo'],r2b,optimize=True)
    # Intermediate 5: X2B(bcje)*Y2B(aeik) -> Z3B(abcijk)
    Int5 = -1.0*np.einsum('mcje,bm->bcje',H2B['ovov'],r1a,optimize=True)
    Int5 -= np.einsum('bmje,cm->bcje',H2B['voov'],r1b,optimize=True)
    Int5 += np.einsum('bcef,ej->bcjf',H2B['vvvv'],r1a,optimize=True)
    Int5 += np.einsum('mnjf,bcmn->bcjf',H2B['ooov'],r2b,optimize=True)
    Int5 += np.einsum('mcef,bejm->bcjf',H2B['ovvv'],r2a,optimize=True)
    Int5 += np.einsum('cmfe,bejm->bcjf',H2C['vovv'],r2b,optimize=True)
    Int5 -= np.einsum('bmef,ecjm->bcjf',H2B['vovv'],r2b,optimize=True)
    I1 = -1.0*np.einsum('nmfe,bfcjnm->bcje',vB['oovv'],r3b,optimize=True)\
         -0.5*np.einsum('mnef,bfcjnm->bcje',vC['oovv'],r3c,optimize=True)
    X3B += np.einsum('bcje,aeik->abcijk',Int5+I1,t2b,optimize=True)
    X3B += np.einsum('bcje,aeik->abcijk',H2B['vvov'],r2b,optimize=True)
    # Intermediate 6: X2B(bnjk)*Y2B(acin) -> Z3B(abcijk)
    Int6 = -1.0*np.einsum('mnjk,bm->bnjk',H2B['oooo'],r1a,optimize=True)
    Int6 += np.einsum('bmje,ek->bmjk',H2B['voov'],r1b,optimize=True)
    Int6 += np.einsum('bmek,ej->bmjk',H2B['vovo'],r1a,optimize=True)
    Int6 += np.einsum('bnef,efjk->bnjk',H2B['vovv'],r2b,optimize=True)
    Int6 += np.einsum('mnek,bejm->bnjk',H2B['oovo'],r2a,optimize=True)
    Int6 += np.einsum('nmke,bejm->bnjk',H2C['ooov'],r2b,optimize=True)
    Int6 -= np.einsum('nmje,benk->bmjk',H2B['ooov'],r2b,optimize=True)
    Int6 += np.einsum('me,bejk->bmjk',Q2,cc_t['t2b'],optimize=True)
    I1 = np.einsum('nmfe,bfejnk->bmjk',vB['oovv'],r3b,optimize=True)\
        +0.5*np.einsum('mnef,befjkn->bmjk',vC['oovv'],r3c,optimize=True)
    X3B -= np.einsum('bnjk,acin->abcijk',Int6+I1,t2b,optimize=True)
    X3B -= np.einsum('bnjk,acin->abcijk',H2B['vooo'],r2b,optimize=True)

    # additional terms with T3 (these contractions mirror the form of
    # the ones with R3 later on)
    I1 = -1.0*np.einsum('me,bm->be',H1A['ov'],r1a,optimize=True)\
         +np.einsum('bnef,fn->be',H2A['vovv'],r1a,optimize=True)\
         +np.einsum('bnef,fn->be',H2B['vovv'],r1b,optimize=True)
    I2 = -0.5*np.einsum('mnef,bfmn->be',vA['oovv'],r2a,optimize=True)\
         -np.einsum('mnef,bfmn->be',vB['oovv'],r2b,optimize=True)
    X3B += 0.5*np.einsum('be,aecijk->abcijk',I1+I2,t3b,optimize=True)

    I1 = -1.0*np.einsum('me,cm->ce',H1B['ov'],r1b,optimize=True)\
         +np.einsum('ncfe,fn->ce',H2B['ovvv'],r1a,optimize=True)\
         +np.einsum('cnef,fn->ce',H2C['vovv'],r1b,optimize=True)
    I2 = -1.0*np.einsum('nmfe,fcnm->ce',vB['oovv'],r2b,optimize=True)\
         -0.5*np.einsum('mnef,fcnm->ce',vC['oovv'],r2c,optimize=True)
    X3B += 0.25*np.einsum('ce,abeijk->abcijk',I1+I2,t3b,optimize=True)

    I1 = np.einsum('me,ej->mj',H1A['ov'],r1a,optimize=True)\
        +np.einsum('mnjf,fn->mj',H2A['ooov'],r1a,optimize=True)\
        +np.einsum('mnjf,fn->mj',H2B['ooov'],r1b,optimize=True)
    I2 = 0.5*np.einsum('mnef,efjn->mj',vA['oovv'],r2a,optimize=True)\
        +np.einsum('mnef,efjn->mj',vB['oovv'],r2b,optimize=True)
    X3B -= 0.5*np.einsum('mj,abcimk->abcijk',I1+I2,t3b,optimize=True)

    I1 = np.einsum('me,ek->mk',H1B['ov'],r1b,optimize=True)\
        +np.einsum('nmfk,fn->mk',H2B['oovo'],r1a,optimize=True)\
        +np.einsum('mnkf,fn->mk',H2C['ooov'],r1b,optimize=True)
    I2 = np.einsum('nmfe,fenk->mk',vB['oovv'],r2b,optimize=True)\
        +0.5*np.einsum('mnef,efkn->mk',vC['oovv'],r2c,optimize=True)
    X3B -= 0.25*np.einsum('mk,abcijm->abcijk',I1+I2,t3b,optimize=True)

    I1 = np.einsum('nmje,ek->nmjk',H2B['ooov'],r1b,optimize=True)\
        +np.einsum('nmek,ej->nmjk',H2B['oovo'],r1a,optimize=True)
    I2 = np.einsum('mnef,efjk->mnjk',vB['oovv'],r2b,optimize=True)
    X3B += 0.5*np.einsum('nmjk,abcinm->abcijk',I1+I2,t3b,optimize=True)

    I1 = np.einsum('mnie,ej->mnij',H2A['ooov'],r1a,optimize=True)
    I1 -= np.transpose(I1,(0,1,3,2))
    I2 = 0.5*np.einsum('mnef,efij->mnij',vA['oovv'],r2a,optimize=True)
    X3B += 0.125*np.einsum('mnij,abcmnk->abcijk',I1+I2,t3b,optimize=True)

    I1 = -1.0*np.einsum('bmfe,cm->bcfe',H2B['vovv'],r1b,optimize=True)\
         -np.einsum('mcfe,bm->bcfe',H2B['ovvv'],r1a,optimize=True)
    I2 = np.einsum('mnef,bcmn->bcef',vB['oovv'],r2b,optimize=True)
    X3B += 0.5*np.einsum('bcfe,afeijk->abcijk',I1+I2,t3b,optimize=True)

    I1 = -1.0*np.einsum('amef,bm->abef',H2A['vovv'],r1a,optimize=True)
    I1 -= np.transpose(I1,(1,0,2,3))
    I2 = 0.5*np.einsum('mnef,abmn->abef',vA['oovv'],r2a,optimize=True)
    X3B += 0.125*np.einsum('abef,efcijk->abcijk',I1+I2,t3b,optimize=True)

    I1 = -1.0*np.einsum('nmfk,cm->ncfk',H2B['oovo'],r1b,optimize=True)\
         +np.einsum('ncfe,ek->ncfk',H2B['ovvv'],r1b,optimize=True)
    I2 = np.einsum('mnef,ecmk->ncfk',vA['oovv'],r2b,optimize=True)\
        +np.einsum('nmfe,ecmk->ncfk',vB['oovv'],r2c,optimize=True)
    X3B += 0.25*np.einsum('ncfk,abfijn->abcijk',I1+I2,t3a,optimize=True)

    I1 = -1.0*np.einsum('mnkf,cm->cnkf',H2C['ooov'],r1b,optimize=True)\
         +np.einsum('cnef,ek->cnkf',H2C['vovv'],r1b,optimize=True)
    I2 = np.einsum('mnef,ecmk->cnkf',vB['oovv'],r2b,optimize=True)\
        +np.einsum('mnef,ecmk->cnkf',vC['oovv'],r2c,optimize=True)
    X3B += 0.25*np.einsum('cnkf,abfijn->abcijk',I1+I2,t3b,optimize=True)

    I1 = np.einsum('bmfe,ek->bmfk',H2B['vovv'],r1b,optimize=True)\
        -np.einsum('nmfk,bn->bmfk',H2B['oovo'],r1a,optimize=True)
    I2 = -1.0*np.einsum('mnef,bfmk->bnek',vB['oovv'],r2b,optimize=True)
    X3B -= 0.5*np.einsum('bmfk,afcijm->abcijk',I1+I2,t3b,optimize=True)

    I1 = -1.0*np.einsum('nmje,cm->ncje',H2B['ooov'],r1b,optimize=True)\
         +np.einsum('ncfe,fj->ncje',H2B['ovvv'],r1a,optimize=True)
    I2 = -1.0*np.einsum('mnef,ecjn->mcjf',vB['oovv'],r2b,optimize=True)
    X3B -= 0.5*np.einsum('ncje,abeink->abcijk',I1+I2,t3b,optimize=True)

    I1 = -1.0*np.einsum('nmje,bn->bmje',H2A['ooov'],r1a,optimize=True)\
         +np.einsum('bmfe,fj->bmje',H2A['vovv'],r1a,optimize=True)
    I2 = np.einsum('mnef,aeim->anif',vA['oovv'],r2a,optimize=True)\
        +np.einsum('nmfe,aeim->anif',vB['oovv'],r2b,optimize=True)
    X3B += np.einsum('bmje,aecimk->abcijk',I1+I2,t3b,optimize=True)

    I1 = -1.0*np.einsum('nmje,bn->bmje',H2B['ooov'],r1a,optimize=True)\
         +np.einsum('bmfe,fj->bmje',H2B['vovv'],r1a,optimize=True)
    I2 = np.einsum('mnef,aeim->anif',vB['oovv'],r2a,optimize=True)\
        +np.einsum('mnef,aeim->anif',vC['oovv'],r2b,optimize=True)
    X3B += np.einsum('bmje,aecimk->abcijk',I1+I2,t3c,optimize=True)

    # < ijk~abc~ | (HR3)_C | 0 >
    X3B -= 0.5*np.einsum('mj,abcimk->abcijk',H1A['oo'],r3b,optimize=True)
    X3B -= 0.25*np.einsum('mk,abcijm->abcijk',H1B['oo'],r3b,optimize=True)
    X3B += 0.5*np.einsum('be,aecijk->abcijk',H1A['vv'],r3b,optimize=True)
    X3B += 0.25*np.einsum('ce,abeijk->abcijk',H1B['vv'],r3b,optimize=True)
    X3B += 0.125*np.einsum('mnij,abcmnk->abcijk',H2A['oooo'],r3b,optimize=True)
    X3B += 0.5*np.einsum('mnjk,abcimn->abcijk',H2B['oooo'],r3b,optimize=True)
    X3B += 0.125*np.einsum('abef,efcijk->abcijk',H2A['vvvv'],r3b,optimize=True)
    X3B += 0.5*np.einsum('bcef,aefijk->abcijk',H2B['vvvv'],r3b,optimize=True)
    X3B += np.einsum('amie,ebcmjk->abcijk',H2A['voov'],r3b,optimize=True)
    X3B += np.einsum('amie,becjmk->abcijk',H2B['voov'],r3c,optimize=True)
    X3B += 0.25*np.einsum('mcek,abeijm->abcijk',H2B['ovvo'],r3a,optimize=True)
    X3B += 0.25*np.einsum('cmke,abeijm->abcijk',H2C['voov'],r3b,optimize=True)
    X3B -= 0.5*np.einsum('bmek,aecijm->abcijk',H2B['vovo'],r3b,optimize=True)
    X3B -= 0.5*np.einsum('mcje,abeimk->abcijk',H2B['ovov'],r3b,optimize=True)

    X3B -= np.transpose(X3B,(0,1,2,4,3,5)) + np.transpose(X3B,(1,0,2,3,4,5)) - np.transpose(X3B,(1,0,2,4,3,5))
    return X3B

def build_HR_3C(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <ij~k~ab~c~|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.

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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    --------
    X3C : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Calculated HR Projection
    """
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']
    t3b = cc_t['t3b']
    t3c = cc_t['t3c']
    t3d = cc_t['t3d']
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']

    # < ij~k~ab~c~ | [ H(R1+R2) ]_C | 0 >
    Q1 = np.einsum('mnef,fn->me',vC['oovv'],r1b,optimize=True)\
                        +np.einsum('nmfe,fn->me',vB['oovv'],r1a,optimize=True)
    Q2 = np.einsum('mnef,fn->me',vB['oovv'],r1b,optimize=True)\
                        +np.einsum('nmfe,fn->me',vA['oovv'],r1a,optimize=True)
    # Intermediate 1: X2B(cbke)*Y2C(aeij) -> Z3C(cbakji)
    Int1 = -1.0*np.einsum('cmke,bm->cbke',H2B['voov'],r1b,optimize=True)
    Int1 -= np.einsum('mbke,cm->cbke',H2B['ovov'],r1a,optimize=True)
    Int1 += np.einsum('cbef,ek->cbkf',H2B['vvvv'],r1a,optimize=True)
    Int1 += np.einsum('nmke,cbnm->cbke',H2B['ooov'],r2b,optimize=True)
    Int1 += np.einsum('bmfe,cekm->cbkf',H2C['vovv'],r2b,optimize=True)
    Int1 += np.einsum('mbef,ecmk->cbkf',H2B['ovvv'],r2a,optimize=True)
    Int1 -= np.einsum('cmef,ebkm->cbkf',H2B['vovv'],r2b,optimize=True)
    I1 = -0.5*np.einsum('mnef,cfbknm->cbke',vC['oovv'],r3c,optimize=True)\
         -np.einsum('nmfe,cfbknm->cbke',vB['oovv'],r3b,optimize=True)
    X3C = 0.5*np.einsum('cbke,aeij->cbakji',Int1+I1,cc_t['t2c'],optimize=True)
    X3C += 0.5*np.einsum('cbke,aeij->cbakji',H2B['vvov'],r2c,optimize=True)
    # Intermediate 2: X2B(cnkj)*Y2C(abin) -> Z3C(cbakji)
    Int2 = -1.0*np.einsum('mnkj,cm->cnkj',H2B['oooo'],r1a,optimize=True)
    Int2 += np.einsum('cmej,ek->cmkj',H2B['vovo'],r1a,optimize=True)
    Int2 += np.einsum('cmke,ej->cmkj',H2B['voov'],r1b,optimize=True)
    Int2 += np.einsum('cmfe,fekj->cmkj',H2B['vovv'],r2b,optimize=True)
    Int2 += np.einsum('nmje,cekm->cnkj',H2C['ooov'],r2b,optimize=True)
    Int2 += np.einsum('mnej,ecmk->cnkj',H2B['oovo'],r2a,optimize=True)
    Int2 -= np.einsum('mnke,cemj->cnkj',H2B['ooov'],r2b,optimize=True)
    I1 = 0.5*np.einsum('mnef,cfeknj->cmkj',vC['oovv'],r3c,optimize=True)\
        +np.einsum('nmfe,cfeknj->cmkj',vB['oovv'],r3b,optimize=True)
    X3C -= 0.5*np.einsum('cnkj,abin->cbakji',Int2+I1,cc_t['t2c'],optimize=True)
    X3C -= 0.5*np.einsum('cmkj,abim->cbakji',H2B['vooo'],r2c,optimize=True)
    # Intermediate 3: X2C(abej)*Y2B(ceki) -> Z3C(cbakji)
    Int3 = np.einsum('amje,bm->abej',H2C['voov'],r1b,optimize=True) #(*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int3 += 0.5*np.einsum('abfe,ej->abfj',H2C['vvvv'],r1b,optimize=True) #(*) added factor 1/2 to compensate A(ab)
    Int3 += 0.25*np.einsum('nmje,abmn->abej',H2C['ooov'],r2c,optimize=True) #(*) added factor 1/2 to compensate A(ab)
    Int3 += np.einsum('amfe,bejm->abfj',H2C['vovv'],r2c,optimize=True)
    Int3 += np.einsum('maef,ebmj->abfj',H2B['ovvv'],r2b,optimize=True)
    Int3 -= 0.5*np.einsum('me,abmj->abej',Q1,cc_t['t2c'],optimize=True) #(*) added factor 1/2 to compensate A(ab)
    Int3 -= np.transpose(Int3,(1,0,2,3))
    I1 = -0.5*np.einsum('mnef,abfmjn->abej',vC['oovv'],r3d,optimize=True)\
         -np.einsum('nmfe,fbanjm->abej',vB['oovv'],r3c,optimize=True)
    X3C += 0.5*np.einsum('abej,ceki->cbakji',Int3+I1,cc_t['t2b'],optimize=True)
    X3C += 0.5*np.einsum('baje,ceki->cbakji',H2C['vvov'],r2b,optimize=True)
    # Intermediate 4: X2C(bnji)*Y2B(cakn) -> Z3C(cbakji)
    Int4 = -0.5*np.einsum('nmij,bm->bnji',H2C['oooo'],r1b,optimize=True) #(*) added factor 1/2 to compenate A(ij)
    Int4 -= np.einsum('bmie,ej->bmji',H2C['voov'],r1b,optimize=True) #(*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int4 += 0.25*np.einsum('bmfe,efij->bmji',H2C['vovv'],r2c,optimize=True) #(*) added factor 1/2 to compensate A(ij)
    Int4 += np.einsum('nmie,bejm->bnji',H2C['ooov'],r2c,optimize=True)
    Int4 += np.einsum('mnei,ebmj->bnji',H2B['oovo'],r2b,optimize=True)
    Int4 += 0.5*np.einsum('me,ebij->bmji',Q1,cc_t['t2c'],optimize=True) # (*) added factor 1/2 to compensate A(ij)
    Int4 -= np.transpose(Int4,(0,1,3,2))
    I1 = 0.5*np.einsum('mnef,aefijn->amij',vC['oovv'],r3d,optimize=True)\
        +np.einsum('nmfe,feanji->amij',vB['oovv'],r3c,optimize=True)
    X3C -= 0.5*np.einsum('bnji,cakn->cbakji',Int4+I1,cc_t['t2b'],optimize=True)
    X3C -= 0.5*np.einsum('bnji,cakn->cbakji',H2C['vooo'],r2b,optimize=True)
    # Intermediate 5: X2B(cbej)*Y2B(eaki) -> Z3C(cbakji)
    Int5 = -1.0*np.einsum('cmej,bm->cbej',H2B['vovo'],r1b,optimize=True)
    Int5 -= np.einsum('mbej,cm->cbej',H2B['ovvo'],r1a,optimize=True)
    Int5 += np.einsum('cbfe,ej->cbfj',H2B['vvvv'],r1b,optimize=True)
    Int5 += np.einsum('nmfj,cbnm->cbfj',H2B['oovo'],r2b,optimize=True)
    Int5 += np.einsum('cmfe,bejm->cbfj',H2B['vovv'],r2c,optimize=True)
    Int5 += np.einsum('cmfe,ebmj->cbfj',H2A['vovv'],r2b,optimize=True)
    Int5 -= np.einsum('mbfe,cemj->cbfj',H2B['ovvv'],r2b,optimize=True)
    I1 = -1.0*np.einsum('mnef,cfbmnj->cbej',vB['oovv'],r3c,optimize=True)\
         -0.5*np.einsum('mnef,cfbmnj->cbej',vA['oovv'],r3b,optimize=True)
    X3C += np.einsum('cbej,eaki->cbakji',Int5+I1,cc_t['t2b'],optimize=True)
    X3C += np.einsum('cbej,eaki->cbakji',H2B['vvvo'],r2b,optimize=True)
    # Intermediate 6: X2B(nbkj)*Y2B(cani) -> Z3C(cbakji)
    Int6 = -1.0*np.einsum('nmkj,bm->nbkj',H2B['oooo'],r1b,optimize=True)
    Int6 += np.einsum('mbej,ek->mbkj',H2B['ovvo'],r1a,optimize=True)
    Int6 += np.einsum('mbke,ej->mbkj',H2B['ovov'],r1b,optimize=True)
    Int6 += np.einsum('nbfe,fekj->nbkj',H2B['ovvv'],r2b,optimize=True)
    Int6 += np.einsum('nmke,bejm->nbkj',H2B['ooov'],r2c,optimize=True)
    Int6 += np.einsum('nmke,ebmj->nbkj',H2A['ooov'],r2b,optimize=True)
    Int6 -= np.einsum('mnej,ebkn->mbkj',H2B['oovo'],r2b,optimize=True)
    Int6 += np.einsum('me,ebkj->mbkj',Q2,cc_t['t2b'],optimize=True)
    I1 = np.einsum('mnef,efbknj->mbkj',vB['oovv'],r3c,optimize=True)\
        +0.5*np.einsum('mnef,febnkj->mbkj',vA['oovv'],r3b,optimize=True)
    X3C -= np.einsum('nbkj,cani->cbakji',Int6+I1,cc_t['t2b'],optimize=True)
    X3C -= np.einsum('nbkj,cani->cbakji',H2B['ovoo'],r2b,optimize=True)

    # additional terms with T3
    I1 = -1.0*np.einsum('me,bm->be',H1B['ov'],r1b,optimize=True)\
         +np.einsum('bnef,fn->be',H2C['vovv'],r1b,optimize=True)\
         +np.einsum('nbfe,fn->be',H2B['ovvv'],r1a,optimize=True)
    I2 = -0.5*np.einsum('mnef,bfmn->be',vC['oovv'],r2c,optimize=True)\
         -np.einsum('nmfe,fbnm->be',vB['oovv'],r2b,optimize=True)
    X3C += 0.5*np.einsum('be,ceakji->cbakji',I1+I2,t3c,optimize=True)

    I1 = -1.0*np.einsum('me,cm->ce',H1A['ov'],r1a,optimize=True)\
         +np.einsum('cnef,fn->ce',H2B['vovv'],r1b,optimize=True)\
         +np.einsum('cnef,fn->ce',H2A['vovv'],r1a,optimize=True)
    I2 = -1.0*np.einsum('mnef,cfmn->ce',vB['oovv'],r2b,optimize=True)\
         -0.5*np.einsum('mnef,fcnm->ce',vA['oovv'],r2a,optimize=True)
    X3C += 0.25*np.einsum('ce,ebakji->cbakji',I1+I2,t3c,optimize=True)

    I1 = np.einsum('me,ej->mj',H1B['ov'],r1b,optimize=True)\
        +np.einsum('mnjf,fn->mj',H2C['ooov'],r1b,optimize=True)\
        +np.einsum('nmfj,fn->mj',H2B['oovo'],r1a,optimize=True)
    I2 = 0.5*np.einsum('mnef,efjn->mj',vC['oovv'],r2c,optimize=True)\
        +np.einsum('nmfe,fenj->mj',vB['oovv'],r2b,optimize=True)
    X3C -= 0.5*np.einsum('mj,cbakmi->cbakji',I1+I2,t3c,optimize=True)

    I1 = np.einsum('me,ek->mk',H1A['ov'],r1a,optimize=True)\
        +np.einsum('mnkf,fn->mk',H2B['ooov'],r1b,optimize=True)\
        +np.einsum('mnkf,fn->mk',H2A['ooov'],r1a,optimize=True)
    I2 = np.einsum('mnef,efkn->mk',vB['oovv'],r2b,optimize=True)\
        +0.5*np.einsum('mnef,efkn->mk',vA['oovv'],r2a,optimize=True)
    X3C -= 0.25*np.einsum('mk,cbamji->cbakji',I1+I2,t3c,optimize=True)

    I1 = np.einsum('mnej,ek->mnkj',H2B['oovo'],r1a,optimize=True)\
        +np.einsum('mnke,ej->mnkj',H2B['ooov'],r1b,optimize=True)
    I2 = np.einsum('nmfe,fekj->nmkj',vB['oovv'],r2b,optimize=True)
    X3C += 0.5*np.einsum('mnkj,cbamni->cbakji',I1+I2,t3c,optimize=True)

    I1 = np.einsum('mnie,ej->mnij',H2C['ooov'],r1b,optimize=True)
    I1 -= np.transpose(I1,(0,1,3,2))
    I2 = 0.5*np.einsum('mnef,efij->mnij',vC['oovv'],r2c,optimize=True)
    X3C += 0.125*np.einsum('mnij,cbaknm->cbakji',I1+I2,t3c,optimize=True)

    I1 = -1.0*np.einsum('mbef,cm->cbef',H2B['ovvv'],r1a,optimize=True)\
         -np.einsum('cmef,bm->cbef',H2B['vovv'],r1b,optimize=True)
    I2 = np.einsum('nmfe,cbnm->cbfe',vB['oovv'],r2b,optimize=True)
    X3C += 0.5*np.einsum('cbef,efakji->cbakji',I1+I2,t3c,optimize=True)

    I1 = -1.0*np.einsum('amef,bm->abef',H2C['vovv'],r1b,optimize=True)
    I1 -= np.transpose(I1,(1,0,2,3))
    I2 = 0.5*np.einsum('mnef,abmn->abef',vC['oovv'],r2c,optimize=True)
    X3C += 0.125*np.einsum('abef,cfekji->cbakji',I1+I2,t3c,optimize=True)

    I1 = -1.0*np.einsum('mnkf,cm->cnkf',H2B['ooov'],r1a,optimize=True)\
         +np.einsum('cnef,ek->cnkf',H2B['vovv'],r1a,optimize=True)
    I2 = np.einsum('mnef,cekm->cnkf',vC['oovv'],r2b,optimize=True)\
        +np.einsum('mnef,ecmk->cnkf',vB['oovv'],r2a,optimize=True)
    X3C += 0.25*np.einsum('cnkf,abfijn->cbakji',I1+I2,t3d,optimize=True)

    I1 = -1.0*np.einsum('mnkf,cm->ncfk',H2A['ooov'],r1a,optimize=True)\
         +np.einsum('cnef,ek->ncfk',H2A['vovv'],r1a,optimize=True)
    I2 = np.einsum('nmfe,cekm->ncfk',vB['oovv'],r2b,optimize=True)\
        +np.einsum('mnef,ecmk->ncfk',vA['oovv'],r2a,optimize=True)
    X3C += 0.25*np.einsum('ncfk,fbanji->cbakji',I1+I2,t3c,optimize=True)

    I1 = np.einsum('mbef,ek->mbkf',H2B['ovvv'],r1a,optimize=True)\
        -np.einsum('mnkf,bn->mbkf',H2B['ooov'],r1b,optimize=True)
    I2 = -1.0*np.einsum('nmfe,fbkm->nbke',vB['oovv'],r2b,optimize=True)
    X3C -= 0.5*np.einsum('mbkf,cfamji->cbakji',I1+I2,t3c,optimize=True)

    I1 = -1.0*np.einsum('mnej,cm->cnej',H2B['oovo'],r1a,optimize=True)\
         +np.einsum('cnef,fj->cnej',H2B['vovv'],r1b,optimize=True)
    I2 = -1.0*np.einsum('nmfe,cenj->cmfj',vB['oovv'],r2b,optimize=True)
    X3C -= 0.5*np.einsum('cnej,ebakni->cbakji',I1+I2,t3c,optimize=True)

    I1 = -1.0*np.einsum('nmje,bn->mbej',H2C['ooov'],r1b,optimize=True)\
         +np.einsum('bmfe,fj->mbej',H2C['vovv'],r1b,optimize=True)
    I2 = np.einsum('mnef,aeim->nafi',vC['oovv'],r2c,optimize=True)\
        +np.einsum('mnef,eami->nafi',vB['oovv'],r2b,optimize=True)
    X3C += np.einsum('mbej,ceakmi->cbakji',I1+I2,t3c,optimize=True)

    I1 = -1.0*np.einsum('mnej,bn->mbej',H2B['oovo'],r1b,optimize=True)\
         +np.einsum('mbef,fj->mbej',H2B['ovvv'],r1b,optimize=True)
    I2 = np.einsum('nmfe,aeim->nafi',vB['oovv'],r2c,optimize=True)\
        +np.einsum('mnef,eami->nafi',vA['oovv'],r2b,optimize=True)
    X3C += np.einsum('mbej,ceakmi->cbakji',I1+I2,t3b,optimize=True)

    # < ijk~abc~ | (HR3)_C | 0 >
    X3C -= 0.5*np.einsum('mj,cbakmi->cbakji',H1B['oo'],r3c,optimize=True)
    X3C -= 0.25*np.einsum('mk,cbamji->cbakji',H1A['oo'],r3c,optimize=True)
    X3C += 0.5*np.einsum('be,ceakji->cbakji',H1B['vv'],r3c,optimize=True)
    X3C += 0.25*np.einsum('ce,ebakji->cbakji',H1A['vv'],r3c,optimize=True)
    X3C += 0.125*np.einsum('mnij,cbaknm->cbakji',H2C['oooo'],r3c,optimize=True)
    X3C += 0.5*np.einsum('nmkj,cbanmi->cbakji',H2B['oooo'],r3c,optimize=True)
    X3C += 0.125*np.einsum('abef,cfekji->cbakji',H2C['vvvv'],r3c,optimize=True)
    X3C += 0.5*np.einsum('cbfe,feakji->cbakji',H2B['vvvv'],r3c,optimize=True)
    X3C += np.einsum('amie,cbekjm->cbakji',H2C['voov'],r3c,optimize=True)
    X3C += np.einsum('maei,cebkmj->cbakji',H2B['ovvo'],r3b,optimize=True)
    X3C += 0.25*np.einsum('cmke,ebamji->cbakji',H2B['voov'],r3d,optimize=True)
    X3C += 0.25*np.einsum('cmke,ebamji->cbakji',H2A['voov'],r3c,optimize=True)
    X3C -= 0.5*np.einsum('mbke,ceamji->cbakji',H2B['ovov'],r3c,optimize=True)
    X3C -= 0.5*np.einsum('cmej,ebakmi->cbakji',H2B['vovo'],r3c,optimize=True)

    X3C -= np.transpose(X3C,(0,1,2,3,5,4)) + np.transpose(X3C,(0,2,1,3,4,5)) - np.transpose(X3C,(0,2,1,3,5,4))
    return X3C

def build_HR_3D(r1a,r1b,r2a,r2b,r2c,r3a,r3b,r3c,r3d,cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Calculate the projection <i~j~k~a~b~c~|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.

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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    --------
    X3D : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Calculated HR Projection
    """
    t2c = cc_t['t2c']
    t3c = cc_t['t3c']
    t3d = cc_t['t3d']
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']

    X3D = 0.0
    # <i~j~k~a~b~c~| [H(R1+R2)]_C | 0 >
    Q1 = np.einsum('mnef,fn->me',vC['oovv'],r1b,optimize=True)
    Q1 += np.einsum('nmfe,fn->me',vB['oovv'],r1a,optimize=True)
    I1 = np.einsum('amje,bm->abej',H2C['voov'],r1b,optimize=True)
    I1 += np.einsum('amfe,bejm->abfj',H2C['vovv'],r2c,optimize=True)
    I1 += np.einsum('maef,ebmj->abfj',H2B['ovvv'],r2b,optimize=True)
    I1 -= np.transpose(I1,(1,0,2,3))
    I2 = np.einsum('abfe,ej->abfj',H2C['vvvv'],r1b,optimize=True)
    I2 += 0.5*np.einsum('nmje,abmn->abej',H2C['ooov'],r2c,optimize=True)
    I2 -= np.einsum('me,abmj->abej',Q1,cc_t['t2c'],optimize=True)
    I3 = -0.5*np.einsum('mnef,abfimn->baei',vC['oovv'],r3d,optimize=True)\
        -np.einsum('nmfe,fbanmi->baei',vB['oovv'],r3c,optimize=True)
    X3D = 0.25*np.einsum('abej,ecik->abcijk',I1+I2+I3,cc_t['t2c'],optimize=True)
    X3D += 0.25*np.einsum('baje,ecik->abcijk',H2C['vvov'],r2c,optimize=True)

    I1 = -np.einsum('bmie,ej->mbij',H2C['voov'],r1b,optimize=True)
    I1 += np.einsum('nmie,bejm->nbij',H2C['ooov'],r2c,optimize=True)
    I1 += np.einsum('mnei,ebmj->nbij',H2B['oovo'],r2b,optimize=True)
    I1 -= np.transpose(I1,(0,1,3,2))
    I2 = -1.0*np.einsum('nmij,bm->nbij',H2C['oooo'],r1b,optimize=True)
    I2 += 0.5*np.einsum('bmfe,efij->mbij',H2C['vovv'],r2c,optimize=True)
    I3 = 0.5*np.einsum('mnef,efcjnk->mcjk',vC['oovv'],r3d,optimize=True)\
        +np.einsum('nmfe,fcenkj->mcjk',vB['oovv'],r3c,optimize=True)
    X3D -= 0.25*np.einsum('mbij,acmk->abcijk',I1+I2+I3,cc_t['t2c'],optimize=True)
    X3D -= 0.25*np.einsum('bmji,acmk->abcijk',H2C['vooo'],r2c,optimize=True)

    # additional terms with T3 in <ijkabc|[ H(R1+R2)]_C | 0>
    I1 = -1.0*np.einsum('me,bm->be',H1B['ov'],r1b,optimize=True)\
         +np.einsum('bnef,fn->be',H2C['vovv'],r1b,optimize=True)\
         +np.einsum('nbfe,fn->be',H2B['ovvv'],r1a,optimize=True)
    I2 = -0.5*np.einsum('mnef,bfmn->be',vC['oovv'],r2c,optimize=True)\
         -np.einsum('nmfe,fbnm->be',vB['oovv'],r2b,optimize=True)
    X3D += (1.0/12.0)*np.einsum('be,aecijk->abcijk',I1+I2,t3d,optimize=True) # A(b/ac)

    I1 = np.einsum('me,ej->mj',H1B['ov'],r1b,optimize=True)\
        +np.einsum('mnjf,fn->mj',H2C['ooov'],r1b,optimize=True)\
        +np.einsum('nmfj,fn->mj',H2B['oovo'],r1a,optimize=True)
    I2 = 0.5*np.einsum('mnef,efjn->mj',vC['oovv'],r2c,optimize=True)\
        +np.einsum('nmfe,fenj->mj',vB['oovv'],r2b,optimize=True)
    X3D -= (1.0/12.0)*np.einsum('mj,abcimk->abcijk',I1+I2,t3d,optimize=True) # A(j/ik)

    I1 = np.einsum('nmje,ei->mnij',H2C['ooov'],r1b,optimize=True)
    I1 -= np.transpose(I1,(0,1,3,2))
    I2 = 0.5*np.einsum('mnef,efij->mnij',vC['oovv'],r2c,optimize=True)
    X3D += (1.0/24.0)*np.einsum('mnij,abcmnk->abcijk',I1+I2,t3d,optimize=True) # A(k/ij)

    I1 = -1.0*np.einsum('amef,bm->abef',H2C['vovv'],r1b,optimize=True)
    I1 -= np.transpose(I1,(1,0,2,3))
    I2 = 0.5*np.einsum('mnef,abmn->abef',vC['oovv'],r2c,optimize=True)
    X3D += (1.0/24.0)*np.einsum('abef,efcijk->abcijk',I1+I2,t3d,optimize=True) # A(c/ab)

    I1 = -1.0*np.einsum('nmje,bn->bmje',H2C['ooov'],r1b,optimize=True)\
         +np.einsum('bmfe,fj->bmje',H2C['vovv'],r1b,optimize=True)
    I2 = np.einsum('mnef,fcnk->cmke',vC['oovv'],r2c,optimize=True)\
        +np.einsum('nmfe,fcnk->cmke',vB['oovv'],r2b,optimize=True)
    X3D += 0.25*np.einsum('bmje,aecimk->abcijk',I1+I2,t3d,optimize=True) # A(j/ik)A(b/ac)

    I1 = -1.0*np.einsum('mnej,bn->bmje',H2B['oovo'],r1b,optimize=True)\
         +np.einsum('mbef,fj->bmje',H2B['ovvv'],r1b,optimize=True)
    I2 = np.einsum('mnef,fcnk->cmke',vB['oovv'],r2c,optimize=True)\
        +np.einsum('mnef,fcnk->cmke',vA['oovv'],r2b,optimize=True)
    X3D += 0.25*np.einsum('bmje,ecamki->abcijk',I1+I2,t3c,optimize=True) # A(j/ik)A(b/ac)

    # < i~j~k~a~b~c~ | (HR3)_C | 0 >
    X3D -= (1.0/12.0)*np.einsum('mj,abcimk->abcijk',H1B['oo'],r3d,optimize=True)
    X3D += (1.0/12.0)*np.einsum('be,aecijk->abcijk',H1B['vv'],r3d,optimize=True)
    X3D += (1.0/24.0)*np.einsum('mnij,abcmnk->abcijk',H2C['oooo'],r3d,optimize=True)
    X3D += (1.0/24.0)*np.einsum('abef,efcijk->abcijk',H2C['vvvv'],r3d,optimize=True)
    X3D += 0.25*np.einsum('amie,ebcmjk->abcijk',H2C['voov'],r3d,optimize=True)
    X3D += 0.25*np.einsum('maei,ecbmkj->abcijk',H2B['ovvo'],r3c,optimize=True)

    # antisymmetrize terms and add up: A(abc)A(ijk) = A(a/bc)A(bc)A(i/jk)A(jk)
    X3D -= np.transpose(X3D,(0,1,2,3,5,4))
    X3D -= np.transpose(X3D,(0,1,2,4,3,5)) + np.transpose(X3D,(0,1,2,5,4,3))
    X3D -= np.transpose(X3D,(0,2,1,3,4,5))
    X3D -= np.transpose(X3D,(1,0,2,3,4,5)) + np.transpose(X3D,(2,1,0,3,4,5))
    return X3D
