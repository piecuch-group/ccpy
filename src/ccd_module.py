"""Module with functions that perform the CC with doubles (CCD) calculation
for a molecular system."""
import numpy as np
from solvers import diis
from cc_energy import calc_cc_energy
import time
import cc_loops

def ccd(sys,ints,maxit=100,tol=1e-08,diis_size=6,shift=0.0,flag_RHF=False):
    """Perform the ground-state CCD calculation.

    Parameters
    ----------
    sys : dict
        System information dictionary
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    maxit : int, optional
        Maximum number of iterations for the CC calculation. Default is 100.
    tol : float, optional
        Convergence tolerance for the CC calculation. Default is 1.0e-08.
    diis_siize : int, optional
        Size of the inversion subspace used in DIIS convergence acceleration. Default is 6.
    shift : float, optional
        Value (in hartree) of the denominator shifting parameter used to converge difficult CC calculations.
        Default is 0.0.

    Returns
    -------
    cc_t : dict
        Contains the converged T2 cluster amplitudes
    Eccd : float
        Total CCD energy
    """
    print('\n==================================++Entering CCD Routine++=================================\n')

    n2a = sys['Nocc_a'] ** 2 * sys['Nunocc_a'] ** 2
    n2b = sys['Nocc_a'] * sys['Nocc_b'] * sys['Nunocc_a'] * sys['Nunocc_b']
    n2c = sys['Nocc_b'] ** 2 * sys['Nunocc_b'] ** 2
    ndim = n2a + n2b + n2c
    idx_2a = slice(0,n2a)
    idx_2b = slice(n2a,n2a+n2b)
    idx_2c = slice(n2a+n2b,n2a+n2b+n2c)

    cc_t = {}
    T = np.zeros(ndim)
    T_list = np.zeros((ndim,diis_size))
    T_resid_list = np.zeros((ndim,diis_size))
    T_old = np.zeros(ndim)

    # Jacobi/DIIS iterations
    it_micro = 0
    flag_conv = False
    it_macro = 0
    Ecorr_old = 0.0

    t_start = time.time()
    print('Iteration    Residuum               deltaE                 Ecorr')
    print('=============================================================================')
    while it_micro < maxit:
        
        # store old T and get current diis dimensions
        T_old = T.copy()
        cc_t['t2a']  = np.reshape(T[idx_2a],(sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a']))
        cc_t['t2b']  = np.reshape(T[idx_2b],(sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b']))
        cc_t['t2c']  = np.reshape(T[idx_2c],(sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b']))

        # CC correlation energy
        Ecorr = calc_cc_energy(cc_t,ints)

        # update T2
        cc_t = update_t2a(cc_t,ints,sys,shift)
        cc_t = update_t2b(cc_t,ints,sys,shift)
        if flag_RHF:
            cc_t['t2c'] = cc_t['t2a']
        else:
            cc_t = update_t2c(cc_t,ints,sys,shift)
        
        # store vectorized results
        T[idx_2a] = cc_t['t2a'].flatten()
        T[idx_2b] = cc_t['t2b'].flatten()
        T[idx_2c] = cc_t['t2c'].flatten()

        # build DIIS residual
        T_resid = T - T_old
        
        # change in Ecorr
        deltaE = Ecorr - Ecorr_old

        # check for exit condition
        ccd_resid = np.linalg.norm(T_resid)
        if ccd_resid < tol and abs(deltaE) < tol:
            flag_conv = True
            break

        # append trial and residual vectors to lists
        T_list[:,it_micro%diis_size] = T
        T_resid_list[:,it_micro%diis_size] = T_resid
        
        if it_micro%diis_size == 0 and it_micro > 1:
            it_macro = it_macro + 1
            print('DIIS Cycle - {}'.format(it_macro))
            T = diis(T_list,T_resid_list)
        
        print('   {}       {:.10f}          {:.10f}          {:.10f}'.format(it_micro,ccd_resid,deltaE,Ecorr))
        
        it_micro += 1
        Ecorr_old = Ecorr

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    if flag_conv:
        print('CCD successfully converged! ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))
        print('')
        print('CCD Correlation Energy = {} Eh'.format(Ecorr))
        print('CCD Total Energy = {} Eh'.format(Ecorr+ints['Escf']))
    else:
        print('Failed to converge CCD in {} iterations'.format(maxit))

    return cc_t, ints['Escf'] + Ecorr

def update_t2a(cc_t,ints,sys,shift):
    """Update t2a amplitudes by calculating the projection <ijab|(H_N e^T2)_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T2
    """
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    fA = ints['fA']
    fB = ints['fB']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']

    # intermediates
    I1A_oo = 0.0
    I1A_oo += 0.5*np.einsum('mnef,efin->mi',vA['oovv'],t2a,optimize=True)
    I1A_oo += np.einsum('mnef,efin->mi',vB['oovv'],t2b,optimize=True)
    I1A_oo += fA['oo']

    I1A_vv = 0.0
    I1A_vv -= 0.5*np.einsum('mnef,afmn->ae',vA['oovv'],t2a,optimize=True)
    I1A_vv -= np.einsum('mnef,afmn->ae',vB['oovv'],t2b,optimize=True)
    I1A_vv += fA['vv']

    I2A_voov = 0.0
    I2A_voov += 0.5*np.einsum('mnef,afin->amie',vA['oovv'],t2a,optimize=True)
    I2A_voov += np.einsum('mnef,afin->amie',vB['oovv'],t2b,optimize=True)
    I2A_voov += vA['voov'] 

    I2A_oooo = 0.0
    I2A_oooo += 0.5*np.einsum('mnef,efij->mnij',vA['oovv'],t2a,optimize=True)
    I2A_oooo += vA['oooo'] 

    I2B_voov = 0.0
    I2B_voov += 0.5*np.einsum('mnef,afin->amie',vC['oovv'],t2b,optimize=True)
    I2B_voov += vB['voov']

    X2A = 0.0
    X2A += vA['vvoo']
    D3 = np.einsum('ae,ebij->abij',I1A_vv,t2a,optimize=True)
    D4 = -np.einsum('mi,abmj->abij',I1A_oo,t2a,optimize=True)
    D5 = np.einsum('amie,ebmj->abij',I2A_voov,t2a,optimize=True)
    D6 = np.einsum('amie,bejm->abij',I2B_voov,t2b,optimize=True)
    X2A += 0.5*np.einsum('abef,efij->abij',vA['vvvv'],t2a,optimize=True)
    X2A += 0.5*np.einsum('mnij,abmn->abij',I2A_oooo,t2a,optimize=True)
    
    # diagrams that have A(ab)
    D13 = D3
    D13 = D13 - np.einsum('abij->baij',D13)
    
    # diagrams that have A(ij)
    D24 = D4
    D24 = D24 - np.einsum('abij->abji',D24)
        
    # diagrams that have A(ab)A(ij)
    D56 = D5 + D6
    D56 = D56 - np.einsum('abij->baij',D56) - np.einsum('abij->abji',D56) + np.einsum('abij->baji',D56)
    
    # total contribution
    X2A += D13 + D24 + D56

    t2a = cc_loops.cc_loops.update_t2a(t2a,X2A,fA['oo'],fA['vv'],shift)

    cc_t['t2a'] = t2a
    return cc_t

def update_t2b(cc_t,ints,sys,shift):
    """Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^T2)_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T2
    """
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    fA = ints['fA']
    fB = ints['fB']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']

    # intermediates
    I1A_vv = 0.0
    I1A_vv -= 0.5*np.einsum('mnef,afmn->ae',vA['oovv'],t2a,optimize=True)
    I1A_vv -= np.einsum('mnef,afmn->ae',vB['oovv'],t2b,optimize=True)
    I1A_vv += fA['vv']

    I1B_vv = 0.0
    I1B_vv -= np.einsum('nmfe,fbnm->be',vB['oovv'],t2b,optimize=True)
    I1B_vv -= 0.5*np.einsum('mnef,fbnm->be',vC['oovv'],t2c,optimize=True)
    I1B_vv += fB['vv']

    I1A_oo = 0.0
    I1A_oo += 0.5*np.einsum('mnef,efin->mi',vA['oovv'],t2a,optimize=True)
    I1A_oo += np.einsum('mnef,efin->mi',vB['oovv'],t2b,optimize=True)
    I1A_oo += fA['oo']

    I1B_oo = 0.0
    I1B_oo += np.einsum('nmfe,fenj->mj',vB['oovv'],t2b,optimize=True)
    I1B_oo += 0.5*np.einsum('mnef,efjn->mj',vC['oovv'],t2c,optimize=True)
    I1B_oo += fB['oo']
    
    I2A_voov = 0.0
    I2A_voov += np.einsum('mnef,aeim->anif',vA['oovv'],t2a,optimize=True)
    I2A_voov += np.einsum('nmfe,aeim->anif',vB['oovv'],t2b,optimize=True)
    I2A_voov += vA['voov']

    I2B_voov = 0.0
    I2B_voov += np.einsum('mnef,aeim->anif',vB['oovv'],t2a,optimize=True)
    I2B_voov += np.einsum('mnef,aeim->anif',vC['oovv'],t2b,optimize=True)
    I2B_voov += vB['voov']

    I2B_oooo = 0.0
    I2B_oooo += np.einsum('mnef,efij->mnij',vB['oovv'],t2b,optimize=True)
    I2B_oooo += vB['oooo']

    I2B_vovo = 0.0
    I2B_vovo -= np.einsum('mnef,afmj->anej',vB['oovv'],t2b,optimize=True)
    I2B_vovo += vB['vovo']

    X2B = 0.0
    X2B += vB['vvoo']
    X2B += np.einsum('ae,ebij->abij',I1A_vv,t2b,optimize=True)
    X2B += np.einsum('be,aeij->abij',I1B_vv,t2b,optimize=True)
    X2B -= np.einsum('mi,abmj->abij',I1A_oo,t2b,optimize=True)
    X2B -= np.einsum('mj,abim->abij',I1B_oo,t2b,optimize=True)
    X2B += np.einsum('amie,ebmj->abij',I2A_voov,t2b,optimize=True)
    X2B += np.einsum('amie,ebmj->abij',I2B_voov,t2c,optimize=True)
    X2B += np.einsum('mbej,aeim->abij',vB['ovvo'],t2a,optimize=True)
    X2B += np.einsum('bmje,aeim->abij',vC['voov'],t2b,optimize=True)
    X2B -= np.einsum('mbie,aemj->abij',vB['ovov'],t2b,optimize=True)
    X2B -= np.einsum('amej,ebim->abij',I2B_vovo,t2b,optimize=True)
    X2B += np.einsum('mnij,abmn->abij',I2B_oooo,t2b,optimize=True)
    X2B += np.einsum('abef,efij->abij',vB['vvvv'],t2b,optimize=True)

    t2b = cc_loops.cc_loops.update_t2b(t2b,X2B,fA['oo'],fA['vv'],fB['oo'],fB['vv'],shift)       

    cc_t['t2b'] = t2b
    return cc_t

def update_t2c(cc_t,ints,sys,shift):
    """Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^T2)_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T2
    """
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    fA = ints['fA']
    fB = ints['fB']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']

    I1B_oo = 0.0
    I1B_oo += 0.5*np.einsum('mnef,efin->mi',vC['oovv'],t2c,optimize=True)
    I1B_oo += np.einsum('nmfe,feni->mi',vB['oovv'],t2b,optimize=True)
    I1B_oo += fB['oo']

    I1B_vv = 0.0
    I1B_vv -= 0.5*np.einsum('mnef,afmn->ae',vC['oovv'],t2c,optimize=True)
    I1B_vv -= np.einsum('nmfe,fanm->ae',vB['oovv'],t2b,optimize=True)
    I1B_vv += fB['vv']
             
    I2C_oooo = 0.0
    I2C_oooo += 0.5*np.einsum('mnef,efij->mnij',vC['oovv'],t2c,optimize=True)
    I2C_oooo += vC['oooo']

    I2B_ovvo = 0.0
    I2B_ovvo += np.einsum('mnef,afin->maei',vB['oovv'],t2c,optimize=True)
    I2B_ovvo += 0.5*np.einsum('mnef,fani->maei',vA['oovv'],t2b,optimize=True)
    I2B_ovvo += vB['ovvo']

    I2C_voov = 0.0
    I2C_voov += 0.5*np.einsum('mnef,afin->amie',vC['oovv'],t2c,optimize=True)
    I2C_voov += vC['voov']
    
    X2C = 0.0
    X2C += vC['vvoo']
    D3 = np.einsum('ae,ebij->abij',I1B_vv,t2c,optimize=True)
    D4 = -np.einsum('mi,abmj->abij',I1B_oo,t2c,optimize=True)
    D5 = np.einsum('amie,ebmj->abij',I2C_voov,t2c,optimize=True)
    D6 = np.einsum('maei,ebmj->abij',I2B_ovvo,t2b,optimize=True)
    X2C += 0.5*np.einsum('abef,efij->abij',vC['vvvv'],t2c,optimize=True)
    X2C += 0.5*np.einsum('mnij,abmn->abij',I2C_oooo,t2c,optimize=True)
    
    # diagrams that have A(ab)
    D13 = D3
    D13 = D13 - np.einsum('abij->baij',D13)
    
    # diagrams that have A(ij)
    D24 = D4
    D24 = D24 - np.einsum('abij->abji',D24)
        
    # diagrams that have A(ab)A(ij)
    D56 = D5 + D6
    D56 = D56 - np.einsum('abij->baij',D56) - np.einsum('abij->abji',D56) + np.einsum('abij->baji',D56)
    
    # total contribution
    X2C += D13 + D24 + D56

    t2c = cc_loops.cc_loops.update_t2c(t2c,X2C,fB['oo'],fB['vv'],shift)

    cc_t['t2c'] = t2c
    return cc_t
