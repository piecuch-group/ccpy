"""Module with functions that perform the CC with singles and 
doubles (CCSD) calculation for a molecular system."""
import numpy as np
from solvers import solve_cc_jacobi, solve_cc_jacobi_out_of_core
from HBar_module import get_ccs_intermediates, get_ccsd_intermediates
from cc_energy import calc_cc_energy
import cc_loops2
from functools import partial

#print(cc_loops.cc_loops.__doc__)

def ccsd(sys,ints,maxit=100,tol=1e-08,diis_size=6,shift=0.0,flag_RHF=False):
    """Perform the ground-state CCSD calculation.

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
        Contains the converged T1, T2 cluster amplitudes
    Eccsd : float
        Total CCSD energy
    """
    print('\n==================================++Entering CCSD Routine++=================================\n')

    n1a = sys['Nocc_a'] * sys['Nunocc_a']
    n1b = sys['Nocc_b'] * sys['Nunocc_b']
    n2a = sys['Nocc_a'] ** 2 * sys['Nunocc_a'] ** 2
    n2b = sys['Nocc_a'] * sys['Nocc_b'] * sys['Nunocc_a'] * sys['Nunocc_b']
    n2c = sys['Nocc_b'] ** 2 * sys['Nunocc_b'] ** 2
    ndim = n1a + n1b + n2a + n2b + n2c

    # Initialize the cc_t dictionary containing the T vectors to 0
    cc_t = {}
    cc_t['t1a']  = np.zeros((sys['Nunocc_a'],sys['Nocc_a']))
    cc_t['t1b']  = np.zeros((sys['Nunocc_b'],sys['Nocc_b']))
    cc_t['t2a']  = np.zeros((sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a']))
    cc_t['t2b']  = np.zeros((sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b']))
    cc_t['t2c']  = np.zeros((sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b']))

    update_t_func = partial(update_t,ints=ints,sys=sys,shift=shift,flag_RHF=flag_RHF)
    cc_t, Eccsd = solve_cc_jacobi(cc_t,update_t_func,ints,maxit,tol,ndim,diis_size)

    return cc_t, Eccsd

def update_t(cc_t,ints,sys,shift,flag_RHF):

    # update T1                        
    cc_t, dt_1a = update_t1a(cc_t,ints,sys,shift)
    if flag_RHF:
        cc_t['t1b'] = cc_t['t1a']
        dt_1b = dt_1a.copy()
    else:
        cc_t, dt_1b = update_t1b(cc_t,ints,sys,shift)
    
    # CCS intermediates
    H1A,H1B,H2A,H2B,H2C = get_ccs_intermediates(cc_t,ints,sys)

    # update T2
    cc_t, dt_2a = update_t2a(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
    cc_t, dt_2b = update_t2b(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
    if flag_RHF:
        cc_t['t2c'] = cc_t['t2a']
        dt_2c = dt_2a.copy()
    else:
        cc_t, dt_2c = update_t2c(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)


    return cc_t, \
        np.concatenate((dt_1a.flatten(),dt_1b.flatten(),dt_2a.flatten(),dt_2b.flatten(),dt_2c.flatten()))


def update_t1a(cc_t,ints,sys,shift):
    """Update t1a amplitudes by calculating the projection <ia|(H_N e^(T1+T2+T3))_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2
    """
    h1A_ov = ints['fA']['ov'] + np.einsum('mnef,fn->me',ints['vA']['oovv'],cc_t['t1a'],optimize=True)\
                    +np.einsum('mnef,fn->me',ints['vB']['oovv'],cc_t['t1b'],optimize=True)
    h1B_ov = ints['fB']['ov'] + np.einsum('mnef,fn->me',ints['vC']['oovv'],cc_t['t1b'],optimize=True)\
                    +np.einsum('nmfe,fn->me',ints['vB']['oovv'],cc_t['t1a'],optimize=True)
    h1A_oo = ints['fA']['oo'] + np.einsum('mnif,fn->mi',ints['vA']['ooov'],cc_t['t1a'],optimize=True)\
                    +np.einsum('mnif,fn->mi',ints['vB']['ooov'],cc_t['t1b'],optimize=True)\
                    +0.5*np.einsum('mnef,efin->mi',ints['vA']['oovv'],cc_t['t2a'],optimize=True)\
                    +np.einsum('mnef,efin->mi',ints['vB']['oovv'],cc_t['t2b'],optimize=True)
    h1A_vv = ints['fA']['vv'] + np.einsum('anef,fn->ae',ints['vA']['vovv'],cc_t['t1a'],optimize=True)\
                    +np.einsum('anef,fn->ae',ints['vB']['vovv'],cc_t['t1b'],optimize=True)\
                    -0.5*np.einsum('mnef,afmn->ae',ints['vA']['oovv'],cc_t['t2a'],optimize=True)\
                    -np.einsum('mnef,afmn->ae',ints['vB']['oovv'],cc_t['t2b'],optimize=True)\
                    -np.einsum('me,am->ae',h1A_ov,cc_t['t1a'],optimize=True)

    X1A = 0.0
    X1A = ints['fA']['vo']+np.einsum('me,aeim->ai',h1A_ov,cc_t['t2a'],optimize=True)\
                    +np.einsum('me,aeim->ai',h1B_ov,cc_t['t2b'],optimize=True)\
                    +np.einsum('amie,em->ai',ints['vA']['voov'],cc_t['t1a'],optimize=True)\
                    +np.einsum('amie,em->ai',ints['vB']['voov'],cc_t['t1b'],optimize=True)\
                    -np.einsum('mi,am->ai',h1A_oo,cc_t['t1a'],optimize=True)\
                    +np.einsum('ae,ei->ai',h1A_vv,cc_t['t1a'],optimize=True)\
                    -0.5*np.einsum('mnif,afmn->ai',ints['vA']['ooov'],cc_t['t2a'],optimize=True)\
                    -np.einsum('mnif,afmn->ai',ints['vB']['ooov'],cc_t['t2b'],optimize=True)\
                    +0.5*np.einsum('anef,efin->ai',ints['vA']['vovv'],cc_t['t2a'],optimize=True)\
                    +np.einsum('anef,efin->ai',ints['vB']['vovv'],cc_t['t2b'],optimize=True)
    cc_t['t1a'], resid = cc_loops2.cc_loops2.update_t1a(cc_t['t1a'],X1A,ints['fA']['oo'],ints['fA']['vv'],shift)
    return cc_t, resid

#@profile
def update_t1b(cc_t,ints,sys,shift):
    """Update t1b amplitudes by calculating the projection <i~a~|(H_N e^(T1+T2+T3))_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2
    """
    h1B_ov = ints['fB']['ov'] + np.einsum('mnef,fn->me',ints['vC']['oovv'],cc_t['t1b'],optimize=True)\
                    +np.einsum('nmfe,fn->me',ints['vB']['oovv'],cc_t['t1a'],optimize=True)
    h1A_ov = ints['fA']['ov'] + np.einsum('mnef,fn->me',ints['vA']['oovv'],cc_t['t1a'],optimize=True)\
                    +np.einsum('mnef,fn->me',ints['vB']['oovv'],cc_t['t1b'],optimize=True)
    h1B_oo = ints['fB']['oo'] + np.einsum('mnif,fn->mi',ints['vC']['ooov'],cc_t['t1b'],optimize=True)\
                    +np.einsum('nmfi,fn->mi',ints['vB']['oovo'],cc_t['t1a'],optimize=True)\
                    +0.5*np.einsum('mnef,efin->mi',ints['vC']['oovv'],cc_t['t2c'],optimize=True)\
                    +np.einsum('nmfe,feni->mi',ints['vB']['oovv'],cc_t['t2b'],optimize=True)
    h1B_vv = ints['fB']['vv'] + np.einsum('anef,fn->ae',ints['vC']['vovv'],cc_t['t1b'],optimize=True)\
                    +np.einsum('nafe,fn->ae',ints['vB']['ovvv'],cc_t['t1a'],optimize=True)\
                    -0.5*np.einsum('mnef,afmn->ae',ints['vC']['oovv'],cc_t['t2c'],optimize=True)\
                    -np.einsum('nmfe,fanm->ae',ints['vB']['oovv'],cc_t['t2b'],optimize=True)\
                    -np.einsum('me,am->ae',h1B_ov,cc_t['t1b'],optimize=True)

    X1B = 0.0
    X1B = ints['fB']['vo']+np.einsum('me,aeim->ai',h1B_ov,cc_t['t2c'],optimize=True)\
                    +np.einsum('me,eami->ai',h1A_ov,cc_t['t2b'],optimize=True)\
                    +np.einsum('amie,em->ai',ints['vC']['voov'],cc_t['t1b'],optimize=True)\
                    +np.einsum('maei,em->ai',ints['vB']['ovvo'],cc_t['t1a'],optimize=True)\
                    -np.einsum('mi,am->ai',h1B_oo,cc_t['t1b'],optimize=True)\
                    +np.einsum('ae,ei->ai',h1B_vv,cc_t['t1b'],optimize=True)\
                    -0.5*np.einsum('mnif,afmn->ai',ints['vC']['ooov'],cc_t['t2c'],optimize=True)\
                    -np.einsum('nmfi,fanm->ai',ints['vB']['oovo'],cc_t['t2b'],optimize=True)\
                    +0.5*np.einsum('anef,efin->ai',ints['vC']['vovv'],cc_t['t2c'],optimize=True)\
                    +np.einsum('nafe,feni->ai',ints['vB']['ovvv'],cc_t['t2b'],optimize=True)
    cc_t['t1b'], resid = cc_loops2.cc_loops2.update_t1b(cc_t['t1b'],X1B,ints['fB']['oo'],ints['fB']['vv'],shift)
    return cc_t, resid

#@profile
def update_t2a(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift):
    """Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2+T3))_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2
    """
    # intermediates
    I1A_oo = 0.5*np.einsum('mnef,efin->mi',ints['vA']['oovv'],cc_t['t2a'],optimize=True)
    I1A_oo += np.einsum('mnef,efin->mi',ints['vB']['oovv'],cc_t['t2b'],optimize=True)
    I1A_oo += H1A['oo']
    I1A_vv = -0.5*np.einsum('mnef,afmn->ae',ints['vA']['oovv'],cc_t['t2a'],optimize=True)
    I1A_vv -= np.einsum('mnef,afmn->ae',ints['vB']['oovv'],cc_t['t2b'],optimize=True)
    I1A_vv += H1A['vv']
    I2A_voov = 0.5*np.einsum('mnef,afin->amie',ints['vA']['oovv'],cc_t['t2a'],optimize=True)
    I2A_voov += np.einsum('mnef,afin->amie',ints['vB']['oovv'],cc_t['t2b'],optimize=True)
    I2A_voov += H2A['voov'] 
    I2A_oooo = 0.5*np.einsum('mnef,efij->mnij',ints['vA']['oovv'],cc_t['t2a'],optimize=True)
    I2A_oooo += H2A['oooo'] 
    I2B_voov = 0.5*np.einsum('mnef,afin->amie',ints['vC']['oovv'],cc_t['t2b'],optimize=True)
    I2B_voov += H2B['voov']

    X2A = 0.25*ints['vA']['vvoo']
    X2A -= 0.5*np.einsum('amij,bm->abij',H2A['vooo'],cc_t['t1a'],optimize=True)
    X2A += 0.5*np.einsum('abie,ej->abij',H2A['vvov'],cc_t['t1a'],optimize=True)
    X2A += 0.5*np.einsum('ae,ebij->abij',I1A_vv,cc_t['t2a'],optimize=True)
    X2A -= 0.5*np.einsum('mi,abmj->abij',I1A_oo,cc_t['t2a'],optimize=True)
    X2A += np.einsum('amie,ebmj->abij',I2A_voov,cc_t['t2a'],optimize=True)
    X2A += np.einsum('amie,bejm->abij',I2B_voov,cc_t['t2b'],optimize=True)
    X2A += 0.125*np.einsum('abef,efij->abij',H2A['vvvv'],cc_t['t2a'],optimize=True)
    X2A += 0.125*np.einsum('mnij,abmn->abij',I2A_oooo,cc_t['t2a'],optimize=True)

    cc_t['t2a'], resid = cc_loops2.cc_loops2.update_t2a(cc_t['t2a'],X2A,ints['fA']['oo'],ints['fA']['vv'],shift)
    return cc_t, resid

#@profile
def update_t2b(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift):
    """Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2+T3))_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2
    """
    # intermediates
    I1A_vv = -0.5*np.einsum('mnef,afmn->ae',ints['vA']['oovv'],cc_t['t2a'],optimize=True)
    I1A_vv -= np.einsum('mnef,afmn->ae',ints['vB']['oovv'],cc_t['t2b'],optimize=True)
    I1A_vv += H1A['vv']
    I1B_vv = -np.einsum('nmfe,fbnm->be',ints['vB']['oovv'],cc_t['t2b'],optimize=True)
    I1B_vv -= 0.5*np.einsum('mnef,fbnm->be',ints['vC']['oovv'],cc_t['t2c'],optimize=True)
    I1B_vv += H1B['vv']
    I1A_oo = 0.5*np.einsum('mnef,efin->mi',ints['vA']['oovv'],cc_t['t2a'],optimize=True)
    I1A_oo += np.einsum('mnef,efin->mi',ints['vB']['oovv'],cc_t['t2b'],optimize=True)
    I1A_oo += H1A['oo']
    I1B_oo = np.einsum('nmfe,fenj->mj',ints['vB']['oovv'],cc_t['t2b'],optimize=True)
    I1B_oo += 0.5*np.einsum('mnef,efjn->mj',ints['vC']['oovv'],cc_t['t2c'],optimize=True)
    I1B_oo += H1B['oo']
    I2A_voov = np.einsum('mnef,aeim->anif',ints['vA']['oovv'],cc_t['t2a'],optimize=True)
    I2A_voov += np.einsum('nmfe,aeim->anif',ints['vB']['oovv'],cc_t['t2b'],optimize=True)
    I2A_voov += H2A['voov']
    I2B_voov = np.einsum('mnef,aeim->anif',ints['vB']['oovv'],cc_t['t2a'],optimize=True)
    I2B_voov += np.einsum('mnef,aeim->anif',ints['vC']['oovv'],cc_t['t2b'],optimize=True)
    I2B_voov += H2B['voov']
    I2B_oooo = np.einsum('mnef,efij->mnij',ints['vB']['oovv'],cc_t['t2b'],optimize=True)
    I2B_oooo += H2B['oooo']
    I2B_vovo = -np.einsum('mnef,afmj->anej',ints['vB']['oovv'],cc_t['t2b'],optimize=True)
    I2B_vovo += H2B['vovo']

    X2B = 0.0
    X2B += ints['vB']['vvoo']
    X2B -= np.einsum('mbij,am->abij',H2B['ovoo'],cc_t['t1a'],optimize=True)
    X2B -= np.einsum('amij,bm->abij',H2B['vooo'],cc_t['t1b'],optimize=True)
    X2B += np.einsum('abej,ei->abij',H2B['vvvo'],cc_t['t1a'],optimize=True)
    X2B += np.einsum('abie,ej->abij',H2B['vvov'],cc_t['t1b'],optimize=True)
    X2B += np.einsum('ae,ebij->abij',I1A_vv,cc_t['t2b'],optimize=True)
    X2B += np.einsum('be,aeij->abij',I1B_vv,cc_t['t2b'],optimize=True)
    X2B -= np.einsum('mi,abmj->abij',I1A_oo,cc_t['t2b'],optimize=True)
    X2B -= np.einsum('mj,abim->abij',I1B_oo,cc_t['t2b'],optimize=True)
    X2B += np.einsum('amie,ebmj->abij',I2A_voov,cc_t['t2b'],optimize=True)
    X2B += np.einsum('amie,ebmj->abij',I2B_voov,cc_t['t2c'],optimize=True)
    X2B += np.einsum('mbej,aeim->abij',H2B['ovvo'],cc_t['t2a'],optimize=True)
    X2B += np.einsum('bmje,aeim->abij',H2C['voov'],cc_t['t2b'],optimize=True)
    X2B -= np.einsum('mbie,aemj->abij',H2B['ovov'],cc_t['t2b'],optimize=True)
    X2B -= np.einsum('amej,ebim->abij',I2B_vovo,cc_t['t2b'],optimize=True)
    X2B += np.einsum('mnij,abmn->abij',I2B_oooo,cc_t['t2b'],optimize=True)
    X2B += np.einsum('abef,efij->abij',H2B['vvvv'],cc_t['t2b'],optimize=True)

    cc_t['t2b'], resid = cc_loops2.cc_loops2.update_t2b(cc_t['t2b'],X2B,ints['fA']['oo'],ints['fA']['vv'],ints['fB']['oo'],ints['fB']['vv'],shift)
    return cc_t, resid

#@profile
def update_t2c(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift):
    """Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2+T3))_C|0>.

    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)

    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2
    """
    # intermediates
    I1B_oo = 0.5*np.einsum('mnef,efin->mi',ints['vC']['oovv'],cc_t['t2c'],optimize=True)
    I1B_oo += np.einsum('nmfe,feni->mi',ints['vB']['oovv'],cc_t['t2b'],optimize=True)
    I1B_oo += H1B['oo']
    I1B_vv = -0.5*np.einsum('mnef,afmn->ae',ints['vC']['oovv'],cc_t['t2c'],optimize=True)
    I1B_vv -= np.einsum('nmfe,fanm->ae',ints['vB']['oovv'],cc_t['t2b'],optimize=True)
    I1B_vv += H1B['vv']
    I2C_oooo = 0.5*np.einsum('mnef,efij->mnij',ints['vC']['oovv'],cc_t['t2c'],optimize=True)
    I2C_oooo += H2C['oooo']
    I2B_ovvo = np.einsum('mnef,afin->maei',ints['vB']['oovv'],cc_t['t2c'],optimize=True)
    I2B_ovvo += 0.5*np.einsum('mnef,fani->maei',ints['vA']['oovv'],cc_t['t2b'],optimize=True)
    I2B_ovvo += H2B['ovvo']
    I2C_voov = 0.5*np.einsum('mnef,afin->amie',ints['vC']['oovv'],cc_t['t2c'],optimize=True)
    I2C_voov += H2C['voov']
    
    X2C = 0.25*ints['vC']['vvoo']
    X2C -= 0.5*np.einsum('bmji,am->abij',H2C['vooo'],cc_t['t1b'],optimize=True)
    X2C += 0.5*np.einsum('baje,ei->abij',H2C['vvov'],cc_t['t1b'],optimize=True)
    X2C += 0.5*np.einsum('ae,ebij->abij',I1B_vv,cc_t['t2c'],optimize=True)
    X2C -= 0.5*np.einsum('mi,abmj->abij',I1B_oo,cc_t['t2c'],optimize=True)
    X2C += np.einsum('amie,ebmj->abij',I2C_voov,cc_t['t2c'],optimize=True)
    X2C += np.einsum('maei,ebmj->abij',I2B_ovvo,cc_t['t2b'],optimize=True)
    X2C += 0.125*np.einsum('abef,efij->abij',H2C['vvvv'],cc_t['t2c'],optimize=True)
    X2C += 0.125*np.einsum('mnij,abmn->abij',I2C_oooo,cc_t['t2c'],optimize=True)

    cc_t['t2c'], resid = cc_loops2.cc_loops2.update_t2c(cc_t['t2c'],X2C,ints['fB']['oo'],ints['fB']['vv'],shift)
    return cc_t, resid
