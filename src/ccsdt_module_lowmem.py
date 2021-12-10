"""Module with functions that perform the CC with singles, doubles,
and triples (CCSDT) calculation for a molecular system."""
import numpy as np
from solvers import diis_out_of_core
from cc_energy import calc_cc_energy
import time
import cc_loops
from utilities import print_memory_usage, clean_up

#print(cc_loops.cc_loops.__doc__)

def ccsdt(sys,ints,work_dir,maxit=100,tol=1e-08,diis_size=6,shift=0.0,flag_RHF=False):
    """Perform the ground-state CCSDT calculation.

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
    flag_save : bool, optional
        Flag to indicate whether the T1, T2, and T3 amplitudes should be saved. If True,
        they will be saved in the path specified by save_location and in the .npy format
        (they will be large!). Default is False
    save_location : str, optional
        Path to directory in which T vectors will be saved. Default is None.

    Returns
    -------
    cc_t : dict
        Contains the converged T1, T2 cluster amplitudes
    Eccsdt : float
        Total CCSDT energy
    """
    print('\n==================================++Entering CCSDT Routine++=================================\n')
    print('>> LOW MEMORY VERSION <<')

    vecfid = work_dir + '/t_diis'
    dvecfid = work_dir + '/dt_diis'

    n1a = sys['Nocc_a'] * sys['Nunocc_a']
    n1b = sys['Nocc_b'] * sys['Nunocc_b']
    n2a = sys['Nocc_a'] ** 2 * sys['Nunocc_a'] ** 2
    n2b = sys['Nocc_a'] * sys['Nocc_b'] * sys['Nunocc_a'] * sys['Nunocc_b']
    n2c = sys['Nocc_b'] ** 2 * sys['Nunocc_b'] ** 2
    n3a = sys['Nocc_a'] ** 3 * sys['Nunocc_a'] ** 3
    n3b = sys['Nocc_a'] ** 2 * sys['Nocc_b'] * sys['Nunocc_a'] ** 2 * sys['Nunocc_b']
    n3c = sys['Nocc_a'] * sys['Nocc_b'] ** 2 * sys['Nunocc_a'] * sys['Nunocc_b'] ** 2
    n3d = sys['Nocc_b'] ** 3 * sys['Nunocc_b'] ** 3

    ndim = n1a + n1b + n2a + n2b + n2c + n3a + n3b + n3c + n3d
    idx_1a = slice(0,n1a)
    idx_1b = slice(n1a,n1a+n1b)
    idx_2a = slice(n1a+n1b,n1a+n1b+n2a)
    idx_2b = slice(n1a+n1b+n2a,n1a+n1b+n2a+n2b)
    idx_2c = slice(n1a+n1b+n2a+n2b,n1a+n1b+n2a+n2b+n2c)
    idx_3a = slice(n1a+n1b+n2a+n2b+n2c,n1a+n1b+n2a+n2b+n2c+n3a)
    idx_3b = slice(n1a+n1b+n2a+n2b+n2c+n3a,n1a+n1b+n2a+n2b+n2c+n3a+n3b)    
    idx_3c = slice(n1a+n1b+n2a+n2b+n2c+n3a+n3b,n1a+n1b+n2a+n2b+n2c+n3a+n3b+n3c)
    idx_3d = slice(n1a+n1b+n2a+n2b+n2c+n3a+n3b+n3c,n1a+n1b+n2a+n2b+n2c+n3a+n3b+n3c+n3d)

    cc_t = {}
    T = np.zeros(ndim)
    T_resid = np.zeros(ndim)
    # Jacobi/DIIS iterations
    it_micro = 0
    flag_conv = False
    it_macro = 0
    Ecorr_old = 0.0

    t_start = time.time()
    print('Iteration    Residuum               deltaE                 Ecorr')
    print('=============================================================================')
    while it_micro < maxit:
        
        cc_t['t1a']  = np.reshape(T[idx_1a],(sys['Nunocc_a'],sys['Nocc_a']))
        cc_t['t1b']  = np.reshape(T[idx_1b],(sys['Nunocc_b'],sys['Nocc_b']))
        cc_t['t2a']  = np.reshape(T[idx_2a],(sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a']))
        cc_t['t2b']  = np.reshape(T[idx_2b],(sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b']))
        cc_t['t2c']  = np.reshape(T[idx_2c],(sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b']))
        cc_t['t3a']  = np.reshape(T[idx_3a],(sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_a']))
        cc_t['t3b']  = np.reshape(T[idx_3b],(sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_b']))
        cc_t['t3c']  = np.reshape(T[idx_3c],(sys['Nunocc_a'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b'],sys['Nocc_b']))
        cc_t['t3d']  = np.reshape(T[idx_3d],(sys['Nunocc_b'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b'],sys['Nocc_b']))
        # CC correlation energy
        Ecorr = calc_cc_energy(cc_t,ints)
       
        # update T1                        
        cc_t, T_resid[idx_1a] = update_t1a(cc_t,ints,sys,shift)
        if flag_RHF:
            cc_t['t1b'] = cc_t['t1a']
            T_resid[idx_1b] = T_resid[idx_1a]
        else:
            cc_t, T_resid[idx_1b] = update_t1b(cc_t,ints,sys,shift)
        # CCS intermediates
        H1A,H1B,H2A,H2B,H2C = get_ccs_intermediates(cc_t,ints,sys)
        # update T2
        cc_t, T_resid[idx_2a] = update_t2a(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        cc_t, T_resid[idx_2b] = update_t2b(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        if flag_RHF:
            cc_t['t2c'] = cc_t['t2a']
            T_resid[idx_2c] = T_resid[idx_2a]
        else:
            cc_t, T_resid[idx_2c] = update_t2c(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        # CCSD intermediates
        H1A,H1B,H2A,H2B,H2C = get_ccsd_intermediates(cc_t,ints,sys)
        # update T3
        cc_t, T_resid[idx_3a] = update_t3a(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        cc_t, T_resid[idx_3b] = update_t3b(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        if flag_RHF:
            cc_t['t3c'] = np.transpose(cc_t['t3b'],(2,0,1,5,3,4))
            T_resid[idx_3c] = T_resid[idx_3b] # THIS PROBABLY DOESN'T WORK!!!
            cc_t['t3d'] = cc_t['t3a']
            T_resid[idx_3d] = T_resid[idx_3a]
        else:
            cc_t, T_resid[idx_3c] = update_t3c(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
            cc_t, T_resid[idx_3d] = update_t3d(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        
        # store vectorized results
        T[idx_1a] = cc_t['t1a'].flatten()
        T[idx_1b] = cc_t['t1b'].flatten()
        T[idx_2a] = cc_t['t2a'].flatten()
        T[idx_2b] = cc_t['t2b'].flatten()
        T[idx_2c] = cc_t['t2c'].flatten()
        T[idx_3a] = cc_t['t3a'].flatten()
        T[idx_3b] = cc_t['t3b'].flatten()
        T[idx_3c] = cc_t['t3c'].flatten()
        T[idx_3d] = cc_t['t3d'].flatten()
        
        # change in Ecorr
        deltaE = Ecorr - Ecorr_old

        # check for exit condition
        resid = np.linalg.norm(T_resid)
        if resid < tol and abs(deltaE) < tol:
            flag_conv = True
            break

        # Save T and dT vectors to disk for out of core DIIS
        ndiis = it_micro%diis_size + 1
        np.save(vecfid+'-'+str(ndiis)+'.npy',T)
        np.save(dvecfid+'-'+str(ndiis)+'.npy',T_resid)
        # Do DIIS extrapolation        
        if it_micro%diis_size == 0 and it_micro > 1:
            it_macro = it_macro + 1
            print('DIIS Cycle - {}'.format(it_macro))
            T = diis_out_of_core(vecfid,dvecfid,ndim,diis_size)
        
        print('   {}       {:.10f}          {:.10f}          {:.10f}'.format(it_micro,resid,deltaE,Ecorr))
        
        it_micro += 1
        Ecorr_old = Ecorr

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    if flag_conv:
        print('CCSDT successfully converged! ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))
        print('')
        print('CCSDT Correlation Energy = {} Eh'.format(Ecorr))
        print('CCSDT Total Energy = {} Eh'.format(Ecorr + ints['Escf']))
    else:
        print('Failed to converge CCSDT in {} iterations'.format(maxit))

    # Clean up the DIIS files from the working directory
    clean_up(vecfid,diis_size)
    clean_up(dvecfid,diis_size)

    return cc_t, ints['Escf'] + Ecorr

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
        New cluster amplitudes T1, T2, T3
    """
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    fA = ints['fA']
    fB = ints['fB']
    t1a = cc_t['t1a']
    t1b = cc_t['t1b']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t3a = cc_t['t3a']
    t3b = cc_t['t3b']
    t3c = cc_t['t3c']

    chi1A_vv = 0.0
    chi1A_vv += fA['vv']
    chi1A_vv += np.einsum('anef,fn->ae',vA['vovv'],t1a,optimize=True)
    chi1A_vv += np.einsum('anef,fn->ae',vB['vovv'],t1b,optimize=True)

    chi1A_oo = 0.0
    chi1A_oo += fA['oo']
    chi1A_oo += np.einsum('mnif,fn->mi',vA['ooov'],t1a,optimize=True)
    chi1A_oo += np.einsum('mnif,fn->mi',vB['ooov'],t1b,optimize=True)

    h1A_ov = 0.0
    h1A_ov += fA['ov']
    h1A_ov += np.einsum('mnef,fn->me',vA['oovv'],t1a,optimize=True) 
    h1A_ov += np.einsum('mnef,fn->me',vB['oovv'],t1b,optimize=True) 

    h1B_ov = 0.0
    h1B_ov += fB['ov'] 
    h1B_ov += np.einsum('nmfe,fn->me',vB['oovv'],t1a,optimize=True) 
    h1B_ov += np.einsum('mnef,fn->me',vC['oovv'],t1b,optimize=True) 

    h1A_oo = 0.0
    h1A_oo += chi1A_oo 
    h1A_oo += np.einsum('me,ei->mi',h1A_ov,t1a,optimize=True) 

    M11 = 0.0
    M11 += fA['vo']
    M11 -= np.einsum('mi,am->ai',h1A_oo,t1a,optimize=True) 
    M11 += np.einsum('ae,ei->ai',chi1A_vv,t1a,optimize=True) 
    M11 += np.einsum('anif,fn->ai',vA['voov'],t1a,optimize=True) 
    M11 += np.einsum('anif,fn->ai',vB['voov'],t1b,optimize=True) 
      
    h2A_ooov = vA['ooov'] + np.einsum('mnfe,fi->mnie',vA['oovv'],t1a,optimize=True) 
    h2B_ooov = vB['ooov'] + np.einsum('mnfe,fi->mnie',vB['oovv'],t1a,optimize=True) 
    h2A_vovv = vA['vovv'] - np.einsum('mnfe,an->amef',vA['oovv'],t1a,optimize=True) 
    h2B_vovv = vB['vovv'] - np.einsum('nmef,an->amef',vB['oovv'],t1a,optimize=True) 

    CCS_T2 = 0.0
    CCS_T2 += np.einsum('me,aeim->ai',h1A_ov,t2a,optimize=True)
    CCS_T2 += np.einsum('me,aeim->ai',h1B_ov,t2b,optimize=True)
    CCS_T2 -= 0.5*np.einsum('mnif,afmn->ai',h2A_ooov,t2a,optimize=True)
    CCS_T2 -= np.einsum('mnif,afmn->ai',h2B_ooov,t2b,optimize=True)
    CCS_T2 += 0.5*np.einsum('anef,efin->ai',h2A_vovv,t2a,optimize=True)
    CCS_T2 += np.einsum('anef,efin->ai',h2B_vovv,t2b,optimize=True)
       
    X1A = M11 + CCS_T2

    X1A += 0.25*np.einsum('mnef,aefimn->ai',vA['oovv'],t3a,optimize=True)
    X1A += np.einsum('mnef,aefimn->ai',vB['oovv'],t3b,optimize=True)
    X1A += 0.25*np.einsum('mnef,aefimn->ai',vC['oovv'],t3c,optimize=True)

    t1a, resid = cc_loops2.cc_loops2.update_t1a(t1a,X1A,fA['oo'],fA['vv'],shift)

    cc_t['t1a'] = t1a

    return cc_t, resid.flatten()

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
        New cluster amplitudes T1, T2, T3
    """
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    fA = ints['fA']
    fB = ints['fB']
    t1a = cc_t['t1a']
    t1b = cc_t['t1b']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']
    t3b = cc_t['t3b']
    t3c = cc_t['t3c']
    t3d = cc_t['t3d']

    chi1B_vv = 0.0
    chi1B_vv += fB['vv']
    chi1B_vv += np.einsum('anef,fn->ae',vC['vovv'],t1b,optimize=True)
    chi1B_vv += np.einsum('nafe,fn->ae',vB['ovvv'],t1a,optimize=True)

    chi1B_oo = 0.0
    chi1B_oo += fB['oo']
    chi1B_oo += np.einsum('mnif,fn->mi',vC['ooov'],t1b,optimize=True)
    chi1B_oo += np.einsum('nmfi,fn->mi',vB['oovo'],t1a,optimize=True)

    h1A_ov = 0.0
    h1A_ov += fA['ov']
    h1A_ov += np.einsum('mnef,fn->me',vA['oovv'],t1a,optimize=True) 
    h1A_ov += np.einsum('mnef,fn->me',vB['oovv'],t1b,optimize=True) 

    h1B_ov = 0.0
    h1B_ov += fB['ov'] 
    h1B_ov += np.einsum('nmfe,fn->me',vB['oovv'],t1a,optimize=True) 
    h1B_ov += np.einsum('mnef,fn->me',vC['oovv'],t1b,optimize=True) 

    h1B_oo = 0.0
    h1B_oo += chi1B_oo + np.einsum('me,ei->mi',h1B_ov,t1b,optimize=True)

    M11 = 0.0
    M11 += fB['vo']
    M11 -= np.einsum('mi,am->ai',h1B_oo,t1b,optimize=True)
    M11 += np.einsum('ae,ei->ai',chi1B_vv,t1b,optimize=True)
    M11 += np.einsum('anif,fn->ai',vC['voov'],t1b,optimize=True)
    M11 += np.einsum('nafi,fn->ai',vB['ovvo'],t1a,optimize=True)

    h2C_ooov = 0.0
    h2C_ooov += vC['ooov']
    h2C_ooov += np.einsum('mnfe,fi->mnie',vC['oovv'],t1b,optimize=True)

    h2B_oovo = 0.0
    h2B_oovo += vB['oovo']
    h2B_oovo += np.einsum('nmef,fi->nmei',vB['oovv'],t1b,optimize=True)
    
    h2C_vovv = 0.0
    h2C_vovv += vC['vovv']
    h2C_vovv -= np.einsum('mnfe,an->amef',vC['oovv'],t1b,optimize=True)
    
    h2B_ovvv = 0.0
    h2B_ovvv += vB['ovvv']
    h2B_ovvv -= np.einsum('mnfe,an->mafe',vB['oovv'],t1b,optimize=True)

    CCS_T2 = 0.0
    CCS_T2 += np.einsum('me,eami->ai',h1A_ov,t2b,optimize=True)
    CCS_T2 += np.einsum('me,aeim->ai',h1B_ov,t2c,optimize=True)
    CCS_T2 -= 0.5*np.einsum('mnif,afmn->ai',h2C_ooov,t2c,optimize=True)
    CCS_T2 -= np.einsum('nmfi,fanm->ai',h2B_oovo,t2b,optimize=True)
    CCS_T2 += 0.5*np.einsum('anef,efin->ai',h2C_vovv,t2c,optimize=True)
    CCS_T2 += np.einsum('nafe,feni->ai',h2B_ovvv,t2b,optimize=True)
       
    X1B = M11 + CCS_T2; 

    X1B += 0.25*np.einsum('mnef,aefimn->ai',vC['oovv'],t3d,optimize=True)
    X1B += 0.25*np.einsum('mnef,efamni->ai',vA['oovv'],t3b,optimize=True)
    X1B += np.einsum('mnef,efamni->ai',vB['oovv'],t3c,optimize=True)
    
    t1b, resid = cc_loops2.cc_loops2.update_t1b(t1b,X1B,fB['oo'],fB['vv'],shift)

    cc_t['t1b'] = t1b        
    return cc_t, resid.flatten()


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
        New cluster amplitudes T1, T2, T3
    """
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    fA = ints['fA']
    t1a = cc_t['t1a']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t3a = cc_t['t3a']
    t3b = cc_t['t3b']

    # intermediates
    I1A_oo = 0.0
    I1A_oo += 0.5*np.einsum('mnef,efin->mi',vA['oovv'],t2a,optimize=True)
    I1A_oo += np.einsum('mnef,efin->mi',vB['oovv'],t2b,optimize=True)
    I1A_oo += H1A['oo']

    I1A_vv = 0.0
    I1A_vv -= 0.5*np.einsum('mnef,afmn->ae',vA['oovv'],t2a,optimize=True)
    I1A_vv -= np.einsum('mnef,afmn->ae',vB['oovv'],t2b,optimize=True)
    I1A_vv += H1A['vv']

    I2A_voov = 0.0
    I2A_voov += 0.5*np.einsum('mnef,afin->amie',vA['oovv'],t2a,optimize=True)
    I2A_voov += np.einsum('mnef,afin->amie',vB['oovv'],t2b,optimize=True)
    I2A_voov += H2A['voov'] 

    I2A_oooo = 0.0
    I2A_oooo += 0.5*np.einsum('mnef,efij->mnij',vA['oovv'],t2a,optimize=True)
    I2A_oooo += H2A['oooo'] 

    I2B_voov = 0.0
    I2B_voov += 0.5*np.einsum('mnef,afin->amie',vC['oovv'],t2b,optimize=True)
    I2B_voov += H2B['voov']

    X2A = 0.25*vA['vvoo']
    X2A -= 0.5*np.einsum('amij,bm->abij',H2A['vooo'],t1a,optimize=True)
    X2A += 0.5*np.einsum('abie,ej->abij',H2A['vvov'],t1a,optimize=True)
    X2A += 0.5*np.einsum('ae,ebij->abij',I1A_vv,t2a,optimize=True)
    X2A -= 0.5*np.einsum('mi,abmj->abij',I1A_oo,t2a,optimize=True)
    X2A += np.einsum('amie,ebmj->abij',I2A_voov,t2a,optimize=True)
    X2A += np.einsum('amie,bejm->abij',I2B_voov,t2b,optimize=True)
    X2A += 0.125*np.einsum('abef,efij->abij',H2A['vvvv'],t2a,optimize=True)
    X2A += 0.125*np.einsum('mnij,abmn->abij',I2A_oooo,t2a,optimize=True)
    X2A += 0.25*np.einsum('me,abeijm->abij',H1A['ov'],t3a,optimize=True)
    X2A += 0.25*np.einsum('me,abeijm->abij',H1B['ov'],t3b,optimize=True)
    X2A -= 0.5*np.einsum('mnif,abfmjn->abij',H2B['ooov'],t3b,optimize=True)
    X2A -= 0.25*np.einsum('mnif,abfmjn->abij',H2A['ooov'],t3a,optimize=True)
    X2A += 0.25*np.einsum('anef,ebfijn->abij',H2A['vovv'],t3a,optimize=True)
    X2A += 0.5*np.einsum('anef,ebfijn->abij',H2B['vovv'],t3b,optimize=True)

    t2a, resid = cc_loops2.cc_loops2.update_t2a(t2a,X2A,fA['oo'],fA['vv'],shift)

    cc_t['t2a'] = t2a
    return cc_t, resid.flatten()

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
        New cluster amplitudes T1, T2, T3
    """
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
    t3b = cc_t['t3b']
    t3c = cc_t['t3c']

    # intermediates
    I1A_vv = 0.0
    I1A_vv -= 0.5*np.einsum('mnef,afmn->ae',vA['oovv'],t2a,optimize=True)
    I1A_vv -= np.einsum('mnef,afmn->ae',vB['oovv'],t2b,optimize=True)
    I1A_vv += H1A['vv']

    I1B_vv = 0.0
    I1B_vv -= np.einsum('nmfe,fbnm->be',vB['oovv'],t2b,optimize=True)
    I1B_vv -= 0.5*np.einsum('mnef,fbnm->be',vC['oovv'],t2c,optimize=True)
    I1B_vv += H1B['vv']

    I1A_oo = 0.0
    I1A_oo += 0.5*np.einsum('mnef,efin->mi',vA['oovv'],t2a,optimize=True)
    I1A_oo += np.einsum('mnef,efin->mi',vB['oovv'],t2b,optimize=True)
    I1A_oo += H1A['oo']

    I1B_oo = 0.0
    I1B_oo += np.einsum('nmfe,fenj->mj',vB['oovv'],t2b,optimize=True)
    I1B_oo += 0.5*np.einsum('mnef,efjn->mj',vC['oovv'],t2c,optimize=True)
    I1B_oo += H1B['oo']
    
    I2A_voov = 0.0
    I2A_voov += np.einsum('mnef,aeim->anif',vA['oovv'],t2a,optimize=True)
    I2A_voov += np.einsum('nmfe,aeim->anif',vB['oovv'],t2b,optimize=True)
    I2A_voov += H2A['voov']

    I2B_voov = 0.0
    I2B_voov += np.einsum('mnef,aeim->anif',vB['oovv'],t2a,optimize=True)
    I2B_voov += np.einsum('mnef,aeim->anif',vC['oovv'],t2b,optimize=True)
    I2B_voov += H2B['voov']

    I2B_oooo = 0.0
    I2B_oooo += np.einsum('mnef,efij->mnij',vB['oovv'],t2b,optimize=True)
    I2B_oooo += H2B['oooo']

    I2B_vovo = 0.0
    I2B_vovo -= np.einsum('mnef,afmj->anej',vB['oovv'],t2b,optimize=True)
    I2B_vovo += H2B['vovo']

    X2B = 0.0
    X2B += vB['vvoo']
    X2B -= np.einsum('mbij,am->abij',H2B['ovoo'],t1a,optimize=True)
    X2B -= np.einsum('amij,bm->abij',H2B['vooo'],t1b,optimize=True)
    X2B += np.einsum('abej,ei->abij',H2B['vvvo'],t1a,optimize=True)
    X2B += np.einsum('abie,ej->abij',H2B['vvov'],t1b,optimize=True)
    X2B += np.einsum('ae,ebij->abij',I1A_vv,t2b,optimize=True)
    X2B += np.einsum('be,aeij->abij',I1B_vv,t2b,optimize=True)
    X2B -= np.einsum('mi,abmj->abij',I1A_oo,t2b,optimize=True)
    X2B -= np.einsum('mj,abim->abij',I1B_oo,t2b,optimize=True)
    X2B += np.einsum('amie,ebmj->abij',I2A_voov,t2b,optimize=True)
    X2B += np.einsum('amie,ebmj->abij',I2B_voov,t2c,optimize=True)
    X2B += np.einsum('mbej,aeim->abij',H2B['ovvo'],t2a,optimize=True)
    X2B += np.einsum('bmje,aeim->abij',H2C['voov'],t2b,optimize=True)
    X2B -= np.einsum('mbie,aemj->abij',H2B['ovov'],t2b,optimize=True)
    X2B -= np.einsum('amej,ebim->abij',I2B_vovo,t2b,optimize=True)
    X2B += np.einsum('mnij,abmn->abij',I2B_oooo,t2b,optimize=True)
    X2B += np.einsum('abef,efij->abij',H2B['vvvv'],t2b,optimize=True)
    X2B -= 0.5*np.einsum('mnif,afbmnj->abij',H2A['ooov'],t3b,optimize=True)
    X2B -= np.einsum('nmfj,afbinm->abij',H2B['oovo'],t3b,optimize=True)
    X2B -= 0.5*np.einsum('mnjf,afbinm->abij',H2C['ooov'],t3c,optimize=True)
    X2B -= np.einsum('mnif,afbmnj->abij',H2B['ooov'],t3c,optimize=True)
    X2B += 0.5*np.einsum('anef,efbinj->abij',H2A['vovv'],t3b,optimize=True)
    X2B += np.einsum('anef,efbinj->abij',H2B['vovv'],t3c,optimize=True)
    X2B += np.einsum('nbfe,afeinj->abij',H2B['ovvv'],t3b,optimize=True)
    X2B += 0.5*np.einsum('bnef,afeinj->abij',H2C['vovv'],t3c,optimize=True)
    X2B += np.einsum('me,aebimj->abij',H1A['ov'],t3b,optimize=True)
    X2B += np.einsum('me,aebimj->abij',H1B['ov'],t3c,optimize=True)

    t2b, flatten = cc_loops2.cc_loops2.update_t2b(t2b,X2B,fA['oo'],fA['vv'],fB['oo'],fB['vv'],shift)

    cc_t['t2b'] = t2b
    return cc_t, resid.flatten()

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
        New cluster amplitudes T1, T2, T3
    """
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    fB = ints['fB']
    t1b = cc_t['t1b']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']
    t3c = cc_t['t3c']
    t3d = cc_t['t3d']

    I1B_oo = 0.0
    I1B_oo += 0.5*np.einsum('mnef,efin->mi',vC['oovv'],t2c,optimize=True)
    I1B_oo += np.einsum('nmfe,feni->mi',vB['oovv'],t2b,optimize=True)
    I1B_oo += H1B['oo']

    I1B_vv = 0.0
    I1B_vv -= 0.5*np.einsum('mnef,afmn->ae',vC['oovv'],t2c,optimize=True)
    I1B_vv -= np.einsum('nmfe,fanm->ae',vB['oovv'],t2b,optimize=True)
    I1B_vv += H1B['vv']
             
    I2C_oooo = 0.0
    I2C_oooo += 0.5*np.einsum('mnef,efij->mnij',vC['oovv'],t2c,optimize=True)
    I2C_oooo += H2C['oooo']

    I2B_ovvo = 0.0
    I2B_ovvo += np.einsum('mnef,afin->maei',vB['oovv'],t2c,optimize=True)
    I2B_ovvo += 0.5*np.einsum('mnef,fani->maei',vA['oovv'],t2b,optimize=True)
    I2B_ovvo += H2B['ovvo']

    I2C_voov = 0.0
    I2C_voov += 0.5*np.einsum('mnef,afin->amie',vC['oovv'],t2c,optimize=True)
    I2C_voov += H2C['voov']
    
    X2C = 0.25*vC['vvoo']
    X2C -= 0.5*np.einsum('mbij,am->abij',H2C['ovoo'],t1b,optimize=True)
    X2C += 0.5*np.einsum('abej,ei->abij',H2C['vvvo'],t1b,optimize=True)
    X2C += 0.5*np.einsum('ae,ebij->abij',I1B_vv,t2c,optimize=True)
    X2C -= 0.5*np.einsum('mi,abmj->abij',I1B_oo,t2c,optimize=True)
    X2C += np.einsum('amie,ebmj->abij',I2C_voov,t2c,optimize=True)
    X2C += np.einsum('maei,ebmj->abij',I2B_ovvo,t2b,optimize=True)
    X2C += 0.125*np.einsum('abef,efij->abij',H2C['vvvv'],t2c,optimize=True)
    X2C += 0.125*np.einsum('mnij,abmn->abij',I2C_oooo,t2c,optimize=True)
    X2C += 0.25*np.einsum('me,eabmij->abij',H1A['ov'],t3c,optimize=True)
    X2C += 0.25*np.einsum('me,abeijm->abij',H1B['ov'],t3d,optimize=True)
    X2C += 0.25*np.einsum('anef,ebfijn->abij',H2C['vovv'],t3d,optimize=True)
    X2C += 0.5*np.einsum('nafe,febnij->abij',H2B['ovvv'],t3c,optimize=True)
    X2C -= 0.25*np.einsum('mnif,abfmjn->abij',H2C['ooov'],t3d,optimize=True)
    X2C -= 0.5*np.einsum('nmfi,fabnmj->abij',H2B['oovo'],t3c,optimize=True)

    t2c, resid = cc_loops2.cc_loops2.update_t2c(t2c,X2C,fB['oo'],fB['vv'],shift)

    cc_t['t2c'] = t2c
    return cc_t, resid.flatten()

#@profile
def update_t3a(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift):
    """Update t3a amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3))_C|0>.

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
        New cluster amplitudes T1, T2, T3
    """
    vA = ints['vA']
    vB = ints['vB']
    fA = ints['fA']
    t2a = cc_t['t2a']
    t3a = cc_t['t3a']
    t3b = cc_t['t3b']
    # new t3a vector
    t3a_new = np.zeros((sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_a']))
    # Intermediates
    I2A_vvov = 0.0
    I2A_vvov -= 0.5*np.einsum('mnef,abfimn->abie',vA['oovv'],t3a,optimize=True)
    I2A_vvov -= np.einsum('mnef,abfimn->abie',vB['oovv'],t3b,optimize=True)
    I2A_vvov += np.einsum('me,abim->abie',H1A['ov'],t2a,optimize=True)
    I2A_vvov += H2A['vvov']
    I2A_vooo = 0.0
    I2A_vooo += 0.5*np.einsum('mnef,aefijn->amij',vA['oovv'],t3a,optimize=True)
    I2A_vooo += np.einsum('mnef,aefijn->amij',vB['oovv'],t3b,optimize=True)
    I2A_vooo += H2A['vooo']  
    # update loop
    for i in range(sys['Nocc_a']):
        for j in range(i+1,sys['Nocc_a']):
            for k in range(j+1,sys['Nocc_a']):

                X3A = -0.5*np.einsum('am,bcm->abc',I2A_vooo[:,:,i,j],t2a[:,:,:,k],optimize=True)
                X3A += 0.5*np.einsum('am,bcm->abc',I2A_vooo[:,:,k,j],t2a[:,:,:,i],optimize=True)
                X3A += 0.5*np.einsum('am,bcm->abc',I2A_vooo[:,:,i,k],t2a[:,:,:,j],optimize=True)

                X3A += 0.5*np.einsum('abe,ec->abc',I2A_vvov[:,:,i,:],t2a[:,:,j,k],optimize=True)
                X3A -= 0.5*np.einsum('abe,ec->abc',I2A_vvov[:,:,j,:],t2a[:,:,i,k],optimize=True)
                X3A -= 0.5*np.einsum('abe,ec->abc',I2A_vvov[:,:,k,:],t2a[:,:,j,i],optimize=True)

                X3A -= (1.0/6.0)*np.einsum('m,abcm->abc',H1A['oo'][:,k],t3a[:,:,:,i,j,:],optimize=True)
                X3A += (1.0/6.0)*np.einsum('m,abcm->abc',H1A['oo'][:,i],t3a[:,:,:,k,j,:],optimize=True)
                X3A += (1.0/6.0)*np.einsum('m,abcm->abc',H1A['oo'][:,j],t3a[:,:,:,i,k,:],optimize=True)

                X3A += 0.5*np.einsum('ce,abe->abc',H1A['vv'],t3a[:,:,:,i,j,k],optimize=True)

                X3A += (1.0/12.0)*np.einsum('mn,abcmn->abc',H2A['oooo'][:,:,i,j],t3a[:,:,:,:,:,k],optimize=True)
                X3A -= (1.0/12.0)*np.einsum('mn,abcmn->abc',H2A['oooo'][:,:,k,j],t3a[:,:,:,:,:,i],optimize=True)
                X3A -= (1.0/12.0)*np.einsum('mn,abcmn->abc',H2A['oooo'][:,:,i,k],t3a[:,:,:,:,:,j],optimize=True)

                X3A += 0.25*np.einsum('abef,efc->abc',H2A['vvvv'],t3a[:,:,:,i,j,k],optimize=True)

                X3A += 0.5*np.einsum('cme,abem->abc',H2A['voov'][:,:,k,:],t3a[:,:,:,i,j,:],optimize=True)
                X3A -= 0.5*np.einsum('cme,abem->abc',H2A['voov'][:,:,i,:],t3a[:,:,:,k,j,:],optimize=True)
                X3A -= 0.5*np.einsum('cme,abem->abc',H2A['voov'][:,:,j,:],t3a[:,:,:,i,k,:],optimize=True)

                X3A += 0.5*np.einsum('cme,abem->abc',H2B['voov'][:,:,k,:],t3b[:,:,:,i,j,:],optimize=True)
                X3A -= 0.5*np.einsum('cme,abem->abc',H2B['voov'][:,:,i,:],t3b[:,:,:,k,j,:],optimize=True)
                X3A -= 0.5*np.einsum('cme,abem->abc',H2B['voov'][:,:,j,:],t3b[:,:,:,i,k,:],optimize=True)

                for a in range(sys['Nunocc_a']):
                    for b in range(a+1,sys['Nunocc_a']):
                        for c in range(b+1,sys['Nunocc_a']):
                            denom = fA['oo'][i,i]+fA['oo'][j,j]+fA['oo'][k,k]-fA['vv'][a,a]-fA['vv'][b,b]-fA['vv'][c,c]
                            val = X3A[a,b,c]-X3A[b,a,c]-X3A[a,c,b]-X3A[c,b,a]+X3A[b,c,a]+X3A[c,a,b]
                            t3a_new[a,b,c,i,j,k] = t3a[a,b,c,i,j,k] + val/(denom-shift)
                            t3a_new[b,c,a,i,j,k] = t3a_new[a,b,c,i,j,k]
                            t3a_new[c,a,b,i,j,k] = t3a_new[a,b,c,i,j,k]
                            t3a_new[a,c,b,i,j,k] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[b,a,c,i,j,k] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[c,b,a,i,j,k] = -t3a_new[a,b,c,i,j,k]

                            t3a_new[a,b,c,j,k,i] = t3a_new[a,b,c,i,j,k]
                            t3a_new[b,c,a,j,k,i] = t3a_new[a,b,c,i,j,k]
                            t3a_new[c,a,b,j,k,i] = t3a_new[a,b,c,i,j,k]
                            t3a_new[a,c,b,j,k,i] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[b,a,c,j,k,i] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[c,b,a,j,k,i] = -t3a_new[a,b,c,i,j,k]
                            
                            t3a_new[a,b,c,k,i,j] = t3a_new[a,b,c,i,j,k]
                            t3a_new[b,c,a,k,i,j] = t3a_new[a,b,c,i,j,k]
                            t3a_new[c,a,b,k,i,j] = t3a_new[a,b,c,i,j,k]
                            t3a_new[a,c,b,k,i,j] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[b,a,c,k,i,j] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[c,b,a,k,i,j] = -t3a_new[a,b,c,i,j,k]
                            
                            t3a_new[a,b,c,j,i,k] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[b,c,a,j,i,k] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[c,a,b,j,i,k] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[a,c,b,j,i,k] = t3a_new[a,b,c,i,j,k]
                            t3a_new[b,a,c,j,i,k] = t3a_new[a,b,c,i,j,k]
                            t3a_new[c,b,a,j,i,k] = t3a_new[a,b,c,i,j,k]

                            t3a_new[a,b,c,k,j,i] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[b,c,a,k,j,i] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[c,a,b,k,j,i] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[a,c,b,k,j,i] = t3a_new[a,b,c,i,j,k]
                            t3a_new[b,a,c,k,j,i] = t3a_new[a,b,c,i,j,k]
                            t3a_new[c,b,a,k,j,i] = t3a_new[a,b,c,i,j,k]

                            t3a_new[a,b,c,i,k,j] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[b,c,a,i,k,j] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[c,a,b,i,k,j] = -t3a_new[a,b,c,i,j,k]
                            t3a_new[a,c,b,i,k,j] = t3a_new[a,b,c,i,j,k]
                            t3a_new[b,a,c,i,k,j] = t3a_new[a,b,c,i,j,k]
                            t3a_new[c,b,a,i,k,j] = t3a_new[a,b,c,i,j,k]
    cc_t['t3a'] = t3a_new
    return cc_t

#@profile
def update_t3b(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift):
    """Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N e^(T1+T2+T3))_C|0>.

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
        New cluster amplitudes T1, T2, T3
    """
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    fA = ints['fA']
    fB = ints['fB']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t3a = cc_t['t3a']
    t3b = cc_t['t3b']
    t3c = cc_t['t3c']
    # New cluster amplitudes
    t3b_new = np.zeros((sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_b']))
    # MM23B + (VT3)_C intermediates
    I2A_vvov = 0.0
    I2A_vvov += -0.5*np.einsum('mnef,abfimn->abie',vA['oovv'],t3a,optimize=True)
    I2A_vvov += -np.einsum('mnef,abfimn->abie',vB['oovv'],t3b,optimize=True)
    I2A_vvov += H2A['vvov']
    I2A_vooo = 0.0
    I2A_vooo += 0.5*np.einsum('mnef,aefijn->amij',vA['oovv'],t3a,optimize=True)
    I2A_vooo += np.einsum('mnef,aefijn->amij',vB['oovv'],t3b,optimize=True)
    I2A_vooo += H2A['vooo']
    I2A_vooo += -np.einsum('me,aeij->amij',H1A['ov'],t2a,optimize=True)
    I2B_vvvo = 0.0
    I2B_vvvo += -0.5*np.einsum('mnef,afbmnj->abej',vA['oovv'],t3b,optimize=True)
    I2B_vvvo += -np.einsum('mnef,afbmnj->abej',vB['oovv'],t3c,optimize=True)
    I2B_vvvo += H2B['vvvo']
    I2B_ovoo = 0.0
    I2B_ovoo += 0.5*np.einsum('mnef,efbinj->mbij',vA['oovv'],t3b,optimize=True)
    I2B_ovoo += np.einsum('mnef,efbinj->mbij',vB['oovv'],t3c,optimize=True)
    I2B_ovoo += H2B['ovoo']
    I2B_ovoo += -np.einsum('me,ecjk->mcjk',H1A['ov'],t2b,optimize=True)
    I2B_vvov = 0.0      
    I2B_vvov += -np.einsum('nmfe,afbinm->abie',vB['oovv'],t3b,optimize=True)
    I2B_vvov += -0.5*np.einsum('nmfe,afbinm->abie',vC['oovv'],t3c,optimize=True)
    I2B_vvov += H2B['vvov']
    I2B_vooo = 0.0         
    I2B_vooo += np.einsum('nmfe,afeinj->amij',vB['oovv'],t3b,optimize=True)
    I2B_vooo += 0.5*np.einsum('nmfe,afeinj->amij',vC['oovv'],t3c,optimize=True)
    I2B_vooo += H2B['vooo']
    I2B_vooo += -np.einsum('me,aeik->amik',H1B['ov'],t2b,optimize=True) 
    # Update loop
    for i in range(sys['Nocc_a']):
        for j in range(i+1,sys['Nocc_a']):
            for k in range(sys['Nocc_b']):
                X3B = np.einsum('bce,ae->abc',I2B_vvvo[:,:,:,k],t2a[:,:,i,j],optimize=True)
   
                X3B -= 0.5*np.einsum('mc,abm->abc',I2B_ovoo[:,:,j,k],t2a[:,:,i,:],optimize=True)
                X3B += 0.5*np.einsum('mc,abm->abc',I2B_ovoo[:,:,i,k],t2a[:,:,j,:],optimize=True)

                X3B += np.einsum('ace,be->abc',I2B_vvov[:,:,i,:],t2b[:,:,j,k],optimize=True)
                X3B -= np.einsum('ace,be->abc',I2B_vvov[:,:,j,:],t2b[:,:,i,k],optimize=True)

                X3B -= np.einsum('am,bcm->abc',I2B_vooo[:,:,i,k],t2b[:,:,j,:],optimize=True)
                X3B += np.einsum('am,bcm->abc',I2B_vooo[:,:,j,k],t2b[:,:,i,:],optimize=True)

                X3B += 0.5*np.einsum('abe,ec->abc',I2A_vvov[:,:,i,:],t2b[:,:,j,k],optimize=True)
                X3B -= 0.5*np.einsum('abe,ec->abc',I2A_vvov[:,:,j,:],t2b[:,:,i,k],optimize=True)

                X3B -= np.einsum('am,bcm->abc',I2A_vooo[:,:,i,j],t2b[:,:,:,k],optimize=True)

                # (HBar*T3)_C
                X3B -= 0.5*np.einsum('m,abcm->abc',H1A['oo'][:,i],t3b[:,:,:,:,j,k],optimize=True)
                X3B += 0.5*np.einsum('m,abcm->abc',H1A['oo'][:,j],t3b[:,:,:,:,i,k],optimize=True)

                X3B -= 0.5*np.einsum('m,abcm->abc',H1B['oo'][:,k],t3b[:,:,:,i,j,:],optimize=True)
                
                X3B += np.einsum('ae,ebc->abc',H1A['vv'],t3b[:,:,:,i,j,k],optimize=True)

                X3B += 0.5*np.einsum('ce,abe->abc',H1B['vv'],t3b[:,:,:,i,j,k],optimize=True)

                X3B += 0.25*np.einsum('mn,abcmn->abc',H2A['oooo'][:,:,i,j],t3b[:,:,:,:,:,k],optimize=True)

                X3B += 0.5*np.einsum('mn,abcmn->abc',H2B['oooo'][:,:,j,k],t3b[:,:,:,i,:,:],optimize=True)
                X3B -= 0.5*np.einsum('mn,abcmn->abc',H2B['oooo'][:,:,i,k],t3b[:,:,:,j,:,:],optimize=True)

                X3B += 0.25*np.einsum('abef,efc->abc',H2A['vvvv'],t3b[:,:,:,i,j,k],optimize=True)

                X3B += np.einsum('bcef,aef->abc',H2B['vvvv'],t3b[:,:,:,i,j,k],optimize=True)

                X3B += np.einsum('ame,ebcm->abc',H2A['voov'][:,:,i,:],t3b[:,:,:,:,j,k],optimize=True)
                X3B -= np.einsum('ame,ebcm->abc',H2A['voov'][:,:,j,:],t3b[:,:,:,:,i,k],optimize=True)

                X3B += np.einsum('ame,becm->abc',H2B['voov'][:,:,i,:],t3c[:,:,:,j,:,k],optimize=True)
                X3B -= np.einsum('ame,becm->abc',H2B['voov'][:,:,j,:],t3c[:,:,:,i,:,k],optimize=True)

                X3B += 0.5*np.einsum('mce,abem->abc',H2B['ovvo'][:,:,:,k],t3a[:,:,:,i,j,:],optimize=True)

                X3B += 0.5*np.einsum('cme,abem->abc',H2C['voov'][:,:,k,:],t3b[:,:,:,i,j,:],optimize=True)

                X3B -= np.einsum('ame,ebcm->abc',H2B['vovo'][:,:,:,k],t3b[:,:,:,i,j,:],optimize=True)

                X3B -= 0.5*np.einsum('mce,abem->abc',H2B['ovov'][:,:,i,:],t3b[:,:,:,:,j,k],optimize=True)
                X3B += 0.5*np.einsum('mce,abem->abc',H2B['ovov'][:,:,j,:],t3b[:,:,:,:,i,k],optimize=True)

                for a in range(sys['Nunocc_a']):
                    for b in range(a+1,sys['Nunocc_a']):
                        for c in range(sys['Nunocc_b']):
                            denom = fA['oo'][i,i]+fA['oo'][j,j]+fB['oo'][k,k]-fA['vv'][a,a]-fA['vv'][b,b]-fB['vv'][c,c]
                            val = X3B[a,b,c] - X3B[b,a,c]
                            t3b_new[a,b,c,i,j,k] = t3b[a,b,c,i,j,k] + val/(denom-shift)
                            t3b_new[b,a,c,i,j,k] = -t3b_new[a,b,c,i,j,k]
                            t3b_new[a,b,c,j,i,k] = -t3b_new[a,b,c,i,j,k]
                            t3b_new[b,a,c,j,i,k] = t3b_new[a,b,c,i,j,k]
    cc_t['t3b'] = t3b_new
    return cc_t

#@profile
def update_t3c(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift):
    """Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N e^(T1+T2+T3))_C|0>.

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
        New cluster amplitudes T1, T2, T3
    """
    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    fA = ints['fA']
    fB = ints['fB']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']
    t3b = cc_t['t3b']
    t3c = cc_t['t3c']
    t3d = cc_t['t3d']
    # New cluster amplitudes
    t3c_new = np.zeros((sys['Nunocc_a'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b'],sys['Nocc_b']))
    # Intermediates
    I2B_vvvo = 0.0
    I2B_vvvo += -0.5*np.einsum('mnef,afbmnj->abej',vA['oovv'],t3b,optimize=True)
    I2B_vvvo += -np.einsum('mnef,afbmnj->abej',vB['oovv'],t3c,optimize=True)
    I2B_vvvo += H2B['vvvo']
    I2B_ovoo = 0.0       
    I2B_ovoo += 0.5*np.einsum('mnef,efbinj->mbij',vA['oovv'],t3b,optimize=True)
    I2B_ovoo += np.einsum('mnef,efbinj->mbij',vB['oovv'],t3c,optimize=True)
    I2B_ovoo += H2B['ovoo']
    I2B_ovoo -= np.einsum('me,ebij->mbij',H1A['ov'],t2b,optimize=True)
    I2B_vvov = 0.0         
    I2B_vvov += -np.einsum('nmfe,afbinm->abie',vB['oovv'],t3b,optimize=True)
    I2B_vvov += -0.5*np.einsum('nmfe,afbinm->abie',vC['oovv'],t3c,optimize=True)
    I2B_vvov += H2B['vvov']
    I2B_vooo = 0.0     
    I2B_vooo += np.einsum('nmfe,afeinj->amij',vB['oovv'],t3b,optimize=True)
    I2B_vooo += 0.5*np.einsum('nmfe,afeinj->amij',vC['oovv'],t3c,optimize=True)
    I2B_vooo += H2B['vooo']
    I2B_vooo -= np.einsum('me,aeij->amij',H1B['ov'],t2b,optimize=True)
    I2C_vvov = 0.0
    I2C_vvov += -0.5*np.einsum('mnef,abfimn->abie',vC['oovv'],t3d,optimize=True)
    I2C_vvov += -np.einsum('nmfe,fabnim->abie',vB['oovv'],t3c,optimize=True)
    I2C_vvov += H2C['vvov']
    I2C_vooo = 0.0             
    I2C_vooo += np.einsum('nmfe,faenij->amij',vB['oovv'],t3c,optimize=True)
    I2C_vooo += 0.5*np.einsum('mnef,aefijn->amij',vC['oovv'],t3d,optimize=True)
    I2C_vooo += H2C['vooo']
    I2C_vooo -= np.einsum('me,cekj->cmkj',H1B['ov'],t2c,optimize=True)
    # Update loop
    for i in range(sys['Nocc_a']):
        for j in range(sys['Nocc_b']):
            for k in range(j+1,sys['Nocc_b']):
                # MM(2,3)C
                X3C = np.einsum('abe,ec->abc',I2B_vvov[:,:,i,:],t2c[:,:,j,k],optimize=True)

                X3C -= 0.5*np.einsum('am,bcm->abc',I2B_vooo[:,:,i,j],t2c[:,:,:,k],optimize=True)
                X3C += 0.5*np.einsum('am,bcm->abc',I2B_vooo[:,:,i,k],t2c[:,:,:,j],optimize=True)

                X3C += 0.5*np.einsum('cbe,ae->abc',I2C_vvov[:,:,k,:],t2b[:,:,i,j],optimize=True)
                X3C -= 0.5*np.einsum('cbe,ae->abc',I2C_vvov[:,:,j,:],t2b[:,:,i,k],optimize=True)

                X3C -= np.einsum('cm,abm->abc',I2C_vooo[:,:,k,j],t2b[:,:,i,:],optimize=True)

                X3C += np.einsum('abe,ec->abc',I2B_vvvo[:,:,:,j],t2b[:,:,i,k],optimize=True)
                X3C -= np.einsum('abe,ec->abc',I2B_vvvo[:,:,:,k],t2b[:,:,i,j],optimize=True)
    
                X3C -= np.einsum('mb,acm->abc',I2B_ovoo[:,:,i,j],t2b[:,:,:,k],optimize=True)
                X3C += np.einsum('mb,acm->abc',I2B_ovoo[:,:,i,k],t2b[:,:,:,j],optimize=True)

                # (HBar*T3)_C
                X3C -= 0.5*np.einsum('m,abcm->abc',H1A['oo'][:,i],t3c[:,:,:,:,j,k],optimize=True)

                X3C -= 0.5*np.einsum('m,abcm->abc',H1B['oo'][:,j],t3c[:,:,:,i,:,k],optimize=True)
                X3C += 0.5*np.einsum('m,abcm->abc',H1B['oo'][:,k],t3c[:,:,:,i,:,j],optimize=True)

                X3C += 0.5*np.einsum('ae,ebc->abc',H1A['vv'],t3c[:,:,:,i,j,k],optimize=True)

                X3C += np.einsum('be,aec->abc',H1B['vv'],t3c[:,:,:,i,j,k],optimize=True)

                X3C += 0.25*np.einsum('mn,abcmn->abc',H2C['oooo'][:,:,j,k],t3c[:,:,:,i,:,:],optimize=True)

                X3C += 0.5*np.einsum('mn,abcmn->abc',H2B['oooo'][:,:,i,j],t3c[:,:,:,:,:,k],optimize=True)
                X3C -= 0.5*np.einsum('mn,abcmn->abc',H2B['oooo'][:,:,i,k],t3c[:,:,:,:,:,j],optimize=True)

                X3C += 0.25*np.einsum('bcef,aef->abc',H2C['vvvv'],t3c[:,:,:,i,j,k],optimize=True)

                X3C += np.einsum('abef,efc->abc',H2B['vvvv'],t3c[:,:,:,i,j,k],optimize=True)

                X3C += 0.5*np.einsum('ame,ebcm->abc',H2A['voov'][:,:,i,:],t3c[:,:,:,:,j,k],optimize=True)

                X3C += 0.5*np.einsum('ame,ebcm->abc',H2B['voov'][:,:,i,:],t3d[:,:,:,:,j,k],optimize=True)

                X3C += np.einsum('mbe,aecm->abc',H2B['ovvo'][:,:,:,j],t3b[:,:,:,i,:,k],optimize=True)
                X3C -= np.einsum('mbe,aecm->abc',H2B['ovvo'][:,:,:,k],t3b[:,:,:,i,:,j],optimize=True)

                X3C += np.einsum('bme,aecm->abc',H2C['voov'][:,:,j,:],t3c[:,:,:,i,:,k],optimize=True)
                X3C -= np.einsum('bme,aecm->abc',H2C['voov'][:,:,k,:],t3c[:,:,:,i,:,j],optimize=True)

                X3C -= np.einsum('mbe,aecm->abc',H2B['ovov'][:,:,i,:],t3c[:,:,:,:,j,k],optimize=True)

                X3C -= 0.5*np.einsum('ame,ebcm->abc',H2B['vovo'][:,:,:,j],t3c[:,:,:,i,:,k],optimize=True)
                X3C += 0.5*np.einsum('ame,ebcm->abc',H2B['vovo'][:,:,:,k],t3c[:,:,:,i,:,j],optimize=True)

                for a in range(sys['Nunocc_a']):
                    for b in range(sys['Nunocc_b']):
                        for c in range(b+1,sys['Nunocc_b']):
                            denom = fA['oo'][i,i]+fB['oo'][j,j]+fB['oo'][k,k]-fA['vv'][a,a]-fB['vv'][b,b]-fB['vv'][c,c]
                            val = X3C[a,b,c] - X3C[a,c,b]
                            t3c_new[a,b,c,i,j,k] = t3c[a,b,c,i,j,k] + val/(denom-shift)
                            t3c_new[a,c,b,i,j,k] = -t3c_new[a,b,c,i,j,k]
                            t3c_new[a,b,c,i,k,j] = -t3c_new[a,b,c,i,j,k]
                            t3c_new[a,c,b,i,k,j] = t3c_new[a,b,c,i,j,k]
    cc_t['t3c'] = t3c_new
    return cc_t

#@profile
def update_t3d(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift):
    """Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N e^(T1+T2+T3))_C|0>.

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
        New cluster amplitudes T1, T2, T3
    """
    vB = ints['vB']
    vC = ints['vC']
    fB = ints['fB']
    t2c = cc_t['t2c']
    t3c = cc_t['t3c']
    t3d = cc_t['t3d']
    # New cluster amplitudes
    t3d_new = np.zeros((sys['Nunocc_b'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b'],sys['Nocc_b']))
    # Intermediates
    I2C_vvov = 0.0
    I2C_vvov -= 0.5*np.einsum('mnef,abfimn->abie',vC['oovv'],t3d,optimize=True)
    I2C_vvov -= np.einsum('nmfe,fabnim->abie',vB['oovv'],t3c,optimize=True)
    I2C_vvov += np.einsum('me,abim->abie',H1B['ov'],t2c,optimize=True)
    I2C_vvov += H2C['vvov']
    I2C_vooo = 0.0
    I2C_vooo += 0.5*np.einsum('mnef,aefijn->amij',vC['oovv'],t3d,optimize=True)
    I2C_vooo += np.einsum('nmfe,faenij->amij',vB['oovv'],t3c,optimize=True)
    I2C_vooo += H2C['vooo']   
    # Update loop
    for i in range(sys['Nocc_b']):
        for j in range(i+1,sys['Nocc_b']):
            for k in range(j+1,sys['Nocc_b']):
                # MM(2,3)D
                X3D = -0.5*np.einsum('am,bcm->abc',I2C_vooo[:,:,i,j],t2c[:,:,:,k],optimize=True)
                X3D += 0.5*np.einsum('am,bcm->abc',I2C_vooo[:,:,k,j],t2c[:,:,:,i],optimize=True)
                X3D += 0.5*np.einsum('am,bcm->abc',I2C_vooo[:,:,i,k],t2c[:,:,:,j],optimize=True)

                X3D += 0.5*np.einsum('abe,ec->abc',I2C_vvov[:,:,i,:],t2c[:,:,j,k],optimize=True)
                X3D -= 0.5*np.einsum('abe,ec->abc',I2C_vvov[:,:,j,:],t2c[:,:,i,k],optimize=True)
                X3D -= 0.5*np.einsum('abe,ec->abc',I2C_vvov[:,:,k,:],t2c[:,:,j,i],optimize=True)

                # (HBar*T3)_C
                X3D -= (1.0/6.0)*np.einsum('m,abcm->abc',H1B['oo'][:,k],t3d[:,:,:,i,j,:],optimize=True)
                X3D += (1.0/6.0)*np.einsum('m,abcm->abc',H1B['oo'][:,i],t3d[:,:,:,k,j,:],optimize=True)
                X3D += (1.0/6.0)*np.einsum('m,abcm->abc',H1B['oo'][:,j],t3d[:,:,:,i,k,:],optimize=True)

                X3D += 0.5*np.einsum('ce,abe->abc',H1B['vv'],t3d[:,:,:,i,j,k],optimize=True)

                X3D += (1.0/12.0)*np.einsum('mn,abcmn->abc',H2C['oooo'][:,:,i,j],t3d[:,:,:,:,:,k],optimize=True)
                X3D -= (1.0/12.0)*np.einsum('mn,abcmn->abc',H2C['oooo'][:,:,k,j],t3d[:,:,:,:,:,i],optimize=True)
                X3D -= (1.0/12.0)*np.einsum('mn,abcmn->abc',H2C['oooo'][:,:,i,k],t3d[:,:,:,:,:,j],optimize=True)

                X3D += 0.25*np.einsum('abef,efc->abc',H2C['vvvv'],t3d[:,:,:,i,j,k],optimize=True)

                X3D += 0.5*np.einsum('mae,ebcm->abc',H2B['ovvo'][:,:,:,i],t3c[:,:,:,:,j,k],optimize=True)
                X3D -= 0.5*np.einsum('mae,ebcm->abc',H2B['ovvo'][:,:,:,j],t3c[:,:,:,:,i,k],optimize=True)
                X3D -= 0.5*np.einsum('mae,ebcm->abc',H2B['ovvo'][:,:,:,k],t3c[:,:,:,:,j,i],optimize=True)

                X3D += 0.5*np.einsum('ame,ebcm->abc',H2C['voov'][:,:,i,:],t3d[:,:,:,:,j,k],optimize=True)
                X3D -= 0.5*np.einsum('ame,ebcm->abc',H2C['voov'][:,:,j,:],t3d[:,:,:,:,i,k],optimize=True)
                X3D -= 0.5*np.einsum('ame,ebcm->abc',H2C['voov'][:,:,k,:],t3d[:,:,:,:,j,i],optimize=True)

                for a in range(sys['Nunocc_b']):
                    for b in range(a+1,sys['Nunocc_b']):
                        for c in range(b+1,sys['Nunocc_b']):
                            denom = fB['oo'][i,i]+fB['oo'][j,j]+fB['oo'][k,k]-fB['vv'][a,a]-fB['vv'][b,b]-fB['vv'][c,c]
                            val = X3D[a,b,c]-X3D[b,a,c]-X3D[a,c,b]-X3D[c,b,a]+X3D[b,c,a]+X3D[c,a,b]
                            t3d_new[a,b,c,i,j,k] = t3d[a,b,c,i,j,k] + val/(denom-shift)
                            t3d_new[b,c,a,i,j,k] = t3d_new[a,b,c,i,j,k]
                            t3d_new[c,a,b,i,j,k] = t3d_new[a,b,c,i,j,k]
                            t3d_new[a,c,b,i,j,k] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[b,a,c,i,j,k] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[c,b,a,i,j,k] = -t3d_new[a,b,c,i,j,k]

                            t3d_new[a,b,c,j,k,i] = t3d_new[a,b,c,i,j,k]
                            t3d_new[b,c,a,j,k,i] = t3d_new[a,b,c,i,j,k]
                            t3d_new[c,a,b,j,k,i] = t3d_new[a,b,c,i,j,k]
                            t3d_new[a,c,b,j,k,i] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[b,a,c,j,k,i] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[c,b,a,j,k,i] = -t3d_new[a,b,c,i,j,k]
                            
                            t3d_new[a,b,c,k,i,j] = t3d_new[a,b,c,i,j,k]
                            t3d_new[b,c,a,k,i,j] = t3d_new[a,b,c,i,j,k]
                            t3d_new[c,a,b,k,i,j] = t3d_new[a,b,c,i,j,k]
                            t3d_new[a,c,b,k,i,j] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[b,a,c,k,i,j] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[c,b,a,k,i,j] = -t3d_new[a,b,c,i,j,k]
                            
                            t3d_new[a,b,c,j,i,k] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[b,c,a,j,i,k] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[c,a,b,j,i,k] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[a,c,b,j,i,k] = t3d_new[a,b,c,i,j,k]
                            t3d_new[b,a,c,j,i,k] = t3d_new[a,b,c,i,j,k]
                            t3d_new[c,b,a,j,i,k] = t3d_new[a,b,c,i,j,k]

                            t3d_new[a,b,c,k,j,i] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[b,c,a,k,j,i] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[c,a,b,k,j,i] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[a,c,b,k,j,i] = t3d_new[a,b,c,i,j,k]
                            t3d_new[b,a,c,k,j,i] = t3d_new[a,b,c,i,j,k]
                            t3d_new[c,b,a,k,j,i] = t3d_new[a,b,c,i,j,k]

                            t3d_new[a,b,c,i,k,j] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[b,c,a,i,k,j] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[c,a,b,i,k,j] = -t3d_new[a,b,c,i,j,k]
                            t3d_new[a,c,b,i,k,j] = t3d_new[a,b,c,i,j,k]
                            t3d_new[b,a,c,i,k,j] = t3d_new[a,b,c,i,j,k]
                            t3d_new[c,b,a,i,k,j] = t3d_new[a,b,c,i,j,k]
    cc_t['t3d'] = t3d_new
    return cc_t

def get_ccs_intermediates(cc_t,ints,sys):
    """Calculate the CCS-like similarity-transformed HBar intermediates (H_N e^T1)_C.

    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    -------
    H1* : dict
        One-body HBar similarity-transformed intermediates. Sorted by occ/unocc blocks.
    H2* : dict
        Two-body HBar similarity-transformed intermediates. Sorted by occ/unocc blocks.
    """

    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    fA = ints['fA']
    fB = ints['fB']
    t1a = cc_t['t1a']
    t1b = cc_t['t1b']


    h1A_ov = 0.0
    h1A_ov += fA['ov']
    h1A_ov += np.einsum('mnef,fn->me',vA['oovv'],t1a,optimize=True) 
    h1A_ov += np.einsum('mnef,fn->me',vB['oovv'],t1b,optimize=True) 

    h1B_ov = 0.0
    h1B_ov += fB['ov'] 
    h1B_ov += np.einsum('nmfe,fn->me',vB['oovv'],t1a,optimize=True) 
    h1B_ov += np.einsum('mnef,fn->me',vC['oovv'],t1b,optimize=True) 

    h1A_vv = 0.0
    h1A_vv += fA['vv']
    h1A_vv += np.einsum('anef,fn->ae',vA['vovv'],t1a,optimize=True)
    h1A_vv += np.einsum('anef,fn->ae',vB['vovv'],t1b,optimize=True)
    h1A_vv -= np.einsum('me,am->ae',h1A_ov,t1a,optimize=True)

    h1A_oo = 0.0
    h1A_oo += fA['oo']
    h1A_oo += np.einsum('mnif,fn->mi',vA['ooov'],t1a,optimize=True)
    h1A_oo += np.einsum('mnif,fn->mi',vB['ooov'],t1b,optimize=True)
    h1A_oo += np.einsum('me,ei->mi',h1A_ov,t1a,optimize=True)

    h1B_vv = 0.0
    h1B_vv += fB['vv']
    h1B_vv += np.einsum('anef,fn->ae',vC['vovv'],t1b,optimize=True)
    h1B_vv += np.einsum('nafe,fn->ae',vB['ovvv'],t1a,optimize=True)
    h1B_vv -= np.einsum('me,am->ae',h1B_ov,t1b,optimize=True)

    h1B_oo = 0.0
    h1B_oo += fB['oo']
    h1B_oo += np.einsum('mnif,fn->mi',vC['ooov'],t1b,optimize=True)
    h1B_oo += np.einsum('nmfi,fn->mi',vB['oovo'],t1a,optimize=True)
    h1B_oo += np.einsum('me,ei->mi',h1B_ov,t1b,optimize=True)

    h2A_oooo = 0.0
    h2A_oooo += vA['oooo']
    h2A_oooo += np.einsum('mnej,ei->mnij',vA['oovo'],t1a,optimize=True) # ij
    h2A_oooo -= np.einsum('mnei,ej->mnij',vA['oovo'],t1a,optimize=True) # ji
    h2A_oooo += np.einsum('mnef,ei,fj->mnij',vA['oovv'],t1a,t1a,optimize=True)

    h2A_vvvv = 0.0
    h2A_vvvv += vA['vvvv']
    h2A_vvvv -= np.einsum('mbef,am->abef',vA['ovvv'],t1a,optimize=True) # ab
    h2A_vvvv += np.einsum('maef,bm->abef',vA['ovvv'],t1a,optimize=True) # ba
    h2A_vvvv += np.einsum('mnef,bn,am->abef',vA['oovv'],t1a,t1a,optimize=True)

    h2A_vooo = 0.0
    h2A_vooo += vA['vooo']
    h2A_vooo -= 0.5*np.einsum('nmij,an->amij',vA['oooo'],t1a,optimize=True)
    h2A_vooo += np.einsum('amef,ei,fj->amij',vA['vovv'],t1a,t1a,optimize=True)
    h2A_vooo += np.einsum('amie,ej->amij',vA['voov'],t1a,optimize=True)
    h2A_vooo -= np.einsum('amje,ei->amij',vA['voov'],t1a,optimize=True)
    h2A_vooo -= 0.5*np.einsum('nmef,fj,an,ei->amij',vA['oovv'],t1a,t1a,t1a,optimize=True)

    h2A_vvov = 0.0
    h2A_vvov += vA['vvov']
    h2A_vvov += 0.5*np.einsum('abfe,fi->abie',vA['vvvv'],t1a,optimize=True)
    h2A_vvov += np.einsum('mnie,am,bn->abie',vA['ooov'],t1a,t1a,optimize=True)

    h2A_voov = 0.0
    h2A_voov += vA['voov']
    h2A_voov -= np.einsum('nmie,an->amie',vA['ooov'],t1a,optimize=True)
    h2A_voov += np.einsum('amfe,fi->amie',vA['vovv'],t1a,optimize=True)
    h2A_voov -= np.einsum('nmfe,fi,an->amie',vA['oovv'],t1a,t1a,optimize=True)

    h2A_ooov = 0.0
    h2A_ooov += vA['ooov']
    h2A_ooov += np.einsum('mnfe,fi->mnie',vA['oovv'],t1a,optimize=True)

    h2A_vovv = 0.0
    h2A_vovv += vA['vovv']
    h2A_vovv -= np.einsum('mnfe,an->amef',vA['oovv'],t1a,optimize=True)

    h2B_oooo = 0.0
    h2B_oooo += vB['oooo'] 
    h2B_oooo += np.einsum('mnej,ei->mnij',vB['oovo'],t1a,optimize=True)
    h2B_oooo += np.einsum('mnif,fj->mnij',vB['ooov'],t1b,optimize=True)
    h2B_oooo += np.einsum('mnef,ei,fj->mnij',vB['oovv'],t1a,t1b,optimize=True)    
        
    h2B_vvvv = 0.0
    h2B_vvvv += vB['vvvv']
    h2B_vvvv -= np.einsum('mbef,am->abef',vB['ovvv'],t1a,optimize=True)
    h2B_vvvv -= np.einsum('anef,bn->abef',vB['vovv'],t1b,optimize=True)
    h2B_vvvv += np.einsum('mnef,am,bn->abef',vB['oovv'],t1a,t1b,optimize=True)

    h2B_voov = 0.0
    h2B_voov += vB['voov']
    h2B_voov -= np.einsum('nmie,an->amie',vB['ooov'],t1a,optimize=True)
    h2B_voov += np.einsum('amfe,fi->amie',vB['vovv'],t1a,optimize=True)
    h2B_voov -= np.einsum('nmfe,fi,an->amie',vB['oovv'],t1a,t1a,optimize=True)
            
    h2B_ovov = 0.0
    h2B_ovov += vB['ovov']
    h2B_ovov += np.einsum('mafe,fi->maie',vB['ovvv'],t1a,optimize=True)
    h2B_ovov -= np.einsum('mnie,an->maie',vB['ooov'],t1b,optimize=True)
    h2B_ovov -= np.einsum('mnfe,an,fi->maie',vB['oovv'],t1b,t1a,optimize=True)
           
    h2B_vovo = 0.0
    h2B_vovo += vB['vovo']
    h2B_vovo -= np.einsum('nmei,an->amei',vB['oovo'],t1a,optimize=True)
    h2B_vovo += np.einsum('amef,fi->amei',vB['vovv'],t1b,optimize=True)
    h2B_vovo -= np.einsum('nmef,fi,an->amei',vB['oovv'],t1b,t1a,optimize=True)
           
    h2B_ovvo = 0.0
    h2B_ovvo += vB['ovvo']
    h2B_ovvo += np.einsum('maef,fi->maei',vB['ovvv'],t1b,optimize=True)
    h2B_ovvo -= np.einsum('mnei,an->maei',vB['oovo'],t1b,optimize=True)
    h2B_ovvo -= np.einsum('mnef,fi,an->maei',vB['oovv'],t1b,t1b,optimize=True)

    h2B_ovoo = 0.0
    h2B_ovoo += vB['ovoo']
    h2B_ovoo += np.einsum('mbej,ei->mbij',vB['ovvo'],t1a,optimize=True)
    h2B_ovoo -= np.einsum('mnij,bn->mbij',vB['oooo'],t1b,optimize=True)
    h2B_ovoo -= np.einsum('mnif,bn,fj->mbij',vB['ooov'],t1b,t1b,optimize=True)
    h2B_ovoo -= np.einsum('mnej,bn,ei->mbij',vB['oovo'],t1b,t1a,optimize=True)
    h2B_ovoo += np.einsum('mbef,fj,ei->mbij',vB['ovvv'],t1b,t1a,optimize=True)
    
    h2B_vooo = 0.0
    h2B_vooo += vB['vooo']
    h2B_vooo += np.einsum('amif,fj->amij',vB['voov'],t1b,optimize=True)
    h2B_vooo -= np.einsum('nmef,an,ei,fj->amij',vB['oovv'],t1a,t1a,t1b,optimize=True)
    h2B_vooo += np.einsum('amef,fj,ei->amij',vB['vovv'],t1b,t1a,optimize=True)

    h2B_vvvo = 0.0
    h2B_vvvo += vB['vvvo']
    h2B_vvvo += np.einsum('abef,fj->abej',vB['vvvv'],t1b,optimize=True)
    h2B_vvvo -= np.einsum('anej,bn->abej',vB['vovo'],t1b,optimize=True)
    
    h2B_vvov = 0.0
    h2B_vvov += vB['vvov']
    h2B_vvov -= np.einsum('mbie,am->abie',vB['ovov'],t1a,optimize=True)

    h2B_ooov = 0.0
    h2B_ooov += vB['ooov']
    h2B_ooov += np.einsum('mnfe,fi->mnie',vB['oovv'],t1a,optimize=True)

    h2B_oovo = 0.0
    h2B_oovo += vB['oovo']
    h2B_oovo += np.einsum('nmef,fi->nmei',vB['oovv'],t1b,optimize=True)

    h2B_vovv = 0.0
    h2B_vovv += vB['vovv']
    h2B_vovv -= np.einsum('nmef,an->amef',vB['oovv'],t1a,optimize=True)

    h2B_ovvv = 0.0
    h2B_ovvv += vB['ovvv']
    h2B_ovvv -= np.einsum('mnfe,an->mafe',vB['oovv'],t1b,optimize=True)
    
    h2C_oooo = 0.0
    h2C_oooo += vC['oooo']
    h2C_oooo += np.einsum('mnie,ej->mnij',vC['ooov'],t1b,optimize=True) # ij
    h2C_oooo -= np.einsum('mnje,ei->mnij',vC['ooov'],t1b,optimize=True) # ji
    h2C_oooo += np.einsum('mnef,ei,fj->mnij',vC['oovv'],t1b,t1b,optimize=True)

    h2C_vvvv = 0.0
    h2C_vvvv += vC['vvvv']
    h2C_vvvv -= np.einsum('mbef,am->abef',vC['ovvv'],t1b,optimize=True) # ab
    h2C_vvvv += np.einsum('maef,bm->abef',vC['ovvv'],t1b,optimize=True) # ba
    h2C_vvvv += np.einsum('mnef,bn,am->abef',vC['oovv'],t1b,t1b,optimize=True)
           
    h2C_voov = 0.0
    h2C_voov += vC['voov']
    h2C_voov -= np.einsum('mnei,an->amie',vC['oovo'],t1b,optimize=True)
    h2C_voov += np.einsum('amfe,fi->amie',vC['vovv'],t1b,optimize=True)
    h2C_voov -= np.einsum('mnef,fi,an->amie',vC['oovv'],t1b,t1b,optimize=True)

    h2C_ovoo = 0.0
    h2C_ovoo += vC['ovoo']
    h2C_ovoo -= 0.5*np.einsum('mnij,bn->mbij',vC['oooo'],t1b,optimize=True)
    h2C_ovoo += np.einsum('mbef,ei,fj->mbij',vC['ovvv'],t1b,t1b,optimize=True)
    h2C_ovoo -= 0.5*np.einsum('mnef,fj,ei,bn->mbij',vC['oovv'],t1b,t1b,t1b,optimize=True)
    h2C_ovoo += np.einsum('mbif,fj->mbij',vC['ovov'],t1b,optimize=True)
    h2C_ovoo -= np.einsum('mbjf,fi->mbij',vC['ovov'],t1b,optimize=True)

    h2C_vvvo = 0.0
    h2C_vvvo += vC['vvvo']
    h2C_vvvo += 0.5*np.einsum('abef,fj->abej',vC['vvvv'],t1b,optimize=True)
    h2C_vvvo += np.einsum('mnej,am,bn->abej',vC['oovo'],t1b,t1b,optimize=True)

    h2C_ooov = 0.0
    h2C_ooov += vC['ooov']
    h2C_ooov += np.einsum('mnfe,fi->mnie',vC['oovv'],t1b,optimize=True)

    h2C_vovv = 0.0
    h2C_vovv += vC['vovv'] 
    h2C_vovv -= np.einsum('mnfe,an->amef',vC['oovv'],t1b,optimize=True)


    H1A = {'ov' : h1A_ov, 'oo' : h1A_oo, 'vv' : h1A_vv}
    H1B = {'ov' : h1B_ov, 'oo' : h1B_oo, 'vv' : h1B_vv}
    H2A = {'oooo' : h2A_oooo, 'vvvv' : h2A_vvvv, 'vvov' : h2A_vvov, 'vooo' : h2A_vooo, 'voov' : h2A_voov,
    'ooov' : h2A_ooov, 'vovv' : h2A_vovv}
    H2B = {'oooo' : h2B_oooo, 'vvvv' : h2B_vvvv, 'ovov' : h2B_ovov, 'voov' : h2B_voov, 
           'ovvo' : h2B_ovvo, 'vovo' : h2B_vovo, 'ovoo' : h2B_ovoo, 'vooo' : h2B_vooo, 
           'vvvo' : h2B_vvvo, 'vvov' : h2B_vvov, 'ooov' : h2B_ooov, 'vovv' : h2B_vovv,
           'ovvv' : h2B_ovvv, 'oovo' : h2B_oovo}
    H2C = {'oooo' : h2C_oooo, 'vvvv' : h2C_vvvv, 'vvvo' : h2C_vvvo, 'ovoo' : h2C_ovoo, 'voov' : h2C_voov,
    'ooov' : h2C_ooov, 'vovv' : h2C_vovv}
           

    return H1A, H1B, H2A, H2B, H2C

def get_ccsd_intermediates(cc_t,ints,sys):
    """Calculate the CCSD-like similarity-transformed HBar intermediates (H_N e^(T1+T2))_C.

    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    -------
    H1* : dict
        One-body HBar similarity-transformed intermediates. Sorted by occ/unocc blocks.
    H2* : dict
        Two-body HBar similarity-transformed intermediates. Sorted by occ/unocc blocks.
    """
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

    h1A_ov = 0.0
    h1A_ov += fA['ov']
    h1A_ov += np.einsum('imae,em->ia',vA['oovv'],t1a,optimize=True)
    h1A_ov += np.einsum('imae,em->ia',vB['oovv'],t1b,optimize=True)
       
    h1A_oo = 0.0
    h1A_oo += fA['oo']
    h1A_oo += np.einsum('je,ei->ji',h1A_ov,t1a,optimize=True)
    h1A_oo += np.einsum('jmie,em->ji',vA['ooov'],t1a,optimize=True)
    h1A_oo += np.einsum('jmie,em->ji',vB['ooov'],t1b,optimize=True)
    h1A_oo += 0.5*np.einsum('jnef,efin->ji',vA['oovv'],t2a,optimize=True)
    h1A_oo += np.einsum('jnef,efin->ji',vB['oovv'],t2b,optimize=True)

    h1A_vv = 0.0
    h1A_vv += fA['vv']
    h1A_vv -= np.einsum('mb,am->ab',h1A_ov,t1a,optimize=True)
    h1A_vv += np.einsum('ambe,em->ab',vA['vovv'],t1a,optimize=True)
    h1A_vv += np.einsum('ambe,em->ab',vB['vovv'],t1b,optimize=True)
    h1A_vv -= 0.5*np.einsum('mnbf,afmn->ab',vA['oovv'],t2a,optimize=True)
    h1A_vv -= np.einsum('mnbf,afmn->ab',vB['oovv'],t2b,optimize=True)

    h1B_ov = 0.0
    h1B_ov += fB['ov']
    h1B_ov += np.einsum('imae,em->ia',vC['oovv'],t1b,optimize=True)
    h1B_ov += np.einsum('miea,em->ia',vB['oovv'],t1a,optimize=True)

    h1B_oo = 0.0
    h1B_oo += fB['oo']
    h1B_oo += np.einsum('je,ei->ji',h1B_ov,t1b,optimize=True)
    h1B_oo += np.einsum('jmie,em->ji',vC['ooov'],t1b,optimize=True)
    h1B_oo += np.einsum('mjei,em->ji',vB['oovo'],t1a,optimize=True)
    h1B_oo += 0.5*np.einsum('jnef,efin->ji',vC['oovv'],t2c,optimize=True)
    h1B_oo += np.einsum('njfe,feni->ji',vB['oovv'],t2b,optimize=True)

    h1B_vv = 0.0
    h1B_vv += fB['vv']
    h1B_vv -= np.einsum('mb,am->ab',h1B_ov,t1b,optimize=True)
    h1B_vv += np.einsum('ambe,em->ab',vC['vovv'],t1b,optimize=True)
    h1B_vv += np.einsum('maeb,em->ab',vB['ovvv'],t1a,optimize=True)
    h1B_vv -= 0.5*np.einsum('mnbf,afmn->ab',vC['oovv'],t2c,optimize=True)
    h1B_vv -= np.einsum('nmfb,fanm->ab',vB['oovv'],t2b,optimize=True)

    Q1 = -np.einsum('mnfe,an->amef',vA['oovv'],t1a,optimize=True)
    I2A_vovv = vA['vovv'] + 0.5*Q1
    h2A_vovv = I2A_vovv + 0.5*Q1
    
    Q1 = -np.einsum('mnfe,an->amef',vA['oovv'],t1a,optimize=True)
    I2A_vovv = vA['vovv'] + 0.5*Q1
    h2A_vovv = I2A_vovv + 0.5*Q1

    Q1 = np.einsum('mnfe,fi->mnie',vA['oovv'],t1a,optimize=True)
    I2A_ooov = vA['ooov'] + 0.5*Q1
    h2A_ooov = I2A_ooov + 0.5*Q1

    Q1 = -np.einsum('nmef,an->amef',vB['oovv'],t1a,optimize=True)
    I2B_vovv = vB['vovv'] + 0.5*Q1
    h2B_vovv = I2B_vovv + 0.5*Q1

    Q1 = np.einsum('mnfe,fi->mnie',vB['oovv'],t1a,optimize=True)
    I2B_ooov = vB['ooov'] + 0.5*Q1
    h2B_ooov = I2B_ooov + 0.5*Q1

    Q1 = -np.einsum('mnef,an->maef',vB['oovv'],t1b,optimize=True)
    I2B_ovvv = vB['ovvv'] + 0.5*Q1
    h2B_ovvv = I2B_ovvv + 0.5*Q1

    Q1 = np.einsum('nmef,fi->nmei',vB['oovv'],t1b,optimize=True)
    I2B_oovo = vB['oovo'] + 0.5*Q1
    h2B_oovo = I2B_oovo + 0.5*Q1

    Q1 = -np.einsum('nmef,an->amef',vC['oovv'],t1b,optimize=True)
    I2C_vovv = vC['vovv'] + 0.5*Q1
    h2C_vovv = I2C_vovv + 0.5*Q1

    Q1 = np.einsum('mnfe,fi->mnie',vC['oovv'],t1b,optimize=True)
    I2C_ooov = vC['ooov'] + 0.5*Q1
    h2C_ooov = I2C_ooov + 0.5*Q1

    Q1 = -np.einsum('bmfe,am->abef',I2A_vovv,t1a,optimize=True)
    Q1 -= np.einsum('abef->baef',Q1,optimize=True)
    h2A_vvvv = 0.0
    h2A_vvvv += vA['vvvv']
    h2A_vvvv += 0.5*np.einsum('mnef,abmn->abef',vA['oovv'],t2a,optimize=True)
    h2A_vvvv += Q1

    h2B_vvvv = 0.0
    h2B_vvvv += vB['vvvv']
    h2B_vvvv -= np.einsum('mbef,am->abef',I2B_ovvv,t1a,optimize=True)
    h2B_vvvv -= np.einsum('amef,bm->abef',I2B_vovv,t1b,optimize=True)
    h2B_vvvv += np.einsum('mnef,abmn->abef',vB['oovv'],t2b,optimize=True)

    Q1 = -np.einsum('bmfe,am->abef',I2C_vovv,t1b,optimize=True)
    Q1 -= np.einsum('abef->baef',Q1,optimize=True)
    h2C_vvvv = 0.0
    h2C_vvvv += vC['vvvv']
    h2C_vvvv += 0.5*np.einsum('mnef,abmn->abef',vC['oovv'],t2c,optimize=True)
    h2C_vvvv += Q1

    Q1 = +np.einsum('nmje,ei->mnij',I2A_ooov,t1a,optimize=True)
    Q1 -= np.einsum('mnij->mnji',Q1,optimize=True)
    h2A_oooo = 0.0
    h2A_oooo += vA['oooo']
    h2A_oooo += 0.5*np.einsum('mnef,efij->mnij',vA['oovv'],t2a,optimize=True)
    h2A_oooo += Q1

    h2B_oooo = 0.0
    h2B_oooo += vB['oooo']
    h2B_oooo += np.einsum('mnej,ei->mnij',I2B_oovo,t1a,optimize=True)
    h2B_oooo += np.einsum('mnie,ej->mnij',I2B_ooov,t1b,optimize=True)
    h2B_oooo += np.einsum('mnef,efij->mnij',vB['oovv'],t2b,optimize=True)

    Q1 = +np.einsum('nmje,ei->mnij',I2C_ooov,t1b,optimize=True)
    Q1 -= np.einsum('mnij->mnji',Q1,optimize=True)
    h2C_oooo = 0.0
    h2C_oooo += vC['oooo']
    h2C_oooo += 0.5*np.einsum('mnef,efij->mnij',vC['oovv'],t2c,optimize=True)
    h2C_oooo += Q1

    h2A_voov = 0.0
    h2A_voov += vA['voov']
    h2A_voov += np.einsum('amfe,fi->amie',I2A_vovv,t1a,optimize=True)
    h2A_voov -= np.einsum('nmie,an->amie',I2A_ooov,t1a,optimize=True)
    h2A_voov += np.einsum('nmfe,afin->amie',vA['oovv'],t2a,optimize=True)
    h2A_voov += np.einsum('mnef,afin->amie',vB['oovv'],t2b,optimize=True)

    h2B_voov = 0.0
    h2B_voov += vB['voov']
    h2B_voov += np.einsum('amfe,fi->amie',I2B_vovv,t1a,optimize=True)
    h2B_voov -= np.einsum('nmie,an->amie',I2B_ooov,t1a,optimize=True)
    h2B_voov += np.einsum('nmfe,afin->amie',vB['oovv'],t2a,optimize=True)
    h2B_voov += np.einsum('nmfe,afin->amie',vC['oovv'],t2b,optimize=True)

    h2B_ovvo = 0.0
    h2B_ovvo += vB['ovvo']
    h2B_ovvo += np.einsum('maef,fi->maei',I2B_ovvv,t1b,optimize=True)
    h2B_ovvo -= np.einsum('mnei,an->maei',I2B_oovo,t1b,optimize=True)
    h2B_ovvo += np.einsum('mnef,afin->maei',vB['oovv'],t2c,optimize=True)
    h2B_ovvo += np.einsum('mnef,fani->maei',vA['oovv'],t2b,optimize=True)

    h2B_ovov = 0.0
    h2B_ovov += vB['ovov']
    h2B_ovov += np.einsum('mafe,fi->maie',I2B_ovvv,t1a,optimize=True)
    h2B_ovov -= np.einsum('mnie,an->maie',I2B_ooov,t1b,optimize=True)
    h2B_ovov -= np.einsum('mnfe,fain->maie',vB['oovv'],t2b,optimize=True)

    h2B_vovo = 0.0
    h2B_vovo += vB['vovo']
    h2B_vovo -= np.einsum('nmei,an->amei',I2B_oovo,t1a,optimize=True)
    h2B_vovo += np.einsum('amef,fi->amei',I2B_vovv,t1b,optimize=True)
    h2B_vovo -= np.einsum('nmef,afni->amei',vB['oovv'],t2b,optimize=True)

    h2C_voov = 0.0
    h2C_voov += vC['voov']
    h2C_voov += np.einsum('amfe,fi->amie',I2C_vovv,t1b,optimize=True)
    h2C_voov -= np.einsum('nmie,an->amie',I2C_ooov,t1b,optimize=True)
    h2C_voov += np.einsum('nmfe,afin->amie',vC['oovv'],t2c,optimize=True)
    h2C_voov += np.einsum('nmfe,fani->amie',vB['oovv'],t2b,optimize=True)


    Q1 = +np.einsum('mnjf,afin->amij',h2A_ooov,t2a,optimize=True)+np.einsum('mnjf,afin->amij',h2B_ooov,t2b,optimize=True)
    Q2 = vA['voov'] + 0.5*np.einsum('amef,ei->amif',vA['vovv'],t1a,optimize=True)
    Q2 = np.einsum('amif,fj->amij',Q2,t1a,optimize=True)
    Q1 += Q2
    Q1 -= np.einsum('amij->amji',Q1,optimize=True)
    h2A_vooo = 0.0
    h2A_vooo += vA['vooo']
    h2A_vooo += np.einsum('me,aeij->amij',h1A_ov,t2a,optimize=True)
    h2A_vooo -= np.einsum('nmij,an->amij',h2A_oooo,t1a,optimize=True)
    h2A_vooo += 0.5*np.einsum('amef,efij->amij',vA['vovv'],t2a,optimize=True)
    h2A_vooo += Q1

    Q1 = vB['voov']+np.einsum('amfe,fi->amie',vB['vovv'],t1a,optimize=True)
    h2B_vooo = 0.0
    h2B_vooo += vB['vooo']
    h2B_vooo += np.einsum('me,aeij->amij',h1B_ov,t2b,optimize=True)
    h2B_vooo -= np.einsum('nmij,an->amij',h2B_oooo,t1a,optimize=True)
    h2B_vooo += np.einsum('mnjf,afin->amij',h2C_ooov,t2b,optimize=True)
    h2B_vooo += np.einsum('nmfj,afin->amij',h2B_oovo,t2a,optimize=True)
    h2B_vooo -= np.einsum('nmif,afnj->amij',h2B_ooov,t2b,optimize=True)
    h2B_vooo += np.einsum('amej,ei->amij',vB['vovo'],t1a,optimize=True)
    h2B_vooo += np.einsum('amie,ej->amij',Q1,t1b,optimize=True)
    h2B_vooo += np.einsum('amef,efij->amij',vB['vovv'],t2b,optimize=True)

    Q1 = vB['ovov']+np.einsum('mafe,fj->maje',vB['ovvv'],t1a,optimize=True)
    h2B_ovoo = 0.0
    h2B_ovoo += vB['ovoo']
    h2B_ovoo += np.einsum('me,eaji->maji',h1A_ov,t2b,optimize=True)
    h2B_ovoo -= np.einsum('mnji,an->maji',h2B_oooo,t1b,optimize=True)
    h2B_ovoo += np.einsum('mnjf,fani->maji',h2A_ooov,t2b,optimize=True)
    h2B_ovoo += np.einsum('mnjf,fani->maji',h2B_ooov,t2c,optimize=True)
    h2B_ovoo -= np.einsum('mnfi,fajn->maji',h2B_oovo,t2b,optimize=True)
    h2B_ovoo += np.einsum('maje,ei->maji',Q1,t1b,optimize=True)
    h2B_ovoo += np.einsum('maei,ej->maji',vB['ovvo'],t1a,optimize=True)
    h2B_ovoo += np.einsum('mafe,feji->maji',vB['ovvv'],t2b,optimize=True)

    Q1 = +np.einsum('mnjf,afin->amij',h2C_ooov,t2c,optimize=True)+np.einsum('nmfj,fani->amij',h2B_oovo,t2b,optimize=True)
    Q2 = vC['voov'] + 0.5*np.einsum('amef,ei->amif',vC['vovv'],t1b,optimize=True)
    Q2 = np.einsum('amif,fj->amij',Q2,t1b,optimize=True)
    Q1 += Q2
    Q1 -= np.einsum('amij->amji',Q1,optimize=True)
    h2C_vooo = 0.0
    h2C_vooo += vC['vooo']
    h2C_vooo += np.einsum('me,aeij->amij',h1B_ov,t2c,optimize=True)
    h2C_vooo -= np.einsum('nmij,an->amij',h2C_oooo,t1b,optimize=True)
    h2C_vooo += 0.5*np.einsum('amef,efij->amij',vC['vovv'],t2c,optimize=True)
    h2C_vooo += Q1

    Q1 = +np.einsum('bnef,afin->abie',h2A_vovv,t2a,optimize=True)+np.einsum('bnef,afin->abie',h2B_vovv,t2b,optimize=True)
    Q2 = vA['ovov'] - 0.5*np.einsum('mnie,bn->mbie',vA['ooov'],t1a,optimize=True)
    Q2 = -np.einsum('mbie,am->abie',Q2,t1a,optimize=True)
    Q1 += Q2
    Q1 -= np.einsum('abie->baie',Q1,optimize=True)
    h2A_vvov = 0.0
    h2A_vvov += vA['vvov']
    h2A_vvov -= np.einsum('me,abim->abie',h1A_ov,t2a,optimize=True)
    h2A_vvov += np.einsum('abfe,fi->abie',h2A_vvvv,t1a,optimize=True)
    h2A_vvov += 0.5*np.einsum('mnie,abmn->abie',vA['ooov'],t2a,optimize=True)
    h2A_vvov += Q1

    Q1 = vB['ovov'] - np.einsum('mnie,bn->mbie',vB['ooov'],t1b,optimize=True)
    Q1 = -np.einsum('mbie,am->abie',Q1,t1a,optimize=True)
    h2B_vvov = 0.0
    h2B_vvov += vB['vvov']
    h2B_vvov -= np.einsum('me,abim->abie',h1B_ov,t2b,optimize=True)
    h2B_vvov += np.einsum('abfe,fi->abie',h2B_vvvv,t1a,optimize=True)
    h2B_vvov += np.einsum('nbfe,afin->abie',h2B_ovvv,t2a,optimize=True)
    h2B_vvov += np.einsum('bnef,afin->abie',h2C_vovv,t2b,optimize=True)
    h2B_vvov -= np.einsum('amfe,fbim->abie',h2B_vovv,t2b,optimize=True)
    h2B_vvov += Q1
    h2B_vvov -= np.einsum('amie,bm->abie',vB['voov'],t1b,optimize=True)
    h2B_vvov += np.einsum('nmie,abnm->abie',vB['ooov'],t2b,optimize=True)

    Q1 = vB['vovo'] - np.einsum('nmei,bn->bmei',vB['oovo'],t1a,optimize=True)
    Q1 = -np.einsum('bmei,am->baei',Q1,t1b,optimize=True)
    h2B_vvvo = 0.0
    h2B_vvvo += vB['vvvo']
    h2B_vvvo -= np.einsum('me,bami->baei',h1A_ov,t2b,optimize=True)
    h2B_vvvo += np.einsum('baef,fi->baei',h2B_vvvv,t1b,optimize=True)
    h2B_vvvo += np.einsum('bnef,fani->baei',h2A_vovv,t2b,optimize=True)
    h2B_vvvo += np.einsum('bnef,fani->baei',h2B_vovv,t2c,optimize=True)
    h2B_vvvo -= np.einsum('maef,bfmi->baei',h2B_ovvv,t2b,optimize=True)
    h2B_vvvo += Q1
    h2B_vvvo -= np.einsum('naei,bn->baei',vB['ovvo'],t1a,optimize=True)
    h2B_vvvo += np.einsum('nmei,banm->baei',vB['oovo'],t2b,optimize=True)

    Q1 = +np.einsum('bnef,afin->abie',h2C_vovv,t2c,optimize=True)+np.einsum('nbfe,fani->abie',h2B_ovvv,t2b,optimize=True)
    Q2 = vC['ovov'] - 0.5*np.einsum('mnie,bn->mbie',vC['ooov'],t1b,optimize=True)
    Q2 = -np.einsum('mbie,am->abie',Q2,t1b,optimize=True)
    Q1 += Q2
    Q1 -= np.einsum('abie->baie',Q1,optimize=True)
    h2C_vvov = 0.0
    h2C_vvov += vC['vvov']
    h2C_vvov -= np.einsum('me,abim->abie',h1B_ov,t2c,optimize=True)
    h2C_vvov += np.einsum('abfe,fi->abie',h2C_vvvv,t1b,optimize=True)
    h2C_vvov += 0.5*np.einsum('mnie,abmn->abie',vC['ooov'],t2c,optimize=True)
    h2C_vvov += Q1

    H1A = {'ov' : h1A_ov, 'oo' : h1A_oo, 'vv' : h1A_vv}

    H1B = {'ov' : h1B_ov, 'oo' : h1B_oo, 'vv' : h1B_vv}

    H2A = {'vovv' : h2A_vovv, 'ooov' : h2A_ooov, 'vvvv' : h2A_vvvv, 'oooo' : h2A_oooo, 'voov' : h2A_voov, 'vooo' : h2A_vooo, 'vvov' : h2A_vvov}

    H2B = {'vovv' : h2B_vovv, 'ooov' : h2B_ooov, 'ovvv' : h2B_ovvv, 'oovo' : h2B_oovo, 'vvvv' : h2B_vvvv, 'oooo' : h2B_oooo, 'voov' : h2B_voov,
    'ovvo' : h2B_ovvo, 'ovov' : h2B_ovov, 'vovo' : h2B_vovo, 'vooo' : h2B_vooo, 'ovoo' : h2B_ovoo, 'vvov' : h2B_vvov, 'vvvo' : h2B_vvvo}

    H2C = {'vovv' : h2C_vovv, 'ooov' : h2C_ooov, 'vvvv' : h2C_vvvv, 'oooo' : h2C_oooo, 'voov' : h2C_voov, 'vooo' : h2C_vooo, 'vvov' : h2C_vvov}

    return H1A,H1B,H2A,H2B,H2C

