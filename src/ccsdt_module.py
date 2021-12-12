"""Module with functions that perform the CC with singles, doubles,
and triples (CCSDT) calculation for a molecular system."""
import numpy as np
from solvers import diis_out_of_core
from cc_energy import calc_cc_energy
from HBar_module import get_ccs_intermediates, get_ccsd_intermediates
import time
import cc_loops2
from utilities import print_memory_usage, clean_up

#print(cc_loops.cc_loops.__doc__)

#@profile
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
        Contains the converged T1, T2, T3 cluster amplitudes
    Eccsdt : float
        Total CCSDT energy
    """
    print('\n==================================++Entering CCSDT Routine++=================================\n')

    vecfid = work_dir+'/t_diis'
    dvecfid = work_dir+'/dt_diis'

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

    # [TODO]: Write the residual dt_diis-*.npy files in chunks using memmap
    #resid_mmap = [None]*diis_size
    #for i in range(diis_size):
    #    resid_mmap[i] = open_memmap(dvecfid+'-'+str(i+1)+'.npy', mode="w+", shape=(ndim))

    # Jacobi/DIIS iterations
    it_micro = 0
    flag_conv = False
    it_macro = 0
    Ecorr_old = 0.0

    t_start = time.time()
    print('Iteration    Residuum               deltaE                 Ecorr')
    print('=============================================================================')
    while it_micro < maxit:
        
        # get DIIS counter
        ndiis = it_micro%diis_size

        # reshape T into tensor form
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
       
        # CCS intermediates
        H1A,H1B,H2A,H2B,H2C = get_ccs_intermediates(cc_t,ints,sys)
        #H1A2,H1B2,H2A2,H2B2,H2C2 = get_ccs_intermediates_v2(cc_t,ints,sys)
        #for key in H2A.keys():
        #    err = np.linalg.norm(H2A[key].flatten() - H2A2[key].flatten())
        #    print('error in H2A({}) = {}'.format(key,err))

        # update T2
        cc_t, T_resid[idx_2a] = update_t2a(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        cc_t, T_resid[idx_2b] = update_t2b(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        if flag_RHF:
            cc_t['t2c'] = cc_t['t2a']
            T_resid[idx_2c] = T_resid[idx_2a]
        else:
            cc_t, T_resid[idx_2c] = update_t2c(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)

        # update T1                        
        cc_t, T_resid[idx_1a] = update_t1a(cc_t,ints,sys,shift)
        if flag_RHF:
            cc_t['t1b'] = cc_t['t1a']
            T_resid[idx_1b] = T_resid[idx_1a]
        else:
            cc_t, T_resid[idx_1b] = update_t1b(cc_t,ints,sys,shift)

        # CCSD intermediates
        H1A,H1B,H2A,H2B,H2C = get_ccsd_intermediates(cc_t,ints,sys)

        # update T3
        cc_t, T_resid[idx_3a] = update_t3a(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        cc_t, T_resid[idx_3b] = update_t3b(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        if flag_RHF:
            cc_t['t3c'] = np.transpose(cc_t['t3b'],(2,0,1,5,3,4))
            T_resid[idx_3c] = T_resid[idx_3b] # THIS PROBABLY DOESN'T WORK!
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

        # build DIIS residual
        #T_resid = T - T_old
        
        # change in Ecorr
        deltaE = Ecorr - Ecorr_old

        # check for exit condition
        resid = np.linalg.norm(T_resid)
        if resid < tol and abs(deltaE) < tol:
            flag_conv = True
            break

        # Save T and dT vectors to disk for out of core DIIS
        np.save(vecfid+'-'+str(ndiis+1)+'.npy',T)
        np.save(dvecfid+'-'+str(ndiis+1)+'.npy',T_resid)
        # Do DIIS extrapolation        
        if ndiis == 0 and it_micro > 1:
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
   
    # Clean up DIIS files from the working directory
    clean_up(vecfid,diis_size)
    clean_up(dvecfid,diis_size)

    return cc_t, ints['Escf'] + Ecorr

#@profile
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
    chi1A_vv = 0.0
    chi1A_vv += ints['fA']['vv']
    chi1A_vv += np.einsum('anef,fn->ae',ints['vA']['vovv'],cc_t['t1a'],optimize=True)
    chi1A_vv += np.einsum('anef,fn->ae',ints['vB']['vovv'],cc_t['t1b'],optimize=True)
    chi1A_oo = 0.0
    chi1A_oo += ints['fA']['oo']
    chi1A_oo += np.einsum('mnif,fn->mi',ints['vA']['ooov'],cc_t['t1a'],optimize=True)
    chi1A_oo += np.einsum('mnif,fn->mi',ints['vB']['ooov'],cc_t['t1b'],optimize=True)
    h1A_ov = 0.0
    h1A_ov += ints['fA']['ov']
    h1A_ov += np.einsum('mnef,fn->me',ints['vA']['oovv'],cc_t['t1a'],optimize=True) 
    h1A_ov += np.einsum('mnef,fn->me',ints['vB']['oovv'],cc_t['t1b'],optimize=True) 
    h1B_ov = 0.0
    h1B_ov += ints['fB']['ov'] 
    h1B_ov += np.einsum('nmfe,fn->me',ints['vB']['oovv'],cc_t['t1a'],optimize=True) 
    h1B_ov += np.einsum('mnef,fn->me',ints['vC']['oovv'],cc_t['t1b'],optimize=True) 
    h1A_oo = 0.0
    h1A_oo += chi1A_oo 
    h1A_oo += np.einsum('me,ei->mi',h1A_ov,cc_t['t1a'],optimize=True) 
    h2A_ooov = ints['vA']['ooov'] + np.einsum('mnfe,fi->mnie',ints['vA']['oovv'],cc_t['t1a'],optimize=True) 
    h2B_ooov = ints['vB']['ooov'] + np.einsum('mnfe,fi->mnie',ints['vB']['oovv'],cc_t['t1a'],optimize=True) 
    h2A_vovv = ints['vA']['vovv'] - np.einsum('mnfe,an->amef',ints['vA']['oovv'],cc_t['t1a'],optimize=True) 
    h2B_vovv = ints['vB']['vovv'] - np.einsum('nmef,an->amef',ints['vB']['oovv'],cc_t['t1a'],optimize=True) 

    X1A = 0.0
    X1A += ints['fA']['vo']
    X1A -= np.einsum('mi,am->ai',h1A_oo,cc_t['t1a'],optimize=True) 
    X1A += np.einsum('ae,ei->ai',chi1A_vv,cc_t['t1a'],optimize=True) 
    X1A += np.einsum('anif,fn->ai',ints['vA']['voov'],cc_t['t1a'],optimize=True) 
    X1A += np.einsum('anif,fn->ai',ints['vB']['voov'],cc_t['t1b'],optimize=True) 
    X1A += np.einsum('me,aeim->ai',h1A_ov,cc_t['t2a'],optimize=True)
    X1A += np.einsum('me,aeim->ai',h1B_ov,cc_t['t2b'],optimize=True)
    X1A -= 0.5*np.einsum('mnif,afmn->ai',h2A_ooov,cc_t['t2a'],optimize=True)
    X1A -= np.einsum('mnif,afmn->ai',h2B_ooov,cc_t['t2b'],optimize=True)
    X1A += 0.5*np.einsum('anef,efin->ai',h2A_vovv,cc_t['t2a'],optimize=True)
    X1A += np.einsum('anef,efin->ai',h2B_vovv,cc_t['t2b'],optimize=True)
    X1A += 0.25*np.einsum('mnef,aefimn->ai',ints['vA']['oovv'],cc_t['t3a'],optimize=True)
    X1A += np.einsum('mnef,aefimn->ai',ints['vB']['oovv'],cc_t['t3b'],optimize=True)
    X1A += 0.25*np.einsum('mnef,aefimn->ai',ints['vC']['oovv'],cc_t['t3c'],optimize=True)

    cc_t['t1a'], resid = cc_loops2.cc_loops2.update_t1a(cc_t['t1a'],X1A,ints['fA']['oo'],ints['fA']['vv'],shift)
    return cc_t, resid.flatten()

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
        New cluster amplitudes T1, T2, T3
    """
    # Intermediates
    chi1B_vv = 0.0
    chi1B_vv += ints['fB']['vv']
    chi1B_vv += np.einsum('anef,fn->ae',ints['vC']['vovv'],cc_t['t1b'],optimize=True)
    chi1B_vv += np.einsum('nafe,fn->ae',ints['vB']['ovvv'],cc_t['t1a'],optimize=True)
    chi1B_oo = 0.0
    chi1B_oo += ints['fB']['oo']
    chi1B_oo += np.einsum('mnif,fn->mi',ints['vC']['ooov'],cc_t['t1b'],optimize=True)
    chi1B_oo += np.einsum('nmfi,fn->mi',ints['vB']['oovo'],cc_t['t1a'],optimize=True)
    h1A_ov = 0.0
    h1A_ov += ints['fA']['ov']
    h1A_ov += np.einsum('mnef,fn->me',ints['vA']['oovv'],cc_t['t1a'],optimize=True) 
    h1A_ov += np.einsum('mnef,fn->me',ints['vB']['oovv'],cc_t['t1b'],optimize=True) 
    h1B_ov = 0.0
    h1B_ov += ints['fB']['ov'] 
    h1B_ov += np.einsum('nmfe,fn->me',ints['vB']['oovv'],cc_t['t1a'],optimize=True) 
    h1B_ov += np.einsum('mnef,fn->me',ints['vC']['oovv'],cc_t['t1b'],optimize=True) 
    h1B_oo = 0.0
    h1B_oo += chi1B_oo + np.einsum('me,ei->mi',h1B_ov,cc_t['t1b'],optimize=True)
    h2C_ooov = 0.0
    h2C_ooov += ints['vC']['ooov']
    h2C_ooov += np.einsum('mnfe,fi->mnie',ints['vC']['oovv'],cc_t['t1b'],optimize=True)
    h2B_oovo = 0.0
    h2B_oovo += ints['vB']['oovo']
    h2B_oovo += np.einsum('nmef,fi->nmei',ints['vB']['oovv'],cc_t['t1b'],optimize=True)
    h2C_vovv = 0.0
    h2C_vovv += ints['vC']['vovv']
    h2C_vovv -= np.einsum('mnfe,an->amef',ints['vC']['oovv'],cc_t['t1b'],optimize=True)
    h2B_ovvv = 0.0
    h2B_ovvv += ints['vB']['ovvv']
    h2B_ovvv -= np.einsum('mnfe,an->mafe',ints['vB']['oovv'],cc_t['t1b'],optimize=True)

    X1B = 0.0
    X1B += ints['fB']['vo']
    X1B -= np.einsum('mi,am->ai',h1B_oo,cc_t['t1b'],optimize=True)
    X1B += np.einsum('ae,ei->ai',chi1B_vv,cc_t['t1b'],optimize=True)
    X1B += np.einsum('anif,fn->ai',ints['vC']['voov'],cc_t['t1b'],optimize=True)
    X1B += np.einsum('nafi,fn->ai',ints['vB']['ovvo'],cc_t['t1a'],optimize=True)
    X1B += np.einsum('me,eami->ai',h1A_ov,cc_t['t2b'],optimize=True)
    X1B += np.einsum('me,aeim->ai',h1B_ov,cc_t['t2c'],optimize=True)
    X1B -= 0.5*np.einsum('mnif,afmn->ai',h2C_ooov,cc_t['t2c'],optimize=True)
    X1B -= np.einsum('nmfi,fanm->ai',h2B_oovo,cc_t['t2b'],optimize=True)
    X1B += 0.5*np.einsum('anef,efin->ai',h2C_vovv,cc_t['t2c'],optimize=True)
    X1B += np.einsum('nafe,feni->ai',h2B_ovvv,cc_t['t2b'],optimize=True)
    X1B += 0.25*np.einsum('mnef,aefimn->ai',ints['vC']['oovv'],cc_t['t3d'],optimize=True)
    X1B += 0.25*np.einsum('mnef,efamni->ai',ints['vA']['oovv'],cc_t['t3b'],optimize=True)
    X1B += np.einsum('mnef,efamni->ai',ints['vB']['oovv'],cc_t['t3c'],optimize=True)
    
    cc_t['t1b'], resid = cc_loops2.cc_loops2.update_t1b(cc_t['t1b'],X1B,ints['fB']['oo'],ints['fB']['vv'],shift)
    return cc_t, resid.flatten()

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
        New cluster amplitudes T1, T2, T3
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
    X2A += 0.25*np.einsum('me,abeijm->abij',H1A['ov'],cc_t['t3a'],optimize=True)
    X2A += 0.25*np.einsum('me,abeijm->abij',H1B['ov'],cc_t['t3b'],optimize=True)
    X2A -= 0.5*np.einsum('mnif,abfmjn->abij',H2B['ooov'],cc_t['t3b'],optimize=True)
    X2A -= 0.25*np.einsum('mnif,abfmjn->abij',H2A['ooov'],cc_t['t3a'],optimize=True)
    X2A += 0.25*np.einsum('anef,ebfijn->abij',H2A['vovv'],cc_t['t3a'],optimize=True)
    X2A += 0.5*np.einsum('anef,ebfijn->abij',H2B['vovv'],cc_t['t3b'],optimize=True)

    cc_t['t2a'], resid = cc_loops2.cc_loops2.update_t2a(cc_t['t2a'],X2A,ints['fA']['oo'],ints['fA']['vv'],shift)
    return cc_t, resid.flatten()

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
        New cluster amplitudes T1, T2, T3
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
    X2B -= 0.5*np.einsum('mnif,afbmnj->abij',H2A['ooov'],cc_t['t3b'],optimize=True)
    X2B -= np.einsum('nmfj,afbinm->abij',H2B['oovo'],cc_t['t3b'],optimize=True)
    X2B -= 0.5*np.einsum('mnjf,afbinm->abij',H2C['ooov'],cc_t['t3c'],optimize=True)
    X2B -= np.einsum('mnif,afbmnj->abij',H2B['ooov'],cc_t['t3c'],optimize=True)
    X2B += 0.5*np.einsum('anef,efbinj->abij',H2A['vovv'],cc_t['t3b'],optimize=True)
    X2B += np.einsum('anef,efbinj->abij',H2B['vovv'],cc_t['t3c'],optimize=True)
    X2B += np.einsum('nbfe,afeinj->abij',H2B['ovvv'],cc_t['t3b'],optimize=True)
    X2B += 0.5*np.einsum('bnef,afeinj->abij',H2C['vovv'],cc_t['t3c'],optimize=True)
    X2B += np.einsum('me,aebimj->abij',H1A['ov'],cc_t['t3b'],optimize=True)
    X2B += np.einsum('me,aebimj->abij',H1B['ov'],cc_t['t3c'],optimize=True)

    cc_t['t2b'], resid = cc_loops2.cc_loops2.update_t2b(cc_t['t2b'],X2B,ints['fA']['oo'],ints['fA']['vv'],ints['fB']['oo'],ints['fB']['vv'],shift)
    return cc_t, resid.flatten()

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
        New cluster amplitudes T1, T2, T3
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
    X2C += 0.25*np.einsum('me,eabmij->abij',H1A['ov'],cc_t['t3c'],optimize=True)
    X2C += 0.25*np.einsum('me,abeijm->abij',H1B['ov'],cc_t['t3d'],optimize=True)
    X2C += 0.25*np.einsum('anef,ebfijn->abij',H2C['vovv'],cc_t['t3d'],optimize=True)
    X2C += 0.5*np.einsum('nafe,febnij->abij',H2B['ovvv'],cc_t['t3c'],optimize=True)
    X2C -= 0.25*np.einsum('mnif,abfmjn->abij',H2C['ooov'],cc_t['t3d'],optimize=True)
    X2C -= 0.5*np.einsum('nmfi,fabnmj->abij',H2B['oovo'],cc_t['t3c'],optimize=True)

    cc_t['t2c'], resid = cc_loops2.cc_loops2.update_t2c(cc_t['t2c'],X2C,ints['fB']['oo'],ints['fB']['vv'],shift)
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
    # Intermediates
    I2A_vvov = -0.5*np.einsum('mnef,abfimn->abie',ints['vA']['oovv'],cc_t['t3a'],optimize=True)
    I2A_vvov -= np.einsum('mnef,abfimn->abie',ints['vB']['oovv'],cc_t['t3b'],optimize=True)
    I2A_vvov += np.einsum('me,abim->abie',H1A['ov'],cc_t['t2a'],optimize=True)
    I2A_vvov += H2A['vvov']
    I2A_vooo = 0.5*np.einsum('mnef,aefijn->amij',ints['vA']['oovv'],cc_t['t3a'],optimize=True)
    I2A_vooo += np.einsum('mnef,aefijn->amij',ints['vB']['oovv'],cc_t['t3b'],optimize=True)
    I2A_vooo += H2A['vooo']   
    # MM(2,3)A
    X3A = -0.25*np.einsum('amij,bcmk->abcijk',I2A_vooo,cc_t['t2a'],optimize=True) 
    X3A += 0.25*np.einsum('abie,ecjk->abcijk',I2A_vvov,cc_t['t2a'],optimize=True)
    # (HBar*T3)_C    
    X3A -= (1.0/12.0)*np.einsum('mk,abcijm->abcijk',H1A['oo'],cc_t['t3a'],optimize=True)
    X3A += (1.0/12.0)*np.einsum('ce,abeijk->abcijk',H1A['vv'],cc_t['t3a'],optimize=True)
    X3A += (1.0/24.0)*np.einsum('mnij,abcmnk->abcijk',H2A['oooo'],cc_t['t3a'],optimize=True)
    X3A += (1.0/24.0)*np.einsum('abef,efcijk->abcijk',H2A['vvvv'],cc_t['t3a'],optimize=True)
    X3A += 0.25*np.einsum('cmke,abeijm->abcijk',H2A['voov'],cc_t['t3a'],optimize=True)
    X3A += 0.25*np.einsum('cmke,abeijm->abcijk',H2B['voov'],cc_t['t3b'],optimize=True)

    cc_t['t3a'], resid = cc_loops2.cc_loops2.update_t3a_v2(cc_t['t3a'],X3A,ints['fA']['oo'],ints['fA']['vv'],shift)
    return cc_t, resid.flatten()

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
    # MM23B + (VT3)_C intermediates
    I2A_vvov = -0.5*np.einsum('mnef,abfimn->abie',ints['vA']['oovv'],cc_t['t3a'],optimize=True)
    I2A_vvov += -np.einsum('mnef,abfimn->abie',ints['vB']['oovv'],cc_t['t3b'],optimize=True)
    I2A_vvov += H2A['vvov']
    I2A_vooo = 0.5*np.einsum('mnef,aefijn->amij',ints['vA']['oovv'],cc_t['t3a'],optimize=True)
    I2A_vooo += np.einsum('mnef,aefijn->amij',ints['vB']['oovv'],cc_t['t3b'],optimize=True)
    I2A_vooo += H2A['vooo']
    I2A_vooo += -np.einsum('me,aeij->amij',H1A['ov'],cc_t['t2a'],optimize=True)
    I2B_vvvo = -0.5*np.einsum('mnef,afbmnj->abej',ints['vA']['oovv'],cc_t['t3b'],optimize=True)
    I2B_vvvo += -np.einsum('mnef,afbmnj->abej',ints['vB']['oovv'],cc_t['t3c'],optimize=True)
    I2B_vvvo += H2B['vvvo']
    I2B_ovoo = 0.5*np.einsum('mnef,efbinj->mbij',ints['vA']['oovv'],cc_t['t3b'],optimize=True)
    I2B_ovoo += np.einsum('mnef,efbinj->mbij',ints['vB']['oovv'],cc_t['t3c'],optimize=True)
    I2B_ovoo += H2B['ovoo']
    I2B_ovoo += -np.einsum('me,ecjk->mcjk',H1A['ov'],cc_t['t2b'],optimize=True)
    I2B_vvov = -np.einsum('nmfe,afbinm->abie',ints['vB']['oovv'],cc_t['t3b'],optimize=True)
    I2B_vvov += -0.5*np.einsum('nmfe,afbinm->abie',ints['vC']['oovv'],cc_t['t3c'],optimize=True)
    I2B_vvov += H2B['vvov']
    I2B_vooo = np.einsum('nmfe,afeinj->amij',ints['vB']['oovv'],cc_t['t3b'],optimize=True)
    I2B_vooo += 0.5*np.einsum('nmfe,afeinj->amij',ints['vC']['oovv'],cc_t['t3c'],optimize=True)
    I2B_vooo += H2B['vooo']
    I2B_vooo += -np.einsum('me,aeik->amik',H1B['ov'],cc_t['t2b'],optimize=True) 
    # MM(2,3)B 
    X3B = 0.5*np.einsum('bcek,aeij->abcijk',I2B_vvvo,cc_t['t2a'],optimize=True)
    X3B -= 0.5*np.einsum('mcjk,abim->abcijk',I2B_ovoo,cc_t['t2a'],optimize=True)
    X3B += np.einsum('acie,bejk->abcijk',I2B_vvov,cc_t['t2b'],optimize=True)
    X3B -= np.einsum('amik,bcjm->abcijk',I2B_vooo,cc_t['t2b'],optimize=True)
    X3B += 0.5*np.einsum('abie,ecjk->abcijk',I2A_vvov,cc_t['t2b'],optimize=True)
    X3B -= 0.5*np.einsum('amij,bcmk->abcijk',I2A_vooo,cc_t['t2b'],optimize=True)
    # (HBar*T3)_C
    X3B -= 0.5*np.einsum('mi,abcmjk->abcijk',H1A['oo'],cc_t['t3b'],optimize=True)
    X3B -= 0.25*np.einsum('mk,abcijm->abcijk',H1B['oo'],cc_t['t3b'],optimize=True)
    X3B += 0.5*np.einsum('ae,ebcijk->abcijk',H1A['vv'],cc_t['t3b'],optimize=True)
    X3B += 0.25*np.einsum('ce,abeijk->abcijk',H1B['vv'],cc_t['t3b'],optimize=True)
    X3B += 0.125*np.einsum('mnij,abcmnk->abcijk',H2A['oooo'],cc_t['t3b'],optimize=True)
    X3B += 0.5*np.einsum('mnjk,abcimn->abcijk',H2B['oooo'],cc_t['t3b'],optimize=True)
    X3B += 0.125*np.einsum('abef,efcijk->abcijk',H2A['vvvv'],cc_t['t3b'],optimize=True)
    X3B += 0.5*np.einsum('bcef,aefijk->abcijk',H2B['vvvv'],cc_t['t3b'],optimize=True)
    X3B += np.einsum('amie,ebcmjk->abcijk',H2A['voov'],cc_t['t3b'],optimize=True)   
    X3B += np.einsum('amie,becjmk->abcijk',H2B['voov'],cc_t['t3c'],optimize=True)    
    X3B += 0.25*np.einsum('mcek,abeijm->abcijk',H2B['ovvo'],cc_t['t3a'],optimize=True)
    X3B += 0.25*np.einsum('cmke,abeijm->abcijk',H2C['voov'],cc_t['t3b'],optimize=True)
    X3B -= 0.5*np.einsum('amek,ebcijm->abcijk',H2B['vovo'],cc_t['t3b'],optimize=True)
    X3B -= 0.5*np.einsum('mcie,abemjk->abcijk',H2B['ovov'],cc_t['t3b'],optimize=True)

    cc_t['t3b'], resid = cc_loops2.cc_loops2.update_t3b_v2(cc_t['t3b'],X3B,ints['fA']['oo'],ints['fA']['vv'],ints['fB']['oo'],ints['fB']['vv'],shift)
    return cc_t, resid.flatten()

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
    # Intermediates
    I2B_vvvo = -0.5*np.einsum('mnef,afbmnj->abej',ints['vA']['oovv'],cc_t['t3b'],optimize=True)
    I2B_vvvo += -np.einsum('mnef,afbmnj->abej',ints['vB']['oovv'],cc_t['t3c'],optimize=True)
    I2B_vvvo += H2B['vvvo']
    I2B_ovoo = 0.5*np.einsum('mnef,efbinj->mbij',ints['vA']['oovv'],cc_t['t3b'],optimize=True)
    I2B_ovoo += np.einsum('mnef,efbinj->mbij',ints['vB']['oovv'],cc_t['t3c'],optimize=True)
    I2B_ovoo += H2B['ovoo']
    I2B_ovoo -= np.einsum('me,ebij->mbij',H1A['ov'],cc_t['t2b'],optimize=True)
    I2B_vvov = -np.einsum('nmfe,afbinm->abie',ints['vB']['oovv'],cc_t['t3b'],optimize=True)
    I2B_vvov += -0.5*np.einsum('nmfe,afbinm->abie',ints['vC']['oovv'],cc_t['t3c'],optimize=True)
    I2B_vvov += H2B['vvov']
    I2B_vooo = np.einsum('nmfe,afeinj->amij',ints['vB']['oovv'],cc_t['t3b'],optimize=True)
    I2B_vooo += 0.5*np.einsum('nmfe,afeinj->amij',ints['vC']['oovv'],cc_t['t3c'],optimize=True)
    I2B_vooo += H2B['vooo']
    I2B_vooo -= np.einsum('me,aeij->amij',H1B['ov'],cc_t['t2b'],optimize=True)
    I2C_vvov = -0.5*np.einsum('mnef,abfimn->abie',ints['vC']['oovv'],cc_t['t3d'],optimize=True)
    I2C_vvov += -np.einsum('nmfe,fabnim->abie',ints['vB']['oovv'],cc_t['t3c'],optimize=True)
    I2C_vvov += H2C['vvov']
    I2C_vooo = np.einsum('nmfe,faenij->amij',ints['vB']['oovv'],cc_t['t3c'],optimize=True)
    I2C_vooo += 0.5*np.einsum('mnef,aefijn->amij',ints['vC']['oovv'],cc_t['t3d'],optimize=True)
    I2C_vooo += H2C['vooo']
    I2C_vooo -= np.einsum('me,cekj->cmkj',H1B['ov'],cc_t['t2c'],optimize=True)
    # MM(2,3)C
    X3C = 0.5*np.einsum('abie,ecjk->abcijk',I2B_vvov,cc_t['t2c'],optimize=True)
    X3C -= 0.5*np.einsum('amij,bcmk->abcijk',I2B_vooo,cc_t['t2c'],optimize=True)
    X3C += 0.5*np.einsum('cbke,aeij->abcijk',I2C_vvov,cc_t['t2b'],optimize=True)
    X3C -= 0.5*np.einsum('cmkj,abim->abcijk',I2C_vooo,cc_t['t2b'],optimize=True)
    X3C += np.einsum('abej,ecik->abcijk',I2B_vvvo,cc_t['t2b'],optimize=True)
    X3C -= np.einsum('mbij,acmk->abcijk',I2B_ovoo,cc_t['t2b'],optimize=True)
    # (HBar*T3)_C
    X3C -= 0.25*np.einsum('mi,abcmjk->abcijk',H1A['oo'],cc_t['t3c'],optimize=True)
    X3C -= 0.5*np.einsum('mj,abcimk->abcijk',H1B['oo'],cc_t['t3c'],optimize=True)
    X3C += 0.25*np.einsum('ae,ebcijk->abcijk',H1A['vv'],cc_t['t3c'],optimize=True)
    X3C += 0.5*np.einsum('be,aecijk->abcijk',H1B['vv'],cc_t['t3c'],optimize=True)
    X3C += 0.125*np.einsum('mnjk,abcimn->abcijk',H2C['oooo'],cc_t['t3c'],optimize=True)
    X3C += 0.5*np.einsum('mnij,abcmnk->abcijk',H2B['oooo'],cc_t['t3c'],optimize=True)
    X3C += 0.125*np.einsum('bcef,aefijk->abcijk',H2C['vvvv'],cc_t['t3c'],optimize=True)
    X3C += 0.5*np.einsum('abef,efcijk->abcijk',H2B['vvvv'],cc_t['t3c'],optimize=True)
    X3C += 0.25*np.einsum('amie,ebcmjk->abcijk',H2A['voov'],cc_t['t3c'],optimize=True)
    X3C += 0.25*np.einsum('amie,ebcmjk->abcijk',H2B['voov'],cc_t['t3d'],optimize=True)
    X3C += np.einsum('mbej,aecimk->abcijk',H2B['ovvo'],cc_t['t3b'],optimize=True)
    X3C += np.einsum('bmje,aecimk->abcijk',H2C['voov'],cc_t['t3c'],optimize=True)
    X3C -= 0.5*np.einsum('mbie,aecmjk->abcijk',H2B['ovov'],cc_t['t3c'],optimize=True)
    X3C -= 0.5*np.einsum('amej,ebcimk->abcijk',H2B['vovo'],cc_t['t3c'],optimize=True)

    cc_t['t3c'], resid = cc_loops2.cc_loops2.update_t3c_v2(cc_t['t3c'],X3C,ints['fA']['oo'],ints['fA']['vv'],ints['fB']['oo'],ints['fB']['vv'],shift)             
    return cc_t, resid.flatten()

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
    # Intermediates
    I2C_vvov = -0.5*np.einsum('mnef,abfimn->abie',ints['vC']['oovv'],cc_t['t3d'],optimize=True)
    I2C_vvov -= np.einsum('nmfe,fabnim->abie',ints['vB']['oovv'],cc_t['t3c'],optimize=True)
    I2C_vvov += np.einsum('me,abim->abie',H1B['ov'],cc_t['t2c'],optimize=True)
    I2C_vvov += H2C['vvov']
    I2C_vooo = 0.5*np.einsum('mnef,aefijn->amij',ints['vC']['oovv'],cc_t['t3d'],optimize=True)
    I2C_vooo += np.einsum('nmfe,faenij->amij',ints['vB']['oovv'],cc_t['t3c'],optimize=True)
    I2C_vooo += H2C['vooo']   
    # MM(2,3)D
    X3D = -0.25*np.einsum('amij,bcmk->abcijk',I2C_vooo,cc_t['t2c'],optimize=True) 
    X3D += 0.25*np.einsum('abie,ecjk->abcijk',I2C_vvov,cc_t['t2c'],optimize=True)
    # (HBar*T3)_C
    X3D -= (1.0/12.0)*np.einsum('mk,abcijm->abcijk',H1B['oo'],cc_t['t3d'],optimize=True)
    X3D += (1.0/12.0)*np.einsum('ce,abeijk->abcijk',H1B['vv'],cc_t['t3d'],optimize=True)
    X3D += (1.0/24.0)*np.einsum('mnij,abcmnk->abcijk',H2C['oooo'],cc_t['t3d'],optimize=True)
    X3D += (1.0/24.0)*np.einsum('abef,efcijk->abcijk',H2C['vvvv'],cc_t['t3d'],optimize=True)
    X3D += 0.25*np.einsum('maei,ebcmjk->abcijk',H2B['ovvo'],cc_t['t3c'],optimize=True)
    X3D += 0.25*np.einsum('amie,ebcmjk->abcijk',H2C['voov'],cc_t['t3d'],optimize=True)

    cc_t['t3d'], resid = cc_loops2.cc_loops2.update_t3d_v2(cc_t['t3d'],X3D,ints['fB']['oo'],ints['fB']['vv'],shift)
    return cc_t, resid.flatten()

