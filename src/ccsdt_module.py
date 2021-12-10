"""Module with functions that perform the CC with singles, doubles,
and triples (CCSDT) calculation for a molecular system."""
import numpy as np
from solvers import diis_out_of_core
from cc_energy import calc_cc_energy
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
    X2C -= 0.5*np.einsum('mbij,am->abij',H2C['ovoo'],cc_t['t1b'],optimize=True)
    X2C += 0.5*np.einsum('abej,ei->abij',H2C['vvvo'],cc_t['t1b'],optimize=True)
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
    #H1A = {}
    #H1A['ov'] = 0.0
    #H1A['ov'] += fA['ov']
    #H1A['ov'] += np.einsum('mnef,fn->me',ints['vA']['oovv'],cc_t['t1a'],optimize=True) 
    #H1A['ov'] += np.einsum('mnef,fn->me',ints['vB']['oovv'],cc_t['t1b'],optimize=True) 
    h1A_ov = 0.0
    h1A_ov += ints['fA']['ov']
    h1A_ov += np.einsum('mnef,fn->me',ints['vA']['oovv'],cc_t['t1a'],optimize=True) 
    h1A_ov += np.einsum('mnef,fn->me',ints['vB']['oovv'],cc_t['t1b'],optimize=True) 

    h1B_ov = 0.0
    h1B_ov += ints['fB']['ov'] 
    h1B_ov += np.einsum('nmfe,fn->me',ints['vB']['oovv'],cc_t['t1a'],optimize=True) 
    h1B_ov += np.einsum('mnef,fn->me',ints['vC']['oovv'],cc_t['t1b'],optimize=True) 

    h1A_vv = 0.0
    h1A_vv += ints['fA']['vv']
    h1A_vv += np.einsum('anef,fn->ae',ints['vA']['vovv'],cc_t['t1a'],optimize=True)
    h1A_vv += np.einsum('anef,fn->ae',ints['vB']['vovv'],cc_t['t1b'],optimize=True)
    h1A_vv -= np.einsum('me,am->ae',h1A_ov,cc_t['t1a'],optimize=True)

    h1A_oo = 0.0
    h1A_oo += ints['fA']['oo']
    h1A_oo += np.einsum('mnif,fn->mi',ints['vA']['ooov'],cc_t['t1a'],optimize=True)
    h1A_oo += np.einsum('mnif,fn->mi',ints['vB']['ooov'],cc_t['t1b'],optimize=True)
    h1A_oo += np.einsum('me,ei->mi',h1A_ov,cc_t['t1a'],optimize=True)

    h1B_vv = 0.0
    h1B_vv += ints['fB']['vv']
    h1B_vv += np.einsum('anef,fn->ae',ints['vC']['vovv'],cc_t['t1b'],optimize=True)
    h1B_vv += np.einsum('nafe,fn->ae',ints['vB']['ovvv'],cc_t['t1a'],optimize=True)
    h1B_vv -= np.einsum('me,am->ae',h1B_ov,cc_t['t1b'],optimize=True)

    h1B_oo = 0.0
    h1B_oo += ints['fB']['oo']
    h1B_oo += np.einsum('mnif,fn->mi',ints['vC']['ooov'],cc_t['t1b'],optimize=True)
    h1B_oo += np.einsum('nmfi,fn->mi',ints['vB']['oovo'],cc_t['t1a'],optimize=True)
    h1B_oo += np.einsum('me,ei->mi',h1B_ov,cc_t['t1b'],optimize=True)

    h2A_oooo = 0.0
    h2A_oooo += ints['vA']['oooo']
    h2A_oooo += np.einsum('mnej,ei->mnij',ints['vA']['oovo'],cc_t['t1a'],optimize=True) # ij
    h2A_oooo -= np.einsum('mnei,ej->mnij',ints['vA']['oovo'],cc_t['t1a'],optimize=True) # ji
    h2A_oooo += np.einsum('mnef,ei,fj->mnij',ints['vA']['oovv'],cc_t['t1a'],cc_t['t1a'],optimize=True) # BAD!!!

    h2A_vvvv = 0.0
    h2A_vvvv += ints['vA']['vvvv']
    h2A_vvvv -= np.einsum('mbef,am->abef',ints['vA']['ovvv'],cc_t['t1a'],optimize=True) # ab
    h2A_vvvv += np.einsum('maef,bm->abef',ints['vA']['ovvv'],cc_t['t1a'],optimize=True) # ba
    h2A_vvvv += np.einsum('mnef,bn,am->abef',ints['vA']['oovv'],cc_t['t1a'],cc_t['t1a'],optimize=True) # BAD!!!

    h2A_vooo = 0.0
    h2A_vooo += ints['vA']['vooo']
    h2A_vooo -= 0.5*np.einsum('nmij,an->amij',ints['vA']['oooo'],cc_t['t1a'],optimize=True)
    h2A_vooo += np.einsum('amef,ei,fj->amij',ints['vA']['vovv'],cc_t['t1a'],cc_t['t1a'],optimize=True)
    h2A_vooo += np.einsum('amie,ej->amij',ints['vA']['voov'],cc_t['t1a'],optimize=True)
    h2A_vooo -= np.einsum('amje,ei->amij',ints['vA']['voov'],cc_t['t1a'],optimize=True)
    h2A_vooo -= 0.5*np.einsum('nmef,fj,an,ei->amij',ints['vA']['oovv'],cc_t['t1a'],cc_t['t1a'],cc_t['t1a'],optimize=True) # NOMINALLY O(N^7) CONTRACTION.. WTF ARE YOU DOING?!?

    h2A_vvov = 0.0
    h2A_vvov += ints['vA']['vvov']
    h2A_vvov += 0.5*np.einsum('abfe,fi->abie',ints['vA']['vvvv'],cc_t['t1a'],optimize=True)
    h2A_vvov += np.einsum('mnie,am,bn->abie',ints['vA']['ooov'],cc_t['t1a'],cc_t['t1a'],optimize=True)

    h2A_voov = 0.0
    h2A_voov += ints['vA']['voov']
    h2A_voov -= np.einsum('nmie,an->amie',ints['vA']['ooov'],cc_t['t1a'],optimize=True)
    h2A_voov += np.einsum('amfe,fi->amie',ints['vA']['vovv'],cc_t['t1a'],optimize=True)
    h2A_voov -= np.einsum('nmfe,fi,an->amie',ints['vA']['oovv'],cc_t['t1a'],cc_t['t1a'],optimize=True)

    h2A_ooov = 0.0
    h2A_ooov += ints['vA']['ooov']
    h2A_ooov += np.einsum('mnfe,fi->mnie',ints['vA']['oovv'],cc_t['t1a'],optimize=True)

    h2A_vovv = 0.0
    h2A_vovv += ints['vA']['vovv']
    h2A_vovv -= np.einsum('mnfe,an->amef',ints['vA']['oovv'],cc_t['t1a'],optimize=True)

    h2B_oooo = 0.0
    h2B_oooo += ints['vB']['oooo'] 
    h2B_oooo += np.einsum('mnej,ei->mnij',ints['vB']['oovo'],cc_t['t1a'],optimize=True)
    h2B_oooo += np.einsum('mnif,fj->mnij',ints['vB']['ooov'],cc_t['t1b'],optimize=True)
    h2B_oooo += np.einsum('mnef,ei,fj->mnij',ints['vB']['oovv'],cc_t['t1a'],cc_t['t1b'],optimize=True)    
        
    h2B_vvvv = 0.0
    h2B_vvvv += ints['vB']['vvvv']
    h2B_vvvv -= np.einsum('mbef,am->abef',ints['vB']['ovvv'],cc_t['t1a'],optimize=True)
    h2B_vvvv -= np.einsum('anef,bn->abef',ints['vB']['vovv'],cc_t['t1b'],optimize=True)
    h2B_vvvv += np.einsum('mnef,am,bn->abef',ints['vB']['oovv'],cc_t['t1a'],cc_t['t1b'],optimize=True)

    h2B_voov = 0.0
    h2B_voov += ints['vB']['voov']
    h2B_voov -= np.einsum('nmie,an->amie',ints['vB']['ooov'],cc_t['t1a'],optimize=True)
    h2B_voov += np.einsum('amfe,fi->amie',ints['vB']['vovv'],cc_t['t1a'],optimize=True)
    h2B_voov -= np.einsum('nmfe,fi,an->amie',ints['vB']['oovv'],cc_t['t1a'],cc_t['t1a'],optimize=True)
            
    h2B_ovov = 0.0
    h2B_ovov += ints['vB']['ovov']
    h2B_ovov += np.einsum('mafe,fi->maie',ints['vB']['ovvv'],cc_t['t1a'],optimize=True)
    h2B_ovov -= np.einsum('mnie,an->maie',ints['vB']['ooov'],cc_t['t1b'],optimize=True)
    h2B_ovov -= np.einsum('mnfe,an,fi->maie',ints['vB']['oovv'],cc_t['t1b'],cc_t['t1a'],optimize=True)
           
    h2B_vovo = 0.0
    h2B_vovo += ints['vB']['vovo']
    h2B_vovo -= np.einsum('nmei,an->amei',ints['vB']['oovo'],cc_t['t1a'],optimize=True)
    h2B_vovo += np.einsum('amef,fi->amei',ints['vB']['vovv'],cc_t['t1b'],optimize=True)
    h2B_vovo -= np.einsum('nmef,fi,an->amei',ints['vB']['oovv'],cc_t['t1b'],cc_t['t1a'],optimize=True)
           
    h2B_ovvo = 0.0
    h2B_ovvo += ints['vB']['ovvo']
    h2B_ovvo += np.einsum('maef,fi->maei',ints['vB']['ovvv'],cc_t['t1b'],optimize=True)
    h2B_ovvo -= np.einsum('mnei,an->maei',ints['vB']['oovo'],cc_t['t1b'],optimize=True)
    h2B_ovvo -= np.einsum('mnef,fi,an->maei',ints['vB']['oovv'],cc_t['t1b'],cc_t['t1b'],optimize=True)

    h2B_ovoo = 0.0
    h2B_ovoo += ints['vB']['ovoo']
    h2B_ovoo += np.einsum('mbej,ei->mbij',ints['vB']['ovvo'],cc_t['t1a'],optimize=True)
    h2B_ovoo -= np.einsum('mnij,bn->mbij',ints['vB']['oooo'],cc_t['t1b'],optimize=True)
    h2B_ovoo -= np.einsum('mnif,bn,fj->mbij',ints['vB']['ooov'],cc_t['t1b'],cc_t['t1b'],optimize=True)
    h2B_ovoo -= np.einsum('mnej,bn,ei->mbij',ints['vB']['oovo'],cc_t['t1b'],cc_t['t1a'],optimize=True)
    h2B_ovoo += np.einsum('mbef,fj,ei->mbij',ints['vB']['ovvv'],cc_t['t1b'],cc_t['t1a'],optimize=True)
    
    h2B_vooo = 0.0
    h2B_vooo += ints['vB']['vooo']
    h2B_vooo += np.einsum('amif,fj->amij',ints['vB']['voov'],cc_t['t1b'],optimize=True)
    h2B_vooo -= np.einsum('nmef,an,ei,fj->amij',ints['vB']['oovv'],cc_t['t1a'],cc_t['t1a'],cc_t['t1b'],optimize=True)
    h2B_vooo += np.einsum('amef,fj,ei->amij',ints['vB']['vovv'],cc_t['t1b'],cc_t['t1a'],optimize=True)

    h2B_vvvo = 0.0
    h2B_vvvo += ints['vB']['vvvo']
    h2B_vvvo += np.einsum('abef,fj->abej',ints['vB']['vvvv'],cc_t['t1b'],optimize=True)
    h2B_vvvo -= np.einsum('anej,bn->abej',ints['vB']['vovo'],cc_t['t1b'],optimize=True)
    
    h2B_vvov = 0.0
    h2B_vvov += ints['vB']['vvov']
    h2B_vvov -= np.einsum('mbie,am->abie',ints['vB']['ovov'],cc_t['t1a'],optimize=True)

    h2B_ooov = 0.0
    h2B_ooov += ints['vB']['ooov']
    h2B_ooov += np.einsum('mnfe,fi->mnie',ints['vB']['oovv'],cc_t['t1a'],optimize=True)

    h2B_oovo = 0.0
    h2B_oovo += ints['vB']['oovo']
    h2B_oovo += np.einsum('nmef,fi->nmei',ints['vB']['oovv'],cc_t['t1b'],optimize=True)

    h2B_vovv = 0.0
    h2B_vovv += ints['vB']['vovv']
    h2B_vovv -= np.einsum('nmef,an->amef',ints['vB']['oovv'],cc_t['t1a'],optimize=True)

    h2B_ovvv = 0.0
    h2B_ovvv += ints['vB']['ovvv']
    h2B_ovvv -= np.einsum('mnfe,an->mafe',ints['vB']['oovv'],cc_t['t1b'],optimize=True)
    
    h2C_oooo = 0.0
    h2C_oooo += ints['vC']['oooo']
    h2C_oooo += np.einsum('mnie,ej->mnij',ints['vC']['ooov'],cc_t['t1b'],optimize=True) # ij
    h2C_oooo -= np.einsum('mnje,ei->mnij',ints['vC']['ooov'],cc_t['t1b'],optimize=True) # ji
    h2C_oooo += np.einsum('mnef,ei,fj->mnij',ints['vC']['oovv'],cc_t['t1b'],cc_t['t1b'],optimize=True)

    h2C_vvvv = 0.0
    h2C_vvvv += ints['vC']['vvvv']
    h2C_vvvv -= np.einsum('mbef,am->abef',ints['vC']['ovvv'],cc_t['t1b'],optimize=True) # ab
    h2C_vvvv += np.einsum('maef,bm->abef',ints['vC']['ovvv'],cc_t['t1b'],optimize=True) # ba
    h2C_vvvv += np.einsum('mnef,bn,am->abef',ints['vC']['oovv'],cc_t['t1b'],cc_t['t1b'],optimize=True)
           
    h2C_voov = 0.0
    h2C_voov += ints['vC']['voov']
    h2C_voov -= np.einsum('mnei,an->amie',ints['vC']['oovo'],cc_t['t1b'],optimize=True)
    h2C_voov += np.einsum('amfe,fi->amie',ints['vC']['vovv'],cc_t['t1b'],optimize=True)
    h2C_voov -= np.einsum('mnef,fi,an->amie',ints['vC']['oovv'],cc_t['t1b'],cc_t['t1b'],optimize=True)

    h2C_ovoo = 0.0
    h2C_ovoo += ints['vC']['ovoo']
    h2C_ovoo -= 0.5*np.einsum('mnij,bn->mbij',ints['vC']['oooo'],cc_t['t1b'],optimize=True)
    h2C_ovoo += np.einsum('mbef,ei,fj->mbij',ints['vC']['ovvv'],cc_t['t1b'],cc_t['t1b'],optimize=True)
    h2C_ovoo -= 0.5*np.einsum('mnef,fj,ei,bn->mbij',ints['vC']['oovv'],cc_t['t1b'],cc_t['t1b'],cc_t['t1b'],optimize=True)
    h2C_ovoo += np.einsum('mbif,fj->mbij',ints['vC']['ovov'],cc_t['t1b'],optimize=True)
    h2C_ovoo -= np.einsum('mbjf,fi->mbij',ints['vC']['ovov'],cc_t['t1b'],optimize=True)

    h2C_vvvo = 0.0
    h2C_vvvo += ints['vC']['vvvo']
    h2C_vvvo += 0.5*np.einsum('abef,fj->abej',ints['vC']['vvvv'],cc_t['t1b'],optimize=True)
    h2C_vvvo += np.einsum('mnej,am,bn->abej',ints['vC']['oovo'],cc_t['t1b'],cc_t['t1b'],optimize=True)

    h2C_ooov = 0.0
    h2C_ooov += ints['vC']['ooov']
    h2C_ooov += np.einsum('mnfe,fi->mnie',ints['vC']['oovv'],cc_t['t1b'],optimize=True)

    h2C_vovv = 0.0
    h2C_vovv += ints['vC']['vovv'] 
    h2C_vovv -= np.einsum('mnfe,an->amef',ints['vC']['oovv'],cc_t['t1b'],optimize=True)


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

    h1A_ov = 0.0
    h1A_ov += ints['fA']['ov']
    h1A_ov += np.einsum('imae,em->ia',ints['vA']['oovv'],cc_t['t1a'],optimize=True)
    h1A_ov += np.einsum('imae,em->ia',ints['vB']['oovv'],cc_t['t1b'],optimize=True)
       
    h1A_oo = 0.0
    h1A_oo += ints['fA']['oo']
    h1A_oo += np.einsum('je,ei->ji',h1A_ov,cc_t['t1a'],optimize=True)
    h1A_oo += np.einsum('jmie,em->ji',ints['vA']['ooov'],cc_t['t1a'],optimize=True)
    h1A_oo += np.einsum('jmie,em->ji',ints['vB']['ooov'],cc_t['t1b'],optimize=True)
    h1A_oo += 0.5*np.einsum('jnef,efin->ji',ints['vA']['oovv'],cc_t['t2a'],optimize=True)
    h1A_oo += np.einsum('jnef,efin->ji',ints['vB']['oovv'],cc_t['t2b'],optimize=True)

    h1A_vv = 0.0
    h1A_vv += ints['fA']['vv']
    h1A_vv -= np.einsum('mb,am->ab',h1A_ov,cc_t['t1a'],optimize=True)
    h1A_vv += np.einsum('ambe,em->ab',ints['vA']['vovv'],cc_t['t1a'],optimize=True)
    h1A_vv += np.einsum('ambe,em->ab',ints['vB']['vovv'],cc_t['t1b'],optimize=True)
    h1A_vv -= 0.5*np.einsum('mnbf,afmn->ab',ints['vA']['oovv'],cc_t['t2a'],optimize=True)
    h1A_vv -= np.einsum('mnbf,afmn->ab',ints['vB']['oovv'],cc_t['t2b'],optimize=True)

    h1B_ov = 0.0
    h1B_ov += ints['fB']['ov']
    h1B_ov += np.einsum('imae,em->ia',ints['vC']['oovv'],cc_t['t1b'],optimize=True)
    h1B_ov += np.einsum('miea,em->ia',ints['vB']['oovv'],cc_t['t1a'],optimize=True)

    h1B_oo = 0.0
    h1B_oo += ints['fB']['oo']
    h1B_oo += np.einsum('je,ei->ji',h1B_ov,cc_t['t1b'],optimize=True)
    h1B_oo += np.einsum('jmie,em->ji',ints['vC']['ooov'],cc_t['t1b'],optimize=True)
    h1B_oo += np.einsum('mjei,em->ji',ints['vB']['oovo'],cc_t['t1a'],optimize=True)
    h1B_oo += 0.5*np.einsum('jnef,efin->ji',ints['vC']['oovv'],cc_t['t2c'],optimize=True)
    h1B_oo += np.einsum('njfe,feni->ji',ints['vB']['oovv'],cc_t['t2b'],optimize=True)

    h1B_vv = 0.0
    h1B_vv += ints['fB']['vv']
    h1B_vv -= np.einsum('mb,am->ab',h1B_ov,cc_t['t1b'],optimize=True)
    h1B_vv += np.einsum('ambe,em->ab',ints['vC']['vovv'],cc_t['t1b'],optimize=True)
    h1B_vv += np.einsum('maeb,em->ab',ints['vB']['ovvv'],cc_t['t1a'],optimize=True)
    h1B_vv -= 0.5*np.einsum('mnbf,afmn->ab',ints['vC']['oovv'],cc_t['t2c'],optimize=True)
    h1B_vv -= np.einsum('nmfb,fanm->ab',ints['vB']['oovv'],cc_t['t2b'],optimize=True)

    Q1 = -np.einsum('mnfe,an->amef',ints['vA']['oovv'],cc_t['t1a'],optimize=True)
    I2A_vovv = ints['vA']['vovv'] + 0.5*Q1
    h2A_vovv = I2A_vovv + 0.5*Q1
    
    Q1 = -np.einsum('mnfe,an->amef',ints['vA']['oovv'],cc_t['t1a'],optimize=True)
    I2A_vovv = ints['vA']['vovv'] + 0.5*Q1
    h2A_vovv = I2A_vovv + 0.5*Q1

    Q1 = np.einsum('mnfe,fi->mnie',ints['vA']['oovv'],cc_t['t1a'],optimize=True)
    I2A_ooov = ints['vA']['ooov'] + 0.5*Q1
    h2A_ooov = I2A_ooov + 0.5*Q1

    Q1 = -np.einsum('nmef,an->amef',ints['vB']['oovv'],cc_t['t1a'],optimize=True)
    I2B_vovv = ints['vB']['vovv'] + 0.5*Q1
    h2B_vovv = I2B_vovv + 0.5*Q1

    Q1 = np.einsum('mnfe,fi->mnie',ints['vB']['oovv'],cc_t['t1a'],optimize=True)
    I2B_ooov = ints['vB']['ooov'] + 0.5*Q1
    h2B_ooov = I2B_ooov + 0.5*Q1

    Q1 = -np.einsum('mnef,an->maef',ints['vB']['oovv'],cc_t['t1b'],optimize=True)
    I2B_ovvv = ints['vB']['ovvv'] + 0.5*Q1
    h2B_ovvv = I2B_ovvv + 0.5*Q1

    Q1 = np.einsum('nmef,fi->nmei',ints['vB']['oovv'],cc_t['t1b'],optimize=True)
    I2B_oovo = ints['vB']['oovo'] + 0.5*Q1
    h2B_oovo = I2B_oovo + 0.5*Q1

    Q1 = -np.einsum('nmef,an->amef',ints['vC']['oovv'],cc_t['t1b'],optimize=True)
    I2C_vovv = ints['vC']['vovv'] + 0.5*Q1
    h2C_vovv = I2C_vovv + 0.5*Q1

    Q1 = np.einsum('mnfe,fi->mnie',ints['vC']['oovv'],cc_t['t1b'],optimize=True)
    I2C_ooov = ints['vC']['ooov'] + 0.5*Q1
    h2C_ooov = I2C_ooov + 0.5*Q1

    Q1 = -np.einsum('bmfe,am->abef',I2A_vovv,cc_t['t1a'],optimize=True)
    Q1 -= np.einsum('abef->baef',Q1,optimize=True)
    h2A_vvvv = 0.0
    h2A_vvvv += ints['vA']['vvvv']
    h2A_vvvv += 0.5*np.einsum('mnef,abmn->abef',ints['vA']['oovv'],cc_t['t2a'],optimize=True)
    h2A_vvvv += Q1

    h2B_vvvv = 0.0
    h2B_vvvv += ints['vB']['vvvv']
    h2B_vvvv -= np.einsum('mbef,am->abef',I2B_ovvv,cc_t['t1a'],optimize=True)
    h2B_vvvv -= np.einsum('amef,bm->abef',I2B_vovv,cc_t['t1b'],optimize=True)
    h2B_vvvv += np.einsum('mnef,abmn->abef',ints['vB']['oovv'],cc_t['t2b'],optimize=True)

    Q1 = -np.einsum('bmfe,am->abef',I2C_vovv,cc_t['t1b'],optimize=True)
    Q1 -= np.einsum('abef->baef',Q1,optimize=True)
    h2C_vvvv = 0.0
    h2C_vvvv += ints['vC']['vvvv']
    h2C_vvvv += 0.5*np.einsum('mnef,abmn->abef',ints['vC']['oovv'],cc_t['t2c'],optimize=True)
    h2C_vvvv += Q1

    Q1 = +np.einsum('nmje,ei->mnij',I2A_ooov,cc_t['t1a'],optimize=True)
    Q1 -= np.einsum('mnij->mnji',Q1,optimize=True)
    h2A_oooo = 0.0
    h2A_oooo += ints['vA']['oooo']
    h2A_oooo += 0.5*np.einsum('mnef,efij->mnij',ints['vA']['oovv'],cc_t['t2a'],optimize=True)
    h2A_oooo += Q1

    h2B_oooo = 0.0
    h2B_oooo += ints['vB']['oooo']
    h2B_oooo += np.einsum('mnej,ei->mnij',I2B_oovo,cc_t['t1a'],optimize=True)
    h2B_oooo += np.einsum('mnie,ej->mnij',I2B_ooov,cc_t['t1b'],optimize=True)
    h2B_oooo += np.einsum('mnef,efij->mnij',ints['vB']['oovv'],cc_t['t2b'],optimize=True)

    Q1 = +np.einsum('nmje,ei->mnij',I2C_ooov,cc_t['t1b'],optimize=True)
    Q1 -= np.einsum('mnij->mnji',Q1,optimize=True)
    h2C_oooo = 0.0
    h2C_oooo += ints['vC']['oooo']
    h2C_oooo += 0.5*np.einsum('mnef,efij->mnij',ints['vC']['oovv'],cc_t['t2c'],optimize=True)
    h2C_oooo += Q1

    h2A_voov = 0.0
    h2A_voov += ints['vA']['voov']
    h2A_voov += np.einsum('amfe,fi->amie',I2A_vovv,cc_t['t1a'],optimize=True)
    h2A_voov -= np.einsum('nmie,an->amie',I2A_ooov,cc_t['t1a'],optimize=True)
    h2A_voov += np.einsum('nmfe,afin->amie',ints['vA']['oovv'],cc_t['t2a'],optimize=True)
    h2A_voov += np.einsum('mnef,afin->amie',ints['vB']['oovv'],cc_t['t2b'],optimize=True)

    h2B_voov = 0.0
    h2B_voov += ints['vB']['voov']
    h2B_voov += np.einsum('amfe,fi->amie',I2B_vovv,cc_t['t1a'],optimize=True)
    h2B_voov -= np.einsum('nmie,an->amie',I2B_ooov,cc_t['t1a'],optimize=True)
    h2B_voov += np.einsum('nmfe,afin->amie',ints['vB']['oovv'],cc_t['t2a'],optimize=True)
    h2B_voov += np.einsum('nmfe,afin->amie',ints['vC']['oovv'],cc_t['t2b'],optimize=True)

    h2B_ovvo = 0.0
    h2B_ovvo += ints['vB']['ovvo']
    h2B_ovvo += np.einsum('maef,fi->maei',I2B_ovvv,cc_t['t1b'],optimize=True)
    h2B_ovvo -= np.einsum('mnei,an->maei',I2B_oovo,cc_t['t1b'],optimize=True)
    h2B_ovvo += np.einsum('mnef,afin->maei',ints['vB']['oovv'],cc_t['t2c'],optimize=True)
    h2B_ovvo += np.einsum('mnef,fani->maei',ints['vA']['oovv'],cc_t['t2b'],optimize=True)

    h2B_ovov = 0.0
    h2B_ovov += ints['vB']['ovov']
    h2B_ovov += np.einsum('mafe,fi->maie',I2B_ovvv,cc_t['t1a'],optimize=True)
    h2B_ovov -= np.einsum('mnie,an->maie',I2B_ooov,cc_t['t1b'],optimize=True)
    h2B_ovov -= np.einsum('mnfe,fain->maie',ints['vB']['oovv'],cc_t['t2b'],optimize=True)

    h2B_vovo = 0.0
    h2B_vovo += ints['vB']['vovo']
    h2B_vovo -= np.einsum('nmei,an->amei',I2B_oovo,cc_t['t1a'],optimize=True)
    h2B_vovo += np.einsum('amef,fi->amei',I2B_vovv,cc_t['t1b'],optimize=True)
    h2B_vovo -= np.einsum('nmef,afni->amei',ints['vB']['oovv'],cc_t['t2b'],optimize=True)

    h2C_voov = 0.0
    h2C_voov += ints['vC']['voov']
    h2C_voov += np.einsum('amfe,fi->amie',I2C_vovv,cc_t['t1b'],optimize=True)
    h2C_voov -= np.einsum('nmie,an->amie',I2C_ooov,cc_t['t1b'],optimize=True)
    h2C_voov += np.einsum('nmfe,afin->amie',ints['vC']['oovv'],cc_t['t2c'],optimize=True)
    h2C_voov += np.einsum('nmfe,fani->amie',ints['vB']['oovv'],cc_t['t2b'],optimize=True)


    Q1 = +np.einsum('mnjf,afin->amij',h2A_ooov,cc_t['t2a'],optimize=True)+np.einsum('mnjf,afin->amij',h2B_ooov,cc_t['t2b'],optimize=True)
    Q2 = ints['vA']['voov'] + 0.5*np.einsum('amef,ei->amif',ints['vA']['vovv'],cc_t['t1a'],optimize=True)
    Q2 = np.einsum('amif,fj->amij',Q2,cc_t['t1a'],optimize=True)
    Q1 += Q2
    Q1 -= np.einsum('amij->amji',Q1,optimize=True)
    h2A_vooo = 0.0
    h2A_vooo += ints['vA']['vooo']
    h2A_vooo += np.einsum('me,aeij->amij',h1A_ov,cc_t['t2a'],optimize=True)
    h2A_vooo -= np.einsum('nmij,an->amij',h2A_oooo,cc_t['t1a'],optimize=True)
    h2A_vooo += 0.5*np.einsum('amef,efij->amij',ints['vA']['vovv'],cc_t['t2a'],optimize=True)
    h2A_vooo += Q1

    Q1 = ints['vB']['voov']+np.einsum('amfe,fi->amie',ints['vB']['vovv'],cc_t['t1a'],optimize=True)
    h2B_vooo = 0.0
    h2B_vooo += ints['vB']['vooo']
    h2B_vooo += np.einsum('me,aeij->amij',h1B_ov,cc_t['t2b'],optimize=True)
    h2B_vooo -= np.einsum('nmij,an->amij',h2B_oooo,cc_t['t1a'],optimize=True)
    h2B_vooo += np.einsum('mnjf,afin->amij',h2C_ooov,cc_t['t2b'],optimize=True)
    h2B_vooo += np.einsum('nmfj,afin->amij',h2B_oovo,cc_t['t2a'],optimize=True)
    h2B_vooo -= np.einsum('nmif,afnj->amij',h2B_ooov,cc_t['t2b'],optimize=True)
    h2B_vooo += np.einsum('amej,ei->amij',ints['vB']['vovo'],cc_t['t1a'],optimize=True)
    h2B_vooo += np.einsum('amie,ej->amij',Q1,cc_t['t1b'],optimize=True)
    h2B_vooo += np.einsum('amef,efij->amij',ints['vB']['vovv'],cc_t['t2b'],optimize=True)

    Q1 = ints['vB']['ovov']+np.einsum('mafe,fj->maje',ints['vB']['ovvv'],cc_t['t1a'],optimize=True)
    h2B_ovoo = 0.0
    h2B_ovoo += ints['vB']['ovoo']
    h2B_ovoo += np.einsum('me,eaji->maji',h1A_ov,cc_t['t2b'],optimize=True)
    h2B_ovoo -= np.einsum('mnji,an->maji',h2B_oooo,cc_t['t1b'],optimize=True)
    h2B_ovoo += np.einsum('mnjf,fani->maji',h2A_ooov,cc_t['t2b'],optimize=True)
    h2B_ovoo += np.einsum('mnjf,fani->maji',h2B_ooov,cc_t['t2c'],optimize=True)
    h2B_ovoo -= np.einsum('mnfi,fajn->maji',h2B_oovo,cc_t['t2b'],optimize=True)
    h2B_ovoo += np.einsum('maje,ei->maji',Q1,cc_t['t1b'],optimize=True)
    h2B_ovoo += np.einsum('maei,ej->maji',ints['vB']['ovvo'],cc_t['t1a'],optimize=True)
    h2B_ovoo += np.einsum('mafe,feji->maji',ints['vB']['ovvv'],cc_t['t2b'],optimize=True)

    Q1 = np.einsum('mnjf,afin->amij',h2C_ooov,cc_t['t2c'],optimize=True)+np.einsum('nmfj,fani->amij',h2B_oovo,cc_t['t2b'],optimize=True)
    Q2 = ints['vC']['voov'] + 0.5*np.einsum('amef,ei->amif',ints['vC']['vovv'],cc_t['t1b'],optimize=True)
    Q2 = np.einsum('amif,fj->amij',Q2,cc_t['t1b'],optimize=True)
    Q1 += Q2
    Q1 -= np.einsum('amij->amji',Q1,optimize=True)
    h2C_vooo = 0.0
    h2C_vooo += ints['vC']['vooo']
    h2C_vooo += np.einsum('me,aeij->amij',h1B_ov,cc_t['t2c'],optimize=True)
    h2C_vooo -= np.einsum('nmij,an->amij',h2C_oooo,cc_t['t1b'],optimize=True)
    h2C_vooo += 0.5*np.einsum('amef,efij->amij',ints['vC']['vovv'],cc_t['t2c'],optimize=True)
    h2C_vooo += Q1

    Q1 = np.einsum('bnef,afin->abie',h2A_vovv,cc_t['t2a'],optimize=True)+np.einsum('bnef,afin->abie',h2B_vovv,cc_t['t2b'],optimize=True)
    Q2 = ints['vA']['ovov'] - 0.5*np.einsum('mnie,bn->mbie',ints['vA']['ooov'],cc_t['t1a'],optimize=True)
    Q2 = -np.einsum('mbie,am->abie',Q2,cc_t['t1a'],optimize=True)
    Q1 += Q2
    Q1 -= np.einsum('abie->baie',Q1,optimize=True)
    h2A_vvov = 0.0
    h2A_vvov += ints['vA']['vvov']
    h2A_vvov -= np.einsum('me,abim->abie',h1A_ov,cc_t['t2a'],optimize=True)
    h2A_vvov += np.einsum('abfe,fi->abie',h2A_vvvv,cc_t['t1a'],optimize=True)
    h2A_vvov += 0.5*np.einsum('mnie,abmn->abie',ints['vA']['ooov'],cc_t['t2a'],optimize=True)
    h2A_vvov += Q1

    Q1 = ints['vB']['ovov'] - np.einsum('mnie,bn->mbie',ints['vB']['ooov'],cc_t['t1b'],optimize=True)
    Q1 = -np.einsum('mbie,am->abie',Q1,cc_t['t1a'],optimize=True)
    h2B_vvov = 0.0
    h2B_vvov += ints['vB']['vvov']
    h2B_vvov -= np.einsum('me,abim->abie',h1B_ov,cc_t['t2b'],optimize=True)
    h2B_vvov += np.einsum('abfe,fi->abie',h2B_vvvv,cc_t['t1a'],optimize=True)
    h2B_vvov += np.einsum('nbfe,afin->abie',h2B_ovvv,cc_t['t2a'],optimize=True)
    h2B_vvov += np.einsum('bnef,afin->abie',h2C_vovv,cc_t['t2b'],optimize=True)
    h2B_vvov -= np.einsum('amfe,fbim->abie',h2B_vovv,cc_t['t2b'],optimize=True)
    h2B_vvov += Q1
    h2B_vvov -= np.einsum('amie,bm->abie',ints['vB']['voov'],cc_t['t1b'],optimize=True)
    h2B_vvov += np.einsum('nmie,abnm->abie',ints['vB']['ooov'],cc_t['t2b'],optimize=True)

    Q1 = ints['vB']['vovo'] - np.einsum('nmei,bn->bmei',ints['vB']['oovo'],cc_t['t1a'],optimize=True)
    Q1 = -np.einsum('bmei,am->baei',Q1,cc_t['t1b'],optimize=True)
    h2B_vvvo = 0.0
    h2B_vvvo += ints['vB']['vvvo']
    h2B_vvvo -= np.einsum('me,bami->baei',h1A_ov,cc_t['t2b'],optimize=True)
    h2B_vvvo += np.einsum('baef,fi->baei',h2B_vvvv,cc_t['t1b'],optimize=True)
    h2B_vvvo += np.einsum('bnef,fani->baei',h2A_vovv,cc_t['t2b'],optimize=True)
    h2B_vvvo += np.einsum('bnef,fani->baei',h2B_vovv,cc_t['t2c'],optimize=True)
    h2B_vvvo -= np.einsum('maef,bfmi->baei',h2B_ovvv,cc_t['t2b'],optimize=True)
    h2B_vvvo += Q1
    h2B_vvvo -= np.einsum('naei,bn->baei',ints['vB']['ovvo'],cc_t['t1a'],optimize=True)
    h2B_vvvo += np.einsum('nmei,banm->baei',ints['vB']['oovo'],cc_t['t2b'],optimize=True)

    Q1 = +np.einsum('bnef,afin->abie',h2C_vovv,cc_t['t2c'],optimize=True)+np.einsum('nbfe,fani->abie',h2B_ovvv,cc_t['t2b'],optimize=True)
    Q2 = ints['vC']['ovov'] - 0.5*np.einsum('mnie,bn->mbie',ints['vC']['ooov'],cc_t['t1b'],optimize=True)
    Q2 = -np.einsum('mbie,am->abie',Q2,cc_t['t1b'],optimize=True)
    Q1 += Q2
    Q1 -= np.einsum('abie->baie',Q1,optimize=True)
    h2C_vvov = 0.0
    h2C_vvov += ints['vC']['vvov']
    h2C_vvov -= np.einsum('me,abim->abie',h1B_ov,cc_t['t2c'],optimize=True)
    h2C_vvov += np.einsum('abfe,fi->abie',h2C_vvvv,cc_t['t1b'],optimize=True)
    h2C_vvov += 0.5*np.einsum('mnie,abmn->abie',ints['vC']['ooov'],cc_t['t2c'],optimize=True)
    h2C_vvov += Q1

    H1A = {'ov' : h1A_ov, 'oo' : h1A_oo, 'vv' : h1A_vv}

    H1B = {'ov' : h1B_ov, 'oo' : h1B_oo, 'vv' : h1B_vv}

    H2A = {'vovv' : h2A_vovv, 'ooov' : h2A_ooov, 'vvvv' : h2A_vvvv, 'oooo' : h2A_oooo, 'voov' : h2A_voov, 'vooo' : h2A_vooo, 'vvov' : h2A_vvov}

    H2B = {'vovv' : h2B_vovv, 'ooov' : h2B_ooov, 'ovvv' : h2B_ovvv, 'oovo' : h2B_oovo, 'vvvv' : h2B_vvvv, 'oooo' : h2B_oooo, 'voov' : h2B_voov,
    'ovvo' : h2B_ovvo, 'ovov' : h2B_ovov, 'vovo' : h2B_vovo, 'vooo' : h2B_vooo, 'ovoo' : h2B_ovoo, 'vvov' : h2B_vvov, 'vvvo' : h2B_vvvo}

    H2C = {'vovv' : h2C_vovv, 'ooov' : h2C_ooov, 'vvvv' : h2C_vvvv, 'oooo' : h2C_oooo, 'voov' : h2C_voov, 'vooo' : h2C_vooo, 'vvov' : h2C_vvov}

    return H1A,H1B,H2A,H2B,H2C
