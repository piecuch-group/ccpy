import numpy as np
from solvers import diis
import time


def ccsdt_p(sys,ints,p_spaces,maxit=100,tol=1e-08,diis_size=6,shift=0.0,initial_guess=None,flag_RHF=False):

    print('\n==================================++Entering CCSDT(P) Routine++=================================\n')
    if flag_RHF:
        print('     >>USING RHF SYMMETRY<<')

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

    T = np.zeros(ndim)
    if initial_guess is not None:
        print('USING INITIAL GUESS')
        T[idx_1a] = initial_guess['t1a'].flatten()
        T[idx_1b] = initial_guess['t1b'].flatten()
        T[idx_2a] = initial_guess['t2a'].flatten()
        T[idx_2b] = initial_guess['t2b'].flatten()
        T[idx_2c] = initial_guess['t2c'].flatten()
        T[idx_3a] = initial_guess['t3a'].flatten()
        T[idx_3b] = initial_guess['t3b'].flatten()
        T[idx_3c] = initial_guess['t3c'].flatten()
        T[idx_3d] = initial_guess['t3d'].flatten()

    cc_t = {}
    T_list = np.zeros((ndim,diis_size))
    T_resid_list = np.zeros((ndim,diis_size))
    T_old = np.zeros(ndim)

    # Jacobi/DIIS iterations
    it_micro = 0
    flag_conv = False
    it_macro = 0
    Ecorr_old = 0.0

    t_start = time.time()

    list_of_triples = get_list_of_triples(p_spaces)

    print('Iteration    Residuum               deltaE                 Ecorr')
    print('=============================================================================')
    while it_micro < maxit:

        t_iter_start = time.perf_counter()
        
        # store old T and get current diis dimensions
        T_old = T.copy()
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
        cc_t = update_t1a(cc_t,ints,sys,shift)
        if flag_RHF:
            cc_t['t1b'] = cc_t['t1a']
        else:
            cc_t = update_t1b(cc_t,ints,sys,shift)

        # CCS intermediates
        H1A,H1B,H2A,H2B,H2C = get_ccs_intermediates(cc_t,ints,sys)

        # update T2
        cc_t = update_t2a(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        cc_t = update_t2b(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        if flag_RHF:
            cc_t['t2c'] = cc_t['t2a']
        else:
            cc_t = update_t2c(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)

        # CCSD intermediates
        H1A,H1B,H2A,H2B,H2C = get_ccsd_intermediates(cc_t,ints,sys)

        # update T3 in the P space
        list_of_triples = get_list_of_triples(p_spaces)
        num_triples_A = len(list_of_triples['A'])
        num_triples_B = len(list_of_triples['B'])
        num_triples_C = len(list_of_triples['C'])
        num_triples_D = len(list_of_triples['D'])

        # test t3 updates
        if num_triples_A > 0:
            list_A = np.asarray(list_of_triples['A'])
            I2A_vvov = H2A['vvov'] + np.einsum('me,abim->abie',H1A['ov'],cc_t['t2a'],optimize=True)
            cc_t['t3a'] = f90_ccp_updates_mkl_omp.ccp_loops.update_t3a(cc_t['t2a'],cc_t['t3a'],cc_t['t3b'],list_A,\
                        ints['vA']['oovv'],ints['vB']['oovv'],H1A['oo'],H1A['vv'],H2A['oooo'],\
                        H2A['vvvv'],H2A['voov'],H2B['voov'],H2A['vooo'],I2A_vvov,\
                        ints['fA']['oo'],ints['fA']['vv'],shift,sys['Nocc_a'],sys['Nunocc_a'],\
                        sys['Nocc_b'],sys['Nunocc_b'],num_triples_A)

        if num_triples_B > 0:
            list_B = np.asarray(list_of_triples['B'])
            I2A_vooo = H2A['vooo'] - np.einsum('me,aeij->amij',H1A['ov'],cc_t['t2a'],optimize=True)
            I2B_ovoo = H2B['ovoo'] - np.einsum('me,ecjk->mcjk',H1A['ov'],cc_t['t2b'],optimize=True)
            I2B_vooo = H2B['vooo'] - np.einsum('me,aeik->amik',H1B['ov'],cc_t['t2b'],optimize=True) 
            cc_t['t3b'] = f90_ccp_updates_mkl_omp.ccp_loops.update_t3b(cc_t['t2a'],cc_t['t2b'],cc_t['t3a'],cc_t['t3b'],cc_t['t3c'],\
                        list_B,ints['vA']['oovv'],ints['vB']['oovv'],ints['vC']['oovv'],\
                        H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],H2A['oooo'],H2A['vvvv'],\
                        H2A['voov'],H2B['oooo'],H2B['vvvv'],H2B['voov'],H2B['ovov'],\
                        H2B['vovo'],H2B['ovvo'],H2C['voov'],I2A_vooo,H2A['vvov'],\
                        I2B_vooo,I2B_ovoo,H2B['vvov'],H2B['vvvo'],\
                        ints['fA']['oo'],ints['fA']['vv'],ints['fB']['oo'],ints['fB']['vv'],shift,\
                        sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'],\
                        num_triples_B)

        if num_triples_C > 0:
            list_C = np.asarray(list_of_triples['C'])
            I2B_ovoo = H2B['ovoo'] - np.einsum('me,ebij->mbij',H1A['ov'],cc_t['t2b'],optimize=True)
            I2B_vooo = H2B['vooo'] - np.einsum('me,aeij->amij',H1B['ov'],cc_t['t2b'],optimize=True)
            I2C_vooo = H2C['vooo'] - np.einsum('me,cekj->cmkj',H1B['ov'],cc_t['t2c'],optimize=True)
            cc_t['t3c'] = f90_ccp_updates_mkl_omp.ccp_loops.update_t3c(cc_t['t2b'],cc_t['t2c'],cc_t['t3b'],cc_t['t3c'],cc_t['t3d'],\
                        list_C,ints['vA']['oovv'],ints['vB']['oovv'],ints['vC']['oovv'],\
                        H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],\
                        H2A['voov'],\
                        H2B['oooo'],H2B['vvvv'],H2B['voov'],H2B['ovov'],\
                        H2B['vovo'],H2B['ovvo'],H2C['oooo'],H2C['vvvv'],\
                        H2C['voov'],I2C_vooo,H2C['vvov'],\
                        I2B_vooo,I2B_ovoo,H2B['vvov'],H2B['vvvo'],\
                        ints['fA']['oo'],ints['fA']['vv'],ints['fB']['oo'],ints['fB']['vv'],shift,\
                        sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'],\
                        num_triples_C)
        
        if num_triples_D > 0:
            list_D = np.asarray(list_of_triples['D'])
            I2C_vvov = H2C['vvov'] + np.einsum('me,abim->abie',H1B['ov'],cc_t['t2c'],optimize=True)
            cc_t['t3d'] = f90_ccp_updates_mkl_omp.ccp_loops.update_t3d(cc_t['t2c'],cc_t['t3c'],cc_t['t3d'],list_D,\
                        ints['vB']['oovv'],ints['vC']['oovv'],H1B['oo'],H1B['vv'],H2C['oooo'],\
                        H2C['vvvv'],H2C['voov'],H2B['ovvo'],H2C['vooo'],I2C_vvov,\
                        ints['fB']['oo'],ints['fB']['vv'],shift,sys['Nocc_a'],sys['Nunocc_a'],\
                        sys['Nocc_b'],sys['Nunocc_b'],num_triples_D)
        
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

        t_iter_end = time.perf_counter()

        # build DIIS residual
        T_resid = T - T_old
        
        # change in Ecorr
        deltaE = Ecorr - Ecorr_old

        # check for exit condition
        resid = np.linalg.norm(T_resid)
        if resid < tol and abs(deltaE) < tol:
            flag_conv = True
            break

        # append trial and residual vectors to lists
        T_list[:,it_micro%diis_size] = T
        T_resid_list[:,it_micro%diis_size] = T_resid
        
        if it_micro%diis_size == 0 and it_micro > 1:
            it_macro = it_macro + 1
            print('DIIS Cycle - {}'.format(it_macro))
            T = diis(T_list,T_resid_list)
        
        print('   {}       {:.10f}          {:.10f}          {:.10f}         ({:.3f}s)'.format(it_micro,resid,deltaE,Ecorr,t_iter_end-t_iter_start))
        
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

    return cc_t, ints['Escf'] + Ecorr

def update_t1a(cc_t,ints,sys,shift):

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
    t3a = cc_t['t3a']
    t3b = cc_t['t3b']
    t3c = cc_t['t3c']
    t3d = cc_t['t3d']

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

    t1a = cc_loops.cc_loops.update_t1a(t1a,X1A,fA['oo'],fA['vv'],shift)

    cc_t['t1a'] = t1a

    return cc_t

def update_t1b(cc_t,ints,sys,shift):

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
    t3a = cc_t['t3a']
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
    
    t1b = cc_loops.cc_loops.update_t1b(t1b,X1B,fB['oo'],fB['vv'],shift)

    cc_t['t1b'] = t1b        
    return cc_t


def update_t2a(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift):

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
    t3a = cc_t['t3a']
    t3b = cc_t['t3b']
    t3c = cc_t['t3c']
    t3d = cc_t['t3d']

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

    X2A = 0.0
    X2A += vA['vvoo']
    D1 = -np.einsum('amij,bm->abij',H2A['vooo'],t1a,optimize=True)
    D2 = np.einsum('abie,ej->abij',H2A['vvov'],t1a,optimize=True)
    D3 = np.einsum('ae,ebij->abij',I1A_vv,t2a,optimize=True)
    D4 = -np.einsum('mi,abmj->abij',I1A_oo,t2a,optimize=True)
    D5 = np.einsum('amie,ebmj->abij',I2A_voov,t2a,optimize=True)
    D6 = np.einsum('amie,bejm->abij',I2B_voov,t2b,optimize=True)
    X2A += 0.5*np.einsum('abef,efij->abij',H2A['vvvv'],t2a,optimize=True)
    X2A += 0.5*np.einsum('mnij,abmn->abij',I2A_oooo,t2a,optimize=True)
    
    # CCSDT contribution
    X2A += np.einsum('me,abeijm->abij',H1A['ov'],t3a,optimize=True)
    X2A += np.einsum('me,abeijm->abij',H1B['ov'],t3b,optimize=True)
    Q3 = -np.einsum('mnif,abfmjn->abij',H2B['ooov'],t3b,optimize=True)
    Q4 = -0.5*np.einsum('mnif,abfmjn->abij',H2A['ooov'],t3a,optimize=True)
    Q5 = 0.5*np.einsum('anef,ebfijn->abij',H2A['vovv'],t3a,optimize=True)
    Q6 = np.einsum('anef,ebfijn->abij',H2B['vovv'],t3b,optimize=True)

    # diagrams that have A(ab)
    D_ab = D1 + D3 + Q5 + Q6
    D_ab -= np.einsum('abij->baij',D_ab,optimize=True)
    
    # diagrams that have A(ij)
    D_ij = D2 + D4 + Q3 + Q4
    D_ij = D_ij - np.einsum('abij->abji',D_ij,optimize=True)
        
    # diagrams that have A(ab)A(ij)
    D56 = D5 + D6
    D56 = D56 - np.einsum('abij->baij',D56,optimize=True) - np.einsum('abij->abji',D56,optimize=True) + np.einsum('abij->baji',D56,optimize=True)
    
    # total contribution
    X2A += D_ij + D_ab + D56

    t2a = cc_loops.cc_loops.update_t2a(t2a,X2A,fA['oo'],fA['vv'],shift)

    cc_t['t2a'] = t2a
    return cc_t

def update_t2b(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift):

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
    t3a = cc_t['t3a']
    t3b = cc_t['t3b']
    t3c = cc_t['t3c']
    t3d = cc_t['t3d']

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

    # CCSDT contribution
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

    t2b = cc_loops.cc_loops.update_t2b(t2b,X2B,fA['oo'],fA['vv'],fB['oo'],fB['vv'],shift)

    cc_t['t2b'] = t2b
    return cc_t

def update_t2c(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift):

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
    t3a = cc_t['t3a']
    t3b = cc_t['t3b']
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
    
    X2C = 0.0
    X2C += vC['vvoo']
    D1 = -np.einsum('mbij,am->abij',H2C['ovoo'],t1b,optimize=True)
    D2 = np.einsum('abej,ei->abij',H2C['vvvo'],t1b,optimize=True)
    D3 = np.einsum('ae,ebij->abij',I1B_vv,t2c,optimize=True)
    D4 = -np.einsum('mi,abmj->abij',I1B_oo,t2c,optimize=True)
    D5 = np.einsum('amie,ebmj->abij',I2C_voov,t2c,optimize=True)
    D6 = np.einsum('maei,ebmj->abij',I2B_ovvo,t2b,optimize=True)
    X2C += 0.5*np.einsum('abef,efij->abij',H2C['vvvv'],t2c,optimize=True)
    X2C += 0.5*np.einsum('mnij,abmn->abij',I2C_oooo,t2c,optimize=True)

    # CCSDT contribution
    X2C += np.einsum('me,eabmij->abij',H1A['ov'],t3c,optimize=True)
    X2C += np.einsum('me,abeijm->abij',H1B['ov'],t3d,optimize=True)
    Q3 = 0.5*np.einsum('anef,ebfijn->abij',H2C['vovv'],t3d,optimize=True)
    Q4 = np.einsum('nafe,febnij->abij',H2B['ovvv'],t3c,optimize=True)
    Q5 = -0.5*np.einsum('mnif,abfmjn->abij',H2C['ooov'],t3d,optimize=True)
    Q6 = -np.einsum('nmfi,fabnmj->abij',H2B['oovo'],t3c,optimize=True)
    
    # diagrams that have A(ab)
    D_ab = D1 + D3 + Q3 + Q4
    D_ab -= np.einsum('abij->baij',D_ab,optimize=True)
    
    # diagrams that have A(ij)
    D_ij = D2 + D4 + Q5 + Q6
    D_ij -= np.einsum('abij->abji',D_ij,optimize=True)
        
    # diagrams that have A(ab)A(ij)
    D56 = D5 + D6
    D56 = D56 - np.einsum('abij->baij',D56,optimize=True) - np.einsum('abij->abji',D56,optimize=True) + np.einsum('abij->baji',D56,optimize=True)
    
    # total contribution
    X2C += D_ij + D_ab + D56

    t2c = cc_loops.cc_loops.update_t2c(t2c,X2C,fB['oo'],fB['vv'],shift)

    cc_t['t2c'] = t2c
    return cc_t

def get_ccs_intermediates(cc_t,ints,sys):

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

def get_list_of_triples(p_spaces):

    noa = p_spaces['A'].shape[3]
    nua = p_spaces['A'].shape[0]
    nob = p_spaces['D'].shape[3]
    nub = p_spaces['D'].shape[0]

    list_of_triples_A = []
    for a in range(nua):
        for b in range(a+1,nua):
            for c in range(b+1,nua):
                for i in range(noa):
                    for j in range(i+1,noa):
                        for k in range(j+1,noa):
                            if p_spaces['A'][a,b,c,i,j,k] == 1:
                                list_of_triples_A.append([a, b, c, i, j, k])

    list_of_triples_B = []
    for a in range(nua):
        for b in range(a+1,nua):
            for c in range(nub):
                for i in range(noa):
                    for j in range(i+1,noa):
                        for k in range(nob):
                            if p_spaces['B'][a,b,c,i,j,k] == 1:
                                list_of_triples_B.append([a, b, c, i, j, k])
                            
    list_of_triples_C = []
    for a in range(nua):
        for b in range(nub):
            for c in range(b+1,nub):
                for i in range(noa):
                    for j in range(nob):
                        for k in range(j+1,nob):
                            if p_spaces['C'][a,b,c,i,j,k] == 1:
                                list_of_triples_C.append([a, b, c, i, j, k])

    list_of_triples_D = []
    for a in range(nub):
        for b in range(a+1,nub):
            for c in range(b+1,nub):
                for i in range(nob):
                    for j in range(i+1,nob):
                        for k in range(j+1,nob):
                            if p_spaces['D'][a,b,c,i,j,k] == 1:
                                list_of_triples_D.append([a, b, c, i, j, k])

    list_of_triples = {'A' : list_of_triples_A,\
                        'B' : list_of_triples_B,\
                        'C' : list_of_triples_C,\
                        'D' : list_of_triples_D}
    return list_of_triples








