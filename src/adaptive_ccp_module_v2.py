import numpy as np
from solvers import diis
from cc_energy import calc_cc_energy
import time
import cc_loops

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
        cc_t = update_t3a_inloop(cc_t,ints,list_of_triples['A'],H1A,H1B,H2A,H2B,H2C,sys,shift)
        cc_t = update_t3b_inloop(cc_t,ints,list_of_triples['B'],H1A,H1B,H2A,H2B,H2C,sys,shift)
        if flag_RHF:
            cc_t['t3c'] = np.transpose(cc_t['t3b'],(2,0,1,5,3,4))
            cc_t['t3d'] = cc_t['t3a']
        else:
            cc_t = update_t3c_inloop(cc_t,ints,list_of_triples['C'],H1A,H1B,H2A,H2B,H2C,sys,shift)
            cc_t = update_t3d_inloop(cc_t,ints,list_of_triples['D'],H1A,H1B,H2A,H2B,H2C,sys,shift)
        
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

def update_t3a(cc_t,ints,p_space,H1A,H1B,H2A,H2B,H2C,sys,shift):

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

    I2A_vvov = 0.0
    I2A_vvov -= 0.5*np.einsum('mnef,abfimn->abie',vA['oovv'],t3a,optimize=True)
    I2A_vvov -= np.einsum('mnef,abfimn->abie',vB['oovv'],t3b,optimize=True)
    I2A_vvov += np.einsum('me,abim->abie',H1A['ov'],t2a,optimize=True)
    I2A_vvov += H2A['vvov']
             
    I2A_vooo = 0.0
    I2A_vooo += 0.5*np.einsum('mnef,aefijn->amij',vA['oovv'],t3a,optimize=True)
    I2A_vooo += np.einsum('mnef,aefijn->amij',vB['oovv'],t3b,optimize=True)
    I2A_vooo += H2A['vooo']   
    
    # MM(2,3)A
    M23_D1 = 0.0
    M23_D1 -= np.einsum('amij,bcmk->abcijk',I2A_vooo,t2a,optimize=True) 

    M23_D1 += -np.einsum('abcijk->abckji',M23_D1,optimize=True) \
    -np.einsum('abcijk->abcikj',M23_D1,optimize=True) \
    -np.einsum('abcijk->cbaijk',M23_D1,optimize=True) \
    -np.einsum('abcijk->bacijk',M23_D1,optimize=True) \
    +np.einsum('abcijk->backji',M23_D1,optimize=True) \
    +np.einsum('abcijk->cbakji',M23_D1,optimize=True) \
    +np.einsum('abcijk->bacikj',M23_D1,optimize=True) \
    +np.einsum('abcijk->cbaikj',M23_D1,optimize=True)

    M23_D2 = 0.0
    M23_D2 += np.einsum('abie,ecjk->abcijk',I2A_vvov,t2a,optimize=True)

    M23_D2 += -np.einsum('abcijk->abcjik',M23_D2,optimize=True)\
    -np.einsum('abcijk->abckji',M23_D2,optimize=True)\
    -np.einsum('abcijk->cbaijk',M23_D2,optimize=True)\
    -np.einsum('abcijk->acbijk',M23_D2,optimize=True)\
    +np.einsum('abcijk->cbajik',M23_D2,optimize=True)\
    +np.einsum('abcijk->acbjik',M23_D2,optimize=True)\
    +np.einsum('abcijk->cbakji',M23_D2,optimize=True)\
    +np.einsum('abcijk->acbkji',M23_D2,optimize=True)\

    MM23A = M23_D1 + M23_D2

    # (HBar*T3)_C    
    D1 = -np.einsum('mk,abcijm->abcijk',H1A['oo'],t3a,optimize=True)
    D2 = np.einsum('ce,abeijk->abcijk',H1A['vv'],t3a,optimize=True)
    D3 = 0.5*np.einsum('mnij,abcmnk->abcijk',H2A['oooo'],t3a,optimize=True)
    D4 = 0.5*np.einsum('abef,efcijk->abcijk',H2A['vvvv'],t3a,optimize=True)
    D5 = np.einsum('cmke,abeijm->abcijk',H2A['voov'],t3a,optimize=True)
    D6 = np.einsum('cmke,abeijm->abcijk',H2B['voov'],t3b,optimize=True)
    
    # A(k/ij)
    D13 = D1 + D3
    D13 += -np.einsum('abcijk->abckji',D13,optimize=True)\
    -np.einsum('abcijk->abcikj',D13,optimize=True)
    
    # A(c/ab)
    D24 = D2 + D4
    D24 += -np.einsum('abcijk->cbaijk',D24,optimize=True)\
    -np.einsum('abcijk->acbijk',D24,optimize=True)
   
    # A(k/ij)A(c/ab)
    D56 = D5 + D6
    D56 += -np.einsum('abcijk->abckji',D56,optimize=True)\
    -np.einsum('abcijk->abcikj',D56,optimize=True)\
    -np.einsum('abcijk->cbaijk',D56,optimize=True)\
    -np.einsum('abcijk->acbijk',D56,optimize=True)\
    +np.einsum('abcijk->cbakji',D56,optimize=True)\
    +np.einsum('abcijk->cbaikj',D56,optimize=True)\
    +np.einsum('abcijk->acbkji',D56,optimize=True)\
    +np.einsum('abcijk->acbikj',D56,optimize=True)   
     
    X3A = D13 + D24 + D56 + MM23A

    t3a = cc_loops.cc_loops.update_t3a_p(t3a,X3A,p_space,fA['oo'],fA['vv'],shift)

    cc_t['t3a'] = t3a
    return cc_t

def update_t3b(cc_t,ints,p_space,H1A,H1B,H2A,H2B,H2C,sys,shift):

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
   
    # MM(2,3)B 
    M23_D1 = np.einsum('bcek,aeij->abcijk',I2B_vvvo,t2a,optimize=True)
    M23_D2 = -np.einsum('mcjk,abim->abcijk',I2B_ovoo,t2a,optimize=True)
    M23_D3 = +np.einsum('acie,bejk->abcijk',I2B_vvov,t2b,optimize=True)
    M23_D4 = -np.einsum('amik,bcjm->abcijk',I2B_vooo,t2b,optimize=True)
    M23_D5 = +np.einsum('abie,ecjk->abcijk',I2A_vvov,t2b,optimize=True)
    M23_D6 = -np.einsum('amij,bcmk->abcijk',I2A_vooo,t2b,optimize=True)

    # (HBar*T3)_C
    D1 = -np.einsum('mi,abcmjk->abcijk',H1A['oo'],t3b,optimize=True)
    D2 = -np.einsum('mk,abcijm->abcijk',H1B['oo'],t3b,optimize=True)
    D3 = np.einsum('ae,ebcijk->abcijk',H1A['vv'],t3b,optimize=True)
    D4 = np.einsum('ce,abeijk->abcijk',H1B['vv'],t3b,optimize=True)
    D5 = 0.5*np.einsum('mnij,abcmnk->abcijk',H2A['oooo'],t3b,optimize=True)
    D6 = np.einsum('mnjk,abcimn->abcijk',H2B['oooo'],t3b,optimize=True)
    D7 = 0.5*np.einsum('abef,efcijk->abcijk',H2A['vvvv'],t3b,optimize=True)
    D8 = np.einsum('bcef,aefijk->abcijk',H2B['vvvv'],t3b,optimize=True)
    D9 = np.einsum('amie,ebcmjk->abcijk',H2A['voov'],t3b,optimize=True)   
    D10 = np.einsum('amie,becjmk->abcijk',H2B['voov'],t3c,optimize=True)    
    D11 = np.einsum('mcek,abeijm->abcijk',H2B['ovvo'],t3a,optimize=True)
    D12 = np.einsum('cmke,abeijm->abcijk',H2C['voov'],t3b,optimize=True)
    D13 = -np.einsum('amek,ebcijm->abcijk',H2B['vovo'],t3b,optimize=True)
    D14 = -np.einsum('mcie,abemjk->abcijk',H2B['ovov'],t3b,optimize=True)
    
    # diagrams that have A(ab)A(ij)
    D_abij = D9 + D10 + M23_D3 + M23_D4
    D_abij += -np.einsum('abcijk->bacijk',D_abij,optimize=True)\
    -np.einsum('abcijk->abcjik',D_abij,optimize=True)\
    +np.einsum('abcijk->bacjik',D_abij,optimize=True)

    # diagrams that have A(ab)
    D_ab = D3 + D8 + D13 + M23_D1 + M23_D6
    D_ab -= np.einsum('abcijk->bacijk',D_ab,optimize=True)

    # diagrams that have A(ij)
    D_ij = D1 + D6 + D14 + M23_D2 + M23_D5
    D_ij -= np.einsum('abcijk->abcjik',D_ij,optimize=True)
     
    X3B = D2 + D4 + D5 + D7 + D11 + D12 + D_ij + D_ab + D_abij
    
    t3b = cc_loops.cc_loops.update_t3b_p(t3b,X3B,p_space,fA['oo'],fA['vv'],fB['oo'],fB['vv'],shift)
    
    cc_t['t3b'] = t3b
    return cc_t

def update_t3c(cc_t,ints,p_space,H1A,H1B,H2A,H2B,H2C,sys,shift):

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
    
    # MM(2,3)C
    M23_D1 = +np.einsum('abie,ecjk->abcijk',I2B_vvov,t2c,optimize=True)
    M23_D2 = -np.einsum('amij,bcmk->abcijk',I2B_vooo,t2c,optimize=True)
    M23_D3 = +np.einsum('cbke,aeij->abcijk',I2C_vvov,t2b,optimize=True)
    M23_D4 = -np.einsum('cmkj,abim->abcijk',I2C_vooo,t2b,optimize=True)
    M23_D5 = +np.einsum('abej,ecik->abcijk',I2B_vvvo,t2b,optimize=True)
    M23_D6 = -np.einsum('mbij,acmk->abcijk',I2B_ovoo,t2b,optimize=True)

    # (HBar*T3)_C
    D1 = -np.einsum('mi,abcmjk->abcijk',H1A['oo'],t3c,optimize=True)
    D2 = -np.einsum('mj,abcimk->abcijk',H1B['oo'],t3c,optimize=True)
    D3 = +np.einsum('ae,ebcijk->abcijk',H1A['vv'],t3c,optimize=True)
    D4 = +np.einsum('be,aecijk->abcijk',H1B['vv'],t3c,optimize=True)
    D5 = 0.5*np.einsum('mnjk,abcimn->abcijk',H2C['oooo'],t3c,optimize=True)
    D6 = np.einsum('mnij,abcmnk->abcijk',H2B['oooo'],t3c,optimize=True)
    D7 = 0.5*np.einsum('bcef,aefijk->abcijk',H2C['vvvv'],t3c,optimize=True)
    D8 = np.einsum('abef,efcijk->abcijk',H2B['vvvv'],t3c,optimize=True)
    D9 = np.einsum('amie,ebcmjk->abcijk',H2A['voov'],t3c,optimize=True)
    D10 = np.einsum('amie,ebcmjk->abcijk',H2B['voov'],t3d,optimize=True)
    D11 = np.einsum('mbej,aecimk->abcijk',H2B['ovvo'],t3b,optimize=True)
    D12 = np.einsum('bmje,aecimk->abcijk',H2C['voov'],t3c,optimize=True)
    D13 = -np.einsum('mbie,aecmjk->abcijk',H2B['ovov'],t3c,optimize=True)
    D14 = -np.einsum('amej,ebcimk->abcijk',H2B['vovo'],t3c,optimize=True)

    D_jk = D2 + D6 + D14 + M23_D2 + M23_D3
    D_jk -= np.einsum('abcijk->abcikj',D_jk,optimize=True)

    D_bc = D4 + D8 + D13 + M23_D1 + M23_D4
    D_bc -= np.einsum('abcijk->acbijk',D_bc,optimize=True)

    D_bcjk = D11 + D12 + M23_D5 + M23_D6
    D_bcjk += -np.einsum('abcijk->acbijk',D_bcjk,optimize=True)\
    -np.einsum('abcijk->abcikj',D_bcjk,optimize=True)\
    +np.einsum('abcijk->acbikj',D_bcjk,optimize=True)

    X3C = D1 + D_jk + D3 + D_bc + D5 + D7 + D9 + D10 + D_bcjk

    t3c = cc_loops.cc_loops.update_t3c_p(t3c,X3C,p_space,fA['oo'],fA['vv'],fB['oo'],fB['vv'],shift)
             
    cc_t['t3c'] = t3c
    return cc_t

def update_t3d(cc_t,ints,p_space,H1A,H1B,H2A,H2B,H2C,sys,shift):

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

    I2C_vvov = 0.0
    I2C_vvov -= 0.5*np.einsum('mnef,abfimn->abie',vC['oovv'],t3d,optimize=True)
    I2C_vvov -= np.einsum('nmfe,fabnim->abie',vB['oovv'],t3c,optimize=True)
    I2C_vvov += np.einsum('me,abim->abie',H1B['ov'],t2c,optimize=True)
    I2C_vvov += H2C['vvov']
             
    I2C_vooo = 0.0
    I2C_vooo += 0.5*np.einsum('mnef,aefijn->amij',vC['oovv'],t3d,optimize=True)
    I2C_vooo += np.einsum('nmfe,faenij->amij',vB['oovv'],t3c,optimize=True)
    I2C_vooo += H2C['vooo']   
    
    # MM(2,3)D
    M23_D1 = 0.0
    M23_D1 -= np.einsum('amij,bcmk->abcijk',I2C_vooo,t2c,optimize=True) 

    M23_D1 += -np.einsum('abcijk->abckji',M23_D1,optimize=True) \
    -np.einsum('abcijk->abcikj',M23_D1,optimize=True) \
    -np.einsum('abcijk->cbaijk',M23_D1,optimize=True) \
    -np.einsum('abcijk->bacijk',M23_D1,optimize=True) \
    +np.einsum('abcijk->backji',M23_D1,optimize=True) \
    +np.einsum('abcijk->cbakji',M23_D1,optimize=True) \
    +np.einsum('abcijk->bacikj',M23_D1,optimize=True) \
    +np.einsum('abcijk->cbaikj',M23_D1,optimize=True)

    M23_D2 = 0.0
    M23_D2 += np.einsum('abie,ecjk->abcijk',I2C_vvov,t2c,optimize=True)

    M23_D2 += -np.einsum('abcijk->abcjik',M23_D2,optimize=True)\
    -np.einsum('abcijk->abckji',M23_D2,optimize=True)\
    -np.einsum('abcijk->cbaijk',M23_D2,optimize=True)\
    -np.einsum('abcijk->acbijk',M23_D2,optimize=True)\
    +np.einsum('abcijk->cbajik',M23_D2,optimize=True)\
    +np.einsum('abcijk->acbjik',M23_D2,optimize=True)\
    +np.einsum('abcijk->cbakji',M23_D2,optimize=True)\
    +np.einsum('abcijk->acbkji',M23_D2,optimize=True)\

    MM23D = M23_D1 + M23_D2

    # (HBar*T3)_C
    D1 = -np.einsum('mk,abcijm->abcijk',H1B['oo'],t3d,optimize=True)
    D2 = np.einsum('ce,abeijk->abcijk',H1B['vv'],t3d,optimize=True)
    D3 = 0.5*np.einsum('mnij,abcmnk->abcijk',H2C['oooo'],t3d,optimize=True)
    D4 = 0.5*np.einsum('abef,efcijk->abcijk',H2C['vvvv'],t3d,optimize=True)
    D5 = np.einsum('maei,ebcmjk->abcijk',H2B['ovvo'],t3c,optimize=True)
    D6 = np.einsum('amie,ebcmjk->abcijk',H2C['voov'],t3d,optimize=True)

    # A(k/ij)
    D13 = D1 + D3
    D13 += -np.einsum('abcijk->abckji',D13,optimize=True)\
    -np.einsum('abcijk->abcikj',D13,optimize=True)
    
    # A(c/ab)
    D24 = D2 + D4
    D24 += -np.einsum('abcijk->cbaijk',D24,optimize=True)\
    -np.einsum('abcijk->acbijk',D24,optimize=True)
   
    # A(i/jk)A(a/bc)
    D56 = D5 + D6
    D56 += -np.einsum('abcijk->abcjik',D56,optimize=True)\
    -np.einsum('abcijk->abckji',D56,optimize=True)\
    -np.einsum('abcijk->bacijk',D56,optimize=True)\
    -np.einsum('abcijk->cbaijk',D56,optimize=True)\
    +np.einsum('abcijk->bacjik',D56,optimize=True)\
    +np.einsum('abcijk->backji',D56,optimize=True)\
    +np.einsum('abcijk->cbajik',D56,optimize=True)\
    +np.einsum('abcijk->cbakji',D56,optimize=True)
     
    X3D = MM23D + D13 + D24 + D56

    t3d = cc_loops.cc_loops.update_t3d_p(t3d,X3D,p_space,fB['oo'],fB['vv'],shift)

    cc_t['t3d'] = t3d
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

def update_t3a_inloop(cc_t,ints,list_of_triples_A,H1A,H1B,H2A,H2B,H2C,sys,shift):

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

    I2A_vvov = np.einsum('me,abim->abie',H1A['ov'],t2a,optimize=True)
    I2A_vvov += H2A['vvov']

    t3a_new = t3a.copy()
    for idx in list_of_triples_A:
        a = idx[0]; b = idx[1]; c = idx[2];
        i = idx[3]; j = idx[4]; k = idx[5];

        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3a[a,:,:,i,j,:],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3b[a,:,:,i,j,:],optimize=True)
        m1 = -np.einsum('m,m->',H2A['vooo'][a,:,i,j]+vt3int,t2a[b,c,:,k],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3a[c,:,:,i,j,:],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3b[c,:,:,i,j,:],optimize=True)
        m1 -= -np.einsum('m,m->',H2A['vooo'][c,:,i,j]+vt3int,t2a[b,a,:,k],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3a[b,:,:,i,j,:],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3b[b,:,:,i,j,:],optimize=True)
        m1 -= -np.einsum('m,m->',H2A['vooo'][b,:,i,j]+vt3int,t2a[a,c,:,k],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3a[a,:,:,k,j,:],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3b[a,:,:,k,j,:],optimize=True)
        m1 -= -np.einsum('m,m->',H2A['vooo'][a,:,k,j]+vt3int,t2a[b,c,:,i],optimize=True)
        
        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3a[c,:,:,k,j,:],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3b[c,:,:,k,j,:],optimize=True)
        m1 += -np.einsum('m,m->',H2A['vooo'][c,:,k,j]+vt3int,t2a[b,a,:,i],optimize=True)
        
        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3a[b,:,:,k,j,:],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3b[b,:,:,k,j,:],optimize=True)
        m1 += -np.einsum('m,m->',H2A['vooo'][b,:,k,j]+vt3int,t2a[a,c,:,i],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3a[a,:,:,i,k,:],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3b[a,:,:,i,k,:],optimize=True)
        m1 -= -np.einsum('m,m->',H2A['vooo'][a,:,i,k]+vt3int,t2a[b,c,:,j],optimize=True)
        
        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3a[c,:,:,i,k,:],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3b[c,:,:,i,k,:],optimize=True)
        m1 += -np.einsum('m,m->',H2A['vooo'][c,:,i,k]+vt3int,t2a[b,a,:,j],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3a[b,:,:,i,k,:],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3b[b,:,:,i,k,:],optimize=True)
        m1 += -np.einsum('m,m->',H2A['vooo'][b,:,i,k]+vt3int,t2a[a,c,:,j],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3a[a,b,:,i,:,:],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3b[a,b,:,i,:,:],optimize=True)
        m2 = np.einsum('e,e->',I2A_vvov[a,b,i,:]+vt3int,t2a[:,c,j,k],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3a[c,b,:,i,:,:],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3b[c,b,:,i,:,:],optimize=True)
        m2 -= np.einsum('e,e->',I2A_vvov[c,b,i,:]+vt3int,t2a[:,a,j,k],optimize=True)
        
        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3a[a,c,:,i,:,:],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3b[a,c,:,i,:,:],optimize=True)
        m2 -= np.einsum('e,e->',I2A_vvov[a,c,i,:]+vt3int,t2a[:,b,j,k],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3a[a,b,:,j,:,:],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3b[a,b,:,j,:,:],optimize=True)
        m2 -= np.einsum('e,e->',I2A_vvov[a,b,j,:]+vt3int,t2a[:,c,i,k],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3a[c,b,:,j,:,:],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3b[c,b,:,j,:,:],optimize=True)
        m2 += np.einsum('e,e->',I2A_vvov[c,b,j,:]+vt3int,t2a[:,a,i,k],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3a[a,c,:,j,:,:],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3b[a,c,:,j,:,:],optimize=True)
        m2 += np.einsum('e,e->',I2A_vvov[a,c,j,:]+vt3int,t2a[:,b,i,k],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3a[a,b,:,k,:,:],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3b[a,b,:,k,:,:],optimize=True)
        m2 -= np.einsum('e,e->',I2A_vvov[a,b,k,:]+vt3int,t2a[:,c,j,i],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3a[c,b,:,k,:,:],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3b[c,b,:,k,:,:],optimize=True)
        m2 += np.einsum('e,e->',I2A_vvov[c,b,k,:]+vt3int,t2a[:,a,j,i],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3a[a,c,:,k,:,:],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3b[a,c,:,k,:,:],optimize=True)
        m2 += np.einsum('e,e->',I2A_vvov[a,c,k,:]+vt3int,t2a[:,b,j,i],optimize=True)

        d1 = -np.einsum('m,m->',H1A['oo'][:,k],t3a[a,b,c,i,j,:],optimize=True)
        d1 -= -np.einsum('m,m->',H1A['oo'][:,i],t3a[a,b,c,k,j,:],optimize=True)
        d1 -= -np.einsum('m,m->',H1A['oo'][:,j],t3a[a,b,c,i,k,:],optimize=True)

        d2 = +np.einsum('e,e->',H1A['vv'][c,:],t3a[a,b,:,i,j,k],optimize=True)
        d2 -= np.einsum('e,e->',H1A['vv'][a,:],t3a[c,b,:,i,j,k],optimize=True)
        d2 -= np.einsum('e,e->',H1A['vv'][b,:],t3a[a,c,:,i,j,k],optimize=True)

        d3 = +0.5*np.einsum('mn,mn->',H2A['oooo'][:,:,i,j],t3a[a,b,c,:,:,k],optimize=True)
        d3 -= +0.5*np.einsum('mn,mn->',H2A['oooo'][:,:,k,j],t3a[a,b,c,:,:,i],optimize=True)
        d3 -= +0.5*np.einsum('mn,mn->',H2A['oooo'][:,:,i,k],t3a[a,b,c,:,:,j],optimize=True)

        d4 = +0.5*np.einsum('ef,ef->',H2A['vvvv'][a,b,:,:],t3a[:,:,c,i,j,k],optimize=True)
        d4 -= +0.5*np.einsum('ef,ef->',H2A['vvvv'][a,c,:,:],t3a[:,:,b,i,j,k],optimize=True)
        d4 -= +0.5*np.einsum('ef,ef->',H2A['vvvv'][c,b,:,:],t3a[:,:,a,i,j,k],optimize=True)

        d5 = +np.einsum('me,em->',H2A['voov'][c,:,k,:],t3a[a,b,:,i,j,:],optimize=True)
        d5 -= +np.einsum('me,em->',H2A['voov'][c,:,i,:],t3a[a,b,:,k,j,:],optimize=True)
        d5 -= +np.einsum('me,em->',H2A['voov'][c,:,j,:],t3a[a,b,:,i,k,:],optimize=True)
        d5 -= +np.einsum('me,em->',H2A['voov'][a,:,k,:],t3a[c,b,:,i,j,:],optimize=True)
        d5 += +np.einsum('me,em->',H2A['voov'][a,:,i,:],t3a[c,b,:,k,j,:],optimize=True)
        d5 += +np.einsum('me,em->',H2A['voov'][a,:,j,:],t3a[c,b,:,i,k,:],optimize=True)
        d5 -= +np.einsum('me,em->',H2A['voov'][b,:,k,:],t3a[a,c,:,i,j,:],optimize=True)
        d5 += +np.einsum('me,em->',H2A['voov'][b,:,i,:],t3a[a,c,:,k,j,:],optimize=True)
        d5 += +np.einsum('me,em->',H2A['voov'][b,:,j,:],t3a[a,c,:,i,k,:],optimize=True)

        d6 = +np.einsum('me,em->',H2B['voov'][c,:,k,:],t3b[a,b,:,i,j,:],optimize=True)
        d6 -= +np.einsum('me,em->',H2B['voov'][c,:,i,:],t3b[a,b,:,k,j,:],optimize=True)
        d6 -= +np.einsum('me,em->',H2B['voov'][c,:,j,:],t3b[a,b,:,i,k,:],optimize=True)
        d6 -= +np.einsum('me,em->',H2B['voov'][a,:,k,:],t3b[c,b,:,i,j,:],optimize=True)
        d6 += +np.einsum('me,em->',H2B['voov'][a,:,i,:],t3b[c,b,:,k,j,:],optimize=True)
        d6 += +np.einsum('me,em->',H2B['voov'][a,:,j,:],t3b[c,b,:,i,k,:],optimize=True)
        d6 -= +np.einsum('me,em->',H2B['voov'][b,:,k,:],t3b[a,c,:,i,j,:],optimize=True)
        d6 += +np.einsum('me,em->',H2B['voov'][b,:,i,:],t3b[a,c,:,k,j,:],optimize=True)
        d6 += +np.einsum('me,em->',H2B['voov'][b,:,j,:],t3b[a,c,:,i,k,:],optimize=True)

        residual = m1 + m2 + d1 + d2 + d3 + d4 + d5 + d6
        denom = fA['oo'][i,i]+fA['oo'][j,j]+fA['oo'][k,k]\
                -fA['vv'][a,a]-fA['vv'][b,b]-fA['vv'][c,c]
        t3a_new[a,b,c,i,j,k] += residual/(denom-shift)
        t3a_new[b,a,c,i,j,k] = -t3a_new[a,b,c,i,j,k]
        t3a_new[a,c,b,i,j,k] = -t3a_new[a,b,c,i,j,k]
        t3a_new[b,c,a,i,j,k] = t3a_new[a,b,c,i,j,k]
        t3a_new[c,a,b,i,j,k] = t3a_new[a,b,c,i,j,k]
        t3a_new[c,b,a,i,j,k] = -t3a_new[a,b,c,i,j,k]
        t3a_new[a,b,c,j,i,k] = -t3a_new[a,b,c,i,j,k] 
        t3a_new[b,a,c,j,i,k] = t3a_new[a,b,c,i,j,k]
        t3a_new[a,c,b,j,i,k] = t3a_new[a,b,c,i,j,k]
        t3a_new[b,c,a,j,i,k] = -t3a_new[a,b,c,i,j,k]
        t3a_new[c,a,b,j,i,k] = -t3a_new[a,b,c,i,j,k]
        t3a_new[c,b,a,j,i,k] = t3a_new[a,b,c,i,j,k]
        t3a_new[a,b,c,i,k,j] = -t3a_new[a,b,c,i,j,k] 
        t3a_new[b,a,c,i,k,j] = t3a_new[a,b,c,i,j,k]
        t3a_new[a,c,b,i,k,j] = t3a_new[a,b,c,i,j,k]
        t3a_new[b,c,a,i,k,j] = -t3a_new[a,b,c,i,j,k]
        t3a_new[c,a,b,i,k,j] = -t3a_new[a,b,c,i,j,k]
        t3a_new[c,b,a,i,k,j] = t3a_new[a,b,c,i,j,k]
        t3a_new[a,b,c,j,k,i] = t3a_new[a,b,c,i,j,k] 
        t3a_new[b,a,c,j,k,i] = -t3a_new[a,b,c,i,j,k]
        t3a_new[a,c,b,j,k,i] = -t3a_new[a,b,c,i,j,k]
        t3a_new[b,c,a,j,k,i] = t3a_new[a,b,c,i,j,k]
        t3a_new[c,a,b,j,k,i] = t3a_new[a,b,c,i,j,k]
        t3a_new[c,b,a,j,k,i] = -t3a_new[a,b,c,i,j,k]
        t3a_new[a,b,c,k,i,j] = t3a_new[a,b,c,i,j,k]
        t3a_new[b,a,c,k,i,j] = -t3a_new[a,b,c,i,j,k]
        t3a_new[a,c,b,k,i,j] = -t3a_new[a,b,c,i,j,k]
        t3a_new[b,c,a,k,i,j] = t3a_new[a,b,c,i,j,k]
        t3a_new[c,a,b,k,i,j] = t3a_new[a,b,c,i,j,k]
        t3a_new[c,b,a,k,i,j] = -t3a_new[a,b,c,i,j,k]
        t3a_new[a,b,c,k,j,i] = -t3a_new[a,b,c,i,j,k] 
        t3a_new[b,a,c,k,j,i] = t3a_new[a,b,c,i,j,k]
        t3a_new[a,c,b,k,j,i] = t3a_new[a,b,c,i,j,k]
        t3a_new[b,c,a,k,j,i] = -t3a_new[a,b,c,i,j,k]
        t3a_new[c,a,b,k,j,i] = -t3a_new[a,b,c,i,j,k]
        t3a_new[c,b,a,k,j,i] = t3a_new[a,b,c,i,j,k]

    cc_t['t3a'] = t3a_new
    return cc_t

def update_t3b_inloop(cc_t,ints,list_of_triples_B,H1A,H1B,H2A,H2B,H2C,sys,shift):

    import time

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

    # MM(2,3)_B intermediates
    I2A_vooo = 0.0
    I2A_vooo += H2A['vooo']
    I2A_vooo += -np.einsum('me,aeij->amij',H1A['ov'],t2a,optimize=True)
             
    I2B_ovoo = 0.0
    I2B_ovoo += H2B['ovoo']
    I2B_ovoo += -np.einsum('me,ecjk->mcjk',H1A['ov'],t2b,optimize=True)

    I2B_vooo = 0.0
    I2B_vooo += H2B['vooo']
    I2B_vooo += -np.einsum('me,aeik->amik',H1B['ov'],t2b,optimize=True) 


    total_time = 0.0
    num_triples = 0
    time_per_diagram = np.zeros(20)

    t3b_new = t3b.copy()
    for idx in list_of_triples_B:
        
        num_triples += 1
        tic_iter = time.perf_counter()

        a = idx[0]; b = idx[1]; c = idx[2];
        i = idx[3]; j = idx[4]; k = idx[5];
    
        tic = time.perf_counter()
        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3b[b,:,c,:,:,k],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3c[b,:,c,:,:,k],optimize=True)
        m1 = np.einsum('e,e->',H2B['vvvo'][b,c,:,k]+vt3int,t2a[a,:,i,j],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3b[a,:,c,:,:,k],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3c[a,:,c,:,:,k],optimize=True)
        m1 -= np.einsum('e,e->',H2B['vvvo'][a,c,:,k]+vt3int,t2a[b,:,i,j],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[0] += toc-tic

        tic = time.perf_counter()
        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3b[:,:,c,i,:,k],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3c[:,:,c,i,:,k],optimize=True)
        m2 = -np.einsum('m,m->',I2B_ovoo[:,c,i,k]+vt3int,t2a[a,b,:,j],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3b[:,:,c,j,:,k],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3c[:,:,c,j,:,k],optimize=True)
        m2 -= -np.einsum('m,m->',I2B_ovoo[:,c,j,k]+vt3int,t2a[a,b,:,i],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[1] += toc-tic

        tic = time.perf_counter()
        vt3int = -np.einsum('nmfe,fnm->e',vB['oovv'],t3b[a,:,c,i,:,:],optimize=True)\
            -0.5*np.einsum('nmfe,fnm->e',vC['oovv'],t3c[a,:,c,i,:,:],optimize=True)
        m3 = +np.einsum('e,e->',H2B['vvov'][a,c,i,:]+vt3int,t2b[b,:,j,k],optimize=True)

        vt3int = -np.einsum('nmfe,fnm->e',vB['oovv'],t3b[b,:,c,i,:,:],optimize=True)\
            -0.5*np.einsum('nmfe,fnm->e',vC['oovv'],t3c[b,:,c,i,:,:],optimize=True)
        m3 -= +np.einsum('e,e->',H2B['vvov'][b,c,i,:]+vt3int,t2b[a,:,j,k],optimize=True)

        vt3int = -np.einsum('nmfe,fnm->e',vB['oovv'],t3b[a,:,c,j,:,:],optimize=True)\
            -0.5*np.einsum('nmfe,fnm->e',vC['oovv'],t3c[a,:,c,j,:,:],optimize=True)
        m3 -= +np.einsum('e,e->',H2B['vvov'][a,c,j,:]+vt3int,t2b[b,:,i,k],optimize=True)

        vt3int = -np.einsum('nmfe,fnm->e',vB['oovv'],t3b[b,:,c,j,:,:],optimize=True)\
            -0.5*np.einsum('nmfe,fnm->e',vC['oovv'],t3c[b,:,c,j,:,:],optimize=True)
        m3 += +np.einsum('e,e->',H2B['vvov'][b,c,j,:]+vt3int,t2b[a,:,i,k],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[2] += toc-tic

        tic = time.perf_counter()
        vt3int = np.einsum('nmfe,fen->m',vB['oovv'],t3b[b,:,:,j,:,k],optimize=True)\
            +0.5*np.einsum('mnef,fen->m',vC['oovv'],t3c[b,:,:,j,:,k],optimize=True)
        m4 = -np.einsum('m,m->',I2B_vooo[b,:,j,k]+vt3int,t2b[a,c,i,:],optimize=True)

        vt3int = np.einsum('nmfe,fen->m',vB['oovv'],t3b[b,:,:,i,:,k],optimize=True)\
            +0.5*np.einsum('mnef,fen->m',vC['oovv'],t3c[b,:,:,i,:,k],optimize=True)
        m4 -= -np.einsum('m,m->',I2B_vooo[b,:,i,k]+vt3int,t2b[a,c,j,:],optimize=True)

        vt3int = np.einsum('nmfe,fen->m',vB['oovv'],t3b[a,:,:,j,:,k],optimize=True)\
            +0.5*np.einsum('mnef,fen->m',vC['oovv'],t3c[a,:,:,j,:,k],optimize=True)
        m4 -= -np.einsum('m,m->',I2B_vooo[a,:,j,k]+vt3int,t2b[b,c,i,:],optimize=True)

        vt3int = np.einsum('nmfe,fen->m',vB['oovv'],t3b[a,:,:,i,:,k],optimize=True)\
            +0.5*np.einsum('mnef,fen->m',vC['oovv'],t3c[a,:,:,i,:,k],optimize=True)
        m4 += -np.einsum('m,m->',I2B_vooo[a,:,i,k]+vt3int,t2b[b,c,j,:],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[3] += toc-tic

        tic = time.perf_counter()
        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3a[a,b,:,i,:,:],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3b[a,b,:,i,:,:],optimize=True)
        m5 = +np.einsum('e,e->',H2A['vvov'][a,b,i,:]+vt3int,t2b[:,c,j,k],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3a[a,b,:,j,:,:],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3b[a,b,:,j,:,:],optimize=True)
        m5 -= +np.einsum('e,e->',H2A['vvov'][a,b,j,:]+vt3int,t2b[:,c,i,k],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[4] += toc-tic

        tic = time.perf_counter() 
        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3a[a,:,:,i,j,:],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3b[a,:,:,i,j,:],optimize=True)
        m6 = -np.einsum('m,m->',I2A_vooo[a,:,i,j]+vt3int,t2b[b,c,:,k],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3a[b,:,:,i,j,:],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3b[b,:,:,i,j,:],optimize=True)
        m6 -= -np.einsum('m,m->',I2A_vooo[b,:,i,j]+vt3int,t2b[a,c,:,k],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[5] += toc-tic

        tic = time.perf_counter()
        d1 = -np.einsum('m,m->',H1A['oo'][:,i],t3b[a,b,c,:,j,k],optimize=True)
        d1 -= -np.einsum('m,m->',H1A['oo'][:,j],t3b[a,b,c,:,i,k],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[6] += toc-tic

        tic = time.perf_counter()
        d2 = -np.einsum('m,m->',H1B['oo'][:,k],t3b[a,b,c,i,j,:],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[7] += toc-tic

        tic = time.perf_counter()
        d3 = np.einsum('e,e->',H1A['vv'][a,:],t3b[:,b,c,i,j,k],optimize=True)
        d3 -= np.einsum('e,e->',H1A['vv'][b,:],t3b[:,a,c,i,j,k],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[8] += toc-tic

        tic = time.perf_counter()
        d4 = np.einsum('e,e->',H1B['vv'][c,:],t3b[a,b,:,i,j,k],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[9] += toc-tic
        
        tic = time.perf_counter()
        d5 = 0.5*np.einsum('mn,mn->',H2A['oooo'][:,:,i,j],t3b[a,b,c,:,:,k],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[10] += toc-tic

        tic = time.perf_counter()
        d6 = np.einsum('mn,mn->',H2B['oooo'][:,:,j,k],t3b[a,b,c,i,:,:],optimize=True)
        d6 -= np.einsum('mn,mn->',H2B['oooo'][:,:,i,k],t3b[a,b,c,j,:,:],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[11] += toc-tic

        tic = time.perf_counter()
        d7 = 0.5*np.einsum('ef,ef->',H2A['vvvv'][a,b,:,:],t3b[:,:,c,i,j,k],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[12] += toc-tic

        tic = time.perf_counter()
        d8 = np.einsum('ef,ef->',H2B['vvvv'][b,c,:,:],t3b[a,:,:,i,j,k],optimize=True)
        d8 -= np.einsum('ef,ef->',H2B['vvvv'][a,c,:,:],t3b[b,:,:,i,j,k],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[13] += toc-tic

        tic = time.perf_counter()
        d9 = np.einsum('me,em->',H2A['voov'][a,:,i,:],t3b[:,b,c,:,j,k],optimize=True)
        d9 -= np.einsum('me,em->',H2A['voov'][b,:,i,:],t3b[:,a,c,:,j,k],optimize=True)
        d9 -= np.einsum('me,em->',H2A['voov'][a,:,j,:],t3b[:,b,c,:,i,k],optimize=True)
        d9 += np.einsum('me,em->',H2A['voov'][b,:,j,:],t3b[:,a,c,:,i,k],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[14] += toc-tic

        tic = time.perf_counter()
        d10 = np.einsum('me,em->',H2B['voov'][a,:,i,:],t3c[b,:,c,j,:,k],optimize=True)   
        d10 -= np.einsum('me,em->',H2B['voov'][b,:,i,:],t3c[a,:,c,j,:,k],optimize=True)   
        d10 -= np.einsum('me,em->',H2B['voov'][a,:,j,:],t3c[b,:,c,i,:,k],optimize=True)   
        d10 += np.einsum('me,em->',H2B['voov'][b,:,j,:],t3c[a,:,c,i,:,k],optimize=True)   
        toc = time.perf_counter()
        time_per_diagram[15] += toc-tic

        tic = time.perf_counter()
        d11 = np.einsum('me,em->',H2B['ovvo'][:,c,:,k],t3a[a,b,:,i,j,:],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[16] += toc-tic

        tic = time.perf_counter()
        d12 = np.einsum('me,em->',H2C['voov'][c,:,k,:],t3b[a,b,:,i,j,:],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[17] += toc-tic

        tic = time.perf_counter()
        d13 = -np.einsum('me,em->',H2B['vovo'][a,:,:,k],t3b[:,b,c,i,j,:],optimize=True)
        d13 -= -np.einsum('me,em->',H2B['vovo'][b,:,:,k],t3b[:,a,c,i,j,:],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[18] += toc-tic

        tic = time.perf_counter()
        d14 = -np.einsum('me,em->',H2B['ovov'][:,c,i,:],t3b[a,b,:,:,j,k],optimize=True)
        d14 -= -np.einsum('me,em->',H2B['ovov'][:,c,j,:],t3b[a,b,:,:,i,k],optimize=True)
        toc = time.perf_counter()
        time_per_diagram[19] += toc-tic
    
        residual = m1 + m2 + m3 + m4 + m5 + m6\
        +d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9 + d10\
        +d11 + d12 + d13 + d14 
        denom = fA['oo'][i,i]+fA['oo'][j,j]+fB['oo'][k,k]\
                -fA['vv'][a,a]-fA['vv'][b,b]-fB['vv'][c,c]

        t3b_new[a,b,c,i,j,k] += residual/(denom-shift)
        t3b_new[b,a,c,i,j,k] = -t3b_new[a,b,c,i,j,k]
        t3b_new[a,b,c,j,i,k] = -t3b_new[a,b,c,i,j,k]
        t3b_new[b,a,c,j,i,k] = t3b_new[a,b,c,i,j,k]
             
        toc_iter = time.perf_counter()
        total_time += toc_iter - tic_iter

    print('================TIMING SUMMARY===============')
    print('Average time per iteration = {} s'.format(total_time/num_triples))
    for i in range(len(time_per_diagram)):
        print('Average time for diagram {} = {} s'.format(i+1,time_per_diagram[i]/num_triples))
    cc_t['t3b'] = t3b_new
    return cc_t

def update_t3c_inloop(cc_t,ints,list_of_triples_C,H1A,H1B,H2A,H2B,H2C,sys,shift):

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

    I2B_ovoo = 0.0       
    I2B_ovoo += H2B['ovoo']
    I2B_ovoo -= np.einsum('me,ebij->mbij',H1A['ov'],t2b,optimize=True)

    I2B_vooo = 0.0     
    I2B_vooo += H2B['vooo']
    I2B_vooo -= np.einsum('me,aeij->amij',H1B['ov'],t2b,optimize=True)

    I2C_vooo = 0.0             
    I2C_vooo += H2C['vooo']
    I2C_vooo -= np.einsum('me,cekj->cmkj',H1B['ov'],t2c,optimize=True)

    t3c_new = t3c.copy()
    for idx in list_of_triples_C:
        a = idx[0]; b = idx[1]; c = idx[2];
        i = idx[3]; j = idx[4]; k = idx[5];

        vt3int = -np.einsum('nmfe,fnm->e',vB['oovv'],t3b[a,:,b,i,:,:],optimize=True)\
            -0.5*np.einsum('nmfe,fnm->e',vC['oovv'],t3c[a,:,b,i,:,:],optimize=True)
        m1 = np.einsum('e,e->',H2B['vvov'][a,b,i,:]+vt3int,t2c[:,c,j,k],optimize=True)

        vt3int = -np.einsum('nmfe,fnm->e',vB['oovv'],t3b[a,:,c,i,:,:],optimize=True)\
            -0.5*np.einsum('nmfe,fnm->e',vC['oovv'],t3c[a,:,c,i,:,:],optimize=True)
        m1 -= np.einsum('e,e->',H2B['vvov'][a,c,i,:]+vt3int,t2c[:,b,j,k],optimize=True)

        vt3int = np.einsum('nmfe,fen->m',vB['oovv'],t3b[a,:,:,i,:,j],optimize=True)\
            +0.5*np.einsum('nmfe,fen->m',vC['oovv'],t3c[a,:,:,i,:,j],optimize=True)
        m2 = -np.einsum('m,m->',I2B_vooo[a,:,i,j]+vt3int,t2c[b,c,:,k],optimize=True)

        vt3int = np.einsum('nmfe,fen->m',vB['oovv'],t3b[a,:,:,i,:,k],optimize=True)\
            +0.5*np.einsum('nmfe,fen->m',vC['oovv'],t3c[a,:,:,i,:,k],optimize=True)
        m2 -= -np.einsum('m,m->',I2B_vooo[a,:,i,k]+vt3int,t2c[b,c,:,j],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vC['oovv'],t3d[c,b,:,k,:,:],optimize=True)\
            -np.einsum('nmfe,fnm->e',vB['oovv'],t3c[:,c,b,:,k,:],optimize=True)
        m3 = np.einsum('e,e->',H2C['vvov'][c,b,k,:]+vt3int,t2b[a,:,i,j],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vC['oovv'],t3d[c,b,:,j,:,:],optimize=True)\
            -np.einsum('nmfe,fnm->e',vB['oovv'],t3c[:,c,b,:,j,:],optimize=True)
        m3 -= np.einsum('e,e->',H2C['vvov'][c,b,j,:]+vt3int,t2b[a,:,i,k],optimize=True)

        vt3int = np.einsum('nmfe,fen->m',vB['oovv'],t3c[:,c,:,:,k,j],optimize=True)\
            +0.5*np.einsum('mnef,efn->m',vC['oovv'],t3d[c,:,:,k,j,:],optimize=True)
        m4 = -np.einsum('m,m->',I2C_vooo[c,:,k,j]+vt3int,t2b[a,b,i,:],optimize=True)

        vt3int = np.einsum('nmfe,fen->m',vB['oovv'],t3c[:,b,:,:,k,j],optimize=True)\
            +0.5*np.einsum('mnef,efn->m',vC['oovv'],t3d[b,:,:,k,j,:],optimize=True)
        m4 -= -np.einsum('m,m->',I2C_vooo[b,:,k,j]+vt3int,t2b[a,c,i,:],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3b[a,:,b,:,:,j],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3c[a,:,b,:,:,j],optimize=True)
        m5 = np.einsum('e,e->',H2B['vvvo'][a,b,:,j]+vt3int,t2b[:,c,i,k],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3b[a,:,c,:,:,j],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3c[a,:,c,:,:,j],optimize=True)
        m5 -= np.einsum('e,e->',H2B['vvvo'][a,c,:,j]+vt3int,t2b[:,b,i,k],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3b[a,:,b,:,:,k],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3c[a,:,b,:,:,k],optimize=True)
        m5 -= np.einsum('e,e->',H2B['vvvo'][a,b,:,k]+vt3int,t2b[:,c,i,j],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vA['oovv'],t3b[a,:,c,:,:,k],optimize=True)\
            -np.einsum('mnef,fmn->e',vB['oovv'],t3c[a,:,c,:,:,k],optimize=True)
        m5 += np.einsum('e,e->',H2B['vvvo'][a,c,:,k]+vt3int,t2b[:,b,i,j],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3b[:,:,b,i,:,j],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3c[:,:,b,i,:,j],optimize=True)
        m6 = -np.einsum('m,m->',I2B_ovoo[:,b,i,j]+vt3int,t2b[a,c,:,k],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3b[:,:,c,i,:,j],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3c[:,:,c,i,:,j],optimize=True)
        m6 -= -np.einsum('m,m->',I2B_ovoo[:,c,i,j]+vt3int,t2b[a,b,:,k],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3b[:,:,b,i,:,k],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3c[:,:,b,i,:,k],optimize=True)
        m6 -= -np.einsum('m,m->',I2B_ovoo[:,b,i,k]+vt3int,t2b[a,c,:,j],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vA['oovv'],t3b[:,:,c,i,:,k],optimize=True)\
            +np.einsum('mnef,efn->m',vB['oovv'],t3c[:,:,c,i,:,k],optimize=True)
        m6 += -np.einsum('m,m->',I2B_ovoo[:,c,i,k]+vt3int,t2b[a,b,:,j],optimize=True)

        d1 = -np.einsum('m,m->',H1A['oo'][:,i],t3c[a,b,c,:,j,k],optimize=True)

        d2 = -np.einsum('m,m->',H1B['oo'][:,j],t3c[a,b,c,i,:,k],optimize=True)
        d2 -= -np.einsum('m,m->',H1B['oo'][:,k],t3c[a,b,c,i,:,j],optimize=True)

        d3 = np.einsum('e,e->',H1A['vv'][a,:],t3c[:,b,c,i,j,k],optimize=True)
        
        d4 = np.einsum('e,e->',H1B['vv'][b,:],t3c[a,:,c,i,j,k],optimize=True)
        d4 -= np.einsum('e,e->',H1B['vv'][c,:],t3c[a,:,b,i,j,k],optimize=True)

        d5 = 0.5*np.einsum('mn,mn->',H2C['oooo'][:,:,j,k],t3c[a,b,c,i,:,:],optimize=True)

        d6 = np.einsum('mn,mn->',H2B['oooo'][:,:,i,j],t3c[a,b,c,:,:,k],optimize=True)
        d6 -= np.einsum('mn,mn->',H2B['oooo'][:,:,i,k],t3c[a,b,c,:,:,j],optimize=True)

        d7 = 0.5*np.einsum('ef,ef->',H2C['vvvv'][b,c,:,:],t3c[a,:,:,i,j,k],optimize=True)

        d8 = np.einsum('ef,ef->',H2B['vvvv'][a,b,:,:],t3c[:,:,c,i,j,k],optimize=True)
        d8 -= np.einsum('ef,ef->',H2B['vvvv'][a,c,:,:],t3c[:,:,b,i,j,k],optimize=True)

        d9 = np.einsum('me,em->',H2A['voov'][a,:,i,:],t3c[:,b,c,:,j,k],optimize=True)

        d10 = np.einsum('me,em->',H2B['voov'][a,:,i,:],t3d[:,b,c,:,j,k],optimize=True)

        d11 = np.einsum('me,em->',H2B['ovvo'][:,b,:,j],t3b[a,:,c,i,:,k],optimize=True)
        d11 -= np.einsum('me,em->',H2B['ovvo'][:,c,:,j],t3b[a,:,b,i,:,k],optimize=True)
        d11 -= np.einsum('me,em->',H2B['ovvo'][:,b,:,k],t3b[a,:,c,i,:,j],optimize=True)
        d11 += np.einsum('me,em->',H2B['ovvo'][:,c,:,k],t3b[a,:,b,i,:,j],optimize=True)

        d12 = np.einsum('me,em->',H2C['voov'][b,:,j,:],t3c[a,:,c,i,:,k],optimize=True)
        d12 -= np.einsum('me,em->',H2C['voov'][c,:,j,:],t3c[a,:,b,i,:,k],optimize=True)
        d12 -= np.einsum('me,em->',H2C['voov'][b,:,k,:],t3c[a,:,c,i,:,j],optimize=True)
        d12 += np.einsum('me,em->',H2C['voov'][c,:,k,:],t3c[a,:,b,i,:,j],optimize=True)

        d13 = -np.einsum('me,em->',H2B['ovov'][:,b,i,:],t3c[a,:,c,:,j,k],optimize=True)
        d13 -= -np.einsum('me,em->',H2B['ovov'][:,c,i,:],t3c[a,:,b,:,j,k],optimize=True)

        d14 = -np.einsum('me,em->',H2B['vovo'][a,:,:,j],t3c[:,b,c,i,:,k],optimize=True)
        d14 -= -np.einsum('me,em->',H2B['vovo'][a,:,:,k],t3c[:,b,c,i,:,j],optimize=True)

        residual = m1 + m2 + m3 + m4 + m5 + m6\
        + d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9 + d10\
        + d11 + d12 + d13 + d14
        denom = fA['oo'][i,i]+fB['oo'][j,j]+fB['oo'][k,k]\
        -fA['vv'][a,a]-fB['vv'][b,b]-fB['vv'][c,c]
        t3c_new[a,b,c,i,j,k] += residual/(denom-shift)
        t3c_new[a,c,b,i,j,k] = -t3c_new[a,b,c,i,j,k]
        t3c_new[a,b,c,i,k,j] = -t3c_new[a,b,c,i,j,k]
        t3c_new[a,c,b,i,k,j] = t3c_new[a,b,c,i,j,k]

    cc_t['t3c'] = t3c_new
    return cc_t

def update_t3d_inloop(cc_t,ints,list_of_triples_D,H1A,H1B,H2A,H2B,H2C,sys,shift):

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

    I2C_vvov = 0.0
    I2C_vvov += np.einsum('me,abim->abie',H1B['ov'],t2c,optimize=True)
    I2C_vvov += H2C['vvov']

    t3d_new = t3d.copy()
    for idx in list_of_triples_D:
        a = idx[0]; b = idx[1]; c = idx[2];
        i = idx[3]; j = idx[4]; k = idx[5];

        vt3int = 0.5*np.einsum('mnef,efn->m',vC['oovv'],t3d[a,:,:,i,j,:],optimize=True)\
            +np.einsum('nmfe,fen->m',vB['oovv'],t3c[:,a,:,:,i,j],optimize=True)
        m1 = -np.einsum('m,m->',H2C['vooo'][a,:,i,j]+vt3int,t2c[b,c,:,k],optimize=True)
        
        vt3int = 0.5*np.einsum('mnef,efn->m',vC['oovv'],t3d[b,:,:,i,j,:],optimize=True)\
            +np.einsum('nmfe,fen->m',vB['oovv'],t3c[:,b,:,:,i,j],optimize=True)
        m1 -= -np.einsum('m,m->',H2C['vooo'][b,:,i,j]+vt3int,t2c[a,c,:,k],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vC['oovv'],t3d[c,:,:,i,j,:],optimize=True)\
            +np.einsum('nmfe,fen->m',vB['oovv'],t3c[:,c,:,:,i,j],optimize=True)
        m1 -= -np.einsum('m,m->',H2C['vooo'][c,:,i,j]+vt3int,t2c[b,a,:,k],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vC['oovv'],t3d[a,:,:,k,j,:],optimize=True)\
            +np.einsum('nmfe,fen->m',vB['oovv'],t3c[:,a,:,:,k,j],optimize=True)
        m1 -= -np.einsum('m,m->',H2C['vooo'][a,:,k,j]+vt3int,t2c[b,c,:,i],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vC['oovv'],t3d[a,:,:,i,k,:],optimize=True)\
            +np.einsum('nmfe,fen->m',vB['oovv'],t3c[:,a,:,:,i,k],optimize=True)
        m1 -= -np.einsum('m,m->',H2C['vooo'][a,:,i,k]+vt3int,t2c[b,c,:,j],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vC['oovv'],t3d[b,:,:,k,j,:],optimize=True)\
            +np.einsum('nmfe,fen->m',vB['oovv'],t3c[:,b,:,:,k,j],optimize=True)
        m1 += -np.einsum('m,m->',H2C['vooo'][b,:,k,j]+vt3int,t2c[a,c,:,i],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vC['oovv'],t3d[b,:,:,i,k,:],optimize=True)\
            +np.einsum('nmfe,fen->m',vB['oovv'],t3c[:,b,:,:,i,k],optimize=True)
        m1 += -np.einsum('m,m->',H2C['vooo'][b,:,i,k]+vt3int,t2c[a,c,:,j],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vC['oovv'],t3d[c,:,:,k,j,:],optimize=True)\
            +np.einsum('nmfe,fen->m',vB['oovv'],t3c[:,c,:,:,k,j],optimize=True)
        m1 += -np.einsum('m,m->',H2C['vooo'][c,:,k,j]+vt3int,t2c[b,a,:,i],optimize=True)

        vt3int = 0.5*np.einsum('mnef,efn->m',vC['oovv'],t3d[c,:,:,i,k,:],optimize=True)\
            +np.einsum('nmfe,fen->m',vB['oovv'],t3c[:,c,:,:,i,k],optimize=True)
        m1 += -np.einsum('m,m->',H2C['vooo'][c,:,i,k]+vt3int,t2c[b,a,:,j],optimize=True)

        vt3int = -0.5*np.einsum('mnef,fmn->e',vC['oovv'],t3d[a,b,:,i,:,:],optimize=True)\
            -np.einsum('nmfe,fnm->e',vB['oovv'],t3c[:,a,b,:,i,:],optimize=True)
        m2 = np.einsum('e,e->',I2C_vvov[a,b,i,:]+vt3int,t2c[:,c,j,k],optimize=True)        

        vt3int = -0.5*np.einsum('mnef,fmn->e',vC['oovv'],t3d[a,c,:,i,:,:],optimize=True)\
            -np.einsum('nmfe,fnm->e',vB['oovv'],t3c[:,a,c,:,i,:],optimize=True)
        m2 -= np.einsum('e,e->',I2C_vvov[a,c,i,:]+vt3int,t2c[:,b,j,k],optimize=True)        
             
        vt3int = -0.5*np.einsum('mnef,fmn->e',vC['oovv'],t3d[c,b,:,i,:,:],optimize=True)\
            -np.einsum('nmfe,fnm->e',vB['oovv'],t3c[:,c,b,:,i,:],optimize=True)
        m2 -= np.einsum('e,e->',I2C_vvov[c,b,i,:]+vt3int,t2c[:,a,j,k],optimize=True)       

        vt3int = -0.5*np.einsum('mnef,fmn->e',vC['oovv'],t3d[a,b,:,j,:,:],optimize=True)\
            -np.einsum('nmfe,fnm->e',vB['oovv'],t3c[:,a,b,:,j,:],optimize=True)
        m2 -= np.einsum('e,e->',I2C_vvov[a,b,j,:]+vt3int,t2c[:,c,i,k],optimize=True)        

        vt3int = -0.5*np.einsum('mnef,fmn->e',vC['oovv'],t3d[a,b,:,k,:,:],optimize=True)\
            -np.einsum('nmfe,fnm->e',vB['oovv'],t3c[:,a,b,:,k,:],optimize=True)
        m2 -= np.einsum('e,e->',I2C_vvov[a,b,k,:]+vt3int,t2c[:,c,j,i],optimize=True)        

        vt3int = -0.5*np.einsum('mnef,fmn->e',vC['oovv'],t3d[c,b,:,j,:,:],optimize=True)\
            -np.einsum('nmfe,fnm->e',vB['oovv'],t3c[:,c,b,:,j,:],optimize=True)
        m2 += np.einsum('e,e->',I2C_vvov[c,b,j,:]+vt3int,t2c[:,a,i,k],optimize=True)        

        vt3int = -0.5*np.einsum('mnef,fmn->e',vC['oovv'],t3d[c,b,:,k,:,:],optimize=True)\
            -np.einsum('nmfe,fnm->e',vB['oovv'],t3c[:,c,b,:,k,:],optimize=True)
        m2 += np.einsum('e,e->',I2C_vvov[c,b,k,:]+vt3int,t2c[:,a,j,i],optimize=True)        

        vt3int = -0.5*np.einsum('mnef,fmn->e',vC['oovv'],t3d[a,c,:,j,:,:],optimize=True)\
            -np.einsum('nmfe,fnm->e',vB['oovv'],t3c[:,a,c,:,j,:],optimize=True)
        m2 += np.einsum('e,e->',I2C_vvov[a,c,j,:]+vt3int,t2c[:,b,i,k],optimize=True)       

        vt3int = -0.5*np.einsum('mnef,fmn->e',vC['oovv'],t3d[a,c,:,k,:,:],optimize=True)\
            -np.einsum('nmfe,fnm->e',vB['oovv'],t3c[:,a,c,:,k,:],optimize=True)
        m2 += np.einsum('e,e->',I2C_vvov[a,c,k,:]+vt3int,t2c[:,b,j,i],optimize=True)        

        d1 = -np.einsum('m,m->',H1B['oo'][:,k],t3d[a,b,c,i,j,:],optimize=True)
        d1 -= -np.einsum('m,m->',H1B['oo'][:,j],t3d[a,b,c,i,k,:],optimize=True)
        d1 -= -np.einsum('m,m->',H1B['oo'][:,i],t3d[a,b,c,k,j,:],optimize=True)

        d2 = np.einsum('e,e->',H1B['vv'][c,:],t3d[a,b,:,i,j,k],optimize=True)
        d2 -= np.einsum('e,e->',H1B['vv'][a,:],t3d[c,b,:,i,j,k],optimize=True)
        d2 -= np.einsum('e,e->',H1B['vv'][b,:],t3d[a,c,:,i,j,k],optimize=True)

        d3 = 0.5*np.einsum('mn,mn->',H2C['oooo'][:,:,i,j],t3d[a,b,c,:,:,k],optimize=True)
        d3 -= 0.5*np.einsum('mn,mn->',H2C['oooo'][:,:,k,j],t3d[a,b,c,:,:,i],optimize=True)
        d3 -= 0.5*np.einsum('mn,mn->',H2C['oooo'][:,:,i,k],t3d[a,b,c,:,:,j],optimize=True)

        d4 = 0.5*np.einsum('ef,ef->',H2C['vvvv'][a,b,:,:],t3d[:,:,c,i,j,k],optimize=True)
        d4 -= 0.5*np.einsum('ef,ef->',H2C['vvvv'][c,b,:,:],t3d[:,:,a,i,j,k],optimize=True)
        d4 -= 0.5*np.einsum('ef,ef->',H2C['vvvv'][a,c,:,:],t3d[:,:,b,i,j,k],optimize=True)

        d5 = np.einsum('me,em->',H2B['ovvo'][:,a,:,i],t3c[:,b,c,:,j,k],optimize=True)
        d5 -= np.einsum('me,em->',H2B['ovvo'][:,b,:,i],t3c[:,a,c,:,j,k],optimize=True)
        d5 -= np.einsum('me,em->',H2B['ovvo'][:,c,:,i],t3c[:,b,a,:,j,k],optimize=True)
        d5 -= np.einsum('me,em->',H2B['ovvo'][:,a,:,j],t3c[:,b,c,:,i,k],optimize=True)
        d5 -= np.einsum('me,em->',H2B['ovvo'][:,a,:,k],t3c[:,b,c,:,j,i],optimize=True)
        d5 += np.einsum('me,em->',H2B['ovvo'][:,c,:,j],t3c[:,b,a,:,i,k],optimize=True)
        d5 += np.einsum('me,em->',H2B['ovvo'][:,c,:,k],t3c[:,b,a,:,j,i],optimize=True)
        d5 += np.einsum('me,em->',H2B['ovvo'][:,b,:,j],t3c[:,a,c,:,i,k],optimize=True)
        d5 += np.einsum('me,em->',H2B['ovvo'][:,b,:,k],t3c[:,a,c,:,j,i],optimize=True)

        d6 = np.einsum('me,em->',H2C['voov'][a,:,i,:],t3d[:,b,c,:,j,k],optimize=True)
        d6 -= np.einsum('me,em->',H2C['voov'][b,:,i,:],t3d[:,a,c,:,j,k],optimize=True)
        d6 -= np.einsum('me,em->',H2C['voov'][c,:,i,:],t3d[:,b,a,:,j,k],optimize=True)
        d6 -= np.einsum('me,em->',H2C['voov'][a,:,j,:],t3d[:,b,c,:,i,k],optimize=True)
        d6 -= np.einsum('me,em->',H2C['voov'][a,:,k,:],t3d[:,b,c,:,j,i],optimize=True)
        d6 += np.einsum('me,em->',H2C['voov'][b,:,j,:],t3d[:,a,c,:,i,k],optimize=True)
        d6 += np.einsum('me,em->',H2C['voov'][b,:,k,:],t3d[:,a,c,:,j,i],optimize=True)
        d6 += np.einsum('me,em->',H2C['voov'][c,:,j,:],t3d[:,b,a,:,i,k],optimize=True)
        d6 += np.einsum('me,em->',H2C['voov'][c,:,k,:],t3d[:,b,a,:,j,i],optimize=True)

        residual = m1 + m2 + d1 + d2 + d3 + d4 + d5 + d6
        denom = fB['oo'][i,i]+fB['oo'][j,j]+fB['oo'][k,k]\
                 -fB['vv'][a,a]-fB['vv'][b,b]-fB['vv'][c,c]
        t3d_new[a,b,c,i,j,k] += residual/(denom-shift)
        t3d_new[b,a,c,i,j,k] = -t3d_new[a,b,c,i,j,k]
        t3d_new[a,c,b,i,j,k] = -t3d_new[a,b,c,i,j,k]
        t3d_new[b,c,a,i,j,k] = t3d_new[a,b,c,i,j,k]
        t3d_new[c,a,b,i,j,k] = t3d_new[a,b,c,i,j,k]
        t3d_new[c,b,a,i,j,k] = -t3d_new[a,b,c,i,j,k]
        t3d_new[a,b,c,j,i,k] = -t3d_new[a,b,c,i,j,k] 
        t3d_new[b,a,c,j,i,k] = t3d_new[a,b,c,i,j,k]
        t3d_new[a,c,b,j,i,k] = t3d_new[a,b,c,i,j,k]
        t3d_new[b,c,a,j,i,k] = -t3d_new[a,b,c,i,j,k]
        t3d_new[c,a,b,j,i,k] = -t3d_new[a,b,c,i,j,k]
        t3d_new[c,b,a,j,i,k] = t3d_new[a,b,c,i,j,k]
        t3d_new[a,b,c,i,k,j] = -t3d_new[a,b,c,i,j,k] 
        t3d_new[b,a,c,i,k,j] = t3d_new[a,b,c,i,j,k]
        t3d_new[a,c,b,i,k,j] = t3d_new[a,b,c,i,j,k]
        t3d_new[b,c,a,i,k,j] = -t3d_new[a,b,c,i,j,k]
        t3d_new[c,a,b,i,k,j] = -t3d_new[a,b,c,i,j,k]
        t3d_new[c,b,a,i,k,j] = t3d_new[a,b,c,i,j,k]
        t3d_new[a,b,c,j,k,i] = t3d_new[a,b,c,i,j,k] 
        t3d_new[b,a,c,j,k,i] = -t3d_new[a,b,c,i,j,k]
        t3d_new[a,c,b,j,k,i] = -t3d_new[a,b,c,i,j,k]
        t3d_new[b,c,a,j,k,i] = t3d_new[a,b,c,i,j,k]
        t3d_new[c,a,b,j,k,i] = t3d_new[a,b,c,i,j,k]
        t3d_new[c,b,a,j,k,i] = -t3d_new[a,b,c,i,j,k]
        t3d_new[a,b,c,k,i,j] = t3d_new[a,b,c,i,j,k]
        t3d_new[b,a,c,k,i,j] = -t3d_new[a,b,c,i,j,k]
        t3d_new[a,c,b,k,i,j] = -t3d_new[a,b,c,i,j,k]
        t3d_new[b,c,a,k,i,j] = t3d_new[a,b,c,i,j,k]
        t3d_new[c,a,b,k,i,j] = t3d_new[a,b,c,i,j,k]
        t3d_new[c,b,a,k,i,j] = -t3d_new[a,b,c,i,j,k]
        t3d_new[a,b,c,k,j,i] = -t3d_new[a,b,c,i,j,k] 
        t3d_new[b,a,c,k,j,i] = t3d_new[a,b,c,i,j,k]
        t3d_new[a,c,b,k,j,i] = t3d_new[a,b,c,i,j,k]
        t3d_new[b,c,a,k,j,i] = -t3d_new[a,b,c,i,j,k]
        t3d_new[c,a,b,k,j,i] = -t3d_new[a,b,c,i,j,k]
        t3d_new[c,b,a,k,j,i] = t3d_new[a,b,c,i,j,k]

    cc_t['t3d'] = t3d_new
    return cc_t

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








