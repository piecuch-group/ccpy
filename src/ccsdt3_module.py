import numpy as np
from solvers import diis
import time
import cc_loops

def ccsdt3(sys,ints,nact_o_alpha,nact_o_beta,nact_u_alpha,nact_u_beta,maxit=100,tol=1e-08,diis_size=6,shift=0.0):

    print('\n==================================++Entering CCSDt(III) Routine++=================================\n')


    n1a = sys['Nocc_a'] * sys['Nunocc_a']
    n1b = sys['Nocc_b'] * sys['Nunocc_b']
    n2a = sys['Nocc_a'] ** 2 * sys['Nunocc_a'] ** 2
    n2b = sys['Nocc_a'] * sys['Nocc_b'] * sys['Nunocc_a'] * sys['Nunocc_b']
    n2c = sys['Nocc_b'] ** 2 * sys['Nunocc_b'] ** 2
    n3a = nact_o_alpha ** 3 * nact_u_alpha ** 3
    n3b = nact_o_alpha ** 2 * nact_o_beta * nact_u_alpha ** 2 * nact_u_beta
    n3c = nact_o_alpha * nact_o_beta ** 2 * nact_u_alpha * nact_u_beta ** 2
    n3d = nact_o_beta ** 3 * nact_u_beta ** 3

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
        cc_t['t1a']  = np.reshape(T[idx_1a],(sys['Nunocc_a'],sys['Nocc_a']))
        cc_t['t1b']  = np.reshape(T[idx_1b],(sys['Nunocc_b'],sys['Nocc_b']))
        cc_t['t2a']  = np.reshape(T[idx_2a],(sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a']))
        cc_t['t2b']  = np.reshape(T[idx_2b],(sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b']))
        cc_t['t2c']  = np.reshape(T[idx_2c],(sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b']))
        cc_t['t3a']  = np.reshape(T[idx_3a],(nact_u_alpha,nact_u_alpha,nact_u_alpha,nact_o_alpha,nact_o_alpha,nact_o_alpha))
        cc_t['t3b']  = np.reshape(T[idx_3b],(nact_u_alpha,nact_u_alpha,nact_u_beta,nact_o_alpha,nact_o_alpha,nact_o_beta))
        cc_t['t3c']  = np.reshape(T[idx_3c],(nact_u_alpha,nact_u_beta,nact_u_beta,nact_o_alpha,nact_o_beta,nact_o_beta))
        cc_t['t3d']  = np.reshape(T[idx_3d],(nact_u_beta,nact_u_beta,nact_u_beta,nact_o_beta,nact_o_beta,nact_o_beta))

        # CC correlation energy
        Ecorr = calc_cc_energy(cc_t,ints)
       
        # update T1                        
        cc_t = update_t1a(cc_t,ints,sys,shift)
        cc_t = update_t1b(cc_t,ints,sys,shift)

        # CCS intermediates
        H1A,H1B,H2A,H2B,H2C = get_ccs_intermediates(cc_t,ints,sys)

        # update T2
        cc_t = update_t2a(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        cc_t = update_t2b(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        cc_t = update_t2c(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)

        # CCSD intermediates
        H1A,H1B,H2A,H2B,H2C = get_ccsd_intermediates(cc_t,ints,sys)

        # update T3
        cc_t = update_t3a(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        cc_t = update_t3b(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        cc_t = update_t3c(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        cc_t = update_t3d(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
        
        # store vectorized results
        T[idx_1a]= cc_t['t1a'].flatten()
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
        print('CCSDt(III) successfully converged! ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))
        print('')
        print('CCSDt(III) Correlation Energy = {} Eh'.format(Ecorr))
        print('CCSDt(III) Total Energy = {} Eh'.format(Ecorr + ints['Escf']))
    else:
        print('Failed to converge CCSDt(III) in {} iterations'.format(maxit))

    return cc_t, ints['Escf'] + Ecorr

def update_t1a(cc_t,ints,sys,shift,nact_o_alpha,nact_o_beta,nact_u_alpha,nact_u_beta):

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

    iHA = slice(sys['Nocc_a']-nact_o_alpha, sys['Nocc_a'])
    iHB = slice(sys['Nocc_b']-nact_o_beta, sys['Nocc_b'])
    iPA = slice(0, nact_u_alpha)
    iPB = slice(0, nact_u_beta)

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

    X1A += 0.25*np.einsum('mnef,aefimn->ai',vA['oovv'][iHA,iHA,iPA,iPA],t3a,optimize=True)
    X1A += np.einsum('mnef,aefimn->ai',vB['oovv'][iHA,iHB,iPA,iPB],t3b,optimize=True)
    X1A += 0.25*np.einsum('mnef,aefimn->ai',vC['oovv'][iHB,iHB,iPB,iPB],t3c,optimize=True)

    t1a = cc_loops.cc_loops.update_t1a(t1a,X1A,fA['oo'],fA['vv'],shift)

    cc_t['t1a'] = t1a

    return cc_t

def update_t1b(cc_t,ints,sys,shift,nact_o_alpha,nact_o_beta,nact_u_alpha,nact_u_beta):

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

    iHA = slice(sys['Nocc_a']-nact_o_alpha, sys['Nocc_a'])
    iHB = slice(sys['Nocc_b']-nact_o_beta, sys['Nocc_b'])
    iPA = slice(0, nact_u_alpha)
    iPB = slice(0, nact_u_beta)

    chi1B_vv = 0.0
    chi1B_vv += fB['vv']
    chi1B_vv += np.einsum('anef,fn->ae',vC['vovv'],t1b,optimize=True)
    chi1B_vv += np.einsum('nafe,fn->ae',vB['ovvv'],t1a,optimize=True)

    chi1B_oo = 0.0
    chi1B_oo += fB['oo']
    chi1B_oo += np.einsum('mnif,fn->mi',vC['ooov'],t1b,optimize=True)
    chi1B_oo += np.einsum('nmfi,fn->mi',vB['oovo'],t1b,optimize=True)

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

def update_t3a(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift):

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

    t3a = cc_loops.cc_loops.update_t3a(t3a,X3A,fA['oo'],fA['vv'],shift)

    cc_t['t3a'] = t3a
    return cc_t

def update_t3b(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift):

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

    t3b = cc_loops.cc_loops.update_t3b(t3b,X3B,fA['oo'],fA['vv'],fB['oo'],fB['vv'],shift)
             
    cc_t['t3b'] = t3b
    return cc_t

def update_t3c(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift):

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

    t3c = cc_loops.cc_loops.update_t3c(t3c,X3C,fA['oo'],fA['vv'],fB['oo'],fB['vv'],shift)
             
    cc_t['t3c'] = t3c
    return cc_t

def update_t3d(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift):

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

    t3d = cc_loops.cc_loops.update_t3d(t3d,X3D,fB['oo'],fB['vv'],shift)

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

def test_updates(matfile,ints,sys):

    from scipy.io import loadmat

    print('')
    print('TEST SUBROUTINE:')
    print('Loading Matlab .mat file from {}'.format(matfile))
    print('')

    data_dict = loadmat(matfile)
    cc_t = data_dict['cc_t']

    t1a = cc_t['t1a'][0,0]
    t1b = cc_t['t1b'][0,0]
    t2a = cc_t['t2a'][0,0]
    t2b = cc_t['t2b'][0,0]
    t2c = cc_t['t2c'][0,0]
    t3a = cc_t['t3a'][0,0]
    t3b = cc_t['t3b'][0,0]
    t3c = cc_t['t3c'][0,0]
    t3d = cc_t['t3d'][0,0]

    cc_t = {'t1a' : t1a, 't1b' : t1b, 't2a' : t2a, 't2b' : t2b, 't2c' : t2c,
            't3a' : t3a, 't3b' : t3b, 't3c' : t3c, 't3d' : t3d}

    Ecorr = calc_cc_energy(cc_t,ints)
    print('Correlation energy = {}'.format(Ecorr))

    shift = 0.0

    # test t1a update
    out = update_t1a(cc_t,ints,sys,shift)
    t1a = out['t1a']
    print('|t1a| = {}'.format(np.linalg.norm(t1a)))

    # test t1b update
    out = update_t1b(cc_t,ints,sys,shift)
    t1a = out['t1b']
    print('|t1b| = {}'.format(np.linalg.norm(t1b)))

    H1A,H1B,H2A,H2B,H2C = get_ccs_intermediates(cc_t,ints,sys)

    # test t2a update
    out = update_t2a(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
    t2a = out['t2a']
    print('|t2a| = {}'.format(np.linalg.norm(t2a)))

    # test t2b update
    out = update_t2b(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
    t1a = out['t2b']
    print('|t2b| = {}'.format(np.linalg.norm(t2b)))

    # test t2c update
    out = update_t2c(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
    t2c = out['t2c']
    print('|t2c| = {}'.format(np.linalg.norm(t2c)))

    H1A,H1B,H2A,H2B,H2C = get_ccsd_intermediates(cc_t,ints,sys)

    # test t3a update
    out = update_t3a(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
    t3a = out['t3a']
    print('|t3a| = {}'.format(np.linalg.norm(t3a)))

    # test t3b update
    out = update_t3b(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
    t3b = out['t3b']
    print('|t3b| = {}'.format(np.linalg.norm(t3b)))

    # test t3c update
    out = update_t3c(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
    t3c = out['t3c']
    print('|t3c| = {}'.format(np.linalg.norm(t3c)))

    # test t3d update
    out = update_t3d(cc_t,ints,H1A,H1B,H2A,H2B,H2C,sys,shift)
    t3d = out['t3d']
    print('|t3d| = {}'.format(np.linalg.norm(t3d)))

    return
