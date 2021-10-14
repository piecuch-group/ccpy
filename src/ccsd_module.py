
import numpy as np
from solvers import diis
import time
import cc_loops

#print(cc_loops.cc_loops.__doc__)


def ccsd(sys,ints,maxit=100,tol=1e-08,diis_size=6,shift=0.0):

    print('\n==================================++Entering CCSD Routine++=================================\n')


    n1a = sys['Nocc_a'] * sys['Nunocc_a']
    n1b = sys['Nocc_b'] * sys['Nunocc_b']
    n2a = sys['Nocc_a'] ** 2 * sys['Nunocc_a'] ** 2
    n2b = sys['Nocc_a'] * sys['Nocc_b'] * sys['Nunocc_a'] * sys['Nunocc_b']
    n2c = sys['Nocc_b'] ** 2 * sys['Nunocc_b'] ** 2
    ndim = n1a + n1b + n2a + n2b + n2c
    idx_1a = slice(0,n1a)
    idx_1b = slice(n1a,n1a+n1b)
    idx_2a = slice(n1a+n1b,n1a+n1b+n2a)
    idx_2b = slice(n1a+n1b+n2a,n1a+n1b+n2a+n2b)
    idx_2c = slice(n1a+n1b+n2a+n2b,n1a+n1b+n2a+n2b+n2c)

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
        
        # store vectorized results
        T[idx_1a]= cc_t['t1a'].flatten()
        T[idx_1b] = cc_t['t1b'].flatten()
        T[idx_2a] = cc_t['t2a'].flatten()
        T[idx_2b] = cc_t['t2b'].flatten()
        T[idx_2c] = cc_t['t2c'].flatten()

        # build DIIS residual
        T_resid = T - T_old
        
        # change in Ecorr
        deltaE = Ecorr - Ecorr_old

        # check for exit condition
        ccsd_resid = np.linalg.norm(T_resid)
        if ccsd_resid < tol and abs(deltaE) < tol:
            flag_conv = True
            break

        # append trial and residual vectors to lists
        T_list[:,it_micro%diis_size] = T
        T_resid_list[:,it_micro%diis_size] = T_resid
        
        if it_micro%diis_size == 0 and it_micro > 1:
            it_macro = it_macro + 1
            print('DIIS Cycle - {}'.format(it_macro))
            T = diis(T_list,T_resid_list)
        
        print('   {}       {:.10f}          {:.10f}          {:.10f}'.format(it_micro,ccsd_resid,deltaE,Ecorr))
        
        it_micro += 1
        Ecorr_old = Ecorr

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    if flag_conv:
        print('CCSD successfully converged! ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))
        print('')
        print('CCSD Correlation Energy = {} Eh'.format(Ecorr))
        print('CCSD Total Energy = {} Eh'.format(Ecorr+ints['Escf']))
    else:
        print('Failed to converge CCSD in {} iterations'.format(maxit))

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
    
    # diagrams that have A(ab)
    D13 = D1 + D3
    D13 = D13 - np.einsum('abij->baij',D13)
    
    # diagrams that have A(ij)
    D24 = D2 + D4
    D24 = D24 - np.einsum('abij->abji',D24)
        
    # diagrams that have A(ab)A(ij)
    D56 = D5 + D6
    D56 = D56 - np.einsum('abij->baij',D56) - np.einsum('abij->abji',D56) + np.einsum('abij->baji',D56)
    
    # total contribution
    X2A += D13 + D24 + D56

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
    
    # diagrams that have A(ab)
    D13 = D1 + D3
    D13 = D13 - np.einsum('abij->baij',D13)
    
    # diagrams that have A(ij)
    D24 = D2 + D4
    D24 = D24 - np.einsum('abij->abji',D24)
        
    # diagrams that have A(ab)A(ij)
    D56 = D5 + D6
    D56 = D56 - np.einsum('abij->baij',D56) - np.einsum('abij->abji',D56) + np.einsum('abij->baji',D56)
    
    # total contribution
    X2C += D13 + D24 + D56

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
                

    H1A = {'ov' : h1A_ov, 'oo' : h1A_oo, 'vv' : h1A_vv}
    H1B = {'ov' : h1B_ov, 'oo' : h1B_oo, 'vv' : h1B_vv}
    H2A = {'oooo' : h2A_oooo, 'vvvv' : h2A_vvvv, 'vvov' : h2A_vvov, 'vooo' : h2A_vooo, 'voov' : h2A_voov}
    H2B = {'oooo' : h2B_oooo, 'vvvv' : h2B_vvvv, 'ovov' : h2B_ovov, 'voov' : h2B_voov, 
           'ovvo' : h2B_ovvo, 'vovo' : h2B_vovo, 'ovoo' : h2B_ovoo, 'vooo' : h2B_vooo, 
           'vvvo' : h2B_vvvo, 'vvov' : h2B_vvov}
    H2C = {'oooo' : h2C_oooo, 'vvvv' : h2C_vvvv, 'vvvo' : h2C_vvvo, 'ovoo' : h2C_ovoo, 'voov' : h2C_voov}
           

    return H1A, H1B, H2A, H2B, H2C

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

    cc_t = {'t1a' : t1a, 't1b' : t1b, 't2a' : t2a, 't2b' : t2b, 't2c' : t2c}

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

    # test CCS HBar components
    for key,item in H1A.items():
        print('|H1A({})| = {}'.format(key,np.linalg.norm(item)))
    for key,item in H1B.items():
        print('|H1B({})| = {}'.format(key,np.linalg.norm(item)))
    for key,item in H2A.items():
        print('|H2A({})| = {}'.format(key,np.linalg.norm(item)))
    for key,item in H2B.items():
        print('|H2B({})| = {}'.format(key,np.linalg.norm(item)))
    for key,item in H2C.items():
        print('|H2C({})| = {}'.format(key,np.linalg.norm(item)))

    return

