import numpy as np
from solvers import diis
import time

def ccs(sys,ints,maxit=100,tol=1e-08,diis_size=6,shift=0.0):

    print('\n==================================++Entering CCS Routine++=================================\n')


    n1a = sys['Nocc_a'] * sys['Nunocc_a']
    n1b = sys['Nocc_b'] * sys['Nunocc_b']
    ndim = n1a + n1b
    idx_1a = slice(0,n1a)
    idx_1b = slice(n1a,n1a+n1b)

    cc_t = {}
    T = np.zeros(ndim)
    T_list = np.zeros((ndim,diis_size))
    T_resid_list = np.zeros((ndim,diis_size))
    T_old = np.zeros(ndim)

    cc_t = second_order_guess(sys,ints)
    T[idx_1a] = cc_t['t1a'].flatten()
    T[idx_1b] = cc_t['t1b'].flatten()

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

        # CC correlation energy
        Ecorr = calc_cc_energy(cc_t,ints)
       
        # update T1                        
        cc_t = update_t1a(cc_t,ints,sys,shift)
        cc_t = update_t1b(cc_t,ints,sys,shift)
        
        # store vectorized results
        T[idx_1a] = cc_t['t1a'].flatten()
        T[idx_1b] = cc_t['t1b'].flatten()

        # build DIIS residual
        T_resid = T - T_old
        
        # change in Ecorr
        deltaE = Ecorr - Ecorr_old

        # check for exit condition
        ccs_resid = np.linalg.norm(T_resid)
        if ccs_resid < tol and abs(deltaE) < tol:
            flag_conv = True
            break

        # append trial and residual vectors to lists
        T_list[:,it_micro%diis_size] = T
        T_resid_list[:,it_micro%diis_size] = T_resid
        
        if it_micro%diis_size == 0 and it_micro > 1:
            it_macro = it_macro + 1
            print('DIIS Cycle - {}'.format(it_macro))
            T = diis(T_list,T_resid_list)
        
        print('   {}       {:.10f}          {:.10f}          {:.10f}'.format(it_micro,ccs_resid,deltaE,Ecorr))
        
        it_micro += 1
        Ecorr_old = Ecorr

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    if flag_conv:
        print('CCS successfully converged! ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))
        print('')
        print('CCS Correlation Energy = {} Eh'.format(Ecorr))
        print('CCS Total Energy = {} Eh'.format(Ecorr+ints['Escf']))
    else:
        print('Failed to converge CCS in {} iterations'.format(maxit))

    return cc_t, ints['Escf'] + Ecorr

def update_t1a(cc_t,ints,sys,shift):

    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    fA = ints['fA']
    fB = ints['fB']
    t1a = cc_t['t1a']
    t1b = cc_t['t1b']

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
       
    X1A = M11

    for a in range(sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            denom = fA['oo'][i,i] - fA['vv'][a,a]
            t1a[a,i] += X1A[a,i]/(denom-shift)

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
       
    X1B = M11
    
    for a in range(sys['Nunocc_b']):
        for i in range(sys['Nocc_b']):
            denom = fB['oo'][i,i] - fB['vv'][a,a]
            t1b[a,i] += X1B[a,i]/(denom-shift)

    cc_t['t1b'] = t1b        
    return cc_t

def second_order_guess(sys,ints):

    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']

    t1a = np.zeros((sys['Nunocc_a'],sys['Nocc_a']))
    t1b = np.zeros((sys['Nunocc_b'],sys['Nocc_b']))

    for a in range(sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            val = 0.0
            for m in range(sys['Nocc_a']):
                for n in range(m+1,sys['Nocc_a']):
                    for f in range(sys['Nunocc_a']):
                        val -= 0.5*vA['ooov'][m,n,i,f]*vA['vvoo'][a,f,m,n]
                for n in range(sys['Nocc_b']):
                    for f in range(sys['Nunocc_b']):
                        val -= vB['ooov'][m,n,i,f]*vB['vvoo'][a,f,m,n]
            for e in range(sys['Nunocc_a']):
                for f in range(e+1,sys['Nunocc_a']):
                    for n in range(sys['Nocc_a']):
                        val += 0.5*vA['vovv'][a,n,e,f]*vA['vvoo'][e,f,i,n]
                for f in range(sys['Nunocc_b']):
                    for n in range(sys['Nocc_b']):
                        val += vB['vovv'][a,n,e,f]*vB['vvoo'][e,f,i,n]
            t1a[a,i] = val
            t1b[a,i] = val

    cc_t = {'t1a' : t1a, 't1b' : t1b}

    return cc_t

def calc_cc_energy(cc_t,ints):

    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    fA = ints['fA']
    fB = ints['fB']
    t1a = cc_t['t1a']
    t1b = cc_t['t1b']

    Ecorr = 0.0
    Ecorr += np.einsum('me,em->',fA['ov'],t1a,optimize=True)
    Ecorr += np.einsum('me,em->',fB['ov'],t1b,optimize=True)
    Ecorr += 0.5*np.einsum('mnef,fn,em->',vA['oovv'],t1a,t1a,optimize=True)
    Ecorr += 0.5*np.einsum('mnef,fn,em->',vC['oovv'],t1b,t1b,optimize=True)
    Ecorr += np.einsum('mnef,em,fn->',vB['oovv'],t1a,t1b,optimize=True)

    return Ecorr

