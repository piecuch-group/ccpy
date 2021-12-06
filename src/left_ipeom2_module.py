import numpy as np
from solvers import diis
import time
import cc_loops

def left_ipeom2(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,nroot,omega,maxit=100,tol=1e-08,diis_size=6,shift=0.0,eom_tol=1.0e-06,eom_lccshift=0.0,eom_maxit=200):

    print('\n==================================++Entering Left-IPEOM(2h-1p) Routine++=================================\n')


    cc_t['l1a'] = [None]*(nroot+1)
    cc_t['l1b'] = [None]*(nroot+1)
    cc_t['l2a'] = [None]*(nroot+1)
    cc_t['l2b'] = [None]*(nroot+1)
    cc_t['l2c'] = [None]*(nroot+1)


    for iroot in range(nroot+1):

        if iroot == 0:
            print('\nPerforming left CC iterations for ground state')
            L = np.hstack((cc_t['t1a'].flatten(),\
            cc_t['t1b'].flatten(),\
            cc_t['t2a'].flatten(),\
            cc_t['t2b'].flatten(),\
            cc_t['t2c'].flatten()))
            L_list = np.zeros((ndim,diis_size))
            L_resid_list = np.zeros((ndim,diis_size))
            omega_mu = 0.0
            shiftval = shift
            tolval = tol
            maxitval = maxit

        else:
            print('\nPerforming left CC iterations for root {}'.format(iroot))
            L = np.hstack((cc_t['r1a'][iroot-1].flatten(),\
            cc_t['r1b'][iroot-1].flatten(),\
            cc_t['r2a'][iroot-1].flatten(),\
            cc_t['r2b'][iroot-1].flatten(),\
            cc_t['r2c'][iroot-1].flatten()))
            R = L.copy()
            #L = Lvec0[:,iroot-1]
            shiftval = eom_lccshift
            tolval = eom_tol
            maxitval = eom_maxit
            L_list = np.zeros((ndim,diis_size))
            L_resid_list = np.zeros((ndim,diis_size))
            omega_mu = omega[iroot-1] 

        # Jacobi/DIIS iterations
        it_micro = 0
        flag_conv = False
        it_macro = 0
        Ecorr = calc_cc_energy(cc_t,ints)

        print('SHIFT = {}'.format(shiftval))
        print('TOLERANCE = {}'.format(tolval))
        print('MAXIT = {}'.format(maxitval))
        print('EXCITATION ENERGY = {}\n'.format(omega_mu))

        t_start = time.time()
        print('Iteration    Residuum            E - Ecorr             omega')
        print('===================================================================')
        while it_micro < maxitval:

            # get current L1 and L2
            cc_t['l1a'][iroot]  = np.reshape(L[idx_1a],(sys['Nunocc_a'],sys['Nocc_a']))
            cc_t['l1b'][iroot]  = np.reshape(L[idx_1b],(sys['Nunocc_b'],sys['Nocc_b']))
            cc_t['l2a'][iroot]  = np.reshape(L[idx_2a],(sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a']))
            cc_t['l2b'][iroot]  = np.reshape(L[idx_2b],(sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b']))
            cc_t['l2c'][iroot]  = np.reshape(L[idx_2c],(sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b']))

            # update L1 and L2 by Jacobi
            X1A = build_LH_1A(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys)
            X1B = build_LH_1B(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys)
            X2A = build_LH_2A(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys)
            X2B = build_LH_2B(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys)
            X2C = build_LH_2C(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys)

            #l1a,l1b,l2a,l2b,l2c = update_L(l1a,l1b,l2a,l2b,l2c,X1A,X1B,X2A,X2B,X2C,omega,H1A,H1B,sys,shift):
            l1a,l1b,l2a,l2b,l2c = cc_loops.cc_loops.update_l(cc_t['l1a'][iroot],cc_t['l1b'][iroot],\
                        cc_t['l2a'][iroot],cc_t['l2b'][iroot],cc_t['l2c'][iroot],\
            X1A,X1B,X2A,X2B,X2C,omega_mu,H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],shiftval)

            cc_t['l1a'][iroot] = l1a; cc_t['l1b'][iroot] = l1b; cc_t['l2a'][iroot] = l2a; cc_t['l2b'][iroot] = l2b; cc_t['l2c'][iroot] = l2c
       
            L = np.hstack((l1a.flatten(),l1b.flatten(),l2a.flatten(),l2b.flatten(),l2c.flatten()))

            # build LH - omega*L residual measure (use full LH)
            X1A = build_LH_1A(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys)
            X1B = build_LH_1B(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys)
            X2A = build_LH_2A(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys)
            X2B = build_LH_2B(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys)
            X2C = build_LH_2C(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys)
            LH = np.hstack((X1A.flatten(),X1B.flatten(),X2A.flatten(),X2B.flatten(),X2C.flatten()))
            L_resid  = LH -  omega_mu * L

            # get lcc energy - returns omega
            E_lcc = np.sqrt(np.sum(LH**2))/np.sqrt(np.sum(L**2))
            err_energy = E_lcc - omega_mu

            # append trial and residual vectors to lists
            L_list[:,it_micro%diis_size] = L
            L_resid_list[:,it_micro%diis_size] = L_resid
        
            if it_micro%diis_size == 0 and it_micro > 1:
                it_macro = it_macro + 1
                print('DIIS Cycle - {}'.format(it_macro))
                L = diis(L_list,L_resid_list)

            # biorthogonalize to R for excited states
            if iroot != 0:
                LR = np.dot(L.T,R)
                L *= 1.0/LR 
            resid = np.linalg.norm(L_resid)

            print('   {}       {:.10f}        {:.10f}        {:.10f}'.format(it_micro,resid,err_energy,E_lcc))
        
            # check for exit condition
            if resid < tolval and abs(err_energy) < tolval:
                flag_conv = True
                break

            it_micro += 1
    
        t_end = time.time()
        minutes, seconds = divmod(t_end-t_start, 60)
        if flag_conv:
            print('Left-CCSD successfully converged! ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))
        else:
            print('Failed to converge left-CCSD in {} iterations'.format(maxitval))
        
        # explicitly enforce birothonormality
        if iroot != 0:
            LR = 0.0
            LR += np.einsum('em,em->',cc_t['r1a'][iroot-1],cc_t['l1a'][iroot],optimize=True)
            LR += np.einsum('em,em->',cc_t['r1b'][iroot-1],cc_t['l1b'][iroot],optimize=True)
            LR += 0.25*np.einsum('efmn,efmn->',cc_t['r2a'][iroot-1],cc_t['l2a'][iroot],optimize=True)
            LR += np.einsum('efmn,efmn->',cc_t['r2b'][iroot-1],cc_t['l2b'][iroot],optimize=True)
            LR += 0.25*np.einsum('efmn,efmn->',cc_t['r2c'][iroot-1],cc_t['l2c'][iroot],optimize=True)
            cc_t['l1a'][iroot] *= 1.0/LR
            cc_t['l1b'][iroot] *= 1.0/LR
            cc_t['l2a'][iroot] *= 1.0/LR
            cc_t['l2b'][iroot] *= 1.0/LR
            cc_t['l2c'][iroot] *= 1.0/LR

    return cc_t

def build_LH_1A(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys):

    l1a = cc_t['l1a'][iroot]
    l1b = cc_t['l1b'][iroot]
    l2a = cc_t['l2a'][iroot]
    l2b = cc_t['l2b'][iroot]
    l2c = cc_t['l2c'][iroot]

    t1a = cc_t['t1a']
    t1b = cc_t['t1b']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']

    X1A = np.zeros((sys['Nocc_a'],sys['Nunocc_a']))

    X1A += np.einsum('ea,ei->ia',H1A['vv'],l1a,optimize=True)
    X1A -= np.einsum('im,am->ia',H1A['oo'],l1a,optimize=True)
    X1A += np.einsum('eima,em->ia',H2A['voov'],l1a,optimize=True)
    X1A += np.einsum('ieam,em->ia',H2B['ovvo'],l1b,optimize=True)
    X1A += 0.5*np.einsum('fena,efin->ia',H2A['vvov'],l2a,optimize=True)
    X1A += np.einsum('efan,efin->ia',H2B['vvvo'],l2b,optimize=True)
    X1A -= 0.5*np.einsum('finm,afmn->ia',H2A['vooo'],l2a,optimize=True)
    X1A -= np.einsum('ifmn,afmn->ia',H2B['ovoo'],l2b,optimize=True)

    I1 = 0.25*np.einsum('efmn,fgnm->ge',l2a,t2a,optimize=True)
    I2 = -0.25*np.einsum('efmn,egnm->gf',l2a,t2a,optimize=True)
    I3 = -0.25*np.einsum('efmo,efno->mn',l2a,t2a,optimize=True)
    I4 = 0.25*np.einsum('efmo,efnm->on',l2a,t2a,optimize=True)
    X1A += np.einsum('ge,eiga->ia',I1,H2A['vovv'],optimize=True)
    X1A += np.einsum('gf,figa->ia',I2,H2A['vovv'],optimize=True)
    X1A += np.einsum('mn,nima->ia',I3,H2A['ooov'],optimize=True)
    X1A += np.einsum('on,nioa->ia',I4,H2A['ooov'],optimize=True)

    I1 = -np.einsum('abij,abin->jn',l2b,t2b,optimize=True)
    I2 = np.einsum('abij,afij->fb',l2b,t2b,optimize=True)
    I3 = np.einsum('abij,fbij->fa',l2b,t2b,optimize=True)
    I4 = -np.einsum('abij,abnj->in',l2b,t2b,optimize=True)
    X1A += np.einsum('jn,mnej->me',I1,H2B['oovo'],optimize=True)
    X1A += np.einsum('fb,mbef->me',I2,H2B['ovvv'],optimize=True)
    X1A += np.einsum('fa,amfe->me',I3,H2A['vovv'],optimize=True)
    X1A += np.einsum('in,nmie->me',I4,H2A['ooov'],optimize=True)

    I1 = 0.25*np.einsum('abij,fbij->fa',l2c,t2c,optimize=True)
    I2 = -0.25*np.einsum('abij,faij->fb',l2c,t2c,optimize=True)
    I3 = -0.25*np.einsum('abij,abnj->in',l2c,t2c,optimize=True)
    I4 = 0.25*np.einsum('abij,abni->jn',l2c,t2c,optimize=True)
    X1A += np.einsum('fa,maef->me',I1,H2B['ovvv'],optimize=True)
    X1A += np.einsum('fb,mbef->me',I2,H2B['ovvv'],optimize=True)
    X1A += np.einsum('in,mnei->me',I3,H2B['oovo'],optimize=True)
    X1A += np.einsum('jn,mnej->me',I4,H2B['oovo'],optimize=True)
    
    if iroot == 0:
        X1A += H1A['ov']

    return X1A.transpose((1,0))

def build_LH_1B(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys):

    l1a = cc_t['l1a'][iroot]
    l1b = cc_t['l1b'][iroot]
    l2a = cc_t['l2a'][iroot]
    l2b = cc_t['l2b'][iroot]
    l2c = cc_t['l2c'][iroot]

    t1a = cc_t['t1a']
    t1b = cc_t['t1b']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']

    X1B = np.zeros((sys['Nocc_b'], sys['Nunocc_b']))
    X1B += np.einsum('ea,ei->ia',H1B['vv'],l1b,optimize=True)
    X1B -= np.einsum('im,am->ia',H1B['oo'],l1b,optimize=True)
    X1B += np.einsum('eima,em->ia',H2B['voov'],l1a,optimize=True)
    X1B += np.einsum('eima,em->ia',H2C['voov'],l1b,optimize=True)
    X1B -= 0.5*np.einsum('finm,afmn->ia',H2C['vooo'],l2c,optimize=True)
    X1B -= np.einsum('finm,fanm->ia',H2B['vooo'],l2b,optimize=True)
    X1B += np.einsum('fena,feni->ia',H2B['vvov'],l2b,optimize=True)
    X1B += 0.5*np.einsum('fena,efin->ia',H2C['vvov'],l2c,optimize=True)
    
    I1 = 0.25*np.einsum('efmn,fgnm->ge',l2c,t2c,optimize=True)
    I2 = -0.25*np.einsum('efmn,egnm->gf',l2c,t2c,optimize=True)
    I3 = -0.25*np.einsum('efmn,efon->mo',l2c,t2c,optimize=True)
    I4 = 0.25*np.einsum('efmn,efom->no',l2c,t2c,optimize=True)
    X1B += np.einsum('ge,eiga->ia',I1,H2C['vovv'],optimize=True)\
    +np.einsum('gf,figa->ia',I2,H2C['vovv'],optimize=True)\
    +np.einsum('mo,oima->ia',I3,H2C['ooov'],optimize=True)\
    +np.einsum('no,oina->ia',I4,H2C['ooov'],optimize=True)

    I1 = 0.25*np.einsum('efmn,fgnm->ge',l2a,t2a,optimize=True)
    I2 = -0.25*np.einsum('efmn,egnm->gf',l2a,t2a,optimize=True)
    I3 = -0.25*np.einsum('efmn,efon->mo',l2a,t2a,optimize=True)
    I4 = 0.25*np.einsum('efmn,efom->no',l2a,t2a,optimize=True)
    X1B += np.einsum('ge,eiga->ia',I1,H2B['vovv'],optimize=True)\
    +np.einsum('gf,figa->ia',I2,H2B['vovv'],optimize=True)\
    +np.einsum('mo,oima->ia',I3,H2B['ooov'],optimize=True)\
    +np.einsum('no,oina->ia',I4,H2B['ooov'],optimize=True)

    I1 = np.einsum('efmn,gfmn->ge',l2b,t2b,optimize=True)
    I2 = np.einsum('fenm,fgnm->ge',l2b,t2b,optimize=True)
    I3 = -np.einsum('efmn,efon->mo',l2b,t2b,optimize=True)
    I4 = -np.einsum('fenm,feno->mo',l2b,t2b,optimize=True)
    X1B += np.einsum('ge,eiga->ia',I1,H2B['vovv'],optimize=True)\
    +np.einsum('ge,eiga->ia',I2,H2C['vovv'],optimize=True)\
    +np.einsum('mo,oima->ia',I3,H2B['ooov'],optimize=True)\
    +np.einsum('mo,oima->ia',I4,H2C['ooov'],optimize=True)
    
    if iroot == 0:
        X1B += H1B['ov']

    return X1B.transpose((1,0))

def build_LH_2A(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys):

    l1a = cc_t['l1a'][iroot]
    l1b = cc_t['l1b'][iroot]
    l2a = cc_t['l2a'][iroot]
    l2b = cc_t['l2b'][iroot]
    l2c = cc_t['l2c'][iroot]

    t1a = cc_t['t1a']
    t1b = cc_t['t1b']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']

    vA = ints['vA']

    X2A = np.zeros((sys['Nocc_a'],sys['Nocc_a'],sys['Nunocc_a'],sys['Nunocc_a']))

    X2A += np.einsum('ea,ebij->ijab',H1A['vv'],l2a,optimize=True)\
    -np.einsum('eb,eaij->ijab',H1A['vv'],l2a,optimize=True)
    X2A += -np.einsum('im,abmj->ijab',H1A['oo'],l2a,optimize=True)\
    +np.einsum('jm,abmi->ijab',H1A['oo'],l2a,optimize=True)
    X2A += np.einsum('jb,ai->ijab',H1A['ov'],l1a,optimize=True)\
    -np.einsum('ja,bi->ijab',H1A['ov'],l1a,optimize=True)\
    -np.einsum('ib,aj->ijab',H1A['ov'],l1a,optimize=True)\
    +np.einsum('ia,bj->ijab',H1A['ov'],l1a,optimize=True)
                
    I1 = np.einsum('afmn,efmn->ea',l2a,t2a,optimize=True)
    I2 = np.einsum('bfmn,efmn->eb',l2a,t2a,optimize=True)
    X2A += -0.5*np.einsum('ea,ijeb->ijab',I1,vA['oovv'],optimize=True)\
    +0.5*np.einsum('eb,ijea->ijab',I2,vA['oovv'],optimize=True)
                
    I1 = np.einsum('afmn,efmn->ea',l2b,t2b,optimize=True)
    I2 = np.einsum('bfmn,efmn->eb',l2b,t2b,optimize=True)
    X2A += -np.einsum('ea,ijeb->ijab',I1,vA['oovv'],optimize=True)\
    +np.einsum('eb,ijea->ijab',I2,vA['oovv'],optimize=True)
                
    I1 = np.einsum('efin,efmn->im',l2a,t2a,optimize=True)
    I2 = np.einsum('efjn,efmn->jm',l2a,t2a,optimize=True)
    X2A += -0.5*np.einsum('im,mjab->ijab',I1,vA['oovv'],optimize=True)\
    +0.5*np.einsum('jm,miab->ijab',I2,vA['oovv'],optimize=True)
                
    I1 = np.einsum('efin,efmn->im',l2b,t2b,optimize=True)
    I2 = np.einsum('efjn,efmn->jm',l2b,t2b,optimize=True)
    X2A += -np.einsum('im,mjab->ijab',I1,vA['oovv'],optimize=True)\
    +np.einsum('jm,miab->ijab',I2,vA['oovv'],optimize=True)

    X2A += np.einsum('eima,ebmj->ijab',H2A['voov'],l2a,optimize=True)\
    -np.einsum('ejma,ebmi->ijab',H2A['voov'],l2a,optimize=True)\
    -np.einsum('eimb,eamj->ijab',H2A['voov'],l2a,optimize=True)\
    +np.einsum('ejmb,eami->ijab',H2A['voov'],l2a,optimize=True)
                
    X2A += +np.einsum('ieam,bejm->ijab',H2B['ovvo'],l2b,optimize=True)\
    -np.einsum('jeam,beim->ijab',H2B['ovvo'],l2b,optimize=True)\
    -np.einsum('iebm,aejm->ijab',H2B['ovvo'],l2b,optimize=True)\
    +np.einsum('jebm,aeim->ijab',H2B['ovvo'],l2b,optimize=True)
                
    X2A += 0.5*np.einsum('ijmn,abmn->ijab',H2A['oooo'],l2a,optimize=True)
    X2A += +0.5*np.einsum('efab,efij->ijab',H2A['vvvv'],l2a,optimize=True)
    X2A += np.einsum('ejab,ei->ijab',H2A['vovv'],l1a,optimize=True)\
    -np.einsum('eiab,ej->ijab',H2A['vovv'],l1a,optimize=True)
    X2A += -np.einsum('ijmb,am->ijab',H2A['ooov'],l1a,optimize=True)\
    +np.einsum('ijma,bm->ijab',H2A['ooov'],l1a,optimize=True)

    if iroot == 0:
        X2A += vA['oovv']

    return X2A.transpose((2,3,0,1))

def build_LH_2B(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys):
    
    l1a = cc_t['l1a'][iroot]
    l1b = cc_t['l1b'][iroot]
    l2a = cc_t['l2a'][iroot]
    l2b = cc_t['l2b'][iroot]
    l2c = cc_t['l2c'][iroot]

    t1a = cc_t['t1a']
    t1b = cc_t['t1b']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']

    vB = ints['vB']

    X2B = np.zeros((sys['Nocc_a'],sys['Nocc_b'],sys['Nunocc_a'],sys['Nunocc_b']))
    
    X2B -= np.einsum('ijmb,am->ijab',H2B['ooov'],l1a,optimize=True)
    X2B -= np.einsum('ijam,bm->ijab',H2B['oovo'],l1b,optimize=True)
    
    X2B += np.einsum('ejab,ei->ijab',H2B['vovv'],l1a,optimize=True)
    X2B += np.einsum('ieab,ej->ijab',H2B['ovvv'],l1b,optimize=True)
     
    X2B += np.einsum('ijmn,abmn->ijab',H2B['oooo'],l2b,optimize=True)
    X2B += np.einsum('efab,efij->ijab',H2B['vvvv'],l2b,optimize=True)
    
    X2B +=  np.einsum('ejmb,aeim->ijab',H2B['voov'],l2a,optimize=True)
    X2B +=  np.einsum('eima,ebmj->ijab',H2A['voov'],l2b,optimize=True)
    X2B +=  np.einsum('ejmb,aeim->ijab',H2C['voov'],l2b,optimize=True)
    X2B +=  np.einsum('ieam,ebmj->ijab',H2B['ovvo'],l2c,optimize=True)
    X2B -=  np.einsum('iemb,aemj->ijab',H2B['ovov'],l2b,optimize=True)
    X2B -=  np.einsum('ejam,ebim->ijab',H2B['vovo'],l2b,optimize=True)

    I1 = -0.5*np.einsum('abij,fbij->fa',l2a,t2a,optimize=True)
    I2 = -np.einsum('afmn,efmn->ea',l2b,t2b,optimize=True)
    I3 = -np.einsum('fbnm,fenm->eb',l2b,t2b,optimize=True)
    I4 = -0.5*np.einsum('bfmn,efmn->eb',l2c,t2c,optimize=True)
    X2B += np.einsum('fa,nmfe->nmae',I1,vB['oovv'],optimize=True)
    X2B += np.einsum('ea,ijeb->ijab',I2,vB['oovv'],optimize=True)
    X2B += np.einsum('eb,ijae->ijab',I3,vB['oovv'],optimize=True)
    X2B += np.einsum('eb,ijae->ijab',I4,vB['oovv'],optimize=True)

    I1 = -0.5*np.einsum('efin,efmn->im',l2a,t2a,optimize=True)
    I2 = -np.einsum('efin,efmn->im',l2b,t2b,optimize=True)
    I3 = -np.einsum('fenj,fenm->jm',l2b,t2b,optimize=True)
    I4 = -0.5*np.einsum('efjn,efmn->jm',l2c,t2c,optimize=True)
    X2B += np.einsum('im,mjab->ijab',I1,vB['oovv'],optimize=True)
    X2B += np.einsum('im,mjab->ijab',I2,vB['oovv'],optimize=True)
    X2B += np.einsum('jm,imab->ijab',I3,vB['oovv'],optimize=True)
    X2B += np.einsum('jm,imab->ijab',I4,vB['oovv'],optimize=True)
    
    X2B += np.einsum('ea,ebij->ijab',H1A['vv'],l2b,optimize=True)
    X2B += np.einsum('eb,aeij->ijab',H1B['vv'],l2b,optimize=True)
    X2B -= np.einsum('im,abmj->ijab',H1A['oo'],l2b,optimize=True)
    X2B -= np.einsum('jm,abim->ijab',H1B['oo'],l2b,optimize=True)
    X2B += np.einsum('jb,ai->ijab',H1B['ov'],l1a,optimize=True)
    X2B += np.einsum('ia,bj->ijab',H1A['ov'],l1b,optimize=True)
              
    if iroot == 0:
        X2B += vB['oovv']  

    return X2B.transpose((2,3,0,1))

def build_LH_2C(cc_t,H1A,H1B,H2A,H2B,H2C,iroot,ints,sys):

    l1a = cc_t['l1a'][iroot]
    l1b = cc_t['l1b'][iroot]
    l2a = cc_t['l2a'][iroot]
    l2b = cc_t['l2b'][iroot]
    l2c = cc_t['l2c'][iroot]

    t1a = cc_t['t1a']
    t1b = cc_t['t1b']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']

    vC = ints['vC']

    X2C = np.zeros((sys['Nocc_b'],sys['Nocc_b'],sys['Nunocc_b'],sys['Nunocc_b']))
    
    X2C += np.einsum('ea,ebij->ijab',H1B['vv'],l2c,optimize=True)
    X2C -= np.einsum('eb,eaij->ijab',H1B['vv'],l2c,optimize=True)
    X2C -= np.einsum('im,abmj->ijab',H1B['oo'],l2c,optimize=True)
    X2C += np.einsum('jm,abmi->ijab',H1B['oo'],l2c,optimize=True) 
    X2C -= np.einsum('ijmb,am->ijab',H2C['ooov'],l1b,optimize=True)
    X2C += np.einsum('ijma,bm->ijab',H2C['ooov'],l1b,optimize=True)
    X2C += np.einsum('ejab,ei->ijab',H2C['vovv'],l1b,optimize=True)
    X2C -= np.einsum('eiab,ej->ijab',H2C['vovv'],l1b,optimize=True)
                
    X2C += 0.5*np.einsum('efab,efij->ijab',H2C['vvvv'],l2c,optimize=True)    
    X2C += 0.5*np.einsum('ijmn,abmn->ijab',H2C['oooo'],l2c,optimize=True)
     
    X2C += np.einsum('ejmb,aeim->ijab',H2C['voov'],l2c,optimize=True)
    X2C -= np.einsum('eimb,aejm->ijab',H2C['voov'],l2c,optimize=True)
    X2C -= np.einsum('ejma,beim->ijab',H2C['voov'],l2c,optimize=True)
    X2C += np.einsum('eima,bejm->ijab',H2C['voov'],l2c,optimize=True)
                
    X2C += np.einsum('ejmb,eami->ijab',H2B['voov'],l2b,optimize=True)
    X2C -= np.einsum('eimb,eamj->ijab',H2B['voov'],l2b,optimize=True)
    X2C -= np.einsum('ejma,ebmi->ijab',H2B['voov'],l2b,optimize=True)
    X2C += np.einsum('eima,ebmj->ijab',H2B['voov'],l2b,optimize=True)

    I1 = np.einsum('fanm,fenm->ea',l2b,t2b,optimize=True)
    I2 = np.einsum('fbnm,fenm->eb',l2b,t2b,optimize=True)
    X2C -= np.einsum('ea,ijeb->ijab',I1,vC['oovv'],optimize=True)
    X2C += np.einsum('eb,ijea->ijab',I2,vC['oovv'],optimize=True)

    I1 = np.einsum('afmn,efmn->ea',l2c,t2c,optimize=True)
    I2 = np.einsum('bfmn,efmn->eb',l2c,t2c,optimize=True)
    X2C -= 0.5*np.einsum('ea,ijeb->ijab',I1,vC['oovv'],optimize=True)
    X2C += 0.5*np.einsum('eb,ijea->ijab',I2,vC['oovv'],optimize=True)

    I1 = np.einsum('feni,fenm->im',l2b,t2b,optimize=True)
    I2 = np.einsum('fenj,fenm->jm',l2b,t2b,optimize=True)
    X2C -= np.einsum('im,mjab->ijab',I1,vC['oovv'],optimize=True)
    X2C += np.einsum('jm,miab->ijab',I2,vC['oovv'],optimize=True)

    I1 = np.einsum('efin,efmn->im',l2c,t2c,optimize=True)
    I2 = np.einsum('efjn,efmn->jm',l2c,t2c,optimize=True)
    X2C -= 0.5*np.einsum('im,mjab->ijab',I1,vC['oovv'],optimize=True)
    X2C += 0.5*np.einsum('jm,miab->ijab',I2,vC['oovv'],optimize=True)
                
    X2C += np.einsum('jb,ai->ijab',H1B['ov'],l1b,optimize=True)
    X2C -= np.einsum('ib,aj->ijab',H1B['ov'],l1b,optimize=True)
    X2C -= np.einsum('ja,bi->ijab',H1B['ov'],l1b,optimize=True)
    X2C += np.einsum('ia,bj->ijab',H1B['ov'],l1b,optimize=True)

    if iroot == 0:
        X2C += vC['oovv']

    return X2C.transpose((2,3,0,1))

def update_L(l1a,l1b,l2a,l2b,l2c,X1A,X1B,X2A,X2B,X2C,omega,H1A,H1B,sys,shift):

    for a in range(sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            denom = H1A['vv'][a,a] - H1A['oo'][i,i]
            l1a[a,i] -= (X1A[a,i]-omega*l1a[a,i])/(denom - omega + shift)


    for a in range(sys['Nunocc_b']):
        for i in range(sys['Nocc_b']):
            denom = H1B['vv'][a,a] - H1B['oo'][i,i]
            l1b[a,i] -= (X1B[a,i]-omega*l1b[a,i])/(denom - omega + shift)

    for a in range(sys['Nunocc_a']):
        for b in range(a+1,sys['Nunocc_a']):
            for i in range(sys['Nocc_a']):
                for j in range(i+1,sys['Nocc_a']):
                    denom = H1A['vv'][a,a] + H1A['vv'][b,b] - H1A['oo'][i,i] - H1A['oo'][j,j]
                    l2a[a,b,i,j] -= (X2A[a,b,i,j]-omega*l2a[a,b,i,j])/(denom - omega + shift)
                    l2a[a,b,j,i] = -l2a[a,b,i,j]
                    l2a[b,a,i,j] = -l2a[a,b,i,j]
                    l2a[b,a,j,i] =  l2a[a,b,i,j]


    for a in range(sys['Nunocc_a']):
        for b in range(sys['Nunocc_b']):
            for i in range(sys['Nocc_a']):
                for j in range(sys['Nocc_b']):
                    denom = H1A['vv'][a,a] + H1B['vv'][b,b] - H1A['oo'][i,i] - H1B['oo'][j,j]
                    l2b[a,b,i,j] -= (X2B[a,b,i,j]-omega*l2b[a,b,i,j])/(denom - omega + shift)
    

    for a in range(sys['Nunocc_b']):
        for b in range(a+1,sys['Nunocc_b']):
            for i in range(sys['Nocc_b']):
                for j in range(i+1,sys['Nocc_b']):
                    denom = H1B['vv'][a,a] + H1B['vv'][b,b] - H1B['oo'][i,i] - H1B['oo'][j,j]
                    l2c[a,b,i,j] -= (X2C[a,b,i,j]-omega*l2c[a,b,i,j])/(denom - omega + shift)
                    l2c[a,b,j,i] = -l2a[a,b,i,j]
                    l2c[b,a,i,j] = -l2a[a,b,i,j]
                    l2c[b,a,j,i] =  l2a[a,b,i,j]

    return l1a, l1b, l2a, l2b, l2c

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
    from HBar_module import HBar_CCSD

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

    l1a = data_dict['l1a']
    l1b = data_dict['l1b']
    l2a = data_dict['l2a']
    l2b = data_dict['l2b']
    l2c = data_dict['l2c']

    cc_t = {'t1a' : t1a, 't1b' : t1b, 't2a' : t2a, 't2b' : t2b, 't2c' : t2c,
            'l1a' : l1a, 'l1b' : l1b, 'l2a' : l2a, 'l2b' : l2b, 'l2c' : l2c}

    shift = 0.0

    H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,ints,sys)

    # test l1a update
    X1A = build_LH_1A(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    print('|X1A| = {}'.format(np.linalg.norm(X1A)))

    # test l1b update
    X1B = build_LH_1B(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    print('|X1B| = {}'.format(np.linalg.norm(X1B)))

    # test l2a update
    X2A = build_LH_2A(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    print('|X2A| = {}'.format(np.linalg.norm(X2A)))

    # test l2b update
    X2B = build_LH_2B(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    print('|X2B| = {}'.format(np.linalg.norm(X2B)))

    # test l2c update
    X2C = build_LH_2C(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys)
    print('|X2C| = {}'.format(np.linalg.norm(X2C)))

    return
