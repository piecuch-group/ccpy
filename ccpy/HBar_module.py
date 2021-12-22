import numpy as np
import time

def get_ccs_intermediates_v2(cc_t,ints,sys):
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

    h1A_vv = 0.0
    h1A_vv += ints['fA']['vv']
    h1A_vv -= np.einsum('mb,am->ab',h1A_ov,cc_t['t1a'],optimize=True)
    h1A_vv += np.einsum('ambe,em->ab',ints['vA']['vovv'],cc_t['t1a'],optimize=True)
    h1A_vv += np.einsum('ambe,em->ab',ints['vB']['vovv'],cc_t['t1b'],optimize=True)

    h1B_ov = 0.0
    h1B_ov += ints['fB']['ov']
    h1B_ov += np.einsum('imae,em->ia',ints['vC']['oovv'],cc_t['t1b'],optimize=True)
    h1B_ov += np.einsum('miea,em->ia',ints['vB']['oovv'],cc_t['t1a'],optimize=True)

    h1B_oo = 0.0
    h1B_oo += ints['fB']['oo']
    h1B_oo += np.einsum('je,ei->ji',h1B_ov,cc_t['t1b'],optimize=True)
    h1B_oo += np.einsum('jmie,em->ji',ints['vC']['ooov'],cc_t['t1b'],optimize=True)
    h1B_oo += np.einsum('mjei,em->ji',ints['vB']['oovo'],cc_t['t1a'],optimize=True)

    h1B_vv = 0.0
    h1B_vv += ints['fB']['vv']
    h1B_vv -= np.einsum('mb,am->ab',h1B_ov,cc_t['t1b'],optimize=True)
    h1B_vv += np.einsum('ambe,em->ab',ints['vC']['vovv'],cc_t['t1b'],optimize=True)
    h1B_vv += np.einsum('maeb,em->ab',ints['vB']['ovvv'],cc_t['t1a'],optimize=True)

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
    Q1 -= np.transpose(Q1,(1,0,2,3))
    h2A_vvvv = 0.0
    h2A_vvvv += ints['vA']['vvvv']
    h2A_vvvv += Q1

    h2B_vvvv = 0.0
    h2B_vvvv += ints['vB']['vvvv']
    h2B_vvvv -= np.einsum('mbef,am->abef',I2B_ovvv,cc_t['t1a'],optimize=True)
    h2B_vvvv -= np.einsum('amef,bm->abef',I2B_vovv,cc_t['t1b'],optimize=True)

    Q1 = -np.einsum('bmfe,am->abef',I2C_vovv,cc_t['t1b'],optimize=True)
    Q1 -= np.transpose(Q1,(1,0,2,3))
    h2C_vvvv = 0.0
    h2C_vvvv += ints['vC']['vvvv']
    h2C_vvvv += Q1

    Q1 = +np.einsum('nmje,ei->mnij',I2A_ooov,cc_t['t1a'],optimize=True)
    Q1 -= np.transpose(Q1,(0,1,3,2))
    h2A_oooo = 0.0
    h2A_oooo += ints['vA']['oooo']
    h2A_oooo += Q1

    h2B_oooo = 0.0
    h2B_oooo += ints['vB']['oooo']
    h2B_oooo += np.einsum('mnej,ei->mnij',I2B_oovo,cc_t['t1a'],optimize=True)
    h2B_oooo += np.einsum('mnie,ej->mnij',I2B_ooov,cc_t['t1b'],optimize=True)

    Q1 = +np.einsum('nmje,ei->mnij',I2C_ooov,cc_t['t1b'],optimize=True)
    Q1 -= np.transpose(Q1,(0,1,3,2))
    h2C_oooo = 0.0
    h2C_oooo += ints['vC']['oooo']
    h2C_oooo += Q1

    h2A_voov = 0.0
    h2A_voov += ints['vA']['voov']
    h2A_voov += np.einsum('amfe,fi->amie',I2A_vovv,cc_t['t1a'],optimize=True)
    h2A_voov -= np.einsum('nmie,an->amie',I2A_ooov,cc_t['t1a'],optimize=True)

    h2B_voov = 0.0
    h2B_voov += ints['vB']['voov']
    h2B_voov += np.einsum('amfe,fi->amie',I2B_vovv,cc_t['t1a'],optimize=True)
    h2B_voov -= np.einsum('nmie,an->amie',I2B_ooov,cc_t['t1a'],optimize=True)

    h2B_ovvo = 0.0
    h2B_ovvo += ints['vB']['ovvo']
    h2B_ovvo += np.einsum('maef,fi->maei',I2B_ovvv,cc_t['t1b'],optimize=True)
    h2B_ovvo -= np.einsum('mnei,an->maei',I2B_oovo,cc_t['t1b'],optimize=True)

    h2B_ovov = 0.0
    h2B_ovov += ints['vB']['ovov']
    h2B_ovov += np.einsum('mafe,fi->maie',I2B_ovvv,cc_t['t1a'],optimize=True)
    h2B_ovov -= np.einsum('mnie,an->maie',I2B_ooov,cc_t['t1b'],optimize=True)

    h2B_vovo = 0.0
    h2B_vovo += ints['vB']['vovo']
    h2B_vovo -= np.einsum('nmei,an->amei',I2B_oovo,cc_t['t1a'],optimize=True)
    h2B_vovo += np.einsum('amef,fi->amei',I2B_vovv,cc_t['t1b'],optimize=True)

    h2C_voov = 0.0
    h2C_voov += ints['vC']['voov']
    h2C_voov += np.einsum('amfe,fi->amie',I2C_vovv,cc_t['t1b'],optimize=True)
    h2C_voov -= np.einsum('nmie,an->amie',I2C_ooov,cc_t['t1b'],optimize=True)


    Q2 = ints['vA']['voov'] + 0.5*np.einsum('amef,ei->amif',ints['vA']['vovv'],cc_t['t1a'],optimize=True)
    Q2 = np.einsum('amif,fj->amij',Q2,cc_t['t1a'],optimize=True)
    Q2 -= np.transpose(Q2,(0,1,3,2))
    h2A_vooo = 0.0
    h2A_vooo += ints['vA']['vooo']
    h2A_vooo -= 0.5*np.einsum('nmij,an->amij',h2A_oooo,cc_t['t1a'],optimize=True)
    h2A_vooo += Q2

    Q1 = ints['vB']['voov']+np.einsum('amfe,fi->amie',ints['vB']['vovv'],cc_t['t1a'],optimize=True)
    h2B_vooo = 0.0
    h2B_vooo += ints['vB']['vooo']
    h2B_vooo -= np.einsum('nmij,an->amij',h2B_oooo,cc_t['t1a'],optimize=True)
    h2B_vooo += np.einsum('amej,ei->amij',ints['vB']['vovo'],cc_t['t1a'],optimize=True)
    h2B_vooo += np.einsum('amie,ej->amij',Q1,cc_t['t1b'],optimize=True)

    Q1 = ints['vB']['ovov']+np.einsum('mafe,fj->maje',ints['vB']['ovvv'],cc_t['t1a'],optimize=True)
    h2B_ovoo = 0.0
    h2B_ovoo += ints['vB']['ovoo']
    h2B_ovoo -= np.einsum('mnji,an->maji',h2B_oooo,cc_t['t1b'],optimize=True)
    h2B_ovoo += np.einsum('maje,ei->maji',Q1,cc_t['t1b'],optimize=True)
    h2B_ovoo += np.einsum('maei,ej->maji',ints['vB']['ovvo'],cc_t['t1a'],optimize=True)

    Q2 = ints['vC']['voov'] + 0.5*np.einsum('amef,ei->amif',ints['vC']['vovv'],cc_t['t1b'],optimize=True)
    Q2 = np.einsum('amif,fj->amij',Q2,cc_t['t1b'],optimize=True)
    Q2 -= np.transpose(Q2,(0,1,3,2))
    h2C_vooo = 0.0
    h2C_vooo += ints['vC']['vooo']
    h2C_vooo -= np.einsum('nmij,an->amij',h2C_oooo,cc_t['t1b'],optimize=True)
    h2C_vooo += Q2

    Q2 = ints['vA']['ovov'] - 0.5*np.einsum('mnie,bn->mbie',ints['vA']['ooov'],cc_t['t1a'],optimize=True)
    Q2 = -np.einsum('mbie,am->abie',Q2,cc_t['t1a'],optimize=True)
    Q2 -= np.transpose(Q2,(1,0,2,3))
    h2A_vvov = 0.0
    h2A_vvov += ints['vA']['vvov']
    h2A_vvov += np.einsum('abfe,fi->abie',h2A_vvvv,cc_t['t1a'],optimize=True)
    h2A_vvov += Q2

    Q1 = ints['vB']['ovov'] - np.einsum('mnie,bn->mbie',ints['vB']['ooov'],cc_t['t1b'],optimize=True)
    Q1 = -np.einsum('mbie,am->abie',Q1,cc_t['t1a'],optimize=True)
    h2B_vvov = 0.0
    h2B_vvov += ints['vB']['vvov']
    h2B_vvov += np.einsum('abfe,fi->abie',h2B_vvvv,cc_t['t1a'],optimize=True)
    h2B_vvov += Q1
    h2B_vvov -= np.einsum('amie,bm->abie',ints['vB']['voov'],cc_t['t1b'],optimize=True)

    Q1 = ints['vB']['vovo'] - np.einsum('nmei,bn->bmei',ints['vB']['oovo'],cc_t['t1a'],optimize=True)
    Q1 = -np.einsum('bmei,am->baei',Q1,cc_t['t1b'],optimize=True)
    h2B_vvvo = 0.0
    h2B_vvvo += ints['vB']['vvvo']
    h2B_vvvo += np.einsum('baef,fi->baei',h2B_vvvv,cc_t['t1b'],optimize=True)
    h2B_vvvo += Q1
    h2B_vvvo -= np.einsum('naei,bn->baei',ints['vB']['ovvo'],cc_t['t1a'],optimize=True)

    Q2 = ints['vC']['ovov'] - 0.5*np.einsum('mnie,bn->mbie',ints['vC']['ooov'],cc_t['t1b'],optimize=True)
    Q2 = -np.einsum('mbie,am->abie',Q2,cc_t['t1b'],optimize=True)
    Q2 -= np.transpose(Q2,(1,0,2,3))
    h2C_vvov = 0.0
    h2C_vvov += ints['vC']['vvov']
    h2C_vvov += np.einsum('abfe,fi->abie',h2C_vvvv,cc_t['t1b'],optimize=True)
    h2C_vvov += Q2

    H1A = {'ov' : h1A_ov, 'oo' : h1A_oo, 'vv' : h1A_vv}

    H1B = {'ov' : h1B_ov, 'oo' : h1B_oo, 'vv' : h1B_vv}

    H2A = {'vovv' : h2A_vovv, 'ooov' : h2A_ooov, 'vvvv' : h2A_vvvv, 'oooo' : h2A_oooo, 'voov' : h2A_voov, 'vooo' : h2A_vooo, 'vvov' : h2A_vvov}

    H2B = {'vovv' : h2B_vovv, 'ooov' : h2B_ooov, 'ovvv' : h2B_ovvv, 'oovo' : h2B_oovo, 'vvvv' : h2B_vvvv, 'oooo' : h2B_oooo, 'voov' : h2B_voov,
    'ovvo' : h2B_ovvo, 'ovov' : h2B_ovov, 'vovo' : h2B_vovo, 'vooo' : h2B_vooo, 'ovoo' : h2B_ovoo, 'vvov' : h2B_vvov, 'vvvo' : h2B_vvvo}

    H2C = {'vovv' : h2C_vovv, 'ooov' : h2C_ooov, 'vvvv' : h2C_vvvv, 'oooo' : h2C_oooo, 'voov' : h2C_voov, 'vooo' : h2C_vooo, 'vvov' : h2C_vvov}

    return H1A,H1B,H2A,H2B,H2C

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

    h2C_vooo = 0.0
    h2C_vooo += ints['vC']['vooo']
    h2C_vooo -= 0.5*np.einsum('mnij,bn->bmji',ints['vC']['oooo'],cc_t['t1b'],optimize=True)
    h2C_vooo += np.einsum('mbef,ei,fj->bmji',ints['vC']['ovvv'],cc_t['t1b'],cc_t['t1b'],optimize=True)
    h2C_vooo -= 0.5*np.einsum('mnef,fj,ei,bn->bmji',ints['vC']['oovv'],cc_t['t1b'],cc_t['t1b'],cc_t['t1b'],optimize=True)
    h2C_vooo += np.einsum('mbif,fj->bmji',ints['vC']['ovov'],cc_t['t1b'],optimize=True)
    h2C_vooo -= np.einsum('mbjf,fi->bmji',ints['vC']['ovov'],cc_t['t1b'],optimize=True)

    h2C_vvov = 0.0
    h2C_vvov += ints['vC']['vvov']
    h2C_vvov += 0.5*np.einsum('abef,fj->baje',ints['vC']['vvvv'],cc_t['t1b'],optimize=True)
    h2C_vvov += np.einsum('mnej,am,bn->baje',ints['vC']['oovo'],cc_t['t1b'],cc_t['t1b'],optimize=True)

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
    H2C = {'oooo' : h2C_oooo, 'vvvv' : h2C_vvvv, 'vvov' : h2C_vvov, 'vooo' : h2C_vooo, 'voov' : h2C_voov,
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
    Q1 -= np.transpose(Q1,(1,0,2,3))
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
    Q1 -= np.transpose(Q1,(1,0,2,3))
    h2C_vvvv = 0.0
    h2C_vvvv += ints['vC']['vvvv']
    h2C_vvvv += 0.5*np.einsum('mnef,abmn->abef',ints['vC']['oovv'],cc_t['t2c'],optimize=True)
    h2C_vvvv += Q1

    Q1 = +np.einsum('nmje,ei->mnij',I2A_ooov,cc_t['t1a'],optimize=True)
    Q1 -= np.transpose(Q1,(0,1,3,2))
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
    Q1 -= np.transpose(Q1,(0,1,3,2))
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
    Q1 -= np.transpose(Q1,(0,1,3,2))
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
    Q1 -= np.transpose(Q1,(0,1,3,2))
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
    Q1 -= np.transpose(Q1,(1,0,2,3))
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
    Q1 -= np.transpose(Q1,(1,0,2,3))
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

def HBar_CCSD(cc_t,ints,sys):
    """Calculate the CCSD similarity-transformed HBar integrals (H_N e^(T1+T2))_C.

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
    print('\nCCSD HBar construction...',end='')

    t_start = time.time()

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

    Q1 = -np.einsum('mnef,am->anef',vB['oovv'],t1a,optimize=True)
    I2B_vovv = vB['vovv'] + 0.5*Q1
    h2B_vovv = vB['vovv'] + Q1

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

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print(' completed in {:0.2f}m  {:0.2f}s'.format(minutes,seconds))

    H1A = {'ov' : h1A_ov, 'oo' : h1A_oo, 'vv' : h1A_vv}

    H1B = {'ov' : h1B_ov, 'oo' : h1B_oo, 'vv' : h1B_vv}

    H2A = {'vovv' : h2A_vovv, 'ooov' : h2A_ooov, 'vvvv' : h2A_vvvv, 'oooo' : h2A_oooo, 'voov' : h2A_voov, 'vooo' : h2A_vooo, 'vvov' : h2A_vvov,\
           'oovv' : vA['oovv']}

    H2B = {'vovv' : h2B_vovv, 'ooov' : h2B_ooov, 'ovvv' : h2B_ovvv, 'oovo' : h2B_oovo, 'vvvv' : h2B_vvvv, 'oooo' : h2B_oooo, 'voov' : h2B_voov,
    'ovvo' : h2B_ovvo, 'ovov' : h2B_ovov, 'vovo' : h2B_vovo, 'vooo' : h2B_vooo, 'ovoo' : h2B_ovoo, 'vvov' : h2B_vvov, 'vvvo' : h2B_vvvo,\
    'oovv' : vB['oovv']}

    H2C = {'vovv' : h2C_vovv, 'ooov' : h2C_ooov, 'vvvv' : h2C_vvvv, 'oooo' : h2C_oooo, 'voov' : h2C_voov, 'vooo' : h2C_vooo, 'vvov' : h2C_vvov,\
           'oovv' : vC['oovv']}

    return H1A,H1B,H2A,H2B,H2C

def HBar_CCSDT(cc_t,ints,sys):
    """Calculate the CCSDT similarity-transformed HBar integrals (H_N e^(T1+T2+T3))_C.

    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2, and T3
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
    print('\nCCSDT HBar construction...',end='')

    t_start = time.time()

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

    Q1 = -np.einsum('mnef,am->anef',vB['oovv'],t1a,optimize=True)
    I2B_vovv = vB['vovv'] + 0.5*Q1
    h2B_vovv = vB['vovv'] + Q1

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
    h2A_vooo += 0.5*np.einsum('mnef,aefijn->amij',vA['oovv'],t3a,optimize=True)
    h2A_vooo += np.einsum('mnef,aefijn->amij',vB['oovv'],t3b,optimize=True)

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
    h2B_vooo += np.einsum('nmfe,afeinj->amij',vB['oovv'],t3b,optimize=True)
    h2B_vooo += 0.5*np.einsum('mnef,aefijn->amij',vC['oovv'],t3c,optimize=True)

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
    h2B_ovoo += 0.5*np.einsum('mnef,efajni->maji',vA['oovv'],t3b,optimize=True)
    h2B_ovoo += np.einsum('mnef,efajni->maji',vB['oovv'],t3c,optimize=True)

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
    h2C_vooo += 0.5*np.einsum('mnef,aefijn->amij',vC['oovv'],t3d,optimize=True)
    h2C_vooo += np.einsum('nmfe,faenij->amij',vB['oovv'],t3c,optimize=True)

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
    h2A_vvov -= 0.5*np.einsum('mnef,abfimn->abie',vA['oovv'],t3a,optimize=True)
    h2A_vvov -= np.einsum('mnef,abfimn->abie',vB['oovv'],t3b,optimize=True)

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
    h2B_vvov -= np.einsum('nmfe,afbinm->abie',vB['oovv'],t3b,optimize=True)
    h2B_vvov -= 0.5*np.einsum('mnef,afbinm->abie',vC['oovv'],t3c,optimize=True)

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
    h2B_vvvo -= 0.5*np.einsum('mnef,bfamni->baei',vA['oovv'],t3b,optimize=True)
    h2B_vvvo -= np.einsum('mnef,bfamni->baei',vB['oovv'],t3c,optimize=True)

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
    h2C_vvov -= 0.5*np.einsum('mnef,abfimn->abie',vC['oovv'],t3d,optimize=True)
    h2C_vvov -= np.einsum('nmfe,fabnim->abie',vB['oovv'],t3c,optimize=True)

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print(' completed in {:0.2f}m  {:0.2f}s'.format(minutes,seconds))

    H1A = {'ov' : h1A_ov, 'oo' : h1A_oo, 'vv' : h1A_vv}

    H1B = {'ov' : h1B_ov, 'oo' : h1B_oo, 'vv' : h1B_vv}

    H2A = {'vovv' : h2A_vovv, 'ooov' : h2A_ooov, 'vvvv' : h2A_vvvv, 'oooo' : h2A_oooo, 'voov' : h2A_voov, 'vooo' : h2A_vooo, 'vvov' : h2A_vvov,\
           'oovv' : vA['oovv']}

    H2B = {'vovv' : h2B_vovv, 'ooov' : h2B_ooov, 'ovvv' : h2B_ovvv, 'oovo' : h2B_oovo, 'vvvv' : h2B_vvvv, 'oooo' : h2B_oooo, 'voov' : h2B_voov,
    'ovvo' : h2B_ovvo, 'ovov' : h2B_ovov, 'vovo' : h2B_vovo, 'vooo' : h2B_vooo, 'ovoo' : h2B_ovoo, 'vvov' : h2B_vvov, 'vvvo' : h2B_vvvo,\
    'oovv' : vB['oovv']}

    H2C = {'vovv' : h2C_vovv, 'ooov' : h2C_ooov, 'vvvv' : h2C_vvvv, 'oooo' : h2C_oooo, 'voov' : h2C_voov, 'vooo' : h2C_vooo, 'vvov' : h2C_vvov,\
           'oovv' : vC['oovv']}

    return H1A,H1B,H2A,H2B,H2C

def test_HBar(matfile,ints,sys):
    """Test the HBar integrals using known results from Matlab code.

    Parameters
    ----------
    matfile : str
        Path to .mat file containing T1, T2 amplitudes from Matlab
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary

    Returns
    -------
    None
    """
    from scipy.io import loadmat
    from cc_energy import calc_cc_energy

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

    H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,ints,sys)

    # test CCSD HBar components
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