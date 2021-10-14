import numpy as np 
import time
from f90_crcc import crcc_loops

def crcc23(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF=False,iroot=0):

    print('\n==================================++Entering CR-CC(2,3) Routine++=============================')
    
    t_start = time.time()

    if flag_RHF:
        print('Using RHF closed-shell symmetry...')
    
    # get fock matrices
    fA = ints['fA']; fB = ints['fB']
    
    # get the 3-body HBar triples diagonal
    D3A, D3B, D3C, D3D = triples_3body_diagonal(cc_t,ints,sys)

    # get cluster ampltidues t1, t2
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']

    # correction containers
    deltaA = 0.0
    deltaB = 0.0
    deltaC = 0.0
    deltaD = 0.0

    MM23A = build_MM23A(cc_t,H1A,H2A,sys)
    L3A = build_L3A(cc_t,H1A,H2A,ints,sys,iroot=iroot)
    dA_AAA, dB_AAA, dC_AAA, dD_AAA = crcc_loops.crcc23a(MM23A,L3A,fA['oo'],fA['vv'],\
                    H1A['oo'],H1A['vv'],H2A['voov'],H2A['oooo'],H2A['vvvv'],D3A['O'],D3A['V'],\
                    sys['Nocc_a'],sys['Nunocc_a'])

    MM23B = build_MM23B(cc_t,H1A,H1B,H2A,H2B,sys)
    L3B = build_L3B(cc_t,H1A,H1B,H2A,H2B,ints,sys,iroot=iroot)
    dA_AAB, dB_AAB, dC_AAB, dD_AAB = crcc_loops.crcc23b(MM23B,L3B,fA['oo'],fA['vv'],fB['oo'],fB['vv'],\
                    H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],H2A['voov'],H2A['oooo'],H2A['vvvv'],\
                    H2B['ovov'],H2B['vovo'],H2B['oooo'],H2B['vvvv'],H2C['voov'],\
                    D3A['O'],D3A['V'],D3B['O'],D3B['V'],D3C['O'],D3C['V'],\
                    sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'])


    if flag_RHF:
        deltaA = 2.0*dA_AAA + 2.0*dA_AAB
        deltaB = 2.0*dB_AAA + 2.0*dB_AAB
        deltaC = 2.0*dC_AAA + 2.0*dC_AAB
        deltaD = 2.0*dD_AAA + 2.0*dD_AAB
    else:
        MM23C = build_MM23C(cc_t,H1A,H1B,H2B,H2C,sys)
        L3C = build_L3C(cc_t,H1A,H1B,H2B,H2C,ints,sys,iroot=iroot)  
        dA_ABB, dB_ABB, dC_ABB, dD_ABB = crcc_loops.crcc23c(MM23C,L3C,fA['oo'],fA['vv'],fB['oo'],fB['vv'],\
                    H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],H2A['voov'],\
                    H2B['ovov'],H2B['vovo'],H2B['oooo'],H2B['vvvv'],H2C['voov'],H2C['oooo'],H2C['vvvv'],\
                    D3B['O'],D3B['V'],D3C['O'],D3C['V'],D3D['O'],D3D['V'],\
                    sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'])
        
        MM23D = build_MM23D(cc_t,H1B,H2C,sys) 
        L3D = build_L3D(cc_t,H1B,H2C,ints,sys,iroot=iroot) 
        dA_BBB, dB_BBB, dC_BBB, dD_BBB = crcc_loops.crcc23d(MM23D,L3D,fB['oo'],fB['vv'],\
                    H1B['oo'],H1B['vv'],H2C['voov'],H2C['oooo'],H2C['vvvv'],D3D['O'],D3D['V'],\
                    sys['Nocc_b'],sys['Nunocc_b'])

        deltaA = dA_AAA + dA_AAB + dA_ABB + dA_BBB
        deltaB = dB_AAA + dB_AAB + dB_ABB + dB_BBB
        deltaC = dC_AAA + dC_AAB + dC_ABB + dC_BBB
        deltaD = dD_AAA + dD_AAB + dD_ABB + dD_BBB

    Ecorr = calc_cc_energy(cc_t,ints)

    EcorrA = Ecorr + deltaA; EcorrB = Ecorr + deltaB; EcorrC = Ecorr + deltaC; EcorrD = Ecorr + deltaD

    E23A = ints['Escf'] + EcorrA
    E23B = ints['Escf'] + EcorrB
    E23C = ints['Escf'] + EcorrC
    E23D = ints['Escf'] + EcorrD
 
    print('CR-CC(2,3)_A = {} Eh     Ecorr_A = {} Eh     Delta_A = {} Eh'.format(E23A,EcorrA,deltaA))
    print('CR-CC(2,3)_B = {} Eh     Ecorr_B = {} Eh     Delta_B = {} Eh'.format(E23B,EcorrB,deltaB))
    print('CR-CC(2,3)_C = {} Eh     Ecorr_C = {} Eh     Delta_C = {} Eh'.format(E23C,EcorrC,deltaC))
    print('CR-CC(2,3)_D = {} Eh     Ecorr_D = {} Eh     Delta_D = {} Eh'.format(E23D,EcorrD,deltaD))

    Ecrcc23 = {'A' : E23A, 'B' : E23B, 'C' : E23C, 'D' : E23D}
    delta23 = {'A' : deltaA, 'B' : deltaB, 'C' : deltaC, 'D' : deltaD}

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))

    return Ecrcc23, delta23

def build_MM23A(cc_t,H1A,H2A,sys):

    
    t2a = cc_t['t2a']
    I2A_vvov = H2A['vvov']+np.einsum('me,abim->abie',H1A['ov'],t2a,optimize=True)

    MM23A = np.zeros((sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_a']))
    
    # < phi_{ijkabc} | H_{CCSD} | 0 >
    # = -A(k/ij)A(a/bc) h(amij)*t(bcmk) + A(i/jk)A(c/ab)(h(abie)-h(me)*t(abim))*t(ecjk)
    MM23A -= np.einsum('amij,bcmk->abcijk',H2A['vooo'],t2a,optimize=True) # (1)
    MM23A += np.einsum('amkj,bcmi->abcijk',H2A['vooo'],t2a,optimize=True) # (ik)
    MM23A += np.einsum('amik,bcmj->abcijk',H2A['vooo'],t2a,optimize=True) # (jk)
    MM23A += np.einsum('cmij,bamk->abcijk',H2A['vooo'],t2a,optimize=True) # (ac)
    MM23A += np.einsum('bmij,acmk->abcijk',H2A['vooo'],t2a,optimize=True) # (ab)
    MM23A -= np.einsum('bmkj,acmi->abcijk',H2A['vooo'],t2a,optimize=True) # (ab)(ik)
    MM23A -= np.einsum('cmkj,bami->abcijk',H2A['vooo'],t2a,optimize=True) # (ac)(ik)
    MM23A -= np.einsum('bmik,acmj->abcijk',H2A['vooo'],t2a,optimize=True) # (ab)(jk)
    MM23A -= np.einsum('cmik,bamj->abcijk',H2A['vooo'],t2a,optimize=True)    # (ac)(jk)

    MM23A += np.einsum('abie,ecjk->abcijk',I2A_vvov,t2a,optimize=True) # (1)
    MM23A -= np.einsum('abje,ecik->abcijk',I2A_vvov,t2a,optimize=True) # (ij)
    MM23A -= np.einsum('abke,ecji->abcijk',I2A_vvov,t2a,optimize=True) # (ik)
    MM23A -= np.einsum('cbie,eajk->abcijk',I2A_vvov,t2a,optimize=True) # (ac)
    MM23A -= np.einsum('acie,ebjk->abcijk',I2A_vvov,t2a,optimize=True) # (bc)
    MM23A += np.einsum('cbje,eaik->abcijk',I2A_vvov,t2a,optimize=True) # (ac)(ij)
    MM23A += np.einsum('acje,ebik->abcijk',I2A_vvov,t2a,optimize=True) # (bc)(ij)
    MM23A += np.einsum('cbke,eaji->abcijk',I2A_vvov,t2a,optimize=True) # (ac)(ik)
    MM23A += np.einsum('acke,ebji->abcijk',I2A_vvov,t2a,optimize=True) # (bc)(ik)

    return MM23A

def build_MM23B(cc_t,H1A,H1B,H2A,H2B,sys):

    t2a = cc_t['t2a']
    t2b = cc_t['t2b']

    MM23B = np.zeros((sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_b']))
    
    I2B_ovoo = H2B['ovoo'] - np.einsum('me,ecjk->mcjk',H1A['ov'],t2b,optimize=True) 
    I2B_vooo = H2B['vooo'] - np.einsum('me,aeik->amik',H1B['ov'],t2b,optimize=True) 
    I2A_vooo = H2A['vooo'] - np.einsum('me,aeij->amij',H1A['ov'],t2a,optimize=True) 
   
    MM23B += np.einsum('bcek,aeij->abcijk',H2B['vvvo'],t2a,optimize=True)
    MM23B -= np.einsum('acek,beij->abcijk',H2B['vvvo'],t2a,optimize=True)
    MM23B -=  np.einsum('mcjk,abim->abcijk',I2B_ovoo,t2a,optimize=True)
    MM23B += np.einsum('mcik,abjm->abcijk',I2B_ovoo,t2a,optimize=True)
        
    MM23B += np.einsum('acie,bejk->abcijk',H2B['vvov'],t2b,optimize=True)
    MM23B -= np.einsum('bcie,aejk->abcijk',H2B['vvov'],t2b,optimize=True)
    MM23B -= np.einsum('acje,beik->abcijk',H2B['vvov'],t2b,optimize=True)
    MM23B += np.einsum('bcje,aeik->abcijk',H2B['vvov'],t2b,optimize=True)
    MM23B -= np.einsum('amik,bcjm->abcijk',I2B_vooo,t2b,optimize=True)
    MM23B += np.einsum('bmik,acjm->abcijk',I2B_vooo,t2b,optimize=True)
    MM23B += np.einsum('amjk,bcim->abcijk',I2B_vooo,t2b,optimize=True)
    MM23B -= np.einsum('bmjk,acim->abcijk',I2B_vooo,t2b,optimize=True)
        
    MM23B += np.einsum('abie,ecjk->abcijk',H2A['vvov'],t2b,optimize=True)
    MM23B -= np.einsum('abje,ecik->abcijk',H2A['vvov'],t2b,optimize=True)
    MM23B -= np.einsum('amij,bcmk->abcijk',I2A_vooo,t2b,optimize=True)
    MM23B += np.einsum('bmij,acmk->abcijk',I2A_vooo,t2b,optimize=True)

    return MM23B

def build_MM23C(cc_t,H1A,H1B,H2B,H2C,sys):

    
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']

    MM23C = np.zeros((sys['Nunocc_a'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b'],sys['Nocc_b']))
    
    I2B_vooo = H2B['vooo'] - np.einsum('me,aeij->amij',H1B['ov'],t2b,optimize=True)
    I2C_vooo = H2C['vooo'] - np.einsum('me,cekj->cmkj',H1B['ov'],t2c,optimize=True)
    I2B_ovoo = H2B['ovoo'] - np.einsum('me,ebij->mbij',H1A['ov'],t2b,optimize=True)
    
    MM23C += np.einsum('abie,ecjk->abcijk',H2B['vvov'],t2c,optimize=True)
    MM23C -= np.einsum('acie,ebjk->abcijk',H2B['vvov'],t2c,optimize=True)
    MM23C -= np.einsum('amij,bcmk->abcijk',I2B_vooo,t2c,optimize=True)
    MM23C += np.einsum('amik,bcmj->abcijk',I2B_vooo,t2c,optimize=True)
        
    MM23C += np.einsum('cbke,aeij->abcijk',H2C['vvov'],t2b,optimize=True)
    MM23C -= np.einsum('cbje,aeik->abcijk',H2C['vvov'],t2b,optimize=True)
    MM23C -= np.einsum('cmkj,abim->abcijk',I2C_vooo,t2b,optimize=True)
    MM23C += np.einsum('bmkj,acim->abcijk',I2C_vooo,t2b,optimize=True)
        
    MM23C += np.einsum('abej,ecik->abcijk',H2B['vvvo'],t2b,optimize=True)
    MM23C -= np.einsum('acej,ebik->abcijk',H2B['vvvo'],t2b,optimize=True)
    MM23C -= np.einsum('abek,ecij->abcijk',H2B['vvvo'],t2b,optimize=True)
    MM23C += np.einsum('acek,ebij->abcijk',H2B['vvvo'],t2b,optimize=True)
    MM23C -= np.einsum('mbij,acmk->abcijk',I2B_ovoo,t2b,optimize=True)
    MM23C += np.einsum('mcij,abmk->abcijk',I2B_ovoo,t2b,optimize=True)
    MM23C += np.einsum('mbik,acmj->abcijk',I2B_ovoo,t2b,optimize=True)
    MM23C -= np.einsum('mcik,abmj->abcijk',I2B_ovoo,t2b,optimize=True)

    return MM23C


def build_MM23D(cc_t,H1B,H2C,sys):

    
    t2c = cc_t['t2c']
    I2C_vvov = H2C['vvov']+np.einsum('me,abim->abie',H1B['ov'],t2c,optimize=True)

    MM23D = np.zeros((sys['Nunocc_b'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b'],sys['Nocc_b']))
    
    # < phi_{ijkabc} | H_{CCSD} | 0 >
    # = -A(k/ij)A(a/bc) h(amij)*t(bcmk) + A(i/jk)A(c/ab)(h(abie)-h(me)*t(abim))*t(ecjk)
    MM23D -= np.einsum('amij,bcmk->abcijk',H2C['vooo'],t2c,optimize=True) # (1)
    MM23D += np.einsum('amkj,bcmi->abcijk',H2C['vooo'],t2c,optimize=True) # (ik)
    MM23D += np.einsum('amik,bcmj->abcijk',H2C['vooo'],t2c,optimize=True) # (jk)
    MM23D += np.einsum('cmij,bamk->abcijk',H2C['vooo'],t2c,optimize=True) # (ac)
    MM23D += np.einsum('bmij,acmk->abcijk',H2C['vooo'],t2c,optimize=True) # (ab)
    MM23D -= np.einsum('bmkj,acmi->abcijk',H2C['vooo'],t2c,optimize=True) # (ab)(ik)
    MM23D -= np.einsum('cmkj,bami->abcijk',H2C['vooo'],t2c,optimize=True) # (ac)(ik)
    MM23D -= np.einsum('bmik,acmj->abcijk',H2C['vooo'],t2c,optimize=True) # (ab)(jk)
    MM23D -= np.einsum('cmik,bamj->abcijk',H2C['vooo'],t2c,optimize=True)    # (ac)(jk)

    MM23D += np.einsum('abie,ecjk->abcijk',I2C_vvov,t2c,optimize=True) # (1)
    MM23D -= np.einsum('abje,ecik->abcijk',I2C_vvov,t2c,optimize=True) # (ij)
    MM23D -= np.einsum('abke,ecji->abcijk',I2C_vvov,t2c,optimize=True) # (ik)
    MM23D -= np.einsum('cbie,eajk->abcijk',I2C_vvov,t2c,optimize=True) # (ac)
    MM23D -= np.einsum('acie,ebjk->abcijk',I2C_vvov,t2c,optimize=True) # (bc)
    MM23D += np.einsum('cbje,eaik->abcijk',I2C_vvov,t2c,optimize=True) # (ac)(ij)
    MM23D += np.einsum('acje,ebik->abcijk',I2C_vvov,t2c,optimize=True) # (bc)(ij)
    MM23D += np.einsum('cbke,eaji->abcijk',I2C_vvov,t2c,optimize=True) # (ac)(ik)
    MM23D += np.einsum('acke,ebji->abcijk',I2C_vvov,t2c,optimize=True) # (bc)(ik)

    return MM23D


def build_L3A(cc_t,H1A,H2A,ints,sys,iroot=0):

    vA = ints['vA']
    l1a = cc_t['l1a'][iroot]
    l2a = cc_t['l2a'][iroot]

    L3A = np.zeros((sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_a']))

    L3A += np.einsum('ck,ijab->abcijk',l1a,vA['oovv'],optimize=True) 
    L3A -= np.einsum('ak,ijcb->abcijk',l1a,vA['oovv'],optimize=True)
    L3A -= np.einsum('bk,ijac->abcijk',l1a,vA['oovv'],optimize=True) 
    L3A -= np.einsum('ci,kjab->abcijk',l1a,vA['oovv'],optimize=True) 
    L3A -= np.einsum('cj,ikab->abcijk',l1a,vA['oovv'],optimize=True) 
    L3A += np.einsum('ai,kjcb->abcijk',l1a,vA['oovv'],optimize=True) 
    L3A += np.einsum('bi,kjac->abcijk',l1a,vA['oovv'],optimize=True) 
    L3A += np.einsum('aj,ikcb->abcijk',l1a,vA['oovv'],optimize=True) 
    L3A += np.einsum('bj,ikac->abcijk',l1a,vA['oovv'],optimize=True)

    L3A += np.einsum('kc,abij->abcijk',H1A['ov'],l2a,optimize=True)
    L3A -= np.einsum('ka,cbij->abcijk',H1A['ov'],l2a,optimize=True) 
    L3A -= np.einsum('kb,acij->abcijk',H1A['ov'],l2a,optimize=True) 
    L3A -= np.einsum('ic,abkj->abcijk',H1A['ov'],l2a,optimize=True) 
    L3A -= np.einsum('jc,abik->abcijk',H1A['ov'],l2a,optimize=True) 
    L3A += np.einsum('ia,cbkj->abcijk',H1A['ov'],l2a,optimize=True) 
    L3A += np.einsum('ib,ackj->abcijk',H1A['ov'],l2a,optimize=True) 
    L3A += np.einsum('ja,cbik->abcijk',H1A['ov'],l2a,optimize=True)
    L3A += np.einsum('jb,acik->abcijk',H1A['ov'],l2a,optimize=True)

    L3A += np.einsum('eiba,ecjk->abcijk',H2A['vovv'],l2a,optimize=True) 
    L3A -= np.einsum('ejba,ecik->abcijk',H2A['vovv'],l2a,optimize=True) 
    L3A -= np.einsum('ekba,ecji->abcijk',H2A['vovv'],l2a,optimize=True) 
    L3A -= np.einsum('eibc,eajk->abcijk',H2A['vovv'],l2a,optimize=True) 
    L3A -= np.einsum('eica,ebjk->abcijk',H2A['vovv'],l2a,optimize=True) 
    L3A += np.einsum('ejbc,eaik->abcijk',H2A['vovv'],l2a,optimize=True) 
    L3A += np.einsum('ejca,ebik->abcijk',H2A['vovv'],l2a,optimize=True) 
    L3A += np.einsum('ekbc,eaji->abcijk',H2A['vovv'],l2a,optimize=True) 
    L3A += np.einsum('ekca,ebji->abcijk',H2A['vovv'],l2a,optimize=True)

    L3A -= np.einsum('jima,bcmk->abcijk',H2A['ooov'],l2a,optimize=True)
    L3A += np.einsum('jkma,bcmi->abcijk',H2A['ooov'],l2a,optimize=True)
    L3A += np.einsum('kima,bcmj->abcijk',H2A['ooov'],l2a,optimize=True) 
    L3A += np.einsum('jimb,acmk->abcijk',H2A['ooov'],l2a,optimize=True)
    L3A += np.einsum('jimc,bamk->abcijk',H2A['ooov'],l2a,optimize=True) 
    L3A -= np.einsum('jkmb,acmi->abcijk',H2A['ooov'],l2a,optimize=True) 
    L3A -= np.einsum('jkmc,bami->abcijk',H2A['ooov'],l2a,optimize=True) 
    L3A -= np.einsum('kimb,acmj->abcijk',H2A['ooov'],l2a,optimize=True) 
    L3A -= np.einsum('kimc,bamj->abcijk',H2A['ooov'],l2a,optimize=True)

    return L3A

def build_L3B(cc_t,H1A,H1B,H2A,H2B,ints,sys,iroot=0):

    vB = ints['vB']
    vA = ints['vA']
    l1a = cc_t['l1a'][iroot]
    l1b = cc_t['l1b'][iroot]
    l2a = cc_t['l2a'][iroot]
    l2b = cc_t['l2b'][iroot]

    L3B = np.zeros((sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_b'])) 

    L3B += np.einsum('ai,jkbc->abcijk',l1a,vB['oovv'],optimize=True)
    L3B -= np.einsum('aj,ikbc->abcijk',l1a,vB['oovv'],optimize=True)
    L3B -= np.einsum('bi,jkac->abcijk',l1a,vB['oovv'],optimize=True)
    L3B += np.einsum('bj,ikac->abcijk',l1a,vB['oovv'],optimize=True)
    L3B += np.einsum('ck,ijab->abcijk',l1b,vA['oovv'],optimize=True)

    L3B += np.einsum('bcjk,ia->abcijk',l2b,H1A['ov'],optimize=True)
    L3B -= np.einsum('bcik,ja->abcijk',l2b,H1A['ov'],optimize=True)
    L3B -= np.einsum('acjk,ib->abcijk',l2b,H1A['ov'],optimize=True)
    L3B += np.einsum('acik,jb->abcijk',l2b,H1A['ov'],optimize=True)
    L3B += np.einsum('abij,kc->abcijk',l2a,H1B['ov'],optimize=True)

    L3B += np.einsum('ekbc,aeij->abcijk',H2B['vovv'],l2a,optimize=True)
    L3B -= np.einsum('ekac,beij->abcijk',H2B['vovv'],l2a,optimize=True)
    L3B += np.einsum('eiba,ecjk->abcijk',H2A['vovv'],l2b,optimize=True)
    L3B -= np.einsum('ejba,ecik->abcijk',H2A['vovv'],l2b,optimize=True)
    L3B += np.einsum('ieac,bejk->abcijk',H2B['ovvv'],l2b,optimize=True)
    L3B -= np.einsum('jeac,beik->abcijk',H2B['ovvv'],l2b,optimize=True)
    L3B -= np.einsum('iebc,aejk->abcijk',H2B['ovvv'],l2b,optimize=True)
    L3B += np.einsum('jebc,aeik->abcijk',H2B['ovvv'],l2b,optimize=True)

    L3B -= np.einsum('jkmc,abim->abcijk',H2B['ooov'],l2a,optimize=True)
    L3B += np.einsum('ikmc,abjm->abcijk',H2B['ooov'],l2a,optimize=True)
    L3B -= np.einsum('jima,bcmk->abcijk',H2A['ooov'],l2b,optimize=True)
    L3B += np.einsum('jimb,acmk->abcijk',H2A['ooov'],l2b,optimize=True)
    L3B -= np.einsum('ikam,bcjm->abcijk',H2B['oovo'],l2b,optimize=True)
    L3B += np.einsum('jkam,bcim->abcijk',H2B['oovo'],l2b,optimize=True)
    L3B += np.einsum('ikbm,acjm->abcijk',H2B['oovo'],l2b,optimize=True)
    L3B -= np.einsum('jkbm,acim->abcijk',H2B['oovo'],l2b,optimize=True)

    return L3B

def build_L3C(cc_t,H1A,H1B,H2B,H2C,ints,sys,iroot=0):

    vB = ints['vB']
    vC = ints['vC']
    l1a = cc_t['l1a'][iroot]
    l1b = cc_t['l1b'][iroot]
    l2b = cc_t['l2b'][iroot]
    l2c = cc_t['l2c'][iroot]

    L3C = np.zeros((sys['Nunocc_a'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b'],sys['Nocc_b'])) 


    L3C += np.einsum('ck,ijab->abcijk',l1b,vB['oovv'],optimize=True)
    L3C -= np.einsum('bk,ijac->abcijk',l1b,vB['oovv'],optimize=True)
    L3C -= np.einsum('cj,ikab->abcijk',l1b,vB['oovv'],optimize=True)
    L3C += np.einsum('bj,ikac->abcijk',l1b,vB['oovv'],optimize=True)
    L3C += np.einsum('ai,jkbc->abcijk',l1a,vC['oovv'],optimize=True)
    
    L3C += np.einsum('kc,abij->abcijk',H1B['ov'],l2b,optimize=True)
    L3C -= np.einsum('kb,acij->abcijk',H1B['ov'],l2b,optimize=True)
    L3C -= np.einsum('jc,abik->abcijk',H1B['ov'],l2b,optimize=True)
    L3C += np.einsum('jb,acik->abcijk',H1B['ov'],l2b,optimize=True)
    L3C += np.einsum('ia,bcjk->abcijk',H1A['ov'],l2c,optimize=True)

    L3C += np.einsum('ieab,ecjk->abcijk',H2B['ovvv'],l2c,optimize=True)
    L3C -= np.einsum('ieac,ebjk->abcijk',H2B['ovvv'],l2c,optimize=True)
    L3C += np.einsum('ekbc,aeij->abcijk',H2C['vovv'],l2b,optimize=True)
    L3C -= np.einsum('ejbc,aeik->abcijk',H2C['vovv'],l2b,optimize=True)
    L3C += np.einsum('ejab,ecik->abcijk',H2B['vovv'],l2b,optimize=True)
    L3C -= np.einsum('ekab,ecij->abcijk',H2B['vovv'],l2b,optimize=True)
    L3C -= np.einsum('ejac,ebik->abcijk',H2B['vovv'],l2b,optimize=True)
    L3C += np.einsum('ekac,ebij->abcijk',H2B['vovv'],l2b,optimize=True)

    L3C -= np.einsum('ijam,bcmk->abcijk',H2B['oovo'],l2c,optimize=True)
    L3C += np.einsum('ikam,bcmj->abcijk',H2B['oovo'],l2c,optimize=True)
    L3C -= np.einsum('jkmc,abim->abcijk',H2C['ooov'],l2b,optimize=True)
    L3C += np.einsum('jkmb,acim->abcijk',H2C['ooov'],l2b,optimize=True)
    L3C -= np.einsum('ijmb,acmk->abcijk',H2B['ooov'],l2b,optimize=True)
    L3C += np.einsum('ikmb,acmj->abcijk',H2B['ooov'],l2b,optimize=True)
    L3C += np.einsum('ijmc,abmk->abcijk',H2B['ooov'],l2b,optimize=True)
    L3C -= np.einsum('ikmc,abmj->abcijk',H2B['ooov'],l2b,optimize=True)

    return L3C

def build_L3D(cc_t,H1B,H2C,ints,sys,iroot=0):

    vC = ints['vC']
    l1b = cc_t['l1b'][iroot]
    l2c = cc_t['l2c'][iroot]

    L3D = np.zeros((sys['Nunocc_b'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b'],sys['Nocc_b']))

    L3D += np.einsum('ck,ijab->abcijk',l1b,vC['oovv'],optimize=True) 
    L3D -= np.einsum('ak,ijcb->abcijk',l1b,vC['oovv'],optimize=True)
    L3D -= np.einsum('bk,ijac->abcijk',l1b,vC['oovv'],optimize=True) 
    L3D -= np.einsum('ci,kjab->abcijk',l1b,vC['oovv'],optimize=True) 
    L3D -= np.einsum('cj,ikab->abcijk',l1b,vC['oovv'],optimize=True) 
    L3D += np.einsum('ai,kjcb->abcijk',l1b,vC['oovv'],optimize=True) 
    L3D += np.einsum('bi,kjac->abcijk',l1b,vC['oovv'],optimize=True) 
    L3D += np.einsum('aj,ikcb->abcijk',l1b,vC['oovv'],optimize=True) 
    L3D += np.einsum('bj,ikac->abcijk',l1b,vC['oovv'],optimize=True)

    L3D += np.einsum('kc,abij->abcijk',H1B['ov'],l2c,optimize=True)
    L3D -= np.einsum('ka,cbij->abcijk',H1B['ov'],l2c,optimize=True) 
    L3D -= np.einsum('kb,acij->abcijk',H1B['ov'],l2c,optimize=True) 
    L3D -= np.einsum('ic,abkj->abcijk',H1B['ov'],l2c,optimize=True) 
    L3D -= np.einsum('jc,abik->abcijk',H1B['ov'],l2c,optimize=True) 
    L3D += np.einsum('ia,cbkj->abcijk',H1B['ov'],l2c,optimize=True) 
    L3D += np.einsum('ib,ackj->abcijk',H1B['ov'],l2c,optimize=True) 
    L3D += np.einsum('ja,cbik->abcijk',H1B['ov'],l2c,optimize=True)
    L3D += np.einsum('jb,acik->abcijk',H1B['ov'],l2c,optimize=True)

    L3D += np.einsum('eiba,ecjk->abcijk',H2C['vovv'],l2c,optimize=True) 
    L3D -= np.einsum('ejba,ecik->abcijk',H2C['vovv'],l2c,optimize=True) 
    L3D -= np.einsum('ekba,ecji->abcijk',H2C['vovv'],l2c,optimize=True) 
    L3D -= np.einsum('eibc,eajk->abcijk',H2C['vovv'],l2c,optimize=True) 
    L3D -= np.einsum('eica,ebjk->abcijk',H2C['vovv'],l2c,optimize=True) 
    L3D += np.einsum('ejbc,eaik->abcijk',H2C['vovv'],l2c,optimize=True) 
    L3D += np.einsum('ejca,ebik->abcijk',H2C['vovv'],l2c,optimize=True) 
    L3D += np.einsum('ekbc,eaji->abcijk',H2C['vovv'],l2c,optimize=True) 
    L3D += np.einsum('ekca,ebji->abcijk',H2C['vovv'],l2c,optimize=True)

    L3D -= np.einsum('jima,bcmk->abcijk',H2C['ooov'],l2c,optimize=True)
    L3D += np.einsum('jkma,bcmi->abcijk',H2C['ooov'],l2c,optimize=True)
    L3D += np.einsum('kima,bcmj->abcijk',H2C['ooov'],l2c,optimize=True) 
    L3D += np.einsum('jimb,acmk->abcijk',H2C['ooov'],l2c,optimize=True)
    L3D += np.einsum('jimc,bamk->abcijk',H2C['ooov'],l2c,optimize=True) 
    L3D -= np.einsum('jkmb,acmi->abcijk',H2C['ooov'],l2c,optimize=True) 
    L3D -= np.einsum('jkmc,bami->abcijk',H2C['ooov'],l2c,optimize=True) 
    L3D -= np.einsum('kimb,acmj->abcijk',H2C['ooov'],l2c,optimize=True) 
    L3D -= np.einsum('kimc,bamj->abcijk',H2C['ooov'],l2c,optimize=True)

    return L3D

def triples_3body_diagonal(cc_t,ints,sys):

    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']

    d3A_V = lambda a,i,b: -np.dot(vA['oovv'][i,:,a,b].T,t2a[a,b,i,:])
    d3A_O = lambda a,i,j:  np.dot(vA['oovv'][i,j,a,:].T,t2a[a,:,i,j])
    
    d3B_V = lambda a,i,c: -np.dot(vB['oovv'][i,:,a,c].T,t2b[a,c,i,:])
    d3B_O = lambda a,i,k:  np.dot(vB['oovv'][i,k,a,:].T,t2b[a,:,i,k])
    
    d3C_V = lambda a,k,c: -np.dot(vB['oovv'][:,k,a,c].T,t2b[a,c,:,k])
    d3C_O = lambda c,i,k:  np.dot(vB['oovv'][i,k,:,c].T,t2b[:,c,i,k])
    
    d3D_V = lambda a,i,b: -np.dot(vC['oovv'][i,:,a,b].T,t2c[a,b,i,:])
    d3D_O = lambda a,i,j:  np.dot(vC['oovv'][i,j,a,:].T,t2c[a,:,i,j])
    
    D3A_V = np.zeros((sys['Nunocc_a'],sys['Nocc_a'],sys['Nunocc_a']))
    D3A_O = np.zeros((sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_a']))
    D3B_V = np.zeros((sys['Nunocc_a'],sys['Nocc_a'],sys['Nunocc_b']))
    D3B_O = np.zeros((sys['Nunocc_a'],sys['Nocc_a'],sys['Nocc_b']))
    D3C_V = np.zeros((sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b']))
    D3C_O = np.zeros((sys['Nunocc_b'],sys['Nocc_a'],sys['Nocc_b']))
    D3D_V = np.zeros((sys['Nunocc_b'],sys['Nocc_b'],sys['Nunocc_b']))
    D3D_O = np.zeros((sys['Nunocc_b'],sys['Nocc_b'],sys['Nocc_b']))

    # A diagonal
    for a in range(sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            for b in range(sys['Nunocc_a']):
                D3A_V[a,i,b] = d3A_V(a,i,b)
    for a in range(sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            for j in range(sys['Nocc_a']):
                D3A_O[a,i,j] = d3A_O(a,i,j)
    
    # B diagonal
    for a in range(sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            for c in range(sys['Nunocc_b']):
                D3B_V[a,i,c] = d3B_V(a,i,c)
    for a in range(sys['Nunocc_a']):
        for i in range(sys['Nocc_a']):
            for k in range(sys['Nocc_b']):
                D3B_O[a,i,k] = d3B_O(a,i,k)
    
   # C diagonal 
    for a in range(sys['Nunocc_a']):
        for k in range(sys['Nocc_b']):
            for c in range(sys['Nunocc_b']):
                D3C_V[a,k,c] = d3C_V(a,k,c)
    for c in range(sys['Nunocc_b']):
        for i in range(sys['Nocc_a']):
            for k in range(sys['Nocc_b']):
                D3C_O[c,i,k] = d3C_O(c,i,k)
    
    # D diagonal 
    for a in range(sys['Nunocc_b']):
        for i in range(sys['Nocc_b']):
            for b in range(sys['Nunocc_b']):
                D3D_V[a,i,b] = d3D_V(a,i,b)
    for a in range(sys['Nunocc_b']):
        for i in range(sys['Nocc_b']):
            for j in range(sys['Nocc_b']):
                D3D_O[a,i,j] = d3D_O(a,i,j)

    D3A = {'O' : D3A_O, 'V' : D3A_V}
    D3B = {'O' : D3B_O, 'V' : D3B_V}
    D3C = {'O' : D3C_O, 'V' : D3C_V}
    D3D = {'O' : D3D_O, 'V' : D3D_V}
    
    return D3A, D3B, D3C, D3D

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

    H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,ints,sys)

    # test MM23A update
    MM23A = build_MM23A(cc_t,H1A,H2A,sys)
    print('|MM23A| = {}'.format(np.linalg.norm(MM23A)))

    # test MM23B update
    MM23B = build_MM23B(cc_t,H1A,H1B,H2A,H2B,sys)
    print('|MM23B| = {}'.format(np.linalg.norm(MM23B)))

    # test MM23C update
    MM23C = build_MM23C(cc_t,H1A,H1B,H2B,H2C,sys)
    print('|MM23C| = {}'.format(np.linalg.norm(MM23C)))

    # test MM23D update
    MM23D = build_MM23D(cc_t,H1B,H2C,sys)
    print('|MM23D| = {}'.format(np.linalg.norm(MM23D)))

    # test L3A update
    L3A = build_L3A(cc_t,H1A,H2A,ints,sys)
    print('|L3A| = {}'.format(np.linalg.norm(L3A)))

    # test L3B update
    L3B = build_L3B(cc_t,H1A,H1B,H2A,H2B,ints,sys)
    print('|L3B| = {}'.format(np.linalg.norm(L3B)))

    # test L3C update
    L3C = build_L3C(cc_t,H1A,H1B,H2B,H2C,ints,sys)
    print('|L3C| = {}'.format(np.linalg.norm(L3C)))

    # test L3D update
    L3D = build_L3D(cc_t,H1B,H2C,ints,sys)
    print('|L3D| = {}'.format(np.linalg.norm(L3D)))

    # test 3-body diagonal
    D3A,D3B,D3C,D3D = triples_3body_diagonal(cc_t,ints,sys)
    print('|D3A_O| = {}'.format(np.linalg.norm(D3A['O'])))
    print('|D3A_V| = {}'.format(np.linalg.norm(D3A['V'])))
    print('|D3B_O| = {}'.format(np.linalg.norm(D3B['O'])))
    print('|D3B_V| = {}'.format(np.linalg.norm(D3B['V'])))
    print('|D3C_O| = {}'.format(np.linalg.norm(D3C['O'])))
    print('|D3C_V| = {}'.format(np.linalg.norm(D3C['V'])))
    print('|D3D_O| = {}'.format(np.linalg.norm(D3D['O'])))
    print('|D3D_V| = {}'.format(np.linalg.norm(D3D['V'])))



    return
