import numpy as np 
import time 
import crcc_loops

def crcc24(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF=True,iroot=0):

    print('\n==================================++Entering CR-CC(2,4) Routine++=============================')
    
    t_start = time.time()

    if flag_RHF:
        print('Using closed-shell RHF symmetry')
    
    # get fock matrices
    fA = ints['fA']; fB = ints['fB']
    
    # get the 3-body HBar triples diagonal
    D3A, D3B, D3C, D3D = triples_3body_diagonal(cc_t,ints,sys)
        
    # MM24A correction
    MM24A = build_MM24A(cc_t,H2A,sys)
    L4A = build_L4A(cc_t,ints,sys,iroot=iroot)
    dA_AAAA, dB_AAAA, dC_AAAA, dD_AAAA = crcc_loops.crcc_loops.crcc24a(MM24A,L4A,fA['oo'],fA['vv'],H1A['oo'],H1A['vv'],\
                    H2A['voov'],H2A['oooo'],H2A['vvvv'],D3A['O'],D3A['V'])
    
    # MM24B correction
    MM24B = build_MM24B(cc_t,H2A,H2B,sys)
    L4B = build_L4B(cc_t,ints,sys,iroot=iroot)
    dA_AAAB, dB_AAAB, dC_AAAB, dD_AAAB = crcc_loops.crcc_loops.crcc24b(MM24B,L4B,fA['oo'],fA['vv'],fB['oo'],fB['vv'],\
                    H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],\
                    H2A['voov'],H2A['oooo'],H2A['vvvv'],\
                    H2B['ovov'],H2B['vovo'],H2B['oooo'],H2B['vvvv'],\
                    H2C['voov'],\
                    D3A['O'],D3B['V'],D3B['O'],D3B['V'],D3C['O'],D3C['V'])

    # MM24C correction
    MM24C = build_MM24C(cc_t,H2A,H2B,H2C,sys)
    L4C = build_L4C(cc_t,ints,sys,iroot=iroot) 
    dA_AABB, dB_AABB, dC_AABB, dD_AABB = crcc_loops.crcc_loops.crcc24c(MM24C,L4C,fA['oo'],fA['vv'],fB['oo'],fB['vv'],\
                    H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],\
                    H2A['voov'],H2A['oooo'],H2A['vvvv'],\
                    H2B['ovov'],H2B['vovo'],H2B['oooo'],H2B['vvvv'],\
                    H2C['voov'],H2C['oooo'],H2C['vvvv'],\
                    D3A['O'],D3B['V'],D3B['O'],D3B['V'],D3C['O'],D3C['V'],D3D['O'],D3D['V'])

    if flag_RHF:
        deltaA = 2.0*dA_AAAA + 2.0*dA_AAAB + dA_AABB
        deltaB = 2.0*dB_AAAA + 2.0*dB_AAAB + dB_AABB
        deltaC = 2.0*dC_AAAA + 2.0*dC_AAAB + dC_AABB
        deltaD = 2.0*dD_AAAA + 2.0*dD_AAAB + dD_AABB

    Ecorr = calc_cc_energy(cc_t,ints)
    EcorrA = Ecorr + deltaA; EcorrB = Ecorr + deltaB; EcorrC = Ecorr + deltaC; EcorrD = Ecorr + deltaD
    E24A = ints['Escf'] + EcorrA
    E24B = ints['Escf'] + EcorrB
    E24C = ints['Escf'] + EcorrC
    E24D = ints['Escf'] + EcorrD
 
    print('CR-CC(2,4)_A = {} Eh     Ecorr_A = {} Eh     Delta_A = {} Eh'.format(E24A,EcorrA,deltaA))
    print('CR-CC(2,4)_B = {} Eh     Ecorr_B = {} Eh     Delta_B = {} Eh'.format(E24B,EcorrB,deltaB))
    print('CR-CC(2,4)_C = {} Eh     Ecorr_C = {} Eh     Delta_C = {} Eh'.format(E24C,EcorrC,deltaC))
    print('CR-CC(2,4)_D = {} Eh     Ecorr_D = {} Eh     Delta_D = {} Eh'.format(E24D,EcorrD,deltaD))

    Ecrcc24 = {'A' : E24A, 'B' : E24B, 'C' : E24C, 'D' : E24D}
    delta24 = {'A' : deltaA, 'B' : deltaB, 'C' : deltaC, 'D' : deltaD}

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))
    return Ecrcc24, delta24

def build_MM24A(cc_t,H2A,sys):

    t2a = cc_t['t2a']

    # (jl/i/k)(bc/a/d)
    D1 = -np.einsum('amie,bcmk,edjl->abcdijkl',H2A['voov'],t2a,t2a,optimize=True)
    # (jl/ik)(ik)
    D1 += -permute(D1,[1,2,3,4,7,6,5,8])
    D1 += -permute(D1,[1,2,3,4,6,5,7,8]) - permute(D1,[1,2,3,4,5,7,6,8]) - permute(D1,[1,2,3,4,8,6,7,5])\
    -permute(D1,[1,2,3,4,5,6,8,7]) + permute(D1,[1,2,3,4,6,5,8,7])
    # A(bc/a/d) = A(bc/ad)A(ad)
    D1 += -permute(D1,[4,2,3,1,5,6,7,8])
    D1 += -permute(D1,[2,1,3,4,5,6,7,8]) - permute(D1,[3,2,1,4,5,6,7,8]) - permute(D1,[1,4,3,2,5,6,7,8])\
    -permute(D1,[1,2,4,3,5,6,7,8]) + permute(D1,[2,1,4,3,5,6,7,8])

    # (ij/kl)(bc/ad)
    D2 = np.einsum('mnij,adml,bcnk->abcdijkl',H2A['oooo'],t2a,t2a,optimize=True)
    # (ij/kl)
    D2 += -permute(D2,[1,2,3,4,7,6,5,8]) - permute(D2,[1,2,3,4,8,6,7,5]) - permute(D2,[1,2,3,4,5,7,6,8])\
    -permute(D2,[1,2,3,4,5,8,7,6]) + permute(D2,[1,2,3,4,7,8,5,6])
    # (bc/ad)
    D2 += -permute(D2,[2,1,3,4,5,6,7,8]) - permute(D2,[3,2,1,4,5,6,7,8]) - permute(D2,[1,4,3,2,5,6,7,8])\
    -permute(D2,[1,2,4,3,5,6,7,8]) + permute(D2,[2,1,4,3,5,6,7,8])

    # (jk/il)(ab/cd)
    D3 = np.einsum('abef,fcjk,edil->abcdijkl',H2A['vvvv'],t2a,t2a,optimize=True)
    # (jk/il)
    D3 += -permute(D3,[1,2,3,4,6,5,7,8]) - permute(D3,[1,2,3,4,7,6,5,8]) - permute(D3,[1,2,3,4,5,6,8,7])\
    -permute(D3,[1,2,3,4,5,8,7,6]) + permute(D3,[1,2,3,4,6,5,8,7])
    # (ab/cd)
    D3 += -permute(D3,[3,2,1,4,5,6,7,8]) - permute(D3,[4,2,3,1,5,6,7,8]) - permute(D3,[1,3,2,4,5,6,7,8])\
    -permute(D3,[1,4,3,2,5,6,7,8]) + permute(D3,[3,4,1,2,5,6,7,8])

    MM24A = D1 + D2 + D3

    return MM24A

def build_MM24B(cc_t,H2A,H2B,sys):

    def i_jk(x):
        return x - permute(x,[1,2,3,4,6,5,7,8]) - permute(x,[1,2,3,4,7,6,5,8])
    def c_ab(x):
        return x - permute(x,[3,2,1,4,5,6,7,8]) - permute(x,[1,3,2,4,5,6,7,8])
    def a_bc(x):
        return x - permute(x,[2,1,3,4,5,6,7,8]) - permute(x,[3,2,1,4,5,6,7,8])
    def k_ij(x):
        return x - permute(x,[1,2,3,4,7,6,5,8]) - permute(x,[1,2,3,4,5,7,6,8])
    def i_kj(x):
        return x - permute(x,[1,2,3,4,7,6,5,8]) - permute(x,[1,2,3,4,6,5,7,8])
    def jk(x):
        return x - permute(x,[1,2,3,4,5,7,6,8])
    def bc(x):
        return x - permute(x,[1,3,2,4,5,6,7,8])

    t2a = cc_t['t2a']
    t2b = cc_t['t2b']

    MM24B = 0.0
    #dm_voov = 0.0
    #dm_oooo = 0.0
    #dm_vvvv = 0.0

    # (i/jk)(c/ab)
    D1 = -np.einsum('mdel,abim,ecjk->abcdijkl',H2B['ovvo'],t2a,t2a,optimize=True)
    D1 = i_jk(c_ab(D1))
    MM24B += D1
    #dm_voov += D1
    # (k/ij)(a/bc)
    D2 = +np.einsum('mnij,bcnk,adml->abcdijkl',H2A['oooo'],t2a,t2b,optimize=True)
    D2 = k_ij(a_bc(D2))
    MM24B += D2
    #dm_oooo += D2
    # (ijk)(c/ab) = (i/jk)(c/ab)(jk)
    D3 = -np.einsum('mdjf,abim,cfkl->abcdijkl',H2B['ovov'],t2a,t2b,optimize=True)
    D3 = i_jk(c_ab(jk(D3)))
    MM24B += D3
    #dm_voov += D3
    # (ijk)(abc) = (i/jk)(a/bc)(jk)(bc)
    D4 = -np.einsum('amie,bejl,cdkm->abcdijkl',H2B['voov'],t2b,t2b,optimize=True)
    D4 = i_jk(a_bc(jk(bc(D4))))
    MM24B += D4
    #dm_voov += D4
    # (ijk)(a/bc) = (i/jk)(a/bc)(jk)
    D5 = +np.einsum('mnjl,bcmk,adin->abcdijkl',H2B['oooo'],t2a,t2b,optimize=True)
    D5 = i_jk(a_bc(jk(D5)))
    MM24B += D5
    #dm_oooo += D5
    # (i/jk)(abc) = (i/jk)(a/bc)(bc)
    D6 = -np.einsum('bmel,ecjk,adim->abcdijkl',H2B['vovo'],t2a,t2b,optimize=True)
    D6 = i_jk(a_bc(bc(D6)))
    MM24B += D6
    #dm_voov += D6
    # (i/kj)(abc) = (i/kj)(a/bc)(bc)
    D7 = -np.einsum('amie,ecjk,bdml->abcdijkl',H2A['voov'],t2a,t2b,optimize=True)
    D7 = i_kj(a_bc(bc(D7)))
    MM24B += D7
    #dm_voov += D7
    # (i/jk)(c/ab) = (i/jk)(c/ab)
    D8 = +np.einsum('abef,fcjk,edil->abcdijkl',H2A['vvvv'],t2a,t2b,optimize=True)
    D8 = i_jk(c_ab(D8))
    MM24B += D8
    #dm_vvvv += D8
    # (ijk)(a/bc) = (i/jk)(a/bc)(jk)
    D9 = -np.einsum('amie,bcmk,edjl->abcdijkl',H2A['voov'],t2a,t2b,optimize=True)
    D9 = i_jk(a_bc(jk(D9)))
    MM24B += D9
    #dm_voov += D9
    # (k/ij)(abc) = (k/ij)(a/bc)(bc)
    D10 = +np.einsum('adef,ebij,cfkl->abcdijkl',H2B['vvvv'],t2a,t2b,optimize=True)
    D10 = k_ij(a_bc(bc(D10)))
    MM24B += D10
    #dm_vvvv += D10

    return MM24B

def build_MM24C(cc_t,H2A,H2B,H2C,sys):

    def ij(x):
        return x - permute(x,[1,2,3,4,6,5,7,8])
    def kl(x): 
        return x - permute(x,[1,2,3,4,5,6,8,7])
    def ab(x):
        return x - permute(x,[2,1,3,4,5,6,7,8])
    def cd(x):
        return x - permute(x,[1,2,4,3,5,6,7,8])

    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']

    MM24C = 0.0
    #dm_voov = 0.0
    #dm_oooo = 0.0
    #dm_vvvv = 0.0

    # 1 - (ij)(kl)(ab)(cd)
    D1 = -np.einsum('cmke,adim,bejl->abcdijkl',H2C['voov'],t2b,t2b,optimize=True)
    D1 = ij(kl(ab(cd(D1))))
    MM24C += D1
    #dm_voov += D1
    # 2 - (ij)(kl)(ab)(cd)
    D2 = -np.einsum('amie,bcmk,edjl->abcdijkl',H2A['voov'],t2b,t2b,optimize=True)
    D2 = ij(kl(ab(cd(D2)))) 
    MM24C += D2
    #dm_voov += D2
    # 3 - (kl)(ab)(cd)
    D3 = -np.einsum('mcek,aeij,bdml->abcdijkl',H2B['ovvo'],t2a,t2b,optimize=True)
    D3 = kl(ab(cd(D3))) 
    MM24C += D3
    #dm_voov += D3
    # 4 - (ij)(ab)(cd)
    D4 = -np.einsum('amie,bdjm,cekl->abcdijkl',H2B['voov'],t2b,t2c,optimize=True)
    D4 = ij(ab(cd(D4))) 
    MM24C += D4
    #dm_voov += D4
    # 5 - (ij)(kl)(cd)
    D5 = -np.einsum('mcek,abim,edjl->abcdijkl',H2B['ovvo'],t2a,t2b,optimize=True)
    D5 = ij(kl(cd(D5))) 
    MM24C += D5 
    #dm_voov += D5
    # 6 - (ij)(kl)(ab)
    D6 = -np.einsum('amie,cdkm,bejl->abcdijkl',H2B['voov'],t2c,t2b,optimize=True)
    D6 = ij(kl(ab(D6))) 
    MM24C += D6 
    #dm_voov += D6
    # 7 - (ij)(kl)(ab)(cd)
    D7 = -np.einsum('bmel,adim,ecjk->abcdijkl',H2B['vovo'],t2b,t2b,optimize=True)
    D7 = ij(kl(ab(cd(D7)))) 
    MM24C += D7 
    #dm_voov += D7
    # 8 - (ij)(kl)(ab)(cd)
    D8 = -np.einsum('mdje,bcmk,aeil->abcdijkl',H2B['ovov'],t2b,t2b,optimize=True)
    D8 = ij(kl(ab(cd(D8)))) 
    MM24C += D8 
    #dm_voov += D8 
    # 9 - (ij)(cd)
    D9 = -np.einsum('mdje,abim,cekl->abcdijkl',H2B['ovov'],t2a,t2c,optimize=True)
    D9 = ij(cd(D9)) 
    MM24C += D9 
    #dm_voov += D9
    # 10 - (kl)(ab)
    D10 = -np.einsum('bmel,cdkm,aeij->abcdijkl',H2B['vovo'],t2c,t2a,optimize=True)
    D10 = kl(ab(D10)) 
    MM24C += D10 
    #dm_voov += D10
    # 11 - (kl)(ab) !!!
    D11 = np.einsum('mnij,acmk,bdnl->abcdijkl',H2A['oooo'],t2b,t2b,optimize=True)
    D11 = kl(ab(D11)) 
    MM24C += D11 
    #dm_oooo += D11
    # 12 - (ij)(kl) !!!
    D12 = np.einsum('abef,ecik,fdjl->abcdijkl',H2A['vvvv'],t2b,t2b,optimize=True)
    D12 = ij(kl(D12))
    MM24C += D12 
    #dm_vvvv += D12
    # 13 - (ij)(kl)
    D13 = np.einsum('mnik,abmj,cdnl->abcdijkl',H2B['oooo'],t2a,t2c,optimize=True)
    D13 = ij(kl(D13)) 
    MM24C += D13
    #dm_oooo += D13
    # 14 - (ab)(cd) 
    D14 = np.einsum('acef,ebij,fdkl->abcdijkl',H2B['vvvv'],t2a,t2c,optimize=True)
    D14 = ab(cd(D14)) 
    MM24C += D14
    #dm_vvvv += D14
    # 15 - (ij)(kl)(ab)(cd) 
    D15 = np.einsum('mnik,adml,bcjn->abcdijkl',H2B['oooo'],t2b,t2b,optimize=True)
    D15 = ij(kl(ab(cd(D15)))) 
    MM24C += D15 
    #dm_oooo += D15
    # 16 - (ij)(kl)(ab)(cd)
    D16 = np.einsum('acef,edil,bfjk->abcdijkl',H2B['vvvv'],t2b,t2b,optimize=True)
    D16 = ij(kl(ab(cd(D16)))) 
    MM24C += D16 
    #dm_vvvv += D16 
    # 17 - (ij)(cd) !!!
    D17 = np.einsum('mnkl,adin,bcjm->abcdijkl',H2C['oooo'],t2b,t2b,optimize=True)
    D17 = ij(cd(D17)) 
    MM24C += D17 
    #dm_oooo += D17 
    # 18 - (ij)(kl) !!!
    D18 = np.einsum('cdef,afil,bejk->abcdijkl',H2C['vvvv'],t2b,t2b,optimize=True)
    D18 = ij(kl(D18)) 
    MM24C += D18
    #dm_vvvv += D18

    return MM24C

def build_MM24D(cc_t,H2B,H2C,sys):

    return MM24D


def build_MM24E(cc_t,H2C,sys):

    return MM24E

def build_L4A(cc_t,ints,sys,iroot=0):

    vA = ints['vA']
    l2a = cc_t['l2a'][iroot]

    L4A = np.einsum('ijab,cdkl->abcdijkl',vA['oovv'],l2a,optimize=True)

    L4A += -permute(L4A,[1,2,3,4,7,6,5,8]) - permute(L4A,[1,2,3,4,8,6,7,5])\
        -permute(L4A,[1,2,3,4,5,7,6,8]) - permute(L4A,[1,2,3,4,5,8,7,6])\
        +permute(L4A,[1,2,3,4,7,8,5,6]) # A(ij/kl)
    L4A += -permute(L4A,[3,2,1,4,5,6,7,8]) - permute(L4A,[4,2,3,1,5,6,7,8])\
        -permute(L4A,[1,3,2,4,5,6,7,8]) - permute(L4A,[1,4,3,2,5,6,7,8])\
        +permute(L4A,[3,4,1,2,5,6,7,8]) # A(ab/cd)

    return L4A

def build_L4B(cc_t,ints,sys,iroot=0):

    vA = ints['vA']
    vB = ints['vB']
    l2a = cc_t['l2a'][iroot]
    l2b = cc_t['l2b'][iroot]

    L4B = np.zeros((sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_b'],\
    sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_b']))

    L4B = np.einsum('ijab,cdkl->abcdijkl',vA['oovv'],l2b,optimize=True)\
        +np.einsum('klcd,abij->abcdijkl',vB['oovv'],l2a,optimize=True)

    L4B += -permute(L4B,[1,2,3,4,7,6,5,8]) - permute(L4B,[1,2,3,4,5,7,6,8]) # A(k/ij)
    L4B += -permute(L4B,[3,2,1,4,5,6,7,8]) - permute(L4B,[1,3,2,4,5,6,7,8]) # A(c/ab)

    return L4B

def build_L4C(cc_t,ints,sys,iroot=0):

    vA = ints['vA']
    vB = ints['vB']
    vC = ints['vC']
    l2a = cc_t['l2a'][iroot]
    l2b = cc_t['l2b'][iroot]
    l2c = cc_t['l2c'][iroot]

    L4C = np.zeros((sys['Nunocc_a'],sys['Nunocc_a'],sys['Nunocc_b'],sys['Nunocc_b'],\
    sys['Nocc_a'],sys['Nocc_a'],sys['Nocc_b'],sys['Nocc_b']))

    L4C += np.einsum('ijab,cdkl->abcdijkl',vA['oovv'],l2c,optimize=True)
    L4C += np.einsum('abij,klcd->abcdijkl',l2a,vC['oovv'],optimize=True)
    D1 = np.einsum('bcjk,ilad->abcdijkl',l2b,vB['oovv'],optimize=True)

    D1 += -permute(D1,[1,2,3,4,6,5,7,8]) # A(ij)
    D1 += -permute(D1,[1,2,3,4,5,6,8,7]) # A(kl)
    D1 += -permute(D1,[2,1,3,4,5,6,7,8]) # A(ab)
    D1 += -permute(D1,[1,2,4,3,5,6,7,8]) # A(cd)
    L4C += D1

    return L4C


def build_L4D(cc_t,ints,sys,iroot=0):

    vB = ints['vB']
    vC = ints['vC']
    l2b = cc_t['l2b'][iroot]
    l2c = cc_t['l2c'][iroot]

    L4D = np.zeros((sys['Nunocc_a'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nunocc_b'],\
    sys['Nocc_a'],sys['Nocc_b'],sys['Nocc_b'],sys['Nocc_b']))

    L4D = np.einsum('klcd,abij->abcdijkl',vC['oovv'],l2b,optimize=True)\
    +np.einsum('ijab,cdkl->abcdijkl',vB['oovv'],l2c,optimize=True)

    L4D += -permute(L4D,[1,2,3,4,5,7,6,8]) - permute(L4D,[1,2,3,4,5,8,7,6]) # A(j/kl)
    L4D += -permute(L4D,[1,3,2,4,5,6,7,8]) - permute(L4D,[1,4,3,2,5,6,7,8]) # A(b/cd)

    return L4D

def build_L4E(cc_t,ints,sys,iroot=0):

    vC = ints['vC']
    l2c = cc_t['l2c'][iroot]

    L4E = np.zeros((sys['Nunocc_b'],sys['Nunocc_b'],sys['Nunocc_b'],sys['Nunocc_b'],\
    sys['Nocc_b'],sys['Nocc_b'],sys['Nocc_b'],sys['Nocc_b']))

    L4E += np.einsum('ijab,cdkl->abcdijkl',vC['oovv'],l2c,optimize=True)

    L4E = L4E - permute(L4E,[1,2,3,4,7,6,5,8]) - permute(L4E,[1,2,3,4,8,6,7,5])\
        -permute(L4E,[1,2,3,4,5,7,6,8]) - permute(L4E,[1,2,3,4,5,8,7,6])\
        +permute(L4E,[1,2,3,4,7,8,5,6]) # A(ij/kl)
    L4E = L4E - permute(L4E,[3,2,1,4,5,6,7,8]) - permute(L4E,[4,2,3,1,5,6,7,8])\
        -permute(L4E,[1,3,2,4,5,6,7,8]) - permute(L4E,[1,4,3,2,5,6,7,8])\
        +permute(L4E,[3,4,1,2,5,6,7,8]) # A(ab/cd)

    return L4E

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

def permute(x,perm_list):
    str1 = ['a','b','c','d','i','j','k','l']
    str2 = ''.join([str1[x-1] for x in perm_list]) 
    str1 = ''.join(s for s in str1)
    contr = str1+'->'+str2
    return np.einsum(contr,x,optimize=True)

def test_updates(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys):

    from scipy.io import loadmat

    print('')
    print('TEST SUBROUTINE:')
    print('')

    # test MM24A update
    MM24A = build_MM24A(cc_t,H2A,sys)
    print('|MM24A| = {}'.format(np.linalg.norm(MM24A)))

    # test MM24B update
    MM24B = build_MM24B(cc_t,H2A,H2B,sys)
    print('|MM24B| = {}'.format(np.linalg.norm(MM24B)))
    #print('|voov| = {}'.format(np.linalg.norm(dm_voov)))
    #print('|oooo| = {}'.format(np.linalg.norm(dm_oooo))) 
    #print('|vvvv| = {}'.format(np.linalg.norm(dm_vvvv)))

    # test MM24C update
    MM24C,dm_voov,dm_oooo,dm_vvvv = build_MM24C(cc_t,H2A,H2B,H2C,sys)
    print('|MM24C| = {}'.format(np.linalg.norm(MM24C)))
    print('|voov| = {}'.format(np.linalg.norm(dm_voov)))
    print('|oooo| = {}'.format(np.linalg.norm(dm_oooo))) 
    print('|vvvv| = {}'.format(np.linalg.norm(dm_vvvv)))

    # # test MM24D update
    # MM24D = build_MM24D(cc_t,H2B,H2C,sys)
    # print('|MM24D| = {}'.format(np.linalg.norm(MM24D)))

    # # test MM24D update
    # MM24E = build_MM24E(cc_t,H2C,sys)
    # print('|MM24E| = {}'.format(np.linalg.norm(MM24E)))

    # test L4A update
    L4A = build_L4A(cc_t,ints,sys)
    print('|L4A| = {}'.format(np.linalg.norm(L4A)))

    # test L4B update
    L4B = build_L4B(cc_t,ints,sys)
    print('|L4B| = {}'.format(np.linalg.norm(L4B)))

    # test L4C update
    L4C = build_L4C(cc_t,ints,sys)
    print('|L4C| = {}'.format(np.linalg.norm(L4C)))

    # test L4D update
    L4D = build_L4D(cc_t,ints,sys)
    print('|L4D| = {}'.format(np.linalg.norm(L4D)))

    # test L4E update
    L4E = build_L4E(cc_t,ints,sys)
    print('|L4E| = {}'.format(np.linalg.norm(L4E)))

    return
