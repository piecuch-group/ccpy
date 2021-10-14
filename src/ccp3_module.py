import numpy as np 
import time 

def ccp3(cc_t,p_space,H1A,H1B,H2A,H2B,H2C,ints,sys,iroot=0):

    print('\n==================================++Entering CC(P;3) Routine++=============================')
    
    t_start = time.time()
    
    # get fock matrices
    fA = ints['fA']; fB = ints['fB']
    
    # get the 3-body HBar triples diagonal
    D3A, D3B, D3C, D3D = triples_3body_diagonal(cc_t,ints,sys)

    # MM correction containers
    deltaA = 0.0; # using MP denominator -(f_aa - f_ii + f_bb - f_jj + f_cc - f_kk)
    deltaB = 0.0; # using EN denominator -<phi_{ijk}^{abc} | H_1(CCSD) | phi_{ijk}^{abc}>
    deltaC = 0.0; # using EN denominator -<phi_{ijk}^{abc} | H_1(CCSD) + H_2(CCSD) | phi_{ijk}^{abc}>
    deltaD = 0.0; # using EN denominator -<phi_{ijk}^{abc} | H_1(CCSD) + H_2(CCSD) + H_3(CCSD) | phi_{ijk}^{abc}>
        
    # MM23A correction
    MM23A = build_MM23A(cc_t,H1A,H2A,sys)
    L3A = build_L3A(cc_t,H1A,H2A,ints,sys,iroot=iroot)
    for a in range(sys['Nunocc_a']):
        for b in range(a+1,sys['Nunocc_a']):
            for c in range(b+1,sys['Nunocc_a']):
                for i in range(sys['Nocc_a']):
                    for j in range(i+1,sys['Nocc_a']):
                        for k in range(j+1,sys['Nocc_a']):

                            if p_space[a,b,c,i,j,k] == 1:
                                continue
                            else:
                                LM = L3A[a,b,c,i,j,k] * MM23A[a,b,c,i,j,k]
                            
                                DMP = fA['vv'][a,a] + fA['vv'][b,b] + fA['vv'][c,c]\
                                -fA['oo'][i,i] - fA['oo'][j,j] - fA['oo'][k,k] 
                                
                                D1 = H1A['vv'][a,a] + H1A['vv'][b,b] + H1A['vv'][c,c]\
                                -H1A['oo'][i,i] - H1A['oo'][j,j] - H1A['oo'][k,k]

                                D2 = -H2A['voov'][a,i,i,a]-H2A['voov'][b,i,i,b]-H2A['voov'][c,i,i,c]\
                                -H2A['voov'][a,j,j,a]-H2A['voov'][b,j,j,b]-H2A['voov'][c,j,j,c]\
                                -H2A['voov'][a,k,k,a]-H2A['voov'][b,k,k,b]-H2A['voov'][c,k,k,c]\
                                -H2A['oooo'][j,i,j,i]-H2A['oooo'][k,i,k,i]-H2A['oooo'][k,j,k,j]\
                                -H2A['vvvv'][b,a,b,a]-H2A['vvvv'][c,a,c,a]-H2A['vvvv'][c,b,c,b]
                                D2 *= -1.0
                            
                                D3 = -D3A['O'][a,i,j]-D3A['O'][a,i,k]-D3A['O'][a,j,k]\
                                -D3A['O'][b,i,j]-D3A['O'][b,i,k]-D3A['O'][b,j,k]\
                                -D3A['O'][c,i,j]-D3A['O'][c,i,k]-D3A['O'][c,j,k]\
                                +D3A['V'][a,i,b]+D3A['V'][a,i,c]+D3A['V'][b,i,c]\
                                +D3A['V'][a,j,b]+D3A['V'][a,j,c]+D3A['V'][b,j,c]\
                                +D3A['V'][a,k,b]+D3A['V'][a,k,c]+D3A['V'][b,k,c]
            
                                D_A = -1.0 * DMP
                                D_B = -1.0 * D1
                                D_C = -1.0 * (D1+D2)
                                D_D = -1.0 * (D1+D2+D3)

                                deltaA += LM/D_A
                                deltaB += LM/D_B
                                deltaC += LM/D_C
                                deltaD += LM/D_D
    
    # MM23B correction
    MM23B = build_MM23B(cc_t,H1A,H1B,H2A,H2B,sys) 
    L3B = build_L3B(cc_t,H1A,H1B,H2A,H2B,ints,sys,iroot=iroot)   
    for a in range(sys['Nunocc_a']):
        for b in range(a+1,sys['Nunocc_a']):
            for c in range(sys['Nunocc_b']):
                for i in range(sys['Nocc_a']):
                    for j in range(i+1,sys['Nocc_a']):
                        for k in range(sys['Nocc_b']):

                            if p_space[a,b,c,i,j,k] == 1:
                                continue
                            else:
                                LM = L3B[a,b,c,i,j,k] * MM23B[a,b,c,i,j,k]
                            
                                DMP = fA['vv'][a,a] + fA['vv'][b,b] + fB['vv'][c,c]\
                                -fA['oo'][i,i] - fA['oo'][j,j] - fB['oo'][k,k] 
                                
                                D1 = H1A['vv'][a,a] + H1A['vv'][b,b] + H1B['vv'][c,c]\
                                -H1A['oo'][i,i] - H1A['oo'][j,j] - H1B['oo'][k,k]
                                
                                D2 = -H2A['voov'][a,i,i,a]-H2A['voov'][b,i,i,b]+H2B['vovo'][c,i,c,i]\
                                -H2A['voov'][a,j,j,a]-H2A['voov'][b,j,j,b]+H2B['vovo'][c,j,c,j]\
                                +H2B['ovov'][k,a,k,a]+H2B['ovov'][k,b,k,b]-H2C['voov'][c,k,k,c]\
                                -H2A['oooo'][j,i,j,i]-H2B['oooo'][k,i,k,i]-H2B['oooo'][k,j,k,j]\
                                -H2A['vvvv'][b,a,b,a]-H2B['vvvv'][c,a,c,a]-H2B['vvvv'][c,b,c,b]
                                D2 *= -1.0
                            
                                D3 = -D3A['O'][a,i,j]-D3B['O'][a,i,k]-D3B['O'][a,j,k]\
                                -D3A['O'][b,i,j]-D3B['O'][b,i,k]-D3B['O'][b,j,k]\
                                -D3C['O'][c,i,k]-D3C['O'][c,j,k]\
                                +D3A['V'][a,i,b]+D3B['V'][a,i,c]+D3B['V'][b,i,c]\
                                +D3A['V'][a,j,b]+D3B['V'][a,j,c]+D3B['V'][b,j,c]\
                                +D3C['V'][a,k,c]+D3C['V'][b,k,c]

                                D_A = -1.0 * DMP
                                D_B = -1.0 * D1
                                D_C = -1.0 * (D1+D2)
                                D_D = -1.0 * (D1+D2+D3)

                                deltaA += LM/D_A
                                deltaB += LM/D_B
                                deltaC += LM/D_C
                                deltaD += LM/D_D
    
    # MM23C correction
    MM23C = build_MM23C(cc_t,H1A,H1B,H2B,H2C,sys) 
    L3C = build_L3C(cc_t,H1A,H1B,H2B,H2C,ints,sys,iroot=iroot)   
    for a in range(sys['Nunocc_a']):
        for b in range(sys['Nunocc_b']):
            for c in range(b+1,sys['Nunocc_b']):
                for i in range(sys['Nocc_a']):
                    for j in range(sys['Nocc_b']):
                        for k in range(j+1,sys['Nocc_b']):
        
                            if p_space[a,b,c,i,j,k] == 1:
                                continue
                            else:
                                LM = L3C[a,b,c,i,j,k] * MM23C[a,b,c,i,j,k]
                            
                                DMP = fA['vv'][a,a] + fB['vv'][b,b] + fB['vv'][c,c]\
                                -fA['oo'][i,i] - fB['oo'][j,j] - fB['oo'][k,k] 
                                
                                D1 = H1A['vv'][a,a] + H1B['vv'][b,b] + H1B['vv'][c,c]\
                                -H1A['oo'][i,i] - H1B['oo'][j,j] - H1B['oo'][k,k]

                                D2 = -H2A['voov'][a,i,i,a]+H2B['vovo'][b,i,b,i]+H2B['vovo'][c,i,c,i]\
                                +H2B['ovov'][j,a,j,a]-H2C['voov'][b,j,j,b]-H2C['voov'][c,j,j,c]\
                                +H2B['ovov'][k,a,k,a]-H2C['voov'][b,k,k,b]-H2C['voov'][c,k,k,c]\
                                -H2B['oooo'][j,i,j,i]-H2B['oooo'][k,i,k,i]-H2C['oooo'][k,j,k,j]\
                                -H2B['vvvv'][b,a,b,a]-H2B['vvvv'][c,a,c,a]-H2C['vvvv'][c,b,c,b]
                                D2 *= -1.0
                            
                                D3 = -D3B['O'][a,i,j]-D3B['O'][a,i,k]\
                                -D3C['O'][b,i,j]-D3C['O'][b,i,k]-D3D['O'][b,j,k]\
                                -D3C['O'][c,i,j]-D3C['O'][c,i,k]-D3D['O'][c,j,k]\
                                +D3B['V'][a,i,b]+D3B['V'][a,i,c]\
                                +D3C['V'][a,j,b]+D3C['V'][a,j,c]+D3D['V'][b,j,c]\
                                +D3C['V'][a,k,b]+D3C['V'][a,k,c]+D3D['V'][b,k,c]

                                D_A = -1.0 * DMP
                                D_B = -1.0 * D1
                                D_C = -1.0 * (D1+D2)
                                D_D = -1.0 * (D1+D2+D3)

                                deltaA += LM/D_A
                                deltaB += LM/D_B
                                deltaC += LM/D_C
                                deltaD += LM/D_D
    
    # MM23D correction
    MM23D = build_MM23D(cc_t,H1B,H2C,sys) 
    L3D = build_L3D(cc_t,H1B,H2C,ints,sys,iroot=iroot)
    for a in range(sys['Nunocc_b']):
        for b in range(a+1,sys['Nunocc_b']):
            for c in range(b+1,sys['Nunocc_b']):
                for i in range(sys['Nocc_b']):
                    for j in range(i+1,sys['Nocc_b']):
                        for k in range(j+1,sys['Nocc_b']):

                            if p_space[a,b,c,i,j,k] == 1:
                                continue
                            else:
                                LM = L3D[a,b,c,i,j,k] * MM23D[a,b,c,i,j,k]
                            
                                DMP = fB['vv'][a,a] + fB['vv'][b,b] + fB['vv'][c,c]\
                                -fB['oo'][i,i] - fB['oo'][j,j] - fB['oo'][k,k] 
                                
                                D1 = H1B['vv'][a,a] + H1B['vv'][b,b] + H1B['vv'][c,c]\
                                -H1B['oo'][i,i] - H1B['oo'][j,j] - H1B['oo'][k,k]

                                D2 = -H2C['voov'][a,i,i,a]-H2C['voov'][b,i,i,b]-H2C['voov'][c,i,i,c]\
                                -H2C['voov'][a,j,j,a]-H2C['voov'][b,j,j,b]-H2C['voov'][c,j,j,c]\
                                -H2C['voov'][a,k,k,a]-H2C['voov'][b,k,k,b]-H2C['voov'][c,k,k,c]\
                                -H2C['oooo'][j,i,j,i]-H2C['oooo'][k,i,k,i]-H2C['oooo'][k,j,k,j]\
                                -H2C['vvvv'][b,a,b,a]-H2C['vvvv'][c,a,c,a]-H2C['vvvv'][c,b,c,b]
                                D2 *= -1.0
                            
                                D3 = -D3D['O'][a,i,j]-D3D['O'][a,i,k]-D3D['O'][a,j,k]\
                                -D3D['O'][b,i,j]-D3D['O'][b,i,k]-D3D['O'][b,j,k]\
                                -D3D['O'][c,i,j]-D3D['O'][c,i,k]-D3D['O'][c,j,k]\
                                +D3D['V'][a,i,b]+D3D['V'][a,i,c]+D3D['V'][b,i,c]\
                                +D3D['V'][a,j,b]+D3D['V'][a,j,c]+D3D['V'][b,j,c]\
                                +D3D['V'][a,k,b]+D3D['V'][a,k,c]+D3D['V'][b,k,c]

                                D_A = -1.0 * DMP
                                D_B = -1.0 * D1
                                D_C = -1.0 * (D1+D2)
                                D_D = -1.0 * (D1+D2+D3)

                                deltaA += LM/D_A
                                deltaB += LM/D_B
                                deltaC += LM/D_C
                                deltaD += LM/D_D
    
    Ecorr = calc_cc_energy(cc_t,ints)

    EcorrA = Ecorr + deltaA; EcorrB = Ecorr + deltaB; EcorrC = Ecorr + deltaC; EcorrD = Ecorr + deltaD

    EP3A = ints['Escf'] + EcorrA
    EP3B = ints['Escf'] + EcorrB
    EP3C = ints['Escf'] + EcorrC
    EP3D = ints['Escf'] + EcorrD
 
    print('CC(P,3)_A = {} Eh     Ecorr_A = {} Eh     Delta_A = {} Eh'.format(EP3A,EcorrA,deltaA))
    print('CC(P,3)_B = {} Eh     Ecorr_B = {} Eh     Delta_B = {} Eh'.format(EP3B,EcorrB,deltaB))
    print('CC(P,3)_C = {} Eh     Ecorr_C = {} Eh     Delta_C = {} Eh'.format(EP3C,EcorrC,deltaC))
    print('CC(P,3)_D = {} Eh     Ecorr_D = {} Eh     Delta_D = {} Eh'.format(EP3D,EcorrD,deltaD))

    EccP3 = {'A' : EP3A, 'B' : EP3B, 'C' : EP3C, 'D' : EP3D}
    delta23 = {'A' : deltaA, 'B' : deltaB, 'C' : deltaC, 'D' : deltaD}

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))

    return EccP3, delta23

def build_MM23A(cc_t,H1A,H2A,sys):

    print('MMCC(2,3)A construction... ')
    
    t_start = time.time()
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

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))

    return MM23A

def build_MM23B(cc_t,H1A,H1B,H2A,H2B,sys):

    print('MMCC(2,3)B construction... ')
    
    t_start = time.time()
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
        
    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))

    return MM23B

def build_MM23C(cc_t,H1A,H1B,H2B,H2C,sys):

    print('MMCC(2,3)C construction... ')
    
    t_start = time.time()
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
    
    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))

    return MM23C


def build_MM23D(cc_t,H1B,H2C,sys):

    print('MMCC(2,3)D construction... ')
    
    t_start = time.time()
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

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))

    return MM23D


def build_L3A(cc_t,H1A,H2A,ints,sys,iroot=0):

    print('Approximate L3A construction... ')
    
    t_start = time.time()

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

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))

    return L3A

def build_L3B(cc_t,H1A,H1B,H2A,H2B,ints,sys,iroot=0):

    print('Approximate L3B construction... ')
    
    t_start = time.time()

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

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))

    return L3B

def build_L3C(cc_t,H1A,H1B,H2B,H2C,ints,sys,iroot=0):

    print('Approximate L3C construction... ')

    t_start = time.time()

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
    
    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))

    return L3C

def build_L3D(cc_t,H1B,H2C,ints,sys,iroot=0):

    print('Approximate L3D construction... ')
    
    t_start = time.time()

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

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))

    return L3D

def triples_3body_diagonal(cc_t,ints,sys):

    print('\nCalculating 3-body triples diagonal... ')
    t_start = time.time()

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
    
    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))
    
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
