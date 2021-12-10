"""
    Module to perform the CC(P;Q) correction to the CC(P) calculation
    in the adaptive-CC(P;Q) scheme using the CCSD(T)-like correction.
"""
import numpy as np
from cc_energy import calc_cc_energy
import time

def ccp3_pertT(cc_t,p_spaces,ints,sys,flag_RHF=False):
    """Calculate the ground-state CCSD(T)-like correction delta(P;Q)_(T) to 
    the CC(P) calculation defined by the P space contained in p_spaces and 
    its coresponding T vectors and HBar matrices.

    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2 and left amplitudes L1, L2
    p_spaces : dict
        Triples included in the P spaces for each spin case (A - D)
    H1*, H2* : dict
        Sliced CCSD-like similarity-transformed HBar matrices
    ints : dict
        Sliced F_N and V_N integrals
    sys : dict
        System information dictionary
    flag_RHF : bool, optional
        Flag used to determine whether closed-shell RHF symmetry should be used.
        Default value is False.

    Returns
    -------
    Eccp3 : float
        The resulting CC(P;Q)_D correction using the Epstein-Nesbet denominator
    mcA : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Individual CC(P;Q)_D corrections for each triple |ijkabc> in both P and Q spaces
    mcB : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Individual CC(P;Q)_D corrections for each triple |ijk~abc~> in both P and Q spaces
    mcC : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Individual CC(P;Q)_D corrections for each triple |ij~k~ab~c~> in both P and Q spaces
    mcD : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Individual CC(P;Q)_D corrections for each triple |i~j~k~a~b~c~> in both P and Q spaces
    """
    print('\n==================================++Entering CC(P;Q)_(T) Routine++=============================')

    t_start = time.time()

    if flag_RHF:
        print('Using RHF closed-shell symmetry...')

    # system dimensions
    nua = sys['Nunocc_a']; noa = sys['Nocc_a'];
    nub = sys['Nunocc_b']; nob = sys['Nocc_b'];

    # get fock and v matrices
    fA = ints['fA']; fB = ints['fB'];
    vA = ints['vA']; vB = ints['vB']; vC = ints['vC'];

    # get cluster amplitudes t1, t2
    t1a = cc_t['t1a']; t1b = cc_t['t1b'];
    t2a = cc_t['t2a']; t2b = cc_t['t2b']; t2c = cc_t['t2c'];

    delta = 0.0

    mcA = np.zeros((nua,nua,nua,noa,noa,noa))
    MM23A = build_MM23A(cc_t,ints)
    for a in range(sys['Nunocc_a']):
        for b in range(a+1,sys['Nunocc_a']):
            for c in range(b+1,sys['Nunocc_a']):
                for i in range(sys['Nocc_a']):
                    for j in range(i+1,sys['Nocc_a']):
                        for k in range(j+1,sys['Nocc_a']):

                            if p_spaces['A'][a,b,c,i,j,k] == 1:
                                continue

                            Z3 = vA['vvoo'][a,b,i,j]*t1a[c,k]\
                                -vA['vvoo'][a,b,k,j]*t1a[c,i]\
                                -vA['vvoo'][a,b,i,k]*t1a[c,j]\
                                -vA['vvoo'][c,b,i,j]*t1a[a,k]\
                                +vA['vvoo'][c,b,k,j]*t1a[a,i]\
                                +vA['vvoo'][c,b,i,k]*t1a[a,j]\
                                -vA['vvoo'][a,c,i,j]*t1a[b,k]\
                                +vA['vvoo'][a,c,k,j]*t1a[b,i]\
                                +vA['vvoo'][a,c,i,k]*t1a[b,j]
                            
                            denom = fA['oo'][i,i]+fA['oo'][j,j]+fA['oo'][k,k]\
                                    -fA['vv'][a,a]-fA['vv'][b,b]-fA['vv'][c,c]
                            L3 = (Z3 + MM23A[a,b,c,i,j,k])/denom

                            mcA[a,b,c,i,j,k] = L3*MM23A[a,b,c,i,j,k]
                            delta += L3*MM23A[a,b,c,i,j,k]

    mcB = np.zeros((nua,nua,nub,noa,noa,nob))
    MM23B = build_MM23B(cc_t,ints)
    for a in range(sys['Nunocc_a']):
        for b in range(a+1,sys['Nunocc_a']):
            for c in range(sys['Nunocc_b']):
                for i in range(sys['Nocc_a']):
                    for j in range(i+1,sys['Nocc_a']):
                        for k in range(sys['Nocc_b']):

                            if p_spaces['B'][a,b,c,i,j,k] == 1:
                                continue

                            Z3 = vA['vvoo'][a,b,i,j]*t1b[c,k]\
                                +vB['vvoo'][a,c,i,k]*t1a[b,j]\
                                -vB['vvoo'][b,c,i,k]*t1a[a,j]\
                                -vB['vvoo'][a,c,j,k]*t1a[b,i]\
                                +vB['vvoo'][b,c,j,k]*t1a[a,i]
                            
                            denom = fA['oo'][i,i]+fA['oo'][j,j]+fB['oo'][k,k]\
                                    -fA['vv'][a,a]-fA['vv'][b,b]-fB['vv'][c,c]
                            L3 = (Z3 + MM23B[a,b,c,i,j,k])/denom

                            mcB[a,b,c,i,j,k] = L3*MM23B[a,b,c,i,j,k]
                            delta += L3*MM23B[a,b,c,i,j,k]

    mcC = np.zeros((nua,nub,nub,noa,nob,nob))
    MM23C = build_MM23C(cc_t,ints)
    for a in range(sys['Nunocc_a']):
        for b in range(sys['Nunocc_b']):
            for c in range(b+1,sys['Nunocc_b']):
                for i in range(sys['Nocc_a']):
                    for j in range(sys['Nocc_b']):
                        for k in range(j+1,sys['Nocc_b']):

                            if p_spaces['C'][a,b,c,i,j,k] == 1:
                                continue

                            Z3 = vC['vvoo'][b,c,j,k]*t1a[a,i]\
                                +vB['vvoo'][a,b,i,j]*t1b[c,k]\
                                -vB['vvoo'][a,c,i,j]*t1b[b,k]\
                                -vB['vvoo'][a,b,i,k]*t1b[c,j]\
                                +vB['vvoo'][a,c,i,k]*t1b[b,j]
                            
                            denom = fA['oo'][i,i]+fB['oo'][j,j]+fB['oo'][k,k]\
                                    -fA['vv'][a,a]-fB['vv'][b,b]-fB['vv'][c,c]
                            L3 = (Z3 + MM23C[a,b,c,i,j,k])/denom

                            mcC[a,b,c,i,j,k] = L3*MM23C[a,b,c,i,j,k]
                            delta += L3*MM23C[a,b,c,i,j,k]

    mcD = np.zeros((nub,nub,nub,nob,nob,nob))
    MM23D = build_MM23D(cc_t,ints)
    for a in range(sys['Nunocc_b']):
        for b in range(a+1,sys['Nunocc_b']):
            for c in range(b+1,sys['Nunocc_b']):
                for i in range(sys['Nocc_b']):
                    for j in range(i+1,sys['Nocc_b']):
                        for k in range(j+1,sys['Nocc_b']):

                            if p_spaces['D'][a,b,c,i,j,k] == 1:
                                continue

                            Z3 = vC['vvoo'][a,b,i,j]*t1b[c,k]\
                                -vC['vvoo'][a,b,k,j]*t1b[c,i]\
                                -vC['vvoo'][a,b,i,k]*t1b[c,j]\
                                -vC['vvoo'][c,b,i,j]*t1b[a,k]\
                                +vC['vvoo'][c,b,k,j]*t1b[a,i]\
                                +vC['vvoo'][c,b,i,k]*t1b[a,j]\
                                -vC['vvoo'][a,c,i,j]*t1b[b,k]\
                                +vC['vvoo'][a,c,k,j]*t1b[b,i]\
                                +vC['vvoo'][a,c,i,k]*t1b[b,j]
                            
                            denom = fB['oo'][i,i]+fB['oo'][j,j]+fB['oo'][k,k]\
                                    -fB['vv'][a,a]-fB['vv'][b,b]-fB['vv'][c,c]
                            L3 = (Z3 + MM23D[a,b,c,i,j,k])/denom

                            mcD[a,b,c,i,j,k] = L3*MM23D[a,b,c,i,j,k]
                            delta += L3*MM23D[a,b,c,i,j,k]

    Ecorr = calc_cc_energy(cc_t,ints)
    Eparenth = ints['Escf'] + Ecorr + delta
 
    print('CCSD(T) = {} Eh     Ecorr(T) = {} Eh     Delta = {} Eh'.format(Eparenth,Ecorr+delta,delta))

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))

    return Eparenth,mcA,mcB,mcC,mcD

def build_MM23A(cc_t,ints):
    """Calculate the projection <ijkabc|(V_N*T2)_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
        
    Returns
    -------
    MM23A : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Array containing the (V*T2)_C)|0> projections for each i,j,k,a,b,c
    """
    t2a = cc_t['t2a']
    vA = ints['vA']

    MM23A = -1.0*np.einsum('amij,bcmk->abcijk',vA['vooo'],t2a,optimize=True) # (1)
    MM23A += np.einsum('amkj,bcmi->abcijk',vA['vooo'],t2a,optimize=True) # (ik)
    MM23A += np.einsum('amik,bcmj->abcijk',vA['vooo'],t2a,optimize=True) # (jk)
    MM23A += np.einsum('cmij,bamk->abcijk',vA['vooo'],t2a,optimize=True) # (ac)
    MM23A += np.einsum('bmij,acmk->abcijk',vA['vooo'],t2a,optimize=True) # (ab)
    MM23A -= np.einsum('bmkj,acmi->abcijk',vA['vooo'],t2a,optimize=True) # (ab)(ik)
    MM23A -= np.einsum('cmkj,bami->abcijk',vA['vooo'],t2a,optimize=True) # (ac)(ik)
    MM23A -= np.einsum('bmik,acmj->abcijk',vA['vooo'],t2a,optimize=True) # (ab)(jk)
    MM23A -= np.einsum('cmik,bamj->abcijk',vA['vooo'],t2a,optimize=True)    # (ac)(jk)

    MM23A += np.einsum('abie,ecjk->abcijk',vA['vvov'],t2a,optimize=True) # (1)
    MM23A -= np.einsum('abje,ecik->abcijk',vA['vvov'],t2a,optimize=True) # (ij)
    MM23A -= np.einsum('abke,ecji->abcijk',vA['vvov'],t2a,optimize=True) # (ik)
    MM23A -= np.einsum('cbie,eajk->abcijk',vA['vvov'],t2a,optimize=True) # (ac)
    MM23A -= np.einsum('acie,ebjk->abcijk',vA['vvov'],t2a,optimize=True) # (bc)
    MM23A += np.einsum('cbje,eaik->abcijk',vA['vvov'],t2a,optimize=True) # (ac)(ij)
    MM23A += np.einsum('acje,ebik->abcijk',vA['vvov'],t2a,optimize=True) # (bc)(ij)
    MM23A += np.einsum('cbke,eaji->abcijk',vA['vvov'],t2a,optimize=True) # (ac)(ik)
    MM23A += np.einsum('acke,ebji->abcijk',vA['vvov'],t2a,optimize=True) # (bc)(ik)

    return MM23A

def build_MM23B(cc_t,ints):
    """Calculate the projection <ijk~abc~|(V_N*T2)_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
        
    Returns
    -------
    MM23B : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Array containing the (V*T2)_C)|0> projections for each i,j,k~,a,b,c~
    """
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    vA = ints['vA']
    vB = ints['vB']
       
    MM23B = np.einsum('bcek,aeij->abcijk',vB['vvvo'],t2a,optimize=True)
    MM23B -= np.einsum('acek,beij->abcijk',vB['vvvo'],t2a,optimize=True)
    MM23B -= np.einsum('mcjk,abim->abcijk',vB['ovoo'],t2a,optimize=True)
    MM23B += np.einsum('mcik,abjm->abcijk',vB['ovoo'],t2a,optimize=True)
        
    MM23B += np.einsum('acie,bejk->abcijk',vB['vvov'],t2b,optimize=True)
    MM23B -= np.einsum('bcie,aejk->abcijk',vB['vvov'],t2b,optimize=True)
    MM23B -= np.einsum('acje,beik->abcijk',vB['vvov'],t2b,optimize=True)
    MM23B += np.einsum('bcje,aeik->abcijk',vB['vvov'],t2b,optimize=True)
    MM23B -= np.einsum('amik,bcjm->abcijk',vB['vooo'],t2b,optimize=True)
    MM23B += np.einsum('bmik,acjm->abcijk',vB['vooo'],t2b,optimize=True)
    MM23B += np.einsum('amjk,bcim->abcijk',vB['vooo'],t2b,optimize=True)
    MM23B -= np.einsum('bmjk,acim->abcijk',vB['vooo'],t2b,optimize=True)
        
    MM23B += np.einsum('abie,ecjk->abcijk',vA['vvov'],t2b,optimize=True)
    MM23B -= np.einsum('abje,ecik->abcijk',vA['vvov'],t2b,optimize=True)
    MM23B -= np.einsum('amij,bcmk->abcijk',vA['vooo'],t2b,optimize=True)
    MM23B += np.einsum('bmij,acmk->abcijk',vA['vooo'],t2b,optimize=True)

    return MM23B

def build_MM23C(cc_t,ints):
    """Calculate the projection <ij~k~ab~c~|(V_N*T2)_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
        
    Returns
    -------
    MM23C : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Array containing the (V*T2)_C)|0> projections for each i,j~,k~,a,b~,c~
    """
    t2b = cc_t['t2b']
    t2c = cc_t['t2c']
    vB = ints['vB']
    vC = ints['vC']
    
    MM23C = np.einsum('abie,ecjk->abcijk',vB['vvov'],t2c,optimize=True)
    MM23C -= np.einsum('acie,ebjk->abcijk',vB['vvov'],t2c,optimize=True)
    MM23C -= np.einsum('amij,bcmk->abcijk',vB['vooo'],t2c,optimize=True)
    MM23C += np.einsum('amik,bcmj->abcijk',vB['vooo'],t2c,optimize=True)
        
    MM23C += np.einsum('cbke,aeij->abcijk',vC['vvov'],t2b,optimize=True)
    MM23C -= np.einsum('cbje,aeik->abcijk',vC['vvov'],t2b,optimize=True)
    MM23C -= np.einsum('cmkj,abim->abcijk',vC['vooo'],t2b,optimize=True)
    MM23C += np.einsum('bmkj,acim->abcijk',vC['vooo'],t2b,optimize=True)
        
    MM23C += np.einsum('abej,ecik->abcijk',vB['vvvo'],t2b,optimize=True)
    MM23C -= np.einsum('acej,ebik->abcijk',vB['vvvo'],t2b,optimize=True)
    MM23C -= np.einsum('abek,ecij->abcijk',vB['vvvo'],t2b,optimize=True)
    MM23C += np.einsum('acek,ebij->abcijk',vB['vvvo'],t2b,optimize=True)
    MM23C -= np.einsum('mbij,acmk->abcijk',vB['ovoo'],t2b,optimize=True)
    MM23C += np.einsum('mcij,abmk->abcijk',vB['ovoo'],t2b,optimize=True)
    MM23C += np.einsum('mbik,acmj->abcijk',vB['ovoo'],t2b,optimize=True)
    MM23C -= np.einsum('mcik,abmj->abcijk',vB['ovoo'],t2b,optimize=True)

    return MM23C

def build_MM23D(cc_t,ints):
    """Calculate the projection <i~j~k~a~b~c~|(V_N*T2)_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
        
    Returns
    -------
    MM23D : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Array containing the (V*T2)_C)|0> projections for each i~,j~,k~,a~,b~,c~
    """
    t2c = cc_t['t2c']
    vC = ints['vC']
    
    # < phi_{ijkabc} | H_{CCSD} | 0 >
    # = -A(k/ij)A(a/bc) h(amij)*t(bcmk) + A(i/jk)A(c/ab)(h(abie)-h(me)*t(abim))*t(ecjk)
    MM23D = -1.0*np.einsum('amij,bcmk->abcijk',vC['vooo'],t2c,optimize=True) # (1)
    MM23D += np.einsum('amkj,bcmi->abcijk',vC['vooo'],t2c,optimize=True) # (ik)
    MM23D += np.einsum('amik,bcmj->abcijk',vC['vooo'],t2c,optimize=True) # (jk)
    MM23D += np.einsum('cmij,bamk->abcijk',vC['vooo'],t2c,optimize=True) # (ac)
    MM23D += np.einsum('bmij,acmk->abcijk',vC['vooo'],t2c,optimize=True) # (ab)
    MM23D -= np.einsum('bmkj,acmi->abcijk',vC['vooo'],t2c,optimize=True) # (ab)(ik)
    MM23D -= np.einsum('cmkj,bami->abcijk',vC['vooo'],t2c,optimize=True) # (ac)(ik)
    MM23D -= np.einsum('bmik,acmj->abcijk',vC['vooo'],t2c,optimize=True) # (ab)(jk)
    MM23D -= np.einsum('cmik,bamj->abcijk',vC['vooo'],t2c,optimize=True)    # (ac)(jk)

    MM23D += np.einsum('abie,ecjk->abcijk',vC['vvov'],t2c,optimize=True) # (1)
    MM23D -= np.einsum('abje,ecik->abcijk',vC['vvov'],t2c,optimize=True) # (ij)
    MM23D -= np.einsum('abke,ecji->abcijk',vC['vvov'],t2c,optimize=True) # (ik)
    MM23D -= np.einsum('cbie,eajk->abcijk',vC['vvov'],t2c,optimize=True) # (ac)
    MM23D -= np.einsum('acie,ebjk->abcijk',vC['vvov'],t2c,optimize=True) # (bc)
    MM23D += np.einsum('cbje,eaik->abcijk',vC['vvov'],t2c,optimize=True) # (ac)(ij)
    MM23D += np.einsum('acje,ebik->abcijk',vC['vvov'],t2c,optimize=True) # (bc)(ij)
    MM23D += np.einsum('cbke,eaji->abcijk',vC['vvov'],t2c,optimize=True) # (ac)(ik)
    MM23D += np.einsum('acke,ebji->abcijk',vC['vvov'],t2c,optimize=True) # (bc)(ik)

    return MM23D

