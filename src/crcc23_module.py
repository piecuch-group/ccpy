"""Module containing functions that calculate the CR-CC(2,3) and CR-EOMCC(2,3)
triples corrections to the ground- and excited-state energetics obtained at the CCSD
and EOMCCSD levels, respectively.
Note: For CR-EOMCC(2,3), closed-shell RHF symmetry is used because C and D projections
      are not put in yet."""
import numpy as np
from cc_energy import calc_cc_energy
import time
from crcc_loops import crcc_loops

def crcc23(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF=False,nroot=0,omega=0.0):
    """Calculate the ground-/excited-state CR-CC(2,3)/CR-EOMCC(2,3) corrections
    to the CCSD/EOMCCSD energetics. 

    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2 for ground state and excitation amplitudes
        R1, R2 for each excited state and left amplitudes L1, L2 for both ground
        and excited states
    H1*, H2* : dict
        Sliced CCSD similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals
    sys : dict
        System information dictionary
    flag_RHF : bool, optional
        Flag used to determine whether closed-shell RHF symmetry should be used.
        Default value is False.
    nroot : int, optional
        Number of roots for which to perform the CR-EOMCC(2,3) correction. Default is 0,
        corresponding to only performing the ground-state correction.
    omega : ndarray(dtype=float, shape=(nroot))
        Vector of EOMCCSD excitation energies for each root

    Returns
    -------
    Ecrcc23 : dict
        The total energies resulting from the CR-CC(2,3) calculation for the ground state
        and excited states. Contains all variants (A-D) based on the choice of perturbative denominator
    delta23 : dict
        The corresponding CR-CC(2,3) corrections for the ground- and excited-state energetics. Contains all variants (A-D) based on the choice of perturbative denominator
    """
    print('\n==================================++Entering CR-CC(2,3) Routine++=============================')
    print('Performing correction for ground state')
    t_start = time.time()

    if nroot == 0:
        Ecrcc23 = [None]
        delta23 = [None]
    else:
        Ecrcc23 = [None]*(nroot+1)
        delta23 = [None]*(nroot+1)

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

    I2A_vvov = H2A['vvov']+np.einsum('me,abim->abie',H1A['ov'],cc_t['t2a'],optimize=True)
    dA_AAA, dB_AAA, dC_AAA, dD_AAA = crcc_loops.crcc23a_opt(cc_t['t2a'],cc_t['l1a'][0],cc_t['l2a'][0],\
                    H2A['vooo'],I2A_vvov,ints['vA']['oovv'],H1A['ov'],H2A['vovv'],H2A['ooov'],fA['oo'],fA['vv'],\
                    H1A['oo'],H1A['vv'],H2A['voov'],H2A['oooo'],H2A['vvvv'],D3A['O'],D3A['V'],\
                    sys['Nocc_a'],sys['Nunocc_a'])

    I2B_ovoo = H2B['ovoo'] - np.einsum('me,ecjk->mcjk',H1A['ov'],t2b,optimize=True) 
    I2B_vooo = H2B['vooo'] - np.einsum('me,aeik->amik',H1B['ov'],t2b,optimize=True) 
    I2A_vooo = H2A['vooo'] - np.einsum('me,aeij->amij',H1A['ov'],t2a,optimize=True) 
    dA_AAB, dB_AAB, dC_AAB, dD_AAB = crcc_loops.crcc23b_opt(cc_t['t2a'],cc_t['t2b'],cc_t['l1a'][0],cc_t['l1b'][0],\
                    cc_t['l2a'][0],cc_t['l2b'][0],I2B_ovoo,I2B_vooo,I2A_vooo,\
                    H2B['vvvo'],H2B['vvov'],H2A['vvov'],\
                    H2B['vovv'],H2B['ovvv'],H2A['vovv'],\
                    H2B['ooov'],H2B['oovo'],H2A['ooov'],\
                    H1A['ov'],H1B['ov'],ints['vA']['oovv'],ints['vB']['oovv'],\
                    fA['oo'],fA['vv'],fB['oo'],fB['vv'],\
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
        I2B_vooo = H2B['vooo'] - np.einsum('me,aeij->amij',H1B['ov'],t2b,optimize=True)
        I2C_vooo = H2C['vooo'] - np.einsum('me,cekj->cmkj',H1B['ov'],t2c,optimize=True)
        I2B_ovoo = H2B['ovoo'] - np.einsum('me,ebij->mbij',H1A['ov'],t2b,optimize=True)
        dA_ABB, dB_ABB, dC_ABB, dD_ABB = crcc_loops.crcc23c_opt(cc_t['t2b'],cc_t['t2c'],\
                    cc_t['l1a'][0],cc_t['l1b'][0],cc_t['l2b'][0],cc_t['l2c'][0],\
                    I2B_vooo,I2C_vooo,I2B_ovoo,H2B['vvov'],H2C['vvov'],H2B['vvvo'],\
                    H2B['ovvv'],H2B['vovv'],H2C['vovv'],H2B['oovo'],H2B['ooov'],H2C['ooov'],\
                    H1A['ov'],H1B['ov'],ints['vB']['oovv'],ints['vC']['oovv'],\
                    fA['oo'],fA['vv'],fB['oo'],fB['vv'],\
                    H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],H2A['voov'],\
                    H2B['ovov'],H2B['vovo'],H2B['oooo'],H2B['vvvv'],H2C['voov'],H2C['oooo'],H2C['vvvv'],\
                    D3B['O'],D3B['V'],D3C['O'],D3C['V'],D3D['O'],D3D['V'],\
                    sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'])
        
        I2C_vvov = H2C['vvov']+np.einsum('me,abim->abie',H1B['ov'],t2c,optimize=True)
        dA_BBB, dB_BBB, dC_BBB, dD_BBB = crcc_loops.crcc23d_opt(cc_t['t2c'],cc_t['l1b'][0],cc_t['l2c'][0],\
                    H2C['vooo'],I2C_vvov,ints['vC']['oovv'],H1B['ov'],H2C['vovv'],H2C['ooov'],fB['oo'],fB['vv'],\
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
 
    print('CCSD = {} Eh'.format(ints['Escf']+Ecorr))
    print('CR-CC(2,3)_A = {} Eh     Ecorr_A = {} Eh     Delta_A = {} Eh'.format(E23A,EcorrA,deltaA))
    print('CR-CC(2,3)_B = {} Eh     Ecorr_B = {} Eh     Delta_B = {} Eh'.format(E23B,EcorrB,deltaB))
    print('CR-CC(2,3)_C = {} Eh     Ecorr_C = {} Eh     Delta_C = {} Eh'.format(E23C,EcorrC,deltaC))
    print('CR-CC(2,3)_D = {} Eh     Ecorr_D = {} Eh     Delta_D = {} Eh'.format(E23D,EcorrD,deltaD))

    Ecrcc23[0] = {'A' : E23A, 'B' : E23B , 'C' : E23C, 'D' : E23D}
    delta23[0] = {'A' : deltaA, 'B' : deltaB, 'C' : deltaC, 'D' : deltaD}

    t_end = time.time()
    minutes, seconds = divmod(t_end-t_start, 60)
    print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))

    for iroot in range(nroot):
        print('Performing correction for root {}'.format(iroot+1))
        t_start = time.time()
        
        # correction containers
        deltaA = 0.0
        deltaB = 0.0
        deltaC = 0.0
        deltaD = 0.0

        chi2A_vvvo, chi2A_ovoo = calc_eomm23a_intermediates(cc_t,H2A,H2B,iroot)
        I2A_vvov = H2A['vvov']+np.einsum('me,abim->abie',H1A['ov'],cc_t['t2a'],optimize=True)
        dA_AAA, dB_AAA, dC_AAA, dD_AAA = crcc_loops.creomcc23a_opt(omega[iroot],cc_t['r0'][iroot],\
                        cc_t['t2a'],cc_t['r2a'][iroot],cc_t['l1a'][iroot+1],cc_t['l2a'][iroot+1],\
                        H2A['vooo'],I2A_vvov,H2A['vvov'],chi2A_vvvo,chi2A_ovoo,ints['vA']['oovv'],H1A['ov'],\
                        H2A['vovv'],H2A['ooov'],fA['oo'],fA['vv'],\
                        H1A['oo'],H1A['vv'],H2A['voov'],H2A['oooo'],H2A['vvvv'],D3A['O'],D3A['V'],\
                        sys['Nocc_a'],sys['Nunocc_a'])

        I2B_ovoo = H2B['ovoo'] - np.einsum('me,ecjk->mcjk',H1A['ov'],t2b,optimize=True) 
        I2B_vooo = H2B['vooo'] - np.einsum('me,aeik->amik',H1B['ov'],t2b,optimize=True) 
        I2A_vooo = H2A['vooo'] - np.einsum('me,aeij->amij',H1A['ov'],t2a,optimize=True) 
        chi2B_vvvo, chi2B_ovoo, chi2A_vvvo, chi2A_vooo, chi2B_vvov, chi2B_vooo = calc_eomcc23b_intermediates(cc_t,H2A,H2B,H2C,iroot)
        dA_AAB, dB_AAB, dC_AAB, dD_AAB = crcc_loops.creomcc23b_opt(omega[iroot],cc_t['r0'][iroot],\
                        cc_t['t2a'],cc_t['t2b'],cc_t['r2a'][iroot],cc_t['r2b'][iroot],\
                        cc_t['l1a'][iroot+1],cc_t['l1b'][iroot+1],cc_t['l2a'][iroot+1],cc_t['l2b'][iroot+1],\
                        I2B_ovoo,I2B_vooo,I2A_vooo,\
                        H2B['vvvo'],H2B['vvov'],H2A['vvov'],\
                        H2B['vovv'],H2B['ovvv'],H2A['vovv'],\
                        H2B['ooov'],H2B['oovo'],H2A['ooov'],\
                        chi2B_vvvo,chi2B_ovoo,chi2A_vvvo,\
                        chi2A_vooo,chi2B_vvov,chi2B_vooo,\
                        H2B['ovoo'],H2A['vooo'],H2B['vooo'],\
                        H1A['ov'],H1B['ov'],ints['vA']['oovv'],ints['vB']['oovv'],\
                        fA['oo'],fA['vv'],fB['oo'],fB['vv'],\
                        H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],H2A['voov'],H2A['oooo'],H2A['vvvv'],\
                        H2B['ovov'],H2B['vovo'],H2B['oooo'],H2B['vvvv'],H2C['voov'],\
                        D3A['O'],D3A['V'],D3B['O'],D3B['V'],D3C['O'],D3C['V'],\
                        sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'])

        #if flag_RHF:
        deltaA = 2.0*dA_AAA + 2.0*dA_AAB
        deltaB = 2.0*dB_AAA + 2.0*dB_AAB
        deltaC = 2.0*dC_AAA + 2.0*dC_AAB
        deltaD = 2.0*dD_AAA + 2.0*dD_AAB

        EcorrA = Ecorr + omega[iroot] + deltaA
        EcorrB = Ecorr + omega[iroot] + deltaB
        EcorrC = Ecorr + omega[iroot] + deltaC
        EcorrD = Ecorr + omega[iroot] + deltaD

        E23A = ints['Escf'] + EcorrA; VEE_A = (E23A - Ecrcc23[0]['A'])*27.211396641308;
        E23B = ints['Escf'] + EcorrB; VEE_B = (E23B - Ecrcc23[0]['B'])*27.211396641308;
        E23C = ints['Escf'] + EcorrC; VEE_C = (E23C - Ecrcc23[0]['C'])*27.211396641308;
        E23D = ints['Escf'] + EcorrD; VEE_D = (E23D - Ecrcc23[0]['D'])*27.211396641308;

        print('EOMCCSD = {} Eh'.format(ints['Escf']+Ecorr+omega[iroot]))
        print('CR-CC(2,3)_A = {} Eh     Ecorr_A = {} Eh     Delta_A = {} Eh     VEE = {} eV'.format(E23A,EcorrA,deltaA,VEE_A))
        print('CR-CC(2,3)_B = {} Eh     Ecorr_B = {} Eh     Delta_B = {} Eh     VEE = {} eV'.format(E23B,EcorrB,deltaB,VEE_B))
        print('CR-CC(2,3)_C = {} Eh     Ecorr_C = {} Eh     Delta_C = {} Eh     VEE = {} eV'.format(E23C,EcorrC,deltaC,VEE_C))
        print('CR-CC(2,3)_D = {} Eh     Ecorr_D = {} Eh     Delta_D = {} Eh     VEE = {} eV'.format(E23D,EcorrD,deltaD,VEE_D))

        Ecrcc23[iroot+1] = {'A' : E23A, 'B' : E23B, 'C' : E23C, 'D' : E23D}
        delta23[iroot+1] = {'A' : deltaA, 'B' : deltaB, 'C' : deltaC, 'D' : deltaD}

        t_end = time.time()
        minutes, seconds = divmod(t_end-t_start, 60)
        print('finished in ({:0.2f}m  {:0.2f}s)'.format(minutes,seconds))

    return Ecrcc23, delta23

def calc_eomm23a_intermediates(cc_t,H2A,H2B,iroot):

    Q1 = np.einsum('mnef,fn->me',H2A['oovv'],cc_t['r1a'][iroot],optimize=True)
    Q1 += np.einsum('mnef,fn->me',H2B['oovv'],cc_t['r1b'][iroot],optimize=True)
    I1 = np.einsum('amje,bm->abej',H2A['voov'],cc_t['r1a'][iroot],optimize=True)
    I1 += np.einsum('amfe,bejm->abfj',H2A['vovv'],cc_t['r2a'][iroot],optimize=True)
    I1 += np.einsum('amfe,bejm->abfj',H2B['vovv'],cc_t['r2b'][iroot],optimize=True)
    I1 -= np.transpose(I1,(1,0,2,3))
    I2 = np.einsum('abfe,ej->abfj',H2A['vvvv'],cc_t['r1a'][iroot],optimize=True)
    I2 += 0.5*np.einsum('nmje,abmn->abej',H2A['ooov'],cc_t['r2a'][iroot],optimize=True)
    I2 -= np.einsum('me,abmj->abej',Q1,cc_t['t2a'],optimize=True)
    chi2A_vvvo = I1 + I2
    I1 = -np.einsum('bmie,ej->mbij',H2A['voov'],cc_t['r1a'][iroot],optimize=True)
    I1 += np.einsum('nmie,bejm->nbij',H2A['ooov'],cc_t['r2a'][iroot],optimize=True)
    I1 += np.einsum('nmie,bejm->nbij',H2B['ooov'],cc_t['r2b'][iroot],optimize=True)
    I1 -= np.transpose(I1,(0,1,3,2))
    I2 = -1.0*np.einsum('nmij,bm->nbij',H2A['oooo'],cc_t['r1a'][iroot],optimize=True)
    I2 += 0.5*np.einsum('bmfe,efij->mbij',H2A['vovv'],cc_t['r2a'][iroot],optimize=True)
    chi2A_ovoo = I1 + I2

    return chi2A_vvvo, chi2A_ovoo

def calc_eomcc23b_intermediates(cc_t,H2A,H2B,H2C,iroot):
    Q1 = np.einsum('mnef,fn->me',H2A['oovv'],cc_t['r1a'][iroot],optimize=True)\
                        +np.einsum('mnef,fn->me',H2B['oovv'],cc_t['r1b'][iroot],optimize=True)
    Q2 = np.einsum('nmfe,fn->me',H2B['oovv'],cc_t['r1a'][iroot],optimize=True)\
                        +np.einsum('nmfe,fn->me',H2C['oovv'],cc_t['r1b'][iroot],optimize=True)
    # Intermediate 1: X2B(bcek)*Y2A(aeij) -> Z3B(abcijk)
    Int1 = -1.0*np.einsum('mcek,bm->bcek',H2B['ovvo'],cc_t['r1a'][iroot],optimize=True)
    Int1 -= np.einsum('bmek,cm->bcek',H2B['vovo'],cc_t['r1b'][iroot],optimize=True)
    Int1 += np.einsum('bcfe,ek->bcfk',H2B['vvvv'],cc_t['r1b'][iroot],optimize=True)
    Int1 += np.einsum('mnek,bcmn->bcek',H2B['oovo'],cc_t['r2b'][iroot],optimize=True)
    Int1 += np.einsum('bmfe,ecmk->bcfk',H2A['vovv'],cc_t['r2b'][iroot],optimize=True)
    Int1 += np.einsum('bmfe,ecmk->bcfk',H2B['vovv'],cc_t['r2c'][iroot],optimize=True)
    Int1 -= np.einsum('mcfe,bemk->bcfk',H2B['ovvv'],cc_t['r2b'][iroot],optimize=True)
    # Intermediate 2: X2B(ncjk)*Y2A(abin) -> Z3B(abcijk)
    Int2 = -1.0*np.einsum('nmjk,cm->ncjk',H2B['oooo'],cc_t['r1b'][iroot],optimize=True)
    Int2 += np.einsum('mcje,ek->mcjk',H2B['ovov'],cc_t['r1b'][iroot],optimize=True)
    Int2 += np.einsum('mcek,ej->mcjk',H2B['ovvo'],cc_t['r1a'][iroot],optimize=True)
    Int2 += np.einsum('mcef,efjk->mcjk',H2B['ovvv'],cc_t['r2b'][iroot],optimize=True)
    Int2 += np.einsum('nmje,ecmk->ncjk',H2A['ooov'],cc_t['r2b'][iroot],optimize=True)
    Int2 += np.einsum('nmje,ecmk->ncjk',H2B['ooov'],cc_t['r2c'][iroot],optimize=True)
    Int2 -= np.einsum('nmek,ecjm->ncjk',H2B['oovo'],cc_t['r2b'][iroot],optimize=True)
    # Intermediate 3: X2A(abej)*Y2B(ecik) -> Z3B(abcijk)
    Int3 = np.einsum('amje,bm->abej',H2A['voov'],cc_t['r1a'][iroot],optimize=True) #(*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int3 += 0.5*np.einsum('abfe,ej->abfj',H2A['vvvv'],cc_t['r1a'][iroot],optimize=True) #(*) added factor 1/2 to compensate A(ab)
    Int3 += 0.25*np.einsum('nmje,abmn->abej',H2A['ooov'],cc_t['r2a'][iroot],optimize=True) #(*) added factor 1/2 to compensate A(ab)
    Int3 += np.einsum('amfe,bejm->abfj',H2A['vovv'],cc_t['r2a'][iroot],optimize=True)
    Int3 += np.einsum('amfe,bejm->abfj',H2B['vovv'],cc_t['r2b'][iroot],optimize=True)
    Int3 -= 0.5*np.einsum('me,abmj->abej',Q1,cc_t['t2a'],optimize=True) #(*) added factor 1/2 to compensate A(ab)
    Int3 -= np.transpose(Int3,(1,0,2,3))
    # Intermediate 4: X2A(bnji)*Y2B(acnk) -> Z3B(abcijk)
    Int4 = -0.5*np.einsum('nmij,bm->bnji',H2A['oooo'],cc_t['r1a'][iroot],optimize=True) #(*) added factor 1/2 to compenate A(ij)
    Int4 -= np.einsum('bmie,ej->bmji',H2A['voov'],cc_t['r1a'][iroot],optimize=True) #(*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int4 += 0.25*np.einsum('bmfe,efij->bmji',H2A['vovv'],cc_t['r2a'][iroot],optimize=True) #(*) added factor 1/2 to compensate A(ij)
    Int4 += np.einsum('nmie,bejm->bnji',H2A['ooov'],cc_t['r2a'][iroot],optimize=True)
    Int4 += np.einsum('nmie,bejm->bnji',H2B['ooov'],cc_t['r2b'][iroot],optimize=True)
    Int4 += 0.5*np.einsum('me,ebij->bmji',Q1,cc_t['t2a'],optimize=True) # (*) added factor 1/2 to compensate A(ij)
    Int4 -= np.transpose(Int4,(0,1,3,2))
    # Intermediate 5: X2B(bcje)*Y2B(aeik) -> Z3B(abcijk)
    Int5 = -1.0*np.einsum('mcje,bm->bcje',H2B['ovov'],cc_t['r1a'][iroot],optimize=True)
    Int5 -= np.einsum('bmje,cm->bcje',H2B['voov'],cc_t['r1b'][iroot],optimize=True)
    Int5 += np.einsum('bcef,ej->bcjf',H2B['vvvv'],cc_t['r1a'][iroot],optimize=True)
    Int5 += np.einsum('mnjf,bcmn->bcjf',H2B['ooov'],cc_t['r2b'][iroot],optimize=True)
    Int5 += np.einsum('mcef,bejm->bcjf',H2B['ovvv'],cc_t['r2a'][iroot],optimize=True)
    Int5 += np.einsum('cmfe,bejm->bcjf',H2C['vovv'],cc_t['r2b'][iroot],optimize=True)
    Int5 -= np.einsum('bmef,ecjm->bcjf',H2B['vovv'],cc_t['r2b'][iroot],optimize=True)
    # Intermediate 6: X2B(bnjk)*Y2B(acin) -> Z3B(abcijk)
    Int6 = -1.0*np.einsum('mnjk,bm->bnjk',H2B['oooo'],cc_t['r1a'][iroot],optimize=True)
    Int6 += np.einsum('bmje,ek->bmjk',H2B['voov'],cc_t['r1b'][iroot],optimize=True)
    Int6 += np.einsum('bmek,ej->bmjk',H2B['vovo'],cc_t['r1a'][iroot],optimize=True)
    Int6 += np.einsum('bnef,efjk->bnjk',H2B['vovv'],cc_t['r2b'][iroot],optimize=True)
    Int6 += np.einsum('mnek,bejm->bnjk',H2B['oovo'],cc_t['r2a'][iroot],optimize=True)
    Int6 += np.einsum('nmke,bejm->bnjk',H2C['ooov'],cc_t['r2b'][iroot],optimize=True)
    Int6 -= np.einsum('nmje,benk->bmjk',H2B['ooov'],cc_t['r2b'][iroot],optimize=True)
    Int6 += np.einsum('me,bejk->bmjk',Q2,cc_t['t2b'],optimize=True)

    return Int1, Int2, Int3, Int4, Int5, Int6

def triples_3body_diagonal(cc_t,ints,sys):
    """Calculate the triples diagonal <ijkabc|H3|ijkabc>, where H3
    is the 3-body component of (H_N e^(T1+T2))_C corresponding to 
    (V_N*T2)_C diagrams.

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
    D3A : dict
        Contains the matrices D3A['O'] (ndarray(dtype=float, shape=(nua,noa,noa)))
        and D3A['V'] (ndarray(dtype=float, shape=(nua,noa,nua)))
    D3B : dict
        Contains the matrices D3B['O'] (ndarray(dtype=float, shape=(nua,noa,nob)))
        and D3B['V'] (ndarray(dtype=float, shape=(nua,noa,nub)))
    D3C : dict
        Contains the matrices D3C['O'] (ndarray(dtype=float, shape=(nub,noa,nob)))
        and D3C['V'] (ndarray(dtype=float, shape=(nua,nob,nub)))
    D3D : dict
        Contains the matrices D3D['O'] (ndarray(dtype=float, shape=(nub,nob,nob)))
        and D3D['V'] (ndarray(dtype=float, shape=(nub,nob,nub)))
    """
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

