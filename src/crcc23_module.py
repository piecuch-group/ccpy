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
        I2A_vvov = H2A['vvov']+np.einsum('me,abim->abie',H1A['ov'],cc_t['t2a'],optimize=True)
        dA_AAA, dB_AAA, dC_AAA, dD_AAA = crcc_loops.creomcc23a_opt(omega[iroot],cc_t['r0'][iroot],\
                        cc_t['t2a'],cc_t['r2a'][iroot],cc_t['l1a'][iroot+1],cc_t['l2a'][iroot+1],\
                        H2A['vooo'],I2A_vvov,H2A['vvov'],chi2A_vvvo,chi2A_ovoo,ints['vA']['oovv'],H1A['ov'],\
                        H2A['vovv'],H2A['ooov'],fA['oo'],fA['vv'],\
                        H1A['oo'],H1A['vv'],H2A['voov'],H2A['oooo'],H2A['vvvv'],D3A['O'],D3A['V'],\
                        sys['Nocc_a'],sys['Nunocc_a'])

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

        I2B_ovoo = H2B['ovoo'] - np.einsum('me,ecjk->mcjk',H1A['ov'],t2b,optimize=True) 
        I2B_vooo = H2B['vooo'] - np.einsum('me,aeik->amik',H1B['ov'],t2b,optimize=True) 
        I2A_vooo = H2A['vooo'] - np.einsum('me,aeij->amij',H1A['ov'],t2a,optimize=True) 
        dA_AAB, dB_AAB, dC_AAB, dD_AAB = crcc_loops.creomcc23b_opt(omega[iroot],cc_t['r0'][iroot],\
                        cc_t['t2a'],cc_t['t2b'],cc_t['r2a'][iroot],cc_t['r2b'][iroot],\
                        cc_t['l1a'][iroot+1],cc_t['l1b'][iroot+1],cc_t['l2a'][iroot+1],cc_t['l2b'][iroot+1],\
                        I2B_ovoo,I2B_vooo,I2A_vooo,\
                        H2B['vvvo'],H2B['vvov'],H2A['vvov'],\
                        H2B['vovv'],H2B['ovvv'],H2A['vovv'],\
                        H2B['ooov'],H2B['oovo'],H2A['ooov'],\
                        Int1,Int2,Int3,Int4,Int5,Int6,\
                        H2B['ovoo'],H2A['vooo'],H2B['vooo'],\
                        H1A['ov'],H1B['ov'],ints['vA']['oovv'],ints['vB']['oovv'],\
                        fA['oo'],fA['vv'],fB['oo'],fB['vv'],\
                        H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],H2A['voov'],H2A['oooo'],H2A['vvvv'],\
                        H2B['ovov'],H2B['vovo'],H2B['oooo'],H2B['vvvv'],H2C['voov'],\
                        D3A['O'],D3A['V'],D3B['O'],D3B['V'],D3C['O'],D3C['V'],\
                        sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'])

        #MM23B = build_MM23B(cc_t,H1A,H1B,H2A,H2B,sys)
        #EOMMM23B = build_EOM_MM23B(cc_t,H2A,H2B,H2C,iroot,sys)
        #L3B = build_L3B(cc_t,H1A,H1B,H2A,H2B,ints,sys,iroot=iroot+1)
        #dA_AAB, dB_AAB, dC_AAB, dD_AAB = crcc_loops.crcc23b(EOMMM23B+cc_t['r0'][iroot]*MM23B,L3B,omega[iroot],fA['oo'],fA['vv'],fB['oo'],fB['vv'],\
        #            H1A['oo'],H1A['vv'],H1B['oo'],H1B['vv'],H2A['voov'],H2A['oooo'],H2A['vvvv'],\
        #            H2B['ovov'],H2B['vovo'],H2B['oooo'],H2B['vvvv'],H2C['voov'],\
        #            D3A['O'],D3A['V'],D3B['O'],D3B['V'],D3C['O'],D3C['V'],\
        #            sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'])


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


def build_MM23A(cc_t,H1A,H2A,sys):
    """Calculate the projection <ijkabc|(H_N e^(T1+T2))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD HBar integrals (H_N e^(T1+T2))_C
    sys : dict
        System information dictionary
        
    Returns
    -------
    MM23A : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Array containing the MM(2,3)A projections for each i,j,k,a,b,c
    """ 
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
    """Calculate the projection <ijk~abc~|(H_N e^(T1+T2))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD HBar integrals (H_N e^(T1+T2))_C
    sys : dict
        System information dictionary
        
    Returns
    -------
    MM23B : ndarray(dtype=float, ndim=shape(nua,nua,nob,noa,noa,nob))
        Array containing the MM(2,3)B projections for each i,j,k~,a,b,c~
    """
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
    """Calculate the projection <ij~k~ab~c~|(H_N e^(T1+T2))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD HBar integrals (H_N e^(T1+T2))_C
    sys : dict
        System information dictionary
        
    Returns
    -------
    MM23C : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Array containing the MM(2,3)C projections for each i,j~,k~,a,b~,c~
    """  
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
    """Calculate the projection <i~j~k~a~b~c~|(H_N e^(T1+T2))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD HBar integrals (H_N e^(T1+T2))_C
    sys : dict
        System information dictionary
        
    Returns
    -------
    MM23D : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Array containing the MM(2,3)D projections for each i~,j~,k~,a~,b~,c~
    """  
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

def build_EOM_MM23A(cc_t,H2A,H2B,iroot,sys):
    """Calculate the projection <ijkabc|(H_N e^(T1+T2))_C (R1+R2)|0> in
    a factorized manner.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2 and excitation amplitudes R1 and R2 for each root
    H1*, H2* : dict
        Sliced similarity-transformed CCSD HBar integrals (H_N e^(T1+T2))_C
    iroot : int
        Index specifying the excited state root of interest (iroot = 0 corresponds
        to the first excited state)
    sys : dict
        System information dictionary
        
    Returns
    -------
    EOMMM23A : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Array containing the EOMMM(2,3)A projections for each i,j,k,a,b,c
    """ 
    t2a = cc_t['t2a']
    r1a = cc_t['r1a'][iroot]
    r1b = cc_t['r1b'][iroot]
    r2a = cc_t['r2a'][iroot]
    r2b = cc_t['r2b'][iroot]

    Q1 = np.einsum('mnef,fn->me',H2A['oovv'],r1a,optimize=True)
    Q1 += np.einsum('mnef,fn->me',H2B['oovv'],r1b,optimize=True)

    I1 = np.einsum('amje,bm->abej',H2A['voov'],r1a,optimize=True)
    I1 += np.einsum('amfe,bejm->abfj',H2A['vovv'],r2a,optimize=True)
    I1 += np.einsum('amfe,bejm->abfj',H2B['vovv'],r2b,optimize=True)
    I1 -= np.transpose(I1,(1,0,2,3))
    I2 = np.einsum('abfe,ej->abfj',H2A['vvvv'],r1a,optimize=True)
    I2 += 0.5*np.einsum('nmje,abmn->abej',H2A['ooov'],r2a,optimize=True)
    I2 -= np.einsum('me,abmj->abej',Q1,t2a,optimize=True)
    Int1 = I1 + I2

    I1 = -np.einsum('bmie,ej->mbij',H2A['voov'],r1a,optimize=True)
    I1 += np.einsum('nmie,bejm->nbij',H2A['ooov'],r2a,optimize=True)
    I1 += np.einsum('nmie,bejm->nbij',H2B['ooov'],r2b,optimize=True)
    I1 -= np.transpose(I1,(0,1,3,2))
    I2 = -1.0*np.einsum('nmij,bm->nbij',H2A['oooo'],r1a,optimize=True)
    I2 += 0.5*np.einsum('bmfe,efij->mbij',H2A['vovv'],r2a,optimize=True)
    Int2 = I1 + I2

    A1 = np.einsum('abej,ecik->abcijk',Int1,t2a,optimize=True)
    A1 += np.einsum('baje,ecik->abcijk',H2A['vvov'],r2a,optimize=True)
    A1 -= np.transpose(A1,(2,1,0,3,4,5)) + np.transpose(A1,(0,2,1,3,4,5))
    A1 -= np.transpose(A1,(0,1,2,4,3,5)) + np.transpose(A1,(0,1,2,3,5,4))

    A2 = -1.0*np.einsum('mbij,acmk->abcijk',Int2,t2a,optimize=True)
    A2 -= np.einsum('bmji,acmk->abcijk',H2A['vooo'],r2a,optimize=True)
    A2 -= np.transpose(A2,(1,0,2,3,4,5)) + np.transpose(A2,(0,2,1,3,4,5))
    A2 -= np.transpose(A2,(0,1,2,5,4,3)) + np.transpose(A2,(0,1,2,3,5,4))

    return A1 + A2


def build_EOM_MM23B(cc_t,H2A,H2B,H2C,iroot,sys):
    """Calculate the projection <ijk~abc~|(H_N e^(T1+T2))_C (R1+R2)|0>
    in a factorized manner.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2 and excitation amplitudes R1 and R2 for each root
    H1*, H2* : dict
        Sliced similarity-transformed CCSD HBar integrals (H_N e^(T1+T2))_C
    iroot : int
        Index specifying the excited state root of interest (iroot = 0 corresponds
        to the first excited state)
    sys : dict
        System information dictionary
        
    Returns
    -------
    EOMMM23B : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Array containing the EOMMM(2,3)B projections for each i,j,k~,a,b,c~
    """ 
    t2a = cc_t['t2a']
    t2b = cc_t['t2b']
    r1a = cc_t['r1a'][iroot]
    r1b = cc_t['r1b'][iroot]
    r2a = cc_t['r2a'][iroot]
    r2b = cc_t['r2b'][iroot]
    r2c = cc_t['r2c'][iroot]

    Q1 = np.einsum('mnef,fn->me',H2A['oovv'],r1a,optimize=True)+np.einsum('mnef,fn->me',H2B['oovv'],r1b,optimize=True)
    Q2 = np.einsum('nmfe,fn->me',H2B['oovv'],r1a,optimize=True)+np.einsum('nmfe,fn->me',H2C['oovv'],r1b,optimize=True)
    # Intermediate 1: X2B(bcek)*Y2A(aeij) -> Z3B(abcijk)
    Int1 = -1.0*np.einsum('mcek,bm->bcek',H2B['ovvo'],r1a,optimize=True)
    Int1 -= np.einsum('bmek,cm->bcek',H2B['vovo'],r1b,optimize=True)
    Int1 += np.einsum('bcfe,ek->bcfk',H2B['vvvv'],r1b,optimize=True)
    Int1 += np.einsum('mnek,bcmn->bcek',H2B['oovo'],r2b,optimize=True)
    Int1 += np.einsum('bmfe,ecmk->bcfk',H2A['vovv'],r2b,optimize=True)
    Int1 += np.einsum('bmfe,ecmk->bcfk',H2B['vovv'],r2c,optimize=True)
    Int1 -= np.einsum('mcfe,bemk->bcfk',H2B['ovvv'],r2b,optimize=True)
    A1 = np.einsum('bcek,aeij->abcijk',Int1,t2a,optimize=True)
    A1 += np.einsum('bcek,aeij->abcijk',H2B['vvvo'],r2a,optimize=True)
    # Intermediate 2: X2B(ncjk)*Y2A(abin) -> Z3B(abcijk)
    Int2 = -1.0*np.einsum('nmjk,cm->ncjk',H2B['oooo'],r1b,optimize=True)
    Int2 += np.einsum('mcje,ek->mcjk',H2B['ovov'],r1b,optimize=True)
    Int2 += np.einsum('mcek,ej->mcjk',H2B['ovvo'],r1a,optimize=True)
    Int2 += np.einsum('mcef,efjk->mcjk',H2B['ovvv'],r2b,optimize=True)
    Int2 += np.einsum('nmje,ecmk->ncjk',H2A['ooov'],r2b,optimize=True)
    Int2 += np.einsum('nmje,ecmk->ncjk',H2B['ooov'],r2c,optimize=True)
    Int2 -= np.einsum('nmek,ecjm->ncjk',H2B['oovo'],r2b,optimize=True)
    A2 = -1.0*np.einsum('ncjk,abin->abcijk',Int2,t2a,optimize=True)
    A2 -= np.einsum('mcjk,abim->abcijk',H2B['ovoo'],r2a,optimize=True)
    # Intermediate 3: X2A(abej)*Y2B(ecik) -> Z3B(abcijk)
    Int3 = np.einsum('amje,bm->abej',H2A['voov'],r1a,optimize=True) #(*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int3 += 0.5*np.einsum('abfe,ej->abfj',H2A['vvvv'],r1a,optimize=True) #(*) added factor 1/2 to compensate A(ab)
    Int3 += 0.25*np.einsum('nmje,abmn->abej',H2A['ooov'],r2a,optimize=True) #(*) added factor 1/2 to compensate A(ab)
    Int3 += np.einsum('amfe,bejm->abfj',H2A['vovv'],r2a,optimize=True)
    Int3 += np.einsum('amfe,bejm->abfj',H2B['vovv'],r2b,optimize=True)
    Int3 -= 0.5*np.einsum('me,abmj->abej',Q1,t2a,optimize=True) #(*) added factor 1/2 to compensate A(ab)
    Int3 -= np.transpose(Int3,(1,0,2,3))
    A3 = np.einsum('abej,ecik->abcijk',Int3,t2b,optimize=True)
    A3 += np.einsum('baje,ecik->abcijk',H2A['vvov'],r2b,optimize=True)
    # Intermediate 4: X2A(bnji)*Y2B(acnk) -> Z3B(abcijk)
    Int4 = -0.5*np.einsum('nmij,bm->bnji',H2A['oooo'],r1a,optimize=True) #(*) added factor 1/2 to compenate A(ij)
    Int4 -= np.einsum('bmie,ej->bmji',H2A['voov'],r1a,optimize=True) #(*) flipped sign to use H2A(voov) instead of H2A(vovo)
    Int4 += 0.25*np.einsum('bmfe,efij->bmji',H2A['vovv'],r2a,optimize=True) #(*) added factor 1/2 to compensate A(ij)
    Int4 += np.einsum('nmie,bejm->bnji',H2A['ooov'],r2a,optimize=True)
    Int4 += np.einsum('nmie,bejm->bnji',H2B['ooov'],r2b,optimize=True)
    Int4 += 0.5*np.einsum('me,ebij->bmji',Q1,t2a,optimize=True) # (*) added factor 1/2 to compensate A(ij)
    Int4 -= np.transpose(Int4,(0,1,3,2))
    A4 = -1.0*np.einsum('bnji,acnk->abcijk',Int4,t2b,optimize=True)
    A4 -= np.einsum('bnji,acnk->abcijk',H2A['vooo'],r2b,optimize=True)
    # Intermediate 5: X2B(bcje)*Y2B(aeik) -> Z3B(abcijk)
    Int5 = -1.0*np.einsum('mcje,bm->bcje',H2B['ovov'],r1a,optimize=True)
    Int5 -= np.einsum('bmje,cm->bcje',H2B['voov'],r1b,optimize=True)
    Int5 += np.einsum('bcef,ej->bcjf',H2B['vvvv'],r1a,optimize=True)
    Int5 += np.einsum('mnjf,bcmn->bcjf',H2B['ooov'],r2b,optimize=True)
    Int5 += np.einsum('mcef,bejm->bcjf',H2B['ovvv'],r2a,optimize=True)
    Int5 += np.einsum('cmfe,bejm->bcjf',H2C['vovv'],r2b,optimize=True)
    Int5 -= np.einsum('bmef,ecjm->bcjf',H2B['vovv'],r2b,optimize=True)
    A5 = np.einsum('bcje,aeik->abcijk',Int5,t2b,optimize=True)
    A5 += np.einsum('bcje,aeik->abcijk',H2B['vvov'],r2b,optimize=True)
    # Intermediate 6: X2B(bnjk)*Y2B(acin) -> Z3B(abcijk)
    Int6 = -1.0*np.einsum('mnjk,bm->bnjk',H2B['oooo'],r1a,optimize=True)
    Int6 += np.einsum('bmje,ek->bmjk',H2B['voov'],r1b,optimize=True)
    Int6 += np.einsum('bmek,ej->bmjk',H2B['vovo'],r1a,optimize=True)
    Int6 += np.einsum('bnef,efjk->bnjk',H2B['vovv'],r2b,optimize=True)
    Int6 += np.einsum('mnek,bejm->bnjk',H2B['oovo'],r2a,optimize=True)
    Int6 += np.einsum('nmke,bejm->bnjk',H2C['ooov'],r2b,optimize=True)
    Int6 -= np.einsum('nmje,benk->bmjk',H2B['ooov'],r2b,optimize=True)
    Int6 += np.einsum('me,bejk->bmjk',Q2,t2b,optimize=True)
    A6 = -1.0*np.einsum('bnjk,acin->abcijk',Int6,t2b,optimize=True)
    A6 -= np.einsum('bnjk,acin->abcijk',H2B['vooo'],r2b,optimize=True)

    D_ij = A2 + A3
    D_ij -= np.transpose(D_ij,(0,1,2,4,3,5))
    D_ab = A1 + A4
    D_ab -= np.transpose(D_ab,(1,0,2,3,4,5))
    D_ij_ab = A5 + A6
    D_ij_ab -= np.transpose(D_ij_ab,(1,0,2,3,4,5))
    D_ij_ab -= np.transpose(D_ij_ab,(0,1,2,4,3,5))

    return D_ij + D_ab + D_ij_ab


def build_EOM_MM23C(cc_t,H2A,H2B,H2C,iroot,sys):
    """Calculate the projection <ij~k~ab~c~|[ (H_N e^(T1+T2))_C (R1+R2) ]_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2 and excitation amplitudes R1 and R2 for each root
    H1*, H2* : dict
        Sliced similarity-transformed CCSD HBar integrals (H_N e^(T1+T2))_C
    iroot : int
        Index specifying the excited state root of interest (iroot = 0 corresponds
        to the first excited state)
    sys : dict
        System information dictionary
        
    Returns
    -------
    EOMMM23C : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Array containing the EOMMM(2,3)C projections for each i,j~,k~,a,b~,c~
    """ 

    t2b = cc_t['t2b']
    t2c = cc_t['t2c']
    r1a = cc_t['r1a'][iroot]
    r1b = cc_t['r1b'][iroot]
    r2a = cc_t['r2a'][iroot]
    r2b = cc_t['r2b'][iroot]
    r2c = cc_t['r2c'][iroot]

    D_jk = 0.0
    D_bc = 0.0
    D_jk_bc = 0.0
    # diagram 1
    I1 = -1.0*np.einsum('mbie,am->abie',H2B['ovov'],r1a,optimize=True)
    I2 = -1.0*np.einsum('mnij,am->anij',H2B['oooo'],r1a,optimize=True)
    I3 = -1.0*np.einsum('mbej,am->abej',H2B['ovvo'],r1a,optimize=True)
    A1 = np.einsum('abie,ecjk->abcijk',I1,t2c,optimize=True)
    D_bc += A1
    A2 = -1.0*np.einsum('anij,bcnk->abcijk',I2,t2c,optimize=True)
    D_jk += A2
    A3 = np.einsum('abej,ecik,->abcijk',I3,t2b,optimize=True)
    D_jk_bc += A3
    # diagram 2
    I1 = -1.0*np.einsum('amej,bm->abej',H2B['vovo'],r1b,optimize=True)
    I2 = -1.0*np.einsum('amie,bm->abie',H2B['voov'],r1b,optimize=True)
    I3 = -1.0*np.einsum('cmke,bm->bcek',H2C['voov'],r1b,optimize=True)
    I3 -= np.transpose(I3,(1,0,2,3))
    I4 = -1.0*np.einsum('mnjk,bm->bnjk',H2C['oooo'],r1b,optimize=True)
    I5 = -1.0*np.einsum('nmij,bm->nbij',H2B['oooo'],r1b,optimize=True)
    A1 = np.einsum('abej,ecik->abcijk',I1,t2b,optimize=True)
    D_jk_bc += A1
    A2 = np.einsum('abie,ecjk->abcijk',I2,t2c,optimize=True)
    D_bc += A2
    A3 = np.einsum('bcek,aeij->abcijk',I3,t2b,optimize=True)
    D_jk += A3
    A4 = -1.0*np.einsum('bnjk,acin->abcijk',I4,t2b,optimize=True)
    D_bc += A4
    A5 = -1.0*np.einsum('nbij,acnk->abcijk',I5,t2b,optimize=True)
    D_jk_bc += A5
    # diagram 3
    I1 = np.einsum('amej,ei->amij',H2B['vovo'],r1a,optimize=True)
    A1 = -1.0*np.einsum('amij,bcmk->abcijk',I1,t2c,optimize=True)
    D_jk += A1
    I2 = np.einsum('abef,ei->abif',H2B['vvvv'],r1a,optimize=True)
    A2 = np.einsum('abif,fcjk->abcijk',I2,t2c,optimize=True)
    D_bc += A2
    I3 = np.einsum('mbej,ei->mbij',H2B['ovvo'],r1a,optimize=True)
    A3 = -1.0*np.einsum('mbij,acmk->abcijk',I3,t2b,optimize=True)
    D_jk_bc += A3
    # diagram 4
    I1 = np.einsum('mbie,ej->mbij',H2B['ovov'],r1b,optimize=True)
    A1 = -1.0*np.einsum('mbij,acmk->abcijk',I1,t2b,optimize=True)
    D_jk_bc += A1
    I2 = -1.0*np.einsum('bmke,ej->bmkj',H2C['voov'],r1b,optimize=True)
    I2 -= np.transpose(I1,(0,1,3,2))
    A2 = -1.0*np.einsum('bmkj,acim->abcijk',I2,t2b,optimize=True)
    D_bc += A2
    I3 = np.einsum('amie,ej->amij',H2B['voov'],r1b,optimize=True)
    A3 = -1.0*np.einsum('amij,bcmk->abcijk',I3,t2c,optimize=True)
    D_jk += A3
    I4 = np.einsum('bcef,ej->bcjf',H2C['vvvv'],r1b,optimize=True)
    A4 = np.einsum('bcjf,afik->abcijk',I4,t2b,optimize=True)
    D_jk += A4
    I5 = np.einsum('abfe,ej->abfj',H2B['vvvv'],r1b,optimize=True)
    A5 = np.einsum('abfj,fcik->abcijk',I5,t2b,optimize=True)
    D_jk_bc += A5
    # diagram 5
    I1 = 0.5*np.einsum('mnje,bcmn->bcje',H2C['ooov'],r2c,optimize=True)
    A1 = np.einsum('bcje,aeik->abcijk',I1,t2b,optimize=True)
    D_jk += A1
    # diagram 6
    I1 = np.einsum('mnie,abmn->abie',H2B['ooov'],r2b,optimize=True)
    A1 = np.einsum('abie,ecjk->abcijk',I1,t2c,optimize=True)
    D_bc += A1
    I2 = np.einsum('mnej,abmn->abej',H2B['oovo'],r2b,optimize=True)
    A2 = np.einsum('abej,ecik->abcijk',I2,t2b,optimize=True)
    D_jk_bc += A2
    # diagram 7
    I1 = 0.5*np.einsum('mbfe,efjk->mbkj',H2B['ovvv'],r2c,optimize=True)
    A1 = -1.0*np.einsum('mbkj,acim->abcijk',I1,t2b,optimize=True)
    D_bc += A1
    # diagram 8
    I1 = np.einsum('amef,efij->amij',H2B['vovv'],r2b,optimize=True)
    A1 = -1.0*np.einsum('amij,bcmk->abcijk',I1,t2c,optimize=True)
    D_jk += A1
    I2 = np.einsum('mbef,efij->mbij',H2B['ovvv'],r2b,optimize=True)
    A2 = -1.0*np.einsum('mbij,acmk->abcijk',I2,t2b,optimize=True)
    D_jk_bc += A2
    # diagram 9
    I1 = np.einsum('mnej,aeim->anij',H2B['oovo'],r2a,optimize=True)
    A1 = -1.0*np.einsum('anij,bcnk->abcijk',I1,t2c,optimize=True)
    D_jk += A1
    I2 = np.einsum('mbef,aeim->abif',H2B['ovvv'],r2a,optimize=True)
    A2 = np.einsum('abif,fcjk->abcijk',I2,t2c,optimize=True)
    D_bc += A2
    # diagram 10
    I1 = np.einsum('mnej,aeim->anij',H2C['oovo'],r2b,optimize=True)
    A1 = -1.0*np.einsum('anij,bcnk->abcijk',I1,t2c,optimize=True)
    D_jk += A1
    I2 = np.einsum('bmfe,aeim->abif',H2C['vovv'],r2b,optimize=True)
    A2 = np.einsum('abif,fcjk->abcijk',I2,t2c,optimize=True)
    D_bc += A2
    # diagram 11
    I1 = np.einsum('amfe,ecmk->acfk',H2A['vovv'],r2b,optimize=True)
    A1 = np.einsum('acfk,fbij->abcijk',I1,t2b,optimize=True)
    D_jk_bc += A1
    I2 = np.einsum('mbef,ecmk->bcfk',H2B['ovvv'],r2b,optimize=True)
    I2 -= np.transpose(I2,(1,0,3,2))
    A2 = np.einsum('bcfk,afij->abcijk',I2,t2b,optimize=True)
    D_jk += A2
    I3 = np.einsum('mnei,ecmk->ncik',H2A['oovo'],r2b,optmize=True)
    A3 = -1.0*np.einsum('ncik,abnj->abcijk',I3,t2b,optimize=True)
    D_jk_bc += A3
    I4 = np.einsum('mnej,ecmk->cnkj',H2B['oovo'],r2b,optimize=True)
    I4 -= np.transpose(I4,(0,1,3,2))
    A4 = -1.0*np.einsum('cnkj,abin->abcijk',I4,t2b,optimize=True)
    D_bc += A4
    # diagram 12
    I1 = np.einsum('amfe,cekm->acfk',H2B['vovv'],r2c,optimize=True)
    A1 = np.einsum('acfk,fbij->abcijk',I1,t2b,optimize=True)
    D_jk_bc += A1
    I2 = np.einsum('bmfe,cekm->bcfk',H2C['vovv'],r2c,optimize=True)
    I2 -= np.transpose(I2,(1,0,2,3))
    A2 = np.einsum('bcfk,afij->abcijk',I2,t2b,optimize=True)
    D_jk += A2
    I3 = np.einsum('nmie,cekm->ncik',H2B['ooov'],r2c,optimize=True)
    A3 = -1.0*np.einsum('ncik,abnj->abcijk',I3,t2b,optimize=True)
    D_jk_bc += A3
    I4 = np.einsum('nmje,cekm->ncjk',H2C['ooov'],r2c,optimize=True)
    I4 -= np.transpose(I4,(0,1,3,2))
    A4 = -1.0*np.einsum('ncjk,abin->abcijk',I4,t2b,optimize=True)
    D_bc += A4
    # diagram 13
    I1 = -1.0*np.einsum('mbfe,aemj->abfj',H2B['ovvv'],r2b,optimize=True)
    A1 = np.einsum('abfj,fcik->abcijk',I1,t2b,optimize=True)
    D_jk_bc += A1
    I2 = -1.0*np.einsum('mnie,aemj->anij',H2B['ooov'],r2b,optimize=True)
    A2 = -1.0*np.einsum('anij,bcnk->abcijk',I2,t2c,optimize=True)
    D_jk += A2
    # diagram 14
    I1 = -1.0*np.einsum('nmej,ebim->nbij',H2B['oovo'],r2b,optimize=True)
    A1 = -1.0*np.einsum('nbij,acnk->abcijk',I1,t2b,optimize=True)
    D_jk_bc += A1
    I2 = -1.0*np.einsum('amef,ebim->abif',H2B['vovv'],r2b,optimize=True)
    A2 = np.einsum('abif,fcjk->abcijk',I2,t2c,optimize=True)
    D_bc += A2
    # diagram 15
    D_jk_bc -= np.einsum('mcik,abmj->abcijk',H2B['ovoo'],r2b,optimize=True)
    # diagram 16
    D_jk -= np.einsum('amij,bcmk->abcijk',H2B['vooo'],r2c,optimize=True)
    # diagram 17
    D_bc -= np.einsum('cmkj,abim->abcijk',H2C['vooo'],r2b,optimize=True)
    # diagram 18
    D_jk_bc += np.einsum('acek,ebij->abcijk',H2B['vvvo'],r2b,optimize=True)
    # diagram 19
    D_bc += np.einsum('abie,ecjk->abcijk',H2B['vvov'],r2c,optimize=True)
    # diagram 20
    D_jk += np.einsum('cbke,aeij->abcijk',H2C['vvov'],r2b,optimize=True)
    # diagram 21 - 23
    I1A = np.einsum('mnef,fn->me',vA['oovv'],r1a,optimize=True)\
                    +np.einsum('mnef,fn->me',vB['oovv'],r1b,optimize=True)
    I1B = np.einsum('nmfe,fn->me',vC['oovv'],r1b,optimize=True)\
                    +np.einsum('nmfe,fn->me',vB['oovv'],r1a,optimize=True)
    D_bc -= np.einsum('me,ecjk,abim->abcijk',I1B,t2c,t2b,optimize=True)
    D_jk -= np.einsum('me,bcmk,aeij->abcijk',I1B,t2c,t2b,optimize=True)
    D_jk_bc -= np.einsum('me,ecik,abmj->abcijk',I1A,t2b,t2b,optimize=True)
     
    D_jk -= np.transpose(D_jk,(0,1,2,3,5,4))
    D_bc -= np.transpose(D_bc,(0,2,1,3,4,5))
    D_jk_bc -= np.transpose(D_jk_bc,(0,2,1,3,4,5)) + np.transpose(D_jk_bc,(0,1,2,3,5,4)) - np.transpose(D_jk_bc,(0,2,1,3,5,4))

    return D_jk + D_bc + D_jk_bc

def build_L3A(cc_t,H1A,H2A,ints,sys,iroot=0):
    """Calculate the projection <0|(L1+L2)(H_N e^(T1+T2))_C|ijkabc>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2 and left amplitudes L1, L2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD HBar integrals (H_N e^(T1+T2))_C
    ints : dict
        Sliced F_N and V_N integrals comprising the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    iroot : int, optional
        Integer of the excited-state (iroot > 0) or ground-state (iroot = 0) root.
        Default is iroot = 0 corresponding to the ground-state calculation.
        
    Returns
    -------
    L3A : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Array containing the L3A projections for each i,j,k,a,b,c
    """ 
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
    """Calculate the projection <0|(L1+L2)(H_N e^(T1+T2))_C|ijk~abc~>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2 and left amplitudes L1, L2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD HBar integrals (H_N e^(T1+T2))_C
    ints : dict
        Sliced F_N and V_N integrals comprising the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    iroot : int, optional
        Integer of the excited-state (iroot > 0) or ground-state (iroot = 0) root.
        Default is iroot = 0 corresponding to the ground-state calculation.
        
    Returns
    -------
    L3B : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Array containing the L3B projections for each i,j,k~,a,b,c~
    """ 
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
    """Calculate the projection <0|(L1+L2)(H_N e^(T1+T2))_C|ij~k~ab~c~>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2 and left amplitudes L1, L2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD HBar integrals (H_N e^(T1+T2))_C
    ints : dict
        Sliced F_N and V_N integrals comprising the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    iroot : int, optional
        Integer of the excited-state (iroot > 0) or ground-state (iroot = 0) root.
        Default is iroot = 0 corresponding to the ground-state calculation.
        
    Returns
    -------
    L3C : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Array containing the L3C projections for each i,j~,k~,a,b~,c~
    """ 
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
    """Calculate the projection <0|(L1+L2)(H_N e^(T1+T2))_C|i~j~k~a~b~c~>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2 and left amplitudes L1, L2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD HBar integrals (H_N e^(T1+T2))_C
    ints : dict
        Sliced F_N and V_N integrals comprising the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    iroot : int, optional
        Integer of the excited-state (iroot > 0) or ground-state (iroot = 0) root.
        Default is iroot = 0 corresponding to the ground-state calculation.
        
    Returns
    -------
    L3D : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Array containing the L3D projections for each i~,j~,k~,a~,b~,c~
    """ 
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

