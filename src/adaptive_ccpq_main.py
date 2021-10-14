import numpy as np
from adaptive_ccp_module import ccsdt_p     # This is the full CCSDT version
#from adaptive_ccp_module_v2 import ccsdt_p # this is the devectorized version in Python
#from adaptive_ccp_module_v3 import ccsdt_p # this is the Fortran version with OpenMP
from adaptive_ccp3_module import ccp3
from adaptive_ccp3pert_module import ccp3_pertT
from HBar_module import HBar_CCSD
from left_ccsd_module import left_ccsd
from ccsd_module import ccsd
import os

def calc_adaptive_ccpq_norelax(sys,ints,tot_triples,workdir,triples_percentages,ccshift=0.0,lccshift=0.0,isRHF=False,\
                ccmaxit=100,tol=1.0e-08,diis_size=6,flag_save=False):

    print('\n==================================++Entering Adaptive CC(P;Q) Routine++=================================\n')
    print('     UNRELAXED ADAPTIVE CC(P;Q) ALGORITHM')
    print('     TOTAL NUMBER OF TRIPLES: {}'.format(tot_triples))
    print('     REQUESTED %T VALUES = {}'.format(triples_percentages))
    print('     SAVE CLUSTER AMPLITUDES & P SPACES = {}'.format(flag_save))
    print('')

    noa = sys['Nocc_a']
    nob = sys['Nocc_b']
    nua = sys['Nunocc_a']
    nub = sys['Nunocc_b']

    flag_USE_PERT_T = True
    if flag_USE_PERT_T:
        print('>>>USING CCSD(T) CORRECTION<<<')

    num_calcs = len(triples_percentages)

    Eccp = np.zeros(num_calcs)
    Eccpq = np.zeros(num_calcs)

    p_spaces_ccsd = {'A' : np.zeros((nua,nua,nua,noa,noa,noa)),\
                     'B' : np.zeros((nua,nua,nub,noa,noa,nob)),\
                     'C' : np.zeros((nua,nub,nub,noa,nob,nob)),\
                     'D' : np.zeros((nub,nub,nub,nob,nob,nob))}

    # Initial CCSD/CR-CC(2,3) calculation
    cc_t_ccsd, Eccsd = ccsd(sys,ints,shift=ccshift,maxit=ccmaxit,tol=tol,diis_size=diis_size)
    if flag_USE_PERT_T:
        _, mcA_ccsd, mcB_ccsd, mcC_ccsd, mcD_ccsd  = ccp3_pertT(cc_t_ccsd,p_spaces_ccsd,ints,sys,flag_RHF=isRHF)
    else:
        H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t_ccsd,ints,sys)
        cc_t_ccsd = left_ccsd(cc_t_ccsd,H1A,H1B,H2A,H2B,H2C,ints,sys,shift=lccshift,maxit=ccmaxit,tol=tol,diis_size=diis_size)
        _, mcA_ccsd, mcB_ccsd, mcC_ccsd, mcD_ccsd  = ccp3(cc_t_ccsd,p_spaces_ccsd,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF=isRHF)

    for ncnt in range(num_calcs):

        print('Triples Percentage - {}'.format(triples_percentages[ncnt])) 
        if flag_save:
            iterdir = workdir+'/Percentage'+str(triples_percentages[ncnt])
            os.mkdir(iterdir)

        # Get the P space in one shot by adding the specified %T
        print('Selecting {}% of triples out of CCSD'.format(triples_percentages[ncnt]))
        add_1perc = int(tot_triples * 0.01)
        num_triples = int( triples_percentages[ncnt] * add_1perc )

        p_spaces_ccsd = {'A' : np.zeros((nua,nua,nua,noa,noa,noa)),\
                     'B' : np.zeros((nua,nua,nub,noa,noa,nob)),\
                     'C' : np.zeros((nua,nub,nub,noa,nob,nob)),\
                     'D' : np.zeros((nub,nub,nub,nob,nob,nob))}

        p_spaces, num_triples_2 = selection_function(sys,mcA_ccsd,mcB_ccsd,mcC_ccsd,mcD_ccsd,p_spaces_ccsd,num_triples,flag_RHF=isRHF)

        # Update triples count in P space
        num_triples = count_triples_in_P(p_spaces) 

        # Save the current P space to the iteration directory
        if flag_save:
            print('Saving P space to {}'.format(iterdir))
            for key, value in p_spaces.items():
                np.save(iterdir+'/pspace_'+key,value)

        # Solve CC(P) equations in current P space
        cc_t, Eccp[ncnt] = ccsdt_p(sys,ints,p_spaces,shift=ccshift,maxit=ccmaxit,tol=tol,diis_size=diis_size,flag_RHF=isRHF)

        # Save cluster amplitudes to current directory
        if flag_save:
            print('Saving cluster amplitudes to {}'.format(iterdir))
            for key, value in cc_t.items():
                np.save(iterdir+'/'+key,value)

        # Left CC and HBar
        if flag_USE_PERT_T:
            Eccpq[ncnt], _, _, _, _  = ccp3_pertT(cc_t,p_spaces,ints,sys,flag_RHF=isRHF)
        else:
            H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,ints,sys)
            cc_t = left_ccsd(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,shift=lccshift,maxit=ccmaxit,tol=tol,diis_size=diis_size)
            # CC(P;Q) moment correction
            Eccpq[ncnt], _, _, _, _  = ccp3(cc_t,p_spaces,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF=isRHF)

        print('SUMMARY AT TRIPLES PERCENTAGE - {}'.format(triples_percentages[ncnt]))
        print('NUMBER OF TRIPLES IN P SPACE - {}/{}  ({}%)'.format(num_triples,num_triples_2,num_triples/tot_triples*100))
        print('E(P) = {} HARTREE'.format(Eccp[ncnt]))
        print('E(P;Q) = {} HARTREE'.format(Eccpq[ncnt]))

    return Eccp, Eccpq

def calc_adaptive_ccpq(sys,ints,tot_triples,workdir,ccshift=0.0,lccshift=0.0,maxit=10,growth_percentage=1.0,\
                restart_dir=None,niter0=0,isRHF=False,ccmaxit=100,tol=1.0e-08,diis_size=6,flag_save=False):

    print('\n==================================++Entering Adaptive CC(P;Q) Routine++=================================\n')
    print('     REXALED ADAPTIVE CC(P;Q) ALGORITHM')
    print('     TOTAL NUMBER OF TRIPLES: {}'.format(tot_triples))
    print('     REQUESTED %T GROWTH PER ITERATION = {}'.format(growth_percentage))
    print('     NUMBER OF ADAPTIVE CC(P;Q) ITERATIONS = {}'.format(maxit))
    print('     SAVE CLUSTER AMPLITUDES & P SPACES = {}'.format(flag_save))
    print('')

    noa = sys['Nocc_a']
    nob = sys['Nocc_b']
    nua = sys['Nunocc_a']
    nub = sys['Nunocc_b']

    Eccp = np.zeros(maxit)
    Eccpq = np.zeros(maxit)

    flag_USE_PERT_T = True
    if flag_USE_PERT_T:
        print('>>>USING CCSD(T) CORRECTION<<<')

    if restart_dir is not None:
        niter0 = int(restart_dir.split('Iter')[1].strip())

        print('RESTARTING CALCULATION FROM DIRECTORY: ',restart_dir)
        print('    RESULTS OF MACRO ITERATION - {}:'.format(niter0))

        p_spaceA = np.load(restart_dir+'/pspace_A.npy')
        p_spaceB = np.load(restart_dir+'/pspace_B.npy')
        p_spaceC = np.load(restart_dir+'/pspace_C.npy')
        p_spaceD = np.load(restart_dir+'/pspace_D.npy')
        p_spaces = {'A' : p_spaceA, 'B' : p_spaceB, 'C' : p_spaceC, 'D' : p_spaceD}

        num_triples = count_triples_in_P(p_spaces)
        print('    NUMBER OF TRIPLES in P SPACE OF THIS DIRECTORY - {}'.format(num_triples))

        t1a = np.load(restart_dir+'/t1a.npy')
        t1b = np.load(restart_dir+'/t1b.npy')
        t2a = np.load(restart_dir+'/t2a.npy')
        t2b = np.load(restart_dir+'/t2b.npy')
        t2c = np.load(restart_dir+'/t2c.npy')
        t3a = np.load(restart_dir+'/t3a.npy')
        t3b = np.load(restart_dir+'/t3b.npy')
        t3c = np.load(restart_dir+'/t3c.npy')
        t3d = np.load(restart_dir+'/t3d.npy')
        cc_t = {'t1a' : t1a, 't1b' : t1b, 't2a' : t2a, 't2b' : t2b, 't2c' : t2c,\
                't3a' : t3a, 't3b' : t3b, 't3c' : t3c, 't3d' : t3d}

        Ecorr = 0.0
        Ecorr += np.einsum('me,em->',ints['fA']['ov'],t1a,optimize=True)
        Ecorr += np.einsum('me,em->',ints['fB']['ov'],t1b,optimize=True)
        Ecorr += 0.25*np.einsum('mnef,efmn->',ints['vA']['oovv'],t2a,optimize=True)
        Ecorr += np.einsum('mnef,efmn->',ints['vB']['oovv'],t2b,optimize=True)
        Ecorr += 0.25*np.einsum('mnef,efmn->',ints['vC']['oovv'],t2c,optimize=True)
        Ecorr += 0.5*np.einsum('mnef,fn,em->',ints['vA']['oovv'],t1a,t1a,optimize=True)
        Ecorr += 0.5*np.einsum('mnef,fn,em->',ints['vC']['oovv'],t1b,t1b,optimize=True)
        Ecorr += np.einsum('mnef,em,fn->',ints['vB']['oovv'],t1a,t1b,optimize=True)

        Eccp_rest = Ecorr + ints['Escf']
        print('    CC(P) ENERGY = {} HARTREE'.format(Eccp_rest))

        # Calculate CC(P;Q) moment correction using 2BA
        if flag_USE_PERT_T:
            Eccpq_rest, mcA, mcB, mcC, mcD  = ccp3_pertT(cc_t,p_spaces,ints,sys,flag_RHF=isRHF)
        else:
            # Calculate HBar using 2BA
            H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,ints,sys)
            # Solve left CCSD equations using 2BA
            cc_t = left_ccsd(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,shift=lccshift,maxit=ccmaxit,tol=tol,diis_size=diis_size)
            Eccpq_rest, mcA, mcB, mcC, mcD  = ccp3(cc_t,p_spaces,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF=isRHF)

        print('SUMMARY AT ITERATION - {}'.format(niter0))
        print('NUMBER OF TRIPLES IN P SPACE - {}  ({}%)'.format(num_triples,num_triples/tot_triples*100))
        print('E(P) = {} HARTREE'.format(Eccp_rest))
        print('E(P;Q) = {} HARTREE'.format(Eccpq_rest))

        niter0 += 1

    else:
        
        # Initialize empty P space
        p_spaces = {'A' : np.zeros((nua,nua,nua,noa,noa,noa)),\
                    'B' : np.zeros((nua,nua,nub,noa,noa,nob)),\
                    'C' : np.zeros((nua,nub,nub,noa,nob,nob)),\
                    'D' : np.zeros((nub,nub,nub,nob,nob,nob ))}

        num_triples = 0

        # Initial CCSD/CR-CC(2,3) calculation
        cc_t, Eccsd = ccsd(sys,ints,shift=ccshift,maxit=ccmaxit,tol=tol,diis_size=diis_size)
        if flag_USE_PERT_T:
            _, mcA, mcB, mcC, mcD  = ccp3_pertT(cc_t,p_spaces,ints,sys,flag_RHF=isRHF)
        else:
            H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,ints,sys)
            cc_t = left_ccsd(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,shift=lccshift,maxit=ccmaxit,tol=tol,diis_size=diis_size)
            _, mcA, mcB, mcC, mcD  = ccp3(cc_t,p_spaces,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF=isRHF)

        cc_t['t3a'] = np.zeros((nua,nua,nua,noa,noa,noa))
        cc_t['t3b'] = np.zeros((nua,nua,nub,noa,noa,nob))
        cc_t['t3c'] = np.zeros((nua,nub,nub,noa,nob,nob))
        cc_t['t3d'] = np.zeros((nub,nub,nub,nob,nob,nob))

        niter0 += 1

    for niter in range(niter0,niter0+maxit):
        
        print('MACRO ITERATION - {}'.format(niter)) 

        if flag_save:
            iterdir = workdir+'/Iter'+str(niter)
            os.mkdir(iterdir)

        # Update the P space by adding the desired %T based on the existing moment corrections
        print('CONSTRUCTING P SPACE BY SELECTION')
        add_1perc = int(tot_triples * 0.01)
        num_add = int( growth_percentage * add_1perc )
        p_spaces,_ = selection_function(sys,mcA,mcB,mcC,mcD,p_spaces,num_add,flag_RHF=isRHF)

        # Update the count of the number of triples in P space
        num_triples = count_triples_in_P(p_spaces)
        print('THERE ARE {} TRIPLES IN THE P SPACE'.format(num_triples))
        #num_triples += num_add

        # Save P space in current Iteration directory
        if flag_save:
            print('SAVING P SPACE TO DIRECTORY {}'.format(iterdir))
            for key, value in p_spaces.items():
                np.save(iterdir+'/pspace_'+key,value)

        # Solve CC(P) equations in updated P space
        if niter > 0:
            cc_t, Eccp[niter-niter0] = ccsdt_p(sys,ints,p_spaces,maxit=ccmaxit,shift=ccshift,initial_guess=cc_t,flag_RHF=isRHF,\
                            tol=tol,diis_size=diis_size)
        else:
            cc_t, Eccp[niter-niter0] = ccsdt_p(sys,ints,p_spaces,maxit=ccmaxit,shift=ccshift,flag_RHF=isRHF,tol=tol,diis_size=diis_size)

        # Save cluster amplitudes in current Iteration directory
        if flag_save:
            print('SAVING CLUSTER AMPLITUDES TO DIRECTORY {}'.format(iterdir))
            for key, value in cc_t.items():
                np.save(iterdir+'/'+key,value)

        if flag_USE_PERT_T:
            Eccpq[niter-niter0], mcA, mcB, mcC, mcD  = ccp3_pertT(cc_t,p_spaces,ints,sys,flag_RHF=isRHF)
        else:
            # Calculate HBar using 2BA
            H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,ints,sys)
            # Solve left CCSD equations using 2BA
            cc_t = left_ccsd(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,shift=lccshift,maxit=ccmaxit,tol=tol,diis_size=diis_size)
            # Calculate CC(P;Q) moment correction using 2BA
            Eccpq[niter-niter0], mcA, mcB, mcC, mcD  = ccp3(cc_t,p_spaces,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF=isRHF)

        print('SUMMARY AT ITERATION - {}'.format(niter))
        print('NUMBER OF TRIPLES IN P SPACE - {}  ({}%)'.format(num_triples,num_triples/tot_triples*100))
        print('E(P) = {} HARTREE'.format(Eccp[niter-niter0]))
        print('E(P;Q) = {} HARTREE'.format(Eccpq[niter-niter0]))

    return Eccp, Eccpq

def calc_adaptive_ccpq_depreciated(sys,ints,tot_triples,workdir,ccshift=0.0,lccshift=0.0,maxit=10,growth_percentage=1.0,\
                restart_dir=None,niter0=0,isRHF=False,ccmaxit=100,tol=1.0e-08,diis_size=6,flag_save=False):

    print('\n==================================++Entering Adaptive CC(P;Q) Routine++=================================\n')
    print('     REXALED ADAPTIVE CC(P;Q) ALGORITHM')
    print('     TOTAL NUMBER OF TRIPLES: {}'.format(tot_triples))
    print('     REQUESTED %T GROWTH PER ITERATION = {}'.format(growth_percentage))
    print('     NUMBER OF ADAPTIVE CC(P;Q) ITERATIONS = {}'.format(maxit))
    print('     SAVE CLUSTER AMPLITUDES & P SPACES = {}'.format(flag_save))
    print('')


    noa = sys['Nocc_a']
    nob = sys['Nocc_b']
    nua = sys['Nunocc_a']
    nub = sys['Nunocc_b']

    Eccp = np.zeros(maxit)
    Eccpq = np.zeros(maxit)

    if restart_dir is not None:
        niter0 = int(restart_dir.split('Iter')[1].strip())

        print('RESTARTING CALCULATION FROM DIRECTORY: ',restart_dir)
        print('    MACRO ITERATION - {}'.format(niter0))

        p_spaceA = np.load(restart_dir+'/pspace_A.npy')
        p_spaceB = np.load(restart_dir+'/pspace_B.npy')
        p_spaceC = np.load(restart_dir+'/pspace_C.npy')
        p_spaceD = np.load(restart_dir+'/pspace_D.npy')
        p_spaces = {'A' : p_spaceA, 'B' : p_spaceB, 'C' : p_spaceC, 'D' : p_spaceD}

        t1a = np.load(restart_dir+'/t1a.npy')
        t1b = np.load(restart_dir+'/t1b.npy')
        t2a = np.load(restart_dir+'/t2a.npy')
        t2b = np.load(restart_dir+'/t2b.npy')
        t2c = np.load(restart_dir+'/t2c.npy')
        t3a = np.load(restart_dir+'/t3a.npy')
        t3b = np.load(restart_dir+'/t3b.npy')
        t3c = np.load(restart_dir+'/t3c.npy')
        t3d = np.load(restart_dir+'/t3d.npy')
        cc_t = {'t1a' : t1a, 't1b' : t1b, 't2a' : t2a, 't2b' : t2b, 't2c' : t2c,\
                't3a' : t3a, 't3b' : t3b, 't3c' : t3c, 't3d' : t3d}

        Ecorr = 0.0
        Ecorr += np.einsum('me,em->',ints['fA']['ov'],t1a,optimize=True)
        Ecorr += np.einsum('me,em->',ints['fB']['ov'],t1b,optimize=True)
        Ecorr += 0.25*np.einsum('mnef,efmn->',ints['vA']['oovv'],t2a,optimize=True)
        Ecorr += np.einsum('mnef,efmn->',ints['vB']['oovv'],t2b,optimize=True)
        Ecorr += 0.25*np.einsum('mnef,efmn->',ints['vC']['oovv'],t2c,optimize=True)
        Ecorr += 0.5*np.einsum('mnef,fn,em->',ints['vA']['oovv'],t1a,t1a,optimize=True)
        Ecorr += 0.5*np.einsum('mnef,fn,em->',ints['vC']['oovv'],t1b,t1b,optimize=True)
        Ecorr += np.einsum('mnef,em,fn->',ints['vB']['oovv'],t1a,t1b,optimize=True)
        print('CC(P) ENERGY = {} HARTREE'.format(Ecorr+ints['Escf']))

        num_triples = count_triples_in_P(p_spaces)
        print('NUMBER OF TRIPLES in P SPACE OF THIS DIRECTORY - {}\n\n'.format(num_triples))
        print('NOTE: THIS NUMBER CORRESPONDS TO THE P SPACE FOR THE **NEXT** ITERATION!!! - MACRO ITERATION {}'.format(niter0+1))
        niter0 += 1

    else:
        p_spaces = {'A' : np.zeros((nua,nua,nua,noa,noa,noa)),\
                    'B' : np.zeros((nua,nua,nub,noa,noa,nob)),\
                    'C' : np.zeros((nua,nub,nub,noa,nob,nob)),\
                    'D' : np.zeros((nub,nub,nub,nob,nob,nob))}

        num_triples = 0

    ntot = tot_triples

    for niter in range(niter0,niter0+maxit):
        
        print('MACRO ITERATION - {}'.format(niter)) 

        if flag_save:
            iterdir = workdir+'/Iter'+str(niter)
            os.mkdir(iterdir)

        if niter > 0:
            cc_t, Eccp[niter-niter0] = ccsdt_p(sys,ints,p_spaces,maxit=ccmaxit,shift=ccshift,initial_guess=cc_t,flag_RHF=isRHF,\
                            tol=tol,diis_size=diis_size)
        else:
            cc_t, Eccp[niter-niter0] = ccsdt_p(sys,ints,p_spaces,maxit=ccmaxit,shift=ccshift,flag_RHF=isRHF,tol=tol,diis_size=diis_size)

        if flag_save:
            for key, value in cc_t.items():
                np.save(iterdir+'/'+key,value)

        H1A,H1B,H2A,H2B,H2C = HBar_CCSD(cc_t,ints,sys)

        cc_t = left_ccsd(cc_t,H1A,H1B,H2A,H2B,H2C,ints,sys,shift=lccshift,maxit=ccmaxit,tol=tol,diis_size=diis_size)

        Eccpq[niter-niter0], mcA, mcB, mcC, mcD  = ccp3(cc_t,p_spaces,H1A,H1B,H2A,H2B,H2C,ints,sys,flag_RHF=isRHF)

        add_1perc = int(tot_triples * 0.01)
        num_add = int( growth_percentage * add_1perc )
        p_spaces,num_add = selection_function(sys,mcA,mcB,mcC,mcD,p_spaces,num_add,flag_RHF=isRHF)

        if flag_save:
            for key, value in p_spaces.items():
                np.save(iterdir+'/pspace_'+key,value)

        print('SUMMARY AT ITERATION - {}'.format(niter))
        print('NUMBER OF TRIPLES IN P SPACE - {}  ({}%)'.format(num_triples,num_triples/tot_triples*100))
        print('NUMBER OF TRIPLES ADDED FOR NEXT ITERATION - {}'.format(num_add))
        print('E(P) = {} HARTREE'.format(Eccp[niter-niter0]))
        print('E(P;Q) = {} HARTREE'.format(Eccpq[niter-niter0]))

        if num_add == 0:
            print('Algorithm stagnated; no more triples added!')
            break
        else:
            num_triples = num_triples + num_add
            ntot = tot_triples - num_triples

    return Eccp, Eccpq

def selection_function(sys,mcA,mcB,mcC,mcD,p_spaces,num_add,flag_RHF=False):

    print('SELECTION STEP USING CR-CC(2,3)_D CORRECTIONS')
    n3a = sys['Nunocc_a']**3 * sys['Nocc_a']**3
    n3b = sys['Nunocc_a']**2 * sys['Nunocc_b'] * sys['Nocc_a']**2 * sys['Nocc_b']
    n3c = sys['Nunocc_a'] * sys['Nunocc_b']**2 * sys['Nocc_a'] * sys['Nocc_b']**2
    n3d = sys['Nunocc_b']**3 * sys['Nocc_b']**3
    ntot = n3a + n3b + n3c + n3d

    mvec = np.zeros(ntot)
    mvec[:n3a] = mcA.flatten()
    mvec[n3a:n3a+n3b] = mcB.flatten()
    mvec[n3a+n3b:n3a+n3b+n3c] = mcC.flatten()
    mvec[n3a+n3b+n3c:] = mcD.flatten()

    idx = np.flip(np.argsort(abs(mvec)))
    
    # should these be copied?
    p_spaces_out = {'A' : p_spaces['A'],\
                    'B' : p_spaces['B'],\
                    'C' : p_spaces['C'],\
                    'D' : p_spaces['D']}

    if not flag_RHF:
        ct = 0
        ct2 = 0
        while ct < num_add:

            if idx[ct2] < n3a:
                a,b,c,i,j,k = np.unravel_index(idx[ct2],mcA.shape)
                if p_spaces['A'][a,b,c,i,j,k] == 1:
                    ct2 += 1
                    continue
                else:
                    ct += 1
                    p_spaces_out['A'][a,b,c,i,j,k] = 1
                    p_spaces_out['A'][a,b,c,i,k,j] = 1
                    p_spaces_out['A'][a,b,c,j,i,k] = 1
                    p_spaces_out['A'][a,b,c,j,k,j] = 1
                    p_spaces_out['A'][a,b,c,k,i,j] = 1
                    p_spaces_out['A'][a,b,c,k,j,i] = 1
            
                    p_spaces_out['A'][a,c,b,i,j,k] = 1
                    p_spaces_out['A'][a,c,b,i,k,j] = 1
                    p_spaces_out['A'][a,c,b,j,i,k] = 1
                    p_spaces_out['A'][a,c,b,j,k,j] = 1
                    p_spaces_out['A'][a,c,b,k,i,j] = 1
                    p_spaces_out['A'][a,c,b,k,j,i] = 1

                    p_spaces_out['A'][b,a,c,i,j,k] = 1
                    p_spaces_out['A'][b,a,c,i,k,j] = 1
                    p_spaces_out['A'][b,a,c,j,i,k] = 1
                    p_spaces_out['A'][b,a,c,j,k,j] = 1
                    p_spaces_out['A'][b,a,c,k,i,j] = 1
                    p_spaces_out['A'][b,a,c,k,j,i] = 1

                    p_spaces_out['A'][b,c,a,i,j,k] = 1
                    p_spaces_out['A'][b,c,a,i,k,j] = 1
                    p_spaces_out['A'][b,c,a,j,i,k] = 1
                    p_spaces_out['A'][b,c,a,j,k,j] = 1
                    p_spaces_out['A'][b,c,a,k,i,j] = 1
                    p_spaces_out['A'][b,c,a,k,j,i] = 1

                    p_spaces_out['A'][c,a,b,i,j,k] = 1
                    p_spaces_out['A'][c,a,b,i,k,j] = 1
                    p_spaces_out['A'][c,a,b,j,i,k] = 1
                    p_spaces_out['A'][c,a,b,j,k,j] = 1
                    p_spaces_out['A'][c,a,b,k,i,j] = 1
                    p_spaces_out['A'][c,a,b,k,j,i] = 1

                    p_spaces_out['A'][c,b,a,i,j,k] = 1
                    p_spaces_out['A'][c,b,a,i,k,j] = 1
                    p_spaces_out['A'][c,b,a,j,i,k] = 1
                    p_spaces_out['A'][c,b,a,j,k,j] = 1
                    p_spaces_out['A'][c,b,a,k,i,j] = 1
                    p_spaces_out['A'][c,b,a,k,j,i] = 1

            elif idx[ct2] < n3a+n3b:
                a,b,c,i,j,k = np.unravel_index(idx[ct2]-n3a,mcB.shape)
                if p_spaces['B'][a,b,c,i,j,k] == 1:
                    ct2 += 1
                    continue
                else:
                    ct += 1
                    p_spaces_out['B'][a,b,c,i,j,k] = 1
                    p_spaces_out['B'][b,a,c,i,j,k] = 1
                    p_spaces_out['B'][a,b,c,j,i,k] = 1
                    p_spaces_out['B'][b,a,c,j,i,k] = 1

            elif idx[ct2] < n3a+n3b+n3c:
                a,b,c,i,j,k = np.unravel_index(idx[ct2]-n3a-n3b,mcC.shape)
                if p_spaces['C'][a,b,c,i,j,k] == 1:
                    ct2 += 1
                    continue
                else:
                    ct += 1
                    p_spaces_out['C'][a,b,c,i,j,k] = 1
                    p_spaces_out['C'][a,c,b,i,j,k] = 1
                    p_spaces_out['C'][a,b,c,i,k,j] = 1
                    p_spaces_out['C'][a,c,b,i,k,j] = 1

            else:
                a,b,c,i,j,k = np.unravel_index(idx[ct2]-n3a-n3b-n3c,mcD.shape)
                if p_spaces['D'][a,b,c,i,j,k] == 1:
                    ct2 += 1
                    continue
                else:
                    ct += 1
                    p_spaces_out['D'][a,b,c,i,j,k] = 1
                    p_spaces_out['D'][a,b,c,i,k,j] = 1
                    p_spaces_out['D'][a,b,c,j,i,k] = 1
                    p_spaces_out['D'][a,b,c,j,k,j] = 1
                    p_spaces_out['D'][a,b,c,k,i,j] = 1
                    p_spaces_out['D'][a,b,c,k,j,i] = 1
        
                    p_spaces_out['D'][a,c,b,i,j,k] = 1
                    p_spaces_out['D'][a,c,b,i,k,j] = 1
                    p_spaces_out['D'][a,c,b,j,i,k] = 1
                    p_spaces_out['D'][a,c,b,j,k,j] = 1
                    p_spaces_out['D'][a,c,b,k,i,j] = 1
                    p_spaces_out['D'][a,c,b,k,j,i] = 1

                    p_spaces_out['D'][b,a,c,i,j,k] = 1
                    p_spaces_out['D'][b,a,c,i,k,j] = 1
                    p_spaces_out['D'][b,a,c,j,i,k] = 1
                    p_spaces_out['D'][b,a,c,j,k,j] = 1
                    p_spaces_out['D'][b,a,c,k,i,j] = 1
                    p_spaces_out['D'][b,a,c,k,j,i] = 1

                    p_spaces_out['D'][b,c,a,i,j,k] = 1
                    p_spaces_out['D'][b,c,a,i,k,j] = 1
                    p_spaces_out['D'][b,c,a,j,i,k] = 1
                    p_spaces_out['D'][b,c,a,j,k,j] = 1
                    p_spaces_out['D'][b,c,a,k,i,j] = 1
                    p_spaces_out['D'][b,c,a,k,j,i] = 1

                    p_spaces_out['D'][c,a,b,i,j,k] = 1
                    p_spaces_out['D'][c,a,b,i,k,j] = 1
                    p_spaces_out['D'][c,a,b,j,i,k] = 1
                    p_spaces_out['D'][c,a,b,j,k,j] = 1
                    p_spaces_out['D'][c,a,b,k,i,j] = 1
                    p_spaces_out['D'][c,a,b,k,j,i] = 1

                    p_spaces_out['D'][c,b,a,i,j,k] = 1
                    p_spaces_out['D'][c,b,a,i,k,j] = 1
                    p_spaces_out['D'][c,b,a,j,i,k] = 1
                    p_spaces_out['D'][c,b,a,j,k,j] = 1
                    p_spaces_out['D'][c,b,a,k,i,j] = 1
                    p_spaces_out['D'][c,b,a,k,j,i] = 1

    else: # USING RHF SYMMETRY, ADDINGS PAIRS OF A/D AND B/C DETS TO P SPACE

        print('    >>USING RHF SYMMETRY<<')
        ct = 0
        ct2 = 0
        while ct < num_add:
            if idx[ct2] < n3a:
                a,b,c,i,j,k = np.unravel_index(idx[ct2],mcA.shape)
                if p_spaces['A'][a,b,c,i,j,k] == 1 or p_spaces['D'][a,b,c,i,j,k] == 1:
                    ct2 += 1
                    continue
                else:
                    ct += 2
                    p_spaces_out['A'][a,b,c,i,j,k] = 1
                    p_spaces_out['A'][a,b,c,i,k,j] = 1
                    p_spaces_out['A'][a,b,c,j,i,k] = 1
                    p_spaces_out['A'][a,b,c,j,k,j] = 1
                    p_spaces_out['A'][a,b,c,k,i,j] = 1
                    p_spaces_out['A'][a,b,c,k,j,i] = 1
            
                    p_spaces_out['A'][a,c,b,i,j,k] = 1
                    p_spaces_out['A'][a,c,b,i,k,j] = 1
                    p_spaces_out['A'][a,c,b,j,i,k] = 1
                    p_spaces_out['A'][a,c,b,j,k,j] = 1
                    p_spaces_out['A'][a,c,b,k,i,j] = 1
                    p_spaces_out['A'][a,c,b,k,j,i] = 1

                    p_spaces_out['A'][b,a,c,i,j,k] = 1
                    p_spaces_out['A'][b,a,c,i,k,j] = 1
                    p_spaces_out['A'][b,a,c,j,i,k] = 1
                    p_spaces_out['A'][b,a,c,j,k,j] = 1
                    p_spaces_out['A'][b,a,c,k,i,j] = 1
                    p_spaces_out['A'][b,a,c,k,j,i] = 1

                    p_spaces_out['A'][b,c,a,i,j,k] = 1
                    p_spaces_out['A'][b,c,a,i,k,j] = 1
                    p_spaces_out['A'][b,c,a,j,i,k] = 1
                    p_spaces_out['A'][b,c,a,j,k,j] = 1
                    p_spaces_out['A'][b,c,a,k,i,j] = 1
                    p_spaces_out['A'][b,c,a,k,j,i] = 1

                    p_spaces_out['A'][c,a,b,i,j,k] = 1
                    p_spaces_out['A'][c,a,b,i,k,j] = 1
                    p_spaces_out['A'][c,a,b,j,i,k] = 1
                    p_spaces_out['A'][c,a,b,j,k,j] = 1
                    p_spaces_out['A'][c,a,b,k,i,j] = 1
                    p_spaces_out['A'][c,a,b,k,j,i] = 1

                    p_spaces_out['A'][c,b,a,i,j,k] = 1
                    p_spaces_out['A'][c,b,a,i,k,j] = 1
                    p_spaces_out['A'][c,b,a,j,i,k] = 1
                    p_spaces_out['A'][c,b,a,j,k,j] = 1
                    p_spaces_out['A'][c,b,a,k,i,j] = 1
                    p_spaces_out['A'][c,b,a,k,j,i] = 1

                    p_spaces_out['D'][a,b,c,i,j,k] = 1
                    p_spaces_out['D'][a,b,c,i,k,j] = 1
                    p_spaces_out['D'][a,b,c,j,i,k] = 1
                    p_spaces_out['D'][a,b,c,j,k,j] = 1
                    p_spaces_out['D'][a,b,c,k,i,j] = 1
                    p_spaces_out['D'][a,b,c,k,j,i] = 1
        
                    p_spaces_out['D'][a,c,b,i,j,k] = 1
                    p_spaces_out['D'][a,c,b,i,k,j] = 1
                    p_spaces_out['D'][a,c,b,j,i,k] = 1
                    p_spaces_out['D'][a,c,b,j,k,j] = 1
                    p_spaces_out['D'][a,c,b,k,i,j] = 1
                    p_spaces_out['D'][a,c,b,k,j,i] = 1

                    p_spaces_out['D'][b,a,c,i,j,k] = 1
                    p_spaces_out['D'][b,a,c,i,k,j] = 1
                    p_spaces_out['D'][b,a,c,j,i,k] = 1
                    p_spaces_out['D'][b,a,c,j,k,j] = 1
                    p_spaces_out['D'][b,a,c,k,i,j] = 1
                    p_spaces_out['D'][b,a,c,k,j,i] = 1

                    p_spaces_out['D'][b,c,a,i,j,k] = 1
                    p_spaces_out['D'][b,c,a,i,k,j] = 1
                    p_spaces_out['D'][b,c,a,j,i,k] = 1
                    p_spaces_out['D'][b,c,a,j,k,j] = 1
                    p_spaces_out['D'][b,c,a,k,i,j] = 1
                    p_spaces_out['D'][b,c,a,k,j,i] = 1

                    p_spaces_out['D'][c,a,b,i,j,k] = 1
                    p_spaces_out['D'][c,a,b,i,k,j] = 1
                    p_spaces_out['D'][c,a,b,j,i,k] = 1
                    p_spaces_out['D'][c,a,b,j,k,j] = 1
                    p_spaces_out['D'][c,a,b,k,i,j] = 1
                    p_spaces_out['D'][c,a,b,k,j,i] = 1

                    p_spaces_out['D'][c,b,a,i,j,k] = 1
                    p_spaces_out['D'][c,b,a,i,k,j] = 1
                    p_spaces_out['D'][c,b,a,j,i,k] = 1
                    p_spaces_out['D'][c,b,a,j,k,j] = 1
                    p_spaces_out['D'][c,b,a,k,i,j] = 1
                    p_spaces_out['D'][c,b,a,k,j,i] = 1

            elif idx[ct2] < n3a+n3b:
                a,b,c,i,j,k = np.unravel_index(idx[ct2]-n3a,mcB.shape)
                if p_spaces['B'][a,b,c,i,j,k] == 1 or p_spaces['C'][c,a,b,k,i,j] == 1:
                    ct2 += 1
                    continue
                else:
                    ct += 2
                    p_spaces_out['B'][a,b,c,i,j,k] = 1
                    p_spaces_out['B'][b,a,c,i,j,k] = 1
                    p_spaces_out['B'][a,b,c,j,i,k] = 1
                    p_spaces_out['B'][b,a,c,j,i,k] = 1

                    p_spaces_out['C'][c,a,b,k,i,j] = 1
                    p_spaces_out['C'][c,b,a,k,i,j] = 1
                    p_spaces_out['C'][c,a,b,k,j,i] = 1
                    p_spaces_out['C'][c,b,a,k,j,i] = 1



    return p_spaces_out, ct

def count_triples_in_P(p_spaces):

    num_triples = 0

    pA = p_spaces['A']
    for a in range(pA.shape[0]):
        for b in range(a+1,pA.shape[1]):
            for c in range(b+1,pA.shape[2]):
                for i in range(pA.shape[3]):
                    for j in range(i+1,pA.shape[4]):
                        for k in range(j+1,pA.shape[5]):
                            if pA[a,b,c,i,j,k] == 1:
                                num_triples += 1

    pB = p_spaces['B']
    for a in range(pB.shape[0]):
        for b in range(a+1,pB.shape[1]):
            for c in range(pB.shape[2]):
                for i in range(pB.shape[3]):
                    for j in range(i+1,pB.shape[4]):
                        for k in range(pB.shape[5]):
                            if pB[a,b,c,i,j,k] == 1:
                                num_triples += 1

    pC = p_spaces['C']
    for a in range(pC.shape[0]):
        for b in range(pC.shape[1]):
            for c in range(b+1,pC.shape[2]):
                for i in range(pC.shape[3]):
                    for j in range(pC.shape[4]):
                        for k in range(j+1,pC.shape[5]):
                            if pC[a,b,c,i,j,k] == 1:
                                num_triples += 1

    pD = p_spaces['D']
    for a in range(pD.shape[0]):
        for b in range(a+1,pD.shape[1]):
            for c in range(b+1,pD.shape[2]):
                for i in range(pD.shape[3]):
                    for j in range(i+1,pD.shape[4]):
                        for k in range(j+1,pD.shape[5]):
                            if pD[a,b,c,i,j,k] == 1:
                                num_triples += 1

    return num_triples








