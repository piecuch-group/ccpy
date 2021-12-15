"""Module for handling the initial EOMCCSd guess for the EOMCC calculations"""
import numpy as np
import eomcc_initial_guess

def get_eomcc_initial_guess(nroot,noact,nuact,ndim,H1A,H1B,H2A,H2B,H2C,ints,sys):
    """Obtains the EOMCCSd initial guess.

    Parameters:
    -----------
    nroot : list
        Number of roots requested for each irrep
    noact : int
        Number of active occupied orbitals in EOMCCSd initial guess
    nuact : int
        Number of active unoccupied orbitals in EOMCCSd initial guess
    ndim : int
        Dimension of the EOMCC method (e.g., triples for EOMCCSDT, doubles for EOMCCSD, etc.,)
    H1*, H2* : dict
        Similarity-transformed Hamiltonian of the underlying ground-state CC theory
    ints : dict
        Sliced F_N and V_N integrals of the bare Hamiltonian
    sys : dict  
        System information dictionary

    Returns:
    --------
    B0 : ndarray(dtype=np.float64,shape=(ndim,sum(nroot)))
        Matrix of initial guess vectors
    E0 : ndarray(dtype=np.float64,shape=nroot))
        Matrix of initial guess energies
    """
    num_roots_total = sum(nroot)

    num_doubles =sys['Nocc_a']*sys['Nunocc_a']\
                 +sys['Nocc_b']*sys['Nunocc_b']\
                 +sys['Nocc_a']**2*sys['Nunocc_a']**2\
                 +sys['Nocc_a']*sys['Nunocc_a']*sys['Nocc_b']*sys['Nunocc_b']\
                 +sys['Nocc_b']**2*sys['Nunocc_b']**2

    B0 = np.zeros((ndim,num_roots_total))
    E0 = np.zeros(num_roots_total)
    root_sym = [None]*num_roots_total

    mo_sym = np.array(sys['sym_nums'])[sys['Nfroz']:]
    mult_table = np.array(sys['pg_mult_table'])
    h_group = mult_table.shape[0]
    sym_ref = 0

    idx1A,idx1B,idx2A,idx2B,idx2C,\
    syms1A,syms1B,syms2A,syms2B,syms2C,\
    n1a_act,n1b_act,n2a_act,n2b_act,n2c_act =\
                        eomcc_initial_guess.eomcc_initial_guess.\
                        get_active_dimensions(noact,nuact,mo_sym,\
                        mult_table,\
                        sys['Nocc_a'],sys['Nunocc_a'],sys['Nocc_b'],sys['Nunocc_b'])
    ndim_act = n1a_act+n1b_act+n2a_act+n2b_act+n2c_act
    print('Dimension of EOMCCSd guess = {}'.format(ndim_act))

    nct = 0
    for sym_target, num_root_sym in enumerate(nroot):

        if num_root_sym == 0: continue

        state_irrep = list(sys['irrep_map'].keys())[sym_target]
        print('Calculating initial guess for {} singlet states of {} symmetry '.format(num_root_sym,state_irrep))

        Cvec, omega_eomccsd, Hmat = eomcc_initial_guess.eomcc_initial_guess.\
                        eomccs_d_matrix(idx1A,idx1B,idx2A,idx2B,idx2C,\
                        H1A['oo'],H1A['vv'],H1A['ov'],H1B['oo'],H1B['vv'],H1B['ov'],\
                        H2A['oooo'],H2A['vvvv'],H2A['voov'],H2A['vooo'],H2A['vvov'],\
                        H2A['ooov'],H2A['vovv'],\
                        H2B['oooo'],H2B['vvvv'],H2B['voov'],H2B['ovvo'],H2B['vovo'],\
                        H2B['ovov'],H2B['vooo'],H2B['ovoo'],H2B['vvov'],H2B['vvvo'],\
                        H2B['ooov'],H2B['oovo'],H2B['vovv'],H2B['ovvv'],\
                        H2C['oooo'],H2C['vvvv'],H2C['voov'],H2C['vooo'],H2C['vvov'],\
                        H2C['ooov'],H2C['vovv'],\
                        n1a_act,n1b_act,n2a_act,n2b_act,n2c_act,ndim_act,\
                        sym_target,sym_ref,syms1A,syms1B,syms2A,syms2B,syms2C,\
                        mult_table)

        # sort the roots
        idx = np.argsort(omega_eomccsd)
        omega_eomccsd = omega_eomccsd[idx]
        Cvec = Cvec[:,idx]

        # locate only singlet roots
        if sys['Nocc_a'] == sys['Nocc_b']: # if closed shell
            ct = 0
            slice_1A = slice(0,n1a_act)
            slice_1B = slice(n1a_act,n1a_act+n1b_act)
            for i in range(len(omega_eomccsd)):
                chk = np.linalg.norm(Cvec[slice_1A,i] - Cvec[slice_1B,i])
                if abs(chk) < 1.0e-01:
                    r1a,r1b,r2a,r2b,r2c = eomcc_initial_guess.eomcc_initial_guess.\
                                unflatten_guess_vector(Cvec[:,i],idx1A,idx1B,idx2A,idx2B,idx2C,\
                                n1a_act,n1b_act,n2a_act,n2b_act,n2c_act)
                    B0[:num_doubles,nct] = flatten_R(r1a,r1b,r2a,r2b,r2c)
                    E0[nct] = omega_eomccsd[i]
                    root_sym[nct] = state_irrep
                    nct += 1
                    ct += 1
                    if ct == num_root_sym:
                        break
            else:
                print('Could not find {} singlet roots of {} symmetry in EOMCCSd guess!'.format(num_root_sym,state_irrep)) 
        else: # open shell
            ct = 0
            for i in range(len(omega_eomccsd)):
                r1a,r1b,r2a,r2b,r2c = eomcc_initial_guess.eomcc_initial_guess.\
                                unflatten_guess_vector(Cvec[:,i],idx1A,idx1B,idx2A,idx2B,idx2C,\
                                n1a_act,n1b_act,n2a_act,n2b_act,n2c_act)
                B0[:num_doubles,nct] = flatten_R(r1a,r1b,r2a,r2b,r2c)
                E0[nct] = omega_eomccsd[i]
                root_sym[nct] = state_irrep
                nct += 1
                ct += 1
                if ct == num_root_sym:
                    break

    print('Initial EOMCCSd energies:')
    for i in range(len(E0)):
        print('Root - {}  (Sym: {})     E = {:.10f}    ({:.10f})'.format(i+1,root_sym[i],E0[i],E0[i]+ints['Escf']))
    print('')

    return B0,E0

def flatten_R(r1a,r1b,r2a,r2b,r2c,order='C'):
    """Flatten the R vector.

    Parameters
    ----------
    r1a : ndarray(dtype=float, shape=(nua,noa))
        Linear EOMCC excitation amplitudes R1(aa)
    r1b : ndarray(dtype=float, shape=(nub,nob))
        Linear EOMCC excitation amplitudes R1(bb)
    r2a : ndarray(dtype=float, shape=(nua,nua,noa,noa))
        Linear EOMCC excitation amplitudes R2(aa)
    r2b : ndarray(dtype=float, shape=(nua,nub,noa,nob))
        Linear EOMCC excitation amplitudes R2(ab)
    r2c : ndarray(dtype=float, shape=(nub,nub,nob,nob))
        Linear EOMCC excitation amplitudes R2(bb)

    Returns
    -------
    R : ndarray(dtype=float, shape=(ndim_ccsd))
        Flattened array of R vector for the given root
    """
    return np.concatenate((r1a.flatten(order=order),r1b.flatten(order=order),r2a.flatten(order=order),r2b.flatten(order=order),r2c.flatten(order=order)),axis=0)
