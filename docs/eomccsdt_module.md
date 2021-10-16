Module eomccsdt_module
======================
Module containing functions to calculate the vertical excitation
energies and linear excitation amplitudes for excited states using
the equation-of-motion (EOM) CC with singles, doubles, and triples (EOMCCSDT).

Functions
---------

    
`HR(R, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys)`
:   Calculate the matrix-vector product H(CCSDT)*R.
    
    Parameters
    ----------
    R : ndarray(dtype=float, shape=(ndim_ccsdt))
        Flattened vector of R amplitudes
    cc_t : dict
        Cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    
    Returns
    -------
    HR : ndarray(dtype=float, shape=(ndim_ccsdt))
        Vector containing the matrix-vector product H(CCSDT)*R

    
`build_HR_1A(r1a, r1b, r2a, r2b, r2c, r3a, r3b, r3c, r3d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys)`
:   Calculate the projection <ia|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.
    
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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    
    Returns
    --------
    X1A : ndarray(dtype=float, shape=(nua,noa))
        Calculated HR Projection

    
`build_HR_1B(r1a, r1b, r2a, r2b, r2c, r3a, r3b, r3c, r3d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys)`
:   Calculate the projection <i~a~|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.
    
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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    
    Returns
    --------
    X1B : ndarray(dtype=float, shape=(nub,nob))
        Calculated HR Projection

    
`build_HR_2A(r1a, r1b, r2a, r2b, r2c, r3a, r3b, r3c, r3d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys)`
:   Calculate the projection <ijab|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.
    
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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    
    Returns
    --------
    X2A : ndarray(dtype=float, shape=(nua,nua,noa,noa))
        Calculated HR Projection

    
`build_HR_2B(r1a, r1b, r2a, r2b, r2c, r3a, r3b, r3c, r3d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys)`
:   Calculate the projection <ij~ab~|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.
    
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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    
    Returns
    --------
    X2B : ndarray(dtype=float, shape=(nua,nub,noa,nob))
        Calculated HR Projection

    
`build_HR_2C(r1a, r1b, r2a, r2b, r2c, r3a, r3b, r3c, r3d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys)`
:   Calculate the projection <i~j~a~b~|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.
    
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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    
    Returns
    --------
    X2C : ndarray(dtype=float, shape=(nub,nub,nob,nob))
        Calculated HR Projection

    
`build_HR_3A(r1a, r1b, r2a, r2b, r2c, r3a, r3b, r3c, r3d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys)`
:   Calculate the projection <ijkabc|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.
    
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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    
    Returns
    --------
    X3A : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Calculated HR Projection

    
`build_HR_3B(r1a, r1b, r2a, r2b, r2c, r3a, r3b, r3c, r3d, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys)`
:   Calculate the projection <ijk~abc~|[ (H_N e^(T1+T2+T3))_C*(R1+R2+R3) ]_C|0>.
    
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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    
    Returns
    --------
    X3B : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Calculated HR Projection

    
`calc_cc_energy(cc_t, ints)`
:   Calculate the CC correlation energy <0|(H_N e^T)_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    ints : dict
        Sliced integrals F_N and V_N that define the bare Hamiltonian H_N
        
    Returns
    -------
    Ecorr : float
        CC correlation energy

    
`calc_r0(r1a, r1b, r2a, r2b, r2c, H1A, H1B, ints, omega)`
:   Calculate the EOMCC overlap <0|[ (H_N e^T)_C * (R1+R2) ]_C|0>.
    
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
    H1*, H2* : dict
        Sliced CCSD similarity-transformed HBar integrals
    ints : dict
        Sliced integrals F_N and V_N that define the bare Hamiltonian H_N
    omega : float
        Vertical excitation energy (in hartree) for the given root
    
    Returns
    -------
    r0 : float
        Overlap of excited state with ground state

    
`cis(ints, sys)`
:   Build and diagonalize the CIS Hamiltonian.
    
    Parameters
    ----------
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    
    Returns
    -------
    C : ndarray(dtype=float, shape=(ndim_cis,ndim_cis))
        Matrix of CIS eigenvectors
    E_cis : ndarray(dtype=float, shape=(ndim_cis))
        Vector of CIS eigenvalues

    
`davidson_solver(H1A, H1B, H2A, H2B, H2C, ints, cc_t, nroot, B0, E0, sys, maxit, tol)`
:   Diagonalize the CCSDT similarity-transformed Hamiltonian HBar using the
    non-Hermitian Davidson algorithm.
    
    Parameters
    ----------
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    cc_t : dict
        Cluster amplitudes T1, T2 of the ground-state
    nroot : int
        Number of excited-states to solve for
    B0 : ndarray(dtype=float, shape=(ndim_ccsdt,nroot))
        Matrix containing the initial guess vectors for the Davidson procedure
    E0 : ndarray(dtype=float, shape=(nroot))
        Vector containing the energies corresponding to the initial guess vectors
    sys : dict
        System information dictionary
    maxit : int, optional
        Maximum number of Davidson iterations in the EOMCC procedure.
    tol : float, optional
        Convergence tolerance for the EOMCC calculation. Default is 1.0e-06.
    
    Returns
    -------
    Rvec : ndarray(dtype=float, shape=(ndim_ccsdt,nroot))
        Matrix containing the final converged R vectors corresponding to the EOMCCSDT linear excitation amplitudes
    omega : ndarray(dtype=float, shape=(nroot))
        Vector of vertical excitation energies (in hartree) for each root
    is_converged : list
        List of boolean indicating whether each root converged to within the specified tolerance

    
`eomccsdt(nroot, H1A, H1B, H2A, H2B, H2C, cc_t, ints, sys, initial_guess='cis', tol=1e-06, maxit=80)`
:   Perform the EOMCCSDT excited-state calculation.
    
    Parameters
    ----------
    nroot : int
        Number of excited-states to solve for in the EOMCCSDT procedure
    H1*, H2* : dict
        Sliced CCSDT similarity-transformed HBar integrals
    cc_t : dict
        Cluster amplitudes T1, T2, and T3 of the ground-state
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    initial_guess : str, optional
        String that specifies the form of the initial guess, including options for
        'cis' and 'eomccsd'. Default is 'cis'.
    tol : float, optional
        Convergence tolerance for the EOMCC calculation. Default is 1.0e-06.
    maxit : int, optional
        Maximum number of Davidson iterations in the EOMCC procedure.
    
    Returns
    -------
    cc_t : dict
        Updated dictionary of cluster amplitudes with r0, R1, R2, and R3 amplitudes for each excited state.
    omega : ndarray(dtype=float, shape=(nroot))
        Vector of vertical excitation energies (in hartree) for each root

    
`flatten_R(r1a, r1b, r2a, r2b, r2c, r3a, r3b, r3c, r3d)`
:   Flatten the R vector.
    
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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)
    
    Returns
    -------
    R : ndarray(dtype=float, shape=(ndim_ccsdt))
        Flattened array of R vector for the given root

    
`orthogonalize(q, B)`
:   Orthogonalize the correction vector to the vectors comprising
    the current subspace.
    
    Parameters
    ----------
    q : ndarray(dtype=float, shape=(ndim_ccsdt))
        Preconditioned residual vector from Davidson procedure
    B : ndarray(dtype=float, shape=(ndim_ccsdt,curr_size))
        Matrix of subspace vectors in Davidson procedure
    
    Returns
    -------
    q : ndarray(dtype=float, shape=(ndim_ccsdt))
        Orthogonalized residual vector

    
`test_updates(matfile, cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys)`
:   Test the EOMCCSDT updates using known results from Matlab code.
    
    Parameters
    ----------
    matfile : str
        Path to .mat file containing R vector amplitudes from Matlab
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    cc_t : dict
        Cluster amplitudes T1, T2, and T3
    sys : dict
        System information dictionary
    
    Returns
    -------
    None

    
`unflatten_R(R, sys, order='C')`
:   Unflatten the R vector into many-body tensor components.
    
    Parameters
    ----------
    R : ndarray(dtype=float, shape=(ndim_ccsdt))
        Flattened array of R vector for the given root
    sys : dict
        System information dictionary
    order : str, optional
        String of value 'C' or 'F' indicating whether row-major or column-major
        flattening should be used. Default is 'C'.
    
    Returns
    -------
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
    r3a : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Linear EOMCC excitation amplitudes R3(aaa)
    r3b : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Linear EOMCC excitation amplitudes R3(aab)
    r3c : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Linear EOMCC excitation amplitudes R3(abb)
    r3d : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Linear EOMCC excitation amplitudes R3(bbb)