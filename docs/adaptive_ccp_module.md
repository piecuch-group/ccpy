Module adaptive_ccp_module
==========================
Module with functions used to perform the ground-state CC(P) 
calculations for a given P space used in the adaptive-CC(P;Q) scheme
aimed at converging the CCSDT energetics.

Functions
---------

    
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

    
`ccsdt_p(sys, ints, p_spaces, maxit=100, tol=1e-08, diis_size=6, shift=0.0, initial_guess=None, flag_RHF=False)`
:   Perform the ground-state CC(P) calculation in the adaptive-CC(P;Q)
    scheme aimed at converging the CCSDT energetics.
    
    Parameters
    ----------
    sys : dict
        System information
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    p_spaces : dict
        Holds the ndarrays that indicate the triples included in the P space for each spin case
    maxit : int, optional
        Maximum number of iterations for the CC(P) calculation. Default is 100.
    tol : float, optional
        Convergence tolerance on the changes in the T vector and CC energy. Default is 1.0e-08.
    diis_siize : int, optional
        Size of the inversion subspace used in DIIS convergence acceleration. Default is 6.
    shift : float, optional
        Value (in hartree) of the denominator shifting parameter used to converge difficult CC calculations.
        Default is 0.0.
    initial_guess : dict, optional
        Dictionary of T1, T2, and T3 amplitudes taken as the initial guess in the CC(P) calculation.
        If present, initial guess is used, otherwise, starting guess of T1 = T2 = T3 = 0 is used. 
        Default is None.
    flag_RHF : bool, optional
        Flag indicating whether closed-shell RHF symmetry should be used. Default is False.
    
    Returns
    -------
    cc_t : dict
        Contains the converged T1, T2, and T3 cluster amplitudes
    Eccp : float
        Total CC(P) energy

    
`get_ccs_intermediates(cc_t, ints, sys)`
:   Calculate the CCS-like similarity-transformed HBar intermediates (H_N e^T1)_C.
    
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

    
`get_ccsd_intermediates(cc_t, ints, sys)`
:   Calculate the CCSD-like similarity-transformed HBar intermediates (H_N e^(T1+T2))_C.
    
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

    
`update_t1a(cc_t, ints, sys, shift)`
:   Update t1a amplitudes by calculating the projection <ia|(H_N e^(T1+T2+T3))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)
    
    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2, and T3

    
`update_t1b(cc_t, ints, sys, shift)`
:   Update t1b amplitudes by calculating the projection <i~a~|(H_N e^(T1+T2+T3))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)
    
    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2, and T3

    
`update_t2a(cc_t, ints, H1A, H1B, H2A, H2B, H2C, sys, shift)`
:   Update t2a amplitudes by calculating the projection <ijab|(H_N e^(T1+T2+T3))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)
    
    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2, and T3

    
`update_t2b(cc_t, ints, H1A, H1B, H2A, H2B, H2C, sys, shift)`
:   Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^(T1+T2+T3))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)
    
    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2, and T3

    
`update_t2c(cc_t, ints, H1A, H1B, H2A, H2B, H2C, sys, shift)`
:   Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^(T1+T2+T3))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)
    
    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2, and T3

    
`update_t3a(cc_t, ints, p_space, H1A, H1B, H2A, H2B, H2C, sys, shift)`
:   Update t3a amplitudes by calculating the projection <ijkabc|(H_N e^(T1+T2+T3))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)
    
    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2, and T3

    
`update_t3b(cc_t, ints, p_space, H1A, H1B, H2A, H2B, H2C, sys, shift)`
:   Update t3b amplitudes by calculating the projection <ijk~abc~|(H_N e^(T1+T2+T3))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)
    
    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2, and T3

    
`update_t3c(cc_t, ints, p_space, H1A, H1B, H2A, H2B, H2C, sys, shift)`
:   Update t3c amplitudes by calculating the projection <ij~k~ab~c~|(H_N e^(T1+T2+T3))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)
    
    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2, and T3

    
`update_t3d(cc_t, ints, p_space, H1A, H1B, H2A, H2B, H2C, sys, shift)`
:   Update t3d amplitudes by calculating the projection <i~j~k~a~b~c~|(H_N e^(T1+T2+T3))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1, T2, and T3
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)
    
    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1, T2, and T3