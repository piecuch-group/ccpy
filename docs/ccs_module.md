Module ccs_module
=================
Module with functions that perform the CC with singles (CCS) calculation
for a molecular system. Note that when using Hartree-Fock orbitals, CCS
recovers no correlation energy and is generally meaningless.

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

    
`ccs(sys, ints, maxit=100, tol=1e-08, diis_size=6, shift=0.0)`
:   Perform the ground-state CCS calculation.
    
    Parameters
    ----------
    sys : dict
        System information dictionary
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    maxit : int, optional
        Maximum number of iterations for the CC calculation. Default is 100.
    tol : float, optional
        Convergence tolerance for the CC calculation. Default is 1.0e-08.
    diis_siize : int, optional
        Size of the inversion subspace used in DIIS convergence acceleration. Default is 6.
    shift : float, optional
        Value (in hartree) of the denominator shifting parameter used to converge difficult CC calculations.
        Default is 0.0.
    
    Returns
    -------
    cc_t : dict
        Contains the converged T1 cluster amplitudes
    Eccs : float
        Total CCA energy

    
`second_order_guess(sys, ints)`
:   Calculate the 2nd-order MBPT approximation of the T1 amplitudes.
    This is done because T1 is 0 at 1st-order, so initiating CC iterations
    with the 1st-order does not go anywhere.
    
    Parameters
    ----------
    sys : dict
        System information dictionary
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    
    Returns
    -------
    cc_t : dict
        Cluster amplitudes T1 at the 2nd-order MBPT approximation level
        Note: Currently, closed-shell RHF symmetry is used to enforce
        t1a = t1b

    
`update_t1a(cc_t, ints, sys, shift)`
:   Update t1a amplitudes by calculating the projection <ia|(H_N e^T1)_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)
    
    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1

    
`update_t1b(cc_t, ints, sys, shift)`
:   Update t1b amplitudes by calculating the projection <i~a~|(H_N e^T1)_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T1
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)
    
    Returns
    --------
    cc_t : dict
        New cluster amplitudes T1