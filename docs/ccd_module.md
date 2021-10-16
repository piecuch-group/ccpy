Module ccd_module
=================
Module with functions that perform the CC with doubles (CCD) calculation
for a molecular system.

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

    
`ccd(sys, ints, maxit=100, tol=1e-08, diis_size=6, shift=0.0)`
:   Perform the ground-state CCD calculation.
    
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
        Contains the converged T2 cluster amplitudes
    Eccd : float
        Total CCD energy

    
`update_t2a(cc_t, ints, sys, shift)`
:   Update t2a amplitudes by calculating the projection <ijab|(H_N e^T2)_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)
    
    Returns
    --------
    cc_t : dict
        New cluster amplitudes T2

    
`update_t2b(cc_t, ints, sys, shift)`
:   Update t2b amplitudes by calculating the projection <ij~ab~|(H_N e^T2)_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)
    
    Returns
    --------
    cc_t : dict
        New cluster amplitudes T2

    
`update_t2c(cc_t, ints, sys, shift)`
:   Update t2c amplitudes by calculating the projection <i~j~a~b~|(H_N e^T2)_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Current cluster amplitudes T2
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    shift : float
        Energy denominator shift (in hartree)
    
    Returns
    --------
    cc_t : dict
        New cluster amplitudes T2