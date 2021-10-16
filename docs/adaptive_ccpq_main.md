Module adaptive_ccpq_main
=========================
Module containing the main driver functions for the adaptive-CC(P;Q)
approach aimed at converging the CCSDT energetics. Contains functions to
perform the relaxed and unrelaxed variants and may use either the 
CR-CC(2,3)-like corrections under the two-body approximation or the
perturbative CCSD(T)-like corrections to grow the P spaces.

Functions
---------

    
`calc_adaptive_ccpq(sys, ints, tot_triples, workdir, ccshift=0.0, lccshift=0.0, maxit=10, growth_percentage=1.0, restart_dir=None, niter0=0, isRHF=False, ccmaxit=100, tol=1e-08, diis_size=6, flag_pert_triples=False, flag_save=False)`
:   Perform the relaxed adaptive-CC(P:Q) calculation aimed at converging the CCSDT energetics.
    
    Parameters
    ----------
    sys : dict
        System information dictionary
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    tot_triples : int
        Total number of triples belonging to symmetry of the ground state
    workdir : str
        Path to working directory where output files will be stored
    ccshift : float, optional
        Energy denominator shift value (in hartree) used to help converge the CC(P) calculations. Default value is 0.0.
    lccshift : float, optional
        Energy denominator shift value (in hartree) used to help converge the left-CC(P) calculations. Default value is 0.0.
    maxit : int, optional
        Number of adaptive-CC(P;Q) growth iterations. Default is 10.
    growth_percentage : float, optional
        Percentage of triples to add into the P space at each adaptive-CC(P;Q) growth step. Default is 1.
    restart_dir : str, optional
        Directory containing the T1 and P space files stored in .npy format used to restart the relaxed adaptive-CC(P;Q)
        calculation. If provided, these files will be looked for. Default is None (no restarting).
    isRHF : bool, optional
        Flag to indicate whether closed-shell RHF symmetry should be used. Default is False.
    ccmaxit : int, optional
        Maximum number of iterations allowed for the CC(P) and left-CC(P) calculations. Default is 100.
    tol : float, optional
        Convergence tolerance for the CC(P) and left-CC(P) calculations. Default is 1.0e-08.
    diis_size : int, optional
        Size of inversion subspace used in DIIS acceleration for CC(P) and left-CC(P) calculations. Default is 6.
    flag_pert_triples : bool, optional
        Flag to indicate whether the perturbative CCSD(T)-like correction should be used in the adaptive-CC(P;Q) procedure. If False,
        the more robust CR-CC(2,3)-like corrections are used. Default is False.
    flag_save : bool, optional
        Flag to indicate whether outputs of T vectors, L vectors, and P spaces should be saved for each percentage. 
        Quantities are stored as .npy file (they are large!) and will be placed in workdir. Default is False.
    
    Returns
    -------
    Eccp : ndarray(dtype=float, shape=(maxit))
        Array of total CC(P) energies for each relaxed adaptive-CC(P;Q) iteration.
    Eccpq : ndarray(dtype=float, shape=(maxit))
        Array of total CC(P;Q) energies for each relaxed adaptive-CC(P;Q) iteration.

    
`calc_adaptive_ccpq_depreciated(sys, ints, tot_triples, workdir, ccshift=0.0, lccshift=0.0, maxit=10, growth_percentage=1.0, restart_dir=None, niter0=0, isRHF=False, ccmaxit=100, tol=1e-08, diis_size=6, flag_save=False)`
:   DEPRECIATED VERSION OF THE RELAXED ADAPTIVE-CC(P;Q) ALGORITHM. IT IS CORRECT, BUT DO NOT USE!

    
`calc_adaptive_ccpq_norelax(sys, ints, tot_triples, workdir, triples_percentages, ccshift=0.0, lccshift=0.0, isRHF=False, ccmaxit=100, tol=1e-08, diis_size=6, flag_pert_triples=False, flag_save=False)`
:   Perform the unrelaxed adaptive-CC(P:Q) calculation aimed at converging the CCSDT energetics.
    
    Parameters
    ----------
    sys : dict
        System information dictionary
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    tot_triples : int
        Total number of triples belonging to symmetry of the ground state
    workdir : str
        Path to working directory where output files will be stored
    triples_percentages : list
        List of the integer percentages of triples that will be included in the P space for the CC(P;Q) calculations
    ccshift : float, optional
        Energy denominator shift value (in hartree) used to help converge the CC(P) calculations. Default value is 0.0.
    lccshift : float, optional
        Energy denominator shift value (in hartree) used to help converge the left-CC(P) calculations. Default value is 0.0.
    isRHF : bool, optional
        Flag to indicate whether closed-shell RHF symmetry should be used. Default is False.
    ccmaxit : int, optional
        Maximum number of iterations allowed for the CC(P) and left-CC(P) calculations. Default is 100.
    tol : float, optional
        Convergence tolerance for the CC(P) and left-CC(P) calculations. Default is 1.0e-08.
    diis_size : int, optional
        Size of inversion subspace used in DIIS acceleration for CC(P) and left-CC(P) calculations. Default is 6.
    flag_pert_T : bool, optional
        Flag to indicate whether the perturbative CCSD(T)-like correction should be used in the adaptive-CC(P;Q) procedure. If False,
        the more robust CR-CC(2,3)-like corrections are used. Default is False.
    flag_save : bool, optional
        Flag to indicate whether outputs of T vectors, L vectors, and P spaces should be saved for each percentage. 
        Quantities are stored as .npy file (they are large!) and will be placed in workdir. Default is False.
    
    Returns
    -------
    Eccp : ndarray(dtype=float, shape=(len(triples_percentages))
        Array of total CC(P) energies for each requested triples percentage
    Eccpq : ndarray(dtype=float, shape=(len(triples_percentages))
        Array of total CC(P;Q) energies for each requested triples percentage

    
`count_triples_in_P(p_spaces)`
:   Count the triples in the P space.
    
    Parameters
    ----------
    p_spaces : dict
        Triples included in the P spaces for each spin case (A - D)
    
    Returns
    -------
    num_triples : int
        Number of triples contained in the P space

    
`selection_function(sys, mcA, mcB, mcC, mcD, p_spaces, num_add, flag_RHF=False)`
:   Select the specified number of triples from the Q space to be included 
    in the P space by choosing those with the largest magntiude of 
    moment-based corrections.
       
    Parameters
    ----------
    sys : dict
        System information dictionary
    mcA : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Individual CC(P;Q)_D corrections for each triple |ijkabc> in both P and Q spaces
    mcB : ndarray(dtype=float, shape=(nua,nua,nub,noa,noa,nob))
        Individual CC(P;Q)_D corrections for each triple |ijk~abc~> in both P and Q spaces
    mcC : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Individual CC(P;Q)_D corrections for each triple |ij~k~ab~c~> in both P and Q spaces
    mcD : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Individual CC(P;Q)_D corrections for each triple |i~j~k~a~b~c~> in both P and Q spaces
    p_spaces : dict
        Triples included in the P spaces for each spin case (A - D)
    num_add : int
        Number of triples from the Q space to add into the P space
    flag_RHF : bool, optional
        Flag to indicate whether closed-shell RHF symmetry should be used. Default is False.