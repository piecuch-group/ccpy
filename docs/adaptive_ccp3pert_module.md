Module adaptive_ccp3pert_module
===============================
Module to perform the CC(P;Q) correction to the CC(P) calculation
in the adaptive-CC(P;Q) scheme using the CCSD(T)-like correction.

Functions
---------

    
`build_MM23A(cc_t, ints)`
:   Calculate the projection <ijkabc|(V_N*T2)_C|0>.
    
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

    
`build_MM23B(cc_t, ints)`
:   Calculate the projection <ijk~abc~|(V_N*T2)_C|0>.
    
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

    
`build_MM23C(cc_t, ints)`
:   Calculate the projection <ij~k~ab~c~|(V_N*T2)_C|0>.
    
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

    
`build_MM23D(cc_t, ints)`
:   Calculate the projection <i~j~k~a~b~c~|(V_N*T2)_C|0>.
    
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

    
`ccp3_pertT(cc_t, p_spaces, ints, sys, flag_RHF=False)`
:   Calculate the ground-state CCSD(T)-like correction delta(P;Q)_(T) to 
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