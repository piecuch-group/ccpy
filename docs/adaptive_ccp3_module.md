Module adaptive_ccp3_module
===========================
Module to perform the CC(P;Q) correction to the CC(P) calculation
in the adaptive-CC(P;Q) scheme using the two-body approximation for
the moment-based CR-CC(2,3)-like correction.

Functions
---------

    
`build_L3A(cc_t, H1A, H2A, ints, sys, iroot=0)`
:   Calculate the projection <0|(L1+L2)(H_N e^(T1+T2))_C|ijkabc>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2 and left amplitudes L1, L2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD-like HBar integrals (H_N e^(T1+T2))_C
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

    
`build_L3B(cc_t, H1A, H1B, H2A, H2B, ints, sys, iroot=0)`
:   Calculate the projection <0|(L1+L2)(H_N e^(T1+T2))_C|ijk~abc~>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2 and left amplitudes L1, L2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD-like HBar integrals (H_N e^(T1+T2))_C
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

    
`build_L3C(cc_t, H1A, H1B, H2B, H2C, ints, sys, iroot=0)`
:   Calculate the projection <0|(L1+L2)(H_N e^(T1+T2))_C|ij~k~ab~c~>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2 and left amplitudes L1, L2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD-like HBar integrals (H_N e^(T1+T2))_C
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

    
`build_L3D(cc_t, H1B, H2C, ints, sys, iroot=0)`
:   Calculate the projection <0|(L1+L2)(H_N e^(T1+T2))_C|i~j~k~a~b~c~>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2 and left amplitudes L1, L2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD-like HBar integrals (H_N e^(T1+T2))_C
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

    
`build_MM23A(cc_t, H1A, H2A, sys)`
:   Calculate the projection <ijkabc|(H_N e^(T1+T2))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD-like HBar integrals (H_N e^(T1+T2))_C
    sys : dict
        System information dictionary
        
    Returns
    -------
    MM23A : ndarray(dtype=float, shape=(nua,nua,nua,noa,noa,noa))
        Array containing the MM(2,3)A projections for each i,j,k,a,b,c

    
`build_MM23B(cc_t, H1A, H1B, H2A, H2B, sys)`
:   Calculate the projection <ijk~abc~|(H_N e^(T1+T2))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD-like HBar integrals (H_N e^(T1+T2))_C
    sys : dict
        System information dictionary
        
    Returns
    -------
    MM23B : ndarray(dtype=float, ndim=shape(nua,nua,nob,noa,noa,nob))
        Array containing the MM(2,3)B projections for each i,j,k~,a,b,c~

    
`build_MM23C(cc_t, H1A, H1B, H2B, H2C, sys)`
:   Calculate the projection <ij~k~ab~c~|(H_N e^(T1+T2))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD-like HBar integrals (H_N e^(T1+T2))_C
    sys : dict
        System information dictionary
        
    Returns
    -------
    MM23C : ndarray(dtype=float, shape=(nua,nub,nub,noa,nob,nob))
        Array containing the MM(2,3)C projections for each i,j~,k~,a,b~,c~

    
`build_MM23D(cc_t, H1B, H2C, sys)`
:   Calculate the projection <i~j~k~a~b~c~|(H_N e^(T1+T2))_C|0>.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    H1*, H2* : dict
        Sliced similarity-transformed CCSD-like HBar integrals (H_N e^(T1+T2))_C
    sys : dict
        System information dictionary
        
    Returns
    -------
    MM23D : ndarray(dtype=float, shape=(nub,nub,nub,nob,nob,nob))
        Array containing the MM(2,3)D projections for each i~,j~,k~,a~,b~,c~

    
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

    
`ccp3(cc_t, p_spaces, H1A, H1B, H2A, H2B, H2C, ints, sys, flag_RHF=False)`
:   Calculate the ground-state CR-CC(2,3)-like correction delta(P;Q) to 
    the CC(P) calculation defined by the P space contained in p_spaces and 
    its coresponding T vectors and HBar matrices. The two-body approximation
    is used.
    
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

    
`triples_3body_diagonal(cc_t, ints, sys)`
:   Calculate the triples diagonal <ijkabc|H3|ijkabc>, where H3
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