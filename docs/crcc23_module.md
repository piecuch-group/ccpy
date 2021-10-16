Module crcc23_module
====================
Module containing functions that calculate the CR-CC(2,3) and CR-EOMCC(2,3)
triples corrections to the ground- and excited-state energetics obtained at the CCSD
and EOMCCSD levels, respectively.
Note: For CR-EOMCC(2,3), closed-shell RHF symmetry is used because C and D projections
      are not put in yet.

Functions
---------

    
`build_EOM_MM23A(cc_t, H2A, H2B, iroot, sys)`
:   Calculate the projection <ijkabc|[ (H_N e^(T1+T2))_C (R1+R2) ]_C|0>.
    
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

    
`build_EOM_MM23B(cc_t, H2A, H2B, H2C, iroot, sys)`
:   Calculate the projection <ijk~abc~|[ (H_N e^(T1+T2))_C (R1+R2) ]_C|0>.
    
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

    
`build_EOM_MM23C(cc_t, H2A, H2B, H2C, iroot, sys)`
:   Calculate the projection <ij~k~ab~c~|[ (H_N e^(T1+T2))_C (R1+R2) ]_C|0>.
    
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

    
`build_L3A(cc_t, H1A, H2A, ints, sys, iroot=0)`
:   Calculate the projection <0|(L1+L2)(H_N e^(T1+T2))_C|ijkabc>.
    
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

    
`build_L3B(cc_t, H1A, H1B, H2A, H2B, ints, sys, iroot=0)`
:   Calculate the projection <0|(L1+L2)(H_N e^(T1+T2))_C|ijk~abc~>.
    
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

    
`build_L3C(cc_t, H1A, H1B, H2B, H2C, ints, sys, iroot=0)`
:   Calculate the projection <0|(L1+L2)(H_N e^(T1+T2))_C|ij~k~ab~c~>.
    
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

    
`build_L3D(cc_t, H1B, H2C, ints, sys, iroot=0)`
:   Calculate the projection <0|(L1+L2)(H_N e^(T1+T2))_C|i~j~k~a~b~c~>.
    
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

    
`build_MM23A(cc_t, H1A, H2A, sys)`
:   Calculate the projection <ijkabc|(H_N e^(T1+T2))_C|0>.
    
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

    
`build_MM23B(cc_t, H1A, H1B, H2A, H2B, sys)`
:   Calculate the projection <ijk~abc~|(H_N e^(T1+T2))_C|0>.
    
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

    
`build_MM23C(cc_t, H1A, H1B, H2B, H2C, sys)`
:   Calculate the projection <ij~k~ab~c~|(H_N e^(T1+T2))_C|0>.
    
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

    
`build_MM23D(cc_t, H1B, H2C, sys)`
:   Calculate the projection <i~j~k~a~b~c~|(H_N e^(T1+T2))_C|0>.
    
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

    
`crcc23(cc_t, H1A, H1B, H2A, H2B, H2C, ints, sys, flag_RHF=False, nroot=0, omega=0.0)`
:   Calculate the ground-/excited-state CR-CC(2,3)/CR-EOMCC(2,3) corrections
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

    
`test_updates(matfile, ints, sys)`
:   Test the CR-CC(2,3) updates using known results from Matlab code.
    
    Parameters
    ----------
    matfile : str
        Path to .mat file containing T1, T2 and L1, L2 amplitudes from Matlab
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    
    Returns
    -------
    None

    
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