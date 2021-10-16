Module HBar_module
==================

Functions
---------

    
`HBar_CCSD(cc_t, ints, sys)`
:   Calculate the CCSD similarity-transformed HBar integrals (H_N e^(T1+T2))_C.
    
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

    
`HBar_CCSDT(cc_t, ints, sys)`
:   Calculate the CCSDT similarity-transformed HBar integrals (H_N e^(T1+T2+T3))_C.
    
    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2, and T3
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

    
`test_HBar(matfile, ints, sys)`
:   Test the HBar integrals using known results from Matlab code.
    
    Parameters
    ----------
    matfile : str
        Path to .mat file containing T1, T2 amplitudes from Matlab
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N
    sys : dict
        System information dictionary
    
    Returns
    -------
    None