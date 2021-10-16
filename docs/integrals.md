Module integrals
================

Functions
---------

    
`build_f(e1int, v, sys)`
:   This function generates the Fock matrix using the formula
    F = Z + G where G is \sum_{i} <pi|v|qi>_A split for different
    spin cases.
    
    Parameters
    ----------
    e1int : ndarray(dtype=float, shape=(norb,norb))
        Onebody MO integrals
    v : dict
        Twobody integral dictionary
    sys : dict
        System information dictionary
    
    Returns
    -------
    f : dict
        Dictionary containing the Fock matrices for the aa and bb cases

    
`build_v(e2int)`
:   Generate the antisymmetrized version of the twobody matrix.
    
    Parameters
    ----------
    e2int : ndarray(dtype=float, shape=(norb,norb,norb,norb))
        Twobody MO integral array
        
    Returns
    -------
    v : dict
        Dictionary with v['A'], v['B'], and v['C'] containing the
        antisymmetrized twobody MO integrals.

    
`get_integrals(onebody_file, twobody_file, sys, **kwargs)`
:   Get the dictionary of onebody and twobody integrals in
    the MO basis.
    
    Parameters
    ----------
    onebody_file : str
        Path to onebody integral file
    twobody_file : str
        Path to twobody integral file
    sys : dict
        System information dictionary
    kwargs : dict, optional
        Keyword dictionary with possible fields: 'mux_file', 'muy_file', and 'muz_file'
        for loading in dipole moment integrals
    
    Returns
    -------
    ints : dict
        Sliced F_N and V_N integrals defining the bare Hamiltonian H_N

    
`parse_onebody(filename, sys)`
:   This function reads the onebody.inp file from GAMESS
    and returns a numpy matrix.
    
    Parameters
    ----------
    filename : str
        Path to onebody integral file
    sys : dict
        System information dict
        
    Returns
    -------
    e1int : ndarray(dtype=float, shape=(norb,norb))
        Onebody part of the bare Hamiltonian in the MO basis (Z)

    
`parse_twobody(filename, sys)`
:   This function reads the twobody.inp file from GAMESS
    and returns a numpy matrix.
    
    Parameters
    ----------
    filename : str
        Path to twobody integral file
    sys : dict
        System information dict
        
    Returns
    -------
    e_nn : float
        Nuclear repulsion energy (in hartree)
    e2int : ndarray(dtype=float, shape=(norb,norb,norb,norb))
        Twobody part of the bare Hamiltonian in the MO basis (V)

    
`slice_onebody_ints(f, sys)`
:   Slice the onebody integrals and sort them by occ/unocc blocks.
    
    Parameters
    ----------
    f : dict
        AA and BB Fock matrices
    sys : dict
        System information dictionary
    
    Returns
    -------
    fA : dict
        Sliced Fock matrices for the A spincase
    fB : dict
        Sliced Fock matrices for the B spincase

    
`slice_twobody_ints(v, sys)`
:   Slice the twobody integrals and sort them by occ/unocc blocks.
    
    Parameters
    ----------
    f : dict
        AA and BB Fock matrices
    sys : dict
        System information dictionary
    
    Returns
    -------
    vA : dict
        Sliced twobody matrices for the AA spincase
    vB : dict
        Sliced twobody matrices for the AB spincase
    vC : dict
        Sliced twobody matrices for the BB spincase