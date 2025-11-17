#####################
Computational Options
#####################

Here, we provide sample code to get started with running any of the computational options
available in CCpy. Before executing any correlated CC/EOMCC steps, the Hartree-Fock
mean field solution and transformed one- and two-electron integrals must be provided via
an external source. Currently, this can be done in one of three ways:

1) PySCF

CCpy is fully interfaced with PySCF and can build the Driver object out of a PySCF mean field. This
approach is arguably the most convenient one since the Hartree-Fock calculation can be run using PySCF
within the same script. ::

    from pyscf import gto, scf
    from ccpy import Driver

    # Set up geometry for symmetrically stretched H2O
    WATER = [["O", (0.0, 0.0, -0.0180)],
             ["H", (0.0, 3.030526, -2.117796)],
             ["H", (0.0, -3.030526, -2.117796)]]
    # Create PySCF molecule object
    mol = gto.M(
        atom=WATER,
        basis="cc-pvdz",
        charge=0,
        spin=0,
        symmetry="C2V",
        cart=True,
        unit="Bohr",
    )
    # Create PySCF RHF mean field object
    mf = scf.RHF(mol)
    # Run RHF
    mf.kernel()

    # Now set up the CCpy driver object using the PySCF mean field
    driver = Driver.from_pyscf(mf, nfrozen=1)
    # Print the system information
    driver.system.print_info()

2) GAMESS

CCpy can read system information and extract transformed one- and two-electron integrals from a completed
GAMESS calculation by passing in the locations of the GAMESS output logfile (``gms_logfile``) and
companion FCIDUMP (``gms_fcidump``). GAMESS can be used to generate FCIDUMP files for RHF and ROHF
calculations using the ``runtyp=fcidump`` option.

`Important Note: The FCIDUMP option in GAMESS has a bug for high-spin ROHF references. The number of
electrons (NELEC) and number of unpaired electrons (MS2) will have incorrect values upon output. Therefore,
to use FCIDUMP files corresponding to ROHF references, one should manually change the NELEC and MS2
fields to their proper values.` ::

    from ccpy import Driver
    # Set up CCpy driver object using GAMESS logfile (gms_logfile) and FCIDUMP (gms_fcidump)
    driver = Driver.from_gamess(gms_logfile,
                                gms_fcidump,
                                nfrozen=0)
    # Print the system information
    driver.system.print_info()

3) FCIDUMP

The most general way to pass in information about a mean field is through a single FCIDUMP file. For ROHF
references, it is a good idea to specify the appropriate canonicalization scheme so that the molecular orbital
energies are output correctly (the ROHF single-particle energies are not actually used in correlated computations,
so in practice, this is optional). If no canonicalization is provided, CCpy will default to using ``Guest-Saunders``,
which is most common, but this is not always the correct choice. For example, when using ROHF FCIDUMP files generated
with GAMESS under default settings, the correct canonicalization scheme is ``Roothaan``.

`Important Note: Currently, CCpy does not use the spatial point group symmetry information contained in the FCIDUMP file. This
will be changed soon.` ::

    from ccpy import Driver
    # Set up CCpy driver object using an FCIDUMP file
    driver = Driver.from_fcidump(fcidump, nfrozen=0, charge=0, rohf_canonicalization="Roothaan")
    # Print the system information
    driver.system.print_info()

CCD
---
Sample code ::

    from ccpy import Driver

    # Run CCD calculation
    driver.run_cc(method="ccd")

CCSD
----
Sample code ::

    from ccpy import Driver

    # Run CCSD calculation
    driver.run_cc(method="ccsd")

CCSD(T)
-------
Sample code ::

    from ccpy import Driver

    # Run CCSD calculation
    driver.run_cc(method="ccsd")
    # Run CCSD(T) triples correction to CCSD energetics
    driver.run_ccp3(method="ccsdpt")

CC3
---
Sample code ::

    from ccpy import Driver

    # Run CC3 calculation
    driver.run_cc(method="cc3")

CCSDt
-----
Sample code ::

    from ccpy import Driver, get_active_triples_space

    # Choose the active space for the problem. Here, we are using (2,2).
    driver.system.set_active_space(nact_occupied=2, nact_unoccupied=2)
    # Obtain the list of triples excitations corresponding to the CCSDt truncation (ground-state symmetry adapted)
    t3_excitations = get_active_triples_space(driver.system, target_irrep=driver.system.reference_symmetry)

    # Run active-space CCSDt calculation via general CC(P) solver
    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)

CCSDT
-----
Sample code ::

    from ccpy import Driver

    # Run CCSDT calculation
    driver.run_cc(method="ccsdt")

Alternatively, full CCSDT calculations are also available by running active-orbital-based CCSDt with full active space.
The advantage of this approach is that it allows for point group symmetry-adapted CCSDT runs. ::

    from ccpy import Driver, get_active_triples_space

    # Choose the active space for the problem. Here, we are using a full active space.
    driver.system.set_active_space(nact_occupied=driver.system.noccupied_alpha, nact_unoccupied=driver.system.nunoccupied_beta)
    # Obtain the list of triples excitations corresponding to the CCSDt truncation (ground-state symmetry adapted)
    t3_excitations = get_active_triples_space(driver.system, target_irrep=driver.system.reference_symmetry)

    # Run full CCSDT calculation via general CC(P) solver
    driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)

CC4
----
`Note: CC4 is available for closed-shell references only!`

Sample code ::

    from ccpy import Driver

    # Run CC4 calculation
    driver.run_cc(method="cc4")

CCSDTQ
------
`Note: CCSDTQ is available for closed-shell references only!`

Sample code ::

    from ccpy import Driver

    # Run CCSDTQ calculation
    driver.run_cc(method="ccsdtq")

EOMCCSD
-------
Sample code ::

    # Run the ground-state CCSD calculation
    driver.run_cc(method="ccsd")
    # Compute and store the CCSD similarity-transformed Hamiltonian (this will overwrite the bare integrals in driver.hamiltonian)
    driver.run_hbar(method="ccsd")
    # Perform an initial CI-like diagonalization to obtain guess vectors
    driver.run_guess(method="cis", multiplicity=1, roots_per_irrep={"A1": 3, "B1": 2, "B2": 2, "A2": 0})
    # Run the EOMCCSD calculation for the specified states. The values `state_index` map one-to-one
    # with the guess vectors, so in this example,
    # states 1, 2, 3 -> A1
    # states 4, 5 -> B1
    # states 6, 7 -> B2
    driver.run_eomcc(method="eomccsd", state_index=[1, 2, 3, 4, 5, 6, 7])

EOM-CC3
-------
Sample code ::

    #
    driver.run_cc(method="cc3")
    driver.run_hbar(method="cc3")
    driver.run_guess(method="cisd", roots_per_irrep={"A1": 3, "B1": 3, "B2": 2, "A2": 1}, multiplicity=1, nact_occupied=2, nact_unoccupied=4)
    driver.run_eomcc(method="eomcc3", state_index=[1, 2, 3, 4, 5, 6, 7, 8, 9])

EOMCCSDT(a)*
------------
Sample code ::

    # Run ground-state CC calculation
    driver.run_cc(method="ccsd")
    # Obtain the CCSD(T)(a) similarity-transformed Hamiltonian
    driver.run_hbar(method="ccsdta")
    # Run EOMCCSD-like calculation using CCSD(T)(a) HBar
    driver.run_guess(method="cisd", multiplicity=1, roots_per_irrep={"A1": 4, "B1": 2, "B2": 0, "A2": 2},  nact_occupied=3, nact_unoccupied=7)
    driver.run_eomcc(method="eomccsd", state_index=[2, 3, 4, 5, 6, 7, 8])
    # Obtain the left eigenstates for each EOMCC root
    driver.run_lefteomcc(method="left_ccsd", state_index=[2, 3, 4, 5, 6, 7, 8])
    # Compute EOMCCSDT(a)* excited-state corrections
    driver.run_ccp3(method="eomccsdta_star", state_index=[0, 2, 3, 4, 5, 6, 7, 8])

EOMCCSDt
--------

EOMCCSDT
--------

IP-EOMCCSD(2h-1p)
-----------------

IP-EOMCCSD(3h-2p)
-----------------

IP-EOMCCSDT(a)*
---------------

EA-EOMCCSD(2p-1h)
-----------------

EA-EOMCCSD(3p-2h)
-----------------

EA-EOMCCSDT(a)*
---------------

DEA-EOMCCSD(2p)
---------------

DEA-EOMCCSD(3p-1h)
------------------

DEA-EOMCCSD(4p-2h)
------------------

SF-EOMCCSD
----------

CR-CC(2,3)
----------

CR-CC(2,4)
----------

CC(t;3)
-------

CIPSI-driven CC(*P* ;\ *Q*) aimed at converging CCSDT
-----------------------------------------------------

Adaptive CC(*P* ;\ *Q*) aimed at converging CCSDT
-------------------------------------------------

CR-EOMCC(2,3) and :math:`\delta`-CR-EOMCC(2,3)
----------------------------------------------

ec-CC-II
--------

ec-CC-II\ :sub:`3`
------------------

ec-CC-II\ :sub:`3,4`
--------------------
