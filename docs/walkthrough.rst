Specific Walkthroughs
###################

Here, we walkthrough some code for performing a few of the standard
CC calculations that are possible using CCpy. In all cases, we will 
assume that the :code:`Driver` object has been instantiated, either 
manually, or using one of the provided interfaces. This way, the 
system is specified within :code:`driver.system` and the molecular 
orbital integrals are loaded and available in :code:`driver.hamiltonian`.

Completely Renormalized (CR) CC Calculations
********************************
Here is an example of how to perform a CR-CC(2,3) calculation in order to 
obtain the nonperturbative correction to the CCSD energetics for the 
effects of missing triples using moment energy expansions. ::

        # Run CCSD calculation
        driver.run_cc(method="ccsd")
        # Obtain CCSD similarity-transformed Hamiltonian
        driver.run_hbar(method="ccsd")
        # Run left-CCSD calculation
        driver.run_leftcc(method="left_ccsd")
        # Run CR-CC(2,3) triples correction
        driver.run_ccp3(method="crcc23")

The above commands, which must be run in sequence, first performs the CCSD calculation in order
to obtain the converged :math:`T_{1}` and :math:`T_{2}` clusters and follows this
by computing the one- and two-body components of the CCSD similarity-transformed
Hamiltonan :math:`\bar{H}=(H_N e^{T_1+T_2})_C`. Note that the :code:`run_hbar` 
method replaces the bare one- and two-body molecular orbital integrals contained 
in :code:`driver.hamiltonian` with their similarity-transformed counterparts. 
Next, the :code:`driver.run_leftcc(method="left_ccsd")` method solves the companion left-CCSD system of 
linear equations to obtain the :math:`\Lambda_{1}` and :math:`\Lambda_{2}` operators,
which are needed to compute the CR-CC(2,3) triples correction using a single
noniterative step with :code:`driver.run_ccp3(method="crcc23")`. The CR-CC(2,3)
calculation returns four distinct energetics, labelled as CR-CC(2,3)\ :sub:`X`\, for 
X = A, B, C, and D, where each variant A-D corresponds to a different treatment of the energy 
denominator :math:`E^{(\text{CCSD})} - \langle \Phi_{ijk}^{abc} | \bar{H} | \Phi_{ijk}^{abc} \rangle`
entering the formula for the CR-CC(2,3) triples correction. Note that the variant using
the simplest Moeller-Plesset form of the energy denominator, denoted as CR-CC(2,3)\ :sub:`A`\, 
is equivalent to the method called CCSD(2)\ :sub:`T` \ and the 
result corresponding to CR-CC(2,3)\ :sub:`D`\, which employs the full Epstein-Nesbet 
energy denominator, is generally most accurate and often simply referred to as the 
CR-CC(2,3) energy (or by its former name, CR-CCSD(T)\ :sub:`L`\).

Active-Orbital-Based CC(*P*; *Q*) Calculations
********************************
Here is an example of how to perform a CC(t;3) calculation in order to 
obtain the nonperturbative correction to the CCSDt energetics for the 
effects of missing triples using the CC(P;Q) moment expansions. ::

        # Set number of occupied and unoccupied orbitals that are active
        nacto = 2
        nactu = 2
        driver.system.set_active_space(nact_occupied=nacto, nact_unoccupied=nactu)

        # Run active-space CCSDt calculation
        driver.run_cc(method="ccsdt1")
        # Obtain CCSD-like similarity-transformed Hamiltonian
        driver.run_hbar(method="ccsd")
        # Run left-CCSD calculation
        driver.run_leftcc(method="left_ccsd")
        # Run CC(t;3) correction
        driver.run_ccp3(method="cct3")

Compared to the above code used to perform CR-CC(2,3), the main difference in CC(t;3) is that the lower-order CCSD
calculation is replaced with the improved active-space CCSDt approach. In order to run
CCSDt, a contiguous set of active orbitals around the Fermi level must be specified using
the :code:`driver.system.set_active_space` method. Here, we are arbitrarily setting up a (2,2)
active space, but the reasonable choice of active orbitals is obviously very dependent on the
specifics of the problem in consideration. This particular example of CC(t;3) relies on what
is known as the two-body approximation, which employs the CR-CC(2,3)-like expressions to formulate
the correction for missing triples to the CCSDt energetics. As a result, the components of the
similarity-transformed Hamiltonian as well as the left-CC deexcitation operator are obtained
with CCSD-like calculations that avoid directly using the of active-orbital-based :math:`t_3`
clusters in the resulting expressions, and instead indirectly incorporates their effects 
by using :math:`T_1` and :math:`T_2` clusters provided by CCSDt, which are
relaxed in the presence of :math:`t_3`, and are more accurate than the one- and two-body 
components of the cluster operator obtained in CCSD calculations. 

Using the more general CC(P) and CC(P;Q) routines, one can also perform the most complete CC(t;3)
correction that does not invoke the two-body approximation. ::

        # Set number of occupied and unoccupied orbitals that are active
        nacto = 2
        nactu = 2
        driver.system.set_active_space(nact_occupied=nacto, nact_unoccupied=nactu)

        # Import and use the helper routine to build the P space corresponding to the choice of active orbitals
        from ccpy.utilities.pspace import get_active_space
        t3_excitations, pspace = get_active_pspace(driver.system, target_irrep=driver.system.reference_symmetry)

        # Run the active-space CCSDt calculation using the general CC(P) solver
        driver.run_ccp(method="ccsdt_p", t3_excitations=t3_excitations)
        # Obtain the CCSDt similarity-transformed Hamiltonian
        driver.run_hbar(method="ccsdt_p", t3_excitations=t3_excitations)
        # Run the left-CCSDt calculation
        driver.run_leftccp(method="left_ccsdt_p", t3_excitations=t3_excitations)
        # Run the complete CC(t;3) correction
        driver.run_ccp3(method="ccp3", state_index=0, t3_excitations=t3_excitations, two_body_approx=False)

Here, the list of triply excited determinants that enter the P space corresponding to the choice of a (2,2) 
orbital active space is constructed with the help of the :code:`get_active_pspace` function, returning
a list of spin-orbital triples excitations stored in the 2D Numpy array called :code:`t3_excitations`. This 
list of triply excited determinants entering the P space can also be constrained to include only those belonging
to a specific symmetry irrep, which is specified using the keyword :code:`target_irrep`.
For ground-state calculations of molcules described using an Abelian point group, the target irrep
should correspond to the symmetry of the reference wave function, which is stored in the attribute
:code:`driver.system.reference_symmetry` (for closed-shell systems, this should correspond to the totally
symmetric representation, however, this may not be the case for open shells). Using the P space containing
all singles and doubles ad the list of triples excitations contained in :code:`t3_excitations`, the CC(P)
calculations can be performed using the general :code:`driver.run_ccp` method. This is followed by the running
the :code:`driver.run_hbar` and :code:`driver.run_leftccp` routines, which are used to compute the CCSDt 
similarity-transformed Hamiltonian and solve the left-CCSDt system of linear equations, respectively.
Finally, the complete CC(t;3) correction is computed using the :code:`driver.run_ccp3` routine, where it 
is important to set the flag :code:`two_body_appox=False` (by default, it is set to :code:`True`). Unlike
the CC(t;3) calculations employing the two-body approximation, which takes advantage of CCSD-like routines
that incorporate the effects of the active-space :math:`t_3` clusters indirectly through relaxation of the 
:math:`T_1` and :math:`T_2` components, the full CC(t;3) correction requires constructing the corresponding 
CCSDt similarity-transformed Hamiltonian using the :math:`T_1`, :math:`T_2`, and :math:`t_3` operators and 
solving for the companion one-, two-, and three-body components of the left-CCSDt deexcitation operator. 
As a result, the full CC(t;3) correction provides energetics that are more accurate than its counterpart 
employing the two-body approximation, but with computational costs that are larger by roughly a factor of 
2-3. For excited-state applications of CC(t;3), the improvement in the energetics often justifies 
this increase in computational effort. On the other hand, the two-body approximation is often sufficient for 
ground-state CC(t;3) calaculations, with previous applications demonstrating its ability to converge the parent
CCSDT energetics to within a fraction of a millihartree, even in challenging situations featuring stronger 
many-electron correlation effects where :math:`T_3` clusters become large and nonperturbative.

Adaptive CC(*P*; *Q*) Calculations
********************************

CIPSI-Driven CC(*P*; *Q*) Calculations
********************************

CIPSI-Based Externally Corrected (ec) CC Calculations
********************************