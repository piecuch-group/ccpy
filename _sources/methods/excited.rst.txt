Excitation, attachment, ionization, ADC and STEOM
=================================================

Energy and sector conventions
-----------------------------

All EOM methods return target **total energies** from ``energy()``. ``omega`` is
the raw eigenvalue relative to the parent determinant's correlated ground state:

.. math::

   \omega = E_{\mathrm{target}}-E_{\mathrm{parent}}.

EE and SF preserve electron number; IP/DIP remove one/two electrons and EA/DEA
add one/two. Positive IP roots are ionization potentials. The conventional
positive electron affinity is ``-omega`` for attachment. For FNO-completed
energies use ``completed_omega``; see :doc:`spaces`.

A neutral EOM example, starting from native SCF ``mf``:

.. code-block:: python

   from ccpy import CCSD, EOMCCSD

   cc = CCSD(mf, nfrozen=1)
   eom = EOMCCSD(cc)
   eom.initial_guess(method="cisd", multiplicity=1, nacto=2, nactu=2,
                     roots_per_irrep={"A": 4}, roots=[0, 2])
   totals = eom.energy()
   first_returned_total = eom.energy(root=0)
   print(eom.omega, eom.root_irreps, eom.root_converged)

EOM raw totals are ``reference_energy + hbar.correlation_energy + omega``.
For ``HBarCCSDTA`` the HBar owns an augmented correlation energy and new
amplitudes, distinct from the original CCSD parent. Consequently
``completed_omega = energy() - cc.energy()`` can include that parent-energy
change even without FNOs. Use the HBar correlation energy when you specifically
want a gap relative to the augmented parent, rather than assuming every
``omega`` uses the original CCSD energy zero.

The same method may be constructed directly from ``mf, nfrozen=1`` to create
its standard parent automatically. Ordinary EOM requires an explicit
``initial_guess`` call. See :doc:`guesses` for spin, symmetry and the difference
between guess indices and returned-root indices.

Neutral EOM catalog
-------------------

.. list-table:: Standard EE chains
   :header-rows: 1
   :widths: 23 28 49

   * - Class
     - Default parent / HBar
     - Excited-state model
   * - ``EOMCC2``
     - ``CC2`` / ``HBarCC2``
     - Approximate second-order EE model.
   * - ``EOMCCSD``
     - ``CCSD`` / ``HBarCCSD``
     - Full R1+R2; native DF equations.
   * - ``EOMCC3``
     - ``CC3`` / ``HBarCC3``
     - Nonlinear omega-dependent DIIS solver, including when multiroot is requested.
   * - ``EOMCCSDT``
     - ``CCSDT`` / ``HBarCCSDT``
     - Full R1+R2+R3.
   * - ``EOMCCSDT_P``
     - ``CCSDT_P`` / ``HBarCCSDT_P``
     - Full R1/R2 and selected R3; separate ground and excited lists.
   * - ``EOMCCSDTA``
     - ``CCSD`` / ``HBarCCSDTA``
     - SD right states on a CCSD(T)(a) HBar; intended parent for ``EOMCCSDTA_star``.

Except for DF EOMCCSD, these wrappers currently materialize their equation
integrals. ``r0`` and ``rel`` are neutral EOM diagnostics: the reference
component and relative excitation level. They are not transition strengths.

Charged EOM catalog
-------------------

The numeral in ``IPEOM3`` or ``EAEOM3`` describes the largest hole/particle
sector in R; it does not automatically change the ground-state parent from
CCSD to CCSDT. The explicitly named ``...CCSDT`` variants do change that default.

.. list-table:: IP/EA excitation spaces
   :header-rows: 1
   :widths: 26 24 50

   * - Classes
     - R sectors
     - Standard parent and notes
   * - ``IPEOM2``
     - 1h + 2h1p
     - CCSD, legacy CCSD HBar; default IP-CISd guess.
   * - ``IPEOM3``
     - 1h + 2h1p + 3h2p
     - CCSD, legacy CCSD HBar.
   * - ``IPEOM3_P``
     - selected 3h2p
     - CCSD; lower R sectors unrestricted.
   * - ``IPEOMCCSDT``
     - through 3h2p
     - CCSDT and CCSDT HBar.
   * - ``IPEOMCCSDT_P``
     - selected 3h2p
     - CCSDT(P) and selected-triples HBar; ground/excited spaces distinct.
   * - ``IPEOMCCSDTA``
     - through 3h2p
     - CCSD, CCSD(T)(a) HBar.
   * - ``EAEOM2``
     - 1p + 2p1h
     - CCSD, legacy CCSD HBar; default EA-CISd guess.
   * - ``EAEOM3``
     - 1p + 2p1h + 3p2h
     - CCSD, legacy CCSD HBar.
   * - ``EAEOM3_P``
     - selected 3p2h
     - CCSD; lower R sectors unrestricted.
   * - ``EAEOMCCSDT``
     - through 3p2h
     - CCSDT and CCSDT HBar.

.. code-block:: python

   from ccpy import IPEOM2, EAEOM2

   ip = IPEOM2(cc)
   ip.initial_guess(method="ipcisd", nacto=2, nactu=2,
                    roots_per_irrep={"A": 2})
   ionized_totals = ip.energy()
   ionization_potentials = ip.completed_omega

   ea = EAEOM2(cc)
   ea.initial_guess(method="eacisd", nacto=2, nactu=2,
                    roots_per_irrep={"A": 2})
   attached_totals = ea.energy()
   electron_affinities = -ea.completed_omega

IP/EA wrappers use the existing alpha-removal/attachment operator branch.
For closed-shell parents, spin-equivalent partners need not be solved separately;
for open-shell parents this is a specific spin projection, not all possible
charged spin sectors. Default charged guesses use ``multiplicity=-1``.

Double attachment/removal and spin flip
---------------------------------------

.. list-table:: Other EOM families
   :header-rows: 1
   :widths: 25 28 47

   * - Classes
     - R model
     - Standard setup
   * - ``DIPEOM3``
     - 2h + 3h1p
     - CCSD / legacy HBar; ``dipcis`` guess.
   * - ``DIPEOM4``
     - through 4h2p
     - CCSD / legacy HBar.
   * - ``DIPEOMCCSDT``
     - through 4h2p
     - CCSDT / CCSDT HBar.
   * - ``DIPEOMCCSDTA``
     - through 4h2p
     - CCSD / CCSD(T)(a) HBar.
   * - ``DEAEOM2``
     - 2p
     - CCSD / legacy HBar; ``deacis`` guess.
   * - ``DEAEOM3``
     - 2p + 3p1h
     - CCSD / legacy HBar.
   * - ``DEAEOM4``
     - through 4p2h
     - CCSD / legacy HBar.
   * - ``DEAEOM4_P``
     - selected 4p2h
     - CCSD; lower R sectors unrestricted.
   * - ``SFEOMCCSD``
     - single-spin-flip singles/doubles
     - High-spin CCSD reference; ``sfcis`` guess; Ms changes by -1.
   * - ``SFEOMCC23``
     - single-spin-flip through order 3
     - Same reference/guess convention; materialized equations.

.. code-block:: python

   from ccpy import DIPEOM3, DEAEOM3, Molecule, ROHF, SFEOMCCSD

   dip = DIPEOM3(cc)
   dip.initial_guess(method="dipcis", nacto=cc.system.noccupied_alpha,
                     roots_per_irrep={"A": 2})
   dication_totals = dip.energy()
   dea = DEAEOM3(cc)
   dea.initial_guess(method="deacis", nactu=min(4, cc.system.nunoccupied_alpha),
                     roots_per_irrep={"A": 2})
   dianion_totals = dea.energy()

   high_spin = ROHF(Molecule("C 0 0 0; H 0 1 .6; H 0 -1 .6",
                            "6-31g", multiplicity=3))
   sf = SFEOMCCSD(high_spin)
   sf.initial_guess(method="sfcis", multiplicity=-1, roots_per_irrep={"A": 2})
   spin_flipped_totals = sf.energy()

The appropriate parent electron number is part of the physical model: DEA
starts from an N-2 parent for an N-electron target, DIP from N+2. Ensure charge,
basis diffuseness and parent spin are appropriate before requesting roots.

Correcting excited states
-------------------------

.. list-table:: Excited correction wrappers
   :header-rows: 1
   :widths: 30 30 40

   * - Correction
     - Intended upstream
     - Notes
   * - ``CREOMCC23``
     - ``EOMCCSD``
     - Default energy reports size-intensive delta-CR-EOMCC(2,3).
   * - ``EOMCCT3``
     - active ``EOMCCSDT_P``
     - ``num_active``, ``nacto/nactu``, ``two_body_approx``.
   * - ``EOMCCP3``
     - ``EOMCCSDT_P``
     - Full selected-triples correction; optional ``target_irrep``.
   * - ``EOMCCSDTA_star``
     - ``EOMCCSDTA``
     - Standard CCSD(T)(a)* neutral chain.
   * - ``CRIPCC23`` / ``CREACC23``
     - ``IPEOM2`` / ``EAEOM2``
     - CR charged triples corrections.
   * - ``IPCCP3`` / ``EACCP3``
     - ``IPEOM3_P`` / ``EAEOM3_P``
     - Requires upstream excited selected-space lists.
   * - ``IPEOMCCSDTA_star``
     - ``IPEOM2`` with ``hbar_class=HBarCCSDTA``
     - Standard lower-rank right/left chain for IP star correction.
   * - ``EAEOMCCSDTA_star``
     - ``EAEOM2`` with ``hbar_class=HBarCCSDTA``
     - Standard lower-rank right/left chain for EA star correction.
   * - ``DIPEOMCCSDTA_star``
     - ``DIPEOM3`` with ``hbar_class=HBarCCSDTA``
     - No left vectors required by this correction kernel.

.. code-block:: python

   from ccpy import CREOMCC23, EOMCCSDTA, EOMCCSDTA_star

   corrected = CREOMCC23(eom, variant="D")
   delta_cr_totals = corrected.energy()
   full_cr_totals = corrected.full_energy()
   print(corrected.ddeltas, corrected.deltas)

   approximate = EOMCCSDTA(cc)
   approximate.initial_guess(method="cis", multiplicity=1,
                             roots_per_irrep={"A": 2})
   star_totals = EOMCCSDTA_star(approximate, variant="A").energy()

``CREOMCC23.energy()`` adds ``ddeltas`` (the size-intensive correction).
``full_energy()`` instead adds ``deltas``, including the non-intensive r0 term.
``raw_energy()`` and ``full_raw_energy()`` omit FNO completion. Other excited
corrections ordinarily add their ``deltas`` to their parent EOM totals.
There is no universal ``completed_omega`` property on correction objects:
subtract the chosen, consistently corrected parent ground total explicitly
when constructing a corrected gap. Do not subtract an unrelated CCSD total
if the desired definition requires a corrected ground state.

ADC(2): full and folded
-----------------------

``ADC2``, ``IPADC2`` and ``EAADC2`` operate directly on a reference, without a CC
solve or HBar. Their guesses use the bare Hamiltonian. RHF, ROHF, UHF and
noncanonical occupied/virtual blocks are supported using the documented
spin-Fock partition including first-order Fov singles.

.. code-block:: python

   from ccpy import ADC2, IPADC2, EAADC2

   adc = ADC2(mf, nfrozen=1)  # formulation="folded" by default
   adc.initial_guess(method="cis", multiplicity=1,
                     roots_per_irrep={"A": 2})
   totals = adc.energy()
   density = adc.virtual_density(root=0)

   satellites = IPADC2(mf, nfrozen=1, formulation="full")
   satellites.initial_guess(method="ipcisd", nacto=2, nactu=2,
                            roots_per_irrep={"A": 4}, roots=[3])
   satellite_total = satellites.energy(root=0)

Folded mode stores primary Davidson vectors and reconstructs satellite
amplitudes using the diagonal resolvent; its default guesses are CIS/IP-CIS/
EA-CIS. It targets primary-dominated states. Full mode stores the expanded
linear eigenvectors and defaults to CISd/IP-CISd/EA-CISd, allowing doubles and
satellite targeting. Both use CCpy Davidson and return reconstructed normalized
full vectors for virtual state densities. Folded poles, unusable primary seeds,
duplicate roots and unconverged solutions raise explicit errors.
See :doc:`../adc` for equations, storage, density order and validation.

STEOM: automatic and manual calculations
----------------------------------------

``STEOMCC`` requires a closed-shell restricted CCSD parent and target
multiplicity 1 or 3. It solves charged auxiliaries, constructs a second
similarity transformation and then a neutral effective problem. It is not an
alias for an EE-EOMCCSD calculation.

.. code-block:: python

   from ccpy import STEOMCC, STEOMCCSDT

   automatic = STEOMCC(cc, nroots=3, multiplicity=1)
   automatic_totals = automatic.energy()
   manual = STEOMCC(cc, nroots=3, nacto=2, nactu=3)
   manual.initial_guess(method="cisd", nacto=2, nactu=3,
                        roots_per_irrep={"A": 5}, roots=[0, 2, 4])
   targeted_totals = manual.energy()
   extended = STEOMCCSDT(cc, nroots=3, nacto=2, nactu=3,
                        auxiliary_solver="pspace", minimum_active=1)
   extended_totals = extended.energy()

Without explicit active counts or lists, ordinary STEOM chooses CIS natural
orbital spaces. ``occupied_threshold`` and ``virtual_threshold`` control
cumulative populations; defaults are 0.99 for singlets and 0.999 for triplets.
Alternatively specify both ``nacto/nactu`` or both zero-based
``active_occupied/active_virtual`` lists. ``ip_threshold/ea_threshold`` filter
auxiliary singles character (ordinary defaults 0.9 singlet, 0.8 triplet);
``condition_limit`` controls model-overlap conditioning.

The neutral guess active space is separate from the charged auxiliary active
space. Ordinary STEOM has a singles-only final projection: a CISd guess can
select a state but cannot turn it into an SD diagonalization. Use
``STEOMCC(..., level="sdt")`` or ``STEOMCCSDT`` for the extended SD projection.
That class retains a CCSD ground state, enlarged charged satellite models,
and optionally higher-rank auxiliary sectors. Its additional controls include
``auxiliary_order=2|3``, ``auxiliary_solver="pspace"|"dense"``,
``minimum_active``, ``ip_pspace``, ``ea_pspace`` and ``dense_limit``.

Inspect ``omega``, ``singles_weights``, auxiliary model weights,
``orbital_rotations`` and convergence diagnostics. The extended implementation
is a correctness-oriented research implementation with omitted neutral
quadruples and no production-performance claim. Ordinary returned singles
are not full property wavefunctions. See :doc:`../steomcc` for normalization,
spin/solver behavior, active-space definitions and limitations.
