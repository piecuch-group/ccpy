FNOs, active spaces and selected determinants
=============================================

Experimental CIM domains are a separate local-correlation model; see
:doc:`../cim`. Current CIM wrappers cannot be composed with FNOs.

Three different reductions
---------------------------

A frozen core removes occupied orbitals from correlation. FNOs rotate and
truncate virtual orbitals for the whole correlated calculation. A P space
selects individual high-rank excitations within those working orbitals.
An initial guess active space restricts only its preliminary CI problem.
These choices are independent; specify which one you intend to change.

Enable DF-MP2 FNOs at the start of a chain
------------------------------------------

.. code-block:: python

   from ccpy import CCSD, EOMCCSD, FNOOptions

   options = FNOOptions(source="mp2", occupation_cutoff=1.e-6)
   cc = CCSD(mf, nfrozen=1, fno=options)
   completed_ground = cc.energy()
   eom = EOMCCSD(cc)
   eom.initial_guess(method="cisd", multiplicity=1, nacto=2, nactu=2,
                     roots_per_irrep={"A": 2})
   completed_states = eom.energy()
   print(cc.raw_energy(), cc.fno_result.delta_ground)
   print(eom.completed_omega)

Preparation computes full-space DF-MP2 energy and virtual density, diagonalizes
the density, selects virtual natural orbitals, semicanonicalizes the retained
virtual Fock block, transforms DF factors and recomputes DF-MP2 in the reduced
space. Occupied orbitals and the SCF determinant are unchanged. CC/EOM kernels
then treat the reduced virtual dimension as the ordinary working dimension.
The ground MP2 provider uses three-index occupied-pair contractions rather
than storing a full MP2 doubles tensor. ROHF uses the specified spin-Fock
partition, including first-order singles when Fov is nonzero.

Use exactly one selection criterion:

.. code-block:: python

   by_occupation = FNOOptions(occupation_cutoff=1.e-6)
   by_population = FNOOptions(retained_population=0.999)
   by_count = FNOOptions(nvirtuals=20)
   full_retention = FNOOptions(occupation_cutoff=0.)

The default is occupation cutoff 1e-6. Restricted occupations use spatial 0--2
normalization; UHF uses spin 0--1. ``nvirtuals`` counts external spatial
virtuals for restricted references. For UHF it specifies a minimum per-spin
count with equal removals enforced. Symmetry, degeneracy and protected-space
closure can increase the final count. Removing all external virtuals is rejected.
FNO thresholds are occupation criteria, not guaranteed energy-error bounds;
check energy/gap convergence as the retained space grows.

Automatic composite energy completion
--------------------------------------

The default ``energy_completion="ground"`` returns

.. math::

   E_{\mathrm{reported}}=E_{\mathrm{high,trunc}}
     +E_{\mathrm{low,full}}-E_{\mathrm{low,trunc}}.

``energy()`` includes this shift automatically. ``raw_energy()``,
``correlation_energy`` and raw ``omega`` do not. Ground corrections and final
STEOM/EOM totals apply the completion once; auxiliary eigenvalues are not
independently shifted. For MP2 selection the same ground correction shifts
all target totals and the parent, so target-minus-parent gaps are unchanged.
A corrected total is still an approximation to the full-space high-level
energy; the composite formula does not recover missing high-level amplitudes.

``energy_completion="none"`` reports uncompleted truncated totals.
``"state_specific"`` requires ADC target energies and applies a distinct
low-level full-minus-reduced shift to each target. ``completed_omega`` is the
completed target-minus-completed-parent difference on EOM/ADC/STEOM objects.
There is no correction for manually deleted virtuals with ``ndelete``.

State-specific ADC-FNOs
-----------------------

.. code-block:: python

   from ccpy import CCSD, EAEOM2, FNOOptions, FNOTarget

   targets = (FNOTarget("EA", root=0, irrep="A", formulation="folded"),)
   policy = FNOOptions(
       source="adc2", targets=targets, space_policy="root_specific",
       occupation_cutoff=1.e-6, energy_completion="state_specific",
   )
   cc = CCSD(mf, nfrozen=1, fno=policy)
   ea = EAEOM2(cc, fno_targets=targets)
   ea.initial_guess(method="eacis", roots_per_irrep={"A": 1})
   completed_target_total = ea.energy(root=0)

The native provider supports EE/IP/EA on RHF/ROHF/UHF, including noncanonical
references. ``FNOTarget`` fields are sector, zero-based root within the
provider's irrep, optional irrep, optional positive multiplicity, and
``formulation="folded"|"full"``. Omit charged multiplicity to use the
provider's unadapted default; ``-1`` is a guess setting, not an allowed
``FNOTarget.multiplicity``. On a symmetry reference, provide the irrep.

Folded selection uses bare CIS/IP-CIS/EA-CIS guesses and is intended for
primary-dominated states. Full selection uses bare CISd/IP-CISd/EA-CISd,
including active doubles, and supports satellite/doubles targeting:

.. code-block:: python

   satellite = FNOTarget("IP", root=3, irrep="A", formulation="full")
   double_excitation = FNOTarget("EE", root=4, irrep="A", multiplicity=1,
                                formulation="full")

The built-in full-space CISd provider uses up to two active occupied/virtual
orbitals; requesting a higher root does not enlarge that active model. For
other active-space choices or more elaborate root identification, register a
custom provider. Its density must be a state virtual block with the documented
ISR order, not a transition density or an arbitrary particle population.
Native DEA/DIP/SF ADC-FNO selection is not implemented; DF-MP2 FNOs still apply
to those high-level methods.

Multiple targets and root identities
------------------------------------

``space_policy="average"`` diagonalizes a weighted average target density;
``weights`` supplies positive weights, normalized by preparation.
``"union"`` selects each target subspace separately and retains their span.
``"root_specific"`` requires exactly one target and a separate chain per target.
The default shared policy is average. Union/average can combine supported
sectors or ADC formulations; all high-level calculations must use the resulting
same reference to share amplitudes.

.. code-block:: python

   targets = (
       FNOTarget("EE", root=0, irrep="A", multiplicity=1),
       FNOTarget("EE", root=2, irrep="A", multiplicity=1),
   )
   policy = FNOOptions(source="adc2", targets=targets, weights=(1., 2.),
                       space_policy="average", nvirtuals=10,
                       energy_completion="state_specific")
   cc = CCSD(mf, nfrozen=1, fno=policy)
   eom = EOMCCSD(cc, fno_targets=targets)
   eom.initial_guess(method="cisd", multiplicity=1, nacto=2, nactu=2,
                     roots_per_irrep={"A": 3}, roots=[0, 2])
   totals = eom.energy()

``fno_targets`` maps low-level state identities into **returned high-level root
order**. Its length, sectors, irreps and specified spins must match that output.
The API cannot prove that a high-level root has the intended low-level physical
character: check the states yourself. Mixed-sector references with state-specific
completion need each EOM object's corresponding subset of targets.

The provider projects full-space physical vectors through the actual orbital
transformation to seed reduced calculations. It follows overlaps and rejects a
normalized projected overlap below 0.5. Degeneracies can require a larger space
or a provider with subspace tracking. A root label is not a guarantee of an
unchanged energy ordering.

Protected orbitals and reference inspection
-------------------------------------------

``protected_virtuals=(0, 2)`` adds specified input virtual directions to the
retained span. These are zero-based external virtual labels (excluding ROHF
singly occupied spatial orbitals); for UHF they are local spin-virtual labels.
This protects directions, not final individual natural orbitals. Dense protected
subspaces can also be supplied by custom providers.

All singly occupied ROHF spatial orbitals are preserved. Only the common
external virtual space is rotated/truncated, preserving the restricted
open-shell determinant. Symmetry and occupation degeneracies are preserved;
``degeneracy_tolerance`` and ``density_tolerance`` control associated checks.

.. code-block:: python

   result = cc.fno_result
   print(result.discarded, result.retained_population)
   print(result.occupations)
   print(result.full.ground_energy, result.truncated.ground_energy)
   full_reference = cc.reference.fno_parent

``fno_result`` is shared through the reduced reference. Occupations are reported
before virtual semicanonicalization. A new threshold requires a new chain from
SCF or the full reference. You cannot add FNOs after allocating an upstream CC
calculation, disable inherited FNOs downstream, combine them with ``ndelete``,
or reuse full-space amplitudes after a dense virtual rotation.

Active and selected spaces after FNO preparation
------------------------------------------------

.. code-block:: python

   from ccpy import (Reference, prepare_fno, CCSDT_P, EOMCCSDT_P,
                     get_active_triples_pspace)

   ref = prepare_fno(Reference(mf, nfrozen=1), FNOOptions(nvirtuals=6))
   ref.system.set_active_space(nact_occupied=2, nact_unoccupied=2)
   t3 = get_active_triples_pspace(ref.system, num_active=1)
   r3 = get_active_triples_pspace(ref.system, num_active=1)
   ccp = CCSDT_P(ref, t3)
   eom_p = EOMCCSDT_P(ccp, r3_excitations=r3)
   eom_p.initial_guess(method="cis", multiplicity=1,
                       roots_per_irrep={"A": 1})
   selected_total = eom_p.energy(root=0)

Active occupied labels are frontier occupied orbitals and active virtual labels
are the lowest working virtuals; ``set_active_space`` derives spin-specific
counts for open-shell references. Setting active counts alone does not constrain
an ordinary CCSDT/EOMCCSDT solve. Generate P lists and choose a selected-space
method to restrict high-rank excitations.

P lists are integer, one-based orbital labels in the **working correlated MO
basis**, with particles first and holes second. Use Fortran-contiguous arrays
and the spin cases/order expected by the kernel:

.. list-table:: Selected high-rank spaces
   :header-rows: 1
   :widths: 25 25 50

   * - Model / helper
     - Dictionary keys
     - Rows
   * - Ground/EE triples, ``get_active_triples_pspace``
     - aaa, aab, abb, bbb
     - ``[a,b,c,i,j,k]``
   * - IP 3h2p, ``get_active_3h2p_pspace``
     - aaa, aab, abb
     - ``[b,c,i,j,k]`` with particle/holes spin strings derived from the sector.
   * - EA 3p2h, ``get_active_3p2h_pspace``
     - aaa, aab, abb
     - ``[a,b,c,j,k]``
   * - DEA 4p2h, ``get_active_4p2h_pspace``
     - abaa, abab, abbb
     - ``[a,b,c,d,k,l]``

For IP the particle/hole spin strings are ``aa/aaa``, ``ab/aab``,
``bb/abb``; for EA they are ``aaa/aa``, ``aab/ab``, ``abb/bb``. For DEA
they are ``abaa/aa``, ``abab/ab``, ``abbb/bb``. Each side gives the spin of
each particle/hole column in order. The triples helper requires at least
``num_active`` active holes **and** active particles; IP counts active holes,
EA/DEA active particles. ``target_irrep`` can filter generated lists.

Empty spin blocks conventionally use a single all-ones row with zero amplitude,
not a zero-row array. Same-spin indices have canonical ordering/antisymmetry;
helpers enforce the kernel's convention. For restricted closed-shell CC(P),
wrappers derive beta-related lists from their alpha partners. Fortran kernels
may sort lists in place together with amplitudes; do not rely on caller order
remaining unchanged.

``CCSDT_P.set_pspace`` retains amplitudes by extending its selected operator.
New excitations must be appended consistently to the current list/order; it
is not a general arbitrary-list remapper. ``EOM..._P.set_pspace`` resets right
states and recomputes from guesses. Ground ``t3_excitations`` and excited
``r3_excitations`` are separate data even if generated by the same helper.

Build charged selected models similarly:

.. code-block:: python

   from ccpy import (IPEOM3_P, EAEOM3_P, DEAEOM4_P,
                     get_active_3h2p_pspace, get_active_3p2h_pspace,
                     get_active_4p2h_pspace)

   ip_space = get_active_3h2p_pspace(ref.system, num_active=1)
   ea_space = get_active_3p2h_pspace(ref.system, num_active=1)
   dea_space = get_active_4p2h_pspace(ref.system, num_active=2)
   ip = IPEOM3_P(ref, r3_excitations=ip_space)
   ea = EAEOM3_P(ref, r3_excitations=ea_space)
   dea = DEAEOM4_P(ref, r3_excitations=dea_space)

These constructors create their standard CCSD parents unless a compatible
parent is supplied. Configure guesses before calling ``energy()``. The name
``r3_excitations`` is also used by the DEA wrapper for its selected 4p2h sector;
it is an API parameter name, not a claim of three-particle rank.

Custom FNO providers
--------------------

``register_fno_provider(name, callable)`` installs a provider accepting
``(reference, targets=(), need_density=True, full_result=None)`` and returning
``FNOEvaluation``. Ground/target energies are absolute totals, including the
parent determinant energy, with signed charged conventions. Density dictionaries
are keyed by targets (or ``None`` for ground selection), then ``a/b`` spin blocks
in the supplied reference's virtual basis. The reduced call receives the full
result for physical-root matching. Honor ``need_density=False`` without changing
the energy convention. Providers may return metadata and protected subspaces.
See :doc:`../fno` for the complete extension contract and an example.

External determinant-space readers
----------------------------------

``get_triples_pspace_from_cipsi(path, system)`` and
``get_quadruples_pspace_from_cipsi(path, system)`` extract the respective ranks
from headerless determinant records with identifier/coefficient followed by
occupied spin-orbital labels. They use correlated-space numbering (alpha odd,
beta even), not AO labels or all-electron frozen-core numbering. Check the
function's returned tuple before passing its excitation dictionary to a method;
the triples reader also reports excitation counts by symmetry.
``get_pspace_from_qmc`` reads integer signed walker populations and filters
by ``threshold_walkers``; the present implementation selects triples. These
readers do not transform determinants between orbital bases. Inspect the source
and a small known determinant before using a new external generator's output.
