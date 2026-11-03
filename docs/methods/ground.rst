Ground states and energy corrections
====================================

MPn and coupled-cluster levels
------------------------------

For any ordinary ground-state level, instantiate it on native SCF or a
``Reference`` and call ``energy()``. Additional required inputs are listed below.

.. code-block:: python

   from ccpy import MP2, CCSD, CCSDT, CCSDpT, CRCC23

   mp2 = MP2(mf, nfrozen=1)
   e_mp2 = mp2.energy()
   cc = CCSD(mf, nfrozen=1)
   e_ccsd = cc.energy()
   e_ccsdt = CCSDT(mf, nfrozen=1).energy()
   e_ccsd_t = CCSDpT(cc).energy()
   e_cr = CRCC23(cc, variant="D").energy()

Each energy includes the SCF determinant energy. ``correlation_energy`` is the
raw correlated contribution, not a second total energy. ``raw_energy()`` omits
FNO completion. For CC, ``T`` contains the converged excitation amplitudes.
``total_energy`` is a reporting property and does not itself trigger a fresh
solve; use ``energy()`` when the method may be unsolved.

.. list-table:: Ground-state method selection
   :header-rows: 1
   :widths: 20 45 35

   * - Class
     - Equations / inputs
     - Implementation notes
   * - ``MP2``
     - Second-order MP energy; native DF occupied-pair contractions.
     - Handles internal semicanonicalization and Fov singles for general references.
   * - ``MP3``, ``MP4``
     - Third-/fourth-order MP energies.
     - Materialized four-index kernels.
   * - ``CCD``
     - Doubles CC.
     - Materialized equations; no default HBar or left method.
   * - ``CC2``
     - Approximate second-order CC.
     - Materialized equations; default ``HBarCC2``.
   * - ``CCSD``
     - Singles and doubles CC.
     - Native DF; default ``HBarCCSD`` and ``LeftCCSD``.
   * - ``CC3``
     - Approximate iterative triples.
     - Materialized; stored operator order 2, with triples handled by the kernel.
   * - ``CCSDT``
     - Full singles, doubles and triples.
     - Materialized; stored order 3, ``HBarCCSDT`` / ``LeftCCSDT``.
   * - ``CC4``
     - Approximate quadruples model.
     - Materialized; stored order 3; explicitly restricted to closed-shell RHF.
   * - ``CCSDTQ``
     - Full singles through quadruples.
     - Materialized; no default HBar/left method exposed by this wrapper.
   * - ``CCSDT_P``
     - Full T1/T2 plus selected T3; requires ``t3_excitations``.
     - Native DF CC solve; downstream HBar/left equations can be materialized.
   * - ``ACCD``, ``ACCSD``
     - Require ``acparray``: five scaling coefficients for T2-squared diagrams.
     - Materialized; no default HBar/left method.
   * - ``ACCSDT_P``
     - Requires ``t3_excitations`` and ``acparray``.
     - Materialized selected-triples approximate coupled-pair equations.
   * - ``ECCC2``
     - Requires ``ci_vectors_file`` for external cluster analysis.
     - Materialized; external T3/T4 are held fixed during CCSD-like iterations.

Higher excitation rank can rapidly increase memory and cost even for small
molecules. The operator rank is not always the formal perturbative level:
``CC3`` and ``CC4`` are examples. Use the equation descriptions and method name,
not ``T.order`` alone, to identify the model.

Approximate coupled-pair and external correction inputs
-------------------------------------------------------

.. code-block:: python

   from ccpy import ACCSD, ECCC2

   acp = ACCSD(mf, acparray=[1., 1., 0., 0., 1.], nfrozen=1)
   acp_total = acp.energy()

   external = ECCC2(mf, ci_vectors_file="ci_wavefunction.dat", nfrozen=1)
   external_total = external.energy()

The example scaling vector is a selectable diagram model, not a universal
recommended setting. ``ECCC2`` uses
``ccpy.extcorr.external_correction.cluster_analysis``'s headerless text format.
Each line contains an identifier, a floating-point CI coefficient, then the
occupied spin-orbital labels of that determinant. For example, in a correlated
two-electron space:

.. code-block:: text

   1  0.98   1 2
   2 -0.10   1 4

Spin-orbital labels are one based with alpha odd and beta even: spatial orbital
p has alpha label ``2*p-1`` and beta label ``2*p``. Frozen electrons are excluded
and correlated orbital labels restart at one. Include the sorted HF determinant
with a nonzero coefficient; the reader uses it for intermediate normalization.
It processes excitations through quadruples. Do not insert headers, comments
or blank records into this parser's input. A generic coefficient vector without
determinant labels is insufficient.

The CI determinants, orbital ordering, frozen core and basis must match the
working reference. FNO rotation requires a CI wavefunction already expressed
in the reduced working basis. This wrapper does not rotate external CI data.

Ground-state noniterative corrections
-------------------------------------

Corrections are separate objects so that the same CC, HBar and left state can
be reused. ``energy()`` is the corrected total, while ``deltas`` contains the
additive increments by variant. Most wrappers default to variant D; the
catalog gives the exact variant tuple and default for each class.

.. list-table:: Standard correction chains
   :header-rows: 1
   :widths: 23 30 47

   * - Correction
     - Standard parent
     - Important settings / meaning
   * - ``CCSDpT``
     - ``CCSD``
     - CCSD(T); A only; bare Hamiltonian, no left state or HBar needed.
   * - ``CRCC23``
     - ``CCSD`` + ``LeftCCSD``
     - CR-CC(2,3); A/B/C/D; native DF; optional semi-stochastic evaluation.
   * - ``CCT3``
     - active ``CCSDT_P``
     - CC(t;3); ``num_active``, optional ``nacto/nactu``; optional stochastic evaluation.
   * - ``CCP3``
     - ``CCSDT_P``
     - CC(P;3); default two-body approximation, optional full selected-triples HBar/left.
   * - ``CRCC24``
     - ``CCSD``
     - CR-CC(2,4), materialized quadruples correction.
   * - ``CRCC34``
     - ``CCSDT`` + ``LeftCCSDT``
     - CR-CC(3,4), RHF kernel; default A. Do not assume every inherited variant is implemented.
   * - ``CCP4``
     - default ``CCSD``
     - Requires ``t4_excitations``; materialized selected-quadruples correction.

.. code-block:: python

   cr = CRCC23(cc, variant="D")
   total = cr.energy()
   print(cr.deltas)                 # e.g. A/B/C/D increments
   print(cr.cc is cc)               # reuse the parent

Changing ``variant`` among computed supported keys changes reporting; it does
not require solving CC again. Scientific kernel restrictions can be narrower
than inherited wrapper options. In particular, inspect returned ``deltas``
before assuming a high-rank correction supplies every A/B/C/D variant.
FNO completion is included once in the final total, in addition to the
high-level correction.

Selected triples, CC(t;3), and CC(P;3)
--------------------------------------

.. code-block:: python

   from ccpy import (Reference, CCSDT_P, CCT3, CCP3,
                     get_active_triples_pspace)

   ref = Reference(mf, nfrozen=1)
   ref.system.set_active_space(nact_occupied=2, nact_unoccupied=2)
   triples = get_active_triples_pspace(ref.system, num_active=1)
   ccp = CCSDT_P(ref, triples)
   e_p = ccp.energy()
   e_t3 = CCT3(ccp, num_active=1, nacto=2, nactu=2).energy()
   e_pq_2ba = CCP3(ccp, two_body_approx=True).energy()
   e_pq_full = CCP3(ccp, two_body_approx=False).energy()

``CCT3``'s active counts configure the correction's ``System``; they do not
retroactively generate the right CC triples. Supply a consistent selected-triples
parent or explicit ``t3_excitations``. ``CCP3`` can also accept
``target_irrep``; ``high_memory`` selects the high-memory full correction only
when ``two_body_approx=False``. Full and two-body-approximation corrections use
different HBars/left states and need not give the same energy.

``CCP4`` requires an explicit ``t4_excitations`` dictionary with keys
``aaaa``, ``aaab``, ``aabb``, ``abbb`` and ``bbbb`` and rows
``[a,b,c,d,i,j,k,l]`` in the expected selected-space convention. The helper
``get_quadruples_pspace_from_cipsi`` reads the supported external CIPSI format;
its signature and file convention should be inspected before using external
lists. It does not create a CCSDTQ(P) parent: no such ground-state wrapper is
exported here. Selected-space conventions are detailed in :doc:`spaces`.

Adaptive CC(P;Q)
-----------------

.. code-block:: python

   from ccpy import AdaptiveCCPQ

   adaptive = AdaptiveCCPQ(
       mf, nfrozen=1, percentage=[0., 1., 2.],
       energy_tolerance=1.e-4,
   )
   total = adaptive.energy()
   print(adaptive.ccp_energies, adaptive.ccpq_energies)
   print(adaptive.adaptive_converged, adaptive.ccpq_errors)

The percentages specify successive P-space sizes relative to all triples in
the reference symmetry. Each macroiteration solves CCSDT(P), constructs a
CCSD-like HBar/left state for the two-body approximation, and selects new triples
from CC(P;3) moment contributions. The final macroiteration computes the
correction without further growth. Without ``percentage``, growth uses
``selection_factor`` times a singles or singles+doubles base
(``base_growth="ccsd"`` by default), for up to
``maximum_macroiterations + 1`` macroiterations.

Controls include ``n_det_max``, ``buffer_factor``, ``minimum_threshold`` and
``reset_amplitudes``. Explicit ``pspaces=[...]`` replays prescribed spaces.
``pspaces``, ``timings`` and ``extrapolated`` record the selection history,
wall times and Goodson extrapolations. ``adaptive_converged`` distinguishes
meeting the adaptive energy criterion from merely finishing the prescribed
schedule. This wrapper is ground-state AdaptiveCCPQ; legacy adaptive excited
state drivers are separate APIs.

Semi-stochastic triples corrections
-----------------------------------

``CRCC23``, ``CCT3`` and two-body ``CCP3`` accept:

.. code-block:: python

   stochastic = CRCC23(cc, stochastic={"target_error": 1.e-5, "seed": 7})
   total = stochastic.energy()
   print(stochastic.errors, stochastic.stochastic_info)

``errors`` reports one-standard-deviation estimates by variant. These are
sampling uncertainties, not estimates of basis-set, FNO or truncation errors.
``stochastic=True`` uses the algorithm defaults; the complete controls live in
``ccpy.moments.stochastic_triples.DEFAULT_OPTIONS``. Unsupported correction
classes reject stochastic evaluation. Full ``CCP3`` requires deterministic
operation.

``AdaptiveCCPQ(stochastic=..., convergence_z=1.96)`` uses stochastic corrections
for convergence decisions while determinant selection remains deterministic.
Its stopping criterion includes the uncertainty of consecutive energies:

.. math::

   |E_n-E_{n-1}|+z\sqrt{\sigma_n^2+\sigma_{n-1}^2}
      < \mathrm{energy\_tolerance}.

The statistical target must be small enough to resolve that tolerance; the
wrapper checks this. Analytic gradients are disabled for stochastic correction
instances; numerical differentiation of sampled energies needs its own noise
assessment.
