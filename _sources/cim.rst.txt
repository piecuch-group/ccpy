Cluster-in-molecule local correlation
=====================================

.. important::

   The CIM wrappers are experimental and are not fully tested. Limited
   canonical-limit, partition, subsystem and multilevel checks exist, but do
   not establish broad accuracy, robustness, scaling or geometry-derivative
   validation. Benchmark the chosen model and thresholds against a canonical
   calculation on representative small systems before scientific use.

   Every CIM calculation issues a ``CIMDevelopmentWarning`` (exported from
   ``ccpy``) to say so; filter it with ``warnings.simplefilter("ignore",
   CIMDevelopmentWarning)`` once the model has been checked for your systems.

Cluster-in-molecule (CIM) splits a ground-state correlated calculation into
independent calculations for small orbital subsystems and assembles the
correlation energy from the contributions of the occupied localized orbitals
(LMOs),

.. math::

   \Delta E = \sum_{i'} \delta E_{i'}, \qquad
   \delta E_{i'} = \frac{1}{M_{i'}} \sum_{\{P_{i'}\}} \delta E_{i'}(\{P_{i'}\}),

where :math:`\{P_{i'}\}` runs over the :math:`M_{i'}` subsystems in which the LMO
:math:`i'` is central. CIM is enabled on any supported method with ``cim=``, in the
same way as FNOs:

.. code-block:: python

   from ccpy import Molecule, RHF, CCSD, CRCC23, MP2, CIMOptions

   mf = RHF(Molecule(geometry, "6-31g*", auxiliary_basis="def2-universal-jkfit"))
   cim = CRCC23(mf, nfrozen=ncore, cim=CIMOptions(zeta=0.003))
   energy = cim.energy()
   cim.domains          # the subsystems (CIMDomains)
   cim.subsystems       # orbital counts, method, and energy of each subsystem

The constructor returns a ``ccpy.methods.cim.CIM`` object; method options (solver
settings, ``variant`` of CR-CC(2,3), ...) are passed on to the subsystem
calculations.

The implemented class filter accepts ground-state ``CCBase`` subclasses,
MP2, and the concrete ``CRCC23`` and ``AdaptiveCCPQ`` classes. Ordinary CCSD/CCSDT/CC3-style energies
are assembled from their T1/T2 energy expression. Acceptance by that filter
is not proof that every subclass or parameterized input is usable: selected
CC(P) lists and external CI data need a subsystem-basis adaptation that this
wrapper does not generally provide. Begin with MP2, CCSD, CCSDT, CRCC23 or
AdaptiveCCPQ, which have focused canonical-limit tests.

Adaptive CC(P;Q) runs its macroiterations unchanged in every subsystem: each
subsystem grows its own P space of triples (in its QCMO basis) by its own moment
contributions and stops by its own energy test (``energy_tolerance`` applies to
the subsystem energy). The central energy is the CC(P) energy from T1/T2 plus the
central part of the CC(P;3)_D correction (two-body approximation) on the P space
of the last macroiteration, partitioned like CR-CC(2,3) with the triples sum
skipping the P space. With ``percentage=[..., 100.0]`` every subsystem reaches
CCSDT, so the calculation converges to CIM-CCSDT. Its limitations:

* the triples are ranked by their contributions to the whole subsystem
  correction, not to its central part, so some are spent on environment-only
  excitations;
* the P spaces are labeled by subsystem QCMOs, which change with the geometry, so
  ``at_geometry`` (fixed domains, finite differences) reselects them rather than
  reusing them;
* semi-stochastic corrections (``stochastic=``) are not supported.

.. code-block:: python

   from ccpy.methods.corrections import AdaptiveCCPQ

   local = AdaptiveCCPQ(mf, nfrozen=ncore, percentage=[0.0, 1.0, 2.0, 5.0],
                        cim=CIMOptions(zeta=0.003))
   energy = local.energy() The reference must be closed-shell
RHF without point-group symmetry or deleted virtuals.

Out-of-the-box setup and what is returned
-------------------------------------------

.. code-block:: python

   from ccpy import CIM, CIMOptions, CCSD, MP2, Molecule, RHF

   mol = Molecule("O 0 0 0; H 0 0 .96; H .93 0 -.24", "6-31g",
                  auxiliary_basis="def2-universal-jkfit", symmetry=False)
   mf = RHF(mol, verbose=0)
   local = CCSD(mf, nfrozen=1, cim=CIMOptions(zeta=0.003))
   assert isinstance(local, CIM)  # this is a CIM wrapper, not a CCSD amplitude object
   total = local.energy()
   print(local.reference_energy, local.correlation_energy, total)
   print(local.domains.subsystems)
   for subsystem in local.subsystems:
       print(subsystem)

``cim=`` is intercepted by ``MethodBase.__new__``. You can also construct
``CIM(CCSD, mf, nfrozen=1, cim=CIMOptions(...))`` explicitly. Start from native
SCF or a ``Reference``, not a previously solved CC/left/correction object;
CIM creates its own subsystem calculations. Omit ``cim`` or pass ``None`` for
a canonical calculation; ``cim=False`` is not the disabling convention here.

The total is the full-system determinant energy plus assembled central-LMO
correlation contributions. Each ``subsystems`` entry contains ``method``,
``ncentral``, ``noccupied``, ``nunoccupied``, ``natoms`` and ``energy``. That
energy is the weighted central contribution, **not** a complete subsystem total
that should have its own SCF energy added again. ``raw_energy()`` equals
``energy()``: no automatic full-minus-truncated composite correction is attached.

``local.domains`` is a ``CIMDomains`` object containing ``subsystems`` and a
``(nLMO,natoms)`` array of atomic populations. Each underlying ``CIMSubsystem``
has ``central``, ``occupied``, ``atoms`` and ``method``. Orbital and atom indices
are zero based; central LMOs are listed first in ``occupied``. The wrapper
exposes summarized subsystem results, not a global canonical T/L/R operator.
It cannot serve as a normal CC parent for EOM or left-state composition.

Use a canonical-limit check before a threshold sweep:

.. code-block:: python

   canonical = CCSD(mf, nfrozen=1).energy()
   limit = CCSD(mf, nfrozen=1, cim=CIMOptions(zeta=0.)).energy()
   print("Canonical-limit error:", limit - canonical)
   for zeta in (0.01, 0.003, 0.001):
       approximation = CCSD(mf, nfrozen=1, cim=CIMOptions(zeta=zeta))
       print(zeta, approximation.energy(), approximation.subsystems)

Smaller zeta generally expands occupied environments, but domain merging can
change discretely; monotonic energy errors and universal accuracy at the
paper's thresholds are not guaranteed. On a small molecule every selected
domain can already span the full system, so a fast small example is not a
local-correlation scaling benchmark.

Method
------

The implementation follows W. Li, P. Piecuch, J. R. Gour, and S. Li, J. Chem.
Phys. **131**, 114109 (2009), with the single-environment subsystems of W. Li and
P. Piecuch, J. Phys. Chem. A **114**, 8644 (2010), and the multilevel scheme of
J. Phys. Chem. A **114**, 6721 (2010).

1. The correlated occupied orbitals are Boys localized.
2. **Subsystems (SECIM).** Every non-hydrogen atom, together with the hydrogen atoms
   bonded to it, collects as central orbitals the LMOs whose Mulliken charge (of two
   electrons) on one of these atoms exceeds ``central_charge`` (0.3). The environment
   are the LMOs :math:`j` with :math:`|\langle i|f|j\rangle| > \zeta` for a central
   :math:`i`. A subsystem whose occupied LMOs are contained in another one is merged
   into it (its central LMOs become central there). A single threshold
   :math:`\zeta` controls the calculation: :math:`\zeta = 0` gives one subsystem and
   the canonical result. Since subsystems are anchored on atoms, they do not
   rearrange unnecessarily as the nuclei move.
3. **Local subsystem orbitals.** The AO domain of an occupied LMO (core included)
   holds its largest-charge atoms whose charges add up to ``domain_charge`` (1.98);
   the AO domain of a subsystem is the union over its LMOs. Its extended domain adds
   every atom whose anchor (itself, or the non-hydrogen atom a hydrogen is bonded to)
   lies within ``buffer_radius`` (4 angstrom) of the AO domain. The occupied LMOs whose
   AO domains lie in the extended domain are projected onto its AOs (least squares)
   and symmetrically orthonormalized; the subsystem's own ones are its occupied
   orbitals. Its unoccupied orbitals span the exact orthogonal complement of those
   projected LMOs in the extended domain (the span of its PAOs), restricted to the
   orbitals that differ from their projection onto the AO-domain atoms by less than
   ``eta`` (0.5) in norm. Every subsystem orbital is a combination of the AOs of its
   extended domain only, and the subsystem orbitals are exactly orthonormal.
4. **Quasi-canonical orbitals.** The occupied and unoccupied orbitals of each
   subsystem are rotated to the QCMOs that diagonalize its Fock blocks. Each
   subsystem is an ordinary ``Reference`` holding only its own orbitals: its Fock
   matrix is that of the full system (the other occupied LMOs enter through it), and
   its two-electron integrals are density fitted with the auxiliary functions of the
   extended domain, formed one auxiliary atom at a time. No full-system MO integrals
   are built, and the unmodified method runs on the subsystem. The occupied and
   unoccupied subsystem orbitals are not exactly the HF ones, so the subsystem Fock
   matrix has small occupied-unoccupied elements; CC methods include them, MP2
   neglects them.
5. **Mixed-index energies.** With :math:`R` the rotation from the LMOs to the QCMOs,
   :math:`\delta E_{i'}(\{P\})` is the energy expression with its first occupied
   index rotated back to the LMO :math:`i'` (Eqs. (15)-(22) of the 2009 paper). For
   the CC energy, :math:`\sum_a f_{i'a} t^a_{i'} + \tfrac12 \sum_{j,a<b}
   v_{i'j}^{ab} \tau^{ab}_{i'j}`. For the triples correction of CR-CC(2,3),
   :math:`\tfrac13 \sum \ell^{i'jk}_{abc} \mathfrak{M}_{i'jk}^{abc}`: the Fortran
   kernels are called with the moment inputs (HBar blocks and :math:`T_2`) replaced
   by their derivatives under the occupied transformation
   :math:`e^{\lambda P}`, which yields :math:`(P_1 + P_2 + P_3)\mathfrak{M}` exactly,
   with :math:`P` the central projector weighted by :math:`1/M_{i'}`. In the QCMO
   basis the triples keep their canonical (diagonal-denominator) form, so the
   corrections remain noniterative.

Potential energy surfaces
-------------------------

By default the subsystems are determined at every geometry (they adapt to the
nuclear arrangement through their atom anchors). Where a reaction coordinate changes
them (e.g. two subsystems merge as a bond breaks), the energy can jump. The
fixed-domain variant reuses the subsystems of one geometry everywhere:

.. code-block:: python

   ref = CRCC23(mf_re, nfrozen=ncore, cim=CIMOptions(zeta=0.01))
   ref.energy()
   for mf in scan:
       e = CRCC23(mf, nfrozen=ncore, cim=CIMOptions(zeta=0.01, domains=ref.domains)).energy()

The LMOs of the new geometry are matched to those of the reference geometry by their
atomic Mulliken populations (an assignment problem), and each subsystem keeps its
AO-domain atoms. ``at_geometry`` (and so finite-difference gradients, which are the
only gradients CIM provides) always use the fixed domains. For a gradient or an optimization, solve the
reference CIM calculation first so its domains exist:

.. code-block:: python

   from ccpy.optimize import optimize_geometry

   local.energy()
   gradient = local.gradient()  # numerical energy differences, no analytic CIM response
   # Potentially expensive; this remains an experimental numerical surface.
   optimized = optimize_geometry(local, thresh="gau_tight")

Frozen domain membership does not freeze the LMOs or QCMO coefficients. The
orbitals are relocalized/rematched and subsystem orbitals rebuilt at each
geometry. Population-based assignments can become ambiguous at degeneracies,
and changes in the number of unoccupied orbitals passing ``eta`` can still affect
smoothness. Verify energy/gradient
continuity before using CIM for minima, transition states, frequencies or IRCs.
Static property convenience methods currently request an analytic property
gradient and therefore are unavailable on CIM; a consistent fixed-domain
finite-field scalar-energy callback is the generic alternative described in
:doc:`methods/multipoles`.

Constructing a fresh CIM object independently at every displaced geometry
without ``domains=...`` is a different, adaptively selected surface. It need
not give the same derivative as ``local.fd_gradient()`` or ``local.at_geometry``.

Multilevel CIM
--------------

``levels`` assigns higher-level methods to atoms; the method CIM is called on treats
the rest:

.. code-block:: python

   cim = MP2(mf, nfrozen=ncore, cim=CIMOptions(levels={CRCC23: [0, 1]}))

A subsystem is treated by the highest level for which one of its central LMOs lies
entirely on that level's atoms (hydrogen atoms follow the atom they are bonded to). A
subsystem is not merged into one of a lower level, and an LMO central in subsystems of
different levels contributes their average. The keys are method classes,
not method instances; atom lists use zero-based full-molecule indices. Dictionary
insertion order defines the ranking: list the highest level first. Hydrogens
follow their heavy-atom anchors. The assignment is based on an LMO's atomic
population set, not simply a per-atom additive correlation energy.

.. code-block:: python

   mixed = MP2(mf, nfrozen=1, cim=CIMOptions(
       zeta=0.003, levels={CCSD: [0]},
   ))
   mixed_total = mixed.energy()
   print([row["method"] for row in mixed.subsystems])

This upgrades qualifying subsystems anchored on atom 0 to CCSD and leaves
the others at MP2. A single water may all qualify, so test a separated-fragment
model when demonstrating mixed levels. Shared solver options reach subsystem
methods, but method-specific kwargs such as ``variant`` are forwarded only
when the subsystem's class is the wrapper's base method. ``levels`` does not
accept per-level constructor option dictionaries; an upgraded CRCC23 class
therefore uses its own default variant.

Options
-------

``zeta`` (0.003)
   environment threshold; SECIM(0.003) and SECIM(0.001) are the settings of the 2010
   paper.
``central_charge`` (0.3), ``domain_charge`` (1.98), ``bond_tolerance`` (0.168 angstrom)
   LMO-to-atom assignments; X-H bonds use covalent radii.
``buffer_radius`` (4.0 angstrom)
   buffer atoms of the extended domains, as in the 2009 paper. Much smaller values
   let occupied orbitals of excluded neighbors leak into the unoccupied space; this is
   detected (an unoccupied QCMO below the occupied ones) and raises a ``ValueError``.
``eta`` (0.5)
   selection of the unoccupied orbitals of a subsystem (step 12 of the 2009 paper,
   which recommends 0.5 when the unoccupied orbitals are strictly orthonormalized).
``levels``, ``domains``
   multilevel methods and fixed subsystems.
``localization_tol`` (1e-12), ``localization_maxiter`` (1000)
   Boys localization convergence and iteration limit.
``verbose`` (False)
   print the output of the subsystem calculations.

Differences from the papers
---------------------------

* The subsystem orbitals are kept in the extended domain (AO domain plus buffer)
  rather than projected back onto the AO domain, and they are strictly orthonormal
  (an option in the 2009 paper). The ``eta`` selection is applied to the whole
  unoccupied space of the extended domain at once (the subspace with weight above
  1 - eta^2 on the AO-domain atoms) instead of to its Boys-localized orbitals. With an
  extended domain covering the molecule and one subsystem, the canonical calculation
  is recovered.
* The two-electron integrals are fitted locally, with the auxiliary functions of the
  extended domain.
* X-H bonds use covalent radii plus ``bond_tolerance``.
* Among the noniterative corrections only CR-CC(2,3) and the CC(P;3) correction of
  adaptive CC(P;Q) have the triples partition; CCSD(T), CC(t;3), CC(P;3) with a given
  P space, EOM, and open-shell references are not supported yet. Adaptive CC(P;Q) is
  not part of the papers.

Restrictions, custom domains and validation status
---------------------------------------------------

CIM is currently closed-shell native RHF, C1, without deleted virtuals, with density
fitting (``cholesky_tei`` molecules are not supported). It does
not support UHF/ROHF, EE/IP/EA/DEA/DIP/SF EOM, ADC or STEOM. FNO+CIM composition
is unsupported, including starting from an already FNO-reduced reference.
Noniterative CCSD(T), CC(t;3), standalone CC(P;3) and quadruples corrections do not
have this wrapper's correction partition. Stochastic CIM correction sampling is not
implemented: the CRCC23 and adaptive CC(P;Q) central correction paths directly call
deterministic kernels. Do not infer availability from kwargs that a subsystem
class otherwise accepts.

For advanced fixed/custom domains, ``CIMDomains`` is exported from CCpy;
``CIMSubsystem`` is available from ``ccpy.methods.cim``. Custom central labels,
occupied lists, atom domains and population references must describe the same
correlated LMO space and molecule. Every occupied LMO must be covered as central
with consistent multiplicity; central orbitals must occupy the leading positions
of each subsystem list. The wrapper is not a general validator/remapper for
arbitrary selected determinants or external orbital labels.

The current result keeps only summarized subsystem counts/energies. Inspect
subsystem solver output with ``CIMOptions(verbose=True)`` and use tight solver
settings: the wrapper's final ``converged`` flag alone is not an independent
check that every inner iterative solve converged.

``tests/methods/test_cim.py`` contains focused checks for full-space limits
(MP2, CCSD, CCSDT, CRCC23), a dense mixed-index CRCC23_A partition oracle,
water-trimer subsystem construction, truncated extended domains (n-decane), the
buffer-too-small check, fixed-domain matching, multilevel
assignment and rejected unsupported classes/references. These are limited
implementation checks, not a broad method/reference/basis benchmark or a
production scaling test. Derivatives, large/localized systems and general
reaction paths remain insufficiently validated.
