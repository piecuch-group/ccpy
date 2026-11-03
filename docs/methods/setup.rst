Molecules, references and composing calculations
================================================

Geometry, integrals and SCF
---------------------------

.. code-block:: python

   from ccpy import Molecule, RHF, ROHF, UHF, Reference

   mol = Molecule(
       "O 0 0 0; H 0 0 .97", "cc-pvdz",
       auxiliary_basis="cc-pvdz-rifit",
       charge=0, multiplicity=2, unit="angstrom", symmetry=True,
   )
   mf = ROHF(mol, e_tol=1.e-10, d_tol=1.e-8, maxiter=150)
   mf.run()
   assert mf.converged
   ref = Reference(mf, nfrozen=1)
   print(ref.system.point_group, ref.system.reference_symmetry)
   print(ref.system.orbital_symmetries)

``Molecule`` accepts semicolon/newline XYZ or Z-matrix text. ``basis`` and
``auxiliary_basis`` can be names, basis files, or per-atom dictionaries, e.g.
``{"O": "cc-pvtz", "default": "cc-pvdz"}``. With a string orbital basis,
omitting the auxiliary basis selects ``<basis>-autoaux``. Supply an auxiliary
basis explicitly with a dictionary orbital basis. ``cartesian=False`` uses
spherical functions. ``overlap_tol`` controls removal of AO linear dependencies;
``df_metric_tol`` optionally controls DF metric orthogonalization.

``cholesky_tei=True, cholesky_tol=1.e-6`` selects pivoted AO Cholesky instead of
auxiliary-basis fitting. The correlated interfaces still receive three-index
factors. Auxiliary basis and Cholesky thresholds affect the Hamiltonian itself;
they are independent of the later FNO occupation cutoff.

Choose RHF for closed-shell restricted orbitals, ROHF for a common restricted
open-shell spatial basis, or UHF for independent alpha/beta orbitals. Molecular
``multiplicity`` describes the reference determinant, not the target EOM spin.
Native SCF accepts ``guess="minao"`` (default) or ``"hcore"``, ``diis_size``,
``diis_start``, ``level_shift`` and ``level_shift_thresh``. UHF accepts a pair of
spin-specific level shifts. SCF uses ``maxiter/e_tol/d_tol``; correlated methods
use ``maximum_iterations/energy_convergence/amp_convergence``.

With symmetry enabled, SCF can accept ``target_symmetry`` or
``irrep_occupations={irrep: electrons}`` / ``{irrep: (nalpha, nbeta)}``.
Unlisted irreps in an occupation dictionary are left empty. These configure the
reference occupations; they are not EOM root selections. Inspect the detected
point group and orbital labels rather than assuming a particular orientation
and irrep assignment.

``Reference`` currently accepts native ``SCFBase`` objects, not PySCF SCF
objects. The older ``Driver.from_pyscf`` workflow is separate. Reference creation
runs SCF when ``e_tot`` is absent and checks the determinant energy against the
DF Hamiltonian. It warns if explicitly given an unconverged SCF; verify
``mf.converged`` before trusting correlated results.

Frozen core and working orbitals
--------------------------------

``nfrozen`` counts lowest core spatial orbitals removed from correlation in
both spins. ``ndelete`` deletes highest virtuals; it is a fixed orbital deletion,
with no automatic missing-space correction. These partition settings belong
to the reference. The determinant energy still contains the frozen electrons.

.. code-block:: python

   from ccpy import CCSD, EOMCCSD

   ref = Reference(mf, nfrozen=1)
   cc = CCSD(ref)
   eom = EOMCCSD(cc)
   assert eom.reference is cc.reference

Do not change ``nfrozen`` or ``ndelete`` downstream. An explicitly conflicting
value raises an error. Start a new chain from SCF or another full reference for
a different partition. FNO selection also creates a new working basis; see
:doc:`spaces` before constructing orbital-indexed inputs.

The working dimensions are ``system.noccupied_alpha``,
``system.noccupied_beta``, ``system.nunoccupied_alpha`` and
``system.nunoccupied_beta``. Frozen/deleted orbitals are excluded. Under FNO,
these are already the reduced dimensions; equation kernels do not carry a
second full-space dimension.

Short form and explicit composition
-----------------------------------

.. code-block:: python

   from ccpy import CCSD, CRCC23, LeftCCSD, EOMCCSD, CREOMCC23

   # Out of the box: creates CCSD, the required HBar and Lambda automatically.
   result = CRCC23(mf, nfrozen=1, variant="D").energy()

   # Explicit reusable chain.
   cc = CCSD(mf, nfrozen=1)
   cc.energy()
   hbar = cc.hbar                   # builds on demand
   left = cc.left.ensure_solved()   # LeftCCSD
   correction = CRCC23(left, variant="D")
   result_again = correction.energy()

   eom = EOMCCSD(cc)
   eom.initial_guess(method="cis", multiplicity=2,
                     roots_per_irrep={ref.system.reference_symmetry: 1})
   totals = eom.energy()
   corrected_totals = CREOMCC23(eom, variant="D").energy()

The example above uses the open-shell reference of this chapter, hence target
multiplicity 2. Closed-shell singlet examples elsewhere use multiplicity 1.
Excited-state corrections require an EOM or left-EOM object; they cannot infer
target states from SCF alone. Left-EOM methods also require an EOM object.

Passing an existing method with ``.cc`` generally reuses that CC object. It does
not automatically replace it with the new method's nominal default CC level.
For example, an EOM class whose default parent is CCSDT will reuse an explicitly
supplied CC object. This makes custom equation/HBar combinations possible, but
the caller must choose a scientifically compatible parent and amplitudes.
Use :doc:`catalog` to see the standard dependencies and :doc:`excited` for
known standard chains.

``hbar_class=...`` deliberately overrides the EOM similarity-transformed
Hamiltonian. It is useful for CCSD(T)(a)* chains, not an instruction to perform
an arbitrary missing ground-state calculation. HBars are cached by class;
building a legacy HBar does not overwrite the cached DF HBar or bare integrals.

Options and restart ownership
-----------------------------

Explicit solver options propagate into automatically created dependencies.
When a method is built on another method, its explicit solver settings are
inherited unless overridden locally. Inheriting a tolerance does not mutate
an already existing upstream object; configure that object directly if it also
needs tighter convergence. ``RHF_symmetry`` is recomputed from the reference
rather than blindly inherited, and EOM/left wrappers adjust it for their spin
sector.

Repeated ``energy()`` calls use cached results. ``solve()`` explicitly reruns
and increments ``version``. Ground CC restarts from its current ``T``. Downstream
objects track versions of their dependencies and rebuild stale HBars, guesses,
left states and corrections. If changing a solver option in ``options`` after
convergence, explicitly call ``solve()`` or mark the method unconverged; changing
a dictionary entry alone is not a new dependency version. There is no general
persistent disk checkpoint/restart API in this layer. Out-of-core solver files
are working storage, not durable checkpoints.
