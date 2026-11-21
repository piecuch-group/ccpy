Solver controls, diagnostics and custom calculations
=====================================================

Shared correlated-method options
---------------------------------

Pass options as constructor keywords. Unknown keywords are rejected, which
helps catch legacy Driver spellings accidentally used with this API.

.. code-block:: python

   from ccpy import CCSD, EOMCCSD

   cc = CCSD(mf, nfrozen=1, maximum_iterations=120,
             amp_convergence=1.e-8, energy_convergence=1.e-9,
             diis_size=8, denominator="mp", energy_shift=0.)
   eom = EOMCCSD(cc, davidson_max_subspace_size=40,
                 davidson_selection_method="overlap")

.. list-table:: Defaults in MethodBase
   :header-rows: 1
   :widths: 32 18 50

   * - Option
     - Default
     - Meaning
   * - ``maximum_iterations``
     - 80
     - Iteration limit for iterative solvers; ADC applies it to inner/outer loops.
   * - ``amp_convergence``
     - 1e-7
     - Residual/amplitude convergence tolerance.
   * - ``energy_convergence``
     - 1e-7
     - Energy/eigenvalue change tolerance, in hartree.
   * - ``energy_shift``
     - 0.0
     - Shift used by kernels that support shifted iterative denominators.
   * - ``denominator``
     - ``"mp"``
     - Ground-CC update denominator scheme; supported schemes depend on the level.
   * - ``diis_size``
     - 6
     - DIIS history; ground CC uses -1 to disable DIIS.
   * - ``diis_out_of_core``
     - False
     - Disk storage for DIIS vectors in solvers that implement it.
   * - ``RHF_symmetry``
     - None
     - Automatic closed-shell spin-block shortcut; not the spatial symmetry switch.
   * - ``amp_print_threshold``
     - 0.09
     - Threshold for amplitude printing, not amplitude truncation.
   * - ``davidson_max_subspace_size``
     - 30
     - Maximum Davidson basis size before restarting.
   * - ``davidson_solver``
     - ``"standard"``
     - Standard sequential EOM or ``"multiroot"`` block EOM; ADC supports standard only.
   * - ``davidson_selection_method``
     - ``"overlap"``
     - Ritz selection by starting-vector overlap or ``"energy"`` proximity.
   * - ``davidson_out_of_core``
     - False
     - Disk-backed Davidson vectors where supported.
   * - ``fd_step``
     - 1e-3
     - Finite-difference displacement, bohr.
   * - ``fd_npoints``
     - 4
     - Central finite-difference stencil size (2, 4 or 6).

The wrapper automatically disables ground DIIS on very small spaces when no
explicit history size was given. SCF has separate defaults and uses
``diis_size=0`` to disable SCF DIIS. Shared option names do not imply that every
kernel uses every option. EOMCC3 is nonlinear DIIS; STEOM has separate auxiliary
and final solvers; MBPT is noniterative.

``CCSD``, ``CCSDT`` and ``CCSDT_P`` accept ``denominator="mp"``,
``"epstein-nesbet"``, ``"quasilinear"`` or ``"newton"`` in their ground-state
updates. Other levels default to MP and reject unsupported schemes. These
select iterative updates/preconditioning; they are not interchangeable with
correction variant A/B/C/D or ADC's perturbation partition.

Convergence and result inspection
---------------------------------

.. code-block:: python

   energy = cc.energy()
   print(cc.converged, cc.version, cc.is_current())
   print(cc.reference_energy, cc.correlation_energy)
   print(cc.T.a.shape, cc.T.ab.shape)

   totals = eom.energy()
   print(eom.root_converged, eom.root_irreps, eom.omega)
   print(eom.guess_energy, eom.guess_irreps)

Do not infer convergence from the fact that ``energy()`` returned a number.
Most CC/EOM wrappers preserve convergence flags even on failure, and some can
return the last iterate. ADC and STEOM have additional fail-fast checks.
For repeated or suspicious EOM roots, inspect state amplitudes and overlap,
not just energy ordering. Tightening only the downstream tolerance does not
resolve an unconverged or inaccurate upstream CC calculation.

To restart ground CC from its current amplitudes:

.. code-block:: python

   cc.options["maximum_iterations"] = 160
   cc.options["amp_convergence"] = 1.e-9
   cc.converged = False
   cc.solve()

An explicit ``solve()`` also reruns without setting ``converged=False``.
Dependency tracking is version based; arbitrary in-place changes to ``T``,
orbital matrices or dictionaries are not automatically detected. After changing
amplitudes or options, explicitly solve the affected upstream object before
requesting downstream results. After changing orbitals, build a consistent
reference and chain rather than mutating a live calculation's basis.

Flexible combinations and HBar choices
--------------------------------------

.. code-block:: python

   from ccpy import CCSD, EOMCCSD, HBarCCSD, HBarCCSDLegacy

   cc = CCSD(mf, nfrozen=1)
   factorized = cc.get_hbar(HBarCCSD)
   materialized = cc.get_hbar(HBarCCSDLegacy)
   custom_eom = EOMCCSD(cc, hbar_class=HBarCCSD)

``get_hbar`` ensures the parent is solved and returns a built HBar wrapper;
its actual integral object is ``.H`` and its associated amplitudes are ``.T``.
Available HBars include ``HBarCC2``, ``HBarCC3``, ``HBarCCSD``,
``HBarCCSDLegacy``, ``HBarCCSDT``, ``HBarCCSDT1``, ``HBarCCSDTA`` and
``HBarCCSDT_P``. Use the HBar specified by the scientific model. Different
HBar classes may require amplitudes of different rank or selected lists;
construction is not a generic type-compatibility proof.

For ordinary runs, use the method's default HBar and left class. Advanced
combinations are useful when deliberately defining an approximation, but
should be tested against a standard chain or independent oracle. Reusing a
CCSD parent in a higher-rank EOM class does not automatically supply converged
CCSDT amplitudes. A correction object is not a replacement right-state
wavefunction: building EOM on it generally reuses its underlying CC method,
not a fictitious corrected T operator.

Implementing another method level
---------------------------------

A ground-state CC level can subclass ``CCBase`` and select its equation module,
operator order, HBar and left class. The module must provide the update protocol
expected by the chosen solver; inspect a nearby implementation before adding
one. A minimal definition is:

.. code-block:: python

   from ccpy.methods.cc import CCBase
   from ccpy.methods.hbar import HBarCCSD

   class MyCCLevel(CCBase):
       module = "my_package.my_cc_equations"
       order = 2
       hbar_class = HBarCCSD
       left_class = None
       requires_df_port = True

This is a protocol sketch; it cannot run until the named update module exists.
Declare ``requires_df_port`` truthfully. DF references prohibit accidental
full ``vvvv`` access by default; a legacy kernel needs materialization or a
proper DF port. If the model has a gradient, name ``gradient_module`` and
``gradient_function`` and implement the correct scheme-specific invocation.

An EOM level chooses the relevant EE/SF/Fock base, ``module`` containing ``HR``
and ``update``, parent/HBar classes and operator dimensions. Selected-space
levels also need their list keys, spin conventions and solver extra arguments.
Override ``_guess_arguments`` only when the new method requires a different
Hamiltonian/guess protocol, as ADC and STEOM do. ``InitialGuessMixin`` owns
spin/irrep/root configuration; avoid inventing an incompatible root numbering
scheme for a new family.

FNO preparation belongs to ``MethodBase``/``Reference``. New correlation
kernels should use the ordinary working dimensions and remain unaware of the
original virtual count. Apply composite energy shifts at reporting boundaries,
not inside CC/EOM residuals or iterative denominators. A new low-level FNO model
can use the provider contract in :doc:`spaces` and :doc:`../fno`.

Troubleshooting common setup mistakes
-------------------------------------

.. list-table:: Symptoms and next checks
   :header-rows: 1
   :widths: 38 62

   * - Symptom
     - Check
   * - Expected native SCF object
     - Use CCpy RHF/ROHF/UHF; PySCF objects belong to the legacy interface.
   * - Unknown option
     - Check constructor signature; do not use Driver option names such as nact_occupied in initial_guess.
   * - Call initial_guess before solving
     - Ordinary EOM/ADC need a configured guess pool and root selection.
   * - Missing irrep or unavailable root
     - Inspect point group, guess dimensions, active counts, spin and actual guess_irreps.
   * - Conflicting nfrozen / FNO settings
     - Start a fresh chain from SCF/full Reference instead of changing a shared basis downstream.
   * - Selected list exceeds dimensions
     - Regenerate after FNO preparation; use one-based particle-first rows.
   * - Property operator belongs to another reference
     - Transform the AO operator with the current reference.
   * - Folded ADC pole / no primary character
     - Select formulation="full" and suitable CISd guesses for the target.
   * - DFPortWarning / unexpected memory use
     - The selected method materializes four-index equations; inspect the catalog/registry.
   * - Repeated EOM states
     - Check overlaps/character, enlarge or change guesses, and consider multiroot where supported.
   * - Unexpected FNO gap correction
     - Ground completion cancels from gaps; state-specific ADC completion needs matching targets.
   * - Expensive gradient
     - Inspect has_analytic_gradient(); FNO and unsupported reference/variant combinations use finite differences.
