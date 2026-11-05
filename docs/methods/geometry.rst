Geometry optimization, transition states and IRC validation
===========================================================

CCpy's ``ccpy.optimize`` interface connects native energy/gradient calculations
to pysisyphus. It provides ``optimize_geometry`` for minima,
``optimize_transition_state`` for first-order saddles,
``fd_energy_and_gradient`` for energy-only functions, and ``make_calculator``
for direct pysisyphus use. ``run_irc`` is exported by both ``ccpy`` and
``ccpy.optimize``; it integrates an Euler predictor-corrector IRC and returns
native endpoint molecules, trajectories and branch diagnostics.

Build the native molecule with **symmetry=False**. Geometry changes and finite
differences preserve the supplied Cartesian frame and atom order; symmetry
detection/reorientation would invalidate that assumption. Use a separate working
directory for each optimizer/IRC calculation, since the external algorithms
write trajectories, logs and Hessian data.

Installation and units
-----------------------

pysisyphus is listed as a CCpy runtime dependency. If it is missing from an
existing environment, install it in that same environment:

.. code-block:: bash

   python -m pip install pysisyphus

The native CCpy calculator does not need an external QC executable. A missing
``.pysisyphusrc`` notice from pysisyphus does not itself prevent this Python
calculator from running.

``Molecule`` defaults to angstrom for text input. Native stored coordinates and
pysisyphus Cartesian arrays are **bohr**. Calculator energies are hartree;
gradients are Eh/bohr and Cartesian Hessians are Eh/bohr squared. The bridge
converts gradients into negative Cartesian forces. Do not negate the gradient
in your callback. ``get_normal_modes`` applies masses and projects rigid motions
before reporting signed vibrational wavenumbers in inverse centimeters.

Minimize an electronic method directly
---------------------------------------

.. code-block:: python

   from ccpy import Molecule, RHF, CCSD
   from ccpy.optimize import optimize_geometry

   mol = Molecule("O 0 0 .12; H 0 .80 -.47; H 0 -.78 -.45",
                  "6-31g", auxiliary_basis="def2-universal-jkfit",
                  symmetry=False)
   cc = CCSD(RHF(mol, verbose=0, e_tol=1.e-12, d_tol=1.e-10), nfrozen=1,
             amp_convergence=1.e-10, energy_convergence=1.e-12)
   minimum = optimize_geometry(cc, coord_type="redund", thresh="gau_tight",
                               max_cycles=100, hessian_init="fischer")
   print(minimum.converged, minimum.energy, minimum.ncycles)
   print(minimum.molecule.atoms)  # optimized coordinates in bohr

This creates a pysisyphus rational-function optimizer and evaluates
``cc.at_geometry(mol).energy_and_gradient()`` at requested geometries. It chooses
analytic gradients where available and otherwise uses the method's finite
differences. The originally supplied ``cc`` object remains the template; its
stored molecule and amplitudes are not the optimized result. Reconstruct with
``cc.at_geometry(minimum.molecule)`` for final properties.

Configure ``nfrozen``, solver thresholds, FNO policy and correction variant on
the template. Geometry copies reproduce that constructor chain. For example,
``optimize_geometry(CRCC23(cc, variant="A"))`` minimizes the analytic-A surface;
variant D instead uses a numerical gradient. An MP2/ADC/FNO template may require
many energy evaluations per optimizer step because it lacks an eligible
analytic derivative. See :doc:`gradients` before choosing a costly model.

A completed optimizer run is not necessarily converged. Inspect
``result.converged`` and verify the final gradient. The wrapper does not
universally reject every unconverged inner electronic iterate. For explicit
control over SCF, CC and left-state convergence, use a callback.

Use a callback for custom calculations
---------------------------------------

.. code-block:: python

   from ccpy import CCSD, CRCC23, RHF

   def energy_and_gradient(molecule):
       mf = RHF(molecule, verbose=0, e_tol=1.e-12, d_tol=1.e-10)
       cc = CCSD(mf, nfrozen=1, amp_convergence=1.e-10,
                 energy_convergence=1.e-12)
       corrected = CRCC23(cc, variant="A")
       energy, gradient = corrected.energy_and_gradient()
       if not (mf.converged and cc.converged and corrected.converged
               and corrected.left.converged):
           raise RuntimeError("Unconverged electronic calculation")
       return float(energy), gradient

   minimum = optimize_geometry(mol, energy_and_gradient, thresh="gau_tight")

The callback contract is ``callable(Molecule) -> (float, array(natoms,3))``.
Do not pass ``gradient_and_dipole`` here: its return is ``(g,mu)``, not ``(E,g)``.
Use the native Hessian/IR helpers in :doc:`multipoles` for dipole intensities.
It lets you select correction variants, check convergence, target an excited
state, or define a custom scalar composite energy consistently. Pass either a
method or a molecule plus callback, not both method and callback.

For an energy-only calculation:

.. code-block:: python

   from ccpy import MP2
   from ccpy.optimize import fd_energy_and_gradient

   def energy_only(molecule):
       method = MP2(RHF(molecule, verbose=0), nfrozen=1)
       return method.energy()

   numerical = fd_energy_and_gradient(energy_only, step=1.e-3, npoints=4)
   minimum = optimize_geometry(mol, numerical, thresh="gau_tight")

``make_calculator`` caches the last Cartesian geometry's energy and gradient,
so an energy request followed by a force request at identical coordinates
needs one callback evaluation. This is not a persistent calculation cache or
an orbital/amplitude warm start. Energy-only requests still invoke the combined
callback and can therefore calculate a gradient.

Optimization controls and result fields
----------------------------------------

.. list-table:: Minimum and saddle wrappers
   :header-rows: 1
   :widths: 26 28 46

   * - Setting
     - Default
     - Meaning
   * - ``coord_type``
     - ``"redund"``
     - Redundant internals. Also supports the installed pysisyphus ``"dlc"``, ``"tric"`` or ``"cart"``.
   * - ``thresh``
     - ``"gau"``
     - Named convergence preset; ``"gau_loose"`` / ``"gau_tight"`` are also available.
   * - ``max_cycles``
     - 100
     - Maximum optimizer cycles, separate from electronic iterations.
   * - ``hessian_init`` (minimum)
     - ``"fischer"``
     - Model Hessian; ``"lindh"`` / ``"swart"`` or ``"calc"`` also supported.
   * - ``hessian_init`` (TS)
     - ``"calc"``
     - Finite-difference Hessian from calculator gradients.
   * - ``root`` (TS)
     - 0
     - Lowest Hessian mode followed uphill; **not** an electronic-state root.
   * - ``hessian_recalc`` (TS)
     - None
     - Periodic Hessian recomputation interval; otherwise uses the optimizer's updates.
   * - ``hessian_step`` (TS / calculator)
     - 5e-3 bohr
     - Cartesian gradient-difference displacement for the Hessian.

Extra keyword arguments go to ``RFOptimizer`` or ``RSPRFOptimizer``. Check the
installed pysisyphus signatures before passing advanced controls; the wrapper
already supplies ``dump=False``, so do not pass ``dump`` a second time.
``verbose=False`` suppresses bridge messages, not necessarily all external
optimizer output. ``optimize_geometry`` does not expose a separate
``hessian_step`` parameter: use ``make_calculator`` and the external optimizer
directly to customize that minimum Hessian displacement.

``OptimizationResult`` exposes ``molecule``, ``energy``, ``converged``,
``ncycles`` and ``optimizer`` (the external optimizer/history).
``hessian_eigenvalues`` is filled only by the TS wrapper. It contains raw
**unweighted Cartesian** Hessian eigenvalues, not vibrational frequencies or
projected mass-weighted mode eigenvalues. The minimum wrapper does not perform
a final frequency calculation automatically.

Hessian evaluation and cost
----------------------------

CCpy's calculator computes a Cartesian Hessian by two-point differences of
Cartesian gradients at +/- ``hessian_step``, then symmetrizes it. This requires
6N displaced gradients for N atoms, plus a central energy/force request as
needed. There is no analytic molecular Hessian exposed by this bridge.

With numerical gradients, the Hessian is a nested finite difference and its
cost is quadratic in atom count times the scalar energy cost. ``fd_step`` /
``fd_npoints`` control the **inner energy gradient**, while ``hessian_step``
controls the **outer gradient derivative**. Tight electronic convergence and
step-size checks are essential when distinguishing weak imaginary modes from
noise. A model Hessian is useful to start minimization; it is not a final
vibrational characterization.

Find a transition state
-------------------------

Supply a chemically plausible near-saddle guess. The following ammonia inversion
example starts close to planar and follows the umbrella mode:

.. code-block:: python

   from ccpy.optimize import optimize_transition_state

   guess = Molecule("N 0 0 .10; H .95 0 0; H -.475 .8227 0; H -.475 -.8227 0",
                    "6-31g", auxiliary_basis="def2-universal-jkfit",
                    symmetry=False)
   method = CCSD(RHF(guess, verbose=0, e_tol=1.e-12, d_tol=1.e-10),
                 nfrozen=1, amp_convergence=1.e-10, energy_convergence=1.e-12)
   ts = optimize_transition_state(
       method, root=0, coord_type="redund", thresh="gau_tight",
       hessian_init="calc", hessian_recalc=5, hessian_step=5.e-3,
       max_cycles=100,
   )
   print(ts.converged, ts.energy, ts.hessian_eigenvalues)

The wrapper uses restricted-step partitioned RFO, maximizing along one mode
and minimizing the others. ``root=0`` is appropriate when the lowest relevant
mode is the desired reaction coordinate. A poor starting guess can have no
negative mode, several negative modes or the wrong unstable motion. Inspect
mode displacements rather than accepting a converged optimizer flag as proof
of the intended transition state.

After the search the wrapper **always** computes a final finite-difference
Cartesian Hessian, even if optimization did not converge. Check ``ts.converged``
first, then characterize its projected vibrational modes and verify the chemical
motion. Do not count tiny negative raw Cartesian eigenvalues as separate
imaginary vibrations; translations/rotations and numerical noise can contribute.

Projected vibrational analysis
--------------------------------

See :doc:`frequencies` for a complete minimum-frequency calculation, mode
displacements, isotope substitution, Hessian reuse and numerical convergence.

Use public pysisyphus objects with CCpy's calculator, on the same electronic
surface and basis as the optimization:

.. code-block:: python

   import numpy as np
   from pysisyphus.Geometry import Geometry
   from ccpy.integrals.atom_data import Z_TO_ATOM_SYMBOL
   from ccpy.optimize import make_calculator

   def at_geometry(molecule):
       calculation = method.at_geometry(molecule)
       energy, gradient = calculation.energy_and_gradient()
       if not calculation.converged:
           raise RuntimeError("Unconverged electronic state")
       return float(energy), gradient

   atoms = [Z_TO_ATOM_SYMBOL[int(z)] for z in ts.molecule.atomic_charges]
   coords = np.array([xyz for _, xyz in ts.molecule.atoms]).ravel()
   geom = Geometry(atoms, coords, coord_type="cart")
   geom.set_calculator(make_calculator(ts.molecule, at_geometry,
                                      hessian_step=5.e-3, verbose=False))
   frequencies, eigenvalues, mass_weighted_modes, cartesian_modes = geom.get_normal_modes()
   print(frequencies)  # cm^-1; negative values denote imaginary modes

``get_normal_modes`` mass-weights the Hessian and projects rigid motions. A
nonlinear isolated molecule has 3N-6 vibrational modes (3N-5 if linear), with
special handling needed for atomic/degenerate limits. A minimum should have
no significant imaginary vibration; a first-order saddle should have exactly
one. The associated displacement must follow the proposed chemical reaction.
Low-frequency negative values require numerical checks rather than an arbitrary
universal cutoff. The runnable example uses -20 cm^-1 only as a practical
small-molecule demonstration threshold.

This standalone analysis computes a Hessian again because the CCpy TS result
stores only its eigenvalues. The resulting ``geom`` retains the Hessian cache
and can serve as the initial geometry for direct pysisyphus use. Keep atom order, charge,
spin, isotopic masses and coordinate frame consistent when reusing Hessians.

Run both IRC branches with run_irc
----------------------------------

Use the converged TS molecule and the same scalar electronic callback used
for the saddle search:

.. code-block:: python

   from ccpy import run_irc

   result = run_irc(
       ts.molecule, at_geometry, forward=True, backward=True,
       hessian_init="calc", root=0, hessian_step=5.e-3,
       step_length=0.05, max_cycles=150, rms_grad_thresh=5.e-4,
       displ="length", displ_length=0.1,
       out_dir="irc", prefix="reaction", verbose=False,
   )
   print(result.converged, result.start_energy)
   for direction, branch in result.branches.items():
       print(direction, branch.converged, branch.energy_converged,
             branch.energy_increased, branch.gradient_rms, branch.ncycles)

Alternatively pass a method rebuilt on the saddle geometry:

.. code-block:: python

   result = run_irc(method.at_geometry(ts.molecule), out_dir="method-irc")

This repeats the method's configured scalar energy/gradient calculation at each
geometry and checks its convergence flag. For an excited-state method select
its **returned electronic root** with ``state``:

.. code-block:: python

   result = run_irc(eom_template.at_geometry(ts.molecule), state=1, root=0,
                    out_dir="excited-irc")

Here ``state=1`` is passed to the method's ``energy(root=1)`` and
``gradient(root=1)``; ``root=0`` selects the Hessian mode. An array-valued energy
without an explicit scalar state raises a clear error before computing the
method gradient. With a molecule/callback input, select the state inside the
callback instead of passing ``state``. The wrapper checks that every energy is
a finite real scalar and every gradient has finite real shape ``(natoms,3)``.
It does not implement universal electronic root tracking; keep the state-following
checks discussed below.

The result types ``IRCResult`` and ``IRCBranch`` are also exported by CCpy:

* ``start_molecule`` and ``start_energy`` describe the input geometry.
* ``branches`` maps requested directions to ``IRCBranch`` objects.
* Each branch stores ``coordinates`` in Cartesian bohr with shape
  ``(npoints,natoms,3)``, ``energies`` in hartree, and ``gradients`` in Eh/bohr.
  Points run from the initial displacement toward the last stored endpoint;
  the separate starting TS is not prepended to each branch.
* ``branch.molecule`` is the native molecule at the last stored point;
  ``result.endpoints`` maps directions to these molecules.
* ``converged`` means all requested branches met the external integrator's
  convergence criteria. A cycle-limited path returns ``False``; it does not
  silently become a validated connection. ``gradient_rms`` is evaluated at the
  last stored branch point. ``ncycles`` is the external cycle index plus one.
* ``result.irc`` exposes the original pysisyphus object for complete histories,
  mass-weighted coordinates and additional diagnostics.

Both directions are enabled by default. Disable one with ``forward=False`` or
``backward=False``. ``downhill=True`` instead returns a single ``"downhill"``
branch and ignores those two switches. At least one direction is required when
not running downhill. Extra keyword arguments are passed to pysisyphus
``EulerPC``/``IRC``, including initial-displacement settings, ``energy_thresh``,
``hessian_recalc`` and ``dump_every``. A saved Hessian path can be supplied as
``hessian_init``; verify matching geometry, atom order, units and electronic
surface before reusing it. The native wrapper starts from a molecule, so it
calculates its own Hessian unless a file is supplied; it does not implicitly
reuse the standalone ``geom`` object's cache from the frequency example.

EulerPC integrates the reaction path in mass-weighted coordinates, starting in
opposite directions along the transition vector. Here ``displ_length`` is the
initial mass-weighted displacement (sqrt(amu) bohr). ``step_length`` is a
separate integrator step control; the installed EulerPC measures its target
integration length in unweighted Cartesian coordinates. Do not confuse either
with ``hessian_step`` or a finite-difference gradient displacement.
``max_cycles`` applies per branch. These explicit values are an example, not a
universal converged IRC protocol. Reduce the integration step and tighten
thresholds to check path/endpoint stability.

A branch may terminate through energy-change criteria rather than a small final
gradient. Inspect the per-branch flags, cycle counts and actual endpoint gradients.
Reaching ``max_cycles`` is not validation. ``downhill=True`` is a different
single downhill trajectory from a nonstationary starting point; it does not
replace both branches from a stationary saddle.

The interface writes pysisyphus trajectories, logs and Hessian/HDF5 data;
``prefix`` and ``out_dir`` control the IRC files, while some external outputs
can still be relative to the process working directory. Run in a dedicated
directory. ``pysisplot --irc`` can visualize the saved IRC data; consult the
installed pysisyphus command help for file arguments and output conventions.

Optimize and validate the endpoints
------------------------------------

``run_irc`` returns the last stored geometries without optimizing or
vibrationally characterizing them. Minimize its native endpoint molecules
on the same energy surface:

.. code-block:: python

   endpoints = {}
   for direction, endpoint in result.endpoints.items():
       relaxed = optimize_geometry(endpoint, at_geometry, thresh="gau_tight")
       if not relaxed.converged:
           raise RuntimeError("Endpoint optimization did not converge")
       endpoints[direction] = relaxed
       print(direction, relaxed.energy, ts.energy - relaxed.energy)

Use the projected frequency procedure on **both** relaxed endpoints and check
that they are minima. Then compare their bonding, structures, atom mapping and
state character with the intended reactant/product basins. Both should lie
below the saddle on that same surface; otherwise inspect optimization,
convergence or state tracking. A single imaginary TS frequency alone does not
establish a connection to the desired pair of minima.

Forward/backward labels are arbitrary: the unstable eigenvector can change
sign. Symmetric reactions may lead to symmetry-equivalent endpoints rather
than different chemical connectivities. For ammonia inversion the preserved
N/H atom labels allow comparing the signed N height relative to the ordered H
plane: opposite signs distinguish the two umbrella minima. Do not require two
arbitrary, unsuperimposed Cartesian coordinate arrays to be equal when matching
geometries.

Electronic excited states and custom surfaces
---------------------------------------------

Use a scalar root-aware callback instead of passing a multiroot EOM method
directly to the wrappers:

.. code-block:: python

   from ccpy import EOMCCSD

   eom_template = EOMCCSD(method)
   eom_template.initial_guess(method="cisd", multiplicity=1, nacto=2, nactu=2,
                              roots_per_irrep={"A": 3}, roots=[0, 2])

   def excited_surface(molecule):
       eom = eom_template.at_geometry(molecule)
       energy, gradient = eom.energy_and_gradient(root=1)
       if not (eom.converged and eom.cc.converged and eom.root_converged[1]):
           raise RuntimeError("Unconverged target")
       return float(energy), gradient

   excited_minimum = optimize_geometry(eom_template.molecule, excited_surface)

Here the selected electronic state is returned root 1, seeded by guess index 2.
The TS/IRC ``root`` instead selects a Hessian mode. Root-specific callbacks also
work for corrected EOM/ADC energies if their gradient route and state identity
are appropriate. Generic ``at_geometry`` repeats guesses; it does not implement
universal electronic root tracking across crossings. Inspect amplitudes,
overlap and neighboring energies along the path. Near degeneracies, a
single smooth adiabatic surface and ordinary first-order saddle/IRC description
can fail; the interface does not supply nonadiabatic couplings or crossing
optimization automatically.

FNO cutoffs and adaptive selected spaces can change discretely along a path.
Use the smoothness checks in :doc:`gradients` and consistent parent/target
completion conventions. Numerical differentiation of an FNO-completed energy
includes the rebuilt low-level correction; it does not make a cutoff-discontinuous
surface differentiable.

Runnable end-to-end examples
-----------------------------

.. code-block:: bash

   python examples/methods_api/geometry_workflows.py --task minimum --method ccsd \
       --basis sto-3g --out-dir minimum-run
   python examples/methods_api/geometry_workflows.py --task minimum --method mp2 \
       --basis sto-3g --out-dir numerical-minimum
   python examples/methods_api/geometry_workflows.py --task ts --method rhf \
       --basis sto-3g --out-dir ts-run
   python examples/methods_api/geometry_workflows.py --task irc --method rhf \
       --basis sto-3g --out-dir irc-run

``--finite-difference`` explicitly replaces analytic gradients by displaced
energies. ``--method crcc23a`` selects the deterministic analytic-A correction.
The IRC task searches for the ammonia inversion saddle, checks its projected
frequencies, integrates both branches, relaxes the endpoints and checks their
frequencies and opposite umbrella signs. ``summary.json`` records the results.
These small-basis examples validate wiring and conventions, not quantitative
reaction energetics. Do not assume a successful RHF path establishes a CC or
excited-state path without repeating that surface's calculations.

Validation of the shipped example
-----------------------------------

The examples were run with pysisyphus 1.0.0, native DF-RHF/CCSD and the default
STO-3G / def2-universal-jkfit demonstration setup. CCSD analytic versus
four-point energy-difference gradients at distorted water differed by about
6.1e-11 Eh/bohr. The optimized CCSD water had three positive vibrational modes.
The RHF ammonia saddle had one imaginary mode near -1081.5 cm^-1; both IRC
branches relaxed to positive-frequency minima near -55.4555641 Eh, below the
saddle near -55.4378011 Eh, with opposite signed umbrella heights. These are
sanity checks for the runnable example, not reference-quality reaction data or
a guarantee of a CCSD IRC from an RHF path.

``tests/scf/test_irc_interface.py`` additionally exercises the public wrapper
on both ammonia branches, native endpoint relaxation, single-branch and
cycle-limited results, electronic-state selection and invalid inputs.

The existing direct-method optimization test in
``tests/methods/test_method_gradients.py::test_optimize_geometry_with_a_method``
also checks the CCpy-object-to-pysisyphus wrapper. Larger methods, basis sets,
excited states and arbitrary reactions require their own validation.

External documentation and version checks
-----------------------------------------

The examples were checked against the locally installed pysisyphus API. Record
your installed version and inspect signatures when transferring a workflow:

.. code-block:: python

   import inspect
   from importlib.metadata import version
   from pysisyphus.irc.EulerPC import EulerPC
   from pysisyphus.irc.IRC import IRC

   print(version("pysisyphus"))
   print(inspect.signature(EulerPC.__init__))
   print(inspect.signature(IRC.__init__))

See the primary `pysisyphus IRC documentation
<https://pysisyphus.readthedocs.io/en/master/irc.html>`_ and
`public Geometry API
<https://pysisyphus.readthedocs.io/en/latest/api/pysisyphus.html>`_. Online and
installed defaults can differ; the examples explicitly specify the important
path controls. CCpy bridge signatures are in
``ccpy/optimize/pysisyphus_interface.py``.
