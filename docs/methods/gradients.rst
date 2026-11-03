Analytic gradients and finite differences
=========================================

An energy gradient is the derivative of a specified scalar energy surface.
Before using a derivative, choose the electronic method, frozen core, correction
variant and (for excited states) returned root. The same choices must be used
at every displaced or optimized geometry. Geometry optimizers need the gradient
of the energy they are minimizing, including any enabled energy completion.

The common method interface
---------------------------

.. code-block:: python

   from ccpy import Molecule, RHF, CCSD

   mol = Molecule("O 0 .05 .1173; H 0 .7572 -.4692; H .1 -.7572 -.4692",
                  "6-31g", auxiliary_basis="def2-universal-jkfit",
                  symmetry=False)
   cc = CCSD(RHF(mol, e_tol=1.e-12, d_tol=1.e-10), nfrozen=1,
             amp_convergence=1.e-10, energy_convergence=1.e-12,
             fd_step=1.e-3, fd_npoints=4)
   print(cc.has_analytic_gradient())
   energy, gradient = cc.energy_and_gradient()
   assert cc.converged
   print(energy, gradient)  # Eh; array (natoms, 3), Eh/bohr

``gradient()`` automatically chooses the analytic route when
``has_analytic_gradient()`` permits it. Otherwise it calls ``fd_gradient()``.
``energy_and_gradient()`` first obtains the energy, then the gradient on the
same calculation object, so analytic routes can share CC, HBar and left-state
work. An optimizer's forces are minus this gradient; pass gradients to CCpy's
adapter, which performs that sign conversion itself.

An analytic routine that raises an import, convergence or numerical error does
**not** silently fall back to finite differences. ``has_analytic_gradient`` is
a dispatch/capability check, not a substitute for testing an installed kernel
and checking all electronic convergence flags. Most ordinary CC/EOM wrappers
can retain unconverged iterates rather than raise exceptions; do not feed those
values into an optimization or derivative comparison.

Which methods use analytic kernels?
-----------------------------------

.. list-table:: Current methods-layer dispatch
   :header-rows: 1
   :widths: 28 34 38

   * - Method
     - Analytic path
     - Additional conditions / interpretation
   * - ``CCSD``
     - Registered CCSD kernel.
     - RHF/UHF; no deleted virtuals or FNOs. Includes required Lambda/orbital response.
   * - ``EOMCCSD``
     - Registered EOMCCSD kernel.
     - Same reference restrictions; select the returned root; left/response work is required.
   * - ``CRCC23``
     - Variant A only.
     - Deterministic correction; other variants use numerical gradients.
   * - ``CREOMCC23``
     - Variant A only.
     - Gradient of the size-intensive delta-CR energy returned by energy(), not full_energy().
   * - ``CCSDT_P``
     - Conditional registration.
     - Requires the selected CC density module; check the actual object. No guarantee from class name alone.
   * - Ground left-CC wrappers
     - Delegate to the parent's analytic gradient when available.
     - Their energy is the right ground-state energy; they do not define a new surface.
   * - MP2/MP3/MP4, ADC, STEOM, other CC/EOM levels and unregistered corrections
     - Finite-difference fallback.
     - Even when their energy equations use DF, they do not thereby have an analytic gradient.

ROHF, FNO references and ``ndelete != 0`` do not currently pass the general
methods-layer analytic eligibility check. Stochastic correction instances also
use the fallback. Frozen **occupied** cores are distinct from deleted virtuals;
a frozen core alone does not disable the registered RHF/UHF analytic routes.

For a deterministic CR-CC(2,3)_A example:

.. code-block:: python

   from ccpy import CRCC23

   cr = CRCC23(cc, variant="A")
   assert cr.has_analytic_gradient()
   corrected_energy, corrected_gradient = cr.energy_and_gradient()

Use the same correction variant in energy and gradient. In particular, changing
the default D correction to A changes the energy model, not just the derivative
algorithm. ``CREOMCC23.full_energy()`` has no separate analytic gradient method
in this wrapper; if that full energy is required, differentiate that scalar
explicitly with a callback.

What an analytic gradient contains
-----------------------------------

CC energy depends on nuclear positions through AO integrals, basis overlap and
molecular orbitals, as well as the correlated state. Analytic methods use left
amplitudes/response equations to account for this dependence without displacing
every coordinate. They include the appropriate integral and orbital response;
contracting an unrelaxed stored RDM with an integral derivative alone generally
does not produce the full molecular gradient.

``cc.rdm1()/rdm2()`` expose derivative densities in the T1-transformed frame;
:doc:`properties` explains that convention. These are not directly relaxed AO
total-state RDMs. ``zvector_tol`` controls orbital-response solves on analytic
routes that accept it. Use it only after checking that the requested route is
analytic: arbitrary analytic-only kwargs are not filtered out when a method
falls back to numerical differentiation and may reach ``energy()`` instead.

Force numerical gradients deliberately
---------------------------------------

.. code-block:: python

   numerical = cc.fd_gradient()  # bypasses analytic dispatch
   analytic = cc.gradient()
   import numpy as np
   print("max difference (Eh/bohr):", np.max(np.abs(analytic - numerical)))
   print("net gradient:", analytic.sum(axis=0))
   xyz = np.array([xyz for _, xyz in mol.atoms])  # bohr
   print("net torque diagnostic:", np.cross(xyz, analytic).sum(axis=0))

Set ``fd_step`` and ``fd_npoints`` on the method constructor. Do not call
``method.fd_gradient(step=...)``: its kwargs specify energy arguments such as
``root``, not stencil settings. To change stencil settings after construction,
recreate the method with explicit constructor options; this also preserves the
settings correctly when ``at_geometry`` rebuilds the chain.

The available central stencils use 2, 4 or 6 points per coordinate, with leading
truncation error of order h squared, h to the fourth or h to the sixth.
For the four-point stencil:

.. math::

   \frac{\partial E}{\partial x}\simeq
   \frac{E(x-2h)-8E(x-h)+8E(x+h)-E(x+2h)}{12h}.

The default is h=1e-3 bohr and four points. A complete p-point gradient of an
N-atom molecule requires p times 3N displaced scalar energy evaluations,
in addition to any central energy requested separately. All displaced
calculations rebuild SCF and the recorded correlated chain. This implementation
does not propagate converged amplitudes/orbitals between geometries as a warm
start.

Compare more than one displacement if precision matters. Making h smaller
reduces truncation error but amplifies electronic energy noise approximately
as energy_error/h. Tighten SCF, CC and left/response convergence before using
very small steps. Translation/rotation sum rules are useful diagnostics at a
fixed isolated-molecule geometry, but alone do not establish correctness of
every gradient component or the electronic state identity.

Energy-only functions and partial checks
----------------------------------------

.. code-block:: python

   from ccpy import MP2
   from ccpy.gradients.finite_difference import fd_gradient
   from ccpy.optimize import fd_energy_and_gradient

   def energy_at(molecule):
       method = MP2(RHF(molecule, verbose=0), nfrozen=1)
       energy = method.energy()
       if not method.converged:
           raise RuntimeError("MP2 setup did not converge")
       return energy

   gradient = fd_gradient(energy_at, mol, step=1.e-3, npoints=4)
   atom0_only = fd_gradient(energy_at, mol, atoms=[0], step=1.e-3, npoints=4)
   callback = fd_energy_and_gradient(energy_at, step=1.e-3, npoints=4)
   energy, gradient = callback(mol)

The function must return a scalar total energy for a supplied native
``Molecule`` and repeat the full electronic calculation. ``atoms`` contains
zero-based atom indices; unchecked rows of the returned gradient are zero.
Such a partial gradient is useful for validation, not an unconstrained geometry
optimization. The helper ``fd_energy_and_gradient`` packages numerical
differentiation for the pysisyphus interface.

Excited-state gradients and state following
-------------------------------------------

.. code-block:: python

   from ccpy import EOMCCSD

   eom = EOMCCSD(cc)
   eom.initial_guess(method="cisd", multiplicity=1, nacto=2, nactu=2,
                     roots_per_irrep={"A": 4}, roots=[1, 3])
   total, gradient = eom.energy_and_gradient(root=1)
   numerical = eom.fd_gradient(root=1)

Here ``root=1`` is the second returned state, seeded by guess index 3. It is
not the second guess in the original pool. Calling ``energy_and_gradient()``
without ``root`` on a multiroot method does not define a single scalar surface:
``energy()`` returns an array while the analytic gradient route defaults to
root 0; numerical differentiation can reject a nonscalar energy. This is not
the scalar energy/gradient pair required by the optimization calculator. Always select one returned root in an explicit
callback for excited-state optimization.

Keep the same physical state at displaced geometries and along an optimization
or IRC. ``at_geometry`` copies guess settings; it does not provide a universal
cross-geometry state-overlap tracker. Inspect character, irreps and neighboring
states near avoided crossings and degeneracies. The TS Hessian mode index is
unrelated to an electronic EOM root index. See :doc:`geometry` for a scalar
excited-state callback and optimizer setup.

FNOs, discrete spaces and smooth surfaces
-----------------------------------------

Numerical differentiation evaluates the **completed** ``energy()`` by default.
For FNOs it rebuilds the density, virtual selection, transformed factors and
low-level completion at each displacement. Analytic gradients do not currently
include these FNO-selection derivatives. A cutoff crossing, changing protected
span or lost target root can make the energy nonsmooth. Check retained counts,
root identity and step/threshold convergence before trusting gradients or
Hessian signs. A fixed retained count alone does not establish smoothness at
degeneracies.

Selected P/Q spaces and external CI data must also remain meaningful across
geometries. ``AdaptiveCCPQ.at_geometry`` reuses its recorded determinant spaces;
that is different from rerunning independent adaptive selection at each
geometry. Ordinary user-supplied determinant labels are not automatic orbital
tracking. For stochastic energies, differentiation can amplify sampling noise;
use deterministic corrections for derivative checks and stationary-point work
unless a consistent noise-controlled protocol has been established.

Runnable validation
---------------------

.. code-block:: bash

   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/methods_api/geometry_workflows.py \
       --task gradient-check --method ccsd --basis sto-3g --out-dir gradient-check

This example compares automatic analytic CCSD gradients with an independently
displaced energy gradient at a distorted water geometry. Method-layer tests are
in ``tests/methods/test_method_gradients.py``; additional SCF and correction
checks are in ``tests/scf``. The example's small basis demonstrates the workflow,
not a benchmark-quality geometry or barrier.
