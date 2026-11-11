Computing vibrational frequencies
=================================

Harmonic vibrational frequencies come from the **mass-weighted molecular
Hessian**, with translations and rotations projected out. CCpy provides
Cartesian energies/gradients and the finite-difference Hessian through
``make_calculator``; pysisyphus performs the mass weighting, projection and
normal-mode analysis. There is no separate method-object ``frequencies()`` call or analytic
molecular Hessian exposed by this interface. CCpy also supplies native
``fd_hessian``, ``harmonic_analysis`` and ``vibrational_analysis`` helpers,
including IR intensities when dipoles are supplied; see :doc:`multipoles`.

Native frequency and IR helpers
--------------------------------

``ccpy.gradients.hessian.vibrational_analysis(molecule, energy=callback)``
computes an energy-only Hessian using second differences.
``vibrational_analysis(molecule, gradient=callback)`` instead differences the
gradient, accepting ``g`` or ``(E,g)``. With ``dipole=True`` the callback must
return ``(g,mu)`` or ``(E,g,mu)`` and native analysis additionally reports IR
intensities. These are distinct callback contracts; see :doc:`multipoles` for
a complete example and finite-field alternatives.

The native helper's default ``step`` is 2.5e-3 bohr, while the pysisyphus
calculator below defaults to ``hessian_step=5e-3``. Native results are a dictionary,
with modes shaped ``(nvib,natoms,3)``, reduced masses, force constants and a
positive-real-mode ZPE; pysisyphus returns the arrays described below. Match mass
tables and numerical steps when comparing them.

A complete optimized-geometry example
-------------------------------------

The following computes a CCSD minimum and its harmonic frequencies using native
DF integrals, analytic CCSD gradients and a numerically differentiated Hessian:

.. code-block:: python

   import numpy as np
   from ccpy import Molecule, RHF, CCSD
   from ccpy.integrals.atom_data import Z_TO_ATOM_SYMBOL
   from ccpy.optimize import optimize_geometry, make_calculator
   from pysisyphus.Geometry import Geometry

   mol = Molecule("O 0 0 .12; H 0 .80 -.47; H 0 -.78 -.45", "sto-3g",
                  auxiliary_basis="def2-universal-jkfit", symmetry=False)
   template = CCSD(RHF(mol, verbose=0, e_tol=1.e-12, d_tol=1.e-10),
                   nfrozen=1, amp_convergence=1.e-10, energy_convergence=1.e-12)
   minimum = optimize_geometry(template, thresh="gau_tight", verbose=False)
   if not minimum.converged:
       raise RuntimeError("Optimize the geometry before characterizing its vibrations")

   def energy_and_gradient(molecule):
       calculation = template.at_geometry(molecule)
       energy, gradient = calculation.energy_and_gradient()
       if not (calculation.converged and calculation.reference.scf.converged
               and calculation.left.converged):
           raise RuntimeError("Unconverged electronic calculation")
       return float(energy), gradient

   optimized = minimum.molecule
   atoms = [Z_TO_ATOM_SYMBOL[int(z)] for z in optimized.atomic_charges]
   xyz_bohr = np.array([xyz for _, xyz in optimized.atoms])
   geom = Geometry(atoms, xyz_bohr.ravel(), coord_type="cart")
   geom.set_calculator(make_calculator(optimized, energy_and_gradient,
                                      hessian_step=5.e-3, verbose=False))

   hessian = geom.cart_hessian  # shape (3N,3N), Eh/bohr^2; cached on geom
   frequencies, eigenvalues, mw_modes, cart_modes = geom.get_normal_modes()
   print("Harmonic frequencies (cm^-1):", frequencies)
   np.savetxt("frequencies_cm-1.txt", frequencies)
   np.save("cartesian_hessian.npy", hessian)
   np.save("cartesian_normal_modes.npy", cart_modes)

STO-3G here keeps the demonstration small; choose an appropriate basis and
converge the electronic model for scientific use. Build with ``symmetry=False``
so displaced geometries retain their frame and atom order. Input geometry text
defaults to angstrom, but stored coordinates and pysisyphus arrays are bohr.
Use the same electronic method, frozen core, correction variant and state at
the optimization and Hessian geometries unless deliberately defining a
mixed-level protocol.

Reading the returned arrays
-----------------------------

``get_normal_modes()`` returns four arrays:

* ``frequencies``: signed vibrational wavenumbers in cm^-1. Negative values
  denote imaginary frequencies; they are not negative real vibrational energies.
* ``eigenvalues``: eigenvalues of the projected mass-weighted Hessian, before
  conversion to wavenumbers. These are not the raw Cartesian Hessian eigenvalues.
* ``mw_modes``: mass-weighted Cartesian mode vectors in columns.
* ``cart_modes``: Cartesian displacement vectors in columns, each normalized
  to unit Euclidean norm by the external routine. Shape is ``(3N,nvib)``.

An isolated nonlinear molecule has 3N-6 vibrational modes, and a linear molecule
has 3N-5. Atomic and degenerate geometries require appropriate special handling.
Water therefore has three modes. Do not manually discard the first six values
from ``frequencies``: rigid motions have already been projected out.
The raw ``hessian`` still has all 3N Cartesian directions.

The normal modes solve the projected version of

.. math::

   H_{\mathrm{mw}}=M^{-1/2}H_{\mathrm{cart}}M^{-1/2}.

The routine performs the unit conversion; do not reinterpret raw Cartesian
eigenvalues as frequencies or apply a second mass weighting. In particular,
``optimize_transition_state(...).hessian_eigenvalues`` are unweighted Cartesian
values and cannot be used directly as cm^-1 frequencies.

For a minimum, expect no significant imaginary vibrational mode. For a
first-order transition state, expect exactly one, with displacement along the
intended reaction coordinate. An unconverged geometry, loose electronic
thresholds, an unsuitable displacement, root switching or a nonsmooth FNO
selection can produce misleading negative modes. There is no universal
frequency cutoff that distinguishes noise from a physical instability.

Inspect a mode displacement
----------------------------

.. code-block:: python

   k = 0  # mode column; inspect frequencies[k] first
   displacement = cart_modes[:, k].reshape(-1, 3)
   amplitude_bohr = 0.1
   plus = optimized.with_geometry(xyz_bohr + amplitude_bohr * displacement,
                                  unit="bohr")
   minus = optimized.with_geometry(xyz_bohr - amplitude_bohr * displacement,
                                   unit="bohr")

These are geometries displaced in opposite directions along the normalized
mode for visualization or follow-up calculations. The arbitrary amplitude is
not a thermal or zero-point normal-coordinate amplitude. For a TS, inspect the
imaginary mode and then follow both IRC branches; see :doc:`geometry`.

Displacement size, cost and accuracy
-------------------------------------

``hessian_step`` (default 5e-3 bohr) controls the two-point central difference
of gradients used to build the Hessian. It is independent of the optimizer's
step size and the ``fd_step`` / ``fd_npoints`` controls used when the gradient
itself comes from displaced energies. For N atoms a complete Cartesian Hessian
requires 6N displaced gradients, plus a central energy/force request as needed.
When analytic gradients are unavailable, the resulting nested finite
differences are substantially more expensive.

Compare, for example, Hessians built using steps 5e-3 and 2.5e-3 bohr on the
**same** optimized geometry with equally tight electronic convergence. Create a
fresh Geometry/calculator for the second step: changing a variable after
``geom.cart_hessian`` was cached does not recompute the Hessian. Check stability
of weak modes, not just large stretching frequencies. Symmetrization of the
computed Hessian does not remove finite-difference or electronic noise.

``geom.get_normal_modes(cart_hessian=H)`` can analyze an explicitly supplied
Cartesian Hessian without recomputing it. To reuse the saved one:

.. code-block:: python

   saved_hessian = np.load("cartesian_hessian.npy")
   frequencies_again, _, _, _ = geom.get_normal_modes(cart_hessian=saved_hessian)

Verify geometry, frame, atom order, shape, units and electronic surface before
reusing a Hessian. A Hessian from another geometry does not characterize the
current stationary point.

Isotopes and other electronic surfaces
---------------------------------------

pysisyphus selects its default atomic masses from its mass table. To request
an isotope, construct another Geometry with zero-based atom-index/mass pairs:

.. code-block:: python

   deuterated = Geometry(atoms, xyz_bohr.ravel(), coord_type="cart",
                         isotopes=[(1, 2), (2, 2)])
   isotope_frequencies, _, _, _ = deuterated.get_normal_modes(cart_hessian=hessian)

An integer requests an isotope mass number; a floating-point value supplies a
mass directly in atomic mass units. Within the Born-Oppenheimer harmonic model,
isotopic substitution changes the mass weighting, so the same electronic
Cartesian Hessian can be reused. Recompute the projection/modes with the new
masses; do not rescale all frequencies by one global factor.

For an excited state, make the scalar callback call
``energy_and_gradient(root=selected_returned_root)``. Select and follow the same
physical root at all displaced geometries. For FNO or selected-space methods,
verify smoothness of the truncated space and completed energy as described in
:doc:`gradients`. The calculator uses the callback's actual gradient route; a
DF energy method does not automatically have an analytic gradient.

Harmonic frequencies alone are not intensities. The native
``vibrational_analysis(..., dipole=True)`` path now supplies dipole derivatives
and double-harmonic IR intensities; see :doc:`multipoles`. This does not provide
Raman activities, anharmonic corrections, empirical scaling or a complete
temperature/pressure-dependent thermochemistry calculation. Native harmonic
analysis reports a positive-real-mode ZPE; additional thermochemistry needs
an appropriate treatment.

Run the shipped example
-------------------------

.. code-block:: bash

   python examples/methods_api/geometry_workflows.py --task minimum --method ccsd \
       --basis sto-3g --out-dir water-frequencies

The example optimizes water and includes projected frequencies in
``water-frequencies/summary.json``. Its ``--task ts`` and ``--task irc`` modes
also characterize the ammonia saddle; the IRC task characterizes the relaxed
endpoints as well. The default CCSD water demonstration gave three positive
modes near 2043, 3575 and 3815 cm^-1. These values verify the small-basis workflow,
not agreement with experimental fundamentals.
