Multipole moments and harmonic IR intensities
==============================================

The derivative-density machinery now supplies relaxed state multipole moments
and combined nuclear gradients/dipoles. Differences of the dipoles along
nuclear displacements give an atomic polar tensor and double-harmonic IR
intensities. This chapter distinguishes static moments, electronic energy
perturbations and vibrational intensities; none is automatically an electronic
transition strength or a Raman spectrum.

Analytic property derivatives
------------------------------

For an electronic one-body perturbation

.. math::

   h(\boldsymbol\lambda)=h+\sum_k\lambda_k V_k,
   \qquad \frac{\partial E}{\partial\lambda_k}=D_{\mathrm{relaxed}}:V_k,

``V`` is a real AO matrix ``(nao,nao)`` or component stack
``(ncomp,nao,nao)``. The analytic gradient kernels use orbital-relaxed density
and response information for this derivative as well as for nuclear gradients.
The operator holds the nuclei and AO basis positions fixed during the field
variation.

.. code-block:: python

   from ccpy import Molecule, RHF, CCSD
   from ccpy.gradients.derivative_integrals import multipole_integrals

   mol = Molecule("O 0 .05 .1173; H 0 .7572 -.4692; H .1 -.7572 -.4692",
                  "sto-3g", auxiliary_basis="def2-universal-jkfit", symmetry=False)
   cc = CCSD(RHF(mol, verbose=0, e_tol=1.e-12, d_tol=1.e-10), nfrozen=1,
             amp_convergence=1.e-10, energy_convergence=1.e-12)
   labels, V = multipole_integrals(mol, order=2, origin=(0., 0., 0.))
   derivatives = cc.gradient(perturbation=V)  # (ncomp,), electronic dE/dlambda
   nuclear_gradient, derivatives = cc.gradient(properties=V)

The three forms of ``gradient`` have different returns:

* ``gradient()``: nuclear gradient ``(natoms,3)``, Eh/bohr.
* ``gradient(perturbation=V)``: property derivatives ``(ncomp,)``.
* ``gradient(properties=V)``: ``(nuclear_gradient, property_derivatives)``
  from one analytic response calculation.

Use ``properties=`` on ``gradient``, not ``energy_and_gradient``: the latter
passes kwargs to ``energy`` as well, and property-integral kwargs are not energy
arguments. Use either perturbation or properties according to the desired
return. For EOM or an excited correction, pass ``root=...`` to select one
returned state. Matching the energy variant matters, e.g. CRCC23/CREOMCC23
analytic property paths are available for variant A under their normal
reference restrictions. See :doc:`gradients` for eligibility and convergence.

State moments, units and origin
--------------------------------

.. code-block:: python

   moments = cc.multipole_moments(order=3, origin=(0.1, -0.2, 0.3))
   print(moments["x"], moments["xy"], moments["xyz"])
   gradient, dipole = cc.gradient_and_dipole()  # dipole (3,), ordered x,y,z

``multipole_moments`` returns a dictionary of all Cartesian moments through
``order``: 3 dipole components for order 1; those plus 6 quadrupole components
for order 2; those plus 10 octupole components for order 3. ``origin`` is in
bohr and defaults to the coordinate origin. Keys follow
``x,y,z,xx,xy,xz,yy,yz,zz,xxx,xxy,xxz,xyy,xyz,xzz,yyy,yyz,yzz,zzz``.
These are primitive Cartesian charge moments, not traceless tensors or
spherical multipoles, with units e bohr, e bohr squared and e bohr cubed.

The helper returns the **total physical charge moment**, nuclear minus the
electronic position-moment derivative:

.. math::

   M_{x^a y^b z^c}=\sum_A Z_A(X_A-O_x)^a(Y_A-O_y)^b(Z_A-O_z)^c
       -\frac{\partial E}{\partial\lambda_{x^a y^b z^c}}.

Do not negate or add the nuclear part again. By comparison,
``get_multipole_integral`` used by LRCC returns positive position operators and
an electronic reference expectation; that is a different return convention.
Charged dipoles depend on the origin, and higher moments generally do too.
Neutral dipoles are origin invariant. Keep a fixed origin when differentiating
moments across geometries.

``gradient_and_dipole`` uses the coordinate origin and returns no energy.
Its kwargs select gradient options such as an electronic ``root``; it does not
expose an origin argument. For a custom origin build ``V`` explicitly and use
``gradient(properties=V)`` plus the charge-moment conversion helper.

Analytic convenience versus a generic finite-field route
---------------------------------------------------------

The convenience methods above currently call ``gradient`` with a property
request. When no analytic property gradient is available,
``gradient(perturbation=...)`` and ``gradient(properties=...)`` raise
``NotImplementedError``; they do **not** automatically differentiate energies
with respect to fields. This includes methods with a numerical nuclear-gradient
fallback. The existence of a nuclear finite-difference gradient does not itself
provide a finite-field moment.

For any method for which a consistent scalar energy can be rebuilt in a field,
the finite-field route is available: recompute SCF and the complete correlated
calculation for each strength, differentiate the electronic total energy and
add the nuclear moment with the helper above. A runnable example callback,
``finite_field_moments``, is provided in
``examples/methods_api/multipole_ir.py``. It is an example function rather than
an additional exported MethodBase method:

.. code-block:: python

   import numpy as np
   from ccpy.gradients.derivative_integrals import multipole_integrals, multipole_moments
   from ccpy.gradients.finite_difference import finite_difference

   def finite_field_moments(molecule, scalar_energy, order=1, origin=None, field_step=1.e-3):
       labels, operators = multipole_integrals(molecule, order=order, origin=origin)
       xyz = np.array([xyz for _, xyz in molecule.atoms])

       def energy_in_field(strengths):
           field_molecule = molecule.with_geometry(xyz, unit="bohr")
           h = field_molecule.ints_hcore() + np.einsum("k,kmn->mn", strengths, operators)
           field_molecule.ints_hcore = lambda: h
           return scalar_energy(field_molecule)  # create fresh SCF and correlation here

       dE = finite_difference(energy_in_field, np.zeros(len(labels)),
                              step=field_step, npoints=4)
       return multipole_moments(dE, molecule, order=order, origin=origin)

For instance ``scalar_energy`` can instantiate MP2 or another energy-only
method on ``RHF(field_molecule)`` and return its ``energy()``. An excited
callback must configure guesses and return ``energy(root=...)`` for the chosen
physical state. Do not reuse a zero-field solved reference or merely perturb
integrals after SCF: orbital relaxation requires re-solving SCF in every field.
Construct the field on each fresh molecule; an arbitrary monkey-patched
``ints_hcore`` is not automatically preserved by ``MethodBase.at_geometry``.

Field strengths are in atomic units conjugate to the selected operator. Check
field-step and electronic-tolerance convergence; cancellation amplifies energy
noise. If FNOs, CIM or selected spaces are used, ensure consistent energy
completion, fixed domains/selections as appropriate and smooth state tracking.
The analytic helpers retain their existing reference/variant limitations; this
finite-field recipe is the general scalar-energy path, not a claim that every
method/reference combination is already validated.

Compute frequencies and IR intensities together
------------------------------------------------

Use the native Hessian/vibrational helpers with an optimized molecule:

.. code-block:: python

   from ccpy.gradients.hessian import vibrational_analysis

   optimized = minimum.molecule  # from a converged, consistently defined optimization

   def gradient_and_dipole_at(molecule):
       method = cc.at_geometry(molecule)
       g, mu = method.gradient_and_dipole()
       if not (method.converged and method.reference.scf.converged):
           raise RuntimeError("Unconverged displaced calculation")
       return g, mu

   result = vibrational_analysis(optimized, gradient=gradient_and_dipole_at,
                                 dipole=True, step=2.5e-3)
   print(result["frequencies"])       # cm^-1
   print(result["ir_intensities"])    # km/mol, same mode ordering
   print(result["atomic_polar_tensor"].shape)  # (3,3*natoms)

With ``dipole=True`` the callback must return **(gradient,dipole)** or
**(energy,gradient,dipole)**. Supplying the ordinary ``(energy,gradient)`` pair
has the wrong meaning. The dipole must be the total physical dipole including
the nuclear contribution, in atomic units, not an electronic position vector.
With ``dipole=False``, the ordinary gradient or ``(energy,gradient)`` callback
remains valid. ``dipole=True`` requires a callback providing dipoles;
``energy=...`` alone cannot supply intensities automatically.

The same +/- nuclear displacements give the Hessian and atomic polar tensor

.. math::

   P_{a,Ab}=\frac{\partial\mu_a}{\partial R_{Ab}},\qquad
   \frac{\partial\mu}{\partial Q_k}=P M^{-1/2}L_k.

IR intensity is proportional to the squared norm of this mode dipole derivative
and is converted to km/mol. The code uses mass-normalized vibrational
coordinates for this contraction; do not contract ``P`` (units e) with the separately
Euclidean-normalized display modes without restoring their normalization.
For a charge q, the translational sum rule is
``sum_A P[a,A,b] = q*delta[a,b]``. This is a useful sign/nuclear-term diagnostic,
subject to finite-difference error.

All Hessians here are numerical derivatives, even when the gradient and dipole
at each displacement are analytic. A generic numerical callback can instead
return a finite-difference nuclear gradient and a finite-field total dipole.
The shipped ``--mode finite-field`` example implements that construction for
MP2 as well as CCSD/CRCC23_A. It costs many more energy evaluations; there are
separate nuclear-gradient, field and Hessian step sizes to converge. Finite
fields on the existing analytic nuclear gradient are not a substitute: moving
AO multipole operators require additional nuclear derivative terms that the
ordinary field-free gradient contraction does not add automatically.

Saved tensors, isotope reanalysis and intensity scope
-----------------------------------------------------

To separate derivative generation from harmonic analysis:

.. code-block:: python

   from ccpy.gradients.hessian import fd_hessian, harmonic_analysis

   H, P = fd_hessian(optimized, gradient=gradient_and_dipole_at,
                      dipole=True, step=2.5e-3)
   result = harmonic_analysis(H, optimized, dipole_derivatives=P)
   np.savez("vibrations_ir.npz", **result, hessian=H, atomic_polar_tensor=P)

``harmonic_analysis`` accepts an optional ``masses`` array in amu for isotope
substitution. Recompute both frequencies and intensities from the same H/P
with the new masses; no new electronic derivatives are needed within this
harmonic Born-Oppenheimer treatment. Native default masses are the most
abundant isotopes through Kr, with a fallback atomic mass beyond that table;
this can differ slightly from external pysisyphus mass-table defaults.

Native results contain ``normal_modes`` with shape ``(nvib,natoms,3)``,
``reduced_masses`` (amu), ``force_constants`` (mdyn/angstrom), ``eigenvalues``
(Eh/(bohr squared amu)), and ``zpe`` (Eh, positive real modes only), as well
as frequencies. With dipoles they also contain mode ``dipole_derivatives``
``(nvib,3)`` in e/sqrt(amu) and ``ir_intensities``. The convenience
``vibrational_analysis`` additionally stores ``hessian`` and
``atomic_polar_tensor``. Native mode arrays differ in layout from pysisyphus'
column-vector return, documented in :doc:`frequencies`.

The implemented intensities are electric-dipole, double-harmonic vibrational
IR intensities. Higher Cartesian moments can be computed but do not by
themselves provide quadrupole/octupole absorption intensities, Raman
activities or electronic transition oscillator strengths. No anharmonic
corrections or spectral line broadening are implied. Do not interpret a TS
imaginary mode as an ordinary absorption line.

Run the examples and inspect validation
----------------------------------------

.. code-block:: bash

   python examples/methods_api/multipole_ir.py --task moments --method ccsd --order 3 \
       --mode analytic --out-dir analytic-moments
   python examples/methods_api/multipole_ir.py --task ir --method ccsd \
       --mode analytic --out-dir ccsd-ir
   python examples/methods_api/multipole_ir.py --task ir --method mp2 \
       --mode finite-field --out-dir mp2-ir

The IR tasks first optimize the demonstration water molecule and save
``summary.json`` and ``harmonic_ir.npz``. These use a small default basis to
illustrate the interface; basis, step-size and electronic-model convergence
remain necessary for quantitative spectra. Existing checks in
``test_property_gradients.py`` compare relaxed derivatives against finite-field
energies, and ``test_ir_intensities.py`` checks the atomic polar tensor against
energy derivatives, charge sum rules and CO2 IR selection rules. These checks
cover particular references and methods, not all possible method combinations.
