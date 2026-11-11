Left states, densities, response and gradients
==============================================

Left ground and excited states
-------------------------------

Ground-state CC and EOM similarity-transformed Hamiltonians are non-Hermitian.
Left states are needed for many corrections and response quantities, even
though a right-state energy can be obtained without them.

.. code-block:: python

   from ccpy import LeftCCSD, LeftEOMCCSD

   left_ground = cc.left.ensure_solved()    # or LeftCCSD(cc).solve()
   Lambda = left_ground.L
   left_excited = eom.left.ensure_solved()  # or LeftEOMCCSD(eom).solve()
   left_vectors = left_excited.L
   print(left_excited.omega, left_excited.LR)

``LeftCCSD`` can start directly from SCF to create CCSD automatically.
``LeftCCSDT`` and ``LeftCC3`` use their corresponding ground-state defaults.
``LeftCCSDT_P`` requires an upstream selected-triples CC method with its lists.
``left.energy()`` returns the associated right-state energy; it is not a new
independent variational energy.

Exported excited left classes are ``LeftEOMCCSD``, ``LeftEOMCCSDT``,
``LeftEOMCCSDT_P``, ``LeftEOMCCSDTA``, ``LeftIPEOM2``, ``LeftIPEOM3``,
``LeftIPEOM3_P``, ``LeftEAEOM2``, ``LeftEAEOM3`` and ``LeftEAEOM3_P``.
They require an upstream EOM object. ``solver="davidson"`` is the default;
``solver="jacobi"`` selects the left Jacobi route when the kernel supports it.
The wrapper normalizes each left vector to its paired right vector using the
sector's overlap function. ``LR`` records the overlap before this rescaling;
inspect the printed right/left eigenvalue difference and ``converged`` as well.

Not every right method has a corresponding left implementation.
``cc.left`` or ``eom.left`` raises ``NotImplementedError`` where the class does
not name a left method. Availability is listed in :doc:`catalog`; an inherited
base method such as ``rdm1`` does not guarantee a density implementation at every
level.

CCSD densities and their frame
-------------------------------

.. code-block:: python

   rdm1 = cc.rdm1()  # solves CCSD and its Lambda state if necessary
   rdm2 = cc.rdm2()
   rdm2_slice = cc.left.rdm2_vvvv_slice(a=0, spin="a")

These CCSD densities are **unrelaxed derivative densities in the T1-transformed
frame**, conjugate to the stored Fock/two-body blocks of H(T1). They are not a
ready-made AO total-state RDM obtained by simply sandwiching a conventional MO
density with the SCF coefficients. The determinant contribution is separate,
and the full ``vvvv`` block is supplied by slices rather than stored densely.
Consult ``ccpy.density.ccsd_rdm`` before contracting these with operators or
using them in orbital optimizations.

ADC ``virtual_density(root=...)`` instead returns spin-resolved virtual blocks
of its unrelaxed second-order ISR state density, normalized from the full
reconstructed eigenvector. Those blocks are appropriate for ADC-FNO selection.
They are not full state RDMs, transition moments or oscillator strengths.
Ordinary STEOM returned singles likewise are not full property wavefunctions.

Relaxed moments and vibrational intensities
---------------------------------------------

``multipole_moments`` and ``gradient_and_dipole`` now use the relaxed analytic
gradient densities. The native Hessian helpers can differentiate dipoles and
compute harmonic IR intensities. See :doc:`multipoles` for units, nuclear signs,
combined gradient/property returns and a finite-field energy recipe for methods
without an analytic property path. These relaxed property helpers are distinct
from the unrelaxed T1-frame density getters above.

One-body operators and LRCCSD
-----------------------------

``LRCCSD`` accepts a first-order ``Integral`` operator in the **same working
basis** as its reference. ``property()`` returns the correlation contribution;
``energy()`` returns the unperturbed ground CC energy.

.. code-block:: python

   from ccpy import LRCCSD
   from ccpy.interfaces.scf_tools import get_multipole_integral

   dipole_operators, reference_dipole = get_multipole_integral(1, cc.reference)
   response = LRCCSD(cc, dipole_operators[2])  # z component
   correlation_contribution = response.property()
   total_component = reference_dipole[2] + correlation_contribution
   print(response.T1)  # first-order perturbed amplitudes, not the ground T1

``get_multipole_integral`` returns positive position operators and their
electronic reference expectations, without nuclear moments or the electronic
charge sign. The example total is therefore an electronic position expectation;
for a physical dipole in atomic units use the nuclear charge-weighted positions
minus that electronic expectation. Higher multipoles need their corresponding
charge/sign and tensor conventions. For a custom
scalar AO operator:

.. code-block:: python

   operator = cc.reference.one_body_operator_from_ao(ao_matrix)
   correlation_contribution = LRCCSD(cc, operator).property()

``one_body_from_ao`` returns spin-resolved working matrices and can handle
leading component axes. ``one_body_operator_from_ao`` wraps a single scalar
matrix as a reference-bound ``Integral`` suitable for LRCCSD. The LR wrapper
checks spin block shapes, finiteness, reality and an operator's reference
binding. Full-space operators must be rebuilt after FNO preparation; identical
dimensions alone do not prove that two MO bases are the same.

LRCCSD currently uses materialized equations and a legacy CCSD HBar. It is a
first-order one-body property interface, not a general frequency-dependent
response/absorption API. FNO energy completion does not add the missing-space
contribution to densities or properties.

Gradients and finite differences
--------------------------------

For the complete derivative guide and availability table, see :doc:`gradients`.
For pysisyphus minima, transition-state searches, frequencies and two-sided
IRC validation, see :doc:`geometry`. Harmonic frequencies and normal modes
are covered in :doc:`frequencies`.

.. code-block:: python

   energy, gradient = cc.energy_and_gradient()
   available = cc.has_analytic_gradient()
   numerical = cc.fd_gradient()

``gradient`` has shape ``(natoms, 3)`` and units Eh/bohr. It selects an analytic
kernel when the method and reference pass ``has_analytic_gradient``; otherwise
it differentiates ``energy()`` with central finite differences. ``fd_step``
(default 1e-3 bohr) and ``fd_npoints`` (default 4) control the stencil. To choose
finite differences explicitly, call ``fd_gradient``; there is no general
``gradient(analytic=False)`` switch.

The wrapper names analytic kernels for CCSD, EOMCCSD, CRCC23 variant A and
CREOMCC23 variant A. CCSDT(P) additionally checks availability of its selected
CC density module. Correction variants other than those explicitly supported,
stochastic correction instances, ROHF references, deleted virtuals and FNO
references use finite differences. These are dispatch conditions, not a claim
that every installed analytic kernel works for every physical case. Inspect
``has_analytic_gradient`` on the actual object; a module import/runtime failure
inside an advertised kernel is not automatically converted to finite differences.

.. code-block:: python

   eom_total, excited_gradient = eom.energy_and_gradient(root=0)
   fd_excited = eom.fd_gradient(root=0)

``root`` is the returned-state index. The left and response quantities required
by analytic gradients are created on demand. ``zvector_tol`` is accepted by
analytic routes where applicable; numerical gradients instead use the method's
finite-difference controls.

Geometry reconstruction
------------------------

.. code-block:: python

   import numpy as np

   xyz_bohr = np.array([xyz for _, xyz in cc.molecule.atoms])
   xyz_bohr[1, 2] += 0.01
   moved_molecule = cc.molecule.with_geometry(xyz_bohr, unit="bohr")
   moved = cc.at_geometry(moved_molecule)
   moved_total = moved.energy()

``at_geometry`` reconstructs the recorded constructor chain, SCF type, reference
partition and guess setup. It is a new calculation, not a transfer of converged
orbitals/amplitudes or a persistent restart. FNO bases and low-level completion
are rebuilt at each geometry. AdaptiveCCPQ reuses recorded selected P spaces
for geometry copies. In general, discrete determinant spaces, symmetry changes,
occupation cutoffs and root crossings can make the numerical energy surface
nonsmooth; check state identity and threshold stability for derivatives.

Use ``Molecule.with_geometry``'s explicit ``unit`` when supplying arrays from
external geometry packages. The native molecule stores Cartesian coordinates
in bohr, even if the original input was angstrom.
