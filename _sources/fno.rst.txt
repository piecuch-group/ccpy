Frozen natural orbitals
=======================

FNO preparation is enabled on any object-oriented method through ``MethodBase``.
It runs the low-level calculation in the full virtual space, builds its virtual
occupation density, selects virtual natural orbitals, semicanonicalizes the
retained subspace, transforms the DF factors, and reruns the low-level calculation
in the reduced space. Occupied orbitals and the SCF determinant energy are
unchanged. CC/EOM kernels receive ordinary reduced orbital dimensions.

DF-MP2 usage
------------

.. code-block:: python

   from ccpy import Molecule, RHF, CCSD, EOMCCSD, FNOOptions

   mol = Molecule("O 0 0 0; H 0 0 .96; H .93 0 -.24", "cc-pvdz",
                  auxiliary_basis="def2-universal-jkfit")
   mf = RHF(mol)
   cc = CCSD(mf, nfrozen=1, fno=FNOOptions(occupation_cutoff=1.e-6))
   energy = cc.energy()
   truncated_energy = cc.raw_energy()
   delta_mp2 = cc.fno_result.delta_ground

   eom = EOMCCSD(cc)  # the identical reduced reference, T, and HBar are shared
   eom.initial_guess(nacto=2, nactu=2, roots_per_irrep={"A": 2})
   total_state_energies = eom.energy()
   gaps = eom.completed_omega

The default completion is

.. math::

   E_{\mathrm{FNO}} = E_{\mathrm{trunc}} +
      E_{\mathrm{MP2,full}} - E_{\mathrm{MP2,trunc}}.

``correlation_energy`` and ``omega`` remain raw truncated-space quantities.
``energy()`` and CC ``total_energy`` include completion; ``raw_energy()`` does not.
The MP2 correction shifts every EOM total by the same amount, leaving all
target-minus-parent gaps unchanged. Corrections such as CR-CC(2,3) and
CR-EOMCC(2,3) include the shift exactly once. STEOM completion applies to its
final neutral energies, not its internal auxiliary eigenvalues.

MP2 is computed directly from occupied-pair contractions of three-index DF
factors, without four-index ERIs or a full doubles-amplitude tensor. Pair
workspace is quadratic in virtual count. The standalone ``MP2`` method uses
the same DF implementation. Occupied/virtual Fock blocks are internally
semicanonicalized for MP2; its density is returned in the original basis.
For ROHF this defines the spin-Fock MP partition including singles, which
can differ from other ROHF-MP2 conventions. Occupied orbitals passed to CC
remain unchanged.

Selection and reference handling
--------------------------------

Specify one of ``occupation_cutoff``, ``retained_population`` (fraction in
``(0, 1]``), or ``nvirtuals``. The default absolute cutoff is ``1.e-6``.
``occupation_cutoff=0`` keeps all orbitals. Restricted occupations use the
spatial 0--2 convention; UHF occupations use the spin 0--1 convention.
Occupation-degenerate groups are retained together, so counts can exceed the
requested minimum. A threshold removing all external virtuals raises an error.

Selection and Fock diagonalization preserve irreps. UHF uses separate spin
rotations but equal removals per irrep. Restricted open-shell references keep
every singly occupied spatial orbital, rotating only external spatial virtuals.
This preserves the determinant for SF-EOM. Accuracy for diffuse attachment,
double-attachment, and spin-flip states must be checked by threshold convergence.

To preserve specific input virtual directions, set, for example,
``FNOOptions(occupation_cutoff=1.e-6, protected_virtuals=(0, 2))``. These are
zero-based external spatial virtual indices for restricted references (excluding
all singly occupied ROHF orbitals) and local virtual indices in each spin for
UHF. Protected directions are added to the selected natural-orbital subspace
before semicanonicalization. This can increase the retained count; the resulting
orbitals span the protected directions but need not individually equal them.
Symmetry closure can add directions in multiple irreps. UHF conservatively
retains a complete irrep in both spins if protected-space ranks would otherwise
give unequal removals. Providers can supply general dense subspaces as described
below.

``fno_result`` exposes the full/truncated low-level evaluations, natural
occupations before semicanonicalization, all-MO parent-to-child transforms,
retained populations, and discarded counts. It is shared through the reference.
The original full reference is ``method.reference.fno_parent``. SCF data are not
mutated. A new threshold requires a new chain starting from SCF or the full
reference; conflicting downstream settings and ``ndelete`` with FNO are rejected.

For active or selected spaces, prepare the reference first and build lists in
its working basis:

.. code-block:: python

   from ccpy import Reference, prepare_fno, CCSDT_P, get_active_triples_pspace

   ref = prepare_fno(Reference(mf, nfrozen=1), FNOOptions(nvirtuals=5))
   ref.system.set_active_space(nact_occupied=2, nact_unoccupied=2)
   triples = get_active_triples_pspace(ref.system)
   cc_p = CCSDT_P(ref, triples)

External CI wavefunctions, guesses, selected excitation lists and orbital labels
must refer to this new basis. Dense virtual rotations cannot be handled by
relabeling old canonical orbital indices. For one-body properties,
``reference.one_body_from_ao(operator)`` returns per-spin correlated-space
matrices, including any leading component axes of the AO operator.
``reference.one_body_operator_from_ao(operator)`` wraps a single AO matrix as
a reference-bound ``Integral`` for LR-CC. Multipole generation accepts the
reference directly:

.. code-block:: python

   from ccpy import LRCCSD
   from ccpy.interfaces.scf_tools import get_multipole_integral

   dipoles, reference_dipole = get_multipole_integral(1, cc.reference)
   response = LRCCSD(cc, dipoles[2]).property()

LR-CC rejects operators bound to a different reference and operators whose
dimensions do not match the working space. CC(P) and EOM(P) reject malformed
or out-of-range selected excitation lists before calling compiled kernels.
Unbound operators and selected lists whose indices fit still require the caller
to establish the correct basis: matching dimensions alone do not prove this.

Existing analytic gradients are disabled for FNO references. Finite differences
rebuild FNOs and the completion at every geometry; cutoff crossings or changed
target roots can make derivatives discontinuous. The energy correction does not
restore full-space densities or transition properties. Legacy four-index methods
still materialize their integrals, but only in the reduced space.

ADC(2) provider hook
--------------------

Strict DF EE/IP/EA-ADC(2) and their unrelaxed ISR virtual densities are supplied by
``ADC2``, ``IPADC2`` and ``EAADC2`` (see :doc:`adc`). The native ``adc2`` FNO provider is installed
automatically and follows projected full-space roots in the reduced space.
RHF, ROHF and UHF, including noncanonical references, are supported.
``FNOTarget(..., formulation="folded")`` is the default primary-state solver;
use ``formulation="full"`` for doubles-dominated EE or charged satellite targets.
Other ADC sectors remain explicit unsupported capabilities.
To extend or replace the native provider, register one before construction:

.. code-block:: python

   from ccpy import (FNOEvaluation, FNOOptions, FNOTarget,
                     register_fno_provider)

   def evaluate_adc2(reference, *, targets=(), need_density=True, full_result=None):
       # Your DF-ADC(2) implementation runs on exactly this reference/basis.
       # Match each target to the same physical root in both evaluations.
       # Return absolute ground/target totals; densities are spin-resolved
       # STATE 1-RDM virtual blocks in this reference's working MO basis.
       ground_energy, states, densities = run_your_adc2(
           reference, targets, need_density=need_density)
       return FNOEvaluation(
           ground_energy=ground_energy,
           state_energies=states,          # {FNOTarget: absolute energy}
           virtual_densities=densities,   # {FNOTarget: {"a": Dvv_a, "b": Dvv_b}}
           converged=True,
           metadata={"density_convention": "document your order/relaxation"})

   register_fno_provider("adc2", evaluate_adc2)
   targets = (FNOTarget("EE", root=0, irrep="A"),)
   options = FNOOptions(source="adc2", targets=targets,
                        occupation_cutoff=1.e-6,
                        energy_completion="state_specific")
   eom = EOMCCSD(mf, nfrozen=1, fno=options, fno_targets=targets)

The provider is called first on the full reference with ``need_density=True``,
then on the fixed reduced reference with ``need_density=False``. Do not enable
FNO recursively inside the provider. Use density key ``None`` for a ground-state
selection with no targets. ADC providers must explicitly reject unsupported
sectors or missing state-density capabilities; there is no silent MP2 fallback.
Complex, non-Hermitian, nonfinite, significantly negative, or symmetry-mixing
occupation densities are rejected. Transition and signed difference densities
are not state occupation densities.

During the second call, ``full_result`` is the full-space ``FNOEvaluation``;
its ``metadata`` can carry eigenvectors or other root-tracking data.
``reference.fno_parent`` and ``reference.orbital_transform`` give the full
reference and rectangular all-MO parent-to-child transforms. Use these to
project and match states before returning the reduced-space energies.

The full-space provider may set ``FNOEvaluation.protected_subspaces`` to
``{"a": V_alpha, "b": V_beta}``, where each finite real matrix has one row per
input spin virtual and one column per direction to retain. Columns need not
be normalized or independent. The union of these spaces and the selected FNOs
is orthonormalized and semicanonicalized. For ROHF, include the singly occupied
beta virtual rows; their occupied spatial orbitals remain unchanged and the
external projection of each supplied direction is also retained. Providers must
compute new matrices for each geometry rather than reuse old-basis vectors.

For target ``s``, supply signed absolute energies consistent with
``E_target = E_ground + omega``. IP/DIP targets have N-1/N-2 electrons and EA/DEA
targets N+1/N+2; a positive electron affinity has the opposite sign from the
attachment ``omega``. State completion is

.. math::

   \delta_s &= E^{L}_{s,\mathrm{full}} - E^{L}_{s,\mathrm{trunc}},\\
   E_s^{\mathrm{FNO}} &= E_{s,\mathrm{trunc}}^{\mathrm{CC/EOM}} + \delta_s,\\
   \omega_s^{\mathrm{completed}} &= \omega_{s,\mathrm{trunc}}^{\mathrm{EOM}}
                                      + \delta_s - \delta_0.

``fno_targets`` identifies targets in EOM output order (defaults to the options'
target order). The method layer checks sector, irrep, and available energies;
the provider and caller must establish physical state correspondence using
overlaps/character rather than sorted energies. Providers must raise on lost
roots or failed matching. Declare the chosen density relaxation/order and
convergence diagnostics in ``metadata``.

``space_policy="average"`` diagonalizes the weighted state-average density;
``weights`` defaults to equal weights. ``"union"`` spans independently selected
state spaces before Fock diagonalization. UHF unions conservatively retain full
irreps when independently selected ranks would give unequal spin removals.
``"root_specific"`` requires one target: create a separate method/reference
chain per target. Ground-state corrections are reevaluated in each target's
own reduced space. Cross-space transition properties are not supplied.
``energy_completion="none"`` exposes an uncompleted FNO calculation; ``"ground"``
uses only the ground-state correction even with state-derived selection.
