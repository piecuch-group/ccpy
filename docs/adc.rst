Density-fitted ADC(2)
====================

``ADC2``, ``IPADC2`` and ``EAADC2`` implement strict second-order ADC for
neutral excitation, ionization and electron attachment from real RHF, ROHF
and UHF references, including noncanonical occupied and virtual orbitals. It uses the same ``MethodBase`` and
``InitialGuessMixin`` contracts as EOMCC. Its CIS/CISd initial guesses operate on
the bare DF Hamiltonian; no coupled-cluster amplitudes or HBar are computed.

.. code-block:: python

   from ccpy import Molecule, RHF, ADC2

   mol = Molecule("O 0 0 0; H 0 0 .96; H .93 0 -.24", "6-31g",
                  auxiliary_basis="def2-universal-jkfit")
   adc = ADC2(RHF(mol), nfrozen=1)
   adc.initial_guess(method="cisd", multiplicity=1, nacto=2, nactu=2,
                     roots_per_irrep={"A": 3}, roots=[0, 2])
   totals = adc.energy()         # SCF + MP2 correlation + ADC excitation energy
   excitation_energies = adc.omega
   virtual_density = adc.virtual_density(root=0)  # {"a": Dvv_alpha, "b": Dvv_beta}

``roots_per_irrep``, ``roots``, ``use_symmetry``, ``nacto`` and ``nactu`` have
the same meaning as in EOMCC. ``method="cis"`` is also supported. Closed-shell restricted
references support singlet/triplet guesses; open-shell references can use ``multiplicity=-1``
for unadapted guesses. Spin-adapted UHF guesses do not make a spin-contaminated
UHF ADC Hamiltonian spin-pure.

Charged states use the existing bare-Hamiltonian guess API:

.. code-block:: python

   from ccpy import IPADC2, EAADC2

   ip = IPADC2(mf, nfrozen=1, formulation="full")
   ip.initial_guess(method="ipcisd", nacto=2, nactu=2,
                    roots_per_irrep={"A": 2})
   ionized_totals = ip.energy()
   ea = EAADC2(mf, nfrozen=1, formulation="full")
   ea.initial_guess(method="eacisd", nacto=2, nactu=2,
                    roots_per_irrep={"A": 2})
   attached_totals = ea.energy()

``ipcis`` and ``eacis`` are also supported. Charged operators use the alpha
removal/attachment branch of CCpy's existing IP/EA API. Their default guess
multiplicity is -1. The signed energy is ``omega = E(target)-E(parent)``:
IP roots are ionization potentials, while positive electron affinities equal
``-ea.omega``. Root indices and irrep selection follow the EOMCC interface.

Formulations and root targeting
------------------------------

All three methods accept ``formulation="folded"`` (default) or
``formulation="full"``. Folded mode targets states dominated by singles (EE),
one hole (IP), or one particle (EA). Its default guesses are respectively
``cis``, ``ipcis`` and ``eacis``. Full mode defaults to ``cisd``, ``ipcisd`` and
``eacisd`` and retains explicit doubles/satellite vectors for targeting states
with strong higher-excitation character. Explicit guess selection remains
available in either mode; a folded guess with no primary component is rejected.

.. code-block:: python

   adc = ADC2(mf)                          # folded, CIS guesses
   satellites = IPADC2(mf, formulation="full")
   satellites.initial_guess(method="ipcisd", nacto=3, nactu=3,
                            roots_per_irrep={"A": 5}, roots=[4])

Writing the full matrix as ``[[A,B],[B.T,D]]``, folded mode evaluates

.. math::

   R_2=(\omega I-D)^{-1}B^\dagger R_1,\qquad
   [A+B(\omega I-D)^{-1}B^\dagger]R_1=\omega R_1.

``D`` is diagonal in the internal semicanonical basis. Each outer energy
iteration solves the primary-space Hermitian problem with CCpy's
``eomcc_davidson`` and uses a Newton energy update including the resolvent
derivative. Only primary vectors enter the Davidson subspace; satellite work
arrays are reconstructed during matrix actions. Final normalized full vectors
are retained for densities and FNO root following. This is not an
amplitude-direct ground-state implementation: MP2 doubles remain stored.
Satellite poles raise an explicit error recommending full mode.

Full mode uses the expanded linear problem with sequential orthogonal root
locking. Folded roots use overlap following without Euclidean primary-space
locking, because different roots are orthogonal in the reconstructed full-space
metric. Duplicate converged roots raise an error. Every final eigenpair is
checked against the original expanded matrix action; unconverged states cannot
supply FNO densities or corrections. ``davidson_solver="standard"`` is the
supported mode.

Equations and storage
---------------------

The implementation uses antisymmetrized spin-orbital integrals
``g_ijab = <ij||ab>`` and first-order MP2 amplitudes

.. math::

   t_{ij}^{ab} = -g_{ijab}/\Delta_{ij}^{ab}, \qquad
   \Delta_{ij}^{ab} = \epsilon_a+\epsilon_b-\epsilon_i-\epsilon_j.

The ADC secular matrix has the strict ADC(2) orders

.. math::

   M = \begin{pmatrix}
       M_{11}^{(0)}+M_{11}^{(1)}+M_{11}^{(2)} & M_{12}^{(1)}\\
       M_{21}^{(1)} & M_{22}^{(0)}
       \end{pmatrix}.

The singles action is evaluated through the CIS action, occupied/virtual
self-energy intermediates, and symmetrized products of the MP2-amplitude and
``g_imae`` singles-space actions. The satellite block is diagonal in the internal semicanonical
orbitals. Singles--doubles couplings are evaluated as staged contractions with
the DF ``oo``, ``ov`` and ``vv`` factors. The native implementation builds
neither the secular matrix nor ``ovvv``/``vvvv`` integral tensors.

The general-reference spin-Fock partition diagonalizes occupied and virtual
spin blocks internally, within irreps, and transforms guesses and state densities
between that basis and the supplied working orbitals. The supplied occupied
orbitals are unchanged. The occupied--virtual Fock block belongs to the
perturbation, with ``t1 = -Fov/(epsilon_v-epsilon_o)``. Both secular matrices
and virtual densities retain its first-order singles contributions. Thus ROHF
support includes nonzero Fov; it uses this specified spin-Fock perturbation
convention rather than assuming Brillouin's theorem. Ground correlation includes
the corresponding ``Fov*t1`` contribution.

Ground-state MP2 doubles and spin-orbital doubles work arrays require
``O(o^2 v^2)`` storage; this is not an amplitude-direct ADC implementation.
Packing retains only independent same-spin occupied/virtual pairs and all
opposite-spin doubles. ``R[k].flatten()`` has a Euclidean independent-determinant
metric. ``R[k].as_cluster()`` converts to CCpy's redundant spin-block amplitudes;
do not compare their unweighted norms.

``correlation_energy`` is the raw DF-MP2 ground-state correlation energy.
``omega`` remains the raw ADC excitation energy, and ``raw_energy()`` returns
uncompleted truncated-space totals. FNO ``energy()`` and ``completed_omega``
apply the configured MethodBase completion policy.

State densities and FNO integration
-----------------------------------

The virtual state-density blocks use the second-order, unrelaxed intermediate
state representation (ISR) formulas, including the quadratic MP2-amplitude
corrections to the singles contribution. They are not transition densities,
MP2 ground densities, or simply CIS particle densities. Full state RDMs,
transition moments, oscillator strengths and analytic ADC gradients are not
implemented by this interface.

The built-in ``adc2`` FNO provider is installed automatically unless a custom
provider has already been registered:

.. code-block:: python

   from ccpy import CCSD, EOMCCSD, FNOOptions, FNOTarget

   targets = (FNOTarget("EE", root=0, irrep="A", multiplicity=1),)
   fno = FNOOptions(source="adc2", targets=targets, occupation_cutoff=1.e-6,
                    energy_completion="state_specific")
   eom = EOMCCSD(mf, nfrozen=1, fno=fno, fno_targets=targets)
   eom.initial_guess(multiplicity=1, nacto=2, nactu=2,
                     roots_per_irrep={"A": 1})
   total = eom.energy()

The full-space provider uses the selected formulation and its corresponding
bare-Hamiltonian guesses, with up to two active occupied and virtual orbitals
for CISd. Set ``FNOTarget(..., formulation="full")`` for doubles/satellite
state selection; its default is ``"folded"``. Targets with different formulations
can be combined in one FNO policy. The reduced-space calculation projects each full-space ISR vector
through the actual orbital transformation and uses it as its Davidson seed.
A normalized projected overlap below 0.5 is treated as a lost root and raises
an error. Degenerate-state identification may require a larger retained space
or a custom provider with subspace tracking. Full/reduced calculations use the
same frozen core, DF factors and perturbation convention. Multiple spin groups
reuse one set of ground-state ADC intermediates.

The native ADC-FNO provider supports EE, IP and EA for RHF/ROHF/UHF and
noncanonical references. DEA, DIP and SF ADC models remain unsupported and
raise explicit capability errors. Ordinary MP2-FNO preparation remains available
for all existing CC/EOM families.

Validation and references
-------------------------

``tests/methods/test_adc2.py`` compares random full matrix actions and virtual
state densities with PySCF 2.14.0's independent UADC spin-block implementation
on RHF and UHF references, using exactly the same DF integrals. EE also
checks the ROHF branch. The charged PySCF ROHF path omits Fov contributions,
so it is not used as an oracle for those terms.
``tests/methods/test_adc2_general_reference.py`` independently constructs the
second-order BCH effective Hamiltonian and virtual density in small full Fock
spaces, checking the charged non-Brillouin terms. It also checks matrix-action,
density and Davidson-energy invariance under occupied/virtual rotations on
RHF, ROHF and UHF references. The tests also check
matrix symmetry, diagonal, determinant metric, Davidson residuals, a dense
small-molecule spectrum, spin/irrep/root targeting, frozen core and full-retention
ADC-FNO completion. PySCF is optional and used only for numerical tests.

The equations are based on J. Schirmer's ADC construction and the transparent
spin-block reference implementations by the PySCF ADC authors:

* J. Schirmer, Phys. Rev. A **26**, 2395 (1982),
  https://doi.org/10.1103/PhysRevA.26.2395
* T. Stahl, S. Banerjee and A. Sokolov, J. Chem. Phys. **157**, 044106 (2022),
  https://doi.org/10.1063/5.0097333 and https://arxiv.org/abs/2204.13164
* PySCF ADC source and documentation:
  https://pyscf.org/_modules/pyscf/adc/uadc_ee.html
  https://pyscf.org/_modules/pyscf/adc/uadc_ip.html,
  https://pyscf.org/_modules/pyscf/adc/uadc_ea.html
  and https://pyscf.org/user/adc.html

Attribution and the Apache-2.0 license are retained in
``THIRD_PARTY_NOTICES.md`` and ``ccpy/adc/LICENSE.apache2``.

``tests/methods/test_adc2_formulations.py`` checks folded/full energies, vectors
and densities, explicit satellite roots, pole/guess guards and FNO selection
with each formulation.
