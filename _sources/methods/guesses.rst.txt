Initial guesses, symmetry and choosing roots
============================================

Guess configuration is separate from the final equation model
-------------------------------------------------------------

``initial_guess`` is shared by EOMCC, ADC and STEOM. It configures a CI-like
problem used to initialize and identify targets. It does not reduce the
subsequent full EOM excitation space. In particular, guess ``nacto/nactu``
restrict guess doubles, not all EOM doubles/triples.

.. code-block:: python

   eom.initial_guess(
       method="cisd", multiplicity=1,
       nacto=3, nactu=4,
       roots_per_irrep={"A1": 3, "B2": 2},
       roots=[0, 2, 4], use_symmetry=True,
   )

This example requires a reference with matching point-group labels, such as
C2v; it cannot be copied onto a C1 reference without changing the labels.
Use ``debug=True`` for supported guess diagnostics.

.. list-table:: Guess choices
   :header-rows: 1
   :widths: 22 38 40

   * - Guess
     - Space
     - Typical use
   * - ``cis``
     - All neutral singles.
     - Primary-dominated EE; default folded EE-ADC(2).
   * - ``cisd`` / ``cisd_chol``
     - All neutral singles plus active doubles.
     - Ordinary EE-EOM default; full EE-ADC default; doubles-character targeting.
   * - ``ipcis`` / ``ipcisd``
     - 1h / 1h plus active 2h1p.
     - Folded IP-ADC / IP-EOM or full IP-ADC.
   * - ``eacis`` / ``eacisd``
     - 1p / 1p plus active 2p1h.
     - Folded EA-ADC / EA-EOM or full EA-ADC.
   * - ``dipcis``
     - Leading 2h configurations.
     - DIP-EOM families.
   * - ``deacis``
     - Leading 2p configurations.
     - DEA-EOM families.
   * - ``sfcis``
     - Leading single spin flips.
     - SF-EOM families.

For native DF methods ``method="cisd"`` is mapped to ``"cisd_chol"``.
It is an implementation selection, not a different user-facing root model.
``nacto=None`` and ``nactu=None`` become zero in the generic mixin, so explicitly
supply positive counts when seeking active-double character. Open-shell guess
builders derive spin-specific active counts from the reference multiplicity,
subject to available dimensions. Consult printed active counts.

EOMCC guesses normally use the selected **HBar** (and T for DF CISd), not the
bare SCF Hamiltonian. ADC and STEOM override this behavior and use the **bare**
Hamiltonian. A bare CISd guess therefore need not have the same energies or
ordering as an EOMCC CISd guess, even on the same reference.

Request a guess pool, then choose entries from it
-------------------------------------------------

``roots_per_irrep={"A1": 3, "B2": 2}`` requests up to three A1 and two B2 guess
states. For symmetry-blocked guesses the concatenation follows dictionary
insertion order. Within each block the guess states are energy ordered.
``roots=[0, 2, 4]`` selects entries of that **concatenated guess pool**, not
root 0/2/4 within every irrep, and not absolute final eigenvalue indices.

.. code-block:: text

   guess index        0    1    2    3    4
   guess irrep       A1   A1   A1   B2   B2
   selected          yes       yes       yes
   returned index     0         1         2

After solving, ``energy(root=1)`` refers to the second returned state, which
was seeded by guess index 2 in this example. Likewise ``R[1]`` and
``omega[1]`` are output indices. All these indices are zero based.
``FNOTarget.root`` is different again: it labels the low-level provider's root
within an irrep; see :doc:`spaces`.

A constructor ``roots=[...]`` provides the same guess-index selection for EOM
and ADC, but ``initial_guess(roots=...)`` takes precedence. Prefer configuring
selection in one place. If the guess pool provides fewer states than requested
because of spin, symmetry or active-space dimension, requesting a missing index
raises an error. Asking for a larger pool does not guarantee those states exist.

.. code-block:: python

   eom.initial_guess(method="cisd", multiplicity=1, nacto=2, nactu=3,
                     roots_per_irrep={"A": 6}, roots=[1, 4])
   eom.run_guess()  # optional inspection before the expensive EOM solve
   print(eom.guess_energy, eom.guess_irreps)
   totals = eom.energy()
   print(eom.root_irreps)

For EOM, ``run_guess()`` can build the ground CC and HBar on demand. The
subsequent solve may regenerate guesses when no right vectors have yet been
initialized; it is an inspection step, not a persistent checkpoint.

Irrep dictionaries, integer counts and disabled symmetry
--------------------------------------------------------

An integer ``roots_per_irrep=4`` asks for the lowest four guess states without
irrep quotas. With symmetry disabled, a dictionary is treated as a total root
count rather than as enforced per-irrep quotas. Use a dictionary with
``use_symmetry=True`` when distinct irrep counts matter.

CIS/CISd, IP-/EA-CISd and DIP-/DEA-CIS are symmetry blocked. IP-CIS, EA-CIS and
SF-CIS are diagonalized in their full leading space and classified afterward
by a leading amplitude. Printed ``guess_irreps`` records the resulting labels.
Final EOM root symmetry is checked against the guess and mismatches produce
warnings; this general check is not itself a projector. ADC additionally
restricts its Davidson action to the requested spatial irrep.

Spin multiplicity
-----------------

For neutral EOM, the default guess multiplicity is the reference multiplicity.
Set 1 for singlets or 3 for triplets on a closed-shell parent. IP, EA, DIP, DEA
and SF default to ``-1``, meaning no guess spin adaptation. ``multiplicity``
selects the CI guess spin subspace when supported; it does not change the
molecule's electron number or reference SCF spin.

EOM disables the closed-shell amplitude shortcut when the target multiplicity
is not 1. ``RHF_symmetry`` is an equation-level spin-block shortcut, not the
point-group symmetry switch and not an instruction to run RHF SCF.
A spin-adapted guess does not make a spin-contaminated UHF Hamiltonian spin pure.
STEOM separately restricts its reference to closed-shell restricted CCSD and its
target multiplicity to 1 or 3.

Following state character
--------------------------

``davidson_selection_method="overlap"`` (default) follows the trial-state
character in the projected problem; ``"energy"`` selects the Ritz eigenvalue
nearest the current energy estimate. Neither guarantees an ordinal final root
of the exact spectrum. Inspect amplitudes, irreps, ``rel`` and overlaps when
states are close, reorder, or have strong double-excitation character.

For ordinary EOM, ``davidson_solver="standard"`` solves roots sequentially;
``"multiroot"`` selects the existing block-Davidson implementation. Sequential
EOM passes previous vectors to the solver, but that is not the same as a
universal strict root-deflation projector. Confirm distinct roots rather than
assuming that an arbitrary set of guesses must converge to distinct states.
EOM-CC3 uses its omega-dependent nonlinear DIIS solver for either setting.
ADC exposes sequential Davidson only, with explicit full-space locking in
full mode and duplicate detection in folded mode. STEOM has its own auxiliary
and final diagonalization procedures.

For satellite or doubles-dominated targets:

1. Choose a model that includes the relevant R sectors.
2. Enlarge the CISd active space and request enough guess states.
3. Select the desired guess indices explicitly.
4. For ADC use ``formulation="full"``; for a neutral SD STEOM projection use
   ``STEOMCCSDT`` rather than ordinary singles-only STEOM.
5. Verify final character and convergence. FNO density selection must follow
   the same physical target in full and reduced spaces.

Changing ``initial_guess`` clears the existing right-state setup. Geometry
reconstruction copies guess settings, but generic EOM finite differences do
not implement a universal cross-geometry overlap tracker. Near crossings,
check the state identity explicitly; a fixed guess index is not proof that
the same physical state was followed.
