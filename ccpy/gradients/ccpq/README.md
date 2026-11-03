# Determinant CC(P;Q) and EOMCC(P;Q) gradient checks

This package integrates the ground/excited extension from `ccpq_audit` into
CCpy's molecular integral and nuclear derivative machinery. It is a dense
correctness oracle for small systems, with fixed determinant selections and
separate stationary derivatives of the base state energy and its correction.

```python
from ccpy import Molecule
from ccpy.gradients.ccpq import CCPQGradient, spaces

mol = Molecule("H 0 0 0; H 0.1 0 1.5; H 0 0.2 3.1; H 0.2 0.1 4.8",
               "sto-3g", symmetry=False, verbose=False)

def selected(S):
    return spaces.union(spaces.ranks(S, (1, 2)), spaces.ranks(S, (3,))[::3])

calc = CCPQGradient(mol, pspace=selected, qspace=(3, 4),
                    denominator="full", state=1)
energy = calc.energy()                           # Eh, nuclear repulsion included
parts = calc.gradient_components()               # base, correction, total; Eh/bohr
checks = calc.check_gradient(components=[(1, 2)]) # one Cartesian component, all three parts
print({key: value["max_error"] for key, value in checks.items()})
```

`gradient()` returns the total gradient; `energy_components()` returns the
three energies. `check_gradient()` defaults to all coordinates with four-point
central differences at a 0.001-bohr step. It re-solves RHF, CC and EOM at every
displacement, using CCpy's `Molecule.with_geometry` and finite-difference utility.
Use `symmetry=False` to keep displaced Cartesian frames consistent.
`stages` exposes the reference, amplitude result, total relaxed densities, and
correction relaxed densities. Only the base gradient includes nuclear repulsion.

## Selections and models

- `state=0`: ground state using `SelectedCC`; `state=1,2,...`: excited roots in
  ascending uncorrected projected-Hbar energy order using `StateOracle`.
  EOM roots are in the fixed alpha/beta electron sector, without a spin filter.
- `pspace`/`qspace` accept tuples of excitation ranks, one-dimensional integer
  arrays of determinant indices, or callables `DeterminantSpace -> indices`.
  A Q rank tuple automatically excludes P. Explicit arrays/callable results
  must already be disjoint. The reference, duplicates and invalid indices are
  rejected. An empty P is allowed for ground-state checks.
- `qspace=None` or an empty selection gives CC(P)/EOMCC(P) without a correction.
  P containing every excited determinant with empty Q gives the FCI limit on
  the physical ground-state branch. Non-downward-closed P is supported.
- `spaces.from_t3_excitations(S, t3_excitations)` converts CCpy selected-triples
  dictionaries into determinant indices. Union this with singles and doubles
  to reuse selections made by CCpy's P-space tools.
- `denominator="full"`: `E_mu - Hbar_KK`.
- `denominator="mp"`: `omega_mu + sum_ref_occ f_pp - sum_K_occ f_pp`.
  Its explicit Fock derivative is included in the correction densities.
  These are two defined denominator models; this interface does not implement
  every named CR-CC/EOMCC denominator approximation.

The correction uses full biorthogonal moments
`sum_K <L_mu|Hbar|K><K|Hbar|R_mu>/D_muK`. The projected EOM matrix includes
the reference and P. The excited right vector includes its reference component;
the CC residual Jacobian is retained for cluster response rather than substituted
for this projected Hamiltonian. The imported complex-step adjoint implementation
returns separate base and correction density matrices. Ground-state
`StateOracle` supplies an independent check against `SelectedCC`.

`DeterminantBackend(..., state=N).run(h, eri)` accepts spatial core Hamiltonian
and chemists' four-index integrals without running SCF. Its `AmplitudeStage`
contains electronic `energy_base`, `delta`, spatial densities, diagnostics and
underlying solver data. Direct `StateOracle` uses the same state numbering;
legacy `EOMOracle(state=0)` means the first excited state. For MP directional
Hamiltonian derivatives, pass `dfock_diag` or contract the integral densities;
`StateResult.gradient_H` alone holds the derivative at fixed Fock diagonal.

## Molecular demonstrations and regression checks

From the repository root, with the CCpy build environment active:

```sh
OPENBLAS_NUM_THREADS=1 python examples/ccpq_determinant_gradients.py --state 0 --denominator full --fd
OPENBLAS_NUM_THREADS=1 python examples/ccpq_determinant_gradients.py --state 1 --denominator mp --fd
OPENBLAS_NUM_THREADS=1 python -m pytest tests/gradients tests/scf/test_ccpq_gradient.py -q
```

The example accepts geometry, basis, frozen core, state, P/Q ranks, a selected
triples stride, denominator, and finite-difference step/stencil/tolerance.
It prints component gradients, residuals, density energy checks, and numerical
errors, and exits unsuccessfully when the requested FD tolerance is exceeded.
Use `--triples-stride 0` for pure rank-selected P, and `--q-ranks` with no values
for an empty Q.

Tests cover synthetic moments, gauge rescaling, orbital rotations, independent
ground adjoints, full-P/empty-Q limits, input validation, actual H4/STO-3G
connected triples, nonclosed molecular P, LiH frozen-core response, and water
ground-state comparisons with existing CCpy CCSD/CR-CC(2,3)_A gradients.
The H4 excited-state MP demonstration with 36 determinants, P=29 and Q=6
agreed with fully re-solved four-point finite differences over all 12 nuclear
coordinates to about 7e-12 Eh/bohr for base, correction and total on this run.
These are derivative consistency checks, not accuracy benchmarks for the method.

## Scope

Molecular gradients currently require a closed-shell singlet RHF reference and
derivative-enabled libint2. With `integral_model="conventional"` (default), the
local RHF solver differentiates unfactorized four-center integrals, irrespective
of a Molecule's DF/Cholesky configuration. With `integral_model="factorized"`,
native RHF and the molecule's DF/Cholesky vectors define the reference and
Hamiltonian. The same determinant densities are contracted with the matching
three-/two-center derivative integrals and native orbital response. Auxiliary
basis and metric derivatives are therefore included in the DF check. This
does not compare a fitted energy with an unfitted analytical derivative.
Both modes use CCpy molecules, derivatives, orbital response and finite differences;
it independently solves determinant amplitudes rather than consuming production
`Driver` amplitudes or production EOM roots.

For an independent tensor-code comparison, run:

```sh
OPENBLAS_NUM_THREADS=1 python examples/ccpq_optimized_comparison.py --basis 6-31g
```

This checks all Cartesian components against fully re-solved optimized DF
energies: CR-CC(2,3) A/D and full CC(P;3) A/D, plus the existing DF-CR-CC(2,3)_A
analytical gradient. P-space triples are imported from CCpy's active-space
list, held fixed, and Q is every remaining triple. The CCSDt selection is
"at least one active hole and one active particle"; it differs from requiring
every hole and particle to be active (`spaces.active_excitations`).

The script also records finite differences of optimized `CCT3`. That method
uses CCSD-like moments/Hbar and left-CCSD (the two-body approximation), while
this determinant functional uses the full selected-space Hbar and left state.
P=SD+selected T and Q=remaining T alone do not make these two functionals
identical. The matching optimized comparator is `CCP3(two_body_approx=False)`;
its current tensor implementation reconstructs four-index integrals from the
same DF vectors. The script reports the CCT3 difference separately rather than
declaring it a successful check of the full functional.
For the CCSDt active-space P list, the full `CCP3(two_body_approx=False)`
calculation is the CC(t;3) comparison without the two-body approximation;
that comparison agrees with the determinant analytical gradient. The
`CCT3` class itself does not expose a `two_body_approx` switch.
Executed comparisons and numerical differences are documented in
`examples/results/ccpq_optimized_comparison.md`.

For the corresponding independent excited-state comparison, run
`examples/ccpq_excited_optimized_comparison.py`. It checks native EOMCCSDt
base energies and full `EOMCCP3` corrections against determinant analytical
gradients, using a fixed active-triples P and its complete triples complement.
The reference spectrum is independently ordered with Arnoldi on optimized
HR, and physical determinant-coordinate right/left eigenvector overlaps
verify the matched singlet/triplet roots. Displaced optimized energies use
orbital sign alignment and overlap/gap guards. See
`examples/results/ccpq_excited_optimized.md` for the full Cartesian water/H4
checks and finite-difference step convergence near the H4 singlet pair.

Use `EOMCCP3` for the full selected-R3/L3 correction. The separate
`EOMCCT3(two_body_approx=False)` path retains VT3 in Hbar but still uses
SD-only correction contractions; the comparison records its different
reference corrections explicitly.

With the default dense backend, full determinant and response matrices limit practical sizes.
This is intended to test arbitrary selected spaces and excitation ranks, not
large-basis production calculations. Root selection uses energy ordering with
no overlap tracking through crossings; degeneracy can make the adjoint singular.
Q denominators smaller than 1e-9 Eh are rejected, with no regularization.
Monitor conditioning and denominators for near-resonant corrections. The tests
do not establish open-shell, spin-flip, electron-attachment/ionization or
near-degenerate-state gradients.

## Optional FCIpy matrix-free backend

The same molecular driver now supports `backend="matrix_free"`. The dense
backend remains the default and an independent complex-step reference. Build
the optional connector with the Python/compiler environment used for CCpy:

```sh
python devtools/build_fcipy_oracle.py /path/to/FCIpy --output build/fcipy
# Run the exact export command printed by the build helper.
# Alternatively pass fcipy_extension="/absolute/path/to/_ccpq_fcipy...so" below.
```

The helper compiles the existing external `fcipy/lib/ci.f90` together with
CCpy's `fortran/fcipy_connected.f90`. It leaves the FCIpy checkout unchanged;
no FCIpy code is vendored or downloaded. NumPy f2py, Meson, Ninja and a Fortran
compiler must be available in the build environment.

```python
calc = CCPQGradient(mol, pspace=selected, qspace=(3, 4), state=1,
                    denominator="mp", backend="matrix_free")
parts = calc.gradient_components()
checks = calc.check_gradient(components=[(1, 2)])
```

The implementation:

- Enumerates connected determinants and calls FCIpy's Slater-rule matrix-element
  and phase routines for each nonzero source coefficient. It never allocates
  the full Hamiltonian. Real/imaginary vector parts are acted on separately.
- Applies finite excitation exponentials as sparse matrix–vector products:
  `Hbar v = exp(-T) H exp(T) v`, including the exact transpose action.
- Solves CC(P) with Newton/GMRES and the exact commutator Jacobian action.
- Uses non-Hermitian Davidson for projected EOM right and left roots, including
  the right reference coefficient and biorthogonal normalization.
- Solves the constrained adjoints with GMRES using an analytical transpose
  Jacobian action. No state-response Jacobian is assembled.
- Pulls outer-product and diagonal factors through the exponentials and contracts
  left/right transition densities with FCIpy. Diagonal factors are streamed.
  No full determinant-pair density coefficient matrix is allocated.
- Includes MP's explicit Fock-diagonal derivative. Full Hbar diagonal elements
  are evaluated by streamed unit-vector actions. An energy-only path avoids
  solving adjoints at finite-difference geometries.

The connector currently accepts real, Hermitian spatial Hamiltonians with equal
alpha/beta electron counts and at most 31 spatial orbitals. It supports frozen
occupied orbitals. The matrix-free backend still stores full-space vectors and
sparse excitation connectivity. FCIpy's transition-RDM kernels scan determinant
pairs, and streamed full-denominator diagonals require one action per Q
determinant; large dense-support P/Q selections may be slow. This removes the
quadratic dense storage bottleneck, without promising production-scale runtime
for every selection. H10/6-31G full-space performance has not been established.

Reproduce the larger molecular check and all regressions with the connector set:

```sh
OPENBLAS_NUM_THREADS=1 python examples/ccpq_h6_631g_matrix_free.py
OPENBLAS_NUM_THREADS=1 python -m pytest tests/gradients tests/scf/test_ccpq_gradient.py -q
```

The H6 example retains all six electrons and all 12 6-31G orbitals: 48,400
determinants, zero frozen/deleted orbitals. Its fixed P contains 6 singles,
6 doubles and 2 triples; Q contains 17 omitted triples. This tests general
selected CC(P;Q), rather than full CCSD with all omitted triples. It checks the
ground state and first two projected excited roots with MP/full denominators,
all 18 Cartesian components using a four-point 0.001-bohr stencil, and two
components again at 0.0005 bohr. Every displaced geometry re-solves RHF, CC and
EOM. Results are saved to `examples/results/ccpq_h6_631g_matrix_free.json`.

This test also exposed and fixed a separate four-center AO derivative bug:
libint2 signals a screened quartet by nulling its first target pointer, while
other derivative pointers may retain previous results. CCpy now skips that
entire quartet. A standalone integral-derivative regression covers this case
without requiring FCIpy.

## Complete SD and complementary high-rank spaces

For conventional CC(P;Q) tests, build P with **every single and double** and
select only the higher ranks. `spaces.active_excitations` selects determinants
whose holes and particles are all inside the specified spatial active sets:

```python
def p_with_active_triples(S):
    sd = spaces.ranks(S, (1, 2))
    triples = spaces.active_excitations(S, (3,), occupied=[1, 2], virtual=[3, 4, 5])
    return spaces.union(sd, triples)

calc = CCPQGradient(mol, pspace=p_with_active_triples, qspace=(3,),
                    state=1, denominator="full", backend="matrix_free",
                    sparse_cache=True)
```

Active indices are zero-based canonical spatial MO indices, including any
frozen orbitals in the numbering. This active selection restricts high-rank
cluster amplitudes only; all singles/doubles and the full determinant space
remain in the calculation. The Q rank tuple automatically takes the full
complement within the requested ranks. For P containing selected quadruples,
union them into P and set `qspace=(3, 4)`.

`examples/ccpq_complete_spaces.py` reproduces the H6/6-31G checks with two models:

| Model | Singles in P | Doubles in P | Triples in P | Quadruples in P | Triples in Q | Quadruples in Q |
|---|---:|---:|---:|---:|---:|---:|
| SD + active T | 54 | 945 | 36 | 0 | 5964 | 0 |
| SD + active T + selected Q | 54 | 945 | 36 | 5 | 5964 | 16195 |

The active set is the highest two occupied and lowest three virtual orbitals;
the second model includes every other active-space quadruple. The script asserts
that P includes **all** singles/doubles, P and Q are disjoint, and their union
exhausts every determinant through the highest requested correction rank.
No electrons/orbitals are frozen or deleted. The complete 48,400-determinant
space is retained for intermediate exponential and Hamiltonian actions.

```sh
OPENBLAS_NUM_THREADS=1 python examples/ccpq_complete_spaces.py --models sdt --output examples/results/h6_sdt_complete.json
OPENBLAS_NUM_THREADS=1 python examples/ccpq_complete_spaces.py --models sdtq --output examples/results/h6_sdtq_complete.json
```

Each model checks states 0, 1 and 2 with MP/full denominators, comparing the
separate base, correction and total analytical gradients against four-point
finite differences at a 0.001-bohr step. The checked components are atom 1/z
and atom 2/x (zero-based): one longitudinal and one transverse displacement.
Displaced RHF, CC and EOM equations are fully re-solved. Unlike the earlier
small-selected-P demonstration, these are complete SD spaces and complete
complementary T/TQ correction spaces. The JSON reports include the full lists,
rank counts, analytic gradients, numerical components and residuals. Use
`--resume` to preserve completed passing state/denominator pairs, or
`--basis sto-3g` for a faster six-electron check.

For these dense-support vectors, `sparse_cache=True` builds the bare connected
Hamiltonian as CSR and uses connected Slater-rule transition densities. This
stores **sparse** determinant-pair matrices, without any dense full-space
matrix. H6/6-31G has 48.4 million structural Hamiltonian entries, about 581 MB
for CSR values/indices; additional excitation and sparse commutator storage
is required. Memory/runtime therefore differ from the cache-free backend.
When the connector provides `update_connected_values`, subsequent geometries
reuse the structural Slater graph and recompute every Hamiltonian value from
the new integrals. This retains structural zero entries needed by integral
derivatives. Older connectors rebuild the graph. Reference states with the
same P/Q selection can share this graph; CC/EOM equations and response are
still solved and checked for each calculation.
The full-denominator diagonal and its derivatives use exact rank-limited
commutators: a two-body H cannot lower excitation rank by more than two, so
only T1, T2 and the quadratic T1 term contribute to diagonal Hbar elements.
This avoids one full-space action per Q determinant. Independent tests compare
the diagonal, its amplitude derivatives, and its integral densities with dense
complex-step expressions, including nonzero triple/quadruple amplitudes.

The adjoint solver now exploits the block-triangular CC/state constraints,
with a sparse bordered diagonal preconditioner. Left Davidson searches are
seeded from the right eigenvector to preserve its symmetry sector, eigenvalues
must agree, and vanishing biorthogonal overlap or unconverged constraints are
rejected. The complete-SD/selected-quadruple tests exposed a root mismatch that
was not exercised by the earlier tiny selected spaces.

Full-support Davidson seeds and a buffer of converged low roots also prevent
the singles-only starting subspace from missing a lower state in a different
symmetry sector. Independent SciPy Arnoldi confirmed the true H6 SDT second
root to 1.1e-14 Eh; the previous candidate was actually third. A dedicated
disconnected-sector regression and the dense projected-eigensystem check pin
both root-selection fixes. Final numerical results for all 12 H6/6-31G cases
are in `examples/results/ccpq_complete_spaces.md` and the associated JSON files.

## Substantial triple/quadruple fractions

The complete-space demonstration accepts explicit active occupied and virtual
indices and a quadruple stride. Setting the stride to 1 includes every active
quadruple. All singles/doubles still span the full MO basis, and Q remains the
complete complementary T+Q set.

```sh
OPENBLAS_NUM_THREADS=1 python examples/ccpq_complete_spaces.py --models sdtq --active-occupied 0 1 2 --active-virtual 3 4 5 6 7 8 --quadruple-stride 1 --output examples/results/h6_healthy_v6.json
OPENBLAS_NUM_THREADS=1 python examples/ccpq_complete_spaces.py --models sdtq --active-occupied 0 1 2 --active-virtual 3 4 5 6 7 8 9 --quadruple-stride 1 --output examples/results/h6_healthy_v7.json
```

| Active virtuals | Triples in P | Quadruples in P | Selected T fraction | Selected Q fraction | Triples in Q | Quadruples in Q |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 1660 | 2745 | 27.7% | 16.9% | 4340 | 13455 |
| 7 | 2716 | 5439 | 45.3% | 33.6% | 3284 | 10761 |

The P spaces have 5,404 and 9,154 excited determinants, respectively, including
all 54 singles and 945 doubles. Both include pure-spin triples and the aaab,
aabb and abbb quadruple patterns. The selected sets are nested; no electrons
or basis orbitals are frozen/deleted.

Displaced RHF orbital signs are aligned with the reference using cross-geometry
AO overlaps (`--no-align-phases` disables this). Only signs are adjusted; the
canonical orbitals are not mixed or reordered. A regression checks invariance
of the selected CC/EOM energies under these phase choices.

For excited-state finite differences the example can follow its independently
ordered reference root. `backend.follow_reference(reference_stage.data)` sets
an excitation-energy target relative to the current CC(P) ground-state energy,
with a guard of one quarter of the nearest buffered reference-root gap, and
requires at least 0.5 normalized right-vector overlap. The relative target
allows common shifts of the electronic energies at displaced geometries.
Degenerate reference roots or a displacement leaving that neighborhood are
rejected. Residual and stationarity checks remain active. Use
`backend.clear_reference()` to restore ordinary energy-ordered root selection,
or `--no-follow-reference` in the example. Reference-root following is local to
a nondegenerate numerical stencil and assumes consistent orbital phases; it
does not implement general near-crossing dynamics.

Executed results for these substantial T/Q fractions are documented in
`examples/results/ccpq_healthy_spaces.md`, with complete machine-readable
selection and numerical-gradient reports for both active virtual spaces.

The water demonstration uses the same molecular and determinant drivers:

```sh
OPENBLAS_NUM_THREADS=1 python examples/ccpq_complete_spaces.py \
  --geometry 'O 0 0 0; H .05 -.75 .59; H -.03 .78 .62' \
  --basis 6-31g --nfrozen 1 --models sdtq --states 0 1 2 \
  --active-occupied 1 2 3 4 --active-virtual 5 6 7 8 9 \
  --quadruple-stride 1 --components 1:2 2:1 --verify-roots \
  --output examples/results/h2o_631g_frozen_core.json --resume
```

This freezes only O(1s) in the correlation calculation and retains all 13
spatial basis orbitals. It has 245,025 determinants and about 4.19 GB of bare
Hamiltonian CSR storage, before excitation/response intermediates. The
all-electron space has 1,656,369 determinants; its structural connected cache
exceeds the current 32-bit CSR connector capacity. The command above is
therefore a frozen-core check. It numerically checks two Cartesian components
and records the complete analytical gradient. The quick water regression in
`tests/gradients/test_ccpq_water.py` explicitly uses three frozen occupied
orbitals, a separate reduced correlation model.
Executed ground/excited results, all-component reduced-model checks and
reproduction details are in `examples/results/h2o_631g_gradients.md`.
