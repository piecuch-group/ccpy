# Independent DF tensor-code / determinant derivative comparison

The determinant code and optimized CCpy tensor code solve their own amplitude
and left equations. They share molecular definitions and integral conventions,
not amplitude solutions. The finite differences below use optimized tensor
energies, not the determinant energy routine.

## Integral and space definitions

- Geometry (angstrom): `O 0 0 0; H .05 -.75 .59; H -.03 .78 .62`.
- Basis sets: STO-3G pilot and 6-31G primary calculation.
- Reference: native RHF, DF with `def2-universal-jkfit`; O(1s) is frozen in
  correlation. No spatial basis orbitals are deleted. These are not
  all-electron correlation calculations.
- All nine Cartesian components are checked by four-point central differences
  at 0.001 bohr. RHF and all tensor CC/left equations are solved afresh at
  every point. Iterative energy/amplitude tolerances are 1e-13/1e-11.
- CR-CC(2,3): P is every single and double; Q is every triple.
- Selected triples: the CCpy CCSDt selection with one active occupied orbital,
  one active virtual orbital, and `num_active=1`. A triple must contain at
  least one active hole and at least one active particle. The actual
  `get_active_triples_pspace` list is imported into the determinant P space;
  Q is its complete triples complement. Lists are fixed across displacements.
  This is not the selection requiring every hole/particle to be active.
- Variant A maps to determinant `denominator="mp"`; variant D maps to
  `denominator="full"` in these ground-state SD/SDt spaces. Variants B/C
  are not claimed as gradient checks here.

`CCPQGradient(..., integral_model="factorized")` uses the same fitted
Hamiltonian for its independent determinant amplitude solve. It contracts
the resulting bare spatial densities with DF-vector derivatives through the
native orbital response and 3-/2-center derivative machinery. Auxiliary-basis
and metric derivatives are included. The default conventional-integral mode
is retained. Comparing the conventional analytical derivative directly with
finite-auxiliary-basis DF energies would test different Hamiltonians.

## Numerical results

Every listed error is the maximum absolute difference over all nine Cartesian
components, in Eh/bohr. Optimized tensor energies supply the finite differences.
The numerical total gradient is the sum of the separately differenced base
and correction energies, using the linearity of the same four-point stencil.

### 6-31G

| Matched optimized functional | Variant | Base error | Correction error | Total error |
|---|---|---:|---:|---:|
| CR-CC(2,3) | A | 1.091e-10 | 5.870e-14 | 1.090e-10 |
| CR-CC(2,3) | D | 1.091e-10 | 5.509e-14 | 1.090e-10 |
| Full CC(P;3) | A | 1.152e-10 | 3.064e-14 | 1.152e-10 |
| Full CC(P;3) | D | 1.152e-10 | 5.232e-14 | 1.152e-10 |

Largest base-energy difference: 2.84e-13 Eh.
Largest correction-energy difference: 2.49e-15 Eh.
CR-CC(2,3)-A determinant/existing DF analytical-gradient difference:
1.43e-13 Eh/bohr.

The primary CR case has P = 64 S + 1,360 D and Q = all 11,200 T.
The selected-triples case has P = 64 S + 1,360 D + 2,436 T and
Q = all 8,764 remaining T. Both use 245,025 determinant support states.

### STO-3G

| Matched optimized functional | Variant | Base error | Correction error | Total error |
|---|---|---:|---:|---:|
| CR-CC(2,3) | A | 7.599e-11 | 5.164e-15 | 7.598e-11 |
| CR-CC(2,3) | D | 7.599e-11 | 3.304e-14 | 7.598e-11 |
| Full CC(P;3) | A | 7.547e-11 | 5.320e-15 | 7.547e-11 |
| Full CC(P;3) | D | 7.547e-11 | 7.735e-15 | 7.548e-11 |

Largest base-energy difference: 2.84e-14 Eh.
Largest correction-energy difference: 1.09e-15 Eh.
CR-CC(2,3)-A determinant/existing DF analytical-gradient difference:
9.83e-13 Eh/bohr.

### Direct optimized CC(t;3) comparison

These are differences from the full determinant functional, rather than errors
between two implementations of the same approximation. The shared CCSDt base
energy/gradient already passes the full-CC(P;3) comparisons above.

| Basis | Variant | Determinant minus CCT3 correction (Eh) | Correction-gradient difference | Total-gradient difference |
|---|---|---:|---:|---:|
| 6-31G | A | +1.351655e-05 | 2.046653e-05 | 2.046645e-05 |
| 6-31G | D | +1.669676e-05 | 2.674901e-05 | 2.674893e-05 |
| STO-3G | A | +3.970946e-07 | 5.314081e-06 | 5.314096e-06 |
| STO-3G | D | +5.608524e-07 | 7.921606e-06 | 7.921622e-06 |

All matched-functionals checks pass the 2e-8 Eh/bohr tolerance. The direct
optimized CCT3 comparison does not pass that tolerance; the functional
distinction below explains the difference. An analytical gradient of the
default two-body-approximate `CCT3` path would require a separate determinant
functional reproducing its
CCSD-like moment and left-state approximations.

## Method distinction: optimized CC(t;3) versus full CC(P;3)

Identical P/Q selections do not define the entire correction functional.
The optimized `CCT3` method uses CCSD-like Hbar/moments and left-CCSD, built
from the converged CCSDt singles/doubles. The determinant oracle uses the
full selected-space similarity transform and the full left-CC(P) state,
including selected triples. These are different approximations and give
different correction energies and derivatives.

The matching optimized full-functional calculation is
`CCP3(cc, two_body_approx=False)`. That implementation currently reconstructs
four-index integrals from the same DF vectors; it is an independent tensor
implementation, but not a fully factorized full-CC(P;3) kernel.

For the active-space CCSDt P selection used here, this is the requested
**CC(t;3) calculation without the two-body approximation**. The earlier
direct-CCT3 mismatch concerns the default approximate `CCT3` implementation,
not this full calculation. Its `CCT3` class does not expose a
`two_body_approx` switch; explicitly use `CCP3` with the same triples list.

A fresh 6-31G finite-difference recheck used `CCP3(two_body_approx=False)`
at every displacement and asserted the `HBarCCSDT_P` and `LeftCCSDT_P`
classes. It compared all nine Cartesian components with the previously
computed determinant analytical gradients, retaining the same geometry,
DF auxiliary basis, frozen O(1s), and fixed P/Q spaces:

| Full CC(t;3) variant | Correction-gradient error | Total-gradient error |
|---|---:|---:|
| A | 3.051e-14 | 1.152e-10 |
| D | 5.246e-14 | 1.152e-10 |

Errors are in Eh/bohr. Base and correction energies agree within 2.85e-13
and 2.49e-15 Eh, respectively. This recheck reuses the saved determinant
analytical arrays and freshly recomputes all optimized finite-difference
energies. Separately, the focused full-selected-triples DF/Cholesky tests
freshly recomputed both analytical and numerical sides and passed (two tests).
The complete recheck data and its analytical provenance are in
[h2o_631g_cct3_full_recheck.json](h2o_631g_cct3_full_recheck.json).

As a control, optimized `CCT3` and `CCP3(two_body_approx=True)` corrections
agree for all A/B/C/D variants: maximum differences are at most 6.8e-21 Eh (STO-3G)
and 2.2e-18 Eh (6-31G). Thus the observed CCT3/full-oracle difference cannot
be removed by relabeling the selected triples or changing a denominator name.
The current determinant analytical gradient is not an analytical gradient
of the optimized CC(t;3) approximation.
The reference control is saved in
[h2o_cct3_two_body_control.json](h2o_cct3_two_body_control.json).

## Reproduction

With CCpy's build environment active, build the optional FCIpy connector with
`devtools/build_fcipy_oracle.py` and set its printed `CCPY_FCIPY_EXTENSION`:

```sh
OPENBLAS_NUM_THREADS=1 python examples/ccpq_optimized_comparison.py \
  --basis sto-3g --output examples/results/h2o_sto3g_optimized_df.json
OPENBLAS_NUM_THREADS=1 python examples/ccpq_optimized_comparison.py \
  --basis 6-31g --output examples/results/h2o_631g_optimized_df.json
OPENBLAS_NUM_THREADS=1 python -m pytest tests/gradients tests/scf/test_ccpq_gradient.py -q
```

The script compares base, correction and total derivatives separately. It
requires the matched CR-CC(2,3)/full-CC(P;3) comparisons to pass and records
the distinct optimized CCT3 result without declaring it the same functional.
`--integrals cholesky` selects a 1e-13-threshold Cholesky Hamiltonian instead
of DF. Focused regression tests cover both integral models and an excited
state in factorized mode.

The complete regression run passed 140 tests. In particular, both the existing
conventional water checks and the new DF/Cholesky checks pass; the original
fresh-reference policy is preserved for conventional energy/gradient checks.

The machine-readable reports contain complete gradient arrays, numerical
components, selected-triples lists, P/Q sizes, correction energies and solver
checks:

- [STO-3G](h2o_sto3g_optimized_df.json)
- [6-31G](h2o_631g_optimized_df.json)
