# CC(P;Q) analytical gradient derivation

The manuscript is [`ccpq_analytical_gradients.tex`](ccpq_analytical_gradients.tex).
The compiled PDF is [`output/pdf/ccpq_analytical_gradients.pdf`](../../output/pdf/ccpq_analytical_gradients.pdf).

It derives the full biorthogonal ground/excited-state moment functional, its
separate base/correction adjoints, explicit ground Sigma/Pi and excited-state
response equations, bare-integral densities, RHF orbital/overlap response, and
matching conventional/DF/Cholesky contractions. Numerical tables come directly
from the saved JSON reports, with component counts and frozen-core conventions
stated per model. It also distinguishes the full CC(P;3)/EOMCC(P;3) paths from
the separate SD-only CCT3/EOMCCT3 correction approximations.

## Notation crosswalk

| Manuscript | Determinant code |
|---|---|
| Ground projected left coefficients, lambda | `SelectedCC` result `lam` |
| Base gradient multiplier, lambda^g | `lambda_grad` |
| Q right coefficient u = M/D | `r` in `SelectedCC.arrays`; `divided` in `MatrixFreeState.objective` |
| Q left coefficient v = N/D | `ell` |
| Q weights w = NM/D^2 | `weights` in `MatrixFreeState.objective` |
| Sigma and Pi correction responses | `sigma`, `pi` |
| Full projected reference+P matrix A | `small` in `StateOracle`, `full = [0, P]` index list |
| Ground residual Jacobian J | `SelectedCC.jacobian` and exact matrix-free cluster action |
| B = Hbar_PP - E0 | Ground projected-left constraint/action |
| State gauge index q | `pivot` |
| Excited adjoint `(z_t, a, b_without_q, nu)` | `StateOracle` result arrays `adj_base`, `adj_corr` |
| Stationary Hbar coefficient | `CoefficientMatrix` / matrix-free `Factors` |
| Bare Hamiltonian coefficient K = U^T Abar V^T | `gradient_H_base`, `gradient_H_corr` in the dense oracle |
| Explicit MP Fock derivative | `d_mp`, `mp_denominator_densities` |

`state=0` denotes the ground state; positive states are ordered in the full
fixed-Ms determinant sector, without a singlet filter. Code variables named
`r` can denote a Q divided coefficient or a P-space eigenvector depending on
the class; the manuscript uses separate symbols to avoid that ambiguity.

## Regenerate the evidence and compile

For the standalone source bundle, extract it and run:

```sh
tectonic --untrusted ccpq_analytical_gradients.tex
```

The `evidence/` folder in that bundle contains report snapshots. Manifest
paths refer to the original repository layout. Running the Python checkers
or regenerating tables requires the CCpy checkout; compiling the supplied
generated LaTeX inputs does not.

From the repository root, with the CCpy environment active:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python devtools/check_ccpq_theory.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python devtools/check_ccpq_lean_response.py
python devtools/build_ccpq_theory_evidence.py
cd docs/theory
tectonic --untrusted --outdir ../../output/pdf ccpq_analytical_gradients.tex
```

An ordinary LaTeX installation can compile the four `.tex` files together;
Tectonic downloads standard resources on its first use. The supplied source
bundle includes generated tables, so PDF compilation does not require running
molecular calculations or importing CCpy.

`ccpq_formula_checks.json` records 12 additional algebra checks (closed and
nonclosed P, ground/two excited states, MP/full), comparing expanded formulas
with independent complex-step derivatives. `ccpq_regression_run.txt` preserves
the 144-test gradient regression summary. These are distinct checks/counts.

`ccpq_validation_manifest.json` records SHA-256 hashes of implementation sources,
reports, and manuscript inputs in this locally modified checkout. It is a
provenance snapshot, not a scientific correctness certificate. It deliberately
does not identify the modified implementation merely by the Git HEAD revision.

The reduced excited-state total-gradient section derives two projected state
responses, their gauge equivalence, one combined cluster adjoint, and three
outer-product density factors plus denominator terms. It gives a production
work sequence and reuse rules. `ccpq_lean_response_checks.json` records eight
closed/nonclosed-P, two-root, full/MP fixed-Fock coefficient checks against
the dense constrained oracle (maximum error 5.551115123125783e-15). The
explicit MP Fock density remains required. This is an algebraically verified
prototype, not a deployed optimization or a molecular timing benchmark.
