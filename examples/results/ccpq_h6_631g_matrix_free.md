# H6/6-31G matrix-free gradient verification

Executed on 2026-11-02 using the optional external FCIpy connector.

All six electrons and 12 spatial orbitals were retained, with no frozen or
deleted orbitals. The full determinant space contains 48,400 determinants.
P contains 6 singles, 6 doubles and 2 triples; Q contains 17 omitted triples.
These fixed lists define a selected CC(P;Q)/EOMCC(P;Q) model. This does not
benchmark complete CCSD/triples spaces or establish the physical accuracy of
the corrected energies. The exact geometry and selections are in the JSON.

Analytical, fully RHF-relaxed nuclear gradients were compared with four-point
central differences of separately re-solved base and correction energies.
All 18 Cartesian components used a 0.001-bohr displacement. A longitudinal
and a transverse component were repeated at 0.0005 bohr. Errors below are the
maximum across both step sizes, in Eh/bohr; correction energies are in Eh.
State 0 is ground; states 1 and 2 are the first two projected EOM roots.

| Denominator | State | Correction energy | Base error | Correction error | Total error |
|---|---:|---:|---:|---:|---:|
| mp | 0 | -1.8702240748e-05 | 3.992e-11 | 1.001e-13 | 4.015e-11 |
| mp | 1 | -3.4066455150e-03 | 4.398e-11 | 3.593e-12 | 4.734e-11 |
| mp | 2 | -7.3825329818e-03 | 1.065e-10 | 6.402e-12 | 1.123e-10 |
| full | 0 | -2.7811358834e-05 | 3.992e-11 | 1.552e-13 | 4.054e-11 |
| full | 1 | -5.2871791255e-03 | 4.398e-11 | 5.781e-12 | 4.891e-11 |
| full | 2 | -1.2805896322e-02 | 1.065e-10 | 1.051e-11 | 1.157e-10 |

All cases pass the requested numerical tolerance of 2e-8 Eh/bohr. Minimum
absolute Q denominators exceed 0.59 Eh in these cases. The complete analytical
and numerical gradient arrays, smaller-step results, force/torque sums and
solver diagnostics are in [the machine-readable report](ccpq_h6_631g_matrix_free.json).

The backend uses FCIpy Slater-rule actions, sparse finite excitation
exponentials, non-Hermitian Davidson and GMRES adjoints. It builds no dense
full-space Hamiltonian, exponential, response Jacobian or determinant-pair
density coefficient matrix. Independent small-system dense comparisons cover
energies, separate 1-/2-RDMs, nonclosed P, empty Q, and nuclear gradients.

The H6 test exposed a separate AO ERI derivative bug: when libint2 screened
out a shell quartet, CCpy consumed stale derivative pointers after the first
null target. Skipping the entire screened quartet repaired errors up to about
1e-3 Eh/bohr in the initial nuclear tests. A standalone AO derivative
regression was added alongside the matrix-free tests.

Reproduce from the repository root using the CCpy Python build environment:

```sh
python devtools/build_fcipy_oracle.py /path/to/FCIpy --output build/fcipy
# Execute the export CCPY_FCIPY_EXTENSION command printed by the helper.
OPENBLAS_NUM_THREADS=1 python examples/ccpq_h6_631g_matrix_free.py
OPENBLAS_NUM_THREADS=1 python -m pytest tests/gradients tests/scf/test_ccpq_gradient.py -q
```

FCIpy's pair-scanning transition-RDM implementation and streaming of each
full-denominator Q diagonal still limit large selections. H10/6-31G and
complete H6 CCSD/all-triples spaces have not been validated here.
