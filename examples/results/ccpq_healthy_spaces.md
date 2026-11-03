# Substantial selected triple/quadruple P spaces

H6/6-31G, all six electrons and 12 spatial orbitals retained. P includes every
single (54) and double (945), plus complete CAS-style triples and quadruples
in all three occupied orbitals and either six or seven of the nine virtual
orbitals. Q is the complete remaining triples+quadruples complement.

| Active virtual MOs | Triples in P | P triple fraction | Quadruples in P | P quadruple fraction | Total P | Triples in Q | Quadruples in Q |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3–8 (six MOs) | 1660 | 27.67% | 2745 | 16.94% | 5404 | 4340 | 13455 |
| 3–9 (seven MOs) | 2716 | 45.27% | 5439 | 33.57% | 9154 | 3284 | 10761 |

All indices are zero-based. The selected sets are nested. Both include all
allowed triple spin patterns and aaab/aabb/abbb quadruples. No orbital or
electron is frozen/deleted; only the higher-rank P amplitudes are restricted.
Full 48,400-determinant intermediates are retained, including ranks five/six
induced by the finite excitation exponentials. P/Q coverage and disjointness
are asserted exactly; there is no subsampling of Q.

## Nuclear derivative results

Ground and the lowest two excited states were checked with MP/full
denominators. Four-point central differences at 0.001 bohr perturb atom 1/z
and atom 2/x (zero-based), independently testing a longitudinal and transverse
component. RHF, CC and EOM equations are fully re-solved at every point.
Separate base, correction and total gradients are checked; these larger-space
runs numerically check two components per case, not all 18 components.

Correction energies are in Eh and gradient errors in Eh/bohr.

| Active virtual count | State | Denominator | Correction energy | Base error | Correction error | Total error |
|---:|---:|---|---:|---:|---:|---:|
| 6 | 0 | mp | -3.6662672260e-04 | 8.847e-12 | 3.860e-13 | 8.754e-12 |
| 6 | 0 | full | -4.5782649173e-04 | 8.847e-12 | 4.812e-13 | 8.811e-12 |
| 6 | 1 | mp | +5.0154743183e-04 | 5.745e-12 | 3.423e-12 | 9.192e-12 |
| 6 | 1 | full | +6.8515931153e-04 | 5.745e-12 | 4.548e-12 | 9.855e-12 |
| 6 | 2 | mp | -5.7244397876e-04 | 9.871e-12 | 1.131e-11 | 2.100e-11 |
| 6 | 2 | full | -7.4134494357e-04 | 9.838e-12 | 1.443e-11 | 2.404e-11 |
| 7 | 0 | mp | -2.3737359034e-04 | 2.005e-11 | 2.038e-13 | 2.027e-11 |
| 7 | 0 | full | -2.9484948208e-04 | 2.005e-11 | 2.564e-13 | 2.013e-11 |
| 7 | 1 | mp | +2.4399256860e-04 | 2.384e-12 | 4.808e-12 | 3.427e-12 |
| 7 | 1 | full | +3.4525146882e-04 | 2.388e-12 | 6.128e-12 | 4.930e-12 |
| 7 | 2 | mp | -2.9375841163e-04 | 1.314e-11 | 5.385e-12 | 1.831e-11 |
| 7 | 2 | full | -3.7638224543e-04 | 1.314e-11 | 6.665e-12 | 1.959e-11 |

All 12 cases pass the 2e-8 Eh/bohr tolerance. Full selections, analytic gradient
arrays, finite-difference components, rank fractions and solver diagnostics:

- [Six active virtuals](h6_healthy_v6.json)
- [Seven active virtuals](h6_healthy_v7.json)
- [Independent root-order check](h6_healthy_root_order.json)

Independent SciPy Arnoldi checks all four buffered reference roots. Maximum
Arnoldi/Davidson energy differences are below 1.2e-13 Eh for both P spaces.
Canonical MO signs are aligned by cross-geometry AO overlaps. This changes
only the phase gauge, and an independent energy-invariance regression covers
it. For the final second-state finite differences, the verified reference
root is followed using an electronic-energy guard of one quarter of the
nearest buffered gap and a normalized right-vector overlap of at least 0.5.
Degenerate references and displacements leaving the guard are rejected.
The same CC/EOM equations and residual tolerances are used. Earlier states
were independently re-ordered at each displaced point. No near-crossing
claim is made.

The regression suite passes 126 tests, including substantial-fraction T/Q
models on H4/6-31G, MO-phase invariance, and comparisons of guarded reference
following with independently energy-ordered solutions (energies and separate
1-/2-RDMs). The root-following tests also reject an out-of-neighborhood shift
and a degenerate reference.

## Reproduce

With the CCpy build environment active and CCPY_FCIPY_EXTENSION set using the
output of devtools/build_fcipy_oracle.py:

```sh
OPENBLAS_NUM_THREADS=1 python examples/ccpq_complete_spaces.py --models sdtq --active-occupied 0 1 2 --active-virtual 3 4 5 6 7 8 --quadruple-stride 1 --output examples/results/h6_healthy_v6.json
OPENBLAS_NUM_THREADS=1 python examples/ccpq_complete_spaces.py --models sdtq --active-occupied 0 1 2 --active-virtual 3 4 5 6 7 8 9 --quadruple-stride 1 --output examples/results/h6_healthy_v7.json
OPENBLAS_NUM_THREADS=1 python -m pytest tests/gradients tests/scf/test_ccpq_gradient.py -q
```

`--resume` preserves completed passing states. `--no-follow-reference` performs
fresh energy ordering at every numerical point; `--no-align-phases` disables
MO sign alignment. Sparse connected-Hamiltonian caching is enabled. These
are consistency checks of the stated selected models, not benchmarks of the
physical accuracy of their corrected energies.
