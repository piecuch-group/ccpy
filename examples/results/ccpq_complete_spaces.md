# Complete-SD CC(P;Q)/EOMCC(P;Q) gradient checks

H6/6-31G; all six electrons and all 12 spatial orbitals retained, zero frozen
or deleted orbitals. Intermediate exponential and Hamiltonian actions use the
complete 48,400-determinant space, including ranks above Q.

| Model | Singles in P | Doubles in P | Triples in P | Quadruples in P | Triples in Q | Quadruples in Q |
|---|---:|---:|---:|---:|---:|---:|
| SDT | 54 | 945 | 36 | 0 | 5964 | 0 |
| SDTQ | 54 | 945 | 36 | 5 | 5964 | 16195 |

P contains **every single and double**. Triples are a CAS-style selection with
all holes/particles in occupied spatial orbitals [1,2] and virtual orbitals
[3,4,5] (zero-based). The second model adds every other active-space quadruple.
Q is the **complete remaining** triples / triples+quadruples set, with no
subsampling. The script checks disjointness and exact coverage of every
requested excitation rank.

States 0, 1 and 2 denote the ground and lowest two projected excited roots.
Each case compares separate base, correction and total analytical nuclear
gradients against four-point central differences of fully re-solved RHF, CC
and EOM energies at a 0.001-bohr step. The components are atom 1/z and atom 2/x
(zero-based), giving longitudinal and transverse tests. These are two nuclear
components per case; the table does not claim a full 18-component numerical
check for these larger P/Q spaces. Both denominator variants use the same
underlying displaced CC/EOM state.

Errors are absolute, in Eh/bohr; correction energies are in Eh.

| Model | State | Denominator | Correction | Base gradient error | Correction gradient error | Total gradient error |
|---|---:|---|---:|---:|---:|---:|
| SDT | 0 | mp | -8.2189571445e-04 | 7.175e-11 | 1.751e-12 | 7.073e-11 |
| SDT | 0 | full | -1.0525220044e-03 | 7.175e-11 | 2.247e-12 | 7.022e-11 |
| SDT | 1 | mp | +7.7791273201e-04 | 5.925e-11 | 4.671e-11 | 1.243e-11 |
| SDT | 1 | full | +1.1046147595e-03 | 5.925e-11 | 6.239e-11 | 1.052e-11 |
| SDT | 2 | mp | -9.6683641451e-04 | 4.549e-11 | 1.879e-11 | 2.673e-11 |
| SDT | 2 | full | -1.2488272760e-03 | 4.550e-11 | 2.616e-11 | 2.176e-11 |
| SDTQ | 0 | mp | -7.6068626236e-04 | 8.298e-11 | 9.819e-13 | 8.279e-11 |
| SDTQ | 0 | full | -9.7052285559e-04 | 8.298e-11 | 1.243e-12 | 8.240e-11 |
| SDTQ | 1 | mp | +8.0900363193e-04 | 4.233e-11 | 5.099e-11 | 8.935e-12 |
| SDTQ | 1 | full | +1.1482514713e-03 | 4.233e-11 | 6.812e-11 | 2.562e-11 |
| SDTQ | 2 | mp | -9.6777482557e-04 | 3.878e-11 | 1.832e-11 | 2.027e-11 |
| SDTQ | 2 | full | -1.2481432186e-03 | 3.880e-11 | 2.555e-11 | 2.310e-11 |

All 12 cases pass the 2e-8 Eh/bohr tolerance. Full definitions, gradient
arrays, finite-difference components and residuals are in:

- [SDT report](h6_sdt_complete.json)
- [SDTQ report](h6_sdtq_complete.json)
- [H6/STO-3G auxiliary checks](h6_sto3g_complete.json)

The exact same full-SD/complement constructions were checked independently on
H4/6-31G for all three states and both denominators. The six-electron STO-3G
auxiliary checks cover both models and all states/denominators as well.

## Issues exposed and fixed

The original full-denominator implementation streamed one full-space action
per Q determinant. Exact sparse commutators now evaluate all Hbar diagonal
values and their density/amplitude derivatives together. The optional sparse
Hamiltonian cache and connected transition-RDM contractions make these large
Q spaces practical; state and CC adjoints use block GMRES preconditioning.
There is no dense full-space determinant-pair matrix allocation.

The larger excited-state tests exposed two Davidson issues. A left search
could converge in a different symmetry sector from its right root; it is now
seeded from the right vector, checked for matching eigenvalue and overlap,
and unconverged constraints are rejected. In addition, a few low-diagonal
singles guesses could miss a lower root in another sector. Full-support seeds
and a buffer of converged low roots now precede selection of the requested
state. Both conditions have dedicated regression checks.

For SDT, independent SciPy Arnoldi gives the lowest projected total energies
as -3.19136109, -3.04715180, -3.01800848 and -3.00917654 Eh. Davidson's corrected
second root agrees within 1.1e-14 Eh. The previous second-root candidate was
actually third. See [the independent ordering check](h6_complete_root_order.json).
Only final, correctly ordered root results appear in the table above.

## Reproduction

Activate the CCpy build environment, build the external FCIpy connector, and
execute the export command printed by the helper:

```sh
python devtools/build_fcipy_oracle.py /path/to/FCIpy --output build/fcipy
OPENBLAS_NUM_THREADS=1 python examples/ccpq_complete_spaces.py --models sdt --output examples/results/h6_sdt_complete.json
OPENBLAS_NUM_THREADS=1 python examples/ccpq_complete_spaces.py --models sdtq --output examples/results/h6_sdtq_complete.json
OPENBLAS_NUM_THREADS=1 python -m pytest tests/gradients tests/scf/test_ccpq_gradient.py -q
```

`--resume` preserves completed passing state/denominator pairs. The sparse
Hamiltonian cache alone uses about 581 MB for this system, with additional
excitation/commutator storage. Runtime and memory remain limitations for
larger systems. These establish derivative consistency for the stated models;
they do not assess the physical accuracy of the corrected energies.
