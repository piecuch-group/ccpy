# CC(P) Cholesky layout regression fixture

`ccsdt_p_chol_layout_reference.npz` contains all updated amplitude and residual blocks from the original implementation, before the shared packing and contraction-order changes. The deterministic tensor setup (NumPy seed 792) is in `tests/test_ccsdt_p_chol_layout.py`. Keys encode RHF flag, empty-P flag, amplitude/residual index, and spin block.

Four cases cover RHF and unrestricted evaluation, each with empty P or a nonempty P in every spin block. Nonempty cases start with nonzero T3 amplitudes. Bare and cluster tensors are synthetic, deliberately retaining fewer physical symmetries to expose incorrect permutations. This is a regression oracle, not a molecular energy reference. Comparisons use 2e-12 absolute and 2e-11 relative tolerances.
