# Rank-local Cholesky CC(P) vvvv layers

## Update order

`ccpy/cc/ccsdt_p_chol.py` now performs:

1. T1 updates and T1-transformed Cholesky factors.
2. Non-vvvv T2 residuals for aa, ab, and (unless RHF) bb.
3. A T2-only vvvv layer, followed by T2 antisymmetrization, denominator
   application, and amplitude updates.
4. CCSD intermediates built from the **updated** T2 amplitudes.
5. Non-vvvv T3 residuals for aaa, aab, and (unless RHF) abb, bbb.
6. A T3-only vvvv layer, followed by T3 denominator and amplitude updates.

This is Gauss–Seidel **between ranks**, with unchanged amplitudes within
each T2/T3 residual build. No integral tiles or residuals are shared between
T1, T2, and T3. There is no separate vvvv*T1 term. RHF copies the redundant
amplitudes/residuals after completing each rank.

`ccpy/cc/chol_ppl.py` holds the two layers. T3 integral tiles serve every
applicable spin case of their spin channel. Same-spin T3 tiles include the
half-weight T2 dressing; opposite-spin tiles include the full dressing.
Connectivity is rebuilt after native residual routines finish sorting P
lists. No mutable connectivity or integral cache survives a call.

## Retained changes

- Open-shell T3b sorting uses the beta occupied dimension, not alpha.
- Empty P-space buckets initialize determinant indices safely.
- One-body exchange is reassociated through an occupied-sized temporary.
- Native-layout packing and straightforward RHF/common-product reuse remain.
- Optional native `residual_only` mode skips vvvv and returns raw residuals.
  Default native behavior retains the original equations for regression checks.
- Whole-iteration Jacobi and the combined T2/T3 contraction are removed.

## Parallel implementation and measurements

Native kernels distribute bounded pair tiles, rather than uneven outer
orbital loops. Mixed-spin T2 avoids an integral transpose. T3 writes one
value per target record, then gathers at most three values per determinant
in fixed order: no atomics and no per-thread copies of the full residual.
Integral scratch is bounded independently of total virtual-space size;
no four-virtual tensor is materialized.

Run `python devtools/benchmark_rank_vvvv.py` in the built environment.
It uses fresh processes and single-threaded BLAS, and reports complete-layer
and native-only timings separately. Defaults: 8 occupied, 96 virtual,
384 auxiliary functions, approximately 60,000 selected triples over four
spin cases; three measured repetitions after a warmup.

Observed on this Mac with Accelerate (median seconds, 32-pair tiles):

| Threads | T2 native | T3 native | T2 speedup | T3 speedup |
| --- | --- | --- | --- | --- |
| 1 | 0.517 | 0.592 | 1.00 | 1.00 |
| 2 | 0.269 | 0.300 | 1.92 | 1.97 |
| 4 | 0.244 | 0.263 | 2.12 | 2.25 |
| 8 | 0.236 | 0.249 | 2.19 | 2.37 |

Near-perfect scaling is **not established** beyond two threads. Smaller
8-pair tiles were slower in absolute time and were not retained. T3 Python
connectivity construction adds roughly 0.075 s in this case. Distinguish
that serial setup cost from native scaling; target-machine measurements
are required before claiming near-perfect scalability.

Validation covers independent contraction oracles at 1/4 threads, two
complete updates against frozen-rank legacy equations, RHF/open-shell and
empty sectors, MP/non-MP denominators, the original layout oracle, and
converged water/CH energies against the original spin-by-spin solver.
## Rectangular mixed-spin T2 experiment (2026-11-08)

`devtools/benchmark_vvvv_tiles.py` builds an instrumented copy of the **current
working-tree** mixed-spin T2 routine and `devtools/prototypes/vvvv_rectangular.f90`
into a temporary standalone library. It does not rebuild CCpy or modify the
production kernels. The experiment supports macOS Accelerate and Homebrew OpenBLAS.

The candidate batches both output virtual indices. One DGEMM forms
`Y((f,b),(e,a))` for an `a x b` tile, followed by explicit packing into
`Z((f,e),(b,a))`, an amplitude DGEMM, and disjoint output scatter. Scratch is
`8*(2*va*vb + oa*ob)*a_tile*b_tile` bytes per worker. Tile choices are explicit,
not automatically limited to the production 64 MiB budget; inspect the reported
scratch before increasing dimensions or tile sizes. No four-virtual tensor is
retained across tiles, ranks, or calls. This prototype covers opposite-spin T2;
same-spin triangular packing and selected-T3 connectivity are not implemented.

Run, from the repository root:

```bash
.venv/bin/python devtools/benchmark_vvvv_tiles.py --repeats 5
.venv/bin/python devtools/benchmark_vvvv_tiles.py --virtuals 192 --occupied 12 \
    --naux 768 --tiles 8x8 16x8 --repeats 3
# Small complete independent oracle, partial tiles, and native bounds checking:
.venv/bin/python devtools/benchmark_vvvv_tiles.py --virtuals 13 --occupied 2 \
    --naux 17 --threads 1 4 --repeats 1 --bounds-check
```

Each thread count runs in a fresh process with BLAS restricted through environment
settings. Cases alternate forward/reverse order. Wall times include native scratch
allocation and all packing. Phase counters are **sums of worker seconds**, not
elapsed phase times: do not add them to wall time or interpret them as a parallel
critical-path breakdown. Phase order is integral GEMM, packing/scatter, amplitude
GEMM (including candidate scatter). Allocation, scheduling and barriers are outside
those counters. The current routine scatters in phase two; the candidate packs in
phase two and scatters in phase three.

Every candidate output is compared in full against the current kernel after
warmup and each timed repetition. Three output pairs are also checked independently
against factor-matrix products; small shapes check the entire independent einsum.
Inputs are nonsymmetric, with `ob=oa+1` and `vb=va+3`, exercising unequal dimensions
and partial rectangular tiles. This is kernel validation, not a complete CC(P)
iteration or energy validation.

Initial measurements on an Apple M2 Max (8 performance + 4 efficiency cores),
GNU Fortran and Accelerate, five repetitions, `oa/ob=8/9`, `va/vb=96/99`, `q=384`:

| OpenMP workers | Current mixed-spin T2 | Rectangular 8x8 | Candidate/current speedup |
| --- | --- | --- | --- |
| 1 | 0.381 s | 0.330 s | 1.15 |
| 2 | 0.191 s | 0.163 s | 1.17 |
| 4 | 0.175 s | 0.164 s | 1.07 |
| 8 | 0.157 s | 0.148 s | 1.06 |

The 8x8 candidate uses about 9.3 MiB scratch per worker at this shape. Its
one-to-eight-worker speedup is only 2.23, versus 2.42 for the current kernel.
The best measured eight-worker tile was 32x4 at 0.137 s, but this was a sweep,
not independent confirmation of a universal winning shape. More efficient
one-worker execution does not by itself improve parallel efficiency.

For the current kernel at this shape, integral-GEMM worker-seconds rise from
0.309 at one worker to 0.446 at four workers while total wall time falls from
0.381 to 0.175 s. This localizes a substantial part of the loss inside BLAS
calls rather than residual reduction. It does not identify whether the cause
is memory traffic, backend resources, core placement, or some combination.
The library's BLAS calls run inside the OpenMP region; these counters do not
measure hardware bandwidth or prove that Accelerate uses only CPU cores.

Raw observations are under `output/vvvv_rectangular/`. Production remains
unchanged: modest elapsed-time gains do not establish near-ideal scaling or
justify extending this packing policy to all spin/rank channels yet.

Larger-shape check, three repetitions, `oa/ob=12/13`, `va/vb=192/195`, `q=768`:

| OpenMP workers | Current mixed-spin T2 | Rectangular 8x8 | Rectangular 16x8 |
| --- | --- | --- | --- |
| 1 | 12.841 s | 11.071 s | 10.639 s |
| 2 | 6.144 s | 5.742 s | 5.483 s |
| 4 | 5.664 s | 5.010 s | 4.728 s |
| 8 | 5.037 s | 4.911 s | 4.528 s |

Here 8x8 uses 36.6 MiB per worker and 16x8 uses 73.3 MiB, exceeding the
production budget. The latter reduces eight-worker elapsed time by 10.1%, but
its one-to-eight-worker speedup is only 2.35 (current: 2.55). At eight workers,
current integral-GEMM worker-seconds are 29.36, versus 8.24 at one worker; the
candidate's are 25.40 versus 6.81. This supports investigating the concurrent
GEMM stage before investing in more output scheduling changes. Packing costs
are significant: 16x8's one-worker phases are 6.81 s integral construction,
1.37 s packing, and 2.45 s amplitude contraction/scatter.

The bounds-checked runs passed at virtual dimensions 1/4, 13/16, and 33/36,
with 1 and 4 workers, five candidate tile shapes, nonsymmetric inputs, and
full candidate-versus-current output comparisons. The first two shapes also
passed a complete independent einsum oracle; the 33/36 shape passed the three
independent pair checks. All measured larger/default outputs passed their
full current-kernel comparisons and independent sampled checks.

Next diagnostic: compare concurrent single-thread BLAS calls with larger
BLAS-parallel GEMMs on the target server/backend, recording actual runtime
configuration and phase throughput. The M2 Max/Accelerate plateau alone is
not evidence that this pair decomposition cannot scale on a different CPU.
No same-spin, T3, full-iteration or converged-energy speedup is claimed by
this experiment.

## Cause of the local plateau: backend-dependent compute resources

A follow-up control runs the **same current mixed-spin T2 routine**, with the
same inputs, instrumentation, and scheduling, against Accelerate and Homebrew
OpenBLAS 0.3.32. Three repetitions, 96/99 virtuals, 8/9 occupied, 384 auxiliaries:

| Workers | Accelerate seconds | OpenBLAS seconds | Accelerate speedup | OpenBLAS speedup |
| --- | --- | --- | --- | --- |
| 1 | 0.3863 | 2.0118 | 1.00 | 1.00 |
| 2 | 0.1973 | 1.0308 | 1.96 | 1.95 |
| 4 | 0.1788 | 0.5213 | 2.16 | 3.86 |
| 8 | 0.1621 | 0.3685 | 2.38 | 5.46 |

This isolates a strong backend dependence: changing the BLAS backend restores
close-to-linear scaling through four workers without changing the decomposition.
OpenBLAS is nevertheless slower in absolute elapsed time at eight workers.
Its eight-worker scaling is still sublinear; the remaining loss has not been
attributed to a specific hardware counter or core-placement effect.

The most strongly supported explanation for the early Accelerate plateau is
shared AMX matrix hardware. The AppleAccelerate.jl maintainers independently
report the same 1-to-2-worker doubling and subsequent plateau for Float64 GEMM
on an M2 Max, and attribute it to matrix coprocessors shared per core cluster.
Their CPU OpenBLAS implementation continues scaling across P-cores:
[maintainer benchmarks](https://julialinearalgebra.github.io/AppleAccelerate.jl/dev/benchmarks/).
Our control demonstrates backend-dependent scaling, not direct observation of
AMX instructions or hardware utilization in this particular kernel.

Consequences:

- On this Mac, optimize absolute throughput: tune a small number of concurrent
  DGEMM producers, feed larger cache/memory-bounded tiles, and use CPU workers for
  packing and selected-T3 record contraction. A double-buffered producer/consumer
  pipeline is an experiment to test, not an established speedup. Do not impose
  a global two-worker limit on all CC(P) work; current eight-worker elapsed times
  are still better than two-worker times.
- On a server CPU, retain disjoint output ownership and benchmark a CPU BLAS,
  actual physical-core placement, and single-thread inner calls. The Mac's
  Accelerate speedup ceiling is not a portable measure of this algorithm's
  parallel efficiency.
- Do not replace Accelerate merely to improve a speedup ratio. Compare absolute
  elapsed time and memory at the requested thread count.
- Selected-T3 connectivity construction remains a separate serial cost; it must
  also be reduced for complete-layer scaling. This control covers mixed-spin
  T2 and does not directly identify the fraction of T3 limited by AMX.

Reproduce with `--blas accelerate` and `--blas openblas` on the tile harness;
use `--tiles 8x8 --repeats 3`. OpenBLAS objects compile with GCC OpenMP directives
but link only LLVM libomp, which also supplies OpenBLAS's runtime. Nested teams
are disabled with `OMP_MAX_ACTIVE_LEVELS=1`; `OPENBLAS_NUM_THREADS=1` alone is not
a reliable control for an OpenMP OpenBLAS build. Small full-oracle/bounds checks
and all timed current-versus-candidate and sampled independent checks passed.
