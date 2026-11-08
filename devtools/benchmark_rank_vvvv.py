"""Measure rank-local vvvv scaling in fresh processes (no molecular calculation).

Reports the complete Python layers and native-only time, including scratch
allocation but excluding Python connectivity construction from native time.
BLAS is restricted to one thread so only OpenMP parallelizes the pair tiles.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def child(args):
    from types import SimpleNamespace as NS
    import numpy as np
    from ccpy.cc import chol_ppl
    from ccpy.cc.chol_ppl import add_vvvv_t2, add_vvvv_t3
    from ccpy.lib.core import vvvv_contraction as vc

    rng = np.random.default_rng(181)
    o, v, q = args.occupied, args.virtuals, args.naux
    T, dT, H, exc = NS(), NS(), NS(), {}
    factors = [np.asfortranarray(rng.normal(size=(q, v, v))) for _ in range(2)]
    X = NS(chol=NS(**{s: NS(vv=r.transpose(0, 2, 1))
                      for s, r in zip('ab', factors)}))
    for spin in ('aa', 'ab', 'bb'):
        amp = rng.normal(size=(v, v, o, o))
        if spin != 'ab':
            amp -= amp.swapaxes(0, 1).copy()
            amp -= amp.swapaxes(2, 3).copy()
        setattr(T, spin, np.asfortranarray(amp))
        setattr(dT, spin, np.zeros_like(amp))
        setattr(H, spin, NS(oovv=np.asfortranarray(rng.normal(size=(o, o, v, v)))))
    for spin in ('aaa', 'aab', 'abb', 'bbb'):
        rows = np.column_stack((rng.integers(1, v + 1, (args.triples, 3)),
                                rng.integers(1, o + 1, (args.triples, 3))))
        for offset in (0, 3):
            for s in set(spin):
                cols = [offset + p for p, x in enumerate(spin) if x == s]
                rows[:, cols] = np.sort(rows[:, cols], axis=1)
        valid = np.ones(len(rows), dtype=bool)
        for i in range(3):
            for j in range(i + 1, 3):
                if spin[i] == spin[j]:
                    valid &= (rows[:, i] < rows[:, j]) & (rows[:, i+3] < rows[:, j+3])
        exc[spin] = np.asfortranarray(np.unique(rows[valid], axis=0), dtype=np.int32)
        setattr(T, spin, rng.normal(size=len(exc[spin])))
        setattr(dT, spin, np.zeros(len(exc[spin])))

    native_times, native_cpu = [], []
    def timed(function):
        def call(*a, **kw):
            start = time.perf_counter()
            cpu = time.process_time()
            result = function(*a, **kw)
            native_times.append(time.perf_counter() - start)
            native_cpu.append(time.process_time() - cpu)
            return result
        return call
    chol_ppl.vvvv_contraction = NS(**{name: timed(getattr(vc, name))
        for name in ('vvvv_t2', 'vvvv_t2_sym', 'vvvv_t3')})
    results = {}
    for rank, layer in [('t2', lambda: add_vvvv_t2(T, dT, X, False)),
                        ('t3', lambda: add_vvvv_t3(T, dT, X, H, exc, False))]:
        layer()
        walls, kernels, cpus = [], [], []
        for _ in range(args.repeats):
            native_times.clear()
            native_cpu.clear()
            start = time.perf_counter()
            layer()
            walls.append(time.perf_counter() - start)
            kernels.append(sum(native_times))
            cpus.append(sum(native_cpu))
        results[rank] = {'layer_seconds': float(np.median(walls)),
                         'native_seconds': float(np.median(kernels)),
                         'native_cpu_seconds': float(np.median(cpus))}
    print(json.dumps({'threads': int(os.environ['OMP_NUM_THREADS']),
        'dimensions': {'o': o, 'v': v, 'q': q},
        'selected': {s: len(e) for s, e in exc.items()}, 'timings': results}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--virtuals', type=int, default=96)
    parser.add_argument('--occupied', type=int, default=8)
    parser.add_argument('--naux', type=int, default=384)
    parser.add_argument('--triples', type=int, default=20000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if min(args.virtuals, args.occupied) < 3 or min(args.naux, args.triples, args.repeats, *args.threads) < 1:
        parser.error('Need at least three orbitals and positive dimensions/repeats/threads')
    if args.child:
        child(args)
        return
    baseline = None
    for threads in args.threads:
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
                   VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        output = subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
            '--child', '--virtuals', str(args.virtuals), '--occupied', str(args.occupied),
            '--naux', str(args.naux), '--triples', str(args.triples),
            '--repeats', str(args.repeats)], env=env, text=True, timeout=120)
        row = json.loads(output.strip().splitlines()[-1])
        if threads == 1:
            baseline = row['timings']
        if baseline is not None:
            for rank, timings in row['timings'].items():
                for region in ('layer', 'native'):
                    speedup = baseline[rank][region + '_seconds'] / timings[region + '_seconds']
                    timings[region + '_speedup'] = speedup
                    timings[region + '_efficiency'] = speedup / threads
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()