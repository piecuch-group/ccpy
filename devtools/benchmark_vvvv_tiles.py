#!/usr/bin/env python3
"""Standalone mixed-spin T2 tile experiment; never changes the installed extension.

Compares current source (instrumented in a temporary copy) and rectangular tiles.
Phase counters sum worker seconds; they are NOT wall times or additive wall fractions.
Fresh processes isolate OpenMP thread settings. macOS Accelerate or Homebrew OpenBLAS.
"""
import argparse
import ctypes as ct
import json
import hashlib
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def build(directory, bounds_check=False, blas="accelerate"):
    source = (ROOT / 'ccpy/lib/fortran/vvvv_contraction.f90').read_text()
    routine = source.split('              subroutine vvvv_t2(')[1].split('              end subroutine vvvv_t2')[0]
    routine = 'subroutine baseline(' + routine
    routine = routine.replace('naux)', 'naux, phase)', 1)
    routine = routine.replace('                 ! resid', ' use omp_lib\n                 ! resid', 1)
    routine = routine.replace('                 integer(kind=8), parameter',
        ' real(8), intent(out) :: phase(3)\n real(8) :: start,p1,p2,p3\n                 integer(kind=8), parameter', 1)
    routine = routine.replace('                 !$omp parallel', ' p1=0; p2=0; p3=0\n                 !$omp parallel', 1)
    routine = routine.replace('private(tile,a,b,nb,k,y,work)',
        'private(tile,a,b,nb,k,y,work,start) reduction(+:p1,p2,p3)')
    routine = routine.replace("                    call dgemm('t'", " start=omp_get_wtime()\n                    call dgemm('t'", 1)
    routine = routine.replace("                    call dgemm('n'", " p1=p1+omp_get_wtime()-start\n start=omp_get_wtime()\n                    call dgemm('n'", 1)
    routine = routine.replace('                    do k = 1, nb',
        ' p3=p3+omp_get_wtime()-start\n start=omp_get_wtime()\n                    do k = 1, nb', 1)
    routine = routine.replace('                 end do\n                 !$omp end do',
        ' p2=p2+omp_get_wtime()-start\n                 end do\n                 !$omp end do', 1)
    routine += '\n phase=[p1,p2,p3]\n end subroutine baseline\n'
    path = directory / 'baseline.f90'
    path.write_text(routine)
    lib = directory / 'tiles.so'
    objects = []
    for index, source_path in enumerate([path, ROOT / 'devtools/prototypes/vvvv_rectangular.f90']):
        obj = directory / f'kernel{index}.o'
        subprocess.run(['gfortran', '-O3', '-fPIC', '-c', '-fopenmp',
            '-ffree-line-length-none', *(['-fcheck=bounds'] if bounds_check else []),
            str(source_path), '-o', str(obj)], check=True)
        objects.append(str(obj))
    # Compile OpenMP directives with GCC; link exactly one runtime. LLVM libomp
    # supplies GOMP entry points and is also the Homebrew OpenBLAS runtime.
    link = (['-fopenmp', '-framework', 'Accelerate'] if blas == 'accelerate' else
            ['-L/opt/homebrew/opt/openblas/lib', '-lopenblas',
             '-L/opt/homebrew/opt/libomp/lib', '-lomp'])
    subprocess.run(['gfortran', '-shared', *objects, *link, '-o', str(lib)], check=True)
    return lib


def child(args):
    import numpy as np
    lib = ct.CDLL(args.library)
    blas_config = None
    if args.blas == 'openblas':
        lib.openblas_get_config.restype = ct.c_char_p
        blas_config = {'config': lib.openblas_get_config().decode(),
                       'parallel_model': lib.openblas_get_parallel(),
                       'configured_threads': lib.openblas_get_num_threads()}
    rng = np.random.default_rng(181)
    oa, ob, va, vb, q = args.occupied, args.occupied + 1, args.virtuals, args.virtuals + 3, args.naux
    ra = np.asfortranarray(rng.normal(size=(q, va, va)))
    rb = np.asfortranarray(rng.normal(size=(q, vb, vb)))
    t = np.asfortranarray(rng.normal(size=(ob, oa, vb, va)))
    result = np.empty_like(t)
    phase = np.zeros(3)
    ptr = lambda x: x.ctypes.data_as(ct.POINTER(ct.c_double))
    integers = lambda xs: [ct.byref(ct.c_int(x)) for x in xs]
    common = [ptr(result), ptr(ra), ptr(rb), ptr(t)]
    # Baseline argument order differs from the prototype.
    baseline = common + integers([oa, va, ob, vb, q]) + [ptr(phase)]
    lib.baseline_(*baseline)
    expected = result.copy()
    # Independent sampled oracle also checks the current source, at production shapes.
    for a, b in [(0, 0), (va-1, vb-1), (va//2, vb//2)]:
        coeff = rb[:, :, b].T @ ra[:, :, a]
        oracle = np.einsum('ijfe,fe->ij', t, coeff)
        np.testing.assert_allclose(expected[:, :, b, a], oracle, rtol=3e-11, atol=1e-9)
    if max(va, vb) <= 35:
        oracle = np.einsum('xea,xfb,ijfe->ijba', ra, rb, t, optimize=True)
        np.testing.assert_allclose(expected, oracle, rtol=3e-11, atol=1e-9)
    cases = [('baseline', lib.baseline_, baseline)]
    for tile in args.tiles:
        ba, bb = map(int, tile.split('x'))
        if min(ba, bb) < 1:
            raise ValueError('Tile dimensions must be positive')
        cases.append((tile, lib.rectangular_, common + integers([oa, ob, va, vb, q, ba, bb]) + [ptr(phase)]))
    rows = {name: {'wall': [], 'cpu': [], 'worker_phases': []} for name, _, _ in cases}
    for _, fn, call in cases:
        fn(*call)
        np.testing.assert_allclose(result, expected, rtol=3e-11, atol=1e-9)
    for repeat in range(args.repeats):
        for name, fn, call in (cases if repeat % 2 == 0 else cases[::-1]):
            start, cpu = time.perf_counter(), time.process_time()
            fn(*call)
            elapsed, cpu = time.perf_counter()-start, time.process_time()-cpu
            rows[name]['wall'].append(elapsed)
            rows[name]['cpu'].append(cpu)
            rows[name]['worker_phases'].append(phase.copy().tolist())
            np.testing.assert_allclose(result, expected, rtol=3e-11, atol=1e-9)
    print(json.dumps({'threads': int(os.environ['OMP_NUM_THREADS']),
        'dimensions': [oa, ob, va, vb, q],
        'backend': args.blas, 'openblas_config': blas_config, 'platform': platform.platform(),
        'linked_libraries': subprocess.check_output(['otool', '-L', args.library], text=True).splitlines()[1:],
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT / 'ccpy/lib/fortran/vvvv_contraction.f90',
                                    ROOT / 'devtools/prototypes/vvvv_rectangular.f90']},
        'compiler': subprocess.check_output(['gfortran', '--version'], text=True).splitlines()[0],
        'thread_environment': {k: os.environ.get(k) for k in
            ['OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OMP_MAX_ACTIVE_LEVELS', 'VECLIB_MAXIMUM_THREADS',
             'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
        'phase_order': ['integral_gemm', 'packing_or_scatter', 'amplitude_gemm_and_prototype_scatter'],
        'scratch_bytes_per_worker': {tile: 8*(2*va*vb+oa*ob)*int(tile.split('x')[0])*int(tile.split('x')[1])
                                     for tile in args.tiles},
        'results': rows}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--virtuals', type=int, default=96)
    p.add_argument('--occupied', type=int, default=8)
    p.add_argument('--naux', type=int, default=384)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--tiles', nargs='+', default=['8x4', '8x8', '16x4', '16x8', '32x4'])
    p.add_argument('--blas', choices=['accelerate', 'openblas'], default='accelerate')
    p.add_argument('--bounds-check', action='store_true', help='Compile with Fortran bounds checks (validation only)')
    p.add_argument('--library', help=argparse.SUPPRESS)
    args = p.parse_args()
    if min(args.virtuals, args.occupied, args.naux, args.repeats, *args.threads) < 1:
        p.error('Dimensions, repetitions and thread counts must be positive')
    if args.library:
        child(args)
        return
    if sys.platform != 'darwin':
        p.error('This experiment currently supports macOS only')
    with tempfile.TemporaryDirectory(prefix='ccpy-tiles-') as temp:
        library = build(Path(temp), args.bounds_check, args.blas)
        for threads in args.threads:
            env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
                OMP_MAX_ACTIVE_LEVELS='1',
                VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--library', str(library),
                '--virtuals', str(args.virtuals), '--occupied', str(args.occupied), '--naux', str(args.naux),
                '--repeats', str(args.repeats), '--blas', args.blas, '--tiles', *args.tiles], env=env, check=True)


if __name__ == '__main__':
    main()
