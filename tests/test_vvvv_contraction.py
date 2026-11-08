"""Independent dense checks of the CD ladder, including unequal spin dimensions.

Use subprocesses because OpenMP initializes its thread count when it loads. Nonsymmetric
factors exercise T1-transformed/right and transposed/left orientations without assuming
RHF or Hermitian virtual factors. See ccpy/lib/meson.build for BLAS/runtime requirements.
"""
import os
import subprocess
import sys

import pytest

CHECK = r"""
import numpy as np
from ccpy.lib.core import vvvv_contraction as vc
rng = np.random.default_rng(0)

# Strided Python inputs also exercise the f2py conversion at the actual callers.
for noa, nob, nua, nub, q in [(2, 3, 5, 9, 11), (3, 2, 33, 35, 7), (1, 1, 1, 1, 3)]:
    ra = rng.standard_normal((q, nua, nua)).transpose(0, 2, 1)
    rb = rng.standard_normal((q, nub, nub)).transpose(0, 2, 1)
    t = rng.standard_normal((nua, nub, noa, nob)).transpose(3, 2, 1, 0)
    expected = np.einsum('xea,xfb,ijfe->ijba', ra, rb, t, optimize=True)
    actual = vc.vvvv_t2(ra, rb, t)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-11)

# Check the raw same-spin contract: only b>a is returned and the e<f sum uses t(:,:,f,e).
# Do not impose antisymmetry on t: that could hide an index/sign/factor error.
for nu in (1, 7, 33):
    no, q = 3, 7
    r = np.asfortranarray(rng.standard_normal((q, nu, nu)))
    t = np.asfortranarray(rng.standard_normal((no, no, nu, nu)))
    ints = np.einsum('xea,xfb->abef', r, r, optimize=True)
    coeff = (ints - ints.swapaxes(2, 3)) * np.triu(np.ones((nu, nu)), 1)
    expected = np.einsum('abef,ijfe->ijba', coeff, t, optimize=True)
    expected *= np.tril(np.ones((nu, nu)), -1)
    np.testing.assert_allclose(vc.vvvv_t2_sym(r, t), expected, rtol=2e-12, atol=2e-11)
# Cross the workspace-limited tile boundary without allocating an O(v^4) oracle.
# Check selected output pairs independently, including diagonal/last/partial tiles.
noa, nob, nua, nub, q = 2, 3, 205, 211, 3
ra = np.asfortranarray(rng.standard_normal((q, nua, nua)))
rb = np.asfortranarray(rng.standard_normal((q, nub, nub)))
t = np.asfortranarray(rng.standard_normal((nob, noa, nub, nua)))
g = vc.vvvv_t2(ra, rb, t)
for a, b in [(0, 0), (194, 199), (204, 210), (199, 3)]:
    coeff = rb[:, :, b].T @ ra[:, :, a]
    expected = np.einsum('fe,ijfe->ij', coeff, t, optimize=True)
    np.testing.assert_allclose(g[:, :, b, a], expected, rtol=2e-12, atol=2e-10)
no, nu, q = 2, 205, 3
r = ra
t = np.asfortranarray(rng.standard_normal((no, no, nu, nu)))
g = vc.vvvv_t2_sym(r, t)
for a, b in [(0, 1), (0, 204), (132, 133), (133, 204), (203, 204)]:
    coeff = r[:, :, a].T @ r[:, :, b]
    coeff = np.triu(coeff - coeff.T, 1)
    expected = np.einsum('ef,ijfe->ij', coeff, t, optimize=True)
    np.testing.assert_allclose(g[:, :, b, a], expected, rtol=2e-12, atol=2e-10)
i, j = np.triu_indices(nu)
assert np.count_nonzero(g[:, :, i, j]) == 0

# Selected T3: independent record-level oracle, including three contributions
# to each determinant, signed targets, nonsymmetric factors, and partial tiles.
for same in (False, True):
    for q in (1, 7):
        oa, ob, va, vb = 2, 2 if same else 3, 35, 35 if same else 33
        ra = np.asfortranarray(rng.normal(size=(q, va, va)))
        rb = ra if same else np.asfortranarray(rng.normal(size=(q, vb, vb)))
        if q == 1:  # Isolate T2 dressing with zero Coulomb factors.
            ra.fill(0.)
            rb.fill(0.)
        td = np.asfortranarray(rng.normal(size=(va, vb, oa, ob)))
        h = np.asfortranarray(rng.normal(size=(oa, ob, va, vb)))
        source = np.asfortranarray(np.vstack((rng.integers(1, va+1, 6),
                                              rng.integers(1, vb+1, 6))), dtype=np.int32)
        amps = rng.normal(size=6)
        goff = np.array([1, 3, 4, 7], dtype=np.int32)
        pairs = [(0, 1), (31, 32), (33, 34)] if same else [(0, 0), (31, 32), (34, 32)]
        pair = np.array([a*vb+b for a, b in pairs] * 2)
        target = np.array([[1, 2, 3, 3, 1, 2], [1, -1, 1, -2, 2, 2]], dtype=np.int32)
        order = np.argsort(pair, kind='stable')
        target = np.asfortranarray(target[:, order])
        poff = (np.r_[0, np.cumsum(np.bincount(pair, minlength=va*vb))]+1).astype(np.int32)
        expected = np.zeros(2)
        for rec, p in enumerate(pair[order]):
            a, b = divmod(p, vb)
            group, signed = target[:, rec]
            for isrc in range(goff[group-1]-1, goff[group]-1):
                e, f = source[:, isrc]-1
                val = ra[:, e, a] @ rb[:, f, b]
                if same:
                    val -= ra[:, f, a] @ rb[:, e, b]
                val += (0.5 if same else 1.) * np.sum(td[a, b] * h[:, :, e, f])
                expected[abs(signed)-1] += np.sign(signed)*val*amps[isrc]
        actual = vc.vvvv_t3(ra, rb, td, h, source, amps, goff, target, poff, same, 2)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-11)
        np.testing.assert_array_equal(actual,
            vc.vvvv_t3(ra, rb, td, h, source, amps, goff, target, poff, same, 2))
print('ok')
"""


@pytest.mark.parametrize("threads", [1, 4])
def test_ladder_kernels_with_threads(threads):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), VECLIB_MAXIMUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    out = subprocess.run([sys.executable, "-c", CHECK], env=env, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")
