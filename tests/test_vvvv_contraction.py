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

# Selected T3/T4 slices: independent record-level oracle, including several contributions
# to each determinant, signed targets, nonsymmetric factors, partial tiles, the h1(vv)
# delta terms and T2*T2 placements sharing fiber columns.
for same in (False, True):
    for q in (1, 7):
        for use_h1 in (False, True):
            oa, ob, va, vb = 2, 2 if same else 3, 35, 35 if same else 33
            ra = np.asfortranarray(rng.normal(size=(q, va, va)))
            rb = ra if same else np.asfortranarray(rng.normal(size=(q, vb, vb)))
            if q == 1:  # Isolate T2 dressing with zero Coulomb factors.
                ra.fill(0.)
                rb.fill(0.)
            td = np.asfortranarray(rng.normal(size=(va, vb, oa, ob)))
            h = np.asfortranarray(rng.normal(size=(oa, ob, va, vb)))
            ha = np.asfortranarray(rng.normal(size=(va, va)))
            hb = ha if same else np.asfortranarray(rng.normal(size=(vb, vb)))
            pairs = [(0, 1), (31, 32), (33, 34)] if same else [(0, 0), (31, 32), (34, 32)]
            # sources hit the h1 deltas of their own group's pair
            src = [(1, 2), (0, 5), (31, 32), (31, 7), (33, 34), (4, 34)] if same else \
                  [(1, 0), (0, 5), (31, 32), (31, 7), (34, 32), (4, 32)]
            source = np.asfortranarray(np.array(src).T + 1, dtype=np.int32)
            amps = rng.normal(size=6)
            goff = np.array([1, 3, 5, 7], dtype=np.int32)
            pair = np.array([a*vb+b for a, b in pairs] * 2)
            target = np.array([[1, 2, 3, 3, 1, 2], [1, -1, 1, -2, 2, 2]], dtype=np.int32)
            texc = np.asfortranarray(np.array([[3, 9, 12, 0, 1, 2, 1, 0], [5, 6, 7, 8, 2, 1, 2, 1]],
                                              dtype=np.int32).T)
            te = np.asfortranarray(rng.normal(size=(va, 40)))
            tf = np.asfortranarray(rng.normal(size=(vb, 30)))
            sece = np.asfortranarray(np.array([[0, 5, 2], [20, 4, 2]], dtype=np.int32).T)
            secf = np.asfortranarray(np.array([[0, 6, 2], [12, 3, 3]], dtype=np.int32).T)
            plint = np.asfortranarray(np.array([[1, 5, 6, 1, 2, 5, 6, 1], [2, 6, 5, 2, 3, 6, 5, 2],
                                                [3, 5, 5, 1, 1, 6, 6, 2]], dtype=np.int32).T)
            plcoef = rng.normal(size=3)
            ploff = np.array([1, 1, 3, 4], dtype=np.int32)
            tslot = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)
            order = np.argsort(pair, kind='stable')
            target = np.asfortranarray(target[:, order])
            tslot = tslot[order]
            poff = (np.r_[0, np.cumsum(np.bincount(pair, minlength=va*vb))]+1).astype(np.int32)

            expected = np.zeros(2)
            for rec, pr in enumerate(pair[order]):
                a, b = divmod(pr, vb)
                group, signed = target[:, rec]
                d = abs(signed) - 1
                lam = 1.0 / (np.count_nonzero(texc[:4, d]) - 1)
                W = ra[:, :, a].T @ rb[:, :, b]          # W[e, f] = (ae|bf)
                if same:
                    W = W - W.T
                W = W + (0.5 if same else 1.) * np.einsum('ij,ijef->ef', td[a, b], h)
                val = 0.0
                for isrc in range(goff[group-1]-1, goff[group]-1):
                    e, f = source[:, isrc]-1
                    hv = 0.0
                    if use_h1:
                        hv += ha[a, e] * (f == b) + hb[b, f] * (e == a)
                        if same:
                            hv -= ha[a, f] * (e == b) + hb[b, e] * (f == a)
                    val += (W[e, f] + lam * hv) * amps[isrc]
                val *= np.sign(signed)
                D = texc[:, d]
                for pl in range(ploff[tslot[rec]-1]-1, ploff[tslot[rec]]-1):
                    r, x, y, es, s_, z, w, fs = plint[:, pl]
                    ce = sece[0, es-1] + D[r-1] + sece[1, es-1]*((D[x-1]-1) + sece[2, es-1]*(D[y-1]-1))
                    cf = secf[0, fs-1] + D[s_-1] + secf[1, fs-1]*((D[z-1]-1) + secf[2, fs-1]*(D[w-1]-1))
                    val += plcoef[pl] * te[:, ce-1] @ W @ tf[:, cf-1]
                expected[d] += val
            args = (ra, rb, td, h, ha, hb, source, amps, goff, target, poff, tslot, texc,
                    plint, plcoef, ploff, sece, secf, te, tf, same, use_h1)
            actual = vc.vvvv_tn(*args)
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-11)
            np.testing.assert_array_equal(actual, vc.vvvv_tn(*args))
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
