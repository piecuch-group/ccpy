"""Optional external FCIpy kernels with CCpy determinant/integral conventions.

Build the connector using devtools/build_fcipy_oracle.py. No FCIpy source is
bundled. The extension path is explicit or supplied via CCPY_FCIPY_EXTENSION.
"""
import importlib.util
import os
from math import comb
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from .densities import SpatialDensities


class FCIpyOperator:
    """Bare Hermitian Hamiltonian action and arbitrary left/right transition RDMs.

    Real molecular integrals are assumed. Complex vectors are handled by
    separate real and imaginary actions, preserving complex-step derivatives.
    """
    def __init__(self, space, h, eri, extension=None, sparse_cache=False):
        if space.nalpha != space.nbeta or space.norb > 31:
            raise ValueError("FCIpy connector currently requires equal alpha/beta counts and norb <= 31")
        path = extension or os.environ.get("CCPY_FCIPY_EXTENSION")
        if not path:
            raise ImportError("Build devtools/build_fcipy_oracle.py and set CCPY_FCIPY_EXTENSION")
        path = Path(path).expanduser().resolve()
        spec = importlib.util.spec_from_file_location("_ccpq_fcipy", path)
        self.lib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.lib)
        self.space = space
        self._validate_integrals(h, eri)
        self.h = np.array(h, order='F', dtype=float, copy=True)
        self.v = np.array(eri.transpose(0, 2, 1, 3), order='F', dtype=float, copy=True)
        mask = (1 << space.norb) - 1
        self.det = np.empty((1, 2, space.ndet), dtype=np.int64, order="F")
        self.det[0, 0] = space.dets & np.uint64(mask)
        self.det[0, 1] = space.dets >> np.uint64(space.norb)
        self.keys = space.sorted_dets.astype(np.int64)
        self.order = (space.order + 1).astype(np.int32)
        self.calls = 0
        self.sparse = None
        if sparse_cache:
            no, nv = space.nalpha - space.nfrozen, space.norb - space.nalpha
            nconn = 1 + 2*no*nv + 2*comb(no, 2)*comb(nv, 2) + (no*nv)**2
            if nconn * space.ndet >= np.iinfo(np.int32).max:
                raise ValueError("Sparse Hamiltonian cache exceeds 32-bit connector indexing")
            indices, values = self.lib.ccpq_fcipy.connected_elements(
                self.det, self.keys, self.order, self.h, self.v, space.nalpha,
                space.nfrozen, nconn)
            if np.min(indices) < 1:
                raise RuntimeError("Incomplete connected Hamiltonian graph")
            indices -= 1
            indptr = np.arange(0, space.ndet*nconn + 1, nconn, dtype=np.int32)
            self.sparse = sp.csr_matrix((values.ravel(order='F'), indices.ravel(order='F'), indptr),
                                        shape=(space.ndet, space.ndet), copy=False)
            self.sparse.sort_indices()
            # Keep structurally connected zero-valued edges for derivative RDMs.

    def _validate_integrals(self, h, eri):
        n = self.space.norb
        if h.shape != (n, n) or eri.shape != (n,) * 4:
            raise ValueError('Integrals must cover the full spatial orbital space')
        if np.iscomplexobj(h) or np.iscomplexobj(eri):
            raise ValueError('The FCIpy connector requires real spatial integrals')
        if not np.allclose(h, h.T, atol=1e-12, rtol=0.):
            raise ValueError("FCIpy connector requires a real Hermitian bare Hamiltonian")
        for permutation in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
            if not np.allclose(eri, eri.transpose(permutation), atol=1e-12, rtol=0.):
                raise ValueError("eri must have the permutation symmetries of physical chemists' integrals")

    def update_integrals(self, h, eri):
        """Refresh numeric CSR values without rebuilding/sorting its fixed graph.

        Return False for older connector builds, allowing the backend to
        rebuild instead. Structural zeros are retained for derivative RDMs.
        """
        if self.sparse is None or not hasattr(self.lib.ccpq_fcipy, 'update_connected_values'):
            return False
        self._validate_integrals(h, eri)
        self.h = np.array(h, order='F', dtype=float, copy=True)
        self.v = np.array(eri.transpose(0, 2, 1, 3), order='F', dtype=float, copy=True)
        H = self.sparse
        self.lib.ccpq_fcipy.update_connected_values(self.det, H.indptr, H.indices,
            self.h, self.v, self.space.nalpha, H.data)
        return True

    def _real_action(self, x):
        self.calls += 1
        return self.lib.ccpq_fcipy.connected_sigma(
            self.det, self.keys, self.order, self.h, self.v, np.ascontiguousarray(x),
            self.space.nalpha, self.space.nfrozen)

    def __call__(self, x):
        x = np.asarray(x)
        if self.sparse is not None:
            self.calls += 1
            return self.sparse @ x
        if np.iscomplexobj(x):
            return self._real_action(x.real) + 1j * self._real_action(x.imag)
        return self._real_action(x)

    def transition_density(self, left, right):
        """Spatial RDMs of left^T O right; prune exact zeros only.

        FCIpy scans determinant pairs for RDMs, so dense-support factors can be
        slow. This allocates no determinant-pair matrix.
        """
        if np.iscomplexobj(left) or np.iscomplexobj(right):
            raise ValueError("Transition densities require real stationary factors")
        if self.sparse is not None:
            H = self.sparse
            D, G = self.lib.ccpq_fcipy.graph_density(self.det, H.indptr, H.indices,
                np.ascontiguousarray(left), np.ascontiguousarray(right), self.space.norb, self.space.nalpha)
            return SpatialDensities(D, G)
        active = np.flatnonzero((left != 0) | (right != 0))
        n = self.space.norb
        if not len(active):
            return SpatialDensities(np.zeros((n, n)), np.zeros((n,) * 4))
        det = np.asfortranarray(self.det[:, :, active])
        l, r = np.ascontiguousarray(left[active]), np.ascontiguousarray(right[active])
        na = self.space.nalpha
        a, b = self.lib.ci.compute_rdm1s(det, r, l, n, na, na)
        aa, ab, bb = self.lib.ci.compute_rdm2s(det, r, l, n, na, na)
        # FCIpy's 1-RDM stores annihilator first, unlike its 2-RDM.
        return SpatialDensities((a + b).T, aa + ab + ab.transpose(1, 0, 3, 2) + bb)

    def coefficient_density(self, coefficient_transpose):
        """RDMs of a sparse determinant-pair coefficient (stored transposed)."""
        C = coefficient_transpose.tocoo()
        D, G = self.lib.ccpq_fcipy.sparse_density(self.det, C.col.astype(np.int32),
            C.row.astype(np.int32), C.data, self.space.norb, self.space.nalpha)
        return SpatialDensities(D, G)
