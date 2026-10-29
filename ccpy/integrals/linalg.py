"""Linear-algebra helpers (orthogonalization, metric inverse square roots, blocked
eigensolver, pivoted Cholesky) used by the integral and SCF code.

Adapted from forte2 (https://github.com/evangelistalab/forte2, LGPL-3.0); see
THIRD_PARTY_NOTICES.md.
"""

import numpy as np
import scipy as sp

MACHEPS = 1e-14


def _eigh_metric_kernel(S, rtol=1e-7):
    info = {}
    sevals, sevecs = np.linalg.eigh(S)
    if np.any(sevals < -MACHEPS):
        raise ValueError("The metric matrix must be positive semi-definite.")
    sevals[sevals < 0] = 0.0
    max_seval = sevals[-1]
    info["max_eigenvalue"] = max_seval
    info["min_eigenvalue"] = sevals[0]
    info["condition_number"] = max_seval / sevals[0] if sevals[0] > 0 else np.inf
    ndiscard = np.searchsorted(sevals, rtol * max_seval)
    info["n_discarded"] = ndiscard
    info["n_kept"] = len(sevals) - ndiscard
    return sevals, sevecs, info


def canonical_orth(S, rtol=1e-7):
    """Canonical orthogonalization X = U s^{-1/2} of the metric S, dropping eigenvalues
    below rtol * max(s). Returns X, its left inverse, and an info dict."""
    sevals, sevecs, info = _eigh_metric_kernel(S, rtol=rtol)
    ndiscard = info["n_discarded"]
    U = sevecs[:, ndiscard:]
    X = U / np.sqrt(sevals[ndiscard:])
    Xm1 = np.sqrt(sevals[ndiscard:])[:, None] * U.T.conj()
    return X, Xm1, info


def invsqrt_matrix(M, rtol=1e-7):
    """Inverse square root (and square root) of a positive semi-definite matrix."""
    evals, evecs, info = _eigh_metric_kernel(M, rtol=rtol)
    ndiscard = info["n_discarded"]
    evecs = evecs[:, ndiscard:]
    evals = evals[ndiscard:]
    return (evecs / np.sqrt(evals)) @ evecs.T.conj(), (evecs * np.sqrt(evals)) @ evecs.T.conj(), info


def block_eigh(A, block_idx, atol=1e-8, rtol=1e-8, sort=False):
    """Diagonalize a Hermitian matrix whose blocks (e.g., irreps) are not coupled.
    Returns eigenvalues, eigenvectors, and the block index of each eigenvector."""
    block_idx = np.asarray(block_idx)
    if len(set(block_idx)) == 1:
        return *(np.linalg.eigh(A)), block_idx

    coupling = np.abs(A[block_idx[:, None] != block_idx[None, :]]).max(initial=0.0)
    tol = atol + rtol * np.abs(A).max(initial=0.0)
    if coupling > tol:
        raise ValueError(
            f"Elements coupling different blocks reach {coupling:.2e}, above the "
            f"tolerance {tol:.2e}."
        )
    eigvals = np.empty(len(block_idx))
    eigvecs = np.zeros(A.shape, dtype=A.dtype)
    for block in np.unique(block_idx):
        idx = np.flatnonzero(block_idx == block)
        eigvals[idx], eigvecs[np.ix_(idx, idx)] = np.linalg.eigh(A[np.ix_(idx, idx)])
    if not sort:
        return eigvals, eigvecs, block_idx
    order = np.argsort(eigvals, kind="stable")
    return eigvals[order], eigvecs[:, order], block_idx[order]


def pivoted_cholesky(M, tol):
    """Pivoted (incomplete) Cholesky decomposition M ~ L^T L of a positive semi-definite
    matrix. Returns the Cholesky vectors as rows, shape (nvec, n). tol < 0 means complete."""
    n = M.shape[0]
    diag = np.diag(M).copy()
    vecs = []
    while True:
        p = int(np.argmax(diag))
        dmax = diag[p]
        if dmax <= max(tol, 0.0) or len(vecs) == n:
            break
        L = M[:, p].copy()
        for v in vecs:
            L -= v * v[p]
        L /= np.sqrt(dmax)
        vecs.append(L)
        diag -= L**2
        diag[diag < 0.0] = 0.0
    return np.array(vecs)


def cholesky_metric_inverse_sqrt(M):
    """Return L^{-1} with M = L L^T. This is the (non-symmetric) M^{-1/2} used to build
    the density-fitting B tensor."""
    try:
        L = sp.linalg.cholesky(M, lower=True)
    except sp.linalg.LinAlgError:
        raise ValueError(
            "Density-fitting Coulomb metric (P|Q) is not positive definite. "
            "Set Molecule(df_metric_tol=...) to a small positive value to orthogonalize it."
        )
    return sp.linalg.solve_triangular(L, np.eye(M.shape[0]), lower=True)
