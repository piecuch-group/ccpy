"""Selection of the P and Q determinant lists (Sec. 9.1, step 1).

P and Q are lists of determinant indices of a DeterminantSpace; the reference is never in
either. They are fixed once and reused at every geometry and every displacement, so the
energy and all of its derivatives refer to the same method.

    ranks(space, (1, 2))                  CCSD
    ranks(space, (1, 2, 3))               CCSDT
    union(ranks(space, (1, 2)), t3)       CC(P) with P = SD + selected triples
    complement(space, P, (3,))            Q = the omitted triples, CC(P;3)-type corrections
"""

import numpy as np
import operator


def validate_indices(space, indices, name):
    """Copy a unique, one-dimensional list of excited determinant indices."""
    values = np.asarray(indices)
    if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(values <= 0) or np.any(values >= space.ndet):
        raise ValueError(f"{name} must exclude the reference and lie in 1..ndet-1")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{name} contains duplicate determinants")
    return values.astype(np.int64, copy=True)


def ranks(space, levels):
    """All determinants whose excitation level relative to the reference is in `levels`."""
    return np.flatnonzero(np.isin(space.rank, list(levels)))


def union(*lists):
    return np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in lists]))


def complement(space, P, levels=None):
    """Determinants outside P and the reference, optionally restricted to excitation levels."""
    pool = np.arange(1, space.ndet) if levels is None else ranks(space, levels)
    return np.setdiff1d(pool, P)


def active_excitations(space, levels, occupied, virtual):
    """Excitations with every hole/particle in the given spatial active sets.

    Indices are zero-based canonical spatial orbital indices (frozen orbitals
    included in the numbering). Both spins are included. This implements a
    closed-shell CAS-style selection of triples, quadruples, or other ranks;
    union with all singles/doubles to build a selected coupled-cluster P.
    """
    if space.nalpha != space.nbeta:
        raise ValueError("Spatial active selections currently require a closed-shell reference")
    occupied = [operator.index(p) for p in occupied]
    virtual = [operator.index(p) for p in virtual]
    if any(p < space.nfrozen or p >= space.nalpha for p in occupied):
        raise ValueError("Active occupied indices must be correlated occupied spatial orbitals")
    if any(p < space.nalpha or p >= space.norb for p in virtual):
        raise ValueError("Active virtual indices must be virtual spatial orbitals")
    holes = np.uint64(sum(1 << p for p in set(occupied)) | sum(1 << (space.norb+p) for p in set(occupied)))
    particles = np.uint64(sum(1 << p for p in set(virtual)) | sum(1 << (space.norb+p) for p in set(virtual)))
    valid = ((space.reference & ~space.dets) & ~holes) == 0
    valid &= ((space.dets & ~space.reference) & ~particles) == 0
    return np.flatnonzero(valid & np.isin(space.rank, list(levels)))


def from_occupations(space, occupations):
    """Determinant indices from (alpha spatial occupations, beta spatial occupations) pairs."""
    out = []
    for occ_a, occ_b in occupations:
        d = sum(1 << p for p in occ_a) | sum(1 << (space.norb + p) for p in occ_b)
        idx, found = space.lookup(np.array([d], dtype=np.uint64))
        if not found[0]:
            raise ValueError(f"Determinant {occ_a}, {occ_b} is not in the space.")
        out.append(idx[0])
    return np.array(out, dtype=np.int64)


def from_t3_excitations(space, excitations):
    """Triples of a ccpy P-space dictionary {"aaa", "aab", "abb", "bbb": rows [a, b, c, i, j, k]}
    (1-based; i, j, k over the correlated occupied orbitals, a, b, c over the virtuals, as made
    by ccpy.utilities.pspace) as determinant indices."""
    nfz, norb = space.nfrozen, space.norb
    nocc = {"a": space.nalpha, "b": space.nbeta}
    out = []
    for case, rows in excitations.items():
        rows = np.atleast_2d(np.asarray(rows, dtype=int))
        spins = case
        for a, b, c, i, j, k in rows:
            if a == b == c == i == j == k == 1:
                continue    # ccpy's placeholder for an empty spin case; never a valid triple
            occ = {"a": set(range(nocc["a"])), "b": set(range(nocc["b"]))}
            for s, o in zip(spins, (i, j, k)):
                occ[s].discard(nfz + o - 1)
            for s, v in zip(spins, (a, b, c)):
                occ[s].add(nocc[s] + v - 1)
            out.append((sorted(occ["a"]), sorted(occ["b"])))
    if not out:
        return np.zeros(0, dtype=np.int64)
    return np.unique(from_occupations(space, out))
