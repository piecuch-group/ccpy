"""Determinant space and signed second-quantized transitions (Secs. 1.1 and 6.3 of
ccpq_gradients_revised.pdf).

Determinants are bit strings over 2 * norb spin orbitals, alpha orbital p -> bit p and beta
orbital p -> bit norb + p. The space holds every determinant with the reference's numbers of
alpha and beta electrons and with the frozen-core orbitals occupied. It is closed under the
spin-free Hamiltonian restricted to it, under every excitation that leaves the core alone,
and under e^{+-T}; every operator projection of the notes therefore lives in it. Operator
strings are applied directly to each determinant, so a matrix element such as
<A|a+_p a+_q a_s a_r|B> with core indices never passes through a determinant outside the
space.

A successful a_p or a+_p multiplies the sign by (-1)^popcount(B & (2^p - 1)) (Eq. 6.9).
"""

from itertools import combinations
import operator

import numpy as np
import scipy.sparse as sp

_ONE = np.uint64(1)


class Transitions:
    """Nonzero matrix elements <dst|O_label|src> = sign of a family of operators O_label."""

    def __init__(self, label, src, dst, sign, nlabel):
        self.label = np.asarray(label, dtype=np.int64)
        self.src = np.asarray(src, dtype=np.int64)
        self.dst = np.asarray(dst, dtype=np.int64)
        self.sign = np.asarray(sign, dtype=float)
        self.nlabel = nlabel

    def matrix(self, coefficients, n):
        """sum_label coefficients[label] * O_label as a sparse (n, n) matrix."""
        data = self.sign * np.asarray(coefficients)[self.label]
        return sp.csr_matrix((data, (self.dst, self.src)), shape=(n, n))

    def contract(self, X):
        """For each label, sum_{AB} X[A, B] <A|O_label|B> = sum sign * X[dst, src]."""
        return np.bincount(self.label, weights=self.sign * X[self.dst, self.src], minlength=self.nlabel)

    def contract_outer(self, x, y):
        """contract() for X = x y^T, without forming X."""
        return np.bincount(self.label, weights=self.sign * x[self.dst] * y[self.src], minlength=self.nlabel)


class DeterminantSpace:
    """All determinants with nalpha / nbeta electrons in norb spatial orbitals whose lowest
    nfrozen orbitals are doubly occupied. Index 0 is the reference."""

    def __init__(self, norb, nalpha, nbeta, nfrozen=0):
        norb, nalpha, nbeta, nfrozen = map(operator.index, (norb, nalpha, nbeta, nfrozen))
        if not (0 <= nfrozen <= min(nalpha, nbeta) and max(nalpha, nbeta) <= norb and norb > 0):
            raise ValueError("Require 0 <= nfrozen <= nalpha, nbeta <= norb with norb > 0")
        if norb > 32:
            raise ValueError("At most 32 spatial orbitals fit in a 64-bit determinant.")
        self.norb, self.nso = norb, 2 * norb
        self.nalpha, self.nbeta, self.nfrozen = nalpha, nbeta, nfrozen
        self.nelectrons = nalpha + nbeta
        core = sum(1 << p for p in range(nfrozen))
        alpha = [core | sum(1 << p for p in occ) for occ in combinations(range(nfrozen, norb), nalpha - nfrozen)]
        beta = [core | sum(1 << p for p in occ) for occ in combinations(range(nfrozen, norb), nbeta - nfrozen)]
        ref = ((1 << nalpha) - 1) | (((1 << nbeta) - 1) << norb)
        dets = [a | (b << norb) for a in alpha for b in beta]
        # reference first, then by excitation level
        dets.sort(key=lambda d: (bin(d & ~ref).count("1"), d))
        self.dets = np.array(dets, dtype=np.uint64)
        self.ndet = len(dets)
        self.reference = np.uint64(ref)
        self.order = np.argsort(self.dets)
        self.sorted_dets = self.dets[self.order]
        self.rank = np.bitwise_count(self.dets & ~self.reference).astype(np.int64)
        # occupation of each spin orbital in the reference
        self.ref_occ = np.array([(ref >> p) & 1 for p in range(self.nso)], dtype=bool)
        self.spin = np.array([0] * norb + [1] * norb)
        self.spatial = np.array(list(range(norb)) * 2)

    # ------------------------------------------------------------------ lookup and apply
    def lookup(self, d):
        """Indices of the bit strings d in the space, and a mask of those found."""
        pos = np.searchsorted(self.sorted_dets, d)
        pos = np.minimum(pos, self.ndet - 1)
        found = self.sorted_dets[pos] == d
        return self.order[pos], found

    def apply(self, ops, src=None, d=None, sign=None):
        """Apply a product of operators, written left to right as in [("+", p), ("-", q)] for
        a+_p a_q (the rightmost acts first), to determinants. Returns (src, dst bit strings,
        sign) of the successful applications."""
        src = np.arange(self.ndet) if src is None else np.asarray(src)
        d = self.dets[src].copy() if d is None else d.copy()
        sign = np.ones(len(src)) if sign is None else sign.copy()
        ok = np.ones(len(src), dtype=bool)
        for kind, p in reversed(ops):
            bit = _ONE << np.uint64(p)
            occupied = (d & bit) != 0
            ok &= occupied if kind == "-" else ~occupied
            sign *= 1.0 - 2.0 * (np.bitwise_count(d & (bit - _ONE)) & 1)
            d ^= bit
        return src[ok], d[ok], sign[ok]

    def transitions_of(self, op_list):
        """Transitions of a list of operator strings (label = position in op_list), keeping
        only images inside the space."""
        label, src, dst, sign = [], [], [], []
        for n, ops in enumerate(op_list):
            s, d, sg = self.apply(ops)
            idx, found = self.lookup(d)
            label.append(np.full(found.sum(), n))
            src.append(s[found])
            dst.append(idx[found])
            sign.append(sg[found])
        cat = (lambda x: np.concatenate(x) if x else np.zeros(0))
        return Transitions(cat(label), cat(src), cat(dst), cat(sign), len(op_list))

    # ------------------------------------------------------------------ excitations
    def holes_particles(self, k):
        """Spin orbitals emptied (holes) and filled (particles) by determinant k relative to
        the reference."""
        d = int(self.dets[k])
        ref = int(self.reference)
        holes = [p for p in range(self.nso) if (ref >> p) & 1 and not (d >> p) & 1]
        parts = [p for p in range(self.nso) if (d >> p) & 1 and not (ref >> p) & 1]
        return holes, parts

    def excitation_string(self, k):
        """E_k as a signed operator string with E_k|Phi> = +|Phi_k>."""
        holes, parts = self.holes_particles(k)
        ops = [("+", a) for a in parts] + [("-", i) for i in reversed(holes)]
        _, d, sign = self.apply(ops, src=np.array([0]))
        assert d[0] == self.dets[k]
        return ops, sign[0]

    def excitation_transitions(self, dets):
        """Transitions of E_I for the determinants I in `dets` (label = position)."""
        strings = [self.excitation_string(k) for k in dets]
        tr = self.transitions_of([ops for ops, _ in strings])
        phase = np.array([ph for _, ph in strings])
        tr.sign = tr.sign * phase[tr.label] if len(strings) else tr.sign
        return tr

    def chi(self, dets):
        """chi^K_p of Eq. (3.5): +1 for holes, -1 for particles, shape (len(dets), nso)."""
        X = np.zeros((len(dets), self.nso))
        for n, k in enumerate(dets):
            holes, parts = self.holes_particles(k)
            X[n, holes] = 1.0
            X[n, parts] = -1.0
        return X

    # ------------------------------------------------------------------ 1- and 2-body operators
    def one_body_transitions(self):
        """a+_p a_q for spin-conserving (p, q); label = p * nso + q."""
        pairs = [(p, q) for p in range(self.nso) for q in range(self.nso) if self.spin[p] == self.spin[q]]
        tr = self.transitions_of([[("+", p), ("-", q)] for p, q in pairs])
        lab = np.array([p * self.nso + q for p, q in pairs])
        return Transitions(lab[tr.label], tr.src, tr.dst, tr.sign, self.nso ** 2)

    def two_body_transitions(self):
        """a+_p a+_q a_s a_r for p < q, r < s, spin conserving; label = ((p*nso+q)*nso+r)*nso+s.
        The pair (r, s) is annihilated first for all determinants, then each (p, q) created."""
        n = self.nso
        label, src, dst, sign = [], [], [], []
        pairs = list(combinations(range(n), 2))
        for r, s in pairs:
            s0, d0, sg0 = self.apply([("-", s), ("-", r)])
            if len(s0) == 0:
                continue
            for p, q in pairs:
                if sorted((self.spin[p], self.spin[q])) != sorted((self.spin[r], self.spin[s])):
                    continue
                s1, d1, sg1 = self.apply([("+", p), ("+", q)], src=s0, d=d0, sign=sg0)
                idx, found = self.lookup(d1)
                if not found.any():
                    continue
                label.append(np.full(found.sum(), ((p * n + q) * n + r) * n + s))
                src.append(s1[found])
                dst.append(idx[found])
                sign.append(sg1[found])
        return Transitions(np.concatenate(label), np.concatenate(src), np.concatenate(dst), np.concatenate(sign),
                           n ** 4)


def spin_orbital_integrals(h, eri):
    """Spin-orbital h[p, q] and v[p, q, r, s] = <pq||rs> from spatial h and chemists' eri[a, b, c, d] =
    (ab|cd), alpha block first."""
    norb = h.shape[0]
    nso = 2 * norb
    spin = np.array([0] * norb + [1] * norb)
    idx = np.array(list(range(norb)) * 2)
    same = (spin[:, None] == spin[None, :]).astype(float)
    h_so = h[np.ix_(idx, idx)] * same
    # <pq|rs> = (pr|qs) delta(s_p, s_r) delta(s_q, s_s)
    g = eri[np.ix_(idx, idx, idx, idx)].transpose(0, 2, 1, 3) * same[:, None, :, None] * same[None, :, None, :]
    v = g - g.transpose(0, 1, 3, 2)
    assert v.shape == (nso,) * 4
    return h_so, v


def hamiltonian_matrix(space, h_so, v_so, one=None, two=None):
    """Electronic H (nuclear repulsion excluded) as a dense matrix over the space (Eq. 1.1)."""
    one = space.one_body_transitions() if one is None else one
    two = space.two_body_transitions() if two is None else two
    n = space.ndet
    H = one.matrix(h_so.ravel(), n) + two.matrix(v_so.ravel(), n)
    return H.toarray()


def fock_diagonal(h_so, v_so, ref_occ):
    """f_pp = h_pp + sum_{i occ} <pi||pi> (Eq. 1.2) in the current orbitals."""
    o = np.flatnonzero(ref_occ)
    return np.diag(h_so) + np.array([v_so[p, o, p, o].sum() for p in range(len(h_so))])
