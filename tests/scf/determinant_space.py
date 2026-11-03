"""Exact determinant-space reference for the response theory of CR-CC(2,3)_A.

For a tiny system (a few hundred determinants) every operator is an explicit matrix in the
N-electron determinant space: Hbar = e^{-T} H e^{T}, the CCSD Jacobian, the moments
M_K = <K|Hbar|0>, the left projections N_K = <0|(1+Lambda) Hbar|K>, and all derivatives of
the CR-CC(2,3)_A Lagrangian (CRCC23 Gradient Method.pdf):

    delta = sum_K M_K N_K / D_K,         D_K = f_ii + f_jj + f_kk - f_aa - f_bb - f_cc
    s_t   = d delta / d t,               s_lambda = d delta / d lambda          (Eq. 7)
    A w   = -s_lambda,                   A^T z = -s_t - B^T w                    (Eqs. 4, 5)
    d delta / dx = delta_x + z.R_x + w.g_x                                       (Eq. 6)

Everything is formed by brute force so that the factorized spin-orbital contractions of the
production code can be checked term by term. It is a validation tool, not a method.

Spin orbitals are ordered [occupied | virtual] as in tests/scf/crcc23_triples.py, and
the Hamiltonian is the normal-ordered H_N = sum f {p+ q} + 1/4 sum <pq||rs> {p+ q+ s r}.
Excitation coordinates are the unique amplitudes t_i^a and t_ij^ab (i<j, a<b).
"""

from itertools import combinations

import numpy as np


class DeterminantSpace:
    """All determinants with nocc electrons in nocc + nvir spin orbitals, and second-
    quantized operators as dense matrices."""

    def __init__(self, nocc, nvir):
        self.nocc, self.nvir = nocc, nvir
        self.norb = nocc + nvir
        self.dets = [sum(1 << p for p in occ) for occ in combinations(range(self.norb), nocc)]
        self.index = {d: n for n, d in enumerate(self.dets)}
        self.ndet = len(self.dets)
        self.ref = np.zeros(self.ndet)
        self.ref[self.index[(1 << nocc) - 1]] = 1.0
        o, v = range(nocc), range(nocc, self.norb)
        self.singles = [(a, i) for i in o for a in v]
        self.doubles = [(a, b, i, j) for i, j in combinations(o, 2) for a, b in combinations(v, 2)]
        self.triples = [(a, b, c, i, j, k) for i, j, k in combinations(o, 3)
                        for a, b, c in combinations(v, 3)]

    # strings of creation/annihilation operators, applied right to left
    def string(self, ops):
        """Matrix of a product of operators, e.g. [("+", a), ("+", b), ("-", j), ("-", i)]
        for a+ b+ j i (the rightmost acts first)."""
        X = np.zeros((self.ndet, self.ndet))
        for col, det in enumerate(self.dets):
            sign, d = 1, det
            for kind, p in reversed(ops):
                occupied = (d >> p) & 1
                if (kind == "-") != bool(occupied):
                    break
                sign *= (-1) ** bin(d & ((1 << p) - 1)).count("1")
                d ^= 1 << p
            else:
                X[self.index[d], col] += sign
        return X

    def excitation(self, mu):
        """tau_mu for a single (a, i), double (a, b, i, j) or triple (a, b, c, i, j, k):
        a+ b+ c+ k j i ordering, so tau|0> = |ijk...abc> with ccpy's sign convention."""
        half = len(mu) // 2
        vir, occ = mu[:half], mu[half:]
        return self.string([("+", a) for a in vir] + [("-", i) for i in reversed(occ)])

    def hamiltonian(self, f, v):
        """Normal-ordered H_N from the Fock matrix f[p,q] and <pq||rs> v[p,q,r,s]."""
        n, o = self.norb, self.nocc
        # un-normal-ordered one-body part h = f - sum_i <pi||qi>, then subtract <0|H|0>
        h = f - np.einsum("piqi->pq", v[:, :o, :, :o])
        H = np.zeros((self.ndet, self.ndet))
        for p in range(n):
            for q in range(n):
                if abs(h[p, q]) > 0:
                    H += h[p, q] * self.string([("+", p), ("-", q)])
        for p, q in combinations(range(n), 2):
            for r, s in combinations(range(n), 2):
                if abs(v[p, q, r, s]) > 0:
                    H += v[p, q, r, s] * self.string([("+", p), ("+", q), ("-", s), ("-", r)])
        return H - (self.ref @ H @ self.ref) * np.eye(self.ndet)


class CCSDReference:
    """CCSD, Lambda-CCSD and the CR-CC(2,3)_A response quantities in determinant space."""

    def __init__(self, space, f, v):
        self.space, self.f, self.v = space, f, v
        self.H = space.hamiltonian(f, v)
        self.taus = [space.excitation(mu) for mu in space.singles + space.doubles]
        self.kets = np.array([tau @ space.ref for tau in self.taus])          # <mu| rows
        self.triple_kets = np.array([space.excitation(K) @ space.ref for K in space.triples])

    def T(self, t):
        return sum(tk * tau for tk, tau in zip(t, self.taus))

    def hbar(self, t, H=None):
        H = self.H if H is None else H
        T = self.T(t)
        # e^{T} is a finite series because T only excites
        expT, term = np.eye(self.space.ndet), np.eye(self.space.ndet)
        for k in range(1, self.space.nocc + 1):
            term = term @ T / k
            expT = expT + term
        expmT, term = np.eye(self.space.ndet), np.eye(self.space.ndet)
        for k in range(1, self.space.nocc + 1):
            term = -(term @ T) / k
            expmT = expmT + term
        return expmT @ H @ expT

    def residual(self, Hbar):
        return self.kets @ Hbar @ self.space.ref

    def jacobian(self, Hbar):
        """A[mu, nu] = <mu|[Hbar, tau_nu]|0>."""
        ref = self.space.ref
        return np.array([self.kets @ (Hbar @ tau - tau @ Hbar) @ ref for tau in self.taus]).T

    def energy_gradient(self, Hbar):
        """E_t[mu] = <0|[Hbar, tau_mu]|0> = <0|Hbar tau_mu|0>."""
        ref = self.space.ref
        return np.array([ref @ Hbar @ tau @ ref for tau in self.taus])

    def solve_ccsd(self, t=None, H=None, tol=1.0e-13):
        t = np.zeros(len(self.taus)) if t is None else t.copy()
        for _ in range(100):
            Hbar = self.hbar(t, H)
            R = self.residual(Hbar)
            if np.linalg.norm(R) < tol:
                break
            t -= np.linalg.solve(self.jacobian(Hbar), R)
        return t

    def solve_lambda(self, t, H=None):
        Hbar = self.hbar(t, H)
        return np.linalg.solve(self.jacobian(Hbar).T, -self.energy_gradient(Hbar))

    def left(self, lam):
        """<0|(1 + Lambda) as a row vector."""
        return self.space.ref + lam @ self.kets

    def fock_denominators(self, f):
        e = np.diag(f)
        return np.array([e[i] + e[j] + e[k] - e[a] - e[b] - e[c] for a, b, c, i, j, k in self.space.triples])

    def crcc23_A(self, t, lam, H=None, f=None):
        """delta_A and its factors M_K, N_K, D_K."""
        Hbar = self.hbar(t, H)
        M = self.triple_kets @ Hbar @ self.space.ref
        N = self.left(lam) @ Hbar @ self.triple_kets.T
        D = self.fock_denominators(self.f if f is None else f)
        return np.sum(M * N / D), M, N, D

    # ---------------------------------------------------------------------------------
    # response quantities of the CR-CC(2,3)_A Lagrangian
    # ---------------------------------------------------------------------------------
    def correction_derivative(self, t, lam, fV, vV):
        """d delta_A / dx for H(x) = H + x V (V given by its Fock part fV and <pq||rs> vV),
        from Eq. 6 with the w and z solves; returns (d delta/dx, details)."""
        ref = self.space.ref
        Hbar = self.hbar(t)
        delta, M, N, D = self.crcc23_A(t, lam)
        r, l = M / D, N / D
        bra_left = self.left(lam)

        # amplitude sources
        s_t = np.array([l @ (self.triple_kets @ (Hbar @ tau - tau @ Hbar) @ ref)
                        + (bra_left @ (Hbar @ tau - tau @ Hbar) @ self.triple_kets.T) @ r
                        for tau in self.taus])
        s_lam = self.kets @ Hbar @ self.triple_kets.T @ r                       # Eq. 7

        A = self.jacobian(Hbar)
        w = np.linalg.solve(A, -s_lam)                                          # Eq. 4
        W = self.T(w)
        BTw = np.array([bra_left @ ((Hbar @ tau - tau @ Hbar) @ W - W @ (Hbar @ tau - tau @ Hbar)) @ ref
                        for tau in self.taus])                                  # Eq. 8
        z = np.linalg.solve(A.T, -s_t - BTw)                                    # Eq. 5

        # explicit (fixed t, lambda) derivatives with respect to x
        Vbar = self.hbar(t, self.space.hamiltonian(fV, vV))
        M_x = self.triple_kets @ Vbar @ ref
        N_x = bra_left @ Vbar @ self.triple_kets.T
        D_x = self.fock_denominators(fV)
        delta_x = np.sum(l * M_x + r * N_x - l * r * D_x)
        R_x = self.residual(Vbar)
        g_x = np.array([bra_left @ (Vbar @ tau - tau @ Vbar) @ ref for tau in self.taus])
        ddelta = delta_x + z @ R_x + w @ g_x
        return ddelta, {"delta": delta, "s_t": s_t, "s_lambda": s_lam, "w": w, "z": z,
                        "BTw": BTw, "delta_x": delta_x, "M": M, "N": N, "D": D}

    def correction_finite_difference(self, t, fV, vV, h=1.0e-4):
        """d delta_A / dx by central differences with CCSD and Lambda reconverged."""
        values = []
        for x in (-h, h):
            f = self.f + x * fV
            H = self.H + x * self.space.hamiltonian(fV, vV)
            tx = self.solve_ccsd(t, H)
            lx = self.solve_lambda(tx, H)
            values.append(self.crcc23_A(tx, lx, H, f)[0])
        return (values[1] - values[0]) / (2 * h)


def trace_string(space, ops, Pt):
    """tr(O Pt) for O the operator string `ops` (as in DeterminantSpace.string), without
    forming O: sum over columns j of sign_j * Pt[j, target_j]."""
    total = 0.0
    for col, det in enumerate(space.dets):
        sign, d = 1, det
        for kind, p in reversed(ops):
            occupied = (d >> p) & 1
            if (kind == "-") != bool(occupied):
                break
            sign *= (-1) ** bin(d & ((1 << p) - 1)).count("1")
            d ^= 1 << p
        else:
            total += sign * Pt[col, space.index[d]]
    return total


def extra_densities(ref, t, lam, info):
    """Exact normal-ordered densities of the explicit CR-CC(2,3)_A Lagrangian terms
    delta_x + z.R_x + w.g_x (PDF Eqs. 7-9), split into the triples, z and w pieces:

        dL_extra = sum_pq dgamma[p,q] df_pq + 1/4 sum_pqrs dGamma[p,q,r,s] d<pq||rs>,

    with f and <pq||rs> independent (f at fixed <pq||rs> and vice versa), spin orbitals in
    [occupied | virtual] order, dGamma antisymmetric in (p,q) and in (r,s). `info` is the
    second output of CCSDReference.correction_derivative. Returns {piece: (dgamma, dGamma)}.

    Every term is <left| Vbar |right> with Vbar = e^{-T} V_N e^{T}, i.e. tr(V_N Pt) with
    Pt = e^{T} (|right><left|) e^{-T}; the triples piece also has the Fock-denominator term
    -sum_K l_K r_K dD_K (Eq. 9)."""
    ds = ref.space
    n, o = ds.norb, ds.nocc
    r, l = info["M"] / info["D"], info["N"] / info["D"]
    bra = ref.left(lam)
    L3, R3 = l @ ref.triple_kets, r @ ref.triple_kets
    Z = info["z"] @ ref.kets
    W = ref.T(info["w"])
    Wref, braW = W @ ds.ref, bra @ W
    T = ref.T(t)
    expT, expmT, term1, term2 = np.eye(ds.ndet), np.eye(ds.ndet), np.eye(ds.ndet), np.eye(ds.ndet)
    for k in range(1, ds.nocc + 1):
        term1 = term1 @ T / k
        expT = expT + term1
        term2 = -(term2 @ T) / k
        expmT = expmT + term2
    P = {
        "triples": np.outer(ds.ref, L3) + np.outer(R3, bra),
        "z": np.outer(ds.ref, Z),
        "w": np.outer(Wref, bra) - np.outer(ds.ref, braW),
    }
    occ = np.arange(n) < o
    out = {}
    for piece, Pk in P.items():
        Pt = expT @ Pk @ expmT
        assert abs(np.trace(Pt)) < 1.0e-10   # no dependence on the scalar <0|H|0>
        t1 = np.array([[trace_string(ds, [("+", p), ("-", q)], Pt) for q in range(n)] for p in range(n)])
        dgamma = t1.copy()                    # {p+ q} = p+ q - delta_pq n_p; trace(Pt) = 0
        dGamma = np.zeros((n,) * 4)
        for p in range(n):
            for q in range(p + 1, n):
                for rr in range(n):
                    for s in range(rr + 1, n):
                        val = trace_string(ds, [("+", p), ("+", q), ("-", s), ("-", rr)], Pt)
                        # normal ordering at fixed f: H_N(v) has h = -sum_i <pi||qi>
                        val -= ((q == s and occ[q]) * t1[p, rr] - (p == s and occ[p]) * t1[q, rr]
                                - (q == rr and occ[q]) * t1[p, s] + (p == rr and occ[p]) * t1[q, s])
                        dGamma[p, q, rr, s] = dGamma[q, p, s, rr] = val
                        dGamma[q, p, rr, s] = dGamma[p, q, s, rr] = -val
        if piece == "triples":
            # Fock-denominator term (Eq. 9): dD_K = df_ii + df_jj + df_kk - df_aa - df_bb - df_cc
            for lr, (a, b, c, i, j, k) in zip(l * r, ds.triples):
                for m in (i, j, k):
                    dgamma[m, m] -= lr
                for e in (a, b, c):
                    dgamma[e, e] += lr
        out[piece] = (dgamma, dGamma)
    return out


def amplitudes_from_spin_orbital(space, t1, t2):
    """Unique-amplitude vector (singles, then doubles a<b, i<j) from ccpy-layout
    spin-orbital t1[a,i] and t2[a,b,i,j] (virtual indices counted from 0)."""
    o = space.nocc
    x = [t1[a - o, i] for a, i in space.singles]
    x += [t2[a - o, b - o, i, j] for a, b, i, j in space.doubles]
    return np.array(x)
