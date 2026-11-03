'''
Amplitude-dependent Jacobi denominators for CC iterations.

P. Piecuch and L. Adamowicz, J. Chem. Phys. 100, 5857 (1994). The Jacobi step is
t <- t - R(t)/D, and D is the diagonal of the part of the CC equations kept on the left-hand
side:
  "mp"              D_MP, orbital-energy differences (the usual choice);
  "epstein-nesbet"  D_L = D_MP + diagonal of the bare two-electron linear terms, i.e. the
                    diagonal of the CI matrix (case 1 of the paper);
  "quasilinear"     D_L + D_NL(T2), where the nonlinear terms T2*T_K of the equations
                    projected on K-fold excitations are quasilinearized with T2 held at its
                    current value (case 2, the best choice of the paper). T2*T2 is linearized
                    symmetrically, 1/2*(T2 U2 + U2 T2), so the doubles receive half of the
                    diagonal of the T2*T2 Jacobian;
  "newton"          D_L + the full diagonal of the T2*T_K Jacobian, i.e. the diagonal of the
                    exact Jacobian without its T1-dependent terms. Differs from "quasilinear"
                    only by the factor 1/2 on the doubles, and converged faster than it for
                    stretched H2O.
In quasidegenerate situations the CI diagonal of the dominant excitations becomes small and
the T2 terms restore it, a natural nonuniform analog of a uniform level shift. Only the
preconditioner changes: the residual, and hence the converged amplitudes, are the same.

All of these diagonals are sums of terms that depend on one or two orbital indices of the
excitation (three for the exchange-like T2 terms), so they are assembled from the same
O(o^2 v^2) pieces for any excitation rank, either for full amplitude arrays or for a list of
excitations (CC(P)). The update kernels take the usual MP step R/(D_MP - shift);
Denominators.correct rescales it to R/(D - shift) afterwards.

In ccpy's convention the Jacobi denominators are e_occ - e_virt < 0, so D = D_MP - J, with J
the corrections to dR_mu/dt_mu.
'''

from itertools import combinations, product

import numpy as np

DENOMINATORS = ("mp", "epstein-nesbet", "quasilinear", "newton")

# Corrected denominators may not fall below this fraction of the MP denominators in magnitude
MIN_DENOMINATOR_FRACTION = 0.5


class Denominators:
    """Jacobi denominators of the given kind at the current T2, for blocks named by the spins
    of their indices ("a", "ab", "aab", ...; same spins for virtual and occupied indices)."""

    def __init__(self, T, H, kind):
        if kind not in DENOMINATORS[1:]:
            raise ValueError(f"denominator must be one of {DENOMINATORS[1:]}")
        self.kind = kind
        self.eps_o = {"a": np.diagonal(H.a.oo), "b": np.diagonal(H.b.oo)}
        self.eps_v = {"a": np.diagonal(H.a.vv), "b": np.diagonal(H.b.vv)}
        self.en = _epstein_nesbet_pieces(H)
        self.t2 = _t2_pieces(T, H) if kind in ("quasilinear", "newton") else None

    def correction(self, spins, excitations=None):
        """J for a block: full array, or one value per row of `excitations` (CC(P) lists,
        1-based rows [virtuals..., occupied...])."""
        J = self._assemble(spins, self.en, excitations)
        if self.t2 is not None:
            weight = 0.5 if (self.kind == "quasilinear" and len(spins) == 2) else 1.0
            J = J + weight * self._assemble(spins, self.t2, excitations)
        return J

    def mp(self, spins, excitations=None):
        n = len(spins)
        terms = [(-1.0, self.eps_v[s], (p,)) for p, s in enumerate(spins)]
        terms += [(1.0, self.eps_o[s], (n + q,)) for q, s in enumerate(spins)]
        return sum(c * _take(x, axes, n, excitations) for c, x, axes in terms)

    def correct(self, T, dT, spins, shift, excitations=None):
        """Rescale the MP step R/(D_MP - shift) just taken for block `spins` to the corrected
        denominators."""
        d_mp = self.mp(spins, excitations)
        d = np.minimum(d_mp - self.correction(spins, excitations), MIN_DENOMINATOR_FRACTION * d_mp) - shift
        step = getattr(dT, spins)
        new = step * ((d_mp - shift) / d)
        setattr(T, spins, getattr(T, spins) + (new - step))
        setattr(dT, spins, new)
        return T, dT

    @staticmethod
    def _assemble(spins, pieces, excitations):
        """Sum of the one-, two-, and three-index pieces over all index combinations of the
        excitation (virtual axes 0..n-1, occupied axes n..2n-1)."""
        n = len(spins)
        V = list(enumerate(spins))
        O = [(n + q, s) for q, s in enumerate(spins)]
        terms = []
        for (p, sp) in V:
            terms.append((pieces.get(("v", sp)), (p,)))
        for (q, sq) in O:
            terms.append((pieces.get(("o", sq)), (q,)))
        for (p, sp), (q, sq) in product(V, O):
            terms.append((pieces.get(("vo", sp, sq)), (p, q)))
        for (p, sp), (p2, sp2) in combinations(V, 2):
            terms.append((pieces.get(("vv", sp, sp2)), (p, p2)))
        for (q, sq), (q2, sq2) in combinations(O, 2):
            terms.append((pieces.get(("oo", sq, sq2)), (q, q2)))
        for ((p, sp), (p2, sp2)), (q, sq) in product(combinations(V, 2), O):
            terms.append((pieces.get(("vvo", sp, sp2, sq)), (p, p2, q)))
        for (p, sp), ((q, sq), (q2, sq2)) in product(V, combinations(O, 2)):
            terms.append((pieces.get(("voo", sp, sq, sq2)), (p, q, q2)))
        return sum(_take(x, axes, n, excitations) for x, axes in terms if x is not None)


def _take(x, axes, n, excitations):
    """x (axes in the order given) broadcast onto the 2n axes of a block, or gathered at the
    rows of a list of excitations."""
    if excitations is not None:
        return x[tuple(excitations[:, a] - 1 for a in axes)]
    order = np.argsort(axes)
    shape = [1] * (2 * n)
    for a, k in zip(np.array(axes)[order], order):
        shape[a] = x.shape[k]
    return x.transpose(order).reshape(shape)


def _with_transposes(pieces):
    """Add the pieces with the spins of their virtual (or occupied) pairs swapped."""
    for key, x in list(pieces.items()):
        kind, spins = key[0], key[1:]
        if kind in ("vv", "oo"):
            pieces.setdefault((kind, spins[1], spins[0]), x.T)
        elif kind == "vvo":
            pieces.setdefault((kind, spins[1], spins[0], spins[2]), x.transpose(1, 0, 2))
        elif kind == "voo":
            pieces.setdefault((kind, spins[0], spins[2], spins[1]), x.transpose(0, 2, 1))
    return pieces


def _epstein_nesbet_pieces(H):
    """Diagonal of the bare two-electron terms linear in T (CI diagonal minus D_MP)."""
    vvvv_aa, vvvv_ab, vvvv_bb = _vvvv_diagonals(H)
    return _with_transposes({
        ("vo", "a", "a"): np.einsum("aiia->ai", H.aa.voov),
        ("vo", "b", "b"): np.einsum("aiia->ai", H.bb.voov),
        ("vo", "a", "b"): -np.einsum("ajaj->aj", H.ab.vovo),
        ("vo", "b", "a"): -np.einsum("ibib->bi", H.ab.ovov),
        ("vv", "a", "a"): vvvv_aa,
        ("vv", "a", "b"): vvvv_ab,
        ("vv", "b", "b"): vvvv_bb,
        ("oo", "a", "a"): np.einsum("ijij->ij", H.aa.oooo),
        ("oo", "a", "b"): np.einsum("ijij->ij", H.ab.oooo),
        ("oo", "b", "b"): np.einsum("ijij->ij", H.bb.oooo),
    })


def _vvvv_diagonals(H):
    """<ab||ab> (same spin) and <ab~|ab~>, from the DF vectors when there are any."""
    chol = getattr(H, "chol", None)
    B_a = getattr(getattr(chol, "a", None), "vv", None)
    if B_a is None:
        return (np.einsum("abab->ab", H.aa.vvvv), np.einsum("abab->ab", H.ab.vvvv),
                np.einsum("abab->ab", H.bb.vvvv))
    B_b = chol.b.vv

    def same_spin(B):
        return np.einsum("xaa,xbb->ab", B, B, optimize=True) - np.einsum("xab,xba->ab", B, B, optimize=True)

    return same_spin(B_a), np.einsum("xaa,xbb->ab", B_a, B_b, optimize=True), same_spin(B_b)


def _t2_pieces(T, H):
    """Diagonal of the T2*T_K Jacobian (T1*T2 for singles, T2*T2 for doubles, T2*T3 for
    triples) in terms of spin-orbital quantities:
      v:   T2 part of Hbar(vv)_aa,          o:   minus T2 part of Hbar(oo)_ii,
      vo:  r(a,i) = sum_{ld} v^{il}_{ad} t^{ad}_{il},
      vv:  1/2 sum_{kl} v^{kl}_{ab} t^{ab}_{kl},    oo: 1/2 sum_{cd} v^{ij}_{cd} t^{cd}_{ij},
      vvo: -sum_l v^{lj}_{ab} t^{ab}_{lj},          voo: -sum_e v^{ij}_{ae} t^{ae}_{ij}."""
    v_aa, v_ab, v_bb = H.aa.oovv, H.ab.oovv, H.bb.oovv
    t_aa, t_ab, t_bb = T.aa, T.ab, T.bb

    def ein(subscripts, v, t):
        return np.einsum(subscripts, v, t, optimize=True)

    return _with_transposes({
        ("v", "a"): -0.5 * ein("mnaf,afmn->a", v_aa, t_aa) - ein("mnaf,afmn->a", v_ab, t_ab),
        ("v", "b"): -0.5 * ein("mnaf,afmn->a", v_bb, t_bb) - ein("nmfa,fanm->a", v_ab, t_ab),
        ("o", "a"): -0.5 * ein("inef,efin->i", v_aa, t_aa) - ein("inef,efin->i", v_ab, t_ab),
        ("o", "b"): -0.5 * ein("inef,efin->i", v_bb, t_bb) - ein("nife,feni->i", v_ab, t_ab),
        ("vo", "a", "a"): ein("ilad,adil->ai", v_aa, t_aa) + ein("ilad,adil->ai", v_ab, t_ab),
        ("vo", "b", "b"): ein("ilad,adil->ai", v_bb, t_bb) + ein("lida,dali->ai", v_ab, t_ab),
        ("vo", "a", "b"): ein("ljad,adlj->aj", v_ab, t_ab),
        ("vo", "b", "a"): ein("ildb,dbil->bi", v_ab, t_ab),
        ("vv", "a", "a"): 0.5 * ein("klab,abkl->ab", v_aa, t_aa),
        ("vv", "a", "b"): ein("klab,abkl->ab", v_ab, t_ab),
        ("vv", "b", "b"): 0.5 * ein("klab,abkl->ab", v_bb, t_bb),
        ("oo", "a", "a"): 0.5 * ein("ijcd,cdij->ij", v_aa, t_aa),
        ("oo", "a", "b"): ein("ijcd,cdij->ij", v_ab, t_ab),
        ("oo", "b", "b"): 0.5 * ein("ijcd,cdij->ij", v_bb, t_bb),
        ("vvo", "a", "a", "a"): -ein("ljab,ablj->abj", v_aa, t_aa),
        ("vvo", "b", "b", "b"): -ein("ljab,ablj->abj", v_bb, t_bb),
        ("vvo", "a", "b", "b"): -ein("ljab,ablj->abj", v_ab, t_ab),
        ("vvo", "a", "b", "a"): -ein("ilab,abil->abi", v_ab, t_ab),
        ("voo", "a", "a", "a"): -ein("ijae,aeij->aij", v_aa, t_aa),
        ("voo", "b", "b", "b"): -ein("ijae,aeij->aij", v_bb, t_bb),
        ("voo", "a", "a", "b"): -ein("ijae,aeij->aij", v_ab, t_ab),
        ("voo", "b", "a", "b"): -ein("ijeb,ebij->bij", v_ab, t_ab),
    })
