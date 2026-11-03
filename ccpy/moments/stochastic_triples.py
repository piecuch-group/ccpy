"""Semi-stochastic CR-CC(2,3)-type triples corrections: CR-CC(2,3), CC(t;3), and CC(P;3)_2BA.

Implements Algorithm 1 of Y. Damour, A. Gallo, and A. Scemama, "Stochastically accelerated
perturbative triples correction in coupled cluster calculations", arXiv:2405.16962, for the
CR-CC(2,3) family of corrections. The correction is a sum over virtual triplets,

    delta = sum_{abc} E(abc),   E(abc) = sum_{ijk in Q} M(abcijk) L(ijkabc) / D(abcijk),

where Q is the complement of the P space (all triples for CR-CC(2,3), the triples outside the
active space for CC(t;3), those not in t3_excitations for CC(P;3)). Each E(abc) costs
O(no^3 nv), and there are O(nv^3) of them, so all of them can be stored (memoization). Triplets
(over all spin cases) are sorted by the importance-sampling probability

    P(abc) = (1/N) / max(eps_min, e_a + e_b + e_c)

and split into contiguous buckets of about equal probability. Each Monte Carlo iteration
computes the first triplet not yet computed (the deterministic sweep) and draws one triplet
per bucket that the sweep has not completed. A triplet is computed the first time it is
drawn; later draws reuse it. With D the buckets completed by the sweep and S the others,

    delta = sum_{B in D} E_B + sum_{B in S} (Q_B / n_B) sum_{abc in B} n(abc) E(abc) / P(abc),

with Q_B the probability of bucket B, n_B its number of draws, and n(abc) the number of times
(a,b,c) was drawn. The buckets are sampled independently, so the variance is the sum of the
per-bucket variances of the mean. The calculation stops when the statistical error of the
chosen variant drops below `target_error`, when `max_fraction` of the triplets has been
computed, after `max_time` seconds, or when every triplet has been computed, in which case the
result is exact (it equals the deterministic correction to round-off).
"""
import time

import numpy as np

from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.lib.core import stochastic_triples_loops

VARIANTS = ("A", "B", "C", "D")

DEFAULT_OPTIONS = {
    "target_error": 1.0e-05,  # statistical error (Eh) of the chosen variant at which to stop
    "variant": "D",  # variant whose error controls convergence
    "max_fraction": 1.0,  # stop once this fraction of the triplets has been computed
    "max_time": None,  # stop after this many seconds (None: no limit)
    "nbuckets": 100,  # number of buckets (of about equal probability)
    "batch_size": None,  # Monte Carlo iterations per round (None: about ntriplets / 500)
    "eps_min": 0.2,  # smallest denominator in the importance-sampling probability (Eh)
    "seed": None,  # seed of the random number generator
    "verbose": True,
    "stop_when": None,  # callable(delta, error) of the chosen variant; True stops the calculation
}


def _f(x):
    return np.asfortranarray(x)


def _diag_voov(x):
    """x(a,i,i,a) -> d(a,i)."""
    return np.einsum("aiia->ai", x)


def _diag_ovov(x):
    """x(i,a,i,a) -> d(i,a)."""
    return np.einsum("iaia->ia", x)


def _diag_vovo(x):
    """x(a,i,a,i) -> d(a,i)."""
    return np.einsum("aiai->ai", x)


def _diag_oooo_same(x):
    """x(j,i,j,i) -> d(i,j)."""
    return np.einsum("jiji->ij", x)


def _diag_oooo_mixed(x):
    """x(i,j,i,j) -> d(i,j)."""
    return np.einsum("ijij->ij", x)


def vvvv_diagonal(H, T):
    """Diagonal h(ab,ab) of the vvvv part of HBar for a four-index or a Cholesky HBar."""
    if hasattr(H, "chol"):
        from ccpy.moments.ccp3_chol import get_vvvv_diagonal

        return get_vvvv_diagonal(H, T)
    return (np.einsum("abab->ab", H.aa.vvvv), np.einsum("abab->ab", H.ab.vvvv), np.einsum("abab->ab", H.bb.vvvv))


class TriplesCorrectionKernel:
    """Contributions E(abc) of virtual triplets to the CR-CC(2,3)-type corrections.

    Triplets of all spin cases are indexed together: aaa (a<b<c), aab (a<b, c), and, without
    RHF symmetry, abb (a, b<c) and bbb (a<b<c). With RHF symmetry, aaa and aab contributions
    are doubled.

    Parameters
    ----------
    T, L, H, H0, system, use_RHF : as in calc_crcc23
    t3_excitations : dict of (n, 6) arrays, optional
        P space of triples (a, b, c, i, j, k; 1-based), excluded from the correction.
    num_active : int
        If positive, also exclude the CC(t;3) active-space triples, i.e., those with at least
        num_active active particles and num_active active holes.
    """

    spincases = ("aaa", "aab", "abb", "bbb")

    def __init__(self, T, L, H, H0, system, use_RHF=False, t3_excitations=None, num_active=0):
        self.use_RHF = use_RHF
        noa, nob = system.noccupied_alpha, system.noccupied_beta
        nua, nub = system.nunoccupied_alpha, system.nunoccupied_beta
        self.act = np.array([num_active, system.num_act_occupied_alpha, system.num_act_unoccupied_alpha,
                             system.num_act_occupied_beta, system.num_act_unoccupied_beta] if num_active > 0
                            else [0, 0, 0, 0, 0], dtype=np.int32)
        self._build_arrays(T, L, H, H0, system)

        # triplets of each spin case (1-based) and their orbital-energy sums
        ea, eb = np.diag(H0.a.vv), np.diag(H0.b.vv)
        trips = {
            "aaa": _strict_triples(nua, nua, nua, (True, True)),
            "aab": _strict_triples(nua, nua, nub, (True, False)),
        }
        if not use_RHF:
            trips["abb"] = _strict_triples(nua, nub, nub, (False, True))
            trips["bbb"] = _strict_triples(nub, nub, nub, (True, True))
        energies = {"a": ea, "b": eb}
        self.trips, self.spin, self.esum, self.offsets = {}, [], [], {}
        offset = 0
        for ispin, s in enumerate(trips):
            t = trips[s]
            self.trips[s] = np.asfortranarray(t.T.astype(np.int32))
            self.offsets[s] = offset
            offset += t.shape[0]
            self.spin.append(np.full(t.shape[0], ispin, dtype=np.int8))
            self.esum.append(sum(energies[s[p]][t[:, p] - 1] for p in range(3)))
        self.spin = np.concatenate(self.spin)
        self.esum = np.concatenate(self.esum)
        self.ntriplets = offset
        self.factor = 2.0 if use_RHF else 1.0

        # P space, sorted by virtual triplet: pexc[s] holds the occupied triples and
        # prange[s][t] the (1-based, inclusive) range of triplet t
        self.pexc, self.prange = {}, {}
        for s in self.trips:
            nv = (nua, nub)
            dims = [nv[0] if s[p] == "a" else nv[1] for p in range(3)]
            excits = None if t3_excitations is None else np.asarray(t3_excitations[s])
            self.pexc[s], self.prange[s] = _pspace_lookup(excits, self.trips[s].T, dims)

    def _build_arrays(self, T, L, H, H0, system):
        d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H, system)
        d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H, system)
        d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H, system)
        d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H, system)
        h_aa_vvvv, h_ab_vvvv, h_bb_vvvv = vvvv_diagonal(H, T)

        I2A_vvov = H.aa.vvov + np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
        I2C_vvov = H.bb.vvov + np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
        I2A_vooo = H.aa.vooo - np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
        I2B_ovoo = H.ab.ovoo - np.einsum("me,ecjk->mcjk", H.a.ov, T.ab, optimize=True)
        I2B_vooo = H.ab.vooo - np.einsum("me,aeik->amik", H.b.ov, T.ab, optimize=True)
        I2C_vooo = H.bb.vooo - np.einsum("me,cekj->cmkj", H.b.ov, T.bb, optimize=True)

        diag = lambda x: _f(np.diag(x).copy())
        common_a = dict(fA_o=diag(H0.a.oo), fA_v=diag(H0.a.vv), H1A_o=diag(H.a.oo), H1A_v=diag(H.a.vv))
        common_b = dict(fB_o=diag(H0.b.oo), fB_v=diag(H0.b.vv), H1B_o=diag(H.b.oo), H1B_v=diag(H.b.vv))
        dvoA, dooA = _f(_diag_voov(H.aa.voov)), _f(_diag_oooo_same(H.aa.oooo))
        dvoC, dooC = _f(_diag_voov(H.bb.voov)), _f(_diag_oooo_same(H.bb.oooo))
        dovB, dvoB = _f(_diag_ovov(H.ab.ovov)), _f(_diag_vovo(H.ab.vovo))
        dooB = _f(_diag_oooo_mixed(H.ab.oooo))

        # rearranged amplitudes and HBar elements shared by several spin cases
        t2b_mkyc = _f(T.ab.transpose(2, 3, 0, 1))
        t2b_ejkc = _f(T.ab.transpose(0, 2, 3, 1))
        t2b_eikx = _f(T.ab.transpose(1, 2, 3, 0))
        t2b_mjyc = _f(T.ab.transpose(3, 2, 0, 1))
        l2b_mkyc = _f(L.ab.transpose(2, 3, 0, 1))
        l2b_ejkc = _f(L.ab.transpose(0, 2, 3, 1))
        l2b_eikx = _f(L.ab.transpose(1, 2, 3, 0))
        l2b_mjyc = _f(L.ab.transpose(3, 2, 0, 1))
        H2B_vvvo_ekyc = _f(H.ab.vvvo.transpose(2, 3, 0, 1))
        H2B_vvov_ejyc = _f(H.ab.vvov.transpose(3, 2, 0, 1))
        H2B_ovvv_ejyc = _f(H.ab.ovvv.transpose(1, 0, 2, 3))
        H2B_vovv = _f(H.ab.vovv)
        H2B_oovo_mikx = _f(H.ab.oovo.transpose(3, 0, 1, 2))
        H2B_ooov_mjkc = _f(H.ab.ooov.transpose(2, 0, 1, 3))
        I2B_ovoo_mjkc = _f(I2B_ovoo.transpose(0, 2, 3, 1))
        I2B_vooo_mikx = _f(I2B_vooo.transpose(1, 2, 3, 0))

        def same_spin(t2, l1, l2, vooo, I_vvov, vovv, ooov, v_oovv, h_ov, f, dvo, doo, hvvvv, d3o, d3v):
            return dict(
                H2A_vooo_mijx=_f(vooo.transpose(1, 2, 3, 0)), t2a_mkyz=_f(t2.transpose(2, 3, 0, 1)),
                I2A_vvov_eixy=_f(I_vvov.transpose(3, 2, 0, 1)), t2a_ejkz=_f(t2.transpose(0, 2, 3, 1)),
                H2A_vovv_eixy=_f(vovv.transpose(0, 1, 3, 2)), l2a_ejkz=_f(l2.transpose(0, 2, 3, 1)),
                H2A_ooov_mijx=_f(ooov.transpose(2, 1, 0, 3)), l2a_mkyz=_f(l2.transpose(2, 3, 0, 1)),
                l1a=_f(l1), l2a=_f(l2), vA_oovv=_f(v_oovv), H1A_ov=_f(h_ov),
                fA_o=f["o"], fA_v=f["v"], H1A_o=f["ho"], H1A_v=f["hv"],
                dvoA=dvo, dooA=doo, H2A_vvvv=_f(hvvvv), D3A_O=_f(d3o), D3A_V=_f(d3v),
            )

        fa = dict(o=common_a["fA_o"], v=common_a["fA_v"], ho=common_a["H1A_o"], hv=common_a["H1A_v"])
        fb = dict(o=common_b["fB_o"], v=common_b["fB_v"], ho=common_b["H1B_o"], hv=common_b["H1B_v"])
        self.args = {}
        self.args["aaa"] = same_spin(T.aa, L.a, L.aa, H.aa.vooo, I2A_vvov, H.aa.vovv, H.aa.ooov, H.aa.oovv, H.a.ov,
                                     fa, dvoA, dooA, h_aa_vvvv, d3aaa_o, d3aaa_v)
        self.args["bbb"] = same_spin(T.bb, L.b, L.bb, H.bb.vooo, I2C_vvov, H.bb.vovv, H.bb.ooov, H.bb.oovv, H.b.ov,
                                     fb, dvoC, dooC, h_bb_vvvv, d3bbb_o, d3bbb_v)
        self.args["aab"] = dict(
            t2a_eijx=_f(T.aa.transpose(1, 2, 3, 0)), H2B_vvvo_ekyc=H2B_vvvo_ekyc,
            I2A_vooo_mijx=_f(I2A_vooo.transpose(1, 2, 3, 0)), t2b_mkyc=t2b_mkyc,
            t2a_mixy=_f(T.aa.transpose(2, 3, 0, 1)), I2B_ovoo_mjkc=I2B_ovoo_mjkc,
            H2A_vvov_eixy=_f(H.aa.vvov.transpose(3, 2, 0, 1)), t2b_ejkc=t2b_ejkc,
            t2b_eikx=t2b_eikx, H2B_vvov_ejyc=H2B_vvov_ejyc, I2B_vooo_mikx=I2B_vooo_mikx, t2b_mjyc=t2b_mjyc,
            l2a_eijx=_f(L.aa.transpose(1, 2, 3, 0)), H2B_vovv=H2B_vovv,
            H2A_ooov_mijx=self.args["aaa"]["H2A_ooov_mijx"], l2b_mkyc=l2b_mkyc,
            H2A_vovv_eixy=self.args["aaa"]["H2A_vovv_eixy"], l2b_ejkc=l2b_ejkc,
            l2a_mixy=_f(L.aa.transpose(2, 3, 0, 1)), H2B_ooov_mjkc=H2B_ooov_mjkc,
            l2b_eikx=l2b_eikx, H2B_ovvv_ejyc=H2B_ovvv_ejyc, H2B_oovo_mikx=H2B_oovo_mikx, l2b_mjyc=l2b_mjyc,
            l1a=_f(L.a), l1b=_f(L.b), l2a=_f(L.aa), l2b=_f(L.ab), vA_oovv=_f(H.aa.oovv), vB_oovv=_f(H.ab.oovv),
            H1A_ov=_f(H.a.ov), H1B_ov=_f(H.b.ov), **common_a, **common_b,
            dvoA=dvoA, dooA=dooA, dovB=dovB, dvoB=dvoB, dooB=dooB, dvoC=dvoC,
            H2A_vvvv=_f(h_aa_vvvv), H2B_vvvv=_f(h_ab_vvvv),
            D3A_O=_f(d3aaa_o), D3A_V=_f(d3aaa_v), D3B_O=_f(d3aab_o), D3B_V=_f(d3aab_v),
            D3C_O=_f(d3abb_o), D3C_V=_f(d3abb_v),
        )
        self.args["abb"] = dict(
            H2B_vvov_eiay=H2B_vvov_ejyc, t2c_ejkz=_f(T.bb.transpose(0, 2, 3, 1)),
            t2b_miay=t2b_mjyc, I2C_vooo_mjkz=_f(I2C_vooo.transpose(1, 3, 2, 0)),
            I2B_vooo_mijx=I2B_vooo_mikx, t2c_mkyz=_f(T.bb.transpose(2, 3, 0, 1)),
            t2b_eijx=t2b_eikx, H2C_vvov_ekzy=_f(H.bb.vvov.transpose(3, 2, 0, 1)),
            H2B_vvvo_ejay=H2B_vvvo_ekyc, t2b_eikz=t2b_ejkc, t2b_mjay=t2b_mkyc, I2B_ovoo_mikz=I2B_ovoo_mjkc,
            H2B_ovvv_eiay=H2B_ovvv_ejyc, l2c_ejkz=_f(L.bb.transpose(0, 2, 3, 1)),
            l2b_miay=l2b_mjyc, H2C_ooov_mjkz=_f(H.bb.ooov.transpose(2, 0, 1, 3)),
            l2b_eijx=l2b_eikx, H2C_vovv=_f(H.bb.vovv), H2B_oovo_mijx=H2B_oovo_mikx,
            l2c_mkyz=_f(L.bb.transpose(2, 3, 0, 1)),
            H2B_vovv=H2B_vovv, l2b_eikz=l2b_ejkc, l2b_mjay=l2b_mkyc, H2B_ooov_mikz=H2B_ooov_mjkc,
            l1a=_f(L.a), l1b=_f(L.b), l2b=_f(L.ab), l2c=_f(L.bb), vB_oovv=_f(H.ab.oovv), vC_oovv=_f(H.bb.oovv),
            H1A_ov=_f(H.a.ov), H1B_ov=_f(H.b.ov), **common_a, **common_b,
            dvoA=dvoA, dovB=dovB, dvoB=dvoB, dooB=dooB, dvoC=dvoC, dooC=dooC,
            H2B_vvvv=_f(h_ab_vvvv), H2C_vvvv=_f(h_bb_vvvv),
            D3B_O=_f(d3aab_o), D3B_V=_f(d3aab_v), D3C_O=_f(d3abb_o), D3C_V=_f(d3abb_v),
            D3D_O=_f(d3bbb_o), D3D_V=_f(d3bbb_v),
        )
        # f2py takes lower-case keyword arguments
        self.args = {s: {k.lower(): v for k, v in args.items()} for s, args in self.args.items()}

    def compute(self, indices):
        """E(abc) of the triplets with global indices `indices`: array (len(indices), 4)."""
        indices = np.asarray(indices)
        out = np.zeros((indices.size, 4))
        routine = {"aaa": stochastic_triples_loops.crcc23a_abc, "aab": stochastic_triples_loops.crcc23b_abc,
                   "abb": stochastic_triples_loops.crcc23c_abc, "bbb": stochastic_triples_loops.crcc23a_abc}
        for ispin, s in enumerate(self.trips):
            mask = self.spin[indices] == ispin
            if not np.any(mask):
                continue
            local = indices[mask] - self.offsets[s]
            trips = np.asfortranarray(self.trips[s][:, local])
            pstart, pend = (np.ascontiguousarray(x[local]) for x in self.prange[s])
            e = routine[s](trips, pstart, pend, self.pexc[s], self.act, **self.args[s])
            out[mask] = self.factor * e.T
        return out


def _strict_triples(n1, n2, n3, ordered):
    """1-based (x, y, z) with x < y if ordered[0] and y < z if ordered[1]."""
    blocks = []
    for x in range(1, n1 + 1):
        y, z = np.meshgrid(np.arange(x + 1 if ordered[0] else 1, n2 + 1), np.arange(1, n3 + 1), indexing="ij")
        keep = y < z if ordered[1] else np.ones(y.shape, dtype=bool)
        blocks.append(np.stack([np.full(keep.sum(), x), y[keep], z[keep]], axis=1))
    return np.concatenate(blocks).astype(np.int32)


def _pspace_lookup(excits, trips, dims):
    """Occupied parts of the P-space excitations, grouped by virtual triplet, and the range
    (1-based, inclusive) of each triplet in `trips`."""
    ntrip = trips.shape[0]
    empty = (np.asfortranarray(np.ones((3, 1), dtype=np.int32)),
             (np.ones(ntrip, dtype=np.int32), np.zeros(ntrip, dtype=np.int32)))
    if excits is None or excits.size == 0:
        return empty
    key = lambda v: ((v[:, 0].astype(np.int64) - 1) * dims[1] + (v[:, 1] - 1)) * dims[2] + (v[:, 2] - 1)
    pkey = key(excits[:, :3])
    order = np.argsort(pkey, kind="stable")
    pkey, excits = pkey[order], excits[order]
    tkey = key(trips)
    start = np.searchsorted(pkey, tkey, side="left")
    end = np.searchsorted(pkey, tkey, side="right")
    pexc = np.asfortranarray(excits[:, 3:6].T.astype(np.int32))
    return pexc, ((start + 1).astype(np.int32), end.astype(np.int32))


def run_semistochastic(kernel, options=None):
    """Algorithm 1 of arXiv:2405.16962 over the triplets of `kernel`.

    Returns (deltas, errors, info): the estimated corrections and their statistical errors
    (one standard deviation) for variants A-D, and a dict with the number and fraction of
    triplets computed, the number of Monte Carlo iterations, and whether the result is exact.
    """
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    unknown = set(opts) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(f"unknown stochastic options {sorted(unknown)}")
    ivar = VARIANTS.index(opts["variant"].upper())
    rng = np.random.default_rng(opts["seed"])
    verbose = opts["verbose"]
    t_start = time.perf_counter()

    n = kernel.ntriplets
    # importance-sampling probability, triplets sorted by decreasing probability
    weight = 1.0 / np.maximum(opts["eps_min"], kernel.esum)
    order = np.argsort(-weight, kind="stable")
    prob = weight[order] / weight.sum()
    cum = np.cumsum(prob)
    cum[-1] = 1.0
    # contiguous buckets of about equal probability
    nbuckets = max(1, min(int(opts["nbuckets"]), n))
    raw = np.minimum(((cum - prob) * nbuckets).astype(np.int64), nbuckets - 1)
    _, bucket = np.unique(raw, return_inverse=True)
    nbuckets = bucket[-1] + 1
    first = np.searchsorted(bucket, np.arange(nbuckets), side="left")
    last = np.searchsorted(bucket, np.arange(nbuckets), side="right") - 1
    bucket_start = np.concatenate([[0.0], cum])[first]
    bucket_prob = np.bincount(bucket, weights=prob, minlength=nbuckets)

    batch = opts["batch_size"] or max(16, n // 500)
    energy = np.zeros((n, 4))  # E(abc) in sorted order
    computed = np.zeros(n, dtype=bool)
    bucket_draws = np.zeros(nbuckets, dtype=np.int64)  # n_B
    # running sums over the draws of each bucket of y = Q_B E(abc) / P(abc) and y^2
    s1, s2 = np.zeros((nbuckets, 4)), np.zeros((nbuckets, 4))
    det_sum, split = np.zeros(4), 0  # sum of E(abc) over the completed buckets, which end at `split`
    imin, niter, ncomputed = 0, 0, 0

    if verbose:
        print(f"   Semi-stochastic triples correction: {n} triplets in {nbuckets} buckets, "
              f"target error {opts['target_error']:.1e} Eh (variant {VARIANTS[ivar]})")
        print("      % computed     delta              error        wall (s)")
    while True:
        # stochastic draws: one per iteration in each bucket not completed by the sweep
        open_buckets = np.nonzero(last >= imin)[0]
        u = rng.random((batch, open_buckets.size))
        x = bucket_start[open_buckets] + u * bucket_prob[open_buckets]
        drawn = np.clip(np.searchsorted(cum, x, side="right"), first[open_buckets], last[open_buckets]).ravel()
        bucket_draws[open_buckets] += batch
        # deterministic sweep: the next `batch` triplets not yet computed
        sweep = _next_uncomputed(computed, imin, batch)
        new = np.union1d(drawn[~computed[drawn]], sweep)
        if new.size:
            energy[new] = kernel.compute(order[new])
            computed[new] = True
            ncomputed += new.size
        niter += batch
        imin = _next_uncomputed(computed, imin, 1)
        imin = imin[0] if imin.size else n

        y = (bucket_prob[bucket[drawn]] / prob[drawn])[:, None] * energy[drawn]
        np.add.at(s1, bucket[drawn], y)
        np.add.at(s2, bucket[drawn], y * y)
        ndet = np.searchsorted(last, imin - 1, side="right")  # buckets with last < imin
        new_split = first[ndet] if ndet < nbuckets else n
        det_sum += energy[split:new_split].sum(axis=0)
        split = new_split
        deltas, errors = det_sum.copy(), np.zeros(4)
        if ndet < nbuckets:
            nb = bucket_draws[ndet:, None].astype(float)
            mean = s1[ndet:] / nb
            with np.errstate(invalid="ignore", divide="ignore"):
                var = np.where(nb > 1, (s2[ndet:] - nb * mean ** 2) / ((nb - 1) * nb), np.inf)
            deltas += mean.sum(axis=0)
            errors = np.sqrt(np.maximum(var.sum(axis=0), 0.0))

        fraction = ncomputed / n
        elapsed = time.perf_counter() - t_start
        if verbose:
            print(f"      {100 * fraction:8.3f}    {deltas[ivar]: .10f}    {errors[ivar]:.3e}    {elapsed:8.2f}")
        if imin >= n:
            break
        if errors[ivar] < opts["target_error"] or fraction >= opts["max_fraction"]:
            break
        if opts["max_time"] is not None and elapsed > opts["max_time"]:
            break
        if opts["stop_when"] is not None and opts["stop_when"](deltas[ivar], errors[ivar]):
            break

    info = {"ntriplets": n, "ncomputed": ncomputed, "fraction": fraction, "iterations": niter,
            "exact": bool(imin >= n), "wall_time": time.perf_counter() - t_start}
    return dict(zip(VARIANTS, deltas)), dict(zip(VARIANTS, errors)), info


def _next_uncomputed(computed, start, count):
    """Indices of the first `count` triplets not computed, from `start` on."""
    found, chunk = [], max(4 * count, 1024)
    while start < computed.size and sum(f.size for f in found) < count:
        found.append(start + np.nonzero(~computed[start:start + chunk])[0])
        start += chunk
    return np.concatenate(found)[:count] if found else np.zeros(0, dtype=np.int64)


def calc_stochastic_correction(T, L, corr_energy, H, H0, system, use_RHF=False, t3_excitations=None, num_active=0,
                               options=None, label="CR-CC(2,3)", reference_label="CCSD"):
    """Semi-stochastic CR-CC(2,3) (default), CC(t;3) (num_active > 0), or CC(P;3)_2BA
    (t3_excitations) correction. Returns (energies, deltas, errors, info)."""
    t_start = time.perf_counter()
    kernel = TriplesCorrectionKernel(T, L, H, H0, system, use_RHF, t3_excitations, num_active)
    t_setup = time.perf_counter() - t_start
    deltas, errors, info = run_semistochastic(kernel, options)
    info["setup_time"] = t_setup

    minutes, seconds = divmod(time.perf_counter() - t_start, 60)
    energies = {v: system.reference_energy + corr_energy + deltas[v] for v in VARIANTS}
    print(f"\n   Semi-stochastic {label} Calculation Summary")
    print("   -------------------------------------")
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print(f"   Triplets computed: {info['ncomputed']} of {info['ntriplets']} ({100 * info['fraction']:.2f}%)"
          + ("  [exact]" if info["exact"] else ""))
    print("   {} = {:>10.10f}".format(reference_label, system.reference_energy + corr_energy))
    for v in VARIANTS:
        print("   {}_{} = {:>10.10f}     δ_{} = {:>10.10f} +/- {:.2e}".format(label, v, energies[v], v, deltas[v],
                                                                            errors[v]))
    print("")
    return energies, deltas, errors, info


def calc_crcc23_stochastic(T, L, corr_energy, H, H0, system, use_RHF=False, options=None):
    """Semi-stochastic CR-CC(2,3) correction to CCSD."""
    return calc_stochastic_correction(T, L, corr_energy, H, H0, system, use_RHF, options=options)


def calc_cct3_stochastic(T, L, corr_energy, H, H0, system, use_RHF=False, num_active=1, options=None):
    """Semi-stochastic CC(t;3) correction to CCSDt (active space of `system`)."""
    return calc_stochastic_correction(T, L, corr_energy, H, H0, system, use_RHF, num_active=num_active,
                                      options=options, label="CC(t;3)", reference_label="CCSDt")


def calc_ccp3_2ba_stochastic(T, L, t3_excitations, corr_energy, H, H0, system, use_RHF=False, options=None):
    """Semi-stochastic CC(P;3)_2BA correction to CC(P) for the P space `t3_excitations`."""
    return calc_stochastic_correction(T, L, corr_energy, H, H0, system, use_RHF, t3_excitations=t3_excitations,
                                      options=options, label="CC(P;3)", reference_label="CC(P)")
