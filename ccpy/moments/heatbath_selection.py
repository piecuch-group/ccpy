"""Heat-bath selection of the triples and quadruples for adaptive CC(P;Q).

The CC(P;Q)_A contribution of an excitation K outside P is -l_K m_K / Delta_K, with m_K and
l_K the two-body moment and left vector of crcc_laplace (the correction is computed there, in
O(N^6), over all of Q). Instead of visiting every K, candidates are generated from the
unantisymmetrized moment diagrams, h * t2 (triples) and h * t2 * t2 (quadruples): each
diagram is a chain of sparse tensors joined on its summed labels, and only the products
whose magnitude exceeds eps are kept (sorted tensor entries make every join stop early, as
in heat-bath CI). eps is lowered until about buffer_factor * num_add distinct candidates
outside P are found. Each product of a diagram with external tuple tau adds coef * parity *
value to m_K, K the canonical (sorted) form of tau, so the moments of the candidates are the
sums of their screened products, with no per-candidate evaluation (an exact quadruples
moment costs ~nv^2 per diagram); l_K and Delta_K are exact. The candidates are ranked,
triples and quadruples together, and the num_add largest are returned as spinorbital rows
(odd indices alpha, even beta; see add_spinorbital_excitations_to_pspace).

With eps -> 0 the screened moments are exact and the candidates are every K with a nonzero
moment, so the selection is the exact top-k of the CC(P;Q)_A contributions.
"""
import time

import numpy as np

from ccpy.moments import crcc_laplace as lap

TRIPLES = {"aaa": lap.triples_aaa, "aab": lap.triples_aab, "abb": lap.triples_abb, "bbb": lap.triples_bbb}
QUADRUPLES = {"aaaa": lap.quadruples_aaaa, "aaab": lap.quadruples_aaab, "aabb": lap.quadruples_aabb,
              "abbb": lap.quadruples_abbb, "bbbb": lap.quadruples_bbbb}


def _coo(a, thresh):
    idx = np.nonzero(np.abs(a) > thresh)
    return np.stack(idx, axis=1), a[idx]


def _join(lab_a, idx_a, val_a, lab_b, idx_b, val_b, thresh):
    """Entries of the product of two sparse tensors sharing one label with |product| > thresh
    (the shared label is dropped)."""
    (c,) = set(lab_a) & set(lab_b)
    ka, kb = idx_a[:, lab_a.index(c)], idx_b[:, lab_b.index(c)]
    abs_b = np.abs(val_b)
    order = np.lexsort((-abs_b, kb))
    idx_b, val_b, kb, abs_b = idx_b[order], val_b[order], kb[order], abs_b[order]
    # kb + 1 - |b| / scale is ascending (by kb, then by decreasing |b|) and stays in (kb, kb + 1]
    scale = 1.0001 * abs_b.max() if len(abs_b) else 1.0
    key = kb + 1.0 - abs_b / scale
    start = np.searchsorted(kb, ka, side="left")
    cut = np.minimum(thresh / np.abs(val_a) / scale, 1.0)
    count = np.maximum(np.searchsorted(key, ka + 1.0 - cut, side="left") - start, 0)
    rows_a = np.repeat(np.arange(len(ka)), count)
    offsets = np.cumsum(count) - count
    rows_b = np.arange(count.sum()) - np.repeat(offsets - start, count)
    keep_b = [k for k, x in enumerate(lab_b) if x != c]
    keep_a = [k for k, x in enumerate(lab_a) if x != c]
    lab = "".join(lab_a[k] for k in keep_a) + "".join(lab_b[k] for k in keep_b)
    idx = np.hstack((idx_a[rows_a][:, keep_a], idx_b[rows_b][:, keep_b]))
    return lab, idx, val_a[rows_a] * val_b[rows_b]


def _term_tuples(factors, tensors, external, eps, maxabs):
    """External tuples (columns ordered as `external`) and values of the products of one moment
    diagram with |product| > eps."""
    factors = list(factors)
    bound = np.prod([maxabs[name] for name, _ in factors])
    if bound <= eps:
        return np.zeros((0, len(external)), dtype=np.int64), np.zeros(0)
    # each factor alone must exceed eps over the largest entries of the others
    coo = [(lab, *_coo(tensors[name], eps * maxabs[name] / bound)) for name, lab in factors]
    lab, idx, val = coo.pop(0)
    rest_bound = bound / maxabs[factors[0][0]]
    names = [name for name, _ in factors[1:]]
    while coo:
        k = next(k for k, f in enumerate(coo) if set(f[0]) & set(lab))
        rest_bound /= maxabs[names.pop(k)]
        lab, idx, val = _join(lab, idx, val, *coo.pop(k), eps / rest_bound)
    return idx[:, [lab.index(c) for c in external]], val


def _canonical(tuples, spincase):
    """Sort the same-spin labels of each tuple and drop the tuples with a repeated index; also
    returns the parity of each sort and the mask of the kept tuples."""
    r = len(spincase)
    x = np.array(tuples, dtype=np.int64)
    keep = np.ones(len(x), dtype=bool)
    sign = np.ones(len(x))
    for base in (0, r):
        for sp in "ab":
            cols = [base + k for k, s in enumerate(spincase) if s == sp]
            for p in range(len(cols)):
                for q in range(p + 1, len(cols)):
                    sign[x[:, cols[p]] > x[:, cols[q]]] *= -1.0
            if len(cols) > 1:
                x[:, cols] = np.sort(x[:, cols], axis=1)
                keep &= np.all(np.diff(x[:, cols], axis=1) != 0, axis=1)
    return x[keep], sign[keep], keep


def _flip_columns(rank):
    # aabb -> its spin flip: c d a b k l i j
    half = rank // 2
    virt = list(range(half, rank)) + list(range(half))
    return virt + [rank + v for v in virt]


def _strides(dims):
    """Mixed-radix strides that encode a tuple of indices < dims as one int64 key."""
    if np.prod(np.asarray(dims, dtype=float)) >= 2.0 ** 63:
        raise ValueError("heat-bath selection: excitation keys do not fit in int64")
    return np.cumprod([1] + list(dims[::-1])[:-1])[::-1].astype(np.int64)


def _decode(keys, strides, dims):
    return (keys[:, None] // strides) % np.asarray(dims, dtype=np.int64)


def _to_spinorbital(x, spincase):
    """0-based spin-block rows -> 1-based spinorbital rows (odd alpha, even beta)."""
    spin = np.array([s == "a" for s in spincase + spincase])
    return np.where(spin, 2 * (x + 1) - 1, 2 * (x + 1)).astype(np.int32)


def select_heatbath(T, L, H, H0, system, t3_excitations, num_add, t4_excitations=None, H4=None, use_RHF=False,
                    buffer_factor=2, min_eps=1.0e-12, eps_step=0.3):
    """
    Heat-bath selection of the num_add triples (and quadruples, with t4_excitations) outside
    the P space of largest |CC(P;Q)_A contribution| (see the module docstring). H provides the
    triples moments, H4 (default H) the quadruples ones, H0 the orbital energies.

    Returns the spinorbital rows of the selected triples [a, b, c, i, j, k] and quadruples
    [a, b, c, d, i, j, k, l] (None without t4_excitations), in decreasing contribution.
    """
    t_start = time.perf_counter()
    num_add = int(num_add)
    target = max(buffer_factor * num_add, num_add)
    e = {"o": {"a": np.diagonal(H0.a.oo), "b": np.diagonal(H0.b.oo)},
         "v": {"a": np.diagonal(H0.a.vv), "b": np.diagonal(H0.b.vv)}}

    cases = [(s, f, H, t3_excitations) for s, f in TRIPLES.items() if not use_RHF or s.count("a") >= 2]
    if t4_excitations is not None:
        H4 = H if H4 is None else H4
        cases += [(s, f, H4, t4_excitations) for s, f in QUADRUPLES.items() if not use_RHF or s.count("a") >= 2]
    cases = [(s, *f(T, L, h), lap._pspace(p, s)) for s, f, h, p in cases]  # (s, spincase, vir, occ, tensors, x, l, P)

    maxabs = {}
    for _, _, _, _, tensors, x, _, _ in cases:
        for name in {n for _, f in x for n, _ in f}:
            maxabs[name] = float(np.abs(tensors[name]).max()) if tensors[name].size else 0.0
    eps = max(np.prod([maxabs[n] for n, _ in f]) for c in cases for _, f in c[5])
    nv = {"a": system.nunoccupied_alpha, "b": system.nunoccupied_beta}
    no = {"a": system.noccupied_alpha, "b": system.noccupied_beta}
    dims = [[nv[x] for x in c[1]] + [no[x] for x in c[1]] for c in cases]
    strides = [_strides(d) for d in dims]
    pkeys = [np.unique(c[7] @ st) for c, st in zip(cases, strides)]

    while True:
        eps = max(eps * eps_step, min_eps)
        candidates = []
        for (s, spincase, vir, occ, tensors, x, _, _), st, pk in zip(cases, strides, pkeys):
            keys, vals = [], []
            for coef, f in x:
                tup, val = _term_tuples(f, tensors, vir + occ, eps, maxabs)
                c, sign, keep = _canonical(tup, spincase)
                val = coef * sign * val[keep]
                if use_RHF and spincase == "aabb":  # its spin flip is added with it
                    keep = _le(c, c[:, _flip_columns(len(spincase))])
                    c, val = c[keep], val[keep]
                keys.append(c @ st)
                vals.append(val)
            # m_K = sum of coef * parity * product over the screened products of every diagram
            keys, inverse = np.unique(np.concatenate(keys), return_inverse=True)
            m = np.bincount(inverse.ravel(), weights=np.concatenate(vals), minlength=len(keys))
            outside = ~np.isin(keys, pk, assume_unique=True)
            candidates.append((keys[outside], m[outside]))
        n_found = sum(len(c[0]) for c in candidates)
        if n_found >= target or eps <= min_eps:
            break

    # exact CC(P;Q)_A contributions of the candidates
    rows, value, rank = [], [], []
    for (s, spincase, vir, occ, tensors, x, l, _), (keys, m), st, d in zip(cases, candidates, strides, dims):
        if len(keys) == 0:
            continue
        c = _decode(keys, st, d)
        lk = lap.tuple_values(lap.antisymmetrize(l, spincase, vir, occ), tensors, c, vir + occ)
        r = len(spincase)
        delta = sum(e["v"][spincase[k]][c[:, k]] for k in range(r)) - sum(e["o"][spincase[k]][c[:, r + k]]
                                                                         for k in range(r))
        rows.append(_to_spinorbital(c, spincase))
        value.append(-lk * m / delta)
        rank.append(np.full(len(c), r))
    if not rows:
        empty = np.zeros((0, 6), dtype=np.int32)
        return empty, (np.zeros((0, 8), dtype=np.int32) if t4_excitations is not None else None)
    value, rank = np.concatenate(value), np.concatenate(rank)
    order = np.argsort(-np.abs(value), kind="stable")[:num_add]
    triples = np.vstack([r for r in rows if r.shape[1] == 6] or [np.zeros((0, 6), dtype=np.int32)])
    quadruples = np.vstack([r for r in rows if r.shape[1] == 8] or [np.zeros((0, 8), dtype=np.int32)])
    n3 = len(triples)
    sel3, sel4 = order[rank[order] == 3], order[rank[order] == 4] - n3

    print("   Heat-bath selection: eps = {:.2e}, {} candidates ({} triples, {} quadruples), {:.2f} s".format(
        eps, n_found, n3, len(quadruples), time.perf_counter() - t_start))
    print("   Selected {} triples and {} quadruples\n".format(len(sel3), len(sel4)))
    return triples[sel3], (quadruples[sel4] if t4_excitations is not None else None)


def _le(x, y):
    """Boolean mask of the rows of x lexicographically <= the rows of y."""
    diff = x - y
    first = np.argmax(diff != 0, axis=1)
    return diff[np.arange(len(x)), first] <= 0
