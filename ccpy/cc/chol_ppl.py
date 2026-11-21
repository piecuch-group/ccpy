"""Rank-local integral-driven vvvv layers for Cholesky CC(P).

T2 never shares integral tiles with the selected T3/T4 layers. Every selected
Hbar(vvvv) contraction of one pair spin (ladders on T3 and T4, and the T4
(vvvv T2 T2) terms) is served by one pass over W_ab(e,f) slices built from the
Cholesky vectors (vvvv_contraction.vvvv_tn). Records index the P lists in their
current order, so the result is either added after the non-vvvv routines have
finished sorting (add_vvvv_t3) or passed into them as the starting residual,
which their sorts carry along (vvvv_t34).
"""
from functools import lru_cache
from itertools import combinations, permutations

import numpy as np
from ccpy.lib.core import vvvv_contraction


def _parity(seq):
    seq, sign = list(seq), 1
    for i in range(len(seq)):
        while seq[i] != i:
            j = seq[i]
            seq[i], seq[j] = seq[j], seq[i]
            sign = -sign
    return sign


def _dense_rank(keys):
    """np.unique(keys, return_inverse=True)[1] for non-negative integer keys."""
    top = int(keys.max()) + 1
    if top > 1 << 24:
        return np.unique(keys, return_inverse=True)[1].reshape(-1)
    present = np.zeros(top, dtype=np.int64)
    present[keys] = 1
    return (np.cumsum(present) - 1)[keys]


def _stable_order(keys, nkeys):
    """np.argsort(keys, kind='stable') for keys in [0, nkeys); 16-bit keys take NumPy's radix sort."""
    if nkeys <= 1 << 16:
        keys = keys.astype(np.uint16)
    return np.argsort(keys, kind='stable')


def _pair_specs(spin, pair):
    """(e col, f col, spectator cols, parity) of every virtual pair of spin `pair` in a
    determinant of spin case `spin`; alpha columns precede beta columns."""
    nv = len(spin)
    out = []
    for p, q in combinations(range(nv), 2):
        if (spin[p], spin[q]) == tuple(pair):
            rest = [c for c in range(nv) if c not in (p, q)]
            out.append((p, q, rest, _parity([p, q] + rest)))
    return out


# (vvvv T2 T2) terms of ccpy/cc/ccsdtq.py update_t4a..e: (coef, einsum, W spin, T2 spins).
# dT is antisymmetrized afterwards over every spin-preserving permutation (cc_loops_t4).
T2T2_TERMS = {
    "aaaa": [(36.0 / 576.0, "abef,fcjk,edil->abcdijkl", "aa", ("aa", "aa"))],
    "aaab": [(9.0 / 36.0, "abef,fcjk,edil->abcdijkl", "aa", ("aa", "ab")),
             (18.0 / 36.0, "adef,ebij,cfkl->abcdijkl", "ab", ("aa", "ab"))],
    "aabb": [(0.25, "abef,ecik,fdjl->abcdijkl", "aa", ("ab", "ab")),
             (0.25, "acef,ebij,fdkl->abcdijkl", "ab", ("aa", "bb")),
             (1.0, "acef,edil,bfjk->abcdijkl", "ab", ("ab", "ab")),
             (0.25, "cdef,afil,bejk->abcdijkl", "bb", ("ab", "ab"))],
    "abbb": [(9.0 / 36.0, "abef,fcjk,deli->dcbalkji", "bb", ("bb", "ab")),
             (18.0 / 36.0, "dafe,ebij,fclk->dcbalkji", "ab", ("bb", "ab"))],
    "bbbb": [(36.0 / 576.0, "abef,fcjk,edil->abcdijkl", "bb", ("bb", "bb"))],
}


def _spin_perms(spin):
    """Spin-preserving permutations of the columns 0..n-1 of a block with spins `spin`."""
    blocks = [[c for c in range(len(spin)) if spin[c] == s] for s in "ab"]
    for pa in permutations(blocks[0]):
        for pb in permutations(blocks[1]):
            perm = list(range(len(spin)))
            for src, dst in zip(blocks[0] + blocks[1], list(pa) + list(pb)):
                perm[src] = dst
            yield perm, _parity(perm)


@lru_cache(maxsize=None)
def _placements(sector):
    """{(p, q): ((r, x, y, efib, s, z, w, ffib, coef), ...)} for the (vvvv T2 T2) terms of a T4
    sector: resid(D) += coef * sum_ef W(D[p], D[q]; e, f) Fe(e; r, x, y) Ff(f; s, z, w), with
    texc rows (virtuals 1-4, occupieds 5-8) and fibers named by T2 spin and contracted slot
    ("aa", "ab": alpha first; "ba": beta of T2(ab); "bb")."""
    n = len(sector)
    acc = {}
    for coef, spec, wspin, t2s in T2T2_TERMS.get(sector, ()):
        lhs, out = spec.split("->")
        w, *amps = lhs.split(",")
        for pv, sv in _spin_perms(sector):
            for po, so in _spin_perms(sector):
                col = {x: (pv[c] if c < n else n + po[c - n]) for c, x in enumerate(out)}
                sign = sv * so
                p, q = col[w[0]], col[w[1]]
                if wspin[0] == wspin[1] and p > q:
                    p, q, sign = q, p, -sign
                sides = {}
                for name, letters in zip(t2s, amps):
                    k = 0 if letters[0] in w[2:] else 1
                    r, x, y = col[letters[1 - k]], col[letters[2]], col[letters[3]]
                    if name[0] == name[1]:
                        fib = name
                        if k == 1:
                            sign = -sign
                        if x > y:
                            x, y, sign = y, x, -sign
                    else:
                        fib = "ab" if k == 0 else "ba"
                    sides[letters[k]] = (r + 1, x + 1, y + 1, fib)
                key = (p, q) + sides[w[2]] + sides[w[3]]
                acc[key] = acc.get(key, 0.0) + coef * sign
    out = {}
    for key, c in acc.items():
        if abs(c) > 1e-12:
            out.setdefault(key[:2], []).append(key[2:] + (c,))
    return {k: tuple(v) for k, v in out.items()}


def _fiber(T, name):
    """T2 as a matrix with the contracted virtual first, and its (d1, d2) column strides."""
    t = {"aa": T.aa, "ab": T.ab, "ba": T.ab.transpose(1, 0, 2, 3), "bb": T.bb}[name]
    return t.reshape(t.shape[0], -1, order="F"), t.shape[1], t.shape[2]


def _texc(excitations, spin):
    """Excitations as texc rows (virtuals 1-4, occupieds 5-8, zero padded)."""
    exc = excitations[spin]
    n = len(spin)
    out = np.zeros((len(exc), 8), dtype=np.int32)
    out[:, :n] = exc[:, :n]
    out[:, 4:4 + n] = exc[:, n:]
    return out


def _records(T, excitations, entries, nua, nub, with_t2t2):
    """Records of one slice call. entries: (sector, pair spin); a sector may appear with
    several pair spins (RHF), all sharing its determinant ids.

    Sources are grouped by occupied tuple and spectators, targets by output pair. Group
    offsets keep unlike entries apart; indices passed to Fortran are 1-based; signed target
    ids carry the pair parity. Each (entry, pair) is a slot type for the T2*T2 placements."""
    offsets, sizes = {}, []
    for sector, _ in entries:
        if sector not in offsets:
            offsets[sector] = sum(sizes)
            sizes.append(len(getattr(T, sector)))
    sources, values, groups, targets, pair_ids, slots = [], [], [], [], [], []
    types = []
    group_offset = 0
    for sector, pair in entries:
        exc = excitations[sector]
        n = len(sector)
        if not len(exc) or np.all(exc[0] == 1):
            continue
        amps = getattr(T, sector)
        occ = exc[:, n:].astype(np.int64)
        base = int(occ.max()) + 1
        okey = occ[:, 0]
        for c in range(1, n):
            okey = okey * base + occ[:, c]
        occupied = _dense_rank(okey)
        vmax = int(exc[:, :n].max()) + 1
        ids = offsets[sector] + np.arange(len(exc), dtype=np.int64) + 1
        for p, q, rest, sign in _pair_specs(sector, pair):
            key = occupied.copy()
            for c in rest:
                key = key * vmax + exc[:, c]
            sources.append(exc[:, [p, q]])
            values.append(sign * amps)
            groups.append(group_offset + key)
            targets.append(sign * ids)
            pair_ids.append((exc[:, p].astype(np.int64) - 1) * nub + exc[:, q] - 1)
            pl = _placements(sector).get((p, q), ()) if with_t2t2 else ()
            types.append(pl)
            slots.append(np.full(len(exc), len(types), dtype=np.int64))
        group_offset += (int(occupied.max()) + 1) * vmax ** (n - 2)
    if not sources:
        return None
    source = np.concatenate(sources)
    amps = np.concatenate(values)
    group = _dense_rank(np.concatenate(groups))
    target = np.concatenate(targets)
    pair = np.concatenate(pair_ids)
    slot = np.concatenate(slots)
    counts = np.bincount(group)
    order = _stable_order(group, len(counts))
    goff = np.r_[0, np.cumsum(counts)] + 1
    source = np.asfortranarray(source[order].T, dtype=np.int32)
    amps = amps[order]
    order = _stable_order(pair, nua * nub)
    poff = np.r_[0, np.cumsum(np.bincount(pair, minlength=nua * nub))] + 1
    nres = sum(sizes)
    # Check before narrowing the addressing integers used by the native kernels.
    if max(len(amps), nres, len(goff)) >= np.iinfo(np.int32).max:
        raise OverflowError('vvvv connectivity exceeds 32-bit native indexing')
    target = np.asfortranarray(np.stack((group[order] + 1, target[order])), dtype=np.int32)
    slot = slot[order].astype(np.int32)
    texc = np.zeros((8, nres), dtype=np.int32, order="F")
    for sector, off in offsets.items():
        texc[:, off:off + len(getattr(T, sector))] = _texc(excitations, sector).T
    return dict(source=source, amps=amps, goff=goff.astype(np.int32), target=target,
                poff=poff.astype(np.int32), tslot=slot, texc=texc, types=types,
                sectors=list(offsets), sizes=sizes)


def _vvvv_call(T, X, H, excitations, spin, entries, h1, with_t2t2):
    """Run one slice spin; returns {sector: raw residual} in the current P-list order."""
    a, b = getattr(X.chol, spin[0]).vv, getattr(X.chol, spin[1]).vv
    amp = getattr(T, spin)
    nua, nub = amp.shape[:2]
    rec = _records(T, excitations, entries, nua, nub, with_t2t2)
    if rec is None:
        return {}
    # Placement table, and the T2 fibers it reads (e: first contracted index, f: second).
    rows = [pl for pls in rec["types"] for pl in pls]
    ploff = np.r_[0, np.cumsum([len(pls) for pls in rec["types"]])] + 1
    fibers = {"e": [], "f": []}
    for row in rows:
        for side, name in (("e", row[3]), ("f", row[7])):
            if name not in fibers[side]:
                fibers[side].append(name)
    mats, secs = {}, {}
    for side, dim in (("e", nua), ("f", nub)):
        blocks, sec, off = [], {}, 0
        for name in fibers[side]:
            m, d1, d2 = _fiber(T, name)
            sec[name] = (off, d1, d2)
            blocks.append(m)
            off += m.shape[1]
        mats[side] = np.asfortranarray(np.hstack(blocks)) if blocks else np.zeros((dim, 1), order="F")
        secs[side] = sec
    sece = np.array([secs["e"][n] for n in fibers["e"]] or [(0, 1, 1)], dtype=np.int32).T
    secf = np.array([secs["f"][n] for n in fibers["f"]] or [(0, 1, 1)], dtype=np.int32).T
    ide = {n: k + 1 for k, n in enumerate(fibers["e"])}
    idf = {n: k + 1 for k, n in enumerate(fibers["f"])}
    plint = np.array([r[:3] + (ide[r[3]],) + r[4:7] + (idf[r[7]],) for r in rows] or [(1,) * 8],
                     dtype=np.int32).T
    plcoef = np.array([r[8] for r in rows] or [0.0])
    ha, hb = (getattr(X, spin[0]).vv, getattr(X, spin[1]).vv) if h1 else (np.zeros((nua, nua)), np.zeros((nub, nub)))
    r = vvvv_contraction.vvvv_tn(
        a.transpose(0, 2, 1), b.transpose(0, 2, 1), amp, getattr(H, spin).oovv,
        np.asfortranarray(ha), np.asfortranarray(hb),
        rec["source"], rec["amps"], rec["goff"], rec["target"], rec["poff"], rec["tslot"], rec["texc"],
        np.asfortranarray(plint), plcoef, ploff.astype(np.int32), np.asfortranarray(sece),
        np.asfortranarray(secf), mats["e"], mats["f"], spin[0] == spin[1], h1)
    out, offset = {}, 0
    for sector, n in zip(rec["sectors"], rec["sizes"]):
        out[sector] = r[offset:offset + n]
        offset += n
    return out


def add_vvvv_t2(T, dT, X, rhf):
    """Add raw vvvv*T2 terms before antisymmetrization and denominators."""
    for spin in (('aa', 'ab') if rhf else ('aa', 'ab', 'bb')):
        amp = getattr(T, spin).transpose(3, 2, 1, 0)
        ra = getattr(X.chol, spin[0]).vv.transpose(0, 2, 1)
        if spin[0] == spin[1]:
            result = vvvv_contraction.vvvv_t2_sym(ra, 0.5 * amp)
        else:
            rb = X.chol.b.vv.transpose(0, 2, 1)
            result = vvvv_contraction.vvvv_t2(ra, rb, amp)
        getattr(dT, spin)[:] += result.transpose(3, 2, 1, 0)


def _channels(sectors, rhf):
    """Slice calls {pair spin: entries} covering every virtual pair of the given sectors.
    Under RHF the beta slices equal the alpha ones, so bb pairs join the aa call."""
    calls = {}
    for sector in sectors:
        for pair in ("aa", "ab", "bb"):
            if _pair_specs(sector, pair):
                spin = "aa" if (rhf and pair == "bb") else pair
                calls.setdefault(spin, []).append((sector, pair))
    return calls


def add_vvvv_t3(T, dT, X, H, excitations, rhf):
    """Add Hbar(vvvv)*T3 across spin cases, in their current P-list order (h1 stays in the
    T3 routines)."""
    sectors = ("aaa", "aab") if rhf else ("aaa", "aab", "abb", "bbb")
    for spin, entries in _channels(sectors, rhf).items():
        for sector, r in _vvvv_call(T, X, H, excitations, spin, entries, False, False).items():
            getattr(dT, sector)[:] += r


def vvvv_t34(T, X, H, t3_excitations, t4_excitations, rhf):
    """{sector: raw residual} of Hbar(vvvv)*T3, Hbar(vvvv)*T4, h1(vv)*T3/T4 and the T4
    (vvvv T2 T2) terms, in the current P-list order, from one slice pass per pair spin."""
    excitations = {**t3_excitations, **t4_excitations}
    sectors = [s for s in (("aaa", "aab", "aaaa", "aaab", "aabb") if rhf else
                           ("aaa", "aab", "abb", "bbb", "aaaa", "aaab", "aabb", "abbb", "bbbb"))
               if not np.all(excitations[s][0] == 1)]
    out = {}
    for spin, entries in _channels(sectors, rhf).items():
        for sector, r in _vvvv_call(T, X, H, excitations, spin, entries, True, True).items():
            out[sector] = out.get(sector, 0.0) + r
    return out
