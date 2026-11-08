"""Rank-local integral-driven vvvv layers for Cholesky CC(P).

T2 and T3 never share integral tiles or amplitude snapshots. Within T3,
each spin channel serves all applicable spin cases. Connectivity is local
to the call, after the non-vvvv routines have finished sorting the P lists.
"""
import numpy as np
from ccpy.lib.core import vvvv_contraction


def _records(T, excitations, sectors, pairs, nua, nub):
    """Group sources by occupied triple/spectator, targets by output pair.

    Each pair specification is (first, second, spectator, parity). Sector
    offsets prevent unlike spin sectors with equal indices from coupling.
    Indices passed to Fortran are 1-based; signed target ids carry parity.
    """
    sources, values, groups, targets, pair_ids = [], [], [], [], []
    offset = group_offset = 0
    sizes = []
    for spin, specs in zip(sectors, pairs):
        exc = excitations[spin]
        n = len(getattr(T, spin))
        sizes.append(n)
        if len(exc) and not np.all(exc[0] == 1):
            # Compress occupied triples independently of orbital dimensions.
            _, occupied = np.unique(exc[:, 3:], axis=0, return_inverse=True)
            vmax = int(exc[:, :3].max()) + 1
            for e, f, spectator, sign in specs:
                sources.append(exc[:, [e, f]])
                values.append(sign * getattr(T, spin))
                groups.append(group_offset + occupied * vmax + exc[:, spectator])
                targets.append(sign * (offset + np.arange(n, dtype=np.int64) + 1))
                pair_ids.append((exc[:, e].astype(np.int64) - 1) * nub + exc[:, f] - 1)
            group_offset += (int(occupied.max()) + 1) * vmax
        offset += n
    if not sources:
        return None
    source = np.concatenate(sources)
    amps = np.concatenate(values)
    _, group = np.unique(np.concatenate(groups), return_inverse=True)
    target = np.concatenate(targets)
    pair = np.concatenate(pair_ids)
    order = np.argsort(group, kind='stable')
    goff = np.r_[0, np.cumsum(np.bincount(group))] + 1
    source = np.asfortranarray(source[order].T, dtype=np.int32)
    amps = amps[order]
    order = np.argsort(pair, kind='stable')
    poff = np.r_[0, np.cumsum(np.bincount(pair, minlength=nua * nub))] + 1
    # Check before narrowing the addressing integers used by the native kernels.
    if max(len(amps), offset, len(goff)) >= np.iinfo(np.int32).max:
        raise OverflowError('T3 vvvv connectivity exceeds 32-bit native indexing')
    target = np.asfortranarray(np.stack((group[order] + 1, target[order])), dtype=np.int32)
    return source, amps, goff.astype(np.int32), target, poff.astype(np.int32), sizes


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


def add_vvvv_t3(T, dT, X, H, excitations, rhf):
    """Add Hbar(vvvv)*T3 across spin cases, in their current P-list order."""
    same = ((0, 1, 2, 1), (0, 2, 1, -1), (1, 2, 0, 1))
    channels = [
        ('aa', ('aaa', 'aab'), (same, ((0, 1, 2, 1),))),
        ('ab', ('aab',) if rhf else ('aab', 'abb'),
         (((1, 2, 0, 1), (0, 2, 1, -1)),) if rhf else
         (((1, 2, 0, 1), (0, 2, 1, -1)), ((0, 2, 1, 1), (0, 1, 2, -1)))),
    ]
    if not rhf:
        channels.append(('bb', ('bbb', 'abb'), (same, ((1, 2, 0, 1),))))
    for spin, sectors, specs in channels:
        amp = getattr(T, spin)
        nua, nub = amp.shape[:2]
        records = _records(T, excitations, sectors, specs, nua, nub)
        if records is None:
            continue
        source, values, goff, target, poff, sizes = records
        ss = spin[0] == spin[1]
        r3 = vvvv_contraction.vvvv_t3(
            getattr(X.chol, spin[0]).vv.transpose(0, 2, 1),
            getattr(X.chol, spin[1]).vv.transpose(0, 2, 1),
            amp, getattr(H, spin).oovv, source, values, goff, target, poff,
            ss, sum(sizes))
        offset = 0
        for sector, n in zip(sectors, sizes):
            getattr(dT, sector)[:] += r3[offset:offset + n]
            offset += n
