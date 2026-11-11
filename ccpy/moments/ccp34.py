"""Ground-state CC(P;Q) correction to CCSDTQ(P) in the two-body approximation, with the selection
of the triples and quadruples of largest moment contributions for adaptive CC(P;Q)."""
import time
from itertools import combinations, product
from types import SimpleNamespace

import numpy as np

from ccpy.hbar.diagonal import aaa_H3_aaa_diagonal, abb_H3_abb_diagonal, aab_H3_aab_diagonal, bbb_H3_bbb_diagonal
from ccpy.lib.core import ccp34_adaptive_loops
from ccpy.moments.ccp3_chol import calc_ccp3_2ba_with_selection
from ccpy.moments.crcc24 import (moments_aaaa_ijkl, moments_aaab_ijkl, moments_aabb_ijkl, moments_abbb_ijkl,
                                 moments_bbbb_ijkl)

# terminates the P-space keys of every occupied block (see ccp34_adaptive_loops)
_LAST_KEY = np.iinfo(np.int64).max


def _pspace_keys(excitations):
    """Sorted keys of the virtual indices of the P-space quadruples of each occupied block, keyed by
    the occupied indices (columns 5-8 of the t4_excitations rows), each terminated by _LAST_KEY."""
    x = np.asarray(excitations, dtype=np.int64)
    if np.all(x[0] == 1):  # empty spincase
        return {}
    keys = (x[:, 0] << 45) | (x[:, 1] << 30) | (x[:, 2] << 15) | x[:, 3]
    order = np.lexsort((keys, x[:, 7], x[:, 6], x[:, 5], x[:, 4]))
    occ, keys = x[order, 4:], keys[order]
    starts = np.flatnonzero(np.r_[True, np.any(occ[1:] != occ[:-1], axis=1)])
    ends = np.r_[starts[1:], len(keys)]
    return {tuple(occ[s]): np.append(keys[s:e], _LAST_KEY) for s, e in zip(starts, ends)}


def _central_moment_inputs(H4, T, P):
    """Occupied derivatives (see ccpy.methods.cim.occupied_derivative) of the HBar blocks and T2
    entering the CR-CC(2,4) moments: the moment is H T2 T2, so (P_1 + ... + P_4) M =
    M(dH, T, T) + [M(H, T + dT, T + dT) - M(H, T - dT, T - dT)] / 2, exactly."""
    from ccpy.methods.cim import occupied_derivative

    def d(x, name):  # +P on the occupied ket (last two) slots, -P on the occupied bra (first two) ones
        return occupied_derivative(x, P, lower=tuple(s for s in (2, 3) if name[s] == "o"),
                                   upper=tuple(s for s in (0, 1) if name[s] == "o"))

    blocks = {"aa": ("voov", "oooo", "vvvv"), "ab": ("voov", "ovov", "vovo", "ovvo", "oooo", "vvvv"),
              "bb": ("voov", "oooo", "vvvv")}
    dH = SimpleNamespace(**{spin: SimpleNamespace(**{n: d(getattr(getattr(H4, spin), n), n) for n in names})
                            for spin, names in blocks.items()})
    dT = {spin: d(getattr(T, spin), "vvoo") for spin in blocks}
    T_plus = SimpleNamespace(**{spin: getattr(T, spin) + dT[spin] for spin in blocks})
    T_minus = SimpleNamespace(**{spin: getattr(T, spin) - dT[spin] for spin in blocks})
    return dH, T_plus, T_minus


def calc_ccp4_with_selection(T, L, t4_excitations, H0, H4, system, num_add=1, use_RHF=False, min_thresh=0.0,
                              buffer_factor=2, cutoff=0.0, central=None):
    """
    CR-CC(2,4)-type quadruples correction (two-body approximation, variants A-D) restricted to the
    quadruples outside the P space ``t4_excitations``, keeping the num_add of largest |moment
    contribution| (variant D) above ``cutoff`` in a selection buffer of buffer_factor * num_add rows.
    H0 provides the Fock operator and H4 is the four-index CCSD HBar.

    With ``central`` (a CIM central projector P in the occupied basis; RHF only), the quadruples are
    ranked by their contributions to the central energy, l (P_1 + P_2 + P_3 + P_4) M / (4 D), whose
    sum (the central part of the D correction) is returned as well; the corrections are unchanged.

    Returns the corrections [A, B, C, D], the central D correction (None without ``central``), and
    the selection buffer (spinorbital quadruples and their ranking quantities, unsorted; unset rows
    are zero).
    """
    if central is not None:
        if not use_RHF:
            raise NotImplementedError("Central-projected quadruples selection needs an RHF reference.")
        dH, T_plus, T_minus = _central_moment_inputs(H4, T, central)
    noa, nob = system.noccupied_alpha, system.noccupied_beta
    d3aaa_v, d3aaa_o = aaa_H3_aaa_diagonal(T, H4, system)
    d3aab_v, d3aab_o = aab_H3_aab_diagonal(T, H4, system)
    d3abb_v, d3abb_o = abb_H3_abb_diagonal(T, H4, system)
    d3bbb_v, d3bbb_o = bbb_H3_bbb_diagonal(T, H4, system)

    num_add = int(num_add)
    moments = np.zeros(buffer_factor * num_add)
    quadruples_list = np.zeros((buffer_factor * num_add, 8), dtype=np.int32, order="F")
    nfill = 1

    def correction(spincase, kernel, moments_ijkl, blocks, args, extra=()):
        # blocks: (i, j, k, l) as in moments_ijkl and the occupied key of t4_excitations
        nonlocal moments, quadruples_list, nfill, cutoff
        pkeys = _pspace_keys(t4_excitations[spincase])
        empty = np.array([_LAST_KEY])
        delta = np.zeros(5)  # A, B, C, D, and the central part of D
        for (i, j, k, l), occ in blocks:
            x4 = [moments_ijkl(i, j, k, l, H4, T)]
            if central is not None:
                x4.append(moments_ijkl(i, j, k, l, dH, T) + 0.5 * (moments_ijkl(i, j, k, l, H4, T_plus)
                                                                   - moments_ijkl(i, j, k, l, H4, T_minus)))
            *d, moments, quadruples_list, nfill, cutoff = kernel(
                moments, quadruples_list, nfill, cutoff, i + 1, j + 1, k + 1, l + 1, *x4, *args,
                pkeys.get(occ, empty), num_add, min_thresh, buffer_factor, *extra)
            delta[:len(d)] += d
        return delta

    fock = (H0.a.oo, H0.a.vv, H0.b.oo, H0.b.vv)
    h1 = (H4.a.oo, H4.a.vv, H4.b.oo, H4.b.vv)
    h2a = (H4.aa.voov, H4.aa.oooo, H4.aa.vvvv, H4.aa.oovv)
    h2b = (H4.ab.voov, H4.ab.ovov, H4.ab.vovo, H4.ab.ovvo, H4.ab.oooo, H4.ab.vvvv, H4.ab.oovv)
    h2c = (H4.bb.voov, H4.bb.oooo, H4.bb.vvvv, H4.bb.oovv)

    def plus1(*idx):
        return tuple(x + 1 for x in idx)

    def kernel(name):
        return getattr(ccp34_adaptive_loops, name + ("_with_central_selection" if central is not None else "_with_selection"))

    delta_aaaa = correction("aaaa", kernel("ccp4a_ijkl"), moments_aaaa_ijkl,
                       ((o, plus1(*o)) for o in combinations(range(noa), 4)),
                       (T.aa, L.aa, H0.a.oo, H0.a.vv, H4.a.oo, H4.a.vv, *h2a, d3aaa_o, d3aaa_v))
    delta_aaab = correction("aaab", kernel("ccp4b_ijkl"), moments_aaab_ijkl,
                            (((i, j, k, l), plus1(i, j, k, l))
                             for (i, j, k), l in product(combinations(range(noa), 3), range(nob))),
                            (T.aa, T.ab, L.aa, L.ab, *fock, *h1, *h2a, *h2b, H4.bb.voov,
                             d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v))
    delta_aabb = correction("aabb", kernel("ccp4c_ijkl"), moments_aabb_ijkl,
                            (((i, j, k, l), plus1(i, j, k, l))
                             for (i, j), (k, l) in product(combinations(range(noa), 2), combinations(range(nob), 2))),
                            (T.aa, T.ab, T.bb, L.aa, L.ab, L.bb, *fock, *h1, *h2a, *h2b, *h2c,
                             d3aaa_o, d3aaa_v, d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v),
                            extra=(use_RHF,))
    if use_RHF:
        delta = 2.0 * delta_aaaa + 2.0 * delta_aaab + delta_aabb
    else:
        # abbb blocks: l alpha, i < j < k beta (occupied key l, i, j, k)
        delta_abbb = correction("abbb", kernel("ccp4d_ijkl"), moments_abbb_ijkl,
                                (((i, j, k, l), plus1(l, i, j, k))
                                 for l, (i, j, k) in product(range(noa), combinations(range(nob), 3))),
                                (T.ab, T.bb, L.ab, L.bb, *fock, *h1, H4.aa.voov, *h2b, *h2c,
                                 d3aab_o, d3aab_v, d3abb_o, d3abb_v, d3bbb_o, d3bbb_v))
        delta_bbbb = correction("bbbb", kernel("ccp4e_ijkl"), moments_bbbb_ijkl,
                                ((o, plus1(*o)) for o in combinations(range(nob), 4)),
                                (T.bb, L.bb, H0.b.oo, H0.b.vv, H4.b.oo, H4.b.vv, *h2c, d3bbb_o, d3bbb_v))
        delta = delta_aaaa + delta_aaab + delta_aabb + delta_abbb + delta_bbbb
    return delta[:4], (delta[4] if central is not None else None), quadruples_list, moments


def calc_ccp34_2ba_with_selection(T, L, t3_excitations, t4_excitations, corr_energy, H, H0, H4, system, num_add,
                                  use_RHF=False, min_thresh=0.0, buffer_factor=2, central=None):
    """
    Calculate the ground-state CC(P;Q)_D correction to the CCSDTQ(P) energy: the CC(P;3) triples
    correction of calc_ccp3_2ba_with_selection plus the CR-CC(2,4) quadruples correction (two-body
    approximation) restricted to the quadruples outside the P space ``t4_excitations``.

    The triples and quadruples are ranked together by |moment contribution|; the num_add largest
    are returned as spinorbital rows (odd indices alpha, even beta; see
    add_spinorbital_excitations_to_pspace), triples [a, b, c, i, j, k] and quadruples
    [a, b, c, d, i, j, k, l] separately. With use_RHF, only the aaa, aab, aaaa, aaab, and aabb
    excitations are ranked (aabb ones not after their spin flips).

    H and H0 are the DF CCSD HBar and bare Hamiltonian (triples), H4 the four-index CCSD HBar
    (quadruples). With ``central`` (CIM), both are ranked by their contributions to the central
    energy (see calc_ccp3_2ba_with_selection and calc_ccp4_with_selection).
    """
    e_ccp3, triples_list, moments3 = calc_ccp3_2ba_with_selection(
        T, L, t3_excitations, corr_energy, H, H0, system, num_add, use_RHF=use_RHF, min_thresh=min_thresh,
        buffer_factor=buffer_factor, central=central, return_moments=True)

    t_start = time.perf_counter()
    # a quadruple below the num_add-th largest triple cannot be selected
    delta, _, quadruples_list, moments = calc_ccp4_with_selection(
        T, L, t4_excitations, H0, H4, system, num_add, use_RHF=use_RHF, min_thresh=min_thresh,
        buffer_factor=buffer_factor, cutoff=abs(moments3[-1]) if num_add > 0 else 0.0, central=central)
    num_add = int(num_add)
    correction_D = delta[3]

    # rank the triples and quadruples together; unset rows have zero moments and sort last
    n3 = len(moments3)
    order = np.argsort(-np.abs(np.concatenate((moments3, moments))), kind="stable")[:num_add]
    triples_list = triples_list[order[order < n3]]
    quadruples_list = quadruples_list[order[order >= n3] - n3]

    minutes, seconds = divmod(time.perf_counter() - t_start, 60)
    print('   CC(P;4) Calculation Summary')
    print('   ---------------------------')
    print("   Total wall time: {:0.2f}m  {:0.2f}s".format(minutes, seconds))
    print("   CC(P;3)_D = {:>10.10f}".format(e_ccp3))
    print("   CC(P;3,4)_D = {:>10.10f}     δ(4)_D = {:>10.10f}".format(e_ccp3 + correction_D, correction_D))
    print("   Selected {} triples and {} quadruples\n".format(np.count_nonzero(np.any(triples_list, axis=1)),
                                                             np.count_nonzero(np.any(quadruples_list, axis=1))))
    return e_ccp3 + correction_D, triples_list, quadruples_list
