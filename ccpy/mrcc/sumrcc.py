'''
State-universal multireference CC (SU-MRCC) model space, reference Hamiltonians, and
effective Hamiltonian [Jeziorski & Monkhorst; Piecuch & Kowalski, IJMS 3, 676 (2002)],
for complete or general (incomplete) model spaces [Li & Paldus, JCP 119, 5320 (2003)].

A model space is a list of determinants, each an array of the N_alpha occupied alpha and
then the N_beta occupied beta correlated spatial orbitals, numbered from 1 (the convention
of ccpy.models.hilbert.Determinant). Orbitals occupied (empty) in every determinant are
core (virtual) for that spin; the others are active. Reference p is stored as its local
orbital lists (occ_a, unocc_a, occ_b, unocc_b) of 0-based orbitals ordered
[core, occupied active] and [unoccupied active, virtual], so T(p) and H(p) use the
ordinary sorted ccpy layout and the core-virtual block sits at the same place for every p.

Internal excitations take a reference into another one. Their amplitudes follow the C
conditions: t1 = 0, t2 = -A t1 t1 (kept in the T(p) arrays), and rank >= 3 amplitudes equal
to minus their disconnected clusters (internal_amplitudes; at most the number of active
electrons in rank), so that
<Phi_r|exp(T(q))|Phi_q> = 0 for r /= q. In a complete model space they all vanish.

Heff and the coupling terms come from one primitive, P(q->p)_K = <Phi_K(p)|exp(T(q))|Phi_q>
(get_projection), a recursion over the partitions of an excitation into cluster blocks that
holds for any model space and any number of active electrons; cluster_terms lists the terms
of any element explicitly.
'''

from itertools import combinations
from types import SimpleNamespace

import numpy as np

from ccpy.models.integrals import Integral
from ccpy.lib.core import sumrcc_loops

def get_cas_determinants(system, nact_electrons, nact_orbitals):
    """Determinants of CAS(nact_electrons, nact_orbitals) with the active orbitals right
    above the core, at the alpha and beta electron numbers of the system."""
    noa, nob = system.noccupied_alpha, system.noccupied_beta
    ncore = (system.nelectrons - nact_electrons) // 2
    nact_el_a, nact_el_b = noa - ncore, nob - ncore
    if (system.nelectrons - nact_electrons) % 2 or ncore < 0 or nact_el_b < 0 or nact_el_a > nact_orbitals \
            or ncore + nact_orbitals > system.norbitals:
        raise ValueError(f"CAS({nact_electrons},{nact_orbitals}) does not fit the system")
    core = list(range(1, ncore + 1))
    active = range(ncore + 1, ncore + nact_orbitals + 1)
    return [np.array(core + list(a) + core + list(b)) for a in combinations(active, nact_el_a)
            for b in combinations(active, nact_el_b)]


def get_model_space(system, determinants):
    """Reference orbital lists, core/virtual counts, and internal excitations of the model
    space spanned by `determinants` (see the module docstring)."""
    noa, nob, norb = system.noccupied_alpha, system.noccupied_beta, system.norbitals
    dets = [np.asarray(d, dtype=int) for d in determinants]
    if not dets:
        raise ValueError("The model space needs at least one determinant")
    occ = []
    for d in dets:
        a, b = set(d[:noa] - 1), set(d[noa:] - 1)
        if len(d) != noa + nob or len(a) != noa or len(b) != nob or min(d) < 1 or max(d) > norb:
            raise ValueError(f"Determinant {list(d)} must list {noa} distinct alpha and then {nob} distinct beta "
                             f"orbitals numbered 1..{norb}")
        occ.append((a, b))
    if len(set((frozenset(a), frozenset(b)) for a, b in occ)) != len(occ):
        raise ValueError("The model space contains a determinant twice")
    core = [set.intersection(*[o[s] for o in occ]) for s in (0, 1)]
    virt = [set(range(norb)) - set.union(*[o[s] for o in occ]) for s in (0, 1)]
    refs = []
    for a, b in occ:
        lists = []
        for s, o in ((0, a), (1, b)):
            lists += [sorted(core[s]) + sorted(o - core[s]), sorted(set(range(norb)) - o - virt[s]) + sorted(virt[s])]
        refs.append(tuple(np.array(x, dtype=np.int32) for x in lists))
    ms = SimpleNamespace(determinants=dets, references=refs, ncore=(len(core[0]), len(core[1])),
                         nvirt=(len(virt[0]), len(virt[1])))
    # longest relative excitation between references (at least 3, for the MM triples)
    spin_orbitals = [a | {norb + g for g in b} for a, b in occ]
    ms.max_rank = max([3] + [len(x - y) for x in spin_orbitals for y in spin_orbitals])
    # core (occupied in every reference) and virtual (empty in every one) spin orbitals
    ms.inactive = np.zeros(2 * norb, dtype=np.int32)
    for s in (0, 1):
        for g in core[s] | virt[s]:
            ms.inactive[s * norb + g] = 1
    ms.symmetry = get_determinant_symmetries(system, occ)
    ms.internal, ms.higher = zip(*[get_internal_excitations(ms, p, system) for p in range(len(refs))])
    return ms


def get_determinant_symmetries(system, occ):
    """Point-group irrep number of each determinant (XOR of its occupied orbitals), or None
    when the system carries no orbital symmetries."""
    labels = getattr(system, "orbital_symmetries", None)
    table = getattr(system, "point_group_irrep_to_number", None)
    if not labels or not table or len(set(labels)) < 2:
        return None
    out = []
    for a, b in occ:
        sym = 0
        for g in list(a) + list(b):
            sym ^= table[labels[g]]
        out.append(sym)
    return out


def determinant_key(det, system):
    """Hashable occupation (alpha, beta) of a determinant in the 1-based list format."""
    det = np.asarray(det)
    return tuple(sorted(det[:system.noccupied_alpha])), tuple(sorted(det[system.noccupied_alpha:]))


def get_external_estimates(e1int, e2int, system, determinants):
    """For each determinant Phi: <Phi|H|Phi>, and its singles and doubles at the orbital-energy
    estimate <Phi|H|Phi> + sum(eps_particles) - sum(eps_holes) (the Jacobi denominators of
    reference Phi), as ascending energies with the keys (determinant_key) of the excited
    determinants. Only the len(determinants) + 1 lowest are kept: enough to find the lowest
    one outside any subset of the determinants. e1int and e2int (physics notation) span all
    orbitals, frozen ones included. Each entry is (energy, excited energies, excited keys, own key)."""
    nf, norb, noa = system.nfrozen, system.norbitals, system.noccupied_alpha
    h = np.diagonal(e1int)
    B = as_df(e2int)
    d = np.einsum("xpp->xp", B)
    J = d.T @ d  # <pj|pj> = (pp|jj)
    K = np.einsum("xpj,xpj->pj", B, B)  # <pj|jp> = (pj|jp)
    nkeep = len(determinants) + 1
    estimates = []
    for det in determinants:
        det = np.asarray(det)
        occ = [list(range(nf)) + [nf + g - 1 for g in sorted(det[:noa])],
               list(range(nf)) + [nf + g - 1 for g in sorted(det[noa:])]]
        eps = [h + (J - K)[:, occ[s]].sum(axis=1) + J[:, occ[1 - s]].sum(axis=1) for s in (0, 1)]
        e_ref = 0.5 * sum((h[occ[s]] + eps[s][occ[s]]).sum() for s in (0, 1)) + system.nuclear_repulsion
        # correlated occupied / unoccupied orbitals (1-based) and their single excitation energies
        o = [np.array(sorted(det[:noa])), np.array(sorted(det[noa:]))]
        u = [np.array([g for g in range(1, norb + 1) if g not in set(x)]) for x in o]
        e1 = [eps[s][nf + u[s] - 1][:, None] - eps[s][nf + o[s] - 1][None, :] for s in (0, 1)]
        energies, moves = [], []  # moves: (spin, particle, hole) of each single excitation
        for s in (0, 1):
            for a in range(len(u[s])):
                for i in range(len(o[s])):
                    energies.append(e1[s][a, i])
                    moves.append(((s, u[s][a], o[s][i]),))
        for s1, s2 in ((0, 0), (0, 1), (1, 1)):
            d = e1[s1][:, :, None, None] + e1[s2][None, None, :, :]
            if s1 == s2:  # a < b and i < j
                nu, no = d.shape[0], d.shape[1]
                d = np.where((np.arange(nu)[:, None, None, None] < np.arange(nu)[None, None, :, None])
                             & (np.arange(no)[None, :, None, None] < np.arange(no)[None, None, None, :]), d, np.inf)
            flat = d.ravel()
            take = np.argsort(flat)[:nkeep]
            for t in take[np.isfinite(flat[take])]:
                a, i, b, j = np.unravel_index(t, d.shape)
                energies.append(flat[t])
                moves.append(((s1, u[s1][a], o[s1][i]), (s2, u[s2][b], o[s2][j])))
        order = np.argsort(energies)[:nkeep]
        keys = []
        for t in order:
            a_occ, b_occ = set(o[0]), set(o[1])
            for s, part, hole in moves[t]:
                target = a_occ if s == 0 else b_occ
                target.discard(hole)
                target.add(part)
            keys.append((tuple(sorted(a_occ)), tuple(sorted(b_occ))))
        estimates.append((e_ref, e_ref + np.array(energies)[order], keys, determinant_key(det, system)))
    return estimates


def get_lowest_external(estimates, keep):
    """Lowest estimated energy of a single or double of the determinants with indices `keep`
    that lies outside the model space they span."""
    inside = {estimates[p][3] for p in keep}
    lowest = np.inf
    for p in keep:
        for energy, key in zip(estimates[p][1], estimates[p][2]):
            if key not in inside:
                lowest = min(lowest, energy)
                break
    return lowest


def prune_intruders(determinants, estimates, gap=0.0):
    """Drop determinants, highest energy first, until every single or double of the remaining
    ones outside their model space lies at least `gap` (Eh) above the highest of them, at the
    orbital-energy estimate (gap = 0 removes what check_intruders flags). Returns the indices
    of the determinants kept and dropped. A positive gap is necessary for the Jacobi
    iterations to converge, but no value guarantees it."""
    keep = list(range(len(determinants)))
    dropped = []
    while len(keep) > 1:
        top = max(keep, key=lambda p: estimates[p][0])
        if get_lowest_external(estimates, keep) >= estimates[top][0] + gap:
            break
        keep.remove(top)
        dropped.append(top)
    return keep, dropped


def get_internal_excitations(ms, p, system):
    """Internal excitations of reference p: boolean masks of the T(p) blocks a, b, aa, ab, bb
    that take Phi_p into another reference, and the rank >= 3 ones as Fortran arrays
    (holes, particles, ranks), spin orbitals ascending, sorted by rank."""
    norb = system.norbitals
    occ_a, unocc_a, occ_b, unocc_b = ms.references[p]
    no, nu = (len(occ_a), len(occ_b)), (len(unocc_a), len(unocc_b))
    lo = {s * norb + g: k for s, lst in ((0, occ_a), (1, occ_b)) for k, g in enumerate(lst)}
    lu = {s * norb + g: k for s, lst in ((0, unocc_a), (1, unocc_b)) for k, g in enumerate(lst)}
    spin_orbitals = lambda ref: set(ref[0]) | {norb + g for g in ref[2]}
    masks = {"a": np.zeros((nu[0], no[0]), bool), "b": np.zeros((nu[1], no[1]), bool),
             "aa": np.zeros((nu[0], nu[0], no[0], no[0]), bool), "ab": np.zeros((nu[0], nu[1], no[0], no[1]), bool),
             "bb": np.zeros((nu[1], nu[1], no[1], no[1]), bool)}
    higher = []
    for r, ref in enumerate(ms.references):
        if r == p:
            continue
        holes = sorted(spin_orbitals(ms.references[p]) - spin_orbitals(ref))
        parts = sorted(spin_orbitals(ref) - spin_orbitals(ms.references[p]))
        h, x = [lo[s] for s in holes], [lu[s] for s in parts]
        spins = "".join("ab"[s // norb] for s in holes)
        if len(holes) == 1:
            masks[spins][x[0], h[0]] = True
        elif len(holes) == 2:
            masks[spins][x[0], x[1], h[0], h[1]] = True
            if spins != "ab":
                masks[spins][x[1], x[0], h[0], h[1]] = masks[spins][x[0], x[1], h[1], h[0]] = True
                masks[spins][x[1], x[0], h[1], h[0]] = True
        else:
            higher.append((len(holes), holes, parts))
    higher.sort(key=lambda e: e[0])
    n = max(len(higher), 1)  # Fortran needs one column; rank 0 marks it empty
    int_h, int_p, int_k = (np.zeros((ms.max_rank, n), np.int32), np.zeros((ms.max_rank, n), np.int32),
                           np.zeros(n, np.int32))
    for e, (k, holes, parts) in enumerate(higher):
        int_k[e], int_h[:k, e], int_p[:k, e] = k, holes, parts
    return masks, (np.asfortranarray(int_h), np.asfortranarray(int_p), int_k)


def get_internal_amplitudes(T_q, ms, q, system):
    """C-condition amplitudes of the rank >= 3 internal excitations of reference q."""
    int_h, int_p, int_k = ms.higher[q]
    return sumrcc_loops.internal_amplitudes(T_q.a, T_q.b, T_q.aa, T_q.ab, T_q.bb, int_h, int_p, int_k,
                                            ms.inactive, int(can_prune(T_q, None, ms, system)), *ms.references[q],
                                            norb=system.norbitals)


def can_prune(T_q, int_amp_q, ms, system):
    """True when every nonzero block of T(q) contains a core hole or a virtual particle: no
    all-active t1/t2 and no rank >= 3 internal amplitudes (always so in a complete model
    space). The recursion may then discard partitions that cannot satisfy this."""
    if int_amp_q is not None and np.any(int_amp_q):
        return False
    oa, ob = slice(ms.ncore[0], None), slice(ms.ncore[1], None)
    ua = slice(0, system.nunoccupied_alpha - ms.nvirt[0])
    ub = slice(0, system.nunoccupied_beta - ms.nvirt[1])
    return not (np.any(T_q.a[ua, oa]) or np.any(T_q.b[ub, ob]) or np.any(T_q.aa[ua, ua, oa, oa])
                or np.any(T_q.ab[ua, ub, oa, ob]) or np.any(T_q.bb[ub, ub, ob, ob]))


def fortran_internal(ms, q, int_amp_q):
    """The nonzero rank >= 3 internal amplitudes of reference q in Fortran form (one empty
    entry when there are none)."""
    int_h, int_p, int_k = ms.higher[q]
    keep = np.flatnonzero(int_amp_q)
    if len(keep) == 0:
        return (np.zeros((ms.max_rank, 1), np.int32, order="F"), np.zeros((ms.max_rank, 1), np.int32, order="F"),
                np.zeros(1, np.int32), np.zeros(1))
    return np.asfortranarray(int_h[:, keep]), np.asfortranarray(int_p[:, keep]), int_k[keep], int_amp_q[keep]


def as_df(e2int):
    """DF/Cholesky vectors B (naux, n, n) with <pq|rs> = sum_x B[x,p,r] B[x,q,s]; e2int may already
    be B, or the four-index physics-notation integrals (then decomposed exactly by diagonalization)."""
    if e2int.ndim == 3:
        return e2int
    # shortcut: dense n^2 x n^2 diagonalization, only for the small four-index inputs of the tests
    n = e2int.shape[0]
    w, U = np.linalg.eigh(e2int.transpose(0, 2, 1, 3).reshape(n * n, n * n))
    keep = w > 1e-12 * w.max()
    return (U[:, keep] * np.sqrt(w[keep])).T.reshape(-1, n, n)


def chol_block(chol, key, name, idx=(slice(None),) * 4):
    """Two-body block `name` (e.g. "vovv") of spin case key: <pq||rs> for aa/bb, <pq|rs> for ab,
    built from the Cholesky vectors chol.a/b.{oo,ov,vo,vv} (bare or T1-transformed), on the index
    slices idx."""
    p, q, r, s = name
    c1, c2 = getattr(chol, key[0]), getattr(chol, key[1])
    v = np.einsum("xpr,xqs->pqrs", getattr(c1, p + r)[:, idx[0], idx[2]], getattr(c2, q + s)[:, idx[1], idx[3]],
                  optimize=True)
    if key[0] == key[1]:
        v -= np.einsum("xps,xqr->pqrs", getattr(c1, p + s)[:, idx[0], idx[3]], getattr(c1, q + r)[:, idx[1], idx[2]],
                       optimize=True)
    return v


def get_reference_hamiltonian(e1int, e2int, system, reference):
    """H_N normal ordered with respect to the reference determinant, in the orbital order of the
    reference, and its total energy <Phi_p|H|Phi_p>. e1int and e2int (DF vectors B, or four-index
    physics-notation integrals, see as_df) span all orbitals, frozen ones included.
    As in ccpy.models.integrals.getCholeskyHamiltonian, only the Fock matrix, the Cholesky vectors
    (H.chol, in the orbital order of the reference) and the two-body blocks with at most two
    unoccupied indices are stored; materialize gives the rest."""
    B = as_df(e2int)
    occ_a, unocc_a, occ_b, unocc_b = reference
    nf = system.nfrozen
    frozen = np.arange(nf, dtype=int)
    oa = np.concatenate((frozen, nf + np.asarray(occ_a, dtype=int)))
    ob = np.concatenate((frozen, nf + np.asarray(occ_b, dtype=int)))
    # <p|f|q> = <p|z|q> + sum_i [(pq|ii) - (pi|iq)] (same spin) + sum_i (pq|ii) (other spin)
    J = lambda o: np.einsum("xpq,x->pq", B, np.einsum("xii->x", B[:, o][:, :, o]), optimize=True)
    K = lambda o: np.einsum("xpi,xiq->pq", B[:, :, o], B[:, o, :], optimize=True)
    f = {"a": e1int + J(oa) + J(ob) - K(oa), "b": e1int + J(oa) + J(ob) - K(ob)}
    # <Phi_p|H|Phi_p> = 1/2 sum_i (z_ii + f_ii)
    e_ref = 0.5 * (np.trace((e1int + f["a"])[np.ix_(oa, oa)]) + np.trace((e1int + f["b"])[np.ix_(ob, ob)]))
    idx = {"a": {"o": nf + np.asarray(occ_a, dtype=int), "v": nf + np.asarray(unocc_a, dtype=int)},
           "b": {"o": nf + np.asarray(occ_b, dtype=int), "v": nf + np.asarray(unocc_b, dtype=int)}}
    H = Integral.from_empty(system, 2, use_none=True)
    H.chol = Integral.from_empty(system, 1, use_none=True)
    for spin in ("a", "b"):
        for name in getattr(H, spin).slices:
            rows, cols = idx[spin][name[0]], idx[spin][name[1]]
            setattr(getattr(H, spin), name, np.asfortranarray(f[spin][np.ix_(rows, cols)]))
            setattr(getattr(H.chol, spin), name, np.ascontiguousarray(B[:, rows][:, :, cols]))
    for key in ("aa", "ab", "bb"):
        block = getattr(H, key)
        for name in block.slices:
            if name.count("v") <= 2:
                setattr(block, name, np.asfortranarray(chol_block(H.chol, key, name)))
    return H, e_ref + system.nuclear_repulsion


def materialize(H):
    """H with all two-body blocks dense (those with three or four unoccupied indices built from
    H.chol); for the routines that need them (MM-SUCC(2,3), dense reference implementations)."""
    from copy import copy
    H = copy(H)
    for key in ("aa", "ab", "bb"):
        block = copy(getattr(H, key))
        for name in block.slices:
            if getattr(block, name) is None:
                setattr(block, name, np.asfortranarray(chol_block(H.chol, key, name)))
        setattr(H, key, block)
    return H


def get_projection(T_q, int_amp_q, ms, p, q, system):
    """P(q->p)_K = <Phi_K(p)|exp(T(q))|Phi_q> on the singles and doubles of reference p,
    returned as (P1a, P1b, P2a, P2b, P2c) in the orbital order of p. int_amp_q are the
    rank >= 3 internal amplitudes of q (get_internal_amplitudes)."""
    return sumrcc_loops.project_cc_wavefunction(T_q.a, T_q.b, T_q.aa, T_q.ab, T_q.bb,
                                                *fortran_internal(ms, q, int_amp_q), ms.inactive,
                                                int(can_prune(T_q, int_amp_q, ms, system)),
                                                *ms.references[p], *ms.references[q], norb=system.norbitals)


def symmetry_allowed(ms, p, q):
    """References of different point-group symmetry have Heff_pq = 0 and no coupling."""
    return ms.symmetry is None or ms.symmetry[p] == ms.symmetry[q]


def get_heff_element(P, H):
    """<Phi_p|H|P(q->p)> for p != q (the coefficient of Phi_p in exp(T(q))|Phi_q> is zero)."""
    P1a, P1b, P2a, P2b, P2c = P
    return (np.einsum("me,em->", H.a.ov, P1a, optimize=True)
            + np.einsum("me,em->", H.b.ov, P1b, optimize=True)
            + 0.25 * np.einsum("mnef,efmn->", H.aa.oovv, P2a, optimize=True)
            + np.einsum("mnef,efmn->", H.ab.oovv, P2b, optimize=True)
            + 0.25 * np.einsum("mnef,efmn->", H.bb.oovv, P2c, optimize=True))


def get_projections(T, ms, system, amps=None):
    """P[p][q] = P(q->p) for all p != q (None on the diagonal and for symmetry-forbidden
    pairs). amps are the rank >= 3 internal amplitudes of every reference, computed here if
    not given."""
    nref = len(ms.references)
    if amps is None:
        amps = [get_internal_amplitudes(T[q], ms, q, system) for q in range(nref)]
    return [[get_projection(T[q], amps[q], ms, p, q, system) if q != p and symmetry_allowed(ms, p, q) else None
             for q in range(nref)] for p in range(nref)]


def get_effective_hamiltonian(T, H, e_ref, ms, system, P=None):
    """Heff_pq = <Phi_p|H exp(T(q))|Phi_q> for the cluster operators T[q] of all references,
    reference Hamiltonians H[p], and reference energies e_ref[p]. P are the projections
    from get_projections, computed here if not given."""
    from ccpy.energy.cc_energy import get_cc_energy

    if P is None:
        P = get_projections(T, ms, system)
    nref = len(ms.references)
    heff = np.zeros((nref, nref))
    for p in range(nref):
        heff[p, p] = e_ref[p] + get_cc_energy(T[p], H[p])
        for q in range(nref):
            if P[p][q] is not None:
                heff[p, q] = get_heff_element(P[p][q], H[p])
    return heff


def get_heff_and_couplings(T, H, e_ref, ms, system, amps, max_bytes=2e9, free=None):
    """Heff and, for every reference p, C(p) = sum_{q != p} Heff_qp P(q->p) (the vector whose
    e^{-T(p)} transform gives the coupling terms), as (C1a, C1b, C2a, C2b, C2c). The M(M-1)
    projections are kept in memory when they fit in max_bytes; otherwise they are computed
    twice (once for Heff, once for C) and never stored. free (boolean masks of the a, b, aa,
    ab, bb blocks) limits C, and the stored projections, to those entries (zero elsewhere);
    Heff always uses the whole projection."""
    from ccpy.energy.cc_energy import get_cc_energy

    nref = len(ms.references)
    keys = ("a", "b", "aa", "ab", "bb")
    if free is None:
        free = {key: np.ones(getattr(T[0], key).shape, bool) for key in keys}
    masks = [free[key] for key in keys]
    zero = lambda p: [np.zeros_like(getattr(T[p], k)) for k in keys]
    pairs = [(p, q) for p in range(nref) for q in range(nref) if q != p and symmetry_allowed(ms, p, q)]
    if len(pairs) * sum(int(m.sum()) for m in masks) * 8 <= max_bytes:
        heff = np.diag([e_ref[p] + get_cc_energy(T[p], H[p]) for p in range(nref)])
        P = {}
        for p, q in pairs:
            full = get_projection(T[q], amps[q], ms, p, q, system)
            heff[p, q] = get_heff_element(full, H[p])
            P[p, q] = [x[m] for x, m in zip(full, masks)]
        C = [zero(p) for p in range(nref)]
        for p, q in pairs:
            for k in range(5):
                C[p][k][masks[k]] += heff[q, p] * P[p, q][k]
        return heff, C
    heff = np.diag([e_ref[p] + get_cc_energy(T[p], H[p]) for p in range(nref)])
    for p, q in pairs:
        heff[p, q] = get_heff_element(get_projection(T[q], amps[q], ms, p, q, system), H[p])
    C = [zero(p) for p in range(nref)]
    for p, q in pairs:
        P = get_projection(T[q], amps[q], ms, p, q, system)
        for k in range(5):
            C[p][k][masks[k]] += heff[q, p] * P[k][masks[k]]
    return heff, C


def cluster_terms(holes, particles, norb, internal=()):
    """The terms the Fortran recursion (sumrcc_loops.cluster_coefficient) sums for the
    coefficient of a+_{P1}..a+_{Pn} a_{Hn}..a_{H1}|Phi_q> in exp(T(q))|Phi_q>, holes H and
    particles P given as spin orbitals of q (alpha g -> g, beta g -> norb + g), ascending.

    Each term is (sign, blocks): a product of cluster amplitudes, one per block (holes,
    particles) of a partition of (H, P). Blocks of one or two holes are t1/t2 amplitudes
    (spin conserving); `internal` lists the (holes, particles) of the rank >= 3 internal
    excitations whose C-condition amplitudes may also appear. Use format_cluster_terms to
    print them, e.g. for a projection element of interest."""
    holes, particles = tuple(holes), tuple(particles)
    if not holes:
        return [(1, ())]
    spin = lambda s: s // norb
    h1, rest_h = holes[0], holes[1:]
    terms = []

    def extend(block, sign, rh, rp):
        for s, blocks in cluster_terms(rh, rp, norb, internal):
            terms.append((sign * s, (block,) + blocks))

    for j, pj in enumerate(particles):  # T1 block (H1 -> Pj)
        if spin(pj) == spin(h1):
            extend(((h1,), (pj,)), (-1) ** j, rest_h, particles[:j] + particles[j + 1:])
    for k in range(1, len(holes)):  # T2 block (H1 Hk -> Pj Pl)
        for j in range(len(particles)):
            for l in range(j + 1, len(particles)):
                if sorted((spin(particles[j]), spin(particles[l]))) != sorted((spin(h1), spin(holes[k]))):
                    continue
                if spin(h1) != spin(holes[k]) and spin(particles[j]) != 0:
                    continue  # mixed spin: alpha particle first, as t2b(a, b, i, j)
                rh = holes[1:k] + holes[k + 1:]
                rp = tuple(p for m, p in enumerate(particles) if m not in (j, l))
                extend(((h1, holes[k]), (particles[j], particles[l])), (-1) ** (j + l + k), rh, rp)
    for bh, bp in internal:  # internal blocks of rank >= 3 starting at H1
        if len(bh) < 3 or bh[0] != h1 or not set(bh) <= set(holes) or not set(bp) <= set(particles):
            continue
        parity = sum(holes.index(x) - m + particles.index(y) - m for m, (x, y) in enumerate(zip(bh, bp)))
        rh = tuple(x for x in holes if x not in bh)
        rp = tuple(y for y in particles if y not in bp)
        extend((tuple(bh), tuple(bp)), (-1) ** parity, rh, rp)
    return terms


def format_cluster_terms(terms, norb):
    """Readable form of cluster_terms, e.g. '+ t1(4a;1a) t2(5a,6b;2a,2b)' (particles;holes,
    1-based spatial orbitals with spin labels; tN for internal blocks of rank N)."""
    label = lambda s: f"{s % norb + 1}{'ab'[s // norb]}"
    lines = []
    for sign, blocks in terms:
        factors = " ".join(f"t{len(h)}({','.join(map(label, p))};{','.join(map(label, h))})" for h, p in blocks)
        lines.append(f"{'+' if sign > 0 else '-'} {factors or '1'}")
    return "\n".join(lines)
