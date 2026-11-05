'''
Method of moments of the SU-MRCC equations, MM-SUCC(2,3) [Piecuch & Kowalski, IJMS 3, 676
(2002), Eqs. 46-54]: noniterative triples corrections to the SU-MRCCSD roots,

    E_mu(2,3) = E_mu + sum_p c_pmu <Psi_mu|Gamma_3(p)|Phi_p> / <Psi_mu|Psi_mu^SU-MRCCSD>,

with the triples moments of the SU-MRCC equations of each reference p,

    Gamma_3(p)_K = <K(p)|(H e^{T(p)})_C|Phi_p> - sum_{q != p} <K(p)|e^{-T(p)} e^{T(q)}|Phi_q> Heff_qp,

over the external triples K(p), and the MRMBPT(2)-like trial wave function (the multireference
analog of CR-CCSD(T))

    Psi_mu = sum_q c_qmu (1 + T1(q) + T2(q) + Z3(q))|Phi_q>,
    Z3(q) = R3(q) [(V_N T2(q))_C + (V_N T1(q))_DC],

R3(q) the orbital-energy denominators of reference q, built from the converged SU-MRCCSD
amplitudes (internal excitations, which lead to other references, are left out of Psi_mu).
c_mu are the right eigenvectors of Heff. For a single reference this is CR-CCSD(T).

mmsucc23 evaluates everything on the fly over the triples of each reference (sumrcc_loops.
mm23_reference), at the CR-CC(2,3)-like cost o^3 v^3 (o + v) per reference plus the
recursion for the projections of the other references; mmsucc23_dense is the dense-array
reference implementation used in the tests.
'''
from copy import deepcopy
from itertools import permutations

import numpy as np

from ccpy.cc import ccsdt
from ccpy.hbar.hbar_ccs import get_pre_ccs_intermediates, get_ccs_intermediates_opt
from ccpy.hbar.hbar_ccsd import get_ccsd_intermediates
from ccpy.lib.core import sumrcc_loops
from ccpy.models.integrals import Integral
from ccpy.models.operators import ClusterOperator
from ccpy.mrcc.sumrcc import chol_block, materialize, get_internal_amplitudes, get_projection, fortran_internal, can_prune, symmetry_allowed
from ccpy.mrcc.sumrccsd import get_active_slices, get_internal_t3_t4, _term

T3_KEYS = ("aaa", "aab", "abb", "bbb")
# sum over unique triples of fully antisymmetric arrays
T3_WEIGHTS = {"aaa": 1 / 36, "aab": 1 / 4, "abb": 1 / 4, "bbb": 1 / 36}
T2_WEIGHTS = {"aa": 0.25, "ab": 1.0, "bb": 0.25}


def _signed_permutations(axes):
    for perm in permutations(axes):
        inv = sum(1 for x in range(len(perm)) for y in range(x + 1, len(perm)) if perm[x] > perm[y])
        yield perm, (-1) ** inv


def antisymmetrize(x, key):
    """sum over signed permutations of the like-spin particle and hole indices of a triples
    array (fills an array holding only its unique entries)."""
    groups = {"aaa": ((0, 1, 2), (3, 4, 5)), "aab": ((0, 1), (3, 4)), "abb": ((1, 2), (4, 5)),
              "bbb": ((0, 1, 2), (3, 4, 5))}[key]
    out = np.zeros_like(x)
    for pp, sp in _signed_permutations(groups[0]):
        for ph, sh in _signed_permutations(groups[1]):
            axes = list(range(6))
            for src, dst in zip(groups[0], pp):
                axes[src] = dst
            for src, dst in zip(groups[1], ph):
                axes[src] = dst
            out += sp * sh * np.transpose(x, axes)
    return out


def _dot(x, y, weights):
    return sum(w * np.vdot(x[k], y[k]) for k, w in weights.items())


def triples_denominators(H):
    n = np.newaxis
    ea_o, ea_v = np.diagonal(H.a.oo), np.diagonal(H.a.vv)
    eb_o, eb_v = np.diagonal(H.b.oo), np.diagonal(H.b.vv)
    def d(v1, v2, v3, o1, o2, o3):
        return (o1[n, n, n, :, n, n] + o2[n, n, n, n, :, n] + o3[n, n, n, n, n, :]
                - v1[:, n, n, n, n, n] - v2[n, :, n, n, n, n] - v3[n, n, :, n, n, n])
    return {"aaa": d(ea_v, ea_v, ea_v, ea_o, ea_o, ea_o), "aab": d(ea_v, ea_v, eb_v, ea_o, ea_o, eb_o),
            "abb": d(ea_v, eb_v, eb_v, ea_o, eb_o, eb_o), "bbb": d(eb_v, eb_v, eb_v, eb_o, eb_o, eb_o)}


def _t3_residuals(T, X, H0):
    """<K3|...|Phi> of ccsdt.update_t3a-d for the operator T (order 3), as full antisymmetric
    arrays (the step returned times the denominators)."""
    D = triples_denominators(H0)
    out = {}
    for key, fn in zip(T3_KEYS, (ccsdt.update_t3a, ccsdt.update_t3b, ccsdt.update_t3c, ccsdt.update_t3d)):
        Tc, dT = deepcopy(T), deepcopy(T)
        fn(Tc, dT, X, H0, 0.0)
        out[key] = getattr(dT, key) * D[key]
    return out


def _order3(T, system, t3=None):
    """ClusterOperator of order 3 with T1, T2 of T and T3 (dense, or zero)."""
    T3 = ClusterOperator(system, 3)
    for key in ("a", "b", "aa", "ab", "bb"):
        setattr(T3, key, getattr(T, key).copy())
    for key in T3_KEYS:
        setattr(T3, key, np.zeros_like(getattr(T3, key)) if t3 is None else t3[key])
    return T3


def get_z3(T, H, system):
    """Z3 = R3 [(V_N T2)_C + (V_N T1)_DC] of a reference (full antisymmetric arrays), and
    (V_N T2)_C."""
    H_bare = deepcopy(H)
    H_bare.a.ov[...] = 0.0  # ccsdt subtracts f_ov t2 from Hbar(vooo); (V_N T2)_C has no F_N
    H_bare.b.ov[...] = 0.0
    W = _t3_residuals(_order3(T, system), H_bare, H)
    v = {"aa": H.aa.vvoo, "ab": H.ab.vvoo, "bb": H.bb.vvoo}
    Y = {}
    x = np.einsum("ck,abij->abcijk", T.a, v["aa"], optimize=True)
    Y["aaa"] = _p3(x)
    x = np.einsum("ck,abij->abcijk", T.b, v["bb"], optimize=True)
    Y["bbb"] = _p3(x)
    x = np.einsum("ai,bcjk->abcijk", T.a, v["ab"], optimize=True)
    Y["aab"] = (np.einsum("ck,abij->abcijk", T.b, v["aa"], optimize=True)
                + x - x.transpose(1, 0, 2, 3, 4, 5) - x.transpose(0, 1, 2, 4, 3, 5) + x.transpose(1, 0, 2, 4, 3, 5))
    x = np.einsum("bj,acik->abcijk", T.b, v["ab"], optimize=True)
    Y["abb"] = (np.einsum("ai,bcjk->abcijk", T.a, v["bb"], optimize=True)
                + x - x.transpose(0, 2, 1, 3, 4, 5) - x.transpose(0, 1, 2, 3, 5, 4) + x.transpose(0, 2, 1, 3, 5, 4))
    D = triples_denominators(H)
    return {k: (W[k] + Y[k]) / D[k] for k in T3_KEYS}, W


def _p3(x):
    """P(c/ab) P(k/ij) x_abcijk for x antisymmetric in (ab) and (ij)."""
    xp = x - x.transpose(2, 1, 0, 3, 4, 5) - x.transpose(0, 2, 1, 3, 4, 5)
    return xp - xp.transpose(0, 1, 2, 5, 4, 3) - xp.transpose(0, 1, 2, 3, 5, 4)


def _embed_active(t34, system, act, T):
    """Internal T3 (dense over active orbitals) embedded in full triples arrays."""
    t3 = {k: np.zeros_like(getattr(ClusterOperator(system, 3), k)) for k in T3_KEYS}
    if t34 is None:
        return t3
    for key in T3_KEYS:
        if key in t34:
            sl = tuple(act[(s, "v")] for s in key) + tuple(act[(s, "o")] for s in key)
            t3[key][sl] = t34[key]
    return t3


def get_direct_moments(T, H, t34, act, system):
    """<K3(p)|(H e^{T(p)})_C|Phi_p> with T(p) = T1 + T2 + internal T3 + internal T4."""
    T3 = _order3(T, system, _embed_active(t34, system, act, T))
    X = Integral.from_empty(system, 2, use_none=True)
    X = get_pre_ccs_intermediates(X, T3, H, False)
    X = get_ccs_intermediates_opt(X, T3, H, False)
    X = get_ccsd_intermediates(T3, X, H, False)
    M = _t3_residuals(T3, X, H)
    if t34 is not None and any(len(k) == 4 for k in t34):
        # internal T4 terms (ccpy.cc.ccsdtq), written for the antisymmetrizers of the T3 updates
        out = {k: np.zeros_like(M[k]) for k in T3_KEYS}
        for coef, sub, blk, case in _T4_TO_T3:
            if case in t34:
                h = getattr(getattr(X, blk[0]), blk[1])
                _term(out, coef, sub, h, blk, t34[case], case, act)
        for k in T3_KEYS:
            M[k] = M[k] + antisymmetrize(out[k], k)
    return M


_T4_TO_T3 = [  # (coef, subscripts, (block, kinds), T4 spin case), from ccpy.cc.ccsdtq.update_t3a-d
    (1 / 36, "me,abceijkm->abcijk", ("a", "ov"), "aaaa"), (1 / 36, "me,abceijkm->abcijk", ("b", "ov"), "aaab"),
    (1 / 24, "cnef,abefijkn->abcijk", ("aa", "vovv"), "aaaa"), (1 / 12, "cnef,abefijkn->abcijk", ("ab", "vovv"), "aaab"),
    (-1 / 24, "mnkf,abcfijmn->abcijk", ("aa", "ooov"), "aaaa"), (-1 / 12, "mnkf,abcfijmn->abcijk", ("ab", "ooov"), "aaab"),
    (0.25, "me,abecijmk->abcijk", ("a", "ov"), "aaab"), (0.25, "me,abecijmk->abcijk", ("b", "ov"), "aabb"),
    (-0.25, "mnjf,abfcimnk->abcijk", ("aa", "ooov"), "aaab"), (-0.5, "mnjf,abfcimnk->abcijk", ("ab", "ooov"), "aabb"),
    (-0.25, "nmfk,abfcijnm->abcijk", ("ab", "oovo"), "aaab"), (-0.125, "mnkf,abfcijnm->abcijk", ("bb", "ooov"), "aabb"),
    (0.25, "bnef,aefcijnk->abcijk", ("aa", "vovv"), "aaab"), (0.5, "bnef,aefcijnk->abcijk", ("ab", "vovv"), "aabb"),
    (0.25, "ncfe,abfeijnk->abcijk", ("ab", "ovvv"), "aaab"), (0.125, "cnef,abfeijnk->abcijk", ("bb", "vovv"), "aabb"),
    (0.25, "me,cebakmji->cbakji", ("b", "ov"), "abbb"), (0.25, "me,cebakmji->cbakji", ("a", "ov"), "aabb"),
    (-0.25, "mnjf,cfbaknmi->cbakji", ("bb", "ooov"), "abbb"), (-0.5, "nmfj,cfbaknmi->cbakji", ("ab", "oovo"), "aabb"),
    (-0.25, "mnkf,cfbamnji->cbakji", ("ab", "ooov"), "abbb"), (-0.125, "mnkf,cfbamnji->cbakji", ("aa", "ooov"), "aabb"),
    (0.25, "bnef,cfeaknji->cbakji", ("bb", "vovv"), "abbb"), (0.5, "nbfe,cfeaknji->cbakji", ("ab", "ovvv"), "aabb"),
    (0.25, "cnef,efbaknji->cbakji", ("ab", "vovv"), "abbb"), (0.125, "cnef,efbaknji->cbakji", ("aa", "vovv"), "aabb"),
    (1 / 36, "me,abceijkm->abcijk", ("b", "ov"), "bbbb"), (1 / 36, "me,ecbamkji->abcijk", ("a", "ov"), "abbb"),
    (1 / 24, "cnef,abefijkn->abcijk", ("bb", "vovv"), "bbbb"), (1 / 12, "ncfe,febankji->abcijk", ("ab", "ovvv"), "abbb"),
    (-1 / 24, "mnkf,abcfijmn->abcijk", ("bb", "ooov"), "bbbb"), (-1 / 12, "nmfk,fcbanmji->abcijk", ("ab", "oovo"), "abbb"),
]


def _internal_triples_mask(ms, p, system):
    """Boolean masks of the triples of reference p that lead to other references."""
    norb = system.norbitals
    occ_a, unocc_a, occ_b, unocc_b = ms.references[p]
    lo = {s * norb + g: k for s, lst in ((0, occ_a), (1, occ_b)) for k, g in enumerate(lst)}
    lu = {s * norb + g: k for s, lst in ((0, unocc_a), (1, unocc_b)) for k, g in enumerate(lst)}
    shape = {k: getattr(ClusterOperator(system, 3), k).shape for k in T3_KEYS}
    mask = {k: np.zeros(shape[k]) for k in T3_KEYS}
    int_h, int_p, int_k = ms.higher[p]
    for e in np.flatnonzero(int_k == 3):
        holes, parts = int_h[:3, e], int_p[:3, e]
        key = {3: "aaa", 2: "aab", 1: "abb", 0: "bbb"}[int(sum(h < norb for h in holes))]
        mask[key][tuple(lu[x] for x in parts) + tuple(lo[x] for x in holes)] = 1.0
    return {k: antisymmetrize(mask[k], k) != 0 for k in T3_KEYS}


def _triples_projection(T_q, amps_q, ms, p, q, system):
    out = sumrcc_loops.project_triples(T_q.a, T_q.b, T_q.aa, T_q.ab, T_q.bb, *fortran_internal(ms, q, amps_q),
                                       ms.inactive, int(can_prune(T_q, amps_q, ms, system)),
                                       *ms.references[p], *ms.references[q], norb=system.norbitals)
    return {k: antisymmetrize(x, k) for k, x in zip(T3_KEYS, out)}


def mmsucc23_dense(T, H, heff, roots, coefficients, ms, system):
    """Reference implementation of mmsucc23 with dense triples arrays (small systems only).
    MM-SUCC(2,3) corrections (Eh) to the SU-MRCCSD roots, for the converged cluster operators T,
    reference Hamiltonians H, effective Hamiltonian and its right eigenvectors (columns of
    `coefficients`, in the order of `roots`), and the model space ms. Returns the corrected
    energies, the corrections, and the per-reference triples moments Gamma_3(p)."""
    nref = len(ms.references)
    act = get_active_slices(system, ms)
    amps = [get_internal_amplitudes(T[q], ms, q, system) for q in range(nref)]
    # projections P(q->p) on singles/doubles/triples of p, including q = p
    P12 = [[get_projection(T[q], amps[q], ms, p, q, system) for q in range(nref)] for p in range(nref)]
    P3 = [[_triples_projection(T[q], amps[q], ms, p, q, system) for q in range(nref)] for p in range(nref)]

    gamma, z3, lin12 = [], [], []
    for p in range(nref):
        t34 = get_internal_t3_t4(amps[p], ms, p, system)
        M3 = get_direct_moments(T[p], materialize(H[p]), t34, act, system)
        # coupling: [e^{-T(p)} C(p)]_3 with C(p) = sum_{q != p} Heff_qp P(q->p)
        C12 = [sum((heff[q, p] * P12[p][q][k] for q in range(nref) if q != p), np.zeros_like(P12[p][p][k]))
               for k in range(5)]
        C3 = {k: sum((heff[q, p] * P3[p][q][k] for q in range(nref) if q != p), np.zeros_like(P3[p][p][k]))
              for k in T3_KEYS}
        X3 = sumrcc_loops.coupling_triples(*C12, *[C3[k] for k in T3_KEYS], T[p].a, T[p].b, T[p].aa, T[p].ab,
                                           T[p].bb, *ms.references[p], system.norbitals)
        mask = _internal_triples_mask(ms, p, system)
        g = {k: np.where(mask[k], 0.0, M3[k] - antisymmetrize(x, k)) for k, x in zip(T3_KEYS, X3)}
        gamma.append(g)
        z, _ = get_z3(T[p], materialize(H[p]), system)
        z3.append({k: np.where(mask[k], 0.0, z[k]) for k in T3_KEYS})
        # T1, T2 of the trial function without internal excitations
        lin12.append({k: np.where(ms.internal[p][k], 0.0, getattr(T[p], k)) for k in ("a", "b", "aa", "ab", "bb")})

    # linear overlaps <(1 + T1 + T2 + Z3)(q) Phi_q | K3(p)> for q != p
    L3 = [[None if q == p else {k: antisymmetrize(x, k) for k, x in zip(T3_KEYS, sumrcc_loops.linear_triples(
        *[lin12[q][k] for k in ("a", "b", "aa", "ab", "bb")], *[z3[q][k] for k in T3_KEYS],
        *ms.references[p], *ms.references[q], system.norbitals))} for q in range(nref)] for p in range(nref)]

    # <lin_q | e^{T(p)} Phi_p> (the Phi_q coefficient is zero for q != p by the C conditions)
    S = np.zeros((nref, nref))
    for q in range(nref):
        for p in range(nref):
            P = P12[q][p]
            S[q, p] = float(q == p) + np.vdot(lin12[q]["a"], P[0]) + np.vdot(lin12[q]["b"], P[1]) \
                + _dot(lin12[q], {"aa": P[2], "ab": P[3], "bb": P[4]}, T2_WEIGHTS) + _dot(z3[q], P3[q][p], T3_WEIGHTS)

    corrections = np.zeros(len(roots))
    for mu in range(len(roots)):
        c = np.real_if_close(coefficients[:, mu])
        numerator = 0.0
        for p in range(nref):
            overlap = {k: c[p] * z3[p][k] + sum((c[q] * L3[p][q][k] for q in range(nref) if q != p), 0.0)
                       for k in T3_KEYS}
            numerator += c[p] * _dot(overlap, gamma[p], T3_WEIGHTS)
        corrections[mu] = numerator / (c @ S.T @ c)
    return np.real(roots) + corrections, corrections, gamma


# (Hbar T3)_C terms of ccpy.cc.ccsdt.update_t3a-d linear in T3 (one- and two-body Hbar), for the
# internal T3: (coef, subscripts, (block, kinds) of the CCSD intermediates, T3 spin case)
_T3_LINEAR = [
    (-1 / 12, "mk,abcijm->abcijk", ("a", "oo"), "aaa"), (1 / 12, "ce,abeijk->abcijk", ("a", "vv"), "aaa"),
    (1 / 24, "mnij,abcmnk->abcijk", ("aa", "oooo"), "aaa"), (1 / 24, "abef,efcijk->abcijk", ("aa", "vvvv"), "aaa"),
    (0.25, "cmke,abeijm->abcijk", ("aa", "voov"), "aaa"), (0.25, "cmke,abeijm->abcijk", ("ab", "voov"), "aab"),
    (-0.5, "mi,abcmjk->abcijk", ("a", "oo"), "aab"), (-0.25, "mk,abcijm->abcijk", ("b", "oo"), "aab"),
    (0.5, "ae,ebcijk->abcijk", ("a", "vv"), "aab"), (0.25, "ce,abeijk->abcijk", ("b", "vv"), "aab"),
    (0.125, "mnij,abcmnk->abcijk", ("aa", "oooo"), "aab"), (0.5, "mnjk,abcimn->abcijk", ("ab", "oooo"), "aab"),
    (0.125, "abef,efcijk->abcijk", ("aa", "vvvv"), "aab"), (0.5, "bcef,aefijk->abcijk", ("ab", "vvvv"), "aab"),
    (1.0, "amie,ebcmjk->abcijk", ("aa", "voov"), "aab"), (1.0, "amie,becjmk->abcijk", ("ab", "voov"), "abb"),
    (0.25, "mcek,abeijm->abcijk", ("ab", "ovvo"), "aaa"), (0.25, "cmke,abeijm->abcijk", ("bb", "voov"), "aab"),
    (-0.5, "amek,ebcijm->abcijk", ("ab", "vovo"), "aab"), (-0.5, "mcie,abemjk->abcijk", ("ab", "ovov"), "aab"),
    (-0.25, "mi,abcmjk->abcijk", ("a", "oo"), "abb"), (-0.5, "mj,abcimk->abcijk", ("b", "oo"), "abb"),
    (0.25, "ae,ebcijk->abcijk", ("a", "vv"), "abb"), (0.5, "be,aecijk->abcijk", ("b", "vv"), "abb"),
    (0.125, "mnjk,abcimn->abcijk", ("bb", "oooo"), "abb"), (0.5, "mnij,abcmnk->abcijk", ("ab", "oooo"), "abb"),
    (0.125, "bcef,aefijk->abcijk", ("bb", "vvvv"), "abb"), (0.5, "abef,efcijk->abcijk", ("ab", "vvvv"), "abb"),
    (0.25, "amie,ebcmjk->abcijk", ("aa", "voov"), "abb"), (0.25, "amie,ebcmjk->abcijk", ("ab", "voov"), "bbb"),
    (1.0, "mbej,aecimk->abcijk", ("ab", "ovvo"), "aab"), (1.0, "bmje,aecimk->abcijk", ("bb", "voov"), "abb"),
    (-0.5, "mbie,aecmjk->abcijk", ("ab", "ovov"), "abb"), (-0.5, "amej,ebcimk->abcijk", ("ab", "vovo"), "abb"),
    (-1 / 12, "mk,abcijm->abcijk", ("b", "oo"), "bbb"), (1 / 12, "ce,abeijk->abcijk", ("b", "vv"), "bbb"),
    (1 / 24, "mnij,abcmnk->abcijk", ("bb", "oooo"), "bbb"), (1 / 24, "abef,efcijk->abcijk", ("bb", "vvvv"), "bbb"),
    (0.25, "maei,ebcmjk->abcijk", ("ab", "ovvo"), "abb"), (0.25, "amie,ebcmjk->abcijk", ("bb", "voov"), "bbb"),
]


def _block(coef, sub, h, hblock, t, tcase, act):
    """coef * einsum(sub, h, t) with the indices of t restricted to the active orbitals: the
    output spin case, the local index ranges of the output, and the block."""
    hin, rest = sub.split(",")
    tin, res = rest.split("->")
    spins = {"aa": "aaaa", "ab": "abab", "bb": "bbbb", "a": "aa", "b": "bb"}[hblock[0]]
    kind_of = dict(zip(tin, ["v"] * (len(tin) // 2) + ["o"] * (len(tin) // 2)))
    spin_of = dict(zip(tin, tcase * 2))
    for l, sp, kd in zip(hin, spins, hblock[1]):
        spin_of.setdefault(l, sp)
        kind_of.setdefault(l, kd)
    sl = lambda l: act[(spin_of[l], kind_of[l])] if l in tin else slice(None)
    block = coef * np.einsum(sub, h[tuple(sl(l) for l in hin)], t, optimize=True)
    key = "".join(spin_of[l] for l in res[:3])
    return key, [sl(l) for l in res], block


def _hbar_intermediates(T, H, t34, act, system):
    """I2 intermediates of ccpy.cc.ccsdt.update_t3a-d (CCSD Hbar, -f_ov t2 parts, and the
    internal-T3 parts), and the CCSD intermediates X."""
    X = Integral.from_empty(system, 2, use_none=True)
    X = get_pre_ccs_intermediates(X, T, H, False)
    X = get_ccs_intermediates_opt(X, T, H, False)
    X = get_ccsd_intermediates(T, X, H, False)
    I = {"ha_vvov": X.aa.vvov.copy(), "ha_vooo": X.aa.vooo - np.einsum("me,aeij->amij", X.a.ov, T.aa, optimize=True),
         "hb_vvvo": X.ab.vvvo.copy(), "hb_ovoo": X.ab.ovoo - np.einsum("me,ecjk->mcjk", X.a.ov, T.ab, optimize=True),
         "hb_vvov": X.ab.vvov.copy(), "hb_vooo": X.ab.vooo - np.einsum("me,aeik->amik", X.b.ov, T.ab, optimize=True),
         "hc_vvov": X.bb.vvov.copy(), "hc_vooo": X.bb.vooo - np.einsum("me,cekj->cmkj", X.b.ov, T.bb, optimize=True)}
    if t34 is not None:
        terms = [("ha_vvov", -0.5, "mnef,abfimn->abie", "aa", "aaa"), ("ha_vvov", -1.0, "mnef,abfimn->abie", "ab", "aab"),
                 ("ha_vooo", 0.5, "mnef,aefijn->amij", "aa", "aaa"), ("ha_vooo", 1.0, "mnef,aefijn->amij", "ab", "aab"),
                 ("hb_vvvo", -0.5, "mnef,afbmnj->abej", "aa", "aab"), ("hb_vvvo", -1.0, "mnef,afbmnj->abej", "ab", "abb"),
                 ("hb_ovoo", 0.5, "mnef,efbinj->mbij", "aa", "aab"), ("hb_ovoo", 1.0, "mnef,efbinj->mbij", "ab", "abb"),
                 ("hb_vvov", -1.0, "nmfe,afbinm->abie", "ab", "aab"), ("hb_vvov", -0.5, "nmfe,afbinm->abie", "bb", "abb"),
                 ("hb_vooo", 1.0, "nmfe,afeinj->amij", "ab", "aab"), ("hb_vooo", 0.5, "nmfe,afeinj->amij", "bb", "abb"),
                 ("hc_vvov", -0.5, "mnef,abfimn->abie", "bb", "bbb"), ("hc_vvov", -1.0, "nmfe,fabnim->abie", "ab", "abb"),
                 ("hc_vooo", 0.5, "mnef,aefijn->amij", "bb", "bbb"), ("hc_vooo", 1.0, "nmfe,faenij->amij", "ab", "abb")]
        for name, coef, sub, blk, case in terms:
            if case in t34:
                key, sls, block = _block(coef, sub, getattr(H, blk).oovv, (blk, "oovv"), t34[case], case, act)
                I[name][tuple(sls)] += block
    return I, X


def _stack(arrays):
    return np.asfortranarray(np.stack(arrays, axis=-1))


def _bare(H, key, name):
    """A two-body block of H, built from H.chol when it is not stored (DF Hamiltonians)."""
    block = getattr(getattr(H, key), name)
    return chol_block(H.chol, key, name) if block is None else block


def mmsucc23(T, H, heff, roots, coefficients, ms, system):
    """MM-SUCC(2,3) corrections (Eh) to the SU-MRCCSD roots, for the converged cluster operators T,
    reference Hamiltonians H, effective Hamiltonian and its right eigenvectors (columns of
    `coefficients`, in the order of `roots`), and the model space ms. Returns the corrected
    energies and the corrections. No triples are stored (sumrcc_loops.mm23_reference)."""
    nref, nroot = len(ms.references), len(roots)
    act = get_active_slices(system, ms)
    c = np.asfortranarray(np.real(coefficients))
    amps = [get_internal_amplitudes(T[q], ms, q, system) for q in range(nref)]
    # P(q->p) on singles/doubles (q = p included; None for symmetry-forbidden pairs)
    P12 = [[get_projection(T[q], amps[q], ms, p, q, system) if q == p or symmetry_allowed(ms, p, q) else None
            for q in range(nref)] for p in range(nref)]
    lin12 = [{k: np.where(ms.internal[p][k], 0.0, getattr(T[p], k)) for k in ("a", "b", "aa", "ab", "bb")}
             for p in range(nref)]

    refs = [_stack([ms.references[r][k] for r in range(nref)]) for k in range(4)]
    t_all = [_stack([getattr(T[r], k) for r in range(nref)]) for k in ("a", "b", "aa", "ab", "bb")]
    nint_ref = np.array([len(ms.higher[r][2]) for r in range(nref)], dtype=np.int32)
    nint = int(nint_ref.max())
    int_h = np.zeros((ms.max_rank, nint, nref), np.int32, order="F")
    int_p = np.zeros((ms.max_rank, nint, nref), np.int32, order="F")
    int_k = np.zeros((nint, nref), np.int32, order="F")
    int_amp = np.zeros((nint, nref), order="F")
    for r in range(nref):
        n = nint_ref[r]
        int_h[:, :n, r], int_p[:, :n, r], int_k[:n, r] = ms.higher[r]
        int_amp[:n, r] = amps[r]
    prune = np.array([can_prune(T[r], amps[r], ms, system) for r in range(nref)], dtype=np.int32)
    bare = [_stack([_bare(H[r], blk, kinds) for r in range(nref)]) for blk, kinds in
            (("aa", "vooo"), ("aa", "vvov"), ("ab", "vvvo"), ("ab", "ovoo"), ("ab", "vvov"), ("ab", "vooo"),
             ("bb", "vvov"), ("bb", "vooo"), ("aa", "vvoo"), ("ab", "vvoo"), ("bb", "vvoo"))]
    fock = [_stack([np.diagonal(getattr(getattr(H[r], s), kinds)) for r in range(nref)])
            for s, kinds in (("a", "oo"), ("a", "vv"), ("b", "oo"), ("b", "vv"))]

    numerator = np.zeros(nroot)
    S = np.zeros((nref, nref))
    for p in range(nref):
        t34 = get_internal_t3_t4(amps[p], ms, p, system)
        I, X = _hbar_intermediates(T[p], materialize(H[p]), t34, act, system)
        C12 = [sum((heff[q, p] * P12[p][q][k] for q in range(nref) if q != p and P12[p][q] is not None),
                   np.zeros_like(P12[p][p][k]))
               for k in range(5)]
        # internal T3 (linear Hbar T3) and T4 terms: sum_K O(K) x(K) over their blocks
        blocks = []
        if t34 is not None:
            for coef, sub, blk, case in _T3_LINEAR:
                if case in t34:
                    blocks.append(_block(coef, sub, getattr(getattr(X, blk[0]), blk[1]), blk, t34[case], case, act))
            for coef, sub, blk, case in _T4_TO_T3:
                if case in t34:
                    blocks.append(_block(coef, sub, getattr(getattr(X, blk[0]), blk[1]), blk, t34[case], case, act))
        klist, kcase, xs = [], [], []
        dims = {"a": (system.nunoccupied_alpha, system.noccupied_alpha),
                "b": (system.nunoccupied_beta, system.noccupied_beta)}
        for key, sls, block in blocks:
            sizes = [dims[s][0] for s in key] + [dims[s][1] for s in key]
            ranges = [np.arange(n)[sl] for n, sl in zip(sizes, sls)]
            grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=0).reshape(6, -1)
            keep = np.flatnonzero(block.ravel())
            klist.append(grid[:, keep] + 1)
            kcase.append(np.full(len(keep), T3_KEYS.index(key) + 1))
            xs.append(block.ravel()[keep])
        if klist:
            klist, kcase, xs = np.concatenate(klist, axis=1), np.concatenate(kcase), np.concatenate(xs)
        else:
            klist, kcase, xs = np.ones((6, 1)), np.zeros(1), np.zeros(1)
        num, s3, olist = sumrcc_loops.mm23_reference(
            p + 1, c, heff[:, p].copy(), *refs, *t_all, int_h, int_p, int_k, int_amp, nint_ref, ms.inactive, prune,
            *bare, *fock,
            *[I[k] for k in ("ha_vooo", "ha_vvov", "hb_vvvo", "hb_ovoo", "hb_vvov", "hb_vooo", "hc_vvov", "hc_vooo")],
            *C12, np.asfortranarray(klist, dtype=np.int32), kcase.astype(np.int32), norb=system.norbitals)
        numerator += num + c[p] * (olist @ xs)
        for q in range(nref):
            P = P12[p][q]
            if P is None:
                S[p, q] = s3[q]
                continue
            S[p, q] = float(p == q) + np.vdot(lin12[p]["a"], P[0]) + np.vdot(lin12[p]["b"], P[1]) \
                + _dot(lin12[p], {"aa": P[2], "ab": P[3], "bb": P[4]}, T2_WEIGHTS) + s3[q]
    corrections = numerator / np.einsum("pm,pq,qm->m", c, S, c)
    return np.real(roots) + corrections, corrections
