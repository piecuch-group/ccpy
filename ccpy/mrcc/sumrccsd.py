'''
State-Universal Multireference Coupled-Cluster Method with Singles and Doubles (SU-MRCCSD)
[Jeziorski & Monkhorst; Piecuch & Kowalski, IJMS 3, 676 (2002)], for complete or general
model spaces [Li & Paldus, JCP 119, 5320 (2003)].

For each reference p,
    <K(p)|(H e^{T(p)})_C|Phi_p> = sum_{q != p} <K(p)|e^{-T(p)} e^{T(q)}|Phi_q> Heff_qp,
for the external singles and doubles K(p) (E_K(p) Phi_p outside the model space). The
internal amplitudes follow the C conditions (see ccpy.mrcc.sumrcc). The direct terms are the
CCSD ones plus those of the internal T3 and T4 (all-active, so kept as small dense arrays over
the active orbitals; higher ranks cannot reach singles and doubles). With
C(p) = sum_{q != p} Heff_qp P(q->p), P(q->p)_K = <Phi_K(p)|e^{T(q)}|Phi_q>, the coupling
terms are
    singles: C1(p),    doubles: C2(p) - A(ij)A(ab) t1(p)_ai C1(p)_bj,
since the coefficient of Phi_p in e^{T(q)}|Phi_q> vanishes by the C conditions. The direct terms are the CCSD
residuals of ccpy.cc.ccsd with the Hamiltonian normal ordered with respect to Phi_p.

frozen_cv=True gives SUMRCCSD(1): core->virtual t1 = 0 and core-core->virtual-virtual t2
fixed at first-order MRMBPT values from the orbital energies of reference 0 (Eqs. 64-65).
'''

import time
from types import SimpleNamespace

import numpy as np

from ccpy.cc import ccsd_chol
from ccpy.drivers.diis import DIIS
from ccpy.lib.core import vvvv_contraction
from ccpy.models.integrals import Integral
from ccpy.models.operators import ClusterOperator
from ccpy.mrcc.sumrcc import as_df, chol_block, get_model_space, get_reference_hamiltonian, get_projections, get_effective_hamiltonian, \
    get_internal_amplitudes, get_external_estimates, get_lowest_external, get_heff_and_couplings
from ccpy.mrcc.sumrcc import prune_intruders as prune_model_space


def get_core_virtual(system, ms):
    """Local-index slices of the all-inactive (core-virtual) blocks, the same for every
    reference: core occupied first, virtuals unoccupied last."""
    (ca, cb), (va, vb) = ms.ncore, ms.nvirt
    va, vb = slice(system.nunoccupied_alpha - va, None), slice(system.nunoccupied_beta - vb, None)
    ca, cb = slice(0, ca), slice(0, cb)
    return {"a": (va, ca), "b": (vb, cb), "aa": (va, va, ca, ca), "ab": (va, vb, ca, cb), "bb": (vb, vb, cb, cb)}


def get_frozen_amplitudes(H, core_virtual):
    """First-order core-virtual amplitudes, t1 = 0 and t2 = <ab||ij> / (e_i + e_j - e_a - e_b),
    from the Fock matrix H of reference 0."""
    ea_o, ea_v = np.diagonal(H.a.oo), np.diagonal(H.a.vv)
    eb_o, eb_v = np.diagonal(H.b.oo), np.diagonal(H.b.vv)
    n = np.newaxis
    d_aa = ea_o[n, n, :, n] + ea_o[n, n, n, :] - ea_v[:, n, n, n] - ea_v[n, :, n, n]
    d_ab = ea_o[n, n, :, n] + eb_o[n, n, n, :] - ea_v[:, n, n, n] - eb_v[n, :, n, n]
    d_bb = eb_o[n, n, :, n] + eb_o[n, n, n, :] - eb_v[:, n, n, n] - eb_v[n, :, n, n]
    t2 = {"aa": H.aa.vvoo / d_aa, "ab": H.ab.vvoo / d_ab, "bb": H.bb.vvoo / d_bb}
    frozen = {key: t2[key][core_virtual[key]] for key in ("aa", "ab", "bb")}
    frozen["a"], frozen["b"] = 0.0, 0.0
    return frozen


def apply_masks(T, dT, internal, core_virtual, frozen, keys=("a", "b", "aa", "ab", "bb")):
    """C conditions on the internal amplitudes (t1 = 0, t2 = -A t1 t1, for the current t1);
    core-virtual amplitudes fixed for SUMRCCSD(1)."""
    for key in keys:
        if len(key) == 1:
            value = 0.0
        else:
            t1 = (T.a, T.b)
            x, y = t1["ab".index(key[0])], t1["ab".index(key[1])]
            W = np.einsum("ai,bj->abij", x, y, optimize=True)
            # same spin: -(t_ai t_bj - t_aj t_bi), half the four-term antisymmetrizer of t1 t1
            value = -0.5 * (W - W.transpose(1, 0, 2, 3) - W.transpose(0, 1, 3, 2) + W.transpose(1, 0, 3, 2)) \
                if key[0] == key[1] else -W
            value = value[internal[key]]
        getattr(T, key)[internal[key]] = value
        getattr(dT, key)[internal[key]] = 0.0
        if frozen is not None:
            getattr(T, key)[core_virtual[key]] = frozen[key]
            getattr(dT, key)[core_virtual[key]] = 0.0


# spin cases of the internal T3 and T4 (ccpy layout: alpha before beta, particles then holes)
KEYS = ("a", "b", "aa", "ab", "bb")
T3_CASES, T4_CASES = ("aaa", "aab", "abb", "bbb"), ("aaaa", "aaab", "aabb", "abbb", "bbbb")


def get_active_slices(system, ms):
    """Local-index slices of the active occupied ("o") and unoccupied ("v") orbitals per spin."""
    return {("a", "o"): slice(ms.ncore[0], None), ("b", "o"): slice(ms.ncore[1], None),
            ("a", "v"): slice(0, system.nunoccupied_alpha - ms.nvirt[0]),
            ("b", "v"): slice(0, system.nunoccupied_beta - ms.nvirt[1])}


def get_internal_t3_t4(int_amp, ms, p, system):
    """Rank-3 and rank-4 internal amplitudes of reference p as dense arrays over its active
    orbitals, keyed by spin case (None when all of them vanish)."""
    from itertools import permutations

    int_h, int_p, int_k = ms.higher[p]
    entries = [e for e in np.flatnonzero(int_amp) if int_k[e] in (3, 4)]
    if not entries:
        return None
    norb = system.norbitals
    occ_a, unocc_a, occ_b, unocc_b = ms.references[p]
    act = get_active_slices(system, ms)
    lo = {s * norb + g: k - act[("ab"[s], "o")].start for s, lst in ((0, occ_a), (1, occ_b)) for k, g in enumerate(lst)}
    lu = {s * norb + g: k for s, lst in ((0, unocc_a), (1, unocc_b)) for k, g in enumerate(lst)}
    nact = {(s, k): (len(lst) - sl.start if k == "o" else sl.stop) for (s, k), sl in act.items()
            for lst in [{"ao": occ_a, "bo": occ_b, "av": unocc_a, "bv": unocc_b}[s + k]]}
    amps = {}
    for e in entries:
        k = int_k[e]
        holes, parts = list(int_h[:k, e]), list(int_p[:k, e])
        na = sum(1 for x in holes if x < norb)
        case = "a" * na + "b" * (k - na)
        if case not in amps:
            amps[case] = np.zeros([nact[(s, "v")] for s in case] + [nact[(s, "o")] for s in case])
        # all antisymmetric copies: permute within the alpha and within the beta groups
        groups = [range(na), range(na, k)]
        def signed_perms(group):
            for perm in permutations(group):
                inv = sum(1 for x in range(len(perm)) for y in range(x + 1, len(perm)) if perm[x] > perm[y])
                yield list(perm), (-1) ** inv
        for pa, sa in signed_perms(groups[0]):
            for pb, sb in signed_perms(groups[1]):
                for ha, ta in signed_perms(groups[0]):
                    for hb, tb in signed_perms(groups[1]):
                        idx = tuple(lu[parts[m]] for m in pa + pb) + tuple(lo[holes[m]] for m in ha + hb)
                        amps[case][idx] = sa * sb * ta * tb * int_amp[e]
    return amps


def _term(out, coef, sub, h, hblock, t, tcase, act):
    """out[key] += coef * einsum(sub, h, t), with every index of t (an internal T3/T4 of spin
    case tcase over active orbitals) restricted to the active orbitals in h and in out."""
    hin, rest = sub.split(",")
    tin, res = rest.split("->")
    spins, kinds = hblock
    spins = {"aa": "aaaa", "ab": "abab", "bb": "bbbb", "a": "aa", "b": "bb"}[spins]
    kind_of = dict(zip(tin, ["v"] * (len(tin) // 2) + ["o"] * (len(tin) // 2)))
    spin_of = dict(zip(tin, tcase * 2))
    for l, sp, kd in zip(hin, spins, kinds):
        spin_of.setdefault(l, sp)
        kind_of.setdefault(l, kd)
    sl = lambda l: act[(spin_of[l], kind_of[l])] if l in tin else slice(None)
    key = "".join(spin_of[l] for l in res[:len(res) // 2])
    out[key][tuple(sl(l) for l in res)] += coef * np.einsum(sub, h[tuple(sl(l) for l in hin)], t, optimize=True)


def get_internal_t3_singles(t34, H0, act, T):
    """<S(p)|(V T3)_C|Phi_p> from the internal T3 (terms of ccpy.cc.ccsdt)."""
    out = {"a": np.zeros_like(T.a), "b": np.zeros_like(T.b)}
    if t34 is None:
        return out
    for coef, sub, blk, case in ((0.25, "mnef,aefimn->ai", "aa", "aaa"), (1.0, "mnef,aefimn->ai", "ab", "aab"),
                                 (0.25, "mnef,aefimn->ai", "bb", "abb"), (0.25, "mnef,aefimn->ai", "bb", "bbb"),
                                 (0.25, "mnef,efamni->ai", "aa", "aab"), (1.0, "mnef,efamni->ai", "ab", "abb")):
        if case in t34:
            _term(out, coef, sub, getattr(H0, blk).oovv, (blk, "oovv"), t34[case], case, act)
    return out


class _Dressed:
    """A T1-dressed two-body block built from the transformed Cholesky vectors only on the index
    slices it is read with (_term slices every T3/T4 index to the active orbitals)."""

    def __init__(self, chol, key, name):
        self.chol, self.key, self.name = chol, key, name

    def __getitem__(self, idx):
        return chol_block(self.chol, self.key, self.name, idx)


def get_internal_t3_t4_doubles(t34, H0, X, act, T):
    """<D(p)|(Hbar_CCS T3 + V T4)_C|Phi_p> from the internal T3 and T4 (terms of ccpy.cc.ccsdt
    and ccpy.cc.ccsdtq, with X the CCS intermediates of ccpy.cc.ccsd), fully antisymmetric."""
    out = {"aa": np.zeros_like(T.aa), "ab": np.zeros_like(T.ab), "bb": np.zeros_like(T.bb)}
    if t34 is None:
        return out
    hbar = lambda blk, kinds: _Dressed(X.chol, blk, kinds)
    one = lambda s: getattr(X, s).ov
    terms = [  # (coef, subscripts, H value, H block, T spin case)
        (0.25, "me,abeijm->abij", lambda: one("a"), ("a", "ov"), "aaa"),
        (0.25, "me,abeijm->abij", lambda: one("b"), ("b", "ov"), "aab"),
        (-0.5, "mnif,abfmjn->abij", lambda: hbar("ab", "ooov"), ("ab", "ooov"), "aab"),
        (-0.25, "mnif,abfmjn->abij", lambda: hbar("aa", "ooov"), ("aa", "ooov"), "aaa"),
        (0.25, "anef,ebfijn->abij", lambda: hbar("aa", "vovv"), ("aa", "vovv"), "aaa"),
        (0.5, "anef,ebfijn->abij", lambda: hbar("ab", "vovv"), ("ab", "vovv"), "aab"),
        (-0.5, "mnif,afbmnj->abij", lambda: hbar("aa", "ooov"), ("aa", "ooov"), "aab"),
        (-1.0, "nmfj,afbinm->abij", lambda: hbar("ab", "oovo"), ("ab", "oovo"), "aab"),
        (-0.5, "mnjf,afbinm->abij", lambda: hbar("bb", "ooov"), ("bb", "ooov"), "abb"),
        (-1.0, "mnif,afbmnj->abij", lambda: hbar("ab", "ooov"), ("ab", "ooov"), "abb"),
        (0.5, "anef,efbinj->abij", lambda: hbar("aa", "vovv"), ("aa", "vovv"), "aab"),
        (1.0, "anef,efbinj->abij", lambda: hbar("ab", "vovv"), ("ab", "vovv"), "abb"),
        (1.0, "nbfe,afeinj->abij", lambda: hbar("ab", "ovvv"), ("ab", "ovvv"), "aab"),
        (0.5, "bnef,afeinj->abij", lambda: hbar("bb", "vovv"), ("bb", "vovv"), "abb"),
        (1.0, "me,aebimj->abij", lambda: one("a"), ("a", "ov"), "aab"),
        (1.0, "me,aebimj->abij", lambda: one("b"), ("b", "ov"), "abb"),
        (0.25, "me,eabmij->abij", lambda: one("a"), ("a", "ov"), "abb"),
        (0.25, "me,abeijm->abij", lambda: one("b"), ("b", "ov"), "bbb"),
        (0.25, "anef,ebfijn->abij", lambda: hbar("bb", "vovv"), ("bb", "vovv"), "bbb"),
        (0.5, "nafe,febnij->abij", lambda: hbar("ab", "ovvv"), ("ab", "ovvv"), "abb"),
        (-0.25, "mnif,abfmjn->abij", lambda: hbar("bb", "ooov"), ("bb", "ooov"), "bbb"),
        (-0.5, "nmfi,fabnmj->abij", lambda: hbar("ab", "oovo"), ("ab", "oovo"), "abb"),
        (0.0625, "mnef,abefijmn->abij", lambda: H0.aa.oovv, ("aa", "oovv"), "aaaa"),
        (0.25, "mnef,abefijmn->abij", lambda: H0.ab.oovv, ("ab", "oovv"), "aaab"),
        (0.0625, "mnef,abefijmn->abij", lambda: H0.bb.oovv, ("bb", "oovv"), "aabb"),
        (0.25, "mnef,aefbimnj->abij", lambda: H0.aa.oovv, ("aa", "oovv"), "aaab"),
        (1.0, "mnef,aefbimnj->abij", lambda: H0.ab.oovv, ("ab", "oovv"), "aabb"),
        (0.25, "mnef,abefijmn->abij", lambda: H0.bb.oovv, ("bb", "oovv"), "abbb"),
        (0.0625, "mnef,abefijmn->abij", lambda: H0.bb.oovv, ("bb", "oovv"), "bbbb"),
        (0.25, "nmfe,febanmji->abij", lambda: H0.ab.oovv, ("ab", "oovv"), "abbb"),
        (0.0625, "mnef,febanmji->abij", lambda: H0.aa.oovv, ("aa", "oovv"), "aabb"),
    ]
    for coef, sub, h, blk, case in terms:
        if case in t34:
            _term(out, coef, sub, h(), blk, t34[case], case, act)
    # aa and bb terms are written for the antisymmetrizer A(ab)A(ij) of the CCSD residual
    for key in ("aa", "bb"):
        x = out[key]
        out[key] = x - x.transpose(1, 0, 2, 3) - x.transpose(0, 1, 3, 2) + x.transpose(1, 0, 3, 2)
    return out


def get_coupling(P_p, heff, p, T):
    """C(p) = sum_{q != p} Heff_qp P(q->p) as (C1a, C1b, C2a, C2b, C2c), from the projections
    P_p[q] = P(q->p) (None for p and symmetry-forbidden pairs)."""
    return [sum((heff[q, p] * P_p[q][k] for q in range(len(P_p)) if P_p[q] is not None), np.zeros_like(x))
            for k, x in enumerate((T.a, T.b, T.aa, T.ab, T.bb))]


def get_denominators(H, shift):
    n = np.newaxis
    ea_o, ea_v = np.diagonal(H.a.oo), np.diagonal(H.a.vv)
    eb_o, eb_v = np.diagonal(H.b.oo), np.diagonal(H.b.vv)
    return {"a": ea_o[n, :] - ea_v[:, n] - shift,
            "b": eb_o[n, :] - eb_v[:, n] - shift,
            "aa": ea_o[n, n, :, n] + ea_o[n, n, n, :] - ea_v[:, n, n, n] - ea_v[n, :, n, n] - shift,
            "ab": ea_o[n, n, :, n] + eb_o[n, n, n, :] - ea_v[:, n, n, n] - eb_v[n, :, n, n] - shift,
            "bb": eb_o[n, n, :, n] + eb_o[n, n, n, :] - eb_v[:, n, n, n] - eb_v[n, :, n, n] - shift}


FULL = (slice(None),) * 4


def _ladder(Ra, Rb, t, a=slice(None), b=slice(None), sym=False):
    """sum_ef (ae|bf) t[e, f, ...] on the output rows a, b (slices), from the (T1-transformed) vv
    Cholesky vectors Ra, Rb (naux, v, v). Restricted rows: the integrals of those rows only. Few
    columns: through the auxiliary index, 2 naux v^3 ncol flops instead of naux v^4. Otherwise
    the vvvv_contraction kernels of ccpy.cc.ccsd_chol; sym (same spin, t antisymmetric in e, f and
    in i, j) uses vvvv_t2_sym over the unique pairs a < b, e < f."""
    full = slice(None)
    nv1, nv2 = Ra.shape[1], Rb.shape[1]
    tk = t.reshape(nv1, nv2, -1)
    ncol = tk.shape[2]
    if a != full or b != full:
        V = np.einsum("xae,xbf->abef", Ra[:, a], Rb[:, b], optimize=True)
        x = np.tensordot(V, tk, axes=([2, 3], [0, 1]))
    elif 6 * ncol < min(nv1, nv2):
        # the flop crossover is ncol = v/2, but the full kernel's large GEMMs run several times
        # faster (measured with Accelerate, naux 408, v 71: equal at ncol ~ 10)
        x = np.zeros((nv1, nv2 * ncol))
        tf = tk.transpose(1, 0, 2).reshape(nv2, nv1 * ncol)
        nx = max(1, int(2e7 // (nv1 * nv2 * ncol)))  # auxiliary batches: the intermediate stays ~160 MB
        for x0 in range(0, Ra.shape[0], nx):
            rb, ra = Rb[x0:x0 + nx], Ra[x0:x0 + nx]
            # Y[x,b,e,k] = sum_f Rb[x,b,f] t[e,f,k]; x[a,b,k] += sum_{x,e} Ra[x,a,e] Y[x,b,e,k]
            Y = (rb.reshape(-1, nv2) @ tf).reshape(len(rb), nv2, nv1, ncol).transpose(0, 2, 1, 3)
            x += ra.transpose(1, 0, 2).reshape(nv1, -1) @ Y.reshape(-1, nv2 * ncol)
        x = x.reshape(nv1, nv2, ncol)
    elif sym:
        # sum_{e<f} <ab||ef> t(efij) for a < b, the rest by antisymmetry
        x = vvvv_contraction.vvvv_t2_sym(Ra.transpose(0, 2, 1), t.transpose(3, 2, 1, 0)).transpose(3, 2, 1, 0)
        return x - x.transpose(1, 0, 2, 3)
    else:
        x = vvvv_contraction.vvvv_t2(Ra.transpose(0, 2, 1), Rb.transpose(0, 2, 1),
                                     tk.reshape(nv1, nv2, ncol, 1).transpose(3, 2, 1, 0)).transpose(3, 2, 1, 0)
    return x.reshape(x.shape[:2] + t.shape[2:])


def _x2a(T, X, H, s):
    """Unantisymmetrized aa doubles residual without the vvvv ladder (the terms of
    ccpy.cc.ccsd_chol.update_t2a: T1-transformed Cholesky vectors X.chol, no integral with more
    than two unoccupied indices), on the output block s = (a, b, i, j) slices."""
    a, b, i, j = s
    C, Cb = X.chol.a, X.chol.b
    x = 0.5 * np.einsum("xai,xbj->abij", C.vo[:, a, i], C.vo[:, b, j], optimize=True)
    x -= 0.5 * np.einsum("mi,abmj->abij", X.a.oo[:, i], T.aa[a, b, :, j], optimize=True)
    x += 0.5 * np.einsum("ae,ebij->abij", X.a.vv[a], T.aa[:, b, i, j], optimize=True)
    x += np.einsum("xai,xme,ebmj->abij", C.vo[:, a, i], C.ov, T.aa[:, b, :, j], optimize=True)
    x -= np.einsum("xae,xmi,ebmj->abij", C.vv[:, a], C.oo[:, :, i], T.aa[:, b, :, j], optimize=True)
    x += 0.5 * np.einsum("mnef,afin,ebmj->abij", H.aa.oovv, T.aa[a, :, i, :], T.aa[:, b, :, j], optimize=True)
    x += np.einsum("mnef,afin,ebmj->abij", H.ab.oovv, T.ab[a, :, i, :], T.aa[:, b, :, j], optimize=True)
    x += np.einsum("xai,xme,bejm->abij", C.vo[:, a, i], Cb.ov, T.ab[b, :, j, :], optimize=True)
    x += 0.5 * np.einsum("mnef,afin,bejm->abij", H.bb.oovv, T.ab[a, :, i, :], T.ab[b, :, j, :], optimize=True)
    I = (np.einsum("xmi,xnj->mnij", C.oo[:, :, i], C.oo[:, :, j], optimize=True)
         - np.einsum("xmj,xni->mnij", C.oo[:, :, j], C.oo[:, :, i], optimize=True)
         + 0.5 * np.einsum("mnef,efij->mnij", H.aa.oovv, T.aa[:, :, i, j], optimize=True))
    return x + 0.125 * np.einsum("mnij,abmn->abij", I, T.aa[a, b], optimize=True)


def _x2b(T, X, H, s):
    """ab doubles residual without the ladder (ccpy.cc.ccsd_chol.update_t2b), on the block s."""
    a, b, i, j = s
    Ca, Cb = X.chol.a, X.chol.b
    x = np.einsum("xai,xbj->abij", Ca.vo[:, a, i], Cb.vo[:, b, j], optimize=True)
    x += np.einsum("ae,ebij->abij", X.a.vv[a], T.ab[:, b, i, j], optimize=True)
    x += np.einsum("be,aeij->abij", X.b.vv[b], T.ab[a, :, i, j], optimize=True)
    x -= np.einsum("mi,abmj->abij", X.a.oo[:, i], T.ab[a, b, :, j], optimize=True)
    x -= np.einsum("mj,abim->abij", X.b.oo[:, j], T.ab[a, b, i, :], optimize=True)
    x += np.einsum("xai,xme,ebmj->abij", Ca.vo[:, a, i], Ca.ov, T.ab[:, b, :, j], optimize=True)
    x -= np.einsum("xae,xmi,ebmj->abij", Ca.vv[:, a], Ca.oo[:, :, i], T.ab[:, b, :, j], optimize=True)
    x += np.einsum("lkdc,adil,cbkj->abij", H.aa.oovv, T.aa[a, :, i, :], T.ab[:, b, :, j], optimize=True)
    x += np.einsum("klcd,adil,cbkj->abij", H.ab.oovv, T.ab[a, :, i, :], T.ab[:, b, :, j], optimize=True)
    x += np.einsum("xai,xme,ebmj->abij", Ca.vo[:, a, i], Cb.ov, T.bb[:, b, :, j], optimize=True)
    x += np.einsum("lkdc,adil,cbkj->abij", H.ab.oovv, T.aa[a, :, i, :], T.bb[:, b, :, j], optimize=True)
    x += np.einsum("lkdc,adil,cbkj->abij", H.bb.oovv, T.ab[a, :, i, :], T.bb[:, b, :, j], optimize=True)
    x += np.einsum("xme,xbj,aeim->abij", Ca.ov, Cb.vo[:, b, j], T.aa[a, :, i, :], optimize=True)
    x += np.einsum("xbj,xme,aeim->abij", Cb.vo[:, b, j], Cb.ov, T.ab[a, :, i, :], optimize=True)
    x -= np.einsum("xbe,xmj,aeim->abij", Cb.vv[:, b], Cb.oo[:, :, j], T.ab[a, :, i, :], optimize=True)
    x -= np.einsum("xmi,xbe,aemj->abij", Ca.oo[:, :, i], Cb.vv[:, b], T.ab[a, :, :, j], optimize=True)
    x -= np.einsum("xae,xmj,ebim->abij", Ca.vv[:, a], Cb.oo[:, :, j], T.ab[:, b, i, :], optimize=True)
    x += np.einsum("mnef,afmj,ebin->abij", H.ab.oovv, T.ab[a, :, :, j], T.ab[:, b, i, :], optimize=True)
    I = (np.einsum("xmi,xnj->mnij", Ca.oo[:, :, i], Cb.oo[:, :, j], optimize=True)
         + np.einsum("mnef,efij->mnij", H.ab.oovv, T.ab[:, :, i, j], optimize=True))
    return x + np.einsum("mnij,abmn->abij", I, T.ab[a, b], optimize=True)


def _x2c(T, X, H, s):
    """Unantisymmetrized bb doubles residual without the ladder (ccpy.cc.ccsd_chol.update_t2c)."""
    a, b, i, j = s
    Ca, Cb = X.chol.a, X.chol.b
    x = 0.5 * np.einsum("xai,xbj->abij", Cb.vo[:, a, i], Cb.vo[:, b, j], optimize=True)
    x -= 0.5 * np.einsum("mi,abmj->abij", X.b.oo[:, i], T.bb[a, b, :, j], optimize=True)
    x += 0.5 * np.einsum("ae,ebij->abij", X.b.vv[a], T.bb[:, b, i, j], optimize=True)
    x += np.einsum("xai,xme,ebmj->abij", Cb.vo[:, a, i], Cb.ov, T.bb[:, b, :, j], optimize=True)
    x -= np.einsum("xae,xmi,ebmj->abij", Cb.vv[:, a], Cb.oo[:, :, i], T.bb[:, b, :, j], optimize=True)
    x += 0.5 * np.einsum("mnef,afin,ebmj->abij", H.bb.oovv, T.bb[a, :, i, :], T.bb[:, b, :, j], optimize=True)
    x += np.einsum("nmfe,fani,ebmj->abij", H.ab.oovv, T.ab[:, a, :, i], T.bb[:, b, :, j], optimize=True)
    x += np.einsum("xai,xme,ebmj->abij", Cb.vo[:, a, i], Ca.ov, T.ab[:, b, :, j], optimize=True)
    x += 0.5 * np.einsum("mnef,fani,ebmj->abij", H.aa.oovv, T.ab[:, a, :, i], T.ab[:, b, :, j], optimize=True)
    I = (np.einsum("xmi,xnj->mnij", Cb.oo[:, :, i], Cb.oo[:, :, j], optimize=True)
         - np.einsum("xmj,xni->mnij", Cb.oo[:, :, j], Cb.oo[:, :, i], optimize=True)
         + 0.5 * np.einsum("mnef,efij->mnij", H.bb.oovv, T.bb[:, :, i, j], optimize=True))
    return x + 0.125 * np.einsum("mnij,abmn->abij", I, T.bb[a, b], optimize=True)


def get_t2_residual(T, X, H, key, core_virtual=None):
    """<ijab|(H_N e^(T1+T2))_C|Phi_p> (fully antisymmetric) of spin case key, with X the pre and
    T1-transformed intermediates of ccpy.cc.ccsd_chol. With core_virtual (SUMRCCSD(1)) only on the
    doubles with at least one active index, zero on the core-virtual block: on the overlapping
    blocks (A, :, :, :), (:, A, :, :), (:, :, O, :), (:, :, :, O), A the active unoccupied and O
    the active occupied orbitals (each assigned its whole value). The frozen core-virtual t2
    still enter every term."""
    full = slice(None)
    asym = lambda x: x - x.transpose(1, 0, 2, 3) - x.transpose(0, 1, 3, 2) + x.transpose(1, 0, 3, 2)
    Ra, Rb = X.chol.a.vv, X.chol.b.vv
    if key == "ab":
        if core_virtual is None:
            return _x2b(T, X, H, FULL) + _ladder(Ra, Rb, T.ab)
        va, vb, ca, cb = core_virtual["ab"]
        A, B, Oa, Ob = slice(0, va.start), slice(0, vb.start), slice(ca.stop, None), slice(cb.stop, None)
        R = np.zeros_like(T.ab)
        for s in ((full, B, full, full), (A, full, full, full)):
            R[s] = _x2b(T, X, H, s) + _ladder(Ra, Rb, T.ab, *s[:2])
        # (i in O, any j) and (i in C, j in O): one ladder over their columns
        cols = [T.ab[:, :, Oa, :], T.ab[:, :, ca, Ob]]
        L = _ladder(Ra, Rb, np.concatenate([c.reshape(c.shape[0], c.shape[1], -1) for c in cols], axis=2))
        n0 = cols[0].shape[2] * cols[0].shape[3]
        R[:, :, Oa] = _x2b(T, X, H, (full, full, Oa, full)) + L[..., :n0].reshape(cols[0].shape)
        R[:, :, ca, Ob] = _x2b(T, X, H, (full, full, ca, Ob)) + L[..., n0:].reshape(cols[1].shape)
        return R
    res, t2, Rv = (_x2a, T.aa, Ra) if key == "aa" else (_x2c, T.bb, Rb)
    # the ladder part of the antisymmetric residual, 1/2 <ab||ef> t2(efij) = (ae|bf) t2(efij)
    if core_virtual is None:
        return asym(res(T, X, H, FULL)) + _ladder(Rv, Rv, t2, sym=True)
    va, _, ca, _ = core_virtual[key]
    A, O = slice(0, va.start), slice(ca.stop, None)
    R = np.zeros_like(t2)
    # A(ab)A(ij) of the unantisymmetrized x needs x on the block and on its mirror
    x, y = res(T, X, H, (A, full, full, full)), res(T, X, H, (full, A, full, full))
    R[A] = x - y.transpose(1, 0, 2, 3) - x.transpose(0, 1, 3, 2) + y.transpose(1, 0, 3, 2) + _ladder(Rv, Rv, t2, A)
    R[:, A] = -R[A].transpose(1, 0, 2, 3)
    x, y = res(T, X, H, (full, full, O, full)), res(T, X, H, (full, full, full, O))
    R[:, :, O] = x - x.transpose(1, 0, 2, 3) - y.transpose(0, 1, 3, 2) + y.transpose(1, 0, 3, 2) \
        + _ladder(Rv, Rv, t2[:, :, O])
    R[:, :, :, O] = -R[:, :, O].transpose(0, 1, 3, 2)
    return R


def update(T, dT, H, X, coupling, shift, internal, core_virtual, frozen, t34=None, act=None):
    """One update of T(p): the DF CCSD direct term (ccpy.cc.ccsd_chol: singles, then the doubles
    of each spin case, Gauss-Seidel), plus the terms of the internal T3/T4 (t34, from
    get_internal_t3_t4), corrected by -coupling / D and masked before the next spin case. For
    SUMRCCSD(1) (frozen) the doubles residual is evaluated only where it is solved."""
    C1a, C1b, C2a, C2b, C2c = coupling
    D = get_denominators(H, shift)
    X1 = get_internal_t3_singles(t34, H, act, T)

    def correct(key, c):
        step = c / D[key]
        getattr(T, key)[...] -= step
        getattr(dT, key)[...] -= step
        apply_masks(T, dT, internal, core_virtual, frozen, keys=(key,))

    X = ccsd_chol.get_pre_intermediates(X, T, H, False)
    T, dT = ccsd_chol.update_t1a(T, dT, X, H, shift)
    correct("a", C1a - X1["a"])
    T, dT = ccsd_chol.update_t1b(T, dT, X, H, shift)
    correct("b", C1b - X1["b"])
    apply_masks(T, dT, internal, core_virtual, frozen, keys=("aa", "ab", "bb"))  # internal t2 from the new t1

    # doubles coupling uses the updated t1(p) and the coupling C1 alone (not the T3 singles terms)
    W = np.einsum("ai,bj->abij", T.a, C1a, optimize=True)
    C2a = C2a - (W - W.transpose(1, 0, 2, 3) - W.transpose(0, 1, 3, 2) + W.transpose(1, 0, 3, 2))
    W = np.einsum("ai,bj->abij", T.b, C1b, optimize=True)
    C2c = C2c - (W - W.transpose(1, 0, 2, 3) - W.transpose(0, 1, 3, 2) + W.transpose(1, 0, 3, 2))
    C2b = C2b - np.einsum("ai,bj->abij", T.a, C1b, optimize=True) - np.einsum("ai,bj->abij", C1a, T.b, optimize=True)

    X = ccsd_chol.get_t1_transformed(X, T, H, False)
    X2 = get_internal_t3_t4_doubles(t34, H, X, act, T)
    for key, c in (("aa", C2a - X2["aa"]), ("ab", C2b - X2["ab"]), ("bb", C2c - X2["bb"])):
        step = (get_t2_residual(T, X, H, key, None if frozen is None else core_virtual) - c) / D[key]
        if frozen is not None:
            step[core_virtual[key]] = 0.0
        t = getattr(T, key)
        new = t + step
        if key != "ab":  # keep t2 exactly antisymmetric: roundoff in it would otherwise grow by 1/D
            new = 0.25 * (new - new.transpose(1, 0, 2, 3) - new.transpose(0, 1, 3, 2) + new.transpose(1, 0, 3, 2))
        setattr(dT, key, new - t)
        t[...] = new
        apply_masks(T, dT, internal, core_virtual, frozen, keys=(key,))
    return T, dT


def check_intruders(estimates, keep=None):
    """Warn when a single or double of some reference, at the orbital-energy (Jacobi
    denominator) estimate, lies below the highest reference: an intruder situation in which
    the Jacobi iterations usually fail. estimates come from
    ccpy.mrcc.sumrcc.get_external_estimates; keep selects the references (all by default)."""
    keep = range(len(estimates)) if keep is None else keep
    lowest, highest = get_lowest_external(estimates, keep), max(estimates[p][0] for p in keep)
    if lowest < highest:
        print(f"   WARNING: external determinant estimated at {lowest:.6f} lies below the highest"
              f" reference at {highest:.6f} (intruder states)")
    return lowest < highest


def kernel(e1int, e2int, system, determinants, frozen_cv=False, solver="auto", prune_intruders=None,
           maxit=200, convergence=1.0e-08, shift=0.0, diis_size=6):
    """Solve the SU-MRCCSD equations for the model space spanned by `determinants` (see
    ccpy.mrcc.sumrcc; get_cas_determinants gives a CAS). e1int and e2int (physics notation)
    span all orbitals, frozen ones included.
    solver="jacobi" iterates the update with DIIS; solver="newton_krylov" finds the zero of
    the same update step with scipy's Jacobian-free Newton-Krylov, which can converge in
    intruder cases where the Jacobi iterations fail (maxit is then Newton iterations);
    solver="auto" runs Jacobi and, if it fails, Newton-Krylov from its best iterate.
    prune_intruders=gap (Eh) first drops determinants, highest energy first, until every
    single or double of the remaining ones lies at least `gap` above the highest of them
    (orbital-energy estimate; 0.0 drops what the intruder warning flags), turning the model
    space into a general one (C conditions). No gap guarantees convergence.
    Returns the cluster operators, the effective Hamiltonian, its eigenvalues (ascending real
    part) and right eigenvectors, and the model space (ccpy.mrcc.sumrcc.get_model_space)."""
    B = as_df(e2int)
    estimates = get_external_estimates(e1int, B, system, determinants)
    if prune_intruders is not None:
        keep, dropped = prune_model_space(determinants, estimates, prune_intruders)
        if dropped:
            print(f"   Removed {len(dropped)} intruder determinant(s) from the model space:")
            for p in dropped:
                print(f"      {list(np.asarray(determinants[p]).tolist())}   E = {estimates[p][0]:.6f}")
        determinants, estimates = [determinants[p] for p in keep], [estimates[p] for p in keep]
    ms = get_model_space(system, determinants)
    nref, internal = len(ms.references), ms.internal
    H, e_ref = zip(*[get_reference_hamiltonian(e1int, B, system, ref) for ref in ms.references])
    core_virtual = get_core_virtual(system, ms)
    act = get_active_slices(system, ms)
    frozen = get_frozen_amplitudes(H[0], core_virtual) if frozen_cv else None

    T = [ClusterOperator(system, 2) for _ in range(nref)]
    dT = [ClusterOperator(system, 2) for _ in range(nref)]
    for p in range(nref):
        apply_masks(T[p], dT[p], internal[p], core_virtual, frozen)
    X = Integral.from_empty(system, 2, use_none=True)
    X.chol = Integral.from_empty(system, 1, use_none=True)
    # the solver, DIIS and stored projections only see the non-frozen amplitudes
    free = {key: np.ones(getattr(T[0], key).shape, bool) for key in KEYS}
    if frozen_cv:
        for key in KEYS:
            free[key][core_virtual[key]] = False
    ndim = sum(int(m.sum()) for m in free.values())
    # the doubles coupling -A t1(p) C1(p) needs C1 on the core-virtual singles too
    stored = dict(free, a=np.ones_like(free["a"]), b=np.ones_like(free["b"]))
    pack = lambda t: np.concatenate([getattr(t, key)[free[key]] for key in KEYS])

    def unpack(t, v):
        offset = 0
        for key in KEYS:
            n = int(free[key].sum())
            getattr(t, key)[free[key]] = v[offset:offset + n]
            offset += n

    def step(x):
        """Update step dT(T) for the stacked amplitudes x (internal ones reset by the C
        conditions); its zero solves the SU equations. Also returns Heff and the updated T."""
        for p in range(nref):
            unpack(T[p], x[p * ndim:(p + 1) * ndim])
            apply_masks(T[p], dT[p], internal[p], core_virtual, frozen)
        amps = [get_internal_amplitudes(T[q], ms, q, system) for q in range(nref)]
        heff, C = get_heff_and_couplings(T, H, e_ref, ms, system, amps, free=stored)
        for p in range(nref):
            update(T[p], dT[p], H[p], X, C[p], shift, internal[p], core_virtual, frozen,
                   get_internal_t3_t4(amps[p], ms, p, system), act)
        return np.concatenate([pack(d) for d in dT]), heff, np.concatenate([pack(t) for t in T])

    print(f"   SU-MRCCSD{'(1)' if frozen_cv else ''}: {nref} references, {solver} solver")
    check_intruders(estimates)
    print("   Iter        Residuum          Lowest energy            dE        Time")
    x = np.concatenate([pack(t) for t in T])
    status = {"niter": 0, "energy": np.zeros(nref), "tic": time.perf_counter()}

    def report(x, residual, heff):
        if not np.all(np.isfinite(heff)):
            raise RuntimeError("SU-MRCCSD diverged")
        energy = np.sort(np.linalg.eigvals(heff).real)
        delta = np.abs(energy - status["energy"]).max()
        print(f"   {status['niter']:4d}  {np.linalg.norm(residual):16.10e}  {energy[0]:20.12f}  {delta:12.4e}"
              f"  {time.perf_counter() - status['tic']:6.2f}s")
        status["niter"] += 1
        status["energy"] = energy
        status["tic"] = time.perf_counter()
        return delta

    best = {"x": x, "norm": np.inf}

    def jacobi(x):
        diis = DIIS(SimpleNamespace(ndim=nref * ndim, a=T[0].a), diis_size, False) if diis_size > 0 else None
        vector = lambda v: SimpleNamespace(flatten=lambda: v)
        for niter in range(maxit):
            residual, heff, x_new = step(x)
            delta = report(x, residual, heff)
            if np.linalg.norm(residual) < best["norm"]:
                best["x"], best["norm"] = x, np.linalg.norm(residual)
            x = x_new
            if np.linalg.norm(residual) < convergence and delta < convergence:
                return x
            if diis is not None:
                diis.push(vector(x), vector(residual), niter)
                if niter >= diis_size:
                    x = diis.extrapolate()
        raise RuntimeError("SU-MRCCSD did not converge")

    def newton(x):
        from scipy.optimize import newton_krylov, NoConvergence
        try:
            return newton_krylov(lambda v: step(v)[0], x, f_tol=convergence, method="lgmres", maxiter=maxit,
                                 callback=lambda v, f: report(v, f, step(v)[1]))
        except NoConvergence as err:
            raise RuntimeError("SU-MRCCSD did not converge") from err

    if solver == "jacobi":
        x = jacobi(x)
    elif solver == "newton_krylov":
        x = newton(x)
    elif solver == "auto":
        try:
            x = jacobi(x)
        except RuntimeError as err:
            print(f"   {err}; Newton-Krylov from the Jacobi iterate with residuum {best['norm']:.4e}")
            x = newton(best["x"])
    else:
        raise ValueError(f"unknown solver {solver}")

    residual, heff, _ = step(x)
    for p in range(nref):
        unpack(T[p], x[p * ndim:(p + 1) * ndim])
        apply_masks(T[p], dT[p], internal[p], core_virtual, frozen)
    energy, coeff = np.linalg.eig(heff)
    order = np.argsort(energy.real)
    print("   SU-MRCCSD energies:", " ".join(f"{e:.10f}" for e in energy.real[order]))
    return T, heff, energy[order], coeff[:, order], ms
