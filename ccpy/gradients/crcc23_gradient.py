"""Analytic nuclear gradient of CR-CC(2,3)_A, factorized with density fitting or Cholesky
vectors, RHF or UHF reference (ccpq_analytical_gradients.pdf).

CR-CC(2,3) is CC(P;Q) with P = singles and doubles, Q = all triples,

    delta = sum_K N_K M_K / D_K,   M_K = <K|Hbar|0>,   N_K = <0|(L1 + L2) Hbar|K>,
    D_K   = f_ii + f_jj + f_kk - f_aa - f_bb - f_cc      (bare Fock diagonal, variant A),

and the gradient of E_CCSD + delta is assembled in five steps:

  1. sources: delta and, with u = M/D, v = N/D, the derivatives of delta with respect to the
     left-CCSD amplitudes, s_lambda = d(u . N)/dLambda, and to T, s_t = d(v . M)/dT (Eqs. 6.3,
     6.4); with them the triples densities of v . M + u . N and the bare-Fock vector d;
  2. responses (the CCSD-level kernels of ccpy.gradients.ccsd_gradient), in this order:
     B sigma = -s_lambda (EOMCCSD right Jacobian, HR), the source
     k = B^T w + B^T c (the doubly connected term B^T w, plus the singles
     c_i^a = sum lambda_ij^ab sigma_j^b), and B^T pi = -s_t - k (left-CCSD Jacobian, LH);
  3. correction densities: pi (left-CCSD-like), sigma (the T-directional derivative of the
     CCSD Lagrangian along sigma, plus c), the triples, and the bare-Fock denominator;
  4. orbital relaxation (ccpy.gradients.orbital_response);
  5. contraction with the derivative integrals (ccpy.gradients.derivative_integrals).

The triples are never stored. The moments are written as M = A(abc) A(ijk) sum_g X_g . t_g,
with intermediates X_g (o v^3 or o^3 v; build_triples_intermediates_* below, the terms of
ccpy.gradients.crcc23_gradient_reference.build_M3A-D grouped by their last T2); the Fortran loops of
ccpy/lib/fortran/crcc23_gradient_loops.f90 build M, N and D for one occupied triple at a time
over all virtual triples and return the cotangents of X_g, T2, Lambda and the H blocks, and
backprop_triples_intermediates_* finishes the chain rule through X_g. The full-array
implementation, ccpy.gradients.crcc23_gradient_reference, is the reference.

Everything that depends on T uses the CCS-transformed H = e^{-T1} H e^{T1}
(ccpy.hbar.hbar_ccs_chol), so the singles of every T source are orbital-rotation derivatives
of the corresponding densities. No quantity with four virtual indices is stored.
"""

import time

import numpy as np

from ccpy.gradients.ccsd_gradient import (add_sd, apply_action, calc_bra_density, calc_BTw, calc_c_singles,
                                          calc_w_density, left_action, right_action, solve_linear)
from ccpy.gradients.derivative_integrals import contract_derivative_integrals, print_gradient
from ccpy.gradients.density_gradient import SPINS, add_densities, build_DY, rotation_gradient
from ccpy.gradients.orbital_response import orbital_relaxation
from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol
from ccpy.integrals import integrals
from ccpy.lib.core import crcc23_gradient_loops
from ccpy.models.integrals import Integral


def calc_crcc23_gradient(T, L, Hbar, reference, delta=None, zvector_tol=1.0e-10, components=False, verbose=True):
    """Analytic CR-CC(2,3)_A nuclear gradient, (natoms, 3) in Eh/bohr, from converged CCSD (T),
    left-CCSD (L) and the CCSD HBar (Hbar) on the SCF reference (ccpy.methods.Reference).
    components=True returns a dict with the base (CCSD, nuclear repulsion included), correction
    and total gradients."""
    t0 = time.perf_counter()
    system, H0 = reference.system, reference.H
    H = build_hbar_ccs_chol(T, H0, system)

    # 1. sources
    X = build_triples_intermediates(T, H)
    tr = calc_triples(X, T, L, H, reference)
    if delta is not None and abs(tr["delta"] - delta) > 1.0e-8:
        raise RuntimeError(f"delta from the moments ({tr['delta']:.12f}) does not match the energy ({delta:.12f}).")
    dT, rdm1, rdm2 = backprop_triples_intermediates(tr["dX"], T, H, system)
    dT = {c: dT[c] + tr["dT"][c] for c in dT}
    rho_triples = build_DY(*add_rdm(tr["rdm"], (rdm1, rdm2), system), None, H.chol, T, reference.B_mo, reference)
    singles = rotation_gradient(*rho_triples, reference.h_mo, reference.B_mo, reference)
    s_t = {"a": singles["a"], "b": singles["b"], **antisymmetrize_doubles(dT)}
    s_lambda = {"a": tr["dL"]["a"], "b": tr["dL"]["b"], **antisymmetrize_doubles(tr["dL"])}

    # 2. responses
    sigma = solve_linear(right_action(T, Hbar, system), {c: -x for c, x in s_lambda.items()}, T, Hbar, zvector_tol)
    k = calc_k(sigma, T, L, H, Hbar, reference)
    pi = solve_linear(left_action(T, Hbar, system), {c: -(s_t[c] + k[c]) for c in s_t}, T, Hbar, zvector_tol)

    # 3. densities
    # <L|Hbar|Sigma> - (lambda . sigma) E = D_sigma L + <C|Hbar|0>, c of calc_k
    base = calc_bra_density(1.0, L, T, H, reference)
    corr = add_densities(
        calc_bra_density(0.0, add_sd(pi, calc_c_singles(sigma, L)), T, H, reference),
        calc_w_density(sigma, T, L, H, reference),
        rho_triples,
        calc_denominator_density(tr["d"], H0, reference),
    )

    # 4. orbital relaxation and 5. derivative integrals
    D, Y, W = orbital_relaxation(*add_densities(base, corr), reference, zvector_tol=zvector_tol, verbose=verbose)
    grad = contract_derivative_integrals(D, Y, W, reference)
    if verbose:
        print_gradient(reference.molecule, grad, "CR-CC(2,3)_A Analytic Gradient", time.perf_counter() - t0)
    if not components:
        return grad
    D, Y, W = orbital_relaxation(*corr, reference, zvector_tol=zvector_tol, verbose=False)
    grad_corr = contract_derivative_integrals(D, Y, W, reference) - integrals.nuclear_repulsion_deriv(reference.molecule)
    return {"base": grad - grad_corr, "correction": grad_corr, "total": grad}


# ======================================================================================
# step 1: the triples
# ======================================================================================
def calc_triples(X, T, L, H, reference, omega=0.0, R=None, r0=0.0, J=None):
    """One pass of the Fortran triples loops over the four spin cases.

    Ground state: X = build_triples_intermediates(T, H), L = left-CCSD. Excited state
    (ccpy.gradients.creomcc23_gradient): also R, r0 and J_g = r0 X_g + D_R X_g, and L the left
    EOMCCSD vector; the moments are then those of R and the denominators include omega.

    Returns delta, W (= sum_K u_K v_K), F_r0 (= v . M(ground)), the bare-Fock vector d, and the
    raw cotangents: dX (of X_g), dT (of T2), dL (of L1, L2), rdm (of the H blocks of N),
    and, for an excited state, dJT (v . J_g -> T2), dXR (v . r_g -> X_g) and c' (<V3|R|X>).
    Raw: every stored array entry is treated as independent (antisymmetrize_doubles gives the
    unique-amplitude gradients)."""
    system = reference.system
    excited = R is not None
    if not excited:
        R, J = T, X
    nfz = system.nfrozen
    noa, nob = system.noccupied_alpha, system.noccupied_beta
    nua, nub = system.nunoccupied_alpha, system.nunoccupied_beta
    eps = {s: reference.mo_energy[s][nfz:] for s in SPINS}
    eoa, eva = eps["a"][:noa], eps["a"][noa:noa + nua]
    eob, evb = eps["b"][:nob], eps["b"][nob:nob + nub]
    f = np.asfortranarray
    z = lambda x: np.zeros(np.shape(x), order="F")  # noqa: E731

    out = {"delta": 0.0, "W": 0.0, "F_r0": 0.0,
           "d": {"a": {"o": np.zeros(noa), "v": np.zeros(nua)}, "b": {"o": np.zeros(nob), "v": np.zeros(nub)}},
           "dX": {}, "dXR": {}, "dT": {c: np.zeros_like(getattr(T, c)) for c in ("aa", "ab", "bb")},
           "dJT": {c: np.zeros_like(getattr(T, c)) for c in ("aa", "ab", "bb")},
           "dL": {c: np.zeros_like(getattr(L, c)) for c in ("a", "b", "aa", "ab", "bb")},
           "cprime": {c: np.zeros_like(getattr(T, c)) for c in ("a", "b", "aa", "ab", "bb")}}
    rdm1 = Integral.from_empty(system, 1, use_none=True)
    rdm2 = Integral.from_empty(system, 2, use_none=True)

    # ---- aaa and bbb: x1(a,m,i,j) . t2(b,c,m,k), x2(a,b,i,e) . t2(e,c,j,k) (passed as (a,b,e,i))
    for case, s, ss, t2, r1, r2, l1, l2, eo, ev, no, nu in (
            ("aaa", "a", "aa", T.aa, R.a, R.aa, L.a, L.aa, eoa, eva, noa, nua),
            ("bbb", "b", "bb", T.bb, R.b, R.bb, L.b, L.bb, eob, evb, nob, nub)):
        x1, x2 = X[case]
        j1, j2 = J[case]
        hvovv_r = f(getattr(H, ss).vovv.transpose(3, 2, 0, 1))
        hooov_r = f(getattr(H, ss).ooov.transpose(3, 2, 0, 1))
        acc = [z(x1), z(x2.transpose(0, 1, 3, 2)), z(t2), z(x1), z(x2.transpose(0, 1, 3, 2)), z(t2),
               z(r1), z(r2), z(l1), z(l2), z(hvovv_r), z(hooov_r), z(getattr(H, ss).oovv), z(getattr(H, s).ov)]
        d_o, d_v = np.zeros(no), np.zeros(nu)
        delta, w, fr0 = crcc23_gradient_loops.grad_aaa(
            f(x1), f(x2.transpose(0, 1, 3, 2)), f(t2), f(j1), f(j2.transpose(0, 1, 3, 2)), f(r1), f(r2),
            hvovv_r, hooov_r, f(getattr(H, ss).oovv), f(getattr(H, s).ov), f(l1), f(l2),
            eo, ev, omega, int(excited), d_o, d_v, *acc, no, nu)
        dx1, dx2r, dt2, dxr1, dxr2r, dtj2, cp1, cp2, dl1, dl2, dhvovv_r, dhooov_r, dhoovv, dhov = acc
        out["delta"] += delta
        out["W"] += w
        out["F_r0"] += fr0
        out["d"][s]["o"] += d_o
        out["d"][s]["v"] += d_v
        out["dX"][case] = [dx1, dx2r.transpose(0, 1, 3, 2)]
        out["dXR"][case] = [dxr1, dxr2r.transpose(0, 1, 3, 2)]
        out["dT"][ss] += dt2
        out["dJT"][ss] += dtj2
        out["cprime"][s] += cp1
        out["cprime"][ss] += cp2
        out["dL"][s] += dl1
        out["dL"][ss] += dl2
        sub1, sub2 = getattr(rdm1, s), getattr(rdm2, ss)
        sub2.vovv = _plus(sub2.vovv, dhvovv_r.transpose(2, 3, 1, 0))
        sub2.ooov = _plus(sub2.ooov, dhooov_r.transpose(2, 3, 1, 0))
        sub2.oovv = _plus(sub2.oovv, dhoovv)
        sub1.ov = _plus(sub1.ov, dhov)

    # ---- aab: groups (layouts as passed to grad_aab)
    #   x1(b,c,e,k).t2a(a,e,i,j)  x2(m,c,j,k).t2a(a,b,i,m)  x3(a,c,i,e).t2b(b,e,j,k)
    #   x4(a,m,i,k).t2b(b,c,j,m)  x5(a,b,i,e).t2b(e,c,j,k)  x6(a,m,i,j).t2b(b,c,m,k)
    x1, x2, x3, x4, x5, x6 = X["aab"]
    j1, j2, j3, j4, j5, j6 = J["aab"]
    h = [f(H.ab.vovv.transpose(0, 2, 3, 1)), f(H.aa.vovv.transpose(3, 2, 0, 1)), f(H.ab.ovvv.transpose(2, 3, 1, 0)),
         f(H.ab.ooov.transpose(2, 3, 0, 1)), f(H.aa.ooov.transpose(3, 2, 0, 1)), f(H.ab.oovo.transpose(2, 3, 0, 1))]
    swap = lambda x: f(x.transpose(0, 1, 3, 2))  # noqa: E731
    xs = [f(x1), f(x2), swap(x3), f(x4), swap(x5), f(x6)]
    js = [f(j1), f(j2), swap(j3), f(j4), swap(j5), f(j6)]
    ts = [f(T.aa), swap(T.aa), f(T.ab), swap(T.ab)]
    rs = [f(R.aa), swap(R.aa), f(R.ab), swap(R.ab)]
    hs = [f(H.ab.oovv), f(H.aa.oovv), f(H.a.ov), f(H.b.ov)]
    ls = [f(L.a), f(L.b), f(L.aa), swap(L.aa), f(L.ab), swap(L.ab)]
    d_ = [np.zeros(noa), np.zeros(nob), np.zeros(nua), np.zeros(nub)]
    dxs, dts, dxrs, dtjs = [z(x) for x in xs], [z(t) for t in ts], [z(x) for x in xs], [z(t) for t in ts]
    cps = [z(R.a), z(R.b), z(R.aa), z(R.ab)]
    dhs, dhs2, dls = [z(x) for x in h], [z(x) for x in hs], [z(x) for x in ls]
    delta, w, fr0 = crcc23_gradient_loops.grad_aab(*xs, *ts, *js, f(R.a), f(R.b), *rs, *h, *hs, *ls,
                                                   eoa, eob, eva, evb, omega, int(excited), *d_,
                                                   *dxs, *dts, *dxrs, *dtjs, *cps, *dhs, *dhs2, *dls,
                                                   noa, nob, nua, nub)
    _collect_mixed(out, "aab", delta, w, fr0, d_, dxs, dts, dxrs, dtjs, cps, dhs, dhs2, dls, rdm1, rdm2)

    # ---- abb: groups (layouts as passed to grad_abb)
    #   x1(c,b,k,e).t2c(a,e,i,j)  x2(c,m,k,j).t2c(a,b,i,m)  x3(c,a,e,i).t2b(e,b,k,j)
    #   x4(m,a,k,i).t2b(c,b,m,j)  x5(a,b,i,e).t2b(c,e,k,j)  x6(a,m,i,j).t2b(c,b,k,m)
    x1, x2, x3, x4, x5, x6 = X["abb"]
    j1, j2, j3, j4, j5, j6 = J["abb"]
    h = [f(H.ab.ovvv.transpose(2, 3, 1, 0)), f(H.bb.vovv.transpose(0, 2, 3, 1)), f(H.ab.vovv.transpose(2, 3, 0, 1)),
         f(H.ab.oovo.transpose(2, 3, 0, 1)), f(H.bb.ooov.transpose(3, 2, 0, 1)), f(H.ab.ooov.transpose(3, 2, 0, 1))]
    xs = [swap(x1), f(x2), f(x3), f(x4), swap(x5), f(x6)]
    js = [swap(j1), f(j2), f(j3), f(j4), swap(j5), f(j6)]
    ts = [f(T.bb), swap(T.bb), f(T.ab), swap(T.ab)]
    rs = [f(R.bb), swap(R.bb), f(R.ab), swap(R.ab)]
    hs = [f(H.ab.oovv), f(H.bb.oovv), f(H.a.ov), f(H.b.ov)]
    ls = [f(L.a), f(L.b), f(L.bb), swap(L.bb), f(L.ab), swap(L.ab)]
    d_ = [np.zeros(noa), np.zeros(nob), np.zeros(nua), np.zeros(nub)]
    dxs, dts, dxrs, dtjs = [z(x) for x in xs], [z(t) for t in ts], [z(x) for x in xs], [z(t) for t in ts]
    cps = [z(R.a), z(R.b), z(R.ab), z(R.bb)]
    dhs, dhs2, dls = [z(x) for x in h], [z(x) for x in hs], [z(x) for x in ls]
    delta, w, fr0 = crcc23_gradient_loops.grad_abb(*xs, *ts, *js, f(R.a), f(R.b), *rs, *h, *hs, *ls,
                                                   eoa, eob, eva, evb, omega, int(excited), *d_,
                                                   *dxs, *dts, *dxrs, *dtjs, *cps, *dhs, *dhs2, *dls,
                                                   noa, nob, nua, nub)
    _collect_mixed(out, "abb", delta, w, fr0, d_, dxs, dts, dxrs, dtjs, cps, dhs, dhs2, dls, rdm1, rdm2)

    out["rdm"] = (rdm1, rdm2)
    return out


def _collect_mixed(out, case, delta, w, fr0, d_, dxs, dts, dxrs, dtjs, cps, dhs, dhs2, dls, rdm1, rdm2):
    """Map the outputs of grad_aab / grad_abb back to the natural layouts."""
    swap = lambda x: x.transpose(0, 1, 3, 2)  # noqa: E731
    out["delta"] += delta
    out["W"] += w
    out["F_r0"] += fr0
    for s, blk, x in (("a", "o", d_[0]), ("b", "o", d_[1]), ("a", "v", d_[2]), ("b", "v", d_[3])):
        out["d"][s][blk] += x
    same = "aa" if case == "aab" else "bb"
    for target, src in (("dX", dxs), ("dXR", dxrs)):
        if case == "aab":
            out[target][case] = [src[0], src[1], swap(src[2]), src[3], swap(src[4]), src[5]]
        else:
            out[target][case] = [swap(src[0]), src[1], src[2], src[3], swap(src[4]), src[5]]
    for target, src in (("dT", dts), ("dJT", dtjs)):
        out[target][same] += src[0] + swap(src[1])
        out[target]["ab"] += src[2] + swap(src[3])
    if case == "aab":
        cp = (("a", cps[0]), ("b", cps[1]), ("aa", cps[2]), ("ab", cps[3]))
    else:
        cp = (("a", cps[0]), ("b", cps[1]), ("ab", cps[2]), ("bb", cps[3]))
    for c, x in cp:
        out["cprime"][c] += x
    out["dL"]["a"] += dls[0]
    out["dL"]["b"] += dls[1]
    out["dL"][same] += dls[2] + swap(dls[3])
    out["dL"]["ab"] += dls[4] + swap(dls[5])
    if case == "aab":
        rdm2.ab.vovv = _plus(rdm2.ab.vovv, dhs[0].transpose(0, 3, 1, 2))
        rdm2.aa.vovv = _plus(rdm2.aa.vovv, dhs[1].transpose(2, 3, 1, 0))
        rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, dhs[2].transpose(3, 2, 0, 1))
        rdm2.ab.ooov = _plus(rdm2.ab.ooov, dhs[3].transpose(2, 3, 0, 1))
        rdm2.aa.ooov = _plus(rdm2.aa.ooov, dhs[4].transpose(2, 3, 1, 0))
        rdm2.ab.oovo = _plus(rdm2.ab.oovo, dhs[5].transpose(2, 3, 0, 1))
        rdm2.aa.oovv = _plus(rdm2.aa.oovv, dhs2[1])
    else:
        rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, dhs[0].transpose(3, 2, 0, 1))
        rdm2.bb.vovv = _plus(rdm2.bb.vovv, dhs[1].transpose(0, 3, 1, 2))
        rdm2.ab.vovv = _plus(rdm2.ab.vovv, dhs[2].transpose(2, 3, 0, 1))
        rdm2.ab.oovo = _plus(rdm2.ab.oovo, dhs[3].transpose(2, 3, 0, 1))
        rdm2.bb.ooov = _plus(rdm2.bb.ooov, dhs[4].transpose(2, 3, 1, 0))
        rdm2.ab.ooov = _plus(rdm2.ab.ooov, dhs[5].transpose(2, 3, 1, 0))
        rdm2.bb.oovv = _plus(rdm2.bb.oovv, dhs2[1])
    rdm2.ab.oovv = _plus(rdm2.ab.oovv, dhs2[0])
    rdm1.a.ov = _plus(rdm1.a.ov, dhs2[2])
    rdm1.b.ov = _plus(rdm1.b.ov, dhs2[3])


def _plus(old, new):
    return np.array(new) if old is None else old + new


def add_rdm(*pairs_and_system):
    """Blockwise sum of (rdm1, rdm2) pairs (the last argument is the System)."""
    *pairs, system = pairs_and_system
    rdm1 = Integral.from_empty(system, 1, use_none=True)
    rdm2 = Integral.from_empty(system, 2, use_none=True)
    for r1, r2 in pairs:
        for out, src, spins in ((rdm1, r1, ("a", "b")), (rdm2, r2, ("aa", "ab", "bb"))):
            for s in spins:
                for blk in getattr(src, s).slices:
                    x = getattr(getattr(src, s), blk)
                    if x is not None and not np.isscalar(x):
                        setattr(getattr(out, s), blk, _plus(getattr(getattr(out, s), blk), x))
    return rdm1, rdm2


def antisymmetrize_doubles(raw):
    """Gradients with respect to unique doubles from raw entry gradients (all copies filled)."""
    out = {}
    for c in ("aa", "ab", "bb"):
        x = raw[c]
        if c != "ab":
            x = x - x.transpose(1, 0, 2, 3)
            x = x - x.transpose(0, 1, 3, 2)
        out[c] = x
    return out


# ======================================================================================
# step 2: responses
# ======================================================================================


def calc_k(sigma, T, L, H, Hbar, reference):
    """k_I = <L|[Hbar, tau_I]|Sigma> - (lambda . sigma) e_I (Eq. 6.6): the doubly connected
    term B^T w at w = sigma plus J^T c for the singles c_i^a = sum lambda_ij^ab sigma_j^b
    (the disconnected terms (lambda . sigma) e_I cancel)."""
    system = reference.system
    LHc = apply_action(left_action(T, Hbar, system), calc_c_singles(sigma, L), T)
    k = calc_BTw(sigma, T, L, H, reference)
    return {x: k[x] + LHc[x] for x in ("a", "b", "aa", "ab", "bb")}


# ======================================================================================
# step 3: densities
# ======================================================================================


def calc_denominator_density(d, H0, reference):
    """(D1, Y) of sum_p d_p f_pp for the bare Fock diagonal of the denominators (Eqs. 8.9-8.10)."""
    system = reference.system
    rdm1 = Integral.from_empty(system, 1, use_none=True)
    rdm2 = Integral.from_empty(system, 2, use_none=True)
    for s in SPINS:
        getattr(rdm1, s).oo = np.diag(d[s]["o"])
        getattr(rdm1, s).vv = np.diag(d[s]["v"])
    return build_DY(rdm1, rdm2, None, H0.chol, None, reference.B_mo, reference)


# ======================================================================================
# intermediates of the moments (the terms of crcc23_gradient_reference.build_M3A-D grouped by their
# last T2) and their chain rule
# ======================================================================================
def build_triples_intermediates(T, H, constant=True):
    """X_g of the four spin cases (constant=False drops the bare H blocks, leaving the part
    linear in T2)."""
    return {"aaa": build_triples_intermediates_aaa(T, H, constant),
            "aab": build_triples_intermediates_aab(T, H, constant),
            "abb": build_triples_intermediates_abb(T, H, constant),
            "bbb": build_triples_intermediates_bbb(T, H, constant)}


def backprop_triples_intermediates(dX, T, H, system, constant=True):
    """Cotangents dX_g of the intermediates -> raw T2 gradient and the H(T1) densities."""
    dT = {c: np.zeros_like(getattr(T, c)) for c in ("aa", "ab", "bb")}
    rdm1 = Integral.from_empty(system, 1, use_none=True)
    rdm2 = Integral.from_empty(system, 2, use_none=True)
    backprop_triples_intermediates_aaa(dX["aaa"], T, H, dT, rdm1, rdm2, constant)
    backprop_triples_intermediates_aab(dX["aab"], T, H, dT, rdm1, rdm2, constant)
    backprop_triples_intermediates_abb(dX["abb"], T, H, dT, rdm1, rdm2, constant)
    backprop_triples_intermediates_bbb(dX["bbb"], T, H, dT, rdm1, rdm2, constant)
    return dT, rdm1, rdm2


def build_triples_intermediates_aaa(T, H, constant=True):
    """X_g of M3(aaa) = A sum_g X_g . t_g: amij . T.aa(bcmk), abie . T.aa(ecjk)."""
    X1 = -0.25 * np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    X1 -= 0.125 * np.einsum("amef,efij->amij", H.aa.vovv, T.aa, optimize=True)
    X1 -= 0.5 * np.einsum("mnjf,afin->amij", H.aa.ooov, T.aa, optimize=True)
    X1 -= 0.5 * np.einsum("mnjf,afin->amij", H.ab.ooov, T.ab, optimize=True)
    if constant:
        X1 -= 0.25 * H.aa.vooo
    X2 = 0.125 * np.einsum("mnie,abmn->abie", H.aa.ooov, T.aa, optimize=True)
    X2 += 0.5 * np.einsum("bnef,afin->abie", H.aa.vovv, T.aa, optimize=True)
    X2 += 0.5 * np.einsum("bnef,afin->abie", H.ab.vovv, T.ab, optimize=True)
    if constant:
        X2 += 0.25 * H.aa.vvov
    return [X1, X2]


def build_triples_intermediates_aab(T, H, constant=True):
    """X_g of M3(aab) = A sum_g X_g . t_g: bcek . T.aa(aeij), mcjk . T.aa(abim), acie . T.ab(bejk), amik . T.ab(bcjm), abie . T.ab(ecjk), amij . T.ab(bcmk)."""
    X1 = -0.5 * np.einsum("me,bcmk->bcek", H.a.ov, T.ab, optimize=True)
    X1 += 0.5 * np.einsum("nmek,bcnm->bcek", H.ab.oovo, T.ab, optimize=True)
    X1 += 0.5 * np.einsum("bmef,cfkm->bcek", H.ab.vovv, T.bb, optimize=True)
    X1 += 0.5 * np.einsum("bmef,fcmk->bcek", H.aa.vovv, T.ab, optimize=True)
    X1 -= 0.5 * np.einsum("mcef,bfmk->bcek", H.ab.ovvv, T.ab, optimize=True)
    if constant:
        X1 += 0.5 * H.ab.vvvo
    X2 = -0.5 * np.einsum("mnjf,fcnk->mcjk", H.aa.ooov, T.ab, optimize=True)
    X2 -= 0.5 * np.einsum("mnjf,fcnk->mcjk", H.ab.ooov, T.bb, optimize=True)
    X2 += 0.5 * np.einsum("mnfk,fcjn->mcjk", H.ab.oovo, T.ab, optimize=True)
    X2 -= 0.5 * np.einsum("mcfe,fejk->mcjk", H.ab.ovvv, T.ab, optimize=True)
    if constant:
        X2 -= 0.5 * H.ab.ovoo
    X3 = -np.einsum("me,acim->acie", H.b.ov, T.ab, optimize=True)
    X3 += np.einsum("nmie,acnm->acie", H.ab.ooov, T.ab, optimize=True)
    X3 += np.einsum("mcfe,afim->acie", H.ab.ovvv, T.aa, optimize=True)
    X3 += np.einsum("cmef,afim->acie", H.bb.vovv, T.ab, optimize=True)
    X3 -= np.einsum("amfe,fcim->acie", H.ab.vovv, T.ab, optimize=True)
    if constant:
        X3 += H.ab.vvov
    X4 = -np.einsum("mnkf,afin->amik", H.bb.ooov, T.ab, optimize=True)
    X4 -= np.einsum("nmfk,afin->amik", H.ab.oovo, T.aa, optimize=True)
    X4 += np.einsum("nmif,afnk->amik", H.ab.ooov, T.ab, optimize=True)
    X4 -= np.einsum("amef,efik->amik", H.ab.vovv, T.ab, optimize=True)
    if constant:
        X4 -= H.ab.vooo
    X5 = -0.5 * np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
    X5 += 0.25 * np.einsum("mnie,abmn->abie", H.aa.ooov, T.aa, optimize=True)
    X5 += np.einsum("bnef,afin->abie", H.aa.vovv, T.aa, optimize=True)
    X5 += np.einsum("bnef,afin->abie", H.ab.vovv, T.ab, optimize=True)
    if constant:
        X5 += 0.5 * H.aa.vvov
    X6 = -0.25 * np.einsum("amef,efij->amij", H.aa.vovv, T.aa, optimize=True)
    X6 -= np.einsum("mnjf,afin->amij", H.aa.ooov, T.aa, optimize=True)
    X6 -= np.einsum("mnjf,afin->amij", H.ab.ooov, T.ab, optimize=True)
    if constant:
        X6 -= 0.5 * H.aa.vooo
    return [X1, X2, X3, X4, X5, X6]


def build_triples_intermediates_abb(T, H, constant=True):
    """X_g of M3(abb) = A sum_g X_g . t_g: cbke . T.bb(aeij), cmkj . T.bb(abim), caei . T.ab(ebkj), maki . T.ab(cbmj), abie . T.ab(cekj), amij . T.ab(cbkm)."""
    X1 = -0.5 * np.einsum("me,cbkm->cbke", H.b.ov, T.ab, optimize=True)
    X1 += 0.5 * np.einsum("mnke,cbmn->cbke", H.ab.ooov, T.ab, optimize=True)
    X1 += 0.5 * np.einsum("mbfe,cfkm->cbke", H.ab.ovvv, T.aa, optimize=True)
    X1 += 0.5 * np.einsum("bmef,cfkm->cbke", H.bb.vovv, T.ab, optimize=True)
    X1 -= 0.5 * np.einsum("cmfe,fbkm->cbke", H.ab.vovv, T.ab, optimize=True)
    if constant:
        X1 += 0.5 * H.ab.vvov
    X2 = -0.5 * np.einsum("mnjf,cfkn->cmkj", H.bb.ooov, T.ab, optimize=True)
    X2 -= 0.5 * np.einsum("nmfj,fcnk->cmkj", H.ab.oovo, T.aa, optimize=True)
    X2 += 0.5 * np.einsum("nmkf,cfnj->cmkj", H.ab.ooov, T.ab, optimize=True)
    X2 -= 0.5 * np.einsum("cmef,efkj->cmkj", H.ab.vovv, T.ab, optimize=True)
    if constant:
        X2 -= 0.5 * H.ab.vooo
    X3 = -np.einsum("me,cami->caei", H.a.ov, T.ab, optimize=True)
    X3 += np.einsum("mnei,camn->caei", H.ab.oovo, T.ab, optimize=True)
    X3 += np.einsum("cmef,afim->caei", H.ab.vovv, T.bb, optimize=True)
    X3 += np.einsum("cmef,fami->caei", H.aa.vovv, T.ab, optimize=True)
    X3 -= np.einsum("maef,cfmi->caei", H.ab.ovvv, T.ab, optimize=True)
    if constant:
        X3 += H.ab.vvvo
    X4 = -np.einsum("mnkf,fani->maki", H.aa.ooov, T.ab, optimize=True)
    X4 -= np.einsum("mnkf,afin->maki", H.ab.ooov, T.bb, optimize=True)
    X4 += np.einsum("mnfi,fakn->maki", H.ab.oovo, T.ab, optimize=True)
    X4 -= np.einsum("mafe,feki->maki", H.ab.ovvv, T.ab, optimize=True)
    if constant:
        X4 -= H.ab.ovoo
    X5 = -0.5 * np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
    X5 += 0.25 * np.einsum("mnie,abmn->abie", H.bb.ooov, T.bb, optimize=True)
    X5 += np.einsum("bnef,afin->abie", H.bb.vovv, T.bb, optimize=True)
    X5 += np.einsum("nbfe,fani->abie", H.ab.ovvv, T.ab, optimize=True)
    if constant:
        X5 += 0.5 * H.bb.vvov
    X6 = -0.25 * np.einsum("amef,efij->amij", H.bb.vovv, T.bb, optimize=True)
    X6 -= np.einsum("mnjf,afin->amij", H.bb.ooov, T.bb, optimize=True)
    X6 -= np.einsum("nmfj,fani->amij", H.ab.oovo, T.ab, optimize=True)
    if constant:
        X6 -= 0.5 * H.bb.vooo
    return [X1, X2, X3, X4, X5, X6]


def build_triples_intermediates_bbb(T, H, constant=True):
    """X_g of M3(bbb) = A sum_g X_g . t_g: amij . T.bb(bcmk), abie . T.bb(ecjk)."""
    X1 = -0.25 * np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    X1 -= 0.125 * np.einsum("amef,efij->amij", H.bb.vovv, T.bb, optimize=True)
    X1 -= 0.5 * np.einsum("mnjf,afin->amij", H.bb.ooov, T.bb, optimize=True)
    X1 -= 0.5 * np.einsum("nmfj,fani->amij", H.ab.oovo, T.ab, optimize=True)
    if constant:
        X1 -= 0.25 * H.bb.vooo
    X2 = 0.125 * np.einsum("mnie,abmn->abie", H.bb.ooov, T.bb, optimize=True)
    X2 += 0.5 * np.einsum("bnef,afin->abie", H.bb.vovv, T.bb, optimize=True)
    X2 += 0.5 * np.einsum("nbfe,fani->abie", H.ab.ovvv, T.ab, optimize=True)
    if constant:
        X2 += 0.25 * H.bb.vvov
    return [X1, X2]


def backprop_triples_intermediates_aaa(dX, T, H, dT, rdm1, rdm2, constant=True):
    """Chain rule of build_triples_intermediates_aaa: dX_g -> dT (raw) and the H densities."""
    dX1, dX2 = dX
    if constant:
        rdm2.aa.vooo = _plus(rdm2.aa.vooo, -0.25 * dX1)
    rdm1.a.ov = _plus(rdm1.a.ov, -0.25 * np.einsum("amij,aeij->me", dX1, T.aa, optimize=True))
    dT["aa"] -= 0.25 * np.einsum("amij,me->aeij", dX1, H.a.ov, optimize=True)
    rdm2.aa.vovv = _plus(rdm2.aa.vovv, -0.125 * np.einsum("amij,efij->amef", dX1, T.aa, optimize=True))
    dT["aa"] -= 0.125 * np.einsum("amij,amef->efij", dX1, H.aa.vovv, optimize=True)
    rdm2.aa.ooov = _plus(rdm2.aa.ooov, -0.5 * np.einsum("amij,afin->mnjf", dX1, T.aa, optimize=True))
    dT["aa"] -= 0.5 * np.einsum("amij,mnjf->afin", dX1, H.aa.ooov, optimize=True)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, -0.5 * np.einsum("amij,afin->mnjf", dX1, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("amij,mnjf->afin", dX1, H.ab.ooov, optimize=True)
    if constant:
        rdm2.aa.vvov = _plus(rdm2.aa.vvov, 0.25 * dX2)
    rdm2.aa.ooov = _plus(rdm2.aa.ooov, 0.125 * np.einsum("abie,abmn->mnie", dX2, T.aa, optimize=True))
    dT["aa"] += 0.125 * np.einsum("abie,mnie->abmn", dX2, H.aa.ooov, optimize=True)
    rdm2.aa.vovv = _plus(rdm2.aa.vovv, 0.5 * np.einsum("abie,afin->bnef", dX2, T.aa, optimize=True))
    dT["aa"] += 0.5 * np.einsum("abie,bnef->afin", dX2, H.aa.vovv, optimize=True)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, 0.5 * np.einsum("abie,afin->bnef", dX2, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("abie,bnef->afin", dX2, H.ab.vovv, optimize=True)


def backprop_triples_intermediates_aab(dX, T, H, dT, rdm1, rdm2, constant=True):
    """Chain rule of build_triples_intermediates_aab: dX_g -> dT (raw) and the H densities."""
    dX1, dX2, dX3, dX4, dX5, dX6 = dX
    if constant:
        rdm2.ab.vvvo = _plus(rdm2.ab.vvvo, 0.5 * dX1)
    rdm1.a.ov = _plus(rdm1.a.ov, -0.5 * np.einsum("bcek,bcmk->me", dX1, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("bcek,me->bcmk", dX1, H.a.ov, optimize=True)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, 0.5 * np.einsum("bcek,bcnm->nmek", dX1, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("bcek,nmek->bcnm", dX1, H.ab.oovo, optimize=True)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, 0.5 * np.einsum("bcek,cfkm->bmef", dX1, T.bb, optimize=True))
    dT["bb"] += 0.5 * np.einsum("bcek,bmef->cfkm", dX1, H.ab.vovv, optimize=True)
    rdm2.aa.vovv = _plus(rdm2.aa.vovv, 0.5 * np.einsum("bcek,fcmk->bmef", dX1, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("bcek,bmef->fcmk", dX1, H.aa.vovv, optimize=True)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, -0.5 * np.einsum("bcek,bfmk->mcef", dX1, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("bcek,mcef->bfmk", dX1, H.ab.ovvv, optimize=True)
    if constant:
        rdm2.ab.ovoo = _plus(rdm2.ab.ovoo, -0.5 * dX2)
    rdm2.aa.ooov = _plus(rdm2.aa.ooov, -0.5 * np.einsum("mcjk,fcnk->mnjf", dX2, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("mcjk,mnjf->fcnk", dX2, H.aa.ooov, optimize=True)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, -0.5 * np.einsum("mcjk,fcnk->mnjf", dX2, T.bb, optimize=True))
    dT["bb"] -= 0.5 * np.einsum("mcjk,mnjf->fcnk", dX2, H.ab.ooov, optimize=True)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, 0.5 * np.einsum("mcjk,fcjn->mnfk", dX2, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("mcjk,mnfk->fcjn", dX2, H.ab.oovo, optimize=True)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, -0.5 * np.einsum("mcjk,fejk->mcfe", dX2, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("mcjk,mcfe->fejk", dX2, H.ab.ovvv, optimize=True)
    if constant:
        rdm2.ab.vvov = _plus(rdm2.ab.vvov, 1 * dX3)
    rdm1.b.ov = _plus(rdm1.b.ov, -1 * np.einsum("acie,acim->me", dX3, T.ab, optimize=True))
    dT["ab"] -= np.einsum("acie,me->acim", dX3, H.b.ov, optimize=True)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, 1 * np.einsum("acie,acnm->nmie", dX3, T.ab, optimize=True))
    dT["ab"] += np.einsum("acie,nmie->acnm", dX3, H.ab.ooov, optimize=True)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, 1 * np.einsum("acie,afim->mcfe", dX3, T.aa, optimize=True))
    dT["aa"] += np.einsum("acie,mcfe->afim", dX3, H.ab.ovvv, optimize=True)
    rdm2.bb.vovv = _plus(rdm2.bb.vovv, 1 * np.einsum("acie,afim->cmef", dX3, T.ab, optimize=True))
    dT["ab"] += np.einsum("acie,cmef->afim", dX3, H.bb.vovv, optimize=True)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, -1 * np.einsum("acie,fcim->amfe", dX3, T.ab, optimize=True))
    dT["ab"] -= np.einsum("acie,amfe->fcim", dX3, H.ab.vovv, optimize=True)
    if constant:
        rdm2.ab.vooo = _plus(rdm2.ab.vooo, -1 * dX4)
    rdm2.bb.ooov = _plus(rdm2.bb.ooov, -1 * np.einsum("amik,afin->mnkf", dX4, T.ab, optimize=True))
    dT["ab"] -= np.einsum("amik,mnkf->afin", dX4, H.bb.ooov, optimize=True)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, -1 * np.einsum("amik,afin->nmfk", dX4, T.aa, optimize=True))
    dT["aa"] -= np.einsum("amik,nmfk->afin", dX4, H.ab.oovo, optimize=True)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, 1 * np.einsum("amik,afnk->nmif", dX4, T.ab, optimize=True))
    dT["ab"] += np.einsum("amik,nmif->afnk", dX4, H.ab.ooov, optimize=True)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, -1 * np.einsum("amik,efik->amef", dX4, T.ab, optimize=True))
    dT["ab"] -= np.einsum("amik,amef->efik", dX4, H.ab.vovv, optimize=True)
    if constant:
        rdm2.aa.vvov = _plus(rdm2.aa.vvov, 0.5 * dX5)
    rdm1.a.ov = _plus(rdm1.a.ov, -0.5 * np.einsum("abie,abim->me", dX5, T.aa, optimize=True))
    dT["aa"] -= 0.5 * np.einsum("abie,me->abim", dX5, H.a.ov, optimize=True)
    rdm2.aa.ooov = _plus(rdm2.aa.ooov, 0.25 * np.einsum("abie,abmn->mnie", dX5, T.aa, optimize=True))
    dT["aa"] += 0.25 * np.einsum("abie,mnie->abmn", dX5, H.aa.ooov, optimize=True)
    rdm2.aa.vovv = _plus(rdm2.aa.vovv, 1 * np.einsum("abie,afin->bnef", dX5, T.aa, optimize=True))
    dT["aa"] += np.einsum("abie,bnef->afin", dX5, H.aa.vovv, optimize=True)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, 1 * np.einsum("abie,afin->bnef", dX5, T.ab, optimize=True))
    dT["ab"] += np.einsum("abie,bnef->afin", dX5, H.ab.vovv, optimize=True)
    if constant:
        rdm2.aa.vooo = _plus(rdm2.aa.vooo, -0.5 * dX6)
    rdm2.aa.vovv = _plus(rdm2.aa.vovv, -0.25 * np.einsum("amij,efij->amef", dX6, T.aa, optimize=True))
    dT["aa"] -= 0.25 * np.einsum("amij,amef->efij", dX6, H.aa.vovv, optimize=True)
    rdm2.aa.ooov = _plus(rdm2.aa.ooov, -1 * np.einsum("amij,afin->mnjf", dX6, T.aa, optimize=True))
    dT["aa"] -= np.einsum("amij,mnjf->afin", dX6, H.aa.ooov, optimize=True)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, -1 * np.einsum("amij,afin->mnjf", dX6, T.ab, optimize=True))
    dT["ab"] -= np.einsum("amij,mnjf->afin", dX6, H.ab.ooov, optimize=True)


def backprop_triples_intermediates_abb(dX, T, H, dT, rdm1, rdm2, constant=True):
    """Chain rule of build_triples_intermediates_abb: dX_g -> dT (raw) and the H densities."""
    dX1, dX2, dX3, dX4, dX5, dX6 = dX
    if constant:
        rdm2.ab.vvov = _plus(rdm2.ab.vvov, 0.5 * dX1)
    rdm1.b.ov = _plus(rdm1.b.ov, -0.5 * np.einsum("cbke,cbkm->me", dX1, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("cbke,me->cbkm", dX1, H.b.ov, optimize=True)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, 0.5 * np.einsum("cbke,cbmn->mnke", dX1, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("cbke,mnke->cbmn", dX1, H.ab.ooov, optimize=True)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, 0.5 * np.einsum("cbke,cfkm->mbfe", dX1, T.aa, optimize=True))
    dT["aa"] += 0.5 * np.einsum("cbke,mbfe->cfkm", dX1, H.ab.ovvv, optimize=True)
    rdm2.bb.vovv = _plus(rdm2.bb.vovv, 0.5 * np.einsum("cbke,cfkm->bmef", dX1, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("cbke,bmef->cfkm", dX1, H.bb.vovv, optimize=True)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, -0.5 * np.einsum("cbke,fbkm->cmfe", dX1, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("cbke,cmfe->fbkm", dX1, H.ab.vovv, optimize=True)
    if constant:
        rdm2.ab.vooo = _plus(rdm2.ab.vooo, -0.5 * dX2)
    rdm2.bb.ooov = _plus(rdm2.bb.ooov, -0.5 * np.einsum("cmkj,cfkn->mnjf", dX2, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("cmkj,mnjf->cfkn", dX2, H.bb.ooov, optimize=True)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, -0.5 * np.einsum("cmkj,fcnk->nmfj", dX2, T.aa, optimize=True))
    dT["aa"] -= 0.5 * np.einsum("cmkj,nmfj->fcnk", dX2, H.ab.oovo, optimize=True)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, 0.5 * np.einsum("cmkj,cfnj->nmkf", dX2, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("cmkj,nmkf->cfnj", dX2, H.ab.ooov, optimize=True)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, -0.5 * np.einsum("cmkj,efkj->cmef", dX2, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("cmkj,cmef->efkj", dX2, H.ab.vovv, optimize=True)
    if constant:
        rdm2.ab.vvvo = _plus(rdm2.ab.vvvo, 1 * dX3)
    rdm1.a.ov = _plus(rdm1.a.ov, -1 * np.einsum("caei,cami->me", dX3, T.ab, optimize=True))
    dT["ab"] -= np.einsum("caei,me->cami", dX3, H.a.ov, optimize=True)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, 1 * np.einsum("caei,camn->mnei", dX3, T.ab, optimize=True))
    dT["ab"] += np.einsum("caei,mnei->camn", dX3, H.ab.oovo, optimize=True)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, 1 * np.einsum("caei,afim->cmef", dX3, T.bb, optimize=True))
    dT["bb"] += np.einsum("caei,cmef->afim", dX3, H.ab.vovv, optimize=True)
    rdm2.aa.vovv = _plus(rdm2.aa.vovv, 1 * np.einsum("caei,fami->cmef", dX3, T.ab, optimize=True))
    dT["ab"] += np.einsum("caei,cmef->fami", dX3, H.aa.vovv, optimize=True)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, -1 * np.einsum("caei,cfmi->maef", dX3, T.ab, optimize=True))
    dT["ab"] -= np.einsum("caei,maef->cfmi", dX3, H.ab.ovvv, optimize=True)
    if constant:
        rdm2.ab.ovoo = _plus(rdm2.ab.ovoo, -1 * dX4)
    rdm2.aa.ooov = _plus(rdm2.aa.ooov, -1 * np.einsum("maki,fani->mnkf", dX4, T.ab, optimize=True))
    dT["ab"] -= np.einsum("maki,mnkf->fani", dX4, H.aa.ooov, optimize=True)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, -1 * np.einsum("maki,afin->mnkf", dX4, T.bb, optimize=True))
    dT["bb"] -= np.einsum("maki,mnkf->afin", dX4, H.ab.ooov, optimize=True)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, 1 * np.einsum("maki,fakn->mnfi", dX4, T.ab, optimize=True))
    dT["ab"] += np.einsum("maki,mnfi->fakn", dX4, H.ab.oovo, optimize=True)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, -1 * np.einsum("maki,feki->mafe", dX4, T.ab, optimize=True))
    dT["ab"] -= np.einsum("maki,mafe->feki", dX4, H.ab.ovvv, optimize=True)
    if constant:
        rdm2.bb.vvov = _plus(rdm2.bb.vvov, 0.5 * dX5)
    rdm1.b.ov = _plus(rdm1.b.ov, -0.5 * np.einsum("abie,abim->me", dX5, T.bb, optimize=True))
    dT["bb"] -= 0.5 * np.einsum("abie,me->abim", dX5, H.b.ov, optimize=True)
    rdm2.bb.ooov = _plus(rdm2.bb.ooov, 0.25 * np.einsum("abie,abmn->mnie", dX5, T.bb, optimize=True))
    dT["bb"] += 0.25 * np.einsum("abie,mnie->abmn", dX5, H.bb.ooov, optimize=True)
    rdm2.bb.vovv = _plus(rdm2.bb.vovv, 1 * np.einsum("abie,afin->bnef", dX5, T.bb, optimize=True))
    dT["bb"] += np.einsum("abie,bnef->afin", dX5, H.bb.vovv, optimize=True)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, 1 * np.einsum("abie,fani->nbfe", dX5, T.ab, optimize=True))
    dT["ab"] += np.einsum("abie,nbfe->fani", dX5, H.ab.ovvv, optimize=True)
    if constant:
        rdm2.bb.vooo = _plus(rdm2.bb.vooo, -0.5 * dX6)
    rdm2.bb.vovv = _plus(rdm2.bb.vovv, -0.25 * np.einsum("amij,efij->amef", dX6, T.bb, optimize=True))
    dT["bb"] -= 0.25 * np.einsum("amij,amef->efij", dX6, H.bb.vovv, optimize=True)
    rdm2.bb.ooov = _plus(rdm2.bb.ooov, -1 * np.einsum("amij,afin->mnjf", dX6, T.bb, optimize=True))
    dT["bb"] -= np.einsum("amij,mnjf->afin", dX6, H.bb.ooov, optimize=True)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, -1 * np.einsum("amij,fani->nmfj", dX6, T.ab, optimize=True))
    dT["ab"] -= np.einsum("amij,nmfj->fani", dX6, H.ab.oovo, optimize=True)


def backprop_triples_intermediates_bbb(dX, T, H, dT, rdm1, rdm2, constant=True):
    """Chain rule of build_triples_intermediates_bbb: dX_g -> dT (raw) and the H densities."""
    dX1, dX2 = dX
    if constant:
        rdm2.bb.vooo = _plus(rdm2.bb.vooo, -0.25 * dX1)
    rdm1.b.ov = _plus(rdm1.b.ov, -0.25 * np.einsum("amij,aeij->me", dX1, T.bb, optimize=True))
    dT["bb"] -= 0.25 * np.einsum("amij,me->aeij", dX1, H.b.ov, optimize=True)
    rdm2.bb.vovv = _plus(rdm2.bb.vovv, -0.125 * np.einsum("amij,efij->amef", dX1, T.bb, optimize=True))
    dT["bb"] -= 0.125 * np.einsum("amij,amef->efij", dX1, H.bb.vovv, optimize=True)
    rdm2.bb.ooov = _plus(rdm2.bb.ooov, -0.5 * np.einsum("amij,afin->mnjf", dX1, T.bb, optimize=True))
    dT["bb"] -= 0.5 * np.einsum("amij,mnjf->afin", dX1, H.bb.ooov, optimize=True)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, -0.5 * np.einsum("amij,fani->nmfj", dX1, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("amij,nmfj->fani", dX1, H.ab.oovo, optimize=True)
    if constant:
        rdm2.bb.vvov = _plus(rdm2.bb.vvov, 0.25 * dX2)
    rdm2.bb.ooov = _plus(rdm2.bb.ooov, 0.125 * np.einsum("abie,abmn->mnie", dX2, T.bb, optimize=True))
    dT["bb"] += 0.125 * np.einsum("abie,mnie->abmn", dX2, H.bb.ooov, optimize=True)
    rdm2.bb.vovv = _plus(rdm2.bb.vovv, 0.5 * np.einsum("abie,afin->bnef", dX2, T.bb, optimize=True))
    dT["bb"] += 0.5 * np.einsum("abie,bnef->afin", dX2, H.bb.vovv, optimize=True)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, 0.5 * np.einsum("abie,fani->nbfe", dX2, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("abie,nbfe->fani", dX2, H.ab.ovvv, optimize=True)
