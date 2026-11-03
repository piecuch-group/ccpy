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
from ccpy.gradients.density_gradient import add_densities, build_DY, get_bare_integrals, rotation_gradient
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

    # 1. sources: delta = sum_K <0|(L1 + L2) Hbar|K> <K|(H(T1) e^T2)_C|0> / D_K, with u = M/D, v = N/D
    #    s_lambda(mu) = d delta / d lambda_mu = sum_K <mu|Hbar|K> u_K
    #    s_t(mu)      = d delta / d t_mu      = sum_K v_K d<K|(H(T1) e^T2)_C|0>/dt_mu (doubles)
    #                                           + orbital rotation of the triples densities (singles)
    X = build_triples_intermediates(T, H)
    tr = calc_triples(X, T, L, H, reference)
    if delta is not None and abs(tr["delta"] - delta) > 1.0e-8:
        raise RuntimeError(f"delta from the moments ({tr['delta']:.12f}) does not match the energy ({delta:.12f}).")
    dT, rdm1, rdm2 = backprop_triples_intermediates(tr["dX"], T, H, system)
    dT = {c: dT[c] + tr["dT"][c] for c in dT}
    h, B = get_bare_integrals(reference)
    rho_triples = build_DY(*add_rdm(tr["rdm"], (rdm1, rdm2), system), None, H.chol, T, B, reference)
    singles = rotation_gradient(*rho_triples, h, B, reference)
    s_t = {"a": singles["a"], "b": singles["b"], **antisymmetrize_doubles(dT)}
    s_lambda = {"a": tr["dL"]["a"], "b": tr["dL"]["b"], **antisymmetrize_doubles(tr["dL"])}

    # 2. responses
    #    sum_nu <mu|Hbar - E0|nu> sigma_nu = -s_lambda(mu)                         (EOMCCSD HR)
    #    k_mu = <0|(1 + Lambda)[Hbar, tau_mu]|Sigma> - (lambda . sigma) <0|Hbar|mu>
    #    sum_nu pi_nu <nu|Hbar - E0|mu> = -s_t(mu) - k_mu                         (left-CCSD LH)
    sigma = solve_linear(right_action(T, Hbar, system), {c: -x for c, x in s_lambda.items()}, T, Hbar, zvector_tol)
    k = calc_k(sigma, T, L, H, Hbar, reference)
    pi = solve_linear(left_action(T, Hbar, system), {c: -(s_t[c] + k[c]) for c in s_t}, T, Hbar, zvector_tol)

    # 3. densities of the Hbar coefficients
    #    base: <0|(1 + Lambda) Hbar|0>
    #    corr: <Pi|Hbar|0> + <0|(1 + Lambda) Hbar|Sigma> - (lambda . sigma) E0
    #          + sum_K (v_K <K|Hbar|0> + u_K <0|(L1 + L2) Hbar|K>) - sum_K u_K v_K D_K,
    #    with <0|(1 + Lambda) Hbar|Sigma> - (lambda . sigma) E0 = D_sigma <0|(1 + Lambda) Hbar|0> + <C|Hbar|0>
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
    """One pass of the Fortran triples loops over the four spin cases. For each occupied triple
    and all virtual triples K (abc, ijk) the loops build

        M_K = <K|(H(T1) e^T2)_C|0>            = A sum_g X_g . t_g          (ground state)
        M_K = <K|(Hbar R)_C|0>                 = A sum_g (J_g . t_g + X_g . r_g)   (excited)
        N_K = <0|(L1 + L2) Hbar|K>            = <0|(L2 V_N)_C|K> + <0|(L1 V_N)_DC|K> + <0|(L2 F_N)_DC|K>
        D_K = f_ii + f_jj + f_kk - f_aa - f_bb - f_cc (+ omega),   u_K = M_K / D_K,  v_K = N_K / D_K

    and accumulate delta = sum_K v_K M_K, W = sum_K u_K v_K, F_r0 = sum_K v_K <K|(H(T1) e^T2)_C|0>
    and d_p = -sum_K u_K v_K chi_Kp (d D_K / d f_pp = chi_Kp), with the cotangents
    dX_g = sum_K v_K d M_K/dX_g, dT = sum_K v_K d M_K/dt, dL = sum_K u_K <mu|Hbar|K> (= s_lambda)
    and rdm = sum_K u_K d N_K/dH(T1). The N_K terms are those of
    ccpy.gradients.crcc23_gradient_reference.build_L3A-D; f is the bare Fock diagonal (reference.H).

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
    noa, nob = system.noccupied_alpha, system.noccupied_beta
    nua, nub = system.nunoccupied_alpha, system.nunoccupied_beta
    # bare Fock diagonal of the denominators, as in the energy (ccpy.moments.crcc23_chol)
    eoa, eva = np.diag(reference.H.a.oo).copy(), np.diag(reference.H.a.vv).copy()
    eob, evb = np.diag(reference.H.b.oo).copy(), np.diag(reference.H.b.vv).copy()
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

    # ---- aaa and bbb: M = <ijkabc|...|0> = A(abc)A(ijk)[X1(amij) t(bc,mk) + X2(abie) t(ec,jk)],
    #      N = <0|(L1 + L2) Hbar|ijkabc> (and the same with all indices beta);
    #      x1(a,m,i,j) . t2(b,c,m,k), x2(a,b,i,e) . t2(e,c,j,k) (passed as (a,b,e,i))
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

    # ---- aab: M = <ijk~abc~|...|0> = A(ab)A(ij) sum_g X_g . t_g, N = <0|(L1 + L2) Hbar|ijk~abc~>;
    #      groups (layouts as passed to grad_aab)
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

    # ---- abb: M = <ij~k~ab~c~|...|0> = A(bc)A(jk) sum_g X_g . t_g, N = <0|(L1 + L2) Hbar|ij~k~ab~c~>;
    #      groups (layouts as passed to grad_abb)
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
    rdm1.a.oo, rdm1.a.vv = np.diag(d["a"]["o"]), np.diag(d["a"]["v"])
    rdm1.b.oo, rdm1.b.vv = np.diag(d["b"]["o"]), np.diag(d["b"]["v"])
    _, B = get_bare_integrals(reference)
    return build_DY(rdm1, rdm2, None, H0.chol, None, B, reference)


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
    """Intermediates X_g of the aaa moments,

        M = <ijkabc|(H(T1) e^T2)_C|0> = A(abc) A(ijk) [X1(amij) t(bc,mk)
            + X2(abie) t(ec,jk)]

    (summed over the repeated index of each product): the T2-dressed blocks of H(T1) that each
    multiply one T2. Each term below is labeled by its diagram, (V_N T2)_C for the bare
    H(T1) block, (F_N T2^2)_C/2 and (V_N T2^2)_C/2 for the terms quadratic in T2; constant=False
    drops the bare H(T1) blocks, leaving the part linear in T2.
    Layout: amij . T.aa(bcmk), abie . T.aa(ecjk)."""
    # X1(amij), times t(bc,mk) in <ijkabc|(H_N e^T2)_C|0>
    #     (F_N T2^2)_C/2: f(me) t(ae,ij) . t(bc,mk)
    X1 = -0.25 * np.einsum("me,aeij->amij", H.a.ov, T.aa, optimize=True)
    #     (V_N T2^2)_C/2: <am||ef> t(ef,ij) . t(bc,mk)
    X1 -= 0.125 * np.einsum("amef,efij->amij", H.aa.vovv, T.aa, optimize=True)
    #     (V_N T2^2)_C/2: <mn||jf> t(af,in) . t(bc,mk)
    X1 -= 0.5 * np.einsum("mnjf,afin->amij", H.aa.ooov, T.aa, optimize=True)
    #     (V_N T2^2)_C/2: <mn~|jf~> t(af~,in~) . t(bc,mk)
    X1 -= 0.5 * np.einsum("mnjf,afin->amij", H.ab.ooov, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <am||ij> . t(bc,mk)
        X1 -= 0.25 * H.aa.vooo
    # X2(abie), times t(ec,jk) in <ijkabc|(H_N e^T2)_C|0>
    #     (V_N T2^2)_C/2: <mn||ie> t(ab,mn) . t(ec,jk)
    X2 = 0.125 * np.einsum("mnie,abmn->abie", H.aa.ooov, T.aa, optimize=True)
    #     (V_N T2^2)_C/2: <bn||ef> t(af,in) . t(ec,jk)
    X2 += 0.5 * np.einsum("bnef,afin->abie", H.aa.vovv, T.aa, optimize=True)
    #     (V_N T2^2)_C/2: <bn~|ef~> t(af~,in~) . t(ec,jk)
    X2 += 0.5 * np.einsum("bnef,afin->abie", H.ab.vovv, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <ab||ie> . t(ec,jk)
        X2 += 0.25 * H.aa.vvov
    return [X1, X2]


def build_triples_intermediates_aab(T, H, constant=True):
    """Intermediates X_g of the aab moments,

        M = <ijk~abc~|(H(T1) e^T2)_C|0> = A(ab) A(ij) [X1(bcek) t(ae,ij)
            + X2(mcjk) t(ab,im)
            + X3(acie) t(be~,jk~)
            + X4(amik) t(bc~,jm~)
            + X5(abie) t(ec~,jk~)
            + X6(amij) t(bc~,mk~)]

    (summed over the repeated index of each product): the T2-dressed blocks of H(T1) that each
    multiply one T2. Each term below is labeled by its diagram, (V_N T2)_C for the bare
    H(T1) block, (F_N T2^2)_C/2 and (V_N T2^2)_C/2 for the terms quadratic in T2; constant=False
    drops the bare H(T1) blocks, leaving the part linear in T2.
    Layout: bcek . T.aa(aeij), mcjk . T.aa(abim), acie . T.ab(bejk), amik . T.ab(bcjm), abie . T.ab(ecjk), amij . T.ab(bcmk)."""
    # X1(bcek), times t(ae,ij) in <ijk~abc~|(H_N e^T2)_C|0>
    #     (F_N T2^2)_C/2: f(me) t(bc~,mk~) . t(ae,ij)
    X1 = -0.5 * np.einsum("me,bcmk->bcek", H.a.ov, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <nm~|ek~> t(bc~,nm~) . t(ae,ij)
    X1 += 0.5 * np.einsum("nmek,bcnm->bcek", H.ab.oovo, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <bm~|ef~> t(c~f~,k~m~) . t(ae,ij)
    X1 += 0.5 * np.einsum("bmef,cfkm->bcek", H.ab.vovv, T.bb, optimize=True)
    #     (V_N T2^2)_C/2: <bm||ef> t(fc~,mk~) . t(ae,ij)
    X1 += 0.5 * np.einsum("bmef,fcmk->bcek", H.aa.vovv, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <mc~|ef~> t(bf~,mk~) . t(ae,ij)
    X1 -= 0.5 * np.einsum("mcef,bfmk->bcek", H.ab.ovvv, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <bc~|ek~> . t(ae,ij)
        X1 += 0.5 * H.ab.vvvo
    # X2(mcjk), times t(ab,im) in <ijk~abc~|(H_N e^T2)_C|0>
    #     (V_N T2^2)_C/2: <mn||jf> t(fc~,nk~) . t(ab,im)
    X2 = -0.5 * np.einsum("mnjf,fcnk->mcjk", H.aa.ooov, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <mn~|jf~> t(f~c~,n~k~) . t(ab,im)
    X2 -= 0.5 * np.einsum("mnjf,fcnk->mcjk", H.ab.ooov, T.bb, optimize=True)
    #     (V_N T2^2)_C/2: <mn~|fk~> t(fc~,jn~) . t(ab,im)
    X2 += 0.5 * np.einsum("mnfk,fcjn->mcjk", H.ab.oovo, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <mc~|fe~> t(fe~,jk~) . t(ab,im)
    X2 -= 0.5 * np.einsum("mcfe,fejk->mcjk", H.ab.ovvv, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <mc~|jk~> . t(ab,im)
        X2 -= 0.5 * H.ab.ovoo
    # X3(acie), times t(be~,jk~) in <ijk~abc~|(H_N e^T2)_C|0>
    #     (F_N T2^2)_C/2: f(m~e~) t(ac~,im~) . t(be~,jk~)
    X3 = -np.einsum("me,acim->acie", H.b.ov, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <nm~|ie~> t(ac~,nm~) . t(be~,jk~)
    X3 += np.einsum("nmie,acnm->acie", H.ab.ooov, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <mc~|fe~> t(af,im) . t(be~,jk~)
    X3 += np.einsum("mcfe,afim->acie", H.ab.ovvv, T.aa, optimize=True)
    #     (V_N T2^2)_C/2: <c~m~||e~f~> t(af~,im~) . t(be~,jk~)
    X3 += np.einsum("cmef,afim->acie", H.bb.vovv, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <am~|fe~> t(fc~,im~) . t(be~,jk~)
    X3 -= np.einsum("amfe,fcim->acie", H.ab.vovv, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <ac~|ie~> . t(be~,jk~)
        X3 += H.ab.vvov
    # X4(amik), times t(bc~,jm~) in <ijk~abc~|(H_N e^T2)_C|0>
    #     (V_N T2^2)_C/2: <m~n~||k~f~> t(af~,in~) . t(bc~,jm~)
    X4 = -np.einsum("mnkf,afin->amik", H.bb.ooov, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <nm~|fk~> t(af,in) . t(bc~,jm~)
    X4 -= np.einsum("nmfk,afin->amik", H.ab.oovo, T.aa, optimize=True)
    #     (V_N T2^2)_C/2: <nm~|if~> t(af~,nk~) . t(bc~,jm~)
    X4 += np.einsum("nmif,afnk->amik", H.ab.ooov, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <am~|ef~> t(ef~,ik~) . t(bc~,jm~)
    X4 -= np.einsum("amef,efik->amik", H.ab.vovv, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <am~|ik~> . t(bc~,jm~)
        X4 -= H.ab.vooo
    # X5(abie), times t(ec~,jk~) in <ijk~abc~|(H_N e^T2)_C|0>
    #     (F_N T2^2)_C/2: f(me) t(ab,im) . t(ec~,jk~)
    X5 = -0.5 * np.einsum("me,abim->abie", H.a.ov, T.aa, optimize=True)
    #     (V_N T2^2)_C/2: <mn||ie> t(ab,mn) . t(ec~,jk~)
    X5 += 0.25 * np.einsum("mnie,abmn->abie", H.aa.ooov, T.aa, optimize=True)
    #     (V_N T2^2)_C/2: <bn||ef> t(af,in) . t(ec~,jk~)
    X5 += np.einsum("bnef,afin->abie", H.aa.vovv, T.aa, optimize=True)
    #     (V_N T2^2)_C/2: <bn~|ef~> t(af~,in~) . t(ec~,jk~)
    X5 += np.einsum("bnef,afin->abie", H.ab.vovv, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <ab||ie> . t(ec~,jk~)
        X5 += 0.5 * H.aa.vvov
    # X6(amij), times t(bc~,mk~) in <ijk~abc~|(H_N e^T2)_C|0>
    #     (V_N T2^2)_C/2: <am||ef> t(ef,ij) . t(bc~,mk~)
    X6 = -0.25 * np.einsum("amef,efij->amij", H.aa.vovv, T.aa, optimize=True)
    #     (V_N T2^2)_C/2: <mn||jf> t(af,in) . t(bc~,mk~)
    X6 -= np.einsum("mnjf,afin->amij", H.aa.ooov, T.aa, optimize=True)
    #     (V_N T2^2)_C/2: <mn~|jf~> t(af~,in~) . t(bc~,mk~)
    X6 -= np.einsum("mnjf,afin->amij", H.ab.ooov, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <am||ij> . t(bc~,mk~)
        X6 -= 0.5 * H.aa.vooo
    return [X1, X2, X3, X4, X5, X6]


def build_triples_intermediates_abb(T, H, constant=True):
    """Intermediates X_g of the abb moments,

        M = <ij~k~ab~c~|(H(T1) e^T2)_C|0> = A(bc) A(jk) [X1(cbke) t(a~e~,i~j~)
            + X2(cmkj) t(a~b~,i~m~)
            + X3(caei) t(eb~,kj~)
            + X4(maki) t(cb~,mj~)
            + X5(abie) t(ce~,kj~)
            + X6(amij) t(cb~,km~)]

    (summed over the repeated index of each product): the T2-dressed blocks of H(T1) that each
    multiply one T2. Each term below is labeled by its diagram, (V_N T2)_C for the bare
    H(T1) block, (F_N T2^2)_C/2 and (V_N T2^2)_C/2 for the terms quadratic in T2; constant=False
    drops the bare H(T1) blocks, leaving the part linear in T2.
    Layout: cbke . T.bb(aeij), cmkj . T.bb(abim), caei . T.ab(ebkj), maki . T.ab(cbmj), abie . T.ab(cekj), amij . T.ab(cbkm)."""
    # X1(cbke), times t(a~e~,i~j~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    #     (F_N T2^2)_C/2: f(m~e~) t(cb~,km~) . t(a~e~,i~j~)
    X1 = -0.5 * np.einsum("me,cbkm->cbke", H.b.ov, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <mn~|ke~> t(cb~,mn~) . t(a~e~,i~j~)
    X1 += 0.5 * np.einsum("mnke,cbmn->cbke", H.ab.ooov, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <mb~|fe~> t(cf,km) . t(a~e~,i~j~)
    X1 += 0.5 * np.einsum("mbfe,cfkm->cbke", H.ab.ovvv, T.aa, optimize=True)
    #     (V_N T2^2)_C/2: <b~m~||e~f~> t(cf~,km~) . t(a~e~,i~j~)
    X1 += 0.5 * np.einsum("bmef,cfkm->cbke", H.bb.vovv, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <cm~|fe~> t(fb~,km~) . t(a~e~,i~j~)
    X1 -= 0.5 * np.einsum("cmfe,fbkm->cbke", H.ab.vovv, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <cb~|ke~> . t(a~e~,i~j~)
        X1 += 0.5 * H.ab.vvov
    # X2(cmkj), times t(a~b~,i~m~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    #     (V_N T2^2)_C/2: <m~n~||j~f~> t(cf~,kn~) . t(a~b~,i~m~)
    X2 = -0.5 * np.einsum("mnjf,cfkn->cmkj", H.bb.ooov, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <nm~|fj~> t(fc,nk) . t(a~b~,i~m~)
    X2 -= 0.5 * np.einsum("nmfj,fcnk->cmkj", H.ab.oovo, T.aa, optimize=True)
    #     (V_N T2^2)_C/2: <nm~|kf~> t(cf~,nj~) . t(a~b~,i~m~)
    X2 += 0.5 * np.einsum("nmkf,cfnj->cmkj", H.ab.ooov, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <cm~|ef~> t(ef~,kj~) . t(a~b~,i~m~)
    X2 -= 0.5 * np.einsum("cmef,efkj->cmkj", H.ab.vovv, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <cm~|kj~> . t(a~b~,i~m~)
        X2 -= 0.5 * H.ab.vooo
    # X3(caei), times t(eb~,kj~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    #     (F_N T2^2)_C/2: f(me) t(ca~,mi~) . t(eb~,kj~)
    X3 = -np.einsum("me,cami->caei", H.a.ov, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <mn~|ei~> t(ca~,mn~) . t(eb~,kj~)
    X3 += np.einsum("mnei,camn->caei", H.ab.oovo, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <cm~|ef~> t(a~f~,i~m~) . t(eb~,kj~)
    X3 += np.einsum("cmef,afim->caei", H.ab.vovv, T.bb, optimize=True)
    #     (V_N T2^2)_C/2: <cm||ef> t(fa~,mi~) . t(eb~,kj~)
    X3 += np.einsum("cmef,fami->caei", H.aa.vovv, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <ma~|ef~> t(cf~,mi~) . t(eb~,kj~)
    X3 -= np.einsum("maef,cfmi->caei", H.ab.ovvv, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <ca~|ei~> . t(eb~,kj~)
        X3 += H.ab.vvvo
    # X4(maki), times t(cb~,mj~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    #     (V_N T2^2)_C/2: <mn||kf> t(fa~,ni~) . t(cb~,mj~)
    X4 = -np.einsum("mnkf,fani->maki", H.aa.ooov, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <mn~|kf~> t(a~f~,i~n~) . t(cb~,mj~)
    X4 -= np.einsum("mnkf,afin->maki", H.ab.ooov, T.bb, optimize=True)
    #     (V_N T2^2)_C/2: <mn~|fi~> t(fa~,kn~) . t(cb~,mj~)
    X4 += np.einsum("mnfi,fakn->maki", H.ab.oovo, T.ab, optimize=True)
    #     (V_N T2^2)_C/2: <ma~|fe~> t(fe~,ki~) . t(cb~,mj~)
    X4 -= np.einsum("mafe,feki->maki", H.ab.ovvv, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <ma~|ki~> . t(cb~,mj~)
        X4 -= H.ab.ovoo
    # X5(abie), times t(ce~,kj~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    #     (F_N T2^2)_C/2: f(m~e~) t(a~b~,i~m~) . t(ce~,kj~)
    X5 = -0.5 * np.einsum("me,abim->abie", H.b.ov, T.bb, optimize=True)
    #     (V_N T2^2)_C/2: <m~n~||i~e~> t(a~b~,m~n~) . t(ce~,kj~)
    X5 += 0.25 * np.einsum("mnie,abmn->abie", H.bb.ooov, T.bb, optimize=True)
    #     (V_N T2^2)_C/2: <b~n~||e~f~> t(a~f~,i~n~) . t(ce~,kj~)
    X5 += np.einsum("bnef,afin->abie", H.bb.vovv, T.bb, optimize=True)
    #     (V_N T2^2)_C/2: <nb~|fe~> t(fa~,ni~) . t(ce~,kj~)
    X5 += np.einsum("nbfe,fani->abie", H.ab.ovvv, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <a~b~||i~e~> . t(ce~,kj~)
        X5 += 0.5 * H.bb.vvov
    # X6(amij), times t(cb~,km~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    #     (V_N T2^2)_C/2: <a~m~||e~f~> t(e~f~,i~j~) . t(cb~,km~)
    X6 = -0.25 * np.einsum("amef,efij->amij", H.bb.vovv, T.bb, optimize=True)
    #     (V_N T2^2)_C/2: <m~n~||j~f~> t(a~f~,i~n~) . t(cb~,km~)
    X6 -= np.einsum("mnjf,afin->amij", H.bb.ooov, T.bb, optimize=True)
    #     (V_N T2^2)_C/2: <nm~|fj~> t(fa~,ni~) . t(cb~,km~)
    X6 -= np.einsum("nmfj,fani->amij", H.ab.oovo, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <a~m~||i~j~> . t(cb~,km~)
        X6 -= 0.5 * H.bb.vooo
    return [X1, X2, X3, X4, X5, X6]


def build_triples_intermediates_bbb(T, H, constant=True):
    """Intermediates X_g of the bbb moments,

        M = <i~j~k~a~b~c~|(H(T1) e^T2)_C|0> = A(abc) A(ijk) [X1(amij) t(b~c~,m~k~)
            + X2(abie) t(e~c~,j~k~)]

    (summed over the repeated index of each product): the T2-dressed blocks of H(T1) that each
    multiply one T2. Each term below is labeled by its diagram, (V_N T2)_C for the bare
    H(T1) block, (F_N T2^2)_C/2 and (V_N T2^2)_C/2 for the terms quadratic in T2; constant=False
    drops the bare H(T1) blocks, leaving the part linear in T2.
    Layout: amij . T.bb(bcmk), abie . T.bb(ecjk)."""
    # X1(amij), times t(b~c~,m~k~) in <i~j~k~a~b~c~|(H_N e^T2)_C|0>
    #     (F_N T2^2)_C/2: f(m~e~) t(a~e~,i~j~) . t(b~c~,m~k~)
    X1 = -0.25 * np.einsum("me,aeij->amij", H.b.ov, T.bb, optimize=True)
    #     (V_N T2^2)_C/2: <a~m~||e~f~> t(e~f~,i~j~) . t(b~c~,m~k~)
    X1 -= 0.125 * np.einsum("amef,efij->amij", H.bb.vovv, T.bb, optimize=True)
    #     (V_N T2^2)_C/2: <m~n~||j~f~> t(a~f~,i~n~) . t(b~c~,m~k~)
    X1 -= 0.5 * np.einsum("mnjf,afin->amij", H.bb.ooov, T.bb, optimize=True)
    #     (V_N T2^2)_C/2: <nm~|fj~> t(fa~,ni~) . t(b~c~,m~k~)
    X1 -= 0.5 * np.einsum("nmfj,fani->amij", H.ab.oovo, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <a~m~||i~j~> . t(b~c~,m~k~)
        X1 -= 0.25 * H.bb.vooo
    # X2(abie), times t(e~c~,j~k~) in <i~j~k~a~b~c~|(H_N e^T2)_C|0>
    #     (V_N T2^2)_C/2: <m~n~||i~e~> t(a~b~,m~n~) . t(e~c~,j~k~)
    X2 = 0.125 * np.einsum("mnie,abmn->abie", H.bb.ooov, T.bb, optimize=True)
    #     (V_N T2^2)_C/2: <b~n~||e~f~> t(a~f~,i~n~) . t(e~c~,j~k~)
    X2 += 0.5 * np.einsum("bnef,afin->abie", H.bb.vovv, T.bb, optimize=True)
    #     (V_N T2^2)_C/2: <nb~|fe~> t(fa~,ni~) . t(e~c~,j~k~)
    X2 += 0.5 * np.einsum("nbfe,fani->abie", H.ab.ovvv, T.ab, optimize=True)
    if constant:
        #     (V_N T2)_C: <a~b~||i~e~> . t(e~c~,j~k~)
        X2 += 0.25 * H.bb.vvov
    return [X1, X2]


def backprop_triples_intermediates_aaa(dX, T, H, dT, rdm1, rdm2, constant=True):
    """Chain rule of build_triples_intermediates_aaa for the cotangents dX_g of the X_g
    (from the <ijkabc|...|0> moments): for every term c H(T1) T2 of X_g, the raw T2 gradient
    dT += c dX_g . H and the H(T1) density rdm += c dX_g . T2 (rdm += c dX_g for the bare blocks)."""
    dX1, dX2 = dX
    # X1(amij), times t(bc,mk) in <ijkabc|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <am||ij>
        rdm2.aa.vooo = _plus(rdm2.aa.vooo, -0.25 * dX1)
    #     d/dH, d/dt of f(me) t(ae,ij)
    rdm1.a.ov = _plus(rdm1.a.ov, -0.25 * np.einsum("amij,aeij->me", dX1, T.aa, optimize=True))
    dT["aa"] -= 0.25 * np.einsum("amij,me->aeij", dX1, H.a.ov, optimize=True)
    #     d/dH, d/dt of <am||ef> t(ef,ij)
    rdm2.aa.vovv = _plus(rdm2.aa.vovv, -0.125 * np.einsum("amij,efij->amef", dX1, T.aa, optimize=True))
    dT["aa"] -= 0.125 * np.einsum("amij,amef->efij", dX1, H.aa.vovv, optimize=True)
    #     d/dH, d/dt of <mn||jf> t(af,in)
    rdm2.aa.ooov = _plus(rdm2.aa.ooov, -0.5 * np.einsum("amij,afin->mnjf", dX1, T.aa, optimize=True))
    dT["aa"] -= 0.5 * np.einsum("amij,mnjf->afin", dX1, H.aa.ooov, optimize=True)
    #     d/dH, d/dt of <mn~|jf~> t(af~,in~)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, -0.5 * np.einsum("amij,afin->mnjf", dX1, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("amij,mnjf->afin", dX1, H.ab.ooov, optimize=True)
    # X2(abie), times t(ec,jk) in <ijkabc|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <ab||ie>
        rdm2.aa.vvov = _plus(rdm2.aa.vvov, 0.25 * dX2)
    #     d/dH, d/dt of <mn||ie> t(ab,mn)
    rdm2.aa.ooov = _plus(rdm2.aa.ooov, 0.125 * np.einsum("abie,abmn->mnie", dX2, T.aa, optimize=True))
    dT["aa"] += 0.125 * np.einsum("abie,mnie->abmn", dX2, H.aa.ooov, optimize=True)
    #     d/dH, d/dt of <bn||ef> t(af,in)
    rdm2.aa.vovv = _plus(rdm2.aa.vovv, 0.5 * np.einsum("abie,afin->bnef", dX2, T.aa, optimize=True))
    dT["aa"] += 0.5 * np.einsum("abie,bnef->afin", dX2, H.aa.vovv, optimize=True)
    #     d/dH, d/dt of <bn~|ef~> t(af~,in~)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, 0.5 * np.einsum("abie,afin->bnef", dX2, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("abie,bnef->afin", dX2, H.ab.vovv, optimize=True)


def backprop_triples_intermediates_aab(dX, T, H, dT, rdm1, rdm2, constant=True):
    """Chain rule of build_triples_intermediates_aab for the cotangents dX_g of the X_g
    (from the <ijk~abc~|...|0> moments): for every term c H(T1) T2 of X_g, the raw T2 gradient
    dT += c dX_g . H and the H(T1) density rdm += c dX_g . T2 (rdm += c dX_g for the bare blocks)."""
    dX1, dX2, dX3, dX4, dX5, dX6 = dX
    # X1(bcek), times t(ae,ij) in <ijk~abc~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <bc~|ek~>
        rdm2.ab.vvvo = _plus(rdm2.ab.vvvo, 0.5 * dX1)
    #     d/dH, d/dt of f(me) t(bc~,mk~)
    rdm1.a.ov = _plus(rdm1.a.ov, -0.5 * np.einsum("bcek,bcmk->me", dX1, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("bcek,me->bcmk", dX1, H.a.ov, optimize=True)
    #     d/dH, d/dt of <nm~|ek~> t(bc~,nm~)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, 0.5 * np.einsum("bcek,bcnm->nmek", dX1, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("bcek,nmek->bcnm", dX1, H.ab.oovo, optimize=True)
    #     d/dH, d/dt of <bm~|ef~> t(c~f~,k~m~)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, 0.5 * np.einsum("bcek,cfkm->bmef", dX1, T.bb, optimize=True))
    dT["bb"] += 0.5 * np.einsum("bcek,bmef->cfkm", dX1, H.ab.vovv, optimize=True)
    #     d/dH, d/dt of <bm||ef> t(fc~,mk~)
    rdm2.aa.vovv = _plus(rdm2.aa.vovv, 0.5 * np.einsum("bcek,fcmk->bmef", dX1, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("bcek,bmef->fcmk", dX1, H.aa.vovv, optimize=True)
    #     d/dH, d/dt of <mc~|ef~> t(bf~,mk~)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, -0.5 * np.einsum("bcek,bfmk->mcef", dX1, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("bcek,mcef->bfmk", dX1, H.ab.ovvv, optimize=True)
    # X2(mcjk), times t(ab,im) in <ijk~abc~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <mc~|jk~>
        rdm2.ab.ovoo = _plus(rdm2.ab.ovoo, -0.5 * dX2)
    #     d/dH, d/dt of <mn||jf> t(fc~,nk~)
    rdm2.aa.ooov = _plus(rdm2.aa.ooov, -0.5 * np.einsum("mcjk,fcnk->mnjf", dX2, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("mcjk,mnjf->fcnk", dX2, H.aa.ooov, optimize=True)
    #     d/dH, d/dt of <mn~|jf~> t(f~c~,n~k~)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, -0.5 * np.einsum("mcjk,fcnk->mnjf", dX2, T.bb, optimize=True))
    dT["bb"] -= 0.5 * np.einsum("mcjk,mnjf->fcnk", dX2, H.ab.ooov, optimize=True)
    #     d/dH, d/dt of <mn~|fk~> t(fc~,jn~)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, 0.5 * np.einsum("mcjk,fcjn->mnfk", dX2, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("mcjk,mnfk->fcjn", dX2, H.ab.oovo, optimize=True)
    #     d/dH, d/dt of <mc~|fe~> t(fe~,jk~)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, -0.5 * np.einsum("mcjk,fejk->mcfe", dX2, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("mcjk,mcfe->fejk", dX2, H.ab.ovvv, optimize=True)
    # X3(acie), times t(be~,jk~) in <ijk~abc~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <ac~|ie~>
        rdm2.ab.vvov = _plus(rdm2.ab.vvov, 1 * dX3)
    #     d/dH, d/dt of f(m~e~) t(ac~,im~)
    rdm1.b.ov = _plus(rdm1.b.ov, -1 * np.einsum("acie,acim->me", dX3, T.ab, optimize=True))
    dT["ab"] -= np.einsum("acie,me->acim", dX3, H.b.ov, optimize=True)
    #     d/dH, d/dt of <nm~|ie~> t(ac~,nm~)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, 1 * np.einsum("acie,acnm->nmie", dX3, T.ab, optimize=True))
    dT["ab"] += np.einsum("acie,nmie->acnm", dX3, H.ab.ooov, optimize=True)
    #     d/dH, d/dt of <mc~|fe~> t(af,im)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, 1 * np.einsum("acie,afim->mcfe", dX3, T.aa, optimize=True))
    dT["aa"] += np.einsum("acie,mcfe->afim", dX3, H.ab.ovvv, optimize=True)
    #     d/dH, d/dt of <c~m~||e~f~> t(af~,im~)
    rdm2.bb.vovv = _plus(rdm2.bb.vovv, 1 * np.einsum("acie,afim->cmef", dX3, T.ab, optimize=True))
    dT["ab"] += np.einsum("acie,cmef->afim", dX3, H.bb.vovv, optimize=True)
    #     d/dH, d/dt of <am~|fe~> t(fc~,im~)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, -1 * np.einsum("acie,fcim->amfe", dX3, T.ab, optimize=True))
    dT["ab"] -= np.einsum("acie,amfe->fcim", dX3, H.ab.vovv, optimize=True)
    # X4(amik), times t(bc~,jm~) in <ijk~abc~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <am~|ik~>
        rdm2.ab.vooo = _plus(rdm2.ab.vooo, -1 * dX4)
    #     d/dH, d/dt of <m~n~||k~f~> t(af~,in~)
    rdm2.bb.ooov = _plus(rdm2.bb.ooov, -1 * np.einsum("amik,afin->mnkf", dX4, T.ab, optimize=True))
    dT["ab"] -= np.einsum("amik,mnkf->afin", dX4, H.bb.ooov, optimize=True)
    #     d/dH, d/dt of <nm~|fk~> t(af,in)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, -1 * np.einsum("amik,afin->nmfk", dX4, T.aa, optimize=True))
    dT["aa"] -= np.einsum("amik,nmfk->afin", dX4, H.ab.oovo, optimize=True)
    #     d/dH, d/dt of <nm~|if~> t(af~,nk~)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, 1 * np.einsum("amik,afnk->nmif", dX4, T.ab, optimize=True))
    dT["ab"] += np.einsum("amik,nmif->afnk", dX4, H.ab.ooov, optimize=True)
    #     d/dH, d/dt of <am~|ef~> t(ef~,ik~)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, -1 * np.einsum("amik,efik->amef", dX4, T.ab, optimize=True))
    dT["ab"] -= np.einsum("amik,amef->efik", dX4, H.ab.vovv, optimize=True)
    # X5(abie), times t(ec~,jk~) in <ijk~abc~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <ab||ie>
        rdm2.aa.vvov = _plus(rdm2.aa.vvov, 0.5 * dX5)
    #     d/dH, d/dt of f(me) t(ab,im)
    rdm1.a.ov = _plus(rdm1.a.ov, -0.5 * np.einsum("abie,abim->me", dX5, T.aa, optimize=True))
    dT["aa"] -= 0.5 * np.einsum("abie,me->abim", dX5, H.a.ov, optimize=True)
    #     d/dH, d/dt of <mn||ie> t(ab,mn)
    rdm2.aa.ooov = _plus(rdm2.aa.ooov, 0.25 * np.einsum("abie,abmn->mnie", dX5, T.aa, optimize=True))
    dT["aa"] += 0.25 * np.einsum("abie,mnie->abmn", dX5, H.aa.ooov, optimize=True)
    #     d/dH, d/dt of <bn||ef> t(af,in)
    rdm2.aa.vovv = _plus(rdm2.aa.vovv, 1 * np.einsum("abie,afin->bnef", dX5, T.aa, optimize=True))
    dT["aa"] += np.einsum("abie,bnef->afin", dX5, H.aa.vovv, optimize=True)
    #     d/dH, d/dt of <bn~|ef~> t(af~,in~)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, 1 * np.einsum("abie,afin->bnef", dX5, T.ab, optimize=True))
    dT["ab"] += np.einsum("abie,bnef->afin", dX5, H.ab.vovv, optimize=True)
    # X6(amij), times t(bc~,mk~) in <ijk~abc~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <am||ij>
        rdm2.aa.vooo = _plus(rdm2.aa.vooo, -0.5 * dX6)
    #     d/dH, d/dt of <am||ef> t(ef,ij)
    rdm2.aa.vovv = _plus(rdm2.aa.vovv, -0.25 * np.einsum("amij,efij->amef", dX6, T.aa, optimize=True))
    dT["aa"] -= 0.25 * np.einsum("amij,amef->efij", dX6, H.aa.vovv, optimize=True)
    #     d/dH, d/dt of <mn||jf> t(af,in)
    rdm2.aa.ooov = _plus(rdm2.aa.ooov, -1 * np.einsum("amij,afin->mnjf", dX6, T.aa, optimize=True))
    dT["aa"] -= np.einsum("amij,mnjf->afin", dX6, H.aa.ooov, optimize=True)
    #     d/dH, d/dt of <mn~|jf~> t(af~,in~)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, -1 * np.einsum("amij,afin->mnjf", dX6, T.ab, optimize=True))
    dT["ab"] -= np.einsum("amij,mnjf->afin", dX6, H.ab.ooov, optimize=True)


def backprop_triples_intermediates_abb(dX, T, H, dT, rdm1, rdm2, constant=True):
    """Chain rule of build_triples_intermediates_abb for the cotangents dX_g of the X_g
    (from the <ij~k~ab~c~|...|0> moments): for every term c H(T1) T2 of X_g, the raw T2 gradient
    dT += c dX_g . H and the H(T1) density rdm += c dX_g . T2 (rdm += c dX_g for the bare blocks)."""
    dX1, dX2, dX3, dX4, dX5, dX6 = dX
    # X1(cbke), times t(a~e~,i~j~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <cb~|ke~>
        rdm2.ab.vvov = _plus(rdm2.ab.vvov, 0.5 * dX1)
    #     d/dH, d/dt of f(m~e~) t(cb~,km~)
    rdm1.b.ov = _plus(rdm1.b.ov, -0.5 * np.einsum("cbke,cbkm->me", dX1, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("cbke,me->cbkm", dX1, H.b.ov, optimize=True)
    #     d/dH, d/dt of <mn~|ke~> t(cb~,mn~)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, 0.5 * np.einsum("cbke,cbmn->mnke", dX1, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("cbke,mnke->cbmn", dX1, H.ab.ooov, optimize=True)
    #     d/dH, d/dt of <mb~|fe~> t(cf,km)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, 0.5 * np.einsum("cbke,cfkm->mbfe", dX1, T.aa, optimize=True))
    dT["aa"] += 0.5 * np.einsum("cbke,mbfe->cfkm", dX1, H.ab.ovvv, optimize=True)
    #     d/dH, d/dt of <b~m~||e~f~> t(cf~,km~)
    rdm2.bb.vovv = _plus(rdm2.bb.vovv, 0.5 * np.einsum("cbke,cfkm->bmef", dX1, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("cbke,bmef->cfkm", dX1, H.bb.vovv, optimize=True)
    #     d/dH, d/dt of <cm~|fe~> t(fb~,km~)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, -0.5 * np.einsum("cbke,fbkm->cmfe", dX1, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("cbke,cmfe->fbkm", dX1, H.ab.vovv, optimize=True)
    # X2(cmkj), times t(a~b~,i~m~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <cm~|kj~>
        rdm2.ab.vooo = _plus(rdm2.ab.vooo, -0.5 * dX2)
    #     d/dH, d/dt of <m~n~||j~f~> t(cf~,kn~)
    rdm2.bb.ooov = _plus(rdm2.bb.ooov, -0.5 * np.einsum("cmkj,cfkn->mnjf", dX2, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("cmkj,mnjf->cfkn", dX2, H.bb.ooov, optimize=True)
    #     d/dH, d/dt of <nm~|fj~> t(fc,nk)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, -0.5 * np.einsum("cmkj,fcnk->nmfj", dX2, T.aa, optimize=True))
    dT["aa"] -= 0.5 * np.einsum("cmkj,nmfj->fcnk", dX2, H.ab.oovo, optimize=True)
    #     d/dH, d/dt of <nm~|kf~> t(cf~,nj~)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, 0.5 * np.einsum("cmkj,cfnj->nmkf", dX2, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("cmkj,nmkf->cfnj", dX2, H.ab.ooov, optimize=True)
    #     d/dH, d/dt of <cm~|ef~> t(ef~,kj~)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, -0.5 * np.einsum("cmkj,efkj->cmef", dX2, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("cmkj,cmef->efkj", dX2, H.ab.vovv, optimize=True)
    # X3(caei), times t(eb~,kj~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <ca~|ei~>
        rdm2.ab.vvvo = _plus(rdm2.ab.vvvo, 1 * dX3)
    #     d/dH, d/dt of f(me) t(ca~,mi~)
    rdm1.a.ov = _plus(rdm1.a.ov, -1 * np.einsum("caei,cami->me", dX3, T.ab, optimize=True))
    dT["ab"] -= np.einsum("caei,me->cami", dX3, H.a.ov, optimize=True)
    #     d/dH, d/dt of <mn~|ei~> t(ca~,mn~)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, 1 * np.einsum("caei,camn->mnei", dX3, T.ab, optimize=True))
    dT["ab"] += np.einsum("caei,mnei->camn", dX3, H.ab.oovo, optimize=True)
    #     d/dH, d/dt of <cm~|ef~> t(a~f~,i~m~)
    rdm2.ab.vovv = _plus(rdm2.ab.vovv, 1 * np.einsum("caei,afim->cmef", dX3, T.bb, optimize=True))
    dT["bb"] += np.einsum("caei,cmef->afim", dX3, H.ab.vovv, optimize=True)
    #     d/dH, d/dt of <cm||ef> t(fa~,mi~)
    rdm2.aa.vovv = _plus(rdm2.aa.vovv, 1 * np.einsum("caei,fami->cmef", dX3, T.ab, optimize=True))
    dT["ab"] += np.einsum("caei,cmef->fami", dX3, H.aa.vovv, optimize=True)
    #     d/dH, d/dt of <ma~|ef~> t(cf~,mi~)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, -1 * np.einsum("caei,cfmi->maef", dX3, T.ab, optimize=True))
    dT["ab"] -= np.einsum("caei,maef->cfmi", dX3, H.ab.ovvv, optimize=True)
    # X4(maki), times t(cb~,mj~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <ma~|ki~>
        rdm2.ab.ovoo = _plus(rdm2.ab.ovoo, -1 * dX4)
    #     d/dH, d/dt of <mn||kf> t(fa~,ni~)
    rdm2.aa.ooov = _plus(rdm2.aa.ooov, -1 * np.einsum("maki,fani->mnkf", dX4, T.ab, optimize=True))
    dT["ab"] -= np.einsum("maki,mnkf->fani", dX4, H.aa.ooov, optimize=True)
    #     d/dH, d/dt of <mn~|kf~> t(a~f~,i~n~)
    rdm2.ab.ooov = _plus(rdm2.ab.ooov, -1 * np.einsum("maki,afin->mnkf", dX4, T.bb, optimize=True))
    dT["bb"] -= np.einsum("maki,mnkf->afin", dX4, H.ab.ooov, optimize=True)
    #     d/dH, d/dt of <mn~|fi~> t(fa~,kn~)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, 1 * np.einsum("maki,fakn->mnfi", dX4, T.ab, optimize=True))
    dT["ab"] += np.einsum("maki,mnfi->fakn", dX4, H.ab.oovo, optimize=True)
    #     d/dH, d/dt of <ma~|fe~> t(fe~,ki~)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, -1 * np.einsum("maki,feki->mafe", dX4, T.ab, optimize=True))
    dT["ab"] -= np.einsum("maki,mafe->feki", dX4, H.ab.ovvv, optimize=True)
    # X5(abie), times t(ce~,kj~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <a~b~||i~e~>
        rdm2.bb.vvov = _plus(rdm2.bb.vvov, 0.5 * dX5)
    #     d/dH, d/dt of f(m~e~) t(a~b~,i~m~)
    rdm1.b.ov = _plus(rdm1.b.ov, -0.5 * np.einsum("abie,abim->me", dX5, T.bb, optimize=True))
    dT["bb"] -= 0.5 * np.einsum("abie,me->abim", dX5, H.b.ov, optimize=True)
    #     d/dH, d/dt of <m~n~||i~e~> t(a~b~,m~n~)
    rdm2.bb.ooov = _plus(rdm2.bb.ooov, 0.25 * np.einsum("abie,abmn->mnie", dX5, T.bb, optimize=True))
    dT["bb"] += 0.25 * np.einsum("abie,mnie->abmn", dX5, H.bb.ooov, optimize=True)
    #     d/dH, d/dt of <b~n~||e~f~> t(a~f~,i~n~)
    rdm2.bb.vovv = _plus(rdm2.bb.vovv, 1 * np.einsum("abie,afin->bnef", dX5, T.bb, optimize=True))
    dT["bb"] += np.einsum("abie,bnef->afin", dX5, H.bb.vovv, optimize=True)
    #     d/dH, d/dt of <nb~|fe~> t(fa~,ni~)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, 1 * np.einsum("abie,fani->nbfe", dX5, T.ab, optimize=True))
    dT["ab"] += np.einsum("abie,nbfe->fani", dX5, H.ab.ovvv, optimize=True)
    # X6(amij), times t(cb~,km~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <a~m~||i~j~>
        rdm2.bb.vooo = _plus(rdm2.bb.vooo, -0.5 * dX6)
    #     d/dH, d/dt of <a~m~||e~f~> t(e~f~,i~j~)
    rdm2.bb.vovv = _plus(rdm2.bb.vovv, -0.25 * np.einsum("amij,efij->amef", dX6, T.bb, optimize=True))
    dT["bb"] -= 0.25 * np.einsum("amij,amef->efij", dX6, H.bb.vovv, optimize=True)
    #     d/dH, d/dt of <m~n~||j~f~> t(a~f~,i~n~)
    rdm2.bb.ooov = _plus(rdm2.bb.ooov, -1 * np.einsum("amij,afin->mnjf", dX6, T.bb, optimize=True))
    dT["bb"] -= np.einsum("amij,mnjf->afin", dX6, H.bb.ooov, optimize=True)
    #     d/dH, d/dt of <nm~|fj~> t(fa~,ni~)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, -1 * np.einsum("amij,fani->nmfj", dX6, T.ab, optimize=True))
    dT["ab"] -= np.einsum("amij,nmfj->fani", dX6, H.ab.oovo, optimize=True)


def backprop_triples_intermediates_bbb(dX, T, H, dT, rdm1, rdm2, constant=True):
    """Chain rule of build_triples_intermediates_bbb for the cotangents dX_g of the X_g
    (from the <i~j~k~a~b~c~|...|0> moments): for every term c H(T1) T2 of X_g, the raw T2 gradient
    dT += c dX_g . H and the H(T1) density rdm += c dX_g . T2 (rdm += c dX_g for the bare blocks)."""
    dX1, dX2 = dX
    # X1(amij), times t(b~c~,m~k~) in <i~j~k~a~b~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <a~m~||i~j~>
        rdm2.bb.vooo = _plus(rdm2.bb.vooo, -0.25 * dX1)
    #     d/dH, d/dt of f(m~e~) t(a~e~,i~j~)
    rdm1.b.ov = _plus(rdm1.b.ov, -0.25 * np.einsum("amij,aeij->me", dX1, T.bb, optimize=True))
    dT["bb"] -= 0.25 * np.einsum("amij,me->aeij", dX1, H.b.ov, optimize=True)
    #     d/dH, d/dt of <a~m~||e~f~> t(e~f~,i~j~)
    rdm2.bb.vovv = _plus(rdm2.bb.vovv, -0.125 * np.einsum("amij,efij->amef", dX1, T.bb, optimize=True))
    dT["bb"] -= 0.125 * np.einsum("amij,amef->efij", dX1, H.bb.vovv, optimize=True)
    #     d/dH, d/dt of <m~n~||j~f~> t(a~f~,i~n~)
    rdm2.bb.ooov = _plus(rdm2.bb.ooov, -0.5 * np.einsum("amij,afin->mnjf", dX1, T.bb, optimize=True))
    dT["bb"] -= 0.5 * np.einsum("amij,mnjf->afin", dX1, H.bb.ooov, optimize=True)
    #     d/dH, d/dt of <nm~|fj~> t(fa~,ni~)
    rdm2.ab.oovo = _plus(rdm2.ab.oovo, -0.5 * np.einsum("amij,fani->nmfj", dX1, T.ab, optimize=True))
    dT["ab"] -= 0.5 * np.einsum("amij,nmfj->fani", dX1, H.ab.oovo, optimize=True)
    # X2(abie), times t(e~c~,j~k~) in <i~j~k~a~b~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <a~b~||i~e~>
        rdm2.bb.vvov = _plus(rdm2.bb.vvov, 0.25 * dX2)
    #     d/dH, d/dt of <m~n~||i~e~> t(a~b~,m~n~)
    rdm2.bb.ooov = _plus(rdm2.bb.ooov, 0.125 * np.einsum("abie,abmn->mnie", dX2, T.bb, optimize=True))
    dT["bb"] += 0.125 * np.einsum("abie,mnie->abmn", dX2, H.bb.ooov, optimize=True)
    #     d/dH, d/dt of <b~n~||e~f~> t(a~f~,i~n~)
    rdm2.bb.vovv = _plus(rdm2.bb.vovv, 0.5 * np.einsum("abie,afin->bnef", dX2, T.bb, optimize=True))
    dT["bb"] += 0.5 * np.einsum("abie,bnef->afin", dX2, H.bb.vovv, optimize=True)
    #     d/dH, d/dt of <nb~|fe~> t(fa~,ni~)
    rdm2.ab.ovvv = _plus(rdm2.ab.ovvv, 0.5 * np.einsum("abie,fani->nbfe", dX2, T.ab, optimize=True))
    dT["ab"] += 0.5 * np.einsum("abie,nbfe->fani", dX2, H.ab.ovvv, optimize=True)
