"""Analytic nuclear gradient of CR-CC(2,3)_A, factorized with density fitting or Cholesky
vectors, RHF or UHF reference (ccpq_analytical_gradients.pdf).

Theory
------
CR-CC(2,3) is CC(P;Q) with P = singles and doubles, Q = all triples K = (abc, ijk):

    delta = sum_K N_K M_K / D_K,   M_K = <K|(H(T1) e^T2)_C|0>,   N_K = <0|(L1 + L2) Hbar|K>,
    D_K   = f_ii + f_jj + f_kk - f_aa - f_bb - f_cc      (bare Fock diagonal, variant A).

delta depends on Lambda, which solves the left-CCSD equations, so those are added with
multipliers sigma; the CCSD equations are added with multipliers pi. The gradient of
E_CCSD + delta is that of the Lagrangian

    E = <0|(1 + Lambda) Hbar|0> + delta + sum_mu sigma_mu <0|(1 + Lambda)[Hbar, tau_mu]|0>
        + sum_mu pi_mu <mu|Hbar|0>,

stationary in Lambda and T when, with u_K = M_K / D_K and v_K = N_K / D_K,

    sum_nu <mu|Hbar - E0|nu> sigma_nu = -s_lambda(mu),   s_lambda(mu) = sum_K <mu|Hbar|K> u_K,
    sum_nu pi_nu <nu|Hbar - E0|mu>    = -s_t(mu) - k(mu),

    s_t(mu) = sum_K [v_K d<K|(H(T1) e^T2)_C|0>/dt_mu + u_K d<0|(L1 + L2) Hbar|K>/dt_mu],
    k(mu)   = <0|(1 + Lambda)[[Hbar, tau_mu], Sigma]|0> + <0|C (Hbar - E0)|mu>,

with C_J = <0|Lambda Sigma1|J> (singles only, c(a,i) = sum_jb l(ab,ij) s(b,j)). Hbar depends on
T1 only through H(T1) = e^{-T1} H e^{T1} (ccpy.hbar.hbar_ccs_chol), d H(T1)/dt(ai) = [H(T1), a+ i],
so every singles derivative is a contraction of bare-frame densities (D, Y) with [h, a+ i] and
[B_x, a+ i]:

    sum_pq D_pq [h, a+ i]_pq + sum_x Y_x . [B_x, a+ i] = (h^T D - D h^T)(a,i) + sum_x (B_x^T Y_x - Y_x B_x^T)(a,i).

The gradient is assembled as

  1. sources: the triples M, N, D in one pass over occupied triples (Fortran,
     ccpy/lib/fortran/crcc23_gradient_loops.f90) with the moments written as
     M = A sum_g X_g . t_g (build_triples_intermediates_*), giving delta, s_lambda and s_t;
  2. responses: sigma (EOMCCSD HR), k (build_k_1A ... build_k_2C), pi (left-CCSD LH_fun);
  3. the bare-frame densities D1 = dE/dh and Y_x = dE/dB_x of the Lagrangian:
     <0|(1 + Lambda) Hbar|0> + <0|(Pi + C) Hbar|0> + <0|(1 + Lambda)[Hbar, Sigma]|0>
     + sum_K (v_K M_K + u_K N_K) - sum_K u_K v_K D_K;
  4. orbital relaxation (ccpy.gradients.orbital_response) and
  5. the derivative integrals (ccpy.gradients.derivative_integrals).

The triples are never stored: M, N and D are built one occupied triple at a time over all
virtual triples, and only intermediates of size o v^3 or o^3 v are kept. Densities and bare
integrals are held in the blocks oo, ov, vo, vv of each spin (ccpy.gradients.density_gradient),
"o" running over all occupied orbitals with the frozen core first: the correlated occupied
orbitals are [nfrozen:]. The full-array implementation, ccpy.gradients.crcc23_gradient_reference,
is the reference.
"""

import copy
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse.linalg as spla

from ccpy.density.ccsd_rdm import calc_ccsd_rdm, calc_ccsd_rdm_derivative, calc_rdm2_vvvv_slice
from ccpy.gradients.density_gradient import build_DY, get_bare_integrals, zero_density
from ccpy.gradients.derivative_integrals import contract_derivative_integrals, print_gradient
from ccpy.gradients.orbital_response import orbital_relaxation
from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol, build_hbar_ccs_chol_derivative
from ccpy.integrals import integrals
from ccpy.lib.core import crcc23_gradient_loops
from ccpy.models.integrals import Integral


def calc_crcc23_gradient(T, L, Hbar, reference, delta=None, zvector_tol=1.0e-10, components=False, verbose=True):
    """Analytic CR-CC(2,3)_A nuclear gradient, (natoms, 3) in Eh/bohr, from converged CCSD (T),
    left-CCSD (L) and the CCSD HBar (Hbar) on the SCF reference (ccpy.methods.Reference).
    components=True returns a dict with the base (CCSD, nuclear repulsion included), correction
    and total gradients."""
    if reference.system.cholesky:
        from ccpy.left.left_ccsd_chol import LH_fun
    else:
        from ccpy.left.left_ccsd import LH_fun

    t0 = time.perf_counter()
    system, nfz = reference.system, reference.nfrozen
    H0 = reference.H
    H = build_hbar_ccs_chol(T, H0, system)
    h, B = get_bare_integrals(reference)

    # 1. sources
    # the triples: M = <K|(H(T1) e^T2)_C|0> = A sum_g X_g . t_g, N = <0|(L1 + L2) Hbar|K>, D_K, in one
    # pass over the occupied triples, accumulating delta and the cotangents (calc_triples)
    X = {"aaa": build_triples_intermediates_aaa(T, H), "aab": build_triples_intermediates_aab(T, H),
         "abb": build_triples_intermediates_abb(T, H), "bbb": build_triples_intermediates_bbb(T, H)}
    rdm1_Q, rdm2_Q = zero_triples_density(system)
    tr = calc_triples(X, T, L, H, reference, rdm1_Q, rdm2_Q)
    if delta is not None and abs(tr["delta"] - delta) > 1.0e-8:
        raise RuntimeError(f"delta from the moments ({tr['delta']:.12f}) does not match the energy ({delta:.12f}).")
    # v . M through the intermediates X_g: the T2 gradient (dT) and the H(T1) densities (rdm_Q)
    dT = {"aa": tr["dT"]["aa"].copy(), "ab": tr["dT"]["ab"].copy(), "bb": tr["dT"]["bb"].copy()}
    backprop_triples_intermediates_aaa(tr["dX"]["aaa"], T, H, dT, rdm1_Q, rdm2_Q)
    backprop_triples_intermediates_aab(tr["dX"]["aab"], T, H, dT, rdm1_Q, rdm2_Q)
    backprop_triples_intermediates_abb(tr["dX"]["abb"], T, H, dT, rdm1_Q, rdm2_Q)
    backprop_triples_intermediates_bbb(tr["dX"]["bbb"], T, H, dT, rdm1_Q, rdm2_Q)
    # bare-frame densities of sum_K (v_K M_K + u_K N_K)
    D_Q, Y_Q = build_DY(rdm1_Q, rdm2_Q, None, H.chol, T, B, reference)
    # s_lambda(mu) = sum_K <mu|Hbar|K> u_K (doubles of the unique amplitudes: antisymmetrized)
    dL = tr["dL"]
    s_lambda = SimpleNamespace(
        a=dL["a"], b=dL["b"], ab=dL["ab"],
        aa=dL["aa"] - dL["aa"].transpose(1, 0, 2, 3) - dL["aa"].transpose(0, 1, 3, 2) + dL["aa"].transpose(1, 0, 3, 2),
        bb=dL["bb"] - dL["bb"].transpose(1, 0, 2, 3) - dL["bb"].transpose(0, 1, 3, 2) + dL["bb"].transpose(1, 0, 3, 2),
    )
    # s_t(mu) = sum_K [v_K dM_K/dt_mu + u_K dN_K/dt_mu]
    s_t = SimpleNamespace(
        a=build_s_t_1A(D_Q, Y_Q, h, B, nfz), b=build_s_t_1B(D_Q, Y_Q, h, B, nfz), ab=dT["ab"],
        aa=dT["aa"] - dT["aa"].transpose(1, 0, 2, 3) - dT["aa"].transpose(0, 1, 3, 2) + dT["aa"].transpose(1, 0, 3, 2),
        bb=dT["bb"] - dT["bb"].transpose(1, 0, 2, 3) - dT["bb"].transpose(0, 1, 3, 2) + dT["bb"].transpose(1, 0, 3, 2),
    )

    # 2. responses
    # sum_nu <mu|Hbar - E0|nu> sigma_nu = -s_lambda(mu)
    S = solve_sigma(s_lambda, T, Hbar, system, zvector_tol)
    # intermediates of k
    # C_J = <0|Lambda Sigma1|J>: c(a,i) = sum_jb l(ab,ij) s(b,j)
    c_a = np.einsum("abij,bj->ai", L.aa, S.a, optimize=True) + np.einsum("abij,bj->ai", L.ab, S.b, optimize=True)
    c_b = np.einsum("baji,bj->ai", L.ab, S.a, optimize=True) + np.einsum("abij,bj->ai", L.bb, S.b, optimize=True)
    # densities of <0|(1 + Lambda) (H(T1) e^{T2})_C|0> (the CCSD Lagrangian, reference included), and
    # their derivative along Sigma2
    rdm1_L, rdm2_L = calc_ccsd_rdm(T, L, system)
    D_L, Y_L = build_DY(rdm1_L, rdm2_L, lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s), H.chol, T, B, reference,
                        ref_weight=True)
    D_S2, Y_S2 = build_DY(*calc_ccsd_rdm_derivative(T, S, L, system), lambda a, s: calc_rdm2_vvvv_slice(L, S, a, s),
                          H.chol, T, B, reference)
    # H1 = [H(T1), S1] in the blocks of H(T1); [h, S1] and [B, S1] in the bare blocks
    # ([M, S1]_pq = sum_e M_pe s(e,q) - sum_m s(p,m) M_mq, s(a,i) nonzero for correlated i)
    H1 = build_hbar_ccs_chol_derivative(S, H, system)
    h1, B1 = zero_density(reference), zero_density(reference, B.a.oo.shape[0])
    h1.a.oo[:, nfz:] = h.a.ov @ S.a
    h1.a.vo[...] = -S.a @ h.a.oo[nfz:, :]
    h1.a.vo[:, nfz:] += h.a.vv @ S.a
    h1.a.vv[...] = -S.a @ h.a.ov[nfz:, :]
    h1.b.oo[:, nfz:] = h.b.ov @ S.b
    h1.b.vo[...] = -S.b @ h.b.oo[nfz:, :]
    h1.b.vo[:, nfz:] += h.b.vv @ S.b
    h1.b.vv[...] = -S.b @ h.b.ov[nfz:, :]
    B1.a.oo[:, :, nfz:] = B.a.ov @ S.a
    B1.a.vo[...] = -S.a @ B.a.oo[:, nfz:, :]
    B1.a.vo[:, :, nfz:] += B.a.vv @ S.a
    B1.a.vv[...] = -S.a @ B.a.ov[:, nfz:, :]
    B1.b.oo[:, :, nfz:] = B.b.ov @ S.b
    B1.b.vo[...] = -S.b @ B.b.oo[:, nfz:, :]
    B1.b.vo[:, :, nfz:] += B.b.vv @ S.b
    B1.b.vv[...] = -S.b @ B.b.ov[:, nfz:, :]
    # Y of the CCSD Lagrangian with H(T1) -> H1 (its T1-transformed vectors [Bt, S1] and the bare
    # [B, S1] of its separable and reference terms)
    _, Y_LS1 = build_DY(rdm1_L, rdm2_L, lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s), H1.chol, T, B1, reference,
                        ref_weight=True)
    # <0|C (Hbar - E0)|mu>: the left-CCSD Jacobian (LH_fun) on C
    Cop, LHc = copy.deepcopy(T), copy.deepcopy(T)
    Cop.a, Cop.b = np.asfortranarray(c_a), np.asfortranarray(c_b)
    Cop.aa, Cop.ab, Cop.bb = np.zeros_like(L.aa, order="F"), np.zeros_like(L.ab, order="F"), np.zeros_like(L.bb, order="F")
    LH_fun(LHc, Cop, T, Hbar, False, system)
    # k(mu) = <0|(1 + Lambda)[[Hbar, tau_mu], Sigma]|0> + <0|C (Hbar - E0)|mu>
    k = SimpleNamespace(
        a=build_k_1A(np.array(LHc.a), h, B, h1, B1, D_L, Y_L, D_S2, Y_S2, Y_LS1, nfz),
        b=build_k_1B(np.array(LHc.b), h, B, h1, B1, D_L, Y_L, D_S2, Y_S2, Y_LS1, nfz),
        aa=build_k_2A(np.array(LHc.aa), S, T, L, H, H1),
        ab=build_k_2B(np.array(LHc.ab), S, T, L, H, H1),
        bb=build_k_2C(np.array(LHc.bb), S, T, L, H, H1),
    )
    # sum_nu pi_nu <nu|Hbar - E0|mu> = -s_t(mu) - k(mu)
    P = solve_pi(s_t, k, T, Hbar, system, zvector_tol)

    # 3. bare-frame densities of the correction
    # <0|(Pi + C) Hbar|0>
    Xb = SimpleNamespace(a=P.a + c_a, b=P.b + c_b, aa=P.aa, ab=P.ab, bb=P.bb)
    D1c, Yc = build_DY(*calc_ccsd_rdm(T, Xb, system, energy=False), lambda a, s: calc_rdm2_vvvv_slice(Xb, T, a, s),
                       H.chol, T, B, reference)
    # <0|(1 + Lambda)[Hbar, Sigma2]|0>, and the Y of [Hbar, Sigma1] through H1
    D1c.a.oo += D_S2.a.oo
    D1c.a.ov += D_S2.a.ov
    D1c.a.vo += D_S2.a.vo
    D1c.a.vv += D_S2.a.vv
    D1c.b.oo += D_S2.b.oo
    D1c.b.ov += D_S2.b.ov
    D1c.b.vo += D_S2.b.vo
    D1c.b.vv += D_S2.b.vv
    Yc.a.oo += Y_S2.a.oo + Y_LS1.a.oo
    Yc.a.ov += Y_S2.a.ov + Y_LS1.a.ov
    Yc.a.vo += Y_S2.a.vo + Y_LS1.a.vo
    Yc.a.vv += Y_S2.a.vv + Y_LS1.a.vv
    Yc.b.oo += Y_S2.b.oo + Y_LS1.b.oo
    Yc.b.ov += Y_S2.b.ov + Y_LS1.b.ov
    Yc.b.vo += Y_S2.b.vo + Y_LS1.b.vo
    Yc.b.vv += Y_S2.b.vv + Y_LS1.b.vv
    # <0|(1 + Lambda)[Hbar, Sigma1]|0>: the CCSD Lagrangian densities rotated, D_L S1^T - S1^T D_L
    D1c.a.oo[nfz:, :] -= S.a.T @ D_L.a.vo
    D1c.a.ov += D_L.a.oo[:, nfz:] @ S.a.T
    D1c.a.ov[nfz:, :] -= S.a.T @ D_L.a.vv
    D1c.a.vv += D_L.a.vo[:, nfz:] @ S.a.T
    D1c.b.oo[nfz:, :] -= S.b.T @ D_L.b.vo
    D1c.b.ov += D_L.b.oo[:, nfz:] @ S.b.T
    D1c.b.ov[nfz:, :] -= S.b.T @ D_L.b.vv
    D1c.b.vv += D_L.b.vo[:, nfz:] @ S.b.T
    Yc.a.oo[:, nfz:, :] -= S.a.T @ Y_L.a.vo
    Yc.a.ov += Y_L.a.oo[:, :, nfz:] @ S.a.T
    Yc.a.ov[:, nfz:, :] -= S.a.T @ Y_L.a.vv
    Yc.a.vv += Y_L.a.vo[:, :, nfz:] @ S.a.T
    Yc.b.oo[:, nfz:, :] -= S.b.T @ Y_L.b.vo
    Yc.b.ov += Y_L.b.oo[:, :, nfz:] @ S.b.T
    Yc.b.ov[:, nfz:, :] -= S.b.T @ Y_L.b.vv
    Yc.b.vv += Y_L.b.vo[:, :, nfz:] @ S.b.T
    # sum_K (v_K M_K + u_K N_K)
    D1c.a.oo += D_Q.a.oo
    D1c.a.ov += D_Q.a.ov
    D1c.a.vo += D_Q.a.vo
    D1c.a.vv += D_Q.a.vv
    D1c.b.oo += D_Q.b.oo
    D1c.b.ov += D_Q.b.ov
    D1c.b.vo += D_Q.b.vo
    D1c.b.vv += D_Q.b.vv
    Yc.a.oo += Y_Q.a.oo
    Yc.a.ov += Y_Q.a.ov
    Yc.a.vo += Y_Q.a.vo
    Yc.a.vv += Y_Q.a.vv
    Yc.b.oo += Y_Q.b.oo
    Yc.b.ov += Y_Q.b.ov
    Yc.b.vo += Y_Q.b.vo
    Yc.b.vv += Y_Q.b.vv
    # -sum_K u_K v_K D_K = sum_p d_p f_pp: the bare Fock diagonal of the denominators
    d = tr["d"]
    rdm1_f = Integral.from_empty(system, 1, use_none=True)
    rdm1_f.a.oo, rdm1_f.a.vv = np.diag(d["a"]["o"]), np.diag(d["a"]["v"])
    rdm1_f.b.oo, rdm1_f.b.vv = np.diag(d["b"]["o"]), np.diag(d["b"]["v"])
    D_f, Y_f = build_DY(rdm1_f, Integral.from_empty(system, 2, use_none=True), None, H0.chol, None, B, reference)
    D1c.a.oo += D_f.a.oo
    D1c.a.ov += D_f.a.ov
    D1c.a.vo += D_f.a.vo
    D1c.a.vv += D_f.a.vv
    D1c.b.oo += D_f.b.oo
    D1c.b.ov += D_f.b.ov
    D1c.b.vo += D_f.b.vo
    D1c.b.vv += D_f.b.vv
    Yc.a.oo += Y_f.a.oo
    Yc.a.ov += Y_f.a.ov
    Yc.a.vo += Y_f.a.vo
    Yc.a.vv += Y_f.a.vv
    Yc.b.oo += Y_f.b.oo
    Yc.b.ov += Y_f.b.ov
    Yc.b.vo += Y_f.b.vo
    Yc.b.vv += Y_f.b.vv
    # total: the CCSD Lagrangian <0|(1 + Lambda) Hbar|0> plus the correction
    D1, Y = zero_density(reference), zero_density(reference, B.a.oo.shape[0])
    D1.a.oo[...], D1.a.ov[...], D1.a.vo[...], D1.a.vv[...] = (D_L.a.oo + D1c.a.oo, D_L.a.ov + D1c.a.ov,
                                                              D_L.a.vo + D1c.a.vo, D_L.a.vv + D1c.a.vv)
    D1.b.oo[...], D1.b.ov[...], D1.b.vo[...], D1.b.vv[...] = (D_L.b.oo + D1c.b.oo, D_L.b.ov + D1c.b.ov,
                                                              D_L.b.vo + D1c.b.vo, D_L.b.vv + D1c.b.vv)
    Y.a.oo[...], Y.a.ov[...], Y.a.vo[...], Y.a.vv[...] = (Y_L.a.oo + Yc.a.oo, Y_L.a.ov + Yc.a.ov,
                                                          Y_L.a.vo + Yc.a.vo, Y_L.a.vv + Yc.a.vv)
    Y.b.oo[...], Y.b.ov[...], Y.b.vo[...], Y.b.vv[...] = (Y_L.b.oo + Yc.b.oo, Y_L.b.ov + Yc.b.ov,
                                                          Y_L.b.vo + Yc.b.vo, Y_L.b.vv + Yc.b.vv)

    # 4. orbital relaxation and 5. derivative integrals
    D, Yr, W = orbital_relaxation(D1, Y, reference, zvector_tol=zvector_tol, verbose=verbose)
    grad = contract_derivative_integrals(D, Yr, W, reference)
    if verbose:
        print_gradient(reference.molecule, grad, "CR-CC(2,3)_A Analytic Gradient", time.perf_counter() - t0)
    if not components:
        return grad
    D, Yr, W = orbital_relaxation(D1c, Yc, reference, zvector_tol=zvector_tol, verbose=False)
    grad_corr = contract_derivative_integrals(D, Yr, W, reference) - integrals.nuclear_repulsion_deriv(reference.molecule)
    return {"base": grad - grad_corr, "correction": grad_corr, "total": grad}


def build_s_t_1A(D_Q, Y_Q, h, B, nfz):
    """s_t(ia) = sum_K [v_K d<K|(H(T1) e^T2)_C|0>/dt(ai) + u_K d<0|(L1 + L2) Hbar|K>/dt(ai)], the
    alpha singles of the triples source. M and N depend on t(ai) only through H(T1),
    d H(T1)/dt(ai) = [H(T1), a+ i], so s_t(ia) is the contraction of the bare-frame densities
    (D_Q, Y_Q) of sum_K (v_K M_K + u_K N_K) with [h, a+ i] and [B_x, a+ i]:

        (h^T D_Q - D_Q h^T)(a,i) + sum_x (B_x^T Y_Q,x - Y_Q,x B_x^T)(a,i),   i correlated."""
    X = h.a.ov.T @ D_Q.a.oo[:, nfz:] + h.a.vv.T @ D_Q.a.vo[:, nfz:]
    X -= D_Q.a.vo @ h.a.oo[nfz:, :].T + D_Q.a.vv @ h.a.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.a.ov, Y_Q.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.a.vv, Y_Q.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Q.a.vo, B.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Q.a.vv, B.a.ov[:, nfz:, :], optimize=True)
    return X


def build_s_t_1B(D_Q, Y_Q, h, B, nfz):
    """s_t(i~a~), the beta singles of the triples source (build_s_t_1A with the beta blocks)."""
    X = h.b.ov.T @ D_Q.b.oo[:, nfz:] + h.b.vv.T @ D_Q.b.vo[:, nfz:]
    X -= D_Q.b.vo @ h.b.oo[nfz:, :].T + D_Q.b.vv @ h.b.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.b.ov, Y_Q.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.vv, Y_Q.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Q.b.vo, B.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Q.b.vv, B.b.ov[:, nfz:, :], optimize=True)
    return X


def build_k_1A(LHc, h, B, h1, B1, D_L, Y_L, D_S2, Y_S2, Y_LS1, nfz):
    """k(ia) = <0|(1 + Lambda)[[Hbar, a+ i], Sigma]|0> + <0|C (Hbar - E0)|ia>, the alpha singles
    of the source of the pi equations. Hbar depends on t(ai) only through H(T1),
    d H(T1)/dt(ai) = [H(T1), a+ i], so the first term splits into

        <0|(1 + Lambda)[[Hbar, a+ i], Sigma2]|0> + <0|(1 + Lambda)[[Hbar, a+ i], Sigma1]|0>,

    contractions of bare-frame densities (D, Y) with the rotated integrals [h, a+ i], [B_x, a+ i]:

        sum_pq D_pq [h, a+ i]_pq + sum_x Y_x . [B_x, a+ i] = (h^T D - D h^T)(a,i) + sum_x (B_x^T Y_x - Y_x B_x^T)(a,i),

    with i correlated (occupied indices [nfrozen:])."""
    # <0|C (Hbar - E0)|ia> (left-CCSD LH_fun)
    X = LHc.copy()
    # <0|(1 + Lambda)[[Hbar, a+ i], Sigma2]|0>: the Sigma2-derivative densities of the CCSD Lagrangian
    # against [h, a+ i], [B, a+ i]
    X += h.a.ov.T @ D_S2.a.oo[:, nfz:] + h.a.vv.T @ D_S2.a.vo[:, nfz:]
    X -= D_S2.a.vo @ h.a.oo[nfz:, :].T + D_S2.a.vv @ h.a.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.a.ov, Y_S2.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.a.vv, Y_S2.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_S2.a.vo, B.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_S2.a.vv, B.a.ov[:, nfz:, :], optimize=True)
    # <0|(1 + Lambda)[[Hbar, a+ i], Sigma1]|0>: the CCSD Lagrangian densities against [[h, S1], a+ i],
    # [[B, S1], a+ i] ...
    X += h1.a.ov.T @ D_L.a.oo[:, nfz:] + h1.a.vv.T @ D_L.a.vo[:, nfz:]
    X -= D_L.a.vo @ h1.a.oo[nfz:, :].T + D_L.a.vv @ h1.a.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B1.a.ov, Y_L.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B1.a.vv, Y_L.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_L.a.vo, B1.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_L.a.vv, B1.a.ov[:, nfz:, :], optimize=True)
    # ... and the Y of the Lagrangian with H(T1) -> [H(T1), S1] against [B, a+ i]
    X += np.einsum("xqa,xqi->ai", B.a.ov, Y_LS1.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.a.vv, Y_LS1.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_LS1.a.vo, B.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_LS1.a.vv, B.a.ov[:, nfz:, :], optimize=True)
    return X


def build_k_1B(LHc, h, B, h1, B1, D_L, Y_L, D_S2, Y_S2, Y_LS1, nfz):
    """k(i~a~), the beta singles of the source of the pi equations (build_k_1A with the beta
    blocks)."""
    # <0|C (Hbar - E0)|i~a~> (left-CCSD LH_fun)
    X = LHc.copy()
    # <0|(1 + Lambda)[[Hbar, a~+ i~], Sigma2]|0>
    X += h.b.ov.T @ D_S2.b.oo[:, nfz:] + h.b.vv.T @ D_S2.b.vo[:, nfz:]
    X -= D_S2.b.vo @ h.b.oo[nfz:, :].T + D_S2.b.vv @ h.b.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.b.ov, Y_S2.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.vv, Y_S2.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_S2.b.vo, B.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_S2.b.vv, B.b.ov[:, nfz:, :], optimize=True)
    # <0|(1 + Lambda)[[Hbar, a~+ i~], Sigma1]|0>
    X += h1.b.ov.T @ D_L.b.oo[:, nfz:] + h1.b.vv.T @ D_L.b.vo[:, nfz:]
    X -= D_L.b.vo @ h1.b.oo[nfz:, :].T + D_L.b.vv @ h1.b.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B1.b.ov, Y_L.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B1.b.vv, Y_L.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_L.b.vo, B1.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_L.b.vv, B1.b.ov[:, nfz:, :], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.ov, Y_LS1.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.vv, Y_LS1.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_LS1.b.vo, B.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_LS1.b.vv, B.b.ov[:, nfz:, :], optimize=True)
    return X


def build_k_2A(LHc, S, T, L, H, H1):
    """k(ijab) = <0|(1 + Lambda)[[Hbar, tau(ab,ij)], Sigma]|0> + <0|C (Hbar - E0)|ijab>, the
    aa doubles of the source of the pi equations (C_J = <0|Lambda Sigma1|J>, singles only).
    Sigma2 enters through the T2^2 terms of the CCSD equations, Sigma1 through H1 = [H(T1), S1]
    in the left-CCSD doubles terms. Terms are built in the stored layout and antisymmetrized
    before the (already antisymmetric) last term is added."""
    # <0|L2 [[(V_N T2^2)_C/2, tau(ab,ij)], S2]|0>: the T2^2 terms of the CCSD equations along S2
    #     <mn||ef> s(af,mn) l(ab,ij)  -> (eb,ij)
    X = -0.25 * np.einsum("mnef,afmn,abij->ebij", H.aa.oovv, S.aa, L.aa, optimize=True)
    #     <mn||ef> s(eb,ij) l(ab,ij)  -> (af,mn)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H.aa.oovv, S.aa, L.aa, optimize=True)
    #     <mn~|ef~> s(af~,mn~) l(ab,ij)  -> (eb,ij)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H.ab.oovv, S.ab, L.aa, optimize=True)
    #     <mn||ef> s(ef,in) l(ab,ij)  -> (ab,mj)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H.aa.oovv, S.aa, L.aa, optimize=True)
    #     <mn||ef> s(ab,mj) l(ab,ij)  -> (ef,in)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H.aa.oovv, S.aa, L.aa, optimize=True)
    #     <mn~|ef~> s(ef~,in~) l(ab,ij)  -> (ab,mj)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H.ab.oovv, S.ab, L.aa, optimize=True)
    #     <mn||ef> s(af,in) l(ab,ij)  -> (eb,mj)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H.aa.oovv, S.aa, L.aa, optimize=True)
    #     <mn||ef> s(eb,mj) l(ab,ij)  -> (af,in)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H.aa.oovv, S.aa, L.aa, optimize=True)
    #     <mn~|ef~> s(af~,in~) l(ab,ij)  -> (eb,mj)
    X += np.einsum("mnef,afin,abij->ebmj", H.ab.oovv, S.ab, L.aa, optimize=True)
    #     <mn||ef> s(ef,ij) l(ab,ij)  -> (ab,mn)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H.aa.oovv, S.aa, L.aa, optimize=True)
    #     <mn||ef> s(ab,mn) l(ab,ij)  -> (ef,ij)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H.aa.oovv, S.aa, L.aa, optimize=True)
    #     <mn||ef> s(eb~,ij~) l(ab~,ij~)  -> (af,mn)
    X -= 0.5 * np.einsum("mnef,ebij,abij->afmn", H.aa.oovv, S.ab, L.ab, optimize=True)
    #     <mn||ef> s(ab~,mj~) l(ab~,ij~)  -> (ef,in)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H.aa.oovv, S.ab, L.ab, optimize=True)
    #     <mn||ef> s(fb~,nj~) l(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H.aa.oovv, S.ab, L.ab, optimize=True)
    #     <mn~|ef~> s(f~b~,n~j~) l(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H.ab.oovv, S.bb, L.ab, optimize=True)
    # <0|(1 + L1 + L2)[[H(T1), S1], tau(ab,ij)]_C|0>: the left-CCSD doubles terms with H1 = [H(T1), S1]
    #     <ij||ab>'
    X += 0.25 * H1.aa.oovv.transpose(2, 3, 0, 1)
    #     f'(me) l(a,i)  -> (ae,im)
    X += np.einsum("me,ai->aeim", H1.a.ov, L.a, optimize=True)
    #     <mn||if>' l(a,i)  -> (af,mn)
    X -= 0.5 * np.einsum("mnif,ai->afmn", H1.aa.ooov, L.a, optimize=True)
    #     <an||ef>' l(a,i)  -> (ef,in)
    X += 0.5 * np.einsum("anef,ai->efin", H1.aa.vovv, L.a, optimize=True)
    #     f'(ae) l(ab,ij)  -> (eb,ij)
    X += 0.5 * np.einsum("ae,abij->ebij", H1.a.vv, L.aa, optimize=True)
    #     <mn||ef>' t(eb,ij) l(ab,ij)  -> (af,mn)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H1.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn||ef>' t(af,mn) l(ab,ij)  -> (eb,ij)
    X -= 0.25 * np.einsum("mnef,afmn,abij->ebij", H1.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn~|ef~>' t(af~,mn~) l(ab,ij)  -> (eb,ij)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H1.ab.oovv, T.ab, L.aa, optimize=True)
    #     f'(mi) l(ab,ij)  -> (ab,mj)
    X -= 0.5 * np.einsum("mi,abij->abmj", H1.a.oo, L.aa, optimize=True)
    #     <mn||ef>' t(ab,mj) l(ab,ij)  -> (ef,in)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H1.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn||ef>' t(ef,in) l(ab,ij)  -> (ab,mj)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H1.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn~|ef~>' t(ef~,in~) l(ab,ij)  -> (ab,mj)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H1.ab.oovv, T.ab, L.aa, optimize=True)
    #     <am||ie>' l(ab,ij)  -> (eb,mj)
    X += np.einsum("amie,abij->ebmj", H1.aa.voov, L.aa, optimize=True)
    #     <mn||ef>' t(eb,mj) l(ab,ij)  -> (af,in)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H1.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn||ef>' t(af,in) l(ab,ij)  -> (eb,mj)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H1.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn~|ef~>' t(af~,in~) l(ab,ij)  -> (eb,mj)
    X += np.einsum("mnef,afin,abij->ebmj", H1.ab.oovv, T.ab, L.aa, optimize=True)
    #     <mn||ij>' l(ab,ij)  -> (ab,mn)
    X += 0.125 * np.einsum("mnij,abij->abmn", H1.aa.oooo, L.aa, optimize=True)
    #     <mn||ef>' t(ab,mn) l(ab,ij)  -> (ef,ij)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H1.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn||ef>' t(ef,ij) l(ab,ij)  -> (ab,mn)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H1.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn||ef>' t(eb~,ij~) l(ab~,ij~)  -> (af,mn)
    X -= 0.5 * np.einsum("mnef,ebij,abij->afmn", H1.aa.oovv, T.ab, L.ab, optimize=True)
    #     <mn||ef>' t(ab~,mj~) l(ab~,ij~)  -> (ef,in)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H1.aa.oovv, T.ab, L.ab, optimize=True)
    #     <mn||ef>' t(fb~,nj~) l(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H1.aa.oovv, T.ab, L.ab, optimize=True)
    #     <mn~|ef~>' t(f~b~,n~j~) l(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H1.ab.oovv, T.bb, L.ab, optimize=True)
    #     <mb~|ej~>' l(ab~,ij~)  -> (ae,im)
    X += np.einsum("mbej,abij->aeim", H1.ab.ovvo, L.ab, optimize=True)
    # vvvv of H1: 1/8 sum_ab L(abij) <ab||ef>' with <ab|ef>' = sum_x (dB_ae B_bf + B_ae dB_bf)
    for B1, B2 in ((H1.chol.a.vv, H.chol.a.vv), (H.chol.a.vv, H1.chol.a.vv)):
        Y = cholesky_ladder(L.aa, B1, B2)
        X += 0.125 * (Y - Y.transpose(1, 0, 2, 3))
    X -= np.transpose(X, (1, 0, 2, 3))
    X -= np.transpose(X, (0, 1, 3, 2))
    # <0|C (Hbar - E0)|ijab> (left-CCSD LH_fun)
    X += LHc
    return X


def build_k_2B(LHc, S, T, L, H, H1):
    """k(ij~ab~) = <0|(1 + Lambda)[[Hbar, tau(ab~,ij~)], Sigma]|0> + <0|C (Hbar - E0)|ij~ab~>, the
    ab doubles of the source of the pi equations (C_J = <0|Lambda Sigma1|J>, singles only).
    Sigma2 enters through the T2^2 terms of the CCSD equations, Sigma1 through H1 = [H(T1), S1]
    in the left-CCSD doubles terms. Terms are built in the stored layout and added to the last term."""
    # <0|L2 [[(V_N T2^2)_C/2, tau(ab~,ij~)], S2]|0>: the T2^2 terms of the CCSD equations along S2
    #     <mn~|ef~> s(eb,ij) l(ab,ij)  -> (af~,mn~)
    X = -0.5 * np.einsum("mnef,ebij,abij->afmn", H.ab.oovv, S.aa, L.aa, optimize=True)
    #     <mn~|ef~> s(ab,mj) l(ab,ij)  -> (ef~,in~)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H.ab.oovv, S.aa, L.aa, optimize=True)
    #     <mn~|ef~> s(eb,mj) l(ab,ij)  -> (af~,in~)
    X += np.einsum("mnef,ebmj,abij->afin", H.ab.oovv, S.aa, L.aa, optimize=True)
    #     <m~n~||e~f~> s(af~,in~) l(ab,ij)  -> (be~,jm~)
    X += 0.5 * np.einsum("mnef,afin,abij->bejm", H.bb.oovv, S.ab, L.aa, optimize=True)
    #     <m~n~||e~f~> s(be~,jm~) l(ab,ij)  -> (af~,in~)
    X += 0.5 * np.einsum("mnef,bejm,abij->afin", H.bb.oovv, S.ab, L.aa, optimize=True)
    #     <mn||ef> s(af,mn) l(ab~,ij~)  -> (eb~,ij~)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H.aa.oovv, S.aa, L.ab, optimize=True)
    #     <mn~|ef~> s(af~,mn~) l(ab~,ij~)  -> (eb~,ij~)
    X -= np.einsum("mnef,afmn,abij->ebij", H.ab.oovv, S.ab, L.ab, optimize=True)
    #     <mn~|ef~> s(eb~,ij~) l(ab~,ij~)  -> (af~,mn~)
    X -= np.einsum("mnef,ebij,abij->afmn", H.ab.oovv, S.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> s(b~f~,m~n~) l(ab~,ij~)  -> (ae~,ij~)
    X -= 0.5 * np.einsum("mnef,bfmn,abij->aeij", H.bb.oovv, S.bb, L.ab, optimize=True)
    #     <nm~|fe~> s(fb~,nm~) l(ab~,ij~)  -> (ae~,ij~)
    X -= np.einsum("nmfe,fbnm,abij->aeij", H.ab.oovv, S.ab, L.ab, optimize=True)
    #     <nm~|fe~> s(ae~,ij~) l(ab~,ij~)  -> (fb~,nm~)
    X -= np.einsum("nmfe,aeij,abij->fbnm", H.ab.oovv, S.ab, L.ab, optimize=True)
    #     <mn||ef> s(ef,in) l(ab~,ij~)  -> (ab~,mj~)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H.aa.oovv, S.aa, L.ab, optimize=True)
    #     <mn~|ef~> s(ef~,in~) l(ab~,ij~)  -> (ab~,mj~)
    X -= np.einsum("mnef,efin,abij->abmj", H.ab.oovv, S.ab, L.ab, optimize=True)
    #     <mn~|ef~> s(ab~,mj~) l(ab~,ij~)  -> (ef~,in~)
    X -= np.einsum("mnef,abmj,abij->efin", H.ab.oovv, S.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> s(e~f~,j~n~) l(ab~,ij~)  -> (ab~,im~)
    X -= 0.5 * np.einsum("mnef,efjn,abij->abim", H.bb.oovv, S.bb, L.ab, optimize=True)
    #     <nm~|fe~> s(fe~,nj~) l(ab~,ij~)  -> (ab~,im~)
    X -= np.einsum("nmfe,fenj,abij->abim", H.ab.oovv, S.ab, L.ab, optimize=True)
    #     <nm~|fe~> s(ab~,im~) l(ab~,ij~)  -> (fe~,nj~)
    X -= np.einsum("nmfe,abim,abij->fenj", H.ab.oovv, S.ab, L.ab, optimize=True)
    #     <mn||ef> s(ae,im) l(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.aa.oovv, S.aa, L.ab, optimize=True)
    #     <nm~|fe~> s(ae~,im~) l(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("nmfe,aeim,abij->fbnj", H.ab.oovv, S.ab, L.ab, optimize=True)
    #     <nm~|fe~> s(fb~,nj~) l(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("nmfe,fbnj,abij->aeim", H.ab.oovv, S.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> s(f~b~,n~j~) l(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("mnef,fbnj,abij->aeim", H.bb.oovv, S.bb, L.ab, optimize=True)
    #     <mn~|ef~> s(af~,mj~) l(ab~,ij~)  -> (eb~,in~)
    X += np.einsum("mnef,afmj,abij->ebin", H.ab.oovv, S.ab, L.ab, optimize=True)
    #     <mn~|ef~> s(eb~,in~) l(ab~,ij~)  -> (af~,mj~)
    X += np.einsum("mnef,ebin,abij->afmj", H.ab.oovv, S.ab, L.ab, optimize=True)
    #     <mn~|ef~> s(ef~,ij~) l(ab~,ij~)  -> (ab~,mn~)
    X += np.einsum("mnef,efij,abij->abmn", H.ab.oovv, S.ab, L.ab, optimize=True)
    #     <mn~|ef~> s(ab~,mn~) l(ab~,ij~)  -> (ef~,ij~)
    X += np.einsum("mnef,abmn,abij->efij", H.ab.oovv, S.ab, L.ab, optimize=True)
    #     <nm~|fe~> s(e~b~,i~j~) l(a~b~,i~j~)  -> (fa~,nm~)
    X -= 0.5 * np.einsum("nmfe,ebij,abij->fanm", H.ab.oovv, S.bb, L.bb, optimize=True)
    #     <nm~|fe~> s(a~b~,m~j~) l(a~b~,i~j~)  -> (fe~,ni~)
    X -= 0.5 * np.einsum("nmfe,abmj,abij->feni", H.ab.oovv, S.bb, L.bb, optimize=True)
    #     <mn~|ef~> s(a~f~,i~n~) l(a~b~,i~j~)  -> (eb~,mj~)
    X += np.einsum("mnef,afin,abij->ebmj", H.ab.oovv, S.bb, L.bb, optimize=True)
    #     <mn||ef> s(fa~,ni~) l(a~b~,i~j~)  -> (eb~,mj~)
    X += 0.5 * np.einsum("mnef,fani,abij->ebmj", H.aa.oovv, S.ab, L.bb, optimize=True)
    #     <mn||ef> s(eb~,mj~) l(a~b~,i~j~)  -> (fa~,ni~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->fani", H.aa.oovv, S.ab, L.bb, optimize=True)
    # <0|(1 + L1 + L2)[[H(T1), S1], tau(ab~,ij~)]_C|0>: the left-CCSD doubles terms with H1 = [H(T1), S1]
    #     <ij~|ab~>'
    X += H1.ab.oovv.transpose(2, 3, 0, 1)
    #     f'(m~e~) l(a,i)  -> (ae~,im~)
    X += np.einsum("me,ai->aeim", H1.b.ov, L.a, optimize=True)
    #     <mn~|if~>' l(a,i)  -> (af~,mn~)
    X -= np.einsum("mnif,ai->afmn", H1.ab.ooov, L.a, optimize=True)
    #     <an~|ef~>' l(a,i)  -> (ef~,in~)
    X += np.einsum("anef,ai->efin", H1.ab.vovv, L.a, optimize=True)
    #     f'(me) l(a~,i~)  -> (ea~,mi~)
    X += np.einsum("me,ai->eami", H1.a.ov, L.b, optimize=True)
    #     <nm~|fi~>' l(a~,i~)  -> (fa~,nm~)
    X -= np.einsum("nmfi,ai->fanm", H1.ab.oovo, L.b, optimize=True)
    #     <na~|fe~>' l(a~,i~)  -> (fe~,ni~)
    X += np.einsum("nafe,ai->feni", H1.ab.ovvv, L.b, optimize=True)
    #     <mn~|ef~>' t(eb,ij) l(ab,ij)  -> (af~,mn~)
    X -= 0.5 * np.einsum("mnef,ebij,abij->afmn", H1.ab.oovv, T.aa, L.aa, optimize=True)
    #     <mn~|ef~>' t(ab,mj) l(ab,ij)  -> (ef~,in~)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H1.ab.oovv, T.aa, L.aa, optimize=True)
    #     <mn~|ef~>' t(eb,mj) l(ab,ij)  -> (af~,in~)
    X += np.einsum("mnef,ebmj,abij->afin", H1.ab.oovv, T.aa, L.aa, optimize=True)
    #     <am~|ie~>' l(ab,ij)  -> (be~,jm~)
    X += np.einsum("amie,abij->bejm", H1.ab.voov, L.aa, optimize=True)
    #     <m~n~||e~f~>' t(be~,jm~) l(ab,ij)  -> (af~,in~)
    X += 0.5 * np.einsum("mnef,bejm,abij->afin", H1.bb.oovv, T.ab, L.aa, optimize=True)
    #     <m~n~||e~f~>' t(af~,in~) l(ab,ij)  -> (be~,jm~)
    X += 0.5 * np.einsum("mnef,afin,abij->bejm", H1.bb.oovv, T.ab, L.aa, optimize=True)
    #     f'(ae) l(ab~,ij~)  -> (eb~,ij~)
    X += np.einsum("ae,abij->ebij", H1.a.vv, L.ab, optimize=True)
    #     <mn||ef>' t(af,mn) l(ab~,ij~)  -> (eb~,ij~)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H1.aa.oovv, T.aa, L.ab, optimize=True)
    #     <mn~|ef~>' t(eb~,ij~) l(ab~,ij~)  -> (af~,mn~)
    X -= np.einsum("mnef,ebij,abij->afmn", H1.ab.oovv, T.ab, L.ab, optimize=True)
    #     <mn~|ef~>' t(af~,mn~) l(ab~,ij~)  -> (eb~,ij~)
    X -= np.einsum("mnef,afmn,abij->ebij", H1.ab.oovv, T.ab, L.ab, optimize=True)
    #     f'(b~e~) l(ab~,ij~)  -> (ae~,ij~)
    X += np.einsum("be,abij->aeij", H1.b.vv, L.ab, optimize=True)
    #     <m~n~||e~f~>' t(b~f~,m~n~) l(ab~,ij~)  -> (ae~,ij~)
    X -= 0.5 * np.einsum("mnef,bfmn,abij->aeij", H1.bb.oovv, T.bb, L.ab, optimize=True)
    #     <nm~|fe~>' t(ae~,ij~) l(ab~,ij~)  -> (fb~,nm~)
    X -= np.einsum("nmfe,aeij,abij->fbnm", H1.ab.oovv, T.ab, L.ab, optimize=True)
    #     <nm~|fe~>' t(fb~,nm~) l(ab~,ij~)  -> (ae~,ij~)
    X -= np.einsum("nmfe,fbnm,abij->aeij", H1.ab.oovv, T.ab, L.ab, optimize=True)
    #     f'(mi) l(ab~,ij~)  -> (ab~,mj~)
    X -= np.einsum("mi,abij->abmj", H1.a.oo, L.ab, optimize=True)
    #     <mn||ef>' t(ef,in) l(ab~,ij~)  -> (ab~,mj~)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H1.aa.oovv, T.aa, L.ab, optimize=True)
    #     <mn~|ef~>' t(ab~,mj~) l(ab~,ij~)  -> (ef~,in~)
    X -= np.einsum("mnef,abmj,abij->efin", H1.ab.oovv, T.ab, L.ab, optimize=True)
    #     <mn~|ef~>' t(ef~,in~) l(ab~,ij~)  -> (ab~,mj~)
    X -= np.einsum("mnef,efin,abij->abmj", H1.ab.oovv, T.ab, L.ab, optimize=True)
    #     f'(m~j~) l(ab~,ij~)  -> (ab~,im~)
    X -= np.einsum("mj,abij->abim", H1.b.oo, L.ab, optimize=True)
    #     <m~n~||e~f~>' t(e~f~,j~n~) l(ab~,ij~)  -> (ab~,im~)
    X -= 0.5 * np.einsum("mnef,efjn,abij->abim", H1.bb.oovv, T.bb, L.ab, optimize=True)
    #     <nm~|fe~>' t(ab~,im~) l(ab~,ij~)  -> (fe~,nj~)
    X -= np.einsum("nmfe,abim,abij->fenj", H1.ab.oovv, T.ab, L.ab, optimize=True)
    #     <nm~|fe~>' t(fe~,nj~) l(ab~,ij~)  -> (ab~,im~)
    X -= np.einsum("nmfe,fenj,abij->abim", H1.ab.oovv, T.ab, L.ab, optimize=True)
    #     <am||ie>' l(ab~,ij~)  -> (eb~,mj~)
    X += np.einsum("amie,abij->ebmj", H1.aa.voov, L.ab, optimize=True)
    #     <mn||ef>' t(ae,im) l(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("mnef,aeim,abij->fbnj", H1.aa.oovv, T.aa, L.ab, optimize=True)
    #     <nm~|fe~>' t(fb~,nj~) l(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("nmfe,fbnj,abij->aeim", H1.ab.oovv, T.ab, L.ab, optimize=True)
    #     <nm~|fe~>' t(ae~,im~) l(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("nmfe,aeim,abij->fbnj", H1.ab.oovv, T.ab, L.ab, optimize=True)
    #     <m~n~||e~f~>' t(f~b~,n~j~) l(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("mnef,fbnj,abij->aeim", H1.bb.oovv, T.bb, L.ab, optimize=True)
    #     <b~m~||j~e~>' l(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("bmje,abij->aeim", H1.bb.voov, L.ab, optimize=True)
    #     <mb~|ie~>' l(ab~,ij~)  -> (ae~,mj~)
    X -= np.einsum("mbie,abij->aemj", H1.ab.ovov, L.ab, optimize=True)
    #     <am~|ej~>' l(ab~,ij~)  -> (eb~,im~)
    X -= np.einsum("amej,abij->ebim", H1.ab.vovo, L.ab, optimize=True)
    #     <mn~|ef~>' t(eb~,in~) l(ab~,ij~)  -> (af~,mj~)
    X += np.einsum("mnef,ebin,abij->afmj", H1.ab.oovv, T.ab, L.ab, optimize=True)
    #     <mn~|ef~>' t(af~,mj~) l(ab~,ij~)  -> (eb~,in~)
    X += np.einsum("mnef,afmj,abij->ebin", H1.ab.oovv, T.ab, L.ab, optimize=True)
    #     <mn~|ij~>' l(ab~,ij~)  -> (ab~,mn~)
    X += np.einsum("mnij,abij->abmn", H1.ab.oooo, L.ab, optimize=True)
    #     <mn~|ef~>' t(ab~,mn~) l(ab~,ij~)  -> (ef~,ij~)
    X += np.einsum("mnef,abmn,abij->efij", H1.ab.oovv, T.ab, L.ab, optimize=True)
    #     <mn~|ef~>' t(ef~,ij~) l(ab~,ij~)  -> (ab~,mn~)
    X += np.einsum("mnef,efij,abij->abmn", H1.ab.oovv, T.ab, L.ab, optimize=True)
    #     <nm~|fe~>' t(e~b~,i~j~) l(a~b~,i~j~)  -> (fa~,nm~)
    X -= 0.5 * np.einsum("nmfe,ebij,abij->fanm", H1.ab.oovv, T.bb, L.bb, optimize=True)
    #     <nm~|fe~>' t(a~b~,m~j~) l(a~b~,i~j~)  -> (fe~,ni~)
    X -= 0.5 * np.einsum("nmfe,abmj,abij->feni", H1.ab.oovv, T.bb, L.bb, optimize=True)
    #     <ma~|ei~>' l(a~b~,i~j~)  -> (eb~,mj~)
    X += np.einsum("maei,abij->ebmj", H1.ab.ovvo, L.bb, optimize=True)
    #     <mn~|ef~>' t(a~f~,i~n~) l(a~b~,i~j~)  -> (eb~,mj~)
    X += np.einsum("mnef,afin,abij->ebmj", H1.ab.oovv, T.bb, L.bb, optimize=True)
    #     <mn||ef>' t(eb~,mj~) l(a~b~,i~j~)  -> (fa~,ni~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->fani", H1.aa.oovv, T.ab, L.bb, optimize=True)
    #     <mn||ef>' t(fa~,ni~) l(a~b~,i~j~)  -> (eb~,mj~)
    X += 0.5 * np.einsum("mnef,fani,abij->ebmj", H1.aa.oovv, T.ab, L.bb, optimize=True)
    # vvvv of H1: sum_ab L(abij) <ab|ef>'
    for B1, B2 in ((H1.chol.a.vv, H.chol.b.vv), (H.chol.a.vv, H1.chol.b.vv)):
        X += cholesky_ladder(L.ab, B1, B2)
    # <0|C (Hbar - E0)|ij~ab~> (left-CCSD LH_fun)
    X += LHc
    return X


def build_k_2C(LHc, S, T, L, H, H1):
    """k(i~j~a~b~) = <0|(1 + Lambda)[[Hbar, tau(a~b~,i~j~)], Sigma]|0> + <0|C (Hbar - E0)|i~j~a~b~>, the
    bb doubles of the source of the pi equations (C_J = <0|Lambda Sigma1|J>, singles only).
    Sigma2 enters through the T2^2 terms of the CCSD equations, Sigma1 through H1 = [H(T1), S1]
    in the left-CCSD doubles terms. Terms are built in the stored layout and antisymmetrized
    before the (already antisymmetric) last term is added."""
    # <0|L2 [[(V_N T2^2)_C/2, tau(a~b~,i~j~)], S2]|0>: the T2^2 terms of the CCSD equations along S2
    #     <m~n~||e~f~> s(ae~,ij~) l(ab~,ij~)  -> (b~f~,m~n~)
    X = -0.5 * np.einsum("mnef,aeij,abij->bfmn", H.bb.oovv, S.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> s(ab~,im~) l(ab~,ij~)  -> (e~f~,j~n~)
    X -= 0.5 * np.einsum("mnef,abim,abij->efjn", H.bb.oovv, S.ab, L.ab, optimize=True)
    #     <mn~|ef~> s(ae,im) l(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.ab.oovv, S.aa, L.ab, optimize=True)
    #     <m~n~||e~f~> s(ae~,im~) l(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.bb.oovv, S.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> s(a~f~,m~n~) l(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.25 * np.einsum("mnef,afmn,abij->ebij", H.bb.oovv, S.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> s(e~b~,i~j~) l(a~b~,i~j~)  -> (a~f~,m~n~)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H.bb.oovv, S.bb, L.bb, optimize=True)
    #     <nm~|fe~> s(fa~,nm~) l(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.5 * np.einsum("nmfe,fanm,abij->ebij", H.ab.oovv, S.ab, L.bb, optimize=True)
    #     <m~n~||e~f~> s(e~f~,i~n~) l(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H.bb.oovv, S.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> s(a~b~,m~j~) l(a~b~,i~j~)  -> (e~f~,i~n~)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H.bb.oovv, S.bb, L.bb, optimize=True)
    #     <nm~|fe~> s(fe~,ni~) l(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.5 * np.einsum("nmfe,feni,abij->abmj", H.ab.oovv, S.ab, L.bb, optimize=True)
    #     <m~n~||e~f~> s(a~f~,i~n~) l(a~b~,i~j~)  -> (e~b~,m~j~)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H.bb.oovv, S.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> s(e~b~,m~j~) l(a~b~,i~j~)  -> (a~f~,i~n~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H.bb.oovv, S.bb, L.bb, optimize=True)
    #     <mn~|ef~> s(eb~,mj~) l(a~b~,i~j~)  -> (a~f~,i~n~)
    X += np.einsum("mnef,ebmj,abij->afin", H.ab.oovv, S.ab, L.bb, optimize=True)
    #     <m~n~||e~f~> s(e~f~,i~j~) l(a~b~,i~j~)  -> (a~b~,m~n~)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H.bb.oovv, S.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> s(a~b~,m~n~) l(a~b~,i~j~)  -> (e~f~,i~j~)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H.bb.oovv, S.bb, L.bb, optimize=True)
    # <0|(1 + L1 + L2)[[H(T1), S1], tau(a~b~,i~j~)]_C|0>: the left-CCSD doubles terms with H1 = [H(T1), S1]
    #     <i~j~||a~b~>'
    X += 0.25 * H1.bb.oovv.transpose(2, 3, 0, 1)
    #     f'(m~e~) l(a~,i~)  -> (a~e~,i~m~)
    X += np.einsum("me,ai->aeim", H1.b.ov, L.b, optimize=True)
    #     <m~n~||i~f~>' l(a~,i~)  -> (a~f~,m~n~)
    X -= 0.5 * np.einsum("mnif,ai->afmn", H1.bb.ooov, L.b, optimize=True)
    #     <a~n~||e~f~>' l(a~,i~)  -> (e~f~,i~n~)
    X += 0.5 * np.einsum("anef,ai->efin", H1.bb.vovv, L.b, optimize=True)
    #     <m~n~||e~f~>' t(ae~,ij~) l(ab~,ij~)  -> (b~f~,m~n~)
    X -= 0.5 * np.einsum("mnef,aeij,abij->bfmn", H1.bb.oovv, T.ab, L.ab, optimize=True)
    #     <m~n~||e~f~>' t(ab~,im~) l(ab~,ij~)  -> (e~f~,j~n~)
    X -= 0.5 * np.einsum("mnef,abim,abij->efjn", H1.bb.oovv, T.ab, L.ab, optimize=True)
    #     <am~|ie~>' l(ab~,ij~)  -> (e~b~,m~j~)
    X += np.einsum("amie,abij->ebmj", H1.ab.voov, L.ab, optimize=True)
    #     <mn~|ef~>' t(ae,im) l(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H1.ab.oovv, T.aa, L.ab, optimize=True)
    #     <m~n~||e~f~>' t(ae~,im~) l(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H1.bb.oovv, T.ab, L.ab, optimize=True)
    #     f'(a~e~) l(a~b~,i~j~)  -> (e~b~,i~j~)
    X += 0.5 * np.einsum("ae,abij->ebij", H1.b.vv, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(e~b~,i~j~) l(a~b~,i~j~)  -> (a~f~,m~n~)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H1.bb.oovv, T.bb, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(a~f~,m~n~) l(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.25 * np.einsum("mnef,afmn,abij->ebij", H1.bb.oovv, T.bb, L.bb, optimize=True)
    #     <nm~|fe~>' t(fa~,nm~) l(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.5 * np.einsum("nmfe,fanm,abij->ebij", H1.ab.oovv, T.ab, L.bb, optimize=True)
    #     f'(m~i~) l(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.5 * np.einsum("mi,abij->abmj", H1.b.oo, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(a~b~,m~j~) l(a~b~,i~j~)  -> (e~f~,i~n~)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H1.bb.oovv, T.bb, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(e~f~,i~n~) l(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H1.bb.oovv, T.bb, L.bb, optimize=True)
    #     <nm~|fe~>' t(fe~,ni~) l(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.5 * np.einsum("nmfe,feni,abij->abmj", H1.ab.oovv, T.ab, L.bb, optimize=True)
    #     <a~m~||i~e~>' l(a~b~,i~j~)  -> (e~b~,m~j~)
    X += np.einsum("amie,abij->ebmj", H1.bb.voov, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(e~b~,m~j~) l(a~b~,i~j~)  -> (a~f~,i~n~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H1.bb.oovv, T.bb, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(a~f~,i~n~) l(a~b~,i~j~)  -> (e~b~,m~j~)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H1.bb.oovv, T.bb, L.bb, optimize=True)
    #     <mn~|ef~>' t(eb~,mj~) l(a~b~,i~j~)  -> (a~f~,i~n~)
    X += np.einsum("mnef,ebmj,abij->afin", H1.ab.oovv, T.ab, L.bb, optimize=True)
    #     <m~n~||i~j~>' l(a~b~,i~j~)  -> (a~b~,m~n~)
    X += 0.125 * np.einsum("mnij,abij->abmn", H1.bb.oooo, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(a~b~,m~n~) l(a~b~,i~j~)  -> (e~f~,i~j~)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H1.bb.oovv, T.bb, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(e~f~,i~j~) l(a~b~,i~j~)  -> (a~b~,m~n~)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H1.bb.oovv, T.bb, L.bb, optimize=True)
    # vvvv of H1: 1/8 sum_ab L(abij) <ab||ef>'
    for B1, B2 in ((H1.chol.b.vv, H.chol.b.vv), (H.chol.b.vv, H1.chol.b.vv)):
        Y = cholesky_ladder(L.bb, B1, B2)
        X += 0.125 * (Y - Y.transpose(1, 0, 2, 3))
    X -= np.transpose(X, (1, 0, 2, 3))
    X -= np.transpose(X, (0, 1, 3, 2))
    # <0|C (Hbar - E0)|i~j~a~b~> (left-CCSD LH_fun)
    X += LHc
    return X


def solve_sigma(s_lambda, T, Hbar, system, tol):
    """sigma with sum_nu <mu|Hbar - E0|nu> sigma_nu = -s_lambda(mu) for mu in singles and doubles
    (EOMCCSD HR on the CCSD Hbar), by GMRES on the full antisymmetric arrays, preconditioned with
    the Hbar(oo, vv) diagonal."""
    if system.cholesky:
        from ccpy.eomcc.eomccsd_chol import HR
    else:
        from ccpy.eomcc.eomccsd import HR

    shapes = [T.a.shape, T.b.shape, T.aa.shape, T.ab.shape, T.bb.shape]
    offsets = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    X, HX = copy.deepcopy(T), copy.deepcopy(T)

    def matvec(x):
        X.a = np.asfortranarray(x[offsets[0]:offsets[1]].reshape(shapes[0]))
        X.b = np.asfortranarray(x[offsets[1]:offsets[2]].reshape(shapes[1]))
        X.aa = np.asfortranarray(x[offsets[2]:offsets[3]].reshape(shapes[2]))
        X.ab = np.asfortranarray(x[offsets[3]:offsets[4]].reshape(shapes[3]))
        X.bb = np.asfortranarray(x[offsets[4]:offsets[5]].reshape(shapes[4]))
        HR(HX, X, T, Hbar, False, system)
        return np.concatenate([np.ravel(HX.a), np.ravel(HX.b), np.ravel(HX.aa), np.ravel(HX.ab), np.ravel(HX.bb)])

    eoa, eva = np.diag(Hbar.a.oo), np.diag(Hbar.a.vv)
    eob, evb = np.diag(Hbar.b.oo), np.diag(Hbar.b.vv)
    d1a = eva[:, None] - eoa[None, :]
    d1b = evb[:, None] - eob[None, :]
    d2aa = eva[:, None, None, None] + eva[None, :, None, None] - eoa[None, None, :, None] - eoa[None, None, None, :]
    d2ab = eva[:, None, None, None] + evb[None, :, None, None] - eoa[None, None, :, None] - eob[None, None, None, :]
    d2bb = evb[:, None, None, None] + evb[None, :, None, None] - eob[None, None, :, None] - eob[None, None, None, :]
    diag = np.concatenate([d1a.ravel(), d1b.ravel(), d2aa.ravel(), d2ab.ravel(), d2bb.ravel()])
    n = offsets[-1]
    A = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
    M = spla.LinearOperator((n, n), matvec=lambda x: x / diag, dtype=float)
    rhs = -np.concatenate([np.ravel(s_lambda.a), np.ravel(s_lambda.b), np.ravel(s_lambda.aa), np.ravel(s_lambda.ab),
                           np.ravel(s_lambda.bb)])
    x, info = spla.gmres(A, rhs, M=M, rtol=tol, atol=0.0, restart=60, maxiter=200)
    if info != 0:
        raise RuntimeError(f"The CR-CC(2,3) sigma equations did not converge (gmres info = {info}).")
    return SimpleNamespace(a=x[offsets[0]:offsets[1]].reshape(shapes[0]), b=x[offsets[1]:offsets[2]].reshape(shapes[1]),
                           aa=x[offsets[2]:offsets[3]].reshape(shapes[2]), ab=x[offsets[3]:offsets[4]].reshape(shapes[3]),
                           bb=x[offsets[4]:offsets[5]].reshape(shapes[4]))


def solve_pi(s_t, k, T, Hbar, system, tol):
    """pi with sum_nu pi_nu <nu|Hbar - E0|mu> = -s_t(mu) - k(mu) for mu in singles and doubles
    (left-CCSD LH_fun on the CCSD Hbar), by GMRES on the full antisymmetric arrays,
    preconditioned with the Hbar(oo, vv) diagonal."""
    if system.cholesky:
        from ccpy.left.left_ccsd_chol import LH_fun
    else:
        from ccpy.left.left_ccsd import LH_fun

    shapes = [T.a.shape, T.b.shape, T.aa.shape, T.ab.shape, T.bb.shape]
    offsets = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    X, LX = copy.deepcopy(T), copy.deepcopy(T)

    def matvec(x):
        X.a = np.asfortranarray(x[offsets[0]:offsets[1]].reshape(shapes[0]))
        X.b = np.asfortranarray(x[offsets[1]:offsets[2]].reshape(shapes[1]))
        X.aa = np.asfortranarray(x[offsets[2]:offsets[3]].reshape(shapes[2]))
        X.ab = np.asfortranarray(x[offsets[3]:offsets[4]].reshape(shapes[3]))
        X.bb = np.asfortranarray(x[offsets[4]:offsets[5]].reshape(shapes[4]))
        LH_fun(LX, X, T, Hbar, False, system)
        return np.concatenate([np.ravel(LX.a), np.ravel(LX.b), np.ravel(LX.aa), np.ravel(LX.ab), np.ravel(LX.bb)])

    eoa, eva = np.diag(Hbar.a.oo), np.diag(Hbar.a.vv)
    eob, evb = np.diag(Hbar.b.oo), np.diag(Hbar.b.vv)
    d1a = eva[:, None] - eoa[None, :]
    d1b = evb[:, None] - eob[None, :]
    d2aa = eva[:, None, None, None] + eva[None, :, None, None] - eoa[None, None, :, None] - eoa[None, None, None, :]
    d2ab = eva[:, None, None, None] + evb[None, :, None, None] - eoa[None, None, :, None] - eob[None, None, None, :]
    d2bb = evb[:, None, None, None] + evb[None, :, None, None] - eob[None, None, :, None] - eob[None, None, None, :]
    diag = np.concatenate([d1a.ravel(), d1b.ravel(), d2aa.ravel(), d2ab.ravel(), d2bb.ravel()])
    n = offsets[-1]
    A = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
    M = spla.LinearOperator((n, n), matvec=lambda x: x / diag, dtype=float)
    rhs = -np.concatenate([np.ravel(s_t.a + k.a), np.ravel(s_t.b + k.b), np.ravel(s_t.aa + k.aa),
                           np.ravel(s_t.ab + k.ab), np.ravel(s_t.bb + k.bb)])
    x, info = spla.gmres(A, rhs, M=M, rtol=tol, atol=0.0, restart=60, maxiter=200)
    if info != 0:
        raise RuntimeError(f"The CR-CC(2,3) pi equations did not converge (gmres info = {info}).")
    return SimpleNamespace(a=x[offsets[0]:offsets[1]].reshape(shapes[0]), b=x[offsets[1]:offsets[2]].reshape(shapes[1]),
                           aa=x[offsets[2]:offsets[3]].reshape(shapes[2]), ab=x[offsets[3]:offsets[4]].reshape(shapes[3]),
                           bb=x[offsets[4]:offsets[5]].reshape(shapes[4]))


# ======================================================================================
# the triples: one pass over occupied triples (Fortran)
# ======================================================================================
def zero_triples_density(system):
    """rdm1, rdm2 with zero arrays for the blocks of H(T1) that the moments M_K (through the
    intermediates X_g) and the projections N_K contain: f(ov), and the vooo, ovoo, ooov, oovo,
    oovv, vvov, vvvo, vovv, ovvv blocks of <pq||rs> (aa, bb) and <pq|rs> (ab)."""
    noa, nob = system.noccupied_alpha, system.noccupied_beta
    nua, nub = system.nunoccupied_alpha, system.nunoccupied_beta
    rdm1 = Integral.from_empty(system, 1, use_none=True)
    rdm2 = Integral.from_empty(system, 2, use_none=True)
    rdm1.a.ov = np.zeros((noa, nua))
    rdm1.b.ov = np.zeros((nob, nub))
    rdm2.aa.vooo = np.zeros((nua, noa, noa, noa))
    rdm2.aa.ooov = np.zeros((noa, noa, noa, nua))
    rdm2.aa.oovv = np.zeros((noa, noa, nua, nua))
    rdm2.aa.vvov = np.zeros((nua, nua, noa, nua))
    rdm2.aa.vovv = np.zeros((nua, noa, nua, nua))
    rdm2.ab.vooo = np.zeros((nua, nob, noa, nob))
    rdm2.ab.ovoo = np.zeros((noa, nub, noa, nob))
    rdm2.ab.ooov = np.zeros((noa, nob, noa, nub))
    rdm2.ab.oovo = np.zeros((noa, nob, nua, nob))
    rdm2.ab.oovv = np.zeros((noa, nob, nua, nub))
    rdm2.ab.vvov = np.zeros((nua, nub, noa, nub))
    rdm2.ab.vvvo = np.zeros((nua, nub, nua, nob))
    rdm2.ab.vovv = np.zeros((nua, nob, nua, nub))
    rdm2.ab.ovvv = np.zeros((noa, nub, nua, nub))
    rdm2.bb.vooo = np.zeros((nub, nob, nob, nob))
    rdm2.bb.ooov = np.zeros((nob, nob, nob, nub))
    rdm2.bb.oovv = np.zeros((nob, nob, nub, nub))
    rdm2.bb.vvov = np.zeros((nub, nub, nob, nub))
    rdm2.bb.vovv = np.zeros((nub, nob, nub, nub))
    return rdm1, rdm2


def calc_triples(X, T, L, H, reference, rdm1, rdm2, omega=0.0, R=None, r0=0.0, J=None):
    """One pass of the Fortran triples loops (ccpy/lib/fortran/crcc23_gradient_loops.f90) over
    the four spin cases. For each occupied triple and all virtual triples K (abc, ijk):

        M_K = <K|(H(T1) e^T2)_C|0>   = A sum_g X_g . t_g                   (ground state)
        M_K = <K|(Hbar R)_C|0>        = A sum_g (J_g . t_g + X_g . r_g)     (excited state)
        N_K = <0|(L1 + L2) Hbar|K>   = <0|(L2 V_N)_C|K> + <0|(L1 V_N)_DC|K> + <0|(L2 F_N)_DC|K>
        D_K = f_ii + f_jj + f_kk - f_aa - f_bb - f_cc (+ omega),   u_K = M_K / D_K,  v_K = N_K / D_K

    accumulating delta = sum_K v_K M_K, W = sum_K u_K v_K, F_r0 = sum_K v_K <K|(H(T1) e^T2)_C|0>,
    d_p = -sum_K u_K v_K chi_Kp (d D_K/d f_pp = chi_Kp), and the raw cotangents (every stored
    array entry independent):
        dX_g = sum_K v_K dM_K/dX_g, dT = sum_K v_K dM_K/dt (trailing t_g), dL = sum_K u_K <mu|Hbar|K>,
        rdm += sum_K u_K dN_K/dH(T1) (added to rdm1, rdm2: zero_triples_density),
    and, for an excited state, dJT = sum_K v_K dM_K/dt (J_g . t_g), dXR = sum_K v_K dM_K/dX_g
    (X_g . r_g) and c'_X = <V3|R|X>. f is the bare Fock diagonal (reference.H), as in the energy.
    Ground state: J = X, R = T (unused); excited state (ccpy.gradients.creomcc23_gradient): R, r0,
    J_g = r0 X_g + X_g(T, H1) + X_g(R, H) (linear part), L the left EOMCCSD vector."""
    system = reference.system
    excited = R is not None
    if not excited:
        R, J = T, X
    noa, nob = system.noccupied_alpha, system.noccupied_beta
    nua, nub = system.nunoccupied_alpha, system.nunoccupied_beta
    eoa, eva = np.diag(reference.H.a.oo).copy(), np.diag(reference.H.a.vv).copy()
    eob, evb = np.diag(reference.H.b.oo).copy(), np.diag(reference.H.b.vv).copy()
    f = np.asfortranarray
    out = {"delta": 0.0, "W": 0.0, "F_r0": 0.0,
           "d": {"a": {"o": np.zeros(noa), "v": np.zeros(nua)}, "b": {"o": np.zeros(nob), "v": np.zeros(nub)}},
           "dX": {}, "dXR": {},
           "dT": {"aa": np.zeros_like(T.aa), "ab": np.zeros_like(T.ab), "bb": np.zeros_like(T.bb)},
           "dJT": {"aa": np.zeros_like(T.aa), "ab": np.zeros_like(T.ab), "bb": np.zeros_like(T.bb)},
           "dL": {"a": np.zeros_like(L.a), "b": np.zeros_like(L.b), "aa": np.zeros_like(L.aa),
                  "ab": np.zeros_like(L.ab), "bb": np.zeros_like(L.bb)},
           "cprime": {"a": np.zeros_like(R.a), "b": np.zeros_like(R.b), "aa": np.zeros_like(R.aa),
                      "ab": np.zeros_like(R.ab), "bb": np.zeros_like(R.bb)}}

    # ---- aaa: M = <ijkabc|...|0> = A(abc)A(ijk)[X1(amij) t(bc,mk) + X2(abie) t(ec,jk)],
    #      N = <0|(L1 + L2) Hbar|ijkabc>; passed as x1(a,m,i,j) . t2(b,c,m,k), x2(a,b,e,i) . t2(e,c,j,k)
    x1, x2 = X["aaa"]
    j1, j2 = J["aaa"]
    hvovv, hooov = f(H.aa.vovv.transpose(3, 2, 0, 1)), f(H.aa.ooov.transpose(3, 2, 0, 1))
    x2r, j2r = f(x2.transpose(0, 1, 3, 2)), f(j2.transpose(0, 1, 3, 2))
    dx1, dx2r, dt2 = np.zeros_like(x1, order="F"), np.zeros_like(x2r, order="F"), np.zeros_like(T.aa, order="F")
    dxr1, dxr2r, dtj2 = np.zeros_like(x1, order="F"), np.zeros_like(x2r, order="F"), np.zeros_like(T.aa, order="F")
    cp1, cp2 = np.zeros_like(R.a, order="F"), np.zeros_like(R.aa, order="F")
    dl1, dl2 = np.zeros_like(L.a, order="F"), np.zeros_like(L.aa, order="F")
    dhvovv, dhooov = np.zeros_like(hvovv, order="F"), np.zeros_like(hooov, order="F")
    dhoovv, dhov = np.zeros_like(H.aa.oovv, order="F"), np.zeros_like(H.a.ov, order="F")
    d_o, d_v = np.zeros(noa), np.zeros(nua)
    delta, w, fr0 = crcc23_gradient_loops.grad_aaa(
        f(x1), x2r, f(T.aa), f(j1), j2r, f(R.a), f(R.aa), hvovv, hooov, f(H.aa.oovv), f(H.a.ov), f(L.a), f(L.aa),
        eoa, eva, omega, int(excited), d_o, d_v, dx1, dx2r, dt2, dxr1, dxr2r, dtj2, cp1, cp2, dl1, dl2,
        dhvovv, dhooov, dhoovv, dhov, noa, nua)
    out["delta"] += delta
    out["W"] += w
    out["F_r0"] += fr0
    out["d"]["a"]["o"] += d_o
    out["d"]["a"]["v"] += d_v
    out["dX"]["aaa"] = [dx1, dx2r.transpose(0, 1, 3, 2)]
    out["dXR"]["aaa"] = [dxr1, dxr2r.transpose(0, 1, 3, 2)]
    out["dT"]["aa"] += dt2
    out["dJT"]["aa"] += dtj2
    out["cprime"]["a"] += cp1
    out["cprime"]["aa"] += cp2
    out["dL"]["a"] += dl1
    out["dL"]["aa"] += dl2
    rdm2.aa.vovv += dhvovv.transpose(2, 3, 1, 0)
    rdm2.aa.ooov += dhooov.transpose(2, 3, 1, 0)
    rdm2.aa.oovv += dhoovv
    rdm1.a.ov += dhov

    # ---- bbb: the aaa terms with all indices beta, N = <0|(L1 + L2) Hbar|i~j~k~a~b~c~>
    x1, x2 = X["bbb"]
    j1, j2 = J["bbb"]
    hvovv, hooov = f(H.bb.vovv.transpose(3, 2, 0, 1)), f(H.bb.ooov.transpose(3, 2, 0, 1))
    x2r, j2r = f(x2.transpose(0, 1, 3, 2)), f(j2.transpose(0, 1, 3, 2))
    dx1, dx2r, dt2 = np.zeros_like(x1, order="F"), np.zeros_like(x2r, order="F"), np.zeros_like(T.bb, order="F")
    dxr1, dxr2r, dtj2 = np.zeros_like(x1, order="F"), np.zeros_like(x2r, order="F"), np.zeros_like(T.bb, order="F")
    cp1, cp2 = np.zeros_like(R.b, order="F"), np.zeros_like(R.bb, order="F")
    dl1, dl2 = np.zeros_like(L.b, order="F"), np.zeros_like(L.bb, order="F")
    dhvovv, dhooov = np.zeros_like(hvovv, order="F"), np.zeros_like(hooov, order="F")
    dhoovv, dhov = np.zeros_like(H.bb.oovv, order="F"), np.zeros_like(H.b.ov, order="F")
    d_o, d_v = np.zeros(nob), np.zeros(nub)
    delta, w, fr0 = crcc23_gradient_loops.grad_aaa(
        f(x1), x2r, f(T.bb), f(j1), j2r, f(R.b), f(R.bb), hvovv, hooov, f(H.bb.oovv), f(H.b.ov), f(L.b), f(L.bb),
        eob, evb, omega, int(excited), d_o, d_v, dx1, dx2r, dt2, dxr1, dxr2r, dtj2, cp1, cp2, dl1, dl2,
        dhvovv, dhooov, dhoovv, dhov, nob, nub)
    out["delta"] += delta
    out["W"] += w
    out["F_r0"] += fr0
    out["d"]["b"]["o"] += d_o
    out["d"]["b"]["v"] += d_v
    out["dX"]["bbb"] = [dx1, dx2r.transpose(0, 1, 3, 2)]
    out["dXR"]["bbb"] = [dxr1, dxr2r.transpose(0, 1, 3, 2)]
    out["dT"]["bb"] += dt2
    out["dJT"]["bb"] += dtj2
    out["cprime"]["b"] += cp1
    out["cprime"]["bb"] += cp2
    out["dL"]["b"] += dl1
    out["dL"]["bb"] += dl2
    rdm2.bb.vovv += dhvovv.transpose(2, 3, 1, 0)
    rdm2.bb.ooov += dhooov.transpose(2, 3, 1, 0)
    rdm2.bb.oovv += dhoovv
    rdm1.b.ov += dhov

    # ---- aab: M = <ijk~abc~|...|0> = A(ab)A(ij) sum_g X_g . t_g, N = <0|(L1 + L2) Hbar|ijk~abc~>;
    #      passed as x1(b,c,e,k).t2a(a,e,i,j), x2(m,c,j,k).t2a(a,b,i,m), x3(a,c,e,i).t2b(b,e,j,k),
    #      x4(a,m,i,k).t2b(b,c,j,m), x5(a,b,e,i).t2b(e,c,j,k), x6(a,m,i,j).t2b(b,c,m,k), with the H
    #      blocks of N reordered as h1(e,b,c,k) = <ek~|bc~> ... (inverse transposes below)
    x1, x2, x3, x4, x5, x6 = X["aab"]
    j1, j2, j3, j4, j5, j6 = J["aab"]
    xs = [f(x1), f(x2), f(x3.transpose(0, 1, 3, 2)), f(x4), f(x5.transpose(0, 1, 3, 2)), f(x6)]
    js = [f(j1), f(j2), f(j3.transpose(0, 1, 3, 2)), f(j4), f(j5.transpose(0, 1, 3, 2)), f(j6)]
    ts = [f(T.aa), f(T.aa.transpose(0, 1, 3, 2)), f(T.ab), f(T.ab.transpose(0, 1, 3, 2))]
    rs = [f(R.aa), f(R.aa.transpose(0, 1, 3, 2)), f(R.ab), f(R.ab.transpose(0, 1, 3, 2))]
    hs = [f(H.ab.vovv.transpose(0, 2, 3, 1)), f(H.aa.vovv.transpose(3, 2, 0, 1)), f(H.ab.ovvv.transpose(2, 3, 1, 0)),
          f(H.ab.ooov.transpose(2, 3, 0, 1)), f(H.aa.ooov.transpose(3, 2, 0, 1)), f(H.ab.oovo.transpose(2, 3, 0, 1))]
    hd = [f(H.ab.oovv), f(H.aa.oovv), f(H.a.ov), f(H.b.ov)]
    ls = [f(L.a), f(L.b), f(L.aa), f(L.aa.transpose(0, 1, 3, 2)), f(L.ab), f(L.ab.transpose(0, 1, 3, 2))]
    d_ = [np.zeros(noa), np.zeros(nob), np.zeros(nua), np.zeros(nub)]
    dxs = [np.zeros_like(x, order="F") for x in xs]
    dts = [np.zeros_like(t, order="F") for t in ts]
    dxrs = [np.zeros_like(x, order="F") for x in xs]
    dtjs = [np.zeros_like(t, order="F") for t in ts]
    cps = [np.zeros_like(R.a, order="F"), np.zeros_like(R.b, order="F"), np.zeros_like(R.aa, order="F"),
           np.zeros_like(R.ab, order="F")]
    dhs = [np.zeros_like(x, order="F") for x in hs]
    dhd = [np.zeros_like(x, order="F") for x in hd]
    dls = [np.zeros_like(x, order="F") for x in ls]
    delta, w, fr0 = crcc23_gradient_loops.grad_aab(*xs, *ts, *js, f(R.a), f(R.b), *rs, *hs, *hd, *ls,
                                                   eoa, eob, eva, evb, omega, int(excited), *d_,
                                                   *dxs, *dts, *dxrs, *dtjs, *cps, *dhs, *dhd, *dls,
                                                   noa, nob, nua, nub)
    out["delta"] += delta
    out["W"] += w
    out["F_r0"] += fr0
    out["d"]["a"]["o"] += d_[0]
    out["d"]["b"]["o"] += d_[1]
    out["d"]["a"]["v"] += d_[2]
    out["d"]["b"]["v"] += d_[3]
    out["dX"]["aab"] = [dxs[0], dxs[1], dxs[2].transpose(0, 1, 3, 2), dxs[3], dxs[4].transpose(0, 1, 3, 2), dxs[5]]
    out["dXR"]["aab"] = [dxrs[0], dxrs[1], dxrs[2].transpose(0, 1, 3, 2), dxrs[3], dxrs[4].transpose(0, 1, 3, 2),
                         dxrs[5]]
    out["dT"]["aa"] += dts[0] + dts[1].transpose(0, 1, 3, 2)
    out["dT"]["ab"] += dts[2] + dts[3].transpose(0, 1, 3, 2)
    out["dJT"]["aa"] += dtjs[0] + dtjs[1].transpose(0, 1, 3, 2)
    out["dJT"]["ab"] += dtjs[2] + dtjs[3].transpose(0, 1, 3, 2)
    out["cprime"]["a"] += cps[0]
    out["cprime"]["b"] += cps[1]
    out["cprime"]["aa"] += cps[2]
    out["cprime"]["ab"] += cps[3]
    out["dL"]["a"] += dls[0]
    out["dL"]["b"] += dls[1]
    out["dL"]["aa"] += dls[2] + dls[3].transpose(0, 1, 3, 2)
    out["dL"]["ab"] += dls[4] + dls[5].transpose(0, 1, 3, 2)
    rdm2.ab.vovv += dhs[0].transpose(0, 3, 1, 2)
    rdm2.aa.vovv += dhs[1].transpose(2, 3, 1, 0)
    rdm2.ab.ovvv += dhs[2].transpose(3, 2, 0, 1)
    rdm2.ab.ooov += dhs[3].transpose(2, 3, 0, 1)
    rdm2.aa.ooov += dhs[4].transpose(2, 3, 1, 0)
    rdm2.ab.oovo += dhs[5].transpose(2, 3, 0, 1)
    rdm2.ab.oovv += dhd[0]
    rdm2.aa.oovv += dhd[1]
    rdm1.a.ov += dhd[2]
    rdm1.b.ov += dhd[3]

    # ---- abb: M = <ij~k~ab~c~|...|0> = A(bc)A(jk) sum_g X_g . t_g, N = <0|(L1 + L2) Hbar|ij~k~ab~c~>;
    #      passed as x1(c,b,e,k).t2c(a,e,i,j), x2(c,m,k,j).t2c(a,b,i,m), x3(c,a,e,i).t2b(e,b,k,j),
    #      x4(m,a,k,i).t2b(c,b,m,j), x5(a,b,e,i).t2b(c,e,k,j), x6(a,m,i,j).t2b(c,b,k,m)
    x1, x2, x3, x4, x5, x6 = X["abb"]
    j1, j2, j3, j4, j5, j6 = J["abb"]
    xs = [f(x1.transpose(0, 1, 3, 2)), f(x2), f(x3), f(x4), f(x5.transpose(0, 1, 3, 2)), f(x6)]
    js = [f(j1.transpose(0, 1, 3, 2)), f(j2), f(j3), f(j4), f(j5.transpose(0, 1, 3, 2)), f(j6)]
    ts = [f(T.bb), f(T.bb.transpose(0, 1, 3, 2)), f(T.ab), f(T.ab.transpose(0, 1, 3, 2))]
    rs = [f(R.bb), f(R.bb.transpose(0, 1, 3, 2)), f(R.ab), f(R.ab.transpose(0, 1, 3, 2))]
    hs = [f(H.ab.ovvv.transpose(2, 3, 1, 0)), f(H.bb.vovv.transpose(0, 2, 3, 1)), f(H.ab.vovv.transpose(2, 3, 0, 1)),
          f(H.ab.oovo.transpose(2, 3, 0, 1)), f(H.bb.ooov.transpose(3, 2, 0, 1)), f(H.ab.ooov.transpose(3, 2, 0, 1))]
    hd = [f(H.ab.oovv), f(H.bb.oovv), f(H.a.ov), f(H.b.ov)]
    ls = [f(L.a), f(L.b), f(L.bb), f(L.bb.transpose(0, 1, 3, 2)), f(L.ab), f(L.ab.transpose(0, 1, 3, 2))]
    d_ = [np.zeros(noa), np.zeros(nob), np.zeros(nua), np.zeros(nub)]
    dxs = [np.zeros_like(x, order="F") for x in xs]
    dts = [np.zeros_like(t, order="F") for t in ts]
    dxrs = [np.zeros_like(x, order="F") for x in xs]
    dtjs = [np.zeros_like(t, order="F") for t in ts]
    cps = [np.zeros_like(R.a, order="F"), np.zeros_like(R.b, order="F"), np.zeros_like(R.ab, order="F"),
           np.zeros_like(R.bb, order="F")]
    dhs = [np.zeros_like(x, order="F") for x in hs]
    dhd = [np.zeros_like(x, order="F") for x in hd]
    dls = [np.zeros_like(x, order="F") for x in ls]
    delta, w, fr0 = crcc23_gradient_loops.grad_abb(*xs, *ts, *js, f(R.a), f(R.b), *rs, *hs, *hd, *ls,
                                                   eoa, eob, eva, evb, omega, int(excited), *d_,
                                                   *dxs, *dts, *dxrs, *dtjs, *cps, *dhs, *dhd, *dls,
                                                   noa, nob, nua, nub)
    out["delta"] += delta
    out["W"] += w
    out["F_r0"] += fr0
    out["d"]["a"]["o"] += d_[0]
    out["d"]["b"]["o"] += d_[1]
    out["d"]["a"]["v"] += d_[2]
    out["d"]["b"]["v"] += d_[3]
    out["dX"]["abb"] = [dxs[0].transpose(0, 1, 3, 2), dxs[1], dxs[2], dxs[3], dxs[4].transpose(0, 1, 3, 2), dxs[5]]
    out["dXR"]["abb"] = [dxrs[0].transpose(0, 1, 3, 2), dxrs[1], dxrs[2], dxrs[3], dxrs[4].transpose(0, 1, 3, 2),
                         dxrs[5]]
    out["dT"]["bb"] += dts[0] + dts[1].transpose(0, 1, 3, 2)
    out["dT"]["ab"] += dts[2] + dts[3].transpose(0, 1, 3, 2)
    out["dJT"]["bb"] += dtjs[0] + dtjs[1].transpose(0, 1, 3, 2)
    out["dJT"]["ab"] += dtjs[2] + dtjs[3].transpose(0, 1, 3, 2)
    out["cprime"]["a"] += cps[0]
    out["cprime"]["b"] += cps[1]
    out["cprime"]["ab"] += cps[2]
    out["cprime"]["bb"] += cps[3]
    out["dL"]["a"] += dls[0]
    out["dL"]["b"] += dls[1]
    out["dL"]["bb"] += dls[2] + dls[3].transpose(0, 1, 3, 2)
    out["dL"]["ab"] += dls[4] + dls[5].transpose(0, 1, 3, 2)
    rdm2.ab.ovvv += dhs[0].transpose(3, 2, 0, 1)
    rdm2.bb.vovv += dhs[1].transpose(0, 3, 1, 2)
    rdm2.ab.vovv += dhs[2].transpose(2, 3, 0, 1)
    rdm2.ab.oovo += dhs[3].transpose(2, 3, 0, 1)
    rdm2.bb.ooov += dhs[4].transpose(2, 3, 1, 0)
    rdm2.ab.ooov += dhs[5].transpose(2, 3, 1, 0)
    rdm2.ab.oovv += dhd[0]
    rdm2.bb.oovv += dhd[1]
    rdm1.a.ov += dhd[2]
    rdm1.b.ov += dhd[3]
    return out


# ======================================================================================
# intermediates of the moments (the terms of crcc23_gradient_reference.build_M3A-D grouped by their
# last T2) and their chain rule
# ======================================================================================
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
        rdm2.aa.vooo -= 0.25 * dX1
    #     d/dH, d/dt of f(me) t(ae,ij)
    rdm1.a.ov -= 0.25 * np.einsum("amij,aeij->me", dX1, T.aa, optimize=True)
    dT["aa"] -= 0.25 * np.einsum("amij,me->aeij", dX1, H.a.ov, optimize=True)
    #     d/dH, d/dt of <am||ef> t(ef,ij)
    rdm2.aa.vovv -= 0.125 * np.einsum("amij,efij->amef", dX1, T.aa, optimize=True)
    dT["aa"] -= 0.125 * np.einsum("amij,amef->efij", dX1, H.aa.vovv, optimize=True)
    #     d/dH, d/dt of <mn||jf> t(af,in)
    rdm2.aa.ooov -= 0.5 * np.einsum("amij,afin->mnjf", dX1, T.aa, optimize=True)
    dT["aa"] -= 0.5 * np.einsum("amij,mnjf->afin", dX1, H.aa.ooov, optimize=True)
    #     d/dH, d/dt of <mn~|jf~> t(af~,in~)
    rdm2.ab.ooov -= 0.5 * np.einsum("amij,afin->mnjf", dX1, T.ab, optimize=True)
    dT["ab"] -= 0.5 * np.einsum("amij,mnjf->afin", dX1, H.ab.ooov, optimize=True)
    # X2(abie), times t(ec,jk) in <ijkabc|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <ab||ie>
        rdm2.aa.vvov += 0.25 * dX2
    #     d/dH, d/dt of <mn||ie> t(ab,mn)
    rdm2.aa.ooov += 0.125 * np.einsum("abie,abmn->mnie", dX2, T.aa, optimize=True)
    dT["aa"] += 0.125 * np.einsum("abie,mnie->abmn", dX2, H.aa.ooov, optimize=True)
    #     d/dH, d/dt of <bn||ef> t(af,in)
    rdm2.aa.vovv += 0.5 * np.einsum("abie,afin->bnef", dX2, T.aa, optimize=True)
    dT["aa"] += 0.5 * np.einsum("abie,bnef->afin", dX2, H.aa.vovv, optimize=True)
    #     d/dH, d/dt of <bn~|ef~> t(af~,in~)
    rdm2.ab.vovv += 0.5 * np.einsum("abie,afin->bnef", dX2, T.ab, optimize=True)
    dT["ab"] += 0.5 * np.einsum("abie,bnef->afin", dX2, H.ab.vovv, optimize=True)


def backprop_triples_intermediates_aab(dX, T, H, dT, rdm1, rdm2, constant=True):
    """Chain rule of build_triples_intermediates_aab for the cotangents dX_g of the X_g
    (from the <ijk~abc~|...|0> moments): for every term c H(T1) T2 of X_g, the raw T2 gradient
    dT += c dX_g . H and the H(T1) density rdm += c dX_g . T2 (rdm += c dX_g for the bare blocks)."""
    dX1, dX2, dX3, dX4, dX5, dX6 = dX
    # X1(bcek), times t(ae,ij) in <ijk~abc~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <bc~|ek~>
        rdm2.ab.vvvo += 0.5 * dX1
    #     d/dH, d/dt of f(me) t(bc~,mk~)
    rdm1.a.ov -= 0.5 * np.einsum("bcek,bcmk->me", dX1, T.ab, optimize=True)
    dT["ab"] -= 0.5 * np.einsum("bcek,me->bcmk", dX1, H.a.ov, optimize=True)
    #     d/dH, d/dt of <nm~|ek~> t(bc~,nm~)
    rdm2.ab.oovo += 0.5 * np.einsum("bcek,bcnm->nmek", dX1, T.ab, optimize=True)
    dT["ab"] += 0.5 * np.einsum("bcek,nmek->bcnm", dX1, H.ab.oovo, optimize=True)
    #     d/dH, d/dt of <bm~|ef~> t(c~f~,k~m~)
    rdm2.ab.vovv += 0.5 * np.einsum("bcek,cfkm->bmef", dX1, T.bb, optimize=True)
    dT["bb"] += 0.5 * np.einsum("bcek,bmef->cfkm", dX1, H.ab.vovv, optimize=True)
    #     d/dH, d/dt of <bm||ef> t(fc~,mk~)
    rdm2.aa.vovv += 0.5 * np.einsum("bcek,fcmk->bmef", dX1, T.ab, optimize=True)
    dT["ab"] += 0.5 * np.einsum("bcek,bmef->fcmk", dX1, H.aa.vovv, optimize=True)
    #     d/dH, d/dt of <mc~|ef~> t(bf~,mk~)
    rdm2.ab.ovvv -= 0.5 * np.einsum("bcek,bfmk->mcef", dX1, T.ab, optimize=True)
    dT["ab"] -= 0.5 * np.einsum("bcek,mcef->bfmk", dX1, H.ab.ovvv, optimize=True)
    # X2(mcjk), times t(ab,im) in <ijk~abc~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <mc~|jk~>
        rdm2.ab.ovoo -= 0.5 * dX2
    #     d/dH, d/dt of <mn||jf> t(fc~,nk~)
    rdm2.aa.ooov -= 0.5 * np.einsum("mcjk,fcnk->mnjf", dX2, T.ab, optimize=True)
    dT["ab"] -= 0.5 * np.einsum("mcjk,mnjf->fcnk", dX2, H.aa.ooov, optimize=True)
    #     d/dH, d/dt of <mn~|jf~> t(f~c~,n~k~)
    rdm2.ab.ooov -= 0.5 * np.einsum("mcjk,fcnk->mnjf", dX2, T.bb, optimize=True)
    dT["bb"] -= 0.5 * np.einsum("mcjk,mnjf->fcnk", dX2, H.ab.ooov, optimize=True)
    #     d/dH, d/dt of <mn~|fk~> t(fc~,jn~)
    rdm2.ab.oovo += 0.5 * np.einsum("mcjk,fcjn->mnfk", dX2, T.ab, optimize=True)
    dT["ab"] += 0.5 * np.einsum("mcjk,mnfk->fcjn", dX2, H.ab.oovo, optimize=True)
    #     d/dH, d/dt of <mc~|fe~> t(fe~,jk~)
    rdm2.ab.ovvv -= 0.5 * np.einsum("mcjk,fejk->mcfe", dX2, T.ab, optimize=True)
    dT["ab"] -= 0.5 * np.einsum("mcjk,mcfe->fejk", dX2, H.ab.ovvv, optimize=True)
    # X3(acie), times t(be~,jk~) in <ijk~abc~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <ac~|ie~>
        rdm2.ab.vvov += dX3
    #     d/dH, d/dt of f(m~e~) t(ac~,im~)
    rdm1.b.ov -= 1 * np.einsum("acie,acim->me", dX3, T.ab, optimize=True)
    dT["ab"] -= np.einsum("acie,me->acim", dX3, H.b.ov, optimize=True)
    #     d/dH, d/dt of <nm~|ie~> t(ac~,nm~)
    rdm2.ab.ooov += 1 * np.einsum("acie,acnm->nmie", dX3, T.ab, optimize=True)
    dT["ab"] += np.einsum("acie,nmie->acnm", dX3, H.ab.ooov, optimize=True)
    #     d/dH, d/dt of <mc~|fe~> t(af,im)
    rdm2.ab.ovvv += 1 * np.einsum("acie,afim->mcfe", dX3, T.aa, optimize=True)
    dT["aa"] += np.einsum("acie,mcfe->afim", dX3, H.ab.ovvv, optimize=True)
    #     d/dH, d/dt of <c~m~||e~f~> t(af~,im~)
    rdm2.bb.vovv += 1 * np.einsum("acie,afim->cmef", dX3, T.ab, optimize=True)
    dT["ab"] += np.einsum("acie,cmef->afim", dX3, H.bb.vovv, optimize=True)
    #     d/dH, d/dt of <am~|fe~> t(fc~,im~)
    rdm2.ab.vovv -= 1 * np.einsum("acie,fcim->amfe", dX3, T.ab, optimize=True)
    dT["ab"] -= np.einsum("acie,amfe->fcim", dX3, H.ab.vovv, optimize=True)
    # X4(amik), times t(bc~,jm~) in <ijk~abc~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <am~|ik~>
        rdm2.ab.vooo -= dX4
    #     d/dH, d/dt of <m~n~||k~f~> t(af~,in~)
    rdm2.bb.ooov -= 1 * np.einsum("amik,afin->mnkf", dX4, T.ab, optimize=True)
    dT["ab"] -= np.einsum("amik,mnkf->afin", dX4, H.bb.ooov, optimize=True)
    #     d/dH, d/dt of <nm~|fk~> t(af,in)
    rdm2.ab.oovo -= 1 * np.einsum("amik,afin->nmfk", dX4, T.aa, optimize=True)
    dT["aa"] -= np.einsum("amik,nmfk->afin", dX4, H.ab.oovo, optimize=True)
    #     d/dH, d/dt of <nm~|if~> t(af~,nk~)
    rdm2.ab.ooov += 1 * np.einsum("amik,afnk->nmif", dX4, T.ab, optimize=True)
    dT["ab"] += np.einsum("amik,nmif->afnk", dX4, H.ab.ooov, optimize=True)
    #     d/dH, d/dt of <am~|ef~> t(ef~,ik~)
    rdm2.ab.vovv -= 1 * np.einsum("amik,efik->amef", dX4, T.ab, optimize=True)
    dT["ab"] -= np.einsum("amik,amef->efik", dX4, H.ab.vovv, optimize=True)
    # X5(abie), times t(ec~,jk~) in <ijk~abc~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <ab||ie>
        rdm2.aa.vvov += 0.5 * dX5
    #     d/dH, d/dt of f(me) t(ab,im)
    rdm1.a.ov -= 0.5 * np.einsum("abie,abim->me", dX5, T.aa, optimize=True)
    dT["aa"] -= 0.5 * np.einsum("abie,me->abim", dX5, H.a.ov, optimize=True)
    #     d/dH, d/dt of <mn||ie> t(ab,mn)
    rdm2.aa.ooov += 0.25 * np.einsum("abie,abmn->mnie", dX5, T.aa, optimize=True)
    dT["aa"] += 0.25 * np.einsum("abie,mnie->abmn", dX5, H.aa.ooov, optimize=True)
    #     d/dH, d/dt of <bn||ef> t(af,in)
    rdm2.aa.vovv += 1 * np.einsum("abie,afin->bnef", dX5, T.aa, optimize=True)
    dT["aa"] += np.einsum("abie,bnef->afin", dX5, H.aa.vovv, optimize=True)
    #     d/dH, d/dt of <bn~|ef~> t(af~,in~)
    rdm2.ab.vovv += 1 * np.einsum("abie,afin->bnef", dX5, T.ab, optimize=True)
    dT["ab"] += np.einsum("abie,bnef->afin", dX5, H.ab.vovv, optimize=True)
    # X6(amij), times t(bc~,mk~) in <ijk~abc~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <am||ij>
        rdm2.aa.vooo -= 0.5 * dX6
    #     d/dH, d/dt of <am||ef> t(ef,ij)
    rdm2.aa.vovv -= 0.25 * np.einsum("amij,efij->amef", dX6, T.aa, optimize=True)
    dT["aa"] -= 0.25 * np.einsum("amij,amef->efij", dX6, H.aa.vovv, optimize=True)
    #     d/dH, d/dt of <mn||jf> t(af,in)
    rdm2.aa.ooov -= 1 * np.einsum("amij,afin->mnjf", dX6, T.aa, optimize=True)
    dT["aa"] -= np.einsum("amij,mnjf->afin", dX6, H.aa.ooov, optimize=True)
    #     d/dH, d/dt of <mn~|jf~> t(af~,in~)
    rdm2.ab.ooov -= 1 * np.einsum("amij,afin->mnjf", dX6, T.ab, optimize=True)
    dT["ab"] -= np.einsum("amij,mnjf->afin", dX6, H.ab.ooov, optimize=True)


def backprop_triples_intermediates_abb(dX, T, H, dT, rdm1, rdm2, constant=True):
    """Chain rule of build_triples_intermediates_abb for the cotangents dX_g of the X_g
    (from the <ij~k~ab~c~|...|0> moments): for every term c H(T1) T2 of X_g, the raw T2 gradient
    dT += c dX_g . H and the H(T1) density rdm += c dX_g . T2 (rdm += c dX_g for the bare blocks)."""
    dX1, dX2, dX3, dX4, dX5, dX6 = dX
    # X1(cbke), times t(a~e~,i~j~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <cb~|ke~>
        rdm2.ab.vvov += 0.5 * dX1
    #     d/dH, d/dt of f(m~e~) t(cb~,km~)
    rdm1.b.ov -= 0.5 * np.einsum("cbke,cbkm->me", dX1, T.ab, optimize=True)
    dT["ab"] -= 0.5 * np.einsum("cbke,me->cbkm", dX1, H.b.ov, optimize=True)
    #     d/dH, d/dt of <mn~|ke~> t(cb~,mn~)
    rdm2.ab.ooov += 0.5 * np.einsum("cbke,cbmn->mnke", dX1, T.ab, optimize=True)
    dT["ab"] += 0.5 * np.einsum("cbke,mnke->cbmn", dX1, H.ab.ooov, optimize=True)
    #     d/dH, d/dt of <mb~|fe~> t(cf,km)
    rdm2.ab.ovvv += 0.5 * np.einsum("cbke,cfkm->mbfe", dX1, T.aa, optimize=True)
    dT["aa"] += 0.5 * np.einsum("cbke,mbfe->cfkm", dX1, H.ab.ovvv, optimize=True)
    #     d/dH, d/dt of <b~m~||e~f~> t(cf~,km~)
    rdm2.bb.vovv += 0.5 * np.einsum("cbke,cfkm->bmef", dX1, T.ab, optimize=True)
    dT["ab"] += 0.5 * np.einsum("cbke,bmef->cfkm", dX1, H.bb.vovv, optimize=True)
    #     d/dH, d/dt of <cm~|fe~> t(fb~,km~)
    rdm2.ab.vovv -= 0.5 * np.einsum("cbke,fbkm->cmfe", dX1, T.ab, optimize=True)
    dT["ab"] -= 0.5 * np.einsum("cbke,cmfe->fbkm", dX1, H.ab.vovv, optimize=True)
    # X2(cmkj), times t(a~b~,i~m~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <cm~|kj~>
        rdm2.ab.vooo -= 0.5 * dX2
    #     d/dH, d/dt of <m~n~||j~f~> t(cf~,kn~)
    rdm2.bb.ooov -= 0.5 * np.einsum("cmkj,cfkn->mnjf", dX2, T.ab, optimize=True)
    dT["ab"] -= 0.5 * np.einsum("cmkj,mnjf->cfkn", dX2, H.bb.ooov, optimize=True)
    #     d/dH, d/dt of <nm~|fj~> t(fc,nk)
    rdm2.ab.oovo -= 0.5 * np.einsum("cmkj,fcnk->nmfj", dX2, T.aa, optimize=True)
    dT["aa"] -= 0.5 * np.einsum("cmkj,nmfj->fcnk", dX2, H.ab.oovo, optimize=True)
    #     d/dH, d/dt of <nm~|kf~> t(cf~,nj~)
    rdm2.ab.ooov += 0.5 * np.einsum("cmkj,cfnj->nmkf", dX2, T.ab, optimize=True)
    dT["ab"] += 0.5 * np.einsum("cmkj,nmkf->cfnj", dX2, H.ab.ooov, optimize=True)
    #     d/dH, d/dt of <cm~|ef~> t(ef~,kj~)
    rdm2.ab.vovv -= 0.5 * np.einsum("cmkj,efkj->cmef", dX2, T.ab, optimize=True)
    dT["ab"] -= 0.5 * np.einsum("cmkj,cmef->efkj", dX2, H.ab.vovv, optimize=True)
    # X3(caei), times t(eb~,kj~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <ca~|ei~>
        rdm2.ab.vvvo += dX3
    #     d/dH, d/dt of f(me) t(ca~,mi~)
    rdm1.a.ov -= 1 * np.einsum("caei,cami->me", dX3, T.ab, optimize=True)
    dT["ab"] -= np.einsum("caei,me->cami", dX3, H.a.ov, optimize=True)
    #     d/dH, d/dt of <mn~|ei~> t(ca~,mn~)
    rdm2.ab.oovo += 1 * np.einsum("caei,camn->mnei", dX3, T.ab, optimize=True)
    dT["ab"] += np.einsum("caei,mnei->camn", dX3, H.ab.oovo, optimize=True)
    #     d/dH, d/dt of <cm~|ef~> t(a~f~,i~m~)
    rdm2.ab.vovv += 1 * np.einsum("caei,afim->cmef", dX3, T.bb, optimize=True)
    dT["bb"] += np.einsum("caei,cmef->afim", dX3, H.ab.vovv, optimize=True)
    #     d/dH, d/dt of <cm||ef> t(fa~,mi~)
    rdm2.aa.vovv += 1 * np.einsum("caei,fami->cmef", dX3, T.ab, optimize=True)
    dT["ab"] += np.einsum("caei,cmef->fami", dX3, H.aa.vovv, optimize=True)
    #     d/dH, d/dt of <ma~|ef~> t(cf~,mi~)
    rdm2.ab.ovvv -= 1 * np.einsum("caei,cfmi->maef", dX3, T.ab, optimize=True)
    dT["ab"] -= np.einsum("caei,maef->cfmi", dX3, H.ab.ovvv, optimize=True)
    # X4(maki), times t(cb~,mj~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <ma~|ki~>
        rdm2.ab.ovoo -= dX4
    #     d/dH, d/dt of <mn||kf> t(fa~,ni~)
    rdm2.aa.ooov -= 1 * np.einsum("maki,fani->mnkf", dX4, T.ab, optimize=True)
    dT["ab"] -= np.einsum("maki,mnkf->fani", dX4, H.aa.ooov, optimize=True)
    #     d/dH, d/dt of <mn~|kf~> t(a~f~,i~n~)
    rdm2.ab.ooov -= 1 * np.einsum("maki,afin->mnkf", dX4, T.bb, optimize=True)
    dT["bb"] -= np.einsum("maki,mnkf->afin", dX4, H.ab.ooov, optimize=True)
    #     d/dH, d/dt of <mn~|fi~> t(fa~,kn~)
    rdm2.ab.oovo += 1 * np.einsum("maki,fakn->mnfi", dX4, T.ab, optimize=True)
    dT["ab"] += np.einsum("maki,mnfi->fakn", dX4, H.ab.oovo, optimize=True)
    #     d/dH, d/dt of <ma~|fe~> t(fe~,ki~)
    rdm2.ab.ovvv -= 1 * np.einsum("maki,feki->mafe", dX4, T.ab, optimize=True)
    dT["ab"] -= np.einsum("maki,mafe->feki", dX4, H.ab.ovvv, optimize=True)
    # X5(abie), times t(ce~,kj~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <a~b~||i~e~>
        rdm2.bb.vvov += 0.5 * dX5
    #     d/dH, d/dt of f(m~e~) t(a~b~,i~m~)
    rdm1.b.ov -= 0.5 * np.einsum("abie,abim->me", dX5, T.bb, optimize=True)
    dT["bb"] -= 0.5 * np.einsum("abie,me->abim", dX5, H.b.ov, optimize=True)
    #     d/dH, d/dt of <m~n~||i~e~> t(a~b~,m~n~)
    rdm2.bb.ooov += 0.25 * np.einsum("abie,abmn->mnie", dX5, T.bb, optimize=True)
    dT["bb"] += 0.25 * np.einsum("abie,mnie->abmn", dX5, H.bb.ooov, optimize=True)
    #     d/dH, d/dt of <b~n~||e~f~> t(a~f~,i~n~)
    rdm2.bb.vovv += 1 * np.einsum("abie,afin->bnef", dX5, T.bb, optimize=True)
    dT["bb"] += np.einsum("abie,bnef->afin", dX5, H.bb.vovv, optimize=True)
    #     d/dH, d/dt of <nb~|fe~> t(fa~,ni~)
    rdm2.ab.ovvv += 1 * np.einsum("abie,fani->nbfe", dX5, T.ab, optimize=True)
    dT["ab"] += np.einsum("abie,nbfe->fani", dX5, H.ab.ovvv, optimize=True)
    # X6(amij), times t(cb~,km~) in <ij~k~ab~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <a~m~||i~j~>
        rdm2.bb.vooo -= 0.5 * dX6
    #     d/dH, d/dt of <a~m~||e~f~> t(e~f~,i~j~)
    rdm2.bb.vovv -= 0.25 * np.einsum("amij,efij->amef", dX6, T.bb, optimize=True)
    dT["bb"] -= 0.25 * np.einsum("amij,amef->efij", dX6, H.bb.vovv, optimize=True)
    #     d/dH, d/dt of <m~n~||j~f~> t(a~f~,i~n~)
    rdm2.bb.ooov -= 1 * np.einsum("amij,afin->mnjf", dX6, T.bb, optimize=True)
    dT["bb"] -= np.einsum("amij,mnjf->afin", dX6, H.bb.ooov, optimize=True)
    #     d/dH, d/dt of <nm~|fj~> t(fa~,ni~)
    rdm2.ab.oovo -= 1 * np.einsum("amij,fani->nmfj", dX6, T.ab, optimize=True)
    dT["ab"] -= np.einsum("amij,nmfj->fani", dX6, H.ab.oovo, optimize=True)


def backprop_triples_intermediates_bbb(dX, T, H, dT, rdm1, rdm2, constant=True):
    """Chain rule of build_triples_intermediates_bbb for the cotangents dX_g of the X_g
    (from the <i~j~k~a~b~c~|...|0> moments): for every term c H(T1) T2 of X_g, the raw T2 gradient
    dT += c dX_g . H and the H(T1) density rdm += c dX_g . T2 (rdm += c dX_g for the bare blocks)."""
    dX1, dX2 = dX
    # X1(amij), times t(b~c~,m~k~) in <i~j~k~a~b~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <a~m~||i~j~>
        rdm2.bb.vooo -= 0.25 * dX1
    #     d/dH, d/dt of f(m~e~) t(a~e~,i~j~)
    rdm1.b.ov -= 0.25 * np.einsum("amij,aeij->me", dX1, T.bb, optimize=True)
    dT["bb"] -= 0.25 * np.einsum("amij,me->aeij", dX1, H.b.ov, optimize=True)
    #     d/dH, d/dt of <a~m~||e~f~> t(e~f~,i~j~)
    rdm2.bb.vovv -= 0.125 * np.einsum("amij,efij->amef", dX1, T.bb, optimize=True)
    dT["bb"] -= 0.125 * np.einsum("amij,amef->efij", dX1, H.bb.vovv, optimize=True)
    #     d/dH, d/dt of <m~n~||j~f~> t(a~f~,i~n~)
    rdm2.bb.ooov -= 0.5 * np.einsum("amij,afin->mnjf", dX1, T.bb, optimize=True)
    dT["bb"] -= 0.5 * np.einsum("amij,mnjf->afin", dX1, H.bb.ooov, optimize=True)
    #     d/dH, d/dt of <nm~|fj~> t(fa~,ni~)
    rdm2.ab.oovo -= 0.5 * np.einsum("amij,fani->nmfj", dX1, T.ab, optimize=True)
    dT["ab"] -= 0.5 * np.einsum("amij,nmfj->fani", dX1, H.ab.oovo, optimize=True)
    # X2(abie), times t(e~c~,j~k~) in <i~j~k~a~b~c~|(H_N e^T2)_C|0>
    if constant:
        #     (V_N T2)_C: <a~b~||i~e~>
        rdm2.bb.vvov += 0.25 * dX2
    #     d/dH, d/dt of <m~n~||i~e~> t(a~b~,m~n~)
    rdm2.bb.ooov += 0.125 * np.einsum("abie,abmn->mnie", dX2, T.bb, optimize=True)
    dT["bb"] += 0.125 * np.einsum("abie,mnie->abmn", dX2, H.bb.ooov, optimize=True)
    #     d/dH, d/dt of <b~n~||e~f~> t(a~f~,i~n~)
    rdm2.bb.vovv += 0.5 * np.einsum("abie,afin->bnef", dX2, T.bb, optimize=True)
    dT["bb"] += 0.5 * np.einsum("abie,bnef->afin", dX2, H.bb.vovv, optimize=True)
    #     d/dH, d/dt of <nb~|fe~> t(fa~,ni~)
    rdm2.ab.ovvv += 0.5 * np.einsum("abie,fani->nbfe", dX2, T.ab, optimize=True)
    dT["ab"] += 0.5 * np.einsum("abie,nbfe->fani", dX2, H.ab.ovvv, optimize=True)


def cholesky_ladder(L2, B1, B2, max_elements=2 ** 26):
    """sum_x sum_ab L2[a,b,i,j] B1[x,a,e] B2[x,b,f] -> [e,f,i,j] (a vvvv contraction through
    Cholesky vectors), in chunks of the auxiliary index so that the x-resolved intermediate
    never exceeds max_elements."""
    nx = B1.shape[0]
    chunk = max(1, max_elements // max(1, L2.size // L2.shape[0] * B1.shape[2]))
    out = np.zeros((B1.shape[2], B2.shape[2]) + L2.shape[2:])
    for x0 in range(0, nx, chunk):
        LB = np.einsum("abij,xae->xebij", L2, B1[x0:x0 + chunk], optimize=True)
        out += np.einsum("xebij,xbf->efij", LB, B2[x0:x0 + chunk], optimize=True)
    return out
