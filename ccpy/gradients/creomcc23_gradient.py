"""Analytic nuclear gradient of an excited state's delta-CR-EOMCC(2,3)_A energy, factorized with
density fitting or Cholesky vectors, RHF or UHF reference (ccpq_analytical_gradients.pdf,
Secs. 7-8).

The CR-EOMCC(2,3) moment <K|(Hbar R)_C|0> contains r0 times the ground-state moment
<K|(H(T1) e^T2)_C|0>, which makes the correction non-size-intensive. delta-CR-EOMCC(2,3) drops it,
and so does this gradient: the r0 terms are commented out where they occur and marked
NON-INTENSIVE TERM (the r0 X_g part of J_g, F_r0, r0 <V3|O|0> in the densities and r0 dT0 in F_t).
The gradient is therefore that of E_EOMCCSD + ddelta (the "ddeltas" of ccpy.methods.CREOMCC23).

Theory
------
The state is an EOMCCSD root with right vector R = r0 + R1 + R2, left vector L = L1 + L2
(<L|R> = 1) and energy E = E0 + omega; Q = all triples K = (abc, ijk) and

    delta = sum_K N_K M_K / D_K,   M_K = <K|(Hbar R)_C|0>,   N_K = <L|Hbar|K>,
    D_K   = omega + f_ii + f_jj + f_kk - f_aa - f_bb - f_cc      (bare Fock, variant A).

R is an excitation operator like T, so M = r0 M3 + D_R M3 for CR-EOMCC(2,3) and M = D_R M3 for
delta-CR-EOMCC(2,3) (used here): with the moment intermediates X_g of
ccpy.gradients.crcc23_gradient, M = A sum_g (J_g . t_g + X_g . r_g), J_g = X_g(T, H1) + X_g(R, H)
(linear part; + r0 X_g for CR-EOMCC(2,3)), H1 = [H(T1), R1]. The gradient of E_EOMCCSD + delta is that of

    E = <L|Hbar|R> + sum_mu z_b,mu <mu|Hbar|0>                                     (EOMCCSD)
        + delta + a^T (A - E) r + l^T (A - E) b + nu (l^T r - 1) + sum_mu z_c,mu <mu|Hbar|0>,

with A = Hbar - E0 on <0| + SD (the EOMCCSD eigenvalue problem), a = a0 <0| + A the multiplier
of its right and b = b0 |0> + b1 + b2 of its left equations. With u_K = M_K / D_K,
v_K = N_K / D_K and W = sum_K u_K v_K, stationarity gives, in the order of Sec. 7.2,

    sum_nu z_b,nu <nu|Hbar - E0|mu> = -q(mu),            q(mu) = <L|[Hbar, tau_mu]|R>;
    (A - E) b = -F_l + delta r,   (A - E)^T a = -F_r + delta l,   a.r + b.l = -W   (nu = -delta);
    sum_nu z_c,nu <nu|Hbar - E0|mu> = -g(mu),   g(mu) = F_t(mu) + <a|[Hbar, tau_mu]|R> + <L|[Hbar, tau_mu]|b>,

with the sources of delta at fixed state

    F_r0 = sum_K v_K <K|(H(T1) e^T2)_C|0> (0 here),    F_r(mu) = sum_K v_K d<K|(H(T1) e^T2)_C|0>/dt_mu,
    F_l(mu) = sum_K <mu|Hbar|K> u_K,
    F_t(mu) = sum_K v_K <K|[Hbar, tau_mu] R|0> + sum_K u_K <L|[Hbar, tau_mu]|K> + W <0|Hbar|mu>.

Off the CC solution <K|Hbar R|0> = r0 M_K + D_R M_K + sum_X c'_X m_X with c'_X = <V3|R|X>, so
the doubles of F_t also contain (J^T c')_doubles. Hbar depends on T1 only through H(T1),
d H(T1)/dt(ai) = [H(T1), a+ i], so every singles derivative is a contraction of bare-frame
densities (D, Y) with [h, a+ i] and [B_x, a+ i],

    (h^T D - D h^T)(a,i) + sum_x (B_x^T Y_x - Y_x B_x^T)(a,i),   i correlated.

The gradient is assembled as

  1. sources: the triples (one Fortran pass, ccpy.gradients.crcc23_gradient.calc_triples) give
     delta, W, F_r0, F_r, F_l and F_t;
  2. responses: z_b (the EOMCCSD multipliers, ccpy.gradients.eomccsd_gradient.build_q_* and
     solve_z), the state multipliers b and a (deflated, solve_state), z_c (build_g_1A ... 2C);
  3. the bare-frame densities D1 = dE/dh and Y_x = dE/dB_x of the Hbar coefficients
     l r^T + z_b e0^T (base) and v r^T + l u^T + W e0 e0^T + z_c e0^T + a r^T + l b^T (correction),
     plus the bare-Fock denominator density (Eqs. 8.9-8.10);
  4. orbital relaxation (ccpy.gradients.orbital_response) and
  5. the derivative integrals (ccpy.gradients.derivative_integrals).

The EOMCCSD part of the Lagrangian is that of ccpy.gradients.eomccsd_gradient and the triples
those of ccpy.gradients.crcc23_gradient; this module adds the correction. Densities and bare
integrals are held in the blocks oo, ov, vo, vv of each spin (ccpy.gradients.density_gradient),
"o" running over all occupied orbitals with the frozen core first: the correlated occupied
orbitals are [nfrozen:].
"""

import copy
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse.linalg as spla

from ccpy.density.ccsd_rdm import calc_ccsd_rdm, calc_ccsd_rdm_derivative, calc_rdm2_vvvv_slice
from ccpy.gradients.crcc23_gradient import (backprop_triples_intermediates_aaa, backprop_triples_intermediates_aab,
                                            backprop_triples_intermediates_abb, backprop_triples_intermediates_bbb,
                                            build_triples_intermediates_aaa, build_triples_intermediates_aab,
                                            build_triples_intermediates_abb, build_triples_intermediates_bbb,
                                            calc_triples, cholesky_ladder, zero_triples_density)
from ccpy.gradients.density_gradient import build_DY, get_bare_integrals, zero_density
from ccpy.gradients.derivative_integrals import contract_derivative_integrals, print_gradient
from ccpy.gradients.eomccsd_gradient import (build_q_1A, build_q_1B, build_q_2A, build_q_2B, build_q_2C,
                                             solve_z)
from ccpy.gradients.orbital_response import orbital_relaxation
from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol, build_hbar_ccs_chol_derivative
from ccpy.integrals import integrals
from ccpy.models.integrals import Integral


def calc_creomcc23_gradient(T, R, r0, L, omega, Hbar, reference, delta=None, zvector_tol=1.0e-10, components=False,
                            perturbation=None, verbose=True):
    """Analytic delta-CR-EOMCC(2,3)_A gradient of one EOMCCSD root, (natoms, 3) in Eh/bohr, from the
    CCSD T, the root's R, r0, L, omega, and the CCSD HBar, on the SCF reference
    (ccpy.methods.Reference). components=True returns a dict with the base (EOMCCSD, nuclear
    repulsion included), correction and total gradients. perturbation: AO one-electron operator(s),
    for which dE/dlambda is returned instead (see contract_derivative_integrals)."""
    if reference.system.cholesky:
        from ccpy.left.left_ccsd_chol import LH_fun
    else:
        from ccpy.left.left_ccsd import LH_fun

    t0 = time.perf_counter()
    system, nfz = reference.system, reference.nfrozen
    H0 = reference.H
    H = build_hbar_ccs_chol(T, H0, system)
    h, B = get_bare_integrals(reference)
    naux = B.a.oo.shape[0]
    R = SimpleNamespace(a=np.array(R.a), b=np.array(R.b), aa=np.array(R.aa), ab=np.array(R.ab), bb=np.array(R.bb))
    L = SimpleNamespace(a=np.array(L.a), b=np.array(L.b), aa=np.array(L.aa), ab=np.array(L.ab), bb=np.array(L.bb))

    # EOMCCSD intermediates (ccpy.gradients.eomccsd_gradient)
    # C_J = <0|L R1|J>: c(a,i) = sum_jb l(ab,ij) r(b,j);  L . R over the unique singles and doubles
    c_a = np.einsum("abij,bj->ai", L.aa, R.a, optimize=True) + np.einsum("abij,bj->ai", L.ab, R.b, optimize=True)
    c_b = np.einsum("baji,bj->ai", L.ab, R.a, optimize=True) + np.einsum("abij,bj->ai", L.bb, R.b, optimize=True)
    lr = (np.sum(L.a * R.a) + np.sum(L.b * R.b) + 0.25 * np.sum(L.aa * R.aa) + np.sum(L.ab * R.ab)
          + 0.25 * np.sum(L.bb * R.bb))
    # densities of <0|L (H(T1) e^{T2})_C|0> and their derivative along R2
    rdm1_L, rdm2_L = calc_ccsd_rdm(T, L, system, energy=0.0)
    D_L, Y_L = build_DY(rdm1_L, rdm2_L, lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s), H.chol, T, B, reference)
    D_R2, Y_R2 = build_DY(*calc_ccsd_rdm_derivative(T, R, L, system, energy=0.0),
                          lambda a, s: calc_rdm2_vvvv_slice(L, R, a, s), H.chol, T, B, reference)
    # H1R = [H(T1), R1] in the blocks of H(T1); [h, R1] and [B, R1] in the bare blocks
    H1R = build_hbar_ccs_chol_derivative(R, H, system)
    h1R, B1R = zero_density(reference), zero_density(reference, naux)
    h1R.a.oo[:, nfz:] = h.a.ov @ R.a
    h1R.a.vo[...] = -R.a @ h.a.oo[nfz:, :]
    h1R.a.vo[:, nfz:] += h.a.vv @ R.a
    h1R.a.vv[...] = -R.a @ h.a.ov[nfz:, :]
    h1R.b.oo[:, nfz:] = h.b.ov @ R.b
    h1R.b.vo[...] = -R.b @ h.b.oo[nfz:, :]
    h1R.b.vo[:, nfz:] += h.b.vv @ R.b
    h1R.b.vv[...] = -R.b @ h.b.ov[nfz:, :]
    B1R.a.oo[:, :, nfz:] = B.a.ov @ R.a
    B1R.a.vo[...] = -R.a @ B.a.oo[:, nfz:, :]
    B1R.a.vo[:, :, nfz:] += B.a.vv @ R.a
    B1R.a.vv[...] = -R.a @ B.a.ov[:, nfz:, :]
    B1R.b.oo[:, :, nfz:] = B.b.ov @ R.b
    B1R.b.vo[...] = -R.b @ B.b.oo[:, nfz:, :]
    B1R.b.vo[:, :, nfz:] += B.b.vv @ R.b
    B1R.b.vv[...] = -R.b @ B.b.ov[:, nfz:, :]
    _, Y_LR1 = build_DY(rdm1_L, rdm2_L, lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s), H1R.chol, T, B1R, reference)
    # <0|(r0 L + C)(Hbar - E0)|mu> (left-CCSD LH_fun)
    Xop, LHx = copy.deepcopy(T), copy.deepcopy(T)
    Xop.a, Xop.b = np.asfortranarray(r0 * L.a + c_a), np.asfortranarray(r0 * L.b + c_b)
    Xop.aa, Xop.ab, Xop.bb = np.asfortranarray(r0 * L.aa), np.asfortranarray(r0 * L.ab), np.asfortranarray(r0 * L.bb)
    LH_fun(LHx, Xop, T, Hbar, False, system)

    # 1. sources
    # the triples: M = <K|(Hbar R)_C|0> = A sum_g (J_g . t_g + X_g . r_g), N = <L|Hbar|K>, D_K
    X = {"aaa": build_triples_intermediates_aaa(T, H), "aab": build_triples_intermediates_aab(T, H),
         "abb": build_triples_intermediates_abb(T, H), "bbb": build_triples_intermediates_bbb(T, H)}
    # J_g = X_g(T, H1R) + X_g(R, H): the moments of delta-CR-EOMCC(2,3), M = <K|(Hbar (R1 + R2))_C|0>.
    # NON-INTENSIVE TERM, dropped: r0 X_g, i.e. r0 times the ground-state moment <K|(H(T1) e^T2)_C|0>
    # (CR-EOMCC(2,3) proper; not size-intensive). With it, J_g = r0 X_g + X_g(T, H1R) + X_g(R, H):
    # J = {"aaa": [r0 * xg + xg1 + xgr for xg, xg1, xgr in zip(X["aaa"], build_triples_intermediates_aaa(T, H1R),
    #                                                          build_triples_intermediates_aaa(R, H, constant=False))],
    #      ... and likewise for "aab", "abb", "bbb"}
    J = {"aaa": [xg1 + xgr for xg1, xgr in zip(build_triples_intermediates_aaa(T, H1R),
                                               build_triples_intermediates_aaa(R, H, constant=False))],
         "aab": [xg1 + xgr for xg1, xgr in zip(build_triples_intermediates_aab(T, H1R),
                                               build_triples_intermediates_aab(R, H, constant=False))],
         "abb": [xg1 + xgr for xg1, xgr in zip(build_triples_intermediates_abb(T, H1R),
                                               build_triples_intermediates_abb(R, H, constant=False))],
         "bbb": [xg1 + xgr for xg1, xgr in zip(build_triples_intermediates_bbb(T, H1R),
                                               build_triples_intermediates_bbb(R, H, constant=False))]}
    rdm1_Lu, rdm2_Lu = zero_triples_density(system)
    tr = calc_triples(X, T, L, H, reference, rdm1_Lu, rdm2_Lu, omega=omega, R=R, r0=r0, J=J)
    del X, J
    delta_Q, W = tr["delta"], tr["W"]
    if delta is not None and abs(delta_Q - delta) > 1.0e-8:
        raise RuntimeError(f"delta from the moments ({delta_Q:.12f}) does not match the energy ({delta:.12f}).")
    # v . (J_g . t_g + X_g . r_g) through the intermediates:
    #   v . M3(T) = v . X_g(T, H) . t_g:        T2 gradient dT0, H(T1) densities rdm_v0
    #   v . X_g(T, H1R) . t_g:                   T2 gradient dTH1
    #   v . X_g(T, H) . r_g + v . X_g(R, H) . t_g: T2 gradient dTR, H(T1) densities rdm_vR
    zeros_T = {"aa": np.zeros_like(T.aa), "ab": np.zeros_like(T.ab), "bb": np.zeros_like(T.bb)}
    dT0, dTH1, dTR, dT_unused = (copy.deepcopy(zeros_T), copy.deepcopy(zeros_T), copy.deepcopy(zeros_T),
                                 copy.deepcopy(zeros_T))
    rdm1_v0, rdm2_v0 = zero_triples_density(system)
    rdm1_vR, rdm2_vR = zero_triples_density(system)
    rdm1_unused, rdm2_unused = zero_triples_density(system)
    backprop_triples_intermediates_aaa(tr["dX"]["aaa"], T, H, dT0, rdm1_v0, rdm2_v0)
    backprop_triples_intermediates_aab(tr["dX"]["aab"], T, H, dT0, rdm1_v0, rdm2_v0)
    backprop_triples_intermediates_abb(tr["dX"]["abb"], T, H, dT0, rdm1_v0, rdm2_v0)
    backprop_triples_intermediates_bbb(tr["dX"]["bbb"], T, H, dT0, rdm1_v0, rdm2_v0)
    backprop_triples_intermediates_aaa(tr["dX"]["aaa"], T, H1R, dTH1, rdm1_unused, rdm2_unused)
    backprop_triples_intermediates_aab(tr["dX"]["aab"], T, H1R, dTH1, rdm1_unused, rdm2_unused)
    backprop_triples_intermediates_abb(tr["dX"]["abb"], T, H1R, dTH1, rdm1_unused, rdm2_unused)
    backprop_triples_intermediates_bbb(tr["dX"]["bbb"], T, H1R, dTH1, rdm1_unused, rdm2_unused)
    backprop_triples_intermediates_aaa(tr["dXR"]["aaa"], T, H, dTR, rdm1_vR, rdm2_vR)
    backprop_triples_intermediates_aab(tr["dXR"]["aab"], T, H, dTR, rdm1_vR, rdm2_vR)
    backprop_triples_intermediates_abb(tr["dXR"]["abb"], T, H, dTR, rdm1_vR, rdm2_vR)
    backprop_triples_intermediates_bbb(tr["dXR"]["bbb"], T, H, dTR, rdm1_vR, rdm2_vR)
    backprop_triples_intermediates_aaa(tr["dX"]["aaa"], R, H, dT_unused, rdm1_vR, rdm2_vR, constant=False)
    backprop_triples_intermediates_aab(tr["dX"]["aab"], R, H, dT_unused, rdm1_vR, rdm2_vR, constant=False)
    backprop_triples_intermediates_abb(tr["dX"]["abb"], R, H, dT_unused, rdm1_vR, rdm2_vR, constant=False)
    backprop_triples_intermediates_bbb(tr["dX"]["bbb"], R, H, dT_unused, rdm1_vR, rdm2_vR, constant=False)
    cprime = SimpleNamespace(**tr["cprime"])
    # bare-frame densities: <V3|O|0> (D_v0), <V3|O|R> (D_vR) and <L|O|U3> (D_Lu)
    D_v0, Y_v0 = build_DY(rdm1_v0, rdm2_v0, None, H.chol, T, B, reference)
    D_Lu, Y_Lu = build_DY(rdm1_Lu, rdm2_Lu, None, H.chol, T, B, reference)
    # <V3|O|R> = D_R2 <V3|O|0> + D_R1 <V3|O|0> + <C'|O|0>
    # (NON-INTENSIVE TERM, dropped: + r0 <V3|O|0>, the density of r0 times the ground-state moment)
    D_vR, Y_vR = build_DY(rdm1_vR, rdm2_vR, None, H.chol, T, B, reference)
    _, Y_v0R1 = build_DY(rdm1_v0, rdm2_v0, None, H1R.chol, T, B1R, reference)
    D_c, Y_c = build_DY(*calc_ccsd_rdm(T, cprime, system, energy=0.0),
                        lambda a, s: calc_rdm2_vvvv_slice(cprime, T, a, s), H.chol, T, B, reference)
    D_vR.a.oo += D_c.a.oo  # + r0 * D_v0.a.oo (non-intensive)
    D_vR.a.ov += D_c.a.ov  # + r0 * D_v0.a.ov (non-intensive)
    D_vR.a.vo += D_c.a.vo  # + r0 * D_v0.a.vo (non-intensive)
    D_vR.a.vv += D_c.a.vv  # + r0 * D_v0.a.vv (non-intensive)
    D_vR.b.oo += D_c.b.oo  # + r0 * D_v0.b.oo (non-intensive)
    D_vR.b.ov += D_c.b.ov  # + r0 * D_v0.b.ov (non-intensive)
    D_vR.b.vo += D_c.b.vo  # + r0 * D_v0.b.vo (non-intensive)
    D_vR.b.vv += D_c.b.vv  # + r0 * D_v0.b.vv (non-intensive)
    Y_vR.a.oo += Y_c.a.oo + Y_v0R1.a.oo  # + r0 * Y_v0.a.oo (non-intensive)
    Y_vR.a.ov += Y_c.a.ov + Y_v0R1.a.ov  # + r0 * Y_v0.a.ov (non-intensive)
    Y_vR.a.vo += Y_c.a.vo + Y_v0R1.a.vo  # + r0 * Y_v0.a.vo (non-intensive)
    Y_vR.a.vv += Y_c.a.vv + Y_v0R1.a.vv  # + r0 * Y_v0.a.vv (non-intensive)
    Y_vR.b.oo += Y_c.b.oo + Y_v0R1.b.oo  # + r0 * Y_v0.b.oo (non-intensive)
    Y_vR.b.ov += Y_c.b.ov + Y_v0R1.b.ov  # + r0 * Y_v0.b.ov (non-intensive)
    Y_vR.b.vo += Y_c.b.vo + Y_v0R1.b.vo  # + r0 * Y_v0.b.vo (non-intensive)
    Y_vR.b.vv += Y_c.b.vv + Y_v0R1.b.vv  # + r0 * Y_v0.b.vv (non-intensive)
    # ... the R1 rotation of <V3|O|0>: D_v0 R1^T - R1^T D_v0 (and Y_v0 likewise)
    D_vR.a.oo[nfz:, :] -= R.a.T @ D_v0.a.vo
    D_vR.a.ov += D_v0.a.oo[:, nfz:] @ R.a.T
    D_vR.a.ov[nfz:, :] -= R.a.T @ D_v0.a.vv
    D_vR.a.vv += D_v0.a.vo[:, nfz:] @ R.a.T
    D_vR.b.oo[nfz:, :] -= R.b.T @ D_v0.b.vo
    D_vR.b.ov += D_v0.b.oo[:, nfz:] @ R.b.T
    D_vR.b.ov[nfz:, :] -= R.b.T @ D_v0.b.vv
    D_vR.b.vv += D_v0.b.vo[:, nfz:] @ R.b.T
    Y_vR.a.oo[:, nfz:, :] -= R.a.T @ Y_v0.a.vo
    Y_vR.a.ov += Y_v0.a.oo[:, :, nfz:] @ R.a.T
    Y_vR.a.ov[:, nfz:, :] -= R.a.T @ Y_v0.a.vv
    Y_vR.a.vv += Y_v0.a.vo[:, :, nfz:] @ R.a.T
    Y_vR.b.oo[:, nfz:, :] -= R.b.T @ Y_v0.b.vo
    Y_vR.b.ov += Y_v0.b.oo[:, :, nfz:] @ R.b.T
    Y_vR.b.ov[:, nfz:, :] -= R.b.T @ Y_v0.b.vv
    Y_vR.b.vv += Y_v0.b.vo[:, :, nfz:] @ R.b.T
    # F_r(mu) = sum_K v_K d<K|(H(T1) e^T2)_C|0>/dt_mu, the derivative of delta along r(mu).
    # NON-INTENSIVE TERM, dropped: the derivative along r0, F_r0 = sum_K v_K <K|(H(T1) e^T2)_C|0>, which
    # delta-CR-EOMCC(2,3) does not have
    # F_r0 = tr["F_r0"]
    F_r0 = 0.0
    dTr = {"aa": dT0["aa"] + tr["dT"]["aa"], "ab": dT0["ab"] + tr["dT"]["ab"], "bb": dT0["bb"] + tr["dT"]["bb"]}
    F_r = SimpleNamespace(
        a=build_F_r_1A(D_v0, Y_v0, h, B, nfz), b=build_F_r_1B(D_v0, Y_v0, h, B, nfz), ab=dTr["ab"],
        aa=(dTr["aa"] - dTr["aa"].transpose(1, 0, 2, 3) - dTr["aa"].transpose(0, 1, 3, 2)
            + dTr["aa"].transpose(1, 0, 3, 2)),
        bb=(dTr["bb"] - dTr["bb"].transpose(1, 0, 2, 3) - dTr["bb"].transpose(0, 1, 3, 2)
            + dTr["bb"].transpose(1, 0, 3, 2)),
    )
    # F_l(mu) = sum_K <mu|Hbar|K> u_K
    dL = tr["dL"]
    F_l = SimpleNamespace(
        a=dL["a"], b=dL["b"], ab=dL["ab"],
        aa=dL["aa"] - dL["aa"].transpose(1, 0, 2, 3) - dL["aa"].transpose(0, 1, 3, 2) + dL["aa"].transpose(1, 0, 3, 2),
        bb=dL["bb"] - dL["bb"].transpose(1, 0, 2, 3) - dL["bb"].transpose(0, 1, 3, 2) + dL["bb"].transpose(1, 0, 3, 2),
    )
    # F_t doubles: the T2 gradient of sum_K v_K M_K (J_g . t_g directly, X_g(T, H1R) and X_g(T, H) . r_g
    # through the intermediates), (J^T c')_doubles and W <0|Hbar|ijab>; the singles are part of
    # build_g_1A / build_g_1B. NON-INTENSIVE TERM, dropped: r0 dT0, the T2 gradient of r0 X_g(T, H) . t_g
    Cp, LHcp = copy.deepcopy(T), copy.deepcopy(T)
    Cp.a, Cp.b = np.asfortranarray(cprime.a), np.asfortranarray(cprime.b)
    Cp.aa, Cp.ab, Cp.bb = np.asfortranarray(cprime.aa), np.asfortranarray(cprime.ab), np.asfortranarray(cprime.bb)
    LH_fun(LHcp, Cp, T, Hbar, False, system)
    # (non-intensive: + r0 * dT0["aa"], + r0 * dT0["ab"], + r0 * dT0["bb"])
    dTt = {"aa": tr["dJT"]["aa"] + dTH1["aa"] + dTR["aa"],
           "ab": tr["dJT"]["ab"] + dTH1["ab"] + dTR["ab"],
           "bb": tr["dJT"]["bb"] + dTH1["bb"] + dTR["bb"]}
    F_t_aa = (dTt["aa"] - dTt["aa"].transpose(1, 0, 2, 3) - dTt["aa"].transpose(0, 1, 3, 2)
              + dTt["aa"].transpose(1, 0, 3, 2)
              + np.array(LHcp.aa) + W * Hbar.aa.oovv.transpose(2, 3, 0, 1))
    F_t_ab = dTt["ab"] + np.array(LHcp.ab) + W * Hbar.ab.oovv.transpose(2, 3, 0, 1)
    F_t_bb = (dTt["bb"] - dTt["bb"].transpose(1, 0, 2, 3) - dTt["bb"].transpose(0, 1, 3, 2)
              + dTt["bb"].transpose(1, 0, 3, 2)
              + np.array(LHcp.bb) + W * Hbar.bb.oovv.transpose(2, 3, 0, 1))

    # 2. responses
    # base: sum_nu z_b,nu <nu|Hbar - E0|mu> = -q(mu),  q(mu) = <L|[Hbar, tau_mu]|R>  (EOMCCSD)
    q = SimpleNamespace(
        a=build_q_1A(np.array(LHx.a), lr, Hbar, h, B, h1R, B1R, D_L, Y_L, D_R2, Y_R2, Y_LR1, nfz),
        b=build_q_1B(np.array(LHx.b), lr, Hbar, h, B, h1R, B1R, D_L, Y_L, D_R2, Y_R2, Y_LR1, nfz),
        aa=build_q_2A(np.array(LHx.aa), lr, Hbar, R, T, L, H, H1R),
        ab=build_q_2B(np.array(LHx.ab), lr, Hbar, R, T, L, H, H1R),
        bb=build_q_2C(np.array(LHx.bb), lr, Hbar, R, T, L, H, H1R),
    )
    z_b = solve_z(q, T, Hbar, system, zvector_tol)
    # state multipliers, nu = -delta (Eqs. 7.10-7.12):
    #   (A - E) b = -F_l + delta r,   (A - E)^T a = -F_r + delta l
    b0, bP = solve_state(delta_Q * r0, SimpleNamespace(a=-F_l.a + delta_Q * R.a, b=-F_l.b + delta_Q * R.b,
                                                       aa=-F_l.aa + delta_Q * R.aa, ab=-F_l.ab + delta_Q * R.ab,
                                                       bb=-F_l.bb + delta_Q * R.bb),
                         False, T, R, r0, L, omega, Hbar, system, zvector_tol)
    a0, aP = solve_state(-F_r0, SimpleNamespace(a=-F_r.a + delta_Q * L.a, b=-F_r.b + delta_Q * L.b,
                                                aa=-F_r.aa + delta_Q * L.aa, ab=-F_r.ab + delta_Q * L.ab,
                                                bb=-F_r.bb + delta_Q * L.bb),
                         True, T, R, r0, L, omega, Hbar, system, zvector_tol)
    # deflation gives l.b = r.a = 0; a + beta l satisfies a.r + b.l = -W
    ar = r0 * a0 + (np.sum(aP.a * R.a) + np.sum(aP.b * R.b) + 0.25 * np.sum(aP.aa * R.aa) + np.sum(aP.ab * R.ab)
                    + 0.25 * np.sum(aP.bb * R.bb))
    lb = (np.sum(L.a * bP.a) + np.sum(L.b * bP.b) + 0.25 * np.sum(L.aa * bP.aa) + np.sum(L.ab * bP.ab)
          + 0.25 * np.sum(L.bb * bP.bb))
    beta = -W - ar - lb
    aP = SimpleNamespace(a=aP.a + beta * L.a, b=aP.b + beta * L.b, aa=aP.aa + beta * L.aa, ab=aP.ab + beta * L.ab,
                         bb=aP.bb + beta * L.bb)
    # intermediates of g
    # C_J = <0|A R1|J> + <0|L b1|J>: c(a,i) = sum_jb [a(ab,ij) r(b,j) + l(ab,ij) b(b,j)];
    # x0 = r0 a0 + (A . R) + (L . b)
    cg_a = (np.einsum("abij,bj->ai", aP.aa, R.a, optimize=True) + np.einsum("abij,bj->ai", aP.ab, R.b, optimize=True)
            + np.einsum("abij,bj->ai", L.aa, bP.a, optimize=True) + np.einsum("abij,bj->ai", L.ab, bP.b, optimize=True))
    cg_b = (np.einsum("baji,bj->ai", aP.ab, R.a, optimize=True) + np.einsum("abij,bj->ai", aP.bb, R.b, optimize=True)
            + np.einsum("baji,bj->ai", L.ab, bP.a, optimize=True) + np.einsum("abij,bj->ai", L.bb, bP.b, optimize=True))
    x0 = r0 * a0 + (np.sum(aP.a * R.a) + np.sum(aP.b * R.b) + 0.25 * np.sum(aP.aa * R.aa) + np.sum(aP.ab * R.ab)
                    + 0.25 * np.sum(aP.bb * R.bb)) + lb
    # <0|(r0 A + b0 L + C)(Hbar - E0)|mu> (left-CCSD LH_fun)
    Yop, LHy = copy.deepcopy(T), copy.deepcopy(T)
    Yop.a, Yop.b = np.asfortranarray(r0 * aP.a + b0 * L.a + cg_a), np.asfortranarray(r0 * aP.b + b0 * L.b + cg_b)
    Yop.aa = np.asfortranarray(r0 * aP.aa + b0 * L.aa)
    Yop.ab = np.asfortranarray(r0 * aP.ab + b0 * L.ab)
    Yop.bb = np.asfortranarray(r0 * aP.bb + b0 * L.bb)
    LH_fun(LHy, Yop, T, Hbar, False, system)
    # densities of <0|(a0 + A) Hbar|0>, their derivative along R2, and the Y with H(T1) -> H1R
    rdm1_A, rdm2_A = calc_ccsd_rdm(T, aP, system, energy=a0)
    D_A, Y_A = build_DY(rdm1_A, rdm2_A, lambda a, s: calc_rdm2_vvvv_slice(aP, T, a, s), H.chol, T, B, reference,
                        ref_weight=a0)
    D_AR2, Y_AR2 = build_DY(*calc_ccsd_rdm_derivative(T, R, aP, system, energy=a0),
                            lambda a, s: calc_rdm2_vvvv_slice(aP, R, a, s), H.chol, T, B, reference)
    _, Y_AR1 = build_DY(rdm1_A, rdm2_A, lambda a, s: calc_rdm2_vvvv_slice(aP, T, a, s), H1R.chol, T, B1R, reference,
                        ref_weight=a0)
    # the derivative of <0|L Hbar|0> along b2, H1b = [H(T1), b1], [h, b1], [B, b1], and the Y of
    # <0|L Hbar|0> with H(T1) -> H1b
    D_Lb2, Y_Lb2 = build_DY(*calc_ccsd_rdm_derivative(T, bP, L, system, energy=0.0),
                            lambda a, s: calc_rdm2_vvvv_slice(L, bP, a, s), H.chol, T, B, reference)
    H1b = build_hbar_ccs_chol_derivative(bP, H, system)
    h1b, B1b = zero_density(reference), zero_density(reference, naux)
    h1b.a.oo[:, nfz:] = h.a.ov @ bP.a
    h1b.a.vo[...] = -bP.a @ h.a.oo[nfz:, :]
    h1b.a.vo[:, nfz:] += h.a.vv @ bP.a
    h1b.a.vv[...] = -bP.a @ h.a.ov[nfz:, :]
    h1b.b.oo[:, nfz:] = h.b.ov @ bP.b
    h1b.b.vo[...] = -bP.b @ h.b.oo[nfz:, :]
    h1b.b.vo[:, nfz:] += h.b.vv @ bP.b
    h1b.b.vv[...] = -bP.b @ h.b.ov[nfz:, :]
    B1b.a.oo[:, :, nfz:] = B.a.ov @ bP.a
    B1b.a.vo[...] = -bP.a @ B.a.oo[:, nfz:, :]
    B1b.a.vo[:, :, nfz:] += B.a.vv @ bP.a
    B1b.a.vv[...] = -bP.a @ B.a.ov[:, nfz:, :]
    B1b.b.oo[:, :, nfz:] = B.b.ov @ bP.b
    B1b.b.vo[...] = -bP.b @ B.b.oo[:, nfz:, :]
    B1b.b.vo[:, :, nfz:] += B.b.vv @ bP.b
    B1b.b.vv[...] = -bP.b @ B.b.ov[:, nfz:, :]
    _, Y_Lb1 = build_DY(rdm1_L, rdm2_L, lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s), H1b.chol, T, B1b, reference)
    # correction: sum_nu z_c,nu <nu|Hbar - E0|mu> = -g(mu)
    g = SimpleNamespace(
        a=build_g_1A(np.array(LHy.a), x0, W, Hbar, h, B, h1R, B1R, h1b, B1b, D_vR, Y_vR, D_Lu, Y_Lu,
                     D_A, Y_A, D_AR2, Y_AR2, Y_AR1, D_L, Y_L, D_Lb2, Y_Lb2, Y_Lb1, nfz),
        b=build_g_1B(np.array(LHy.b), x0, W, Hbar, h, B, h1R, B1R, h1b, B1b, D_vR, Y_vR, D_Lu, Y_Lu,
                     D_A, Y_A, D_AR2, Y_AR2, Y_AR1, D_L, Y_L, D_Lb2, Y_Lb2, Y_Lb1, nfz),
        aa=build_g_2A(F_t_aa, np.array(LHy.aa), x0, Hbar, R, aP, a0, bP, L, T, H, H1R, H1b),
        ab=build_g_2B(F_t_ab, np.array(LHy.ab), x0, Hbar, R, aP, a0, bP, L, T, H, H1R, H1b),
        bb=build_g_2C(F_t_bb, np.array(LHy.bb), x0, Hbar, R, aP, a0, bP, L, T, H, H1R, H1b),
    )
    z_c = solve_z(g, T, Hbar, system, zvector_tol)

    # 3. bare-frame densities
    # base: <L|Hbar|R> + <Z_b|Hbar|0> = <0|((L . R) + r0 L + C + Z_b) Hbar|0> + <0|L [Hbar, R]|0>
    Xb = SimpleNamespace(a=r0 * L.a + c_a + z_b.a, b=r0 * L.b + c_b + z_b.b, aa=r0 * L.aa + z_b.aa,
                         ab=r0 * L.ab + z_b.ab, bb=r0 * L.bb + z_b.bb)
    D1b, Yb = build_DY(*calc_ccsd_rdm(T, Xb, system, energy=lr), lambda a, s: calc_rdm2_vvvv_slice(Xb, T, a, s),
                       H.chol, T, B, reference, ref_weight=lr)
    D1b.a.oo += D_R2.a.oo
    D1b.a.ov += D_R2.a.ov
    D1b.a.vo += D_R2.a.vo
    D1b.a.vv += D_R2.a.vv
    D1b.b.oo += D_R2.b.oo
    D1b.b.ov += D_R2.b.ov
    D1b.b.vo += D_R2.b.vo
    D1b.b.vv += D_R2.b.vv
    Yb.a.oo += Y_R2.a.oo + Y_LR1.a.oo
    Yb.a.ov += Y_R2.a.ov + Y_LR1.a.ov
    Yb.a.vo += Y_R2.a.vo + Y_LR1.a.vo
    Yb.a.vv += Y_R2.a.vv + Y_LR1.a.vv
    Yb.b.oo += Y_R2.b.oo + Y_LR1.b.oo
    Yb.b.ov += Y_R2.b.ov + Y_LR1.b.ov
    Yb.b.vo += Y_R2.b.vo + Y_LR1.b.vo
    Yb.b.vv += Y_R2.b.vv + Y_LR1.b.vv
    D1b.a.oo[nfz:, :] -= R.a.T @ D_L.a.vo
    D1b.a.ov += D_L.a.oo[:, nfz:] @ R.a.T
    D1b.a.ov[nfz:, :] -= R.a.T @ D_L.a.vv
    D1b.a.vv += D_L.a.vo[:, nfz:] @ R.a.T
    D1b.b.oo[nfz:, :] -= R.b.T @ D_L.b.vo
    D1b.b.ov += D_L.b.oo[:, nfz:] @ R.b.T
    D1b.b.ov[nfz:, :] -= R.b.T @ D_L.b.vv
    D1b.b.vv += D_L.b.vo[:, nfz:] @ R.b.T
    Yb.a.oo[:, nfz:, :] -= R.a.T @ Y_L.a.vo
    Yb.a.ov += Y_L.a.oo[:, :, nfz:] @ R.a.T
    Yb.a.ov[:, nfz:, :] -= R.a.T @ Y_L.a.vv
    Yb.a.vv += Y_L.a.vo[:, :, nfz:] @ R.a.T
    Yb.b.oo[:, nfz:, :] -= R.b.T @ Y_L.b.vo
    Yb.b.ov += Y_L.b.oo[:, :, nfz:] @ R.b.T
    Yb.b.ov[:, nfz:, :] -= R.b.T @ Y_L.b.vv
    Yb.b.vv += Y_L.b.vo[:, :, nfz:] @ R.b.T
    # correction: v r^T + l u^T (the triples), W e0 e0^T + z_c e0^T and the bra parts of a r^T and
    # l b^T, <0|(W + x0 + Z_c + r0 A + C + b0 L) Hbar|0> ...
    Xc = SimpleNamespace(a=z_c.a + r0 * aP.a + cg_a + b0 * L.a, b=z_c.b + r0 * aP.b + cg_b + b0 * L.b,
                         aa=z_c.aa + r0 * aP.aa + b0 * L.aa, ab=z_c.ab + r0 * aP.ab + b0 * L.ab,
                         bb=z_c.bb + r0 * aP.bb + b0 * L.bb)
    D1c, Yc = build_DY(*calc_ccsd_rdm(T, Xc, system, energy=W + x0), lambda a, s: calc_rdm2_vvvv_slice(Xc, T, a, s),
                       H.chol, T, B, reference, ref_weight=W + x0)
    D1c.a.oo += D_vR.a.oo + D_Lu.a.oo
    D1c.a.ov += D_vR.a.ov + D_Lu.a.ov
    D1c.a.vo += D_vR.a.vo + D_Lu.a.vo
    D1c.a.vv += D_vR.a.vv + D_Lu.a.vv
    D1c.b.oo += D_vR.b.oo + D_Lu.b.oo
    D1c.b.ov += D_vR.b.ov + D_Lu.b.ov
    D1c.b.vo += D_vR.b.vo + D_Lu.b.vo
    D1c.b.vv += D_vR.b.vv + D_Lu.b.vv
    Yc.a.oo += Y_vR.a.oo + Y_Lu.a.oo
    Yc.a.ov += Y_vR.a.ov + Y_Lu.a.ov
    Yc.a.vo += Y_vR.a.vo + Y_Lu.a.vo
    Yc.a.vv += Y_vR.a.vv + Y_Lu.a.vv
    Yc.b.oo += Y_vR.b.oo + Y_Lu.b.oo
    Yc.b.ov += Y_vR.b.ov + Y_Lu.b.ov
    Yc.b.vo += Y_vR.b.vo + Y_Lu.b.vo
    Yc.b.vv += Y_vR.b.vv + Y_Lu.b.vv
    # ... <0|(a0 + A)[Hbar, R]|0>: along R2, and through R1 (the rotated densities and H1R)
    D1c.a.oo += D_AR2.a.oo
    D1c.a.ov += D_AR2.a.ov
    D1c.a.vo += D_AR2.a.vo
    D1c.a.vv += D_AR2.a.vv
    D1c.b.oo += D_AR2.b.oo
    D1c.b.ov += D_AR2.b.ov
    D1c.b.vo += D_AR2.b.vo
    D1c.b.vv += D_AR2.b.vv
    Yc.a.oo += Y_AR2.a.oo + Y_AR1.a.oo
    Yc.a.ov += Y_AR2.a.ov + Y_AR1.a.ov
    Yc.a.vo += Y_AR2.a.vo + Y_AR1.a.vo
    Yc.a.vv += Y_AR2.a.vv + Y_AR1.a.vv
    Yc.b.oo += Y_AR2.b.oo + Y_AR1.b.oo
    Yc.b.ov += Y_AR2.b.ov + Y_AR1.b.ov
    Yc.b.vo += Y_AR2.b.vo + Y_AR1.b.vo
    Yc.b.vv += Y_AR2.b.vv + Y_AR1.b.vv
    D1c.a.oo[nfz:, :] -= R.a.T @ D_A.a.vo
    D1c.a.ov += D_A.a.oo[:, nfz:] @ R.a.T
    D1c.a.ov[nfz:, :] -= R.a.T @ D_A.a.vv
    D1c.a.vv += D_A.a.vo[:, nfz:] @ R.a.T
    D1c.b.oo[nfz:, :] -= R.b.T @ D_A.b.vo
    D1c.b.ov += D_A.b.oo[:, nfz:] @ R.b.T
    D1c.b.ov[nfz:, :] -= R.b.T @ D_A.b.vv
    D1c.b.vv += D_A.b.vo[:, nfz:] @ R.b.T
    Yc.a.oo[:, nfz:, :] -= R.a.T @ Y_A.a.vo
    Yc.a.ov += Y_A.a.oo[:, :, nfz:] @ R.a.T
    Yc.a.ov[:, nfz:, :] -= R.a.T @ Y_A.a.vv
    Yc.a.vv += Y_A.a.vo[:, :, nfz:] @ R.a.T
    Yc.b.oo[:, nfz:, :] -= R.b.T @ Y_A.b.vo
    Yc.b.ov += Y_A.b.oo[:, :, nfz:] @ R.b.T
    Yc.b.ov[:, nfz:, :] -= R.b.T @ Y_A.b.vv
    Yc.b.vv += Y_A.b.vo[:, :, nfz:] @ R.b.T
    # ... <0|L [Hbar, b]|0>: along b2, and through b1 (the rotated densities and H1b)
    D1c.a.oo += D_Lb2.a.oo
    D1c.a.ov += D_Lb2.a.ov
    D1c.a.vo += D_Lb2.a.vo
    D1c.a.vv += D_Lb2.a.vv
    D1c.b.oo += D_Lb2.b.oo
    D1c.b.ov += D_Lb2.b.ov
    D1c.b.vo += D_Lb2.b.vo
    D1c.b.vv += D_Lb2.b.vv
    Yc.a.oo += Y_Lb2.a.oo + Y_Lb1.a.oo
    Yc.a.ov += Y_Lb2.a.ov + Y_Lb1.a.ov
    Yc.a.vo += Y_Lb2.a.vo + Y_Lb1.a.vo
    Yc.a.vv += Y_Lb2.a.vv + Y_Lb1.a.vv
    Yc.b.oo += Y_Lb2.b.oo + Y_Lb1.b.oo
    Yc.b.ov += Y_Lb2.b.ov + Y_Lb1.b.ov
    Yc.b.vo += Y_Lb2.b.vo + Y_Lb1.b.vo
    Yc.b.vv += Y_Lb2.b.vv + Y_Lb1.b.vv
    D1c.a.oo[nfz:, :] -= bP.a.T @ D_L.a.vo
    D1c.a.ov += D_L.a.oo[:, nfz:] @ bP.a.T
    D1c.a.ov[nfz:, :] -= bP.a.T @ D_L.a.vv
    D1c.a.vv += D_L.a.vo[:, nfz:] @ bP.a.T
    D1c.b.oo[nfz:, :] -= bP.b.T @ D_L.b.vo
    D1c.b.ov += D_L.b.oo[:, nfz:] @ bP.b.T
    D1c.b.ov[nfz:, :] -= bP.b.T @ D_L.b.vv
    D1c.b.vv += D_L.b.vo[:, nfz:] @ bP.b.T
    Yc.a.oo[:, nfz:, :] -= bP.a.T @ Y_L.a.vo
    Yc.a.ov += Y_L.a.oo[:, :, nfz:] @ bP.a.T
    Yc.a.ov[:, nfz:, :] -= bP.a.T @ Y_L.a.vv
    Yc.a.vv += Y_L.a.vo[:, :, nfz:] @ bP.a.T
    Yc.b.oo[:, nfz:, :] -= bP.b.T @ Y_L.b.vo
    Yc.b.ov += Y_L.b.oo[:, :, nfz:] @ bP.b.T
    Yc.b.ov[:, nfz:, :] -= bP.b.T @ Y_L.b.vv
    Yc.b.vv += Y_L.b.vo[:, :, nfz:] @ bP.b.T
    # ... and -sum_K u_K v_K D_K = sum_p d_p f_pp: the bare Fock diagonal of the denominators
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
    # total
    D1, Y = zero_density(reference), zero_density(reference, naux)
    D1.a.oo[...], D1.a.ov[...], D1.a.vo[...], D1.a.vv[...] = (D1b.a.oo + D1c.a.oo, D1b.a.ov + D1c.a.ov,
                                                              D1b.a.vo + D1c.a.vo, D1b.a.vv + D1c.a.vv)
    D1.b.oo[...], D1.b.ov[...], D1.b.vo[...], D1.b.vv[...] = (D1b.b.oo + D1c.b.oo, D1b.b.ov + D1c.b.ov,
                                                              D1b.b.vo + D1c.b.vo, D1b.b.vv + D1c.b.vv)
    Y.a.oo[...], Y.a.ov[...], Y.a.vo[...], Y.a.vv[...] = (Yb.a.oo + Yc.a.oo, Yb.a.ov + Yc.a.ov,
                                                          Yb.a.vo + Yc.a.vo, Yb.a.vv + Yc.a.vv)
    Y.b.oo[...], Y.b.ov[...], Y.b.vo[...], Y.b.vv[...] = (Yb.b.oo + Yc.b.oo, Yb.b.ov + Yc.b.ov,
                                                          Yb.b.vo + Yc.b.vo, Yb.b.vv + Yc.b.vv)

    # 4. orbital relaxation and 5. derivative integrals
    D, Yr, Wd = orbital_relaxation(D1, Y, reference, zvector_tol=zvector_tol, verbose=verbose)
    grad = contract_derivative_integrals(D, Yr, Wd, reference, perturbation)
    if verbose:
        print_gradient(reference.molecule, grad, "delta-CR-EOMCC(2,3)_A Analytic Gradient", time.perf_counter() - t0)
    if not components:
        return grad
    D, Yr, Wd = orbital_relaxation(D1c, Yc, reference, zvector_tol=zvector_tol, verbose=False)
    grad_corr = contract_derivative_integrals(D, Yr, Wd, reference, perturbation)
    if perturbation is None:
        grad_corr -= integrals.nuclear_repulsion_deriv(reference.molecule)
    return {"base": grad - grad_corr, "correction": grad_corr, "total": grad}


def build_F_r_1A(D_v0, Y_v0, h, B, nfz):
    """F_r(ia) = sum_K v_K d<K|(H(T1) e^T2)_C|0>/dt(ai), the alpha singles of the triples source of
    the left state equation: the bare-frame densities of sum_K v_K <K|(H(T1) e^T2)_C|0> (D_v0, Y_v0)
    against [h, a+ i] and [B_x, a+ i],

        (h^T D - D h^T)(a,i) + sum_x (B_x^T Y_x - Y_x B_x^T)(a,i),   i correlated."""
    X = np.zeros((h.a.vv.shape[0], h.a.oo.shape[0] - nfz))
    X += h.a.ov.T @ D_v0.a.oo[:, nfz:] + h.a.vv.T @ D_v0.a.vo[:, nfz:]
    X -= D_v0.a.vo @ h.a.oo[nfz:, :].T + D_v0.a.vv @ h.a.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.a.ov, Y_v0.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.a.vv, Y_v0.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_v0.a.vo, B.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_v0.a.vv, B.a.ov[:, nfz:, :], optimize=True)
    return X


def build_F_r_1B(D_v0, Y_v0, h, B, nfz):
    """F_r(i~a~) = sum_K v_K d<K|(H(T1) e^T2)_C|0>/dt(a~i~), the beta singles (build_F_r_1A with the
    beta blocks)."""
    X = np.zeros((h.b.vv.shape[0], h.b.oo.shape[0] - nfz))
    X += h.b.ov.T @ D_v0.b.oo[:, nfz:] + h.b.vv.T @ D_v0.b.vo[:, nfz:]
    X -= D_v0.b.vo @ h.b.oo[nfz:, :].T + D_v0.b.vv @ h.b.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.b.ov, Y_v0.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.vv, Y_v0.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_v0.b.vo, B.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_v0.b.vv, B.b.ov[:, nfz:, :], optimize=True)
    return X


def build_g_1A(LHy, x0, W, Hbar, h, B, h1R, B1R, h1b, B1b, D_vR, Y_vR, D_Lu, Y_Lu, D_A, Y_A, D_AR2, Y_AR2, Y_AR1,
               D_L, Y_L, D_Lb2, Y_Lb2, Y_Lb1, nfz):
    """g(ia), the alpha singles of the source of the correction multiplier equations:

        g(ia) = F_t(ia) + <a|[Hbar, a+ i]|R> + <L|[Hbar, a+ i]|b>
              = sum_K v_K <K|[Hbar, a+ i] R|0> + sum_K u_K <L|[Hbar, a+ i]|K> + W <0|Hbar|ia>
                + <0|(r0 A + b0 L + C)(Hbar - E0)|ia> + x0 <0|Hbar|ia>
                + <0|(a0 + A)[[Hbar, a+ i], R]|0> + <0|L [[Hbar, a+ i], b]|0>,

    with C_J = <0|A R1|J> + <0|L b1|J> and x0 = r0 a0 + (A . R) + (L . b). Hbar depends on t(ai)
    only through H(T1), d H(T1)/d t(ai) = [H(T1), a+ i], so all but the LH_fun and <0|Hbar|ia> terms
    are contractions of bare-frame densities (D, Y) with [h, a+ i], [B_x, a+ i],

        (h^T D - D h^T)(a,i) + sum_x (B_x^T Y_x - Y_x B_x^T)(a,i),   i correlated."""
    # <0|(r0 A + b0 L + C)(Hbar - E0)|ia> (left-CCSD LH_fun) and (x0 + W) <0|Hbar|ia>
    X = LHy + (x0 + W) * Hbar.a.ov.T
    # sum_K v_K <K|[[Hbar, a+ i], R]|0> + sum_K u_K <L|[Hbar, a+ i]|K>: the densities <V3|O|R>, <L|O|U3>
    # against [h, a+ i], [B, a+ i]
    X += h.a.ov.T @ D_vR.a.oo[:, nfz:] + h.a.vv.T @ D_vR.a.vo[:, nfz:]
    X -= D_vR.a.vo @ h.a.oo[nfz:, :].T + D_vR.a.vv @ h.a.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.a.ov, Y_vR.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.a.vv, Y_vR.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_vR.a.vo, B.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_vR.a.vv, B.a.ov[:, nfz:, :], optimize=True)
    X += h.a.ov.T @ D_Lu.a.oo[:, nfz:] + h.a.vv.T @ D_Lu.a.vo[:, nfz:]
    X -= D_Lu.a.vo @ h.a.oo[nfz:, :].T + D_Lu.a.vv @ h.a.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.a.ov, Y_Lu.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.a.vv, Y_Lu.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Lu.a.vo, B.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Lu.a.vv, B.a.ov[:, nfz:, :], optimize=True)
    # <0|(a0 + A)[[Hbar, a+ i], R2]|0>: the R2-derivative densities of <0|(a0 + A) Hbar|0>
    X += h.a.ov.T @ D_AR2.a.oo[:, nfz:] + h.a.vv.T @ D_AR2.a.vo[:, nfz:]
    X -= D_AR2.a.vo @ h.a.oo[nfz:, :].T + D_AR2.a.vv @ h.a.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.a.ov, Y_AR2.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.a.vv, Y_AR2.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_AR2.a.vo, B.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_AR2.a.vv, B.a.ov[:, nfz:, :], optimize=True)
    # <0|(a0 + A)[[Hbar, a+ i], R1]|0>: the densities of <0|(a0 + A) Hbar|0> against [[h, R1], a+ i],
    # [[B, R1], a+ i] ...
    X += h1R.a.ov.T @ D_A.a.oo[:, nfz:] + h1R.a.vv.T @ D_A.a.vo[:, nfz:]
    X -= D_A.a.vo @ h1R.a.oo[nfz:, :].T + D_A.a.vv @ h1R.a.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B1R.a.ov, Y_A.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B1R.a.vv, Y_A.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_A.a.vo, B1R.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_A.a.vv, B1R.a.ov[:, nfz:, :], optimize=True)
    # ... and the Y of <0|(a0 + A) Hbar|0> with H(T1) -> [H(T1), R1] against [B, a+ i]
    X += np.einsum("xqa,xqi->ai", B.a.ov, Y_AR1.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.a.vv, Y_AR1.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_AR1.a.vo, B.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_AR1.a.vv, B.a.ov[:, nfz:, :], optimize=True)
    # <0|L [[Hbar, a+ i], b2]|0>: the b2-derivative densities of <0|L Hbar|0>
    X += h.a.ov.T @ D_Lb2.a.oo[:, nfz:] + h.a.vv.T @ D_Lb2.a.vo[:, nfz:]
    X -= D_Lb2.a.vo @ h.a.oo[nfz:, :].T + D_Lb2.a.vv @ h.a.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.a.ov, Y_Lb2.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.a.vv, Y_Lb2.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Lb2.a.vo, B.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Lb2.a.vv, B.a.ov[:, nfz:, :], optimize=True)
    # <0|L [[Hbar, a+ i], b1]|0>: the densities of <0|L Hbar|0> against [[h, b1], a+ i], [[B, b1], a+ i] ...
    X += h1b.a.ov.T @ D_L.a.oo[:, nfz:] + h1b.a.vv.T @ D_L.a.vo[:, nfz:]
    X -= D_L.a.vo @ h1b.a.oo[nfz:, :].T + D_L.a.vv @ h1b.a.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B1b.a.ov, Y_L.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B1b.a.vv, Y_L.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_L.a.vo, B1b.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_L.a.vv, B1b.a.ov[:, nfz:, :], optimize=True)
    # ... and the Y of <0|L Hbar|0> with H(T1) -> [H(T1), b1] against [B, a+ i]
    X += np.einsum("xqa,xqi->ai", B.a.ov, Y_Lb1.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.a.vv, Y_Lb1.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Lb1.a.vo, B.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Lb1.a.vv, B.a.ov[:, nfz:, :], optimize=True)
    return X


def build_g_1B(LHy, x0, W, Hbar, h, B, h1R, B1R, h1b, B1b, D_vR, Y_vR, D_Lu, Y_Lu, D_A, Y_A, D_AR2, Y_AR2, Y_AR1,
               D_L, Y_L, D_Lb2, Y_Lb2, Y_Lb1, nfz):
    """g(i~a~), the beta singles of the source of the correction multiplier equations (the terms of
    build_g_1A with the beta blocks)."""
    # <0|(r0 A + b0 L + C)(Hbar - E0)|i~a~> (left-CCSD LH_fun) and (x0 + W) <0|Hbar|i~a~>
    X = LHy + (x0 + W) * Hbar.b.ov.T
    # sum_K v_K <K|[[Hbar, a~+ i~], R]|0> + sum_K u_K <L|[Hbar, a~+ i~]|K>
    X += h.b.ov.T @ D_vR.b.oo[:, nfz:] + h.b.vv.T @ D_vR.b.vo[:, nfz:]
    X -= D_vR.b.vo @ h.b.oo[nfz:, :].T + D_vR.b.vv @ h.b.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.b.ov, Y_vR.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.vv, Y_vR.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_vR.b.vo, B.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_vR.b.vv, B.b.ov[:, nfz:, :], optimize=True)
    X += h.b.ov.T @ D_Lu.b.oo[:, nfz:] + h.b.vv.T @ D_Lu.b.vo[:, nfz:]
    X -= D_Lu.b.vo @ h.b.oo[nfz:, :].T + D_Lu.b.vv @ h.b.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.b.ov, Y_Lu.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.vv, Y_Lu.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Lu.b.vo, B.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Lu.b.vv, B.b.ov[:, nfz:, :], optimize=True)
    # <0|(a0 + A)[[Hbar, a~+ i~], R2]|0>
    X += h.b.ov.T @ D_AR2.b.oo[:, nfz:] + h.b.vv.T @ D_AR2.b.vo[:, nfz:]
    X -= D_AR2.b.vo @ h.b.oo[nfz:, :].T + D_AR2.b.vv @ h.b.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.b.ov, Y_AR2.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.vv, Y_AR2.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_AR2.b.vo, B.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_AR2.b.vv, B.b.ov[:, nfz:, :], optimize=True)
    # <0|(a0 + A)[[Hbar, a~+ i~], R1]|0>
    X += h1R.b.ov.T @ D_A.b.oo[:, nfz:] + h1R.b.vv.T @ D_A.b.vo[:, nfz:]
    X -= D_A.b.vo @ h1R.b.oo[nfz:, :].T + D_A.b.vv @ h1R.b.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B1R.b.ov, Y_A.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B1R.b.vv, Y_A.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_A.b.vo, B1R.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_A.b.vv, B1R.b.ov[:, nfz:, :], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.ov, Y_AR1.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.vv, Y_AR1.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_AR1.b.vo, B.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_AR1.b.vv, B.b.ov[:, nfz:, :], optimize=True)
    # <0|L [[Hbar, a~+ i~], b2]|0>
    X += h.b.ov.T @ D_Lb2.b.oo[:, nfz:] + h.b.vv.T @ D_Lb2.b.vo[:, nfz:]
    X -= D_Lb2.b.vo @ h.b.oo[nfz:, :].T + D_Lb2.b.vv @ h.b.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.b.ov, Y_Lb2.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.vv, Y_Lb2.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Lb2.b.vo, B.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Lb2.b.vv, B.b.ov[:, nfz:, :], optimize=True)
    # <0|L [[Hbar, a~+ i~], b1]|0>
    X += h1b.b.ov.T @ D_L.b.oo[:, nfz:] + h1b.b.vv.T @ D_L.b.vo[:, nfz:]
    X -= D_L.b.vo @ h1b.b.oo[nfz:, :].T + D_L.b.vv @ h1b.b.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B1b.b.ov, Y_L.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B1b.b.vv, Y_L.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_L.b.vo, B1b.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_L.b.vv, B1b.b.ov[:, nfz:, :], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.ov, Y_Lb1.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.vv, Y_Lb1.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Lb1.b.vo, B.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_Lb1.b.vv, B.b.ov[:, nfz:, :], optimize=True)
    return X

def build_g_2A(F_t, LHy, x0, Hbar, R, aP, a0, bP, L, T, H, H1R, H1b):
    """g(ijab) = F_t(ijab) + <a|[Hbar, tau(ab,ij)]|R> + <L|[Hbar, tau(ab,ij)]|b>, the aa doubles
    of the source of the correction cluster multipliers (Eq. 7.13). With a = a0 <0| + A,
    b = b0 |0> + b1 + b2, C_J = <0|A R1|J> + <0|L b1|J> (singles only) and
    x0 = r0 a0 + (A . R) + (L . b),

        <a|[Hbar, tau(ab,ij)]|R> + <L|[Hbar, tau(ab,ij)]|b>
            = <0|(r0 A + b0 L + C)(Hbar - E0)|ijab> + x0 <0|Hbar|ijab>
              + <a|[[Hbar, tau(ab,ij)], R]|0> + <L|[[Hbar, tau(ab,ij)], b]|0>,

    the last two through the T2^2 terms of the CCSD equations (R2, b2) and the left-CCSD doubles
    terms with H(T1) -> [H(T1), R1], [H(T1), b1]. F_t holds the triples part (the T2 gradient of
    sum_K v_K M_K, (J^T c')_doubles and W <0|Hbar|ijab>). Terms are built in the stored layout and
    antisymmetrized before the (already antisymmetric) remaining ones are added."""
    # <a|[[Hbar, tau(ab,ij)], R2]|0> = <0|A2 [[(V_N T2^2)_C/2, tau(ab,ij)], R2]|0>
    #     <mn||ef> r(af,mn) a(ab,ij)  -> (eb,ij)
    X = -0.25 * np.einsum("mnef,afmn,abij->ebij", H.aa.oovv, R.aa, aP.aa, optimize=True)
    #     <mn||ef> r(eb,ij) a(ab,ij)  -> (af,mn)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H.aa.oovv, R.aa, aP.aa, optimize=True)
    #     <mn~|ef~> r(af~,mn~) a(ab,ij)  -> (eb,ij)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H.ab.oovv, R.ab, aP.aa, optimize=True)
    #     <mn||ef> r(ef,in) a(ab,ij)  -> (ab,mj)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H.aa.oovv, R.aa, aP.aa, optimize=True)
    #     <mn||ef> r(ab,mj) a(ab,ij)  -> (ef,in)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H.aa.oovv, R.aa, aP.aa, optimize=True)
    #     <mn~|ef~> r(ef~,in~) a(ab,ij)  -> (ab,mj)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H.ab.oovv, R.ab, aP.aa, optimize=True)
    #     <mn||ef> r(af,in) a(ab,ij)  -> (eb,mj)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H.aa.oovv, R.aa, aP.aa, optimize=True)
    #     <mn||ef> r(eb,mj) a(ab,ij)  -> (af,in)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H.aa.oovv, R.aa, aP.aa, optimize=True)
    #     <mn~|ef~> r(af~,in~) a(ab,ij)  -> (eb,mj)
    X += np.einsum("mnef,afin,abij->ebmj", H.ab.oovv, R.ab, aP.aa, optimize=True)
    #     <mn||ef> r(ef,ij) a(ab,ij)  -> (ab,mn)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H.aa.oovv, R.aa, aP.aa, optimize=True)
    #     <mn||ef> r(ab,mn) a(ab,ij)  -> (ef,ij)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H.aa.oovv, R.aa, aP.aa, optimize=True)
    #     <mn||ef> r(eb~,ij~) a(ab~,ij~)  -> (af,mn)
    X -= 0.5 * np.einsum("mnef,ebij,abij->afmn", H.aa.oovv, R.ab, aP.ab, optimize=True)
    #     <mn||ef> r(ab~,mj~) a(ab~,ij~)  -> (ef,in)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H.aa.oovv, R.ab, aP.ab, optimize=True)
    #     <mn||ef> r(fb~,nj~) a(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H.aa.oovv, R.ab, aP.ab, optimize=True)
    #     <mn~|ef~> r(f~b~,n~j~) a(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H.ab.oovv, R.bb, aP.ab, optimize=True)
    # <a|[[Hbar, tau(ab,ij)], R1]|0>: the left-CCSD doubles terms of (a0 + A1 + A2) with H1R = [H(T1), R1]
    #     a0 <ij||ab>'
    X += 0.25 * a0 * H1R.aa.oovv.transpose(2, 3, 0, 1)
    #     f'(me) a(a,i)  -> (ae,im)
    X += np.einsum("me,ai->aeim", H1R.a.ov, aP.a, optimize=True)
    #     <mn||if>' a(a,i)  -> (af,mn)
    X -= 0.5 * np.einsum("mnif,ai->afmn", H1R.aa.ooov, aP.a, optimize=True)
    #     <an||ef>' a(a,i)  -> (ef,in)
    X += 0.5 * np.einsum("anef,ai->efin", H1R.aa.vovv, aP.a, optimize=True)
    #     f'(ae) a(ab,ij)  -> (eb,ij)
    X += 0.5 * np.einsum("ae,abij->ebij", H1R.a.vv, aP.aa, optimize=True)
    #     <mn||ef>' t(eb,ij) a(ab,ij)  -> (af,mn)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H1R.aa.oovv, T.aa, aP.aa, optimize=True)
    #     <mn||ef>' t(af,mn) a(ab,ij)  -> (eb,ij)
    X -= 0.25 * np.einsum("mnef,afmn,abij->ebij", H1R.aa.oovv, T.aa, aP.aa, optimize=True)
    #     <mn~|ef~>' t(af~,mn~) a(ab,ij)  -> (eb,ij)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H1R.ab.oovv, T.ab, aP.aa, optimize=True)
    #     f'(mi) a(ab,ij)  -> (ab,mj)
    X -= 0.5 * np.einsum("mi,abij->abmj", H1R.a.oo, aP.aa, optimize=True)
    #     <mn||ef>' t(ab,mj) a(ab,ij)  -> (ef,in)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H1R.aa.oovv, T.aa, aP.aa, optimize=True)
    #     <mn||ef>' t(ef,in) a(ab,ij)  -> (ab,mj)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H1R.aa.oovv, T.aa, aP.aa, optimize=True)
    #     <mn~|ef~>' t(ef~,in~) a(ab,ij)  -> (ab,mj)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H1R.ab.oovv, T.ab, aP.aa, optimize=True)
    #     <am||ie>' a(ab,ij)  -> (eb,mj)
    X += np.einsum("amie,abij->ebmj", H1R.aa.voov, aP.aa, optimize=True)
    #     <mn||ef>' t(eb,mj) a(ab,ij)  -> (af,in)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H1R.aa.oovv, T.aa, aP.aa, optimize=True)
    #     <mn||ef>' t(af,in) a(ab,ij)  -> (eb,mj)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H1R.aa.oovv, T.aa, aP.aa, optimize=True)
    #     <mn~|ef~>' t(af~,in~) a(ab,ij)  -> (eb,mj)
    X += np.einsum("mnef,afin,abij->ebmj", H1R.ab.oovv, T.ab, aP.aa, optimize=True)
    #     <mn||ij>' a(ab,ij)  -> (ab,mn)
    X += 0.125 * np.einsum("mnij,abij->abmn", H1R.aa.oooo, aP.aa, optimize=True)
    #     <mn||ef>' t(ab,mn) a(ab,ij)  -> (ef,ij)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H1R.aa.oovv, T.aa, aP.aa, optimize=True)
    #     <mn||ef>' t(ef,ij) a(ab,ij)  -> (ab,mn)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H1R.aa.oovv, T.aa, aP.aa, optimize=True)
    #     <mn||ef>' t(eb~,ij~) a(ab~,ij~)  -> (af,mn)
    X -= 0.5 * np.einsum("mnef,ebij,abij->afmn", H1R.aa.oovv, T.ab, aP.ab, optimize=True)
    #     <mn||ef>' t(ab~,mj~) a(ab~,ij~)  -> (ef,in)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H1R.aa.oovv, T.ab, aP.ab, optimize=True)
    #     <mn||ef>' t(fb~,nj~) a(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H1R.aa.oovv, T.ab, aP.ab, optimize=True)
    #     <mn~|ef~>' t(f~b~,n~j~) a(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H1R.ab.oovv, T.bb, aP.ab, optimize=True)
    #     <mb~|ej~>' a(ab~,ij~)  -> (ae,im)
    X += np.einsum("mbej,abij->aeim", H1R.ab.ovvo, aP.ab, optimize=True)
    # vvvv of H1R: 1/8 sum_ab L(abij) <ab||ef>' with <ab|ef>' = sum_x (dB_ae B_bf + B_ae dB_bf)
    for B1, B2 in ((H1R.chol.a.vv, H.chol.a.vv), (H.chol.a.vv, H1R.chol.a.vv)):
        Y = cholesky_ladder(aP.aa, B1, B2)
        X += 0.125 * (Y - Y.transpose(1, 0, 2, 3))
    # <L|[[Hbar, tau(ab,ij)], b2]|0> = <0|L2 [[(V_N T2^2)_C/2, tau(ab,ij)], b2]|0>
    #     <mn||ef> b(af,mn) l(ab,ij)  -> (eb,ij)
    X += -0.25 * np.einsum("mnef,afmn,abij->ebij", H.aa.oovv, bP.aa, L.aa, optimize=True)
    #     <mn||ef> b(eb,ij) l(ab,ij)  -> (af,mn)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H.aa.oovv, bP.aa, L.aa, optimize=True)
    #     <mn~|ef~> b(af~,mn~) l(ab,ij)  -> (eb,ij)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H.ab.oovv, bP.ab, L.aa, optimize=True)
    #     <mn||ef> b(ef,in) l(ab,ij)  -> (ab,mj)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H.aa.oovv, bP.aa, L.aa, optimize=True)
    #     <mn||ef> b(ab,mj) l(ab,ij)  -> (ef,in)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H.aa.oovv, bP.aa, L.aa, optimize=True)
    #     <mn~|ef~> b(ef~,in~) l(ab,ij)  -> (ab,mj)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H.ab.oovv, bP.ab, L.aa, optimize=True)
    #     <mn||ef> b(af,in) l(ab,ij)  -> (eb,mj)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H.aa.oovv, bP.aa, L.aa, optimize=True)
    #     <mn||ef> b(eb,mj) l(ab,ij)  -> (af,in)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H.aa.oovv, bP.aa, L.aa, optimize=True)
    #     <mn~|ef~> b(af~,in~) l(ab,ij)  -> (eb,mj)
    X += np.einsum("mnef,afin,abij->ebmj", H.ab.oovv, bP.ab, L.aa, optimize=True)
    #     <mn||ef> b(ef,ij) l(ab,ij)  -> (ab,mn)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H.aa.oovv, bP.aa, L.aa, optimize=True)
    #     <mn||ef> b(ab,mn) l(ab,ij)  -> (ef,ij)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H.aa.oovv, bP.aa, L.aa, optimize=True)
    #     <mn||ef> b(eb~,ij~) l(ab~,ij~)  -> (af,mn)
    X -= 0.5 * np.einsum("mnef,ebij,abij->afmn", H.aa.oovv, bP.ab, L.ab, optimize=True)
    #     <mn||ef> b(ab~,mj~) l(ab~,ij~)  -> (ef,in)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H.aa.oovv, bP.ab, L.ab, optimize=True)
    #     <mn||ef> b(fb~,nj~) l(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H.aa.oovv, bP.ab, L.ab, optimize=True)
    #     <mn~|ef~> b(f~b~,n~j~) l(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H.ab.oovv, bP.bb, L.ab, optimize=True)
    # <L|[[Hbar, tau(ab,ij)], b1]|0>: the left-CCSD doubles terms of (L1 + L2) with H1b = [H(T1), b1]
    #     f'(me) l(a,i)  -> (ae,im)
    X += np.einsum("me,ai->aeim", H1b.a.ov, L.a, optimize=True)
    #     <mn||if>' l(a,i)  -> (af,mn)
    X -= 0.5 * np.einsum("mnif,ai->afmn", H1b.aa.ooov, L.a, optimize=True)
    #     <an||ef>' l(a,i)  -> (ef,in)
    X += 0.5 * np.einsum("anef,ai->efin", H1b.aa.vovv, L.a, optimize=True)
    #     f'(ae) l(ab,ij)  -> (eb,ij)
    X += 0.5 * np.einsum("ae,abij->ebij", H1b.a.vv, L.aa, optimize=True)
    #     <mn||ef>' t(eb,ij) l(ab,ij)  -> (af,mn)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H1b.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn||ef>' t(af,mn) l(ab,ij)  -> (eb,ij)
    X -= 0.25 * np.einsum("mnef,afmn,abij->ebij", H1b.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn~|ef~>' t(af~,mn~) l(ab,ij)  -> (eb,ij)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H1b.ab.oovv, T.ab, L.aa, optimize=True)
    #     f'(mi) l(ab,ij)  -> (ab,mj)
    X -= 0.5 * np.einsum("mi,abij->abmj", H1b.a.oo, L.aa, optimize=True)
    #     <mn||ef>' t(ab,mj) l(ab,ij)  -> (ef,in)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H1b.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn||ef>' t(ef,in) l(ab,ij)  -> (ab,mj)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H1b.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn~|ef~>' t(ef~,in~) l(ab,ij)  -> (ab,mj)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H1b.ab.oovv, T.ab, L.aa, optimize=True)
    #     <am||ie>' l(ab,ij)  -> (eb,mj)
    X += np.einsum("amie,abij->ebmj", H1b.aa.voov, L.aa, optimize=True)
    #     <mn||ef>' t(eb,mj) l(ab,ij)  -> (af,in)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H1b.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn||ef>' t(af,in) l(ab,ij)  -> (eb,mj)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H1b.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn~|ef~>' t(af~,in~) l(ab,ij)  -> (eb,mj)
    X += np.einsum("mnef,afin,abij->ebmj", H1b.ab.oovv, T.ab, L.aa, optimize=True)
    #     <mn||ij>' l(ab,ij)  -> (ab,mn)
    X += 0.125 * np.einsum("mnij,abij->abmn", H1b.aa.oooo, L.aa, optimize=True)
    #     <mn||ef>' t(ab,mn) l(ab,ij)  -> (ef,ij)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H1b.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn||ef>' t(ef,ij) l(ab,ij)  -> (ab,mn)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H1b.aa.oovv, T.aa, L.aa, optimize=True)
    #     <mn||ef>' t(eb~,ij~) l(ab~,ij~)  -> (af,mn)
    X -= 0.5 * np.einsum("mnef,ebij,abij->afmn", H1b.aa.oovv, T.ab, L.ab, optimize=True)
    #     <mn||ef>' t(ab~,mj~) l(ab~,ij~)  -> (ef,in)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H1b.aa.oovv, T.ab, L.ab, optimize=True)
    #     <mn||ef>' t(fb~,nj~) l(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H1b.aa.oovv, T.ab, L.ab, optimize=True)
    #     <mn~|ef~>' t(f~b~,n~j~) l(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H1b.ab.oovv, T.bb, L.ab, optimize=True)
    #     <mb~|ej~>' l(ab~,ij~)  -> (ae,im)
    X += np.einsum("mbej,abij->aeim", H1b.ab.ovvo, L.ab, optimize=True)
    # vvvv of H1b: 1/8 sum_ab L(abij) <ab||ef>' with <ab|ef>' = sum_x (dB_ae B_bf + B_ae dB_bf)
    for B1, B2 in ((H1b.chol.a.vv, H.chol.a.vv), (H.chol.a.vv, H1b.chol.a.vv)):
        Y = cholesky_ladder(L.aa, B1, B2)
        X += 0.125 * (Y - Y.transpose(1, 0, 2, 3))
    X -= np.transpose(X, (1, 0, 2, 3))
    X -= np.transpose(X, (0, 1, 3, 2))
    # F_t, <0|(r0 A + b0 L + C)(Hbar - E0)|ijab> (left-CCSD LH_fun) and x0 <0|Hbar|ijab>
    X += F_t + LHy + x0 * Hbar.aa.oovv.transpose(2, 3, 0, 1)
    return X


def build_g_2B(F_t, LHy, x0, Hbar, R, aP, a0, bP, L, T, H, H1R, H1b):
    """g(ij~ab~) = F_t(ij~ab~) + <a|[Hbar, tau(ab~,ij~)]|R> + <L|[Hbar, tau(ab~,ij~)]|b>, the ab doubles
    of the source of the correction cluster multipliers (Eq. 7.13). With a = a0 <0| + A,
    b = b0 |0> + b1 + b2, C_J = <0|A R1|J> + <0|L b1|J> (singles only) and
    x0 = r0 a0 + (A . R) + (L . b),

        <a|[Hbar, tau(ab~,ij~)]|R> + <L|[Hbar, tau(ab~,ij~)]|b>
            = <0|(r0 A + b0 L + C)(Hbar - E0)|ij~ab~> + x0 <0|Hbar|ij~ab~>
              + <a|[[Hbar, tau(ab~,ij~)], R]|0> + <L|[[Hbar, tau(ab~,ij~)], b]|0>,

    the last two through the T2^2 terms of the CCSD equations (R2, b2) and the left-CCSD doubles
    terms with H(T1) -> [H(T1), R1], [H(T1), b1]. F_t holds the triples part (the T2 gradient of
    sum_K v_K M_K, (J^T c')_doubles and W <0|Hbar|ij~ab~>). Terms are built in the stored layout;
    the remaining ones are added at the end."""
    # <a|[[Hbar, tau(ab~,ij~)], R2]|0> = <0|A2 [[(V_N T2^2)_C/2, tau(ab~,ij~)], R2]|0>
    #     <mn~|ef~> r(eb,ij) a(ab,ij)  -> (af~,mn~)
    X = -0.5 * np.einsum("mnef,ebij,abij->afmn", H.ab.oovv, R.aa, aP.aa, optimize=True)
    #     <mn~|ef~> r(ab,mj) a(ab,ij)  -> (ef~,in~)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H.ab.oovv, R.aa, aP.aa, optimize=True)
    #     <mn~|ef~> r(eb,mj) a(ab,ij)  -> (af~,in~)
    X += np.einsum("mnef,ebmj,abij->afin", H.ab.oovv, R.aa, aP.aa, optimize=True)
    #     <m~n~||e~f~> r(af~,in~) a(ab,ij)  -> (be~,jm~)
    X += 0.5 * np.einsum("mnef,afin,abij->bejm", H.bb.oovv, R.ab, aP.aa, optimize=True)
    #     <m~n~||e~f~> r(be~,jm~) a(ab,ij)  -> (af~,in~)
    X += 0.5 * np.einsum("mnef,bejm,abij->afin", H.bb.oovv, R.ab, aP.aa, optimize=True)
    #     <mn||ef> r(af,mn) a(ab~,ij~)  -> (eb~,ij~)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H.aa.oovv, R.aa, aP.ab, optimize=True)
    #     <mn~|ef~> r(af~,mn~) a(ab~,ij~)  -> (eb~,ij~)
    X -= np.einsum("mnef,afmn,abij->ebij", H.ab.oovv, R.ab, aP.ab, optimize=True)
    #     <mn~|ef~> r(eb~,ij~) a(ab~,ij~)  -> (af~,mn~)
    X -= np.einsum("mnef,ebij,abij->afmn", H.ab.oovv, R.ab, aP.ab, optimize=True)
    #     <m~n~||e~f~> r(b~f~,m~n~) a(ab~,ij~)  -> (ae~,ij~)
    X -= 0.5 * np.einsum("mnef,bfmn,abij->aeij", H.bb.oovv, R.bb, aP.ab, optimize=True)
    #     <nm~|fe~> r(fb~,nm~) a(ab~,ij~)  -> (ae~,ij~)
    X -= np.einsum("nmfe,fbnm,abij->aeij", H.ab.oovv, R.ab, aP.ab, optimize=True)
    #     <nm~|fe~> r(ae~,ij~) a(ab~,ij~)  -> (fb~,nm~)
    X -= np.einsum("nmfe,aeij,abij->fbnm", H.ab.oovv, R.ab, aP.ab, optimize=True)
    #     <mn||ef> r(ef,in) a(ab~,ij~)  -> (ab~,mj~)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H.aa.oovv, R.aa, aP.ab, optimize=True)
    #     <mn~|ef~> r(ef~,in~) a(ab~,ij~)  -> (ab~,mj~)
    X -= np.einsum("mnef,efin,abij->abmj", H.ab.oovv, R.ab, aP.ab, optimize=True)
    #     <mn~|ef~> r(ab~,mj~) a(ab~,ij~)  -> (ef~,in~)
    X -= np.einsum("mnef,abmj,abij->efin", H.ab.oovv, R.ab, aP.ab, optimize=True)
    #     <m~n~||e~f~> r(e~f~,j~n~) a(ab~,ij~)  -> (ab~,im~)
    X -= 0.5 * np.einsum("mnef,efjn,abij->abim", H.bb.oovv, R.bb, aP.ab, optimize=True)
    #     <nm~|fe~> r(fe~,nj~) a(ab~,ij~)  -> (ab~,im~)
    X -= np.einsum("nmfe,fenj,abij->abim", H.ab.oovv, R.ab, aP.ab, optimize=True)
    #     <nm~|fe~> r(ab~,im~) a(ab~,ij~)  -> (fe~,nj~)
    X -= np.einsum("nmfe,abim,abij->fenj", H.ab.oovv, R.ab, aP.ab, optimize=True)
    #     <mn||ef> r(ae,im) a(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.aa.oovv, R.aa, aP.ab, optimize=True)
    #     <nm~|fe~> r(ae~,im~) a(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("nmfe,aeim,abij->fbnj", H.ab.oovv, R.ab, aP.ab, optimize=True)
    #     <nm~|fe~> r(fb~,nj~) a(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("nmfe,fbnj,abij->aeim", H.ab.oovv, R.ab, aP.ab, optimize=True)
    #     <m~n~||e~f~> r(f~b~,n~j~) a(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("mnef,fbnj,abij->aeim", H.bb.oovv, R.bb, aP.ab, optimize=True)
    #     <mn~|ef~> r(af~,mj~) a(ab~,ij~)  -> (eb~,in~)
    X += np.einsum("mnef,afmj,abij->ebin", H.ab.oovv, R.ab, aP.ab, optimize=True)
    #     <mn~|ef~> r(eb~,in~) a(ab~,ij~)  -> (af~,mj~)
    X += np.einsum("mnef,ebin,abij->afmj", H.ab.oovv, R.ab, aP.ab, optimize=True)
    #     <mn~|ef~> r(ef~,ij~) a(ab~,ij~)  -> (ab~,mn~)
    X += np.einsum("mnef,efij,abij->abmn", H.ab.oovv, R.ab, aP.ab, optimize=True)
    #     <mn~|ef~> r(ab~,mn~) a(ab~,ij~)  -> (ef~,ij~)
    X += np.einsum("mnef,abmn,abij->efij", H.ab.oovv, R.ab, aP.ab, optimize=True)
    #     <nm~|fe~> r(e~b~,i~j~) a(a~b~,i~j~)  -> (fa~,nm~)
    X -= 0.5 * np.einsum("nmfe,ebij,abij->fanm", H.ab.oovv, R.bb, aP.bb, optimize=True)
    #     <nm~|fe~> r(a~b~,m~j~) a(a~b~,i~j~)  -> (fe~,ni~)
    X -= 0.5 * np.einsum("nmfe,abmj,abij->feni", H.ab.oovv, R.bb, aP.bb, optimize=True)
    #     <mn~|ef~> r(a~f~,i~n~) a(a~b~,i~j~)  -> (eb~,mj~)
    X += np.einsum("mnef,afin,abij->ebmj", H.ab.oovv, R.bb, aP.bb, optimize=True)
    #     <mn||ef> r(fa~,ni~) a(a~b~,i~j~)  -> (eb~,mj~)
    X += 0.5 * np.einsum("mnef,fani,abij->ebmj", H.aa.oovv, R.ab, aP.bb, optimize=True)
    #     <mn||ef> r(eb~,mj~) a(a~b~,i~j~)  -> (fa~,ni~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->fani", H.aa.oovv, R.ab, aP.bb, optimize=True)
    # <a|[[Hbar, tau(ab~,ij~)], R1]|0>: the left-CCSD doubles terms of (a0 + A1 + A2) with H1R = [H(T1), R1]
    #     a0 <ij~|ab~>'
    X += a0 * H1R.ab.oovv.transpose(2, 3, 0, 1)
    #     f'(m~e~) a(a,i)  -> (ae~,im~)
    X += np.einsum("me,ai->aeim", H1R.b.ov, aP.a, optimize=True)
    #     <mn~|if~>' a(a,i)  -> (af~,mn~)
    X -= np.einsum("mnif,ai->afmn", H1R.ab.ooov, aP.a, optimize=True)
    #     <an~|ef~>' a(a,i)  -> (ef~,in~)
    X += np.einsum("anef,ai->efin", H1R.ab.vovv, aP.a, optimize=True)
    #     f'(me) a(a~,i~)  -> (ea~,mi~)
    X += np.einsum("me,ai->eami", H1R.a.ov, aP.b, optimize=True)
    #     <nm~|fi~>' a(a~,i~)  -> (fa~,nm~)
    X -= np.einsum("nmfi,ai->fanm", H1R.ab.oovo, aP.b, optimize=True)
    #     <na~|fe~>' a(a~,i~)  -> (fe~,ni~)
    X += np.einsum("nafe,ai->feni", H1R.ab.ovvv, aP.b, optimize=True)
    #     <mn~|ef~>' t(eb,ij) a(ab,ij)  -> (af~,mn~)
    X -= 0.5 * np.einsum("mnef,ebij,abij->afmn", H1R.ab.oovv, T.aa, aP.aa, optimize=True)
    #     <mn~|ef~>' t(ab,mj) a(ab,ij)  -> (ef~,in~)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H1R.ab.oovv, T.aa, aP.aa, optimize=True)
    #     <mn~|ef~>' t(eb,mj) a(ab,ij)  -> (af~,in~)
    X += np.einsum("mnef,ebmj,abij->afin", H1R.ab.oovv, T.aa, aP.aa, optimize=True)
    #     <am~|ie~>' a(ab,ij)  -> (be~,jm~)
    X += np.einsum("amie,abij->bejm", H1R.ab.voov, aP.aa, optimize=True)
    #     <m~n~||e~f~>' t(be~,jm~) a(ab,ij)  -> (af~,in~)
    X += 0.5 * np.einsum("mnef,bejm,abij->afin", H1R.bb.oovv, T.ab, aP.aa, optimize=True)
    #     <m~n~||e~f~>' t(af~,in~) a(ab,ij)  -> (be~,jm~)
    X += 0.5 * np.einsum("mnef,afin,abij->bejm", H1R.bb.oovv, T.ab, aP.aa, optimize=True)
    #     f'(ae) a(ab~,ij~)  -> (eb~,ij~)
    X += np.einsum("ae,abij->ebij", H1R.a.vv, aP.ab, optimize=True)
    #     <mn||ef>' t(af,mn) a(ab~,ij~)  -> (eb~,ij~)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H1R.aa.oovv, T.aa, aP.ab, optimize=True)
    #     <mn~|ef~>' t(eb~,ij~) a(ab~,ij~)  -> (af~,mn~)
    X -= np.einsum("mnef,ebij,abij->afmn", H1R.ab.oovv, T.ab, aP.ab, optimize=True)
    #     <mn~|ef~>' t(af~,mn~) a(ab~,ij~)  -> (eb~,ij~)
    X -= np.einsum("mnef,afmn,abij->ebij", H1R.ab.oovv, T.ab, aP.ab, optimize=True)
    #     f'(b~e~) a(ab~,ij~)  -> (ae~,ij~)
    X += np.einsum("be,abij->aeij", H1R.b.vv, aP.ab, optimize=True)
    #     <m~n~||e~f~>' t(b~f~,m~n~) a(ab~,ij~)  -> (ae~,ij~)
    X -= 0.5 * np.einsum("mnef,bfmn,abij->aeij", H1R.bb.oovv, T.bb, aP.ab, optimize=True)
    #     <nm~|fe~>' t(ae~,ij~) a(ab~,ij~)  -> (fb~,nm~)
    X -= np.einsum("nmfe,aeij,abij->fbnm", H1R.ab.oovv, T.ab, aP.ab, optimize=True)
    #     <nm~|fe~>' t(fb~,nm~) a(ab~,ij~)  -> (ae~,ij~)
    X -= np.einsum("nmfe,fbnm,abij->aeij", H1R.ab.oovv, T.ab, aP.ab, optimize=True)
    #     f'(mi) a(ab~,ij~)  -> (ab~,mj~)
    X -= np.einsum("mi,abij->abmj", H1R.a.oo, aP.ab, optimize=True)
    #     <mn||ef>' t(ef,in) a(ab~,ij~)  -> (ab~,mj~)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H1R.aa.oovv, T.aa, aP.ab, optimize=True)
    #     <mn~|ef~>' t(ab~,mj~) a(ab~,ij~)  -> (ef~,in~)
    X -= np.einsum("mnef,abmj,abij->efin", H1R.ab.oovv, T.ab, aP.ab, optimize=True)
    #     <mn~|ef~>' t(ef~,in~) a(ab~,ij~)  -> (ab~,mj~)
    X -= np.einsum("mnef,efin,abij->abmj", H1R.ab.oovv, T.ab, aP.ab, optimize=True)
    #     f'(m~j~) a(ab~,ij~)  -> (ab~,im~)
    X -= np.einsum("mj,abij->abim", H1R.b.oo, aP.ab, optimize=True)
    #     <m~n~||e~f~>' t(e~f~,j~n~) a(ab~,ij~)  -> (ab~,im~)
    X -= 0.5 * np.einsum("mnef,efjn,abij->abim", H1R.bb.oovv, T.bb, aP.ab, optimize=True)
    #     <nm~|fe~>' t(ab~,im~) a(ab~,ij~)  -> (fe~,nj~)
    X -= np.einsum("nmfe,abim,abij->fenj", H1R.ab.oovv, T.ab, aP.ab, optimize=True)
    #     <nm~|fe~>' t(fe~,nj~) a(ab~,ij~)  -> (ab~,im~)
    X -= np.einsum("nmfe,fenj,abij->abim", H1R.ab.oovv, T.ab, aP.ab, optimize=True)
    #     <am||ie>' a(ab~,ij~)  -> (eb~,mj~)
    X += np.einsum("amie,abij->ebmj", H1R.aa.voov, aP.ab, optimize=True)
    #     <mn||ef>' t(ae,im) a(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("mnef,aeim,abij->fbnj", H1R.aa.oovv, T.aa, aP.ab, optimize=True)
    #     <nm~|fe~>' t(fb~,nj~) a(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("nmfe,fbnj,abij->aeim", H1R.ab.oovv, T.ab, aP.ab, optimize=True)
    #     <nm~|fe~>' t(ae~,im~) a(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("nmfe,aeim,abij->fbnj", H1R.ab.oovv, T.ab, aP.ab, optimize=True)
    #     <m~n~||e~f~>' t(f~b~,n~j~) a(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("mnef,fbnj,abij->aeim", H1R.bb.oovv, T.bb, aP.ab, optimize=True)
    #     <b~m~||j~e~>' a(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("bmje,abij->aeim", H1R.bb.voov, aP.ab, optimize=True)
    #     <mb~|ie~>' a(ab~,ij~)  -> (ae~,mj~)
    X -= np.einsum("mbie,abij->aemj", H1R.ab.ovov, aP.ab, optimize=True)
    #     <am~|ej~>' a(ab~,ij~)  -> (eb~,im~)
    X -= np.einsum("amej,abij->ebim", H1R.ab.vovo, aP.ab, optimize=True)
    #     <mn~|ef~>' t(eb~,in~) a(ab~,ij~)  -> (af~,mj~)
    X += np.einsum("mnef,ebin,abij->afmj", H1R.ab.oovv, T.ab, aP.ab, optimize=True)
    #     <mn~|ef~>' t(af~,mj~) a(ab~,ij~)  -> (eb~,in~)
    X += np.einsum("mnef,afmj,abij->ebin", H1R.ab.oovv, T.ab, aP.ab, optimize=True)
    #     <mn~|ij~>' a(ab~,ij~)  -> (ab~,mn~)
    X += np.einsum("mnij,abij->abmn", H1R.ab.oooo, aP.ab, optimize=True)
    #     <mn~|ef~>' t(ab~,mn~) a(ab~,ij~)  -> (ef~,ij~)
    X += np.einsum("mnef,abmn,abij->efij", H1R.ab.oovv, T.ab, aP.ab, optimize=True)
    #     <mn~|ef~>' t(ef~,ij~) a(ab~,ij~)  -> (ab~,mn~)
    X += np.einsum("mnef,efij,abij->abmn", H1R.ab.oovv, T.ab, aP.ab, optimize=True)
    #     <nm~|fe~>' t(e~b~,i~j~) a(a~b~,i~j~)  -> (fa~,nm~)
    X -= 0.5 * np.einsum("nmfe,ebij,abij->fanm", H1R.ab.oovv, T.bb, aP.bb, optimize=True)
    #     <nm~|fe~>' t(a~b~,m~j~) a(a~b~,i~j~)  -> (fe~,ni~)
    X -= 0.5 * np.einsum("nmfe,abmj,abij->feni", H1R.ab.oovv, T.bb, aP.bb, optimize=True)
    #     <ma~|ei~>' a(a~b~,i~j~)  -> (eb~,mj~)
    X += np.einsum("maei,abij->ebmj", H1R.ab.ovvo, aP.bb, optimize=True)
    #     <mn~|ef~>' t(a~f~,i~n~) a(a~b~,i~j~)  -> (eb~,mj~)
    X += np.einsum("mnef,afin,abij->ebmj", H1R.ab.oovv, T.bb, aP.bb, optimize=True)
    #     <mn||ef>' t(eb~,mj~) a(a~b~,i~j~)  -> (fa~,ni~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->fani", H1R.aa.oovv, T.ab, aP.bb, optimize=True)
    #     <mn||ef>' t(fa~,ni~) a(a~b~,i~j~)  -> (eb~,mj~)
    X += 0.5 * np.einsum("mnef,fani,abij->ebmj", H1R.aa.oovv, T.ab, aP.bb, optimize=True)
    # vvvv of H1R: sum_ab L(abij) <ab|ef>'
    for B1, B2 in ((H1R.chol.a.vv, H.chol.b.vv), (H.chol.a.vv, H1R.chol.b.vv)):
        X += cholesky_ladder(aP.ab, B1, B2)
    # <L|[[Hbar, tau(ab~,ij~)], b2]|0> = <0|L2 [[(V_N T2^2)_C/2, tau(ab~,ij~)], b2]|0>
    #     <mn~|ef~> b(eb,ij) l(ab,ij)  -> (af~,mn~)
    X += -0.5 * np.einsum("mnef,ebij,abij->afmn", H.ab.oovv, bP.aa, L.aa, optimize=True)
    #     <mn~|ef~> b(ab,mj) l(ab,ij)  -> (ef~,in~)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H.ab.oovv, bP.aa, L.aa, optimize=True)
    #     <mn~|ef~> b(eb,mj) l(ab,ij)  -> (af~,in~)
    X += np.einsum("mnef,ebmj,abij->afin", H.ab.oovv, bP.aa, L.aa, optimize=True)
    #     <m~n~||e~f~> b(af~,in~) l(ab,ij)  -> (be~,jm~)
    X += 0.5 * np.einsum("mnef,afin,abij->bejm", H.bb.oovv, bP.ab, L.aa, optimize=True)
    #     <m~n~||e~f~> b(be~,jm~) l(ab,ij)  -> (af~,in~)
    X += 0.5 * np.einsum("mnef,bejm,abij->afin", H.bb.oovv, bP.ab, L.aa, optimize=True)
    #     <mn||ef> b(af,mn) l(ab~,ij~)  -> (eb~,ij~)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H.aa.oovv, bP.aa, L.ab, optimize=True)
    #     <mn~|ef~> b(af~,mn~) l(ab~,ij~)  -> (eb~,ij~)
    X -= np.einsum("mnef,afmn,abij->ebij", H.ab.oovv, bP.ab, L.ab, optimize=True)
    #     <mn~|ef~> b(eb~,ij~) l(ab~,ij~)  -> (af~,mn~)
    X -= np.einsum("mnef,ebij,abij->afmn", H.ab.oovv, bP.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> b(b~f~,m~n~) l(ab~,ij~)  -> (ae~,ij~)
    X -= 0.5 * np.einsum("mnef,bfmn,abij->aeij", H.bb.oovv, bP.bb, L.ab, optimize=True)
    #     <nm~|fe~> b(fb~,nm~) l(ab~,ij~)  -> (ae~,ij~)
    X -= np.einsum("nmfe,fbnm,abij->aeij", H.ab.oovv, bP.ab, L.ab, optimize=True)
    #     <nm~|fe~> b(ae~,ij~) l(ab~,ij~)  -> (fb~,nm~)
    X -= np.einsum("nmfe,aeij,abij->fbnm", H.ab.oovv, bP.ab, L.ab, optimize=True)
    #     <mn||ef> b(ef,in) l(ab~,ij~)  -> (ab~,mj~)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H.aa.oovv, bP.aa, L.ab, optimize=True)
    #     <mn~|ef~> b(ef~,in~) l(ab~,ij~)  -> (ab~,mj~)
    X -= np.einsum("mnef,efin,abij->abmj", H.ab.oovv, bP.ab, L.ab, optimize=True)
    #     <mn~|ef~> b(ab~,mj~) l(ab~,ij~)  -> (ef~,in~)
    X -= np.einsum("mnef,abmj,abij->efin", H.ab.oovv, bP.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> b(e~f~,j~n~) l(ab~,ij~)  -> (ab~,im~)
    X -= 0.5 * np.einsum("mnef,efjn,abij->abim", H.bb.oovv, bP.bb, L.ab, optimize=True)
    #     <nm~|fe~> b(fe~,nj~) l(ab~,ij~)  -> (ab~,im~)
    X -= np.einsum("nmfe,fenj,abij->abim", H.ab.oovv, bP.ab, L.ab, optimize=True)
    #     <nm~|fe~> b(ab~,im~) l(ab~,ij~)  -> (fe~,nj~)
    X -= np.einsum("nmfe,abim,abij->fenj", H.ab.oovv, bP.ab, L.ab, optimize=True)
    #     <mn||ef> b(ae,im) l(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.aa.oovv, bP.aa, L.ab, optimize=True)
    #     <nm~|fe~> b(ae~,im~) l(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("nmfe,aeim,abij->fbnj", H.ab.oovv, bP.ab, L.ab, optimize=True)
    #     <nm~|fe~> b(fb~,nj~) l(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("nmfe,fbnj,abij->aeim", H.ab.oovv, bP.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> b(f~b~,n~j~) l(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("mnef,fbnj,abij->aeim", H.bb.oovv, bP.bb, L.ab, optimize=True)
    #     <mn~|ef~> b(af~,mj~) l(ab~,ij~)  -> (eb~,in~)
    X += np.einsum("mnef,afmj,abij->ebin", H.ab.oovv, bP.ab, L.ab, optimize=True)
    #     <mn~|ef~> b(eb~,in~) l(ab~,ij~)  -> (af~,mj~)
    X += np.einsum("mnef,ebin,abij->afmj", H.ab.oovv, bP.ab, L.ab, optimize=True)
    #     <mn~|ef~> b(ef~,ij~) l(ab~,ij~)  -> (ab~,mn~)
    X += np.einsum("mnef,efij,abij->abmn", H.ab.oovv, bP.ab, L.ab, optimize=True)
    #     <mn~|ef~> b(ab~,mn~) l(ab~,ij~)  -> (ef~,ij~)
    X += np.einsum("mnef,abmn,abij->efij", H.ab.oovv, bP.ab, L.ab, optimize=True)
    #     <nm~|fe~> b(e~b~,i~j~) l(a~b~,i~j~)  -> (fa~,nm~)
    X -= 0.5 * np.einsum("nmfe,ebij,abij->fanm", H.ab.oovv, bP.bb, L.bb, optimize=True)
    #     <nm~|fe~> b(a~b~,m~j~) l(a~b~,i~j~)  -> (fe~,ni~)
    X -= 0.5 * np.einsum("nmfe,abmj,abij->feni", H.ab.oovv, bP.bb, L.bb, optimize=True)
    #     <mn~|ef~> b(a~f~,i~n~) l(a~b~,i~j~)  -> (eb~,mj~)
    X += np.einsum("mnef,afin,abij->ebmj", H.ab.oovv, bP.bb, L.bb, optimize=True)
    #     <mn||ef> b(fa~,ni~) l(a~b~,i~j~)  -> (eb~,mj~)
    X += 0.5 * np.einsum("mnef,fani,abij->ebmj", H.aa.oovv, bP.ab, L.bb, optimize=True)
    #     <mn||ef> b(eb~,mj~) l(a~b~,i~j~)  -> (fa~,ni~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->fani", H.aa.oovv, bP.ab, L.bb, optimize=True)
    # <L|[[Hbar, tau(ab~,ij~)], b1]|0>: the left-CCSD doubles terms of (L1 + L2) with H1b = [H(T1), b1]
    #     f'(m~e~) l(a,i)  -> (ae~,im~)
    X += np.einsum("me,ai->aeim", H1b.b.ov, L.a, optimize=True)
    #     <mn~|if~>' l(a,i)  -> (af~,mn~)
    X -= np.einsum("mnif,ai->afmn", H1b.ab.ooov, L.a, optimize=True)
    #     <an~|ef~>' l(a,i)  -> (ef~,in~)
    X += np.einsum("anef,ai->efin", H1b.ab.vovv, L.a, optimize=True)
    #     f'(me) l(a~,i~)  -> (ea~,mi~)
    X += np.einsum("me,ai->eami", H1b.a.ov, L.b, optimize=True)
    #     <nm~|fi~>' l(a~,i~)  -> (fa~,nm~)
    X -= np.einsum("nmfi,ai->fanm", H1b.ab.oovo, L.b, optimize=True)
    #     <na~|fe~>' l(a~,i~)  -> (fe~,ni~)
    X += np.einsum("nafe,ai->feni", H1b.ab.ovvv, L.b, optimize=True)
    #     <mn~|ef~>' t(eb,ij) l(ab,ij)  -> (af~,mn~)
    X -= 0.5 * np.einsum("mnef,ebij,abij->afmn", H1b.ab.oovv, T.aa, L.aa, optimize=True)
    #     <mn~|ef~>' t(ab,mj) l(ab,ij)  -> (ef~,in~)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H1b.ab.oovv, T.aa, L.aa, optimize=True)
    #     <mn~|ef~>' t(eb,mj) l(ab,ij)  -> (af~,in~)
    X += np.einsum("mnef,ebmj,abij->afin", H1b.ab.oovv, T.aa, L.aa, optimize=True)
    #     <am~|ie~>' l(ab,ij)  -> (be~,jm~)
    X += np.einsum("amie,abij->bejm", H1b.ab.voov, L.aa, optimize=True)
    #     <m~n~||e~f~>' t(be~,jm~) l(ab,ij)  -> (af~,in~)
    X += 0.5 * np.einsum("mnef,bejm,abij->afin", H1b.bb.oovv, T.ab, L.aa, optimize=True)
    #     <m~n~||e~f~>' t(af~,in~) l(ab,ij)  -> (be~,jm~)
    X += 0.5 * np.einsum("mnef,afin,abij->bejm", H1b.bb.oovv, T.ab, L.aa, optimize=True)
    #     f'(ae) l(ab~,ij~)  -> (eb~,ij~)
    X += np.einsum("ae,abij->ebij", H1b.a.vv, L.ab, optimize=True)
    #     <mn||ef>' t(af,mn) l(ab~,ij~)  -> (eb~,ij~)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H1b.aa.oovv, T.aa, L.ab, optimize=True)
    #     <mn~|ef~>' t(eb~,ij~) l(ab~,ij~)  -> (af~,mn~)
    X -= np.einsum("mnef,ebij,abij->afmn", H1b.ab.oovv, T.ab, L.ab, optimize=True)
    #     <mn~|ef~>' t(af~,mn~) l(ab~,ij~)  -> (eb~,ij~)
    X -= np.einsum("mnef,afmn,abij->ebij", H1b.ab.oovv, T.ab, L.ab, optimize=True)
    #     f'(b~e~) l(ab~,ij~)  -> (ae~,ij~)
    X += np.einsum("be,abij->aeij", H1b.b.vv, L.ab, optimize=True)
    #     <m~n~||e~f~>' t(b~f~,m~n~) l(ab~,ij~)  -> (ae~,ij~)
    X -= 0.5 * np.einsum("mnef,bfmn,abij->aeij", H1b.bb.oovv, T.bb, L.ab, optimize=True)
    #     <nm~|fe~>' t(ae~,ij~) l(ab~,ij~)  -> (fb~,nm~)
    X -= np.einsum("nmfe,aeij,abij->fbnm", H1b.ab.oovv, T.ab, L.ab, optimize=True)
    #     <nm~|fe~>' t(fb~,nm~) l(ab~,ij~)  -> (ae~,ij~)
    X -= np.einsum("nmfe,fbnm,abij->aeij", H1b.ab.oovv, T.ab, L.ab, optimize=True)
    #     f'(mi) l(ab~,ij~)  -> (ab~,mj~)
    X -= np.einsum("mi,abij->abmj", H1b.a.oo, L.ab, optimize=True)
    #     <mn||ef>' t(ef,in) l(ab~,ij~)  -> (ab~,mj~)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H1b.aa.oovv, T.aa, L.ab, optimize=True)
    #     <mn~|ef~>' t(ab~,mj~) l(ab~,ij~)  -> (ef~,in~)
    X -= np.einsum("mnef,abmj,abij->efin", H1b.ab.oovv, T.ab, L.ab, optimize=True)
    #     <mn~|ef~>' t(ef~,in~) l(ab~,ij~)  -> (ab~,mj~)
    X -= np.einsum("mnef,efin,abij->abmj", H1b.ab.oovv, T.ab, L.ab, optimize=True)
    #     f'(m~j~) l(ab~,ij~)  -> (ab~,im~)
    X -= np.einsum("mj,abij->abim", H1b.b.oo, L.ab, optimize=True)
    #     <m~n~||e~f~>' t(e~f~,j~n~) l(ab~,ij~)  -> (ab~,im~)
    X -= 0.5 * np.einsum("mnef,efjn,abij->abim", H1b.bb.oovv, T.bb, L.ab, optimize=True)
    #     <nm~|fe~>' t(ab~,im~) l(ab~,ij~)  -> (fe~,nj~)
    X -= np.einsum("nmfe,abim,abij->fenj", H1b.ab.oovv, T.ab, L.ab, optimize=True)
    #     <nm~|fe~>' t(fe~,nj~) l(ab~,ij~)  -> (ab~,im~)
    X -= np.einsum("nmfe,fenj,abij->abim", H1b.ab.oovv, T.ab, L.ab, optimize=True)
    #     <am||ie>' l(ab~,ij~)  -> (eb~,mj~)
    X += np.einsum("amie,abij->ebmj", H1b.aa.voov, L.ab, optimize=True)
    #     <mn||ef>' t(ae,im) l(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("mnef,aeim,abij->fbnj", H1b.aa.oovv, T.aa, L.ab, optimize=True)
    #     <nm~|fe~>' t(fb~,nj~) l(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("nmfe,fbnj,abij->aeim", H1b.ab.oovv, T.ab, L.ab, optimize=True)
    #     <nm~|fe~>' t(ae~,im~) l(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("nmfe,aeim,abij->fbnj", H1b.ab.oovv, T.ab, L.ab, optimize=True)
    #     <m~n~||e~f~>' t(f~b~,n~j~) l(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("mnef,fbnj,abij->aeim", H1b.bb.oovv, T.bb, L.ab, optimize=True)
    #     <b~m~||j~e~>' l(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("bmje,abij->aeim", H1b.bb.voov, L.ab, optimize=True)
    #     <mb~|ie~>' l(ab~,ij~)  -> (ae~,mj~)
    X -= np.einsum("mbie,abij->aemj", H1b.ab.ovov, L.ab, optimize=True)
    #     <am~|ej~>' l(ab~,ij~)  -> (eb~,im~)
    X -= np.einsum("amej,abij->ebim", H1b.ab.vovo, L.ab, optimize=True)
    #     <mn~|ef~>' t(eb~,in~) l(ab~,ij~)  -> (af~,mj~)
    X += np.einsum("mnef,ebin,abij->afmj", H1b.ab.oovv, T.ab, L.ab, optimize=True)
    #     <mn~|ef~>' t(af~,mj~) l(ab~,ij~)  -> (eb~,in~)
    X += np.einsum("mnef,afmj,abij->ebin", H1b.ab.oovv, T.ab, L.ab, optimize=True)
    #     <mn~|ij~>' l(ab~,ij~)  -> (ab~,mn~)
    X += np.einsum("mnij,abij->abmn", H1b.ab.oooo, L.ab, optimize=True)
    #     <mn~|ef~>' t(ab~,mn~) l(ab~,ij~)  -> (ef~,ij~)
    X += np.einsum("mnef,abmn,abij->efij", H1b.ab.oovv, T.ab, L.ab, optimize=True)
    #     <mn~|ef~>' t(ef~,ij~) l(ab~,ij~)  -> (ab~,mn~)
    X += np.einsum("mnef,efij,abij->abmn", H1b.ab.oovv, T.ab, L.ab, optimize=True)
    #     <nm~|fe~>' t(e~b~,i~j~) l(a~b~,i~j~)  -> (fa~,nm~)
    X -= 0.5 * np.einsum("nmfe,ebij,abij->fanm", H1b.ab.oovv, T.bb, L.bb, optimize=True)
    #     <nm~|fe~>' t(a~b~,m~j~) l(a~b~,i~j~)  -> (fe~,ni~)
    X -= 0.5 * np.einsum("nmfe,abmj,abij->feni", H1b.ab.oovv, T.bb, L.bb, optimize=True)
    #     <ma~|ei~>' l(a~b~,i~j~)  -> (eb~,mj~)
    X += np.einsum("maei,abij->ebmj", H1b.ab.ovvo, L.bb, optimize=True)
    #     <mn~|ef~>' t(a~f~,i~n~) l(a~b~,i~j~)  -> (eb~,mj~)
    X += np.einsum("mnef,afin,abij->ebmj", H1b.ab.oovv, T.bb, L.bb, optimize=True)
    #     <mn||ef>' t(eb~,mj~) l(a~b~,i~j~)  -> (fa~,ni~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->fani", H1b.aa.oovv, T.ab, L.bb, optimize=True)
    #     <mn||ef>' t(fa~,ni~) l(a~b~,i~j~)  -> (eb~,mj~)
    X += 0.5 * np.einsum("mnef,fani,abij->ebmj", H1b.aa.oovv, T.ab, L.bb, optimize=True)
    # vvvv of H1b: sum_ab L(abij) <ab|ef>'
    for B1, B2 in ((H1b.chol.a.vv, H.chol.b.vv), (H.chol.a.vv, H1b.chol.b.vv)):
        X += cholesky_ladder(L.ab, B1, B2)
    # F_t, <0|(r0 A + b0 L + C)(Hbar - E0)|ij~ab~> (left-CCSD LH_fun) and x0 <0|Hbar|ij~ab~>
    X += F_t + LHy + x0 * Hbar.ab.oovv.transpose(2, 3, 0, 1)
    return X


def build_g_2C(F_t, LHy, x0, Hbar, R, aP, a0, bP, L, T, H, H1R, H1b):
    """g(i~j~a~b~) = F_t(i~j~a~b~) + <a|[Hbar, tau(a~b~,i~j~)]|R> + <L|[Hbar, tau(a~b~,i~j~)]|b>, the bb doubles
    of the source of the correction cluster multipliers (Eq. 7.13). With a = a0 <0| + A,
    b = b0 |0> + b1 + b2, C_J = <0|A R1|J> + <0|L b1|J> (singles only) and
    x0 = r0 a0 + (A . R) + (L . b),

        <a|[Hbar, tau(a~b~,i~j~)]|R> + <L|[Hbar, tau(a~b~,i~j~)]|b>
            = <0|(r0 A + b0 L + C)(Hbar - E0)|i~j~a~b~> + x0 <0|Hbar|i~j~a~b~>
              + <a|[[Hbar, tau(a~b~,i~j~)], R]|0> + <L|[[Hbar, tau(a~b~,i~j~)], b]|0>,

    the last two through the T2^2 terms of the CCSD equations (R2, b2) and the left-CCSD doubles
    terms with H(T1) -> [H(T1), R1], [H(T1), b1]. F_t holds the triples part (the T2 gradient of
    sum_K v_K M_K, (J^T c')_doubles and W <0|Hbar|i~j~a~b~>). Terms are built in the stored layout and
    antisymmetrized before the (already antisymmetric) remaining ones are added."""
    # <a|[[Hbar, tau(a~b~,i~j~)], R2]|0> = <0|A2 [[(V_N T2^2)_C/2, tau(a~b~,i~j~)], R2]|0>
    #     <m~n~||e~f~> r(ae~,ij~) a(ab~,ij~)  -> (b~f~,m~n~)
    X = -0.5 * np.einsum("mnef,aeij,abij->bfmn", H.bb.oovv, R.ab, aP.ab, optimize=True)
    #     <m~n~||e~f~> r(ab~,im~) a(ab~,ij~)  -> (e~f~,j~n~)
    X -= 0.5 * np.einsum("mnef,abim,abij->efjn", H.bb.oovv, R.ab, aP.ab, optimize=True)
    #     <mn~|ef~> r(ae,im) a(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.ab.oovv, R.aa, aP.ab, optimize=True)
    #     <m~n~||e~f~> r(ae~,im~) a(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.bb.oovv, R.ab, aP.ab, optimize=True)
    #     <m~n~||e~f~> r(a~f~,m~n~) a(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.25 * np.einsum("mnef,afmn,abij->ebij", H.bb.oovv, R.bb, aP.bb, optimize=True)
    #     <m~n~||e~f~> r(e~b~,i~j~) a(a~b~,i~j~)  -> (a~f~,m~n~)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H.bb.oovv, R.bb, aP.bb, optimize=True)
    #     <nm~|fe~> r(fa~,nm~) a(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.5 * np.einsum("nmfe,fanm,abij->ebij", H.ab.oovv, R.ab, aP.bb, optimize=True)
    #     <m~n~||e~f~> r(e~f~,i~n~) a(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H.bb.oovv, R.bb, aP.bb, optimize=True)
    #     <m~n~||e~f~> r(a~b~,m~j~) a(a~b~,i~j~)  -> (e~f~,i~n~)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H.bb.oovv, R.bb, aP.bb, optimize=True)
    #     <nm~|fe~> r(fe~,ni~) a(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.5 * np.einsum("nmfe,feni,abij->abmj", H.ab.oovv, R.ab, aP.bb, optimize=True)
    #     <m~n~||e~f~> r(a~f~,i~n~) a(a~b~,i~j~)  -> (e~b~,m~j~)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H.bb.oovv, R.bb, aP.bb, optimize=True)
    #     <m~n~||e~f~> r(e~b~,m~j~) a(a~b~,i~j~)  -> (a~f~,i~n~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H.bb.oovv, R.bb, aP.bb, optimize=True)
    #     <mn~|ef~> r(eb~,mj~) a(a~b~,i~j~)  -> (a~f~,i~n~)
    X += np.einsum("mnef,ebmj,abij->afin", H.ab.oovv, R.ab, aP.bb, optimize=True)
    #     <m~n~||e~f~> r(e~f~,i~j~) a(a~b~,i~j~)  -> (a~b~,m~n~)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H.bb.oovv, R.bb, aP.bb, optimize=True)
    #     <m~n~||e~f~> r(a~b~,m~n~) a(a~b~,i~j~)  -> (e~f~,i~j~)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H.bb.oovv, R.bb, aP.bb, optimize=True)
    # <a|[[Hbar, tau(a~b~,i~j~)], R1]|0>: the left-CCSD doubles terms of (a0 + A1 + A2) with H1R = [H(T1), R1]
    #     a0 <i~j~||a~b~>'
    X += 0.25 * a0 * H1R.bb.oovv.transpose(2, 3, 0, 1)
    #     f'(m~e~) a(a~,i~)  -> (a~e~,i~m~)
    X += np.einsum("me,ai->aeim", H1R.b.ov, aP.b, optimize=True)
    #     <m~n~||i~f~>' a(a~,i~)  -> (a~f~,m~n~)
    X -= 0.5 * np.einsum("mnif,ai->afmn", H1R.bb.ooov, aP.b, optimize=True)
    #     <a~n~||e~f~>' a(a~,i~)  -> (e~f~,i~n~)
    X += 0.5 * np.einsum("anef,ai->efin", H1R.bb.vovv, aP.b, optimize=True)
    #     <m~n~||e~f~>' t(ae~,ij~) a(ab~,ij~)  -> (b~f~,m~n~)
    X -= 0.5 * np.einsum("mnef,aeij,abij->bfmn", H1R.bb.oovv, T.ab, aP.ab, optimize=True)
    #     <m~n~||e~f~>' t(ab~,im~) a(ab~,ij~)  -> (e~f~,j~n~)
    X -= 0.5 * np.einsum("mnef,abim,abij->efjn", H1R.bb.oovv, T.ab, aP.ab, optimize=True)
    #     <am~|ie~>' a(ab~,ij~)  -> (e~b~,m~j~)
    X += np.einsum("amie,abij->ebmj", H1R.ab.voov, aP.ab, optimize=True)
    #     <mn~|ef~>' t(ae,im) a(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H1R.ab.oovv, T.aa, aP.ab, optimize=True)
    #     <m~n~||e~f~>' t(ae~,im~) a(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H1R.bb.oovv, T.ab, aP.ab, optimize=True)
    #     f'(a~e~) a(a~b~,i~j~)  -> (e~b~,i~j~)
    X += 0.5 * np.einsum("ae,abij->ebij", H1R.b.vv, aP.bb, optimize=True)
    #     <m~n~||e~f~>' t(e~b~,i~j~) a(a~b~,i~j~)  -> (a~f~,m~n~)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H1R.bb.oovv, T.bb, aP.bb, optimize=True)
    #     <m~n~||e~f~>' t(a~f~,m~n~) a(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.25 * np.einsum("mnef,afmn,abij->ebij", H1R.bb.oovv, T.bb, aP.bb, optimize=True)
    #     <nm~|fe~>' t(fa~,nm~) a(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.5 * np.einsum("nmfe,fanm,abij->ebij", H1R.ab.oovv, T.ab, aP.bb, optimize=True)
    #     f'(m~i~) a(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.5 * np.einsum("mi,abij->abmj", H1R.b.oo, aP.bb, optimize=True)
    #     <m~n~||e~f~>' t(a~b~,m~j~) a(a~b~,i~j~)  -> (e~f~,i~n~)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H1R.bb.oovv, T.bb, aP.bb, optimize=True)
    #     <m~n~||e~f~>' t(e~f~,i~n~) a(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H1R.bb.oovv, T.bb, aP.bb, optimize=True)
    #     <nm~|fe~>' t(fe~,ni~) a(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.5 * np.einsum("nmfe,feni,abij->abmj", H1R.ab.oovv, T.ab, aP.bb, optimize=True)
    #     <a~m~||i~e~>' a(a~b~,i~j~)  -> (e~b~,m~j~)
    X += np.einsum("amie,abij->ebmj", H1R.bb.voov, aP.bb, optimize=True)
    #     <m~n~||e~f~>' t(e~b~,m~j~) a(a~b~,i~j~)  -> (a~f~,i~n~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H1R.bb.oovv, T.bb, aP.bb, optimize=True)
    #     <m~n~||e~f~>' t(a~f~,i~n~) a(a~b~,i~j~)  -> (e~b~,m~j~)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H1R.bb.oovv, T.bb, aP.bb, optimize=True)
    #     <mn~|ef~>' t(eb~,mj~) a(a~b~,i~j~)  -> (a~f~,i~n~)
    X += np.einsum("mnef,ebmj,abij->afin", H1R.ab.oovv, T.ab, aP.bb, optimize=True)
    #     <m~n~||i~j~>' a(a~b~,i~j~)  -> (a~b~,m~n~)
    X += 0.125 * np.einsum("mnij,abij->abmn", H1R.bb.oooo, aP.bb, optimize=True)
    #     <m~n~||e~f~>' t(a~b~,m~n~) a(a~b~,i~j~)  -> (e~f~,i~j~)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H1R.bb.oovv, T.bb, aP.bb, optimize=True)
    #     <m~n~||e~f~>' t(e~f~,i~j~) a(a~b~,i~j~)  -> (a~b~,m~n~)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H1R.bb.oovv, T.bb, aP.bb, optimize=True)
    # vvvv of H1R: 1/8 sum_ab L(abij) <ab||ef>'
    for B1, B2 in ((H1R.chol.b.vv, H.chol.b.vv), (H.chol.b.vv, H1R.chol.b.vv)):
        Y = cholesky_ladder(aP.bb, B1, B2)
        X += 0.125 * (Y - Y.transpose(1, 0, 2, 3))
    # <L|[[Hbar, tau(a~b~,i~j~)], b2]|0> = <0|L2 [[(V_N T2^2)_C/2, tau(a~b~,i~j~)], b2]|0>
    #     <m~n~||e~f~> b(ae~,ij~) l(ab~,ij~)  -> (b~f~,m~n~)
    X += -0.5 * np.einsum("mnef,aeij,abij->bfmn", H.bb.oovv, bP.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> b(ab~,im~) l(ab~,ij~)  -> (e~f~,j~n~)
    X -= 0.5 * np.einsum("mnef,abim,abij->efjn", H.bb.oovv, bP.ab, L.ab, optimize=True)
    #     <mn~|ef~> b(ae,im) l(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.ab.oovv, bP.aa, L.ab, optimize=True)
    #     <m~n~||e~f~> b(ae~,im~) l(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.bb.oovv, bP.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> b(a~f~,m~n~) l(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.25 * np.einsum("mnef,afmn,abij->ebij", H.bb.oovv, bP.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> b(e~b~,i~j~) l(a~b~,i~j~)  -> (a~f~,m~n~)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H.bb.oovv, bP.bb, L.bb, optimize=True)
    #     <nm~|fe~> b(fa~,nm~) l(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.5 * np.einsum("nmfe,fanm,abij->ebij", H.ab.oovv, bP.ab, L.bb, optimize=True)
    #     <m~n~||e~f~> b(e~f~,i~n~) l(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H.bb.oovv, bP.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> b(a~b~,m~j~) l(a~b~,i~j~)  -> (e~f~,i~n~)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H.bb.oovv, bP.bb, L.bb, optimize=True)
    #     <nm~|fe~> b(fe~,ni~) l(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.5 * np.einsum("nmfe,feni,abij->abmj", H.ab.oovv, bP.ab, L.bb, optimize=True)
    #     <m~n~||e~f~> b(a~f~,i~n~) l(a~b~,i~j~)  -> (e~b~,m~j~)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H.bb.oovv, bP.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> b(e~b~,m~j~) l(a~b~,i~j~)  -> (a~f~,i~n~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H.bb.oovv, bP.bb, L.bb, optimize=True)
    #     <mn~|ef~> b(eb~,mj~) l(a~b~,i~j~)  -> (a~f~,i~n~)
    X += np.einsum("mnef,ebmj,abij->afin", H.ab.oovv, bP.ab, L.bb, optimize=True)
    #     <m~n~||e~f~> b(e~f~,i~j~) l(a~b~,i~j~)  -> (a~b~,m~n~)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H.bb.oovv, bP.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> b(a~b~,m~n~) l(a~b~,i~j~)  -> (e~f~,i~j~)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H.bb.oovv, bP.bb, L.bb, optimize=True)
    # <L|[[Hbar, tau(a~b~,i~j~)], b1]|0>: the left-CCSD doubles terms of (L1 + L2) with H1b = [H(T1), b1]
    #     f'(m~e~) l(a~,i~)  -> (a~e~,i~m~)
    X += np.einsum("me,ai->aeim", H1b.b.ov, L.b, optimize=True)
    #     <m~n~||i~f~>' l(a~,i~)  -> (a~f~,m~n~)
    X -= 0.5 * np.einsum("mnif,ai->afmn", H1b.bb.ooov, L.b, optimize=True)
    #     <a~n~||e~f~>' l(a~,i~)  -> (e~f~,i~n~)
    X += 0.5 * np.einsum("anef,ai->efin", H1b.bb.vovv, L.b, optimize=True)
    #     <m~n~||e~f~>' t(ae~,ij~) l(ab~,ij~)  -> (b~f~,m~n~)
    X -= 0.5 * np.einsum("mnef,aeij,abij->bfmn", H1b.bb.oovv, T.ab, L.ab, optimize=True)
    #     <m~n~||e~f~>' t(ab~,im~) l(ab~,ij~)  -> (e~f~,j~n~)
    X -= 0.5 * np.einsum("mnef,abim,abij->efjn", H1b.bb.oovv, T.ab, L.ab, optimize=True)
    #     <am~|ie~>' l(ab~,ij~)  -> (e~b~,m~j~)
    X += np.einsum("amie,abij->ebmj", H1b.ab.voov, L.ab, optimize=True)
    #     <mn~|ef~>' t(ae,im) l(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H1b.ab.oovv, T.aa, L.ab, optimize=True)
    #     <m~n~||e~f~>' t(ae~,im~) l(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H1b.bb.oovv, T.ab, L.ab, optimize=True)
    #     f'(a~e~) l(a~b~,i~j~)  -> (e~b~,i~j~)
    X += 0.5 * np.einsum("ae,abij->ebij", H1b.b.vv, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(e~b~,i~j~) l(a~b~,i~j~)  -> (a~f~,m~n~)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H1b.bb.oovv, T.bb, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(a~f~,m~n~) l(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.25 * np.einsum("mnef,afmn,abij->ebij", H1b.bb.oovv, T.bb, L.bb, optimize=True)
    #     <nm~|fe~>' t(fa~,nm~) l(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.5 * np.einsum("nmfe,fanm,abij->ebij", H1b.ab.oovv, T.ab, L.bb, optimize=True)
    #     f'(m~i~) l(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.5 * np.einsum("mi,abij->abmj", H1b.b.oo, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(a~b~,m~j~) l(a~b~,i~j~)  -> (e~f~,i~n~)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H1b.bb.oovv, T.bb, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(e~f~,i~n~) l(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H1b.bb.oovv, T.bb, L.bb, optimize=True)
    #     <nm~|fe~>' t(fe~,ni~) l(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.5 * np.einsum("nmfe,feni,abij->abmj", H1b.ab.oovv, T.ab, L.bb, optimize=True)
    #     <a~m~||i~e~>' l(a~b~,i~j~)  -> (e~b~,m~j~)
    X += np.einsum("amie,abij->ebmj", H1b.bb.voov, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(e~b~,m~j~) l(a~b~,i~j~)  -> (a~f~,i~n~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H1b.bb.oovv, T.bb, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(a~f~,i~n~) l(a~b~,i~j~)  -> (e~b~,m~j~)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H1b.bb.oovv, T.bb, L.bb, optimize=True)
    #     <mn~|ef~>' t(eb~,mj~) l(a~b~,i~j~)  -> (a~f~,i~n~)
    X += np.einsum("mnef,ebmj,abij->afin", H1b.ab.oovv, T.ab, L.bb, optimize=True)
    #     <m~n~||i~j~>' l(a~b~,i~j~)  -> (a~b~,m~n~)
    X += 0.125 * np.einsum("mnij,abij->abmn", H1b.bb.oooo, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(a~b~,m~n~) l(a~b~,i~j~)  -> (e~f~,i~j~)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H1b.bb.oovv, T.bb, L.bb, optimize=True)
    #     <m~n~||e~f~>' t(e~f~,i~j~) l(a~b~,i~j~)  -> (a~b~,m~n~)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H1b.bb.oovv, T.bb, L.bb, optimize=True)
    # vvvv of H1b: 1/8 sum_ab L(abij) <ab||ef>'
    for B1, B2 in ((H1b.chol.b.vv, H.chol.b.vv), (H.chol.b.vv, H1b.chol.b.vv)):
        Y = cholesky_ladder(L.bb, B1, B2)
        X += 0.125 * (Y - Y.transpose(1, 0, 2, 3))
    X -= np.transpose(X, (1, 0, 2, 3))
    X -= np.transpose(X, (0, 1, 3, 2))
    # F_t, <0|(r0 A + b0 L + C)(Hbar - E0)|i~j~a~b~> (left-CCSD LH_fun) and x0 <0|Hbar|i~j~a~b~>
    X += F_t + LHy + x0 * Hbar.bb.oovv.transpose(2, 3, 0, 1)
    return X


def solve_state(rhs0, rhs, transpose, T, R, r0, L, omega, Hbar, system, tol):
    """(x0, X) with (A - E + r l^T) x = rhs, or its transpose, on reference + SD vectors, where
    A = Hbar - E0 on <0| + SD (Eq. 3.1), E - E0 = omega and l0 = 0:

        y0 = -omega x0 + sum_mu <0|Hbar|mu> x_mu + (l . x) r0,
        y  = (Hbar - E0) x - omega x + (l . x) r                           (transpose=False)
        y0 = -omega x0
        y  = x (Hbar - E0) - omega x + x0 <0|Hbar|mu> + (r . x) l           (transpose=True)

    (l . x) and (r . x) run over the unique singles and doubles. GMRES on the full antisymmetric
    arrays, preconditioned with the Hbar(oo, vv) diagonal minus omega (kept away from zero)."""
    if system.cholesky:
        from ccpy.eomcc.eomccsd_chol import HR
        from ccpy.left.left_ccsd_chol import LH_fun
    else:
        from ccpy.eomcc.eomccsd import HR
        from ccpy.left.left_ccsd import LH_fun

    shapes = [T.a.shape, T.b.shape, T.aa.shape, T.ab.shape, T.bb.shape]
    sizes = [int(np.prod(s)) for s in shapes]
    offsets = np.cumsum([1] + sizes)
    Xv, AX = copy.deepcopy(T), copy.deepcopy(T)

    def matvec(x):
        x0 = x[0]
        Xv.a = np.asfortranarray(x[offsets[0]:offsets[1]].reshape(shapes[0]))
        Xv.b = np.asfortranarray(x[offsets[1]:offsets[2]].reshape(shapes[1]))
        Xv.aa = np.asfortranarray(x[offsets[2]:offsets[3]].reshape(shapes[2]))
        Xv.ab = np.asfortranarray(x[offsets[3]:offsets[4]].reshape(shapes[3]))
        Xv.bb = np.asfortranarray(x[offsets[4]:offsets[5]].reshape(shapes[4]))
        if transpose:
            LH_fun(AX, Xv, T, Hbar, False, system)
            # (r . x) l
            rx = r0 * x0 + (np.sum(R.a * Xv.a) + np.sum(R.b * Xv.b) + 0.25 * np.sum(R.aa * Xv.aa)
                            + np.sum(R.ab * Xv.ab) + 0.25 * np.sum(R.bb * Xv.bb))
            y0 = -omega * x0
            y_a = AX.a - omega * Xv.a + x0 * Hbar.a.ov.T + rx * L.a
            y_b = AX.b - omega * Xv.b + x0 * Hbar.b.ov.T + rx * L.b
            y_aa = AX.aa - omega * Xv.aa + x0 * Hbar.aa.oovv.transpose(2, 3, 0, 1) + rx * L.aa
            y_ab = AX.ab - omega * Xv.ab + x0 * Hbar.ab.oovv.transpose(2, 3, 0, 1) + rx * L.ab
            y_bb = AX.bb - omega * Xv.bb + x0 * Hbar.bb.oovv.transpose(2, 3, 0, 1) + rx * L.bb
        else:
            HR(AX, Xv, T, Hbar, False, system)
            # (l . x) r
            lx = (np.sum(L.a * Xv.a) + np.sum(L.b * Xv.b) + 0.25 * np.sum(L.aa * Xv.aa) + np.sum(L.ab * Xv.ab)
                  + 0.25 * np.sum(L.bb * Xv.bb))
            # <0|Hbar|mu> x_mu
            ex = (np.sum(Hbar.a.ov.T * Xv.a) + np.sum(Hbar.b.ov.T * Xv.b)
                  + 0.25 * np.sum(Hbar.aa.oovv.transpose(2, 3, 0, 1) * Xv.aa)
                  + np.sum(Hbar.ab.oovv.transpose(2, 3, 0, 1) * Xv.ab)
                  + 0.25 * np.sum(Hbar.bb.oovv.transpose(2, 3, 0, 1) * Xv.bb))
            y0 = -omega * x0 + ex + lx * r0
            y_a = AX.a - omega * Xv.a + lx * R.a
            y_b = AX.b - omega * Xv.b + lx * R.b
            y_aa = AX.aa - omega * Xv.aa + lx * R.aa
            y_ab = AX.ab - omega * Xv.ab + lx * R.ab
            y_bb = AX.bb - omega * Xv.bb + lx * R.bb
        return np.concatenate([[y0], np.ravel(y_a), np.ravel(y_b), np.ravel(y_aa), np.ravel(y_ab), np.ravel(y_bb)])

    eoa, eva = np.diag(Hbar.a.oo), np.diag(Hbar.a.vv)
    eob, evb = np.diag(Hbar.b.oo), np.diag(Hbar.b.vv)
    d1a = eva[:, None] - eoa[None, :]
    d1b = evb[:, None] - eob[None, :]
    d2aa = eva[:, None, None, None] + eva[None, :, None, None] - eoa[None, None, :, None] - eoa[None, None, None, :]
    d2ab = eva[:, None, None, None] + evb[None, :, None, None] - eoa[None, None, :, None] - eob[None, None, None, :]
    d2bb = evb[:, None, None, None] + evb[None, :, None, None] - eob[None, None, :, None] - eob[None, None, None, :]
    diag = np.concatenate([[0.0], d1a.ravel(), d1b.ravel(), d2aa.ravel(), d2ab.ravel(), d2bb.ravel()]) - omega
    diag = np.where(np.abs(diag) < 0.1, np.copysign(0.1, diag), diag)
    n = offsets[-1]
    A = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
    M = spla.LinearOperator((n, n), matvec=lambda x: x / diag, dtype=float)
    b = np.concatenate([[rhs0], np.ravel(rhs.a), np.ravel(rhs.b), np.ravel(rhs.aa), np.ravel(rhs.ab), np.ravel(rhs.bb)])
    x, info = spla.gmres(A, b, M=M, rtol=tol, atol=0.0, restart=60, maxiter=200)
    if info != 0:
        raise RuntimeError(f"The CR-EOMCC(2,3) state equations did not converge (gmres info = {info}).")
    return x[0], SimpleNamespace(a=x[offsets[0]:offsets[1]].reshape(shapes[0]),
                                 b=x[offsets[1]:offsets[2]].reshape(shapes[1]),
                                 aa=x[offsets[2]:offsets[3]].reshape(shapes[2]),
                                 ab=x[offsets[3]:offsets[4]].reshape(shapes[3]),
                                 bb=x[offsets[4]:offsets[5]].reshape(shapes[4]))
