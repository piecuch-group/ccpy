"""Analytic nuclear gradient of an EOMCCSD excited state, factorized with density fitting or
Cholesky vectors, RHF or UHF reference.

Theory
------
The state has right vector R = r0 + R1 + R2, left vector L = L1 + L2 (<L|R> = 1) and energy
E = E0 + omega = <L|Hbar|R>, which is stationary in R and L. The cluster amplitudes enter
through Hbar = e^{-T} H e^{T}; the CCSD equations <mu|Hbar|0> = 0 are added with multipliers
z, so the gradient is that of

    E = <L|Hbar|R> + sum_mu z_mu <mu|Hbar|0>,

stationary in t when

    sum_nu z_nu <nu|Hbar - E0|mu> = -q(mu),   q(mu) = <L|[Hbar, tau_mu]|R>.

With C_J = <0|L R1|J> (singles only, c(a,i) = sum_jb l(ab,ij) r(b,j)), the matrix element is

    <L|Hbar|R> = <0|((L . R) + r0 L + C) Hbar|0> + <0|L [Hbar, R1]|0> + <0|L [Hbar, R2]|0>,

and Hbar depends on T1 only through H(T1) = e^{-T1} H e^{T1} (ccpy.hbar.hbar_ccs_chol), so
[Hbar, R1] is Hbar with H(T1) -> H1 = [H(T1), R1] and [Hbar, R2] is the derivative of
(H(T1) e^{T2})_C along R2. The gradient is assembled as

  1. intermediates: C, L . R, the densities of <0|L Hbar|0> (ccpy.density.ccsd_rdm), their
     derivative along R2, H1, and the bare integrals rotated by R1, [h, R1] and [B, R1];
  2. the source q (build_q_1A ... build_q_2C) and the multipliers z (left-CCSD LH_fun);
  3. the bare-frame densities D1 = dE/dh and Y_x = dE/dB_x of the Lagrangian;
  4. orbital relaxation (ccpy.gradients.orbital_response) and
  5. the derivative integrals (ccpy.gradients.derivative_integrals).

Densities and bare integrals are held in the blocks oo, ov, vo, vv of each spin
(ccpy.gradients.density_gradient), "o" running over all occupied orbitals with the frozen core
first: the correlated occupied orbitals are [nfrozen:].
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


def calc_eomccsd_gradient(T, R, r0, L, Hbar, reference, zvector_tol=1.0e-10, perturbation=None,
                          properties=None, verbose=True):
    """Analytic EOMCCSD nuclear gradient of one root, (natoms, 3) in Eh/bohr, from the CCSD T,
    the root's R, r0 and left vector L, and the CCSD HBar, on the SCF reference
    (ccpy.methods.Reference). perturbation: AO one-electron operator(s), for which dE/dlambda is
    returned instead (see contract_derivative_integrals); with properties V, (gradient, dE/dlambda)."""
    if reference.system.cholesky:
        from ccpy.left.left_ccsd_chol import LH_fun
    else:
        from ccpy.left.left_ccsd import LH_fun

    t0 = time.perf_counter()
    system, nfz = reference.system, reference.nfrozen
    H = build_hbar_ccs_chol(T, reference.H, system)
    h, B = get_bare_integrals(reference)
    R = SimpleNamespace(a=np.array(R.a), b=np.array(R.b), aa=np.array(R.aa), ab=np.array(R.ab), bb=np.array(R.bb))
    L = SimpleNamespace(a=np.array(L.a), b=np.array(L.b), aa=np.array(L.aa), ab=np.array(L.ab), bb=np.array(L.bb))

    # 1. intermediates
    # C_J = <0|L R1|J>: c(a,i) = sum_jb l(ab,ij) r(b,j)
    c_a = np.einsum("abij,bj->ai", L.aa, R.a, optimize=True) + np.einsum("abij,bj->ai", L.ab, R.b, optimize=True)
    c_b = np.einsum("baji,bj->ai", L.ab, R.a, optimize=True) + np.einsum("abij,bj->ai", L.bb, R.b, optimize=True)
    # L . R over the unique singles and doubles
    lr = (np.sum(L.a * R.a) + np.sum(L.b * R.b) + 0.25 * np.sum(L.aa * R.aa) + np.sum(L.ab * R.ab)
          + 0.25 * np.sum(L.bb * R.bb))
    # densities of <0|L (H(T1) e^{T2})_C|0> (no reference part), and their derivative along R2
    rdm1_L, rdm2_L = calc_ccsd_rdm(T, L, system, energy=0.0)
    D_L, Y_L = build_DY(rdm1_L, rdm2_L, lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s), H.chol, T, B, reference)
    D_R2, Y_R2 = build_DY(*calc_ccsd_rdm_derivative(T, R, L, system, energy=0.0),
                          lambda a, s: calc_rdm2_vvvv_slice(L, R, a, s), H.chol, T, B, reference)
    # H1 = [H(T1), R1] in the blocks of H(T1); [h, R1] and [B, R1] in the bare blocks
    # ([M, R1]_pq = sum_e M_pe r(e,q) - sum_m r(p,m) M_mq, r(a,i) nonzero for correlated i)
    H1 = build_hbar_ccs_chol_derivative(R, H, system)
    h1, B1 = zero_density(reference), zero_density(reference, B.a.oo.shape[0])
    h1.a.oo[:, nfz:] = h.a.ov @ R.a
    h1.a.vo[...] = -R.a @ h.a.oo[nfz:, :]
    h1.a.vo[:, nfz:] += h.a.vv @ R.a
    h1.a.vv[...] = -R.a @ h.a.ov[nfz:, :]
    h1.b.oo[:, nfz:] = h.b.ov @ R.b
    h1.b.vo[...] = -R.b @ h.b.oo[nfz:, :]
    h1.b.vo[:, nfz:] += h.b.vv @ R.b
    h1.b.vv[...] = -R.b @ h.b.ov[nfz:, :]
    B1.a.oo[:, :, nfz:] = B.a.ov @ R.a
    B1.a.vo[...] = -R.a @ B.a.oo[:, nfz:, :]
    B1.a.vo[:, :, nfz:] += B.a.vv @ R.a
    B1.a.vv[...] = -R.a @ B.a.ov[:, nfz:, :]
    B1.b.oo[:, :, nfz:] = B.b.ov @ R.b
    B1.b.vo[...] = -R.b @ B.b.oo[:, nfz:, :]
    B1.b.vo[:, :, nfz:] += B.b.vv @ R.b
    B1.b.vv[...] = -R.b @ B.b.ov[:, nfz:, :]
    # Y of <0|L (H(T1) e^{T2})_C|0> with H(T1) -> H1 (its T1-transformed vectors [Bt, R1] and the
    # bare [B, R1] of its separable terms)
    _, Y_LR1 = build_DY(rdm1_L, rdm2_L, lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s), H1.chol, T, B1, reference)
    # <0|(r0 L + C)(Hbar - E0)|mu>: the left-CCSD Jacobian (LH_fun) on X = r0 L + C
    X, LHx = copy.deepcopy(T), copy.deepcopy(T)
    X.a, X.b = np.asfortranarray(r0 * L.a + c_a), np.asfortranarray(r0 * L.b + c_b)
    X.aa, X.ab, X.bb = np.asfortranarray(r0 * L.aa), np.asfortranarray(r0 * L.ab), np.asfortranarray(r0 * L.bb)
    LH_fun(LHx, X, T, Hbar, False, system)

    # 2. the source q(mu) = <L|[Hbar, tau_mu]|R> and the multipliers z
    q = SimpleNamespace(
        a=build_q_1A(np.array(LHx.a), lr, Hbar, h, B, h1, B1, D_L, Y_L, D_R2, Y_R2, Y_LR1, nfz),
        b=build_q_1B(np.array(LHx.b), lr, Hbar, h, B, h1, B1, D_L, Y_L, D_R2, Y_R2, Y_LR1, nfz),
        aa=build_q_2A(np.array(LHx.aa), lr, Hbar, R, T, L, H, H1),
        ab=build_q_2B(np.array(LHx.ab), lr, Hbar, R, T, L, H, H1),
        bb=build_q_2C(np.array(LHx.bb), lr, Hbar, R, T, L, H, H1),
    )
    z = solve_z(q, T, Hbar, system, zvector_tol)

    # 3. bare-frame densities of <L|Hbar|R> + <Z|Hbar|0>
    # <0|((L . R) + r0 L + C + Z) Hbar|0>
    Xb = SimpleNamespace(a=r0 * L.a + c_a + z.a, b=r0 * L.b + c_b + z.b, aa=r0 * L.aa + z.aa,
                         ab=r0 * L.ab + z.ab, bb=r0 * L.bb + z.bb)
    D1, Y = build_DY(*calc_ccsd_rdm(T, Xb, system, energy=lr), lambda a, s: calc_rdm2_vvvv_slice(Xb, T, a, s),
                     H.chol, T, B, reference, ref_weight=lr)
    # <0|L [Hbar, R2]|0>
    D1.a.oo += D_R2.a.oo
    D1.a.ov += D_R2.a.ov
    D1.a.vo += D_R2.a.vo
    D1.a.vv += D_R2.a.vv
    D1.b.oo += D_R2.b.oo
    D1.b.ov += D_R2.b.ov
    D1.b.vo += D_R2.b.vo
    D1.b.vv += D_R2.b.vv
    Y.a.oo += Y_R2.a.oo + Y_LR1.a.oo
    Y.a.ov += Y_R2.a.ov + Y_LR1.a.ov
    Y.a.vo += Y_R2.a.vo + Y_LR1.a.vo
    Y.a.vv += Y_R2.a.vv + Y_LR1.a.vv
    Y.b.oo += Y_R2.b.oo + Y_LR1.b.oo
    Y.b.ov += Y_R2.b.ov + Y_LR1.b.ov
    Y.b.vo += Y_R2.b.vo + Y_LR1.b.vo
    Y.b.vv += Y_R2.b.vv + Y_LR1.b.vv
    # <0|L [Hbar, R1]|0>: the densities of <0|L Hbar|0> rotated, D_L R1^T - R1^T D_L (and Y_L
    # likewise), plus the Y_LR1 above
    D1.a.oo[nfz:, :] -= R.a.T @ D_L.a.vo
    D1.a.ov += D_L.a.oo[:, nfz:] @ R.a.T
    D1.a.ov[nfz:, :] -= R.a.T @ D_L.a.vv
    D1.a.vv += D_L.a.vo[:, nfz:] @ R.a.T
    D1.b.oo[nfz:, :] -= R.b.T @ D_L.b.vo
    D1.b.ov += D_L.b.oo[:, nfz:] @ R.b.T
    D1.b.ov[nfz:, :] -= R.b.T @ D_L.b.vv
    D1.b.vv += D_L.b.vo[:, nfz:] @ R.b.T
    Y.a.oo[:, nfz:, :] -= R.a.T @ Y_L.a.vo
    Y.a.ov += Y_L.a.oo[:, :, nfz:] @ R.a.T
    Y.a.ov[:, nfz:, :] -= R.a.T @ Y_L.a.vv
    Y.a.vv += Y_L.a.vo[:, :, nfz:] @ R.a.T
    Y.b.oo[:, nfz:, :] -= R.b.T @ Y_L.b.vo
    Y.b.ov += Y_L.b.oo[:, :, nfz:] @ R.b.T
    Y.b.ov[:, nfz:, :] -= R.b.T @ Y_L.b.vv
    Y.b.vv += Y_L.b.vo[:, :, nfz:] @ R.b.T

    # 4. orbital relaxation and 5. derivative integrals
    D, Y, W = orbital_relaxation(D1, Y, reference, zvector_tol=zvector_tol, verbose=verbose)
    grad = contract_derivative_integrals(D, Y, W, reference, perturbation)
    if verbose:
        print_gradient(reference.molecule, grad, "EOMCCSD Analytic Gradient", time.perf_counter() - t0)
    if properties is not None:
        return grad, contract_derivative_integrals(D, Y, W, reference, properties)
    return grad


def build_q_1A(LHx, lr, Hbar, h, B, h1, B1, D_L, Y_L, D_R2, Y_R2, Y_LR1, nfz):
    """q(ia) = <L|[Hbar, a+ i]|R>, the alpha singles of the source of the multiplier equations:

        <L|[Hbar, a+ i]|R> = <0|(r0 L + C)(Hbar - E0)|ia> + (L . R) <0|Hbar|ia>
                             + <0|L [[Hbar, a+ i], R2]|0> + <0|L [[Hbar, a+ i], R1]|0>.

    Hbar depends on t(ai) only through H(T1), d H(T1)/d t(ai) = [H(T1), a+ i], so the last two are
    contractions of bare-frame densities (D, Y) with the rotated integrals [h, a+ i], [B_x, a+ i]:

        sum_pq D_pq [h, a+ i]_pq + sum_x Y_x . [B_x, a+ i] = (h^T D - D h^T)(a,i) + sum_x (B_x^T Y_x - Y_x B_x^T)(a,i),

    with i correlated (occupied indices [nfrozen:])."""
    # <0|(r0 L + C)(Hbar - E0)|ia> (left-CCSD LH_fun) and (L . R) <0|Hbar|ia>
    X = LHx + lr * Hbar.a.ov.T
    # <0|L [[Hbar, a+ i], R2]|0>: the R2-derivative densities of <0|L Hbar|0> against [h, a+ i], [B, a+ i]
    X += h.a.ov.T @ D_R2.a.oo[:, nfz:] + h.a.vv.T @ D_R2.a.vo[:, nfz:]
    X -= D_R2.a.vo @ h.a.oo[nfz:, :].T + D_R2.a.vv @ h.a.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.a.ov, Y_R2.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.a.vv, Y_R2.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_R2.a.vo, B.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_R2.a.vv, B.a.ov[:, nfz:, :], optimize=True)
    # <0|L [[Hbar, a+ i], R1]|0>: the densities of <0|L Hbar|0> against [[h, R1], a+ i], [[B, R1], a+ i] ...
    X += h1.a.ov.T @ D_L.a.oo[:, nfz:] + h1.a.vv.T @ D_L.a.vo[:, nfz:]
    X -= D_L.a.vo @ h1.a.oo[nfz:, :].T + D_L.a.vv @ h1.a.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B1.a.ov, Y_L.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B1.a.vv, Y_L.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_L.a.vo, B1.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_L.a.vv, B1.a.ov[:, nfz:, :], optimize=True)
    # ... and the Y of <0|L Hbar|0> with H(T1) -> [H(T1), R1] against [B, a+ i]
    X += np.einsum("xqa,xqi->ai", B.a.ov, Y_LR1.a.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.a.vv, Y_LR1.a.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_LR1.a.vo, B.a.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_LR1.a.vv, B.a.ov[:, nfz:, :], optimize=True)
    return X


def build_q_1B(LHx, lr, Hbar, h, B, h1, B1, D_L, Y_L, D_R2, Y_R2, Y_LR1, nfz):
    """q(i~a~) = <L|[Hbar, a~+ i~]|R>, the beta singles of the source of the multiplier equations
    (the terms of build_q_1A with the beta blocks)."""
    # <0|(r0 L + C)(Hbar - E0)|i~a~> (left-CCSD LH_fun) and (L . R) <0|Hbar|i~a~>
    X = LHx + lr * Hbar.b.ov.T
    # <0|L [[Hbar, a~+ i~], R2]|0>
    X += h.b.ov.T @ D_R2.b.oo[:, nfz:] + h.b.vv.T @ D_R2.b.vo[:, nfz:]
    X -= D_R2.b.vo @ h.b.oo[nfz:, :].T + D_R2.b.vv @ h.b.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B.b.ov, Y_R2.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.vv, Y_R2.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_R2.b.vo, B.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_R2.b.vv, B.b.ov[:, nfz:, :], optimize=True)
    # <0|L [[Hbar, a~+ i~], R1]|0>
    X += h1.b.ov.T @ D_L.b.oo[:, nfz:] + h1.b.vv.T @ D_L.b.vo[:, nfz:]
    X -= D_L.b.vo @ h1.b.oo[nfz:, :].T + D_L.b.vv @ h1.b.ov[nfz:, :].T
    X += np.einsum("xqa,xqi->ai", B1.b.ov, Y_L.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B1.b.vv, Y_L.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_L.b.vo, B1.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_L.b.vv, B1.b.ov[:, nfz:, :], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.ov, Y_LR1.b.oo[:, :, nfz:], optimize=True)
    X += np.einsum("xqa,xqi->ai", B.b.vv, Y_LR1.b.vo[:, :, nfz:], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_LR1.b.vo, B.b.oo[:, nfz:, :], optimize=True)
    X -= np.einsum("xaq,xiq->ai", Y_LR1.b.vv, B.b.ov[:, nfz:, :], optimize=True)
    return X


def build_q_2A(LHx, lr, Hbar, R, T, L, H, H1):
    """q(ijab) = <L|[Hbar, tau(ab,ij)]|R>, the aa doubles of the source of the multiplier
    equations. With R = r0 + R1 + R2 and C_J = <0|L R1|J> (singles only),

        <L|[Hbar, tau(ab,ij)]|R> = <0|(r0 L + C)(Hbar - E0)|ijab> + (L . R) <0|Hbar|ijab>
                                 + <0|L [[Hbar, tau(ab,ij)], R1]|0> + <0|L [[Hbar, tau(ab,ij)], R2]|0>.

    R2 enters through the T2^2 terms of the CCSD equations, R1 through H1 = [H(T1), R1] in
    the left-CCSD doubles terms. Terms are built in the stored tau(ab,ij) layout and
    antisymmetrized before the (already antisymmetric) last two are added."""
    # <0|L2 [[(V_N T2^2)_C/2, tau(ab,ij)], R2]|0>: the T2^2 terms of the CCSD equations along R2
    #     <mn||ef> r(af,mn) l(ab,ij)  -> (eb,ij)
    X = -0.25 * np.einsum("mnef,afmn,abij->ebij", H.aa.oovv, R.aa, L.aa, optimize=True)
    #     <mn||ef> r(eb,ij) l(ab,ij)  -> (af,mn)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H.aa.oovv, R.aa, L.aa, optimize=True)
    #     <mn~|ef~> r(af~,mn~) l(ab,ij)  -> (eb,ij)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H.ab.oovv, R.ab, L.aa, optimize=True)
    #     <mn||ef> r(ef,in) l(ab,ij)  -> (ab,mj)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H.aa.oovv, R.aa, L.aa, optimize=True)
    #     <mn||ef> r(ab,mj) l(ab,ij)  -> (ef,in)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H.aa.oovv, R.aa, L.aa, optimize=True)
    #     <mn~|ef~> r(ef~,in~) l(ab,ij)  -> (ab,mj)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H.ab.oovv, R.ab, L.aa, optimize=True)
    #     <mn||ef> r(af,in) l(ab,ij)  -> (eb,mj)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H.aa.oovv, R.aa, L.aa, optimize=True)
    #     <mn||ef> r(eb,mj) l(ab,ij)  -> (af,in)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H.aa.oovv, R.aa, L.aa, optimize=True)
    #     <mn~|ef~> r(af~,in~) l(ab,ij)  -> (eb,mj)
    X += np.einsum("mnef,afin,abij->ebmj", H.ab.oovv, R.ab, L.aa, optimize=True)
    #     <mn||ef> r(ef,ij) l(ab,ij)  -> (ab,mn)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H.aa.oovv, R.aa, L.aa, optimize=True)
    #     <mn||ef> r(ab,mn) l(ab,ij)  -> (ef,ij)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H.aa.oovv, R.aa, L.aa, optimize=True)
    #     <mn||ef> r(eb~,ij~) l(ab~,ij~)  -> (af,mn)
    X -= 0.5 * np.einsum("mnef,ebij,abij->afmn", H.aa.oovv, R.ab, L.ab, optimize=True)
    #     <mn||ef> r(ab~,mj~) l(ab~,ij~)  -> (ef,in)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H.aa.oovv, R.ab, L.ab, optimize=True)
    #     <mn||ef> r(fb~,nj~) l(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H.aa.oovv, R.ab, L.ab, optimize=True)
    #     <mn~|ef~> r(f~b~,n~j~) l(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H.ab.oovv, R.bb, L.ab, optimize=True)
    # <0|(L1 + L2)[[H(T1), R1], tau(ab,ij)]_C|0>: the left-CCSD doubles terms with H1 = [H(T1), R1]
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
    # <0|(r0 L + C)(Hbar - E0)|ijab> (left-CCSD LH_fun) and (L . R) <0|Hbar|ijab>
    X += LHx + lr * Hbar.aa.oovv.transpose(2, 3, 0, 1)
    return X


def build_q_2B(LHx, lr, Hbar, R, T, L, H, H1):
    """q(ij~ab~) = <L|[Hbar, tau(ab~,ij~)]|R>, the ab doubles of the source of the multiplier
    equations. With R = r0 + R1 + R2 and C_J = <0|L R1|J> (singles only),

        <L|[Hbar, tau(ab~,ij~)]|R> = <0|(r0 L + C)(Hbar - E0)|ij~ab~> + (L . R) <0|Hbar|ij~ab~>
                                 + <0|L [[Hbar, tau(ab~,ij~)], R1]|0> + <0|L [[Hbar, tau(ab~,ij~)], R2]|0>.

    R2 enters through the T2^2 terms of the CCSD equations, R1 through H1 = [H(T1), R1] in
    the left-CCSD doubles terms. Terms are built in the stored tau(ab~,ij~) layout and
    added to the last two."""
    # <0|L2 [[(V_N T2^2)_C/2, tau(ab~,ij~)], R2]|0>: the T2^2 terms of the CCSD equations along R2
    #     <mn~|ef~> r(eb,ij) l(ab,ij)  -> (af~,mn~)
    X = -0.5 * np.einsum("mnef,ebij,abij->afmn", H.ab.oovv, R.aa, L.aa, optimize=True)
    #     <mn~|ef~> r(ab,mj) l(ab,ij)  -> (ef~,in~)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H.ab.oovv, R.aa, L.aa, optimize=True)
    #     <mn~|ef~> r(eb,mj) l(ab,ij)  -> (af~,in~)
    X += np.einsum("mnef,ebmj,abij->afin", H.ab.oovv, R.aa, L.aa, optimize=True)
    #     <m~n~||e~f~> r(af~,in~) l(ab,ij)  -> (be~,jm~)
    X += 0.5 * np.einsum("mnef,afin,abij->bejm", H.bb.oovv, R.ab, L.aa, optimize=True)
    #     <m~n~||e~f~> r(be~,jm~) l(ab,ij)  -> (af~,in~)
    X += 0.5 * np.einsum("mnef,bejm,abij->afin", H.bb.oovv, R.ab, L.aa, optimize=True)
    #     <mn||ef> r(af,mn) l(ab~,ij~)  -> (eb~,ij~)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H.aa.oovv, R.aa, L.ab, optimize=True)
    #     <mn~|ef~> r(af~,mn~) l(ab~,ij~)  -> (eb~,ij~)
    X -= np.einsum("mnef,afmn,abij->ebij", H.ab.oovv, R.ab, L.ab, optimize=True)
    #     <mn~|ef~> r(eb~,ij~) l(ab~,ij~)  -> (af~,mn~)
    X -= np.einsum("mnef,ebij,abij->afmn", H.ab.oovv, R.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> r(b~f~,m~n~) l(ab~,ij~)  -> (ae~,ij~)
    X -= 0.5 * np.einsum("mnef,bfmn,abij->aeij", H.bb.oovv, R.bb, L.ab, optimize=True)
    #     <nm~|fe~> r(fb~,nm~) l(ab~,ij~)  -> (ae~,ij~)
    X -= np.einsum("nmfe,fbnm,abij->aeij", H.ab.oovv, R.ab, L.ab, optimize=True)
    #     <nm~|fe~> r(ae~,ij~) l(ab~,ij~)  -> (fb~,nm~)
    X -= np.einsum("nmfe,aeij,abij->fbnm", H.ab.oovv, R.ab, L.ab, optimize=True)
    #     <mn||ef> r(ef,in) l(ab~,ij~)  -> (ab~,mj~)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H.aa.oovv, R.aa, L.ab, optimize=True)
    #     <mn~|ef~> r(ef~,in~) l(ab~,ij~)  -> (ab~,mj~)
    X -= np.einsum("mnef,efin,abij->abmj", H.ab.oovv, R.ab, L.ab, optimize=True)
    #     <mn~|ef~> r(ab~,mj~) l(ab~,ij~)  -> (ef~,in~)
    X -= np.einsum("mnef,abmj,abij->efin", H.ab.oovv, R.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> r(e~f~,j~n~) l(ab~,ij~)  -> (ab~,im~)
    X -= 0.5 * np.einsum("mnef,efjn,abij->abim", H.bb.oovv, R.bb, L.ab, optimize=True)
    #     <nm~|fe~> r(fe~,nj~) l(ab~,ij~)  -> (ab~,im~)
    X -= np.einsum("nmfe,fenj,abij->abim", H.ab.oovv, R.ab, L.ab, optimize=True)
    #     <nm~|fe~> r(ab~,im~) l(ab~,ij~)  -> (fe~,nj~)
    X -= np.einsum("nmfe,abim,abij->fenj", H.ab.oovv, R.ab, L.ab, optimize=True)
    #     <mn||ef> r(ae,im) l(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.aa.oovv, R.aa, L.ab, optimize=True)
    #     <nm~|fe~> r(ae~,im~) l(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("nmfe,aeim,abij->fbnj", H.ab.oovv, R.ab, L.ab, optimize=True)
    #     <nm~|fe~> r(fb~,nj~) l(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("nmfe,fbnj,abij->aeim", H.ab.oovv, R.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> r(f~b~,n~j~) l(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("mnef,fbnj,abij->aeim", H.bb.oovv, R.bb, L.ab, optimize=True)
    #     <mn~|ef~> r(af~,mj~) l(ab~,ij~)  -> (eb~,in~)
    X += np.einsum("mnef,afmj,abij->ebin", H.ab.oovv, R.ab, L.ab, optimize=True)
    #     <mn~|ef~> r(eb~,in~) l(ab~,ij~)  -> (af~,mj~)
    X += np.einsum("mnef,ebin,abij->afmj", H.ab.oovv, R.ab, L.ab, optimize=True)
    #     <mn~|ef~> r(ef~,ij~) l(ab~,ij~)  -> (ab~,mn~)
    X += np.einsum("mnef,efij,abij->abmn", H.ab.oovv, R.ab, L.ab, optimize=True)
    #     <mn~|ef~> r(ab~,mn~) l(ab~,ij~)  -> (ef~,ij~)
    X += np.einsum("mnef,abmn,abij->efij", H.ab.oovv, R.ab, L.ab, optimize=True)
    #     <nm~|fe~> r(e~b~,i~j~) l(a~b~,i~j~)  -> (fa~,nm~)
    X -= 0.5 * np.einsum("nmfe,ebij,abij->fanm", H.ab.oovv, R.bb, L.bb, optimize=True)
    #     <nm~|fe~> r(a~b~,m~j~) l(a~b~,i~j~)  -> (fe~,ni~)
    X -= 0.5 * np.einsum("nmfe,abmj,abij->feni", H.ab.oovv, R.bb, L.bb, optimize=True)
    #     <mn~|ef~> r(a~f~,i~n~) l(a~b~,i~j~)  -> (eb~,mj~)
    X += np.einsum("mnef,afin,abij->ebmj", H.ab.oovv, R.bb, L.bb, optimize=True)
    #     <mn||ef> r(fa~,ni~) l(a~b~,i~j~)  -> (eb~,mj~)
    X += 0.5 * np.einsum("mnef,fani,abij->ebmj", H.aa.oovv, R.ab, L.bb, optimize=True)
    #     <mn||ef> r(eb~,mj~) l(a~b~,i~j~)  -> (fa~,ni~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->fani", H.aa.oovv, R.ab, L.bb, optimize=True)
    # <0|(L1 + L2)[[H(T1), R1], tau(ab~,ij~)]_C|0>: the left-CCSD doubles terms with H1 = [H(T1), R1]
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
    # <0|(r0 L + C)(Hbar - E0)|ij~ab~> (left-CCSD LH_fun) and (L . R) <0|Hbar|ij~ab~>
    X += LHx + lr * Hbar.ab.oovv.transpose(2, 3, 0, 1)
    return X


def build_q_2C(LHx, lr, Hbar, R, T, L, H, H1):
    """q(i~j~a~b~) = <L|[Hbar, tau(a~b~,i~j~)]|R>, the bb doubles of the source of the multiplier
    equations. With R = r0 + R1 + R2 and C_J = <0|L R1|J> (singles only),

        <L|[Hbar, tau(a~b~,i~j~)]|R> = <0|(r0 L + C)(Hbar - E0)|i~j~a~b~> + (L . R) <0|Hbar|i~j~a~b~>
                                 + <0|L [[Hbar, tau(a~b~,i~j~)], R1]|0> + <0|L [[Hbar, tau(a~b~,i~j~)], R2]|0>.

    R2 enters through the T2^2 terms of the CCSD equations, R1 through H1 = [H(T1), R1] in
    the left-CCSD doubles terms. Terms are built in the stored tau(ab,ij) layout and
    antisymmetrized before the (already antisymmetric) last two are added."""
    # <0|L2 [[(V_N T2^2)_C/2, tau(a~b~,i~j~)], R2]|0>: the T2^2 terms of the CCSD equations along R2
    #     <m~n~||e~f~> r(ae~,ij~) l(ab~,ij~)  -> (b~f~,m~n~)
    X = -0.5 * np.einsum("mnef,aeij,abij->bfmn", H.bb.oovv, R.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> r(ab~,im~) l(ab~,ij~)  -> (e~f~,j~n~)
    X -= 0.5 * np.einsum("mnef,abim,abij->efjn", H.bb.oovv, R.ab, L.ab, optimize=True)
    #     <mn~|ef~> r(ae,im) l(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.ab.oovv, R.aa, L.ab, optimize=True)
    #     <m~n~||e~f~> r(ae~,im~) l(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.bb.oovv, R.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> r(a~f~,m~n~) l(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.25 * np.einsum("mnef,afmn,abij->ebij", H.bb.oovv, R.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> r(e~b~,i~j~) l(a~b~,i~j~)  -> (a~f~,m~n~)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H.bb.oovv, R.bb, L.bb, optimize=True)
    #     <nm~|fe~> r(fa~,nm~) l(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.5 * np.einsum("nmfe,fanm,abij->ebij", H.ab.oovv, R.ab, L.bb, optimize=True)
    #     <m~n~||e~f~> r(e~f~,i~n~) l(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H.bb.oovv, R.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> r(a~b~,m~j~) l(a~b~,i~j~)  -> (e~f~,i~n~)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H.bb.oovv, R.bb, L.bb, optimize=True)
    #     <nm~|fe~> r(fe~,ni~) l(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.5 * np.einsum("nmfe,feni,abij->abmj", H.ab.oovv, R.ab, L.bb, optimize=True)
    #     <m~n~||e~f~> r(a~f~,i~n~) l(a~b~,i~j~)  -> (e~b~,m~j~)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H.bb.oovv, R.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> r(e~b~,m~j~) l(a~b~,i~j~)  -> (a~f~,i~n~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H.bb.oovv, R.bb, L.bb, optimize=True)
    #     <mn~|ef~> r(eb~,mj~) l(a~b~,i~j~)  -> (a~f~,i~n~)
    X += np.einsum("mnef,ebmj,abij->afin", H.ab.oovv, R.ab, L.bb, optimize=True)
    #     <m~n~||e~f~> r(e~f~,i~j~) l(a~b~,i~j~)  -> (a~b~,m~n~)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H.bb.oovv, R.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> r(a~b~,m~n~) l(a~b~,i~j~)  -> (e~f~,i~j~)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H.bb.oovv, R.bb, L.bb, optimize=True)
    # <0|(L1 + L2)[[H(T1), R1], tau(a~b~,i~j~)]_C|0>: the left-CCSD doubles terms with H1 = [H(T1), R1]
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
    # <0|(r0 L + C)(Hbar - E0)|i~j~a~b~> (left-CCSD LH_fun) and (L . R) <0|Hbar|i~j~a~b~>
    X += LHx + lr * Hbar.bb.oovv.transpose(2, 3, 0, 1)
    return X


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


def solve_z(q, T, Hbar, system, tol):
    """z with sum_nu z_nu <nu|Hbar - E0|mu> = -q(mu) for mu in singles and doubles (left-CCSD
    LH_fun on the CCSD Hbar), by GMRES on the full antisymmetric arrays, preconditioned with the
    Hbar(oo, vv) diagonal."""
    if system.cholesky:
        from ccpy.left.left_ccsd_chol import LH_fun
    else:
        from ccpy.left.left_ccsd import LH_fun

    shapes = [T.a.shape, T.b.shape, T.aa.shape, T.ab.shape, T.bb.shape]
    sizes = [int(np.prod(s)) for s in shapes]
    offsets = np.cumsum([0] + sizes)
    Z, LZ = copy.deepcopy(T), copy.deepcopy(T)

    def matvec(x):
        Z.a = np.asfortranarray(x[offsets[0]:offsets[1]].reshape(shapes[0]))
        Z.b = np.asfortranarray(x[offsets[1]:offsets[2]].reshape(shapes[1]))
        Z.aa = np.asfortranarray(x[offsets[2]:offsets[3]].reshape(shapes[2]))
        Z.ab = np.asfortranarray(x[offsets[3]:offsets[4]].reshape(shapes[3]))
        Z.bb = np.asfortranarray(x[offsets[4]:offsets[5]].reshape(shapes[4]))
        LH_fun(LZ, Z, T, Hbar, False, system)
        return np.concatenate([np.ravel(LZ.a), np.ravel(LZ.b), np.ravel(LZ.aa), np.ravel(LZ.ab), np.ravel(LZ.bb)])

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
    rhs = -np.concatenate([np.ravel(q.a), np.ravel(q.b), np.ravel(q.aa), np.ravel(q.ab), np.ravel(q.bb)])
    x, info = spla.gmres(A, rhs, M=M, rtol=tol, atol=0.0, restart=60, maxiter=200)
    if info != 0:
        raise RuntimeError(f"The EOMCCSD multiplier equations did not converge (gmres info = {info}).")
    return SimpleNamespace(a=x[offsets[0]:offsets[1]].reshape(shapes[0]), b=x[offsets[1]:offsets[2]].reshape(shapes[1]),
                           aa=x[offsets[2]:offsets[3]].reshape(shapes[2]), ab=x[offsets[3]:offsets[4]].reshape(shapes[3]),
                           bb=x[offsets[4]:offsets[5]].reshape(shapes[4]))
