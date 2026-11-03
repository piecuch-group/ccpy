"""Analytic nuclear gradient of an excited state's CR-EOMCC(2,3)_A energy, factorized with
density fitting or Cholesky vectors, RHF or UHF reference (ccpq_analytical_gradients.pdf,
Secs. 7-8).

The state is an EOMCCSD root with right vector R = r0 + R1 + R2, left vector L = L1 + L2
(<L|R> = 1) and energy E = E0 + omega; Q = all triples and

    delta = sum_K N_K M_K / D_K,   M_K = <K|Hbar|R>,   N_K = <L|Hbar|K>,
    D_K   = omega + f_ii + f_jj + f_kk - f_aa - f_bb - f_cc      (bare Fock, variant A).

Because R is an excitation operator like T, M = r0 M3 + D_R M3: with the moment intermediates
X_g of ccpy.gradients.crcc23_gradient, M = A sum_g (J_g . t_g + X_g . r_g), where
J_g = r0 X_g + X_g(T, H1) + X_g(R, H) (linear part) and H1 = d H(T1 + e R1)/de. The Fortran
triples loops build M, N and D one occupied triple at a time in this form.

The gradient of E_EOMCCSD + delta is assembled in five steps:

  1. sources: delta, W = sum_K u_K v_K, and the gradients of delta at fixed state with respect
     to r (F_r), l (F_l) and t (F_t) (Eqs. 7.6-7.8). Off the CC solution
     <K|Hbar R|0> = r0 M_K + D_R M_K + sum_X c'_X m_X with c'_X = <V3|R|X>, so the doubles of
     F_t also contain (J^T c')_doubles;
  2. responses, in the block order of Sec. 7.2: the base cluster multiplier
     J^T z_b = -<L|[Hbar, tau]|R> (Eq. 7.15); the state multipliers b, a of the correction
     (Eqs. 7.10-7.12, nu = -delta), from the deflated state equations; its cluster multiplier
     J^T z_c = -F_t - <a|[Hbar, tau]|R> - <L|[Hbar, tau]|b> (Eq. 7.13);
  3. densities: l r^T + z_b e0^T (base) and v r^T + l u^T + W e0 e0^T + z_c e0^T + a r^T
     + l b^T (correction, Eq. 7.19), plus the bare-Fock denominator density (Eqs. 8.9-8.10);
  4. orbital relaxation (ccpy.gradients.orbital_response);
  5. contraction with the derivative integrals (ccpy.gradients.derivative_integrals).

Every Hbar coefficient of steps 2-3 is a pairing <b|Hbar|w> of reference + SD vectors; the base
part l r^T + z_b e0^T and the pairing algebra are those of the EOMCCSD gradient
(ccpy.gradients.eomccsd_gradient).
"""

import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse.linalg as spla

from ccpy.gradients.ccsd_gradient import (add_sd, apply_action, calc_bra_density, calc_c_singles, calc_w_density,
                                          embed_vo, left_action, right_action, solve_linear)
from ccpy.gradients.crcc23_gradient import (add_rdm, antisymmetrize_doubles, backprop_triples_intermediates,
                                            build_triples_intermediates, calc_denominator_density, calc_triples)
from ccpy.gradients.derivative_integrals import contract_derivative_integrals, print_gradient
from ccpy.gradients.eomccsd_gradient import (SD, as_sd, calc_eomccsd_density, calc_pairing_t_gradient,
                                             calc_reference_row, scale_sd, sd_dot)
from ccpy.gradients.density_gradient import SPINS, add_densities, build_DY, rotation_gradient
from ccpy.gradients.orbital_response import orbital_relaxation
from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol, build_hbar_ccs_chol_derivative
from ccpy.integrals import integrals


def calc_creomcc23_gradient(T, R, r0, L, omega, Hbar, reference, delta=None, zvector_tol=1.0e-10, components=False,
                            verbose=True):
    """Analytic CR-EOMCC(2,3)_A gradient of one EOMCCSD root, (natoms, 3) in Eh/bohr, from the
    CCSD T, the root's R, r0, L, omega, and the CCSD HBar, on the SCF reference
    (ccpy.methods.Reference). components=True returns a dict with the base (EOMCCSD, nuclear
    repulsion included), correction and total gradients."""
    t0 = time.perf_counter()
    system, H0 = reference.system, reference.H
    H = build_hbar_ccs_chol(T, H0, system)
    H1 = build_hbar_ccs_chol_derivative(R, H, system)
    R, L = as_sd(R), as_sd(L)
    e = calc_reference_row(Hbar)
    LH = left_action(T, Hbar, system)

    # 1. sources
    X = build_triples_intermediates(T, H)
    XH1 = build_triples_intermediates(T, H1)
    XR = build_triples_intermediates(R, H, constant=False)
    J = {case: [r0 * x + y + z for x, y, z in zip(X[case], XH1[case], XR[case])] for case in X}
    del XH1, XR
    tr = calc_triples(X, T, L, H, reference, omega=omega, R=R, r0=r0, J=J)
    del X, J
    delta_Q, W = tr["delta"], tr["W"]
    if delta is not None and abs(delta_Q - delta) > 1.0e-8:
        raise RuntimeError(f"delta from the moments ({delta_Q:.12f}) does not match the energy ({delta:.12f}).")
    # v . M3(T) through the intermediates, at (T, H), (T, H1) and along R2
    dT0, *rdm_v0 = backprop_triples_intermediates(tr["dX"], T, H, system)
    dTH1, _, _ = backprop_triples_intermediates(tr["dX"], T, H1, system)
    dTR, *rdm_XR = backprop_triples_intermediates(tr["dXR"], T, H, system)
    _, *rdm_R2 = backprop_triples_intermediates(tr["dX"], R, H, system, constant=False)
    cprime = SimpleNamespace(**tr["cprime"])
    rho_v0 = build_DY(*rdm_v0, None, H.chol, T, reference.B_mo, reference)                     # <V3|O|0>
    rho_vR = add_densities(scale_density(r0, rho_v0),                                    # <V3|O|R>
                           build_DY(*add_rdm(rdm_XR, rdm_R2, system), None, H.chol, T, reference.B_mo, reference),
                           calc_rotation_density(*rdm_v0, R, T, H, reference),
                           calc_bra_density(0.0, cprime, T, H, reference))
    rho_Lu = build_DY(*tr["rdm"], None, H.chol, T, reference.B_mo, reference)                  # <L|O|U3>
    # F_r = (v . M3(T), d(v . M3)/dt), F_l = d(u . N)/dl
    singles = rotation_gradient(*rho_v0, reference.h_mo, reference.B_mo, reference)
    F_r0 = tr["F_r0"]
    F_rP = {"a": singles["a"], "b": singles["b"],
            **antisymmetrize_doubles({c: dT0[c] + tr["dT"][c] for c in dT0})}
    F_l = {"a": tr["dL"]["a"], "b": tr["dL"]["b"], **antisymmetrize_doubles(tr["dL"])}
    # F_t: rotations of the densities (singles), T2 gradient of v . M plus (J^T c')_doubles, W e
    singles = rotation_gradient(*add_densities(rho_vR, rho_Lu), reference.h_mo, reference.B_mo, reference)
    JTc = apply_action(LH, cprime, T)
    F_t2 = antisymmetrize_doubles({c: tr["dJT"][c] + r0 * dT0[c] + dTH1[c] + dTR[c] for c in dT0})
    F_t = {"a": singles["a"] + W * e["a"], "b": singles["b"] + W * e["b"],
           **{c: F_t2[c] + JTc[c] + W * e[c] for c in ("aa", "ab", "bb")}}

    # 2. responses (the base cluster multiplier of Eq. 7.15 is solved in calc_eomccsd_density)
    # state multipliers: (A - E) b = -F_l + delta r, (A - E)^T a = -F_r + delta l    (Eqs. 7.10-7.11)
    b0, bP = solve_state(delta_Q * r0, {c: -F_l[c] + delta_Q * getattr(R, c) for c in SD}, False,
                         T, R, r0, L, omega, e, Hbar, system, zvector_tol)
    a0, aP = solve_state(-F_r0, {c: -F_rP[c] + delta_Q * getattr(L, c) for c in SD}, True,
                         T, R, r0, L, omega, e, Hbar, system, zvector_tol)
    # deflation gives l.b = r.a = 0; a + beta l satisfies a.r + b.l = -W               (Eq. 7.12)
    beta = -W - (r0 * a0 + sd_dot(R, aP)) - sd_dot(L, bP)
    aP = add_sd(aP, scale_sd(beta, L))
    # correction cluster multiplier                                                     (Eq. 7.13)
    qa = calc_pairing_t_gradient(a0, aP, r0, R, T, H, e, LH, reference)
    qb = calc_pairing_t_gradient(0.0, L, b0, bP, T, H, e, LH, reference)
    z_c = solve_linear(LH, {c: -(F_t[c] + qa[c] + qb[c]) for c in SD}, T, Hbar, zvector_tol)

    # 3. densities
    base = calc_eomccsd_density(T, R, r0, L, Hbar, H, reference, zvector_tol)   # l r^T + z_b e0^T
    corr = add_densities(
        rho_vR,                                                               # v r^T
        rho_Lu,                                                               # l u^T
        calc_bra_density(W + r0 * a0 + sd_dot(aP, R) + sd_dot(L, bP),         # W e0 e0^T + z_c e0^T + bra parts
                         add_sd(z_c, scale_sd(r0, aP), calc_c_singles(R, aP),  # of a r^T and l b^T
                                scale_sd(b0, L), calc_c_singles(bP, L)), T, H, reference),
        calc_w_density(R, T, aP, H, reference, l0=a0),                       # a r^T
        calc_w_density(bP, T, L, H, reference, l0=0.0),                      # l b^T
        calc_denominator_density(tr["d"], H0, reference),                    # Eqs. (8.9)-(8.10)
    )

    # 4. orbital relaxation and 5. derivative integrals
    D, Y, Wd = orbital_relaxation(*add_densities(base, corr), reference, zvector_tol=zvector_tol, verbose=verbose)
    grad = contract_derivative_integrals(D, Y, Wd, reference)
    if verbose:
        print_gradient(reference.molecule, grad, "CR-EOMCC(2,3)_A Analytic Gradient", time.perf_counter() - t0)
    if not components:
        return grad
    D, Y, Wd = orbital_relaxation(*corr, reference, zvector_tol=zvector_tol, verbose=False)
    grad_corr = contract_derivative_integrals(D, Y, Wd, reference) - integrals.nuclear_repulsion_deriv(reference.molecule)
    return {"base": grad - grad_corr, "correction": grad_corr, "total": grad}


def calc_rotation_density(rdm1, rdm2, R, T, H, reference):
    """d/de of the bare-frame (D1, Y) of fixed densities of H(T1) when T1 -> T1 + e R1 (the
    rotation part of calc_w_density)."""
    system = reference.system
    W1 = {s: embed_vo(getattr(R, s), reference, s) for s in SPINS}
    H1 = build_hbar_ccs_chol_derivative(R, H, system)
    dB = {s: reference.B_mo[s] @ W1[s] - W1[s] @ reference.B_mo[s] for s in SPINS}
    D1, Y = build_DY(rdm1, rdm2, None, H.chol, T, reference.B_mo, reference)
    _, YdB = build_DY(rdm1, rdm2, None, H1.chol, T, dB, reference)
    return ({s: D1[s] @ W1[s].T - W1[s].T @ D1[s] for s in SPINS},
            {s: Y[s] @ W1[s].T - W1[s].T @ Y[s] + YdB[s] for s in SPINS})


def scale_density(c, DY):
    D1, Y = DY
    return {s: c * D1[s] for s in SPINS}, {s: c * Y[s] for s in SPINS}


# ======================================================================================
# the deflated state equations
# ======================================================================================
def solve_state(rhs0, rhs, transpose, T, R, r0, L, omega, e, Hbar, system, tol):
    """(x0, X) with (A - E + r l^T) x = rhs, or its transpose, on reference + SD vectors
    (A = Hbar - E0 on <0| + SD, Eq. 3.1; l0 = 0). GMRES on full antisymmetric arrays,
    preconditioned with the Hbar(oo, vv) diagonal minus omega (kept away from zero)."""
    shapes = [getattr(T, k).shape for k in SD]
    sizes = [int(np.prod(s)) for s in shapes]
    action = left_action(T, Hbar, system) if transpose else right_action(T, Hbar, system)

    def unpack(x):
        out, k0 = SimpleNamespace(), 1
        for k, shp, n in zip(SD, shapes, sizes):
            setattr(out, k, x[k0:k0 + n].reshape(shp))
            k0 += n
        return x[0], out

    def matvec(x):
        x0, xP = unpack(x)
        AX = apply_action(action, xP, T)
        if transpose:
            rx = r0 * x0 + sd_dot(R, xP)                                       # (r^T x) l
            y0 = -omega * x0
            yP = {c: AX[c] - omega * getattr(xP, c) + x0 * e[c] + rx * getattr(L, c) for c in SD}
        else:
            lx = sd_dot(L, xP)                                                 # (l^T x) r
            y0 = -omega * x0 + sd_dot(e, xP) + lx * r0
            yP = {c: AX[c] - omega * getattr(xP, c) + lx * getattr(R, c) for c in SD}
        return np.concatenate([[y0]] + [np.ravel(yP[c]) for c in SD])

    eo = {s: np.diag(getattr(Hbar, s).oo) for s in SPINS}
    ev = {s: np.diag(getattr(Hbar, s).vv) for s in SPINS}
    d = [np.array([0.0]), ev["a"][:, None] - eo["a"][None, :], ev["b"][:, None] - eo["b"][None, :]]
    for s, t in (("a", "a"), ("a", "b"), ("b", "b")):
        d.append(ev[s][:, None, None, None] + ev[t][None, :, None, None]
                 - eo[s][None, None, :, None] - eo[t][None, None, None, :])
    diag = np.concatenate([np.ravel(x) for x in d]) - omega
    diag = np.where(np.abs(diag) < 0.1, np.copysign(0.1, diag), diag)
    n = 1 + sum(sizes)
    A = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
    M = spla.LinearOperator((n, n), matvec=lambda x: x / diag, dtype=float)
    b = np.concatenate([[rhs0]] + [np.ravel(rhs[k]) for k in SD])
    x, info = spla.gmres(A, b, M=M, rtol=tol, atol=0.0, restart=60, maxiter=200)
    if info != 0:
        raise RuntimeError(f"The CR-EOMCC(2,3) state equations did not converge (gmres info = {info}).")
    x0, xP = unpack(x)
    return x0, SimpleNamespace(**{k: np.array(getattr(xP, k)) for k in SD})
