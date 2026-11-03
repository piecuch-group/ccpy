"""Analytic nuclear gradient of an EOMCCSD excited state, factorized with density fitting or
Cholesky vectors, RHF or UHF reference (ccpq_analytical_gradients.pdf, Sec. 7, base part).

The state has right vector R = r0 + R1 + R2, left vector L = L1 + L2 (<L|R> = 1) and energy
E = E0 + omega = <L|Hbar|R>. Its Lagrangian adds the CCSD equations with the cluster
multiplier z, J^T z = -<L|[Hbar, tau]|R> (Eq. 7.15), so the Hbar coefficient is
l r^T + z e0^T, pulled back to the bare integrals and relaxed as for CCSD.

Every Hbar coefficient of the EOMCCSD-based gradients is a pairing <b|Hbar|w> of
b = b0 <0| + bP and w = w0 |0> + wP (singles and doubles). With c_J = <bP|W|J> (singles only,
calc_c_singles),

    <b|Hbar|w>              = <x0 + X|Hbar|0> + D_wP <0|(b0 + bP) Hbar|0>,
    <b|[Hbar, tau_I]|w>     = (J^T X)_I + x0 e_I + <0|(b0 + bP)[[Hbar, tau_I], WP]|0>,
    x0 = w0 b0 + bP . wP,   X  = w0 bP + c,

so the densities come from calc_bra_density and calc_w_density and the T gradients from
left_action and calc_BTw (ccpy.gradients.ccsd_gradient). CR-EOMCC(2,3)
(ccpy.gradients.creomcc23_gradient) adds its correction to these base densities.
"""

import time
from types import SimpleNamespace

import numpy as np

from ccpy.gradients.ccsd_gradient import (add_sd, apply_action, calc_bra_density, calc_BTw, calc_c_singles,
                                          calc_w_density, left_action, sd_block, solve_linear)
from ccpy.gradients.density_gradient import add_densities
from ccpy.gradients.derivative_integrals import contract_derivative_integrals, print_gradient
from ccpy.gradients.orbital_response import orbital_relaxation
from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol

SD = ("a", "b", "aa", "ab", "bb")


def calc_eomccsd_gradient(T, R, r0, L, Hbar, reference, zvector_tol=1.0e-10, verbose=True):
    """Analytic EOMCCSD nuclear gradient of one root, (natoms, 3) in Eh/bohr, from the CCSD T,
    the root's R, r0 and left vector L, and the CCSD HBar, on the SCF reference
    (ccpy.methods.Reference)."""
    t0 = time.perf_counter()
    H = build_hbar_ccs_chol(T, reference.H, reference.system)
    R, L = as_sd(R), as_sd(L)
    D1, Y = calc_eomccsd_density(T, R, r0, L, Hbar, H, reference, zvector_tol)
    D, Y, W = orbital_relaxation(D1, Y, reference, zvector_tol=zvector_tol, verbose=verbose)
    grad = contract_derivative_integrals(D, Y, W, reference)
    if verbose:
        print_gradient(reference.molecule, grad, "EOMCCSD Analytic Gradient", time.perf_counter() - t0)
    return grad


def calc_eomccsd_density(T, R, r0, L, Hbar, H, reference, tol):
    """(D1, Y) of l r^T + z e0^T, with J^T z = -<L|[Hbar, tau]|R> (Eq. 7.15)."""
    LH = left_action(T, Hbar, reference.system)
    # q_I = <L|[Hbar, tau_I]|R>, and sum_J z_J <J|Hbar - E0|I> = -q_I
    q = calc_pairing_t_gradient(0.0, L, r0, R, T, H, calc_reference_row(Hbar), LH, reference)
    z = solve_linear(LH, {c: -q[c] for c in SD}, T, Hbar, tol)
    # <L|Hbar|R> = <(L . R) + r0 L + c + z|Hbar|0> + D_R <0|L Hbar|0>, c_J = <L|R|J>
    return add_densities(
        calc_bra_density(sd_dot(L, R), add_sd(scale_sd(r0, L), calc_c_singles(R, L), z), T, H, reference),
        calc_w_density(R, T, L, H, reference, l0=0.0),
    )


# ======================================================================================
# pairings <b|Hbar|w>
# ======================================================================================
def calc_pairing_t_gradient(b0, bP, w0, wP, T, H, e, LH, reference):
    """<b|[Hbar, tau_I]|w> for I in SD, b = b0 <0| + bP, w = w0 |0> + wP."""
    x0 = w0 * b0 + sd_dot(bP, wP)
    # (J^T X)_I = <0|X (Hbar - E0)|I> for the de-excitation X = w0 bP + c, c_J = <bP|WP|J>
    JTx = apply_action(LH, add_sd(scale_sd(w0, bP), calc_c_singles(wP, bP)), T)
    # <0|(b0 + bP)[[Hbar, tau_I], WP]|0>
    BTw = calc_BTw(wP, T, bP, H, reference, l0=b0)
    # x0 <0|Hbar|I>
    return {c: JTx[c] + BTw[c] + x0 * e[c] for c in SD}


def calc_reference_row(Hbar):
    """e_I = <0|Hbar|I> for I in SD (Eq. 2.2)."""
    return {"a": Hbar.a.ov.T, "b": Hbar.b.ov.T, "aa": Hbar.aa.oovv.transpose(2, 3, 0, 1),
            "ab": Hbar.ab.oovv.transpose(2, 3, 0, 1), "bb": Hbar.bb.oovv.transpose(2, 3, 0, 1)}


def as_sd(X):
    """Singles and doubles of an EOM vector (ccpy operator) as plain arrays."""
    return SimpleNamespace(**{c: np.array(getattr(X, c)) for c in SD})


def scale_sd(c, x):
    return SimpleNamespace(**{k: c * sd_block(x, k) for k in SD})


def sd_dot(x, y):
    """sum over unique singles and doubles of x * y."""
    return float(np.sum(sd_block(x, "a") * sd_block(y, "a")) + np.sum(sd_block(x, "b") * sd_block(y, "b"))
                 + 0.25 * np.sum(sd_block(x, "aa") * sd_block(y, "aa")) + np.sum(sd_block(x, "ab") * sd_block(y, "ab"))
                 + 0.25 * np.sum(sd_block(x, "bb") * sd_block(y, "bb")))
