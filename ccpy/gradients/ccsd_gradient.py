"""Analytic nuclear gradient of CCSD (RHF or UHF reference, with or without frozen core),
factorized with density fitting or Cholesky vectors: two-electron integrals and densities are
only ever handled through the three-index vectors B_x[p, q] (<pq|rs> = sum_x B_x[p,r] B_x[q,s])
and nothing with four virtual indices is stored.

Theory
------
With the CCS-transformed Hamiltonian H(T1) = e^{-T1} H e^{T1} (ccpy.hbar.hbar_ccs_chol: again
one- plus two-body, with T1-transformed Cholesky vectors), the CCSD Lagrangian is that of
T1 = 0,

    L = E_ref(H(T1)) + E(T2; H(T1)) + sum_mu lambda_mu R_mu(T2; H(T1)),

so the densities are those of ccpy.density.ccsd_rdm (dL/d(blocks of H(T1))). The gradient is
assembled in three stages, each per spin:

  1. calc_ccsd_density: amplitude-relaxed D1[s] = dL/dh[s], Y[s] = dL/dB[s] in the bare MO
     frame (ccpy.gradients.density_gradient.build_DY);
  2. ccpy.gradients.orbital_response.orbital_relaxation: Z-vector of the reference conditions,
     giving the relaxed D[s], Y[s] and the energy-weighted W[s];
  3. ccpy.gradients.derivative_integrals.contract_derivative_integrals.

The CCSD-level kernels below are shared by the gradients built on CCSD (EOMCCSD, CR-CC(2,3),
CR-EOMCC(2,3)): the actions of the CCSD Jacobian (EOMCCSD HR) and its transpose (left-CCSD
LH_fun) and a GMRES solver for them; the bare-frame densities of <0|(x0 + X) Hbar|0>
(calc_bra_density) and of the T-directional derivative of <0|(l0 + Lambda) Hbar|0>
(calc_w_density); and <0|(l0 + Lambda)[[Hbar, tau_I], W]|0> (calc_BTw).
"""

import copy
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse.linalg as spla

from ccpy.density.ccsd_rdm import calc_ccsd_rdm, calc_ccsd_rdm_derivative, calc_rdm2_vvvv_slice
from ccpy.gradients.density_gradient import (add_blocks, build_DY, commutator, get_bare_integrals, rotate_density,
                                             rotation_gradient, zero_like)
from ccpy.gradients.derivative_integrals import contract_derivative_integrals, print_gradient
from ccpy.gradients.orbital_response import orbital_relaxation
from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol, build_hbar_ccs_chol_derivative


def calc_ccsd_gradient(T, L, reference, rdm=None, zvector_tol=1.0e-10, verbose=True):
    """Analytic CCSD nuclear gradient, (natoms, 3) in Eh/bohr, from converged CCSD (T) and
    left-CCSD (L) amplitudes on the SCF reference (ccpy.methods.Reference).

    rdm : (rdm1, rdm2) of ccpy.density.ccsd_rdm.calc_ccsd_rdm(T, L, system), if available
    """
    t0 = time.perf_counter()
    if T.order != 2:
        raise NotImplementedError("Only CCSD gradients are implemented.")
    H = build_hbar_ccs_chol(T, reference.H, reference.system)
    D1, Y = calc_ccsd_density(T, L, H, reference, rdm=rdm)
    D, Y, W = orbital_relaxation(D1, Y, reference, zvector_tol=zvector_tol, verbose=verbose)
    grad = contract_derivative_integrals(D, Y, W, reference)
    if verbose:
        print_gradient(reference.molecule, grad, "CCSD Analytic Gradient", time.perf_counter() - t0)
    return grad


def calc_ccsd_density(T, L, H, reference, rdm=None):
    """Amplitude-relaxed (D1, Y) of the CCSD Lagrangian (reference included), per spin, in the
    bare MO frame (H: build_hbar_ccs_chol)."""
    rdm1, rdm2 = rdm if rdm is not None else calc_ccsd_rdm(T, L, reference.system)
    _, B = get_bare_integrals(reference)
    return build_DY(rdm1, rdm2, lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s), H.chol, T, B, reference,
                    ref_weight=True)


# ======================================================================================
# CCSD-level kernels shared by the gradients built on CCSD
# ======================================================================================
def sd_block(x, k):
    """Block k ("a", "b", "aa", "ab", "bb") of a singles-and-doubles vector (dict or namespace)."""
    return x[k] if isinstance(x, dict) else getattr(x, k)


def add_sd(*xs):
    """Sum of singles-and-doubles vectors."""
    return SimpleNamespace(**{c: sum(sd_block(x, c) for x in xs) for c in ("a", "b", "aa", "ab", "bb")})


def calc_c_singles(w, L):
    """c_J = <L|W|J>: singles only, c_i^a = sum_jb lambda_ij^ab w_j^b."""
    a = np.einsum("abij,bj->ai", L.aa, w.a, optimize=True) + np.einsum("abij,bj->ai", L.ab, w.b, optimize=True)
    b = np.einsum("baji,bj->ai", L.ab, w.a, optimize=True) + np.einsum("abij,bj->ai", L.bb, w.b, optimize=True)
    return SimpleNamespace(a=a, b=b, aa=np.zeros_like(L.aa), ab=np.zeros_like(L.ab), bb=np.zeros_like(L.bb))


def right_action(T, Hbar, system):
    """X -> B X = (Hbar - E0) X on singles and doubles (EOMCCSD HR)."""
    if system.cholesky:
        from ccpy.eomcc.eomccsd_chol import HR
    else:
        from ccpy.eomcc.eomccsd import HR
    return lambda X, out: HR(out, X, T, Hbar, False, system)


def left_action(T, Hbar, system):
    """X -> X (Hbar - E0) on singles and doubles (left-CCSD LH without the inhomogeneous
    terms)."""
    if system.cholesky:
        from ccpy.left.left_ccsd_chol import LH_fun
    else:
        from ccpy.left.left_ccsd import LH_fun
    return lambda X, out: LH_fun(out, X, T, Hbar, False, system)


def apply_action(action, x, T):
    """action (right_action or left_action) applied to one singles-and-doubles vector."""
    X, out = copy.deepcopy(T), copy.deepcopy(T)
    for c in ("a", "b", "aa", "ab", "bb"):
        setattr(X, c, np.asfortranarray(sd_block(x, c)))
    action(X, out)
    return {c: np.array(getattr(out, c)) for c in ("a", "b", "aa", "ab", "bb")}


def solve_linear(action, rhs, T, Hbar, tol):
    """x with action(x) = rhs (singles and doubles, full antisymmetric arrays), by GMRES
    preconditioned with the Hbar(oo, vv) diagonal."""
    keys = ("a", "b", "aa", "ab", "bb")
    shapes = [getattr(T, k).shape for k in keys]
    sizes = [int(np.prod(s)) for s in shapes]
    X, out = copy.deepcopy(T), copy.deepcopy(T)

    def unpack(x, target):
        k0 = 0
        for k, shp, n in zip(keys, shapes, sizes):
            setattr(target, k, np.asfortranarray(x[k0:k0 + n].reshape(shp)))
            k0 += n
        return target

    def matvec(x):
        action(unpack(x, X), out)
        return np.concatenate([np.ravel(getattr(out, k)) for k in keys])

    eoa, eva = np.diag(Hbar.a.oo), np.diag(Hbar.a.vv)
    eob, evb = np.diag(Hbar.b.oo), np.diag(Hbar.b.vv)
    d1a = eva[:, None] - eoa[None, :]
    d1b = evb[:, None] - eob[None, :]
    d2aa = eva[:, None, None, None] + eva[None, :, None, None] - eoa[None, None, :, None] - eoa[None, None, None, :]
    d2ab = eva[:, None, None, None] + evb[None, :, None, None] - eoa[None, None, :, None] - eob[None, None, None, :]
    d2bb = evb[:, None, None, None] + evb[None, :, None, None] - eob[None, None, :, None] - eob[None, None, None, :]
    diag = np.concatenate([d1a.ravel(), d1b.ravel(), d2aa.ravel(), d2ab.ravel(), d2bb.ravel()])
    n = sum(sizes)
    A = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
    M = spla.LinearOperator((n, n), matvec=lambda x: x / diag, dtype=float)
    b = np.concatenate([np.ravel(rhs[k]) for k in keys])
    x, info = spla.gmres(A, b, M=M, rtol=tol, atol=0.0, restart=60, maxiter=200)
    if info != 0:
        raise RuntimeError(f"The response equations did not converge (gmres info = {info}).")
    k0, sol = 0, SimpleNamespace()
    for k, shp, nk in zip(keys, shapes, sizes):
        setattr(sol, k, x[k0:k0 + nk].reshape(shp))
        k0 += nk
    return sol


def calc_bra_density(x0, X, T, H, reference):
    """(D1, Y) of <0|(x0 + X) e^{-T} H e^{T}|0> for a de-excitation operator X (singles and
    doubles); x0 = 1, X = Lambda is the CCSD Lagrangian."""
    system = reference.system
    vvvv = None
    if np.any(X.aa) or np.any(X.ab) or np.any(X.bb):
        vvvv = lambda a, s: calc_rdm2_vvvv_slice(X, T, a, s)  # noqa: E731
    _, B = get_bare_integrals(reference)
    return build_DY(*calc_ccsd_rdm(T, X, system, energy=x0), vvvv, H.chol, T, B, reference, ref_weight=x0)


def calc_w_density(w, T, L, H, reference, l0=1.0):
    """(D1, Y) of d/de <0|(l0 + Lambda) e^{-T-ew} H e^{T+ew}|0>: the T1 part of w rotates the
    Lagrangian densities, the T2 part comes from calc_ccsd_rdm_derivative."""
    system, nfz = reference.system, reference.nfrozen
    W = SimpleNamespace(a=sd_block(w, "a"), b=sd_block(w, "b"), aa=sd_block(w, "aa"), ab=sd_block(w, "ab"),
                        bb=sd_block(w, "bb"))
    _, B = get_bare_integrals(reference)
    H1 = build_hbar_ccs_chol_derivative(W, H, system)
    dB = commutator(B, W.a, W.b, nfz)
    # densities of the Lagrangian <0|(l0 + Lambda) Hbar|0> (D_L), and of the same Lagrangian with
    # the bare B rotated by W1, <0|(l0 + Lambda) Hbar([B, W1])|0>
    rdm1, rdm2 = calc_ccsd_rdm(T, L, system, energy=l0)
    vvvv = lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s)
    D1L, YL = build_DY(rdm1, rdm2, vvvv, H.chol, T, B, reference, ref_weight=l0)
    _, YLdB = build_DY(rdm1, rdm2, vvvv, H1.chol, T, dB, reference, ref_weight=l0)
    # T2 part of w: d/de <0|(l0 + Lambda) (H(T1) e^{T2 + e W2})_C|0>
    D1w, Yw = build_DY(*calc_ccsd_rdm_derivative(T, W, L, system, energy=l0),
                       lambda a, s: calc_rdm2_vvvv_slice(L, W, a, s), H.chol, T, B, reference)
    # T1 part of w: D_L rotated by W1 (D W1^T - W1^T D), plus the B rotation above
    D1 = add_blocks(D1w, rotate_density(D1L, W.a, W.b, nfz))
    Y = add_blocks(add_blocks(Yw, rotate_density(YL, W.a, W.b, nfz)), YLdB)
    return D1, Y


def calc_BTw(w, T, L, H, reference, l0=1.0):
    """<0|(l0 + Lambda)[[Hbar, tau_I], W]|0> for I in SD. Doubles: the T2 part of w through the
    second derivative of the CCSD Lagrangian, the T1 part through H1 = [H, W1]. Singles:
    rotation derivatives of the directional densities and the double rotation
    D_L . [[H, W1], a+ i]."""
    system, nfz = reference.system, reference.nfrozen
    W = SimpleNamespace(a=sd_block(w, "a"), b=sd_block(w, "b"), aa=sd_block(w, "aa"), ab=sd_block(w, "ab"),
                        bb=sd_block(w, "bb"))
    h, B = get_bare_integrals(reference)
    H1 = build_hbar_ccs_chol_derivative(W, H, system)
    h1 = commutator(h, W.a, W.b, nfz)
    dB = commutator(B, W.a, W.b, nfz)

    rdm1, rdm2 = calc_ccsd_rdm(T, L, system, energy=l0)
    vvvv = lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s)
    D1L, YL = build_DY(rdm1, rdm2, vvvv, H.chol, T, B, reference, ref_weight=l0)
    _, YLdB = build_DY(rdm1, rdm2, vvvv, H1.chol, T, dB, reference, ref_weight=l0)
    D1w, Yw = build_DY(*calc_ccsd_rdm_derivative(T, W, L, system, energy=l0),
                       lambda a, s: calc_rdm2_vvvv_slice(L, W, a, s), H.chol, T, B, reference)
    # singles <0|(l0 + Lambda)[[Hbar, a+ i], W]|0>: the T1 derivative is the rotation H -> [H, a+ i],
    #   r1: rotation of the W2-directional densities, d/de <0|(l0 + Lambda)(H(T1) e^{T2 + e W2})_C|0>
    #   r2: rotation of D_L, Y_L against the W1-rotated integrals [h, W1], [B, W1]
    #   r3: rotation of the densities of the W1-rotated B (the H1 dependence through B)
    r1 = rotation_gradient(D1w, Yw, h, B, reference)
    r2 = rotation_gradient(D1L, YL, h1, dB, reference)
    r3 = rotation_gradient(zero_like(D1L), YLdB, h, B, reference)
    # doubles <0|(l0 + Lambda)[[Hbar, tau(ab,ij)], W]|0>: build_BTw_2A-2C

    return {"a": r1["a"] + r2["a"] + r3["a"], "b": r1["b"] + r2["b"] + r3["b"],
            "aa": build_BTw_2A(W, T, L, H, H1, l0), "ab": build_BTw_2B(W, T, L, H, H1, l0),
            "bb": build_BTw_2C(W, T, L, H, H1, l0)}


def build_BTw_2A(W, T, L, H, H1, l0=1.0):
    """(B^T w) (aa) doubles: <0|(l0 + Lambda)[[Hbar, tau(ab,ij)], W]|0>, the derivative of the left-CCSD
    (aa) doubles residual <0|(l0 + Lambda)[Hbar, tau(ab,ij)]|0> along w = (W1, W2). W2 enters through
    the T2^2 terms of the CCSD equations; W1 through H1 = d H(T1 + e W1)/de = [H(T1), W1].
    Indices of every term are those of the stored tau(ab,ij), antisymmetrized at the end."""
    # <0|L2 [(V_N T2^2)_C/2, tau(ab,ij)] along W2|0>: second T2 derivative
    #     <mn||ef> w(af,mn) l(ab,ij)  -> (eb,ij)
    X = -0.25 * np.einsum("mnef,afmn,abij->ebij", H.aa.oovv, W.aa, L.aa, optimize=True)
    #     <mn||ef> w(eb,ij) l(ab,ij)  -> (af,mn)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H.aa.oovv, W.aa, L.aa, optimize=True)
    #     <mn~|ef~> w(af~,mn~) l(ab,ij)  -> (eb,ij)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H.ab.oovv, W.ab, L.aa, optimize=True)
    #     <mn||ef> w(ef,in) l(ab,ij)  -> (ab,mj)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H.aa.oovv, W.aa, L.aa, optimize=True)
    #     <mn||ef> w(ab,mj) l(ab,ij)  -> (ef,in)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H.aa.oovv, W.aa, L.aa, optimize=True)
    #     <mn~|ef~> w(ef~,in~) l(ab,ij)  -> (ab,mj)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H.ab.oovv, W.ab, L.aa, optimize=True)
    #     <mn||ef> w(af,in) l(ab,ij)  -> (eb,mj)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H.aa.oovv, W.aa, L.aa, optimize=True)
    #     <mn||ef> w(eb,mj) l(ab,ij)  -> (af,in)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H.aa.oovv, W.aa, L.aa, optimize=True)
    #     <mn~|ef~> w(af~,in~) l(ab,ij)  -> (eb,mj)
    X += np.einsum("mnef,afin,abij->ebmj", H.ab.oovv, W.ab, L.aa, optimize=True)
    #     <mn||ef> w(ef,ij) l(ab,ij)  -> (ab,mn)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H.aa.oovv, W.aa, L.aa, optimize=True)
    #     <mn||ef> w(ab,mn) l(ab,ij)  -> (ef,ij)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H.aa.oovv, W.aa, L.aa, optimize=True)
    #     <mn||ef> w(eb~,ij~) l(ab~,ij~)  -> (af,mn)
    X -= 0.5 * np.einsum("mnef,ebij,abij->afmn", H.aa.oovv, W.ab, L.ab, optimize=True)
    #     <mn||ef> w(ab~,mj~) l(ab~,ij~)  -> (ef,in)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H.aa.oovv, W.ab, L.ab, optimize=True)
    #     <mn||ef> w(fb~,nj~) l(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H.aa.oovv, W.ab, L.ab, optimize=True)
    #     <mn~|ef~> w(f~b~,n~j~) l(ab~,ij~)  -> (ae,im)
    X += np.einsum("mnef,fbnj,abij->aeim", H.ab.oovv, W.bb, L.ab, optimize=True)
    # <0|(l0 + L1 + L2)[H1, tau(ab,ij)]_C|0>, H1 = [H(T1), W1]
    #     l0 <ij||ab>'
    X += 0.25 * l0 * H1.aa.oovv.transpose(2, 3, 0, 1)
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
    return X


def build_BTw_2B(W, T, L, H, H1, l0=1.0):
    """(B^T w) (ab) doubles: <0|(l0 + Lambda)[[Hbar, tau(ab~,ij~)], W]|0>, the derivative of the left-CCSD
    (ab) doubles residual <0|(l0 + Lambda)[Hbar, tau(ab~,ij~)]|0> along w = (W1, W2). W2 enters through
    the T2^2 terms of the CCSD equations; W1 through H1 = d H(T1 + e W1)/de = [H(T1), W1].
    Indices of every term are those of the stored tau(ab,ij), antisymmetrized at the end."""
    # <0|L2 [(V_N T2^2)_C/2, tau(ab~,ij~)] along W2|0>: second T2 derivative
    #     <mn~|ef~> w(eb,ij) l(ab,ij)  -> (af~,mn~)
    X = -0.5 * np.einsum("mnef,ebij,abij->afmn", H.ab.oovv, W.aa, L.aa, optimize=True)
    #     <mn~|ef~> w(ab,mj) l(ab,ij)  -> (ef~,in~)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H.ab.oovv, W.aa, L.aa, optimize=True)
    #     <mn~|ef~> w(eb,mj) l(ab,ij)  -> (af~,in~)
    X += np.einsum("mnef,ebmj,abij->afin", H.ab.oovv, W.aa, L.aa, optimize=True)
    #     <m~n~||e~f~> w(af~,in~) l(ab,ij)  -> (be~,jm~)
    X += 0.5 * np.einsum("mnef,afin,abij->bejm", H.bb.oovv, W.ab, L.aa, optimize=True)
    #     <m~n~||e~f~> w(be~,jm~) l(ab,ij)  -> (af~,in~)
    X += 0.5 * np.einsum("mnef,bejm,abij->afin", H.bb.oovv, W.ab, L.aa, optimize=True)
    #     <mn||ef> w(af,mn) l(ab~,ij~)  -> (eb~,ij~)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H.aa.oovv, W.aa, L.ab, optimize=True)
    #     <mn~|ef~> w(af~,mn~) l(ab~,ij~)  -> (eb~,ij~)
    X -= np.einsum("mnef,afmn,abij->ebij", H.ab.oovv, W.ab, L.ab, optimize=True)
    #     <mn~|ef~> w(eb~,ij~) l(ab~,ij~)  -> (af~,mn~)
    X -= np.einsum("mnef,ebij,abij->afmn", H.ab.oovv, W.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> w(b~f~,m~n~) l(ab~,ij~)  -> (ae~,ij~)
    X -= 0.5 * np.einsum("mnef,bfmn,abij->aeij", H.bb.oovv, W.bb, L.ab, optimize=True)
    #     <nm~|fe~> w(fb~,nm~) l(ab~,ij~)  -> (ae~,ij~)
    X -= np.einsum("nmfe,fbnm,abij->aeij", H.ab.oovv, W.ab, L.ab, optimize=True)
    #     <nm~|fe~> w(ae~,ij~) l(ab~,ij~)  -> (fb~,nm~)
    X -= np.einsum("nmfe,aeij,abij->fbnm", H.ab.oovv, W.ab, L.ab, optimize=True)
    #     <mn||ef> w(ef,in) l(ab~,ij~)  -> (ab~,mj~)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H.aa.oovv, W.aa, L.ab, optimize=True)
    #     <mn~|ef~> w(ef~,in~) l(ab~,ij~)  -> (ab~,mj~)
    X -= np.einsum("mnef,efin,abij->abmj", H.ab.oovv, W.ab, L.ab, optimize=True)
    #     <mn~|ef~> w(ab~,mj~) l(ab~,ij~)  -> (ef~,in~)
    X -= np.einsum("mnef,abmj,abij->efin", H.ab.oovv, W.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> w(e~f~,j~n~) l(ab~,ij~)  -> (ab~,im~)
    X -= 0.5 * np.einsum("mnef,efjn,abij->abim", H.bb.oovv, W.bb, L.ab, optimize=True)
    #     <nm~|fe~> w(fe~,nj~) l(ab~,ij~)  -> (ab~,im~)
    X -= np.einsum("nmfe,fenj,abij->abim", H.ab.oovv, W.ab, L.ab, optimize=True)
    #     <nm~|fe~> w(ab~,im~) l(ab~,ij~)  -> (fe~,nj~)
    X -= np.einsum("nmfe,abim,abij->fenj", H.ab.oovv, W.ab, L.ab, optimize=True)
    #     <mn||ef> w(ae,im) l(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.aa.oovv, W.aa, L.ab, optimize=True)
    #     <nm~|fe~> w(ae~,im~) l(ab~,ij~)  -> (fb~,nj~)
    X += np.einsum("nmfe,aeim,abij->fbnj", H.ab.oovv, W.ab, L.ab, optimize=True)
    #     <nm~|fe~> w(fb~,nj~) l(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("nmfe,fbnj,abij->aeim", H.ab.oovv, W.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> w(f~b~,n~j~) l(ab~,ij~)  -> (ae~,im~)
    X += np.einsum("mnef,fbnj,abij->aeim", H.bb.oovv, W.bb, L.ab, optimize=True)
    #     <mn~|ef~> w(af~,mj~) l(ab~,ij~)  -> (eb~,in~)
    X += np.einsum("mnef,afmj,abij->ebin", H.ab.oovv, W.ab, L.ab, optimize=True)
    #     <mn~|ef~> w(eb~,in~) l(ab~,ij~)  -> (af~,mj~)
    X += np.einsum("mnef,ebin,abij->afmj", H.ab.oovv, W.ab, L.ab, optimize=True)
    #     <mn~|ef~> w(ef~,ij~) l(ab~,ij~)  -> (ab~,mn~)
    X += np.einsum("mnef,efij,abij->abmn", H.ab.oovv, W.ab, L.ab, optimize=True)
    #     <mn~|ef~> w(ab~,mn~) l(ab~,ij~)  -> (ef~,ij~)
    X += np.einsum("mnef,abmn,abij->efij", H.ab.oovv, W.ab, L.ab, optimize=True)
    #     <nm~|fe~> w(e~b~,i~j~) l(a~b~,i~j~)  -> (fa~,nm~)
    X -= 0.5 * np.einsum("nmfe,ebij,abij->fanm", H.ab.oovv, W.bb, L.bb, optimize=True)
    #     <nm~|fe~> w(a~b~,m~j~) l(a~b~,i~j~)  -> (fe~,ni~)
    X -= 0.5 * np.einsum("nmfe,abmj,abij->feni", H.ab.oovv, W.bb, L.bb, optimize=True)
    #     <mn~|ef~> w(a~f~,i~n~) l(a~b~,i~j~)  -> (eb~,mj~)
    X += np.einsum("mnef,afin,abij->ebmj", H.ab.oovv, W.bb, L.bb, optimize=True)
    #     <mn||ef> w(fa~,ni~) l(a~b~,i~j~)  -> (eb~,mj~)
    X += 0.5 * np.einsum("mnef,fani,abij->ebmj", H.aa.oovv, W.ab, L.bb, optimize=True)
    #     <mn||ef> w(eb~,mj~) l(a~b~,i~j~)  -> (fa~,ni~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->fani", H.aa.oovv, W.ab, L.bb, optimize=True)
    # <0|(l0 + L1 + L2)[H1, tau(ab~,ij~)]_C|0>, H1 = [H(T1), W1]
    #     l0 <ij~|ab~>'
    X += l0 * H1.ab.oovv.transpose(2, 3, 0, 1)
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
    return X


def build_BTw_2C(W, T, L, H, H1, l0=1.0):
    """(B^T w) (bb) doubles: <0|(l0 + Lambda)[[Hbar, tau(a~b~,i~j~)], W]|0>, the derivative of the left-CCSD
    (bb) doubles residual <0|(l0 + Lambda)[Hbar, tau(a~b~,i~j~)]|0> along w = (W1, W2). W2 enters through
    the T2^2 terms of the CCSD equations; W1 through H1 = d H(T1 + e W1)/de = [H(T1), W1].
    Indices of every term are those of the stored tau(ab,ij), antisymmetrized at the end."""
    # <0|L2 [(V_N T2^2)_C/2, tau(a~b~,i~j~)] along W2|0>: second T2 derivative
    #     <m~n~||e~f~> w(ae~,ij~) l(ab~,ij~)  -> (b~f~,m~n~)
    X = -0.5 * np.einsum("mnef,aeij,abij->bfmn", H.bb.oovv, W.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> w(ab~,im~) l(ab~,ij~)  -> (e~f~,j~n~)
    X -= 0.5 * np.einsum("mnef,abim,abij->efjn", H.bb.oovv, W.ab, L.ab, optimize=True)
    #     <mn~|ef~> w(ae,im) l(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.ab.oovv, W.aa, L.ab, optimize=True)
    #     <m~n~||e~f~> w(ae~,im~) l(ab~,ij~)  -> (f~b~,n~j~)
    X += np.einsum("mnef,aeim,abij->fbnj", H.bb.oovv, W.ab, L.ab, optimize=True)
    #     <m~n~||e~f~> w(a~f~,m~n~) l(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.25 * np.einsum("mnef,afmn,abij->ebij", H.bb.oovv, W.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> w(e~b~,i~j~) l(a~b~,i~j~)  -> (a~f~,m~n~)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H.bb.oovv, W.bb, L.bb, optimize=True)
    #     <nm~|fe~> w(fa~,nm~) l(a~b~,i~j~)  -> (e~b~,i~j~)
    X -= 0.5 * np.einsum("nmfe,fanm,abij->ebij", H.ab.oovv, W.ab, L.bb, optimize=True)
    #     <m~n~||e~f~> w(e~f~,i~n~) l(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H.bb.oovv, W.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> w(a~b~,m~j~) l(a~b~,i~j~)  -> (e~f~,i~n~)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H.bb.oovv, W.bb, L.bb, optimize=True)
    #     <nm~|fe~> w(fe~,ni~) l(a~b~,i~j~)  -> (a~b~,m~j~)
    X -= 0.5 * np.einsum("nmfe,feni,abij->abmj", H.ab.oovv, W.ab, L.bb, optimize=True)
    #     <m~n~||e~f~> w(a~f~,i~n~) l(a~b~,i~j~)  -> (e~b~,m~j~)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H.bb.oovv, W.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> w(e~b~,m~j~) l(a~b~,i~j~)  -> (a~f~,i~n~)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H.bb.oovv, W.bb, L.bb, optimize=True)
    #     <mn~|ef~> w(eb~,mj~) l(a~b~,i~j~)  -> (a~f~,i~n~)
    X += np.einsum("mnef,ebmj,abij->afin", H.ab.oovv, W.ab, L.bb, optimize=True)
    #     <m~n~||e~f~> w(e~f~,i~j~) l(a~b~,i~j~)  -> (a~b~,m~n~)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H.bb.oovv, W.bb, L.bb, optimize=True)
    #     <m~n~||e~f~> w(a~b~,m~n~) l(a~b~,i~j~)  -> (e~f~,i~j~)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H.bb.oovv, W.bb, L.bb, optimize=True)
    # <0|(l0 + L1 + L2)[H1, tau(a~b~,i~j~)]_C|0>, H1 = [H(T1), W1]
    #     l0 <i~j~||a~b~>'
    X += 0.25 * l0 * H1.bb.oovv.transpose(2, 3, 0, 1)
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
