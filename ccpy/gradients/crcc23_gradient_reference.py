"""Analytic nuclear gradient of CR-CC(2,3)_A, factorized (density fitting or Cholesky): the
full-array reference implementation (o^3 v^3 triples, kept for validation and pedagogy; the
production gradient is ccpy.gradients.crcc23_gradient).

Theory (CRCC23 Gradient Method.pdf)
-----------------------------------
    delta_A = sum_K M_K N_K / D_K,   K = (i<j<k, a<b<c)
    M_K = <K|Hbar|0>,   N_K = <0|(L1 + L2) Hbar|K>,   D_K = f_ii + f_jj + f_kk - f_aa - f_bb - f_cc

With r_K = M_K / D_K and l_K = N_K / D_K held fixed, delta_A = sum_K (l_K M_K + r_K N_K -
l_K r_K D_K) is stationary in r and l, and

    s_lambda = d delta / d lambda = <mu| Hbar R3 |0>,   s_t = d delta / d t
    A w = -s_lambda,   A^T z = -(s_t + B^T w)        (A: the CCSD Jacobian, B^T w: d g/dT . w)
    d delta / dx = delta_x + z.R_x + w.g_x            (g: the left-CCSD residual)

Everything that depends on T is evaluated with the CCS-transformed Hamiltonian
H = e^{-T1} H e^{T1} (ccpy.hbar.hbar_ccs_chol) and T1 = 0, because Hbar = e^{-T2} H e^{T2}:

  * M = <K|(H T2)_C + 1/2 (H T2^2)_C|0> and N = <0|(L1 + L2) H|K> (build_M3A-D, build_L3A-D);
  * s_lambda = d(R3 . N)/dL (build_s_lambda_*); the doubles of s_t = d(L3 . M)/dT2
    (build_s_t_2*); the singles of s_t are orbital-rotation derivatives of the triples
    densities (a T1 step rotates H: H -> H + e [H, a+ i]);
  * B^T w = d/de of the left-CCSD residual along w: the T2 part of w through the second
    derivative of the CCSD equations (build_BTw_2*), the T1 part through
    [H, W1] (ccpy.hbar.hbar_ccs_chol.build_hbar_ccs_chol_derivative);
  * the densities (ccpy.density.ccsd_rdm, ccpy.density.crcc23_rdm) are those of the blocks of
    H(T1); ccpy.gradients.density_gradient.build_DY takes them to the bare frame.
w and z are solved with ccpy's EOMCCSD (HR) and left-CCSD (LH) Jacobian actions on Hbar.

As for CCSD (ccpy.gradients.ccsd_gradient), the gradient is assembled in three stages, per spin:
the amplitude-relaxed densities of E_CCSD + delta_A (calc_crcc23A_density: the CCSD
Lagrangian, the triples, the z and w multipliers, and the denominators), their orbital
relaxation (ccpy.gradients.orbital_response), and the contraction with the derivative
integrals (ccpy.gradients.derivative_integrals).
No quantity with four virtual indices is stored; the triples are held as o^3 v^3 blocks.
"""

import copy
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse.linalg as spla

from ccpy.density.ccsd_rdm import calc_ccsd_rdm, calc_ccsd_rdm_derivative, calc_rdm2_vvvv_slice
from ccpy.density.crcc23_rdm import calc_crcc23_rdm, calc_crcc23_rdm1_denominator
from ccpy.gradients.ccsd_gradient import calc_ccsd_density
from ccpy.gradients.derivative_integrals import contract_derivative_integrals, print_gradient
from ccpy.gradients.density_gradient import (add_blocks, add_densities, build_DY, commutator, get_bare_integrals,
                                             rotate_density, rotation_gradient, zero_like)
from ccpy.gradients.orbital_response import orbital_relaxation
from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol, build_hbar_ccs_chol_derivative
from ccpy.models.integrals import Integral


def calc_crcc23_gradient(T, L, Hbar, reference, zvector_tol=1.0e-10, verbose=True):
    """Analytic CR-CC(2,3)_A nuclear gradient from converged CCSD (T), left-CCSD (L), and the
    CCSD HBar (Hbar), on the SCF reference (ccpy.methods.Reference)."""
    t0 = time.perf_counter()
    H = build_hbar_ccs_chol(T, reference.H, reference.system)
    response = calc_crcc23A_response(T, L, Hbar, H, reference.H, reference, tol=zvector_tol)
    D1, Y = calc_crcc23A_density(T, L, response, H, reference)
    D, Y, W = orbital_relaxation(D1, Y, reference, zvector_tol=zvector_tol, verbose=verbose)
    grad = contract_derivative_integrals(D, Y, W, reference)
    if verbose:
        print_gradient(reference.molecule, grad, "CR-CC(2,3)_A Analytic Gradient", time.perf_counter() - t0)
    return grad


def calc_crcc23A_response(T, L, Hbar, H, H0, reference, tol=1.0e-10):
    """R3, L3, s_lambda, s_t, w, B^T w and z of CR-CC(2,3)_A (H: build_hbar_ccs_chol, H0: bare)."""
    system = reference.system
    R3, L3 = calc_r3_l3(T, L, H, H0, system)
    s_lambda = calc_s_lambda(R3, H, system)
    s_t = calc_s_t(R3, L3, T, L, H, reference)
    w = solve_w(s_lambda, T, Hbar, system, tol)
    BTw = calc_BTw(w, T, L, H, reference)
    z = solve_z({k: -(s_t[k] + BTw[k]) for k in s_t}, T, L, Hbar, system, tol)
    return {"R3": R3, "L3": L3, "s_lambda": s_lambda, "s_t": s_t, "w": w, "BTw": BTw, "z": z}


def calc_crcc23A_density(T, L, response, H, reference):
    """Amplitude-relaxed (D1, Y) of E_CCSD + delta_A, per spin, conjugate to the bare (h, B):
    the CCSD Lagrangian, the triples (R3, L3), the z (CCSD amplitude equations) and w
    (left-CCSD equations) multipliers, and the orbital-energy denominators."""
    system = reference.system
    R3, L3, w = response["R3"], response["L3"], response["w"]
    z = SimpleNamespace(**{k: _get(response["z"], k) for k in ("a", "b", "aa", "ab", "bb")})
    _, B = get_bare_integrals(reference)
    return add_densities(
        calc_ccsd_density(T, L, H, reference),
        build_DY(*calc_crcc23_rdm(R3, L3, T, L, system), None, H.chol, T, B, reference),
        build_DY(*calc_ccsd_rdm(T, z, system, energy=False), lambda a, s: calc_rdm2_vvvv_slice(z, T, a, s),
                 H.chol, T, B, reference),
        calc_w_density(w, T, L, H, reference),
        calc_denominator_density(R3, L3, H, reference),
    )


def calc_denominator_density(R3, L3, H, reference):
    """(D1, Y) of -sum_K l_K r_K D_K (Eq. 9); the denominators are bare Fock diagonals, so
    there is no T1 to undo (and no two-body density: H.chol only sets the shapes)."""
    system = reference.system
    _, B = get_bare_integrals(reference)
    return build_DY(calc_crcc23_rdm1_denominator(R3, L3, system), Integral.from_empty(system, 2, use_none=True),
                    None, H.chol, None, B, reference)


def calc_w_density(w, T, L, H, reference):
    """(D1, Y) of d L(T + e w)/de at fixed Lambda: the T1 part of w rotates the Lagrangian
    densities, the T2 part comes from calc_ccsd_rdm_derivative."""
    system, nfz = reference.system, reference.nfrozen
    W = SimpleNamespace(**{k: _get(w, k) for k in ("a", "b", "aa", "ab", "bb")})
    _, B = get_bare_integrals(reference)
    H1 = build_hbar_ccs_chol_derivative(W, H, system)
    dB = commutator(B, W.a, W.b, nfz)
    rdm1, rdm2 = calc_ccsd_rdm(T, L, system)
    vvvv = lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s)
    D1L, YL = build_DY(rdm1, rdm2, vvvv, H.chol, T, B, reference, ref_weight=True)
    _, YLdB = build_DY(rdm1, rdm2, vvvv, H1.chol, T, dB, reference, ref_weight=True)
    D1w, Yw = build_DY(*calc_ccsd_rdm_derivative(T, W, L, system), lambda a, s: calc_rdm2_vvvv_slice(L, W, a, s),
                       H.chol, T, B, reference)
    D1 = add_blocks(D1w, rotate_density(D1L, W.a, W.b, nfz))
    Y = add_blocks(add_blocks(Yw, rotate_density(YL, W.a, W.b, nfz)), YLdB)
    return D1, Y


def _get(x, k):
    return x[k] if isinstance(x, dict) else getattr(x, k)


# ======================================================================================
# R3 and L3
# ======================================================================================
def calc_r3_l3(T, L, H, H0, system):
    """r_K = M_K / D_K and l_K = N_K / D_K (all antisymmetric copies), D_K from the bare
    Fock diagonal (H0)."""
    eo = {s: np.diag(getattr(H0, s).oo) for s in ("a", "b")}
    ev = {s: np.diag(getattr(H0, s).vv) for s in ("a", "b")}
    R3, L3 = SimpleNamespace(), SimpleNamespace()
    for spin, M3, N3 in (("aaa", build_M3A(T, H), build_L3A(L, H)),
                         ("aab", build_M3B(T, H), build_L3B(L, H)),
                         ("abb", build_M3C(T, H), build_L3C(L, H)),
                         ("bbb", build_M3D(T, H), build_L3D(L, H))):
        s1, s2, s3 = spin
        D = (eo[s1][None, None, None, :, None, None] + eo[s2][None, None, None, None, :, None]
             + eo[s3][None, None, None, None, None, :] - ev[s1][:, None, None, None, None, None]
             - ev[s2][None, :, None, None, None, None] - ev[s3][None, None, :, None, None, None])
        setattr(R3, spin, M3 / D)
        setattr(L3, spin, N3 / D)
    return R3, L3


def build_M3A(T, H):
    """<ijkabc|Hbar|0> (aaa), with H = H(T1) (T1 = 0 in the contractions)."""
    # <ijkabc|(V_N T2)_C|0>
    #     <am||ij> t(bc,mk)
    M3 = -0.25 * np.einsum("amij,bcmk->abcijk", H.aa.vooo, T.aa, optimize=True)
    # <ijkabc|(F_N T2^2)_C|0> / 2
    #     f(me) t(ae,ij) t(bc,mk)
    M3 -= 0.25 * np.einsum("me,aeij,bcmk->abcijk", H.a.ov, T.aa, T.aa, optimize=True)
    # <ijkabc|(V_N T2^2)_C|0> / 2
    #     <am||ef> t(ef,ij) t(bc,mk)
    M3 -= 0.125 * np.einsum("amef,efij,bcmk->abcijk", H.aa.vovv, T.aa, T.aa, optimize=True)
    #     <mn||jf> t(af,in) t(bc,mk)
    M3 -= 0.5 * np.einsum("mnjf,afin,bcmk->abcijk", H.aa.ooov, T.aa, T.aa, optimize=True)
    #     <mn~|jf~> t(af~,in~) t(bc,mk)
    M3 -= 0.5 * np.einsum("mnjf,afin,bcmk->abcijk", H.ab.ooov, T.ab, T.aa, optimize=True)
    # <ijkabc|(V_N T2)_C|0>
    #     <ab||ie> t(ec,jk)
    M3 += 0.25 * np.einsum("abie,ecjk->abcijk", H.aa.vvov, T.aa, optimize=True)
    # <ijkabc|(V_N T2^2)_C|0> / 2
    #     <mn||ie> t(ab,mn) t(ec,jk)
    M3 += 0.125 * np.einsum("mnie,abmn,ecjk->abcijk", H.aa.ooov, T.aa, T.aa, optimize=True)
    #     <bn||ef> t(af,in) t(ec,jk)
    M3 += 0.5 * np.einsum("bnef,afin,ecjk->abcijk", H.aa.vovv, T.aa, T.aa, optimize=True)
    #     <bn~|ef~> t(af~,in~) t(ec,jk)
    M3 += 0.5 * np.einsum("bnef,afin,ecjk->abcijk", H.ab.vovv, T.ab, T.aa, optimize=True)
    M3 -= np.transpose(M3, (0, 2, 1, 3, 4, 5)) + np.transpose(M3, (2, 1, 0, 3, 4, 5))
    M3 -= np.transpose(M3, (1, 0, 2, 3, 4, 5))
    M3 -= np.transpose(M3, (0, 1, 2, 3, 5, 4)) + np.transpose(M3, (0, 1, 2, 5, 4, 3))
    M3 -= np.transpose(M3, (0, 1, 2, 4, 3, 5))
    return M3


def build_M3B(T, H):
    """<ijk~abc~|Hbar|0> (aab)."""
    # <ijk~abc~|(V_N T2)_C|0>
    #     <bc~|ek~> t(ae,ij)
    M3 = 0.5 * np.einsum("bcek,aeij->abcijk", H.ab.vvvo, T.aa, optimize=True)
    # <ijk~abc~|(F_N T2^2)_C|0> / 2
    #     f(me) t(bc~,mk~) t(ae,ij)
    M3 -= 0.5 * np.einsum("me,bcmk,aeij->abcijk", H.a.ov, T.ab, T.aa, optimize=True)
    # <ijk~abc~|(V_N T2^2)_C|0> / 2
    #     <nm~|ek~> t(bc~,nm~) t(ae,ij)
    M3 += 0.5 * np.einsum("nmek,bcnm,aeij->abcijk", H.ab.oovo, T.ab, T.aa, optimize=True)
    #     <bm~|ef~> t(c~f~,k~m~) t(ae,ij)
    M3 += 0.5 * np.einsum("bmef,cfkm,aeij->abcijk", H.ab.vovv, T.bb, T.aa, optimize=True)
    #     <bm||ef> t(fc~,mk~) t(ae,ij)
    M3 += 0.5 * np.einsum("bmef,fcmk,aeij->abcijk", H.aa.vovv, T.ab, T.aa, optimize=True)
    #     <mc~|ef~> t(bf~,mk~) t(ae,ij)
    M3 -= 0.5 * np.einsum("mcef,bfmk,aeij->abcijk", H.ab.ovvv, T.ab, T.aa, optimize=True)
    # <ijk~abc~|(V_N T2)_C|0>
    #     <mc~|jk~> t(ab,im)
    M3 -= 0.5 * np.einsum("mcjk,abim->abcijk", H.ab.ovoo, T.aa, optimize=True)
    # <ijk~abc~|(V_N T2^2)_C|0> / 2
    #     <mn||jf> t(fc~,nk~) t(ab,im)
    M3 -= 0.5 * np.einsum("mnjf,fcnk,abim->abcijk", H.aa.ooov, T.ab, T.aa, optimize=True)
    #     <mn~|jf~> t(f~c~,n~k~) t(ab,im)
    M3 -= 0.5 * np.einsum("mnjf,fcnk,abim->abcijk", H.ab.ooov, T.bb, T.aa, optimize=True)
    #     <mn~|fk~> t(fc~,jn~) t(ab,im)
    M3 += 0.5 * np.einsum("mnfk,fcjn,abim->abcijk", H.ab.oovo, T.ab, T.aa, optimize=True)
    #     <mc~|fe~> t(fe~,jk~) t(ab,im)
    M3 -= 0.5 * np.einsum("mcfe,fejk,abim->abcijk", H.ab.ovvv, T.ab, T.aa, optimize=True)
    # <ijk~abc~|(V_N T2)_C|0>
    #     <ac~|ie~> t(be~,jk~)
    M3 += np.einsum("acie,bejk->abcijk", H.ab.vvov, T.ab, optimize=True)
    # <ijk~abc~|(F_N T2^2)_C|0> / 2
    #     f(m~e~) t(ac~,im~) t(be~,jk~)
    M3 -= np.einsum("me,acim,bejk->abcijk", H.b.ov, T.ab, T.ab, optimize=True)
    # <ijk~abc~|(V_N T2^2)_C|0> / 2
    #     <nm~|ie~> t(ac~,nm~) t(be~,jk~)
    M3 += np.einsum("nmie,acnm,bejk->abcijk", H.ab.ooov, T.ab, T.ab, optimize=True)
    #     <mc~|fe~> t(af,im) t(be~,jk~)
    M3 += np.einsum("mcfe,afim,bejk->abcijk", H.ab.ovvv, T.aa, T.ab, optimize=True)
    #     <c~m~||e~f~> t(af~,im~) t(be~,jk~)
    M3 += np.einsum("cmef,afim,bejk->abcijk", H.bb.vovv, T.ab, T.ab, optimize=True)
    #     <am~|fe~> t(fc~,im~) t(be~,jk~)
    M3 -= np.einsum("amfe,fcim,bejk->abcijk", H.ab.vovv, T.ab, T.ab, optimize=True)
    # <ijk~abc~|(V_N T2)_C|0>
    #     <am~|ik~> t(bc~,jm~)
    M3 -= np.einsum("amik,bcjm->abcijk", H.ab.vooo, T.ab, optimize=True)
    # <ijk~abc~|(V_N T2^2)_C|0> / 2
    #     <m~n~||k~f~> t(af~,in~) t(bc~,jm~)
    M3 -= np.einsum("mnkf,afin,bcjm->abcijk", H.bb.ooov, T.ab, T.ab, optimize=True)
    #     <nm~|fk~> t(af,in) t(bc~,jm~)
    M3 -= np.einsum("nmfk,afin,bcjm->abcijk", H.ab.oovo, T.aa, T.ab, optimize=True)
    #     <nm~|if~> t(af~,nk~) t(bc~,jm~)
    M3 += np.einsum("nmif,afnk,bcjm->abcijk", H.ab.ooov, T.ab, T.ab, optimize=True)
    #     <am~|ef~> t(ef~,ik~) t(bc~,jm~)
    M3 -= np.einsum("amef,efik,bcjm->abcijk", H.ab.vovv, T.ab, T.ab, optimize=True)
    # <ijk~abc~|(V_N T2)_C|0>
    #     <ab||ie> t(ec~,jk~)
    M3 += 0.5 * np.einsum("abie,ecjk->abcijk", H.aa.vvov, T.ab, optimize=True)
    # <ijk~abc~|(F_N T2^2)_C|0> / 2
    #     f(me) t(ab,im) t(ec~,jk~)
    M3 -= 0.5 * np.einsum("me,abim,ecjk->abcijk", H.a.ov, T.aa, T.ab, optimize=True)
    # <ijk~abc~|(V_N T2^2)_C|0> / 2
    #     <mn||ie> t(ab,mn) t(ec~,jk~)
    M3 += 0.25 * np.einsum("mnie,abmn,ecjk->abcijk", H.aa.ooov, T.aa, T.ab, optimize=True)
    #     <bn||ef> t(af,in) t(ec~,jk~)
    M3 += np.einsum("bnef,afin,ecjk->abcijk", H.aa.vovv, T.aa, T.ab, optimize=True)
    #     <bn~|ef~> t(af~,in~) t(ec~,jk~)
    M3 += np.einsum("bnef,afin,ecjk->abcijk", H.ab.vovv, T.ab, T.ab, optimize=True)
    # <ijk~abc~|(V_N T2)_C|0>
    #     <am||ij> t(bc~,mk~)
    M3 -= 0.5 * np.einsum("amij,bcmk->abcijk", H.aa.vooo, T.ab, optimize=True)
    # <ijk~abc~|(V_N T2^2)_C|0> / 2
    #     <am||ef> t(ef,ij) t(bc~,mk~)
    M3 -= 0.25 * np.einsum("amef,efij,bcmk->abcijk", H.aa.vovv, T.aa, T.ab, optimize=True)
    #     <mn||jf> t(af,in) t(bc~,mk~)
    M3 -= np.einsum("mnjf,afin,bcmk->abcijk", H.aa.ooov, T.aa, T.ab, optimize=True)
    #     <mn~|jf~> t(af~,in~) t(bc~,mk~)
    M3 -= np.einsum("mnjf,afin,bcmk->abcijk", H.ab.ooov, T.ab, T.ab, optimize=True)
    M3 -= np.transpose(M3, (1, 0, 2, 3, 4, 5))
    M3 -= np.transpose(M3, (0, 1, 2, 4, 3, 5))
    return M3


def build_M3C(T, H):
    """<ij~k~ab~c~|Hbar|0> (abb)."""
    # <ij~k~ab~c~|(V_N T2)_C|0>
    #     <cb~|ke~> t(a~e~,i~j~)
    M3 = 0.5 * np.einsum("cbke,aeij->cabkij", H.ab.vvov, T.bb, optimize=True)
    # <ij~k~ab~c~|(F_N T2^2)_C|0> / 2
    #     f(m~e~) t(cb~,km~) t(a~e~,i~j~)
    M3 -= 0.5 * np.einsum("me,cbkm,aeij->cabkij", H.b.ov, T.ab, T.bb, optimize=True)
    # <ij~k~ab~c~|(V_N T2^2)_C|0> / 2
    #     <mn~|ke~> t(cb~,mn~) t(a~e~,i~j~)
    M3 += 0.5 * np.einsum("mnke,cbmn,aeij->cabkij", H.ab.ooov, T.ab, T.bb, optimize=True)
    #     <mb~|fe~> t(cf,km) t(a~e~,i~j~)
    M3 += 0.5 * np.einsum("mbfe,cfkm,aeij->cabkij", H.ab.ovvv, T.aa, T.bb, optimize=True)
    #     <b~m~||e~f~> t(cf~,km~) t(a~e~,i~j~)
    M3 += 0.5 * np.einsum("bmef,cfkm,aeij->cabkij", H.bb.vovv, T.ab, T.bb, optimize=True)
    #     <cm~|fe~> t(fb~,km~) t(a~e~,i~j~)
    M3 -= 0.5 * np.einsum("cmfe,fbkm,aeij->cabkij", H.ab.vovv, T.ab, T.bb, optimize=True)
    # <ij~k~ab~c~|(V_N T2)_C|0>
    #     <cm~|kj~> t(a~b~,i~m~)
    M3 -= 0.5 * np.einsum("cmkj,abim->cabkij", H.ab.vooo, T.bb, optimize=True)
    # <ij~k~ab~c~|(V_N T2^2)_C|0> / 2
    #     <m~n~||j~f~> t(cf~,kn~) t(a~b~,i~m~)
    M3 -= 0.5 * np.einsum("mnjf,cfkn,abim->cabkij", H.bb.ooov, T.ab, T.bb, optimize=True)
    #     <nm~|fj~> t(fc,nk) t(a~b~,i~m~)
    M3 -= 0.5 * np.einsum("nmfj,fcnk,abim->cabkij", H.ab.oovo, T.aa, T.bb, optimize=True)
    #     <nm~|kf~> t(cf~,nj~) t(a~b~,i~m~)
    M3 += 0.5 * np.einsum("nmkf,cfnj,abim->cabkij", H.ab.ooov, T.ab, T.bb, optimize=True)
    #     <cm~|ef~> t(ef~,kj~) t(a~b~,i~m~)
    M3 -= 0.5 * np.einsum("cmef,efkj,abim->cabkij", H.ab.vovv, T.ab, T.bb, optimize=True)
    # <ij~k~ab~c~|(V_N T2)_C|0>
    #     <ca~|ei~> t(eb~,kj~)
    M3 += np.einsum("caei,ebkj->cabkij", H.ab.vvvo, T.ab, optimize=True)
    # <ij~k~ab~c~|(F_N T2^2)_C|0> / 2
    #     f(me) t(ca~,mi~) t(eb~,kj~)
    M3 -= np.einsum("me,cami,ebkj->cabkij", H.a.ov, T.ab, T.ab, optimize=True)
    # <ij~k~ab~c~|(V_N T2^2)_C|0> / 2
    #     <mn~|ei~> t(ca~,mn~) t(eb~,kj~)
    M3 += np.einsum("mnei,camn,ebkj->cabkij", H.ab.oovo, T.ab, T.ab, optimize=True)
    #     <cm~|ef~> t(a~f~,i~m~) t(eb~,kj~)
    M3 += np.einsum("cmef,afim,ebkj->cabkij", H.ab.vovv, T.bb, T.ab, optimize=True)
    #     <cm||ef> t(fa~,mi~) t(eb~,kj~)
    M3 += np.einsum("cmef,fami,ebkj->cabkij", H.aa.vovv, T.ab, T.ab, optimize=True)
    #     <ma~|ef~> t(cf~,mi~) t(eb~,kj~)
    M3 -= np.einsum("maef,cfmi,ebkj->cabkij", H.ab.ovvv, T.ab, T.ab, optimize=True)
    # <ij~k~ab~c~|(V_N T2)_C|0>
    #     <ma~|ki~> t(cb~,mj~)
    M3 -= np.einsum("maki,cbmj->cabkij", H.ab.ovoo, T.ab, optimize=True)
    # <ij~k~ab~c~|(V_N T2^2)_C|0> / 2
    #     <mn||kf> t(fa~,ni~) t(cb~,mj~)
    M3 -= np.einsum("mnkf,fani,cbmj->cabkij", H.aa.ooov, T.ab, T.ab, optimize=True)
    #     <mn~|kf~> t(a~f~,i~n~) t(cb~,mj~)
    M3 -= np.einsum("mnkf,afin,cbmj->cabkij", H.ab.ooov, T.bb, T.ab, optimize=True)
    #     <mn~|fi~> t(fa~,kn~) t(cb~,mj~)
    M3 += np.einsum("mnfi,fakn,cbmj->cabkij", H.ab.oovo, T.ab, T.ab, optimize=True)
    #     <ma~|fe~> t(fe~,ki~) t(cb~,mj~)
    M3 -= np.einsum("mafe,feki,cbmj->cabkij", H.ab.ovvv, T.ab, T.ab, optimize=True)
    # <ij~k~ab~c~|(V_N T2)_C|0>
    #     <a~b~||i~e~> t(ce~,kj~)
    M3 += 0.5 * np.einsum("abie,cekj->cabkij", H.bb.vvov, T.ab, optimize=True)
    # <ij~k~ab~c~|(F_N T2^2)_C|0> / 2
    #     f(m~e~) t(a~b~,i~m~) t(ce~,kj~)
    M3 -= 0.5 * np.einsum("me,abim,cekj->cabkij", H.b.ov, T.bb, T.ab, optimize=True)
    # <ij~k~ab~c~|(V_N T2^2)_C|0> / 2
    #     <m~n~||i~e~> t(a~b~,m~n~) t(ce~,kj~)
    M3 += 0.25 * np.einsum("mnie,abmn,cekj->cabkij", H.bb.ooov, T.bb, T.ab, optimize=True)
    #     <b~n~||e~f~> t(a~f~,i~n~) t(ce~,kj~)
    M3 += np.einsum("bnef,afin,cekj->cabkij", H.bb.vovv, T.bb, T.ab, optimize=True)
    #     <nb~|fe~> t(fa~,ni~) t(ce~,kj~)
    M3 += np.einsum("nbfe,fani,cekj->cabkij", H.ab.ovvv, T.ab, T.ab, optimize=True)
    # <ij~k~ab~c~|(V_N T2)_C|0>
    #     <a~m~||i~j~> t(cb~,km~)
    M3 -= 0.5 * np.einsum("amij,cbkm->cabkij", H.bb.vooo, T.ab, optimize=True)
    # <ij~k~ab~c~|(V_N T2^2)_C|0> / 2
    #     <a~m~||e~f~> t(e~f~,i~j~) t(cb~,km~)
    M3 -= 0.25 * np.einsum("amef,efij,cbkm->cabkij", H.bb.vovv, T.bb, T.ab, optimize=True)
    #     <m~n~||j~f~> t(a~f~,i~n~) t(cb~,km~)
    M3 -= np.einsum("mnjf,afin,cbkm->cabkij", H.bb.ooov, T.bb, T.ab, optimize=True)
    #     <nm~|fj~> t(fa~,ni~) t(cb~,km~)
    M3 -= np.einsum("nmfj,fani,cbkm->cabkij", H.ab.oovo, T.ab, T.ab, optimize=True)
    M3 -= np.transpose(M3, (0, 2, 1, 3, 4, 5))
    M3 -= np.transpose(M3, (0, 1, 2, 3, 5, 4))
    return M3


def build_M3D(T, H):
    """<i~j~k~a~b~c~|Hbar|0> (bbb)."""
    # <i~j~k~a~b~c~|(V_N T2)_C|0>
    #     <a~m~||i~j~> t(b~c~,m~k~)
    M3 = -0.25 * np.einsum("amij,bcmk->abcijk", H.bb.vooo, T.bb, optimize=True)
    # <i~j~k~a~b~c~|(F_N T2^2)_C|0> / 2
    #     f(m~e~) t(a~e~,i~j~) t(b~c~,m~k~)
    M3 -= 0.25 * np.einsum("me,aeij,bcmk->abcijk", H.b.ov, T.bb, T.bb, optimize=True)
    # <i~j~k~a~b~c~|(V_N T2^2)_C|0> / 2
    #     <a~m~||e~f~> t(e~f~,i~j~) t(b~c~,m~k~)
    M3 -= 0.125 * np.einsum("amef,efij,bcmk->abcijk", H.bb.vovv, T.bb, T.bb, optimize=True)
    #     <m~n~||j~f~> t(a~f~,i~n~) t(b~c~,m~k~)
    M3 -= 0.5 * np.einsum("mnjf,afin,bcmk->abcijk", H.bb.ooov, T.bb, T.bb, optimize=True)
    #     <nm~|fj~> t(fa~,ni~) t(b~c~,m~k~)
    M3 -= 0.5 * np.einsum("nmfj,fani,bcmk->abcijk", H.ab.oovo, T.ab, T.bb, optimize=True)
    # <i~j~k~a~b~c~|(V_N T2)_C|0>
    #     <a~b~||i~e~> t(e~c~,j~k~)
    M3 += 0.25 * np.einsum("abie,ecjk->abcijk", H.bb.vvov, T.bb, optimize=True)
    # <i~j~k~a~b~c~|(V_N T2^2)_C|0> / 2
    #     <m~n~||i~e~> t(a~b~,m~n~) t(e~c~,j~k~)
    M3 += 0.125 * np.einsum("mnie,abmn,ecjk->abcijk", H.bb.ooov, T.bb, T.bb, optimize=True)
    #     <b~n~||e~f~> t(a~f~,i~n~) t(e~c~,j~k~)
    M3 += 0.5 * np.einsum("bnef,afin,ecjk->abcijk", H.bb.vovv, T.bb, T.bb, optimize=True)
    #     <nb~|fe~> t(fa~,ni~) t(e~c~,j~k~)
    M3 += 0.5 * np.einsum("nbfe,fani,ecjk->abcijk", H.ab.ovvv, T.ab, T.bb, optimize=True)
    M3 -= np.transpose(M3, (0, 2, 1, 3, 4, 5)) + np.transpose(M3, (2, 1, 0, 3, 4, 5))
    M3 -= np.transpose(M3, (1, 0, 2, 3, 4, 5))
    M3 -= np.transpose(M3, (0, 1, 2, 3, 5, 4)) + np.transpose(M3, (0, 1, 2, 5, 4, 3))
    M3 -= np.transpose(M3, (0, 1, 2, 4, 3, 5))
    return M3


def build_L3A(L, H):
    """<0|(L1 + L2) Hbar|ijkabc> (aaa)."""
    # <0|(L2 V_N)_C|ijkabc>
    #     <ei||ba> l(ec,jk)
    L3 = 0.25 * np.einsum("eiba,ecjk->abcijk", H.aa.vovv, L.aa, optimize=True)
    #     <ji||ma> l(bc,mk)
    L3 -= 0.25 * np.einsum("jima,bcmk->abcijk", H.aa.ooov, L.aa, optimize=True)
    # <0|(L1 V_N)_DC|ijkabc>
    #     l(c,k) <ij||ab>
    L3 += 0.25 * np.einsum("ck,ijab->abcijk", L.a, H.aa.oovv, optimize=True)
    # <0|(L2 F_N)_DC|ijkabc>
    #     f(kc) l(ab,ij)
    L3 += 0.25 * np.einsum("kc,abij->abcijk", H.a.ov, L.aa, optimize=True)
    L3 -= np.transpose(L3, (0, 2, 1, 3, 4, 5)) + np.transpose(L3, (2, 1, 0, 3, 4, 5))
    L3 -= np.transpose(L3, (1, 0, 2, 3, 4, 5))
    L3 -= np.transpose(L3, (0, 1, 2, 3, 5, 4)) + np.transpose(L3, (0, 1, 2, 5, 4, 3))
    L3 -= np.transpose(L3, (0, 1, 2, 4, 3, 5))
    return L3


def build_L3B(L, H):
    """<0|(L1 + L2) Hbar|ijk~abc~> (aab)."""
    # <0|(L2 V_N)_C|ijk~abc~>
    #     <ek~|bc~> l(ae,ij)
    L3 = 0.5 * np.einsum("ekbc,aeij->abcijk", H.ab.vovv, L.aa, optimize=True)
    #     <ei||ba> l(ec~,jk~)
    L3 += 0.5 * np.einsum("eiba,ecjk->abcijk", H.aa.vovv, L.ab, optimize=True)
    #     <ie~|ac~> l(be~,jk~)
    L3 += np.einsum("ieac,bejk->abcijk", H.ab.ovvv, L.ab, optimize=True)
    #     <jk~|mc~> l(ab,im)
    L3 -= 0.5 * np.einsum("jkmc,abim->abcijk", H.ab.ooov, L.aa, optimize=True)
    #     <ji||ma> l(bc~,mk~)
    L3 -= 0.5 * np.einsum("jima,bcmk->abcijk", H.aa.ooov, L.ab, optimize=True)
    #     <ik~|am~> l(bc~,jm~)
    L3 -= np.einsum("ikam,bcjm->abcijk", H.ab.oovo, L.ab, optimize=True)
    # <0|(L1 V_N)_DC|ijk~abc~>
    #     l(a,i) <jk~|bc~>
    L3 += np.einsum("ai,jkbc->abcijk", L.a, H.ab.oovv, optimize=True)
    #     l(c~,k~) <ij||ab>
    L3 += 0.25 * np.einsum("ck,ijab->abcijk", L.b, H.aa.oovv, optimize=True)
    # <0|(L2 F_N)_DC|ijk~abc~>
    #     l(bc~,jk~) f(ia)
    L3 += np.einsum("bcjk,ia->abcijk", L.ab, H.a.ov, optimize=True)
    #     l(ab,ij) f(k~c~)
    L3 += 0.25 * np.einsum("abij,kc->abcijk", L.aa, H.b.ov, optimize=True)
    L3 -= np.transpose(L3, (1, 0, 2, 3, 4, 5))
    L3 -= np.transpose(L3, (0, 1, 2, 4, 3, 5))
    return L3


def build_L3C(L, H):
    """<0|(L1 + L2) Hbar|ij~k~ab~c~> (abb)."""
    # <0|(L2 V_N)_C|ij~k~ab~c~>
    #     <ke~|cb~> l(a~e~,i~j~)
    L3 = 0.5 * np.einsum("kecb,aeij->cabkij", H.ab.ovvv, L.bb, optimize=True)
    #     <e~i~||b~a~> l(ce~,kj~)
    L3 += 0.5 * np.einsum("eiba,cekj->cabkij", H.bb.vovv, L.ab, optimize=True)
    #     <ei~|ca~> l(eb~,kj~)
    L3 += np.einsum("eica,ebkj->cabkij", H.ab.vovv, L.ab, optimize=True)
    #     <kj~|cm~> l(a~b~,i~m~)
    L3 -= 0.5 * np.einsum("kjcm,abim->cabkij", H.ab.oovo, L.bb, optimize=True)
    #     <j~i~||m~a~> l(cb~,km~)
    L3 -= 0.5 * np.einsum("jima,cbkm->cabkij", H.bb.ooov, L.ab, optimize=True)
    #     <ki~|ma~> l(cb~,mj~)
    L3 -= np.einsum("kima,cbmj->cabkij", H.ab.ooov, L.ab, optimize=True)
    # <0|(L1 V_N)_DC|ij~k~ab~c~>
    #     l(a~,i~) <kj~|cb~>
    L3 += np.einsum("ai,kjcb->cabkij", L.b, H.ab.oovv, optimize=True)
    #     l(c,k) <i~j~||a~b~>
    L3 += 0.25 * np.einsum("ck,ijab->cabkij", L.a, H.bb.oovv, optimize=True)
    # <0|(L2 F_N)_DC|ij~k~ab~c~>
    #     l(cb~,kj~) f(i~a~)
    L3 += np.einsum("cbkj,ia->cabkij", L.ab, H.b.ov, optimize=True)
    #     l(a~b~,i~j~) f(kc)
    L3 += 0.25 * np.einsum("abij,kc->cabkij", L.bb, H.a.ov, optimize=True)
    L3 -= np.transpose(L3, (0, 2, 1, 3, 4, 5))
    L3 -= np.transpose(L3, (0, 1, 2, 3, 5, 4))
    return L3


def build_L3D(L, H):
    """<0|(L1 + L2) Hbar|i~j~k~a~b~c~> (bbb)."""
    # <0|(L2 V_N)_C|i~j~k~a~b~c~>
    #     <e~i~||b~a~> l(e~c~,j~k~)
    L3 = 0.25 * np.einsum("eiba,ecjk->abcijk", H.bb.vovv, L.bb, optimize=True)
    #     <j~i~||m~a~> l(b~c~,m~k~)
    L3 -= 0.25 * np.einsum("jima,bcmk->abcijk", H.bb.ooov, L.bb, optimize=True)
    # <0|(L1 V_N)_DC|i~j~k~a~b~c~>
    #     l(c~,k~) <i~j~||a~b~>
    L3 += 0.25 * np.einsum("ck,ijab->abcijk", L.b, H.bb.oovv, optimize=True)
    # <0|(L2 F_N)_DC|i~j~k~a~b~c~>
    #     f(k~c~) l(a~b~,i~j~)
    L3 += 0.25 * np.einsum("kc,abij->abcijk", H.b.ov, L.bb, optimize=True)
    L3 -= np.transpose(L3, (0, 2, 1, 3, 4, 5)) + np.transpose(L3, (2, 1, 0, 3, 4, 5))
    L3 -= np.transpose(L3, (1, 0, 2, 3, 4, 5))
    L3 -= np.transpose(L3, (0, 1, 2, 3, 5, 4)) + np.transpose(L3, (0, 1, 2, 5, 4, 3))
    L3 -= np.transpose(L3, (0, 1, 2, 4, 3, 5))
    return L3


# ======================================================================================
# s_lambda = <mu|Hbar R3|0>
# ======================================================================================
def calc_s_lambda(R3, H, system):
    """s_lambda(mu) = sum_K <mu|Hbar|K> r_K = <mu|(Hbar R3)_C|0> for mu in SD, R3 = M/D."""
    return {"a": build_s_lambda_1A(R3, H), "b": build_s_lambda_1B(R3, H), "aa": build_s_lambda_2A(R3, H),
            "ab": build_s_lambda_2B(R3, H), "bb": build_s_lambda_2C(R3, H)}


def build_s_lambda_1A(R3, H):
    """<ia|(V_N R3)_C|0>: the (ia) singles of s_lambda (output (a,i))."""
    # <ia|(V_N R3)_C|0>
    #     <ij||ab> r(abc,ijk)  -> (c,k)
    X = 0.25 * np.einsum("ijab,abcijk->ck", H.aa.oovv, R3.aaa, optimize=True)
    #     <jk~|bc~> r(abc~,ijk~)  -> (a,i)
    X += np.einsum("jkbc,abcijk->ai", H.ab.oovv, R3.aab, optimize=True)
    #     <i~j~||a~b~> r(ca~b~,ki~j~)  -> (c,k)
    X += 0.25 * np.einsum("ijab,cabkij->ck", H.bb.oovv, R3.abb, optimize=True)
    return X


def build_s_lambda_1B(R3, H):
    """<i~a~|(V_N R3)_C|0>: the (i~a~) singles of s_lambda (output (a~,i~))."""
    # <i~a~|(V_N R3)_C|0>
    #     <ij||ab> r(abc~,ijk~)  -> (c~,k~)
    X = 0.25 * np.einsum("ijab,abcijk->ck", H.aa.oovv, R3.aab, optimize=True)
    #     <kj~|cb~> r(ca~b~,ki~j~)  -> (a~,i~)
    X += np.einsum("kjcb,cabkij->ai", H.ab.oovv, R3.abb, optimize=True)
    #     <i~j~||a~b~> r(a~b~c~,i~j~k~)  -> (c~,k~)
    X += 0.25 * np.einsum("ijab,abcijk->ck", H.bb.oovv, R3.bbb, optimize=True)
    return X


def build_s_lambda_2A(R3, H):
    """<ijab|(H_N R3)_C|0>: the (aa) doubles of s_lambda, antisymmetrized at the end."""
    # <ijab|(V_N R3)_C|0>
    #     <ei||ba> r(abc,ijk)  -> (ec,jk)
    X = 0.25 * np.einsum("eiba,abcijk->ecjk", H.aa.vovv, R3.aaa, optimize=True)
    #     <ji||ma> r(abc,ijk)  -> (bc,mk)
    X -= 0.25 * np.einsum("jima,abcijk->bcmk", H.aa.ooov, R3.aaa, optimize=True)
    # <ijab|(F_N R3)_C|0>
    #     f(kc) r(abc,ijk)  -> (ab,ij)
    X += 0.25 * np.einsum("kc,abcijk->abij", H.a.ov, R3.aaa, optimize=True)
    # <ijab|(V_N R3)_C|0>
    #     <ek~|bc~> r(abc~,ijk~)  -> (ae,ij)
    X += 0.5 * np.einsum("ekbc,abcijk->aeij", H.ab.vovv, R3.aab, optimize=True)
    #     <jk~|mc~> r(abc~,ijk~)  -> (ab,im)
    X -= 0.5 * np.einsum("jkmc,abcijk->abim", H.ab.ooov, R3.aab, optimize=True)
    # <ijab|(F_N R3)_C|0>
    #     f(k~c~) r(abc~,ijk~)  -> (ab,ij)
    X += 0.25 * np.einsum("kc,abcijk->abij", H.b.ov, R3.aab, optimize=True)
    X -= np.transpose(X, (1, 0, 2, 3))
    X -= np.transpose(X, (0, 1, 3, 2))
    return X


def build_s_lambda_2B(R3, H):
    """<ij~ab~|(H_N R3)_C|0>: the (ab) doubles of s_lambda."""
    # <ij~ab~|(V_N R3)_C|0>
    #     <ei||ba> r(abc~,ijk~)  -> (ec~,jk~)
    X = 0.5 * np.einsum("eiba,abcijk->ecjk", H.aa.vovv, R3.aab, optimize=True)
    #     <ie~|ac~> r(abc~,ijk~)  -> (be~,jk~)
    X += np.einsum("ieac,abcijk->bejk", H.ab.ovvv, R3.aab, optimize=True)
    #     <ji||ma> r(abc~,ijk~)  -> (bc~,mk~)
    X -= 0.5 * np.einsum("jima,abcijk->bcmk", H.aa.ooov, R3.aab, optimize=True)
    #     <ik~|am~> r(abc~,ijk~)  -> (bc~,jm~)
    X -= np.einsum("ikam,abcijk->bcjm", H.ab.oovo, R3.aab, optimize=True)
    # <ij~ab~|(F_N R3)_C|0>
    #     f(ia) r(abc~,ijk~)  -> (bc~,jk~)
    X += np.einsum("ia,abcijk->bcjk", H.a.ov, R3.aab, optimize=True)
    # <ij~ab~|(V_N R3)_C|0>
    #     <e~i~||b~a~> r(ca~b~,ki~j~)  -> (ce~,kj~)
    X += 0.5 * np.einsum("eiba,cabkij->cekj", H.bb.vovv, R3.abb, optimize=True)
    #     <ei~|ca~> r(ca~b~,ki~j~)  -> (eb~,kj~)
    X += np.einsum("eica,cabkij->ebkj", H.ab.vovv, R3.abb, optimize=True)
    #     <j~i~||m~a~> r(ca~b~,ki~j~)  -> (cb~,km~)
    X -= 0.5 * np.einsum("jima,cabkij->cbkm", H.bb.ooov, R3.abb, optimize=True)
    #     <ki~|ma~> r(ca~b~,ki~j~)  -> (cb~,mj~)
    X -= np.einsum("kima,cabkij->cbmj", H.ab.ooov, R3.abb, optimize=True)
    # <ij~ab~|(F_N R3)_C|0>
    #     f(i~a~) r(ca~b~,ki~j~)  -> (cb~,kj~)
    X += np.einsum("ia,cabkij->cbkj", H.b.ov, R3.abb, optimize=True)
    return X


def build_s_lambda_2C(R3, H):
    """<i~j~a~b~|(H_N R3)_C|0>: the (bb) doubles of s_lambda, antisymmetrized at the end."""
    # <i~j~a~b~|(V_N R3)_C|0>
    #     <ke~|cb~> r(ca~b~,ki~j~)  -> (a~e~,i~j~)
    X = 0.5 * np.einsum("kecb,cabkij->aeij", H.ab.ovvv, R3.abb, optimize=True)
    #     <kj~|cm~> r(ca~b~,ki~j~)  -> (a~b~,i~m~)
    X -= 0.5 * np.einsum("kjcm,cabkij->abim", H.ab.oovo, R3.abb, optimize=True)
    # <i~j~a~b~|(F_N R3)_C|0>
    #     f(kc) r(ca~b~,ki~j~)  -> (a~b~,i~j~)
    X += 0.25 * np.einsum("kc,cabkij->abij", H.a.ov, R3.abb, optimize=True)
    # <i~j~a~b~|(V_N R3)_C|0>
    #     <e~i~||b~a~> r(a~b~c~,i~j~k~)  -> (e~c~,j~k~)
    X += 0.25 * np.einsum("eiba,abcijk->ecjk", H.bb.vovv, R3.bbb, optimize=True)
    #     <j~i~||m~a~> r(a~b~c~,i~j~k~)  -> (b~c~,m~k~)
    X -= 0.25 * np.einsum("jima,abcijk->bcmk", H.bb.ooov, R3.bbb, optimize=True)
    # <i~j~a~b~|(F_N R3)_C|0>
    #     f(k~c~) r(a~b~c~,i~j~k~)  -> (a~b~,i~j~)
    X += 0.25 * np.einsum("kc,abcijk->abij", H.b.ov, R3.bbb, optimize=True)
    X -= np.transpose(X, (1, 0, 2, 3))
    X -= np.transpose(X, (0, 1, 3, 2))
    return X


# ======================================================================================
# s_t = d delta / dT
# ======================================================================================
def calc_s_t(R3, L3, T, L, H, reference):
    """Doubles: d(L3 . M)/dT2 (N does not depend on T2 in the transformed frame). Singles: the
    T1 derivative of sum_K (l_K M_K + r_K N_K), the rotation derivative of its densities."""
    system = reference.system
    h, B = get_bare_integrals(reference)
    D1, Y = build_DY(*calc_crcc23_rdm(R3, L3, T, L, system), None, H.chol, T, B, reference)
    singles = rotation_gradient(D1, Y, h, B, reference)
    return {"a": singles["a"], "b": singles["b"], "aa": build_s_t_2A(L3, T, H),
            "ab": build_s_t_2B(L3, T, H), "bb": build_s_t_2C(L3, T, H)}


def build_s_t_2A(L3, T, H):
    """d/dt(ab,ij) of sum_K l_K <K|(H(T1) e^T2)_C|0>, L3 = N/D: the (aa) doubles of s_t."""
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <am||ij> l(abc,ijk)  -> (bc,mk)
    X = -0.25 * np.einsum("amij,abcijk->bcmk", H.aa.vooo, L3.aaa, optimize=True)
    # d/dt of <0|L3 (F_N T2^2)_C|0> / 2
    #     f(me) t(bc,mk) l(abc,ijk)  -> (ae,ij)
    X -= 0.25 * np.einsum("me,bcmk,abcijk->aeij", H.a.ov, T.aa, L3.aaa, optimize=True)
    #     f(me) t(ae,ij) l(abc,ijk)  -> (bc,mk)
    X -= 0.25 * np.einsum("me,aeij,abcijk->bcmk", H.a.ov, T.aa, L3.aaa, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <am||ef> t(bc,mk) l(abc,ijk)  -> (ef,ij)
    X -= 0.125 * np.einsum("amef,bcmk,abcijk->efij", H.aa.vovv, T.aa, L3.aaa, optimize=True)
    #     <am||ef> t(ef,ij) l(abc,ijk)  -> (bc,mk)
    X -= 0.125 * np.einsum("amef,efij,abcijk->bcmk", H.aa.vovv, T.aa, L3.aaa, optimize=True)
    #     <mn||jf> t(bc,mk) l(abc,ijk)  -> (af,in)
    X -= 0.5 * np.einsum("mnjf,bcmk,abcijk->afin", H.aa.ooov, T.aa, L3.aaa, optimize=True)
    #     <mn||jf> t(af,in) l(abc,ijk)  -> (bc,mk)
    X -= 0.5 * np.einsum("mnjf,afin,abcijk->bcmk", H.aa.ooov, T.aa, L3.aaa, optimize=True)
    #     <mn~|jf~> t(af~,in~) l(abc,ijk)  -> (bc,mk)
    X -= 0.5 * np.einsum("mnjf,afin,abcijk->bcmk", H.ab.ooov, T.ab, L3.aaa, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <ab||ie> l(abc,ijk)  -> (ec,jk)
    X += 0.25 * np.einsum("abie,abcijk->ecjk", H.aa.vvov, L3.aaa, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <mn||ie> t(ec,jk) l(abc,ijk)  -> (ab,mn)
    X += 0.125 * np.einsum("mnie,ecjk,abcijk->abmn", H.aa.ooov, T.aa, L3.aaa, optimize=True)
    #     <mn||ie> t(ab,mn) l(abc,ijk)  -> (ec,jk)
    X += 0.125 * np.einsum("mnie,abmn,abcijk->ecjk", H.aa.ooov, T.aa, L3.aaa, optimize=True)
    #     <bn||ef> t(ec,jk) l(abc,ijk)  -> (af,in)
    X += 0.5 * np.einsum("bnef,ecjk,abcijk->afin", H.aa.vovv, T.aa, L3.aaa, optimize=True)
    #     <bn||ef> t(af,in) l(abc,ijk)  -> (ec,jk)
    X += 0.5 * np.einsum("bnef,afin,abcijk->ecjk", H.aa.vovv, T.aa, L3.aaa, optimize=True)
    #     <bn~|ef~> t(af~,in~) l(abc,ijk)  -> (ec,jk)
    X += 0.5 * np.einsum("bnef,afin,abcijk->ecjk", H.ab.vovv, T.ab, L3.aaa, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <bc~|ek~> l(abc~,ijk~)  -> (ae,ij)
    X += 0.5 * np.einsum("bcek,abcijk->aeij", H.ab.vvvo, L3.aab, optimize=True)
    # d/dt of <0|L3 (F_N T2^2)_C|0> / 2
    #     f(me) t(bc~,mk~) l(abc~,ijk~)  -> (ae,ij)
    X -= 0.5 * np.einsum("me,bcmk,abcijk->aeij", H.a.ov, T.ab, L3.aab, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <nm~|ek~> t(bc~,nm~) l(abc~,ijk~)  -> (ae,ij)
    X += 0.5 * np.einsum("nmek,bcnm,abcijk->aeij", H.ab.oovo, T.ab, L3.aab, optimize=True)
    #     <bm~|ef~> t(c~f~,k~m~) l(abc~,ijk~)  -> (ae,ij)
    X += 0.5 * np.einsum("bmef,cfkm,abcijk->aeij", H.ab.vovv, T.bb, L3.aab, optimize=True)
    #     <bm||ef> t(fc~,mk~) l(abc~,ijk~)  -> (ae,ij)
    X += 0.5 * np.einsum("bmef,fcmk,abcijk->aeij", H.aa.vovv, T.ab, L3.aab, optimize=True)
    #     <mc~|ef~> t(bf~,mk~) l(abc~,ijk~)  -> (ae,ij)
    X -= 0.5 * np.einsum("mcef,bfmk,abcijk->aeij", H.ab.ovvv, T.ab, L3.aab, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <mc~|jk~> l(abc~,ijk~)  -> (ab,im)
    X -= 0.5 * np.einsum("mcjk,abcijk->abim", H.ab.ovoo, L3.aab, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <mn||jf> t(fc~,nk~) l(abc~,ijk~)  -> (ab,im)
    X -= 0.5 * np.einsum("mnjf,fcnk,abcijk->abim", H.aa.ooov, T.ab, L3.aab, optimize=True)
    #     <mn~|jf~> t(f~c~,n~k~) l(abc~,ijk~)  -> (ab,im)
    X -= 0.5 * np.einsum("mnjf,fcnk,abcijk->abim", H.ab.ooov, T.bb, L3.aab, optimize=True)
    #     <mn~|fk~> t(fc~,jn~) l(abc~,ijk~)  -> (ab,im)
    X += 0.5 * np.einsum("mnfk,fcjn,abcijk->abim", H.ab.oovo, T.ab, L3.aab, optimize=True)
    #     <mc~|fe~> t(fe~,jk~) l(abc~,ijk~)  -> (ab,im)
    X -= 0.5 * np.einsum("mcfe,fejk,abcijk->abim", H.ab.ovvv, T.ab, L3.aab, optimize=True)
    #     <mc~|fe~> t(be~,jk~) l(abc~,ijk~)  -> (af,im)
    X += np.einsum("mcfe,bejk,abcijk->afim", H.ab.ovvv, T.ab, L3.aab, optimize=True)
    #     <nm~|fk~> t(bc~,jm~) l(abc~,ijk~)  -> (af,in)
    X -= np.einsum("nmfk,bcjm,abcijk->afin", H.ab.oovo, T.ab, L3.aab, optimize=True)
    # d/dt of <0|L3 (F_N T2^2)_C|0> / 2
    #     f(me) t(ec~,jk~) l(abc~,ijk~)  -> (ab,im)
    X -= 0.5 * np.einsum("me,ecjk,abcijk->abim", H.a.ov, T.ab, L3.aab, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <mn||ie> t(ec~,jk~) l(abc~,ijk~)  -> (ab,mn)
    X += 0.25 * np.einsum("mnie,ecjk,abcijk->abmn", H.aa.ooov, T.ab, L3.aab, optimize=True)
    #     <bn||ef> t(ec~,jk~) l(abc~,ijk~)  -> (af,in)
    X += np.einsum("bnef,ecjk,abcijk->afin", H.aa.vovv, T.ab, L3.aab, optimize=True)
    #     <am||ef> t(bc~,mk~) l(abc~,ijk~)  -> (ef,ij)
    X -= 0.25 * np.einsum("amef,bcmk,abcijk->efij", H.aa.vovv, T.ab, L3.aab, optimize=True)
    #     <mn||jf> t(bc~,mk~) l(abc~,ijk~)  -> (af,in)
    X -= np.einsum("mnjf,bcmk,abcijk->afin", H.aa.ooov, T.ab, L3.aab, optimize=True)
    #     <mb~|fe~> t(a~e~,i~j~) l(ca~b~,ki~j~)  -> (cf,km)
    X += 0.5 * np.einsum("mbfe,aeij,cabkij->cfkm", H.ab.ovvv, T.bb, L3.abb, optimize=True)
    #     <nm~|fj~> t(a~b~,i~m~) l(ca~b~,ki~j~)  -> (fc,nk)
    X -= 0.5 * np.einsum("nmfj,abim,cabkij->fcnk", H.ab.oovo, T.bb, L3.abb, optimize=True)
    X -= np.transpose(X, (1, 0, 2, 3))
    X -= np.transpose(X, (0, 1, 3, 2))
    return X


def build_s_t_2B(L3, T, H):
    """d/dt(ab~,ij~) of sum_K l_K <K|(H(T1) e^T2)_C|0>, L3 = N/D: the (ab) doubles of s_t."""
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <mn~|jf~> t(bc,mk) l(abc,ijk)  -> (af~,in~)
    X = -0.5 * np.einsum("mnjf,bcmk,abcijk->afin", H.ab.ooov, T.aa, L3.aaa, optimize=True)
    #     <bn~|ef~> t(ec,jk) l(abc,ijk)  -> (af~,in~)
    X += 0.5 * np.einsum("bnef,ecjk,abcijk->afin", H.ab.vovv, T.aa, L3.aaa, optimize=True)
    # d/dt of <0|L3 (F_N T2^2)_C|0> / 2
    #     f(me) t(ae,ij) l(abc~,ijk~)  -> (bc~,mk~)
    X -= 0.5 * np.einsum("me,aeij,abcijk->bcmk", H.a.ov, T.aa, L3.aab, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <nm~|ek~> t(ae,ij) l(abc~,ijk~)  -> (bc~,nm~)
    X += 0.5 * np.einsum("nmek,aeij,abcijk->bcnm", H.ab.oovo, T.aa, L3.aab, optimize=True)
    #     <bm||ef> t(ae,ij) l(abc~,ijk~)  -> (fc~,mk~)
    X += 0.5 * np.einsum("bmef,aeij,abcijk->fcmk", H.aa.vovv, T.aa, L3.aab, optimize=True)
    #     <mc~|ef~> t(ae,ij) l(abc~,ijk~)  -> (bf~,mk~)
    X -= 0.5 * np.einsum("mcef,aeij,abcijk->bfmk", H.ab.ovvv, T.aa, L3.aab, optimize=True)
    #     <mn||jf> t(ab,im) l(abc~,ijk~)  -> (fc~,nk~)
    X -= 0.5 * np.einsum("mnjf,abim,abcijk->fcnk", H.aa.ooov, T.aa, L3.aab, optimize=True)
    #     <mn~|fk~> t(ab,im) l(abc~,ijk~)  -> (fc~,jn~)
    X += 0.5 * np.einsum("mnfk,abim,abcijk->fcjn", H.ab.oovo, T.aa, L3.aab, optimize=True)
    #     <mc~|fe~> t(ab,im) l(abc~,ijk~)  -> (fe~,jk~)
    X -= 0.5 * np.einsum("mcfe,abim,abcijk->fejk", H.ab.ovvv, T.aa, L3.aab, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <ac~|ie~> l(abc~,ijk~)  -> (be~,jk~)
    X += np.einsum("acie,abcijk->bejk", H.ab.vvov, L3.aab, optimize=True)
    # d/dt of <0|L3 (F_N T2^2)_C|0> / 2
    #     f(m~e~) t(be~,jk~) l(abc~,ijk~)  -> (ac~,im~)
    X -= np.einsum("me,bejk,abcijk->acim", H.b.ov, T.ab, L3.aab, optimize=True)
    #     f(m~e~) t(ac~,im~) l(abc~,ijk~)  -> (be~,jk~)
    X -= np.einsum("me,acim,abcijk->bejk", H.b.ov, T.ab, L3.aab, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <nm~|ie~> t(be~,jk~) l(abc~,ijk~)  -> (ac~,nm~)
    X += np.einsum("nmie,bejk,abcijk->acnm", H.ab.ooov, T.ab, L3.aab, optimize=True)
    #     <nm~|ie~> t(ac~,nm~) l(abc~,ijk~)  -> (be~,jk~)
    X += np.einsum("nmie,acnm,abcijk->bejk", H.ab.ooov, T.ab, L3.aab, optimize=True)
    #     <mc~|fe~> t(af,im) l(abc~,ijk~)  -> (be~,jk~)
    X += np.einsum("mcfe,afim,abcijk->bejk", H.ab.ovvv, T.aa, L3.aab, optimize=True)
    #     <c~m~||e~f~> t(be~,jk~) l(abc~,ijk~)  -> (af~,im~)
    X += np.einsum("cmef,bejk,abcijk->afim", H.bb.vovv, T.ab, L3.aab, optimize=True)
    #     <c~m~||e~f~> t(af~,im~) l(abc~,ijk~)  -> (be~,jk~)
    X += np.einsum("cmef,afim,abcijk->bejk", H.bb.vovv, T.ab, L3.aab, optimize=True)
    #     <am~|fe~> t(be~,jk~) l(abc~,ijk~)  -> (fc~,im~)
    X -= np.einsum("amfe,bejk,abcijk->fcim", H.ab.vovv, T.ab, L3.aab, optimize=True)
    #     <am~|fe~> t(fc~,im~) l(abc~,ijk~)  -> (be~,jk~)
    X -= np.einsum("amfe,fcim,abcijk->bejk", H.ab.vovv, T.ab, L3.aab, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <am~|ik~> l(abc~,ijk~)  -> (bc~,jm~)
    X -= np.einsum("amik,abcijk->bcjm", H.ab.vooo, L3.aab, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <m~n~||k~f~> t(bc~,jm~) l(abc~,ijk~)  -> (af~,in~)
    X -= np.einsum("mnkf,bcjm,abcijk->afin", H.bb.ooov, T.ab, L3.aab, optimize=True)
    #     <m~n~||k~f~> t(af~,in~) l(abc~,ijk~)  -> (bc~,jm~)
    X -= np.einsum("mnkf,afin,abcijk->bcjm", H.bb.ooov, T.ab, L3.aab, optimize=True)
    #     <nm~|fk~> t(af,in) l(abc~,ijk~)  -> (bc~,jm~)
    X -= np.einsum("nmfk,afin,abcijk->bcjm", H.ab.oovo, T.aa, L3.aab, optimize=True)
    #     <nm~|if~> t(bc~,jm~) l(abc~,ijk~)  -> (af~,nk~)
    X += np.einsum("nmif,bcjm,abcijk->afnk", H.ab.ooov, T.ab, L3.aab, optimize=True)
    #     <nm~|if~> t(af~,nk~) l(abc~,ijk~)  -> (bc~,jm~)
    X += np.einsum("nmif,afnk,abcijk->bcjm", H.ab.ooov, T.ab, L3.aab, optimize=True)
    #     <am~|ef~> t(bc~,jm~) l(abc~,ijk~)  -> (ef~,ik~)
    X -= np.einsum("amef,bcjm,abcijk->efik", H.ab.vovv, T.ab, L3.aab, optimize=True)
    #     <am~|ef~> t(ef~,ik~) l(abc~,ijk~)  -> (bc~,jm~)
    X -= np.einsum("amef,efik,abcijk->bcjm", H.ab.vovv, T.ab, L3.aab, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <ab||ie> l(abc~,ijk~)  -> (ec~,jk~)
    X += 0.5 * np.einsum("abie,abcijk->ecjk", H.aa.vvov, L3.aab, optimize=True)
    # d/dt of <0|L3 (F_N T2^2)_C|0> / 2
    #     f(me) t(ab,im) l(abc~,ijk~)  -> (ec~,jk~)
    X -= 0.5 * np.einsum("me,abim,abcijk->ecjk", H.a.ov, T.aa, L3.aab, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <mn||ie> t(ab,mn) l(abc~,ijk~)  -> (ec~,jk~)
    X += 0.25 * np.einsum("mnie,abmn,abcijk->ecjk", H.aa.ooov, T.aa, L3.aab, optimize=True)
    #     <bn||ef> t(af,in) l(abc~,ijk~)  -> (ec~,jk~)
    X += np.einsum("bnef,afin,abcijk->ecjk", H.aa.vovv, T.aa, L3.aab, optimize=True)
    #     <bn~|ef~> t(ec~,jk~) l(abc~,ijk~)  -> (af~,in~)
    X += np.einsum("bnef,ecjk,abcijk->afin", H.ab.vovv, T.ab, L3.aab, optimize=True)
    #     <bn~|ef~> t(af~,in~) l(abc~,ijk~)  -> (ec~,jk~)
    X += np.einsum("bnef,afin,abcijk->ecjk", H.ab.vovv, T.ab, L3.aab, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <am||ij> l(abc~,ijk~)  -> (bc~,mk~)
    X -= 0.5 * np.einsum("amij,abcijk->bcmk", H.aa.vooo, L3.aab, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <am||ef> t(ef,ij) l(abc~,ijk~)  -> (bc~,mk~)
    X -= 0.25 * np.einsum("amef,efij,abcijk->bcmk", H.aa.vovv, T.aa, L3.aab, optimize=True)
    #     <mn||jf> t(af,in) l(abc~,ijk~)  -> (bc~,mk~)
    X -= np.einsum("mnjf,afin,abcijk->bcmk", H.aa.ooov, T.aa, L3.aab, optimize=True)
    #     <mn~|jf~> t(bc~,mk~) l(abc~,ijk~)  -> (af~,in~)
    X -= np.einsum("mnjf,bcmk,abcijk->afin", H.ab.ooov, T.ab, L3.aab, optimize=True)
    #     <mn~|jf~> t(af~,in~) l(abc~,ijk~)  -> (bc~,mk~)
    X -= np.einsum("mnjf,afin,abcijk->bcmk", H.ab.ooov, T.ab, L3.aab, optimize=True)
    # d/dt of <0|L3 (F_N T2^2)_C|0> / 2
    #     f(m~e~) t(a~e~,i~j~) l(ca~b~,ki~j~)  -> (cb~,km~)
    X -= 0.5 * np.einsum("me,aeij,cabkij->cbkm", H.b.ov, T.bb, L3.abb, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <mn~|ke~> t(a~e~,i~j~) l(ca~b~,ki~j~)  -> (cb~,mn~)
    X += 0.5 * np.einsum("mnke,aeij,cabkij->cbmn", H.ab.ooov, T.bb, L3.abb, optimize=True)
    #     <b~m~||e~f~> t(a~e~,i~j~) l(ca~b~,ki~j~)  -> (cf~,km~)
    X += 0.5 * np.einsum("bmef,aeij,cabkij->cfkm", H.bb.vovv, T.bb, L3.abb, optimize=True)
    #     <cm~|fe~> t(a~e~,i~j~) l(ca~b~,ki~j~)  -> (fb~,km~)
    X -= 0.5 * np.einsum("cmfe,aeij,cabkij->fbkm", H.ab.vovv, T.bb, L3.abb, optimize=True)
    #     <m~n~||j~f~> t(a~b~,i~m~) l(ca~b~,ki~j~)  -> (cf~,kn~)
    X -= 0.5 * np.einsum("mnjf,abim,cabkij->cfkn", H.bb.ooov, T.bb, L3.abb, optimize=True)
    #     <nm~|kf~> t(a~b~,i~m~) l(ca~b~,ki~j~)  -> (cf~,nj~)
    X += 0.5 * np.einsum("nmkf,abim,cabkij->cfnj", H.ab.ooov, T.bb, L3.abb, optimize=True)
    #     <cm~|ef~> t(a~b~,i~m~) l(ca~b~,ki~j~)  -> (ef~,kj~)
    X -= 0.5 * np.einsum("cmef,abim,cabkij->efkj", H.ab.vovv, T.bb, L3.abb, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <ca~|ei~> l(ca~b~,ki~j~)  -> (eb~,kj~)
    X += np.einsum("caei,cabkij->ebkj", H.ab.vvvo, L3.abb, optimize=True)
    # d/dt of <0|L3 (F_N T2^2)_C|0> / 2
    #     f(me) t(eb~,kj~) l(ca~b~,ki~j~)  -> (ca~,mi~)
    X -= np.einsum("me,ebkj,cabkij->cami", H.a.ov, T.ab, L3.abb, optimize=True)
    #     f(me) t(ca~,mi~) l(ca~b~,ki~j~)  -> (eb~,kj~)
    X -= np.einsum("me,cami,cabkij->ebkj", H.a.ov, T.ab, L3.abb, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <mn~|ei~> t(eb~,kj~) l(ca~b~,ki~j~)  -> (ca~,mn~)
    X += np.einsum("mnei,ebkj,cabkij->camn", H.ab.oovo, T.ab, L3.abb, optimize=True)
    #     <mn~|ei~> t(ca~,mn~) l(ca~b~,ki~j~)  -> (eb~,kj~)
    X += np.einsum("mnei,camn,cabkij->ebkj", H.ab.oovo, T.ab, L3.abb, optimize=True)
    #     <cm~|ef~> t(a~f~,i~m~) l(ca~b~,ki~j~)  -> (eb~,kj~)
    X += np.einsum("cmef,afim,cabkij->ebkj", H.ab.vovv, T.bb, L3.abb, optimize=True)
    #     <cm||ef> t(eb~,kj~) l(ca~b~,ki~j~)  -> (fa~,mi~)
    X += np.einsum("cmef,ebkj,cabkij->fami", H.aa.vovv, T.ab, L3.abb, optimize=True)
    #     <cm||ef> t(fa~,mi~) l(ca~b~,ki~j~)  -> (eb~,kj~)
    X += np.einsum("cmef,fami,cabkij->ebkj", H.aa.vovv, T.ab, L3.abb, optimize=True)
    #     <ma~|ef~> t(eb~,kj~) l(ca~b~,ki~j~)  -> (cf~,mi~)
    X -= np.einsum("maef,ebkj,cabkij->cfmi", H.ab.ovvv, T.ab, L3.abb, optimize=True)
    #     <ma~|ef~> t(cf~,mi~) l(ca~b~,ki~j~)  -> (eb~,kj~)
    X -= np.einsum("maef,cfmi,cabkij->ebkj", H.ab.ovvv, T.ab, L3.abb, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <ma~|ki~> l(ca~b~,ki~j~)  -> (cb~,mj~)
    X -= np.einsum("maki,cabkij->cbmj", H.ab.ovoo, L3.abb, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <mn||kf> t(cb~,mj~) l(ca~b~,ki~j~)  -> (fa~,ni~)
    X -= np.einsum("mnkf,cbmj,cabkij->fani", H.aa.ooov, T.ab, L3.abb, optimize=True)
    #     <mn||kf> t(fa~,ni~) l(ca~b~,ki~j~)  -> (cb~,mj~)
    X -= np.einsum("mnkf,fani,cabkij->cbmj", H.aa.ooov, T.ab, L3.abb, optimize=True)
    #     <mn~|kf~> t(a~f~,i~n~) l(ca~b~,ki~j~)  -> (cb~,mj~)
    X -= np.einsum("mnkf,afin,cabkij->cbmj", H.ab.ooov, T.bb, L3.abb, optimize=True)
    #     <mn~|fi~> t(cb~,mj~) l(ca~b~,ki~j~)  -> (fa~,kn~)
    X += np.einsum("mnfi,cbmj,cabkij->fakn", H.ab.oovo, T.ab, L3.abb, optimize=True)
    #     <mn~|fi~> t(fa~,kn~) l(ca~b~,ki~j~)  -> (cb~,mj~)
    X += np.einsum("mnfi,fakn,cabkij->cbmj", H.ab.oovo, T.ab, L3.abb, optimize=True)
    #     <ma~|fe~> t(cb~,mj~) l(ca~b~,ki~j~)  -> (fe~,ki~)
    X -= np.einsum("mafe,cbmj,cabkij->feki", H.ab.ovvv, T.ab, L3.abb, optimize=True)
    #     <ma~|fe~> t(fe~,ki~) l(ca~b~,ki~j~)  -> (cb~,mj~)
    X -= np.einsum("mafe,feki,cabkij->cbmj", H.ab.ovvv, T.ab, L3.abb, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <a~b~||i~e~> l(ca~b~,ki~j~)  -> (ce~,kj~)
    X += 0.5 * np.einsum("abie,cabkij->cekj", H.bb.vvov, L3.abb, optimize=True)
    # d/dt of <0|L3 (F_N T2^2)_C|0> / 2
    #     f(m~e~) t(a~b~,i~m~) l(ca~b~,ki~j~)  -> (ce~,kj~)
    X -= 0.5 * np.einsum("me,abim,cabkij->cekj", H.b.ov, T.bb, L3.abb, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <m~n~||i~e~> t(a~b~,m~n~) l(ca~b~,ki~j~)  -> (ce~,kj~)
    X += 0.25 * np.einsum("mnie,abmn,cabkij->cekj", H.bb.ooov, T.bb, L3.abb, optimize=True)
    #     <b~n~||e~f~> t(a~f~,i~n~) l(ca~b~,ki~j~)  -> (ce~,kj~)
    X += np.einsum("bnef,afin,cabkij->cekj", H.bb.vovv, T.bb, L3.abb, optimize=True)
    #     <nb~|fe~> t(ce~,kj~) l(ca~b~,ki~j~)  -> (fa~,ni~)
    X += np.einsum("nbfe,cekj,cabkij->fani", H.ab.ovvv, T.ab, L3.abb, optimize=True)
    #     <nb~|fe~> t(fa~,ni~) l(ca~b~,ki~j~)  -> (ce~,kj~)
    X += np.einsum("nbfe,fani,cabkij->cekj", H.ab.ovvv, T.ab, L3.abb, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <a~m~||i~j~> l(ca~b~,ki~j~)  -> (cb~,km~)
    X -= 0.5 * np.einsum("amij,cabkij->cbkm", H.bb.vooo, L3.abb, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <a~m~||e~f~> t(e~f~,i~j~) l(ca~b~,ki~j~)  -> (cb~,km~)
    X -= 0.25 * np.einsum("amef,efij,cabkij->cbkm", H.bb.vovv, T.bb, L3.abb, optimize=True)
    #     <m~n~||j~f~> t(a~f~,i~n~) l(ca~b~,ki~j~)  -> (cb~,km~)
    X -= np.einsum("mnjf,afin,cabkij->cbkm", H.bb.ooov, T.bb, L3.abb, optimize=True)
    #     <nm~|fj~> t(cb~,km~) l(ca~b~,ki~j~)  -> (fa~,ni~)
    X -= np.einsum("nmfj,cbkm,cabkij->fani", H.ab.oovo, T.ab, L3.abb, optimize=True)
    #     <nm~|fj~> t(fa~,ni~) l(ca~b~,ki~j~)  -> (cb~,km~)
    X -= np.einsum("nmfj,fani,cabkij->cbkm", H.ab.oovo, T.ab, L3.abb, optimize=True)
    #     <nm~|fj~> t(b~c~,m~k~) l(a~b~c~,i~j~k~)  -> (fa~,ni~)
    X -= 0.5 * np.einsum("nmfj,bcmk,abcijk->fani", H.ab.oovo, T.bb, L3.bbb, optimize=True)
    #     <nb~|fe~> t(e~c~,j~k~) l(a~b~c~,i~j~k~)  -> (fa~,ni~)
    X += 0.5 * np.einsum("nbfe,ecjk,abcijk->fani", H.ab.ovvv, T.bb, L3.bbb, optimize=True)
    return X


def build_s_t_2C(L3, T, H):
    """d/dt(a~b~,i~j~) of sum_K l_K <K|(H(T1) e^T2)_C|0>, L3 = N/D: the (bb) doubles of s_t."""
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <bm~|ef~> t(ae,ij) l(abc~,ijk~)  -> (c~f~,k~m~)
    X = 0.5 * np.einsum("bmef,aeij,abcijk->cfkm", H.ab.vovv, T.aa, L3.aab, optimize=True)
    #     <mn~|jf~> t(ab,im) l(abc~,ijk~)  -> (f~c~,n~k~)
    X -= 0.5 * np.einsum("mnjf,abim,abcijk->fcnk", H.ab.ooov, T.aa, L3.aab, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <cb~|ke~> l(ca~b~,ki~j~)  -> (a~e~,i~j~)
    X += 0.5 * np.einsum("cbke,cabkij->aeij", H.ab.vvov, L3.abb, optimize=True)
    # d/dt of <0|L3 (F_N T2^2)_C|0> / 2
    #     f(m~e~) t(cb~,km~) l(ca~b~,ki~j~)  -> (a~e~,i~j~)
    X -= 0.5 * np.einsum("me,cbkm,cabkij->aeij", H.b.ov, T.ab, L3.abb, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <mn~|ke~> t(cb~,mn~) l(ca~b~,ki~j~)  -> (a~e~,i~j~)
    X += 0.5 * np.einsum("mnke,cbmn,cabkij->aeij", H.ab.ooov, T.ab, L3.abb, optimize=True)
    #     <mb~|fe~> t(cf,km) l(ca~b~,ki~j~)  -> (a~e~,i~j~)
    X += 0.5 * np.einsum("mbfe,cfkm,cabkij->aeij", H.ab.ovvv, T.aa, L3.abb, optimize=True)
    #     <b~m~||e~f~> t(cf~,km~) l(ca~b~,ki~j~)  -> (a~e~,i~j~)
    X += 0.5 * np.einsum("bmef,cfkm,cabkij->aeij", H.bb.vovv, T.ab, L3.abb, optimize=True)
    #     <cm~|fe~> t(fb~,km~) l(ca~b~,ki~j~)  -> (a~e~,i~j~)
    X -= 0.5 * np.einsum("cmfe,fbkm,cabkij->aeij", H.ab.vovv, T.ab, L3.abb, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <cm~|kj~> l(ca~b~,ki~j~)  -> (a~b~,i~m~)
    X -= 0.5 * np.einsum("cmkj,cabkij->abim", H.ab.vooo, L3.abb, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <m~n~||j~f~> t(cf~,kn~) l(ca~b~,ki~j~)  -> (a~b~,i~m~)
    X -= 0.5 * np.einsum("mnjf,cfkn,cabkij->abim", H.bb.ooov, T.ab, L3.abb, optimize=True)
    #     <nm~|fj~> t(fc,nk) l(ca~b~,ki~j~)  -> (a~b~,i~m~)
    X -= 0.5 * np.einsum("nmfj,fcnk,cabkij->abim", H.ab.oovo, T.aa, L3.abb, optimize=True)
    #     <nm~|kf~> t(cf~,nj~) l(ca~b~,ki~j~)  -> (a~b~,i~m~)
    X += 0.5 * np.einsum("nmkf,cfnj,cabkij->abim", H.ab.ooov, T.ab, L3.abb, optimize=True)
    #     <cm~|ef~> t(ef~,kj~) l(ca~b~,ki~j~)  -> (a~b~,i~m~)
    X -= 0.5 * np.einsum("cmef,efkj,cabkij->abim", H.ab.vovv, T.ab, L3.abb, optimize=True)
    #     <cm~|ef~> t(eb~,kj~) l(ca~b~,ki~j~)  -> (a~f~,i~m~)
    X += np.einsum("cmef,ebkj,cabkij->afim", H.ab.vovv, T.ab, L3.abb, optimize=True)
    #     <mn~|kf~> t(cb~,mj~) l(ca~b~,ki~j~)  -> (a~f~,i~n~)
    X -= np.einsum("mnkf,cbmj,cabkij->afin", H.ab.ooov, T.ab, L3.abb, optimize=True)
    # d/dt of <0|L3 (F_N T2^2)_C|0> / 2
    #     f(m~e~) t(ce~,kj~) l(ca~b~,ki~j~)  -> (a~b~,i~m~)
    X -= 0.5 * np.einsum("me,cekj,cabkij->abim", H.b.ov, T.ab, L3.abb, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <m~n~||i~e~> t(ce~,kj~) l(ca~b~,ki~j~)  -> (a~b~,m~n~)
    X += 0.25 * np.einsum("mnie,cekj,cabkij->abmn", H.bb.ooov, T.ab, L3.abb, optimize=True)
    #     <b~n~||e~f~> t(ce~,kj~) l(ca~b~,ki~j~)  -> (a~f~,i~n~)
    X += np.einsum("bnef,cekj,cabkij->afin", H.bb.vovv, T.ab, L3.abb, optimize=True)
    #     <a~m~||e~f~> t(cb~,km~) l(ca~b~,ki~j~)  -> (e~f~,i~j~)
    X -= 0.25 * np.einsum("amef,cbkm,cabkij->efij", H.bb.vovv, T.ab, L3.abb, optimize=True)
    #     <m~n~||j~f~> t(cb~,km~) l(ca~b~,ki~j~)  -> (a~f~,i~n~)
    X -= np.einsum("mnjf,cbkm,cabkij->afin", H.bb.ooov, T.ab, L3.abb, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <a~m~||i~j~> l(a~b~c~,i~j~k~)  -> (b~c~,m~k~)
    X -= 0.25 * np.einsum("amij,abcijk->bcmk", H.bb.vooo, L3.bbb, optimize=True)
    # d/dt of <0|L3 (F_N T2^2)_C|0> / 2
    #     f(m~e~) t(b~c~,m~k~) l(a~b~c~,i~j~k~)  -> (a~e~,i~j~)
    X -= 0.25 * np.einsum("me,bcmk,abcijk->aeij", H.b.ov, T.bb, L3.bbb, optimize=True)
    #     f(m~e~) t(a~e~,i~j~) l(a~b~c~,i~j~k~)  -> (b~c~,m~k~)
    X -= 0.25 * np.einsum("me,aeij,abcijk->bcmk", H.b.ov, T.bb, L3.bbb, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <a~m~||e~f~> t(b~c~,m~k~) l(a~b~c~,i~j~k~)  -> (e~f~,i~j~)
    X -= 0.125 * np.einsum("amef,bcmk,abcijk->efij", H.bb.vovv, T.bb, L3.bbb, optimize=True)
    #     <a~m~||e~f~> t(e~f~,i~j~) l(a~b~c~,i~j~k~)  -> (b~c~,m~k~)
    X -= 0.125 * np.einsum("amef,efij,abcijk->bcmk", H.bb.vovv, T.bb, L3.bbb, optimize=True)
    #     <m~n~||j~f~> t(b~c~,m~k~) l(a~b~c~,i~j~k~)  -> (a~f~,i~n~)
    X -= 0.5 * np.einsum("mnjf,bcmk,abcijk->afin", H.bb.ooov, T.bb, L3.bbb, optimize=True)
    #     <m~n~||j~f~> t(a~f~,i~n~) l(a~b~c~,i~j~k~)  -> (b~c~,m~k~)
    X -= 0.5 * np.einsum("mnjf,afin,abcijk->bcmk", H.bb.ooov, T.bb, L3.bbb, optimize=True)
    #     <nm~|fj~> t(fa~,ni~) l(a~b~c~,i~j~k~)  -> (b~c~,m~k~)
    X -= 0.5 * np.einsum("nmfj,fani,abcijk->bcmk", H.ab.oovo, T.ab, L3.bbb, optimize=True)
    # d/dt of <0|L3 (V_N T2)_C|0>
    #     <a~b~||i~e~> l(a~b~c~,i~j~k~)  -> (e~c~,j~k~)
    X += 0.25 * np.einsum("abie,abcijk->ecjk", H.bb.vvov, L3.bbb, optimize=True)
    # d/dt of <0|L3 (V_N T2^2)_C|0> / 2
    #     <m~n~||i~e~> t(e~c~,j~k~) l(a~b~c~,i~j~k~)  -> (a~b~,m~n~)
    X += 0.125 * np.einsum("mnie,ecjk,abcijk->abmn", H.bb.ooov, T.bb, L3.bbb, optimize=True)
    #     <m~n~||i~e~> t(a~b~,m~n~) l(a~b~c~,i~j~k~)  -> (e~c~,j~k~)
    X += 0.125 * np.einsum("mnie,abmn,abcijk->ecjk", H.bb.ooov, T.bb, L3.bbb, optimize=True)
    #     <b~n~||e~f~> t(e~c~,j~k~) l(a~b~c~,i~j~k~)  -> (a~f~,i~n~)
    X += 0.5 * np.einsum("bnef,ecjk,abcijk->afin", H.bb.vovv, T.bb, L3.bbb, optimize=True)
    #     <b~n~||e~f~> t(a~f~,i~n~) l(a~b~c~,i~j~k~)  -> (e~c~,j~k~)
    X += 0.5 * np.einsum("bnef,afin,abcijk->ecjk", H.bb.vovv, T.bb, L3.bbb, optimize=True)
    #     <nb~|fe~> t(fa~,ni~) l(a~b~c~,i~j~k~)  -> (e~c~,j~k~)
    X += 0.5 * np.einsum("nbfe,fani,abcijk->ecjk", H.ab.ovvv, T.ab, L3.bbb, optimize=True)
    X -= np.transpose(X, (1, 0, 2, 3))
    X -= np.transpose(X, (0, 1, 3, 2))
    return X


# ======================================================================================
# B^T w = d/de g(T + e w), g the left-CCSD residual
# ======================================================================================
def calc_BTw(w, T, L, H, reference):
    """Doubles: the T2 part of w through the second derivative of the CCSD Lagrangian, the T1
    part through H1 = [H, W1]. Singles: rotation derivatives of the directional densities and
    the double rotation D_L . [[H, W1], a+ i]."""
    system, nfz = reference.system, reference.nfrozen
    W = SimpleNamespace(**{k: _get(w, k) for k in ("a", "b", "aa", "ab", "bb")})
    h, B = get_bare_integrals(reference)
    H1 = build_hbar_ccs_chol_derivative(W, H, system)
    h1 = commutator(h, W.a, W.b, nfz)
    dB = commutator(B, W.a, W.b, nfz)

    rdm1, rdm2 = calc_ccsd_rdm(T, L, system)
    vvvv = lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s)
    D1L, YL = build_DY(rdm1, rdm2, vvvv, H.chol, T, B, reference, ref_weight=True)
    _, YLdB = build_DY(rdm1, rdm2, vvvv, H1.chol, T, dB, reference, ref_weight=True)
    D1w, Yw = build_DY(*calc_ccsd_rdm_derivative(T, W, L, system), lambda a, s: calc_rdm2_vvvv_slice(L, W, a, s),
                       H.chol, T, B, reference)
    r1 = rotation_gradient(D1w, Yw, h, B, reference)
    r2 = rotation_gradient(D1L, YL, h1, dB, reference)
    r3 = rotation_gradient(zero_like(D1L), YLdB, h, B, reference)

    return {"a": r1["a"] + r2["a"] + r3["a"], "b": r1["b"] + r2["b"] + r3["b"],
            "aa": build_BTw_2A(W, T, L, H, H1), "ab": build_BTw_2B(W, T, L, H, H1),
            "bb": build_BTw_2C(W, T, L, H, H1)}


def build_BTw_2A(W, T, L, H, H1):
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
        LB = np.einsum("abij,xae->xebij", L.aa, B1, optimize=True)
        X += 0.125 * np.einsum("xebij,xbf->efij", LB, B2, optimize=True)
        X -= 0.125 * np.einsum("xebij,xbf->feij", LB, B2, optimize=True)
    X -= np.transpose(X, (1, 0, 2, 3))
    X -= np.transpose(X, (0, 1, 3, 2))
    return X


def build_BTw_2B(W, T, L, H, H1):
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
        LB = np.einsum("abij,xae->xebij", L.ab, B1, optimize=True)
        X += np.einsum("xebij,xbf->efij", LB, B2, optimize=True)
    return X


def build_BTw_2C(W, T, L, H, H1):
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
        LB = np.einsum("abij,xae->xebij", L.bb, B1, optimize=True)
        X += 0.125 * np.einsum("xebij,xbf->efij", LB, B2, optimize=True)
        X -= 0.125 * np.einsum("xebij,xbf->feij", LB, B2, optimize=True)
    X -= np.transpose(X, (1, 0, 2, 3))
    X -= np.transpose(X, (0, 1, 3, 2))
    return X


# ======================================================================================
# the two CCSD-sized linear equations (ccpy's EOMCCSD and left-CCSD Jacobians on Hbar)
# ======================================================================================
def solve_w(s_lambda, T, Hbar, system, tol=1.0e-10):
    """A w = -s_lambda, A the EOMCCSD right Jacobian (HR without the energy)."""
    if system.cholesky:
        from ccpy.eomcc.eomccsd_chol import HR
    else:
        from ccpy.eomcc.eomccsd import HR

    def action(R, dR):
        HR(dR, R, T, Hbar, False, system)

    return _solve(action, {k: -v for k, v in s_lambda.items()}, T, Hbar, tol)


def solve_z(rhs, T, L, Hbar, system, tol=1.0e-10):
    """A^T z = rhs, A^T the left-CCSD Jacobian (LH without the Hbar(ov), Hbar(oovv) terms)."""
    if system.cholesky:
        from ccpy.left.left_ccsd_chol import LH_fun
    else:
        from ccpy.left.left_ccsd import LH_fun

    def action(Z, LZ):
        LH_fun(LZ, Z, T, Hbar, False, system)

    return _solve(action, rhs, L, Hbar, tol)


def _solve(action, rhs, template, Hbar, tol):
    keys = ("a", "b", "aa", "ab", "bb")
    shapes = [_get(template, k).shape for k in keys]
    sizes = [int(np.prod(s)) for s in shapes]
    op = copy.deepcopy(template)
    out = copy.deepcopy(template)

    def unpack(x, target):
        k0 = 0
        for k, shp, n in zip(keys, shapes, sizes):
            setattr(target, k, np.asfortranarray(x[k0:k0 + n].reshape(shp)))
            k0 += n
        return target

    def matvec(x):
        action(unpack(x, op), out)
        return np.concatenate([np.ravel(getattr(out, k)) for k in keys])

    # preconditioner: Hbar(oo, vv) diagonal differences
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
        raise RuntimeError(f"The CR-CC(2,3) response equations did not converge (gmres info = {info}).")
    k0, sol = 0, {}
    for k, shp, nk in zip(keys, shapes, sizes):
        sol[k] = x[k0:k0 + nk].reshape(shp)
        k0 += nk
    return sol
