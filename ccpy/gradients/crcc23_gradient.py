"""Analytic nuclear gradient of CR-CC(2,3)_A, factorized (density fitting or Cholesky).

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
    H(T1); ccsd_gradient.build_DY takes them to the bare frame.
w and z are solved with ccpy's EOMCCSD (HR) and left-CCSD (LH) Jacobian actions on Hbar.
No quantity with four virtual indices is stored; the triples are held as o^3 v^3 blocks.
"""

import copy
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse.linalg as spla

from ccpy.density.ccsd_rdm import calc_ccsd_rdm, calc_ccsd_rdm_derivative, calc_rdm2_vvvv_slice
from ccpy.density.crcc23_rdm import calc_crcc23_rdm, calc_crcc23_rdm1_denominator
from ccpy.gradients.ccsd_gradient import build_DY, get_mo_integrals, relaxed_gradient, rotation_gradient
from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol, build_hbar_ccs_chol_derivative
from ccpy.models.integrals import Integral


def crcc23A_gradient(driver, meanfield=None, zvector_tol=1.0e-10, verbose=True):
    """Analytic RHF CR-CC(2,3)_A nuclear gradient, shape (natoms, 3), Eh/bohr.

    The driver must come from Driver.from_scf (RHF; density fitting or Cholesky) after
    run_cc, run_hbar and run_leftcc (ccsd or ccsd_chol); driver.hamiltonian is the Hbar."""
    from ccpy.interfaces.scf_tools import load_scf_integrals

    t0 = time.perf_counter()
    mf = meanfield if meanfield is not None else driver.scf
    T, L, Hbar, system = driver.T, driver.L[0], driver.hamiltonian, driver.system

    _, H0 = load_scf_integrals(mf, system.nfrozen, system.ndelete, use_cholesky=True)
    H = build_hbar_ccs_chol(T, H0, system)
    mo = get_mo_integrals(mf, system)
    response = calc_crcc23A_response(T, L, Hbar, H, H0, mo, system, tol=zvector_tol)
    D1, Y = calc_crcc23A_densities(T, L, response, H, mo, system)
    return relaxed_gradient(mf, system.nfrozen, D1["a"] + D1["b"], Y["a"] + Y["b"],
                            zvector_tol=zvector_tol, verbose=verbose,
                            title="CR-CC(2,3)_A Analytic Gradient", t0=t0)


def calc_crcc23A_response(T, L, Hbar, H, H0, mo, system, tol=1.0e-10):
    """R3, L3, s_lambda, s_t, w, B^T w and z of CR-CC(2,3)_A (H: build_hbar_ccs_chol, H0: bare)."""
    R3, L3 = calc_r3_l3(T, L, H, H0, system)
    s_lambda = calc_s_lambda(R3, H, system)
    s_t = calc_s_t(R3, L3, T, L, H, mo, system)
    w = solve_w(s_lambda, T, Hbar, system, tol)
    BTw = calc_BTw(w, T, L, H, mo, system)
    z = solve_z({k: -(s_t[k] + BTw[k]) for k in s_t}, T, L, Hbar, system, tol)
    return {"R3": R3, "L3": L3, "s_lambda": s_lambda, "s_t": s_t, "w": w, "BTw": BTw, "z": z}


def calc_crcc23A_densities(T, L, response, H, mo, system):
    """(D1, Y) of E_CCSD + delta_A, relaxed in the amplitudes, conjugate to the bare (h, B)."""
    R3, L3, w = response["R3"], response["L3"], response["w"]
    z = SimpleNamespace(**{k: _get(response["z"], k) for k in ("a", "b", "aa", "ab", "bb")})
    pieces = [
        build_DY(*calc_ccsd_rdm(T, L, system), lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s),
                 H.chol, T, mo["B"], mo, system, reference=True),
        build_DY(*calc_crcc23_rdm(R3, L3, T, L, system), None, H.chol, T, mo["B"], mo, system),
        build_DY(*calc_ccsd_rdm(T, z, system, energy=False), lambda a, s: calc_rdm2_vvvv_slice(z, T, a, s),
                 H.chol, T, mo["B"], mo, system),
        calc_w_density(w, T, L, H, mo, system),
        calc_denominator_density(R3, L3, H, mo, system),
    ]
    return ({s: sum(p[0][s] for p in pieces) for s in ("a", "b")},
            {s: sum(p[1][s] for p in pieces) for s in ("a", "b")})


def calc_denominator_density(R3, L3, H, mo, system):
    """(D1, Y) of -sum_K l_K r_K D_K (Eq. 9); the denominators are bare Fock diagonals, so
    there is no T1 to undo (and no two-body density: H.chol only sets the shapes)."""
    return build_DY(calc_crcc23_rdm1_denominator(R3, L3, system), Integral.from_empty(system, 2, use_none=True),
                    None, H.chol, None, mo["B"], mo, system)


def calc_w_density(w, T, L, H, mo, system):
    """(D1, Y) of d L(T + e w)/de at fixed Lambda: the T1 part of w rotates the Lagrangian
    densities, the T2 part comes from calc_ccsd_rdm_derivative."""
    W = SimpleNamespace(**{k: _get(w, k) for k in ("a", "b", "aa", "ab", "bb")})
    W1 = {s: _embed_vo(_get(w, s), mo, s) for s in ("a", "b")}
    H1 = build_hbar_ccs_chol_derivative(W, H, system)
    dB = {s: mo["B"][s] @ W1[s] - W1[s] @ mo["B"][s] for s in ("a", "b")}
    rdm1, rdm2 = calc_ccsd_rdm(T, L, system)
    vvvv = lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s)
    D1L, YL = build_DY(rdm1, rdm2, vvvv, H.chol, T, mo["B"], mo, system, reference=True)
    _, YLdB = build_DY(rdm1, rdm2, vvvv, H1.chol, T, dB, mo, system, reference=True)
    D1w, Yw = build_DY(*calc_ccsd_rdm_derivative(T, W, L, system), lambda a, s: calc_rdm2_vvvv_slice(L, W, a, s),
                       H.chol, T, mo["B"], mo, system)
    D1 = {s: D1w[s] + D1L[s] @ W1[s].T - W1[s].T @ D1L[s] for s in ("a", "b")}
    Y = {s: Yw[s] + YL[s] @ W1[s].T - W1[s].T @ YL[s] + YLdB[s] for s in ("a", "b")}
    return D1, Y


def _embed_vo(x, mo, s):
    W = np.zeros((mo["nmo"], mo["nmo"]))
    W[mo["slices"][s]["v"], mo["slices"][s]["o"]] = x
    return W


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
    M3 = -0.25 * np.einsum("amij,bcmk->abcijk", H.aa.vooo, T.aa, optimize=True)
    M3 -= 0.25 * np.einsum("me,aeij,bcmk->abcijk", H.a.ov, T.aa, T.aa, optimize=True)
    M3 -= 0.125 * np.einsum("amef,efij,bcmk->abcijk", H.aa.vovv, T.aa, T.aa, optimize=True)
    M3 -= 0.5 * np.einsum("mnjf,afin,bcmk->abcijk", H.aa.ooov, T.aa, T.aa, optimize=True)
    M3 -= 0.5 * np.einsum("mnjf,afin,bcmk->abcijk", H.ab.ooov, T.ab, T.aa, optimize=True)
    M3 += 0.25 * np.einsum("abie,ecjk->abcijk", H.aa.vvov, T.aa, optimize=True)
    M3 += 0.125 * np.einsum("mnie,abmn,ecjk->abcijk", H.aa.ooov, T.aa, T.aa, optimize=True)
    M3 += 0.5 * np.einsum("bnef,afin,ecjk->abcijk", H.aa.vovv, T.aa, T.aa, optimize=True)
    M3 += 0.5 * np.einsum("bnef,afin,ecjk->abcijk", H.ab.vovv, T.ab, T.aa, optimize=True)
    M3 -= np.transpose(M3, (0, 2, 1, 3, 4, 5)) + np.transpose(M3, (2, 1, 0, 3, 4, 5))
    M3 -= np.transpose(M3, (1, 0, 2, 3, 4, 5))
    M3 -= np.transpose(M3, (0, 1, 2, 3, 5, 4)) + np.transpose(M3, (0, 1, 2, 5, 4, 3))
    M3 -= np.transpose(M3, (0, 1, 2, 4, 3, 5))
    return M3


def build_M3B(T, H):
    """<ijk~abc~|Hbar|0> (aab)."""
    M3 = 0.5 * np.einsum("bcek,aeij->abcijk", H.ab.vvvo, T.aa, optimize=True)
    M3 -= 0.5 * np.einsum("me,bcmk,aeij->abcijk", H.a.ov, T.ab, T.aa, optimize=True)
    M3 += 0.5 * np.einsum("nmek,bcnm,aeij->abcijk", H.ab.oovo, T.ab, T.aa, optimize=True)
    M3 += 0.5 * np.einsum("bmef,cfkm,aeij->abcijk", H.ab.vovv, T.bb, T.aa, optimize=True)
    M3 += 0.5 * np.einsum("bmef,fcmk,aeij->abcijk", H.aa.vovv, T.ab, T.aa, optimize=True)
    M3 -= 0.5 * np.einsum("mcef,bfmk,aeij->abcijk", H.ab.ovvv, T.ab, T.aa, optimize=True)
    M3 -= 0.5 * np.einsum("mcjk,abim->abcijk", H.ab.ovoo, T.aa, optimize=True)
    M3 -= 0.5 * np.einsum("mnjf,fcnk,abim->abcijk", H.aa.ooov, T.ab, T.aa, optimize=True)
    M3 -= 0.5 * np.einsum("mnjf,fcnk,abim->abcijk", H.ab.ooov, T.bb, T.aa, optimize=True)
    M3 += 0.5 * np.einsum("mnfk,fcjn,abim->abcijk", H.ab.oovo, T.ab, T.aa, optimize=True)
    M3 -= 0.5 * np.einsum("mcfe,fejk,abim->abcijk", H.ab.ovvv, T.ab, T.aa, optimize=True)
    M3 += np.einsum("acie,bejk->abcijk", H.ab.vvov, T.ab, optimize=True)
    M3 -= np.einsum("me,acim,bejk->abcijk", H.b.ov, T.ab, T.ab, optimize=True)
    M3 += np.einsum("nmie,acnm,bejk->abcijk", H.ab.ooov, T.ab, T.ab, optimize=True)
    M3 += np.einsum("mcfe,afim,bejk->abcijk", H.ab.ovvv, T.aa, T.ab, optimize=True)
    M3 += np.einsum("cmef,afim,bejk->abcijk", H.bb.vovv, T.ab, T.ab, optimize=True)
    M3 -= np.einsum("amfe,fcim,bejk->abcijk", H.ab.vovv, T.ab, T.ab, optimize=True)
    M3 -= np.einsum("amik,bcjm->abcijk", H.ab.vooo, T.ab, optimize=True)
    M3 -= np.einsum("mnkf,afin,bcjm->abcijk", H.bb.ooov, T.ab, T.ab, optimize=True)
    M3 -= np.einsum("nmfk,afin,bcjm->abcijk", H.ab.oovo, T.aa, T.ab, optimize=True)
    M3 += np.einsum("nmif,afnk,bcjm->abcijk", H.ab.ooov, T.ab, T.ab, optimize=True)
    M3 -= np.einsum("amef,efik,bcjm->abcijk", H.ab.vovv, T.ab, T.ab, optimize=True)
    M3 += 0.5 * np.einsum("abie,ecjk->abcijk", H.aa.vvov, T.ab, optimize=True)
    M3 -= 0.5 * np.einsum("me,abim,ecjk->abcijk", H.a.ov, T.aa, T.ab, optimize=True)
    M3 += 0.25 * np.einsum("mnie,abmn,ecjk->abcijk", H.aa.ooov, T.aa, T.ab, optimize=True)
    M3 += np.einsum("bnef,afin,ecjk->abcijk", H.aa.vovv, T.aa, T.ab, optimize=True)
    M3 += np.einsum("bnef,afin,ecjk->abcijk", H.ab.vovv, T.ab, T.ab, optimize=True)
    M3 -= 0.5 * np.einsum("amij,bcmk->abcijk", H.aa.vooo, T.ab, optimize=True)
    M3 -= 0.25 * np.einsum("amef,efij,bcmk->abcijk", H.aa.vovv, T.aa, T.ab, optimize=True)
    M3 -= np.einsum("mnjf,afin,bcmk->abcijk", H.aa.ooov, T.aa, T.ab, optimize=True)
    M3 -= np.einsum("mnjf,afin,bcmk->abcijk", H.ab.ooov, T.ab, T.ab, optimize=True)
    M3 -= np.transpose(M3, (1, 0, 2, 3, 4, 5))
    M3 -= np.transpose(M3, (0, 1, 2, 4, 3, 5))
    return M3


def build_M3C(T, H):
    """<ij~k~ab~c~|Hbar|0> (abb)."""
    M3 = 0.5 * np.einsum("cbke,aeij->cabkij", H.ab.vvov, T.bb, optimize=True)
    M3 -= 0.5 * np.einsum("me,cbkm,aeij->cabkij", H.b.ov, T.ab, T.bb, optimize=True)
    M3 += 0.5 * np.einsum("mnke,cbmn,aeij->cabkij", H.ab.ooov, T.ab, T.bb, optimize=True)
    M3 += 0.5 * np.einsum("mbfe,cfkm,aeij->cabkij", H.ab.ovvv, T.aa, T.bb, optimize=True)
    M3 += 0.5 * np.einsum("bmef,cfkm,aeij->cabkij", H.bb.vovv, T.ab, T.bb, optimize=True)
    M3 -= 0.5 * np.einsum("cmfe,fbkm,aeij->cabkij", H.ab.vovv, T.ab, T.bb, optimize=True)
    M3 -= 0.5 * np.einsum("cmkj,abim->cabkij", H.ab.vooo, T.bb, optimize=True)
    M3 -= 0.5 * np.einsum("mnjf,cfkn,abim->cabkij", H.bb.ooov, T.ab, T.bb, optimize=True)
    M3 -= 0.5 * np.einsum("nmfj,fcnk,abim->cabkij", H.ab.oovo, T.aa, T.bb, optimize=True)
    M3 += 0.5 * np.einsum("nmkf,cfnj,abim->cabkij", H.ab.ooov, T.ab, T.bb, optimize=True)
    M3 -= 0.5 * np.einsum("cmef,efkj,abim->cabkij", H.ab.vovv, T.ab, T.bb, optimize=True)
    M3 += np.einsum("caei,ebkj->cabkij", H.ab.vvvo, T.ab, optimize=True)
    M3 -= np.einsum("me,cami,ebkj->cabkij", H.a.ov, T.ab, T.ab, optimize=True)
    M3 += np.einsum("mnei,camn,ebkj->cabkij", H.ab.oovo, T.ab, T.ab, optimize=True)
    M3 += np.einsum("cmef,afim,ebkj->cabkij", H.ab.vovv, T.bb, T.ab, optimize=True)
    M3 += np.einsum("cmef,fami,ebkj->cabkij", H.aa.vovv, T.ab, T.ab, optimize=True)
    M3 -= np.einsum("maef,cfmi,ebkj->cabkij", H.ab.ovvv, T.ab, T.ab, optimize=True)
    M3 -= np.einsum("maki,cbmj->cabkij", H.ab.ovoo, T.ab, optimize=True)
    M3 -= np.einsum("mnkf,fani,cbmj->cabkij", H.aa.ooov, T.ab, T.ab, optimize=True)
    M3 -= np.einsum("mnkf,afin,cbmj->cabkij", H.ab.ooov, T.bb, T.ab, optimize=True)
    M3 += np.einsum("mnfi,fakn,cbmj->cabkij", H.ab.oovo, T.ab, T.ab, optimize=True)
    M3 -= np.einsum("mafe,feki,cbmj->cabkij", H.ab.ovvv, T.ab, T.ab, optimize=True)
    M3 += 0.5 * np.einsum("abie,cekj->cabkij", H.bb.vvov, T.ab, optimize=True)
    M3 -= 0.5 * np.einsum("me,abim,cekj->cabkij", H.b.ov, T.bb, T.ab, optimize=True)
    M3 += 0.25 * np.einsum("mnie,abmn,cekj->cabkij", H.bb.ooov, T.bb, T.ab, optimize=True)
    M3 += np.einsum("bnef,afin,cekj->cabkij", H.bb.vovv, T.bb, T.ab, optimize=True)
    M3 += np.einsum("nbfe,fani,cekj->cabkij", H.ab.ovvv, T.ab, T.ab, optimize=True)
    M3 -= 0.5 * np.einsum("amij,cbkm->cabkij", H.bb.vooo, T.ab, optimize=True)
    M3 -= 0.25 * np.einsum("amef,efij,cbkm->cabkij", H.bb.vovv, T.bb, T.ab, optimize=True)
    M3 -= np.einsum("mnjf,afin,cbkm->cabkij", H.bb.ooov, T.bb, T.ab, optimize=True)
    M3 -= np.einsum("nmfj,fani,cbkm->cabkij", H.ab.oovo, T.ab, T.ab, optimize=True)
    M3 -= np.transpose(M3, (0, 2, 1, 3, 4, 5))
    M3 -= np.transpose(M3, (0, 1, 2, 3, 5, 4))
    return M3


def build_M3D(T, H):
    """<i~j~k~a~b~c~|Hbar|0> (bbb)."""
    M3 = -0.25 * np.einsum("amij,bcmk->abcijk", H.bb.vooo, T.bb, optimize=True)
    M3 -= 0.25 * np.einsum("me,aeij,bcmk->abcijk", H.b.ov, T.bb, T.bb, optimize=True)
    M3 -= 0.125 * np.einsum("amef,efij,bcmk->abcijk", H.bb.vovv, T.bb, T.bb, optimize=True)
    M3 -= 0.5 * np.einsum("mnjf,afin,bcmk->abcijk", H.bb.ooov, T.bb, T.bb, optimize=True)
    M3 -= 0.5 * np.einsum("nmfj,fani,bcmk->abcijk", H.ab.oovo, T.ab, T.bb, optimize=True)
    M3 += 0.25 * np.einsum("abie,ecjk->abcijk", H.bb.vvov, T.bb, optimize=True)
    M3 += 0.125 * np.einsum("mnie,abmn,ecjk->abcijk", H.bb.ooov, T.bb, T.bb, optimize=True)
    M3 += 0.5 * np.einsum("bnef,afin,ecjk->abcijk", H.bb.vovv, T.bb, T.bb, optimize=True)
    M3 += 0.5 * np.einsum("nbfe,fani,ecjk->abcijk", H.ab.ovvv, T.ab, T.bb, optimize=True)
    M3 -= np.transpose(M3, (0, 2, 1, 3, 4, 5)) + np.transpose(M3, (2, 1, 0, 3, 4, 5))
    M3 -= np.transpose(M3, (1, 0, 2, 3, 4, 5))
    M3 -= np.transpose(M3, (0, 1, 2, 3, 5, 4)) + np.transpose(M3, (0, 1, 2, 5, 4, 3))
    M3 -= np.transpose(M3, (0, 1, 2, 4, 3, 5))
    return M3


def build_L3A(L, H):
    """<0|(L1 + L2) Hbar|ijkabc> (aaa)."""
    L3 = 0.25 * np.einsum("eiba,ecjk->abcijk", H.aa.vovv, L.aa, optimize=True)
    L3 -= 0.25 * np.einsum("jima,bcmk->abcijk", H.aa.ooov, L.aa, optimize=True)
    L3 += 0.25 * np.einsum("ck,ijab->abcijk", L.a, H.aa.oovv, optimize=True)
    L3 += 0.25 * np.einsum("kc,abij->abcijk", H.a.ov, L.aa, optimize=True)
    L3 -= np.transpose(L3, (0, 2, 1, 3, 4, 5)) + np.transpose(L3, (2, 1, 0, 3, 4, 5))
    L3 -= np.transpose(L3, (1, 0, 2, 3, 4, 5))
    L3 -= np.transpose(L3, (0, 1, 2, 3, 5, 4)) + np.transpose(L3, (0, 1, 2, 5, 4, 3))
    L3 -= np.transpose(L3, (0, 1, 2, 4, 3, 5))
    return L3


def build_L3B(L, H):
    """<0|(L1 + L2) Hbar|ijk~abc~> (aab)."""
    L3 = 0.5 * np.einsum("ekbc,aeij->abcijk", H.ab.vovv, L.aa, optimize=True)
    L3 += 0.5 * np.einsum("eiba,ecjk->abcijk", H.aa.vovv, L.ab, optimize=True)
    L3 += np.einsum("ieac,bejk->abcijk", H.ab.ovvv, L.ab, optimize=True)
    L3 -= 0.5 * np.einsum("jkmc,abim->abcijk", H.ab.ooov, L.aa, optimize=True)
    L3 -= 0.5 * np.einsum("jima,bcmk->abcijk", H.aa.ooov, L.ab, optimize=True)
    L3 -= np.einsum("ikam,bcjm->abcijk", H.ab.oovo, L.ab, optimize=True)
    L3 += np.einsum("ai,jkbc->abcijk", L.a, H.ab.oovv, optimize=True)
    L3 += 0.25 * np.einsum("ck,ijab->abcijk", L.b, H.aa.oovv, optimize=True)
    L3 += np.einsum("bcjk,ia->abcijk", L.ab, H.a.ov, optimize=True)
    L3 += 0.25 * np.einsum("abij,kc->abcijk", L.aa, H.b.ov, optimize=True)
    L3 -= np.transpose(L3, (1, 0, 2, 3, 4, 5))
    L3 -= np.transpose(L3, (0, 1, 2, 4, 3, 5))
    return L3


def build_L3C(L, H):
    """<0|(L1 + L2) Hbar|ij~k~ab~c~> (abb)."""
    L3 = 0.5 * np.einsum("kecb,aeij->cabkij", H.ab.ovvv, L.bb, optimize=True)
    L3 += 0.5 * np.einsum("eiba,cekj->cabkij", H.bb.vovv, L.ab, optimize=True)
    L3 += np.einsum("eica,ebkj->cabkij", H.ab.vovv, L.ab, optimize=True)
    L3 -= 0.5 * np.einsum("kjcm,abim->cabkij", H.ab.oovo, L.bb, optimize=True)
    L3 -= 0.5 * np.einsum("jima,cbkm->cabkij", H.bb.ooov, L.ab, optimize=True)
    L3 -= np.einsum("kima,cbmj->cabkij", H.ab.ooov, L.ab, optimize=True)
    L3 += np.einsum("ai,kjcb->cabkij", L.b, H.ab.oovv, optimize=True)
    L3 += 0.25 * np.einsum("ck,ijab->cabkij", L.a, H.bb.oovv, optimize=True)
    L3 += np.einsum("cbkj,ia->cabkij", L.ab, H.b.ov, optimize=True)
    L3 += 0.25 * np.einsum("abij,kc->cabkij", L.bb, H.a.ov, optimize=True)
    L3 -= np.transpose(L3, (0, 2, 1, 3, 4, 5))
    L3 -= np.transpose(L3, (0, 1, 2, 3, 5, 4))
    return L3


def build_L3D(L, H):
    """<0|(L1 + L2) Hbar|i~j~k~a~b~c~> (bbb)."""
    L3 = 0.25 * np.einsum("eiba,ecjk->abcijk", H.bb.vovv, L.bb, optimize=True)
    L3 -= 0.25 * np.einsum("jima,bcmk->abcijk", H.bb.ooov, L.bb, optimize=True)
    L3 += 0.25 * np.einsum("ck,ijab->abcijk", L.b, H.bb.oovv, optimize=True)
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
    return {"a": build_s_lambda_1A(R3, H), "b": build_s_lambda_1B(R3, H), "aa": build_s_lambda_2A(R3, H),
            "ab": build_s_lambda_2B(R3, H), "bb": build_s_lambda_2C(R3, H)}


def build_s_lambda_1A(R3, H):
    X = 0.25 * np.einsum("ijab,abcijk->ck", H.aa.oovv, R3.aaa, optimize=True)
    X += np.einsum("jkbc,abcijk->ai", H.ab.oovv, R3.aab, optimize=True)
    X += 0.25 * np.einsum("ijab,cabkij->ck", H.bb.oovv, R3.abb, optimize=True)
    return X


def build_s_lambda_1B(R3, H):
    X = 0.25 * np.einsum("ijab,abcijk->ck", H.aa.oovv, R3.aab, optimize=True)
    X += np.einsum("kjcb,cabkij->ai", H.ab.oovv, R3.abb, optimize=True)
    X += 0.25 * np.einsum("ijab,abcijk->ck", H.bb.oovv, R3.bbb, optimize=True)
    return X


def build_s_lambda_2A(R3, H):
    X = 0.25 * np.einsum("eiba,abcijk->ecjk", H.aa.vovv, R3.aaa, optimize=True)
    X -= 0.25 * np.einsum("jima,abcijk->bcmk", H.aa.ooov, R3.aaa, optimize=True)
    X += 0.25 * np.einsum("kc,abcijk->abij", H.a.ov, R3.aaa, optimize=True)
    X += 0.5 * np.einsum("ekbc,abcijk->aeij", H.ab.vovv, R3.aab, optimize=True)
    X -= 0.5 * np.einsum("jkmc,abcijk->abim", H.ab.ooov, R3.aab, optimize=True)
    X += 0.25 * np.einsum("kc,abcijk->abij", H.b.ov, R3.aab, optimize=True)
    X -= np.transpose(X, (1, 0, 2, 3))
    X -= np.transpose(X, (0, 1, 3, 2))
    return X


def build_s_lambda_2B(R3, H):
    X = 0.5 * np.einsum("eiba,abcijk->ecjk", H.aa.vovv, R3.aab, optimize=True)
    X += np.einsum("ieac,abcijk->bejk", H.ab.ovvv, R3.aab, optimize=True)
    X -= 0.5 * np.einsum("jima,abcijk->bcmk", H.aa.ooov, R3.aab, optimize=True)
    X -= np.einsum("ikam,abcijk->bcjm", H.ab.oovo, R3.aab, optimize=True)
    X += np.einsum("ia,abcijk->bcjk", H.a.ov, R3.aab, optimize=True)
    X += 0.5 * np.einsum("eiba,cabkij->cekj", H.bb.vovv, R3.abb, optimize=True)
    X += np.einsum("eica,cabkij->ebkj", H.ab.vovv, R3.abb, optimize=True)
    X -= 0.5 * np.einsum("jima,cabkij->cbkm", H.bb.ooov, R3.abb, optimize=True)
    X -= np.einsum("kima,cabkij->cbmj", H.ab.ooov, R3.abb, optimize=True)
    X += np.einsum("ia,cabkij->cbkj", H.b.ov, R3.abb, optimize=True)
    return X


def build_s_lambda_2C(R3, H):
    X = 0.5 * np.einsum("kecb,cabkij->aeij", H.ab.ovvv, R3.abb, optimize=True)
    X -= 0.5 * np.einsum("kjcm,cabkij->abim", H.ab.oovo, R3.abb, optimize=True)
    X += 0.25 * np.einsum("kc,cabkij->abij", H.a.ov, R3.abb, optimize=True)
    X += 0.25 * np.einsum("eiba,abcijk->ecjk", H.bb.vovv, R3.bbb, optimize=True)
    X -= 0.25 * np.einsum("jima,abcijk->bcmk", H.bb.ooov, R3.bbb, optimize=True)
    X += 0.25 * np.einsum("kc,abcijk->abij", H.b.ov, R3.bbb, optimize=True)
    X -= np.transpose(X, (1, 0, 2, 3))
    X -= np.transpose(X, (0, 1, 3, 2))
    return X


# ======================================================================================
# s_t = d delta / dT
# ======================================================================================
def calc_s_t(R3, L3, T, L, H, mo, system):
    """Doubles: d(L3 . M)/dT2 (N does not depend on T2 in the transformed frame). Singles: the
    T1 derivative of sum_K (l_K M_K + r_K N_K), the rotation derivative of its densities."""
    D1, Y = build_DY(*calc_crcc23_rdm(R3, L3, T, L, system), None, H.chol, T, mo["B"], mo, system)
    singles = rotation_gradient(D1, Y, mo["h"], mo["B"], mo)
    return {"a": singles["a"], "b": singles["b"], "aa": build_s_t_2A(L3, T, H),
            "ab": build_s_t_2B(L3, T, H), "bb": build_s_t_2C(L3, T, H)}


def build_s_t_2A(L3, T, H):
    X = -0.25 * np.einsum("amij,abcijk->bcmk", H.aa.vooo, L3.aaa, optimize=True)
    X -= 0.25 * np.einsum("me,bcmk,abcijk->aeij", H.a.ov, T.aa, L3.aaa, optimize=True)
    X -= 0.25 * np.einsum("me,aeij,abcijk->bcmk", H.a.ov, T.aa, L3.aaa, optimize=True)
    X -= 0.125 * np.einsum("amef,bcmk,abcijk->efij", H.aa.vovv, T.aa, L3.aaa, optimize=True)
    X -= 0.125 * np.einsum("amef,efij,abcijk->bcmk", H.aa.vovv, T.aa, L3.aaa, optimize=True)
    X -= 0.5 * np.einsum("mnjf,bcmk,abcijk->afin", H.aa.ooov, T.aa, L3.aaa, optimize=True)
    X -= 0.5 * np.einsum("mnjf,afin,abcijk->bcmk", H.aa.ooov, T.aa, L3.aaa, optimize=True)
    X -= 0.5 * np.einsum("mnjf,afin,abcijk->bcmk", H.ab.ooov, T.ab, L3.aaa, optimize=True)
    X += 0.25 * np.einsum("abie,abcijk->ecjk", H.aa.vvov, L3.aaa, optimize=True)
    X += 0.125 * np.einsum("mnie,ecjk,abcijk->abmn", H.aa.ooov, T.aa, L3.aaa, optimize=True)
    X += 0.125 * np.einsum("mnie,abmn,abcijk->ecjk", H.aa.ooov, T.aa, L3.aaa, optimize=True)
    X += 0.5 * np.einsum("bnef,ecjk,abcijk->afin", H.aa.vovv, T.aa, L3.aaa, optimize=True)
    X += 0.5 * np.einsum("bnef,afin,abcijk->ecjk", H.aa.vovv, T.aa, L3.aaa, optimize=True)
    X += 0.5 * np.einsum("bnef,afin,abcijk->ecjk", H.ab.vovv, T.ab, L3.aaa, optimize=True)
    X += 0.5 * np.einsum("bcek,abcijk->aeij", H.ab.vvvo, L3.aab, optimize=True)
    X -= 0.5 * np.einsum("me,bcmk,abcijk->aeij", H.a.ov, T.ab, L3.aab, optimize=True)
    X += 0.5 * np.einsum("nmek,bcnm,abcijk->aeij", H.ab.oovo, T.ab, L3.aab, optimize=True)
    X += 0.5 * np.einsum("bmef,cfkm,abcijk->aeij", H.ab.vovv, T.bb, L3.aab, optimize=True)
    X += 0.5 * np.einsum("bmef,fcmk,abcijk->aeij", H.aa.vovv, T.ab, L3.aab, optimize=True)
    X -= 0.5 * np.einsum("mcef,bfmk,abcijk->aeij", H.ab.ovvv, T.ab, L3.aab, optimize=True)
    X -= 0.5 * np.einsum("mcjk,abcijk->abim", H.ab.ovoo, L3.aab, optimize=True)
    X -= 0.5 * np.einsum("mnjf,fcnk,abcijk->abim", H.aa.ooov, T.ab, L3.aab, optimize=True)
    X -= 0.5 * np.einsum("mnjf,fcnk,abcijk->abim", H.ab.ooov, T.bb, L3.aab, optimize=True)
    X += 0.5 * np.einsum("mnfk,fcjn,abcijk->abim", H.ab.oovo, T.ab, L3.aab, optimize=True)
    X -= 0.5 * np.einsum("mcfe,fejk,abcijk->abim", H.ab.ovvv, T.ab, L3.aab, optimize=True)
    X += np.einsum("mcfe,bejk,abcijk->afim", H.ab.ovvv, T.ab, L3.aab, optimize=True)
    X -= np.einsum("nmfk,bcjm,abcijk->afin", H.ab.oovo, T.ab, L3.aab, optimize=True)
    X -= 0.5 * np.einsum("me,ecjk,abcijk->abim", H.a.ov, T.ab, L3.aab, optimize=True)
    X += 0.25 * np.einsum("mnie,ecjk,abcijk->abmn", H.aa.ooov, T.ab, L3.aab, optimize=True)
    X += np.einsum("bnef,ecjk,abcijk->afin", H.aa.vovv, T.ab, L3.aab, optimize=True)
    X -= 0.25 * np.einsum("amef,bcmk,abcijk->efij", H.aa.vovv, T.ab, L3.aab, optimize=True)
    X -= np.einsum("mnjf,bcmk,abcijk->afin", H.aa.ooov, T.ab, L3.aab, optimize=True)
    X += 0.5 * np.einsum("mbfe,aeij,cabkij->cfkm", H.ab.ovvv, T.bb, L3.abb, optimize=True)
    X -= 0.5 * np.einsum("nmfj,abim,cabkij->fcnk", H.ab.oovo, T.bb, L3.abb, optimize=True)
    X -= np.transpose(X, (1, 0, 2, 3))
    X -= np.transpose(X, (0, 1, 3, 2))
    return X


def build_s_t_2B(L3, T, H):
    X = -0.5 * np.einsum("mnjf,bcmk,abcijk->afin", H.ab.ooov, T.aa, L3.aaa, optimize=True)
    X += 0.5 * np.einsum("bnef,ecjk,abcijk->afin", H.ab.vovv, T.aa, L3.aaa, optimize=True)
    X -= 0.5 * np.einsum("me,aeij,abcijk->bcmk", H.a.ov, T.aa, L3.aab, optimize=True)
    X += 0.5 * np.einsum("nmek,aeij,abcijk->bcnm", H.ab.oovo, T.aa, L3.aab, optimize=True)
    X += 0.5 * np.einsum("bmef,aeij,abcijk->fcmk", H.aa.vovv, T.aa, L3.aab, optimize=True)
    X -= 0.5 * np.einsum("mcef,aeij,abcijk->bfmk", H.ab.ovvv, T.aa, L3.aab, optimize=True)
    X -= 0.5 * np.einsum("mnjf,abim,abcijk->fcnk", H.aa.ooov, T.aa, L3.aab, optimize=True)
    X += 0.5 * np.einsum("mnfk,abim,abcijk->fcjn", H.ab.oovo, T.aa, L3.aab, optimize=True)
    X -= 0.5 * np.einsum("mcfe,abim,abcijk->fejk", H.ab.ovvv, T.aa, L3.aab, optimize=True)
    X += np.einsum("acie,abcijk->bejk", H.ab.vvov, L3.aab, optimize=True)
    X -= np.einsum("me,bejk,abcijk->acim", H.b.ov, T.ab, L3.aab, optimize=True)
    X -= np.einsum("me,acim,abcijk->bejk", H.b.ov, T.ab, L3.aab, optimize=True)
    X += np.einsum("nmie,bejk,abcijk->acnm", H.ab.ooov, T.ab, L3.aab, optimize=True)
    X += np.einsum("nmie,acnm,abcijk->bejk", H.ab.ooov, T.ab, L3.aab, optimize=True)
    X += np.einsum("mcfe,afim,abcijk->bejk", H.ab.ovvv, T.aa, L3.aab, optimize=True)
    X += np.einsum("cmef,bejk,abcijk->afim", H.bb.vovv, T.ab, L3.aab, optimize=True)
    X += np.einsum("cmef,afim,abcijk->bejk", H.bb.vovv, T.ab, L3.aab, optimize=True)
    X -= np.einsum("amfe,bejk,abcijk->fcim", H.ab.vovv, T.ab, L3.aab, optimize=True)
    X -= np.einsum("amfe,fcim,abcijk->bejk", H.ab.vovv, T.ab, L3.aab, optimize=True)
    X -= np.einsum("amik,abcijk->bcjm", H.ab.vooo, L3.aab, optimize=True)
    X -= np.einsum("mnkf,bcjm,abcijk->afin", H.bb.ooov, T.ab, L3.aab, optimize=True)
    X -= np.einsum("mnkf,afin,abcijk->bcjm", H.bb.ooov, T.ab, L3.aab, optimize=True)
    X -= np.einsum("nmfk,afin,abcijk->bcjm", H.ab.oovo, T.aa, L3.aab, optimize=True)
    X += np.einsum("nmif,bcjm,abcijk->afnk", H.ab.ooov, T.ab, L3.aab, optimize=True)
    X += np.einsum("nmif,afnk,abcijk->bcjm", H.ab.ooov, T.ab, L3.aab, optimize=True)
    X -= np.einsum("amef,bcjm,abcijk->efik", H.ab.vovv, T.ab, L3.aab, optimize=True)
    X -= np.einsum("amef,efik,abcijk->bcjm", H.ab.vovv, T.ab, L3.aab, optimize=True)
    X += 0.5 * np.einsum("abie,abcijk->ecjk", H.aa.vvov, L3.aab, optimize=True)
    X -= 0.5 * np.einsum("me,abim,abcijk->ecjk", H.a.ov, T.aa, L3.aab, optimize=True)
    X += 0.25 * np.einsum("mnie,abmn,abcijk->ecjk", H.aa.ooov, T.aa, L3.aab, optimize=True)
    X += np.einsum("bnef,afin,abcijk->ecjk", H.aa.vovv, T.aa, L3.aab, optimize=True)
    X += np.einsum("bnef,ecjk,abcijk->afin", H.ab.vovv, T.ab, L3.aab, optimize=True)
    X += np.einsum("bnef,afin,abcijk->ecjk", H.ab.vovv, T.ab, L3.aab, optimize=True)
    X -= 0.5 * np.einsum("amij,abcijk->bcmk", H.aa.vooo, L3.aab, optimize=True)
    X -= 0.25 * np.einsum("amef,efij,abcijk->bcmk", H.aa.vovv, T.aa, L3.aab, optimize=True)
    X -= np.einsum("mnjf,afin,abcijk->bcmk", H.aa.ooov, T.aa, L3.aab, optimize=True)
    X -= np.einsum("mnjf,bcmk,abcijk->afin", H.ab.ooov, T.ab, L3.aab, optimize=True)
    X -= np.einsum("mnjf,afin,abcijk->bcmk", H.ab.ooov, T.ab, L3.aab, optimize=True)
    X -= 0.5 * np.einsum("me,aeij,cabkij->cbkm", H.b.ov, T.bb, L3.abb, optimize=True)
    X += 0.5 * np.einsum("mnke,aeij,cabkij->cbmn", H.ab.ooov, T.bb, L3.abb, optimize=True)
    X += 0.5 * np.einsum("bmef,aeij,cabkij->cfkm", H.bb.vovv, T.bb, L3.abb, optimize=True)
    X -= 0.5 * np.einsum("cmfe,aeij,cabkij->fbkm", H.ab.vovv, T.bb, L3.abb, optimize=True)
    X -= 0.5 * np.einsum("mnjf,abim,cabkij->cfkn", H.bb.ooov, T.bb, L3.abb, optimize=True)
    X += 0.5 * np.einsum("nmkf,abim,cabkij->cfnj", H.ab.ooov, T.bb, L3.abb, optimize=True)
    X -= 0.5 * np.einsum("cmef,abim,cabkij->efkj", H.ab.vovv, T.bb, L3.abb, optimize=True)
    X += np.einsum("caei,cabkij->ebkj", H.ab.vvvo, L3.abb, optimize=True)
    X -= np.einsum("me,ebkj,cabkij->cami", H.a.ov, T.ab, L3.abb, optimize=True)
    X -= np.einsum("me,cami,cabkij->ebkj", H.a.ov, T.ab, L3.abb, optimize=True)
    X += np.einsum("mnei,ebkj,cabkij->camn", H.ab.oovo, T.ab, L3.abb, optimize=True)
    X += np.einsum("mnei,camn,cabkij->ebkj", H.ab.oovo, T.ab, L3.abb, optimize=True)
    X += np.einsum("cmef,afim,cabkij->ebkj", H.ab.vovv, T.bb, L3.abb, optimize=True)
    X += np.einsum("cmef,ebkj,cabkij->fami", H.aa.vovv, T.ab, L3.abb, optimize=True)
    X += np.einsum("cmef,fami,cabkij->ebkj", H.aa.vovv, T.ab, L3.abb, optimize=True)
    X -= np.einsum("maef,ebkj,cabkij->cfmi", H.ab.ovvv, T.ab, L3.abb, optimize=True)
    X -= np.einsum("maef,cfmi,cabkij->ebkj", H.ab.ovvv, T.ab, L3.abb, optimize=True)
    X -= np.einsum("maki,cabkij->cbmj", H.ab.ovoo, L3.abb, optimize=True)
    X -= np.einsum("mnkf,cbmj,cabkij->fani", H.aa.ooov, T.ab, L3.abb, optimize=True)
    X -= np.einsum("mnkf,fani,cabkij->cbmj", H.aa.ooov, T.ab, L3.abb, optimize=True)
    X -= np.einsum("mnkf,afin,cabkij->cbmj", H.ab.ooov, T.bb, L3.abb, optimize=True)
    X += np.einsum("mnfi,cbmj,cabkij->fakn", H.ab.oovo, T.ab, L3.abb, optimize=True)
    X += np.einsum("mnfi,fakn,cabkij->cbmj", H.ab.oovo, T.ab, L3.abb, optimize=True)
    X -= np.einsum("mafe,cbmj,cabkij->feki", H.ab.ovvv, T.ab, L3.abb, optimize=True)
    X -= np.einsum("mafe,feki,cabkij->cbmj", H.ab.ovvv, T.ab, L3.abb, optimize=True)
    X += 0.5 * np.einsum("abie,cabkij->cekj", H.bb.vvov, L3.abb, optimize=True)
    X -= 0.5 * np.einsum("me,abim,cabkij->cekj", H.b.ov, T.bb, L3.abb, optimize=True)
    X += 0.25 * np.einsum("mnie,abmn,cabkij->cekj", H.bb.ooov, T.bb, L3.abb, optimize=True)
    X += np.einsum("bnef,afin,cabkij->cekj", H.bb.vovv, T.bb, L3.abb, optimize=True)
    X += np.einsum("nbfe,cekj,cabkij->fani", H.ab.ovvv, T.ab, L3.abb, optimize=True)
    X += np.einsum("nbfe,fani,cabkij->cekj", H.ab.ovvv, T.ab, L3.abb, optimize=True)
    X -= 0.5 * np.einsum("amij,cabkij->cbkm", H.bb.vooo, L3.abb, optimize=True)
    X -= 0.25 * np.einsum("amef,efij,cabkij->cbkm", H.bb.vovv, T.bb, L3.abb, optimize=True)
    X -= np.einsum("mnjf,afin,cabkij->cbkm", H.bb.ooov, T.bb, L3.abb, optimize=True)
    X -= np.einsum("nmfj,cbkm,cabkij->fani", H.ab.oovo, T.ab, L3.abb, optimize=True)
    X -= np.einsum("nmfj,fani,cabkij->cbkm", H.ab.oovo, T.ab, L3.abb, optimize=True)
    X -= 0.5 * np.einsum("nmfj,bcmk,abcijk->fani", H.ab.oovo, T.bb, L3.bbb, optimize=True)
    X += 0.5 * np.einsum("nbfe,ecjk,abcijk->fani", H.ab.ovvv, T.bb, L3.bbb, optimize=True)
    return X


def build_s_t_2C(L3, T, H):
    X = 0.5 * np.einsum("bmef,aeij,abcijk->cfkm", H.ab.vovv, T.aa, L3.aab, optimize=True)
    X -= 0.5 * np.einsum("mnjf,abim,abcijk->fcnk", H.ab.ooov, T.aa, L3.aab, optimize=True)
    X += 0.5 * np.einsum("cbke,cabkij->aeij", H.ab.vvov, L3.abb, optimize=True)
    X -= 0.5 * np.einsum("me,cbkm,cabkij->aeij", H.b.ov, T.ab, L3.abb, optimize=True)
    X += 0.5 * np.einsum("mnke,cbmn,cabkij->aeij", H.ab.ooov, T.ab, L3.abb, optimize=True)
    X += 0.5 * np.einsum("mbfe,cfkm,cabkij->aeij", H.ab.ovvv, T.aa, L3.abb, optimize=True)
    X += 0.5 * np.einsum("bmef,cfkm,cabkij->aeij", H.bb.vovv, T.ab, L3.abb, optimize=True)
    X -= 0.5 * np.einsum("cmfe,fbkm,cabkij->aeij", H.ab.vovv, T.ab, L3.abb, optimize=True)
    X -= 0.5 * np.einsum("cmkj,cabkij->abim", H.ab.vooo, L3.abb, optimize=True)
    X -= 0.5 * np.einsum("mnjf,cfkn,cabkij->abim", H.bb.ooov, T.ab, L3.abb, optimize=True)
    X -= 0.5 * np.einsum("nmfj,fcnk,cabkij->abim", H.ab.oovo, T.aa, L3.abb, optimize=True)
    X += 0.5 * np.einsum("nmkf,cfnj,cabkij->abim", H.ab.ooov, T.ab, L3.abb, optimize=True)
    X -= 0.5 * np.einsum("cmef,efkj,cabkij->abim", H.ab.vovv, T.ab, L3.abb, optimize=True)
    X += np.einsum("cmef,ebkj,cabkij->afim", H.ab.vovv, T.ab, L3.abb, optimize=True)
    X -= np.einsum("mnkf,cbmj,cabkij->afin", H.ab.ooov, T.ab, L3.abb, optimize=True)
    X -= 0.5 * np.einsum("me,cekj,cabkij->abim", H.b.ov, T.ab, L3.abb, optimize=True)
    X += 0.25 * np.einsum("mnie,cekj,cabkij->abmn", H.bb.ooov, T.ab, L3.abb, optimize=True)
    X += np.einsum("bnef,cekj,cabkij->afin", H.bb.vovv, T.ab, L3.abb, optimize=True)
    X -= 0.25 * np.einsum("amef,cbkm,cabkij->efij", H.bb.vovv, T.ab, L3.abb, optimize=True)
    X -= np.einsum("mnjf,cbkm,cabkij->afin", H.bb.ooov, T.ab, L3.abb, optimize=True)
    X -= 0.25 * np.einsum("amij,abcijk->bcmk", H.bb.vooo, L3.bbb, optimize=True)
    X -= 0.25 * np.einsum("me,bcmk,abcijk->aeij", H.b.ov, T.bb, L3.bbb, optimize=True)
    X -= 0.25 * np.einsum("me,aeij,abcijk->bcmk", H.b.ov, T.bb, L3.bbb, optimize=True)
    X -= 0.125 * np.einsum("amef,bcmk,abcijk->efij", H.bb.vovv, T.bb, L3.bbb, optimize=True)
    X -= 0.125 * np.einsum("amef,efij,abcijk->bcmk", H.bb.vovv, T.bb, L3.bbb, optimize=True)
    X -= 0.5 * np.einsum("mnjf,bcmk,abcijk->afin", H.bb.ooov, T.bb, L3.bbb, optimize=True)
    X -= 0.5 * np.einsum("mnjf,afin,abcijk->bcmk", H.bb.ooov, T.bb, L3.bbb, optimize=True)
    X -= 0.5 * np.einsum("nmfj,fani,abcijk->bcmk", H.ab.oovo, T.ab, L3.bbb, optimize=True)
    X += 0.25 * np.einsum("abie,abcijk->ecjk", H.bb.vvov, L3.bbb, optimize=True)
    X += 0.125 * np.einsum("mnie,ecjk,abcijk->abmn", H.bb.ooov, T.bb, L3.bbb, optimize=True)
    X += 0.125 * np.einsum("mnie,abmn,abcijk->ecjk", H.bb.ooov, T.bb, L3.bbb, optimize=True)
    X += 0.5 * np.einsum("bnef,ecjk,abcijk->afin", H.bb.vovv, T.bb, L3.bbb, optimize=True)
    X += 0.5 * np.einsum("bnef,afin,abcijk->ecjk", H.bb.vovv, T.bb, L3.bbb, optimize=True)
    X += 0.5 * np.einsum("nbfe,fani,abcijk->ecjk", H.ab.ovvv, T.ab, L3.bbb, optimize=True)
    X -= np.transpose(X, (1, 0, 2, 3))
    X -= np.transpose(X, (0, 1, 3, 2))
    return X


# ======================================================================================
# B^T w = d/de g(T + e w), g the left-CCSD residual
# ======================================================================================
def calc_BTw(w, T, L, H, mo, system):
    """Doubles: the T2 part of w through the second derivative of the CCSD Lagrangian, the T1
    part through H1 = [H, W1]. Singles: rotation derivatives of the directional densities and
    the double rotation D_L . [[H, W1], a+ i]."""
    W = SimpleNamespace(**{k: _get(w, k) for k in ("a", "b", "aa", "ab", "bb")})
    W1 = {s: _embed_vo(_get(w, s), mo, s) for s in ("a", "b")}
    H1 = build_hbar_ccs_chol_derivative(W, H, system)
    h1 = {s: mo["h"][s] @ W1[s] - W1[s] @ mo["h"][s] for s in ("a", "b")}
    dB = {s: mo["B"][s] @ W1[s] - W1[s] @ mo["B"][s] for s in ("a", "b")}

    rdm1, rdm2 = calc_ccsd_rdm(T, L, system)
    vvvv = lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s)
    D1L, YL = build_DY(rdm1, rdm2, vvvv, H.chol, T, mo["B"], mo, system, reference=True)
    _, YLdB = build_DY(rdm1, rdm2, vvvv, H1.chol, T, dB, mo, system, reference=True)
    D1w, Yw = build_DY(*calc_ccsd_rdm_derivative(T, W, L, system), lambda a, s: calc_rdm2_vvvv_slice(L, W, a, s),
                       H.chol, T, mo["B"], mo, system)
    r1 = rotation_gradient(D1w, Yw, mo["h"], mo["B"], mo)
    r2 = rotation_gradient(D1L, YL, h1, dB, mo)
    r3 = rotation_gradient({s: np.zeros_like(D1L[s]) for s in ("a", "b")}, YLdB, mo["h"], mo["B"], mo)

    return {"a": r1["a"] + r2["a"] + r3["a"], "b": r1["b"] + r2["b"] + r3["b"],
            "aa": build_BTw_2A(W, T, L, H, H1), "ab": build_BTw_2B(W, T, L, H, H1),
            "bb": build_BTw_2C(W, T, L, H, H1)}


def build_BTw_2A(W, T, L, H, H1):
    X = -0.25 * np.einsum("mnef,afmn,abij->ebij", H.aa.oovv, W.aa, L.aa, optimize=True)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H.aa.oovv, W.aa, L.aa, optimize=True)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H.ab.oovv, W.ab, L.aa, optimize=True)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H.aa.oovv, W.aa, L.aa, optimize=True)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H.aa.oovv, W.aa, L.aa, optimize=True)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H.ab.oovv, W.ab, L.aa, optimize=True)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H.aa.oovv, W.aa, L.aa, optimize=True)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H.aa.oovv, W.aa, L.aa, optimize=True)
    X += np.einsum("mnef,afin,abij->ebmj", H.ab.oovv, W.ab, L.aa, optimize=True)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H.aa.oovv, W.aa, L.aa, optimize=True)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H.aa.oovv, W.aa, L.aa, optimize=True)
    X -= 0.5 * np.einsum("mnef,ebij,abij->afmn", H.aa.oovv, W.ab, L.ab, optimize=True)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H.aa.oovv, W.ab, L.ab, optimize=True)
    X += np.einsum("mnef,fbnj,abij->aeim", H.aa.oovv, W.ab, L.ab, optimize=True)
    X += np.einsum("mnef,fbnj,abij->aeim", H.ab.oovv, W.bb, L.ab, optimize=True)
    X += 0.25 * H1.aa.oovv.transpose(2, 3, 0, 1)
    X += np.einsum("me,ai->aeim", H1.a.ov, L.a, optimize=True)
    X -= 0.5 * np.einsum("mnif,ai->afmn", H1.aa.ooov, L.a, optimize=True)
    X += 0.5 * np.einsum("anef,ai->efin", H1.aa.vovv, L.a, optimize=True)
    X += 0.5 * np.einsum("ae,abij->ebij", H1.a.vv, L.aa, optimize=True)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H1.aa.oovv, T.aa, L.aa, optimize=True)
    X -= 0.25 * np.einsum("mnef,afmn,abij->ebij", H1.aa.oovv, T.aa, L.aa, optimize=True)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H1.ab.oovv, T.ab, L.aa, optimize=True)
    X -= 0.5 * np.einsum("mi,abij->abmj", H1.a.oo, L.aa, optimize=True)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H1.aa.oovv, T.aa, L.aa, optimize=True)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H1.aa.oovv, T.aa, L.aa, optimize=True)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H1.ab.oovv, T.ab, L.aa, optimize=True)
    X += np.einsum("amie,abij->ebmj", H1.aa.voov, L.aa, optimize=True)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H1.aa.oovv, T.aa, L.aa, optimize=True)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H1.aa.oovv, T.aa, L.aa, optimize=True)
    X += np.einsum("mnef,afin,abij->ebmj", H1.ab.oovv, T.ab, L.aa, optimize=True)
    X += 0.125 * np.einsum("mnij,abij->abmn", H1.aa.oooo, L.aa, optimize=True)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H1.aa.oovv, T.aa, L.aa, optimize=True)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H1.aa.oovv, T.aa, L.aa, optimize=True)
    X -= 0.5 * np.einsum("mnef,ebij,abij->afmn", H1.aa.oovv, T.ab, L.ab, optimize=True)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H1.aa.oovv, T.ab, L.ab, optimize=True)
    X += np.einsum("mnef,fbnj,abij->aeim", H1.aa.oovv, T.ab, L.ab, optimize=True)
    X += np.einsum("mnef,fbnj,abij->aeim", H1.ab.oovv, T.bb, L.ab, optimize=True)
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
    X = -0.5 * np.einsum("mnef,ebij,abij->afmn", H.ab.oovv, W.aa, L.aa, optimize=True)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H.ab.oovv, W.aa, L.aa, optimize=True)
    X += np.einsum("mnef,ebmj,abij->afin", H.ab.oovv, W.aa, L.aa, optimize=True)
    X += 0.5 * np.einsum("mnef,afin,abij->bejm", H.bb.oovv, W.ab, L.aa, optimize=True)
    X += 0.5 * np.einsum("mnef,bejm,abij->afin", H.bb.oovv, W.ab, L.aa, optimize=True)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H.aa.oovv, W.aa, L.ab, optimize=True)
    X -= np.einsum("mnef,afmn,abij->ebij", H.ab.oovv, W.ab, L.ab, optimize=True)
    X -= np.einsum("mnef,ebij,abij->afmn", H.ab.oovv, W.ab, L.ab, optimize=True)
    X -= 0.5 * np.einsum("mnef,bfmn,abij->aeij", H.bb.oovv, W.bb, L.ab, optimize=True)
    X -= np.einsum("nmfe,fbnm,abij->aeij", H.ab.oovv, W.ab, L.ab, optimize=True)
    X -= np.einsum("nmfe,aeij,abij->fbnm", H.ab.oovv, W.ab, L.ab, optimize=True)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H.aa.oovv, W.aa, L.ab, optimize=True)
    X -= np.einsum("mnef,efin,abij->abmj", H.ab.oovv, W.ab, L.ab, optimize=True)
    X -= np.einsum("mnef,abmj,abij->efin", H.ab.oovv, W.ab, L.ab, optimize=True)
    X -= 0.5 * np.einsum("mnef,efjn,abij->abim", H.bb.oovv, W.bb, L.ab, optimize=True)
    X -= np.einsum("nmfe,fenj,abij->abim", H.ab.oovv, W.ab, L.ab, optimize=True)
    X -= np.einsum("nmfe,abim,abij->fenj", H.ab.oovv, W.ab, L.ab, optimize=True)
    X += np.einsum("mnef,aeim,abij->fbnj", H.aa.oovv, W.aa, L.ab, optimize=True)
    X += np.einsum("nmfe,aeim,abij->fbnj", H.ab.oovv, W.ab, L.ab, optimize=True)
    X += np.einsum("nmfe,fbnj,abij->aeim", H.ab.oovv, W.ab, L.ab, optimize=True)
    X += np.einsum("mnef,fbnj,abij->aeim", H.bb.oovv, W.bb, L.ab, optimize=True)
    X += np.einsum("mnef,afmj,abij->ebin", H.ab.oovv, W.ab, L.ab, optimize=True)
    X += np.einsum("mnef,ebin,abij->afmj", H.ab.oovv, W.ab, L.ab, optimize=True)
    X += np.einsum("mnef,efij,abij->abmn", H.ab.oovv, W.ab, L.ab, optimize=True)
    X += np.einsum("mnef,abmn,abij->efij", H.ab.oovv, W.ab, L.ab, optimize=True)
    X -= 0.5 * np.einsum("nmfe,ebij,abij->fanm", H.ab.oovv, W.bb, L.bb, optimize=True)
    X -= 0.5 * np.einsum("nmfe,abmj,abij->feni", H.ab.oovv, W.bb, L.bb, optimize=True)
    X += np.einsum("mnef,afin,abij->ebmj", H.ab.oovv, W.bb, L.bb, optimize=True)
    X += 0.5 * np.einsum("mnef,fani,abij->ebmj", H.aa.oovv, W.ab, L.bb, optimize=True)
    X += 0.5 * np.einsum("mnef,ebmj,abij->fani", H.aa.oovv, W.ab, L.bb, optimize=True)
    X += H1.ab.oovv.transpose(2, 3, 0, 1)
    X += np.einsum("me,ai->aeim", H1.b.ov, L.a, optimize=True)
    X -= np.einsum("mnif,ai->afmn", H1.ab.ooov, L.a, optimize=True)
    X += np.einsum("anef,ai->efin", H1.ab.vovv, L.a, optimize=True)
    X += np.einsum("me,ai->eami", H1.a.ov, L.b, optimize=True)
    X -= np.einsum("nmfi,ai->fanm", H1.ab.oovo, L.b, optimize=True)
    X += np.einsum("nafe,ai->feni", H1.ab.ovvv, L.b, optimize=True)
    X -= 0.5 * np.einsum("mnef,ebij,abij->afmn", H1.ab.oovv, T.aa, L.aa, optimize=True)
    X -= 0.5 * np.einsum("mnef,abmj,abij->efin", H1.ab.oovv, T.aa, L.aa, optimize=True)
    X += np.einsum("mnef,ebmj,abij->afin", H1.ab.oovv, T.aa, L.aa, optimize=True)
    X += np.einsum("amie,abij->bejm", H1.ab.voov, L.aa, optimize=True)
    X += 0.5 * np.einsum("mnef,bejm,abij->afin", H1.bb.oovv, T.ab, L.aa, optimize=True)
    X += 0.5 * np.einsum("mnef,afin,abij->bejm", H1.bb.oovv, T.ab, L.aa, optimize=True)
    X += np.einsum("ae,abij->ebij", H1.a.vv, L.ab, optimize=True)
    X -= 0.5 * np.einsum("mnef,afmn,abij->ebij", H1.aa.oovv, T.aa, L.ab, optimize=True)
    X -= np.einsum("mnef,ebij,abij->afmn", H1.ab.oovv, T.ab, L.ab, optimize=True)
    X -= np.einsum("mnef,afmn,abij->ebij", H1.ab.oovv, T.ab, L.ab, optimize=True)
    X += np.einsum("be,abij->aeij", H1.b.vv, L.ab, optimize=True)
    X -= 0.5 * np.einsum("mnef,bfmn,abij->aeij", H1.bb.oovv, T.bb, L.ab, optimize=True)
    X -= np.einsum("nmfe,aeij,abij->fbnm", H1.ab.oovv, T.ab, L.ab, optimize=True)
    X -= np.einsum("nmfe,fbnm,abij->aeij", H1.ab.oovv, T.ab, L.ab, optimize=True)
    X -= np.einsum("mi,abij->abmj", H1.a.oo, L.ab, optimize=True)
    X -= 0.5 * np.einsum("mnef,efin,abij->abmj", H1.aa.oovv, T.aa, L.ab, optimize=True)
    X -= np.einsum("mnef,abmj,abij->efin", H1.ab.oovv, T.ab, L.ab, optimize=True)
    X -= np.einsum("mnef,efin,abij->abmj", H1.ab.oovv, T.ab, L.ab, optimize=True)
    X -= np.einsum("mj,abij->abim", H1.b.oo, L.ab, optimize=True)
    X -= 0.5 * np.einsum("mnef,efjn,abij->abim", H1.bb.oovv, T.bb, L.ab, optimize=True)
    X -= np.einsum("nmfe,abim,abij->fenj", H1.ab.oovv, T.ab, L.ab, optimize=True)
    X -= np.einsum("nmfe,fenj,abij->abim", H1.ab.oovv, T.ab, L.ab, optimize=True)
    X += np.einsum("amie,abij->ebmj", H1.aa.voov, L.ab, optimize=True)
    X += np.einsum("mnef,aeim,abij->fbnj", H1.aa.oovv, T.aa, L.ab, optimize=True)
    X += np.einsum("nmfe,fbnj,abij->aeim", H1.ab.oovv, T.ab, L.ab, optimize=True)
    X += np.einsum("nmfe,aeim,abij->fbnj", H1.ab.oovv, T.ab, L.ab, optimize=True)
    X += np.einsum("mnef,fbnj,abij->aeim", H1.bb.oovv, T.bb, L.ab, optimize=True)
    X += np.einsum("bmje,abij->aeim", H1.bb.voov, L.ab, optimize=True)
    X -= np.einsum("mbie,abij->aemj", H1.ab.ovov, L.ab, optimize=True)
    X -= np.einsum("amej,abij->ebim", H1.ab.vovo, L.ab, optimize=True)
    X += np.einsum("mnef,ebin,abij->afmj", H1.ab.oovv, T.ab, L.ab, optimize=True)
    X += np.einsum("mnef,afmj,abij->ebin", H1.ab.oovv, T.ab, L.ab, optimize=True)
    X += np.einsum("mnij,abij->abmn", H1.ab.oooo, L.ab, optimize=True)
    X += np.einsum("mnef,abmn,abij->efij", H1.ab.oovv, T.ab, L.ab, optimize=True)
    X += np.einsum("mnef,efij,abij->abmn", H1.ab.oovv, T.ab, L.ab, optimize=True)
    X -= 0.5 * np.einsum("nmfe,ebij,abij->fanm", H1.ab.oovv, T.bb, L.bb, optimize=True)
    X -= 0.5 * np.einsum("nmfe,abmj,abij->feni", H1.ab.oovv, T.bb, L.bb, optimize=True)
    X += np.einsum("maei,abij->ebmj", H1.ab.ovvo, L.bb, optimize=True)
    X += np.einsum("mnef,afin,abij->ebmj", H1.ab.oovv, T.bb, L.bb, optimize=True)
    X += 0.5 * np.einsum("mnef,ebmj,abij->fani", H1.aa.oovv, T.ab, L.bb, optimize=True)
    X += 0.5 * np.einsum("mnef,fani,abij->ebmj", H1.aa.oovv, T.ab, L.bb, optimize=True)
    # vvvv of H1: sum_ab L(abij) <ab|ef>'
    for B1, B2 in ((H1.chol.a.vv, H.chol.b.vv), (H.chol.a.vv, H1.chol.b.vv)):
        LB = np.einsum("abij,xae->xebij", L.ab, B1, optimize=True)
        X += np.einsum("xebij,xbf->efij", LB, B2, optimize=True)
    return X


def build_BTw_2C(W, T, L, H, H1):
    X = -0.5 * np.einsum("mnef,aeij,abij->bfmn", H.bb.oovv, W.ab, L.ab, optimize=True)
    X -= 0.5 * np.einsum("mnef,abim,abij->efjn", H.bb.oovv, W.ab, L.ab, optimize=True)
    X += np.einsum("mnef,aeim,abij->fbnj", H.ab.oovv, W.aa, L.ab, optimize=True)
    X += np.einsum("mnef,aeim,abij->fbnj", H.bb.oovv, W.ab, L.ab, optimize=True)
    X -= 0.25 * np.einsum("mnef,afmn,abij->ebij", H.bb.oovv, W.bb, L.bb, optimize=True)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H.bb.oovv, W.bb, L.bb, optimize=True)
    X -= 0.5 * np.einsum("nmfe,fanm,abij->ebij", H.ab.oovv, W.ab, L.bb, optimize=True)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H.bb.oovv, W.bb, L.bb, optimize=True)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H.bb.oovv, W.bb, L.bb, optimize=True)
    X -= 0.5 * np.einsum("nmfe,feni,abij->abmj", H.ab.oovv, W.ab, L.bb, optimize=True)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H.bb.oovv, W.bb, L.bb, optimize=True)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H.bb.oovv, W.bb, L.bb, optimize=True)
    X += np.einsum("mnef,ebmj,abij->afin", H.ab.oovv, W.ab, L.bb, optimize=True)
    X += 0.0625 * np.einsum("mnef,efij,abij->abmn", H.bb.oovv, W.bb, L.bb, optimize=True)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H.bb.oovv, W.bb, L.bb, optimize=True)
    X += 0.25 * H1.bb.oovv.transpose(2, 3, 0, 1)
    X += np.einsum("me,ai->aeim", H1.b.ov, L.b, optimize=True)
    X -= 0.5 * np.einsum("mnif,ai->afmn", H1.bb.ooov, L.b, optimize=True)
    X += 0.5 * np.einsum("anef,ai->efin", H1.bb.vovv, L.b, optimize=True)
    X -= 0.5 * np.einsum("mnef,aeij,abij->bfmn", H1.bb.oovv, T.ab, L.ab, optimize=True)
    X -= 0.5 * np.einsum("mnef,abim,abij->efjn", H1.bb.oovv, T.ab, L.ab, optimize=True)
    X += np.einsum("amie,abij->ebmj", H1.ab.voov, L.ab, optimize=True)
    X += np.einsum("mnef,aeim,abij->fbnj", H1.ab.oovv, T.aa, L.ab, optimize=True)
    X += np.einsum("mnef,aeim,abij->fbnj", H1.bb.oovv, T.ab, L.ab, optimize=True)
    X += 0.5 * np.einsum("ae,abij->ebij", H1.b.vv, L.bb, optimize=True)
    X -= 0.25 * np.einsum("mnef,ebij,abij->afmn", H1.bb.oovv, T.bb, L.bb, optimize=True)
    X -= 0.25 * np.einsum("mnef,afmn,abij->ebij", H1.bb.oovv, T.bb, L.bb, optimize=True)
    X -= 0.5 * np.einsum("nmfe,fanm,abij->ebij", H1.ab.oovv, T.ab, L.bb, optimize=True)
    X -= 0.5 * np.einsum("mi,abij->abmj", H1.b.oo, L.bb, optimize=True)
    X -= 0.25 * np.einsum("mnef,abmj,abij->efin", H1.bb.oovv, T.bb, L.bb, optimize=True)
    X -= 0.25 * np.einsum("mnef,efin,abij->abmj", H1.bb.oovv, T.bb, L.bb, optimize=True)
    X -= 0.5 * np.einsum("nmfe,feni,abij->abmj", H1.ab.oovv, T.ab, L.bb, optimize=True)
    X += np.einsum("amie,abij->ebmj", H1.bb.voov, L.bb, optimize=True)
    X += 0.5 * np.einsum("mnef,ebmj,abij->afin", H1.bb.oovv, T.bb, L.bb, optimize=True)
    X += 0.5 * np.einsum("mnef,afin,abij->ebmj", H1.bb.oovv, T.bb, L.bb, optimize=True)
    X += np.einsum("mnef,ebmj,abij->afin", H1.ab.oovv, T.ab, L.bb, optimize=True)
    X += 0.125 * np.einsum("mnij,abij->abmn", H1.bb.oooo, L.bb, optimize=True)
    X += 0.0625 * np.einsum("mnef,abmn,abij->efij", H1.bb.oovv, T.bb, L.bb, optimize=True)
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
    eo = {s: np.diag(getattr(Hbar, s).oo) for s in ("a", "b")}
    ev = {s: np.diag(getattr(Hbar, s).vv) for s in ("a", "b")}
    d = [ev["a"][:, None] - eo["a"][None, :], ev["b"][:, None] - eo["b"][None, :]]
    for s, t in (("a", "a"), ("a", "b"), ("b", "b")):
        d.append(ev[s][:, None, None, None] + ev[t][None, :, None, None]
                 - eo[s][None, None, :, None] - eo[t][None, None, None, :])
    diag = np.concatenate([np.ravel(x) for x in d])
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
