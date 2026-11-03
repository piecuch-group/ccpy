"""Orbital relaxation (Z-vector / CPHF) of amplitude-relaxed CC densities, per spin.

Given the amplitude-relaxed densities of a CC Lagrangian L(h, B) in the bare MO frame
(ccpy.gradients.density_gradient), D1 = dL/dh and Y = dL/dB, this adds the response of
the reference orbitals. L is not stationary with respect to orbital rotations, so the
conditions that fix the reference orbitals of each spin s are added with multipliers z[s],

    F[s]_ai = 0   (all occupied i, virtual a; Brillouin),
    F[s]_jc = 0   (active occupied j, frozen core c; with frozen core),
    F[s]_pq = 0   (p != q within the active occupied and within the virtual space;
                   canonical orbitals, which energies with orbital-energy denominators
                   such as CR-CC(2,3)_A need; the multipliers vanish for CCSD),

with F[s] = h[s] + sum_t J[t] - K[s] the UHF Fock operator. The multipliers are collected in
symmetric matrices Z[s] (Z[s]_pq = Z[s]_qp = z[s]_pq / 2), and the relaxed densities are

    D[s] = D1[s] + Z[s],
    Y[s] += dF/dB[s] contracted with Z (Coulomb with both spins, exchange within s),
    W[s]  = energy-weighted density (conjugate to the overlap derivative, U = -1/2 S^x).

The equations are those of UHF, with alpha and beta orbitals rotating independently. For a
closed-shell RHF reference they give the RHF response (the perturbed UHF solution of a
restricted, stable reference stays restricted), so one code serves RHF and UHF. ROHF orbitals
satisfy different conditions and are not handled.

Every quantity is held in the blocks oo, ov, vo, vv of each spin (ccpy Integral containers), "o"
running over all occupied orbitals with the frozen core first; the active occupied orbitals
are the occupied indices [nfrozen:].
"""

import numpy as np
import scipy.sparse.linalg as spla

from ccpy.gradients.density_gradient import get_bare_integrals, zero_density


def orbital_relaxation(D1, Y, reference, zvector_tol=1.0e-10, verbose=True):
    """Orbital-relaxed densities (D, Y, W) from the amplitude-relaxed (D1, Y).

    Parameters
    ----------
    D1 : blocks D1.a.oo, ... with D1_pq = dL/dh_pq, E1 = sum_pq h_pq D1_pq (each spin)
    Y : blocks Y.a.oo, ... with Y_x[p, r] = dL/dB_x[p, r]
    reference : ccpy.methods.Reference
    zvector_tol : float
        Convergence threshold of the Z-vector (conjugate-gradient) solver.
    """
    kind = type(reference.scf).__name__
    if kind not in ("RHF", "UHF"):
        raise NotImplementedError(f"Orbital relaxation is implemented for RHF and UHF references, not {kind}.")
    h, B = get_bare_integrals(reference)
    nfz = reference.nfrozen
    noa, nob = nfz + reference.system.noccupied_alpha, nfz + reference.system.noccupied_beta
    eoa, eva = reference.mo_energy["a"][:noa], reference.mo_energy["a"][noa:]
    eob, evb = reference.mo_energy["b"][:nob], reference.mo_energy["b"][nob:]
    naux = B.a.oo.shape[0]

    # only the symmetric parts are seen by the symmetric h and B_x
    D1s, Ys = zero_density(reference), zero_density(reference, naux)
    _symmetrize(D1s.a, D1.a)
    _symmetrize(D1s.b, D1.b)
    _symmetrize(Ys.a, Y.a)
    _symmetrize(Ys.b, Y.b)
    # generalized Fock F_tp = sum_q h_tq D_pq + sum_x (B_x Y_x^T)_tp; dL/dkappa_tp = g_tp
    F, g = zero_density(reference), zero_density(reference)
    _generalized_fock(F.a, g.a, h.a, B.a, D1s.a, Ys.a)
    _generalized_fock(F.b, g.b, h.b, B.b, D1s.b, Ys.b)

    # multipliers of the canonical conditions within the active occupied and the virtual spaces,
    # z_pq = -g_pq / (eps_p - eps_q) for p > q. Rotations within a degenerate pair leave such
    # energies invariant, so g_pq = 0 there.
    z_oo_a, deg_oa = _canonical_multipliers(g.a.oo[nfz:, nfz:], eoa[nfz:])
    z_vv_a, deg_va = _canonical_multipliers(g.a.vv, eva)
    z_oo_b, deg_ob = _canonical_multipliers(g.b.oo[nfz:, nfz:], eob[nfz:])
    z_vv_b, deg_vb = _canonical_multipliers(g.b.vv, evb)
    degenerate_g = max(deg_oa, deg_va, deg_ob, deg_vb)
    # frozen core / active occupied
    z_fc_a = -g.a.oo[nfz:, :nfz] / (eoa[nfz:, None] - eoa[None, :nfz])
    z_fc_b = -g.b.oo[nfz:, :nfz] / (eob[nfz:, None] - eob[None, :nfz])

    if verbose:
        g_core = max(np.abs(g.a.oo[:nfz, :nfz]).max(initial=0.0), np.abs(g.b.oo[:nfz, :nfz]).max(initial=0.0))
        print("\n   Orbital relaxation")
        print("   ------------------")
        print(f"   Orbital gradient within the frozen core (should vanish): {g_core:.2e}")
        print("   Occupied-occupied / virtual-virtual multipliers: |z| = "
              f"{np.hypot(np.linalg.norm(z_oo_a), np.linalg.norm(z_vv_a)):.3e} (a), "
              f"{np.hypot(np.linalg.norm(z_oo_b), np.linalg.norm(z_vv_b)):.3e} (b)")
    if degenerate_g > 1.0e-6:
        print(f"   WARNING: orbital gradient {degenerate_g:.2e} between degenerate orbitals is not zero; "
              "the energy is not invariant within a degenerate subspace.")

    # Z-vector for the occupied-virtual conditions of both spins:
    # (eps_a - eps_i) z_ai + R(z)_ai = -g_ai - R(z_fixed)_ai,
    # with R(z) = hf_response(z + z^T) the change of F (z: the full multipliers)
    Zfix = zero_density(reference)
    _fixed_multipliers(Zfix.a, z_oo_a, z_vv_a, z_fc_a, nfz)
    _fixed_multipliers(Zfix.b, z_oo_b, z_vv_b, z_fc_b, nfz)
    R_fixed = hf_response(Zfix, B, reference)
    rhs = np.concatenate([(-g.a.vo - R_fixed.a.vo).ravel(), (-g.b.vo - R_fixed.b.vo).ravel()])
    e_ai_a, e_ai_b = eva[:, None] - eoa[None, :], evb[:, None] - eob[None, :]
    na = e_ai_a.size

    def unpack(zflat):
        Zs = zero_density(reference)
        Zs.a.vo[...] = zflat[:na].reshape(e_ai_a.shape)
        Zs.a.ov[...] = Zs.a.vo.T
        Zs.b.vo[...] = zflat[na:].reshape(e_ai_b.shape)
        Zs.b.ov[...] = Zs.b.vo.T
        return Zs

    def A_matvec(zflat):
        Zs = unpack(zflat)
        Rz = hf_response(Zs, B, reference)
        return np.concatenate([(e_ai_a * Zs.a.vo + Rz.a.vo).ravel(), (e_ai_b * Zs.b.vo + Rz.b.vo).ravel()])

    n = rhs.size
    diag = np.concatenate([e_ai_a.ravel(), e_ai_b.ravel()])
    A = spla.LinearOperator((n, n), matvec=A_matvec, dtype=float)
    M = spla.LinearOperator((n, n), matvec=lambda x: x / diag, dtype=float)
    z_flat, info = spla.cg(A, rhs, rtol=zvector_tol, atol=0.0, M=M, maxiter=200)
    if info != 0:
        raise RuntimeError(f"The Z-vector equations did not converge (cg info = {info}).")
    z_ov_a, z_ov_b = z_flat[:na].reshape(e_ai_a.shape), z_flat[na:].reshape(e_ai_b.shape)

    # symmetric Z with Z_pq = Z_qp = z_pq / 2 for each condition
    Z = zero_density(reference)
    _relaxation_multipliers(Z.a, z_ov_a, Zfix.a)
    _relaxation_multipliers(Z.b, z_ov_b, Zfix.b)

    # relaxed densities: D = D1 + Z; Y of the Fock conditions sum_t Z[t].F[t],
    # Z[s]_pr P[t]_qs (all t) + P[s]_pr Z[t]_qs (all t) - Z[s]_ps P[s]_qr - P[s]_ps Z[s]_qr
    # with P the occupied projector
    trP = np.einsum("xqq->x", B.a.oo) + np.einsum("xqq->x", B.b.oo)
    trZ = _trace(Z.a, B.a) + _trace(Z.b, B.b)
    D = zero_density(reference)
    _add(D.a, D1s.a, Z.a)
    _add(D.b, D1s.b, Z.b)
    _relaxed_Y(Ys.a, Z.a, B.a, trP, trZ)
    _relaxed_Y(Ys.b, Z.b, B.b, trP, trZ)

    # energy-weighted densities
    RZ = hf_response(Z, B, reference)
    W = zero_density(reference)
    _energy_weighted(W.a, F.a, Z.a, RZ.a, eoa, eva)
    _energy_weighted(W.b, F.b, Z.b, RZ.b, eob, evb)

    if verbose:
        print(f"   Z-vector converged; |z_ov| = {np.linalg.norm(z_ov_a):.3e} (a), {np.linalg.norm(z_ov_b):.3e} (b)"
              f"; |z_core| = {np.linalg.norm(z_fc_a):.3e} (a), {np.linalg.norm(z_fc_b):.3e} (b)")
    return D, Ys, W


def hf_response(X, B, reference):
    """Change of the UHF Fock operators, sum_t J[X[t]] - K[X[s]], for symmetric block matrices
    X (a change of the occupied density of each spin)."""
    rho = _trace(X.a, B.a) + _trace(X.b, B.b)
    out = zero_density(reference)
    _jk(out.a, X.a, B.a, rho)
    _jk(out.b, X.b, B.b, rho)
    return out


# ======================================================================================
# blocks of one spin
# ======================================================================================
def _symmetrize(out, M):
    """Symmetric part, (M + M^T) / 2, of blocks M (Y-type blocks transpose their last two axes)."""
    out.oo[...] = 0.5 * (M.oo + np.swapaxes(M.oo, -1, -2))
    out.ov[...] = 0.5 * (M.ov + np.swapaxes(M.vo, -1, -2))
    out.vo[...] = np.swapaxes(out.ov, -1, -2)
    out.vv[...] = 0.5 * (M.vv + np.swapaxes(M.vv, -1, -2))


def _generalized_fock(F, g, h, B, D, Y):
    """F_tp = sum_q h_tq D_pq + sum_xq B_x[t, q] Y_x[p, q]; g = 2 (F - F^T)."""
    F.oo[...] = _fock_block(h.oo, h.ov, D.oo, D.ov, B.oo, B.ov, Y.oo, Y.ov)
    F.ov[...] = _fock_block(h.oo, h.ov, D.vo, D.vv, B.oo, B.ov, Y.vo, Y.vv)
    F.vo[...] = _fock_block(h.vo, h.vv, D.oo, D.ov, B.vo, B.vv, Y.oo, Y.ov)
    F.vv[...] = _fock_block(h.vo, h.vv, D.vo, D.vv, B.vo, B.vv, Y.vo, Y.vv)
    g.oo[...] = 2.0 * (F.oo - F.oo.T)
    g.vo[...] = 2.0 * (F.vo - F.ov.T)
    g.ov[...] = -g.vo.T
    g.vv[...] = 2.0 * (F.vv - F.vv.T)


def _fock_block(h_to, h_tv, D_po, D_pv, B_to, B_tv, Y_po, Y_pv):
    return (h_to @ D_po.T + h_tv @ D_pv.T + np.einsum("xtq,xpq->tp", B_to, Y_po, optimize=True)
            + np.einsum("xtq,xpq->tp", B_tv, Y_pv, optimize=True))


def _canonical_multipliers(g, eps):
    """z_pq = -g_pq / (eps_p - eps_q) for p > q within one space (zero for degenerate pairs),
    and the largest |g_pq| between degenerate orbitals."""
    de = eps[:, None] - eps[None, :]
    lower = np.tril(np.ones_like(de, dtype=bool), -1)
    nondeg = lower & (np.abs(de) > 1.0e-8)
    z = np.zeros_like(de)
    z[nondeg] = -g[nondeg] / de[nondeg]
    return z, np.abs(g[lower & ~nondeg]).max(initial=0.0)


def _fixed_multipliers(Z, z_oo, z_vv, z_fc, nfz):
    """z + z^T of the canonical (active occupied, virtual) and frozen-core multipliers."""
    Z.oo[nfz:, nfz:] = z_oo
    Z.oo[nfz:, :nfz] = z_fc
    Z.oo[...] = Z.oo + Z.oo.T
    Z.vv[...] = z_vv + z_vv.T


def _relaxation_multipliers(Z, z_ov, Zfix):
    """Z with Z_pq = Z_qp = z_pq / 2 for every condition."""
    Z.vo[...] = 0.5 * z_ov
    Z.ov[...] = Z.vo.T
    Z.oo[...] = 0.5 * Zfix.oo
    Z.vv[...] = 0.5 * Zfix.vv


def _jk(R, X, B, rho):
    """R = J - K: J_pq = B_x[p, q] rho_x, K_pq = sum_x (B_x X B_x)_pq."""
    BX_oo = B.oo @ X.oo + B.ov @ X.vo
    BX_ov = B.oo @ X.ov + B.ov @ X.vv
    BX_vo = B.vo @ X.oo + B.vv @ X.vo
    BX_vv = B.vo @ X.ov + B.vv @ X.vv
    R.oo[...] = np.einsum("xpq,x->pq", B.oo, rho, optimize=True) - np.einsum("xps,xsq->pq", BX_oo, B.oo, optimize=True) \
        - np.einsum("xps,xsq->pq", BX_ov, B.vo, optimize=True)
    R.ov[...] = np.einsum("xpq,x->pq", B.ov, rho, optimize=True) - np.einsum("xps,xsq->pq", BX_oo, B.ov, optimize=True) \
        - np.einsum("xps,xsq->pq", BX_ov, B.vv, optimize=True)
    R.vo[...] = np.einsum("xpq,x->pq", B.vo, rho, optimize=True) - np.einsum("xps,xsq->pq", BX_vo, B.oo, optimize=True) \
        - np.einsum("xps,xsq->pq", BX_vv, B.vo, optimize=True)
    R.vv[...] = np.einsum("xpq,x->pq", B.vv, rho, optimize=True) - np.einsum("xps,xsq->pq", BX_vo, B.ov, optimize=True) \
        - np.einsum("xps,xsq->pq", BX_vv, B.vv, optimize=True)


def _trace(M, B):
    """sum_pq M_pq B_x[p, q] over the four blocks."""
    return (np.einsum("pq,xpq->x", M.oo, B.oo, optimize=True) + np.einsum("pq,xpq->x", M.ov, B.ov, optimize=True)
            + np.einsum("pq,xpq->x", M.vo, B.vo, optimize=True) + np.einsum("pq,xpq->x", M.vv, B.vv, optimize=True))


def _add(out, M, N):
    out.oo[...] = M.oo + N.oo
    out.ov[...] = M.ov + N.ov
    out.vo[...] = M.vo + N.vo
    out.vv[...] = M.vv + N.vv


def _relaxed_Y(Y, Z, B, trP, trZ):
    """Y_x += Z trP_x + P trZ_x - Z B_x P - P B_x Z for the occupied projector P."""
    Y.oo += Z.oo[None] * trP[:, None, None] + np.eye(Y.oo.shape[1])[None] * trZ[:, None, None]
    Y.ov += Z.ov[None] * trP[:, None, None]
    Y.vo += Z.vo[None] * trP[:, None, None]
    Y.vv += Z.vv[None] * trP[:, None, None]
    Y.oo -= Z.oo @ B.oo + Z.ov @ B.vo + B.oo @ Z.oo + B.ov @ Z.vo
    Y.vo -= Z.vo @ B.oo + Z.vv @ B.vo
    Y.ov -= B.oo @ Z.ov + B.ov @ Z.vv


def _energy_weighted(W, F, Z, RZ, eo, ev):
    """W = (F + F^T) / 2 + Z (eps_p + eps_q) / 2 + (P_pp + P_qq) RZ / 2."""
    W.oo[...] = 0.5 * (F.oo + F.oo.T) + 0.5 * Z.oo * (eo[:, None] + eo[None, :]) + RZ.oo
    W.ov[...] = 0.5 * (F.ov + F.vo.T) + 0.5 * Z.ov * (eo[:, None] + ev[None, :]) + 0.5 * RZ.ov
    W.vo[...] = 0.5 * (F.vo + F.ov.T) + 0.5 * Z.vo * (ev[:, None] + eo[None, :]) + 0.5 * RZ.vo
    W.vv[...] = 0.5 * (F.vv + F.vv.T) + 0.5 * Z.vv * (ev[:, None] + ev[None, :])
