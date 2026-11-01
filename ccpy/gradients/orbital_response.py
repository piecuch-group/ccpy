"""Orbital relaxation (Z-vector / CPHF) of amplitude-relaxed CC densities, per spin.

Given the amplitude-relaxed densities of a CC Lagrangian L(h, B) in the bare MO frame
(ccpy.gradients.mo_density), D1[s] = dL/dh[s] and Y[s] = dL/dB[s], this adds the response of
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
"""

import numpy as np
import scipy.sparse.linalg as spla

from ccpy.gradients.mo_density import SPINS


def hf_response(X, B):
    """Change of the UHF Fock operators, sum_t J[X[t]] - K[X[s]], for symmetric MO matrices X[s]
    (a change of the occupied density of each spin)."""
    rho = sum(np.einsum("xrs,rs->x", B[s], X[s], optimize=True) for s in SPINS)
    out = {}
    for s in SPINS:
        J = np.einsum("xpq,x->pq", B[s], rho, optimize=True)
        K = np.einsum("xps,xsq->pq", np.einsum("xpr,rs->xps", B[s], X[s], optimize=True), B[s], optimize=True)
        out[s] = J - K
    return out


def orbital_relaxation(D1, Y, mo, zvector_tol=1.0e-10, verbose=True):
    """Orbital-relaxed densities (D, Y, W), per spin, from the amplitude-relaxed (D1, Y).

    Parameters
    ----------
    D1 : {spin: (nmo, nmo)}
        D1[s][p, q] = dL/dh[s][p, q], E1 = sum_s sum_pq h[s]_pq D1[s]_pq.
    Y : {spin: (naux, nmo, nmo)}
        Y[s]_x[p, r] = dL/dB[s]_x[p, r].
    mo : ccpy.gradients.mo_density.get_mo_integrals
    zvector_tol : float
        Convergence threshold of the Z-vector (conjugate-gradient) solver.
    """
    if mo["reference"] not in ("RHF", "UHF"):
        raise NotImplementedError(f"Orbital relaxation is implemented for RHF and UHF references, not {mo['reference']}.")
    nmo, nfz = mo["nmo"], mo["nfrozen"]
    B, h, eps, P = mo["B"], mo["h"], mo["eps"], mo["P"]
    occ = {s: slice(0, mo["nocc"][s]) for s in SPINS}
    vir = {s: slice(mo["nocc"][s], nmo) for s in SPINS}
    act = {s: slice(nfz, mo["nocc"][s]) for s in SPINS}
    core = slice(0, nfz)

    # only the symmetric parts are seen by the symmetric h and B_x
    D1 = {s: 0.5 * (D1[s] + D1[s].T) for s in SPINS}
    Y = {s: 0.5 * (Y[s] + Y[s].transpose(0, 2, 1)) for s in SPINS}
    # generalized Fock F[s]_tp = sum_q h_tq D_pq + sum_x (B_x Y_x^T)_tp; dL/dkappa[s]_tp = g[s]_tp
    F = {s: h[s] @ D1[s] + np.einsum("xtq,xpq->tp", B[s], Y[s], optimize=True) for s in SPINS}
    g = {s: 2.0 * (F[s] - F[s].T) for s in SPINS}

    # multipliers of the canonical conditions within the active occupied and the virtual spaces,
    # z_pq = -g_pq / (eps_p - eps_q) for p > q. Rotations within a degenerate pair leave such
    # energies invariant, so g_pq = 0 there.
    z_int = {s: np.zeros((nmo, nmo)) for s in SPINS}
    degenerate_g = 0.0
    for s in SPINS:
        for space in (act[s], vir[s]):
            idx = np.arange(nmo)[space]
            de = eps[s][idx, None] - eps[s][None, idx]
            lower = np.tril(np.ones_like(de, dtype=bool), -1)
            nondeg = lower & (np.abs(de) > 1.0e-8)
            block = np.zeros_like(de)
            block[nondeg] = -g[s][np.ix_(idx, idx)][nondeg] / de[nondeg]
            z_int[s][np.ix_(idx, idx)] = block
            degenerate_g = max(degenerate_g, np.abs(g[s][np.ix_(idx, idx)][lower & ~nondeg]).max(initial=0.0))

    # frozen core / active occupied
    z_fc = {s: -g[s][act[s], core] / (eps[s][act[s], None] - eps[s][None, core]) for s in SPINS}

    if verbose:
        g_core = max(np.abs(g[s][core, core]).max(initial=0.0) for s in SPINS)
        print("\n   Orbital relaxation")
        print("   ------------------")
        print(f"   Orbital gradient within the frozen core (should vanish): {g_core:.2e}")
        print("   Occupied-occupied / virtual-virtual multipliers: |z| = "
              + ", ".join(f"{np.linalg.norm(z_int[s]):.3e} ({s})" for s in SPINS))
    if degenerate_g > 1.0e-6:
        print(f"   WARNING: orbital gradient {degenerate_g:.2e} between degenerate orbitals is not zero; "
              "the energy is not invariant within a degenerate subspace.")

    # Z-vector for the occupied-virtual conditions of both spins:
    # (eps_a - eps_i) z[s]_ai + R(z)[s]_ai = -g[s]_ai - R(z_fixed)[s]_ai,
    # with R(z)[s] = hf_response(z + z^T) the change of F[s] (z: the full multipliers)
    Ws = {}
    for s in SPINS:
        Ws[s] = z_int[s].copy()
        Ws[s][act[s], core] = z_fc[s]
        Ws[s] += Ws[s].T
    R_fixed = hf_response(Ws, B)
    rhs = np.concatenate([(-g[s][vir[s], occ[s]] - R_fixed[s][vir[s], occ[s]]).ravel() for s in SPINS])

    eps_ai = {s: eps[s][vir[s], None] - eps[s][None, occ[s]] for s in SPINS}
    shapes = [eps_ai[s].shape for s in SPINS]
    sizes = [eps_ai[s].size for s in SPINS]

    def unpack(zflat):
        z = {"a": zflat[:sizes[0]].reshape(shapes[0]), "b": zflat[sizes[0]:].reshape(shapes[1])}
        Zs = {}
        for s in SPINS:
            Zs[s] = np.zeros((nmo, nmo))
            Zs[s][vir[s], occ[s]] = z[s]
            Zs[s] += Zs[s].T
        return z, Zs

    def A_matvec(zflat):
        z, Zs = unpack(zflat)
        Rz = hf_response(Zs, B)
        return np.concatenate([(eps_ai[s] * z[s] + Rz[s][vir[s], occ[s]]).ravel() for s in SPINS])

    n = sum(sizes)
    diag = np.concatenate([eps_ai[s].ravel() for s in SPINS])
    A = spla.LinearOperator((n, n), matvec=A_matvec, dtype=float)
    M = spla.LinearOperator((n, n), matvec=lambda x: x / diag, dtype=float)
    z_flat, info = spla.cg(A, rhs, rtol=zvector_tol, atol=0.0, M=M, maxiter=200)
    if info != 0:
        raise RuntimeError(f"The Z-vector equations did not converge (cg info = {info}).")
    z_ov, _ = unpack(z_flat)

    # symmetric Z[s] with Z_pq = Z_qp = z_pq / 2 for each condition
    Z = {}
    for s in SPINS:
        Z[s] = np.zeros((nmo, nmo))
        Z[s][vir[s], occ[s]] = 0.5 * z_ov[s]
        Z[s][act[s], core] = 0.5 * z_fc[s]
        Z[s] += 0.5 * z_int[s]
        Z[s] += Z[s].T

    # relaxed densities: D[s] = D1[s] + Z[s]; Y[s] of the Fock conditions sum_t Z[t].F[t],
    # Z[s]_pr P[t]_qs (all t) + P[s]_pr Z[t]_qs (all t) - Z[s]_ps P[s]_qr - P[s]_ps Z[s]_qr
    trP = sum(np.einsum("xqq->x", B[s][:, occ[s], occ[s]]) for s in SPINS)
    trZ = sum(np.einsum("qs,xqs->x", Z[s], B[s], optimize=True) for s in SPINS)
    D = {s: D1[s] + Z[s] for s in SPINS}
    Y = {s: Y[s] + np.einsum("pr,x->xpr", Z[s], trP, optimize=True)
         + np.einsum("pr,x->xpr", P[s], trZ, optimize=True)
         - Z[s] @ B[s] @ P[s] - P[s] @ B[s] @ Z[s] for s in SPINS}

    # energy-weighted densities
    RZ = hf_response(Z, B)
    W = {}
    for s in SPINS:
        occ_mask = np.diag(P[s])[:, None] + np.diag(P[s])[None, :]  # 2 occ-occ, 1 occ-vir
        W[s] = (0.5 * (F[s] + F[s].T) + Z[s] * 0.5 * (eps[s][:, None] + eps[s][None, :])
                + 0.5 * occ_mask * RZ[s])

    if verbose:
        print("   Z-vector converged; |z_ov| = "
              + ", ".join(f"{np.linalg.norm(z_ov[s]):.3e} ({s})" for s in SPINS)
              + "; |z_core| = " + ", ".join(f"{np.linalg.norm(z_fc[s]):.3e} ({s})" for s in SPINS))
    return D, Y, W
