"""Orbital response of a canonical RHF reference with full four-center integrals
(Sec. 8 of ccpq_gradients_revised.pdf).

Given the total amplitude-relaxed spatial densities (D, G) of F = L_P + dL, the canonical
RHF conditions c_alpha = f_pq = 0 are added with multipliers zeta (Eq. 8.3):

    F_tp  = sum_q h_tq D_pq + sum_qrs (tr|qs) G_pqrs,   g_tp = 2 (F_tp - F_pt)       (8.7)
    zeta_pq = -g_pq / (eps_p - eps_q)       active-occ, virtual, and core/active-occ pairs (8.9)
    (eps_a - eps_i) zeta_ai + G(S(zeta))_ai = -g_ai - G(S_int)_ai                      (8.10)
    D^rel = D + Z,   G^rel = G + 2(Z_pr P_qs + P_pr Z_qs) - (Z_ps P_qr + P_ps Z_qr)     (8.12-8.13)
    W     = (F + F^T)/2 + (eps Z + Z eps)/2 + [P G(Z) + G(Z) P]/2                     (8.17)

with G(X)_pq = sum_rs [2(pq|rs) - (pr|qs)] X_rs and Z the symmetric half-multiplier matrix.
"""

from dataclasses import dataclass

import numpy as np

from ccpy.gradients.ccpq.densities import SpatialDensities


@dataclass
class RelaxedDensities:
    D: np.ndarray
    G: np.ndarray
    W: np.ndarray
    Z: np.ndarray
    F: np.ndarray
    g: np.ndarray


def fock_response(X, eri):
    """G(X)_pq = sum_rs [2 (pq|rs) - (pr|qs)] X_rs, eri[p, q, r, s] = (pq|rs)."""
    return 2.0 * np.einsum("pqrs,rs->pq", eri, X, optimize=True) - np.einsum("prqs,rs->pq", eri, X, optimize=True)


def orbital_relaxation(rho, h, eri, eps, nocc, nfrozen=0, degeneracy_tol=1.0e-8):
    """Relaxed (D, G, W) from amplitude-relaxed spatial densities `rho` (SpatialDensities).

    h, eri : MO core Hamiltonian and (pq|rs); eps : canonical orbital energies;
    nocc : doubly occupied orbitals (frozen core included); nfrozen : frozen-core orbitals.
    """
    rho = rho.symmetrized()
    D, G = rho.D, rho.G
    nmo = len(eps)
    occ, vir = np.arange(nocc), np.arange(nocc, nmo)
    act, core = np.arange(nfrozen, nocc), np.arange(nfrozen)

    F = h @ D.T + np.einsum("trqs,pqrs->tp", eri, G, optimize=True)
    g = 2.0 * (F - F.T)

    # internal multipliers (Eq. 8.9) as a full symmetric matrix S_int
    S_int = np.zeros((nmo, nmo))
    for block in (act, vir):
        for x, p in enumerate(block):
            for q in block[:x]:
                de = eps[p] - eps[q]
                if abs(de) < degeneracy_tol:
                    if abs(g[p, q]) > 1.0e-6:
                        raise RuntimeError(f"Degenerate canonical pair ({p}, {q}) with orbital gradient {g[p, q]:.2e}.")
                    continue
                S_int[p, q] = S_int[q, p] = -g[p, q] / de
    for p in act:
        for c in core:
            S_int[p, c] = S_int[c, p] = -g[p, c] / (eps[p] - eps[c])

    # occupied-virtual Z-vector (Eq. 8.10), solved directly
    nv, no = len(vir), len(occ)
    A = np.zeros((nv * no, nv * no))
    for x in range(nv * no):
        a, i = divmod(x, no)
        S = np.zeros((nmo, nmo))
        S[vir[a], occ[i]] = S[occ[i], vir[a]] = 1.0
        A[:, x] = fock_response(S, eri)[np.ix_(vir, occ)].ravel()
    A += np.diag((eps[vir][:, None] - eps[occ][None, :]).ravel())
    rhs = -g[np.ix_(vir, occ)] - fock_response(S_int, eri)[np.ix_(vir, occ)]
    zeta_ov = np.linalg.solve(A, rhs.ravel()).reshape(nv, no)

    S_full = S_int.copy()
    S_full[np.ix_(vir, occ)] = zeta_ov
    S_full[np.ix_(occ, vir)] = zeta_ov.T
    Z = 0.5 * S_full

    P = np.zeros((nmo, nmo))
    P[occ, occ] = 1.0
    D_rel = D + Z
    dG = (2.0 * (np.einsum("pr,qs->pqrs", Z, P) + np.einsum("pr,qs->pqrs", P, Z))
          - (np.einsum("ps,qr->pqrs", Z, P) + np.einsum("ps,qr->pqrs", P, Z)))
    G_rel = G + dG
    GZ = fock_response(Z, eri)
    E = np.diag(eps)
    W = 0.5 * (F + F.T) + 0.5 * (E @ Z + Z @ E) + 0.5 * (P @ GZ + GZ @ P)
    return RelaxedDensities(D_rel, G_rel, W, Z, F, g)


def relaxed_energy_check(rel, h, eri):
    """sum D^rel h + 1/2 sum G^rel (pr|qs), for comparisons only."""
    return SpatialDensities(rel.D, rel.G).energy(h, eri)
