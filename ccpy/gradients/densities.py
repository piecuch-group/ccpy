"""Unrelaxed densities for the analytic gradients, as plain arrays over all MOs (frozen core
included), spin-traced (closed shell), in physicists' order and conjugate to the bare h and
<pq|rs>:

    E = V_nn + sum_pq h_pq D1[p, q] + 1/2 sum_pqrs <pq|rs> D2[p, q, r, s],

    D1[p, q] = sum_sigma <p_sigma^+ q_sigma>,
    D2[p, q, r, s] = sum_{sigma, tau} <p_sigma^+ q_tau^+ s_tau r_sigma>.

They are assembled from the normal-ordered densities of ccpy.density (rdm1, rdm2), which are
conjugate to the Fock matrix f and the antisymmetrized <pq||rs> of the normal-ordered
Hamiltonian, E_corr = sum f gamma^N + 1/4 sum <pq||rs> Gamma^N. The conversion uses
f_pq = h_pq + sum_i <pi||qi> (i over all occupied spin orbitals, frozen core included).
"""

import numpy as np

# the 8 index permutations that leave <pq|rs> unchanged for real orbitals
SYMMETRY_PERMUTATIONS = [
    (0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0),
    (2, 1, 0, 3), (0, 3, 2, 1), (1, 2, 3, 0), (3, 0, 1, 2),
]


def symmetrize(D2):
    """The 8-fold <pq|rs> symmetrization, which is all that real integrals see."""
    return sum(D2.transpose(perm) for perm in SYMMETRY_PERMUTATIONS) / 8.0


def separable(Z, nocc):
    """Two-body density of sum_pq Z_pq [2 <pi|qi> - <pi|iq>] (i occupied) for a symmetric
    MO matrix Z: 2 (Z_pr P_qs + P_pr Z_qs) - (Z_ps P_qr + P_ps Z_qr), P the occupied projector."""
    P = np.zeros_like(Z)
    P[:nocc, :nocc] = np.eye(nocc)
    return (2.0 * np.einsum("pr,qs->pqrs", Z, P) + 2.0 * np.einsum("pr,qs->pqrs", P, Z)
            - np.einsum("ps,qr->pqrs", Z, P) - np.einsum("ps,qr->pqrs", P, Z))


def hf_densities(nocc, nmo):
    """D1 = 2 P and D2 = 4 d_pr d_qs - 2 d_ps d_qr (occupied) of the closed-shell determinant."""
    D1 = np.zeros((nmo, nmo))
    D1[:nocc, :nocc] = 2.0 * np.eye(nocc)
    P = np.eye(nocc)
    D2 = np.zeros((nmo,) * 4)
    D2[:nocc, :nocc, :nocc, :nocc] = 4.0 * np.einsum("pr,qs->pqrs", P, P) - 2.0 * np.einsum("ps,qr->pqrs", P, P)
    return D1, D2


def spin_integrated_to_spin_traced(G_aa, G_ab, G_bb):
    """Spin-traced D2N[p,q,r,s] = sum_{sigma,tau} Gamma_{p sigma, q tau, r sigma, s tau} from
    spin-integrated Gamma[p,q,r,s] conjugate to <pq||rs> (aa, bb, with 1/4) and <pq|rs> (ab)."""
    return G_aa + G_bb + G_ab + G_ab.transpose(1, 0, 3, 2)


def normal_ordered_correction(gamma_s, D2N, nfrozen, nocc, nmo):
    """(D1, D2) of a normal-ordered correlation contribution.

    gamma_s : (n, n) spin-traced normal-ordered one-body density over the correlated orbitals,
              gamma_s[p, q] conjugate to f_pq.
    D2N : (n,)*4 spin-traced normal-ordered two-body density over the correlated orbitals,
          D2N[p,q,r,s] = sum_{sigma,tau} <{p_sigma^+ q_tau^+ s_tau r_sigma}>.
    Embeds the frozen core and adds the f -> h + sum_i <pi||qi> separable terms."""
    n = gamma_s.shape[0]
    corr = slice(nfrozen, nfrozen + n)
    D1 = np.zeros((nmo, nmo))
    D1[corr, corr] = gamma_s
    D2 = np.zeros((nmo,) * 4)
    D2[corr, corr, corr, corr] = D2N
    D2 += separable(0.5 * (D1 + D1.T), nocc)
    return D1, D2


def ccsd_densities(T, L, system, nmo):
    """(D1, D2) of the closed-shell CCSD Lagrangian from ccpy.density.rdm1 / rdm2, HF part
    and frozen core included: V_nn + h.D1 + 1/2 <pq|rs>.D2 is the CCSD energy."""
    from ccpy.density.rdm1 import calc_rdm1
    from ccpy.density.rdm2 import calc_rdm2_normal_order, calc_rdm2_vvvv_slice

    nfz, n, no = system.nfrozen, system.norbitals, system.noccupied_alpha
    if no != system.noccupied_beta:
        raise NotImplementedError("Closed-shell (RHF) densities only.")
    sl = {"o": slice(0, no), "v": slice(no, n)}

    # one body: rdm1.a[p, q] = <p+ q> (HF included) -> normal ordered, spin traced
    rdm1 = calc_rdm1(T, L, system)
    gamma = np.zeros((n, n))
    for b in ("oo", "ov", "vo", "vv"):
        gamma[sl[b[0]], sl[b[1]]] = getattr(rdm1.a, b)
    gamma_s = 2.0 * (gamma - np.diag([1.0] * no + [0.0] * (n - no)))

    # two body: rdm2.xx.<block>[p,q,r,s] = <{r+ s+ q p}>, so Gamma[p,q,r,s] = <{p+ q+ s r}>
    # (conjugate to <pq||rs>) is its transpose (2, 3, 0, 1). The stored aa blocks are
    # completed by antisymmetry; vvvv is built slice by slice.
    rdm2 = calc_rdm2_normal_order(T, L, system, flag_RHF=True)
    R_aa, R_ab = np.zeros((n,) * 4), np.zeros((n,) * 4)
    for key in rdm2.ab.slices:
        idx = tuple(sl[c] for c in key)
        arr = getattr(rdm2.aa, key)
        if arr is not None and not np.isscalar(arr):
            R_aa[idx] = arr
            R_aa[idx[1], idx[0], idx[2], idx[3]] = -arr.transpose(1, 0, 2, 3)
            R_aa[idx[0], idx[1], idx[3], idx[2]] = -arr.transpose(0, 1, 3, 2)
            R_aa[idx[1], idx[0], idx[3], idx[2]] = arr.transpose(1, 0, 3, 2)
        arr = getattr(rdm2.ab, key)
        if arr is not None and not np.isscalar(arr):
            R_ab[idx] = arr
    for a in range(n - no):
        R_aa[no + a, no:, no:, no:], R_ab[no + a, no:, no:, no:] = calc_rdm2_vvvv_slice(T, L, a)
    G_aa, G_ab = R_aa.transpose(2, 3, 0, 1), R_ab.transpose(2, 3, 0, 1)
    D2N = spin_integrated_to_spin_traced(G_aa, G_ab, G_aa)

    D1c, D2c = normal_ordered_correction(gamma_s, D2N, nfz, nfz + no, nmo)
    D1h, D2h = hf_densities(nfz + no, nmo)
    return D1h + D1c, D2h + D2c
