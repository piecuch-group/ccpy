"""Bare-integral 1- and 2-body derivative densities of the base and correction functionals
(Secs. 2.3, 5 and 6 of ccpq_gradients_revised.pdf) and their spin-traced spatial form
(Sec. 8.2).

Spin-orbital convention (Eq. 1.5): dE = sum_pq gamma_pq dh_pq + 1/4 sum_pqrs Gamma_pqrs dv_pqrs,
with gamma ~ a+_p a_q and Gamma ~ a+_p a+_q a_s a_r. Spatial convention (Eq. 8.6):
dE = sum_pq D_pq dh_pq + 1/2 sum_pqrs G_pqrs d(pr|qs).

Each coefficient matrix A of amplitudes.py gives rho[O] = sum_AB K_AB O_AB with
K = U^T A V^T (Eq. 6.7), so the projections are evaluated on the bare operators, and the
reference and frozen core come out of the determinant algebra directly (no separate
normal-ordering step, Sec. 5.3). The bare-Fock denominator adds the tensors of Eqs. (5.7)-(5.8).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SpinOrbitalDensities:
    gamma: np.ndarray   # (nso, nso)
    Gamma: np.ndarray   # (nso,)*4, antisymmetric

    def __add__(self, other):
        return SpinOrbitalDensities(self.gamma + other.gamma, self.Gamma + other.Gamma)

    def energy(self, h_so, v_so):
        return np.sum(self.gamma * h_so) + 0.25 * np.sum(self.Gamma * v_so)


@dataclass
class SpatialDensities:
    D: np.ndarray       # (norb, norb)
    G: np.ndarray       # (norb,)*4, pairs with (pr|qs)

    def __add__(self, other):
        return SpatialDensities(self.D + other.D, self.G + other.G)

    def energy(self, h, eri):
        """sum D h + 1/2 sum G_pqrs (pr|qs), with chemists' eri[a, b, c, d] = (ab|cd)."""
        return np.sum(self.D * h) + 0.5 * np.einsum("pqrs,prqs->", self.G, eri, optimize=True)

    def symmetrized(self):
        """Representatives seen by real integrals: D <- (D + D^T)/2 and G averaged over the
        8-fold symmetry of (pr|qs): p<->r, q<->s, (pr)<->(qs)."""
        D = 0.5 * (self.D + self.D.T)
        G = self.G
        G = 0.5 * (G + G.transpose(2, 1, 0, 3))   # p <-> r
        G = 0.5 * (G + G.transpose(0, 3, 2, 1))   # q <-> s
        G = 0.5 * (G + G.transpose(1, 0, 3, 2))   # (pr) <-> (qs)
        return SpatialDensities(D, G)


def transition_densities(K, space, one, two):
    """gamma_pq = sum_AB K_AB <A|a+_p a_q|B> and Gamma_pqrs likewise (Sec. 6.3)."""
    nso = space.nso
    gamma = one.contract(K).reshape(nso, nso)
    upper = two.contract(K).reshape(nso, nso, nso, nso)   # filled for p<q, r<s only
    Gamma = upper - upper.transpose(1, 0, 2, 3) - upper.transpose(0, 1, 3, 2) + upper.transpose(1, 0, 3, 2)
    return SpinOrbitalDensities(gamma, Gamma)


def coefficient_densities(A, U, V, space, one, two):
    """Densities of a functional sum_AB A_AB Hbar_AB, Hbar = U H V (Eqs. 6.7-6.8)."""
    K = U.T @ A.dense() @ V.T
    return transition_densities(K, space, one, two)


def mp_denominator_densities(d, ref_occ):
    """Bare tensors of the bare-Fock denominator, Eqs. (5.7)-(5.8): gamma^D = diag(d) and
    Gamma^D_pqrs = d_pr n_qs - d_ps n_qr - d_qr n_ps + d_qs n_pr."""
    dm = np.diag(d)
    nm = np.diag(ref_occ.astype(float))
    Gamma = (np.einsum("pr,qs->pqrs", dm, nm) - np.einsum("ps,qr->pqrs", dm, nm)
             - np.einsum("qr,ps->pqrs", dm, nm) + np.einsum("qs,pr->pqrs", dm, nm))
    return SpinOrbitalDensities(dm, Gamma)


def spin_trace(rho, norb):
    """Eq. (8.5): D_pq = sum_s gamma_{ps,qs}, G_pqrs = sum_{st} Gamma_{ps,qt,rs,st}."""
    g, G4 = rho.gamma, rho.Gamma
    a, b = slice(0, norb), slice(norb, 2 * norb)
    D = g[a, a] + g[b, b]
    G = np.zeros((norb,) * 4)
    for s in (a, b):
        for t in (a, b):
            G += G4[s, t, s, t]
    return SpatialDensities(D, G)
