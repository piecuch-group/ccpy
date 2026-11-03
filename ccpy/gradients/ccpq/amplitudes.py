"""The two separate stationary functionals of a CC(P;Q) energy and their amplitude
response (Secs. 2-4 and 6.1 of ccpq_gradients_revised.pdf), in an exact determinant space.

    L_P   = <Phi|(1 + Lambda_grad) Hbar|Phi>                                   (2.2)
    dL    = <Phi|Pi Hbar|Phi> + <Phi|(1 + Lambda)(Hbar - E) Sigma|Phi> + C(P;Q)  (3.9)
    C     = sum_K [ l_K M_K + r_K N_K - l_K r_K D_K ]                           (3.6)

Both functionals are linear in Hbar_AB = <A|Hbar|B> at fixed multipliers. Each is therefore
kept as a coefficient matrix A with L = sum_AB A_AB Hbar_AB (+ the bare-Fock denominator
term), Eq. (6.1), and every amplitude source is the derivative of such a pairing:

    d/dt_I sum_AB A_AB Hbar_AB = sum_AB A_AB H^I_AB,   H^I = Hbar E_I - E_I Hbar   (1.9)
                               = sum_{<C|E_I|B> != 0} <C|E_I|B> X_BC,  X = A^T Hbar - Hbar A^T.

The determinant P and Q lists are fixed (Sec. 9.1, step 1). Nothing here assumes P to be
closed under de-excitation; the base multiplier Lambda_grad (J^T lambda_grad = -e) and the
projected left state Lambda (B^T lambda = -Hbar_0P) are solved separately (Sec. 2.2).
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from .spaces import validate_indices


class CoefficientMatrix:
    """A = sum_k u_k v_k^T + diag(w): the coefficient matrices of Eqs. (6.2)-(6.4)."""

    def __init__(self, n, terms=(), w=None):
        self.n = n
        self.terms = list(terms)
        self.w = np.zeros(n) if w is None else np.asarray(w, dtype=float)

    def __add__(self, other):
        return CoefficientMatrix(self.n, self.terms + other.terms, self.w + other.w)

    def dense(self):
        A = np.diag(self.w)
        for u, v in self.terms:
            A += np.outer(u, v)
        return A

    def pair(self, M):
        """sum_AB A_AB M_AB."""
        return sum(u @ M @ v for u, v in self.terms) + self.w @ np.diag(M)

    def t_gradient(self, Hbar, exc):
        """sum_AB A_AB H^I_AB for every excitation I of the Transitions `exc` (Eq. 1.9)."""
        g = np.zeros(exc.nlabel)
        for u, v in self.terms:
            # X = (Hbar^T u) v^T - u (Hbar v)^T
            g += exc.contract_outer(Hbar.T @ u, v) - exc.contract_outer(u, Hbar @ v)
        if np.any(self.w):
            # X_BC = Hbar_CB (w_B - w_C) at (dst C, src B) -> Hbar[src, dst] (w[src] - w[dst])
            g += np.bincount(exc.label, weights=exc.sign * Hbar[exc.src, exc.dst] * (self.w[exc.src] - self.w[exc.dst]),
                             minlength=exc.nlabel)
        return g


def similarity_transform(H, T, max_rank):
    """U = e^{-T}, V = e^{T} as finite series (T is nilpotent, Eq. 6.6) and Hbar = U H V."""
    n = H.shape[0]
    V = np.eye(n, dtype=np.result_type(H,T))
    U = np.eye(n, dtype=np.result_type(H,T))
    term_v = np.eye(n, dtype=np.result_type(H,T))
    term_u = np.eye(n, dtype=np.result_type(H,T))
    for k in range(1, max_rank + 1):
        term_v = T @ term_v / k
        term_u = -(T @ term_u) / k
        if not np.any(term_v):
            break
        V += term_v
        U += term_u
    return U, V, U @ H @ V


@dataclass
class AmplitudeResult:
    """Everything the density stage needs, plus the checks of Sec. 9.1."""
    t: np.ndarray
    U: np.ndarray
    V: np.ndarray
    Hbar: np.ndarray
    energy: float                      # E^(P), electronic
    lambda_grad: np.ndarray
    lam: np.ndarray
    delta: float = 0.0
    A_base: CoefficientMatrix = None
    A_corr: CoefficientMatrix = None
    d_mp: np.ndarray = None            # d_p of Eq. (5.6), MP denominator only
    arrays: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)


class SelectedCC:
    """CC(P) and CC(P;Q) in a DeterminantSpace with a fixed Hamiltonian matrix.

    Parameters
    ----------
    space : DeterminantSpace
    H : (ndet, ndet) electronic Hamiltonian
    P, Q : determinant indices (reference excluded; Q disjoint from P)
    fock_diag : spin-orbital f_pp in the current orbitals (bare-Fock denominator)
    denominator : "full" (Eq. 3.4) or "mp" (Eq. 3.5)
    """

    def __init__(self, space, H, P, Q=(), fock_diag=None, denominator="full", t_guess=None,
                 r_tol=1.0e-12, max_iter=100):
        self.space, self.H = space, H
        self.P = validate_indices(space, P, "P")
        self.Q = validate_indices(space, Q, "Q")
        if len(np.intersect1d(self.P, self.Q)) or 0 in self.P or 0 in self.Q:
            raise ValueError("P and Q must be disjoint and exclude the reference.")
        if denominator not in ("full", "mp"):
            raise ValueError("denominator must be 'full' or 'mp'.")
        if denominator == "mp" and len(self.Q) and fock_diag is None:
            raise ValueError("The MP denominator needs the Fock diagonal.")
        self.fock_diag, self.denominator = fock_diag, denominator
        self.t_guess, self.r_tol, self.max_iter = t_guess, r_tol, max_iter
        self.excP = space.excitation_transitions(self.P)
        self.max_rank = int(space.rank.max())

    def embed(self, x, idx):
        out = np.zeros(self.space.ndet)
        out[idx] = x
        return out

    def T_matrix(self, t):
        return self.excP.matrix(t, self.space.ndet)

    def jacobian(self, Hbar):
        """J_AI = H^I_A0 = Hbar_AI - (E_I Hbar_{.0})_A for A, I in P (Eq. 2.4)."""
        P, exc = self.P, self.excP
        h0 = Hbar[:, 0]
        EH0 = sp.csr_matrix((exc.sign * h0[exc.src], (exc.dst, exc.label)), shape=(self.space.ndet, len(P)))
        return Hbar[np.ix_(P, P)] - EH0[P].toarray()

    # ------------------------------------------------------------------ Sec. 2.1, right equations
    def solve_right(self, verbose=False):
        """Newton iterations on M_I(P) = Hbar_I0 = 0 with the exact Jacobian."""
        t = np.zeros(len(self.P)) if self.t_guess is None else np.array(self.t_guess, dtype=float)
        for it in range(self.max_iter):
            U, V, Hbar = similarity_transform(self.H, self.T_matrix(t), self.max_rank)
            R = Hbar[self.P, 0]
            rnorm = np.abs(R).max(initial=0.0)
            if verbose:
                print(f"   CC(P) Newton {it:3d}  E = {Hbar[0, 0]:.12f}  |M_P| = {rnorm:.2e}")
            if rnorm < self.r_tol:
                return t, U, V, Hbar
            t = t - np.linalg.solve(self.jacobian(Hbar), R)
        raise RuntimeError("CC(P) right equations did not converge.")

    # ------------------------------------------------------------------ the full solve
    def run(self, verbose=False):
        sp_, P, Q, n = self.space, self.P, self.Q, self.space.ndet
        e0 = self.embed(1.0, [0])
        t, U, V, Hbar = self.solve_right(verbose)
        E = Hbar[0, 0]
        J = self.jacobian(Hbar)

        # Sec. 2.1 base gradient multiplier, J^T lambda_grad = -e, e_I = H^I_00 = Hbar_0I
        lambda_grad = np.linalg.solve(J.T, -Hbar[0, P])
        # Sec. 2.2 projected left state, B^T lambda = -Hbar_0P
        B = Hbar[np.ix_(P, P)] - E * np.eye(len(P))
        lam = np.linalg.solve(B.T, -Hbar[0, P])
        b = e0 + self.embed(lam, P)

        A_base = CoefficientMatrix(n, [(e0 + self.embed(lambda_grad, P), e0)])
        res = AmplitudeResult(t, U, V, Hbar, E, lambda_grad, lam, A_base=A_base)
        # Eq. (2.9): the base stationarity residual of Lambda itself
        pi_defect = A_base.t_gradient(Hbar, self.excP)
        lam_defect = CoefficientMatrix(n, [(b, e0)]).t_gradient(Hbar, self.excP)
        res.checks.update(
            right_residual=np.abs(Hbar[P, 0]).max(initial=0.0),
            base_stationarity=np.abs(pi_defect).max(initial=0.0),
            lambda_minus_lambda_grad=np.linalg.norm(lam - lambda_grad),
            eq_2_9_defect=np.abs(lam_defect).max(initial=0.0),
        )

        if len(Q) == 0:
            res.A_corr = CoefficientMatrix(n)
            return res

        # Sec. 3.1 moments, left numerators, denominators, divided coefficients
        M = Hbar[Q, 0]
        N = b @ Hbar[:, Q]
        if self.denominator == "full":
            D = E - np.diag(Hbar)[Q]
            chi = None
        else:
            chi = sp_.chi(Q)
            D = chi @ self.fock_diag
        if np.any(np.abs(D) < 1.0e-9):
            raise ValueError("near-zero Q-space denominator")
        ell, r = N / D, M / D
        res.delta = float(ell @ M)
        ell_f, r_f = self.embed(ell, Q), self.embed(r, Q)

        # Sec. 4.2 Sigma: B sigma = -s_lambda, s_lambda = Hbar_PQ r
        s_lambda = Hbar[np.ix_(P, Q)] @ r
        sigma = np.linalg.solve(B, -s_lambda)
        s = lam @ sigma
        sigma_f = self.embed(sigma, P)

        # Sec. 4.3 cluster sources: s_t from C (with d_KI for the full denominator), k from
        # the projected-left constraint, both as t-gradients of coefficient matrices
        A_st = CoefficientMatrix(n, [(ell_f, e0), (b, r_f)])
        if self.denominator == "full":
            w = ell_f * r_f
            w[0] = -np.sum(ell * r)
            A_st = A_st + CoefficientMatrix(n, w=w)                       # Eq. (6.4)
        A_k = CoefficientMatrix(n, [(b, sigma_f), (-s * e0, e0)])
        s_t = A_st.t_gradient(Hbar, self.excP)
        k = A_k.t_gradient(Hbar, self.excP)
        # Sec. 4.3 Pi: J^T pi = -s_t - k
        pi = np.linalg.solve(J.T, -s_t - k)
        A_pi = CoefficientMatrix(n, [(self.embed(pi, P), e0)])
        res.A_corr = A_pi + A_k + A_st                                    # Eq. (6.3) + (6.4)
        if self.denominator == "mp":
            res.d_mp = -(ell * r) @ chi                                   # Eq. (5.6)

        # Sec. 4.4 residuals and stationarity of the whole correction functional
        lam_grad_corr = (Hbar[np.ix_(P, P)] - E * np.eye(len(P))) @ sigma + Hbar[np.ix_(P, Q)] @ r
        value = res.A_corr.pair(Hbar)
        if self.denominator == "mp":
            value -= np.sum(ell * r * D)
        res.checks.update(
            sigma_residual=np.abs(B @ sigma + s_lambda).max(initial=0.0),
            pi_residual=np.abs(J.T @ pi + s_t + k).max(initial=0.0),
            corr_t_stationarity=np.abs(res.A_corr.t_gradient(Hbar, self.excP)).max(initial=0.0),
            corr_lambda_stationarity=np.abs(lam_grad_corr).max(initial=0.0),
            corr_functional_minus_delta=abs(value - res.delta),
        )
        res.arrays.update(M=M, N=N, D=D, ell=ell, r=r, s_lambda=s_lambda, s_t=s_t, k=k, sigma=sigma, pi=pi)
        return res
