"""Matrix-free CC(P;Q)/EOMCC(P;Q) stationary derivatives.

All determinant-space objects are vectors or sparse connected matrices.
Bare H actions and transition densities use externally compiled FCIpy kernels.
CC Newton and constrained adjoints use GMRES; projected EOM roots use Davidson.
No full determinant-pair Hamiltonian, exponential, Jacobian, or density
coefficient matrix is constructed. Full-denominator diagonals are streamed or
evaluated with exact sparse rank-limited commutators when sparse_cache=True.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import operator

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import LinearOperator, gmres

from . import spaces
from .densities import SpatialDensities, mp_denominator_densities, spin_trace
from .determinants import DeterminantSpace, spin_orbital_integrals, fock_diagonal
from .fcipy_adapter import FCIpyOperator


def linear_solve(action, rhs, *, tol=2e-11, diagonal=None, preconditioner=None):
    """Checked matrix-free GMRES. No silent use of unconverged responses."""
    rhs = np.asarray(rhs, dtype=float)
    if not rhs.size or np.max(abs(rhs)) < 1e-15:
        return np.zeros_like(rhs)
    n = len(rhs)
    A = LinearOperator((n, n), matvec=action, dtype=float)
    M = None
    if preconditioner is not None:
        M = LinearOperator((n, n), matvec=preconditioner, dtype=float)
    elif diagonal is not None:
        d = np.asarray(diagonal).copy()
        d[abs(d) < 1e-7] = 1.0
        M = LinearOperator((n, n), matvec=lambda x: x / d, dtype=float)
    result, info = gmres(A, rhs, M=M, rtol=tol, atol=1e-13,
                         restart=min(n, 100), maxiter=40)
    error = np.max(abs(action(result) - rhs))
    if info or error > max(2e-10, 5 * tol * np.max(abs(rhs))):
        raise RuntimeError(f"Matrix-free response failed: info={info}, residual={error:.3e}")
    return result


def davidson(action, diagonal, nroots=1, target=None, tol=2e-11, maxiter=150, maxspace=40, guess=None):
    """Non-Hermitian Davidson with explicit residual checks and root ordering.

    Lowest-real roots are selected unless a target eigenvalue is supplied.
    Complex trial vectors are allowed; a converged complex target is rejected.
    """
    diagonal = np.asarray(diagonal)
    n = len(diagonal)
    if nroots > n or nroots < 1:
        raise ValueError("Invalid Davidson root count")
    maxspace = min(n, max(maxspace, 2 * nroots + 2))
    order = np.argsort(diagonal if target is None else abs(diagonal - target))
    count = min(n, max(4, nroots + 2))
    B = np.zeros((n, count), dtype=complex)
    B[order[:count], np.arange(count)] = 1.0
    if target is None and count < n:
        # Deterministic full-support seeds include spin/spatial sectors absent
        # from a few lowest diagonal (often singles-only) determinant guesses.
        random = np.random.default_rng(731).normal(size=(n, min(2, n-count)))
        B = la.orth(np.column_stack((B, random)))
    if guess is not None:
        B = la.orth(np.column_stack((np.asarray(guess).reshape(n, -1), B)))
    seed = B.copy()
    S = np.column_stack([action(v) for v in B.T])
    for iteration in range(maxiter):
        values, coeff = la.eig(B.conj().T @ S)
        idx = np.argsort(values.real if target is None else abs(values - target))[:nroots]
        values, coeff = values[idx], coeff[:, idx]
        vectors = B @ coeff
        residuals = S @ coeff - vectors * values
        if np.max(la.norm(residuals, axis=0)) < tol:
            if np.max(abs(values.imag)) > 1e-8:
                raise ValueError("Complex EOM root")
            for j in range(nroots):
                pivot = np.argmax(abs(vectors[:, j]))
                vectors[:, j] *= np.exp(-1j * np.angle(vectors[pivot, j]))
            if np.max(abs(vectors.imag)) > 1e-7:
                raise ValueError("EOM root has a non-real eigenvector after phase fixing")
            return values.real, vectors.real, iteration + 1
        additions = []
        for j in range(nroots):
            if la.norm(residuals[:, j]) < tol:
                continue
            den = values[j] - diagonal
            den[abs(den) < 1e-7] = 1e-7
            q = residuals[:, j] / den
            for _ in range(2):
                q -= B @ (B.conj().T @ q)
                for b in additions:
                    q -= b * np.vdot(b, q)
            norm = la.norm(q)
            if norm > 1e-12:
                additions.append(q / norm)
        if not additions:
            raise RuntimeError("Davidson subspace stalled before convergence")
        if B.shape[1] + len(additions) > maxspace:
            # Retain the target seed so a restart cannot discard its spin or
            # spatial symmetry sector in favor of a nearby converged root.
            B = la.orth(np.column_stack((vectors, seed)))
            S = np.column_stack([action(v) for v in B.T])
        else:
            new = np.column_stack(additions)
            B = np.column_stack((B, new))
            S = np.column_stack((S, np.column_stack([action(v) for v in new.T])))
    raise RuntimeError("Davidson did not converge")


@dataclass
class Factors:
    """Coefficient of Hbar as a sum of outer products plus a sparse diagonal."""
    terms: list
    diagonal: dict

    def pairs(self, ndet):
        """Stream diagonal factors with O(ndet) scratch, even for large Q."""
        yield from self.terms
        for k, weight in self.diagonal.items():
            if weight:
                unit = np.zeros(ndet)
                unit[k] = 1.0
                yield weight * unit, unit

    def __add__(self, other):
        diagonal = dict(self.diagonal)
        for k, w in other.diagonal.items():
            diagonal[k] = diagonal.get(k, 0.0) + w
        return Factors(self.terms + other.terms, diagonal)


class MatrixFreeState:
    def __init__(self, space, P, Q, H, fock, state, denominator, t_guess=None, excitations=None,
                 eom_guesses=None, target_root=None):
        self.space, self.P, self.Q = space, P, Q
        self.H, self.fock = H, fock
        self.state, self.denominator = state, denominator
        self.full = np.r_[0, P]
        self.m, self.k = len(P), len(P) + 1
        self.exc = space.excitation_transitions(P) if excitations is None else excitations
        self.rank = int(space.rank.max())
        self.e0 = self.embed([1.0], [0])
        self.chi = space.chi(Q)
        self.t_guess = t_guess
        self.eom_guesses = eom_guesses
        self.target_root = target_root
        self.t = None
        self.T = None

    def embed(self, values, indices=None):
        values = np.asarray(values)
        out = np.zeros(self.space.ndet, dtype=values.dtype)
        out[self.full if indices is None else indices] = values
        return out

    def set_t(self, t):
        if self.t is None or not np.array_equal(t, self.t):
            self.t = np.array(t, copy=True)
            self.T = self.exc.matrix(t, self.space.ndet)
            self._diagonal_prepared = False
            self._diagonal_t_coeff = None

    def prepare_diagonal(self):
        """Only T1/T2 can contribute to diagonal elements of a two-body H.

        Rank(H) can decrease by at most two. Thus the exact diagonal of
        exp(-T) H exp(T) terminates at [H,T1+T2] and [[H,T1],T1]/2.
        Products below are sparse, and are shared by all Q determinants.
        """
        if self._diagonal_prepared:
            return
        n = self.space.ndet
        levels = self.space.rank[self.P]
        self.T1 = self.exc.matrix(np.where(levels == 1, self.t, 0.), n)
        self.T12 = self.exc.matrix(np.where(levels <= 2, self.t, 0.), n)
        self.T1.eliminate_zeros()
        self.T12.eliminate_zeros()
        self.T11 = self.T1 @ self.T1
        H = self.H.sparse
        rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(H.indptr))
        mask = self.space.rank[H.indices] - self.space.rank[rows] == 2
        self.Hminus2 = sp.csr_matrix((H.data[mask], (rows[mask], H.indices[mask])), shape=H.shape)
        self.Hminus2.eliminate_zeros()
        self.T1Hminus2 = self.T1 @ self.Hminus2
        self._diagonal_prepared = True

    def hbar_diagonal(self):
        self.prepare_diagonal()
        H, T, T11 = self.H.sparse, self.T12, self.T11
        diagonal = H.diagonal() + np.asarray(H.multiply(T.T).sum(axis=1)).ravel()
        diagonal -= np.asarray(T.multiply(H.T).sum(axis=1)).ravel()
        diagonal += .5 * np.asarray(H.multiply(T11.T).sum(axis=1)).ravel()
        diagonal += .5 * np.asarray(T11.multiply(H.T).sum(axis=1)).ravel()
        diagonal -= np.asarray(self.T1Hminus2.multiply(self.T1.T).sum(axis=1)).ravel()
        return diagonal

    def diagonal_t_gradient(self, weights):
        self.prepare_diagonal()
        tr = self.exc
        levels = self.space.rank[self.P][tr.label]
        valid = levels <= 2
        if self._diagonal_t_coeff is None:
            coeff = np.zeros(len(tr.label))
            coeff[valid] = np.asarray(self.H.sparse[tr.src[valid], tr.dst[valid]]).ravel()
            singles = levels == 1
            commutator = self.Hminus2 @ self.T1 - self.T1Hminus2
            coeff[singles] += np.asarray(commutator[tr.src[singles], tr.dst[singles]]).ravel()
            self._diagonal_t_coeff = tr.sign * coeff
        return np.bincount(tr.label, weights=(weights[tr.src] - weights[tr.dst]) * self._diagonal_t_coeff,
                           minlength=self.m)

    def diagonal_density(self, weights):
        self.prepare_diagonal()
        W = sp.diags(weights, format='csr')
        # Transpose of the coefficient paired with H. Terms above excitation
        # rank two cannot contract a one-/two-body operator and vanish exactly.
        C = W + self.T12 @ W - W @ self.T12
        C += .5 * (self.T11 @ W + W @ self.T11) - self.T1 @ W @ self.T1
        C.eliminate_zeros()
        return self.H.coefficient_density(C)

    def diagonal_weights(self, factors):
        weights = np.zeros(self.space.ndet)
        for k, w in factors.diagonal.items():
            weights[k] = w
        return weights

    def exponential(self, vector, sign=1, transpose=False):
        T = self.T.T if transpose else self.T
        result = np.array(vector, copy=True)
        term = result.copy()
        for j in range(1, self.rank + 1):
            term = sign * (T @ term) / j
            result = result + term
        return result

    def bar(self, vector, transpose=False):
        if transpose:
            return self.exponential(self.H(self.exponential(vector, -1, True)), 1, True)
        return self.exponential(self.H(self.exponential(vector)), -1)

    def solve_right(self, verbose=False):
        t = np.zeros(self.m) if self.t_guess is None else self.t_guess.copy()
        diagonal = -self.space.chi(self.P) @ self.fock
        for iteration in range(60):
            self.set_t(t)
            b0 = self.bar(self.e0)
            residual = b0[self.P]
            norm = np.max(abs(residual), initial=0.0)
            if verbose:
                print(f"   Matrix-free CC(P) {iteration:3d} E={b0[0]:.12f} residual={norm:.3e}", flush=True)
            if norm < 2e-12:
                return t, b0

            def jacobian(x):
                response = np.bincount(self.exc.dst,
                    weights=self.exc.sign * x[self.exc.label] * b0[self.exc.src],
                    minlength=self.space.ndet)
                return self.bar(self.embed(x, self.P))[self.P] - response[self.P]

            step = linear_solve(jacobian, -residual, diagonal=diagonal)
            for scale in (1., .5, .25, .125, .0625):
                trial = t + scale * step
                self.set_t(trial)
                if np.max(abs(self.bar(self.e0)[self.P])) < norm:
                    t = trial
                    break
            else:
                raise RuntimeError("CC(P) Newton line search failed")
        raise RuntimeError("Matrix-free CC(P) did not converge")

    def initialize(self, verbose=False):
        t, self.b0 = self.solve_right(verbose)
        self.set_t(t)
        self.E0 = self.b0[0]
        # Slater diagonal is a preconditioner only, never an Hbar substitute.
        diag = self.H.lib.ci.calc_diagonal(self.H.det, self.space.nalpha, self.space.nbeta,
                                          self.H.h, self.H.v)[self.P]
        self.preconditioner_diagonal = diag
        self.iterations = 0
        self.left_root_error = 0.
        self.eom_values = np.zeros(0)
        if self.state == 0:
            row = self.bar(self.e0, True)[self.P]
            lam = linear_solve(lambda x: self.bar(self.embed(x, self.P), True)[self.P] - self.E0 * x,
                               -row, diagonal=diag - self.E0)
            self.r = np.r_[1., np.zeros(self.m)]
            self.l = np.r_[1., lam]
            self.E, self.pivot = self.E0, None
        else:
            action = lambda x: self.bar(self.embed(x, self.P))[self.P]
            transpose = lambda x: self.bar(self.embed(x, self.P), True)[self.P]
            guesses = self.eom_guesses or (None, None)
            # Converge a buffer of low roots so that an unrepresented lower
            # state cannot silently change the requested root across FD steps.
            if self.target_root is None:
                work_roots = min(self.m, max(4, self.state + 2))
                vals, vecs, nr = davidson(action, diag, nroots=work_roots, guess=guesses[0])
                selected = self.state - 1
            else:
                target = self.target_root
                guess = target['right'][:, None] if guesses[0] is None else np.column_stack((target['right'], guesses[0]))
                target_energy = self.E0 + target['omega']
                vals, vecs, nr = davidson(action, diag, target=target_energy, guess=guess)
                overlap = abs(np.vdot(target['right'], vecs[:, 0])) / (la.norm(target['right']) * la.norm(vecs[:, 0]))
                if abs(vals[0] - target_energy) > target['radius'] or overlap < .5:
                    raise RuntimeError('Tracked EOM root left its guarded reference neighborhood')
                selected = 0
            self.eom_values = vals.copy()
            self.E = vals[selected]
            rP = vecs[:, selected]
            left_guess = rP[:, None] if guesses[1] is None else np.column_stack((rP, guesses[1]))
            left_values, left, nl = davidson(transpose, diag, target=self.E, guess=left_guess)
            if abs(left_values[0] - self.E) > 1e-9:
                raise RuntimeError("Left Davidson root does not match the right EOM root")
            self.left_root_error = abs(left_values[0] - self.E)
            self.eom_guesses = (vecs.copy(), left.copy())
            self.r = np.r_[self.bar(self.embed(rP, self.P))[0] / (self.E - self.E0), rP]
            self.l = np.r_[0., left[:, 0]]
            self.pivot = int(np.argmax(abs(self.r[1:]))) + 1
            self.r /= self.r[self.pivot]
            overlap = self.l @ self.r
            if abs(overlap) < 1e-9 * la.norm(self.l) * la.norm(self.r):
                raise RuntimeError("Left/right EOM overlap vanishes; cannot biorthogonalize")
            self.l /= overlap
            self.iterations = nr + nl
        self.rhs, self.lhs = self.embed(self.r), self.embed(self.l)
        self.br, self.btl = self.bar(self.rhs), self.bar(self.lhs, True)
        self.M, self.N = self.br[self.Q], self.btl[self.Q]
        if self.denominator == "full":
            if self.H.sparse is not None and len(self.Q) > 8:
                diagonal = self.hbar_diagonal()[self.Q]
            else:
                diagonal = []
                for k in self.Q:
                    unit = self.embed([1.], [k])
                    diagonal.append(self.bar(unit)[k])
            self.D = self.E - np.array(diagonal)
        else:
            self.D = self.E - self.E0 + self.chi @ self.fock
        if np.any(abs(self.D) < 1e-9):
            raise ValueError("near-zero Q-space denominator")
        self.delta = np.sum(self.M * self.N / self.D)
        return t

    def constraint(self):
        if self.state == 0:
            return np.r_[self.b0[self.P], self.btl[self.P] - self.E0 * self.l[1:]]
        return np.r_[self.b0[self.P], self.br[self.full] - self.E * self.r,
                     np.delete(self.btl[self.full] - self.E * self.l, self.pivot), self.l @ self.r - 1]

    def constraint_factors(self, z):
        terms = [(self.embed(z[:self.m], self.P), self.e0)]
        if self.state == 0:
            zz = z[self.m:]
            terms.append((self.lhs, self.embed(zz, self.P)))
            return Factors(terms, {0: -zz @ self.l[1:]})
        zr = self.embed(z[self.m:self.m + self.k])
        zl = self.embed(np.insert(z[self.m + self.k:-1], self.pivot, 0.))
        terms.extend([(zr, self.rhs), (self.lhs, zl)])
        return Factors(terms, {})

    def t_gradient(self, factors):
        result = np.zeros(self.m)
        fast = self.H.sparse is not None and len(factors.diagonal) > 8
        if fast:
            result += self.diagonal_t_gradient(self.diagonal_weights(factors))
        for u, v in (factors.terms if fast else factors.pairs(self.space.ndet)):
            if np.any(u) and np.any(v):
                result += self.exc.contract_outer(self.bar(u, True), v) - self.exc.contract_outer(u, self.bar(v))
        return result

    def pack_gradient(self, gt, gr, gl, gE=0.):
        if self.state == 0:
            return np.r_[gt, gl[1:]]
        return np.r_[gt, np.delete(gr, self.pivot), gl, gE]

    def vjp(self, z):
        gt = self.t_gradient(self.constraint_factors(z))
        if self.state == 0:
            zz = z[self.m:]
            gl = np.r_[0., self.bar(self.embed(zz, self.P))[self.P] - self.E0 * zz]
            return self.pack_gradient(gt, None, gl)
        zr = z[self.m:self.m + self.k]
        zl = np.insert(z[self.m + self.k:-1], self.pivot, 0.)
        gr = self.bar(self.embed(zr), True)[self.full] - self.E * zr + z[-1] * self.l
        gl = self.bar(self.embed(zl))[self.full] - self.E * zl + z[-1] * self.r
        return self.pack_gradient(gt, gr, gl, -zr @ self.r - zl @ self.l)

    def solve_adjoint(self, gradient):
        """Exploit the block-triangular CC/state constraint Jacobian.

        First solve the state-vector response, then the cluster response.
        The diagonal/bordered state preconditioner has O(len(P)) nonzeros;
        the exact state and cluster operators remain matrix-free.
        """
        if self.state == 0:
            action = lambda z: self.bar(self.embed(z, self.P))[self.P] - self.E0 * z
            zstate = linear_solve(action, -gradient[self.m:],
                                  diagonal=self.preconditioner_diagonal - self.E0)
        else:
            def action(z):
                zr = z[:self.k]
                zl = np.insert(z[self.k:-1], self.pivot, 0.)
                gr = self.bar(self.embed(zr), True)[self.full] - self.E * zr + z[-1] * self.l
                gl = self.bar(self.embed(zl))[self.full] - self.E * zl + z[-1] * self.r
                return np.r_[np.delete(gr, self.pivot), gl, -zr @ self.r - zl @ self.l]

            free = np.delete(np.arange(self.k), self.pivot)
            D = sp.diags(np.r_[self.E0, self.preconditioner_diagonal] - self.E, format='csr')
            K = sp.bmat([[D[:, free], sp.csr_matrix((self.k, self.k)), sp.csr_matrix(-self.r[:, None])],
                         [sp.csr_matrix((self.k-1, self.k-1)), D[free, :], sp.csr_matrix(-self.l[free, None])],
                         [sp.csr_matrix(self.l[free][None, :]), sp.csr_matrix(self.r[None, :]), sp.csr_matrix((1, 1))]], format='csc')
            pre = splu(K.T.tocsc())
            zstate = linear_solve(action, -gradient[self.m:], preconditioner=pre.solve)
        only_state = np.r_[np.zeros(self.m), zstate]
        rhs = -gradient[:self.m] - self.t_gradient(self.constraint_factors(only_state))
        def cluster_transpose(z):
            u = self.embed(z, self.P)
            return self.bar(u, True)[self.P] - self.exc.contract_outer(u, self.b0)
        zcc = linear_solve(cluster_transpose, rhs, diagonal=-self.space.chi(self.P) @ self.fock)
        return np.r_[zcc, zstate]

    def objective(self, correction):
        if not correction:
            if self.state == 0:
                factors = Factors([(self.e0, self.e0)], {})
                return factors, self.pack_gradient(self.t_gradient(factors), None, np.zeros(self.k))
            factors = Factors([(self.lhs, self.rhs)], {})
            return factors, self.pack_gradient(self.t_gradient(factors), self.btl[self.full], self.br[self.full])
        ell = self.embed(self.N / self.D, self.Q)
        divided = self.embed(self.M / self.D, self.Q)
        weights = self.N * self.M / self.D**2
        diagonal = {}
        gE = 0.
        if self.denominator == "full":
            diagonal = dict(zip(self.Q, weights))
            if self.state == 0:
                diagonal[0] = -sum(weights)
            else:
                gE = -sum(weights)
        elif self.state:
            diagonal[0] = sum(weights)
            gE = -sum(weights)
        factors = Factors([(self.lhs, divided), (ell, self.rhs)], diagonal)
        gr = self.bar(ell, True)[self.full]
        gl = self.bar(divided)[self.full]
        return factors, self.pack_gradient(self.t_gradient(factors), gr, gl, gE)

    def density(self, factors):
        n = self.space.norb
        density = SpatialDensities(np.zeros((n, n)), np.zeros((n,) * 4))
        fast = self.H.sparse is not None and len(factors.diagonal) > 8
        if fast:
            density = density + self.diagonal_density(self.diagonal_weights(factors))
        for left, right in (factors.terms if fast else factors.pairs(self.space.ndet)):
            if np.any(left) and np.any(right):
                u = self.exponential(left, -1, True)
                v = self.exponential(right)
                density = density + self.H.transition_density(u, v)
        return density

    def run(self, verbose=False, densities=True):
        t = self.initialize(verbose)
        base = self.E0 if self.state == 0 else self.l @ self.br[self.full]
        checks = dict(constraint=float(np.max(abs(self.constraint()), initial=0.)),
                      left_right=float(self.l @ self.r), davidson_iterations=self.iterations,
                      left_root_error=float(self.left_root_error))
        if checks['constraint'] > 1e-8:
            raise RuntimeError(f"CC/EOM state constraints are not converged: {checks['constraint']:.3e}")
        data = SimpleNamespace(t=t, r=self.r, l=self.l, E_base=base, delta=self.delta,
                               state=self.state, P=self.P.copy(),
                               E0=self.E0, omega=self.E-self.E0,
                               eom_values=self.eom_values,
                               arrays=dict(M=self.M, N=self.N, D=self.D,
                                           D_mp=self.E-self.E0+self.chi@self.fock), checks=checks)
        if not densities:
            return data, None, None
        rho = []
        for part in (0, 1):
            factors, gradient = self.objective(part)
            adjoint = self.solve_adjoint(gradient)
            checks["stationarity_corr" if part else "stationarity_base"] = float(np.max(abs(self.vjp(adjoint) + gradient), initial=0.))
            stationary = factors + self.constraint_factors(adjoint)
            rho.append(self.density(stationary))
        if self.denominator == "mp" and len(self.Q):
            dmp = -(self.N * self.M / self.D**2) @ self.chi
            rho[1] = rho[1] + spin_trace(mp_denominator_densities(dmp, self.space.ref_occ), self.space.norb)
        return data, rho[0], rho[1]


class MatrixFreeBackend:
    """AmplitudeStage-compatible FCIpy backend; full determinant-space vectors.

    Supply an extension built by devtools/build_fcipy_oracle.py through
    ``fcipy_extension`` or CCPY_FCIPY_EXTENSION. FCIpy is optional for dense runs.
    With ``sparse_cache=True``, cache the connected bare Hamiltonian as CSR
    and use connected transition RDMs and exact sparse Hbar diagonal algebra.
    """
    def __init__(self, norb, nalpha, nbeta, nfrozen, pspace, qspace, denominator, state=0, fcipy_extension=None,
                 sparse_cache=False):
        from .driver import DeterminantBackend
        self.space = DeterminantSpace(norb, nalpha, nbeta, nfrozen)
        self._select = DeterminantBackend._select.__get__(self)
        self.P = self._select(pspace)
        self.Q = self._select(qspace, P=self.P) if qspace is not None else np.zeros(0, dtype=np.int64)
        self.state = operator.index(state)
        if not 0 <= self.state <= len(self.P):
            raise ValueError("state must be 0 (ground) or 1..len(P) (excited)")
        if denominator not in ("mp", "full") or np.intersect1d(self.P, self.Q).size:
            raise ValueError("Require disjoint P/Q and denominator 'mp' or 'full'")
        self.denominator, self.fcipy_extension = denominator, fcipy_extension
        self.sparse_cache = sparse_cache
        self.t = None
        self._exc = None
        self._eom_guesses = None
        self._target_root = None
        self._operator = None
        self._operator_integrals = None

    def follow_reference(self, result):
        """Follow a checked nondegenerate root for nearby FD displacements.

        ``result`` is the data from this backend's reference AmplitudeStage.
        The reference must contain the buffer of low roots. Its excitation
        energy relative to the current CC(P) ground state is targeted, so a
        common Hamiltonian energy shift does not change branch selection.
        The target is guarded by one quarter of the nearest gap and by >=0.5
        normalized right-vector overlap; constraints/residuals remain checked.
        This is local branch following, not general crossing/degeneracy handling.
        Reference and displaced vectors must use a consistent orbital phase gauge.
        """
        if not self.state:
            raise ValueError('Reference-root following is for excited states')
        if result.state != self.state or not np.array_equal(result.P, self.P):
            raise ValueError('Reference result must have the same state and P space')
        values = np.asarray(result.eom_values)
        if len(values) < self.state or abs(values[self.state-1] - result.E_base) > 1e-8:
            raise ValueError('Reference root ordering has not been established')
        gaps = abs(np.delete(values, self.state-1) - result.E_base)
        if np.any(gaps < 1e-8):
            raise ValueError('Cannot follow a degenerate reference root')
        if not len(gaps):
            raise ValueError('No resolved neighboring roots to guard local tracking')
        radius = .25 * float(np.min(gaps))
        omega = float(result.E_base-result.E0)
        self._target_root = dict(energy=float(result.E_base), omega=omega, radius=radius, right=result.r[1:].copy())
        return dict(energy=float(result.E_base), excitation_energy=omega, radius=radius)

    def clear_reference(self):
        self._target_root = None

    def run(self, h, eri, verbose=False, densities=True):
        from .driver import AmplitudeStage
        h, eri = np.asarray(h), np.asarray(eri)
        n = self.space.norb
        if h.shape != (n, n) or eri.shape != (n,) * 4 or np.iscomplexobj(h) or np.iscomplexobj(eri):
            raise ValueError("Require real full-space spatial h and chemists' eri")
        H = self._operator if self.sparse_cache else None
        if H is not None:
            unchanged = all(np.array_equal(a, b) for a, b in zip((h, eri), self._operator_integrals))
            if not unchanged and not H.update_integrals(h, eri):
                self._operator = None
                H = None
        if H is None:
            H = FCIpyOperator(self.space, h, eri, self.fcipy_extension, sparse_cache=self.sparse_cache)
        if self.sparse_cache:
            self._operator = H
            self._operator_integrals = (h.copy(), eri.copy())
        start_calls = H.calls
        hs, vs = spin_orbital_integrals(h, eri)
        fock = fock_diagonal(hs, vs, self.space.ref_occ)
        calculation = MatrixFreeState(self.space, self.P, self.Q, H, fock, self.state,
                                      self.denominator, t_guess=self.t, excitations=self._exc,
                                      eom_guesses=self._eom_guesses, target_root=self._target_root)
        # Excitation connectivity is fixed across geometries.
        if self._exc is None:
            self._exc = calculation.exc
        data, base, corr = calculation.run(verbose, densities=densities)
        self.t = data.t.copy()
        self._eom_guesses = calculation.eom_guesses
        data.checks['hamiltonian_actions'] = H.calls - start_calls
        if not densities:
            return data
        checks = data.checks
        checks["base_density_energy"] = abs(base.energy(h, eri) - data.E_base)
        checks["corr_density_energy"] = abs(corr.energy(h, eri) - data.delta)
        return AmplitudeStage(data.E_base, data.delta, base, corr, checks, data)

    def energies(self, h, eri):
        data = self.run(h, eri, densities=False)
        return data.E_base, data.delta
