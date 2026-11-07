"""Density fitting (or Cholesky) factorization of the two-electron integrals and the
Coulomb/exchange builder used by the SCF solvers.

The factorization is

    (mn|rs) ~= \\sum_P B^P_{mn} B^P_{rs},    B^P_{mn} = \\sum_Q [M^{-1/2}]_{PQ} (Q|mn),

where M_{PQ} = (P|Q) is the Coulomb metric of the auxiliary basis. In the MO basis the
B tensor plays exactly the role of the Cholesky vectors used by the ``*_chol`` methods
of ccpy, so ``mo_cholesky_vectors(C)`` can be handed straight to
``getCholeskyHamiltonian``.

Adapted from forte2 (https://github.com/evangelistalab/forte2, LGPL-3.0); see
THIRD_PARTY_NOTICES.md.
"""

from collections import OrderedDict

import numpy as np
import scipy.linalg

from ccpy.integrals import integrals
from ccpy.integrals.linalg import invsqrt_matrix, cholesky_metric_inverse_sqrt

# Above this size of the B tensor (GB), SCF Coulomb and exchange builds are integral-direct.
DIRECT_DF_MEMORY_GB = 4.0


class DensityFitting:
    """B tensor of a Molecule (built on first use), plus J/K builds and MO transformations.

    B_Pmn : the B tensor, shape (naux, nbf, nbf)
    Mm12 : X with B = X (P|mn) and X^T X = M^{-1}, shape (naux, naux). For density fitting
           M = (P|Q); for the Cholesky decomposition the "auxiliary functions" are the pivot
           pairs J = (r s) and M = (J|K) (incomplete pivoted Cholesky = Nystrom: (mn|rs) ~
           sum_JK (mn|J) M^-1_JK (K|rs)).
    pivots : the Cholesky pivot function pairs [(r, s), ...] (Cholesky only)
    direct : True when the J/K builds and SCF gradients recompute (P|mn) instead of storing
             B: set by Molecule(df_direct=...), by default when B would exceed
             DIRECT_DF_MEMORY_GB. Correlated methods still build B on first use.
    """

    def __init__(self, molecule):
        self.molecule = molecule
        self.nbf = molecule.nbf
        self._B_Pmn = None
        self._B_nPm = None
        self.Mm12 = None
        self.pivots = None
        direct = getattr(molecule, "df_direct", None)
        if direct is None:
            direct = (not molecule.cholesky_tei
                      and 8 * molecule.naux * molecule.nbf**2 / 1e9 > DIRECT_DF_MEMORY_GB)
        if direct and molecule.cholesky_tei:
            raise ValueError("df_direct requires density fitting, not cholesky_tei.")
        self.direct = direct
        self._fitting = None

    @property
    def fitting(self):
        """The integral-direct DirectFitting used when ``direct`` is set."""
        if self._fitting is None:
            self._fitting = DirectFitting(self.molecule)
        return self._fitting

    @property
    def B_Pmn(self):
        if self._B_Pmn is None:
            if self.molecule.cholesky_tei:
                self._B_Pmn = self._build_B_cholesky()
            else:
                self._B_Pmn = self._build_B_density_fitting()
            self.naux = self._B_Pmn.shape[0]
        return self._B_Pmn

    @property
    def B_nPm(self):
        """B tensor reordered as (nbf, naux, nbf) for fast K builds."""
        if self._B_nPm is None:
            self._B_nPm = np.ascontiguousarray(self.B_Pmn.transpose((2, 0, 1)))
        return self._B_nPm

    def _build_B_density_fitting(self):
        mol = self.molecule
        nbf, naux = mol.nbf, mol.naux
        memory_gb = 8 * (naux**2 + 2 * naux * nbf**2) / 1024**3
        if mol.verbose:
            print(f"   Density fitting: {nbf} orbital / {naux} auxiliary functions "
                  f"({memory_gb:.2f} GB for the B tensor)")
        M = integrals.coulomb_2c(mol)
        if mol.df_metric_tol is not None:
            X, _, info = invsqrt_matrix(M, rtol=mol.df_metric_tol)
            if info["n_discarded"] > 0 and mol.verbose:
                print(f"   Removed {info['n_discarded']} near-linearly-dependent auxiliary functions.")
        else:
            X = cholesky_metric_inverse_sqrt(M)
        self.Mm12 = X
        Pmn = integrals.coulomb_3c(mol)
        B = np.tensordot(X, Pmn, axes=([1], [0]))
        return B

    def _build_B_cholesky(self):
        """Pivoted Cholesky decomposition of (mn|rs) computed on the fly from the diagonal and
        single shell-pair columns (never the full four-index array). With
        molecule.cholesky_pivots set, the decomposition uses those pivots instead (for finite
        differences that must share the pivots of a reference geometry)."""
        mol = self.molecule
        nbf = mol.nbf
        first, sizes = zip(*mol.basis.shell_first_and_size)
        shell_of = np.repeat(np.arange(len(first)), sizes)
        # Store only unique AO pairs. This halves the projection work and vector
        # storage without changing the (unweighted) pivot criterion.
        pair_r, pair_s = np.triu_indices(nbf)
        npair = len(pair_r)
        columns = OrderedDict()
        cache_bytes = 0
        cache_limit = 64 * 1024**2

        def column(r, s):
            """(mn|rs) as a vector over the pairs mn (cached per shell pair)."""
            nonlocal cache_bytes
            if shell_of[r] > shell_of[s]:
                r, s = s, r
            key = (shell_of[r], shell_of[s])
            if key not in columns:
                block = integrals.coulomb_4c_shell_column(mol, *key)[pair_r, pair_s]
                while columns and cache_bytes + block.nbytes > cache_limit:
                    _, old = columns.popitem(last=False)
                    cache_bytes -= old.nbytes
                columns[key] = block
                cache_bytes += block.nbytes
            columns.move_to_end(key)
            return columns[key][:, r - first[key[0]], s - first[key[1]]]

        pair_index = np.empty((nbf, nbf), dtype=np.intp)
        pair_index[pair_r, pair_s] = np.arange(npair)
        pair_index[pair_s, pair_r] = np.arange(npair)

        if mol.cholesky_pivots is not None:
            pivots = [(int(r), int(s)) for r, s in mol.cholesky_pivots]
            A = np.array([column(r, s) for r, s in pivots])                      # (J|mn)
            M = A[:, [pair_index[r, s] for r, s in pivots]]                      # (J|K)
            Lm = np.linalg.cholesky(M)
            B = scipy.linalg.solve_triangular(Lm, A, lower=True)
        else:
            diag = integrals.coulomb_4c_diagonal(mol)[pair_r, pair_s].copy()
            V = np.empty((4 * nbf, npair))  # Cholesky vectors (rows), grown as needed
            pivots = []
            while len(pivots) < npair:
                p = int(np.argmax(diag))
                if diag[p] <= mol.cholesky_tol:
                    break
                k = len(pivots)
                if k == V.shape[0]:
                    V = np.concatenate([V, np.empty_like(V)])
                r, s = int(pair_r[p]), int(pair_s[p])
                V[k] = (column(r, s) - V[:k, p] @ V[:k]) / np.sqrt(diag[p])
                pivots.append((r, s))
                diag -= V[k]**2
                diag[diag < 0.0] = 0.0
            B = V[: len(pivots)]
            Lm = B[:, [pair_index[r, s] for r, s in pivots]].T                   # M = Lm Lm^T
        self.pivots = pivots
        self.Mm12 = scipy.linalg.solve_triangular(Lm, np.eye(len(pivots)), lower=True)
        mol.naux = len(pivots)
        if mol.verbose:
            print(f"   Cholesky decomposition of the AO ERIs (tol = {mol.cholesky_tol}): "
                  f"{mol.naux} vectors")
        # Preserve the contiguous public layout used by SCF/MO transformations,
        # without an additional full-size temporary from advanced indexing.
        full = np.empty((len(B), nbf, nbf))
        full[:, pair_r, pair_s] = B
        full[:, pair_s, pair_r] = B
        return full

    # Coulomb and exchange
    def build_J(self, D):
        """Coulomb matrices J[D]_{mn} = (mn|rs) D_{sr} for each density in the list ``D``."""
        if self.direct:
            return self.fitting.build_J(D)
        B = self.B_Pmn
        return [np.tensordot(B, np.tensordot(B, Di.T, axes=([1, 2], [0, 1])), axes=([0], [0])) for Di in D]

    def build_K(self, C):
        """Exchange matrices K_{mn} = \\sum_i (mi|in) for each set of occupied orbitals."""
        if self.direct:
            return self.fitting.build_K(C)
        K = []
        for Ci in C:
            Y = self.B_nPm @ Ci
            K.append(np.tensordot(Y, Y, axes=([1, 2], [1, 2])))
        return K

    def build_K_pair(self, pairs):
        """Exchange matrices K[L R^T + R L^T] for each (L, R) pair, e.g., the first-order
        density change C_v x C_o^T + C_o x^T C_v^T of an orbital rotation."""
        if self.direct:
            return self.fitting.build_K_pair(pairs)
        return [exchange_pair(self.B_nPm, L, R) for L, R in pairs]

    def build_JK(self, C):
        """J and K for each occupied-coefficient matrix in the list ``C``."""
        D = [Ci @ Ci.T for Ci in C]
        return self.build_J(D), self.build_K(C)

    # MO-basis tensors
    def mo_cholesky_vectors(self, C1, C2=None):
        """B^P_{pq} = \\sum_{mn} C1_{mp} B^P_{mn} C2_{nq}, shape (naux, n1, n2)."""
        if C2 is None:
            C2 = C1
        tmp = np.tensordot(self.B_Pmn, C2, axes=([2], [0]))  # P m q
        return np.einsum("mp,Pmq->Ppq", C1, tmp, optimize=True)

    def mo_eri(self, C1, C2=None, C3=None, C4=None):
        """(pq|rs) = \\sum_P B^P_{pq} B^P_{rs} in chemists' notation."""
        C2 = C1 if C2 is None else C2
        C3 = C1 if C3 is None else C3
        C4 = C3 if C4 is None else C4
        B12 = self.mo_cholesky_vectors(C1, C2)
        B34 = B12 if (C3 is C1 and C4 is C2) else self.mo_cholesky_vectors(C3, C4)
        return np.tensordot(B12, B34, axes=([0], [0]))

    # Quantities needed by DF analytic gradients
    def _check_df_metric(self):
        if self.molecule.cholesky_tei:
            raise NotImplementedError("Analytic gradients require density fitting.")
        self.B_Pmn  # make sure Mm12 exists

    def build_metric_inverted_density_contraction(self, D):
        """rho^P = \\sum_Q [M^{-1/2}]_{QP} \\sum_{mn} D_{mn} B^Q_{mn}."""
        if self.direct:
            return self.fitting.fit([D])[0]
        self._check_df_metric()
        bP = np.einsum("Pmn,nm->P", self.B_Pmn, D, optimize=True)
        return self.Mm12.T @ bP

    def build_metric_inverted_mo_block(self, *pairs):
        """Z^P_{ij} = \\sum_Q [M^{-1/2}]_{QP} \\sum_k (C_bra^k)^T B^Q C_ket^k, summed over the
        (C_bra, C_ket) pairs."""
        if self.direct:
            return self.fitting.mo_block(pairs)
        self._check_df_metric()
        raw = self.mo_cholesky_vectors(*pairs[0])
        for Cbra, Cket in pairs[1:]:
            raw += self.mo_cholesky_vectors(Cbra, Cket)
        return np.einsum("QP,Qij->Pij", self.Mm12, raw, optimize=True)


def exchange_pair(B_nPm, L, R):
    """K[L R^T + R L^T] from a B tensor laid out as (nbf, naux, nbf)."""
    K = np.tensordot(B_nPm @ L, B_nPm @ R, axes=([1, 2], [1, 2]))
    return K + K.T


def _shell_max(values, offsets):
    """Largest value per shell (1D) or shell pair (2D) of a per-function array."""
    starts = np.asarray(offsets)[:-1]
    out = np.maximum.reduceat(values, starts, axis=0)
    return np.maximum.reduceat(out, starts, axis=1) if values.ndim == 2 else out


class DirectFitting:
    """Density fitting of the kernel w = 1/r (op 0), erf(omega r)/r (op 1) or
    erfc(omega r)/r (op 2) in its own metric, with the three-center integrals recomputed by
    each build (Schwarz-screened) or, with ``store=True``, kept as B = X (P|w|mn).

    X is the inverse Cholesky factor of the metric M (eigenvalue cutoff 1e-9 if M is
    numerically singular, as attenuated metrics often are); M^-1 = X^T X is always applied
    as X then X^T, since forming it squares the condition number.
    """

    def __init__(self, molecule, op=0, omega=0.0, store=False, threshold=1.0e-12):
        mol = self.molecule = molecule
        self.op, self.omega, self.threshold = op, omega, threshold
        two_center = (integrals.coulomb_2c, integrals.erf_coulomb_2c, integrals.erfc_coulomb_2c)[op]
        M = two_center(mol, *([omega] if op else []))
        try:
            self.X = cholesky_metric_inverse_sqrt(M)
        except ValueError:
            s, U = np.linalg.eigh(M)
            keep = s > 1.0e-9
            self.X = (U[:, keep] / np.sqrt(s[keep])).T
        self.aux_bound = _shell_max(np.sqrt(np.abs(np.diag(M))), mol.auxiliary_basis.shell_offsets)
        pair = np.sqrt(np.abs(integrals.coulomb_4c_diagonal(mol)))
        self.pair_bound = np.ascontiguousarray(_shell_max(pair, mol.basis.shell_offsets))
        self.B_nPm = None
        if store:
            three_center = (integrals.coulomb_3c, integrals.erf_coulomb_3c, integrals.erfc_coulomb_3c)[op]
            Pmn = three_center(mol, *([omega] if op else []))
            self.B_nPm = np.ascontiguousarray(np.tensordot(self.X, Pmn, axes=([1], [0])).transpose((2, 0, 1)))

    def _pass(self, D=None, coef=None, C=None):
        nbf, naux = self.molecule.nbf, self.molecule.naux
        D = np.zeros((0, nbf, nbf)) if D is None else np.ascontiguousarray(D, dtype=float)
        coef = np.zeros((0, naux)) if coef is None else np.ascontiguousarray(coef, dtype=float)
        C = np.zeros((nbf, 0)) if C is None else np.ascontiguousarray(C, dtype=float)
        return integrals.ints.coulomb_3c_direct(
            self.molecule.auxiliary_basis, self.molecule.basis, D, coef, C, self.pair_bound,
            self.aux_bound, self.threshold, self.op, self.omega)

    def _half_transformed(self, C):
        """B^P_{mk} = sum_Q X_PQ (Q|w|m n) C_nk, shape (nfit, nbf, ncol)."""
        if self.B_nPm is not None:
            return np.ascontiguousarray((self.B_nPm @ C).transpose((1, 0, 2)))
        return np.tensordot(self.X, self._pass(C=C)[2], axes=([1], [0]))

    def fit(self, D):
        """Metric-inverted density contractions rho = M^-1 (P|w|D) of each density."""
        return [self.X.T @ (self.X @ g) for g in self._pass(D=np.array(D))[0]]

    def build_J(self, D):
        return list(self._pass(coef=np.array(self.fit(D)))[1])

    def build_K(self, C):
        B = self._half_transformed(np.hstack(C))
        K, start = [], 0
        for c in C:
            b = B[:, :, start : start + c.shape[1]]
            start += c.shape[1]
            K.append(np.tensordot(b, b, axes=([0, 2], [0, 2])))
        return K

    def build_K_pair(self, pairs):
        B = self._half_transformed(np.hstack([M for pair in pairs for M in pair]))
        K, start = [], 0
        for L, R in pairs:
            BL = B[:, :, start : start + L.shape[1]]
            BR = B[:, :, start + L.shape[1] : start + L.shape[1] + R.shape[1]]
            start += L.shape[1] + R.shape[1]
            k = np.tensordot(BL, BR, axes=([0, 2], [0, 2]))
            K.append(k + k.T)
        return K

    def mo_block(self, pairs):
        """Z^P_ij = sum_Q [M^-1]_PQ (Q|w|ij) summed over the (C_bra, C_ket) pairs."""
        Z = sum(np.einsum("mi,Pmj->Pij", Cbra, self._half_transformed(Cket), optimize=True)
                for Cbra, Cket in pairs)
        return np.tensordot(self.X.T, Z, axes=([1], [0]))

    def gradient(self, D, rho, U, C):
        """sum W^P_mn d(P|w|mn)/dR, W = D rho - U C^T, flat (3 natoms,)."""
        mol = self.molecule
        return integrals.ints.coulomb_3c_deriv_factored(
            mol.auxiliary_basis, mol.basis, np.ascontiguousarray(D), np.ascontiguousarray(rho),
            np.ascontiguousarray(U), np.ascontiguousarray(C), mol.atoms, self.pair_bound,
            self.aux_bound, self.threshold, self.op, self.omega)

    def gradient_2c(self, W2):
        """sum W_PQ d(P|w|Q)/dR, flat (3 natoms,)."""
        if self.op == 0:
            return integrals.coulomb_2c_deriv(self.molecule, W2)
        return integrals.attenuated_coulomb_2c_deriv(self.molecule, W2, self.omega if self.op == 1 else -self.omega)

    def exchange_gradient(self, occupied, D=None, rho=None, scale=1.0):
        """Derivative of the fitted Coulomb energy 1/2 rho M rho (when D and rho are given)
        minus scale/2 sum_s sum_ij (ij|w|ji) over the occupied sets, as (natoms, 3)."""
        nbf, nfit = self.molecule.nbf, self.X.shape[1]
        D = np.zeros((nbf, nbf)) if D is None else D
        rho = np.zeros(nfit) if rho is None else rho
        W2 = -0.5 * np.outer(rho, rho)
        U, C = [], []
        for Cocc in occupied if scale != 0.0 else ():
            Z = self.mo_block([(Cocc, Cocc)])
            W2 += 0.5 * scale * np.einsum("Pij,Qji->PQ", Z, Z, optimize=True)
            U.append(scale * np.einsum("mi,Pij->Pmj", Cocc, Z, optimize=True))
            C.append(Cocc)
        U = np.concatenate(U, axis=2) if U else np.zeros((nfit, nbf, 0))
        C = np.hstack(C) if C else np.zeros((nbf, 0))
        g = self.gradient(D, rho, U, C) + self.gradient_2c(W2)
        return g.reshape(self.molecule.natoms, 3)
