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

import numpy as np
import scipy.linalg

from ccpy.integrals import integrals
from ccpy.integrals.linalg import invsqrt_matrix, cholesky_metric_inverse_sqrt


class DensityFitting:
    """B tensor of a Molecule (built on first use), plus J/K builds and MO transformations.

    B_Pmn : the B tensor, shape (naux, nbf, nbf)
    Mm12 : X with B = X (P|mn) and X^T X = M^{-1}, shape (naux, naux). For density fitting
           M = (P|Q); for the Cholesky decomposition the "auxiliary functions" are the pivot
           pairs J = (r s) and M = (J|K) (incomplete pivoted Cholesky = Nystrom: (mn|rs) ~
           sum_JK (mn|J) M^-1_JK (K|rs)).
    pivots : the Cholesky pivot function pairs [(r, s), ...] (Cholesky only)
    """

    def __init__(self, molecule):
        self.molecule = molecule
        self.nbf = molecule.nbf
        self._B_Pmn = None
        self._B_nPm = None
        self.Mm12 = None
        self.pivots = None

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
        columns = {}

        def column(r, s):
            """(mn|rs) as a vector over the pairs mn (cached per shell pair)."""
            key = (shell_of[r], shell_of[s])
            if key not in columns:
                if len(columns) > 64:
                    columns.clear()
                columns[key] = integrals.coulomb_4c_shell_column(mol, *key)
            return columns[key][:, :, r - first[key[0]], s - first[key[1]]].reshape(nbf * nbf)

        if mol.cholesky_pivots is not None:
            pivots = [(int(r), int(s)) for r, s in mol.cholesky_pivots]
            A = np.array([column(r, s) for r, s in pivots])                      # (J|mn)
            M = A[:, [r * nbf + s for r, s in pivots]]                           # (J|K)
            Lm = np.linalg.cholesky(M)
            B = scipy.linalg.solve_triangular(Lm, A, lower=True)
        else:
            diag = integrals.coulomb_4c_diagonal(mol).reshape(nbf * nbf).copy()
            V = np.empty((4 * nbf, nbf * nbf))  # Cholesky vectors (rows), grown as needed
            pivots = []
            while len(pivots) < nbf * nbf:
                p = int(np.argmax(diag))
                if diag[p] <= mol.cholesky_tol:
                    break
                k = len(pivots)
                if k == V.shape[0]:
                    V = np.concatenate([V, np.empty_like(V)])
                r, s = divmod(p, nbf)
                V[k] = (column(r, s) - V[:k, p] @ V[:k]) / np.sqrt(diag[p])
                pivots.append((r, s))
                diag -= V[k]**2
                diag[diag < 0.0] = 0.0
            B = V[: len(pivots)]
            Lm = B[:, [r * nbf + s for r, s in pivots]].T                        # M = Lm Lm^T
        self.pivots = pivots
        self.Mm12 = scipy.linalg.solve_triangular(Lm, np.eye(len(pivots)), lower=True)
        mol.naux = len(pivots)
        if mol.verbose:
            print(f"   Cholesky decomposition of the AO ERIs (tol = {mol.cholesky_tol}): "
                  f"{mol.naux} vectors")
        return B.reshape((-1, nbf, nbf))

    # Coulomb and exchange
    def build_J(self, D):
        """Coulomb matrices J[D]_{mn} = (mn|rs) D_{sr} for each density in the list ``D``."""
        B = self.B_Pmn
        return [np.tensordot(B, np.tensordot(B, Di.T, axes=([1, 2], [0, 1])), axes=([0], [0])) for Di in D]

    def build_K(self, C):
        """Exchange matrices K_{mn} = \\sum_i (mi|in) for each set of occupied orbitals."""
        K = []
        for Ci in C:
            Y = self.B_nPm @ Ci
            K.append(np.tensordot(Y, Y, axes=([1, 2], [1, 2])))
        return K

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
        self._check_df_metric()
        bP = np.einsum("Pmn,nm->P", self.B_Pmn, D, optimize=True)
        return self.Mm12.T @ bP

    def build_metric_inverted_mo_block(self, *pairs):
        """Z^P_{ij} = \\sum_Q [M^{-1/2}]_{QP} \\sum_k (C_bra^k)^T B^Q C_ket^k, summed over the
        (C_bra, C_ket) pairs."""
        self._check_df_metric()
        raw = self.mo_cholesky_vectors(*pairs[0])
        for Cbra, Cket in pairs[1:]:
            raw += self.mo_cholesky_vectors(Cbra, Cket)
        return np.einsum("QP,Qij->Pij", self.Mm12, raw, optimize=True)
