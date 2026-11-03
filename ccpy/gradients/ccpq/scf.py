"""Conventional (unfactorized) closed-shell RHF and MO integrals for the CC(P;Q) gradient.

The gradient notes (ccpq_gradients_revised.pdf, Sec. 8) are written for full four-center
integrals; the native ccpy.scf solvers are density fitted or Cholesky factorized, so this
small solver supplies a reference whose energy is exactly the one the four-center gradient
differentiates. It stores the AO integrals (O(N^4) memory) and is meant for the small
systems the determinant backend can treat.
"""

from dataclasses import dataclass

import numpy as np

from ccpy.integrals import integrals


def align_orbital_phases(reference, displaced):
    """Align only the signs of canonical MOs across nearby geometries.

    This is a pure orbital-phase gauge: no mixing, rotations, occupation
    changes or root reordering. Cross-geometry AO overlaps keep amplitude and
    EOM warm starts continuous. Orbital degeneracies/crossings need separate
    orbital matching and are outside this helper's scope.
    """
    if reference.C.shape[1] != displaced.C.shape[1]:
        raise ValueError('Phase alignment requires equal MO counts')
    overlap = integrals.overlap(displaced.mol, basis1=reference.mol.basis, basis2=displaced.mol.basis)
    diagonal = np.diag(reference.C.T @ overlap @ displaced.C)
    if np.any(abs(diagonal) < 1e-6):
        raise ValueError('Canonical orbitals cannot be matched by phase alone')
    signs = np.where(diagonal < 0., -1., 1.)
    displaced.C = displaced.C * signs[None, :]
    return signs


@dataclass
class ConventionalRHF:
    mol: object
    C: np.ndarray          # (nbf, nmo)
    eps: np.ndarray        # (nmo,)
    nocc: int
    e_tot: float
    hcore_ao: np.ndarray
    eri_ao: np.ndarray     # (mn|rs)

    @property
    def nmo(self):
        return self.C.shape[1]

    def mo_integrals(self):
        """h[p, q] and chemists' eri[a, b, c, d] = (ab|cd) over all MOs."""
        C = self.C
        h = C.T @ self.hcore_ao @ C
        eri = np.einsum("mnrs,mp->pnrs", self.eri_ao, C, optimize=True)
        eri = np.einsum("pnrs,nq->pqrs", eri, C, optimize=True)
        eri = np.einsum("pqrs,rt->pqts", eri, C, optimize=True)
        eri = np.einsum("pqts,su->pqtu", eri, C, optimize=True)
        return h, eri


def run_rhf(mol, e_tol=1.0e-12, g_tol=1.0e-10, max_iter=200, guess=None, verbose=False):
    """Closed-shell RHF with four-center integrals and DIIS. `guess` is an (nbf, nmo) MO
    coefficient matrix (e.g. from a nearby geometry) used to build the starting density."""
    if mol.multiplicity != 1 or mol.nelectrons % 2:
        raise ValueError("ConventionalRHF needs a closed-shell singlet.")
    S = integrals.overlap(mol)
    H = integrals.hcore(mol)
    eri = integrals.coulomb_4c(mol)
    X = mol.Xorth
    nocc = mol.nelectrons // 2

    def diag(F):
        e, Cp = np.linalg.eigh(X.T @ F @ X)
        return e, X @ Cp

    if guess is None:
        eps, C = diag(H)
    else:
        C = guess
    Fs, errs = [], []
    e_old = 0.0
    for it in range(max_iter):
        Cocc = C[:, :nocc]
        Dm = Cocc @ Cocc.T
        J = np.einsum("mnrs,rs->mn", eri, Dm, optimize=True)
        K = np.einsum("mrns,rs->mn", eri, Dm, optimize=True)
        F = H + 2.0 * J - K
        e = np.sum(Dm * (H + F)) + mol.nuclear_repulsion
        err = X.T @ (F @ Dm @ S - S @ Dm @ F) @ X
        gnorm = np.abs(err).max()
        if verbose:
            print(f"   RHF {it:3d} {e:20.12f} {gnorm:10.2e}")
        if it > 0 and abs(e - e_old) < e_tol and gnorm < g_tol:
            break
        e_old = e
        Fs.append(F)
        errs.append(err)
        Fs, errs = Fs[-8:], errs[-8:]
        if len(Fs) > 1:
            m = len(Fs)
            B = -np.ones((m + 1, m + 1))
            B[m, m] = 0.0
            for i in range(m):
                for j in range(m):
                    B[i, j] = np.sum(errs[i] * errs[j])
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            F = sum(ci * Fi for ci, Fi in zip(c, Fs))
        eps, C = diag(F)
    else:
        raise RuntimeError("Conventional RHF did not converge.")
    # canonical orbitals of the converged Fock matrix
    eps, C = diag(F)
    return ConventionalRHF(mol, C, eps, nocc, e, H, eri)
