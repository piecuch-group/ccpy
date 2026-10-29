import numpy as np

from ccpy.scf.rhf import RHF
from ccpy.scf.scf_base import SCFBase
from ccpy.scf.uhf import UHF


class ROHF(SCFBase):
    """Density-fitted restricted open-shell Hartree-Fock.

    The orbitals are eigenfunctions of the Guest-Saunders canonical Fock operator, the
    same canonicalization forte2 uses. ``mo_energy`` holds its eigenvalues, and the
    alpha and beta Fock matrices of the converged determinant are kept in ``Fa``/``Fb``.
    See ``SCFBase`` for the options.
    """

    _initial_guess = RHF._initial_guess
    _diis_update = RHF._diis_update
    _energy = UHF._energy

    def _build_fock(self, H, df, S):
        Cocc = self.C[0][:, : self.na]
        (Ja, Jb), K = df.build_JK([Cocc, self.C[0][:, : self.nb]])
        F = [H + Ja + Jb - k for k in K]
        self.Fa, self.Fb = F
        return F, self._build_canonical_fock(F, S)

    def _build_canonical_fock(self, F, S):
        # projectors onto the doubly occupied, singly occupied, and virtual spaces
        U_core = self.D[1] @ S
        U_open = (self.D[0] - self.D[1]) @ S
        U_virt = np.eye(self.nbf) - self.D[0] @ S
        F_cs = 0.5 * (F[0] + F[1])

        def project(u, v, f):
            return u.T @ f @ v

        # diagonal blocks are halved because fock + fock.T is taken below
        fock = 0.5 * (project(U_core, U_core, F_cs) + project(U_open, U_open, F_cs) + project(U_virt, U_virt, F_cs))
        fock += project(U_open, U_core, F[1])
        fock += project(U_open, U_virt, F[0])
        fock += project(U_virt, U_core, F_cs)
        return [fock + fock.T]

    def _build_density_matrix(self):
        Ca = self.C[0][:, : self.na]
        Cb = self.C[0][:, : self.nb]
        return [Ca @ Ca.T, Cb @ Cb.T]

    def _build_ao_grad(self, S, F):
        Deff = 0.5 * (self.D[0] + self.D[1])
        grad = F[0] @ Deff @ S - S @ Deff @ F[0]
        return self.Xorth.T @ grad @ self.Xorth

    def _apply_level_shift(self, F, S):
        if self._current_level_shift is None or self._current_level_shift < 1e-4:
            return F
        return [F[0] + self._current_level_shift * (S - S @ self.D[0] @ S)]

    def _spin(self, S):
        return self.ms * (self.ms + 1)

    def _print_orbital_energies(self):
        print("\n   Orbital energies (Eh, Guest-Saunders canonicalization)")
        labels = self.irrep_labels[0]
        self._print_orbital_set("Doubly occupied", self.eps[0], labels, 0, self.nb)
        self._print_orbital_set("Singly occupied", self.eps[0], labels, self.nb, self.na)
        self._print_orbital_set("Virtual", self.eps[0], labels, self.na, self.nmo)

    # PySCF-like views
    @property
    def mo_coeff(self):
        return self.C[0]

    @property
    def mo_energy(self):
        return self.eps[0]

    @property
    def mo_occ(self):
        occ = np.zeros(self.nmo)
        occ[: self.na] = 1.0
        occ[: self.nb] += 1.0
        return occ
