import numpy as np

from ccpy.scf.scf_base import SCFBase


class RHF(SCFBase):
    """Density-fitted restricted Hartree-Fock (closed shells).

    See ``SCFBase`` for the options. After ``kernel()`` the PySCF-like attributes
    ``mo_coeff``, ``mo_energy``, ``mo_occ`` and ``e_tot`` are available.
    """

    def _validate_state(self):
        if self.mol.multiplicity != 1:
            raise ValueError("RHF requires a singlet; use ROHF or UHF for open shells.")

    def _initial_guess(self, H):
        eps, C, irreps = self._eigh(self._guess_hamiltonian(H))
        return [eps], [C], [irreps]

    def _build_fock(self, H, df, S):
        J, K = df.build_JK([self.C[0][:, : self.na]])
        F = H + 2.0 * J[0] - K[0]
        return [F], [F]

    def _build_density_matrix(self):
        Cocc = self.C[0][:, : self.na]
        return [Cocc @ Cocc.T]

    def _build_ao_grad(self, S, F):
        grad = F[0] @ self.D[0] @ S - S @ self.D[0] @ F[0]
        return self.Xorth.T @ grad @ self.Xorth

    def _energy(self, H, F):
        return np.sum(self.D[0] * (H + F[0]))

    def _spin(self, S):
        return 0.0

    def _diis_update(self, diis, F, AO_grad):
        return [diis.update(F[0], AO_grad)]

    def _apply_level_shift(self, F, S):
        if self._current_level_shift is None or self._current_level_shift < 1e-4:
            return F
        return [F[0] + self._current_level_shift * (S - S @ self.D[0] @ S)]

    def _print_orbital_energies(self):
        print("\n   Orbital energies (Eh)")
        self._print_orbital_set("Doubly occupied", self.eps[0], self.irrep_labels[0], 0, self.na)
        self._print_orbital_set("Virtual", self.eps[0], self.irrep_labels[0], self.na, self.nmo)

    def gradient(self):
        """Density-fitted analytic RHF nuclear gradient, shape (natoms, 3), Eh/bohr."""
        from ccpy.scf.hf_gradient import rhf_gradient

        return rhf_gradient(self)

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
        occ[: self.na] = 2.0
        return occ
