"""Restricted open-shell Hartree-Fock.

Adapted from forte2 (https://github.com/evangelistalab/forte2, LGPL-3.0); see
THIRD_PARTY_NOTICES.md.
"""

import numpy as np

from ccpy.scf.rhf import RHF
from ccpy.scf.scf_base import SCFBase
from ccpy.scf.uhf import UHF


class ROHF(SCFBase):
    """Density-fitted restricted open-shell Hartree-Fock.

    The orbitals are eigenfunctions of a canonical Fock operator: Guest-Saunders by default
    (as in PySCF), or canonicalization="gamess" for GAMESS's default coefficients
    (needed to reproduce frozen-core results based on GAMESS ROHF orbitals). ``mo_energy`` holds its eigenvalues, and the
    alpha and beta Fock matrices of the converged determinant are kept in ``Fa``/``Fb``.
    See ``SCFBase`` for the options.
    """

    # (alpha, beta) Fock coefficients of the doubly occupied, singly occupied, and virtual
    # diagonal blocks of the canonical Fock operator. Only these blocks fix the orbitals
    # within each space; frozen-core results depend on the doubly occupied one.
    CANONICALIZATIONS = {
        "guest-saunders": ((0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
        "gamess": ((-0.5, 1.5), (0.5, 0.5), (1.5, -0.5)),
    }

    def __init__(self, molecule, canonicalization="guest-saunders", **kwargs):
        if canonicalization not in self.CANONICALIZATIONS:
            raise ValueError(f"canonicalization must be one of {list(self.CANONICALIZATIONS)}.")
        self.canonicalization = canonicalization
        super().__init__(molecule, **kwargs)

    def _init_options(self):
        return {**super()._init_options(), "canonicalization": self.canonicalization}

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
        (ac, bc), (ao, bo), (av, bv) = self.CANONICALIZATIONS[self.canonicalization]
        fock = 0.5 * (project(U_core, U_core, ac * F[0] + bc * F[1])
                      + project(U_open, U_open, ao * F[0] + bo * F[1])
                      + project(U_virt, U_virt, av * F[0] + bv * F[1]))
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
        print(f"\n   Orbital energies (Eh, {self.canonicalization} canonicalization)")
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
