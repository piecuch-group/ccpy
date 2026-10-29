"""Unrestricted Hartree-Fock.

Adapted from forte2 (https://github.com/evangelistalab/forte2, LGPL-3.0); see
THIRD_PARTY_NOTICES.md.
"""

import numpy as np

from ccpy.integrals.symmetry.sym_utils import COTTON_LABELS
from ccpy.scf.scf_base import SCFBase
from ccpy.scf.scf_utils import guess_mix, smallest_gap_same_irrep


class UHF(SCFBase):
    """Density-fitted unrestricted Hartree-Fock.

    Parameters
    ----------
    guess_mix : bool, optional, default=False
        For ms = 0, rotate the HOMO and LUMO of each spin in opposite directions to
        break the alpha/beta degeneracy of the guess (e.g., for open-shell singlets or
        stretched bonds). With symmetry, the closest occupied/virtual pair of the same
        irrep is mixed instead if the HOMO and LUMO differ in symmetry.

    See ``SCFBase`` for the remaining options.
    """

    def __init__(self, molecule, guess_mix=False, **kwargs):
        self.guess_mix = guess_mix
        super().__init__(molecule, **kwargs)

    def _initial_guess(self, H):
        eps, C, irreps = self._eigh(self._guess_hamiltonian(H))
        return [eps, eps], [C, C], [irreps, irreps]

    def _mix_guess(self, eps, C, irreps):
        if self.na != self.nb or not self.guess_mix:
            return C
        mixed, messages = [], set()
        for spin in (0, 1):
            pair, message = _guess_mix_pair(eps[spin], irreps[spin], self.na, self.mol.point_group)
            if message is not None:
                messages.add(message)
            mixed.append(C[spin] if pair is None else guess_mix(C[spin], *pair)[spin])
        for message in messages:
            print(f"   WARNING: {message}")
        return mixed

    def _build_fock(self, H, df, S):
        (Ja, Jb), K = df.build_JK([self.C[0][:, : self.na], self.C[1][:, : self.nb]])
        F = [H + Ja + Jb - k for k in K]
        return F, F

    def _build_density_matrix(self):
        Ca = self.C[0][:, : self.na]
        Cb = self.C[1][:, : self.nb]
        return [Ca @ Ca.T, Cb @ Cb.T]

    def _build_ao_grad(self, S, F):
        return np.hstack(
            [
                (self.Xorth.T @ (f @ d @ S - S @ d @ f) @ self.Xorth).ravel()
                for d, f in zip(self.D, F)
            ]
        )

    def _energy(self, H, F):
        return 0.5 * (
            np.sum((self.D[0] + self.D[1]) * H)
            + np.sum(self.D[0] * F[0])
            + np.sum(self.D[1] * F[1])
        )

    def _spin(self, S):
        # <S^2> = ms(ms+1) + N_b - sum_ij |<i_a|j_b>|^2
        Sab = self.C[0][:, : self.na].T @ S @ self.C[1][:, : self.nb]
        return self.ms * (self.ms + 1) + self.nb - np.sum(Sab**2)

    def _diis_update(self, diis, F, AO_grad):
        n2 = self.nbf**2
        F_flat = diis.update(np.hstack([f.ravel() for f in F]), AO_grad)
        return [F_flat[:n2].reshape(self.nbf, self.nbf), F_flat[n2:].reshape(self.nbf, self.nbf)]

    def _apply_level_shift(self, F, S):
        if self._current_level_shift is None or all(ls < 1e-4 for ls in self._current_level_shift):
            return F
        return [
            f + ls * (S - S @ d @ S) for ls, f, d in zip(self._current_level_shift, F, self.D)
        ]

    def _print_orbital_energies(self):
        print("\n   Orbital energies (Eh)")
        self._print_orbital_set("Alpha occupied", self.eps[0], self.irrep_labels[0], 0, self.na)
        self._print_orbital_set("Alpha virtual", self.eps[0], self.irrep_labels[0], self.na, self.nmo)
        self._print_orbital_set("Beta occupied", self.eps[1], self.irrep_labels[1], 0, self.nb)
        self._print_orbital_set("Beta virtual", self.eps[1], self.irrep_labels[1], self.nb, self.nmo)

    def gradient(self):
        """Density-fitted analytic UHF nuclear gradient, shape (natoms, 3), Eh/bohr."""
        from ccpy.scf.hf_gradient import uhf_gradient

        return uhf_gradient(self)

    # PySCF-like views (alpha, beta)
    @property
    def mo_coeff(self):
        return (self.C[0], self.C[1])

    @property
    def mo_energy(self):
        return (self.eps[0], self.eps[1])

    @property
    def mo_occ(self):
        occ = np.zeros((2, self.nmo))
        occ[0, : self.na] = 1.0
        occ[1, : self.nb] = 1.0
        return occ


def _guess_mix_pair(eps, irreps, nocc, point_group):
    """Orbitals that guess_mix rotates, and a warning if they are not the HOMO and LUMO."""
    homo, lumo = nocc - 1, nocc
    pair = smallest_gap_same_irrep(eps, irreps, nocc)
    if pair == (homo, lumo):
        return pair, None
    names = {index: label for label, index in COTTON_LABELS[point_group].items()}
    if pair is None:
        return None, (
            f"guess_mix: no occupied and virtual orbitals share a {point_group} irrep, so "
            "the initial guess is not mixed. Run with symmetry=False to mix the HOMO and LUMO."
        )
    return pair, (
        f"guess_mix: the HOMO ({names[irreps[homo]]}) and LUMO ({names[irreps[lumo]]}) are in "
        f"different irreps; mixing orbitals {pair[0]} ({names[irreps[pair[0]]]}) and "
        f"{pair[1]} ({names[irreps[pair[1]]]}) instead."
    )
