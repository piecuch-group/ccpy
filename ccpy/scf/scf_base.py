"""Common machinery of the RHF/ROHF/UHF solvers.

SAP (or core-Hamiltonian) initial guess, Pulay DIIS on the FDS - SDF orbital gradient,
optional level shifting, point-group blocking of the Fock diagonalization, and optional
constraints on the irrep occupations. Two-electron contributions are built from the
density-fitted (or Cholesky) B tensor of the Molecule.

Usage mirrors PySCF, which ccpy users are used to:

    mol = Molecule(geometry, "cc-pvdz", multiplicity=1, symmetry=True)
    mf = RHF(mol)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=1)

Adapted from forte2 (https://github.com/evangelistalab/forte2, LGPL-3.0); see
THIRD_PARTY_NOTICES.md.
"""

import time
from collections import deque

import numpy as np

from ccpy.integrals.molecule import Molecule
from ccpy.integrals.symmetry.sym_utils import COTTON_LABELS, irrep_product
from ccpy.scf.occupations import OccupationConstraints, validate_occupation_options
from ccpy.scf.scf_utils import sap_guess_hamiltonian

# DIIS extrapolation coefficients become numerical noise above this condition number
_MAX_B_CONDITION = 1.0e12


class SCFDIIS:
    """Pulay DIIS (CDIIS) on the orthonormal-basis orbital gradient FDS - SDF."""

    def __init__(self, diis_start=1, diis_size=8, diis_min=2):
        self.do_diis = diis_size > 0 and diis_start >= 0
        self.diis_start = diis_start
        self.diis_min = min(diis_min, max(diis_size, 1))
        self.p_diis = deque(maxlen=max(diis_size, 1))
        self.e_diis = deque(maxlen=max(diis_size, 1))
        self.iter = 0
        self.status = ""

    def update(self, p, e):
        self.status = ""
        if not self.do_diis:
            return p
        self.iter += 1
        if self.iter < self.diis_start:
            return p
        self.p_diis.append(p.copy())
        self.e_diis.append(e.copy())
        self.status = "S"
        if len(self.p_diis) < self.diis_min:
            return p
        B = self._build_B()
        # drop the oldest vectors until B is well conditioned
        while not np.isfinite(np.linalg.cond(B)) or np.linalg.cond(B) > _MAX_B_CONDITION:
            if len(self.p_diis) <= self.diis_min:
                return p
            self.p_diis.popleft()
            self.e_diis.popleft()
            B = self._build_B()
        rhs = np.zeros(B.shape[0])
        rhs[-1] = -1.0
        x = np.linalg.solve(B, rhs)
        self.status = "S/E"
        return sum(x[i] * self.p_diis[i] for i in range(len(self.p_diis)))

    def _build_B(self):
        n = len(self.e_diis)
        B = -np.ones((n + 1, n + 1))
        B[-1, -1] = 0.0
        for i in range(n):
            for j in range(i, n):
                B[i, j] = B[j, i] = np.dot(self.e_diis[i].ravel(), self.e_diis[j].ravel())
        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
        return B


class SCFBase:
    """Base class of the RHF, ROHF, and UHF solvers.

    Parameters
    ----------
    molecule : Molecule
        Geometry, basis sets, charge, multiplicity, and point group.
    e_tol : float, optional, default=1e-10
        Energy convergence threshold.
    d_tol : float, optional, default=1e-7
        Convergence threshold on the change of the density matrix (Frobenius norm).
    maxiter : int, optional, default=100
    guess : str, optional, default="minao"
        Initial guess, "minao" (superposition of atomic potentials) or "hcore".
    diis_size : int, optional, default=8
        Number of DIIS vectors (0 turns DIIS off).
    diis_start : int, optional, default=1
        Iteration at which DIIS vectors start being collected.
    level_shift : float | tuple, optional
        Virtual-orbital level shift (Eh). UHF accepts an (alpha, beta) pair. The shift
        is turned off once the energy change drops below ``level_shift_thresh``.
    level_shift_thresh : float, optional, default=1e-5
    target_symmetry : str | int, optional
        Irrep of the HF determinant (requires symmetry).
    irrep_occupations : dict, optional
        {irrep: number of electrons} or {irrep: (n_alpha, n_beta)}, akin to PySCF's
        ``irrep_nelec``. Irreps that are not listed are left empty.
    die_if_not_converged : bool, optional, default=True
    verbose : int, optional, default=1
        0 suppresses the iteration printout.
    """

    def __init__(
        self,
        molecule,
        e_tol=1.0e-10,
        d_tol=1.0e-07,
        maxiter=100,
        guess="minao",
        diis_size=8,
        diis_start=1,
        diis_min=2,
        level_shift=None,
        level_shift_thresh=1.0e-05,
        target_symmetry=None,
        irrep_occupations=None,
        die_if_not_converged=True,
        verbose=1,
    ):
        if not isinstance(molecule, Molecule):
            raise TypeError("SCF solvers take a ccpy.integrals.Molecule.")
        validate_occupation_options(target_symmetry, irrep_occupations)
        self.mol = molecule
        self.e_tol = e_tol
        self.d_tol = d_tol
        self.maxiter = maxiter
        self.guess = guess
        self.diis_size = diis_size
        self.diis_start = diis_start
        self.diis_min = diis_min
        self.level_shift = level_shift
        self.level_shift_thresh = level_shift_thresh
        self.target_symmetry = target_symmetry
        self.irrep_occupations = irrep_occupations
        self.die_if_not_converged = die_if_not_converged
        self.verbose = verbose

        self.method = type(self).__name__.upper()
        self.nel = molecule.nelectrons
        self.na = molecule.nalpha
        self.nb = molecule.nbeta
        self.ms = 0.5 * (self.na - self.nb)
        self.nbf = molecule.nbf
        self.nmo = molecule.nmo
        self.Xorth = molecule.Xorth
        self._validate_state()
        self._validate_level_shift()

        self.C = None  # optional user-supplied guess, list of coefficient matrices
        self.converged = False
        self.e_tot = None

    # set-up
    def _validate_state(self):
        pass

    def _validate_level_shift(self):
        ls = self.level_shift
        if ls is None:
            return
        if isinstance(ls, tuple):
            if self.method != "UHF" or len(ls) != 2:
                raise ValueError("A tuple level_shift is only valid for UHF, as (alpha, beta).")
        elif ls < 0.0:
            raise ValueError("level_shift must be non-negative.")
        elif self.method == "UHF":
            self.level_shift = (float(ls), float(ls))

    def _set_occupation_constraints(self):
        if self.target_symmetry is None and self.irrep_occupations is None:
            self._occupations = None
            return
        self._occupations = OccupationConstraints(
            self.method in ("RHF", "ROHF"),
            self.mol.point_group,
            (self.na, self.nb),
            self.target_symmetry,
            self.irrep_occupations,
        )
        if self.mol.point_group != "C1":
            irreps = self.mol.symmetry_basis.irreps
        else:
            irreps = np.zeros(self.nmo, dtype=int)
        self._occupations.check_capacity(irreps)

    # helpers shared by all references
    def _occupy(self, eps, C, irreps):
        """Reorder each orbital set so that its occupied orbitals come first."""
        if self._occupations is None:
            return eps, C, irreps
        orders = self._occupations.order(eps, irreps)
        return (
            [e[o] for e, o in zip(eps, orders)],
            [c[:, o] for c, o in zip(C, orders)],
            [h[o] for h, o in zip(irreps, orders)],
        )

    def _eigh(self, F):
        """Diagonalize F (by irrep with symmetry) -> energies, AO coefficients, irreps."""
        if self._symmetry_basis is not None:
            return self._symmetry_basis.eigh(F)
        e, c = np.linalg.eigh(self.Xorth.T @ F @ self.Xorth)
        return e, self.Xorth @ c, np.zeros(len(e), dtype=int)

    def _guess_hamiltonian(self, H):
        if self.guess == "minao":
            return sap_guess_hamiltonian(self.mol, H)
        elif self.guess == "hcore":
            return H
        raise ValueError(f"Unknown initial guess type: {self.guess}")

    def _diagonalize_fock(self, F):
        eps, C, irreps = (list(x) for x in zip(*(self._eigh(f) for f in F)))
        return self._occupy(eps, C, irreps)

    def _mix_guess(self, eps, C, irreps):
        return C

    def _guess_occupations(self):
        index = np.arange(self.C[0].shape[1])
        if len(self.C) == 1:
            return [(index < self.na).astype(float) + (index < self.nb)]
        return [(index < n).astype(float) for n in (self.na, self.nb)]

    # main loop
    def kernel(self):
        """Run the SCF iterations and return the total energy."""
        self.run()
        return self.e_tot

    def run(self):
        t_start = time.perf_counter()
        self._set_occupation_constraints()
        self._current_level_shift = self.level_shift

        mol = self.mol
        Vnn = mol.nuclear_repulsion
        S = mol.ints_overlap()
        H = mol.ints_hcore()
        df = mol.df
        self._symmetry_basis = mol.symmetry_basis if mol.point_group != "C1" else None

        if self.verbose:
            print("")
            print(f"   {self.method} Calculation")
            print("   " + "=" * (len(self.method) + 12))
            print(f"   Point group: {mol.point_group}")
            print(f"   Number of electrons: {self.nel} (alpha = {self.na}, beta = {self.nb})")
            print(f"   Charge: {mol.charge}   Multiplicity: {mol.multiplicity}")
            print(f"   Basis functions: {self.nbf} ({self.nmo} independent)")
            if mol.cholesky_tei:
                print(f"   Two-electron integrals: Cholesky (tol = {mol.cholesky_tol})")
            else:
                print(f"   Auxiliary basis functions: {mol.naux}")
            print(f"   Energy / density tolerance: {self.e_tol:.1e} / {self.d_tol:.1e}")

        diis = SCFDIIS(self.diis_start, self.diis_size, self.diis_min)

        self._orbital_irreps = None
        if self.C is None:
            eps, C, irreps = self._occupy(*self._initial_guess(H))
            self.eps, self._orbital_irreps = eps, irreps
            self.C = self._mix_guess(eps, C, irreps)
        elif self._symmetry_basis is not None:
            # A supplied guess may break the symmetry; keep its symmetric part.
            self.C = [
                self._symmetry_basis.adapt(C, n)
                for C, n in zip(self.C, self._guess_occupations())
            ]
        self.D = self._build_density_matrix()
        F, F_canon = self._build_fock(H, df, S)
        self.E = Vnn + self._energy(H, F)
        Eold, Dold = self.E, self.D

        if self.verbose:
            print("")
            print("    Iter            Energy             ΔE        ‖ΔD‖     ‖grad‖      <S^2>   DIIS")
            print("   " + "-" * 82)
        self.converged = False
        for it in range(1, self.maxiter + 1):
            AO_grad = self._build_ao_grad(S, F_canon)
            F_canon = self._diis_update(diis, F_canon, AO_grad)
            F_canon = self._apply_level_shift(F_canon, S)
            self.eps, self.C, self._orbital_irreps = self._diagonalize_fock(F_canon)
            self.D = self._build_density_matrix()
            F, F_canon = self._build_fock(H, df, S)
            self.E = Vnn + self._energy(H, F)

            dE = self.E - Eold
            if abs(dE) < self.level_shift_thresh:
                self._current_level_shift = None
            dD = sum(np.linalg.norm(d - dold) for d, dold in zip(self.D, Dold))
            self.S2 = self._spin(S)
            if self.verbose:
                print(
                    f"   {it:4d} {self.E:22.12f} {dE:12.4e} {dD:10.3e} "
                    f"{np.linalg.norm(AO_grad):10.3e} {self.S2:10.5f} {diis.status:>5s}"
                )
            if abs(dE) < self.e_tol and dD < self.d_tol:
                self.converged = True
                break
            Eold, Dold = self.E, self.D

        # final diagonalization: canonical orbitals of the converged Fock operator
        self.eps, self.C, self._orbital_irreps = self._diagonalize_fock(F_canon)
        self.D = self._build_density_matrix()
        F, F_canon = self._build_fock(H, df, S)
        self.F = F_canon
        self.E = Vnn + self._energy(H, F)
        self.e_tot = self.E

        if self.verbose:
            print("   " + "-" * 82)
        if not self.converged:
            msg = f"{self.method} did not converge in {self.maxiter} iterations."
            if self.die_if_not_converged:
                raise RuntimeError(msg)
            print(f"   WARNING: {msg}")

        self._assign_orbital_symmetries()
        if self.verbose:
            print(f"   {self.method} converged in {it} iterations ({time.perf_counter() - t_start:.2f} s)")
            print(f"   Total {self.method} energy = {self.e_tot:.12f}")
            print(f"   HF determinant symmetry: {self.determinant_symmetry}")
            self._print_orbital_energies()
            print("")
        return self

    def _assign_orbital_symmetries(self):
        point_group = "C1" if self._symmetry_basis is None else self._symmetry_basis.point_group
        names = {index: label for label, index in COTTON_LABELS[point_group].items()}
        self.irrep_indices = [np.asarray(h).tolist() for h in self._orbital_irreps]
        self.irrep_labels = [[names[i] for i in h] for h in self.irrep_indices]
        occupied = [
            np.asarray(self._orbital_irreps[0][: self.na]),
            np.asarray(self._orbital_irreps[-1][: self.nb]),
        ]
        self.determinant_symmetry = names[irrep_product(np.concatenate(occupied))]

    def _print_orbital_set(self, title, eps, labels, start, stop):
        if stop <= start:
            return
        print(f"\n   {title}:")
        line = "  "
        for n, i in enumerate(range(start, stop)):
            line += f" {i + 1:>4d} {labels[i]:<4} {eps[i]:>12.6f}"
            if n % 4 == 3:
                print(line)
                line = "  "
        if line.strip():
            print(line)

    # PySCF-like accessors used by ccpy's interfaces
    def energy_tot(self):
        return self.e_tot

    def get_hcore(self):
        return self.mol.ints_hcore()

    def get_ovlp(self):
        return self.mol.ints_overlap()

    def orbital_symmetries(self, spin=0):
        """Irrep label of each MO in ccpy's convention (upper case, e.g. "B1U", "A'")."""
        return [to_ccpy_irrep(label, self.mol.point_group) for label in self.irrep_labels[spin]]


def to_ccpy_irrep(label, point_group):
    """Convert a Cotton-order irrep label (e.g. "b1u", "a'") to the convention of
    ccpy.utilities.symmetry ("B1U", "A'")."""
    if point_group.upper() == "CS":
        return {"a'": "A'", "a''": 'A"'}[label]
    if point_group.upper() == "CI":
        return {"g": "AG", "u": "AU"}[label]
    return label.upper()
