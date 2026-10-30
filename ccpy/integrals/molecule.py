"""The Molecule object: geometry, point group, orbital and auxiliary basis sets, and the
factorized two-electron integrals used by ccpy.scf and Driver.from_scf.

Adapted from forte2 (https://github.com/evangelistalab/forte2, LGPL-3.0); see
THIRD_PARTY_NOTICES.md.
"""

import re

import numpy as np

from ccpy.integrals.atom_data import ATOM_DATA, ATOM_SYMBOL_TO_Z, ANGSTROM_TO_BOHR, Z_TO_ATOM_SYMBOL
from ccpy.integrals.linalg import canonical_orth
from ccpy.integrals.symmetry import PGSymmetryDetector, rotation_mat


class Molecule:
    """Molecular system for the libint2-based integrals and the native SCF.

    Arguments:
    ----------
    geometry : str -> atoms in XYZ ("O 0.0 0.0 0.0; H ...") or Z-matrix form; lines can be
                      separated by newlines or semicolons
    basis : str | dict -> orbital basis, e.g. "cc-pvdz", a path to a .json/.gbs file, or
                          a per-atom dict such as {"O": "cc-pvtz", "default": "cc-pvdz"}
    auxiliary_basis : str | dict -> density-fitting basis; defaults to "<basis>-autoaux"
                                    (ignored when cholesky_tei=True)
    charge : int -> total charge
    multiplicity : int -> spin multiplicity 2S+1 of the reference determinant
    unit : str -> "angstrom" or "bohr"
    symmetry : bool -> detect the largest Abelian point group, reorient the molecule
                       along its principal axes, and keep that symmetry in the SCF
    symmetry_tol : float -> tolerance (bohr) for accepting a symmetry operation
    cartesian : bool -> use Cartesian d, f, ... functions (PySCF cart=True, GAMESS ISPHER=-1)
    cholesky_tei : bool -> factorize the four-index AO integrals by a pivoted Cholesky
                           decomposition instead of density fitting
    cholesky_tol : float -> threshold of the Cholesky decomposition
    cholesky_pivots : list of (r, s) -> use these Cholesky pivot function pairs instead of
                      selecting them (e.g. mol.df.pivots of a reference geometry, so that
                      finite differences see a smooth energy); kept by with_geometry
    overlap_tol : float -> AO combinations with overlap eigenvalues below
                           overlap_tol * max eigenvalue are removed as linearly dependent
    df_metric_tol : float -> if given, the DF metric is orthogonalized with this relative
                             threshold instead of being Cholesky decomposed
    verbose : bool -> print the geometry analysis and integral information
    """

    def __init__(
        self,
        geometry,
        basis,
        auxiliary_basis=None,
        charge=0,
        multiplicity=1,
        unit="angstrom",
        symmetry=False,
        symmetry_tol=1.0e-04,
        cartesian=False,
        cholesky_tei=False,
        cholesky_tol=1.0e-06,
        cholesky_pivots=None,
        overlap_tol=1.0e-08,
        df_metric_tol=None,
        verbose=True,
    ):
        unit = unit.lower()
        if unit not in ("angstrom", "bohr"):
            raise ValueError(f"unit must be 'angstrom' or 'bohr', got {unit}.")
        if auxiliary_basis is None and not cholesky_tei:
            if not isinstance(basis, str):
                raise ValueError("An auxiliary_basis must be given when the orbital basis is a dict.")
            auxiliary_basis = f"{basis}-autoaux"

        self.geometry = geometry
        self.unit = unit
        self.basis_set = basis
        self.auxiliary_basis_set = None if cholesky_tei else auxiliary_basis
        self.charge = charge
        self.multiplicity = multiplicity
        self.symmetry = symmetry
        self.symmetry_tol = symmetry_tol
        self.cartesian = cartesian
        self.cholesky_tei = cholesky_tei
        self.cholesky_tol = cholesky_tol
        self.cholesky_pivots = cholesky_pivots
        self.overlap_tol = overlap_tol
        self.df_metric_tol = df_metric_tol
        self.verbose = verbose

        self._build(parse_geometry(geometry, unit))

    def _build(self, atoms):
        # deferred imports: these modules need the compiled extension
        from ccpy.integrals import integrals
        from ccpy.integrals.basis import build_basis
        from ccpy.integrals.df import DensityFitting

        self._set_geometry(atoms)

        self.nelectrons = int(sum(self.atomic_charges)) - self.charge
        if self.nelectrons < 0:
            raise ValueError("Number of electrons must be non-negative.")
        if (self.nelectrons + self.multiplicity - 1) % 2 != 0:
            raise ValueError(f"{self.nelectrons} electrons is incompatible with multiplicity {self.multiplicity}.")

        self.basis = build_basis(self.basis_set, self, cartesian=self.cartesian)
        self.nbf = self.basis.size
        if self.cholesky_tei:
            self.auxiliary_basis = None
            self.naux = None
        else:
            self.auxiliary_basis = build_basis(self.auxiliary_basis_set, self)
            self.naux = self.auxiliary_basis.size

        self.nuclear_repulsion = integrals.nuclear_repulsion(self)
        # canonical orthogonalization of the AO basis; nmo <= nbf independent orbitals
        self.Xorth, _, info = canonical_orth(integrals.overlap(self), self.overlap_tol)
        self.nmo = int(info["n_kept"])
        if info["n_discarded"] > 0:
            print(f"   WARNING: removed {info['n_discarded']} near-linearly-dependent AO combinations.")
        self._symmetry_basis = None
        self.df = DensityFitting(self)

    def _set_geometry(self, atoms):
        """Atomic data, and the point group and principal-axis orientation if requested."""
        self.atoms = atoms
        self.natoms = len(atoms)
        self.atomic_charges = np.array([Z for Z, _ in atoms])
        self.atomic_masses = np.array([ATOM_DATA[Z]["mass"] for Z in self.atomic_charges])
        self.atomic_positions = np.array([xyz for _, xyz in atoms])
        if not self.symmetry:
            self.point_group = "C1"
            self.atom_map = {"E": np.arange(self.natoms)}
            return

        center_of_mass = self.atomic_masses @ self.atomic_positions / np.sum(self.atomic_masses)
        inertia = np.zeros((3, 3))
        for m, r in zip(self.atomic_masses, self.atomic_positions - center_of_mass):
            inertia += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        detector = PGSymmetryDetector(
            inertia,
            self.atomic_positions - center_of_mass,
            self.atomic_charges,
            self.atomic_masses,
            tol=self.symmetry_tol,
            verbose=self.verbose,
        )
        detector.run()
        self.point_group = detector.pg_name
        self.atom_map = detector.atom_map
        if self.verbose:
            print(f"   Detected point group: {self.point_group}")
        if detector.max_displacement > 1.0e-10:
            print(f"   WARNING: symmetrized the geometry to {self.point_group}; the largest "
                  f"atomic displacement is {detector.max_displacement:.2e} bohr.")
        # the molecule is reoriented along its principal axes
        self.atomic_positions = detector.prin_atomic_positions.copy()
        for i in range(self.natoms):
            self.atoms[i][1] = self.atomic_positions[i, :]

    def with_geometry(self, coordinates, unit="bohr"):
        """New Molecule with the same options at the coordinates (natoms x 3), atoms in
        the same order. Used for finite differences, so symmetry must be off."""
        if self.symmetry:
            raise NotImplementedError("with_geometry requires symmetry=False.")
        conv = 1.0 if unit == "bohr" else ANGSTROM_TO_BOHR
        coordinates = np.asarray(coordinates, dtype=float) * conv
        new = object.__new__(Molecule)
        new.__dict__.update({k: v for k, v in self.__dict__.items() if k not in ("df", "_symmetry_basis")})
        new.unit = "bohr"
        new.geometry = "\n".join(f"{Z_TO_ATOM_SYMBOL[int(Z)]} {x:.16f} {y:.16f} {z:.16f}"
                                 for Z, (x, y, z) in zip(self.atomic_charges, coordinates))
        new._build([[int(Z), xyz] for Z, xyz in zip(self.atomic_charges, coordinates)])
        return new

    @property
    def nalpha(self):
        return (self.nelectrons + self.multiplicity - 1) // 2

    @property
    def nbeta(self):
        return (self.nelectrons - self.multiplicity + 1) // 2

    def energy_nuc(self):
        return self.nuclear_repulsion

    def ints_overlap(self):
        from ccpy.integrals import integrals
        return integrals.overlap(self)

    def ints_hcore(self):
        from ccpy.integrals import integrals
        return integrals.kinetic(self) + integrals.nuclear(self)

    @property
    def symmetry_basis(self):
        """Symmetry-adapted AO basis of the point group (None in C1), built on first use."""
        if self.point_group == "C1":
            return None
        if self._symmetry_basis is None:
            from ccpy.integrals import integrals
            from ccpy.integrals.symmetry import SymmetryBasis
            self._symmetry_basis = SymmetryBasis(self, integrals.overlap(self), self.Xorth)
        return self._symmetry_basis

    def print_info(self):
        print("")
        print("   Molecule")
        print("   --------")
        print(f"   Point group: {self.point_group}")
        print(f"   Charge: {self.charge}   Multiplicity: {self.multiplicity}")
        print(f"   Number of electrons: {self.nelectrons}")
        print("   Geometry (bohr):")
        for Z, xyz in self.atoms:
            print(f"      {Z_TO_ATOM_SYMBOL[Z]:<3}{xyz[0]:>16.10f}{xyz[1]:>16.10f}{xyz[2]:>16.10f}")
        print(f"   Orbital basis: {self.basis_set} ({self.nbf} functions, {self.nmo} independent)")
        if self.cholesky_tei:
            print(f"   Two-electron integrals: Cholesky decomposition (tol = {self.cholesky_tol})")
        else:
            print(f"   Auxiliary basis: {self.auxiliary_basis_set} ({self.naux} functions)")
        print(f"   Nuclear repulsion energy: {self.nuclear_repulsion:.10f}")
        print("")


def parse_geometry(geometry, unit):
    """List of [Z, xyz (bohr)] from an XYZ or Z-matrix string. A Z-matrix is recognized
    by a first line holding only an element symbol."""
    lines = [line.strip() for line in geometry.replace(";", "\n").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty geometry string provided.")
    conv = 1.0 if unit == "bohr" else ANGSTROM_TO_BOHR
    if re.match(r"^[a-zA-Z][a-zA-Z]?$", lines[0]):
        return _parse_zmatrix(lines, conv)
    return _parse_xyz(lines, conv)


def _parse_xyz(lines, conv):
    atoms = []
    for line in lines:
        fields = line.split()
        if not re.match(r"^[a-zA-Z][a-zA-Z]?$", fields[0]):
            continue  # e.g., the atom count or a title line of an .xyz file
        if len(fields) != 4:
            raise ValueError(f"Invalid XYZ line '{line}': expected an element symbol and 3 coordinates.")
        xyz = np.array([float(x.replace("d", "e").replace("D", "e")) for x in fields[1:]]) * conv
        atoms.append([ATOM_SYMBOL_TO_Z[fields[0].upper()], xyz])
    return atoms


def _parse_zmatrix(lines, conv):
    """Z-matrix lines: "A", "A i r", "A i r j angle", "A i r j angle k dihedral",
    with 1-based atom indices and angles in degrees."""
    atoms = []
    for n, line in enumerate(lines):
        fields = line.split()
        if len(fields) != [1, 3, 5, 7][min(n, 3)]:
            raise ValueError(f"Invalid Z-matrix line {n + 1}: {line}")
        symbol = fields[0]
        if n == 0:
            xyz = np.zeros(3)
        elif n == 1:
            xyz = np.array([float(fields[2]) * conv, 0.0, 0.0])
        elif n == 2:
            i1, r, i2, angle = int(fields[1]) - 1, float(fields[2]), int(fields[3]) - 1, float(fields[4])
            p1, p2 = atoms[i1][1], atoms[i2][1]
            v1 = (p2 - p1) / np.linalg.norm(p2 - p1)
            axis = np.cross(v1, [0.0, 0.0, 1.0]) if not np.allclose(v1[:2], 0) else np.array([0.0, 0.0, 1.0])
            xyz = p1 + rotation_mat(axis, np.radians(angle)) @ v1 * r * conv
        else:
            i1, r, i2, angle, i3, dihedral = (int(fields[1]) - 1, float(fields[2]), int(fields[3]) - 1,
                                              float(fields[4]), int(fields[5]) - 1, float(fields[6]))
            p1, p2, p3 = atoms[i1][1], atoms[i2][1], atoms[i3][1]
            v1 = p2 - p1
            c = conv * r * v1 / np.linalg.norm(v1)
            c = rotation_mat(np.cross(v1, p2 - p3), np.radians(angle)) @ c
            c = rotation_mat(v1, np.radians(dihedral)) @ c
            xyz = c + p1
        atoms.append([ATOM_SYMBOL_TO_Z[symbol.upper()], xyz])
    return atoms
