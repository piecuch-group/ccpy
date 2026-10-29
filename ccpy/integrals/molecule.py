import re
from dataclasses import dataclass
import numpy as np

from ccpy.integrals.utils import logger, canonical_orth
from ccpy.integrals.symmetry import rotation_mat, PGSymmetryDetector
from ccpy.integrals.atom_data import (
    ATOM_DATA,
    ATOM_SYMBOL_TO_Z,
    ANGSTROM_TO_BOHR,
    Z_TO_ATOM_SYMBOL,
)


def coords_to_atoms(atomic_charges, coordinates, unit="bohr"):
    """
    Build a parsed atom list from atomic numbers and Cartesian coordinates.

    Parameters
    ----------
    atomic_charges : array_like
        Atomic numbers, shape ``(natoms,)``.
    coordinates : array_like
        Cartesian coordinates, shape ``(natoms, 3)``.
    unit : str, optional, default="bohr"
        The unit of `coordinates`, either "bohr" or "angstrom".

    Returns
    -------
    list[list[int, NDArray]]
        A list of ``[atomic number, coordinates]`` pairs, in Bohr.
    """
    if unit not in ("bohr", "angstrom"):
        raise ValueError(f"unit must be 'bohr' or 'angstrom', but got {unit}.")

    atomic_charges = np.asarray(atomic_charges, dtype=int).reshape(-1)
    coordinates = np.asarray(coordinates, dtype=float)
    expected_shape = (atomic_charges.size, 3)
    if coordinates.shape != expected_shape:
        raise ValueError(
            f"Expected coordinates of shape {expected_shape}, got {coordinates.shape}."
        )

    conv = 1.0 if unit == "bohr" else ANGSTROM_TO_BOHR
    return [[int(Z), xyz * conv] for Z, xyz in zip(atomic_charges, coordinates)]


def coords_to_xyz(atomic_charges, coordinates):
    """
    Format atomic numbers and Cartesian coordinates as an XYZ geometry string.

    Parameters
    ----------
    atomic_charges : array_like
        Atomic numbers, shape ``(natoms,)``.
    coordinates : array_like
        Cartesian coordinates, shape ``(natoms, 3)``.

    Returns
    -------
    str
        An XYZ geometry string, one atom per line.
    """
    atomic_charges = np.asarray(atomic_charges, dtype=int).reshape(-1)
    coordinates = np.asarray(coordinates, dtype=float)
    expected_shape = (atomic_charges.size, 3)
    if coordinates.shape != expected_shape:
        raise ValueError(
            f"Expected coordinates of shape {expected_shape}, got {coordinates.shape}."
        )

    return "\n".join(
        f"{Z_TO_ATOM_SYMBOL[int(Z)]} {xyz[0]:.16f} {xyz[1]:.16f} {xyz[2]:.16f}"
        for Z, xyz in zip(atomic_charges, coordinates)
    )


def parse_geometry(geom, unit):
    """
    Parse a geometry string (XYZ or Z-matrix) into a list of atoms.

    Parameters
    ----------
    geom : str
        Geometry string in XYZ or Z-matrix format.
    unit : str
        Coordinate unit, "bohr" or "angstrom".

    Returns
    -------
    atoms : list[tuple(int, NDArray)]
        List of (Z, coords) tuples in Bohr.
    """
    # Gives the user the option to break line with either newline or semicolon
    geom = geom.replace(";", "\n")
    lines = [line.strip() for line in geom.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty geometry string provided.")

    # Z-matrix should have only the element name on the first line
    if re.match(r"^([a-zA-Z][a-zA-Z]?)$", lines[0]):
        return parse_zmatrix(lines, unit)
    else:
        return parse_xyz(lines, unit)


def parse_xyz(lines, unit):
    r"""
    Parse an XYZ string into a list of atoms.

    Parameters
    ----------
    lines : list[str]
        The lines containing the XYZ atom symbols and coordinates.
    unit : str
        The unit of the coordinates, either "bohr" or "angstrom".

    Returns
    -------
    atoms : list[tuple(int, NDArray)]
        A list of tuples, each containing the atomic number and a numpy array of coordinates.

    Raises
    ------
    ValueError
        If a line in the XYZ string does not match the expected format or has an incorrect number of coordinates.

    Examples
    --------
    >>> xyz = "Li 0.0 0.0 0.0\nN -10 0 0\n"
    >>> parse_xyz(xyz, "bohr")
    [(3, array([0., 0., 0.])), (7, array([-10., 0., 0.]))]

    >>> xyz = "Li 0.0 0.0\nN -10 0\n"
    >>> parse_xyz(xyz, "angstrom")
    Traceback (most recent call last):
        ... ValueError: Invalid line in XYZ file: Li 0.0 0.0. Expected 3 coordinates, found 2.

    """
    atoms = []
    for line in lines:
        # look for lines of the form "Li 0.0 0.0 0.0" or "N -10 0 0" and capture the element symbol and coordinates
        # Use regex to match the expected format
        m = re.match(
            r"^\s*([a-zA-Z][a-zA-Z]?)\s+([-+]?\d*\.\d+|[-+]?\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)\s*$",
            line,
        )
        # Skip lines that do not match the expected format
        if not m:
            # Test if one or two coordinates are missing, e.g., "Li 0.0 0.0" or "Li 0.0"
            # This regex captures the element symbol and up to three coordinates
            check_missing_coordinate = re.match(
                r"^\s*([a-zA-Z][a-zA-Z]?)\s+([-+]?\d*\.\d+|[-+]?\d+)(?:\s+([-+]?\d*\.\d+|[-+]?\d+))?(?:\s+([-+]?\d*\.\d+|[-+]?\d+))?\s*$",
                line,
            )
            if check_missing_coordinate:
                n = len(check_missing_coordinate.groups()) - 2
                raise ValueError(
                    f"Invalid line in XYZ file: {line}. Expected 3 coordinates, found {n}."
                )
            continue

        parts = m.groups()
        atomic_number = ATOM_SYMBOL_TO_Z[parts[0].upper()]
        conv = 1.0 if unit == "bohr" else ANGSTROM_TO_BOHR
        coords = np.array([float(x) * conv for x in parts[1:]])
        atoms.append([atomic_number, coords])

    return atoms


def parse_zmatrix(lines, unit):
    """Parse a Z-matrix string into a list of atoms.

    Parameters
    ----------
    lines : list[str]
        The lines containing the Z-matrix atom symbols and coordinates.
    unit : str
        The unit of the coordinates, either "bohr" or "angstrom".

    Returns
    -------
    atoms : list[tuple(int, NDArray)]
        A list of tuples, each containing the atomic number and a numpy array of coordinates.

    Raises
    ------
    ValueError
        If a line in the Z-matrix string does not match the expected format.

    Examples
    --------
    Unit bohr:
    H
    O 1 0.9
    O 2 1.4 1 105.0
    H 3 0.9 2 105.0 1 120.0

    Should return:
    [[ 0.          0.          0.        ]
    [ 1.70075351  0.          0.        ]
    [ 2.38548947  0.         -2.55546938]
    [ 3.29283067  1.42270804 -2.76806356]]

    """

    atoms = []
    coords = []
    conv = 1.0 if unit == "bohr" else ANGSTROM_TO_BOHR

    for i, line in enumerate(lines):
        if i == 0:
            # Atom with no internal coordinates
            m = re.match(r"^([a-zA-Z][a-zA-Z]?)$", line)
            if not m:
                raise ValueError(f"Invalid Z-matrix line {i+1}: {line}")
            symbol = m.group(1)
            coord = np.array([0.0, 0.0, 0.0])
        elif i == 1:
            # Atom with bond length
            m = re.match(
                r"^([a-zA-Z][a-zA-Z]?)\s+(\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)$", line
            )
            if not m:
                raise ValueError(f"Invalid Z-matrix line {i+1}: {line}")
            symbol, i1, r = m.groups()
            coord = np.array([float(r) * conv, 0.0, 0.0])
        elif i == 2:
            # Atom with bond length and angle
            m = re.match(
                r"^([a-zA-Z][a-zA-Z]?)\s+(\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)\s+(\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)$",
                line,
            )
            if not m:
                raise ValueError(f"Invalid Z-matrix line {i+1}: {line}")
            symbol, i1, r, i2, angle = m.groups()
            theta = float(angle) / 180 * np.pi
            i1, i2 = int(i1) - 1, int(i2) - 1

            p1 = coords[i1]
            p2 = coords[i2]
            # Vector from p1 to p2
            v1 = p2 - p1
            v1 = v1 / np.linalg.norm(v1)
            if not np.allclose(v1[:2], 0):
                v_rot = np.cross(v1, np.array((0.0, 0.0, 1.0)))
            else:
                v_rot = np.array((0.0, 0.0, 1.0))
            rmat = rotation_mat(v_rot, theta)
            c = np.dot(rmat, v1) * float(r) * conv
            coord = c + p1
        else:
            m = re.match(
                r"^([a-zA-Z][a-zA-Z]?)\s+(\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)"
                r"\s+(\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)"
                r"\s+(\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)$",
                line,
            )
            if not m:
                raise ValueError(f"Invalid Z-matrix line {i+1}: {line}")
            symbol, i1, r, i2, angle, i3, dihedral = m.groups()
            theta = float(angle) / 180 * np.pi
            phi = float(dihedral) / 180 * np.pi

            i1, i2, i3 = int(i1) - 1, int(i2) - 1, int(i3) - 1
            p1, p2, p3 = coords[i1], coords[i2], coords[i3]
            # Vector from p1 to p2
            v1 = p2 - p1
            # Vector from p3 to p2
            v2 = p2 - p3
            c = conv * float(r) * v1 / np.linalg.norm(v1)
            normal = np.cross(v1, v2)
            c = np.dot(rotation_mat(normal, theta), c)
            c = np.dot(rotation_mat(v1, phi), c)
            coord = c + p1
        atomic_number = ATOM_SYMBOL_TO_Z[symbol.upper()]
        atoms.append([atomic_number, coord])
        coords.append(coord)

    return atoms


@dataclass
class GeometryHelper:
    """Helper class to process geometry data."""

    atoms: list[tuple[int, np.ndarray]]
    tol: float = 1e-4
    symmetry: bool = False

    def __post_init__(self):
        self.Zsum = round(np.sum([x[0] for x in self.atoms]))
        self.natoms = len(self.atoms)
        self.atomic_charges = np.array([atom[0] for atom in self.atoms])
        self.atomic_masses = np.array(
            [ATOM_DATA[Z]["mass"] for Z in self.atomic_charges]
        )
        self.atomic_positions = np.array([atom[1] for atom in self.atoms])
        self.centroid = np.mean(self.atomic_positions, axis=0)

        self.center_of_mass = np.einsum(
            "a,ax->x", self.atomic_masses, self.atomic_positions
        ) / np.sum(self.atomic_masses)

        self.atom_counts = {}
        for atom in self.atoms:
            if atom[0] not in self.atom_counts:
                self.atom_counts[atom[0]] = 0
            self.atom_counts[atom[0]] += 1

        self.atom_to_center = {}
        for i, atom in enumerate(self.atoms):
            if atom[0] not in self.atom_to_center:
                self.atom_to_center[atom[0]] = []
            self.atom_to_center[atom[0]].append(i)

        self.inertia_matrix = np.zeros((3, 3))
        for m, r in zip(self.atomic_masses, self.atomic_positions):
            x = r - self.center_of_mass
            r2 = np.dot(x, x)
            self.inertia_matrix += m * ((r2 * np.eye(3)) - np.outer(x, x))

        if self.symmetry:
            com_atomic_positions = self.atomic_positions - self.center_of_mass[None, :]
            sym_detector = PGSymmetryDetector(
                self.inertia_matrix,
                com_atomic_positions,
                self.atomic_charges,
                self.atomic_masses,
                tol=self.tol,
            )
            sym_detector.run()
            self.prinrot = sym_detector.prinrot
            self.point_group = sym_detector.pg_name
            self.atom_map = sym_detector.atom_map
            logger.log_info1(f"Detected point group: {self.point_group}")
            self.prin_atomic_positions = sym_detector.prin_atomic_positions
            if sym_detector.max_displacement > 1e-10:
                logger.log_warning(
                    f"Symmetrized the geometry to {self.point_group}; the largest atomic "
                    f"displacement is {sym_detector.max_displacement:.2e} bohr."
                )

            # Overwrite original inputted atomic positions with principal atomic positions
            self.atomic_positions = self.prin_atomic_positions.copy()
            for i in range(self.natoms):
                self.atoms[i][1] = self.prin_atomic_positions[i, :]
            logger.log_info2(
                "Original atomic coordinates overwritten with principal atomic positions"
            )
        else:
            self.prinrot = np.eye(3)
            self.prin_atomic_positions = self.atomic_positions.copy()
            self.point_group = "C1"
            self.atom_map = {"E": np.arange(self.natoms)}
            logger.log_info2(
                "Point group symmetry detection not performed. Running in C1 symmetry."
            )


class Molecule:
    """Molecular system (geometry, basis sets, symmetry) for libint2-based integrals.

    This is the ccpy counterpart of forte2's ``System``. It owns the orbital and
    auxiliary basis sets, the canonical orthogonalizer of the AO overlap, the point
    group (if requested), and lazily builds the density-fitting tensors used by the
    SCF solvers and the correlated methods.

    Parameters
    ----------
    geometry : str
        Atoms in XYZ ("O 0.0 0.0 0.0; H ...") or Z-matrix form.
    basis : str | dict
        Orbital basis set, e.g. "cc-pvdz", a path to a .json/.gbs file, or a per-atom
        dict such as {"O": "cc-pvtz", "default": "cc-pvdz"}.
    auxiliary_basis : str | dict, optional
        Density-fitting basis. If None, an AutoAux basis is generated from the
        orbital basis ("<basis>-autoaux"), which is accurate for both SCF and
        correlated calculations.
    charge : int, optional, default=0
    multiplicity : int, optional, default=1
        Spin multiplicity 2S+1 of the reference determinant.
    unit : str, optional, default="angstrom"
        "angstrom" or "bohr".
    symmetry : bool, optional, default=False
        Detect the largest Abelian point group, reorient the molecule along its
        principal axes, and keep that symmetry in the SCF.
    symmetry_tol : float, optional, default=1e-4
        Tolerance (bohr) for accepting a symmetry operation.
    overlap_ortho_rtol : float, optional, default=1e-8
        Relative threshold for removing near-linear dependencies of the AO basis.
    df_ortho_rtol : float, optional
        If set, the DF Coulomb metric is orthogonalized with this relative threshold
        instead of being Cholesky decomposed.
    cholesky_tei : bool, optional, default=False
        Use a pivoted Cholesky decomposition of the four-index AO ERIs instead of an
        auxiliary basis.
    cholesky_tol : float, optional, default=1e-6
        Threshold for the Cholesky decomposition (only with cholesky_tei=True).
    cartesian : bool, optional, default=False
        Use Cartesian d, f, ... functions in the orbital basis (PySCF cart=True,
        GAMESS ISPHER=-1).
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
        overlap_ortho_rtol=1.0e-08,
        df_ortho_rtol=None,
        cholesky_tei=False,
        cholesky_tol=1.0e-06,
        cartesian=False,
    ):
        unit = unit.lower()
        if unit not in ("angstrom", "bohr"):
            raise ValueError(f"unit must be 'angstrom' or 'bohr', got {unit}.")
        self.geometry = geometry
        self.unit = unit
        self.basis_set = basis
        if auxiliary_basis is None and not cholesky_tei:
            if not isinstance(basis, str):
                raise ValueError(
                    "An auxiliary_basis must be given when the orbital basis is a dict."
                )
            auxiliary_basis = f"{basis}-autoaux"
        self.auxiliary_basis_set = auxiliary_basis
        self.charge = charge
        self.multiplicity = multiplicity
        self.symmetry = symmetry
        self.symmetry_tol = symmetry_tol
        self.overlap_ortho_rtol = overlap_ortho_rtol
        self.df_ortho_rtol = df_ortho_rtol
        self.cholesky_tei = cholesky_tei
        self.cholesky_tol = cholesky_tol
        self.cartesian = cartesian
        # forte2-compatible flags read by the integral wrappers
        self.use_gaussian_charges = False
        self.two_component = False

        self._build(parse_geometry(geometry, unit))

    def _build(self, atoms):
        # deferred imports: these modules need the compiled extension
        from ccpy.integrals import integrals
        from ccpy.integrals.basis import build_basis
        from ccpy.integrals.df import DensityFitting

        self.geom_helper = GeometryHelper(atoms, symmetry=self.symmetry, tol=self.symmetry_tol)
        self.atoms = self.geom_helper.atoms
        self.natoms = self.geom_helper.natoms
        self.Zsum = self.geom_helper.Zsum
        self.atomic_charges = self.geom_helper.atomic_charges
        self.atomic_masses = self.geom_helper.atomic_masses
        self.atomic_positions = self.geom_helper.atomic_positions
        self.atom_counts = self.geom_helper.atom_counts
        self.atom_to_center = self.geom_helper.atom_to_center
        self.point_group = self.geom_helper.point_group
        self.atom_map = self.geom_helper.atom_map

        self.nelectrons = int(self.Zsum - self.charge)
        if self.nelectrons < 0:
            raise ValueError("Number of electrons must be non-negative.")
        if (self.nelectrons + self.multiplicity - 1) % 2 != 0:
            raise ValueError(
                f"{self.nelectrons} electrons is incompatible with multiplicity {self.multiplicity}."
            )

        self.basis = build_basis(self.basis_set, self.geom_helper, cartesian=self.cartesian)
        if self.cholesky_tei:
            self.auxiliary_basis = None
        else:
            self.auxiliary_basis = build_basis(self.auxiliary_basis_set, self.geom_helper)
        self.nbf = self.basis.size
        self.naux = self.auxiliary_basis.size if self.auxiliary_basis is not None else None

        self.nuclear_repulsion = integrals.nuclear_repulsion(self)
        self.Xorth, _, info = canonical_orth(integrals.overlap(self), self.overlap_ortho_rtol)
        self.nmo = int(info["n_kept"])
        if info["n_discarded"] > 0:
            logger.log_warning(
                f"Removed {info['n_discarded']} near-linearly-dependent AO combinations."
            )
        self._symmetry_basis = None
        self.df = DensityFitting(self)

    def with_geometry(self, coordinates, unit="bohr"):
        """Return a new Molecule with the same options at displaced coordinates
        (shape (natoms, 3), atoms in the same order). Requires symmetry=False."""
        if self.symmetry:
            raise NotImplementedError("with_geometry requires symmetry=False.")
        new = object.__new__(Molecule)
        new.__dict__.update(
            {k: v for k, v in self.__dict__.items() if k not in ("df", "_symmetry_basis")}
        )
        new.unit = "bohr"
        new.geometry = coords_to_xyz(self.atomic_charges, np.asarray(coordinates) * (
            1.0 if unit == "bohr" else ANGSTROM_TO_BOHR))
        new._build(coords_to_atoms(self.atomic_charges, coordinates, unit=unit))
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

    def get_Xorth(self):
        return self.Xorth

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
