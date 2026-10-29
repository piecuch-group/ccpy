"""Basis sets: loading (bundled JSON files, Gaussian94 .gbs files, Basis Set Exchange,
AutoAux generation) and assembly of libint2 Basis objects for a Molecule.

Adapted from forte2 (https://github.com/evangelistalab/forte2, LGPL-3.0); see
THIRD_PARTY_NOTICES.md.
"""

import json
import itertools
from pathlib import Path
import re
import os

import numpy as np

from ccpy.integrals._libint import Basis, Shell
from ccpy.integrals.atom_data import ATOM_SYMBOL_TO_Z
from ccpy.integrals.autoaux import autoaux_electron_shells

BASIS_DATA_DIR = Path(__file__).parent / "basis_data"

try:
    import basis_set_exchange as bse

    BSE_AVAILABLE = True
except ImportError:
    BSE_AVAILABLE = False


def _is_basis_file(name):
    return name.split("::")[0].lower().endswith((".json", ".gbs", ".g94"))


def build_basis_from_dict(basis_data):
    """
    Build a Basis object from a dictionary containing the basis set data.

    The basis_data is expected to have the following format for each shell:
    {"schema_version": 1,
     "nshells": int,
     "shells": [
        {"nprim": int,
         "l": int,
         "exponents": list[float],
         "coefficients": list[float],
         "center": list[float]},
         "is_pure": bool
        ...
     ]}

    Parameters
    ----------
    basis_data : dict
        A dictionary containing the basis set data for all shells.

    Returns
    -------
    basis : ccpy.integrals.Basis
        The constructed Basis object.
    """
    if not isinstance(basis_data, dict):
        raise TypeError(
            f"Expected basis_data to be a dictionary, but got {type(basis_data)}."
        )

    if "schema_version" not in basis_data:
        raise ValueError("Basis data is missing 'schema_version' key.")
    if basis_data["schema_version"] != 1:
        raise ValueError(
            f"Unsupported basis data schema version: {basis_data['schema_version']}. Expected version 1."
        )
    nshells = int(basis_data["nshells"])
    if len(basis_data["shells"]) != nshells:
        raise ValueError(
            f"The number of shells ({len(basis_data['shells'])}) does not match nshells ({nshells})"
        )
    basis = Basis()
    for i in range(nshells):
        try:
            idata = basis_data["shells"][i]
            is_pure = bool(idata["is_pure"])
            nprim = int(idata["nprim"])
            l = int(idata["l"])
            exponents = np.array(idata["exponents"], dtype=float)
            if len(exponents) != nprim:
                raise ValueError(
                    f"Number of exponents does not match nprim for shell {i+1}. Expected {nprim}, got {len(exponents)}."
                )
            coefficients = np.array(idata["coefficients"], dtype=float)
            if len(coefficients) != nprim:
                raise ValueError(
                    f"Number of coefficients does not match nprim for shell {i+1}. Expected {nprim}, got {len(coefficients)}."
                )
            center = np.array(idata["center"], dtype=float)
            if len(center) != 3:
                raise ValueError(
                    f"Center coordinates must be a list of 3 floats for shell {i+1}. Got {len(center)} coordinates."
                )
        except KeyError as e:
            raise ValueError(
                f"Error parsing basis_data at shell {i+1}. Check that the dictionary is correctly formatted. Original error: {e}"
            ) from e

        basis.add(
            Shell(
                l,
                exponents,
                coefficients,
                center,
                is_pure=is_pure,
                embed_normalization_into_coefficients=False,  # assume the coefficients are already normalized if provided as an array
            )
        )
    return basis


def build_basis(basis_assignment, geometry, embed_normalization_into_coefficients=True,
                decontract=False, cartesian=False):
    """
    Assemble the basis set from JSON data or Basis Set Exchange, depending on availability.

    Parameters
    ----------
    basis_assignment : str or dict
        The basis set name or a dictionary with per-atom basis assignments.
        It is possible to assign different basis sets to specific atomic indices (e.g. "O2" for the second oxygen atom).
        It is also possible to assign basis set for a different element than the atom itself using the syntax "basis::element", e.g. "O": "sto-6g::N" to assign nitrogen sto-6g basis functions to oxygen.
        There are three sources for basis set data:
            1. If the basis set name ends with ``.json``, it is treated as a local JSON file path.
            2. If a JSON file with the basis set name exists in the ``ccpy/integrals/basis_data`` directory, it is used.
            3. If neither of the above is found, Basis Set Exchange is used (if available).

        If a basis set name ends with ``-autoaux``, for example ``"cc-pvtz-autoaux"``,
        an AutoAux auxiliary basis (Stoychev, Auer, and Neese, 2017) is generated from
        the orbital basis set named before the suffix, which is loaded from the sources
        listed here.

        For example, a basis set assignment dictionary can look like::

            {
                "H": "cc-pvtz-truncated.json::Li",
                "O2": "decon-cc-pvqz::N",
                "default": "cc-pvdz"
            }

        where ``"default"`` is used for atoms not explicitly listed.
    geometry : ccpy.integrals.Molecule (only atoms and natoms are used)
        The molecular geometry.
    embed_normalization_into_coefficients : bool, optional, default=True
        Whether to embed normalization factors into the contraction coefficients.
    decontract : bool, optional, default=False
        If True, all basis functions will be decontracted regardless of the basis set assignment.
    cartesian : bool, optional, default=False
        If True, shells with l >= 2 use Cartesian Gaussians (6 d, 10 f, ...) instead of
        solid harmonics, as in GAMESS ISPHER=-1 or PySCF cart=True.

    Returns
    -------
    basis : ccpy.integrals.Basis
        The basis set.
    """
    prefix = "decon-" if decontract else ""
    natoms = geometry.natoms

    # get the mapping of atomic index to basis set names
    if isinstance(basis_assignment, str):
        basis_name = basis_assignment
        basis_per_atom = [basis_name for _ in range(natoms)]
    elif isinstance(basis_assignment, dict):
        basis_name = "custom_basis"
        basis_per_atom = _parse_custom_basis_assignment(geometry, basis_assignment)
    else:
        raise TypeError(
            f"Invalid basis assignment type: {type(basis_assignment)}. "
            "Expected str or dict."
        )

    # convert the basis set names to lowercase
    # (file paths keep their case)
    basis_per_atom = [b if _is_basis_file(b) else b.lower() for b in basis_per_atom]

    if_decontract_atom_basis = ["decon-" in b or decontract for b in basis_per_atom]
    # remove the "decon-" prefix to prepare for fetching of the basis set data
    basis_per_atom = [b.replace("decon-", "") for b in basis_per_atom]

    # get the 'shopping list' for _get_atom_basis
    # e.g. {"cc-pvdz": {1, 2}, "sto-6g": {8}} (fetch H, He from cc-pvdz and O from sto-6g)
    fetch_map = {}
    for i, b in enumerate(basis_per_atom):
        if "::" in b:
            # basis set with element specification (e.g., assign N basis to C)
            bset, atom_symbol = b.split("::")
            Z = ATOM_SYMBOL_TO_Z[atom_symbol.upper()]
        else:
            bset = b
            Z = geometry.atoms[i][0]
        if bset not in fetch_map:
            fetch_map[bset] = set()
        fetch_map[bset].add(Z)

    # set the basis name
    basis = Basis()
    basis.set_name(prefix + basis_name)
    # get the unique basis set data
    atom_basis = _get_atom_basis(fetch_map)
    for (Z, coords), decon, bset in zip(
        geometry.atoms, if_decontract_atom_basis, basis_per_atom
    ):
        if "::" in bset:
            bset_loc, atom_symbol = bset.split("::")
            Z_loc = ATOM_SYMBOL_TO_Z[atom_symbol.upper()]
        else:
            bset_loc = bset
            Z_loc = Z
        if decon:
            basis = _add_atom_basis_to_basis_decontracted(
                basis,
                atom_basis[bset_loc][Z_loc],
                coords,
                embed_normalization_into_coefficients,
                cartesian,
            )
        else:
            basis = _add_atom_basis_to_basis(
                basis,
                atom_basis[bset_loc][Z_loc],
                coords,
                embed_normalization_into_coefficients,
                cartesian,
            )

    return basis


def decontract_basis(basis, embed_normalization_into_coefficients=True):
    """
    Decontract a Basis object and return a new Basis object.

    For generally contracted basis sets where multiple shells share the same
    primitives (same l, center, and exponents), this function avoids duplicates
    so that each unique primitive appears only once.

    Parameters
    ----------
    basis : ccpy.integrals.Basis
        The Basis object to be decontracted.
    embed_normalization_into_coefficients : bool, optional, default=True
        Whether to embed normalization factors into the contraction coefficients.

    Returns
    -------
    basis_decon : ccpy.integrals.Basis
        The decontracted Basis object.
    """
    nsh = basis.nshells
    basis_decon = Basis()
    seen = set()
    for i in range(nsh):
        sh = basis[i]
        l = sh.l
        center = np.asarray(sh.center, dtype=np.float64)
        is_pure = sh.is_pure
        for alpha in sh.exponents:
            # lists/ndarrays cannot be hashed but raw bytes can
            key = (l, center.tobytes(), alpha)
            if key in seen:
                continue
            seen.add(key)
            basis_decon.add(
                Shell(
                    l,
                    [alpha],
                    [1.0],
                    center.tolist(),
                    is_pure,
                    embed_normalization_into_coefficients,
                )
            )
    return basis_decon


def build_sap_potential_basis(basis_name, geometry):
    r"""Build a Gaussian charge-density representation of an SAP potential.

    SAP basis coefficients use the convention of Lehtola, where each primitive
    coefficient multiplies a unit-charge Gaussian potential. Three-center
    Coulomb integrals instead expect a charge density. For an unnormalized
    s-type primitive, the conversion is

    .. math::
        c_k e^{-\alpha_k r^2}
        \longrightarrow
        -c_k \left(\frac{\alpha_k}{\pi}\right)^{3/2}
        e^{-\alpha_k r^2}.

    The minus sign converts the electronic-potential coefficient convention to
    a positive screening charge density.

    Parameters
    ----------
    basis_name : str
        Name of an SAP basis available locally or from Basis Set Exchange.
    geometry : ccpy.integrals.Molecule
        Molecular geometry on which to center the SAP functions.

    Returns
    -------
    Basis
        Unnormalized, contracted s-type density basis suitable for
        three-center Coulomb integrals.
    """
    sap_basis = build_basis(
        basis_name,
        geometry,
        embed_normalization_into_coefficients=False,
    )

    density_basis = Basis()
    density_basis.set_name(f"{basis_name}-density")
    for shell in sap_basis:
        if shell.l != 0:
            raise ValueError(
                f"SAP basis '{basis_name}' contains a shell with l={shell.l}; "
                "only s-type SAP functions are supported."
            )
        exponents = np.asarray(shell.exponents, dtype=float)
        coefficients = np.asarray(shell.coeff, dtype=float)
        density_coefficients = -coefficients * (exponents / np.pi) ** 1.5
        density_basis.add(
            Shell(
                shell.l,
                exponents,
                density_coefficients,
                shell.center,
                shell.is_pure,
                embed_normalization_into_coefficients=False,
            )
        )

    return density_basis


def _parse_custom_basis_assignment(geometry, basis_assignment):
    # explicit_basis contains all entries except "default"
    explicit_basis = basis_assignment.copy()
    default_basis = explicit_basis.pop("default", None)
    # atom_to_center[Z] = indices of the atoms of element Z, in order
    atom_to_center = {}
    for i, (Z, _) in enumerate(geometry.atoms):
        atom_to_center.setdefault(Z, []).append(i)
    atom_counts = {Z: len(idx) for Z, idx in atom_to_center.items()}

    assigned_basis = {}
    for k, v in explicit_basis.items():
        # "could be "O" or "O2" or "H2-12"
        m = re.match(r"([a-zA-Z]{1,2})([0-9]+)?-?([0-9]+)?", k).groups()
        # m[0]: atom symbol
        # m[1]: atom index start (optional)
        # m[2]: atom index end (optional)
        Z = ATOM_SYMBOL_TO_Z[m[0].upper()]
        if m[1] is None:
            assert (
                m[2] is None
            ), f"Basis assignment {k} has an end index without a start index."
            for i in atom_to_center[Z]:
                assigned_basis[i] = v
        elif m[2] is None:
            # single atom index
            i = int(m[1]) - 1
            assert i < atom_counts[Z], f"Atom {k} not found in geometry."
            assigned_basis[atom_to_center[Z][i]] = v
        else:
            # range of atom indices
            start = int(m[1]) - 1
            end = int(m[2])
            assert end <= atom_counts[Z], f"Atom range {k} is out of bounds."
            for i in range(start, end):
                assigned_basis[atom_to_center[Z][i]] = v

    if default_basis is None:
        assert (
            len(assigned_basis) == geometry.natoms
        ), "Not all atoms have a basis set assigned. Provide a default basis set or assign a basis set to all atoms."
    else:
        # ensure all atoms have a basis set assigned
        for i in range(geometry.natoms):
            if i not in assigned_basis:
                assigned_basis[i] = default_basis

    basis_per_atom = []
    for i in range(geometry.natoms):
        basis_per_atom.append(assigned_basis[i])

    return basis_per_atom


def _bse_file_name(basis_name):
    # Basis Set Exchange file-name convention: 6-31g** -> 6-31g_st__st_
    return basis_name.replace("*", "_st_")


def _load_basis(basis_name, Z):
    """
    Load the basis set data for a given atomic number.

    This can be from four sources:
    1. A local JSON file specified by `basis_name` (if it ends with .json), or a local
       Gaussian94-format file (if it ends with .gbs or .g94)
    2. A JSON file supplied in the `ccpy/integrals/basis_data` directory
    3. AutoAux generation (if it ends with -autoaux) from the orbital basis set named
       before the suffix, itself loaded from one of the other sources
    4. Basis Set Exchange (if available)

    Parameters
    ----------
    basis_name : str
        The name of the basis set or the path to a local JSON file.
    Z : int
        The atomic number of the element.

    Returns
    -------
    res : list[dict]
        The basis set data for the specified atomic number.
    """
    if basis_name.endswith(".json"):
        assert os.path.isfile(basis_name), f"Basis file {basis_name} not found."
        with open(basis_name, "r") as f:
            bfile = json.load(f)
            assert (
                str(Z) in bfile["elements"]
            ), f"Element {Z} not found in basis set {basis_name}."
            res = bfile["elements"][str(Z)]["electron_shells"]
    elif basis_name.endswith((".gbs", ".g94")):
        assert os.path.isfile(basis_name), f"Basis file {basis_name} not found."
        res = parse_gaussian94(basis_name, Z)
    elif (BASIS_DATA_DIR / f"{_bse_file_name(basis_name)}.json").is_file():
        with open(BASIS_DATA_DIR / f"{_bse_file_name(basis_name)}.json", "r") as f:
            bfile = json.load(f)
            assert (
                str(Z) in bfile["elements"]
            ), f"Element {Z} not found in basis set {basis_name}."
            res = bfile["elements"][str(Z)]["electron_shells"]
    elif basis_name.endswith("-autoaux"):
        orbital_basis_name = basis_name.removesuffix("-autoaux")
        res = autoaux_electron_shells(Z, _load_basis(orbital_basis_name, Z))
    else:
        assert (
            BSE_AVAILABLE
        ), f"Basis file {basis_name}.json not found locally, and basis_set_exchange is not available."
        try:
            bse_basis = bse.get_basis(basis_name, elements=Z)
        except KeyError:
            raise RuntimeError(
                f"Basis Set Exchange does not have data for element Z={Z} in basis set {basis_name}!"
            )
        res = bse_basis["elements"][str(Z)]["electron_shells"]

    return res


def _get_atom_basis(fetch_map):
    # atom_basis = {basis_name: {Z: basis_data}}
    atom_basis = {}
    for basis_name, atoms in fetch_map.items():
        atom_basis[basis_name] = {}
        for Z in atoms:
            atom_basis[basis_name][Z] = _load_basis(basis_name, Z)
    return atom_basis


def _add_atom_basis_to_basis_decontracted(
    basis, atom_basis, coords, embed_normalization_into_coefficients, cartesian=False
):
    for shell in atom_basis:
        angular_momentum = list(map(int, shell["angular_momentum"]))
        exponents = list(map(float, shell["exponents"]))
        for l in angular_momentum:
            for alpha in exponents:
                basis.add(
                    Shell(
                        l,
                        [alpha],
                        [1.0],
                        coords,
                        is_pure=not (cartesian and l >= 2),
                        embed_normalization_into_coefficients=embed_normalization_into_coefficients,
                    )
                )
    return basis


def _add_atom_basis_to_basis(
    basis, atom_basis, coords, embed_normalization_into_coefficients, cartesian=False
):
    for shell in atom_basis:
        angular_momentum = list(map(int, shell["angular_momentum"]))
        exponents = list(map(float, shell["exponents"]))

        for l, subshell_coefficients in itertools.zip_longest(
            angular_momentum,
            shell["coefficients"],
            fillvalue=angular_momentum[-1],
        ):
            coefficients = list(map(float, subshell_coefficients))
            basis.add(
                Shell(
                    l,
                    exponents,
                    coefficients,
                    coords,
                    is_pure=not (cartesian and l >= 2),
                    embed_normalization_into_coefficients=embed_normalization_into_coefficients,
                )
            )
    return basis


_G94_L = {"S": [0], "P": [1], "D": [2], "F": [3], "G": [4], "H": [5], "I": [6], "SP": [0, 1]}


def parse_gaussian94(filename, Z):
    """
    Read the shells of element ``Z`` from a Gaussian94-format (.gbs) basis-set file and
    return them in Basis Set Exchange ``electron_shells`` format.

    Exponents and coefficients may use Fortran ``D`` exponents. ``SP`` shells are split
    into an s and a p contraction that share exponents.
    """
    symbol = {v: k for k, v in ATOM_SYMBOL_TO_Z.items()}[Z]
    with open(filename, "r") as f:
        lines = [line.strip() for line in f]

    shells = []
    found = False
    i = 0
    while i < len(lines):
        line = lines[i]
        tokens = line.split()
        # An element block starts with "<symbol> 0" and ends with "****".
        if (
            not found
            and len(tokens) == 2
            and tokens[0].upper() == symbol.upper()
            and tokens[1] == "0"
        ):
            found = True
            i += 1
            continue
        if found:
            if line.startswith("****"):
                break
            if not line or line.startswith("!"):
                i += 1
                continue
            l_label, nprim = tokens[0].upper(), int(tokens[1])
            exps, coefs = [], [[] for _ in _G94_L[l_label]]
            for k in range(nprim):
                vals = [float(x.replace("D", "E").replace("d", "e")) for x in lines[i + 1 + k].split()]
                exps.append(vals[0])
                for j in range(len(coefs)):
                    coefs[j].append(vals[1 + j])
            shells.append(
                {
                    "angular_momentum": _G94_L[l_label],
                    "exponents": exps,
                    "coefficients": coefs,
                }
            )
            i += nprim + 1
            continue
        i += 1
    if not found:
        raise ValueError(f"Element {symbol} not found in basis file {filename}.")
    return shells
