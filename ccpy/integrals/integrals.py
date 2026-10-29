"""AO integrals computed with libint2 (ported from forte2/integrals/integrals.py).

Every function takes a ``Molecule`` (or any object exposing ``basis``,
``auxiliary_basis``, ``atoms`` and ``natoms``) as its first argument. Basis-set
arguments default to the molecule's orbital basis for one-electron and four-center
integrals and to its auxiliary basis for the auxiliary indices of two- and
three-center integrals, so for instance ``coulomb_3c(mol)`` returns (P|mn) with P in
the auxiliary basis and m, n in the orbital basis.

Derivative integrals are returned already contracted with a weight (density-like)
matrix, as flat vectors of length 3 * natoms in atom-major Cartesian order (bohr).
"""

import numpy as np

from ccpy.integrals import _libint as ints

LIBINT2_MAX_AM = ints.libint2_max_am


def _check_am(max_l, integral_name):
    if max_l > LIBINT2_MAX_AM:
        raise NotImplementedError(
            f"{integral_name} integrals need angular momentum {max_l}, but the libint2 "
            f"library ccpy was built against supports up to {LIBINT2_MAX_AM}."
        )


def _parse_basis_args_1e(molecule, basis1, basis2):
    if basis1 is None and basis2 is not None:
        raise ValueError("If basis2 is provided, basis1 must also be provided.")
    if basis1 is None:
        basis1 = molecule.basis
    if basis2 is None:
        basis2 = basis1
    return basis1, basis2


def _parse_basis_args_2c2e(molecule, basis1, basis2):
    if basis1 is None and basis2 is not None:
        raise ValueError("If basis2 is provided, basis1 must also be provided.")
    if basis1 is None:
        basis1 = molecule.auxiliary_basis
    if basis2 is None:
        basis2 = basis1
    return basis1, basis2


def _parse_basis_args_3c2e(molecule, basis1, basis2, basis3):
    if basis2 is None and basis3 is not None:
        raise ValueError("If basis3 is provided, basis2 must also be provided.")
    if basis1 is None:
        basis1 = molecule.auxiliary_basis
    if basis2 is None:
        basis2 = molecule.basis
    if basis3 is None:
        basis3 = basis2
    return basis1, basis2, basis3


def _parse_basis_args_4c2e(molecule, basis1, basis2, basis3, basis4):
    if basis1 is None and any(b is not None for b in (basis2, basis3, basis4)):
        raise ValueError(
            "If any of basis2, basis3, or basis4 is provided, basis1 must also be provided."
        )
    if basis1 is None:
        basis1 = molecule.basis
    if all(b is None for b in (basis2, basis3, basis4)):
        basis2 = basis3 = basis4 = basis1
    return basis1, basis2, basis3, basis4


def _max_l(*bases):
    return max(b.max_l for b in bases)


# --------------------------------------------------------------------------- #
# Scalars
# --------------------------------------------------------------------------- #
def nuclear_repulsion(molecule):
    """Point-charge nuclear repulsion energy."""
    return ints.nuclear_repulsion(molecule.atoms)


def nuclear_repulsion_deriv(molecule):
    """First derivative of the nuclear repulsion energy, shape (natoms, 3)."""
    charges = np.asarray([atom[0] for atom in molecule.atoms], dtype=float)
    positions = np.asarray([atom[1] for atom in molecule.atoms], dtype=float)
    gradient = np.zeros((len(charges), 3))
    for a in range(len(charges)):
        for b in range(a + 1, len(charges)):
            rab = positions[a] - positions[b]
            contribution = charges[a] * charges[b] * rab / np.linalg.norm(rab) ** 3
            gradient[a] -= contribution
            gradient[b] += contribution
    return gradient


# --------------------------------------------------------------------------- #
# One-electron integrals
# --------------------------------------------------------------------------- #
def overlap(molecule, basis1=None, basis2=None):
    r"""S_{mn} = <m|n>."""
    basis1, basis2 = _parse_basis_args_1e(molecule, basis1, basis2)
    _check_am(_max_l(basis1, basis2), "Overlap")
    return ints.overlap(basis1, basis2)


def kinetic(molecule, basis1=None, basis2=None):
    r"""T_{mn} = <m|-\nabla^2/2|n>."""
    basis1, basis2 = _parse_basis_args_1e(molecule, basis1, basis2)
    _check_am(_max_l(basis1, basis2), "Kinetic")
    return ints.kinetic(basis1, basis2)


def nuclear(molecule, basis1=None, basis2=None):
    r"""V_{mn} = <m| -\sum_A Z_A / |r - R_A| |n> (point nuclei)."""
    basis1, basis2 = _parse_basis_args_1e(molecule, basis1, basis2)
    _check_am(_max_l(basis1, basis2), "Nuclear attraction")
    return ints.nuclear(basis1, basis2, molecule.atoms)


def hcore(molecule, basis1=None, basis2=None):
    """Core Hamiltonian T + V."""
    return kinetic(molecule, basis1, basis2) + nuclear(molecule, basis1, basis2)


def emultipole1(molecule, basis1=None, basis2=None, origin=None):
    """[S, mu_x, mu_y, mu_z]: overlap and electronic dipole integrals <m|r_a|n>
    about ``origin`` (default: the coordinate origin)."""
    basis1, basis2 = _parse_basis_args_1e(molecule, basis1, basis2)
    _check_am(_max_l(basis1, basis2), "Multipole")
    return ints.emultipole1(basis1, basis2, [0.0] * 3 if origin is None else list(origin))


def emultipole2(molecule, basis1=None, basis2=None, origin=None):
    """[S, x, y, z, xx, xy, xz, yy, yz, zz]: multipole integrals up to quadrupoles."""
    basis1, basis2 = _parse_basis_args_1e(molecule, basis1, basis2)
    _check_am(_max_l(basis1, basis2), "Multipole")
    return ints.emultipole2(basis1, basis2, [0.0] * 3 if origin is None else list(origin))


def emultipole3(molecule, basis1=None, basis2=None, origin=None):
    """Multipole integrals up to octupoles (20 matrices, libint2 ordering)."""
    basis1, basis2 = _parse_basis_args_1e(molecule, basis1, basis2)
    _check_am(_max_l(basis1, basis2), "Multipole")
    return ints.emultipole3(basis1, basis2, [0.0] * 3 if origin is None else list(origin))


def opVop(molecule, basis1=None, basis2=None):
    r"""Small-component nuclear potential [p.Vp, (pxVp)_x, (pxVp)_y, (pxVp)_z]."""
    basis1, basis2 = _parse_basis_args_1e(molecule, basis1, basis2)
    _check_am(_max_l(basis1, basis2), "opVop")
    return ints.opVop(basis1, basis2, molecule.atoms)


def erf_nuclear(molecule, omega, basis1=None, basis2=None):
    r"""Long-range nuclear attraction with erf(omega r)/r."""
    basis1, basis2 = _parse_basis_args_1e(molecule, basis1, basis2)
    _check_am(_max_l(basis1, basis2), "erf-attenuated nuclear")
    return ints.erf_nuclear(basis1, basis2, (omega, molecule.atoms))


def erfc_nuclear(molecule, omega, basis1=None, basis2=None):
    r"""Short-range nuclear attraction with erfc(omega r)/r."""
    basis1, basis2 = _parse_basis_args_1e(molecule, basis1, basis2)
    _check_am(_max_l(basis1, basis2), "erfc-attenuated nuclear")
    return ints.erfc_nuclear(basis1, basis2, (omega, molecule.atoms))


# --------------------------------------------------------------------------- #
# One-electron derivative integrals (contracted with a weight matrix)
# --------------------------------------------------------------------------- #
def _check_weights(weights, shape, name):
    weights = np.ascontiguousarray(np.asarray(weights, dtype=float))
    if weights.shape != shape:
        raise ValueError(f"Expected {name} weights of shape {shape}, got {weights.shape}.")
    return weights


def overlap_deriv(molecule, weights, basis1=None, basis2=None):
    r"""G_{Ax} = \sum_{mn} W_{mn} d S_{mn} / d R_{Ax}, flat (3 natoms,)."""
    basis1, basis2 = _parse_basis_args_1e(molecule, basis1, basis2)
    _check_am(_max_l(basis1, basis2), "Overlap derivative")
    W = _check_weights(weights, (basis1.size, basis2.size), "overlap")
    return ints.overlap_deriv(basis1, basis2, W, molecule.atoms)


def kinetic_deriv(molecule, weights, basis1=None, basis2=None):
    r"""G_{Ax} = \sum_{mn} W_{mn} d T_{mn} / d R_{Ax}, flat (3 natoms,)."""
    basis1, basis2 = _parse_basis_args_1e(molecule, basis1, basis2)
    _check_am(_max_l(basis1, basis2), "Kinetic derivative")
    W = _check_weights(weights, (basis1.size, basis2.size), "kinetic")
    return ints.kinetic_deriv(basis1, basis2, W, molecule.atoms)


def nuclear_deriv(molecule, weights, basis1=None, basis2=None):
    r"""G_{Ax} = \sum_{mn} W_{mn} d V_{mn} / d R_{Ax} (basis-function and operator
    centers), flat (3 natoms,)."""
    basis1, basis2 = _parse_basis_args_1e(molecule, basis1, basis2)
    _check_am(_max_l(basis1, basis2), "Nuclear derivative")
    W = _check_weights(weights, (basis1.size, basis2.size), "nuclear")
    return ints.nuclear_deriv(basis1, basis2, W, molecule.atoms)


def hcore_deriv(molecule, weights, basis1=None, basis2=None):
    """Kinetic + nuclear-attraction derivatives contracted with ``weights``."""
    return kinetic_deriv(molecule, weights, basis1, basis2) + nuclear_deriv(
        molecule, weights, basis1, basis2
    )


# --------------------------------------------------------------------------- #
# Two-electron integrals
# --------------------------------------------------------------------------- #
def coulomb_4c(molecule, basis1=None, basis2=None, basis3=None, basis4=None):
    r"""(mn|rs) in chemists' notation, shape (n1, n2, n3, n4). O(N^4) memory."""
    b1, b2, b3, b4 = _parse_basis_args_4c2e(molecule, basis1, basis2, basis3, basis4)
    _check_am(_max_l(b1, b2, b3, b4), "Four-center Coulomb")
    return ints.coulomb_4c(b1, b2, b3, b4)


def coulomb_3c(molecule, basis1=None, basis2=None, basis3=None):
    r"""(P|mn), shape (naux, n2, n3)."""
    b1, b2, b3 = _parse_basis_args_3c2e(molecule, basis1, basis2, basis3)
    _check_am(_max_l(b1, b2, b3), "Three-center Coulomb")
    return ints.coulomb_3c(b1, b2, b3)


def coulomb_3c_by_shell(molecule, shell_slices, basis1=None, basis2=None, basis3=None):
    r"""(P|mn) for the shell ranges [(P0, P1), (m0, m1), (n0, n1)]."""
    b1, b2, b3 = _parse_basis_args_3c2e(molecule, basis1, basis2, basis3)
    _check_am(_max_l(b1, b2, b3), "Three-center Coulomb")
    return ints.coulomb_3c_by_shell(b1, b2, b3, shell_slices)


def coulomb_2c(molecule, basis1=None, basis2=None):
    r"""(P|Q), shape (n1, n2)."""
    b1, b2 = _parse_basis_args_2c2e(molecule, basis1, basis2)
    _check_am(_max_l(b1, b2), "Two-center Coulomb")
    return ints.coulomb_2c(b1, b2)


def coulomb_3c_opVop(molecule, basis1=None, basis2=None, basis3=None):
    r"""\sum_P (P|(sigma.p m)(sigma.p n)) in scalar, x, y, z Pauli-component order."""
    b1, b2, b3 = _parse_basis_args_3c2e(molecule, basis1, basis2, basis3)
    _check_am(_max_l(b1, b2, b3), "Three-center opVop")
    return ints.coulomb_3c_opVop(b1, b2, b3)


def erf_coulomb_3c(molecule, omega, basis1=None, basis2=None, basis3=None):
    b1, b2, b3 = _parse_basis_args_3c2e(molecule, basis1, basis2, basis3)
    _check_am(_max_l(b1, b2, b3), "erf-attenuated three-center Coulomb")
    return ints.erf_coulomb_3c(b1, b2, b3, omega)


def erf_coulomb_2c(molecule, omega, basis1=None, basis2=None):
    b1, b2 = _parse_basis_args_2c2e(molecule, basis1, basis2)
    _check_am(_max_l(b1, b2), "erf-attenuated two-center Coulomb")
    return ints.erf_coulomb_2c(b1, b2, omega)


def erfc_coulomb_3c(molecule, omega, basis1=None, basis2=None, basis3=None):
    b1, b2, b3 = _parse_basis_args_3c2e(molecule, basis1, basis2, basis3)
    _check_am(_max_l(b1, b2, b3), "erfc-attenuated three-center Coulomb")
    return ints.erfc_coulomb_3c(b1, b2, b3, omega)


def erfc_coulomb_2c(molecule, omega, basis1=None, basis2=None):
    b1, b2 = _parse_basis_args_2c2e(molecule, basis1, basis2)
    _check_am(_max_l(b1, b2), "erfc-attenuated two-center Coulomb")
    return ints.erfc_coulomb_2c(b1, b2, omega)


# --------------------------------------------------------------------------- #
# Two-electron derivative integrals (contracted with weights)
# --------------------------------------------------------------------------- #
def coulomb_3c_deriv(molecule, W3, basis1=None, basis2=None, basis3=None):
    r"""G_{Ax} = \sum_{Pmn} W^P_{mn} d(P|mn)/dR_{Ax}, flat (3 natoms,)."""
    b1, b2, b3 = _parse_basis_args_3c2e(molecule, basis1, basis2, basis3)
    _check_am(_max_l(b1, b2, b3), "Three-center Coulomb derivative")
    W3 = np.ascontiguousarray(W3)
    if W3.shape != (b1.size, b2.size, b3.size):
        raise ValueError(
            f"Expected W3 of shape {(b1.size, b2.size, b3.size)}, got {W3.shape}."
        )
    return ints.coulomb_3c_deriv(b1, b2, b3, W3, molecule.atoms)


def coulomb_4c_deriv(molecule, W4, symmetrize=True):
    r"""G_{Ax} = \sum_{mnrs} W_{mnrs} d(mn|rs)/dR_{Ax}, flat (3 natoms,), orbital basis only.

    Only symmetry-unique shell quartets are computed, so the kernel needs W with the 8-fold
    permutational symmetry of (mn|rs); with ``symmetrize=True`` (default) W is symmetrized
    first, which leaves the contraction unchanged.
    """
    basis = molecule.basis
    _check_am(basis.max_l, "Four-center Coulomb derivative")
    W4 = np.asarray(W4, dtype=float)
    n = basis.size
    if W4.shape != (n, n, n, n):
        raise ValueError(f"Expected W4 of shape {(n, n, n, n)}, got {W4.shape}.")
    if symmetrize:
        W4 = 0.5 * (W4 + W4.transpose(1, 0, 2, 3))
        W4 = 0.5 * (W4 + W4.transpose(0, 1, 3, 2))
        W4 = 0.5 * (W4 + W4.transpose(2, 3, 0, 1))
    return ints.coulomb_4c_deriv(basis, np.ascontiguousarray(W4), molecule.atoms)


def coulomb_2c_deriv(molecule, W2, basis1=None, basis2=None):
    r"""G_{Ax} = \sum_{PQ} W_{PQ} d(P|Q)/dR_{Ax}, flat (3 natoms,)."""
    b1, b2 = _parse_basis_args_2c2e(molecule, basis1, basis2)
    _check_am(_max_l(b1, b2), "Two-center Coulomb derivative")
    W2 = np.ascontiguousarray(W2)
    if W2.shape != (b1.size, b2.size):
        raise ValueError(f"Expected W2 of shape {(b1.size, b2.size)}, got {W2.shape}.")
    return ints.coulomb_2c_deriv(b1, b2, W2, molecule.atoms)


# --------------------------------------------------------------------------- #
# Basis functions on a grid
# --------------------------------------------------------------------------- #
def basis_at_points(molecule, points, basis=None):
    """Values of the AO basis functions at ``points`` (npoints x 3, bohr)."""
    return ints.basis_at_points(molecule.basis if basis is None else basis, points)


def orbitals_at_points(molecule, points, C, basis=None):
    """Values of the orbitals with AO coefficients ``C`` at ``points``."""
    return ints.orbitals_at_points(molecule.basis if basis is None else basis, points, C)
