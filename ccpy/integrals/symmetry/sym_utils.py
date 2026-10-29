"""Abelian point groups: operations, character tables, irrep labels and products.

Adapted from forte2 (https://github.com/evangelistalab/forte2, LGPL-3.0); see
THIRD_PARTY_NOTICES.md.
"""

from numbers import Integral

import numpy as np

SYMMETRY_OPS = {
    "C2V": ["E", "C2z", "σ_xz", "σ_yz"],
    "C2H": ["E", "C2z", "i", "σ_xy"],
    "D2": ["E", "C2z", "C2y", "C2x"],
    "D2H": ["E", "C2z", "C2y", "C2x", "i", "σ_xy", "σ_xz", "σ_yz"],
    "CS": ["E", "σ_xy"],
    "CI": ["E", "i"],
    "C2": ["E", "C2z"],
    "C1": ["E"],
}

# Full 1-D character tables (±1 per operation, in the same order as _SYMMETRY_OPS[group])
CHARACTER_TABLE = {
    "C2V": {
        "a1": [+1, +1, +1, +1],
        "a2": [+1, +1, -1, -1],
        "b1": [+1, -1, +1, -1],
        "b2": [+1, -1, -1, +1],
    },
    "C2H": {
        "ag": [+1, +1, +1, +1],
        "au": [+1, +1, -1, -1],
        "bg": [+1, -1, +1, -1],
        "bu": [+1, -1, -1, +1],
    },
    "D2": {
        "a": [+1, +1, +1, +1],
        "b1": [+1, +1, -1, -1],
        "b2": [+1, -1, +1, -1],
        "b3": [+1, -1, -1, +1],
    },
    "D2H": {
        "ag": [+1, +1, +1, +1, +1, +1, +1, +1],
        "au": [+1, +1, +1, +1, -1, -1, -1, -1],
        "b1g": [+1, +1, -1, -1, +1, +1, -1, -1],
        "b1u": [+1, +1, -1, -1, -1, -1, +1, +1],
        "b2g": [+1, -1, +1, -1, +1, -1, +1, -1],
        "b2u": [+1, -1, +1, -1, -1, +1, -1, +1],
        "b3g": [+1, -1, -1, +1, +1, -1, -1, +1],
        "b3u": [+1, -1, -1, +1, -1, +1, +1, -1],
    },
    "CS": {"a'": [+1, +1], "a''": [+1, -1]},
    "CI": {"g": [+1, +1], "u": [+1, -1]},
    "C2": {"a": [+1, +1], "b": [+1, -1]},
    "C1": {"a": [+1]},
}

COTTON_LABELS = {
    "C1": {"a": 0},
    "CI": {"g": 0, "u": 1},
    "C2": {"a": 0, "b": 1},
    "CS": {"a'": 0, "a''": 1},
    "D2": {"a": 0, "b1": 1, "b2": 2, "b3": 3},
    "C2V": {"a1": 0, "a2": 1, "b1": 2, "b2": 3},
    "C2H": {"ag": 0, "bg": 1, "au": 2, "bu": 3},
    "D2H": {
        "ag": 0,
        "b1g": 1,
        "b2g": 2,
        "b3g": 3,
        "au": 4,
        "b1u": 5,
        "b2u": 6,
        "b3u": 7,
    },
}


def equivalent_under_operation(coords, charges, op, tol):
    """
    Check if a set of coordinates and charges is equivalent to itself under a
    given symmetry operation.

    Parameters
    ----------
    coords : ndarray of shape (N, 3)
        The coordinates of the atoms.
    charges : list of length N
        The nuclear charges of the atoms.
    op : callable
        A function representing the symmetry operation that takes a 3-vector and returns the transformed 3-vector.
    tol : float
        The tolerance for comparing distances.

    Returns
    -------
    has_op : bool
        True if the system is invariant under the operation, False otherwise.
    """
    has_op = True
    for Za, Ra in zip(charges, coords):
        found = False
        for Zb, Rb in zip(charges, coords):
            if Za == Zb and np.linalg.norm(Ra - op(Rb)) < tol:
                found = True
                break
        if not found:
            has_op = False
            break
    return has_op


def rotation_mat(axis, theta):
    """
    Euler-Rodrigues formula for rotation matrix

    Parameters
    ----------
    axis : ndarray of shape (3,)
        The axis to rotate around. Will be normalized internally.
    theta : float
        Rotation angle

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)

    R = np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )
    return R


def reflection_mat(plane):
    """
    Return the 3x3 reflection matrix for the mirror plane σ(x_i, x_j).

    Parameters
    ----------
    plane: tuple (i, j) with i != j and i,j ∈ {0,1,2} mapping to x,y,z.
           Example: (0,1) -> σ_xy (flip z).

    Returns
    -------
    R : ndarray of shape (3, 3)
    The reflection matrix.
    """
    i, j = plane
    if i == j:
        raise ValueError("plane must use two distinct axes")
    R = np.eye(3)
    k = 3 - i - j  # the axis not in the plane (since {0,1,2})
    R[k, k] = -1.0
    return R


def local_sign(l, m, op):
    """
    Return phase describing how the spherical harmonic Y_lm transforms under Abelian symmetry
    operations `op`, which is one of [E, C2z, C2x, C2y, σ_xy, σ_xz, σ_yz].
    """
    ma = abs(m)
    if op == "E":
        return 1
    if op == "i":
        return (-1) ** l
    if op == "C2z":
        return (-1) ** ma
    if op == "C2x":
        if m == 0:
            return (-1) ** l
        return (-1) ** (l + ma) if m > 0 else (-1) ** (l + ma + 1)
    if op == "C2y":
        if m == 0:
            return (-1) ** l
        return (-1) ** l if m > 0 else (-1) ** (l + 1)
    if op == "σ_xz":
        return 1 if m >= 0 else -1
    if op == "σ_yz":
        if m == 0:
            return 1
        return (-1) ** ma if m > 0 else (-1) ** (ma + 1)
    if op == "σ_xy":
        return (-1) ** (l + ma)
    raise ValueError(f"Unknown op {op}")


def get_symmetry_ops(point_group):
    """
    Compute 3x3 matrix representations for the symmetry operators in `point_group`.
    These representation perform reflections/rotations in the molecular principal frame.
    """
    symmetry_ops = {}

    axes = {"x": 0, "y": 1, "z": 2}
    I = np.eye(3)

    ops = SYMMETRY_OPS[point_group]
    for op in ops:
        if op == "E":
            symmetry_ops[op] = I
        elif "C2" in op:
            symmetry_ops[op] = rotation_mat(I[:, axes[op[-1]]], np.deg2rad(180.0))
        elif op == "i":
            symmetry_ops[op] = -I
        elif "σ_" in op:
            symmetry_ops[op] = reflection_mat((axes[op[-2]], axes[op[-1]]))
    return symmetry_ops


def get_irrep_index(pg, value):
    """
    Return the Cotton index of an irrep given by its label or index.

    Parameters
    ----------
    pg : str
        The Abelian point group.
    value : str | int
        The irrep's label, in any case, or its Cotton index.

    Returns
    -------
    int
        The irrep's Cotton index.

    Raises
    ------
    ValueError
        If ``value`` is not an irrep of ``pg``.
    """
    labels = COTTON_LABELS[pg]
    if isinstance(value, str):
        # also accept the labels of ccpy.utilities.symmetry ('A"' for Cs, "AG"/"AU" for Ci)
        key = value.strip().lower().replace('"', "''")
        if pg == "CI":
            key = {"ag": "g", "au": "u"}.get(key, key)
        if key in labels:
            return labels[key]
    is_index = isinstance(value, Integral) and not isinstance(value, bool)
    if is_index and 0 <= value < len(labels):
        return int(value)
    raise ValueError(
        f"Unknown irrep {value!r} for {pg}. The irreps are {list(labels)}."
    )


def irrep_product(irreps):
    """
    Return the direct product of irreps of an Abelian point group.

    Parameters
    ----------
    irreps : ArrayLike
        The Cotton indices of the irreps.

    Returns
    -------
    int
        The Cotton index of their product. An empty product is totally symmetric.

    Raises
    ------
    ValueError
        If an irrep is not an integer from 0 to 7.
    """
    irreps = np.asarray(irreps)
    if irreps.size == 0:
        return 0
    if not np.issubdtype(irreps.dtype, np.integer):
        raise ValueError(f"Irreps must be integers, but got {irreps.dtype} values.")
    bad = irreps[(irreps < 0) | (irreps > 7)]
    if bad.size:
        raise ValueError(
            f"Irreps must be Cotton indices from 0 to 7, but got {bad.tolist()}."
        )
    # In Cotton order, the product of irreps i and j has index i ^ j.
    return int(np.bitwise_xor.reduce(irreps))
