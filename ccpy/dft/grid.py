"""Molecular integration grids for DFT.

Treutler-Ahlrichs (M4) radial grids with element-specific scaling, Lebedev angular grids, and Becke's fuzzy-cell
partitioning with Treutler's atomic-size adjustment. The defaults follow PySCF's
level-3 grids (unpruned), so XC energies can be checked against PySCF on identical grids.

    O. Treutler and R. Ahlrichs, J. Chem. Phys. 102, 346 (1995)
    A. D. Becke, J. Chem. Phys. 88, 2547 (1988)
"""

import numpy as np
from scipy.integrate import lebedev_rule

from ccpy.integrals.atom_data import ANGSTROM_TO_BOHR

# Bragg-Slater radii (Angstrom) by atomic number, index 0 for ghost atoms; J. C. Slater,
# J. Chem. Phys. 41, 3199 (1964), with H = 0.35 as recommended by Becke.
BRAGG_RADII = ANGSTROM_TO_BOHR * np.array([
    2.00, 0.35, 1.40, 1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50, 1.80, 1.50, 1.25, 1.10,
    1.00, 1.00, 1.00, 1.80, 2.20, 1.80, 1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35,
    1.35, 1.30, 1.25, 1.15, 1.15, 1.15, 1.90, 2.35, 2.00, 1.80, 1.55, 1.45, 1.45, 1.35, 1.30,
    1.35, 1.40, 1.60, 1.55, 1.55, 1.45, 1.45, 1.40, 1.40, 2.10, 2.60, 2.15, 1.95, 1.85, 1.85,
    1.85, 1.85, 1.85, 1.85, 1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.55, 1.45, 1.35,
    1.35, 1.30, 1.35, 1.35, 1.35, 1.50, 1.90, 1.80, 1.60, 1.90, 1.45, 2.10,
])

# Treutler-Ahlrichs radial scaling xi by atomic number (index 0 for ghost atoms): H-Kr from
# the paper, heavier elements as in Psi4 and PySCF
TREUTLER_XI = np.array([
    1.0, 0.8, 0.9, 1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9, 1.4, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0,
    1.0, 1.5, 1.4, 1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1, 1.1, 1.0, 0.9, 0.9, 0.9,
    0.9, 2.0, 1.7, 1.5, 1.5, 1.35, 1.35, 1.25, 1.2, 1.25, 1.3, 1.5, 1.5, 1.3, 1.2, 1.2, 1.15,
    1.15, 1.15, 2.5, 2.2, 2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5,
    2.1, 3.685, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
])

# Default (radial, Lebedev) points by period 1-7: PySCF's grid level 3
DEFAULT_RADIAL = (50, 75, 80, 90, 95, 100, 105)
DEFAULT_ANGULAR = (302, 302, 434, 434, 434, 434, 434)
_LAST_Z_OF_PERIOD = np.array((2, 10, 18, 36, 54, 86, 118))

# Lebedev rules: number of points -> algebraic order
LEBEDEV_ORDER = {
    6: 3, 14: 5, 26: 7, 38: 9, 50: 11, 74: 13, 86: 15, 110: 17, 146: 19, 170: 21, 194: 23,
    230: 25, 266: 27, 302: 29, 350: 31, 434: 35, 590: 41, 770: 47, 974: 53, 1202: 59,
    1454: 65, 1730: 71, 2030: 77, 2354: 83, 2702: 89, 3074: 95, 3470: 101, 3890: 107,
    4334: 113, 4802: 119, 5294: 125, 5810: 131,
}


def treutler_ahlrichs(n, xi=1.0):
    """Radial points r and weights r^2 dr of the M4 mapping, r ascending."""
    step = np.pi / (n + 1)
    x = np.cos(np.arange(1, n + 1) * step)
    scale = xi * (1.0 + x) ** 0.6 / np.log(2.0)
    r = -scale * np.log(0.5 * (1.0 - x))
    dr = step * np.sin(np.arange(1, n + 1) * step) * scale * (
        -0.6 / (1.0 + x) * np.log(0.5 * (1.0 - x)) + 1.0 / (1.0 - x))
    return r[::-1], (r**2 * dr)[::-1]


def lebedev(npoints):
    """Unit vectors (npoints, 3) and weights (summing to 4 pi) of a Lebedev rule."""
    if npoints not in LEBEDEV_ORDER:
        raise ValueError(f"No Lebedev rule with {npoints} points; choose from {sorted(LEBEDEV_ORDER)}.")
    x, w = lebedev_rule(LEBEDEV_ORDER[npoints])
    return x.T, w


def nwchem_prune(Z, r, nang):
    """Lebedev points of each radial shell: fewer near the nucleus and far out, by the
    shell radius relative to the Bragg radius (NWChem's scheme, PySCF's default)."""
    sizes = sorted(LEBEDEV_ORDER)[3:]  # 38, 50, 74, ...
    if nang < 50:
        return np.full(len(r), nang)
    levels = [1, 2, 2, 2, 1] if nang == 50 else [1, 3, sizes.index(nang) - 1, sizes.index(nang), sizes.index(nang) - 1]
    alphas = (0.25, 0.5, 1.0, 4.5) if Z <= 2 else (0.1667, 0.5, 0.9, 3.5) if Z <= 10 else (0.1, 0.4, 0.8, 2.5)
    region = np.sum(r[:, None] / BRAGG_RADII[Z] > np.array(alphas)[None, :], axis=1)
    return np.array(sizes)[np.array(levels)[region]]


def atomic_grid(nrad, nang, xi=1.0, Z=None):
    """Points and weights of one atom-centered grid (before partitioning), pruned when
    the atomic number Z is given."""
    r, wr = treutler_ahlrichs(nrad, xi)
    angular = np.full(nrad, nang) if Z is None else nwchem_prune(Z, r, nang)
    coords, weights = [], []
    for n in np.unique(angular):
        u, wu = lebedev(int(n))
        shells = angular == n
        coords.append((r[shells, None, None] * u[None, :, :]).reshape(-1, 3))
        weights.append(np.outer(wr[shells], wu).ravel())
    return np.vstack(coords), np.concatenate(weights)


def becke_weights(coords, atom_coords, Z):
    """Becke cell functions P_A(r) of every atom, shape (natoms, npoints), not normalized."""
    natoms = len(Z)
    dist = np.linalg.norm(coords[None, :, :] - atom_coords[:, None, :], axis=2)
    R = np.linalg.norm(atom_coords[:, None, :] - atom_coords[None, :, :], axis=2)
    # Treutler's size adjustment from sqrt(Bragg radii), a_ij = (chi_ji - chi_ij) / 4
    rad = np.sqrt(BRAGG_RADII[Z])
    chi = rad[:, None] / rad[None, :]
    a = np.clip(0.25 * (chi.T - chi), -0.5, 0.5)
    P = np.ones((natoms, len(coords)))
    for i in range(natoms):
        for j in range(i):
            g = (dist[i] - dist[j]) / R[i, j]
            g = g + a[i, j] * (1.0 - g * g)
            for _ in range(3):
                g = 0.5 * g * (3.0 - g * g)
            P[i] *= 0.5 * (1.0 - g)
            P[j] *= 0.5 * (1.0 + g)
    return P


class MolecularGrid:
    """Quadrature grid of a Molecule: ``coords`` (npoints, 3, bohr) and ``weights``.

    Parameters
    ----------
    molecule : ccpy.integrals.Molecule
    grid : tuple[int, int], optional
        (radial, Lebedev) points used for every atom. The default depends on the period
        of each atom (PySCF's level 3).
    prune : bool, optional, default=True
        Reduce the Lebedev points of the inner and outer radial shells (NWChem scheme).
    """

    def __init__(self, molecule, grid=None, prune=True):
        Z = np.asarray(molecule.atomic_charges, dtype=int)
        xyz = np.asarray(molecule.atomic_positions, dtype=float)
        coords, weights = [], []
        for A, (z, center) in enumerate(zip(Z, xyz)):
            if grid is None:
                period = int(np.sum(z > _LAST_Z_OF_PERIOD))
                nrad, nang = DEFAULT_RADIAL[period], DEFAULT_ANGULAR[period]
            else:
                nrad, nang = grid
            c, w = atomic_grid(nrad, nang, TREUTLER_XI[z], z if prune else None)
            c = c + center
            P = becke_weights(c, xyz, Z)
            coords.append(c)
            weights.append(w * P[A] / P.sum(axis=0))
        coords, weights = np.vstack(coords), np.concatenate(weights)
        order = _morton_order(coords)
        self.coords, self.weights = coords[order], weights[order]

    def __len__(self):
        return len(self.weights)

    def blocks(self, size=1024):
        """(coords, weights) in blocks of spatially close points."""
        for start in range(0, len(self), size):
            yield self.coords[start : start + size], self.weights[start : start + size]


def _morton_order(coords, box=1.0):
    """Order of the points along a Z-order curve through boxes of side ``box`` bohr, so
    that consecutive points are close in space."""
    q = np.floor((coords - coords.min(axis=0)) / box).astype(np.int64)

    def spread(v):  # insert two zero bits between the bits of v (v < 2^21)
        v = (v | (v << 32)) & 0x1F00000000FFFF
        v = (v | (v << 16)) & 0x1F0000FF0000FF
        v = (v | (v << 8)) & 0x100F00F00F00F00F
        v = (v | (v << 4)) & 0x10C30C30C30C30C3
        return (v | (v << 2)) & 0x1249249249249249

    return np.argsort(spread(q[:, 0]) | (spread(q[:, 1]) << 1) | (spread(q[:, 2]) << 2), kind="stable")
