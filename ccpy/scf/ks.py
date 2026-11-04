"""Restricted and unrestricted Kohn-Sham DFT.

The Fock matrix is F = h + J - K_x + V_xc with the density-fitted J (and exact exchange
K_x = hyb K + (alpha - hyb) K_omega for hybrids and range-separated hybrids), and V_xc
integrated on a Becke/Lebedev grid. The SCF energy expressions of RHF/UHF count
1/2 sum_s tr(D_s F_s); replacing their 1/2 sum_s tr(D_s V_xc,s) by E_xc gives the KS energy.

    mf = RKS(mol, xc="b3lyp")
    mf.kernel()
"""

import numpy as np

from ccpy.dft.grid import MolecularGrid
from ccpy.dft.xc import XCFunctional
from ccpy.integrals.df import DirectFitting
from ccpy.scf.rhf import RHF
from ccpy.scf.uhf import UHF


class _KohnSham:
    """Functional, grid and exact exchange shared by RKS and UKS.

    Parameters
    ----------
    xc : str
        Functional name understood by libxc, e.g. "svwn", "pbe", "b3lyp", "pbe0", "wb97x",
        "camb3lyp", "tpss", "r2scan", or "hf".
    grid : tuple[int, int], optional
        (radial, Lebedev) points for every atom; default PySCF's level-3 grid.
    prune : bool, optional, default=True
        NWChem pruning of the angular grids (as PySCF's default).
    """

    def __init__(self, molecule, xc, grid=None, prune=True, **kwargs):
        self.xc = xc
        self.grid = grid
        self.prune = prune
        self.functional = XCFunctional(xc)
        if molecule.cholesky_tei and abs(self.functional.omega) > 1e-10:
            raise NotImplementedError("Range-separated hybrids require density fitting.")
        self._grid = None
        self._omega_fitting = None
        super().__init__(molecule, **kwargs)

    def _init_options(self):
        return {**super()._init_options(), "xc": self.xc, "grid": self.grid, "prune": self.prune}

    def _xc(self, densities):
        """E_xc and V_xc of the densities; the grid is built on first use."""
        if self._grid is None:
            self._grid = MolecularGrid(self.mol, self.grid, self.prune)
            if self.verbose:
                print(f"   XC functional: {self.xc} ({self.functional.kind}, exact exchange "
                      f"{self.functional.hyb:.4f}, omega {self.functional.omega:.4f})")
                print(f"   Integration grid: {len(self._grid)} points")
        return self.functional.potential(self.mol, self._grid, densities)

    def _exact_exchange(self, df, occupied):
        """hyb K + (alpha - hyb) K_omega for each set of occupied orbitals."""
        f = self.functional
        K = [f.hyb * k for k in df.build_K(occupied)] if f.hyb != 0.0 else [0.0] * len(occupied)
        if self._attenuated is not None:
            K = [k + (f.alpha - f.hyb) * kw for k, kw in zip(K, self._attenuated.build_K(occupied))]
        return [k + np.zeros((self.nbf, self.nbf)) for k in K]

    def _exact_exchange_pair(self, df, pairs):
        """Exact-exchange part of the response Fock: hyb K + (alpha - hyb) K_omega of the
        density changes L R^T + R L^T."""
        f = self.functional
        K = [f.hyb * k for k in df.build_K_pair(pairs)] if f.hyb != 0.0 else [0.0] * len(pairs)
        if self._attenuated is not None:
            K = [k + (f.alpha - f.hyb) * kw for k, kw in zip(K, self._attenuated.build_K_pair(pairs))]
        return [k + np.zeros((self.nbf, self.nbf)) for k in K]

    @property
    def _attenuated(self):
        """Fitting of the erf (omega > 0) or erfc (omega < 0) exchange kernel in its own
        metric, or None without range-separated exchange. Stored unless the Coulomb fitting
        is integral-direct."""
        f = self.functional
        if abs(f.omega) <= 1e-10 or f.alpha == f.hyb:
            return None
        if self._omega_fitting is None:
            self._omega_fitting = DirectFitting(self.mol, op=1 if f.omega > 0 else 2, omega=abs(f.omega),
                                                store=not self.mol.df.direct)
        return self._omega_fitting

    def _energy(self, H, F):
        return super()._energy(H, F) + self._exc_shift

    def gradient(self):
        """Density-fitted analytic KS nuclear gradient, shape (natoms, 3), Eh/bohr.

        The grid weights are held fixed (no grid response, as PySCF's default), which
        leaves an error of the order of the grid's integration error.
        """
        from ccpy.scf.hf_gradient import _check_df, scf_gradient

        _check_df(self, self.method)
        f = self.functional
        spin = [min(s, len(self.C) - 1) for s in (0, 1)]  # RKS uses its orbitals for both spins
        occ = [self.C[s][:, :n] for s, n in zip(spin, (self.na, self.nb))]
        eps = [self.eps[s][:n] for s, n in zip(spin, (self.na, self.nb))]
        grad = scf_gradient(self.mol, occ, eps, k_scale=f.hyb)
        if self._attenuated is not None:
            grad += self._attenuated.exchange_gradient(occ, scale=f.alpha - f.hyb)
        densities = [sum(C @ C.T for C in occ)] if len(self.C) == 1 else [C @ C.T for C in occ]
        return grad + f.nuclear_gradient(self.mol, self._grid, densities)


class RKS(_KohnSham, RHF):
    """Density-fitted restricted Kohn-Sham DFT (closed shells). See ``_KohnSham`` for the
    functional and grid options and ``SCFBase`` for the SCF options."""

    def _build_fock(self, H, df, S):
        D = self.D[0]
        exc, (V,) = self._xc([2.0 * D])
        (J,) = df.build_J([D])
        (K,) = self._exact_exchange(df, [self.C[0][:, : self.na]])
        F = H + 2.0 * J - K + V
        self._exc_shift = exc - np.sum(D * V)
        return [F], [F]

    def _response_fock(self, df, pairs):
        ((L, R),) = pairs
        dD = L @ R.T + R @ L.T
        (J,) = df.build_J([dD])
        (K,) = self._exact_exchange_pair(df, pairs)
        (dV,) = self.functional.response(self.mol, self._grid, [2.0 * self.D[0]], [2.0 * dD])
        return [2.0 * J - K + dV]


class UKS(_KohnSham, UHF):
    """Density-fitted unrestricted Kohn-Sham DFT. See ``_KohnSham`` for the functional and
    grid options, and ``UHF``/``SCFBase`` for the SCF options."""

    def _build_fock(self, H, df, S):
        exc, V = self._xc(self.D)
        Ja, Jb = df.build_J(self.D)
        K = self._exact_exchange(df, [self.C[0][:, : self.na], self.C[1][:, : self.nb]])
        F = [H + Ja + Jb - k + v for k, v in zip(K, V)]
        self._exc_shift = exc - 0.5 * sum(np.sum(d * v) for d, v in zip(self.D, V))
        return F, F

    def _response_fock(self, df, pairs):
        dD = [L @ R.T + R @ L.T for L, R in pairs]
        (J,) = df.build_J([dD[0] + dD[1]])
        K = self._exact_exchange_pair(df, pairs)
        dV = self.functional.response(self.mol, self._grid, self.D, dD)
        return [J - k + v for k, v in zip(K, dV)]
