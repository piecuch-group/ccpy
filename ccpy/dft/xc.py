"""Exchange-correlation functionals: energies and Kohn-Sham potential matrices on a grid.

Densities are carried as the rows [rho, d/dx rho, d/dy rho, d/dz rho, tau] (as many as the
functional needs, tau = 1/2 sum_i |grad phi_i|^2), and libxc returns the derivatives of
e_xc with respect to those rows, so

    V_mn = sum_g w_g [ v_rho phi_m phi_n + v_grad . grad(phi_m phi_n)
                       + 1/2 v_tau grad phi_m . grad phi_n ].

shortcut: libxc is reached through PySCF's bundled wrapper (pyscf.dft.libxc, libxc 6.x),
which also parses functional names; link libxc into ccpy.integrals._libint once PySCF
should stop being a runtime dependency of the DFT code.
"""

import numpy as np

from ccpy.integrals import integrals

_NVAR = {"LDA": 1, "GGA": 4, "MGGA": 5}


class XCFunctional:
    """An exchange-correlation functional by name, e.g. "b3lyp", "pbe0", "wb97x", "svwn".

    Attributes
    ----------
    kind : str
        "LDA", "GGA", "MGGA", or "HF" (no semilocal part).
    hyb : float
        Fraction of full-range exact exchange.
    omega, alpha : float
        Range-separation parameter and coefficient: exact exchange is
        hyb * K + (alpha - hyb) * K_omega, with erf(omega r)/r for omega > 0 and
        erfc(|omega| r)/r for omega < 0 (PySCF's convention).
    """

    def __init__(self, name):
        from pyscf.dft import libxc
        from pyscf.dft.numint import NumInt

        self._eval = NumInt().eval_xc_eff
        self.name = name
        self.kind = libxc.xc_type(name)
        if self.kind == "NLC" or libxc.is_nlc(name):
            raise NotImplementedError("Nonlocal (VV10) correlation is not implemented.")
        if any(libxc.needs_laplacian(fid) for fid, _ in libxc.parse_xc(name)[1]):
            raise NotImplementedError("Laplacian-dependent meta-GGAs are not implemented.")
        self.omega, self.alpha, beta = libxc.rsh_coeff(name)
        self.hyb = self.alpha + beta if abs(self.omega) > 1e-10 else libxc.hybrid_coeff(name)

    @property
    def deriv(self):
        """Order of AO derivatives needed on the grid."""
        return 0 if self.kind == "LDA" else 1

    def _on_grid(self, molecule, grid, densities, extra_deriv=0, xc_deriv=1):
        """Per grid block: the indices of the basis functions that are not negligible on it,
        their AOs (with ``extra_deriv`` more derivative orders), the weights, the density
        rows (nspin, nvar, npoints), and libxc's e_xc per particle followed by the
        derivatives of e_xc with respect to the rows up to order ``xc_deriv``."""
        nvar = _NVAR[self.kind]
        deriv = self.deriv + extra_deriv
        if getattr(grid, "_screening", None) is None:
            grid._screening = _ShellScreening(molecule)
        for coords, weights in grid.blocks():
            shells, idx = grid._screening.select(coords)
            if len(idx) == 0:
                continue
            ao = integrals.basis_at_points(molecule, coords, deriv=deriv, shells=shells)
            if deriv == 0:
                ao = ao[None]
            rho = np.array([density_on_grid(ao, D[np.ix_(idx, idx)], nvar) for D in densities])
            xc = self._eval(self.name, rho[0] if len(densities) == 1 else rho, deriv=xc_deriv, xctype=self.kind)
            yield idx, ao, weights, rho, xc[: xc_deriv + 1]

    def potential(self, molecule, grid, densities):
        """E_xc and V_xc for one total density [P] (restricted) or [Da, Db] (unrestricted).

        Returns (exc, [V]) or (exc, [Va, Vb]).
        """
        nbf = densities[0].shape[0]
        V = [np.zeros((nbf, nbf)) for _ in densities]
        if self.kind == "HF":
            return 0.0, V
        exc = 0.0
        for idx, ao, weights, rho, (e, v) in self._on_grid(molecule, grid, densities):
            exc += weights @ (e * rho[:, 0].sum(axis=0))
            block = np.ix_(idx, idx)
            for Vs, vs in zip(V, v.reshape(len(densities), _NVAR[self.kind], -1)):
                Vs[block] += potential_matrix(ao, vs * weights, _NVAR[self.kind])
        return exc, V

    def nuclear_gradient(self, molecule, grid, densities):
        """dE_xc/dR (natoms, 3) through the basis-function centers, for the densities of
        ``potential``. The grid moving with the atoms (weight derivatives) is not included.

        dE/dR_Ax = -2 sum_{m on A, n} D_mn sum_g w_g [ v_rho d_x phi_m phi_n
                   + v_grad . (d_x grad phi_m phi_n + d_x phi_m grad phi_n)
                   + 1/2 v_tau d_x grad phi_m . grad phi_n ]
        """
        grad = np.zeros((molecule.natoms, 3))
        if self.kind == "HF":
            return grad
        nvar = _NVAR[self.kind]
        nbf = densities[0].shape[0]
        M = np.zeros((len(densities), 3, nbf, nbf))
        hess = ((4, 5, 6), (5, 7, 8), (6, 8, 9))  # d_x d_k components of deriv=2 AOs
        for idx, ao, weights, rho, (e, v) in self._on_grid(molecule, grid, densities, extra_deriv=1):
            for Mfull, vs in zip(M, v.reshape(len(densities), nvar, -1)):
                Ms = np.zeros((3, len(idx), len(idx)))
                wv = vs * weights
                # v_rho phi_n + v_grad . grad phi_n, contracted with d_x phi_m
                aow = wv[0][:, None] * ao[0]
                if nvar > 1:
                    aow += np.einsum("kp,kpn->pn", wv[1:4], ao[1:4])
                for x in range(3):
                    Ms[x] += ao[1 + x].T @ aow
                    if nvar > 1:
                        Ms[x] += sum(ao[hess[x][k]].T @ (wv[1 + k][:, None] * ao[0]) for k in range(3))
                    if nvar > 4:
                        Ms[x] += 0.5 * sum(ao[hess[x][k]].T @ (wv[4][:, None] * ao[1 + k]) for k in range(3))
                Mfull[:, idx[:, None], idx[None, :]] += Ms
        for A, (first, last) in enumerate(molecule.basis.center_first_and_last):
            for Ms, D in zip(M, densities):
                grad[A] -= 2.0 * np.einsum("xmn,mn->x", Ms[:, first:last], D[first:last])
        return grad


    def response(self, molecule, grid, densities, perturbations):
        """First-order change of V_xc, sum f_xc . delta rho, for density changes given like
        ``densities`` (one total density, or alpha and beta). Symmetric perturbations only."""
        nbf = densities[0].shape[0]
        dV = [np.zeros((nbf, nbf)) for _ in densities]
        if self.kind == "HF":
            return dV
        nvar, nspin = _NVAR[self.kind], len(densities)
        for idx, ao, weights, rho, (e, v, f) in self._on_grid(molecule, grid, densities, xc_deriv=2):
            block = np.ix_(idx, idx)
            drho = np.array([density_on_grid(ao, dD[block], nvar) for dD in perturbations])
            f = f.reshape(nspin, nvar, nspin, nvar, -1)
            dv = np.einsum("sxtyp,typ->sxp", f, drho)
            for dVs, dvs in zip(dV, dv):
                dVs[block] += potential_matrix(ao, dvs * weights, nvar)
        return dV


class _ShellScreening:
    """The shells of a Molecule whose functions are not negligible (> 1e-12) on a block of
    grid points, from the block's bounding sphere and each shell's extent."""

    def __init__(self, molecule, threshold=1.0e-12):
        basis = molecule.basis
        offsets = np.asarray(basis.shell_offsets)
        self.extent = integrals.shell_extents(molecule, threshold)
        self.functions = [np.arange(offsets[s], offsets[s + 1]) for s in range(basis.nshells)]
        self.center = np.zeros((basis.nshells, 3))
        for A, (first, last) in enumerate(basis.center_first_and_last_shell):
            self.center[first:last] = molecule.atomic_positions[A]

    def select(self, coords):
        mid = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
        radius = np.sqrt(np.max(np.sum((coords - mid) ** 2, axis=1)))
        shells = np.flatnonzero(np.linalg.norm(self.center - mid, axis=1) - radius < self.extent)
        idx = np.concatenate([self.functions[s] for s in shells]) if len(shells) else np.zeros(0, dtype=int)
        return shells, idx


def density_on_grid(ao, D, nvar):
    """Rows [rho, grad rho, tau][:nvar] of the (symmetric) density matrix D."""
    c = ao[0] @ D
    rows = [np.einsum("pm,pm->p", c, ao[0])]
    if nvar > 1:
        rows += [2.0 * np.einsum("pm,pm->p", c, ao[k]) for k in (1, 2, 3)]
    if nvar > 4:
        rows.append(0.5 * sum(np.einsum("pm,pm->p", ao[k] @ D, ao[k]) for k in (1, 2, 3)))
    return np.array(rows)


def potential_matrix(ao, wv, nvar):
    """V_mn from the weighted functional derivatives wv[:nvar] at the grid points."""
    aow = 0.5 * wv[0][:, None] * ao[0]
    if nvar > 1:
        aow += np.einsum("kp,kpm->pm", wv[1:4], ao[1:4])
    V = ao[0].T @ aow
    V = V + V.T
    if nvar > 4:
        V += 0.5 * sum(ao[k].T @ (wv[4][:, None] * ao[k]) for k in (1, 2, 3))
    return V
