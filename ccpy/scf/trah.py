"""Trust-region augmented Hessian (TRAH) second-order SCF, and orbital stability analysis.

Each orbital set (one for RHF/RKS, alpha and beta for UHF/UKS) is rotated as
C <- C exp(K), with K_ai = x_ai = -K_ia between virtual a and occupied i of the same irrep.
With the MO Fock matrix F and the response Fock G[dD] of the first-order density change
dD = C_v x C_o^T + C_o x^T C_v^T, the energy gradient and Hessian are

    g = f F_vo,    H x = f (F_vv x - x F_oo + C_v^T G[dD] C_o),

with f = 4 for closed shells (one orbital set holding both spins) and f = 2 per spin set.
Each macro-iteration minimizes the quadratic model within a trust radius by solving the
augmented-Hessian eigenproblem [[0, a g^T], [a g, H]] in a Davidson subspace, with the
scaling a >= 1 chosen so that the step stays inside the trust radius.

    T. Helmich-Paris, J. Chem. Phys. 154, 164104 (2021)
"""

import numpy as np
import scipy.linalg

_MAX_RADIUS = 1.0


class OrbitalHessian:
    """Energy, gradient, Hessian products, and Hessian diagonal at the current orbitals of
    an RHF/UHF/RKS/UKS object, over the symmetry-allowed rotations packed in one vector."""

    def __init__(self, mf, H, df, S):
        self.mf, self.df = mf, df
        sets = mf._orbital_sets()
        self.scale = 4.0 if len(sets) == 1 else 2.0
        self.C = [C for C, _ in sets]
        self.nocc = [n for _, n in sets]
        self.irreps = [np.asarray(h) for h in _orbital_irreps(mf)]
        self.masks = [h[n:, None] == h[None, :n] for h, n in zip(self.irreps, self.nocc)]
        mf.D = mf._build_density_matrix()
        self.F, _ = mf._build_fock(H, df, S)
        self.energy = mf.mol.nuclear_repulsion + mf._energy(H, self.F)
        self.Fmo = [C.T @ f @ C for C, f in zip(self.C, self.F)]
        self.gradient = self.scale * self.pack([f[n:, :n] for f, n in zip(self.Fmo, self.nocc)])
        self.diagonal = self.scale * self.pack(
            [np.diag(f)[n:, None] - np.diag(f)[None, :n] for f, n in zip(self.Fmo, self.nocc)])

    def pack(self, blocks):
        return np.concatenate([b[m] for b, m in zip(blocks, self.masks)])

    def unpack(self, v):
        blocks, start = [], 0
        for m in self.masks:
            x = np.zeros(m.shape)
            x[m] = v[start : start + m.sum()]
            start += m.sum()
            blocks.append(x)
        return blocks

    def __call__(self, v):
        """Hessian times v."""
        xs = self.unpack(v)
        pairs = [(C[:, n:] @ x, C[:, :n]) for C, n, x in zip(self.C, self.nocc, xs)]
        G = self.mf._response_fock(self.df, pairs)
        return self.scale * self.pack([
            f[n:, n:] @ x - x @ f[:n, :n] + C[:, n:].T @ g @ C[:, :n]
            for f, n, x, C, g in zip(self.Fmo, self.nocc, xs, self.C, G)
        ])

    def rotate(self, v):
        """The orbital coefficients rotated by exp(K(v))."""
        rotated = []
        for C, n, x in zip(self.C, self.nocc, self.unpack(v)):
            K = np.zeros((C.shape[1], C.shape[1]))
            K[n:, :n] = x
            K[:n, n:] = -x.T
            rotated.append(C @ scipy.linalg.expm(K))
        return rotated


def _orbital_irreps(mf):
    if mf._orbital_irreps is not None:
        return mf._orbital_irreps
    if mf._symmetry_basis is None:
        return [np.zeros(C.shape[1], dtype=int) for C in mf.C]
    return [mf._symmetry_basis.orbital_irreps(C) for C in mf.C]


def _subspace_step(Hs, gs, radius):
    """Lowest solution of the scaled augmented Hessian in the subspace with step length at
    most ``radius``: returns (mu, y) with (Hs - mu) y = -gs."""

    def solve(alpha):
        M = np.zeros((len(gs) + 1, len(gs) + 1))
        M[0, 1:] = M[1:, 0] = alpha * gs
        M[1:, 1:] = Hs
        w, U = np.linalg.eigh(M)
        c0 = U[0, 0] if abs(U[0, 0]) > 1e-14 else 1e-14
        return w[0] / alpha, U[1:, 0] / (alpha * c0)

    mu, y = solve(1.0)
    if np.linalg.norm(y) <= radius:
        return mu, y
    lo, hi = 1.0, 2.0
    while np.linalg.norm(solve(hi)[1]) > radius and hi < 1e8:
        lo, hi = hi, 2.0 * hi
    for _ in range(50):
        mid = np.sqrt(lo * hi)
        if np.linalg.norm(solve(mid)[1]) > radius:
            lo = mid
        else:
            hi = mid
    return solve(hi)


class _AugmentedHessianSolver:
    """Davidson solution of the trust-region step; the subspace is kept so the step can be
    recomputed for a smaller radius without new Hessian products."""

    def __init__(self, hess, radius, tol, max_micro=50):
        self.hess = hess
        g, d = hess.gradient, hess.diagonal
        self.V, self.HV = [], []
        t = -g / np.where(np.abs(d) > 1e-2, np.abs(d), 1e-2)
        self.nmicro = 0
        for self.nmicro in range(1, max_micro + 1):
            for _ in range(2):
                for v in self.V:
                    t -= (v @ t) * v
            norm = np.linalg.norm(t)
            if norm < 1e-12:
                break
            self.V.append(t / norm)
            self.HV.append(hess(self.V[-1]))
            x, Hx, mu = self._step(radius)
            residual = Hx + g - mu * x
            if np.linalg.norm(residual) < tol:
                break
            denom = d - mu
            t = -residual / np.where(np.abs(denom) > 1e-8, denom, 1e-8)

    def _step(self, radius):
        V, HV = np.array(self.V), np.array(self.HV)
        Hs = V @ HV.T
        mu, y = _subspace_step(0.5 * (Hs + Hs.T), V @ self.hess.gradient, radius)
        return V.T @ y, HV.T @ y, mu

    def step(self, radius):
        """(step, predicted energy change) for the trust radius."""
        x, Hx, _ = self._step(radius)
        return x, self.hess.gradient @ x + 0.5 * x @ Hx


def trah(mf, H, df, S, maxiter=50):
    """Minimize the energy of ``mf`` from its current orbitals; returns the final
    OrbitalHessian (energy, Fock matrices) and the number of macro-iterations."""
    gtol = 0.1 * np.sqrt(mf.e_tol)
    radius = 0.4
    hess = OrbitalHessian(mf, H, df, S)
    if mf.verbose:
        print("\n   Trust-region augmented Hessian (TRAH) iterations")
        print("    Iter            Energy             ΔE      ‖grad‖     radius  micro")
        print("   " + "-" * 70)
    mf.converged = False
    for it in range(1, maxiter + 1):
        gnorm = np.linalg.norm(hess.gradient)
        if gnorm < gtol:
            mf.converged = True
            break
        # inexact Newton: residual tolerance eta |g| with eta -> 0 for superlinear convergence
        solver = _AugmentedHessianSolver(hess, radius, tol=max(min(0.5, np.sqrt(gnorm)) * gnorm, 0.1 * gtol))
        # energy changes below this are rounding and grid noise, not a test of the model
        noise = 1.0e-11 * max(1.0, abs(hess.energy))
        while True:
            x, predicted = solver.step(radius)
            mf.C = hess.rotate(x)
            trial = OrbitalHessian(mf, H, df, S)
            dE = trial.energy - hess.energy
            ratio = dE / predicted if predicted < 0 else -1.0
            if abs(predicted) < noise and dE < noise:
                ratio = 1.0
            if ratio > 0.0:
                break
            radius *= 0.5  # the energy went up: reject the step
            mf.C = hess.C
            if radius < 1e-6:
                raise RuntimeError("TRAH could not find an energy-lowering step.")
        if ratio > 0.75 and np.linalg.norm(x) > 0.99 * radius:
            radius = min(1.2 * radius, _MAX_RADIUS)
        elif ratio < 0.25:
            radius *= 0.7
        if mf.verbose:
            print(f"   {it:4d} {trial.energy:22.12f} {dE:12.4e} {gnorm:10.3e} {radius:10.3e} {solver.nmicro:6d}")
        hess = trial
    if mf.verbose:
        print("   " + "-" * 70)
    _canonicalize(mf, hess)
    return hess, it


def _canonicalize(mf, hess):
    """Diagonalize the Fock matrix within the occupied and the virtual orbitals of each set
    and irrep, keeping the occupations of the TRAH solution."""
    C, eps, irreps = [], [], []
    for c, n, f, h in zip(hess.C, hess.nocc, hess.Fmo, hess.irreps):
        nmo = c.shape[1]
        new_c, e = c.copy(), np.zeros(nmo)
        for block in (np.arange(n), np.arange(n, nmo)):
            for irrep in np.unique(h[block]):
                sel = block[h[block] == irrep]
                e[sel], U = np.linalg.eigh(f[np.ix_(sel, sel)])
                new_c[:, sel] = c[:, sel] @ U
        order = np.concatenate([b[np.argsort(e[b], kind="stable")] for b in (np.arange(n), np.arange(n, nmo))])
        C.append(new_c[:, order])
        eps.append(e[order])
        irreps.append(h[order])
    mf.C, mf.eps, mf._orbital_irreps = C, eps, irreps
    mf.D = mf._build_density_matrix()
    mf.F = hess.F
    mf.E = hess.energy


def _lowest_eigenpairs(hess, nroots, tol=1.0e-6, maxiter=100):
    """Lowest eigenpairs of the orbital Hessian by Davidson."""
    d = hess.diagonal
    V = [np.eye(len(d))[i] for i in np.argsort(d)[: min(len(d), max(2 * nroots, nroots + 4))]]
    HV = []
    for _ in range(maxiter):
        HV += [hess(v) for v in V[len(HV):]]
        Vm, HVm = np.array(V), np.array(HV)
        Hs = Vm @ HVm.T
        w, U = np.linalg.eigh(0.5 * (Hs + Hs.T))
        w, U = w[:nroots], U[:, :nroots]
        X = Vm.T @ U
        R = HVm.T @ U - X * w
        new = []
        for k in range(nroots):
            if np.linalg.norm(R[:, k]) < tol:
                continue
            denom = d - w[k]
            t = -R[:, k] / np.where(np.abs(denom) > 1e-8, denom, 1e-8)
            for v in V + new:
                t -= (v @ t) * v
            if np.linalg.norm(t) > 1e-8:
                new.append(t / np.linalg.norm(t))
        if not new:
            break
        V += new
    return w, X


def stability(mf, external=False, follow=False, nroots=1, step=0.3):
    """Lowest eigenvalues of the orbital Hessian of a converged RHF/UHF/RKS/UKS solution.

    ``external=True`` (RHF/RKS only) tests the RHF -> UHF (RKS -> UKS) instability, as the
    internal Hessian of the unrestricted counterpart at the restricted solution. With
    ``follow=True``, a negative eigenvalue is followed downhill by rotating the orbitals
    along its eigenvector and reconverging with TRAH.

    Returns (eigenvalues, scf): ``scf`` is ``mf`` itself, or the new unrestricted solution
    when an external instability was followed. Only symmetry-preserving rotations are tested.
    """
    from ccpy.scf.ks import RKS, UKS
    from ccpy.scf.rhf import RHF
    from ccpy.scf.uhf import UHF

    if not hasattr(mf, "_orbital_sets"):
        raise NotImplementedError(f"Stability analysis is not implemented for {mf.method}.")
    if mf.e_tot is None:
        mf.run()
    target = mf
    if external:
        if not isinstance(mf, RHF):
            raise ValueError("external=True tests restricted solutions (RHF or RKS).")
        target = (UKS if isinstance(mf, RKS) else UHF)(mf.mol, **mf._init_options())
        target.C = [mf.C[0].copy(), mf.C[0].copy()]
        target.eps = [mf.eps[0], mf.eps[0]]
        target._orbital_irreps = [mf._orbital_irreps[0], mf._orbital_irreps[0]]
        target._symmetry_basis = mf._symmetry_basis
        target._grid = getattr(mf, "_grid", None)
        target._omega_fitting = getattr(mf, "_omega_fitting", None)
    mol = mf.mol
    hess = OrbitalHessian(target, mol.ints_hcore(), mol.df, mol.ints_overlap())
    w, X = _lowest_eigenpairs(hess, nroots)
    if mf.verbose:
        kind = "external (restricted -> unrestricted)" if external else "internal"
        print(f"   Lowest {kind} orbital Hessian eigenvalues: " + " ".join(f"{e:.6f}" for e in w))
    if follow and w[0] < -1.0e-5:
        target.C = hess.rotate(step * X[:, 0])
        target._orbital_irreps = hess.irreps
        target.trah_after = 0
        target.run()
    return w, target
