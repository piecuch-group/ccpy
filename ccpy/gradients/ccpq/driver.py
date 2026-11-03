"""CC(P) and CC(P;Q) analytic nuclear gradients in the separate-functional framework of
ccpq_gradients_revised.pdf, organized by the stages of its Sec. 9.1:

    1   fix the reference, frozen orbitals, P, Q and the denominator     (this class)
    2-8 amplitude stage: right CC(P), lambda_grad, lambda, moments,
        sources, Sigma and Pi, base and correction bare densities        (an amplitude backend)
    9   total spatial densities and the orbital source                   (densities.spin_trace)
    10-11 orbital Z-vector, relaxed densities, overlap coefficient W      (orbital_response)
    12  conventional or factorized derivative integrals                 (nuclear_gradient)

Stages 9-12 only see the spatial densities (D, G) in the MO basis, so any amplitude backend
that returns them can be used. The backend here, DeterminantBackend, is the exact
determinant-space algorithm of Sec. 6 ("correctness oracle"): it treats CCSD, CCSDT, CC(P)
with any P and CC(P;Q) with any Q, both denominators, but its cost grows with the full
determinant space. A production backend would return the same AmplitudeStage from connected
tensor contractions (Sec. 6.4).

Example
-------
    from ccpy import Molecule
    from ccpy.gradients.ccpq import CCPQGradient

    mol = Molecule("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", "sto-3g",
                   cholesky_tei=True, verbose=False)
    # CR-CC(2,3)_A: P = SD, Q = all triples, bare-Fock denominator
    calc = CCPQGradient(mol, pspace=(1, 2), qspace=(3,), denominator="mp", nfrozen=1)
    grad = calc.gradient()
"""

import time
import operator
from dataclasses import dataclass, field

import numpy as np

from . import spaces
from .amplitudes import SelectedCC
from .densities import (SpatialDensities, coefficient_densities, mp_denominator_densities,
                                           spin_trace)
from .determinants import (DeterminantSpace, fock_diagonal, hamiltonian_matrix,
                                              spin_orbital_integrals)
from .state_oracle import StateOracle


@dataclass
class AmplitudeStage:
    """What an amplitude backend returns (Sec. 9.1, steps 2-8), all electronic energies."""
    energy_base: float                 # E^(P)
    delta: float                       # delta(P;Q)
    rho_base: SpatialDensities         # gamma^P, Gamma^P (reference and core included)
    rho_corr: SpatialDensities         # Delta gamma, Delta Gamma (denominator terms included)
    checks: dict = field(default_factory=dict)
    data: object = None


class DeterminantBackend:
    """Exact determinant-space amplitude stage (Secs. 2-6)."""

    def __init__(self, norb, nalpha, nbeta, nfrozen, pspace, qspace, denominator, state=0):
        if denominator not in ("mp", "full"):
            raise ValueError("denominator must be 'mp' or 'full'")
        self.space = DeterminantSpace(norb, nalpha, nbeta, nfrozen)
        self.P = self._select(pspace)
        self.Q = self._select(qspace, P=self.P) if qspace is not None else np.zeros(0, dtype=np.int64)
        self.denominator = denominator
        self.state = operator.index(state)
        if np.intersect1d(self.P, self.Q).size:
            raise ValueError("P and Q must be disjoint")
        if not 0 <= self.state <= len(self.P):
            raise ValueError("state must be 0 (ground) or 1..len(P) (excited)")
        self.one = self.space.one_body_transitions()
        self.two = self.space.two_body_transitions()
        self.t = None

    def _select(self, spec, P=None):
        """A tuple of excitation levels (for Q: those levels minus P), a callable
        space -> indices, or an array of determinant indices."""
        if callable(spec):
            spec = spec(self.space)
        elif isinstance(spec, tuple):
            spec = spaces.ranks(self.space, spec) if P is None else spaces.complement(self.space, P, spec)
        return spaces.validate_indices(self.space, spec, "P" if P is None else "Q")

    def run(self, h, eri, verbose=False):
        S = self.space
        h, eri = np.asarray(h), np.asarray(eri)
        if h.shape != (S.norb, S.norb) or eri.shape != (S.norb,) * 4:
            raise ValueError("h and eri must cover the backend's full spatial orbital space")
        h_so, v_so = spin_orbital_integrals(h, eri)
        H = hamiltonian_matrix(S, h_so, v_so, self.one, self.two)
        if self.state > 0:
            oracle = StateOracle(S, self.P, self.Q, state=self.state, denominator=self.denominator)
            result = oracle.solve(H, fock_diag=fock_diagonal(h_so, v_so, S.ref_occ))
            from .densities import SpinOrbitalDensities
            base = spin_trace(SpinOrbitalDensities(result.gamma_base, result.Gamma_base), S.norb)
            corr = spin_trace(SpinOrbitalDensities(result.gamma_corr, result.Gamma_corr), S.norb)
            checks = dict(result.checks)
            checks["base_density_energy"] = abs(base.energy(h, eri) - result.E_base)
            checks["corr_density_energy"] = abs(corr.energy(h, eri) - result.delta)
            return AmplitudeStage(result.E_base, result.delta, base, corr, checks, result)
        cc = SelectedCC(S, H, self.P, self.Q, fock_diag=fock_diagonal(h_so, v_so, S.ref_occ),
                        denominator=self.denominator, t_guess=self.t)
        res = cc.run(verbose=verbose)
        self.t = res.t
        base = coefficient_densities(res.A_base, res.U, res.V, S, self.one, self.two)
        corr = coefficient_densities(res.A_corr, res.U, res.V, S, self.one, self.two)
        if res.d_mp is not None:
            corr = corr + mp_denominator_densities(res.d_mp, S.ref_occ)
        checks = dict(res.checks)
        checks["base_density_energy"] = abs(base.energy(h_so, v_so) - res.energy)
        checks["corr_density_energy"] = abs(corr.energy(h_so, v_so) - res.delta)
        return AmplitudeStage(res.energy, res.delta, spin_trace(base, S.norb), spin_trace(corr, S.norb), checks, res)


class CCPQGradient:
    """Analytic nuclear gradient of E^(P+Q) = E^(P) + delta(P;Q) for a closed-shell RHF reference
    with conventional or DF/Cholesky integrals.

    Parameters
    ----------
    mol : ccpy Molecule
    integral_model : "conventional" (default) or "factorized"
        Conventional RHF and four-center derivatives, or native RHF with the
        molecule's DF/Cholesky integrals and their matching derivatives.
    pspace : excitation levels, e.g. (1, 2) for CCSD or (1, 2, 3) for CCSDT; a callable
        DeterminantSpace -> indices (e.g. lambda S: spaces.union(spaces.ranks(S, (1, 2)), t3));
        or determinant indices
    qspace : None for CC(P); excitation levels of the omitted determinants, e.g. (3,), for
        Q = (those levels) minus P; or a callable / indices
    denominator : "mp" (bare Fock, Eq. 3.5; variant A) or "full" (E^(P) - <K|Hbar|K>, Eq. 3.4)
    nfrozen : frozen-core spatial orbitals
    state : 0 for the ground state, 1 for the lowest excited EOM root, etc.
        The excited-state correction uses full projected-Hbar eigenvectors,
        including the reference component of the right eigenvector.
    backend : "dense" (independent complex-step oracle) or "matrix_free"
        (FCIpy Hamiltonian actions, Davidson EOM roots and GMRES response).
    fcipy_extension : path to the optional connector built with
        devtools/build_fcipy_oracle.py; otherwise use CCPY_FCIPY_EXTENSION.
    sparse_cache : cache the connected bare Hamiltonian as CSR for the
        matrix-free backend; speeds dense-support P/Q and full denominators.
    """

    def __init__(self, mol, pspace=(1, 2), qspace=None, denominator="mp", nfrozen=0, verbose=False, state=0,
                 backend="dense", fcipy_extension=None, sparse_cache=False, integral_model="conventional"):
        if mol.multiplicity != 1 or mol.nelectrons % 2:
            raise ValueError("CCPQGradient requires a closed-shell singlet RHF reference")
        self.mol, self.nfrozen, self.verbose, self.state = mol, nfrozen, verbose, state
        if integral_model not in ("conventional", "factorized"):
            raise ValueError("integral_model must be 'conventional' or 'factorized'")
        self.integral_model = integral_model
        nocc = mol.nelectrons // 2
        if backend == "dense":
            self.backend = DeterminantBackend(mol.nmo, nocc, nocc, nfrozen, pspace, qspace, denominator, state=state)
        elif backend == "matrix_free":
            from .matrix_free import MatrixFreeBackend
            self.backend = MatrixFreeBackend(mol.nmo, nocc, nocc, nfrozen, pspace, qspace, denominator,
                                             state=state, fcipy_extension=fcipy_extension,
                                             sparse_cache=sparse_cache)
        else:
            raise ValueError("backend must be 'dense' or 'matrix_free'")
        self.state = self.backend.state
        self.mf = None
        self.stages = {}

    def _reference(self, mol):
        if self.integral_model == "factorized":
            from ccpy import RHF
            mf = RHF(mol, verbose=0, e_tol=1e-13, d_tol=1e-11)
            mf.kernel()
            return mf
        from .scf import run_rhf
        # Preserve the fresh-reference policy of the conventional molecular
        # energy and gradient checks.
        return run_rhf(mol)

    def _integrals(self, mf):
        if self.integral_model == "conventional":
            return mf.mo_integrals()
        C = mf.C[0]
        B = mf.mol.df.mo_cholesky_vectors(C)
        return C.T @ mf.get_hcore() @ C, np.einsum("xpq,xrs->pqrs", B, B, optimize=True)

    def energy(self, mol=None):
        """Total E^(P+Q) (nuclear repulsion included) at `mol` (default: the stored geometry)."""
        return self.energy_components(mol)["total"]

    def energy_components(self, mol=None):
        """Return base (including nuclear repulsion), correction, and total energies."""
        mol = self.mol if mol is None else mol
        if (mol.nmo != self.backend.space.norb or mol.nelectrons != self.mol.nelectrons
                or mol.multiplicity != 1):
            raise ValueError("Displaced molecule must retain orbital and electron counts and singlet reference")
        mf = self._reference(mol)
        h, eri = self._integrals(mf)
        if hasattr(self.backend, "energies"):
            electronic, delta = self.backend.energies(h, eri)
        else:
            amp = self.backend.run(h, eri)
            electronic, delta = amp.energy_base, amp.delta
        base = mf.mol.nuclear_repulsion + electronic
        return dict(base=base, correction=delta, total=base + delta)

    def gradient(self):
        """Return the total state gradient in Eh/bohr, shape (natoms, 3)."""
        return self.gradient_components()["total"]

    def gradient_components(self):
        """Separate orbital-relaxed base/correction gradients and their sum.

        The base includes nuclear repulsion; the correction is purely electronic.
        P/Q indices remain fixed across displaced geometries. State selection uses
        energy order, so a finite-difference stencil must avoid root crossings.
        """
        from .orbital_response import orbital_relaxation
        from .nuclear_gradient import contract_derivative_integrals
        t0 = time.perf_counter()
        mf = self.mf = self._reference(self.mol)
        h, eri = self._integrals(mf)
        amp = self.backend.run(h, eri, verbose=self.verbose)
        if self.integral_model == "factorized":
            grad, rel = self._factorized_gradient(amp.rho_base + amp.rho_corr)
            corr, rel_corr = self._factorized_gradient(amp.rho_corr, include_nuclear=False)
        else:
            rel = orbital_relaxation(amp.rho_base + amp.rho_corr, h, eri, mf.eps, mf.nocc, self.nfrozen)
            rel_corr = orbital_relaxation(amp.rho_corr, h, eri, mf.eps, mf.nocc, self.nfrozen)
            grad = contract_derivative_integrals(self.mol, mf.C, rel)
            corr = contract_derivative_integrals(self.mol, mf.C, rel_corr, include_nuclear=False)
        self.stages = dict(reference=mf, amplitudes=amp, relaxed=rel, relaxed_correction=rel_corr)
        self.gradients = dict(base=grad - corr, correction=corr, total=grad)
        self.e_tot = self.mol.nuclear_repulsion + amp.energy_base + amp.delta
        if self.verbose:
            self.report(grad, time.perf_counter() - t0)
        return self.gradients

    def _factorized_gradient(self, rho, include_nuclear=True):
        """Contract the same determinant densities with fitted-integral derivatives.

        In restricted RHF the shared-orbital derivative is split equally over
        spins for the native response machinery. Symmetrization retains exactly
        the components seen by the real symmetric h and B integrals.
        """
        from ccpy.gradients.orbital_response import orbital_relaxation
        from ccpy.gradients.derivative_integrals import contract_derivative_integrals
        from ccpy.gradients.density_gradient import zero_density
        from ccpy.integrals import integrals
        from ccpy.methods.reference import Reference
        reference = Reference(self.mf, self.nfrozen)
        rho = rho.symmetrized()
        # determinant-space (full-MO) densities, half per spin, into ccpy's oo/ov/vo/vv blocks
        Dfull = rho.D / 2
        Yfull = np.einsum("pqrs,xqs->xpr", rho.G, reference.B_mo["a"], optimize=True) / 2
        nocc = self.nfrozen + reference.system.noccupied_alpha
        D, Y = zero_density(reference), zero_density(reference, Yfull.shape[0])
        for X, full in ((D.a, Dfull), (D.b, Dfull), (Y.a, Yfull), (Y.b, Yfull)):
            X.oo[...], X.ov[...] = full[..., :nocc, :nocc], full[..., :nocc, nocc:]
            X.vo[...], X.vv[...] = full[..., nocc:, :nocc], full[..., nocc:, nocc:]
        relaxed = orbital_relaxation(D, Y, reference, verbose=self.verbose)
        grad = contract_derivative_integrals(*relaxed, reference)
        if not include_nuclear:
            grad -= integrals.nuclear_repulsion_deriv(self.mol)
        return grad, relaxed

    def report(self, grad, elapsed):
        amp = self.stages["amplitudes"]
        b = self.backend
        print(f"\n   {'CC' if self.state == 0 else 'EOMCC'}(P;Q) analytic gradient (state={self.state})")
        print(f"   |P| = {len(b.P)}, |Q| = {len(b.Q)}, denominator = {b.denominator}, ndet = {b.space.ndet}")
        print(f"   E_HF       = {self.stages['reference'].e_tot:.12f}")
        print(f"   E_state^(P) = {self.mol.nuclear_repulsion + amp.energy_base:.12f}")
        print(f"   delta(P;Q) = {amp.delta:.12f}")
        print(f"   E_state^(P;Q) = {self.e_tot:.12f}")
        for k, v in amp.checks.items():
            print(f"   {k:28s} {v:.2e}")
        print(f"   Gradient time: {elapsed:.2f} s")
        for gA in grad:
            print(f"   {gA[0]:17.10f}{gA[1]:17.10f}{gA[2]:17.10f}")

    def check_gradient(self, step=1.0e-3, npoints=4, components=None):
        """Compare all three gradients with fully re-solved displaced RHF energies.

        ``components`` is a list of (atom, Cartesian-axis) pairs; None checks
        all coordinates. Return analytic/numerical arrays and maximum absolute
        errors, separately for base, correction and total. Energies are in Eh,
        gradients and errors in Eh/bohr. Root ordering must remain unchanged.
        """
        from ccpy.gradients.finite_difference import finite_difference

        analytic = self.gradient_components()
        xyz = np.array([atom[1] for atom in self.mol.atoms])
        indices = list(np.ndindex(xyz.shape)) if components is None else list(components)
        if not indices:
            raise ValueError("At least one Cartesian component must be checked")
        keys = ("base", "correction", "total")

        def energies(coords):
            values = self.energy_components(self.mol.with_geometry(coords, unit="bohr"))
            return np.array([values[key] for key in keys])

        numerical = finite_difference(energies, xyz, step=step, npoints=npoints, components=indices)
        result = {}
        for column, key in enumerate(keys):
            values = np.array([analytic[key][index] for index in indices])
            fd = numerical[:, column]
            result[key] = dict(analytic=values, numerical=fd, max_error=float(np.max(abs(values - fd))))
        return result
