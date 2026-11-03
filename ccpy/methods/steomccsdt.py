"""Enlarged-model STEOM on a CCSD ground state, with an SD neutral eigenproblem."""

from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from ccpy.eomcc.steomccsd import (
    diagonalize_singles,
    natural_orbital_space,
    rotate_hbar,
    singles_matrix,
)
from ccpy.eomcc.steomccsdt import (
    ExtendedAction,
    active_triples,
    auxiliary_model,
    enlarged_transformation,
    normalize_auxiliary,
)
from ccpy.methods.steomcc import STEOMCC
from ccpy.models.operators import FockOperator


def _stable_pspace_hr(module, excitations):
    """Keep Davidson coordinates fixed despite in-place Fortran excitation sorts."""

    def HR(dR, R, T, H, rhf, system):
        local = {s: e.copy(order="F") for s, e in excitations.items()}
        module.HR(dR, R, T, H, rhf, system, None, local)
        for spin, original in excitations.items():
            lookup = {tuple(row): k for k, row in enumerate(local[spin])}
            permutation = [lookup[tuple(row)] for row in original]
            setattr(R, spin, np.asfortranarray(getattr(R, spin)[permutation]))
            setattr(dR, spin, np.asfortranarray(getattr(dR, spin)[permutation]))
        return dR.flatten()

    return HR


def _auxiliary_davidson(
    HR,
    update,
    guesses,
    energies,
    new_operator,
    T,
    H,
    system,
    options,
    selection_vectors=None,
    selection_transform=None,
    label="Auxiliary",
    allow_complex_ritz=False,
):
    """Block Davidson with one-to-one root assignment and bounded restarts.

    All model roots are required. Independent overlap selection can otherwise
    converge multiple starting vectors to the same eigenvector.
    """
    nroot = len(guesses)
    initial = np.column_stack([r.flatten() for r in guesses])
    initial /= np.linalg.norm(initial, axis=0)
    if np.linalg.matrix_rank(initial, tol=1.0e-12) != nroot:
        raise ValueError("Target starting vectors are linearly dependent")
    reference = initial if selection_vectors is None else selection_vectors
    reference = reference / np.linalg.norm(reference, axis=0)
    maximum = min(
        initial.shape[0], max(options["davidson_max_subspace_size"], 2 * nroot + 1)
    )
    V, _ = np.linalg.qr(initial)
    R, dR = new_operator(), new_operator()

    def sigma(vector):
        R.unflatten(vector.copy())
        return HR(dR, R, T, H, False, system).copy()

    W = np.column_stack([sigma(v) for v in V.T])
    previous = np.array(energies)
    flags = np.zeros(nroot, dtype=bool)
    eigenvectors = initial
    for iteration in range(options["maximum_iterations"]):
        values, coefficients = diagonalize_singles(
            V.T @ W, min(1.0e-8, options["amp_convergence"])
        )
        candidates = V @ coefficients
        comparison = (
            candidates
            if selection_transform is None
            else selection_transform(candidates)
        )
        comparison = comparison / np.linalg.norm(comparison, axis=0)
        _, selected = linear_sum_assignment(-(abs(reference.T @ comparison) ** 2))
        if not allow_complex_ritz and (
            np.max(abs(values[selected].imag)) > 1.0e-8
            or np.max(abs(candidates[:, selected].imag)) > 1.0e-8
        ):
            raise RuntimeError(f"Selected {label.lower()} Ritz roots are complex")
        omega, alpha = values[selected].real, coefficients[:, selected].real
        eigenvectors = V @ alpha
        residuals = W @ alpha - eigenvectors * omega
        norms = np.linalg.norm(residuals, axis=0)
        flags = (norms < options["amp_convergence"]) & (
            abs(omega - previous) < options["energy_convergence"]
        )
        print(
            f"   {label} block iteration {iteration:3d}: max residual {max(norms):.3e}"
        )
        if all(flags):
            break
        previous = omega.copy()
        corrections = []
        for k in np.flatnonzero(~flags):
            dR.unflatten(residuals[:, k].copy())
            q = update(dR, omega[k], H, False, system).flatten()
            # Two orthogonalization passes protect nearly degenerate roots.
            for _ in range(2):
                q -= V @ (V.T @ q)
                for correction in corrections:
                    q -= correction * np.dot(correction, q)
            norm = np.linalg.norm(q)
            if norm > 1.0e-12:
                corrections.append(q / norm)
        if not corrections:
            if np.max(norms) < options["amp_convergence"]:
                continue
            break
        corrections = np.column_stack(corrections)
        if V.shape[1] + corrections.shape[1] > maximum:
            V, _ = np.linalg.qr(np.column_stack((eigenvectors, corrections)))
            V = V[:, :maximum]
            W = np.column_stack([sigma(v) for v in V.T])
        else:
            W = np.column_stack((W, *[sigma(v) for v in corrections.T]))
            V = np.column_stack((V, corrections))
    result = []
    for vector in eigenvectors.T:
        root = new_operator()
        root.unflatten(vector.copy())
        result.append(root)
    return result, omega, flags.tolist()


class STEOMCCSDT(STEOMCC):
    """STEOMCC(..., level='sdt'): enlarged auxiliary models and an SD G problem.

    Both model spaces contain every 1h/1p determinant plus active 2h1p/2p1h
    doublets. ``ipcisd``/``eacisd`` diagonalizations supply one guess per model
    configuration. ``auxiliary_order=3`` (default) uses IP/EA-EOM3 on CCSD,
    with active P-space triples by default; ``auxiliary_order=2`` is also
    available. This enlarged-model extension is broader than the S2-only
    extended STEOM of Nooijen & Lotrich (2000).

    ``ip_threshold``/``ea_threshold`` now bound the total model-space weight,
    including active satellites, and default to zero. No root is discarded:
    an incomplete or singular model raises an error. ``minimum_active`` is
    the number of active holes (IP) / particles (EA) in auxiliary triples.
    ``ip_pspace``/``ea_pspace`` can override generated lists in the working
    basis. Neutral implicit quadruples are omitted. G is a LinearOperator;
    small problems use a dense diagonalization, larger ones use Arnoldi.
    """

    def __init__(
        self,
        upstream,
        nroots=1,
        multiplicity=1,
        level="sdt",
        auxiliary_order=3,
        auxiliary_solver="pspace",
        minimum_active=1,
        ip_pspace=None,
        ea_pspace=None,
        dense_limit=80,
        ip_threshold=None,
        ea_threshold=None,
        **kwargs,
    ):
        if level != "sdt":
            raise ValueError("STEOMCCSDT requires level='sdt'")
        if not isinstance(nroots, (int, np.integer)) or nroots < 1:
            raise ValueError("nroots must be a positive integer")
        if auxiliary_order not in (2, 3):
            raise ValueError("auxiliary_order must be 2 or 3")
        if auxiliary_solver not in ("pspace", "dense"):
            raise ValueError("auxiliary_solver must be 'pspace' or 'dense'")
        if minimum_active not in (1, 2, 3):
            raise ValueError("minimum_active must be 1, 2, or 3")
        if not isinstance(dense_limit, (int, np.integer)) or dense_limit < 0:
            raise ValueError("dense_limit must be a nonnegative integer")
        if auxiliary_order == 2 and (ip_pspace is not None or ea_pspace is not None):
            raise ValueError("Auxiliary triples lists require auxiliary_order=3")
        if auxiliary_solver == "dense" and (
            ip_pspace is not None or ea_pspace is not None
        ):
            raise ValueError('Custom triples lists require auxiliary_solver="pspace"')
        super().__init__(
            upstream,
            nroots=1,
            multiplicity=multiplicity,
            ip_threshold=0.0 if ip_threshold is None else ip_threshold,
            ea_threshold=0.0 if ea_threshold is None else ea_threshold,
            **kwargs,
        )
        self.nroots, self.level = int(nroots), level
        self.auxiliary_order, self.auxiliary_solver = auxiliary_order, auxiliary_solver
        self.minimum_active, self.dense_limit = minimum_active, dense_limit
        self._ip_pspace, self._ea_pspace = deepcopy(ip_pspace), deepcopy(ea_pspace)
        self.action = None

    @staticmethod
    def _validate_pspace(excitations, no, nv, sector):
        if set(excitations) != {"aaa", "aab", "abb"}:
            raise ValueError("P-space lists require aaa, aab, and abb blocks")
        result = {}
        for spin, data in excitations.items():
            data = np.asarray(data)
            if (
                data.ndim != 2
                or data.shape[1] != 5
                or not np.all(np.isfinite(data))
                or np.any(data != data.astype(int))
            ):
                raise ValueError(
                    "Each auxiliary triples list must be an integer (n,5) array"
                )
            tuples = [tuple(row) for row in data]
            if not len(data) or (len(data) == 1 and np.all(data == 1)):
                result[spin] = np.ones((1, 5), dtype=np.int32, order="F")
            else:
                na = spin.count("a")
                pa, ha = na - (sector == "ip"), na - (sector == "ea")
                npart = 2 if sector == "ip" else 3
                groups = (
                    data[:, :pa],
                    data[:, pa:npart],
                    data[:, npart : npart + ha],
                    data[:, npart + ha :],
                )
                invalid = any(
                    np.any(g < 1)
                    or np.any(g > bound)
                    or np.any(np.diff(g, axis=1) <= 0)
                    for g, bound in zip(groups, (nv, nv, no, no))
                )
                if len(set(tuples)) != len(tuples) or invalid:
                    raise ValueError(
                        f"Invalid, duplicate, or noncanonical {sector.upper()} {spin} triples"
                    )
                result[spin] = np.asfortranarray(data, dtype=np.int32)
        return result

    def _extended_auxiliary(self, sector, occupied, virtual, H, T):
        no, nv = self.system.noccupied_alpha, self.system.nunoccupied_alpha
        model_states, model_basis = auxiliary_model(no, nv, occupied, virtual, sector)
        count = model_basis.shape[1]
        guess_module = import_module(
            "ccpy.eom_guess." + ("ipcisd" if sector == "ip" else "eacisd")
        )
        energies, guess = guess_module.run_diagonalization(
            self.system,
            H,
            2,
            {"A": count},
            len(occupied),
            len(virtual),
            use_symmetry=False,
        )
        if len(energies) != count or np.any(np.linalg.norm(guess, axis=0) < 1.0e-12):
            raise RuntimeError(
                "Auxiliary CISD guess does not span the complete doublet model"
            )
        pspace = None
        if self.auxiliary_order == 3 and self.auxiliary_solver == "pspace":
            custom = self._ip_pspace if sector == "ip" else self._ea_pspace
            pspace = (
                active_triples(no, nv, occupied, virtual, sector, self.minimum_active)
                if custom is None
                else self._validate_pspace(custom, no, nv, sector)
            )
        stem = ("ipeom" if sector == "ip" else "eaeom") + str(self.auxiliary_order)
        module = import_module(
            "ccpy.eomcc." + stem + ("_p" if pspace is not None else "")
        )
        particles, holes = (
            (self.auxiliary_order - 1, self.auxiliary_order)
            if sector == "ip"
            else (self.auxiliary_order, self.auxiliary_order - 1)
        )

        def new_operator():
            extra = (
                {}
                if pspace is None
                else dict(
                    p_orders=[2],
                    pspace_sizes=[[len(pspace[s]) for s in ("aaa", "aab", "abb")]],
                )
            )
            return FockOperator(self.system, particles, holes, **extra)

        guesses = []
        for column in guess.T:
            R = new_operator()
            R.unflatten(column, order=2)
            guesses.append(R)
        HR = module.HR if pspace is None else _stable_pspace_hr(module, pspace)
        update = (
            module.update
            if pspace is None
            else lambda R, w, H, rhf, sys: module.update(R, w, H, rhf, sys, pspace)
        )
        print(
            f"   STEOM {sector.upper()}: {count} enlarged-model doublet roots ({stem}{'_p' if pspace is not None else ''})"
        )
        vectors, omega, flags = _auxiliary_davidson(
            HR, update, guesses, energies, new_operator, T, H, self.system, self.options
        )
        result = SimpleNamespace(
            R=vectors,
            omega=omega,
            root_converged=flags,
            excitations=None if pspace is None else [deepcopy(pspace) for _ in vectors],
            model_states=model_states,
            model_basis=model_basis,
        )
        setattr(self, sector, result)
        if not all(flags):
            raise RuntimeError(
                f"STEOM {sector.upper()} enlarged-model auxiliaries did not converge"
            )
        normalized = normalize_auxiliary(
            result, model_states, model_basis, no, nv, sector, self.condition_limit
        )
        threshold = self.ip_threshold if sector == "ip" else self.ea_threshold
        if np.any(result.model_weights < threshold):
            raise ValueError(
                f"{sector.upper()} root below enlarged-model weight threshold; cannot discard a model root"
            )
        return normalized

    def _targeted_neutral(self, cfg, H, Uo, Uv):
        """CI guesses in the working determinant metric, followed in the EOM picture."""
        targets = []
        for guess in self.guess_vectors:
            R = deepcopy(guess)
            for spin in ("a", "b", "aa", "ab", "bb"):
                value = getattr(R, spin)
                for axis, rotation in enumerate([Uv] * len(spin) + [Uo] * len(spin)):
                    value = np.moveaxis(
                        np.tensordot(value, rotation, axes=(axis, 0)), -1, axis
                    )
                setattr(R, spin, value)
            targets.append(cfg.from_operator(R))
        self._guess_targets = np.column_stack(targets)
        self._guess_targets /= np.linalg.norm(self._guess_targets, axis=0)

        def comparison(vectors):
            return self.metric @ (cfg.basis @ vectors)

        if self.matrix is not None:
            omega, vectors = diagonalize_singles(
                self.matrix, min(1.0e-7, self.options["amp_convergence"])
            )
            r = comparison(vectors)
            r /= np.linalg.norm(r, axis=0)
            overlap = abs(self._guess_targets.T @ r) ** 2
            _, selected = linear_sum_assignment(-overlap)
            self.guess_overlaps = overlap[np.arange(self.nroots), selected]
            return omega[selected], vectors[:, selected]
        # Initial C = M^-1 guess; final selection compares M C against CI SD.
        initial = self.action.factor.solve(
            self._guess_targets - self.action.singles_to_doubles @ self._guess_targets
        )
        initial = cfg.basis.T @ initial
        dimension = cfg.basis.shape[1]

        class PackedVector:
            def __init__(self):
                self.values = np.zeros(dimension)

            def flatten(self):
                return self.values

            def unflatten(self, values):
                self.values = values.copy()

        guesses = []
        for column in initial.T:
            R = PackedVector()
            R.unflatten(column)
            guesses.append(R)
        diagonal = []
        for spin, row, _ in cfg.entries:
            rank = len(spin)
            diagonal.append(
                sum(H.a.vv[a, a] for a in row[:rank])
                - sum(H.a.oo[i, i] for i in row[rank:])
            )
        diagonal = np.asarray(cfg.basis.power(2).T @ np.array(diagonal)).ravel()

        def HR(dR, R, T, H, rhf, system):
            return self.G @ R.flatten()

        def update(R, omega, H, rhf, system):
            denominator = omega - diagonal
            denominator = np.where(
                abs(denominator) < 1.0e-8,
                np.where(denominator < 0, -1.0e-8, 1.0e-8),
                denominator,
            )
            R.unflatten(R.flatten() / denominator)
            return R

        roots, omega, flags = _auxiliary_davidson(
            HR,
            update,
            guesses,
            self.guess_energy,
            PackedVector,
            None,
            H,
            self.system,
            self.options,
            selection_vectors=self._guess_targets,
            selection_transform=comparison,
            label="Neutral",
            allow_complex_ritz=True,
        )
        if not all(flags):
            raise RuntimeError("Targeted neutral STEOM Davidson solve did not converge")
        vectors = np.column_stack([R.flatten() for R in roots])
        r = comparison(vectors)
        r /= np.linalg.norm(r, axis=0)
        self.guess_overlaps = abs(np.sum(self._guess_targets.conj() * r, axis=0)) ** 2
        return omega, vectors

    def solve(self):
        self.converged = False
        self._warn_df_port()
        hbar = self.hbar
        if not self.cc.converged:
            raise RuntimeError("STEOM requires converged CCSD amplitudes")
        no, nv = self.system.noccupied_alpha, self.system.nunoccupied_alpha
        if self.guess_settings is not None:
            self._run_neutral_guess()
        if self.automatic:
            bare = self.reference.H
            energy, cis = np.linalg.eigh(
                singles_matrix(
                    SimpleNamespace(
                        oo=bare.a.oo, vv=bare.a.vv, vovo=bare.ab.vovo, voov=bare.ab.voov
                    ),
                    self.multiplicity,
                )
            )
            count = min(self.nroots, no * nv)
            self.cis_omega, self.cis_vectors = (
                energy[:count],
                cis[:, :count].T.reshape(count, nv, no),
            )
            if self.guess_settings is not None:
                Uo, Uv, occupied, virtual, wo, wv = self._guess_orbitals()
            else:
                Uo, Uv, occupied, virtual, wo, wv = natural_orbital_space(
                    self.cis_vectors,
                    bare.a.oo,
                    bare.a.vv,
                    self.occupied_threshold,
                    self.virtual_threshold,
                )
            self.no_occupations = wo, wv
        else:
            # Contiguous active blocks are required by ipcisd/eacisd guesses.
            occ, vir = self._manual_occupied, self._manual_virtual
            Uo = np.eye(no)[:, np.r_[np.setdiff1d(np.arange(no), occ), occ]]
            Uv = np.eye(nv)[:, np.r_[vir, np.setdiff1d(np.arange(nv), vir)]]
            occupied, virtual = np.arange(no - len(occ), no), np.arange(len(vir))
        H, T = rotate_hbar(hbar.H, hbar.T, Uo, Uv)
        self.orbital_rotations = Uo, Uv
        self.active_occupied, self.active_virtual = occupied, virtual
        ip = self._extended_auxiliary("ip", occupied, virtual, H, T)
        ea = self._extended_auxiliary("ea", occupied, virtual, H, T)
        self.S = enlarged_transformation(ip, ea, no, nv)
        self.action = ExtendedAction(
            self.system, H, T, self.S, self.multiplicity, self.condition_limit
        )
        cfg = self.action.configurations
        dimension = cfg.basis.shape[1]
        if self.nroots > dimension:
            raise ValueError(
                f"nroots exceeds the {dimension} spin-adapted SD configurations"
            )
        self.G, self.metric = self.action.operator, self.action.metric
        self.matrix = None
        if dimension <= self.dense_limit or self.nroots >= dimension - 1:
            self.matrix = np.column_stack(
                [self.G @ column for column in np.eye(dimension)]
            )
            if self.guess_settings is not None:
                omega, vectors = self._targeted_neutral(cfg, H, Uo, Uv)
            else:
                omega, vectors = diagonalize_singles(
                    self.matrix, min(1.0e-7, self.options["amp_convergence"])
                )
                selected = np.argsort(omega.real)[: self.nroots]
                omega, vectors = omega[selected], vectors[:, selected]
        elif self.guess_settings is not None:
            omega, vectors = self._targeted_neutral(cfg, H, Uo, Uv)
        else:
            try:
                omega, vectors = eigs(
                    self.G,
                    k=self.nroots,
                    which="SR",
                    tol=self.options["amp_convergence"] * 0.1,
                    maxiter=self.options["maximum_iterations"],
                    ncv=min(
                        dimension,
                        max(
                            self.options["davidson_max_subspace_size"],
                            2 * self.nroots + 1,
                        ),
                    ),
                    v0=np.random.default_rng(0).normal(size=dimension),
                )
            except ArpackNoConvergence as error:
                raise RuntimeError(
                    "Extended STEOM neutral Arnoldi solve did not converge"
                ) from error
            order = np.argsort(omega.real)
            omega, vectors = omega[order], vectors[:, order]
        if np.max(abs(omega.imag)) > 1.0e-7 or np.max(abs(vectors.imag)) > 1.0e-7:
            raise RuntimeError("Extended STEOM has complex selected neutral roots")
        self.omega, self.working_vectors = omega.real, vectors.real
        self.R, self.residual_norms, self.singles_weights = [], [], []
        self.spin_leakage_norms = []
        for k, vector in enumerate(self.working_vectors.T):
            vector = vector / np.linalg.norm(vector)
            raw = cfg.basis @ vector
            raw_sigma = self.action.raw_matvec(raw)
            self.residual_norms.append(np.linalg.norm(raw_sigma - self.omega[k] * raw))
            self.spin_leakage_norms.append(
                np.linalg.norm(raw_sigma - cfg.basis @ (cfg.basis.T @ raw_sigma))
            )
            lifted, _ = self.action.lift(raw)
            n1 = sum(
                abs(v) ** 2 for s, v in zip(cfg.states, lifted) if s.bit_count() == 2
            )
            self.singles_weights.append(n1 / np.dot(lifted, lifted))
            R = cfg.to_operator(raw)
            for spin in ("a", "b", "aa", "ab", "bb"):
                value = getattr(R, spin)
                for axis, rotation in enumerate([Uv] * len(spin) + [Uo] * len(spin)):
                    value = np.moveaxis(
                        np.tensordot(value, rotation.T, axes=(axis, 0)), -1, axis
                    )
                setattr(R, spin, np.asfortranarray(value))
            self.R.append(R)
        self.residual_norms, self.singles_weights = (
            np.array(self.residual_norms),
            np.array(self.singles_weights),
        )
        self.spin_leakage_norms = np.array(self.spin_leakage_norms)
        if np.max(self.spin_leakage_norms) > self.options["amp_convergence"]:
            raise RuntimeError(
                "Extended STEOM transformation does not preserve the requested spin space; check auxiliary spin completeness"
            )
        if self.root_irreps is not None:
            self.check_root_symmetry(self.R, self.root_irreps)
        print(
            "\n   Extended STEOM roots: excitation energy (Eh), singles weight, residual"
        )
        for k in range(self.nroots):
            print(
                f"   {k:4d}  {self.omega[k]:16.10f}  {self.singles_weights[k]:10.6f}  {self.residual_norms[k]:10.3e}"
            )
        self.root_converged = list(
            self.residual_norms < self.options["amp_convergence"]
        )
        self._finish_solve(all(self.root_converged), self.cc, hbar)
        return self
