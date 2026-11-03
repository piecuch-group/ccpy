"""Spin-integrated STEOM-CCSD with manual or CIS natural-orbital active spaces."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

from ccpy.eomcc.steomccsd import (
    amplitudes_from_eom,
    build_g,
    diagonalize_singles,
    guess_densities,
    natural_orbital_space_from_densities,
    natural_orbital_space,
    rotate_hbar,
    singles_character,
    singles_matrix,
)
from ccpy.methods.base import MethodBase
from ccpy.methods.cc import CCSD
from ccpy.methods.guess import GUESS_ORDER, InitialGuessMixin
from ccpy.methods.hbar import HBarCCSDLegacy
from ccpy.models.operators import ClusterOperator, FockOperator


class STEOMCC(InitialGuessMixin, MethodBase):
    """Closed-shell STEOM-CCSD, with excitation energies in ``omega`` (hartree).

    ``STEOMCC(ccsd_or_scf, nroots=3).energy()`` uses automatic CIS-NO selection.
    Supply both ``nacto`` and ``nactu`` for canonical HOMO/LUMO active spaces, or
    both ``active_occupied`` and ``active_virtual`` for explicit zero-based labels
    in the correlated orbital space (frozen/deleted orbitals excluded).

    ``multiplicity`` is 1 or 3. Automatic cumulative NO thresholds default to
    0.99 for singlets and 0.999 for triplets; auxiliary singles-character cutoffs
    default to 0.9 and 0.8 respectively (Dutta et al., JCP 146, 074103, 2017).
    IP/EA diagonalizations retain the full 2h1p/2p1h correlation spaces, targeting
    one principal root per active orbital. Low-character roots are discarded;
    unconverged roots and ill-conditioned model overlaps raise errors.

    G and S are in ``orbital_rotations``' working basis; returned ClusterOperator
    singles ``R`` are in the upstream reference basis. S1 is omitted, so these
    singles are not full STEOM wavefunctions or property/transition densities.
    """

    requires_df_port = True
    # neutral target guess on the bare Hamiltonian (ccpy.methods.guess), as for EOMCC
    default_guess = "cisd"
    guess_methods = ("cis", "cisd")

    def __new__(cls, *args, **kwargs):
        if cls is STEOMCC and kwargs.get("level", "sd") == "sdt":
            from ccpy.methods.steomccsdt import STEOMCCSDT

            return MethodBase.__new__(STEOMCCSDT, *args, **kwargs)
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        upstream,
        nroots=1,
        multiplicity=1,
        nacto=None,
        nactu=None,
        active_occupied=None,
        active_virtual=None,
        occupied_threshold=None,
        virtual_threshold=None,
        ip_threshold=None,
        ea_threshold=None,
        condition_limit=1.0e10,
        nfrozen=None,
        ndelete=None,
        level="sd",
        **options,
    ):
        if level != "sd":
            raise ValueError("level must be sd or sdt")
        self.level = level
        super().__init__(upstream, nfrozen, ndelete, **options)
        if not self.reference.rhf_symmetry or self.system.multiplicity != 1:
            raise ValueError(
                "Spin-integrated STEOM-CCSD requires a closed-shell restricted reference"
            )
        if multiplicity not in (1, 3):
            raise ValueError("multiplicity must be 1 (singlet) or 3 (triplet)")
        no, nv = self.system.noccupied_alpha, self.system.nunoccupied_alpha
        if not isinstance(nroots, (int, np.integer)) or not 1 <= nroots <= no * nv:
            raise ValueError("nroots must be between 1 and noccupied * nvirtual")
        counts = nacto is not None or nactu is not None
        labels = active_occupied is not None or active_virtual is not None
        if counts and labels:
            raise ValueError("Specify active counts or active orbital lists, not both")
        if counts:
            if (
                not isinstance(nacto, (int, np.integer))
                or not 1 <= nacto <= no
                or not isinstance(nactu, (int, np.integer))
                or not 1 <= nactu <= nv
            ):
                raise ValueError(
                    "Both nacto and nactu must be positive counts within the correlated space"
                )
            active_occupied, active_virtual = (
                np.arange(no - nacto, no),
                np.arange(nactu),
            )
        self.automatic = not (counts or labels)
        self._manual_occupied = (
            self._labels(active_occupied, no, "occupied")
            if not self.automatic
            else None
        )
        self._manual_virtual = (
            self._labels(active_virtual, nv, "virtual") if not self.automatic else None
        )
        default_no, default_singles = (0.99, 0.9) if multiplicity == 1 else (0.999, 0.8)
        self.occupied_threshold = (
            default_no if occupied_threshold is None else occupied_threshold
        )
        self.virtual_threshold = (
            default_no if virtual_threshold is None else virtual_threshold
        )
        self.ip_threshold = default_singles if ip_threshold is None else ip_threshold
        self.ea_threshold = default_singles if ea_threshold is None else ea_threshold
        for name in ("occupied_threshold", "virtual_threshold"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        for name in ("ip_threshold", "ea_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if not np.isfinite(condition_limit) or condition_limit < 1:
            raise ValueError("condition_limit must be finite and at least 1")
        self.condition_limit = condition_limit
        self.nroots, self.multiplicity = int(nroots), multiplicity
        self.cc = self.resolve_cc(self, upstream, CCSD)
        if type(self.cc) is not CCSD:
            raise ValueError("STEOMCC requires a CCSD ground-state calculation")
        self.omega = self.R = self.G = self.ip = self.ea = None
        self.root_converged = None
        self.guess_settings = None
        self.guess_energy = self.guess_vectors = self.guess_irreps = None
        self.root_irreps = None

    # ------------------------------------------------------------------ guess
    # initial_guess(method="cisd", nacto=, nactu=, roots_per_irrep=, roots=, use_symmetry=) of
    # InitialGuessMixin targets neutral spin-adapted CIS/CISd states of the bare Hamiltonian:
    # CISd keeps all singles and active doubles (its active space is independent of the
    # auxiliary model; explicit counts are required with automatic orbital selection).
    # Without roots_per_irrep, the lowest roots up to max(roots) (or nroots) are computed.
    # Doubles-dominated neutral states require level='sdt'.
    def _default_guess_multiplicity(self):
        return self.multiplicity

    def _configure_guess(self, settings):
        if settings["multiplicity"] != self.multiplicity:
            raise ValueError("The guess multiplicity must be that of the STEOM states.")
        no, nv = self.system.noccupied_alpha, self.system.nunoccupied_alpha
        if settings["method"] == "cisd":
            nacto, nactu = settings["nacto"], settings["nactu"]
            if not self.automatic:
                nacto = len(self._manual_occupied) if nacto is None else nacto
                nactu = len(self._manual_virtual) if nactu is None else nactu
            if (not isinstance(nacto, (int, np.integer)) or not 1 <= nacto <= no
                    or not isinstance(nactu, (int, np.integer)) or not 1 <= nactu <= nv):
                raise ValueError("CISd needs positive guess nacto/nactu within the correlated space")
            settings["nacto"], settings["nactu"] = int(nacto), int(nactu)
        else:
            settings["nacto"], settings["nactu"] = 0, 0
        roots, counts = settings["roots"], settings["roots_per_irrep"]
        if counts is not None:
            count = self._count(counts)
        else:
            count = max(roots) + 1 if roots is not None else self.nroots
        selected = list(range(count)) if roots is None else roots
        if self.level == "sd" and len(selected) > no * nv:
            raise ValueError("The singles-only STEOM problem cannot return more than nocc*nvirt roots")
        self.nroots = len(selected)
        return settings

    def _default_root_count(self):
        return self.nroots

    def _guess_arguments(self):
        return (self.system, self.reference.legacy_hamiltonian(copy_=False))

    def _new_guess_operator(self):
        return ClusterOperator(self.system, order=2)

    def _copy_setup(self, new):
        if self.guess_settings is not None:
            new.initial_guess(**self.guess_settings)

    def _run_neutral_guess(self):
        """The selected guess roots as normalized ClusterOperators (guess_vectors), with their
        energies, irreps and singles weights."""
        self.run_guess()
        roots = self.selected_guess_roots()
        order = GUESS_ORDER[self.guess_settings["method"]]
        result, weights = [], []
        for root in roots:
            R = self._new_guess_operator()
            R.unflatten(self.guess_vectors[:, root], order=order)
            norm = sum(
                weight * np.vdot(getattr(R, s), getattr(R, s)).real
                for s, weight in [("a", 1), ("b", 1), ("aa", 0.25), ("ab", 1), ("bb", 0.25)]
            )
            if norm < 1.0e-14:
                raise ValueError("Requested neutral guess is unavailable in this spin/active space")
            for spin in ("a", "b", "aa", "ab", "bb"):
                setattr(R, spin, getattr(R, spin) / np.sqrt(norm))
            result.append(R)
            weights.append(np.vdot(R.a, R.a).real + np.vdot(R.b, R.b).real)
        self.guess_energy = self.guess_energy[roots].copy()
        self.guess_irreps = self.root_irreps = [self.guess_irreps[k] for k in roots]
        self.guess_vectors, self.guess_singles_weights = result, np.array(weights)
        return result

    def _guess_orbitals(self):
        do, dv = guess_densities(self.guess_vectors)
        bare = self.reference.H
        return natural_orbital_space_from_densities(
            do,
            dv,
            bare.a.oo,
            bare.a.vv,
            self.occupied_threshold,
            self.virtual_threshold,
        )

    @staticmethod
    def _labels(values, size, name):
        if values is None:
            raise ValueError("Supply both active_occupied and active_virtual")
        labels = np.asarray(values)
        if (
            labels.ndim != 1
            or not len(labels)
            or labels.dtype.kind not in "iu"
            or len(np.unique(labels)) != len(labels)
            or np.any(labels < 0)
            or np.any(labels >= size)
        ):
            raise ValueError(
                f"Active {name} labels must be distinct integers in [0,{size})"
            )
        return labels.astype(int, copy=True)

    @property
    def hbar(self):
        return self.cc.get_hbar(HBarCCSDLegacy)

    def _auxiliary(self, sector, active, H, T):
        from ccpy.drivers.solvers import eomcc_davidson

        module = import_module(
            "ccpy.eomcc." + ("ipeom2" if sector == "ip" else "eaeom2")
        )
        particles, holes = (1, 2) if sector == "ip" else (2, 1)
        vectors, energies, flags = [], [], []
        options = {
            **self.options,
            "RHF_symmetry": False,
            "davidson_selection_method": "overlap",
        }
        for k in active:
            R = FockOperator(self.system, particles, holes)
            dR = FockOperator(self.system, particles, holes)
            R.a[k] = 1.0
            omega = -H.a.oo[k, k] if sector == "ip" else H.a.vv[k, k]
            print(f"   STEOM {sector.upper()} auxiliary root for orbital {k}")
            R, omega, converged = eomcc_davidson(
                module.HR,
                module.update,
                R.flatten()[:, None],
                R,
                dR,
                omega,
                T,
                H,
                self.system,
                options,
            )
            vectors.append(R)
            energies.append(omega)
            flags.append(converged)
        result = SimpleNamespace(
            R=vectors,
            omega=np.array(energies),
            root_converged=flags,
            singles_character=np.array([singles_character(r) for r in vectors]),
        )
        # Persist diagnostics even when a solve fails.
        setattr(self, sector, result)
        if not all(flags):
            raise RuntimeError(
                f"STEOM {sector.upper()} auxiliary calculation did not converge"
            )
        return result

    def solve(self):
        self.converged = False
        self._warn_df_port()
        hbar = self.hbar
        if not self.cc.converged:
            raise RuntimeError("STEOM requires converged CCSD amplitudes")
        no, nv = self.system.noccupied_alpha, self.system.nunoccupied_alpha
        bare = self.reference.H
        if self.guess_settings is not None:
            self._run_neutral_guess()
            if np.any(self.guess_singles_weights < 1.0e-16):
                raise ValueError(
                    "Neutral guess has no singles projection; use level='sdt' for doubly excited states"
                )
        self.cis_omega, cis = np.linalg.eigh(
            singles_matrix(
                SimpleNamespace(
                    oo=bare.a.oo, vv=bare.a.vv, vovo=bare.ab.vovo, voov=bare.ab.voov
                ),
                self.multiplicity,
            )
        )
        self.cis_omega = self.cis_omega[: self.nroots]
        cis = cis[:, : self.nroots].T.reshape(self.nroots, nv, no)
        self.cis_vectors = cis.copy()
        if self.automatic:
            if self.guess_settings is not None:
                Uo, Uv, occupied, virtual, wo, wv = self._guess_orbitals()
            else:
                Uo, Uv, occupied, virtual, wo, wv = natural_orbital_space(
                    cis,
                    bare.a.oo,
                    bare.a.vv,
                    self.occupied_threshold,
                    self.virtual_threshold,
                )
            self.no_occupations = (wo, wv)
            H, T = rotate_hbar(hbar.H, hbar.T, Uo, Uv)
        else:
            Uo, Uv = np.eye(no), np.eye(nv)
            occupied, virtual = (
                self._manual_occupied.copy(),
                self._manual_virtual.copy(),
            )
            H, T = hbar.H, hbar.T
        print(
            f"   STEOM-CCSD multiplicity {self.multiplicity}: "
            f"{len(occupied)} active occupied, {len(virtual)} active virtual orbitals"
        )
        self.orbital_rotations = (Uo, Uv)
        ip = self._auxiliary("ip", occupied, H, T)
        ea = self._auxiliary("ea", virtual, H, T)
        self.S_ip, occupied, ip.retained_roots = amplitudes_from_eom(
            ip.R, occupied, "ip", self.ip_threshold, self.condition_limit
        )
        self.S_ea, virtual, ea.retained_roots = amplitudes_from_eom(
            ea.R, virtual, "ea", self.ea_threshold, self.condition_limit
        )
        self.active_occupied, self.active_virtual = occupied, virtual
        self.G = build_g(H, self.S_ip, self.S_ea, occupied, virtual)
        self.matrix = singles_matrix(self.G, self.multiplicity)
        omega, vectors = diagonalize_singles(
            self.matrix, min(1.0e-7, self.options["amp_convergence"])
        )
        if self.automatic or self.guess_settings is not None:
            spatial = (
                cis
                if self.guess_settings is None
                else np.array(
                    [
                        (R.a + (1 if self.multiplicity == 1 else -1) * R.b) / np.sqrt(2)
                        for R in self.guess_vectors
                    ]
                )
            )
            guesses = np.array([Uv.T @ c @ Uo for c in spatial]).reshape(
                self.nroots, -1
            )
            norms = np.linalg.norm(guesses, axis=1)
            if np.any(norms < 1.0e-8):
                raise ValueError(
                    "Neutral guess has no singles projection; use level='sdt' for doubly excited states"
                )
            guesses /= norms[:, None]
            overlaps = abs(guesses @ vectors) ** 2
            _, selected = linear_sum_assignment(-overlaps)
            self.guess_overlaps = overlaps[np.arange(self.nroots), selected]
        else:
            selected = np.argsort(omega.real)[: self.nroots]
        if np.any(np.abs(omega[selected].imag) > 1.0e-7) or np.any(
            np.abs(vectors[:, selected].imag) > 1.0e-7
        ):
            raise RuntimeError(
                "STEOM produced complex selected roots; inspect the active space and auxiliary states"
            )
        self.omega = omega[selected].real
        vectors = vectors[:, selected].real
        self.R, weights, residuals = [], [], []
        for k in range(self.nroots):
            vector = vectors[:, k].reshape(nv, no)
            residuals.append(
                np.linalg.norm(
                    self.matrix @ vectors[:, k] - self.omega[k] * vectors[:, k]
                )
            )
            weights.append(
                np.linalg.norm(vector[np.ix_(virtual, occupied)]) ** 2
                / np.linalg.norm(vector) ** 2
            )
            R = ClusterOperator(self.system, order=1)
            R.a = Uv @ vector @ Uo.T / np.sqrt(2)
            R.b = (1 if self.multiplicity == 1 else -1) * R.a
            self.R.append(R)
        self.active_weights = np.array(weights)
        self.residual_norms = np.array(residuals)
        self.root_converged = list(
            self.residual_norms < self.options["amp_convergence"]
        )
        if self.root_irreps is not None:
            self.check_root_symmetry(self.R, self.root_irreps)
        print("\n   STEOM-CCSD roots: excitation energy (Eh), active weight, residual")
        for k in range(self.nroots):
            print(
                f"   {k:4d}  {self.omega[k]:16.10f}  "
                f"{self.active_weights[k]:10.6f}  {self.residual_norms[k]:10.3e}"
            )
        self._finish_solve(all(self.root_converged), self.cc, hbar)
        return self

    def energy(self, root=None):
        """Total excited-state energies; ``omega`` contains excitation energies."""
        self.ensure_solved()
        total = self.reference_energy + self.cc.correlation_energy + self.omega
        return total if root is None else total[root]
