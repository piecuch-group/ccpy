"""Noniterative corrections to CC and EOMCC energies (CR-CC(2,3), CC(P;Q), CCSD(T), ...).

    cr = CRCC23(mf, nfrozen=1)                 # variant "D" by default
    e_crcc23 = cr.energy()                     # runs CCSD, HBar, left-CCSD, then the correction
    cr.deltas                                  # all variants {"A": ..., "B": ..., "C": ..., "D": ...}

Ground-state corrections accept an SCF object or an upstream CC/left-CC method;
excited-state corrections take an EOMCC (or left-EOMCC) method.
"""
import copy
from importlib import import_module

import numpy as np

from ccpy.methods.base import MethodBase
from ccpy.methods.cc import CCPBase, CCSD, CCSDT, CCSDT_P
from ccpy.methods.eomcc import EOMCCBase
from ccpy.methods.hbar import HBarCCSD, HBarCCSDLegacy, HBarCCSDT, HBarCCSDT_P
from ccpy.methods.left import LeftCCSD, LeftCCSDT, LeftCCSDT_P, LeftEOMCCBase


class CorrectionBase(MethodBase):
    """Base class of noniterative corrections (scheme: noniterative correction).

    Parameters
    ----------
    variant : str, optional
        Which variant ("A", "B", "C", "D") energy() reports; all are kept in ``deltas``.
    """

    variants = ("A", "B", "C", "D")
    default_variant = "D"
    module = function = None
    gradient_variants = ()          # variants with an analytic gradient

    def __init__(self, upstream, variant=None, nfrozen=None, ndelete=None, **options):
        super().__init__(upstream, nfrozen, ndelete, **options)
        self.variant = (variant or self.default_variant).upper()
        if self.variant not in self.variants:
            raise ValueError(f"{type(self).__name__}: variant must be one of {self.variants}.")
        self.deltas = None

    def _function(self, module=None, function=None):
        return getattr(import_module(module or self.module), function or self.function)

    def _fock(self):
        """Bare Fock operator (orbital-energy denominators)."""
        return self.reference.legacy_hamiltonian(copy_=False) if self.requires_df_port else self.reference.H

    def solve(self):
        return self.build()

    def has_analytic_gradient(self, **kwargs):
        return super().has_analytic_gradient(**kwargs) and self.variant in self.gradient_variants


class GroundStateCorrectionBase(CorrectionBase):
    """Correction to a ground-state CC energy built from T, HBar, and L.

    Corrections with ``supports_stochastic`` accept ``stochastic``: True, or a dict of options
    of the semi-stochastic algorithm (see ccpy.moments.stochastic_triples.DEFAULT_OPTIONS,
    e.g. ``{"target_error": 1e-5, "seed": 1}``). The deltas are then Monte Carlo estimates
    whose statistical errors (one standard deviation) are kept in ``errors``.
    """

    cc_class = CCSD
    hbar_class = HBarCCSD
    left_class = LeftCCSD
    supports_stochastic = False

    def __init__(self, upstream, variant=None, nfrozen=None, ndelete=None, stochastic=None, **options):
        super().__init__(upstream, variant, nfrozen, ndelete, **options)
        if stochastic and not self.supports_stochastic:
            raise ValueError(f"{type(self).__name__} has no semi-stochastic implementation.")
        self.stochastic = (dict(stochastic) if isinstance(stochastic, dict) else {}) if stochastic else None
        self.errors = self.stochastic_info = None
        self.cc = self._resolve_cc(upstream)
        self.left = self._resolve_left(upstream)

    def _resolve_cc(self, upstream):
        return self.resolve_cc(self, upstream, self.cc_class)

    def _resolve_left(self, upstream):
        if self.left_class is None:
            return None
        if isinstance(upstream, self.left_class):
            return upstream
        if isinstance(self.cc._left, self.left_class):
            return self.cc._left
        if self.cc._left is None and self.cc.left_class == self.left_class.__name__:
            # shared with other methods built on the same CC
            return self.cc.left
        return self.left_class(self.cc, **self.user_options)

    @property
    def hbar(self):
        return self.cc.get_hbar(self.hbar_class) if self.hbar_class is not None else None

    def build(self):
        """Compute the correction from the (converged on demand) upstream quantities."""
        self._warn_df_port()
        cc = self.cc.ensure_solved()
        hbar = self.hbar
        left = self.left.ensure_solved() if self.left is not None else None
        self.deltas = self._compute(cc, hbar, left)
        self._finish_solve(True, *[x for x in (cc, hbar, left) if x is not None])
        return self

    def _compute(self, cc, hbar, left):
        _, deltas = self._function()(hbar.T, left.L, hbar.correlation_energy, hbar.H, self._fock(), self.system,
                                     self.options["RHF_symmetry"])
        return deltas

    def _compute_stochastic(self, function, *args, **kwargs):
        _, deltas, self.errors, self.stochastic_info = function(*args, options=self.stochastic, **kwargs)
        return deltas

    def energy(self):
        """Total energy: CC energy plus the correction of the chosen variant."""
        return self._complete_ground_energy(self.raw_energy())

    def raw_energy(self):
        self.ensure_solved()
        return self.reference_energy + self.cc.correlation_energy + self.deltas[self.variant]

    def has_analytic_gradient(self, **kwargs):
        return super().has_analytic_gradient(**kwargs) and self.stochastic is None

    def _analytic_gradient(self, zvector_tol=1.0e-10, perturbation=None, properties=None,
                           **kwargs):
        self.ensure_solved()
        hbar, left = self.hbar, self.left
        return self._gradient_function()(hbar.T, left.L, hbar.H, self.reference, delta=self.deltas[self.variant],
                                         zvector_tol=zvector_tol, perturbation=perturbation, properties=properties)


class CRCC23(GroundStateCorrectionBase):
    """CR-CC(2,3) triples correction to CCSD (DF/Cholesky); semi-stochastic with ``stochastic``."""
    module, function = "ccpy.moments.crcc23_chol", "calc_crcc23"
    gradient_module, gradient_function = "ccpy.gradients.crcc23_gradient", "calc_crcc23_gradient"
    gradient_variants = ("A",)
    supports_stochastic = True

    def _compute(self, cc, hbar, left):
        if self.stochastic is None:
            return super()._compute(cc, hbar, left)
        from ccpy.moments.stochastic_triples import calc_crcc23_stochastic

        return self._compute_stochastic(calc_crcc23_stochastic, hbar.T, left.L, hbar.correlation_energy, hbar.H,
                                        self._fock(), self.system, self.options["RHF_symmetry"])


class CCSDpT(GroundStateCorrectionBase):
    """CCSD(T), from the bare Hamiltonian (no HBar or left-CC needed)."""
    module, function = "ccpy.moments.crcc23", "calc_ccsdpt"
    requires_df_port = True
    variants = ("A",)
    default_variant = "A"
    hbar_class = left_class = None

    def _compute(self, cc, hbar, left):
        _, deltas = self._function()(cc.T, cc.correlation_energy, self._fock(), self.system,
                                     self.options["RHF_symmetry"])
        return deltas


class PspaceCorrectionMixin:
    """Corrections defined with respect to a P space of triples (``t3_excitations``),
    taken from the upstream CC(P) method or given explicitly."""

    def _resolve_cc(self, upstream):
        if isinstance(upstream, MethodBase) and hasattr(upstream, "cc"):
            return upstream.cc
        if self._t3 is None:
            raise ValueError(f"{type(self).__name__} needs t3_excitations or an upstream CC(P) method.")
        return CCSDT_P(self.reference, self._t3, **self.user_options)

    @property
    def t3_excitations(self):
        return self.cc.t3_excitations if isinstance(self.cc, CCPBase) else self._t3


class CCT3(PspaceCorrectionMixin, GroundStateCorrectionBase):
    """CC(t;3): CR-CC(2,3)-like correction to CCSDt (active-space CCSDT(P)) (DF/Cholesky).

    nacto, nactu : active occupied/unoccupied orbitals defining the CCSDt triples (as used
                   to build t3_excitations); applied to the System if given
    num_active : minimum number of active indices of the CCSDt triples
    stochastic : semi-stochastic evaluation (see GroundStateCorrectionBase)
    """
    module, function = "ccpy.moments.cct3_chol", "calc_cct3"
    supports_stochastic = True

    def __init__(self, upstream, t3_excitations=None, num_active=1, nacto=None, nactu=None, variant=None,
                 nfrozen=None, ndelete=None, stochastic=None, **options):
        self._t3 = t3_excitations
        self.num_active = num_active
        self.active_space = (nacto, nactu)
        super().__init__(upstream, variant, nfrozen, ndelete, stochastic, **options)

    def _compute(self, cc, hbar, left):
        if self.active_space != (None, None):
            self.system.set_active_space(*(n or 0 for n in self.active_space))
        if self.stochastic is not None:
            from ccpy.moments.stochastic_triples import calc_cct3_stochastic

            return self._compute_stochastic(calc_cct3_stochastic, hbar.T, left.L, cc.correlation_energy, hbar.H,
                                            self._fock(), self.system, self.options["RHF_symmetry"],
                                            num_active=self.num_active)
        _, deltas = self._function()(hbar.T, left.L, cc.correlation_energy, hbar.H, self._fock(), self.system,
                                     self.options["RHF_symmetry"], num_active=[self.num_active])
        return deltas


class CCP3(PspaceCorrectionMixin, GroundStateCorrectionBase):
    """CC(P;3) correction. With two_body_approx=True (default, DF/Cholesky) it uses the
    CCSD-like HBar and left-CCSD; otherwise the full correction with the CCSDT(P) HBar and
    left-CCSDT(P) (legacy four-index). The two-body approximation can be evaluated
    semi-stochastically with ``stochastic`` (see GroundStateCorrectionBase)."""
    module, function = "ccpy.moments.ccp3_chol", "calc_ccp3_2ba"
    supports_stochastic = True

    def __init__(self, upstream, t3_excitations=None, two_body_approx=True, target_irrep=None, high_memory=False,
                 variant=None, nfrozen=None, ndelete=None, stochastic=None, **options):
        self._t3 = t3_excitations
        self.two_body_approx = two_body_approx
        self.target_irrep = target_irrep
        self.high_memory = high_memory
        if not two_body_approx:
            if stochastic:
                raise ValueError("CCP3: the semi-stochastic correction requires two_body_approx=True.")
            self.requires_df_port = True
            self.hbar_class, self.left_class = HBarCCSDT_P, LeftCCSDT_P
        super().__init__(upstream, variant, nfrozen, ndelete, stochastic, **options)

    def _compute(self, cc, hbar, left):
        if self.stochastic is not None:
            from ccpy.moments.stochastic_triples import calc_ccp3_2ba_stochastic

            return self._compute_stochastic(calc_ccp3_2ba_stochastic, hbar.T, left.L, self.t3_excitations,
                                            cc.correlation_energy, hbar.H, self._fock(), self.system,
                                            self.options["RHF_symmetry"])
        if self.two_body_approx:
            function = self._function()
        else:
            function = self._function("ccpy.moments.ccp3", "calc_ccp3_high_memory" if self.high_memory else "calc_ccp3")
        _, deltas = function(hbar.T, left.L, self.t3_excitations, cc.correlation_energy, hbar.H, self._fock(),
                             self.system, self.options["RHF_symmetry"], target_irrep=self.target_irrep)
        return deltas


class AdaptiveCCPQ(MethodBase):
    """Adaptive CC(P;Q) (DF/Cholesky): CCSDT(P) on a P space of triples grown, macroiteration
    by macroiteration, with the triples of largest CC(P;3) moment contributions.

    Each macroiteration solves CCSDT(P), HBar and left-CCSD (two-body approximation), and
    computes the CC(P;3)_D correction while selecting the triples to add. The last
    macroiteration only computes the correction. CC(P) restarts from the amplitudes of the
    previous P space.

    The calculation stops when two consecutive CC(P;Q) energies agree within the energy
    tolerance delta at the confidence level of convergence_z,

        |E(n) - E(n-1)| + z sqrt(sigma(n)^2 + sigma(n-1)^2) < delta,

    where sigma are the statistical errors of the CC(P;3) corrections (zero for the
    deterministic ones). With ``stochastic``, every macroiteration that may end the
    calculation first evaluates its correction semi-stochastically, stopping the Monte Carlo
    as soon as the test above is decided either way. If it passes, the calculation ends
    without the (deterministic) selection pass; otherwise the selection pass is run and its
    exact energy is kept.

        adapt = AdaptiveCCPQ(mf, nfrozen=2, percentage=[0.0, 1.0, 2.0])
        e = adapt.energy()                 # CC(P;3)_D energy of the last macroiteration
        adapt.ccp_energies, adapt.ccpq_energies

    Parameters
    ----------
    percentage : list of float, optional
        P space sizes as percentages of all triples (of the reference symmetry); one
        macroiteration each. Without it, each macroiteration adds as many triples as there
        are singles (and doubles, base_growth="ccsd") times selection_factor, for at most
        maximum_macroiterations + 1 macroiterations.
    pspaces : list of dict, optional
        P spaces to use instead of selecting them (e.g. ``pspaces`` of an earlier run); one
        macroiteration each. Copies made by at_geometry() reuse the P spaces of this run, so
        finite-difference energies are smooth functions of the geometry.
    energy_tolerance : float
        Convergence threshold on the change of the CC(P;Q) energy between macroiterations.
    n_det_max : int
        Stop once the P space (reference, singles, doubles, and triples) reaches this size.
    buffer_factor, minimum_threshold : selection buffer size (in units of the number of
        triples added) and smallest moment contribution considered.
    reset_amplitudes : bool
        Restart CC(P) and left-CCSD from zero amplitudes in every macroiteration.
    stochastic : bool or dict, optional
        Semi-stochastic CC(P;3) corrections (see CCP3 and the stopping test above) for the
        convergence tests and the last macroiteration; the triples selection stays
        deterministic. The statistical error target ("target_error") defaults to
        delta / (2 sqrt(2) convergence_z), half of the tolerance when both energies are
        stochastic, and must be below delta / (sqrt(2) convergence_z).
    convergence_z : float
        Number of standard deviations of the energy difference in the stopping test (1.96:
        95% confidence).

    Attributes
    ----------
    ccp_energies, ccpq_energies : list of float, CC(P) and CC(P;3)_D energies
    ccpq_errors : list of float, statistical errors of ccpq_energies (zero when deterministic)
    extrapolated : dict of lists, Goodson extrapolations ("ccq", "ccr", "cccf")
    pspaces : list of dict, P space of each macroiteration
    timings : list of dict, wall times (s) of the CC(P) and CC(P;Q)/selection steps
    cc : CCSDT_P, left : LeftCCSD
    """

    def __init__(self, upstream, percentage=None, pspaces=None, energy_tolerance=1.0e-04,
                 maximum_macroiterations=10, n_det_max=100000000, selection_factor=1.0, base_growth="ccsd",
                 buffer_factor=2, minimum_threshold=0.0, reset_amplitudes=False, nfrozen=None, ndelete=None,
                 stochastic=None, convergence_z=1.96, **options):
        super().__init__(upstream, nfrozen, ndelete, **options)
        self.convergence_z = convergence_z
        self.stochastic = self._stochastic_options(stochastic, energy_tolerance, convergence_z)
        self.percentage = percentage
        self.adaptive_options = {"energy_tolerance": energy_tolerance, "n_det_max": n_det_max,
                                 "selection_factor": selection_factor, "base_growth": base_growth,
                                 "buffer_factor": buffer_factor, "minimum_threshold": minimum_threshold,
                                 "reset_amplitudes": reset_amplitudes}
        self.nmacro = maximum_macroiterations + 1 if percentage is None else len(percentage)
        self.fixed_pspaces = None
        self._set_fixed_pspaces(pspaces)
        empty = {s: np.ones((1, 6), dtype=np.int32) for s in ("aaa", "aab", "abb", "bbb")}
        self.cc = CCSDT_P(self.reference, empty, **self.user_options)
        self.left = LeftCCSD(self.cc, **self.user_options)
        self.ccp3 = None
        self.ccp_energies, self.ccpq_energies, self.ccpq_errors = [], [], []
        self.extrapolated = {"ccq": [], "ccr": [], "cccf": []}
        self.pspaces, self.timings = [], []
        self.adaptive_converged = False

    @staticmethod
    def _stochastic_options(stochastic, tolerance, z):
        if not stochastic:
            return None
        options = dict(stochastic) if isinstance(stochastic, dict) else {}
        if options.setdefault("variant", "D").upper() != "D":
            raise ValueError("AdaptiveCCPQ: the stochastic corrections must target variant D.")
        limit = tolerance / (np.sqrt(2.0) * z)
        options.setdefault("target_error", 0.5 * limit)
        if not options["target_error"] < limit:
            raise ValueError(f"AdaptiveCCPQ: stochastic target_error = {options['target_error']:.2e} cannot pass the "
                             f"stopping test; it must be below energy_tolerance / (sqrt(2) z) = {limit:.2e}.")
        return options

    def _converged(self, energy, error):
        """Stopping test against the CC(P;Q) energy of the previous macroiteration."""
        change = abs(energy - self.ccpq_energies[-1])
        margin = self.convergence_z * np.hypot(error, self.ccpq_errors[-1])
        return change + margin < self.adaptive_options["energy_tolerance"]

    def _stochastic_test(self):
        """Semi-stochastic CC(P;3) energy, stopped once the stopping test is decided."""
        tolerance, z = self.adaptive_options["energy_tolerance"], self.convergence_z
        base = self._complete_ground_energy(self.reference_energy + self.cc.correlation_energy)
        previous, previous_error = self.ccpq_energies[-1], self.ccpq_errors[-1]

        def decided(delta, error):
            change, margin = abs(base + delta - previous), z * np.hypot(error, previous_error)
            return change + margin < tolerance or change - margin > tolerance

        ccp3 = CCP3(self.left, stochastic={**self.stochastic, "stop_when": decided})
        return ccp3, ccp3.energy(), ccp3.errors["D"]

    def _set_fixed_pspaces(self, pspaces):
        if pspaces is not None and len(pspaces) > self.nmacro:
            raise ValueError(f"{len(pspaces)} fixed P spaces given for {self.nmacro} macroiterations.")
        self.fixed_pspaces = pspaces

    def _copy_setup(self, new):
        new._set_fixed_pspaces(copy.deepcopy(self.pspaces or self.fixed_pspaces))

    def _growth_plan(self):
        """Number of triples added after each macroiteration and the size of the base P space."""
        from ccpy.utilities.symmetry import count_doubles, count_singles, count_triples

        system = self.system
        irrep = system.point_group_irrep_to_number[system.reference_symmetry]
        if self.percentage is None:
            counts, _ = count_singles(system)
            if self.adaptive_options["base_growth"].lower() == "ccsd":
                doubles, _ = count_doubles(system)
                counts = [x + y for x, y in zip(counts, doubles)]
            increment = counts[irrep]
            num_add = [int(increment * self.adaptive_options["selection_factor"]) for _ in range(1, self.nmacro)]
            base_size = increment + 1
        else:
            counts, _ = count_triples(system)
            total = counts[irrep]
            increment = int(0.01 * total)
            num_add = np.zeros(len(self.percentage))
            for i in range(len(self.percentage) - 1):
                if self.percentage[i + 1] == 100.0:
                    num_add[i] = total - increment * self.percentage[i]
                else:
                    num_add[i] = increment * (self.percentage[i + 1] - self.percentage[i])
            num_add[-1] = 1
            base_size = 0
        if self.options["RHF_symmetry"]:
            num_add = [int(x / 2) for x in num_add]
        return list(num_add), base_size

    def _pspace_size(self, base_size):
        t3 = self.cc.t3_excitations
        return base_size + sum(t3[s].shape[0] for s in t3 if not np.all(t3[s][0, :] == 1))

    def _load_pspace(self, t3_excitations):
        """Use a given P space, restarting CC(P) from scratch unless it extends the current one."""
        current = self.cc.t3_excitations
        extends = all(t3_excitations[s].shape[0] >= current[s].shape[0]
                      and np.array_equal(np.asarray(t3_excitations[s])[:current[s].shape[0]], current[s])
                      for s in current)
        if not extends:
            self.cc.T = None
        self.cc.set_pspace(copy.deepcopy(t3_excitations))

    def solve(self):
        import time

        from ccpy.extrapolation.goodson_extrapolation import goodson_extrapolation
        from ccpy.moments.ccp3_chol import calc_ccp3_2ba_with_selection
        from ccpy.utilities.selection import add_spinorbital_triples_to_pspace

        self._print_header()
        for key, value in self.adaptive_options.items():
            print("  ", key, "=", value)
        num_add, base_size = self._growth_plan()
        counts = [{"aaa": 0, "aab": 0, "abb": 0, "bbb": 0} for _ in self.system.point_group_irrep_to_number]
        self.ccp_energies, self.ccpq_energies, self.ccpq_errors = [], [], []
        self.extrapolated = {"ccq": [], "ccr": [], "cccf": []}
        self.pspaces, self.timings = [], []
        self.adaptive_converged = False
        e_ref = self._complete_ground_energy(self.reference_energy)
        nmacro = self.nmacro if self.fixed_pspaces is None else len(self.fixed_pspaces)
        for imacro in range(nmacro):
            if self.fixed_pspaces is not None:
                self._load_pspace(self.fixed_pspaces[imacro])
            self.pspaces.append(copy.deepcopy(self.cc.t3_excitations))
            n_det = self._pspace_size(base_size)
            print(f"\n   Adaptive CC(P;Q) macroiteration {imacro}: {n_det} determinants in the P space")

            t0 = time.perf_counter()
            e_ccp = self.cc.energy()
            left = self.left.ensure_solved()
            hbar = self.cc.get_hbar(HBarCCSD)
            t1 = time.perf_counter()
            # macroiterations after the first may end the calculation (stopping test, n_det_max)
            check = imacro > 0 and self.fixed_pspaces is None
            last = imacro == nmacro - 1 or (check and n_det >= self.adaptive_options["n_det_max"])
            e_ccpq = selected = None
            converged = False
            if self.stochastic is not None and check and not last:
                ccp3, e_test, error = self._stochastic_test()
                converged = self._converged(e_test, error)
                print(f"   Stochastic stopping test: CC(P;Q) = {e_test:.10f} +/- {error:.1e}, "
                      + ("converged" if converged else "not converged"))
                if converged:
                    self.ccp3, e_ccpq = ccp3, e_test
            t2 = time.perf_counter()
            if converged:
                pass  # the stochastic energy of the stopping test is final
            elif last:
                self.ccp3 = CCP3(self.left, stochastic=self.stochastic)
                e_ccpq, error = self.ccp3.energy(), (self.ccp3.errors or {"D": 0.0})["D"]
            else:
                e_ccpq, selected = calc_ccp3_2ba_with_selection(
                    hbar.T, left.L, self.cc.t3_excitations, self.cc.correlation_energy, hbar.H, self.reference.H,
                    self.system, num_add[imacro], use_RHF=self.options["RHF_symmetry"],
                    min_thresh=self.adaptive_options["minimum_threshold"],
                    buffer_factor=self.adaptive_options["buffer_factor"])
                e_ccpq = self._complete_ground_energy(e_ccpq)
                error = 0.0
            if check and not converged:
                converged = self._converged(e_ccpq, error)
            t_end = time.perf_counter()
            self.timings.append({"ccp": t1 - t0, "stopping_test": t2 - t1, "ccpq_selection": t_end - t2})
            self.ccp_energies.append(e_ccp)
            self.ccpq_energies.append(e_ccpq)
            self.ccpq_errors.append(error)
            for approximant in self.extrapolated:
                self.extrapolated[approximant].append(goodson_extrapolation(e_ref, e_ccp, e_ccpq,
                                                                            approximant=approximant))
            print(f"   CC(P) = {e_ccp:.10f}   CC(P;Q) = {e_ccpq:.10f}" + (f" +/- {error:.1e}" if error else ""))
            print(f"   (CC(P) {t1 - t0:.2f} s, stopping test {t2 - t1:.2f} s, CC(P;Q) + selection {t_end - t2:.2f} s)")

            if converged:
                print("   Adaptive CC(P;Q) converged to within the energy tolerance.")
                self.adaptive_converged = True
                break
            if last:
                if check and n_det >= self.adaptive_options["n_det_max"]:
                    print(f"   Adaptive CC(P;Q) reached n_det_max = {self.adaptive_options['n_det_max']}.")
                break
            if self.fixed_pspaces is None:
                t3, counts = add_spinorbital_triples_to_pspace(selected, self.cc.t3_excitations, counts, self.system,
                                                               self.options["RHF_symmetry"])
                if self.adaptive_options["reset_amplitudes"]:
                    self.cc.T = None
                    self.left.L = None
                self.cc.set_pspace(t3)
            elif self.adaptive_options["reset_amplitudes"]:
                self.cc.T = None
                self.left.L = None

        print("\n   Iteration           E(P)             E(P;Q)       sigma(P;Q)         ex-CCq             ex-CCr"
              "             ex-CCcf")
        for i in range(len(self.ccp_energies)):
            print("   %8d    %.10f     %.10f     %.1e     %.10f     %.10f     %.10f" %
                  (i, self.ccp_energies[i], self.ccpq_energies[i], self.ccpq_errors[i], self.extrapolated["ccq"][i],
                   self.extrapolated["ccr"][i], self.extrapolated["cccf"][i]))
        self._finish_solve(True)
        return self

    def energy(self):
        """CC(P;3)_D energy of the last macroiteration (runs the adaptive loop if needed)."""
        self.ensure_solved()
        return self.ccpq_energies[-1]

    def raw_energy(self):
        return self.energy() - self._complete_ground_energy(0.0)


class CRCC24(GroundStateCorrectionBase):
    """CR-CC(2,4) quadruples correction to CCSD."""
    module, function = "ccpy.moments.crcc24", "calc_crcc24"
    requires_df_port = True
    hbar_class = HBarCCSDLegacy


class CRCC34(GroundStateCorrectionBase):
    """CR-CC(3,4) quadruples correction to CCSDT (RHF only; variant A)."""
    module, function = "ccpy.moments.crcc34", "calc_crcc34"
    requires_df_port = True
    default_variant = "A"
    cc_class, hbar_class, left_class = CCSDT, HBarCCSDT, LeftCCSDT


class CCP4(GroundStateCorrectionBase):
    """CC(P;4) quadruples correction for the P space ``t4_excitations``."""
    module, function = "ccpy.moments.ccp4", "calc_ccp4"
    requires_df_port = True
    hbar_class = HBarCCSDLegacy

    def __init__(self, upstream, t4_excitations, variant=None, nfrozen=None, ndelete=None, **options):
        super().__init__(upstream, variant, nfrozen, ndelete, **options)
        self.t4_excitations = t4_excitations

    def _compute(self, cc, hbar, left):
        _, deltas = self._function()(hbar.T, left.L, cc.correlation_energy, hbar.H, self._fock(), self.system,
                                     self.t4_excitations, self.options["RHF_symmetry"])
        return deltas


# ---------------------------------------------------------------------- excited states
class ExcitedStateCorrectionBase(CorrectionBase):
    """Corrections to EOMCC energies, from an upstream EOMCC or left-EOMCC method."""

    needs_left = True

    def __init__(self, upstream, variant=None, **options):
        if isinstance(upstream, LeftEOMCCBase):
            eom, left = upstream.eom, upstream
        elif isinstance(upstream, EOMCCBase):
            eom, left = upstream, (upstream.left if self.needs_left else None)
        else:
            raise TypeError(f"{type(self).__name__} needs an upstream EOMCC or left-EOMCC method.")
        super().__init__(upstream, variant, **options)
        self.eom, self.left, self.cc = eom, left, eom.cc
        self.options["RHF_symmetry"] = eom.options["RHF_symmetry"]
        self.ddeltas = None

    def build(self):
        self._warn_df_port()
        eom = self.eom.ensure_solved()
        left = self.left.ensure_solved() if self.left is not None else None
        hbar = eom.hbar
        results = [self._compute(i, eom, hbar, left) for i in range(eom.nroots)]
        self.deltas = [r[0] for r in results]
        self.ddeltas = [r[1] for r in results]
        self._finish_solve(True, *[x for x in (eom, hbar, left) if x is not None])
        return self

    def _compute(self, i, eom, hbar, left):
        out = self._function()(hbar.T, eom.R[i], left.L[i], eom.omega[i], hbar.correlation_energy, hbar.H,
                               self._fock(), self.system, self.options["RHF_symmetry"])
        return out[1], None

    def energy(self, root=None):
        """Corrected total energies of the roots (or of one root)."""
        self.ensure_solved()
        total = self.eom.energy() + np.array([d[self.variant] for d in self.deltas])
        return total if root is None else total[root]

    def raw_energy(self, root=None):
        self.ensure_solved()
        total = self.eom.raw_energy() + np.array([d[self.variant] for d in self.deltas])
        return total if root is None else total[root]

    def _analytic_gradient(self, root=0, zvector_tol=1.0e-10, perturbation=None, properties=None,
                           **kwargs):
        self.ensure_solved()
        eom, left, hbar = self.eom, self.left, self.eom.hbar
        return self._gradient_function()(hbar.T, eom.R[root], eom.r0[root], left.L[root], eom.omega[root], hbar.H,
                                         self.reference, delta=self.deltas[root][self.variant],
                                         zvector_tol=zvector_tol, perturbation=perturbation, properties=properties)


class CREOMCC23(ExcitedStateCorrectionBase):
    """delta-CR-EOMCC(2,3) triples corrections to EOMCCSD (DF/Cholesky).

    energy() and gradient() are those of delta-CR-EOMCC(2,3), E_EOMCCSD + ddeltas: the
    size-intensive correction without r0 times the ground-state moment. The full CR-EOMCC(2,3)
    corrections are kept in ``deltas`` and their energies are returned by full_energy() and
    full_raw_energy().
    """
    module, function = "ccpy.moments.creomcc23_chol", "calc_creomcc23"
    gradient_module, gradient_function = "ccpy.gradients.creomcc23_gradient", "calc_creomcc23_gradient"
    gradient_variants = ("A",)

    def _compute(self, i, eom, hbar, left):
        _, delta, ddelta = self._function()(hbar.T, eom.R[i], left.L[i], eom.r0[i], eom.omega[i],
                                            hbar.correlation_energy, hbar.H, self._fock(), self.system,
                                            self.options["RHF_symmetry"])
        return delta, ddelta

    def energy(self, root=None):
        """delta-CR-EOMCC(2,3) total energies of the roots (or of one root)."""
        self.ensure_solved()
        total = self.eom.energy() + np.array([d[self.variant] for d in self.ddeltas])
        return total if root is None else total[root]

    def raw_energy(self, root=None):
        self.ensure_solved()
        total = self.eom.raw_energy() + np.array([d[self.variant] for d in self.ddeltas])
        return total if root is None else total[root]

    def full_energy(self, root=None):
        """Full CR-EOMCC(2,3) total energies (with the non-intensive r0 term), E_EOMCCSD + deltas."""
        self.ensure_solved()
        total = self.eom.energy() + np.array([d[self.variant] for d in self.deltas])
        return total if root is None else total[root]

    def full_raw_energy(self, root=None):
        self.ensure_solved()
        total = self.eom.raw_energy() + np.array([d[self.variant] for d in self.deltas])
        return total if root is None else total[root]

    def _analytic_gradient(self, root=0, zvector_tol=1.0e-10, perturbation=None, properties=None,
                           **kwargs):
        """Gradient of delta-CR-EOMCC(2,3)_A, E_EOMCCSD + ddeltas[root]["A"]: the analytic gradient
        drops the non-size-intensive r0 (ground-state moment) part of the correction."""
        self.ensure_solved()
        eom, left, hbar = self.eom, self.left, self.eom.hbar
        return self._gradient_function()(hbar.T, eom.R[root], eom.r0[root], left.L[root], eom.omega[root], hbar.H,
                                         self.reference, delta=self.ddeltas[root][self.variant],
                                         zvector_tol=zvector_tol, perturbation=perturbation, properties=properties)


class EOMCCT3(ExcitedStateCorrectionBase):
    """CC(t;3) corrections to EOMCCSDt excitation energies."""
    module, function = "ccpy.moments.cct3", "calc_eomcct3"
    requires_df_port = True

    def __init__(self, upstream, num_active=1, two_body_approx=True, nacto=None, nactu=None, variant=None,
                 **options):
        super().__init__(upstream, variant, **options)
        self.num_active = num_active
        self.two_body_approx = two_body_approx
        self.active_space = (nacto, nactu)

    def _hamiltonian(self, hbar):
        if self.two_body_approx and isinstance(hbar.cc, CCPBase) and hbar.T.order > 2 and \
                isinstance(hbar, HBarCCSDT_P):
            from copy import deepcopy

            from ccpy.hbar.hbar_ccsdt_p import remove_VT3_intermediates

            return remove_VT3_intermediates(hbar.T, hbar.cc.t3_excitations, deepcopy(hbar.H))
        return hbar.H

    def _compute(self, i, eom, hbar, left):
        if self.active_space != (None, None):
            self.system.set_active_space(*(n or 0 for n in self.active_space))
        _, delta, ddelta = self._function()(hbar.T, eom.R[i], left.L[i], eom.r0[i], eom.omega[i],
                                            hbar.correlation_energy, self._hamiltonian(hbar), self._fock(),
                                            self.system, self.options["RHF_symmetry"], num_active=[self.num_active])
        return delta, ddelta


class EOMCCP3(ExcitedStateCorrectionBase):
    """Full CC(P;3) corrections to EOMCCSDT(P) energies."""
    module, function = "ccpy.moments.ccp3", "calc_eomccp3"
    requires_df_port = True

    def __init__(self, upstream, target_irrep=None, variant=None, **options):
        super().__init__(upstream, variant, **options)
        self.target_irrep = target_irrep

    def _compute(self, i, eom, hbar, left):
        _, delta = self._function()(hbar.T, eom.R[i], left.L[i], eom.cc.t3_excitations, eom.r3_excitations,
                                    eom.r0[i], eom.omega[i], hbar.correlation_energy, hbar.H, self._fock(),
                                    self.system, self.options["RHF_symmetry"], target_irrep=self.target_irrep)
        return delta, None


class PspaceExcitedCorrectionBase(ExcitedStateCorrectionBase):
    def _compute(self, i, eom, hbar, left):
        _, delta = self._function()(hbar.T, eom.R[i], left.L[i], eom.r3_excitations, eom.omega[i],
                                    hbar.correlation_energy, hbar.H, self._fock(), self.system,
                                    self.options["RHF_symmetry"])
        return delta, None


class EOMCCSDTA_star(ExcitedStateCorrectionBase):
    """EOMCCSD(T)(a)* corrections (upstream: EOMCCSDTA or its left method)."""
    module, function = "ccpy.moments.eomccsdta_star", "calc_eomccsdta_star"
    requires_df_port = True
    default_variant = "A"


class CRIPCC23(ExcitedStateCorrectionBase):
    """CR-IP-EOMCC(2,3) corrections to IP-EOMCCSD(2h-1p)."""
    module, function = "ccpy.moments.cripcc23", "calc_cripcc23"
    requires_df_port = True


class IPCCP3(PspaceExcitedCorrectionBase):
    """IP-CC(P;3) corrections to IP-EOMCC(P)."""
    module, function = "ccpy.moments.ipccp3", "calc_ipccp3_full"
    requires_df_port = True


class IPEOMCCSDTA_star(ExcitedStateCorrectionBase):
    module, function = "ccpy.moments.ipeomccsdta_star", "calc_ipeomccsdta_star"
    requires_df_port = True
    default_variant = "A"


class CREACC23(ExcitedStateCorrectionBase):
    """CR-EA-EOMCC(2,3) corrections to EA-EOMCCSD(2p-1h)."""
    module, function = "ccpy.moments.creacc23", "calc_creacc23"
    requires_df_port = True


class EACCP3(PspaceExcitedCorrectionBase):
    """EA-CC(P;3) corrections to EA-EOMCC(P)."""
    module, function = "ccpy.moments.eaccp3", "calc_eaccp3"
    requires_df_port = True


class EAEOMCCSDTA_star(ExcitedStateCorrectionBase):
    module, function = "ccpy.moments.eaeomccsdta_star", "calc_eaeomccsdta_star"
    requires_df_port = True
    default_variant = "A"


class DIPEOMCCSDTA_star(ExcitedStateCorrectionBase):
    """DIP-EOMCCSD(T)(a)* (4h-2p) corrections to DIP-EOMCC(3h-1p); no left vectors needed."""
    module, function = "ccpy.moments.dipeomccsdta_star", "calc_dipeomccsdta_star"
    requires_df_port = True
    default_variant = "A"
    needs_left = False

    def _compute(self, i, eom, hbar, left):
        _, delta = self._function()(hbar.T, eom.R[i], None, eom.omega[i], hbar.correlation_energy, hbar.H,
                                    self._fock(), self.system, self.options["RHF_symmetry"])
        return delta, None
