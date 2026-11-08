import copy
from importlib import import_module

import numpy as np

from ccpy.methods.base import MethodBase
from ccpy.methods.cc import CCPBase, CCSD, CCSDT, CCSDT_P
from ccpy.methods.eomcc import EOMCCBase
from ccpy.methods.hbar import HBarCCSD, HBarCCSDLegacy, HBarCCSDT, HBarCCSDT_P
from ccpy.methods.left import LeftCCSD, LeftCCSDT, LeftCCSDT_P, LeftEOMCCBase
from ccpy.methods.corrections import CCP3

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

