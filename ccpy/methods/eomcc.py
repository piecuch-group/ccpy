"""Equation-of-motion CC methods (EE, SF, IP, EA, DIP, DEA) and their initial guesses.

    eom = EOMCCSD(mf, nfrozen=1)
    eom.initial_guess(method="cisd", multiplicity=1, nacto=3, nactu=3, roots_per_irrep={"A1": 2})
    energies = eom.energy()          # total energies of all roots

The ground-state CC, its HBar, and the guess are computed on demand.
"""
from importlib import import_module

import numpy as np

from ccpy.methods.base import MethodBase
from ccpy.methods.cc import CC2, CC3, CCSD, CCSDT, CCSDT_P
from ccpy.methods.hbar import HBarCC2, HBarCC3, HBarCCSD, HBarCCSDLegacy, HBarCCSDT, HBarCCSDTA, HBarCCSDT_P
from ccpy.models.operators import ClusterOperator, FockOperator, SpinFlipOperator
from ccpy.utilities.printing import get_timestamp

# guess methods that generate singles (order 1) or singles and doubles (order 2) only
GUESS_ORDER = {"cis": 1, "sfcis": 1, "eacis": 1, "ipcis": 1,
               "cisd": 2, "cisd_chol": 2, "deacis": 2, "dipcis": 2, "eacisd": 2, "ipcisd": 2}


def resolve_cc(method, upstream, cc_class):
    """Ground-state CC object held by `upstream`, or a new `cc_class` on method's reference."""
    if isinstance(upstream, MethodBase) and hasattr(upstream, "cc"):
        return upstream.cc
    return cc_class(method.reference, **method.user_options)


class EOMCCBase(MethodBase):
    """Base class of EOMCC methods (scheme: EOMCC); levels set the equation module, the
    operator type, the ground-state CC/HBar they need, and the default guess.

    Parameters
    ----------
    upstream : SCF object or a method holding a ground-state CC (``.cc``)
    roots : list of int, optional
        Guess vectors (0-based, in the order returned by the guess) to converge;
        all of them by default.
    hbar_class : HBar class, optional
        Overrides the HBar of the level (e.g., HBarCCSDTA for the CCSD(T)(a)* methods).

    Attributes
    ----------
    R : list of right eigenvectors, omega : excitation (ionization, ...) energies,
    root_converged : convergence flag of each root.
    """

    module = None
    cc_class = CCSD
    hbar_class = HBarCCSD
    default_guess = "cisd"
    left_class = None
    pspace_orders = [None]

    def __init__(self, upstream, roots=None, hbar_class=None, nfrozen=None, ndelete=None, **options):
        super().__init__(upstream, nfrozen, ndelete, **options)
        if hbar_class is not None:
            self.hbar_class = hbar_class
        self.cc = self._resolve_cc(upstream)
        self.roots = None if roots is None else list(roots)
        self.R = None
        self.omega = None
        self.root_converged = None
        self.guess_settings = None
        self.guess_energy = self.guess_vectors = None
        self._guess_hbar_version = None
        self._left = None

    def _resolve_cc(self, upstream):
        return resolve_cc(self, upstream, self.cc_class)

    # ------------------------------------------------------------------ accessors
    @property
    def hbar(self):
        return self.cc.get_hbar(self.hbar_class)

    @property
    def left(self):
        if self._left is None:
            if self.left_class is None:
                raise NotImplementedError(f"No left-EOMCC method is available for {type(self).__name__}.")
            self._left = getattr(import_module("ccpy.methods.left"), self.left_class)(self, **self.user_options)
        return self._left

    @property
    def nroots(self):
        return len(self.R) if self.R is not None else 0

    # ------------------------------------------------------------------ guess
    def initial_guess(self, method=None, multiplicity=1, nacto=0, nactu=0, roots_per_irrep=None,
                      use_symmetry=True, debug=False):
        """Set up the initial guess (a CI-like diagonalization of HBar in a small space),
        run when the EOM equations are first solved.

        Parameters
        ----------
        method : str, guess method (e.g., "cis", "cisd", "ipcisd", "eacisd", "sfcis")
        multiplicity : int, spin multiplicity of the targeted states (-1 for IP/EA/DIP/DEA)
        nacto, nactu : int, active occupied/unoccupied orbitals of the guess space
        roots_per_irrep : dict, number of roots per irreducible representation
        """
        method = (method or self.default_guess).lower()
        if not self.requires_df_port and method == "cisd":
            method = "cisd_chol"
        if method not in GUESS_ORDER:
            raise NotImplementedError(f"Guess {method} is not available.")
        if roots_per_irrep is None:
            raise ValueError("initial_guess needs roots_per_irrep, e.g., {'A1': 2}.")
        self.guess_settings = dict(method=method, multiplicity=multiplicity, nacto=nacto, nactu=nactu,
                                   roots_per_irrep=dict(roots_per_irrep), use_symmetry=use_symmetry, debug=debug)
        # spin-adapted closed-shell equations only apply to singlets
        if multiplicity != 1:
            self.options["RHF_symmetry"] = False
        self.R = None
        self.converged = False
        return self

    def _copy_setup(self, new):
        if self.guess_settings is not None:
            new.initial_guess(**self.guess_settings)
        new.options["RHF_symmetry"] = self.options["RHF_symmetry"]

    def run_guess(self):
        if self.guess_settings is None:
            raise RuntimeError(f"Call {type(self).__name__}.initial_guess(...) before solving.")
        hbar = self.hbar
        s = self.guess_settings
        function = getattr(import_module("ccpy.eom_guess." + s["method"]), "run_diagonalization")
        args = (self.system, hbar.H) + ((hbar.T,) if s["method"] == "cisd_chol" else ())
        self.guess_energy, self.guess_vectors = function(*args, s["multiplicity"], s["roots_per_irrep"],
                                                         s["nacto"], s["nactu"], debug=s["debug"],
                                                         use_symmetry=s["use_symmetry"])
        self._guess_hbar_version = (hbar, hbar.version)
        return self.guess_energy

    # ------------------------------------------------------------------ operators
    def _new_operator(self):
        raise NotImplementedError

    def _extra_args(self):
        """(t3_excitations, r3_excitations) for P-space methods, () otherwise."""
        return ()

    def _postprocess(self, i):
        """r0, REL, and the summary printout of root i."""

    def _print_amplitudes(self, R):
        pass

    # ------------------------------------------------------------------ solve
    def solve(self):
        self._warn_df_port()
        hbar = self.hbar
        if self.R is None or self._guess_hbar_version != (hbar, hbar.version):
            self.run_guess()
            roots = self.roots if self.roots is not None else list(range(len(self.guess_energy)))
            order = GUESS_ORDER[self.guess_settings["method"]]
            self.R = []
            for k in roots:
                R = self._new_operator()
                R.unflatten(self.guess_vectors[:, k], order=order)
                self.R.append(R)
            self.omega = np.array([self.guess_energy[k] for k in roots], dtype=float)
            self.root_converged = [False] * len(roots)
            self.r0 = np.zeros(len(roots))
            self.rel = np.zeros(len(roots))
        self._print_header()
        HR = getattr(import_module(self.module), "HR")
        update = getattr(import_module(self.module), "update")
        dR = self._new_operator()
        if self.options["davidson_solver"] == "multiroot":
            self._solve_multiroot(HR, update, dR, hbar)
        else:
            self._solve_sequential(HR, update, dR, hbar)
        self._finish_solve(all(self.root_converged), self.cc, hbar)
        return self

    def _solve_sequential(self, HR, update, dR, hbar):
        from ccpy.drivers.solvers import eomcc_davidson

        previous = []
        for i, R in enumerate(self.R):
            print(f"   {type(self).__name__} calculation for root {i} started on", get_timestamp())
            print("\n   Energy of initial guess = {:>10.10f}".format(self.omega[i]))
            self._print_amplitudes(R)
            r = R.flatten() / np.linalg.norm(R.flatten())
            B = np.column_stack([r] + ([] if self._extra_args() else previous))
            self.R[i], self.omega[i], self.root_converged[i] = eomcc_davidson(
                HR, update, B, R, dR, self.omega[i], hbar.T, hbar.H, self.system, self.options, *self._extra_args())
            previous.append(self.R[i].flatten() / np.linalg.norm(self.R[i].flatten()))
            self._postprocess(i)
            print(f"   {type(self).__name__} calculation for root {i} ended on", get_timestamp(), "\n")

    def _solve_multiroot(self, HR, update, dR, hbar):
        from ccpy.drivers.solvers import eomcc_block_davidson

        index = list(range(len(self.R)))
        B0, _ = np.linalg.qr(np.asarray([R.flatten() for R in self.R]).T)
        extra = dict(zip(("t3_excitations", "r3_excitations"), self._extra_args()))
        self.R, self.omega, self.root_converged = eomcc_block_davidson(
            HR, update, B0, self.R, dR, self.omega, hbar.T, hbar.H, self.system, index, self.options, **extra)
        for i in index:
            self._postprocess(i)

    # ------------------------------------------------------------------ energies
    def energy(self, root=None):
        """Total energies of the roots (or of one root), solving the equations if needed."""
        self.ensure_solved()
        total = self.reference_energy + self.hbar.correlation_energy + self.omega
        return total if root is None else total[root]



class PspaceEOMMixin:
    """EOMCC(P): the highest excitation rank of R is restricted to ``r3_excitations``.
    The ground-state CC(P) (``t3_excitations``) is taken from the upstream method when it
    is one, otherwise it is created from ``t3_excitations``."""

    r3_spincases = ("aaa", "aab", "abb")

    def _resolve_cc(self, upstream):
        if isinstance(upstream, MethodBase) and hasattr(upstream, "cc"):
            return upstream.cc
        return None

    def _setup_pspace(self, upstream, r3_excitations, t3_excitations):
        from ccpy.methods.cc import CCPBase

        if self.cc is None:
            if issubclass(self.cc_class, CCPBase):
                if t3_excitations is None:
                    raise ValueError(f"{type(self).__name__} needs t3_excitations or an upstream CC(P) method.")
                self.cc = self.cc_class(self.reference, t3_excitations, **self.user_options)
            else:
                self.cc = self.cc_class(self.reference, **self.user_options)
        self.r3_excitations = self._prepare_r3(r3_excitations)

    def set_pspace(self, r3_excitations):
        """Change the P space of R; the roots are recomputed from the initial guess."""
        self.r3_excitations = self._prepare_r3(r3_excitations)
        self.R = None
        self.converged = False

    def _prepare_r3(self, excitations):
        return {k: np.asfortranarray(np.array(v)) for k, v in excitations.items()}

    def _pspace_sizes(self):
        return [[self.r3_excitations[s].shape[0] for s in self.r3_spincases]]

    def _extra_args(self):
        return (getattr(self.cc, "t3_excitations", None), self.r3_excitations)


# ---------------------------------------------------------------------- EE-EOMCC
class EEEOMCCBase(EOMCCBase):
    order = 2

    def _new_operator(self):
        return ClusterOperator(self.system, order=self.order)

    def _print_amplitudes(self, R):
        from ccpy.utilities.printing import print_ee_amplitudes

        print_ee_amplitudes(R, self.system, R.order, self.options["amp_print_threshold"])

    def _postprocess(self, i):
        from ccpy.energy.cc_energy import get_r0, get_rel
        from ccpy.utilities.printing import eomcc_calculation_summary

        self.r0[i] = get_r0(self.R[i], self.hbar.H, self.omega[i])
        self.rel[i] = get_rel(self.R[i], self.r0[i])
        eomcc_calculation_summary(self.R[i], self.omega[i], self.hbar.correlation_energy, self.r0[i], self.rel[i],
                                  self.root_converged[i], i, self.system, self.options["amp_print_threshold"])


class EOMCCSD(EEEOMCCBase):
    """EOMCCSD with DF/Cholesky integrals."""
    module = "ccpy.eomcc.eomccsd_chol"
    left_class = "LeftEOMCCSD"


class EOMCCSDTA(EEEOMCCBase):
    """EOMCCSD equations on the CCSD(T)(a) HBar (right states of EOMCCSD(T)(a)*)."""
    module = "ccpy.eomcc.eomccsd"
    requires_df_port = True
    hbar_class = HBarCCSDTA
    left_class = "LeftEOMCCSDTA"


class EOMCC2(EEEOMCCBase):
    module = "ccpy.eomcc.eomcc2"
    requires_df_port = True
    cc_class = CC2
    hbar_class = HBarCC2


class EOMCCSDT(EEEOMCCBase):
    module = "ccpy.eomcc.eomccsdt"
    order = 3
    requires_df_port = True
    cc_class = CCSDT
    hbar_class = HBarCCSDT
    left_class = "LeftEOMCCSDT"


class EOMCC3(EEEOMCCBase):
    """EOM-CC3, solved with the nonlinear (omega-dependent) DIIS solver."""
    module = "ccpy.eomcc.eomcc3"
    requires_df_port = True
    cc_class = CC3
    hbar_class = HBarCC3

    def _solve_multiroot(self, HR, update, dR, hbar):
        self._solve_sequential(HR, update, dR, hbar)

    def _solve_sequential(self, HR, update, dR, hbar):
        from ccpy.drivers.solvers import eomcc_nonlinear_diis

        fock = self.reference.legacy_hamiltonian(copy_=False)
        for i, R in enumerate(self.R):
            print(f"   EOM-CC3 calculation for root {i} started on", get_timestamp())
            self._print_amplitudes(R)
            self.R[i], self.omega[i], self.root_converged[i] = eomcc_nonlinear_diis(
                HR, update, R.flatten() / np.linalg.norm(R.flatten()), R, dR, self.omega[i], hbar.T, hbar.H,
                hbar.cc3_intermediates, fock, self.system, self.options)
            self._postprocess(i)


class EOMCCSDT_P(PspaceEOMMixin, EEEOMCCBase):
    """EOMCCSDT(P) on top of CCSDT(P); triples of R restricted to ``r3_excitations``."""
    module = "ccpy.eomcc.eomccsdt_p"
    order = 3
    requires_df_port = True
    cc_class = CCSDT_P
    hbar_class = HBarCCSDT_P
    left_class = "LeftEOMCCSDT_P"
    r3_spincases = ("aaa", "aab", "abb", "bbb")

    def __init__(self, upstream, r3_excitations, t3_excitations=None, roots=None, hbar_class=None, nfrozen=None,
                 ndelete=None, **options):
        super().__init__(upstream, roots, hbar_class, nfrozen, ndelete, **options)
        self._setup_pspace(upstream, r3_excitations, t3_excitations)

    def _prepare_r3(self, excitations):
        excitations = super()._prepare_r3(excitations)
        if self.options["RHF_symmetry"]:
            excitations["bbb"] = excitations["aaa"].copy()
            excitations["abb"] = np.asfortranarray(excitations["aab"][:, [2, 0, 1, 5, 3, 4]])
        return excitations

    def _new_operator(self):
        return ClusterOperator(self.system, order=3, p_orders=[3], pspace_sizes=self._pspace_sizes())


# ---------------------------------------------------------------------- SF-EOMCC
class SFEOMCCBase(EOMCCBase):
    requires_df_port = True
    hbar_class = HBarCCSDLegacy
    default_guess = "sfcis"
    order = 2

    def _new_operator(self):
        return SpinFlipOperator(self.system, Ms=-1, order=self.order)

    def _print_amplitudes(self, R):
        from ccpy.utilities.printing import print_sf_amplitudes

        print_sf_amplitudes(R, self.system, R.order, self.options["amp_print_threshold"])

    def _postprocess(self, i):
        from ccpy.utilities.printing import sfeomcc_calculation_summary

        sfeomcc_calculation_summary(self.R[i], self.omega[i], self.hbar.correlation_energy, self.root_converged[i],
                                    self.system, self.options["amp_print_threshold"])


class SFEOMCCSD(SFEOMCCBase):
    module = "ccpy.eomcc.sfeomccsd"


class SFEOMCC23(SFEOMCCBase):
    module = "ccpy.eomcc.sfeomcc23"
    order = 3


# ---------------------------------------------------------------------- particle-nonconserving
class FockEOMCCBase(EOMCCBase):
    """EOMCC with particle-nonconserving R (IP, EA, DIP, DEA)."""
    requires_df_port = True
    hbar_class = HBarCCSDLegacy
    num_particles = num_holes = None
    printer = summary = None

    def _new_operator(self):
        return FockOperator(self.system, self.num_particles, self.num_holes)

    def _print_amplitudes(self, R):
        import ccpy.utilities.printing as printing

        getattr(printing, self.printer)(R, self.system, R.order, self.options["amp_print_threshold"])

    def _postprocess(self, i):
        import ccpy.utilities.printing as printing

        getattr(printing, self.summary)(self.R[i], self.omega[i], self.hbar.correlation_energy, self.root_converged[i],
                                        self.system, self.options["amp_print_threshold"])


class PspaceFockEOMMixin(PspaceEOMMixin):
    def __init__(self, upstream, r3_excitations, t3_excitations=None, roots=None, hbar_class=None, nfrozen=None,
                 ndelete=None, **options):
        super().__init__(upstream, roots, hbar_class, nfrozen, ndelete, **options)
        self._setup_pspace(upstream, r3_excitations, t3_excitations)

    def _new_operator(self):
        return FockOperator(self.system, self.num_particles, self.num_holes, p_orders=[2],
                            pspace_sizes=self._pspace_sizes())


class IPEOMCCBase(FockEOMCCBase):
    default_guess = "ipcisd"
    printer = "print_ip_amplitudes"

    def _postprocess(self, i):
        from ccpy.energy.cc_energy import get_rel_ip
        from ccpy.utilities.printing import ipeomcc_calculation_summary

        self.rel[i] = get_rel_ip(self.R[i])
        ipeomcc_calculation_summary(self.R[i], self.omega[i], self.hbar.correlation_energy, self.rel[i],
                                    self.root_converged[i], self.system, self.options["amp_print_threshold"])


class IPEOM2(IPEOMCCBase):
    module, num_particles, num_holes, left_class = "ccpy.eomcc.ipeom2", 1, 2, "LeftIPEOM2"


class IPEOM3(IPEOMCCBase):
    module, num_particles, num_holes, left_class = "ccpy.eomcc.ipeom3", 2, 3, "LeftIPEOM3"


class IPEOMCCSDT(IPEOMCCBase):
    module, num_particles, num_holes = "ccpy.eomcc.ipeomccsdt", 2, 3
    cc_class, hbar_class = CCSDT, HBarCCSDT


class IPEOMCCSDTA(IPEOMCCBase):
    module, num_particles, num_holes = "ccpy.eomcc.ipeomccsdta", 2, 3
    hbar_class = HBarCCSDTA


class IPEOM3_P(PspaceFockEOMMixin, IPEOMCCBase):
    module, num_particles, num_holes, left_class = "ccpy.eomcc.ipeom3_p", 2, 3, "LeftIPEOM3_P"


class IPEOMCCSDT_P(PspaceFockEOMMixin, IPEOMCCBase):
    module, num_particles, num_holes = "ccpy.eomcc.ipeomccsdt_p", 2, 3
    cc_class, hbar_class = CCSDT_P, HBarCCSDT_P


class EAEOMCCBase(FockEOMCCBase):
    default_guess = "eacisd"
    printer = "print_ea_amplitudes"

    def _postprocess(self, i):
        from ccpy.energy.cc_energy import get_rel_ea
        from ccpy.utilities.printing import eaeomcc_calculation_summary

        self.rel[i] = get_rel_ea(self.R[i])
        eaeomcc_calculation_summary(self.R[i], self.omega[i], self.hbar.correlation_energy, self.rel[i],
                                    self.root_converged[i], self.system, self.options["amp_print_threshold"])


class EAEOM2(EAEOMCCBase):
    module, num_particles, num_holes, left_class = "ccpy.eomcc.eaeom2", 2, 1, "LeftEAEOM2"


class EAEOM3(EAEOMCCBase):
    module, num_particles, num_holes, left_class = "ccpy.eomcc.eaeom3", 3, 2, "LeftEAEOM3"


class EAEOMCCSDT(EAEOMCCBase):
    module, num_particles, num_holes = "ccpy.eomcc.eaeomccsdt", 3, 2
    cc_class, hbar_class = CCSDT, HBarCCSDT


class EAEOM3_P(PspaceFockEOMMixin, EAEOMCCBase):
    module, num_particles, num_holes, left_class = "ccpy.eomcc.eaeom3_p", 3, 2, "LeftEAEOM3_P"


class DIPEOMCCBase(FockEOMCCBase):
    default_guess = "dipcis"
    printer, summary = "print_dip_amplitudes", "dipeomcc_calculation_summary"


class DIPEOM3(DIPEOMCCBase):
    module, num_particles, num_holes = "ccpy.eomcc.dipeom3", 1, 3


class DIPEOM4(DIPEOMCCBase):
    module, num_particles, num_holes = "ccpy.eomcc.dipeom4", 2, 4


class DIPEOMCCSDT(DIPEOMCCBase):
    module, num_particles, num_holes = "ccpy.eomcc.dipeomccsdt", 2, 4
    cc_class, hbar_class = CCSDT, HBarCCSDT


class DIPEOMCCSDTA(DIPEOMCCBase):
    module, num_particles, num_holes = "ccpy.eomcc.dipeomccsdta", 2, 4
    hbar_class = HBarCCSDTA


class DEAEOMCCBase(FockEOMCCBase):
    default_guess = "deacis"
    printer, summary = "print_dea_amplitudes", "deaeomcc_calculation_summary"


class DEAEOM2(DEAEOMCCBase):
    module, num_particles, num_holes = "ccpy.eomcc.deaeom2", 2, 0


class DEAEOM3(DEAEOMCCBase):
    module, num_particles, num_holes = "ccpy.eomcc.deaeom3", 3, 1


class DEAEOM4(DEAEOMCCBase):
    module, num_particles, num_holes = "ccpy.eomcc.deaeom4", 4, 2


class DEAEOM4_P(PspaceFockEOMMixin, DEAEOMCCBase):
    module, num_particles, num_holes = "ccpy.eomcc.deaeom4_p", 4, 2
    r3_spincases = ("abaa", "abab", "abbb")
