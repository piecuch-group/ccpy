"""Ground-state coupled-cluster methods.

    mol = Molecule(geometry, "cc-pvdz", auxiliary_basis="cc-pvdz-rifit")
    mf = RHF(mol)
    cc = CCSD(mf, nfrozen=1)
    e_ccsd = cc.energy()

``energy()`` solves the CC equations if needed, restarting from the current ``cc.T``.
"""
from functools import partial
from importlib import import_module

import numpy as np

from ccpy.methods.base import MethodBase
from ccpy.methods.hbar import HBarCC2, HBarCC3, HBarCCSD, HBarCCSDT, HBarCCSDT_P
from ccpy.models.integrals import Integral
from ccpy.models.operators import ClusterOperator
from ccpy.utilities.printing import cc_calculation_summary, get_timestamp


class CCBase(MethodBase):
    """Base class of ground-state CC methods (scheme: ground-state CC).

    Subclasses (levels) set the equation module and the excitation order, as well as
    the HBar and left-CC classes used by downstream methods.

    Attributes
    ----------
    T : ClusterOperator
        Cluster amplitudes (None before the first solve).
    correlation_energy : float
    converged : bool
    """

    module = None
    order = 2
    hbar_class = HBarCCSD
    left_class = None
    # Jacobi denominators the update of this level implements (option "denominator"; see
    # ccpy.cc.denominators): "mp" (orbital energies), "epstein-nesbet", "quasilinear", "newton"
    denominators = ("mp",)

    def __init__(self, upstream, nfrozen=None, ndelete=None, **options):
        super().__init__(upstream, nfrozen, ndelete, **options)
        if self.options["denominator"] not in self.denominators:
            raise ValueError(f"{type(self).__name__}: denominator must be one of {self.denominators}.")
        self.T = None
        self.correlation_energy = 0.0
        self._hbars = {}
        self._left = None

    # ------------------------------------------------------------------ accessors
    @property
    def cc(self):
        return self

    @property
    def hamiltonian(self):
        """Bare Hamiltonian the CC equations are solved with."""
        if self.requires_df_port:
            return self.reference.legacy_hamiltonian(copy_=False)
        return self.reference.H

    @property
    def total_energy(self):
        return self._complete_ground_energy(self.reference_energy + self.correlation_energy)

    def get_hbar(self, hbar_class=None):
        """HBar of this method's T (built, or rebuilt after T changed, on demand)."""
        cls = hbar_class or self.hbar_class
        if cls is None:
            raise NotImplementedError(f"No HBar is available for {type(self).__name__}.")
        if cls not in self._hbars:
            self._hbars[cls] = cls(self)
        return self._hbars[cls].ensure_built()

    @property
    def hbar(self):
        return self.get_hbar()

    @property
    def left(self):
        """Left-CC (Lambda) method of this level, created on first access (not solved)."""
        if self._left is None:
            if self.left_class is None:
                raise NotImplementedError(f"No left-CC method is available for {type(self).__name__}.")
            self._left = getattr(import_module("ccpy.methods.left"), self.left_class)(self, **self.user_options)
        return self._left

    # ------------------------------------------------------------------ solve
    def _new_operator(self):
        return ClusterOperator(self.system, order=self.order)

    def _initialize(self, T):
        """Initial amplitudes of a fresh calculation (zero by default)."""

    def _intermediates(self):
        X = Integral.from_empty(self.system, 2, use_none=True)
        if not self.requires_df_port:
            X.chol = Integral.from_empty(self.system, 1, use_none=True)
        return X

    def _run(self, update, dT, X):
        from ccpy.drivers.solvers import cc_jacobi

        return cc_jacobi(update, self.T, dT, self.hamiltonian, X, self.options)

    def solve(self):
        """Solve the CC equations, starting from the current T if there is one."""
        self._warn_df_port()
        self._print_header()
        print("   CC calculation started on", get_timestamp())
        if self.T is None:
            self.T = self._new_operator()
            self._initialize(self.T)
        update = getattr(import_module(self.module), "update")
        if self.options["denominator"] != "mp":
            update = partial(update, denominator=self.options["denominator"])
        self.T, self.correlation_energy, converged = self._run(update, self._new_operator(), self._intermediates())
        cc_calculation_summary(self.T, self.reference_energy, self.correlation_energy, self.system,
                               self.options["amp_print_threshold"])
        print("   CC calculation ended on", get_timestamp())
        self._finish_solve(converged)
        return self

    def energy(self):
        """Total CC energy (solves the equations if needed)."""
        self.ensure_solved()
        return self.total_energy

    def raw_energy(self):
        self.ensure_solved()
        return self.reference_energy + self.correlation_energy

    # ------------------------------------------------------------------ gradients
    def _analytic_gradient(self, zvector_tol=1.0e-10, **kwargs):
        left = self.left.ensure_solved()
        return self._gradient_function()(self.T, left.L, self.reference, zvector_tol=zvector_tol)

    def rdm1(self):
        return self.left.rdm1()

    def rdm2(self):
        return self.left.rdm2()


class CCSD(CCBase):
    """CCSD with DF/Cholesky integrals."""
    module = "ccpy.cc.ccsd_chol"
    order = 2
    denominators = ("mp", "epstein-nesbet", "quasilinear", "newton")
    left_class = "LeftCCSD"
    gradient_module, gradient_function = "ccpy.gradients.ccsd_gradient", "calc_ccsd_gradient"


class CCSDT(CCBase):
    """CCSDT. ccpy.cc.ccsdt_chol is an unfinished DF version (outdated Fortran
    interfaces, not installed), so the four-index equations are used for now."""
    module = "ccpy.cc.ccsdt"
    order = 3
    denominators = ("mp", "epstein-nesbet", "quasilinear", "newton")
    requires_df_port = True
    hbar_class = HBarCCSDT
    left_class = "LeftCCSDT"


class CCPBase(CCBase):
    """CC(P) methods: the highest excitation rank is restricted to a list of excitations
    (the P space), e.g., ``t3_excitations = {"aaa": ..., "aab": ..., "abb": ..., "bbb": ...}``
    with rows [a, b, c, i, j, k]."""

    pspace_orders = [3]

    def __init__(self, upstream, t3_excitations, nfrozen=None, ndelete=None, **options):
        super().__init__(upstream, nfrozen, ndelete, **options)
        self.t3_excitations = self._prepare_excitations(t3_excitations)

    def _prepare_excitations(self, excitations):
        excitations = {k: np.asfortranarray(np.array(v)) for k, v in excitations.items()}
        spins = {s: s for s in ("aaa", "aab", "abb", "bbb")}
        self._validate_working_excitations(excitations, spins, spins)
        if self.options["RHF_symmetry"]:
            assert excitations["aaa"].shape[0] == excitations["bbb"].shape[0]
            assert excitations["aab"].shape[0] == excitations["abb"].shape[0]
            excitations["bbb"] = excitations["aaa"].copy()
            # abb excitations must be ordered as a b~<c~ i j~<k~
            excitations["abb"] = np.asfortranarray(excitations["aab"][:, [2, 0, 1, 5, 3, 4]])
        return excitations

    def _pspace_sizes(self):
        return [[self.t3_excitations[s].shape[0] for s in ("aaa", "aab", "abb", "bbb")]]

    def set_pspace(self, t3_excitations):
        """Enlarge the P space. New excitations must be appended to the previous ones; the
        current amplitudes are kept and the next solve restarts from them."""
        self.t3_excitations = self._prepare_excitations(t3_excitations)
        if self.T is not None:
            self.T.extend_pspace_t3_operator(self._pspace_sizes()[0])
        self.converged = False

    def _new_operator(self):
        return ClusterOperator(self.system, order=self.order, p_orders=self.pspace_orders,
                               pspace_sizes=self._pspace_sizes())

    def _run(self, update, dT, X):
        from ccpy.drivers.solvers import cc_jacobi

        # t3_excitations is reordered in place along with T
        return cc_jacobi(update, self.T, dT, self.hamiltonian, X, self.options, self.t3_excitations)

    def _analytic_gradient(self, zvector_tol=1.0e-10, **kwargs):
        left = self.left.ensure_solved()
        return self._gradient_function()(self.T, left.L, self.t3_excitations, self.reference,
                                         zvector_tol=zvector_tol)


class CCSDT_P(CCPBase):
    """CCSDT(P) (CCSD plus the triples of the P space) with DF/Cholesky integrals. Its analytic
    gradient needs the CC(P) densities of ccpy.density.ccsdt_p_rdm."""
    module = "ccpy.cc.ccsdt_p_chol"
    denominators = ("mp", "epstein-nesbet", "quasilinear", "newton")
    order = 3
    hbar_class = HBarCCSDT_P
    left_class = "LeftCCSDT_P"
    gradient_module, gradient_function = "ccpy.gradients.ccsdt_p_gradient", "calc_ccsdt_p_gradient"

    def has_analytic_gradient(self, **kwargs):
        from importlib.util import find_spec

        return super().has_analytic_gradient(**kwargs) and find_spec("ccpy.density.ccsdt_p_rdm") is not None


# ---------------------------------------------------------------------- legacy levels
# The following methods still use four-index integrals and need DF ports.
class CCD(CCBase):
    module = "ccpy.cc.ccd"
    requires_df_port = True
    hbar_class = None


class CC2(CCBase):
    module = "ccpy.cc.cc2"
    requires_df_port = True
    hbar_class = HBarCC2


class CC3(CCBase):
    module = "ccpy.cc.cc3"
    order = 2
    requires_df_port = True
    hbar_class = HBarCC3
    left_class = "LeftCC3"


class CC4(CCBase):
    module = "ccpy.cc.cc4"
    order = 3
    requires_df_port = True
    hbar_class = None

    def solve(self):
        if not self.options["RHF_symmetry"]:
            raise NotImplementedError("CC4 is only implemented for closed-shell RHF references.")
        return super().solve()


class CCSDTQ(CCBase):
    module = "ccpy.cc.ccsdtq"
    order = 4
    requires_df_port = True
    hbar_class = None


class ACCBase(CCBase):
    """Approximate coupled-pair methods: acparray (5 floats) scales the five T2**2 diagrams."""

    def __init__(self, upstream, acparray, nfrozen=None, ndelete=None, **options):
        super().__init__(upstream, nfrozen, ndelete, **options)
        self.acparray = acparray

    def _run(self, update, dT, X):
        from ccpy.drivers.solvers import cc_jacobi

        return cc_jacobi(update, self.T, dT, self.hamiltonian, X, self.options, acparray=self.acparray)


class ACCD(ACCBase):
    module = "ccpy.cc.accd"
    requires_df_port = True
    hbar_class = None


class ACCSD(ACCBase):
    module = "ccpy.cc.accsd"
    requires_df_port = True
    hbar_class = None


class ACCSDT_P(CCPBase):
    module = "ccpy.cc.accsdt_p"
    order = 3
    requires_df_port = True
    hbar_class = HBarCCSDT_P
    left_class = "LeftCCSDT_P"

    def __init__(self, upstream, t3_excitations, acparray, nfrozen=None, ndelete=None, **options):
        super().__init__(upstream, t3_excitations, nfrozen, ndelete, **options)
        self.acparray = acparray

    def _run(self, update, dT, X):
        from ccpy.drivers.solvers import cc_jacobi

        return cc_jacobi(update, self.T, dT, self.hamiltonian, X, self.options, self.t3_excitations,
                         acparray=self.acparray)


class ECCC2(CCBase):
    """Externally corrected CCSD: T3 and T4 from the cluster analysis of a CI wave
    function (``ci_vectors_file``) are held fixed."""
    module = "ccpy.cc.eccc2"
    requires_df_port = True

    def __init__(self, upstream, ci_vectors_file, nfrozen=None, ndelete=None, **options):
        super().__init__(upstream, nfrozen, ndelete, **options)
        self.ci_vectors_file = ci_vectors_file
        self.T_ext = self.VT_ext = None

    def _external(self):
        if self.T_ext is None:
            from ccpy.extcorr.external_correction import cluster_analysis

            self.T_ext, self.VT_ext = cluster_analysis(self.ci_vectors_file, self.hamiltonian, self.system)
        return self.T_ext, self.VT_ext

    def _initialize(self, T):
        T_ext, _ = self._external()
        for s in ("a", "b", "aa", "ab", "bb"):
            setattr(T, s, getattr(T_ext, s))

    def _run(self, update, dT, X):
        from ccpy.drivers.solvers import eccc_jacobi

        T_ext, VT_ext = self._external()
        return eccc_jacobi(update, self.T, dT, self.hamiltonian, X, T_ext, VT_ext, self.options)
