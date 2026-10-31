"""Linear-response CC for first-order properties of one-body operators."""
from importlib import import_module

from ccpy.methods.base import MethodBase
from ccpy.methods.cc import CCSD
from ccpy.methods.eomcc import resolve_cc
from ccpy.methods.hbar import HBarCCSDLegacy
from ccpy.models.operators import ClusterOperator
from ccpy.utilities.printing import get_timestamp


class LRCCBase(MethodBase):
    """Base class of LR-CC methods (scheme: linear response).

    Parameters
    ----------
    upstream : SCF object or a method holding the ground-state CC
    prop : one-body property operator (ccpy.models.integrals.Integral, order 1) in the
           correlated MO basis, e.g., from ccpy.interfaces.scf_tools.get_multipole_integral

    Attributes
    ----------
    T1 : first-order (perturbed) amplitudes, value : correlation contribution to the property
    """

    module = None
    order = 2
    cc_class = CCSD
    hbar_class = HBarCCSDLegacy

    def __init__(self, upstream, prop, nfrozen=None, ndelete=None, **options):
        super().__init__(upstream, nfrozen, ndelete, **options)
        self.cc = resolve_cc(self, upstream, self.cc_class)
        self.prop = prop
        self.T1 = None
        self.value = 0.0

    @property
    def hbar(self):
        return self.cc.get_hbar(self.hbar_class)

    def solve(self):
        from ccpy.drivers.solvers import lrcc_jacobi
        from ccpy.models.integrals import Integral
        from ccpy.utilities.printing import lrcc_calculation_summary

        self._warn_df_port()
        hbar = self.hbar
        self._print_header()
        print("   LR-CC calculation started on", get_timestamp())
        update = getattr(import_module(self.module), "update")
        self.T1 = ClusterOperator(self.system, order=self.order)
        dT = ClusterOperator(self.system, order=self.order)
        X = Integral.from_empty(self.system, 2, use_none=True)
        self.T1, self.value, converged = lrcc_jacobi(update, self.T1, self.prop, dT, hbar.T, hbar.H, X,
                                                     self.system, self.options)
        lrcc_calculation_summary(self.T1, self.value, self.system, self.options["amp_print_threshold"])
        print("   LR-CC calculation ended on", get_timestamp())
        self._finish_solve(converged, self.cc, hbar)
        return self

    def property(self):
        """Correlation contribution to the expectation value of `prop`."""
        self.ensure_solved()
        return self.value

    def energy(self):
        """Ground-state CC energy."""
        return self.cc.energy()


class LRCCSD(LRCCBase):
    module = "ccpy.lrcc.lrccsd"
    requires_df_port = True
