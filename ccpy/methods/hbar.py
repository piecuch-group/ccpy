"""Similarity-transformed Hamiltonians HBar = (H e^T)_C of a converged ground-state CC.

The DF HBar keeps the T1-transformed DF vectors in ``H.chol`` and the one- and two-body
blocks of HBar without vvvv; the bare Hamiltonian of the Reference is never modified.
"""
from importlib import import_module

from ccpy.methods.reference import warn_df_port
from ccpy.utilities.printing import get_timestamp


class HBarBase:
    """HBar of the ground-state CC method ``cc`` (built from cc.T on ``build()``).

    Attributes
    ----------
    H : HBar (DFHamiltonian for DF methods, four-index Integral for legacy ones)
    T : cluster operator HBar was built with (normally cc.T)
    """

    module = None
    function = None
    requires_df_port = False

    def __init__(self, cc):
        self.cc = cc
        self.H = None
        self.T = None
        self.version = 0
        self._cc_version = None

    @property
    def reference(self):
        return self.cc.reference

    @property
    def chol(self):
        """T1-transformed DF vectors (DF HBar only)."""
        return self.H.chol

    @property
    def correlation_energy(self):
        return self.cc.correlation_energy

    def is_current(self):
        return self.H is not None and self._cc_version == self.cc.version

    def build(self):
        self.cc.ensure_solved()
        if self.requires_df_port:
            warn_df_port(type(self).__name__)
        function = getattr(import_module(self.module), self.function)
        print(f"\n   {type(self).__name__} construction began on", get_timestamp())
        self._build(function)
        print("   HBar construction completed on", get_timestamp(), "\n")
        self.version += 1
        self._cc_version = self.cc.version
        return self

    def ensure_built(self):
        if not self.is_current():
            self.build()
        return self

    def _bare(self):
        if self.requires_df_port:
            return self.reference.legacy_hamiltonian()
        return self.reference.H.copy()

    def _build(self, function):
        self.T = self.cc.T
        self.H = function(self.T, self._bare(), self.cc.options["RHF_symmetry"], self.cc.system, *self._extra_args())

    def _extra_args(self):
        return ()


class HBarCCSD(HBarBase):
    """CCSD HBar from DF vectors (also used with T1 and T2 of higher-order methods)."""
    module, function = "ccpy.hbar.hbar_ccsd_chol", "build_hbar_ccsd_chol"


class HBarCCSDLegacy(HBarBase):
    module, function = "ccpy.hbar.hbar_ccsd", "build_hbar_ccsd"
    requires_df_port = True


class HBarCC2(HBarBase):
    module, function = "ccpy.hbar.hbar_cc2", "build_hbar_cc2"
    requires_df_port = True


class HBarCCSDT(HBarBase):
    module, function = "ccpy.hbar.hbar_ccsdt", "build_hbar_ccsdt"
    requires_df_port = True


class HBarCCSDT1(HBarBase):
    module, function = "ccpy.hbar.hbar_ccsdt1", "build_hbar_ccsdt1"
    requires_df_port = True


class HBarCCSDT_P(HBarBase):
    """HBar of CCSDT(P), including the V*T3 parts of the vooo and vvov blocks."""
    module, function = "ccpy.hbar.hbar_ccsdt_p", "build_hbar_ccsdt_p"
    requires_df_port = True

    def _extra_args(self):
        return (self.cc.t3_excitations,)


class HBarCC3(HBarBase):
    """CC3 HBar; also keeps the CCS-transformed intermediates used by EOM-CC3."""
    module, function = "ccpy.hbar.hbar_cc3", "build_hbar_cc3"
    requires_df_port = True

    def _build(self, function):
        self.T = self.cc.T
        self.H, self.cc3_intermediates = function(self.T, self._bare(), self.cc.options["RHF_symmetry"], self.cc.system)


class HBarCCSDTA(HBarBase):
    """CCSD HBar augmented with the CCSD(T)(a) triples; this also produces new T (with T3)
    and the CCSD(T)(a) correlation energy, kept on this object."""
    module, function = "ccpy.hbar.hbar_ccsdta", "build_hbar_ccsdta"
    requires_df_port = True

    @property
    def correlation_energy(self):
        return self._correlation_energy

    def _build(self, function):
        from copy import deepcopy

        self.H, self.T, self._correlation_energy = function(deepcopy(self.cc.T), self._bare(),
                                                            self.cc.options["RHF_symmetry"], self.cc.system)
