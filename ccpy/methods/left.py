"""Left-CC (Lambda) and left-EOMCC methods.

    left = LeftCCSD(mf, nfrozen=1)       # or LeftCCSD(cc), or cc.left
    left.solve()
    rdm1, rdm2 = left.rdm1(), left.rdm2()
"""
from importlib import import_module

import numpy as np

from ccpy.methods.base import MethodBase
from ccpy.methods.cc import CC3, CCSD, CCSDT, CCSDT_P
from ccpy.methods.eomcc import EOMCCBase, resolve_cc
from ccpy.methods.hbar import HBarCC3, HBarCCSD, HBarCCSDT, HBarCCSDT_P
from ccpy.models.operators import ClusterOperator
from ccpy.utilities.printing import get_timestamp


class LeftCCBase(MethodBase):
    """Base class of ground-state left-CC methods (scheme: left-CC).

    Attributes
    ----------
    L : ClusterOperator, the de-excitation amplitudes (Lambda)
    """

    module = None
    order = 2
    cc_class = CCSD
    hbar_class = HBarCCSD

    def __init__(self, upstream, nfrozen=None, ndelete=None, **options):
        super().__init__(upstream, nfrozen, ndelete, **options)
        self.cc = resolve_cc(self, upstream, self.cc_class)
        self.L = None
        self._rdm = None

    @property
    def hbar(self):
        return self.cc.get_hbar(self.hbar_class)

    def _new_operator(self):
        return ClusterOperator(self.system, order=self.order)

    def _extra_args(self):
        return ()

    def _align(self, extra):
        """Bring L into the order of the excitations of T after the solve (P-space methods)."""

    def solve(self):
        from ccpy.drivers.solvers import left_cc_jacobi
        from ccpy.utilities.printing import leftcc_calculation_summary

        self._warn_df_port()
        hbar = self.hbar
        self._print_header()
        print("   Left CC calculation started on", get_timestamp())
        if self.L is None:
            self.L = self._new_operator()
            self.L.unflatten(hbar.T.flatten()[:self.L.ndim])
        LH = self._new_operator()
        update = getattr(import_module(self.module), "update")
        extra = self._extra_args()
        self.L, _, _, converged = left_cc_jacobi(update, self.L, LH, hbar.T, hbar.H, None, 0.0, True,
                                                 self.system, self.options, *extra)
        self._align(extra)
        leftcc_calculation_summary(self.L, 0.0, 0.0, converged, self.system, self.options["amp_print_threshold"])
        print("   Left CC calculation ended on", get_timestamp(), "\n")
        self._rdm = None
        self._finish_solve(converged, self.cc, hbar)
        return self

    def energy(self):
        """Ground-state CC energy (the left-CC equations are solved if needed)."""
        self.ensure_solved()
        return self.cc.energy()

    def has_analytic_gradient(self, **kwargs):
        return self.cc.has_analytic_gradient(**kwargs)

    def _analytic_gradient(self, **kwargs):
        return self.cc._analytic_gradient(**kwargs)


class LeftCCSD(LeftCCBase):
    """Left-CCSD with DF/Cholesky integrals."""
    module = "ccpy.left.left_ccsd_chol"

    def _densities(self):
        from ccpy.density.ccsd_rdm import calc_ccsd_rdm

        self.ensure_solved()
        if self._rdm is None:
            self._rdm = calc_ccsd_rdm(self.cc.T, self.L, self.system)
        return self._rdm

    def rdm1(self):
        """Unrelaxed one-body CCSD density in the T1-transformed frame, conjugate to the
        Fock blocks of H(T1) (ccpy.density.ccsd_rdm)."""
        return self._densities()[0]

    def rdm2(self):
        """Unrelaxed two-body CCSD density in the T1-transformed frame, conjugate to the
        two-body blocks of H(T1); vvvv blocks are given by rdm2_vvvv_slice."""
        return self._densities()[1]

    def rdm2_vvvv_slice(self, a, spin):
        from ccpy.density.ccsd_rdm import calc_rdm2_vvvv_slice

        return calc_rdm2_vvvv_slice(self.L, self.cc.T, a, spin)


class LeftCCSDT(LeftCCBase):
    module = "ccpy.left.left_ccsdt"
    order = 3
    requires_df_port = True
    cc_class, hbar_class = CCSDT, HBarCCSDT


class LeftCC3(LeftCCBase):
    module = "ccpy.left.left_cc3"
    requires_df_port = True
    cc_class, hbar_class = CC3, HBarCC3


class LeftCCSDT_P(LeftCCBase):
    """Left-CCSDT(P) on top of CCSDT(P) (L3 has the excitations of T3)."""
    module = "ccpy.left.left_ccsdt_p"
    order = 3
    requires_df_port = True
    cc_class, hbar_class = CCSDT_P, HBarCCSDT_P

    def __init__(self, upstream, nfrozen=None, ndelete=None, **options):
        if not (isinstance(upstream, MethodBase) and hasattr(upstream, "cc")):
            raise TypeError("LeftCCSDT_P needs an upstream CCSDT_P method.")
        super().__init__(upstream, nfrozen, ndelete, **options)

    def _new_operator(self):
        return ClusterOperator(self.system, order=3, p_orders=[3], pspace_sizes=self.cc._pspace_sizes())

    def _extra_args(self):
        t3 = self.cc.t3_excitations
        return (t3, {k: v.copy() for k, v in t3.items()})

    def _align(self, extra):
        # the left-CC(P) update reorders l3 (and L3); downstream corrections expect L3 in the
        # order of t3_excitations, which depends on the CC(P) solver that produced T
        from ccpy.utilities.utilities import reorder_triples_amplitudes

        t3, l3 = extra
        if any(not np.array_equal(t3[s], l3[s]) for s in t3):
            self.L = reorder_triples_amplitudes(self.L, l3, t3)


class LeftEOMCCBase(MethodBase):
    """Base class of left-EOMCC methods (scheme: left-EOMCC), built on a right EOMCC
    method; each left vector is normalized so that <L|R> = 1.

    Parameters
    ----------
    upstream : EOMCC method
    solver : "davidson" (default) or "jacobi" (EE methods with a left-CC update only)
    """

    module = None
    lr_function = "get_LR"
    printer = "print_ee_amplitudes"
    summary = "leftcc_calculation_summary"

    def __init__(self, upstream, solver="davidson", **options):
        if not isinstance(upstream, EOMCCBase):
            raise TypeError(f"{type(self).__name__} needs an upstream EOMCC method.")
        super().__init__(upstream, **options)
        self.options["RHF_symmetry"] = upstream.options["RHF_symmetry"]
        self.eom = upstream
        self.cc = upstream.cc
        self.solver = solver
        self.L = None
        self.omega = None
        self.LR = None

    @property
    def hbar(self):
        return self.eom.hbar

    def _extra_args(self):
        """(t3_excitations, l3_excitations) for P-space methods."""
        if not hasattr(self.eom, "r3_excitations"):
            return ()
        r3 = self.eom.r3_excitations
        return (getattr(self.cc, "t3_excitations", None), {k: v.copy() for k, v in r3.items()})

    def _lr(self, R, L, l3=None):
        import ccpy.energy.cc_energy as cc_energy

        r3 = getattr(self.eom, "r3_excitations", None)
        return getattr(cc_energy, self.lr_function)(R, L, l3_excitations=l3, r3_excitations=r3 if l3 else None)

    def solve(self):
        import ccpy.utilities.printing as printing
        from ccpy.drivers.solvers import eomcc_davidson, left_cc_jacobi

        self._warn_df_port()
        eom = self.eom.ensure_solved()
        hbar = self.hbar
        module = import_module(self.module)
        extra = self._extra_args()
        l3 = extra[1] if extra else None
        if self.L is None or len(self.L) != eom.nroots:
            self.L = []
            for R in eom.R:
                L = eom._new_operator()
                L.unflatten(R.flatten())
                self.L.append(L)
        self.omega = np.array(eom.omega, dtype=float)
        self.LR = np.zeros(eom.nroots)
        converged = []
        self._print_header()
        for i in range(eom.nroots):
            print(f"   {type(self).__name__} calculation for root {i} started on", get_timestamp())
            getattr(printing, self.printer)(self.L[i], self.system, self.L[i].order, self.options["amp_print_threshold"])
            LH = eom._new_operator()
            if self.solver == "jacobi":
                lr = lambda L, l3_excitations, R=eom.R[i]: self._lr(R, L, l3_excitations)
                self.L[i], _, _, ok = left_cc_jacobi(module.update, self.L[i], LH, hbar.T, hbar.H, lr,
                                                     eom.omega[i], False, self.system, self.options, *extra)
            else:
                L0 = self.L[i].flatten() / np.linalg.norm(self.L[i].flatten())
                self.L[i], self.omega[i], ok = eomcc_davidson(module.LH_fun, module.update_l, L0, self.L[i], LH,
                                                              eom.omega[i], hbar.T, hbar.H, self.system,
                                                              self.options, *extra)
            self.LR[i] = self._lr(eom.R[i], self.L[i], l3)
            self.L[i].unflatten(self.L[i].flatten() / self.LR[i])
            getattr(printing, self.summary)(self.L[i], self.omega[i], self.LR[i], ok, self.system,
                                            self.options["amp_print_threshold"])
            print(f"   |omega(R) - omega(L)| = {abs(self.omega[i] - eom.omega[i]):.3e}")
            converged.append(ok)
        self._finish_solve(all(converged), eom, hbar)
        return self

    def energy(self, root=None):
        self.ensure_solved()
        return self.eom.energy(root)


class LeftEOMCCSD(LeftEOMCCBase):
    """Left-EOMCCSD with DF/Cholesky integrals."""
    module = "ccpy.left.left_ccsd_chol"


class LeftEOMCCSDTA(LeftEOMCCBase):
    module = "ccpy.left.left_ccsd"
    requires_df_port = True


class LeftEOMCCSDT(LeftEOMCCBase):
    module = "ccpy.left.left_ccsdt"
    requires_df_port = True


class LeftEOMCCSDT_P(LeftEOMCCBase):
    module = "ccpy.left.left_ccsdt_p"
    requires_df_port = True


class LeftIPEOMCCBase(LeftEOMCCBase):
    lr_function, printer, summary = "get_LR_ipeom", "print_ip_amplitudes", "leftipcc_calculation_summary"
    requires_df_port = True


class LeftIPEOM2(LeftIPEOMCCBase):
    module = "ccpy.left.left_ipeom2"


class LeftIPEOM3(LeftIPEOMCCBase):
    module = "ccpy.left.left_ipeom3"


class LeftIPEOM3_P(LeftIPEOMCCBase):
    module = "ccpy.left.left_ipeom3_p"


class LeftEAEOMCCBase(LeftEOMCCBase):
    lr_function, printer, summary = "get_LR_eaeom", "print_ea_amplitudes", "lefteacc_calculation_summary"
    requires_df_port = True


class LeftEAEOM2(LeftEAEOMCCBase):
    module = "ccpy.left.left_eaeom2"


class LeftEAEOM3(LeftEAEOMCCBase):
    module = "ccpy.left.left_eaeom3"


class LeftEAEOM3_P(LeftEAEOMCCBase):
    module = "ccpy.left.left_eaeom3_p"
