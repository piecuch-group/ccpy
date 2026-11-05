"""Common contract of all CCpy methods.

Every method exposes ``solve()``, ``energy()``, ``gradient()`` and
``energy_and_gradient()``. ``energy()`` (and ``gradient()``) call ``solve()`` whenever the
method is not converged or one of the quantities it was solved with (T, HBar, L, ...)
has changed since; ``solve()`` always restarts from the current amplitudes, if any.

Analytic gradients follow the energies: a method names its gradient kernel with
``gradient_module`` and ``gradient_function`` (as ``module`` and ``function`` name the energy
kernels), and its scheme's ``_analytic_gradient`` passes it the converged quantities and the
Reference, e.g. calc_crcc23_gradient(T, L, Hbar, reference, delta=...).
"""
from importlib import import_module

import numpy as np

from ccpy.methods.reference import Reference, warn_df_port

# options understood by the iterative solvers of ccpy.drivers.solvers
SOLVER_OPTIONS = {
    "maximum_iterations": 80,
    "amp_convergence": 1.0e-07,
    "energy_convergence": 1.0e-07,
    "energy_shift": 0.0,
    "denominator": "mp",
    "diis_size": 6,
    "RHF_symmetry": None,
    "diis_out_of_core": False,
    "davidson_out_of_core": False,
    "amp_print_threshold": 0.09,
    "davidson_max_subspace_size": 30,
    "davidson_solver": "standard",
    "davidson_selection_method": "overlap",
}
# finite-difference gradient settings (bohr, stencil size)
METHOD_OPTIONS = {"fd_step": 1.0e-3, "fd_npoints": 4}


class MethodBase:
    """Base class of all methods.

    Parameters
    ----------
    upstream : SCF object (ccpy.scf.RHF/ROHF/UHF), Reference, or another method
        Calculation this method builds on. Methods that need other methods (e.g.,
        CR-CC(2,3) needs CCSD, its HBar, and left-CCSD) create them when given an SCF
        object, and reuse them when given a method that holds them.
    nfrozen, ndelete : int, optional
        Frozen core and deleted virtual orbitals (only with an SCF upstream).
    fno : FNOOptions, optional
        Prepare a DF-MP2 (or registered provider) frozen-natural-orbital reference
        before allocating correlated quantities. Inherited from an upstream method.
    fno_targets : sequence of FNOTarget, optional
        Explicit target identities in returned EOM root order for state-specific
        completion. Defaults to the prepared reference's configured target order.
    **options
        Solver options (see SOLVER_OPTIONS) and fd_step/fd_npoints for finite-difference
        gradients. Solver options are passed on to the methods created internally.
    """

    name = None
    # True for methods whose equations still need four-index integrals (see DFPortWarning)
    requires_df_port = False
    # analytic gradient kernel (ccpy.gradients), if there is one
    gradient_module = gradient_function = None

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._ctor_args = (args, kwargs)
        return obj

    def __init__(self, upstream, nfrozen=None, ndelete=None, *, fno=None, fno_targets=None, **options):
        unknown = set(options) - set(SOLVER_OPTIONS) - set(METHOD_OPTIONS)
        if unknown:
            raise TypeError(f"{type(self).__name__}: unknown options {sorted(unknown)}")
        self.upstream = upstream
        self.reference = self.resolve_reference(upstream, nfrozen, ndelete)
        from ccpy.methods.fno import FNOTarget, prepare_fno

        existing = self.reference.fno_result
        if fno is False and existing is not None:
            raise ValueError("Cannot disable upstream FNOs; start a new calculation from SCF.")
        if fno is not None and fno is not False:
            if isinstance(upstream, MethodBase) and existing is None:
                raise ValueError("Cannot change an upstream method's orbital basis; enable FNOs on its SCF/reference.")
            self.reference = prepare_fno(self.reference, fno)
        self.fno_targets = None if fno_targets is None else tuple(fno_targets)
        if self.fno_targets is not None and not all(isinstance(t, FNOTarget) for t in self.fno_targets):
            raise TypeError("fno_targets must contain FNOTarget objects in EOM output order.")
        if self.fno_targets is not None and len(set(self.fno_targets)) != len(self.fno_targets):
            raise ValueError("fno_targets must identify distinct EOM states.")
        # solver options given to an upstream method carry over unless overridden
        inherited = upstream.user_options if isinstance(upstream, MethodBase) else {}
        options = {**{k: v for k, v in inherited.items() if k != "RHF_symmetry"}, **options}
        self.user_options = {k: v for k, v in options.items() if k in SOLVER_OPTIONS}
        self.options = {**SOLVER_OPTIONS, **METHOD_OPTIONS, **options}
        self.options["method"] = self.name or type(self).__name__
        if self.options["RHF_symmetry"] is None:
            self.options["RHF_symmetry"] = self.reference.rhf_symmetry
        system = self.system
        if "diis_size" not in options and system.noccupied_alpha * system.nunoccupied_beta <= 4:
            # DIIS is singular for very small problems
            self.options["diis_size"] = -1
        self.converged = False
        self.version = 0
        self._dependencies = []

    # ------------------------------------------------------------------ properties
    @property
    def system(self):
        return self.reference.system

    @property
    def molecule(self):
        return self.reference.molecule

    @property
    def reference_energy(self):
        return self.reference.energy

    @property
    def fno_result(self):
        return self.reference.fno_result

    @property
    def fno_options(self):
        return None if self.fno_result is None else self.fno_result.options

    def _complete_ground_energy(self, raw):
        result = self.fno_result
        if result is None or result.options.energy_completion == "none":
            return raw
        return raw + result.delta_ground

    def _state_shifts(self, count, sector, irreps=None):
        """Reporting-only state shifts; kernels always retain raw eigenvalues."""
        import numpy as np

        result = self.fno_result
        if result is None or result.options.energy_completion == "none":
            return np.zeros(count)
        if result.options.energy_completion == "ground":
            return np.full(count, result.delta_ground)
        targets = self.fno_targets if self.fno_targets is not None else result.options.targets
        if len(targets) != count or any(t.sector != sector for t in targets):
            raise ValueError("State-specific FNO completion requires one matching-sector FNOTarget per returned root; "
                             "pass fno_targets in EOM result order.")
        if irreps is not None and any(t.irrep is not None and t.irrep != irrep
                                      for t, irrep in zip(targets, irreps)):
            raise ValueError("FNO target irreps do not match the EOM root order.")
        multiplicity = getattr(self, "multiplicity", None)
        if multiplicity is None and getattr(self, "guess_settings", None) is not None:
            multiplicity = self.guess_settings["multiplicity"]
        if multiplicity is not None and multiplicity > 0 and any(
                t.multiplicity is not None and t.multiplicity != multiplicity for t in targets):
            raise ValueError("FNO target multiplicities do not match the targeted EOM spin.")
        try:
            return np.array([result.delta_state(t) for t in targets])
        except KeyError as exc:
            raise ValueError("Missing matched low-level energy for an EOM FNO target.") from exc

    # ------------------------------------------------------------------ contract
    def solve(self):
        raise NotImplementedError

    def energy(self):
        raise NotImplementedError

    def raw_energy(self):
        """Ordinary truncated-space energy, before FNO completion."""
        raise NotImplementedError

    def ensure_solved(self):
        """Solve unless converged and up to date with the quantities it depends on."""
        if not self.is_current():
            self.solve()
        return self

    def is_current(self):
        return self.converged and all(dep.version == v for dep, v in self._dependencies)

    def gradient(self, perturbation=None, properties=None, **kwargs):
        """Nuclear gradient (natoms, 3) in Eh/bohr: analytic when available, otherwise by
        central differences of energy(). With perturbation (AO integrals V, (nao, nao) or
        (ncomp, nao, nao), of one-electron operators added to h with strengths lambda), the
        analytic dE/dlambda, (ncomp,), from the same orbital-relaxed density; with properties V,
        both, as (gradient, dE/dlambda)."""
        if self.has_analytic_gradient(**kwargs):
            return self._analytic_gradient(perturbation=perturbation, properties=properties, **kwargs)
        if perturbation is not None or properties is not None:
            raise NotImplementedError(f"{type(self).__name__} has no analytic gradient for this reference.")
        print(f"   {type(self).__name__}: no analytic gradient available; using finite differences.")
        return self.fd_gradient(**kwargs)

    def multipole_moments(self, order=1, origin=None, **kwargs):
        """Orbital-relaxed Cartesian multipole moments (atomic units, not traceless) of orders
        1..order (<= 3) about origin (bohr), as {"x": ..., "xx": ..., ...}: the analytic derivatives
        of the energy with respect to the electronic moment operators, plus the nuclear moments."""
        from ccpy.gradients.derivative_integrals import multipole_integrals, multipole_moments

        _, V = multipole_integrals(self.molecule, order, origin)
        return multipole_moments(self.gradient(perturbation=V, **kwargs), self.molecule, order, origin)

    def gradient_and_dipole(self, **kwargs):
        """Nuclear gradient (natoms, 3) and orbital-relaxed dipole moment (3,), atomic units, from
        one analytic gradient calculation (for fd_hessian(..., dipole=True))."""
        from ccpy.gradients.derivative_integrals import multipole_integrals, multipole_moments

        _, V = multipole_integrals(self.molecule, 1)
        grad, dE = self.gradient(properties=V, **kwargs)
        return grad, np.array(list(multipole_moments(dE, self.molecule, 1).values()))

    def energy_and_gradient(self, **kwargs):
        return self.energy(**kwargs), self.gradient(**kwargs)

    def has_analytic_gradient(self, **kwargs):
        """True when the method has a gradient kernel and the reference supports it (RHF or UHF
        orbitals, not Kohn-Sham ones; no deleted virtuals)."""
        from ccpy.scf.rhf import RHF
        from ccpy.scf.uhf import UHF

        return (self.gradient_function is not None and type(self.reference.scf) in (RHF, UHF)
                and self.reference.ndelete == 0 and self.fno_result is None)

    def _gradient_function(self):
        return getattr(import_module(self.gradient_module), self.gradient_function)

    def _analytic_gradient(self, **kwargs):
        raise NotImplementedError

    def fd_gradient(self, **kwargs):
        from ccpy.gradients.finite_difference import fd_gradient

        return fd_gradient(lambda mol: self.at_geometry(mol).energy(**kwargs), self.molecule,
                           step=self.options["fd_step"], npoints=self.options["fd_npoints"])

    def at_geometry(self, molecule):
        """The same calculation (method, options, upstream chain) for another Molecule."""
        args, kwargs = self._ctor_args[0], dict(self._ctor_args[1])
        upstream = args[0] if args else kwargs.pop("upstream")
        if isinstance(upstream, (MethodBase, Reference)):
            new_upstream = upstream.at_geometry(molecule) if isinstance(upstream, MethodBase) \
                else upstream.with_molecule(molecule)
        else:
            new_upstream = upstream.with_molecule(molecule)
        new = type(self)(new_upstream, *args[1:], **kwargs)
        self._copy_setup(new)
        new.fno_targets = self.fno_targets
        return new

    def _copy_setup(self, new):
        """Transfer setup done after construction (e.g., an EOM initial guess) to `new`."""

    def _validate_working_excitations(self, excitations, particle_spins, hole_spins):
        """Check selected-space indices before calling compiled FNO-space kernels.

        Lists remain the caller's responsibility to generate in the new basis;
        dimensions alone cannot detect a stale list whose indices happen to fit.
        """
        if self.fno_result is None:
            return
        import numpy as np

        if set(excitations) != set(particle_spins):
            raise ValueError("Selected excitations must contain every required spin block in the working basis.")
        nv = {"a": self.system.nunoccupied_alpha, "b": self.system.nunoccupied_beta}
        no = {"a": self.system.noccupied_alpha, "b": self.system.noccupied_beta}
        for key, values in excitations.items():
            values = np.asarray(values)
            bounds = [nv[s] for s in particle_spins[key]] + [no[s] for s in hole_spins[key]]
            if (values.ndim != 2 or values.shape[1] != len(bounds) or values.shape[0] == 0 or
                    not np.all(np.isfinite(values)) or not np.all(values == np.floor(values))):
                raise ValueError(f"Invalid selected excitation block {key}; supply integer rows in the working basis "
                                 "or one all-ones row for an empty spin block.")
            if values.shape[0] == 1 and np.all(values == 1):
                continue
            if np.any(values < 1) or np.any(values > np.asarray(bounds)):
                raise ValueError(f"Selected excitation block {key} exceeds the working orbital dimensions; "
                                 "regenerate selected spaces after FNO preparation.")

    # ------------------------------------------------------------------ helpers
    def _depends_on(self, *objects):
        self._dependencies = [(obj, obj.version) for obj in objects]

    def _finish_solve(self, converged, *dependencies):
        self.converged = bool(converged)
        self.version += 1
        self._depends_on(*dependencies)

    def _warn_df_port(self):
        if self.requires_df_port:
            warn_df_port(type(self).__name__)

    def _print_header(self):
        print(f"\n   {type(self).__name__}" + (" [legacy four-index equations]" if self.requires_df_port else ""))
        print("   ------------------------------------------")
        for key, value in self.options.items():
            print("  ", key, "=", value)
        if self.fno_result is not None:
            result = self.fno_result
            print(f"   FNO source = {result.options.source}; completion = {result.options.energy_completion}")
            print(f"   FNO virtuals (alpha, beta) = ({self.system.nunoccupied_alpha}, "
                  f"{self.system.nunoccupied_beta}); discarded = {result.discarded}")
            print(f"   FNO low-level energies: full = {result.full.ground_energy:.12f}, "
                  f"truncated = {result.truncated.ground_energy:.12f}")
            print(f"   FNO ground correction = {result.delta_ground:.12f}")
        print("   ------------------------------------------\n")

    @staticmethod
    def resolve_cc(method, upstream, cc_class):
        """Ground-state CC object held by `upstream`, or a new `cc_class` on method's reference."""
        if isinstance(upstream, MethodBase) and hasattr(upstream, "cc"):
            if upstream.cc.reference is not method.reference:
                raise ValueError("Cannot share CC amplitudes between different orbital references.")
            return upstream.cc
        return cc_class(method.reference, **method.user_options)

    @staticmethod
    def resolve_reference(upstream, nfrozen=None, ndelete=None):
        if isinstance(upstream, MethodBase) or isinstance(upstream, Reference):
            ref = upstream.reference if isinstance(upstream, MethodBase) else upstream
            for name, value in (("nfrozen", nfrozen), ("ndelete", ndelete)):
                if value is not None and value != getattr(ref, name):
                    raise ValueError(f"{name}={value} conflicts with the upstream calculation ({getattr(ref, name)}).")
            return ref
        return Reference(upstream, nfrozen or 0, ndelete or 0)
