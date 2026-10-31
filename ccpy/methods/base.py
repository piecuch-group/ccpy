"""Common contract of all CCpy methods.

Every method exposes ``solve()``, ``energy()``, ``gradient()`` and
``energy_and_gradient()``. ``energy()`` (and ``gradient()``) call ``solve()`` whenever the
method is not converged or one of the quantities it was solved with (T, HBar, L, ...)
has changed since; ``solve()`` always restarts from the current amplitudes, if any.
"""
from ccpy.methods.reference import Reference, warn_df_port

# options understood by the iterative solvers of ccpy.drivers.solvers
SOLVER_OPTIONS = {
    "maximum_iterations": 80,
    "amp_convergence": 1.0e-07,
    "energy_convergence": 1.0e-07,
    "energy_shift": 0.0,
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
    **options
        Solver options (see SOLVER_OPTIONS) and fd_step/fd_npoints for finite-difference
        gradients. Solver options are passed on to the methods created internally.
    """

    name = None
    # True for methods whose equations still need four-index integrals (see DFPortWarning)
    requires_df_port = False

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._ctor_args = (args, kwargs)
        return obj

    def __init__(self, upstream, nfrozen=None, ndelete=None, **options):
        unknown = set(options) - set(SOLVER_OPTIONS) - set(METHOD_OPTIONS)
        if unknown:
            raise TypeError(f"{type(self).__name__}: unknown options {sorted(unknown)}")
        self.upstream = upstream
        self.reference = resolve_reference(upstream, nfrozen, ndelete)
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

    # ------------------------------------------------------------------ contract
    def solve(self):
        raise NotImplementedError

    def energy(self):
        raise NotImplementedError

    def ensure_solved(self):
        """Solve unless converged and up to date with the quantities it depends on."""
        if not self.is_current():
            self.solve()
        return self

    def is_current(self):
        return self.converged and all(dep.version == v for dep, v in self._dependencies)

    def gradient(self, **kwargs):
        """Nuclear gradient (natoms, 3) in Eh/bohr: analytic when available, otherwise by
        central differences of energy()."""
        if self.has_analytic_gradient(**kwargs):
            return self._analytic_gradient(**kwargs)
        print(f"   {type(self).__name__}: no analytic gradient available; using finite differences.")
        return self.fd_gradient(**kwargs)

    def energy_and_gradient(self, **kwargs):
        return self.energy(**kwargs), self.gradient(**kwargs)

    def has_analytic_gradient(self, **kwargs):
        return False

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
        return new

    def _copy_setup(self, new):
        """Transfer setup done after construction (e.g., an EOM initial guess) to `new`."""

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
        print("   ------------------------------------------\n")


def resolve_reference(upstream, nfrozen=None, ndelete=None):
    if isinstance(upstream, MethodBase) or isinstance(upstream, Reference):
        ref = upstream.reference if isinstance(upstream, MethodBase) else upstream
        for name, value in (("nfrozen", nfrozen), ("ndelete", ndelete)):
            if value is not None and value != getattr(ref, name):
                raise ValueError(f"{name}={value} conflicts with the upstream calculation ({getattr(ref, name)}).")
        return ref
    return Reference(upstream, nfrozen or 0, ndelete or 0)
