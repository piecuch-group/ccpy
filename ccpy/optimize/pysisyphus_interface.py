"""Geometry optimization, transition states and IRCs through pysisyphus
(https://github.com/eljost/pysisyphus).

The method is given as one Python callable of a ccpy.integrals.Molecule that returns the
energy and its gradient together, so that work shared by both (SCF, CC, HBar, left-CC, ...)
is done once per geometry:

    energy_and_gradient(mol) -> (energy in Eh, gradient ndarray (natoms, 3) in Eh/bohr)

Alternatively, a ccpy.methods method (CCSD, CRCC23, ...) is passed in place of the Molecule;
its calculation is repeated at every geometry with energy_and_gradient(), which uses the
analytic gradient when available and central differences otherwise. For an arbitrary
energy-only callable, fd_energy_and_gradient(energy) provides the central-difference
gradient (ccpy.gradients.fd_gradient). Hessians, needed to start a transition-state search,
are central differences of the gradient.

    from ccpy import CCSDT, Molecule, RHF
    from ccpy.optimize import optimize_geometry

    mol = Molecule(geometry, "cc-pvdz", auxiliary_basis="cc-pvtz-rifit")
    result = optimize_geometry(CCSDT(RHF(mol, verbose=0), nfrozen=1))
    print(result.energy, result.molecule.atoms)

The Molecule must be built with symmetry=False, since every geometry step is a new
Molecule made with Molecule.with_geometry.

``run_irc(method_at_saddle)`` or ``run_irc(molecule, callback)`` integrates both
IRC directions and returns IRCResult with native endpoint molecules. Set
``state`` for an electronic root and ``root`` for a Hessian mode independently.
"""

import time
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from ccpy.integrals.atom_data import Z_TO_ATOM_SYMBOL


def _import_pysisyphus():
    try:
        import pysisyphus  # noqa: F401
    except ImportError as e:
        raise ImportError("Geometry optimization needs pysisyphus: pip install pysisyphus") from e


def make_calculator(molecule, energy_and_gradient, hessian_step=5.0e-3, verbose=True):
    """pysisyphus Calculator that evaluates `energy_and_gradient` at the geometries
    pysisyphus asks for. See CCpyCalculator below for the arguments."""
    _import_pysisyphus()
    from pysisyphus.calculators.Calculator import Calculator

    class CCpyCalculator(Calculator):
        """Arguments:
        ----------
        molecule : ccpy.integrals.Molecule -> template (basis, charge, multiplicity, ...),
                                              built with symmetry=False
        energy_and_gradient : callable(Molecule) -> (energy in Eh, gradient (natoms, 3) in Eh/bohr)
        hessian_step : displacement (bohr) of the finite-difference Hessian

        The last geometry's result is cached, so an energy request followed by a gradient
        request at the same geometry costs one evaluation.
        """

        def __init__(self):
            super().__init__(charge=molecule.charge, mult=molecule.multiplicity,
                             base_name="ccpy", out_dir=".")
            self.molecule = molecule
            self.function = energy_and_gradient
            self.ncalls = 0
            self._cache = (None, None)

        def _molecule_at(self, coords):
            return self.molecule.with_geometry(np.reshape(coords, (-1, 3)), unit="bohr")

        def _evaluate(self, coords):
            coords = np.asarray(coords, dtype=float).ravel()
            key = coords.tobytes()
            if self._cache[0] == key:
                return self._cache[1]
            self.ncalls += 1
            t0 = time.perf_counter()
            e, g = self.function(self._molecule_at(coords))
            e, g = float(e), np.asarray(g, dtype=float)
            if verbose:
                print(f"   [opt] evaluation {self.ncalls:3d}: E = {e:.10f}  "
                      f"|g|max = {np.abs(g).max():.2e}  ({time.perf_counter() - t0:.1f} s)")
            self._cache = (key, (e, g))
            return e, g

        def get_energy(self, atoms, coords, **kwargs):
            return {"energy": self._evaluate(coords)[0]}

        def get_forces(self, atoms, coords, **kwargs):
            e, g = self._evaluate(coords)
            return {"energy": e, "forces": -g.ravel()}

        def get_hessian(self, atoms, coords, **kwargs):
            """Cartesian Hessian by central differences of the gradient (6 natoms gradients)."""
            coords = np.asarray(coords, dtype=float).ravel()
            n = coords.size
            if verbose:
                print(f"   [opt] finite-difference Hessian: {2 * n} gradient evaluations")
            H = np.zeros((n, n))
            for i in range(n):
                step = np.zeros(n)
                step[i] = hessian_step
                g_plus = self._evaluate(coords + step)[1].ravel()
                g_minus = self._evaluate(coords - step)[1].ravel()
                H[i] = (g_plus - g_minus) / (2.0 * hessian_step)
            results = self.get_forces(atoms, coords)
            results["hessian"] = 0.5 * (H + H.T)
            return results

    return CCpyCalculator()


def fd_energy_and_gradient(energy, step=1.0e-3, npoints=4):
    """energy_and_gradient callable for an energy-only method: the energy and its
    central-difference gradient (ccpy.gradients.fd_gradient; step in bohr)."""
    from ccpy.gradients.finite_difference import fd_gradient

    def energy_and_gradient(mol):
        return energy(mol), fd_gradient(energy, mol, step=step, npoints=npoints, verbose=False)

    return energy_and_gradient


@dataclass
class OptimizationResult:
    molecule: object          # ccpy.integrals.Molecule at the final geometry
    energy: float             # Eh
    converged: bool
    ncycles: int
    optimizer: object         # the pysisyphus optimizer, for its full history
    hessian_eigenvalues: object = None  # TS search only: eigenvalues of the final Cartesian Hessian


@dataclass
class IRCBranch:
    """One integrated branch, ordered from its initial displacement to its endpoint.

    Coordinates are Cartesian bohr, energies Eh, gradients Eh/bohr. Convergence
    flags retain pysisyphus' criteria; a converged branch need not be a minimum.
    ``molecule`` is the unoptimized last stored geometry.
    """

    molecule: object
    coordinates: np.ndarray
    energies: np.ndarray
    gradients: np.ndarray
    converged: bool
    energy_converged: bool
    energy_increased: bool
    ncycles: int

    @property
    def gradient_rms(self):
        """RMS Cartesian gradient at the last stored point, in Eh/bohr."""
        return float(np.sqrt(np.mean(self.gradients[-1] ** 2)))


@dataclass
class IRCResult:
    """IRC data and native endpoint molecules; no endpoint optimization is implied."""

    start_molecule: object
    start_energy: float
    branches: dict
    irc: object

    @property
    def converged(self):
        return bool(self.branches) and all(branch.converged for branch in self.branches.values())

    @property
    def endpoints(self):
        return {name: branch.molecule for name, branch in self.branches.items()}


def _geometry(molecule, coord_type):
    from pysisyphus.Geometry import Geometry

    if molecule.symmetry:
        raise ValueError("Build the Molecule with symmetry=False for geometry optimization.")
    atoms = [Z_TO_ATOM_SYMBOL[int(Z)] for Z in molecule.atomic_charges]
    coords = np.array([xyz for _, xyz in molecule.atoms]).ravel()
    return Geometry(atoms, coords, coord_type=coord_type)


def _finish(calc, geom, opt, extra=None):
    mol = calc._molecule_at(geom.cart_coords)
    result = OptimizationResult(mol, float(geom.energy), bool(opt.is_converged),
                                int(opt.cur_cycle) + 1, opt, extra)
    print("")
    print(f"   Optimization {'converged' if result.converged else 'NOT converged'} "
          f"in {result.ncycles} cycles; final energy = {result.energy:.10f} Eh")
    print("   Final geometry (bohr):")
    for Z, xyz in mol.atoms:
        print(f"      {Z_TO_ATOM_SYMBOL[Z]:<3}{xyz[0]:>16.10f}{xyz[1]:>16.10f}{xyz[2]:>16.10f}")
    print("")
    return result


def _resolve_target(molecule, energy_and_gradient):
    """(Molecule, energy_and_gradient callable) from either a Molecule and a callable, or a
    ccpy.methods method (whose calculation is repeated at every geometry)."""
    from ccpy.methods.base import MethodBase

    if isinstance(molecule, MethodBase):
        method = molecule
        if energy_and_gradient is not None:
            raise ValueError("Pass either a method or a Molecule with an energy_and_gradient callable.")
        return method.molecule, lambda mol: method.at_geometry(mol).energy_and_gradient()
    if energy_and_gradient is None:
        raise ValueError("energy_and_gradient is required when a Molecule is given.")
    return molecule, energy_and_gradient


def optimize_geometry(molecule, energy_and_gradient=None, coord_type="redund", thresh="gau",
                      max_cycles=100, hessian_init="fischer", verbose=True, **optimizer_kwargs):
    """Minimize the energy with pysisyphus' rational-function optimizer (RFO).

    Arguments:
    ----------
    molecule : Molecule -> starting geometry (symmetry=False), or a ccpy.methods method
               (e.g., CCSD(mf)), whose energy_and_gradient() is then used
    energy_and_gradient : callable(Molecule) -> (energy, gradient); see the module docstring
    coord_type : str -> "redund" (redundant internals, recommended), "dlc", "tric" or "cart"
    thresh : str -> pysisyphus convergence criterion, e.g. "gau_loose", "gau", "gau_tight"
    hessian_init : str -> starting Hessian: a model Hessian ("fischer", "lindh", "swart")
                          or "calc" for a finite-difference Hessian
    optimizer_kwargs -> passed on to pysisyphus.optimizers.RFOptimizer
    Returns an OptimizationResult.
    """
    _import_pysisyphus()
    from pysisyphus.optimizers.RFOptimizer import RFOptimizer

    molecule, energy_and_gradient = _resolve_target(molecule, energy_and_gradient)
    calc = make_calculator(molecule, energy_and_gradient, verbose=verbose)
    geom = _geometry(molecule, coord_type)
    geom.set_calculator(calc)
    opt = RFOptimizer(geom, thresh=thresh, max_cycles=max_cycles, hessian_init=hessian_init,
                      dump=False, **optimizer_kwargs)
    opt.run()
    return _finish(calc, geom, opt)


def optimize_transition_state(molecule, energy_and_gradient=None, coord_type="redund", thresh="gau",
                              max_cycles=100, root=0, hessian_init="calc", hessian_recalc=None,
                              hessian_step=5.0e-3, verbose=True, **optimizer_kwargs):
    """First-order saddle point search with pysisyphus' restricted-step partitioned RFO
    (RS-P-RFO), maximizing along one Hessian eigenvector and minimizing along the rest.

    Arguments:
    ----------
    molecule : Molecule -> guess of the transition-state geometry (symmetry=False), or a
               ccpy.methods method at that geometry
    energy_and_gradient : callable(Molecule) -> (energy, gradient); see the module docstring
    root : int -> index (in ascending eigenvalue order) of the Hessian mode to follow uphill;
                  0 follows the lowest mode, which should be the imaginary mode of a good guess
    hessian_init : str -> "calc" (finite-difference Hessian, recommended) or a model Hessian
    hessian_recalc : int -> recompute the Hessian every n cycles (None: only Bofill updates)
    optimizer_kwargs -> passed on to pysisyphus.tsoptimizers.RSPRFOptimizer
    Returns an OptimizationResult whose hessian_eigenvalues (of a final finite-difference
    Hessian, in Eh/bohr^2) should contain exactly one negative value after the six
    (five for linear molecules) near-zero translations and rotations are discarded.
    """
    _import_pysisyphus()
    from pysisyphus.tsoptimizers.RSPRFOptimizer import RSPRFOptimizer

    molecule, energy_and_gradient = _resolve_target(molecule, energy_and_gradient)
    calc = make_calculator(molecule, energy_and_gradient, hessian_step=hessian_step, verbose=verbose)
    geom = _geometry(molecule, coord_type)
    geom.set_calculator(calc)
    opt = RSPRFOptimizer(geom, root=root, thresh=thresh, max_cycles=max_cycles, hessian_init=hessian_init,
                         hessian_recalc=hessian_recalc, dump=False, **optimizer_kwargs)
    opt.run()

    # characterize the stationary point with a final Cartesian Hessian
    H = calc.get_hessian(geom.atoms, geom.cart_coords)["hessian"]
    eigenvalues = np.linalg.eigvalsh(H)
    if verbose:
        print("   Lowest Cartesian Hessian eigenvalues (Eh/bohr^2):",
              np.array2string(eigenvalues[:8], precision=5))
    return _finish(calc, geom, opt, extra=eigenvalues)


def run_irc(molecule, energy_and_gradient=None, *, state=None, root=0,
            forward=True, backward=True, downhill=False, step_length=0.1,
            max_cycles=125, rms_grad_thresh=1.e-3, hessian_init="calc",
            hessian_step=5.e-3, out_dir=".", prefix="", verbose=True, **irc_kwargs):
    """Integrate an Euler predictor-corrector IRC with pysisyphus.

    Parameters
    ----------
    molecule : Molecule or MethodBase
        Starting saddle (symmetry=False), or a method defined on that geometry.
        Method inputs are rebuilt with at_geometry at each evaluation.
    energy_and_gradient : callable, optional
        Required with a Molecule: returns a scalar energy (Eh) and a real finite
        Cartesian gradient of shape (natoms, 3) (Eh/bohr). Do not negate it.
    state : int, optional
        Returned electronic root for a method input, passed as root to its energy
        and gradient methods. Required for array-valued excited-state energies.
        For callback inputs select the electronic state inside the callback.
    root : int
        Hessian mode for the initial displacement, distinct from electronic state.
    forward, backward : bool
        Integrate both directions by default. At least one must be enabled unless
        downhill=True, which instead runs one branch from a nonstationary point.
    step_length, max_cycles, rms_grad_thresh
        EulerPC integration controls; max_cycles applies separately to each branch.
    hessian_init : str or path
        "calc" or a Hessian file supported by pysisyphus. Calculated Hessians are
        central differences of gradients using hessian_step (bohr).
    out_dir, prefix
        IRC output directory and filename prefix. Some pysisyphus files can still
        be written in the process working directory; use a dedicated directory.
    **irc_kwargs
        Extra EulerPC/IRC settings, e.g. displ="length", displ_length=0.1,
        hessian_recalc, energy_thresh or dump_every.

    Returns
    -------
    IRCResult
        Copies of per-branch trajectories, native endpoint molecules, convergence
        diagnostics and the underlying pysisyphus integrator. A cycle-limited run
        returns converged=False rather than being labeled successful. Endpoint
        minimization and vibrational characterization remain separate steps.
    """
    from ccpy.methods.base import MethodBase

    for name, value in (("root", root), ("max_cycles", max_cycles), ("state", state)):
        if value is None and name == "state":
            continue
        if not isinstance(value, Integral) or isinstance(value, bool) or value < (1 if name == "max_cycles" else 0):
            raise ValueError(f"{name} must be a {'positive' if name == 'max_cycles' else 'nonnegative'} integer.")
    for name, value in (("step_length", step_length), ("rms_grad_thresh", rms_grad_thresh),
                        ("hessian_step", hessian_step)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not downhill and not (forward or backward):
        raise ValueError("Enable forward or backward, or set downhill=True.")
    if isinstance(molecule, MethodBase):
        method = molecule
        if energy_and_gradient is not None:
            raise ValueError("Pass either a method or a Molecule with an energy_and_gradient callable.")
        molecule = method.molecule

        def evaluate(mol):
            calculation = method.at_geometry(mol)
            kwargs = {} if state is None else {"root": state}
            energy = calculation.energy(**kwargs)
            if np.ndim(energy) != 0:
                raise ValueError("IRC requires a scalar energy; select a returned electronic root with state=... .")
            gradient = calculation.gradient(**kwargs)
            if not calculation.converged:
                raise RuntimeError("IRC electronic calculation did not converge.")
            return energy, gradient

        function = evaluate
    else:
        if state is not None:
            raise ValueError("Select the electronic state inside the callback; state is for method inputs.")
        molecule, function = _resolve_target(molecule, energy_and_gradient)
    if molecule.symmetry:
        raise ValueError("Build the Molecule with symmetry=False for IRC integration.")

    def checked_function(mol):
        energy, gradient = function(mol)
        energy, gradient = np.asarray(energy), np.asarray(gradient)
        if energy.ndim != 0 or np.iscomplexobj(energy) or not np.isfinite(energy):
            raise ValueError("IRC callback must return one finite real scalar energy.")
        if (gradient.shape != (mol.natoms, 3) or np.iscomplexobj(gradient)
                or not np.all(np.isfinite(gradient))):
            raise ValueError("IRC gradient must be finite and real with shape (natoms, 3), in Eh/bohr.")
        return float(energy), np.asarray(gradient, dtype=float)

    _import_pysisyphus()
    from pysisyphus.irc.EulerPC import EulerPC

    geom = _geometry(molecule, "cart")
    calc = make_calculator(molecule, checked_function, hessian_step=hessian_step, verbose=verbose)
    geom.set_calculator(calc)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    irc = EulerPC(geom, root=root, forward=forward, backward=backward, downhill=downhill,
                  step_length=step_length, max_cycles=max_cycles, rms_grad_thresh=rms_grad_thresh,
                  hessian_init=None if hessian_init is None else str(hessian_init),
                  out_dir=out_dir, prefix=prefix, **irc_kwargs)
    irc.run()
    directions = ("downhill",) if downhill else tuple(name for name, enabled in
                                                    (("forward", forward), ("backward", backward)) if enabled)
    branches = {}
    for name in directions:
        coords = np.asarray(getattr(irc, f"{name}_coords"), dtype=float).reshape(-1, molecule.natoms, 3).copy()
        gradients = np.asarray(getattr(irc, f"{name}_gradients"), dtype=float).reshape(coords.shape).copy()
        branches[name] = IRCBranch(
            molecule.with_geometry(coords[-1], unit="bohr"), coords,
            np.asarray(getattr(irc, f"{name}_energies"), dtype=float).copy(), gradients,
            bool(getattr(irc, f"{name}_is_converged")), bool(getattr(irc, f"{name}_energy_converged")),
            bool(getattr(irc, f"{name}_energy_increased")), int(getattr(irc, f"{name}_cycle")) + 1,
        )
    return IRCResult(molecule, float(irc.ts_energy), branches, irc)


