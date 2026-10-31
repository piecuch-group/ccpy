"""Geometry and transition-state optimization of any ccpy method through pysisyphus
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
"""

import time
from dataclasses import dataclass

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


