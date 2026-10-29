"""Geometry and transition-state optimization of any ccpy method through pysisyphus
(https://github.com/eljost/pysisyphus).

The method is given as plain Python callables of a ccpy.integrals.Molecule:

    energy(mol) -> float                                       (always required)
    gradient(mol) -> (float, ndarray (natoms, 3) in Eh/bohr)   (optional)

Without a gradient callable, the gradient is obtained by central differences of the
energy (ccpy.gradients.fd_gradient). Hessians, needed to start a transition-state
search, are central differences of the gradient.

    from ccpy import Driver, Molecule, RHF
    from ccpy.optimize import optimize_geometry, optimize_transition_state

    def ccsdt_energy(mol):
        mf = RHF(mol, verbose=0); mf.kernel()
        driver = Driver.from_scf(mf, nfrozen=1)
        driver.run_cc(method="ccsdt")
        return driver.system.reference_energy + driver.correlation_energy

    mol = Molecule(geometry, "cc-pvdz", cholesky_tei=True, cholesky_tol=1.0e-10)
    result = optimize_geometry(mol, ccsdt_energy)
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


def make_calculator(molecule, energy, gradient=None, fd_step=1.0e-3, fd_npoints=4,
                    hessian_step=5.0e-3, verbose=True):
    """pysisyphus Calculator that evaluates `energy`/`gradient` at the geometries
    pysisyphus asks for. See CCpyCalculator below for the arguments."""
    _import_pysisyphus()
    from pysisyphus.calculators.Calculator import Calculator

    class CCpyCalculator(Calculator):
        """Arguments:
        ----------
        molecule : ccpy.integrals.Molecule -> template (basis, charge, multiplicity, ...),
                                              built with symmetry=False
        energy : callable(Molecule) -> energy in Eh
        gradient : callable(Molecule) -> (energy, gradient (natoms, 3) in Eh/bohr);
                   if None, the gradient is computed by finite differences of `energy`
        fd_step, fd_npoints : displacement (bohr) and stencil of the finite-difference gradient
        hessian_step : displacement (bohr) of the finite-difference Hessian
        """

        def __init__(self):
            super().__init__(charge=molecule.charge, mult=molecule.multiplicity,
                             base_name="ccpy", out_dir=".")
            self.molecule = molecule
            self.energy_function = energy
            self.gradient_function = gradient
            self.ncalls = {"energy": 0, "gradient": 0}

        def _molecule_at(self, coords):
            return self.molecule.with_geometry(np.reshape(coords, (-1, 3)), unit="bohr")

        def get_energy(self, atoms, coords, **kwargs):
            self.ncalls["energy"] += 1
            return {"energy": float(self.energy_function(self._molecule_at(coords)))}

        def get_forces(self, atoms, coords, **kwargs):
            self.ncalls["gradient"] += 1
            t0 = time.perf_counter()
            mol = self._molecule_at(coords)
            if self.gradient_function is not None:
                e, g = self.gradient_function(mol)
            else:
                from ccpy.gradients.finite_difference import fd_gradient
                g = fd_gradient(self.energy_function, mol, step=fd_step, npoints=fd_npoints, verbose=False)
                e = self.energy_function(mol)
            g = np.asarray(g, dtype=float)
            if verbose:
                print(f"   [opt] gradient {self.ncalls['gradient']:3d}: E = {e:.10f}  "
                      f"|g|max = {np.abs(g).max():.2e}  ({time.perf_counter() - t0:.1f} s)")
            return {"energy": float(e), "forces": -g.ravel()}

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
                g_plus = -self.get_forces(atoms, coords + step)["forces"]
                g_minus = -self.get_forces(atoms, coords - step)["forces"]
                H[i] = (g_plus - g_minus) / (2.0 * hessian_step)
            results = self.get_forces(atoms, coords)
            results["hessian"] = 0.5 * (H + H.T)
            return results

    return CCpyCalculator()


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


def optimize_geometry(molecule, energy, gradient=None, coord_type="redund", thresh="gau",
                      max_cycles=100, hessian_init="fischer", fd_step=1.0e-3, fd_npoints=4,
                      verbose=True, **optimizer_kwargs):
    """Minimize the energy with pysisyphus' rational-function optimizer (RFO).

    Arguments:
    ----------
    molecule : Molecule -> starting geometry (symmetry=False)
    energy, gradient : see the module docstring
    coord_type : str -> "redund" (redundant internals, recommended), "dlc", "tric" or "cart"
    thresh : str -> pysisyphus convergence criterion, e.g. "gau_loose", "gau", "gau_tight"
    hessian_init : str -> starting Hessian: a model Hessian ("fischer", "lindh", "swart")
                          or "calc" for a finite-difference Hessian
    optimizer_kwargs -> passed on to pysisyphus.optimizers.RFOptimizer
    Returns an OptimizationResult.
    """
    _import_pysisyphus()
    from pysisyphus.optimizers.RFOptimizer import RFOptimizer

    calc = make_calculator(molecule, energy, gradient, fd_step=fd_step, fd_npoints=fd_npoints, verbose=verbose)
    geom = _geometry(molecule, coord_type)
    geom.set_calculator(calc)
    opt = RFOptimizer(geom, thresh=thresh, max_cycles=max_cycles, hessian_init=hessian_init,
                      dump=False, **optimizer_kwargs)
    opt.run()
    return _finish(calc, geom, opt)


def optimize_transition_state(molecule, energy, gradient=None, coord_type="redund", thresh="gau",
                              max_cycles=100, root=0, hessian_init="calc", hessian_recalc=None,
                              fd_step=1.0e-3, fd_npoints=4, hessian_step=5.0e-3, verbose=True,
                              **optimizer_kwargs):
    """First-order saddle point search with pysisyphus' restricted-step partitioned RFO
    (RS-P-RFO), maximizing along one Hessian eigenvector and minimizing along the rest.

    Arguments:
    ----------
    molecule : Molecule -> guess of the transition-state geometry (symmetry=False)
    energy, gradient : see the module docstring
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

    calc = make_calculator(molecule, energy, gradient, fd_step=fd_step, fd_npoints=fd_npoints,
                           hessian_step=hessian_step, verbose=verbose)
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


def ccsd_energy_and_gradient(nfrozen=0, scf_kwargs=None, cc_options=None):
    """(energy, gradient) callables of RHF-CCSD with the analytic CCSD gradient, for use as
    `energy` and `gradient` of optimize_geometry / optimize_transition_state."""
    from ccpy import Driver
    from ccpy.scf import RHF

    scf_kwargs = {"verbose": 0, "e_tol": 1.0e-12, "d_tol": 1.0e-10, **(scf_kwargs or {})}
    cc_options = {"energy_convergence": 1.0e-12, "amp_convergence": 1.0e-10, **(cc_options or {})}

    def _run(mol):
        mf = RHF(mol, **scf_kwargs)
        mf.kernel()
        driver = Driver.from_scf(mf, nfrozen=nfrozen)
        driver.options.update(cc_options)
        driver.run_cc(method="ccsd")
        return driver

    def energy(mol):
        driver = _run(mol)
        return driver.system.reference_energy + driver.correlation_energy

    def gradient(mol):
        driver = _run(mol)
        driver.run_hbar(method="ccsd")
        driver.run_leftcc(method="left_ccsd")
        return driver.system.reference_energy + driver.correlation_energy, driver.run_gradient()

    return energy, gradient
