"""Gradients, minima, ammonia inversion TS and two-sided IRC validation.

Examples (run from an installed CCpy checkout):
  python examples/methods_api/geometry_workflows.py --task gradient-check --method ccsd
  python examples/methods_api/geometry_workflows.py --task minimum --method ccsd
  python examples/methods_api/geometry_workflows.py --task irc --method rhf

The default STO-3G examples demonstrate the interface, not quantitative barriers.
All pysisyphus files are written inside --out-dir. Each IRC includes a TS search,
projected frequency checks and reoptimization/characterization of both endpoints.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from ccpy import CCSD, CRCC23, MP2, Molecule, RHF, run_irc
from ccpy.optimize import make_calculator, optimize_geometry, optimize_transition_state


TIGHT = dict(amp_convergence=1.e-10, energy_convergence=1.e-12, maximum_iterations=150)


def electronic_method(molecule, level):
    mf = RHF(molecule, e_tol=1.e-12, d_tol=1.e-10, verbose=0)
    if level == "rhf":
        return mf
    cls = {"ccsd": CCSD, "crcc23a": CRCC23, "mp2": MP2}[level]
    extra = {"variant": "A"} if level == "crcc23a" else {}
    return cls(mf, nfrozen=1, **TIGHT, **extra)


def evaluator(level, finite_difference=False):
    """Scalar energy/Cartesian gradient callback, with convergence checks."""
    def evaluate(molecule):
        method = electronic_method(molecule, level)
        if level == "rhf":
            energy = method.kernel()
            if finite_difference:
                from ccpy.gradients.finite_difference import fd_gradient
                gradient = fd_gradient(lambda m: electronic_method(m, level).kernel(), molecule,
                                       step=1.e-3, npoints=4, verbose=False)
            else:
                gradient = method.gradient()
        elif finite_difference:
            energy, gradient = method.energy(), method.fd_gradient()
        else:
            energy, gradient = method.energy_and_gradient()
        if not method.converged:
            raise RuntimeError(f"Unconverged {level} calculation")
        return float(energy), np.asarray(gradient)
    return evaluate


def pysis_geometry(molecule, function, hessian_step=5.e-3):
    """Use public pysisyphus Geometry and CCpy's calculator bridge."""
    from pysisyphus.Geometry import Geometry
    from ccpy.integrals.atom_data import Z_TO_ATOM_SYMBOL
    atoms = [Z_TO_ATOM_SYMBOL[int(z)] for z in molecule.atomic_charges]
    coords = np.array([xyz for _, xyz in molecule.atoms]).ravel()
    geom = Geometry(atoms, coords, coord_type="cart")
    geom.set_calculator(make_calculator(molecule, function, hessian_step=hessian_step, verbose=False))
    return geom


def characterize(molecule, function, hessian_step=5.e-3):
    """Projected, mass-weighted vibrational frequencies, not Cartesian eigenvalues."""
    geom = pysis_geometry(molecule, function, hessian_step)
    frequencies, _, _, _ = geom.get_normal_modes()
    return geom, frequencies


def signed_height(molecule):
    """N distance from the ordered H plane; labels must remain unchanged."""
    xyz = np.array([xyz for _, xyz in molecule.atoms])
    normal = np.cross(xyz[2] - xyz[1], xyz[3] - xyz[1])
    return float(np.dot(xyz[0] - xyz[1], normal / np.linalg.norm(normal)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("gradient-check", "minimum", "ts", "irc"), default="minimum")
    parser.add_argument("--method", choices=("rhf", "ccsd", "crcc23a", "mp2"), default="rhf")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--finite-difference", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("geometry-run"))
    args = parser.parse_args()
    if args.task == "gradient-check" and args.method == "mp2":
        parser.error("MP2 has no analytic gradient here; choose rhf, ccsd or crcc23a for this comparison")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    geometry = ("O 0 0 .12; H 0 .80 -.47; H 0 -.78 -.45" if args.task in ("minimum", "gradient-check")
                else "N 0 0 .10; H .95 0 0; H -.475 .8227 0; H -.475 -.8227 0")
    mol = Molecule(geometry, args.basis, auxiliary_basis="def2-universal-jkfit",
                   symmetry=False, verbose=False)
    function = evaluator(args.method, args.finite_difference)
    if args.task == "gradient-check":
        energy, analytic = evaluator(args.method)(mol)
        _, numerical = evaluator(args.method, True)(mol)
        report = {"energy": energy, "max_gradient_difference": float(np.max(np.abs(analytic - numerical)))}
    elif args.task == "minimum":
        result = optimize_geometry(mol, function, thresh="gau_tight", verbose=False)
        if not result.converged:
            raise RuntimeError("Minimum optimization did not converge")
        _, frequencies = characterize(result.molecule, function)
        if np.any(frequencies < -20.):
            raise RuntimeError("Optimized geometry has a significant imaginary frequency")
        report = {"energy": result.energy, "frequencies_cm-1": frequencies.tolist()}
    else:
        ts = optimize_transition_state(mol, function, thresh="gau_tight", root=0,
                                       hessian_init="calc", hessian_step=5.e-3, verbose=False)
        if not ts.converged:
            raise RuntimeError("TS optimization did not converge")
        _, frequencies = characterize(ts.molecule, function)
        if np.count_nonzero(frequencies < -20.) != 1:
            raise RuntimeError("TS must have exactly one significant imaginary vibrational mode")
        report = {"ts_energy": ts.energy, "ts_frequencies_cm-1": frequencies.tolist()}
        if args.task == "irc":
            irc = run_irc(ts.molecule, function, forward=True, backward=True, hessian_init="calc",
                          step_length=.05, max_cycles=150, rms_grad_thresh=5.e-4,
                          displ="length", displ_length=.1, out_dir=".", prefix="ammonia", verbose=False)
            report["branches"] = {}
            heights = []
            for direction in ("forward", "backward"):
                branch = irc.branches[direction]
                if not branch.converged:
                    raise RuntimeError(f"{direction} IRC exhausted its cycle limit")
                endpoint = branch.molecule
                minimum = optimize_geometry(endpoint, function, thresh="gau_tight", verbose=False)
                if not minimum.converged or minimum.energy >= ts.energy:
                    raise RuntimeError("Endpoint must converge below the TS energy")
                _, freqs = characterize(minimum.molecule, function)
                if np.any(freqs < -20.):
                    raise RuntimeError("IRC endpoint did not optimize to a minimum")
                height = signed_height(minimum.molecule)
                heights.append(height)
                report["branches"][direction] = {"energy": minimum.energy, "signed_height_bohr": height,
                                                 "frequencies_cm-1": freqs.tolist()}
            if heights[0] * heights[1] >= 0:
                raise RuntimeError("The ammonia endpoints must have opposite inversion signs")
    Path("summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
