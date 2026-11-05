"""Relaxed multipole moments and harmonic IR intensities.

Examples:
  python examples/methods_api/multipole_ir.py --task moments --method ccsd --mode analytic
  python examples/methods_api/multipole_ir.py --task ir --method ccsd --mode analytic
  python examples/methods_api/multipole_ir.py --task ir --method mp2 --mode finite-field

Finite-field mode uses only scalar energies, with a fresh SCF in every field.
The helpers here are example callbacks, not additional exported MethodBase APIs.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from ccpy import CCSD, CRCC23, MP2, Molecule, RHF
from ccpy.gradients.derivative_integrals import multipole_integrals, multipole_moments
from ccpy.gradients.finite_difference import fd_gradient, finite_difference
from ccpy.gradients.hessian import vibrational_analysis
from ccpy.optimize import optimize_geometry


TIGHT = dict(amp_convergence=1.e-10, energy_convergence=1.e-12)


def make_method(molecule, level):
    mf = RHF(molecule, verbose=0, e_tol=1.e-12, d_tol=1.e-10)
    cls = {"ccsd": CCSD, "crcc23a": CRCC23, "mp2": MP2}[level]
    kwargs = {"variant": "A"} if level == "crcc23a" else {}
    return cls(mf, nfrozen=1, **TIGHT, **kwargs)


def scalar_energy(molecule, level):
    method = make_method(molecule, level)
    value = method.energy()
    if not (method.converged and method.reference.scf.converged):
        raise RuntimeError("Unconverged electronic energy")
    return float(value)


def finite_field_moments(molecule, energy, order=1, origin=None, field_step=1.e-3):
    """Nuclear minus electronic Cartesian moments from scalar energy derivatives.

    energy must rebuild the complete calculation on its supplied Molecule.
    All fields use one fixed geometry, AO basis, origin and operator stack.
    """
    _, operators = multipole_integrals(molecule, order=order, origin=origin)
    coords = np.array([xyz for _, xyz in molecule.atoms])

    def in_field(strengths):
        perturbed = molecule.with_geometry(coords, unit="bohr")
        h = perturbed.ints_hcore() + np.einsum("k,kmn->mn", strengths, operators)
        perturbed.ints_hcore = lambda: h
        return energy(perturbed)

    derivatives = finite_difference(in_field, np.zeros(len(operators)), step=field_step, npoints=4)
    return multipole_moments(derivatives, molecule, order=order, origin=origin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("moments", "ir"), default="moments")
    parser.add_argument("--method", choices=("ccsd", "crcc23a", "mp2"), default="ccsd")
    parser.add_argument("--mode", choices=("analytic", "finite-field"), default="analytic")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--order", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("multipole-run"))
    args = parser.parse_args()
    if args.mode == "analytic" and args.method == "mp2":
        parser.error("The MP2 convenience property API needs --mode finite-field")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.out_dir)
    molecule = Molecule("O 0 .05 .1173; H 0 .7572 -.4692; H .1 -.7572 -.4692",
                        args.basis, auxiliary_basis="def2-universal-jkfit", symmetry=False, verbose=False)
    template = make_method(molecule, args.method)
    if args.task == "moments":
        moments = (template.multipole_moments(order=args.order) if args.mode == "analytic" else
                   finite_field_moments(molecule, lambda mol: scalar_energy(mol, args.method), order=args.order))
        report = {"moments_au": {key: float(value) for key, value in moments.items()}}
    else:
        minimum = optimize_geometry(template, thresh="gau_tight", verbose=False)
        if not minimum.converged:
            raise RuntimeError("Optimization did not converge")
        molecule = minimum.molecule

        def gradient_and_dipole(mol):
            if args.mode == "analytic":
                method = make_method(mol, args.method)
                gradient, dipole = method.gradient_and_dipole()
                if not method.converged:
                    raise RuntimeError("Unconverged electronic calculation")
                return gradient, dipole
            def energy(m):
                return scalar_energy(m, args.method)
            gradient = fd_gradient(energy, mol, step=1.e-3, npoints=4, verbose=False)
            moments = finite_field_moments(mol, energy)
            return gradient, np.array([moments[c] for c in "xyz"])

        result = vibrational_analysis(molecule, gradient=gradient_and_dipole, dipole=True, step=2.5e-3)
        np.savez("harmonic_ir.npz", **result)
        report = {"frequencies_cm-1": result["frequencies"].tolist(),
                  "ir_intensities_km_mol": result["ir_intensities"].tolist(),
                  "zpe_Eh": float(result["zpe"])}
    Path("summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
