"""Small native-SCF recipes for the ccpy.methods guide.

Run from an installed checkout, for example:
    python examples/methods_api/run_calculation.py --recipe fno --basis 6-31g
    python examples/methods_api/run_calculation.py --recipe all

Energies are hartree. Defaults demonstrate interfaces, not benchmark convergence.
"""
import argparse
import json

import numpy as np

import ccpy as cc


RECIPES = ("ground", "eom", "ip", "ea", "adc", "fno", "selected", "steom", "response")
OPTIONS = dict(amp_convergence=1.e-8, energy_convergence=1.e-9, maximum_iterations=100)


def run(recipe, reference, parent):
    if recipe == "ground":
        result = cc.CRCC23(parent)
        value = result.energy()
    elif recipe in ("eom", "ip", "ea"):
        cls, guess = {"eom": (cc.EOMCCSD, "cisd"), "ip": (cc.IPEOM2, "ipcisd"),
                      "ea": (cc.EAEOM2, "eacisd")}[recipe]
        result = cls(parent)
        result.initial_guess(method=guess, nacto=2, nactu=2, roots_per_irrep={"A": 1},
                             multiplicity=1 if recipe == "eom" else -1)
        value = result.energy()
    elif recipe == "adc":
        result = cc.ADC2(reference, **OPTIONS)
        result.initial_guess(method="cis", multiplicity=1, roots_per_irrep={"A": 1})
        value = result.energy()
    elif recipe == "fno":
        # Explicit count makes the dimension reduction visible in a small example.
        nv = reference.system.nunoccupied_alpha
        result = cc.CCSD(reference, fno=cc.FNOOptions(nvirtuals=max(1, nv - 2)), **OPTIONS)
        value = result.energy()
        assert np.isclose(value, result.raw_energy() + result.fno_result.delta_ground)
    elif recipe == "selected":
        reference.system.set_active_space(nact_occupied=1, nact_unoccupied=1)
        triples = cc.get_active_triples_pspace(reference.system, num_active=1)
        result = cc.CCSDT_P(reference, triples, **OPTIONS)
        value = result.energy()
    elif recipe == "steom":
        result = cc.STEOMCC(parent, nroots=1, nacto=2, nactu=2,
                            ip_threshold=0., ea_threshold=0.)
        value = result.energy()
    elif recipe == "response":
        from ccpy.interfaces.scf_tools import get_multipole_integral
        operators, expectation = get_multipole_integral(1, reference)
        result = cc.LRCCSD(parent, operators[2])
        value = expectation[2] + result.property()  # electronic <sum z>, not physical dipole
    else:
        raise ValueError(recipe)
    if not result.converged:
        raise RuntimeError(f"{recipe} did not converge")
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"Nonfinite result from {recipe}")
    return np.asarray(value).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=RECIPES + ("all",), default="ground")
    parser.add_argument("--basis", default="6-31g")
    args = parser.parse_args()
    molecule = cc.Molecule("O 0 0 0; H 0 0 .96; H .93 0 -.24", args.basis,
                           auxiliary_basis="def2-universal-jkfit", verbose=False)
    mf = cc.RHF(molecule, verbose=0)
    reference = cc.Reference(mf, nfrozen=1)
    parent = cc.CCSD(reference, **OPTIONS)
    recipes = RECIPES if args.recipe == "all" else (args.recipe,)
    results = {recipe: run(recipe, reference, parent) for recipe in recipes}
    print("\nRecipe results (Eh; response is electronic position in bohr):")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
