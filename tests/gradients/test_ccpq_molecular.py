"""End-to-end RHF response, overlap and nuclear-integral derivative checks."""
import numpy as np
import pytest

from ccpy import Molecule
from ccpy.gradients.ccpq import CCPQGradient, spaces
from ccpy.gradients.finite_difference import finite_difference


def selected(space):
    # Four electrons/four orbitals: actual connected triples and quadruples exist.
    return spaces.union(spaces.ranks(space, (1, 2)), spaces.ranks(space, (3,))[::3])


@pytest.mark.parametrize("state", [0, 1])
@pytest.mark.parametrize("denominator", ["full", "mp"])
@pytest.mark.parametrize("selection", [(1,), selected])
def test_nuclear_components(state, denominator, selection):
    mol = Molecule("H 0 0 0; H 0.1 0 1.5; H 0 0.2 3.1; H 0.2 0.1 4.8",
                   "sto-3g", symmetry=False, verbose=False)
    calc = CCPQGradient(mol, pspace=selection, qspace=(1, 2, 3, 4),
                        denominator=denominator, state=state)
    analytical = calc.gradient_components()
    xyz = np.array([atom[1] for atom in mol.atoms])

    def energies(z):
        displaced = xyz.copy()
        displaced[1, 2] = z
        values = calc.energy_components(mol.with_geometry(displaced, unit="bohr"))
        return np.array([values[key] for key in ("base", "correction", "total")])

    numerical = finite_difference(energies, xyz[1, 2], step=1e-3, npoints=4)
    for key, value in zip(("base", "correction", "total"), numerical):
        assert analytical[key][1, 2] == pytest.approx(value, abs=2e-7), key
        assert np.max(abs(analytical[key].sum(axis=0))) < 1e-9
    if selection is selected:
        assert np.any(calc.backend.space.rank[calc.backend.P] == 3)
        assert np.any(calc.backend.space.rank[calc.backend.Q] == 3)


@pytest.mark.parametrize("state", [0, 1])
def test_frozen_core_check_helper(state):
    mol = Molecule("Li 0 0 0; H 0.1 0.2 1.6", "sto-3g", symmetry=False, verbose=False)
    calc = CCPQGradient(mol, pspace=(1,), qspace=(2,), denominator="mp", nfrozen=1, state=state)
    checks = calc.check_gradient(components=[(1, 0), (1, 2)])
    for key, result in checks.items():
        assert result["max_error"] < 2e-7, (key, result)


def test_nonclosed_nuclear_gradient():
    mol = Molecule("H 0 0 0; H 0.1 0 1.5; H 0 0.2 3.1; H 0.2 0.1 4.8",
                   "sto-3g", symmetry=False, verbose=False)

    def nonclosed(space):
        return spaces.union(spaces.ranks(space, (1, 2))[1:], spaces.ranks(space, (3,))[::3])

    calc = CCPQGradient(mol, pspace=nonclosed, qspace=(1, 2, 3, 4), denominator="full", state=1)
    for key, result in calc.check_gradient(components=[(1, 2)]).items():
        assert result["max_error"] < 2e-7, (key, result)
