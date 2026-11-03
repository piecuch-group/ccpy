"""Public IRC interface: real ammonia path and scalar/state/result contracts."""
from types import SimpleNamespace

import numpy as np
import pytest

from ccpy import IRCResult, Molecule, RHF, run_irc
from ccpy.methods.base import MethodBase
from ccpy.optimize import optimize_geometry, optimize_transition_state


@pytest.fixture
def molecule():
    return Molecule("H 0 0 0; H 0 0 .75", "sto-3g", verbose=False)


def test_ammonia_two_sided_irc(tmp_path, monkeypatch):
    pytest.importorskip("pysisyphus")
    monkeypatch.chdir(tmp_path)
    mol = Molecule("N 0 0 .10; H .95 0 0; H -.475 .8227 0; H -.475 -.8227 0",
                   "sto-3g", auxiliary_basis="def2-universal-jkfit", verbose=False)

    def evaluate(molecule):
        mf = RHF(molecule, e_tol=1.e-12, d_tol=1.e-10, verbose=0)
        energy = mf.kernel()
        assert mf.converged
        return energy, mf.gradient()

    ts = optimize_transition_state(mol, evaluate, thresh="gau_tight", verbose=False)
    assert ts.converged
    result = run_irc(ts.molecule, evaluate, step_length=.05, max_cycles=150,
                     rms_grad_thresh=5.e-4, displ="length", displ_length=.1,
                     out_dir=tmp_path / "nested" / "irc", prefix="nh3", verbose=False)
    assert isinstance(result, IRCResult) and result.converged
    assert set(result.branches) == {"forward", "backward"}
    assert result.start_energy == pytest.approx(ts.energy, abs=1.e-9)
    heights = []
    for direction, branch in result.branches.items():
        assert branch.coordinates.shape[1:] == (4, 3)
        assert branch.gradients.shape == branch.coordinates.shape
        assert len(branch.energies) == len(branch.coordinates)
        np.testing.assert_allclose([xyz for _, xyz in branch.molecule.atoms], branch.coordinates[-1])
        assert result.endpoints[direction] is branch.molecule
        assert branch.energies[-1] < result.start_energy
        assert branch.gradient_rms == pytest.approx(np.sqrt(np.mean(branch.gradients[-1] ** 2)))
        optimized = optimize_geometry(branch.molecule, evaluate, thresh="gau_tight", verbose=False)
        assert optimized.converged and optimized.energy < ts.energy
        xyz = np.array([xyz for _, xyz in optimized.molecule.atoms])
        normal = np.cross(xyz[2] - xyz[1], xyz[3] - xyz[1])
        heights.append(np.dot(xyz[0] - xyz[1], normal / np.linalg.norm(normal)))
    assert heights[0] * heights[1] < 0
    limited = run_irc(ts.molecule, evaluate, backward=False, max_cycles=1, step_length=.01,
                      rms_grad_thresh=1.e-12, energy_thresh=0., displ="length", displ_length=.1,
                      out_dir=tmp_path / "limited", verbose=False)
    assert not limited.converged and set(limited.branches) == {"forward"}


@pytest.fixture
def fake_integrator(monkeypatch):
    """Exercise public callback/state contracts without solving a fictitious PES."""
    pytest.importorskip("pysisyphus")
    from importlib import import_module
    module = import_module("pysisyphus.irc.EulerPC")

    class FakeIRC:
        def __init__(self, geometry, **kwargs):
            self.geometry, self.settings = geometry, kwargs

        def run(self):
            coords = self.geometry.cart_coords.copy()
            data = self.geometry.calculator.get_forces(self.geometry.atoms, coords)
            self.ts_energy = data["energy"]
            directions = ("downhill",) if self.settings["downhill"] else tuple(
                name for name in ("forward", "backward") if self.settings[name])
            for name in directions:
                for key, value in dict(coords=[coords], energies=[self.ts_energy],
                                       gradients=[-data["forces"]], is_converged=False,
                                       energy_converged=True, energy_increased=False, cycle=2).items():
                    setattr(self, f"{name}_{key}", value)

    monkeypatch.setattr(module, "EulerPC", FakeIRC)


def test_method_state_and_nonconvergence(molecule, fake_integrator, tmp_path):
    method = object.__new__(MethodBase)
    method.reference = SimpleNamespace(molecule=molecule)
    seen = []

    def at_geometry(mol):
        def energy(root=None):
            seen.append(root)
            return np.array([-1., -2.]) if root is None else -1. - root
        return SimpleNamespace(energy=energy, gradient=lambda root=None: np.full((2, 3), root),
                               converged=True)

    method.at_geometry = at_geometry
    with pytest.raises(ValueError, match="state="):
        run_irc(method, out_dir=tmp_path, verbose=False)
    result = run_irc(method, state=1, root=0, backward=False, out_dir=tmp_path, verbose=False)
    assert seen[-1] == 1 and result.irc.settings["root"] == 0
    assert set(result.branches) == {"forward"}
    assert not result.converged
    np.testing.assert_array_equal(result.branches["forward"].gradients, np.ones((1, 2, 3)))
    assert result.branches["forward"].ncycles == 3
    assert result.start_energy == -2.


@pytest.mark.parametrize("kwargs", [dict(forward=False, backward=False), dict(root=-1), dict(state=-1),
                                   dict(max_cycles=0), dict(hessian_step=0), dict(step_length=np.nan)])
def test_invalid_controls(molecule, kwargs):
    with pytest.raises(ValueError):
        run_irc(molecule, lambda m: (-1., np.zeros((2, 3))), **kwargs)


@pytest.mark.parametrize("energy,gradient", [(np.array([-1.]), np.zeros((2, 3))),
                                            (np.nan, np.zeros((2, 3))),
                                            (-1., np.zeros(6)), (-1., np.full((2, 3), np.inf))])
def test_invalid_callback(molecule, fake_integrator, tmp_path, energy, gradient):
    with pytest.raises(ValueError, match="scalar energy|gradient"):
        run_irc(molecule, lambda m: (energy, gradient), out_dir=tmp_path, verbose=False)


def test_downhill_and_callback_state(molecule, fake_integrator, tmp_path):
    def callback(m):
        return -1., np.zeros((2, 3))
    result = run_irc(molecule, callback, downhill=True, forward=False, backward=False,
                     out_dir=tmp_path, verbose=False)
    assert set(result.branches) == {"downhill"}
    with pytest.raises(ValueError, match="inside the callback"):
        run_irc(molecule, callback, state=0)


def test_symmetry_rejected(molecule):
    molecule.symmetry = True
    with pytest.raises(ValueError, match="symmetry=False"):
        run_irc(molecule, lambda m: (-1., np.zeros((2, 3))))
