"""CC(P;Q) gradient kernels against the exact determinant-space reference of ccp3_harness.py;
P = SD + active-space triples, Q = the other triples, variants "A" (bare-Fock denominator) and
"D" (full diagonal).

The harness tests run now: they check the layouts (the reference T and Lambda reproduce ccpy's)
and the correction (delta equals ccpy's CCP3 with two_body_approx=False). The kernel tests are
skipped until ccpy/moments/ccp3_response.py and ccpy/density/ccp3_rdm.py exist (listed in their
meson.build files), and the gradient test until ccpy/gradients/ccp3_gradient.py does; each kernel
receives REFERENCE inputs, so an error in one kernel does not cascade.

    pytest -q tests/scf/test_ccp3_harness.py
    pytest -q tests/scf/test_ccp3_harness.py -k "s_t and A"
"""
import importlib

import numpy as np
import pytest

from ccp3_harness import CASES, CCP3Case
from ccpy.gradients.finite_difference import fd_gradient
from ccsdt_p_harness import _quiet

TOL = 1.0e-10
_cases, _fd = {}, {}


def case(name):
    if name not in _cases:
        _cases[name] = CCP3Case(name)
    return _cases[name]


def fd(name):
    if name not in _fd:
        c = case(name)
        _fd[name] = fd_gradient(c.energy_at, c.mol, step=1.0e-3, npoints=4, verbose=False)
    return _fd[name]


def kernel(module, name):
    try:
        return getattr(importlib.import_module(module), name)
    except ModuleNotFoundError:
        pytest.skip(f"{module} is not written (or not in its meson.build) yet")


def response(name):
    return kernel("ccpy.moments.ccp3_response", name)


def density(name):
    return kernel("ccpy.density.ccp3_rdm", name)


def assert_small(errors, what, tol=TOL):
    bad = {k: f"{e:.2e}" for k, e in errors.items() if e > tol}
    assert not bad, f"{what}: blocks off by {bad}"


NAMES = list(CASES)


# ---------------------------------------------------------------------- harness
@pytest.mark.parametrize("name", NAMES)
def test_reference_delta_and_layouts(name):
    c = case(name)
    assert abs(c.corr_det.delta - c.delta) < 1.0e-9
    assert_small(c.compare_p(c.T, c.to_p(c.det.t)), "T", 1.0e-9)
    assert_small(c.compare_p(c.L, c.to_p(c.det.lam)), "L", 1.0e-9)


# ---------------------------------------------------------------------- step 1
@pytest.mark.parametrize("name", NAMES)
def test_moments(name):
    c = case(name)
    M3, N3 = response("moments")(c.T, c.L, c.hamiltonian, c.t3_excitations, c.system)
    assert_small(c.compare_q(M3, "M3"), "M3")
    assert_small(c.compare_q(N3, "N3"), "N3")


def test_hbar_diagonal():
    c = case("nh3_ccp3_D")
    diag = response("hbar_diagonal")(c.T, c.hamiltonian, c.t3_excitations, c.system)
    assert_small(c.compare_q(diag, "hbar_diagonal"), "<K|Hbar_N|K>")


# ---------------------------------------------------------------------- step 2
@pytest.mark.parametrize("name", NAMES)
def test_s_lambda(name):
    c = case(name)
    x = response("s_lambda")(c.cref["R3"], c.T, c.L, c.hamiltonian, c.t3_excitations, c.system)
    assert_small(c.compare_p(x, "s_lambda"), "s_lambda")


@pytest.mark.parametrize("name", NAMES)
def test_s_t(name):
    c = case(name)
    x = response("s_t")(c.cref["R3"], c.cref["L3"], c.T, c.L, c.hamiltonian, c.t3_excitations, c.system, c.variant)
    assert_small(c.compare_p(x, "s_t"), "s_t")


@pytest.mark.parametrize("name", NAMES)
def test_k(name):
    c = case(name)
    x = response("k")(c.cref["sigma"], c.T, c.L, c.hamiltonian, c.t3_excitations, c.system)
    assert_small(c.compare_p(x, "k"), "k")


@pytest.mark.parametrize("action", ["B_action", "BT_action"])
def test_B_actions(action):
    c = case("nh3_ccp3_A")
    x = c.random_p()
    mine = response(action)(x, c.T, c.L, c.hamiltonian, c.t3_excitations, c.system, c.ecorr)
    assert_small(c.compare_p(mine, getattr(c, action)(x)), action)


# ---------------------------------------------------------------------- step 3
@pytest.mark.parametrize("name", NAMES)
def test_rdm_pi(name):
    c = case(name)
    r1, r2, vvvv = density("rdm_pi")(c.cref["pi"], c.T, c.t3_excitations, c.system)
    assert c.report(c.canonical(r1, r2, vvvv), "rdm_pi") < TOL


@pytest.mark.parametrize("name", NAMES)
def test_rdm_sigma(name):
    c = case(name)
    r1, r2, vvvv = density("rdm_sigma")(c.cref["sigma"], c.T, c.L, c.t3_excitations, c.system)
    assert c.report(c.canonical(r1, r2, vvvv), "rdm_sigma") < TOL


@pytest.mark.parametrize("name", NAMES)
def test_rdm_moments(name):
    c = case(name)
    r1, r2, vvvv = density("rdm_moments")(c.cref["R3"], c.cref["L3"], c.T, c.L, c.t3_excitations, c.system,
                                          c.variant)
    assert c.report(c.canonical(r1, r2, vvvv), "rdm_moments") < TOL


# ---------------------------------------------------------------------- steps 1-5
@pytest.mark.parametrize("name", NAMES)
def test_gradient_vs_finite_difference(name):
    response("moments")
    density("rdm_pi")
    kernel("ccpy.density.ccsdt_p_rdm", "calc_ccsdt_p_rdm")
    ccp3_gradient = kernel("ccpy.gradients.ccp3_gradient", "ccp3_gradient")
    c = case(name)
    assert np.allclose(_quiet(ccp3_gradient, c.ccp3), fd(name), atol=1.0e-8)
