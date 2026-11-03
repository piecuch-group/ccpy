"""CC(P) densities and gradient (P = SD + a list of triples), against the exact
determinant-space reference of ccsdt_p_harness.py.

The harness tests run without ccpy/density/ccsdt_p_rdm.py: they pin the density conventions
(the reference with T3 = L3 = 0 is calc_ccsd_rdm) and check the gradient plumbing with the
reference densities. The density tests are skipped until ccsdt_p_rdm.py exists (it must also
be listed in ccpy/density/meson.build to be importable).

    pytest -q tests/scf/test_ccsdt_p_harness.py
    pytest -q tests/scf/test_ccsdt_p_harness.py -k "rdm and ccsd"
"""
import importlib

import numpy as np
import pytest

from ccpy.density.ccsd_rdm import calc_ccsd_rdm, calc_rdm2_vvvv_slice
from ccpy.gradients.finite_difference import fd_gradient
from ccsdt_p_harness import CASES, PspaceCase, zeroed

_cases, _fd = {}, {}


def case(name):
    if name not in _cases:
        _cases[name] = PspaceCase(name)
    return _cases[name]


def fd(name):
    if name not in _fd:
        c = case(name)
        _fd[name] = fd_gradient(c.energy_at, c.mol, step=1.0e-3, npoints=4, verbose=False)
    return _fd[name]


def rdm_module():
    try:
        return importlib.import_module("ccpy.density.ccsdt_p_rdm")
    except ModuleNotFoundError:
        pytest.skip("ccpy/density/ccsdt_p_rdm.py is not written (or not in ccpy/density/meson.build) yet")


def assert_blocks(c, canon, part, tol=1.0e-10):
    errors = c.compare(canon, part)
    bad = {k: f"{e:.2e}" for k, e in errors.items() if e > tol}
    assert not bad, f"{c.name} {part}: blocks off by {bad}"


# ---------------------------------------------------------------------- harness
@pytest.mark.parametrize("name", list(CASES))
def test_reference_energy(name):
    c = case(name)
    assert abs(c.energy - c.oracle_energy) < 1.0e-9
    assert c.det.checks["lambda_minus_lambda_grad"] < 1.0e-10


@pytest.mark.parametrize("name", list(CASES))
def test_reference_reproduces_ccsd_rdm(name):
    c = case(name)
    r1, r2 = calc_ccsd_rdm(c.T, c.L, c.system)
    assert_blocks(c, c.canonical(r1, r2, lambda a, s: calc_rdm2_vvvv_slice(c.L, c.T, a, s)), "ccsd")


@pytest.mark.parametrize("name", list(CASES))
def test_gradient_plumbing_with_reference_densities(name):
    c = case(name)
    r1, r2, vvvv = c.to_blocks(c.ref["full"])
    assert np.allclose(c.gradient(rdm=(r1, r2), vvvv_slice=vvvv), fd(name), atol=1.0e-8)


# ---------------------------------------------------------------------- ccsdt_p_rdm.py
def _mine(c, T, L):
    m = rdm_module()
    r1, r2 = m.calc_ccsdt_p_rdm(T, L, c.t3_excitations, c.system)
    return c.canonical(r1, r2, lambda a, s: m.calc_ccsdt_p_rdm2_vvvv_slice(T, L, c.t3_excitations, a, s))


@pytest.mark.parametrize("name", list(CASES))
def test_rdm_ccsd_terms(name):
    """T3 = L3 = 0: only the CCSD terms survive."""
    c = case(name)
    assert_blocks(c, _mine(c, zeroed(c.T), zeroed(c.L)), "ccsd")


@pytest.mark.parametrize("name", list(CASES))
def test_rdm_without_l3(name):
    """L3 = 0: CCSD terms plus the terms with T3 and no L3."""
    c = case(name)
    assert_blocks(c, _mine(c, c.T, zeroed(c.L)), "no_l3")


@pytest.mark.parametrize("name", list(CASES))
def test_rdm_full(name):
    c = case(name)
    assert_blocks(c, _mine(c, c.T, c.L), "full")


@pytest.mark.parametrize("name", list(CASES))
def test_gradient_vs_finite_difference(name):
    rdm_module()
    c = case(name)
    assert np.allclose(c.gradient(), fd(name), atol=1.0e-8)
