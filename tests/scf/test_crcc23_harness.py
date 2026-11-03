"""The factorized analytic CR-CC(2,3)_A gradient (ccpy/gradients/crcc23_gradient_reference.py), piece
by piece against the exact determinant-space reference (crcc23_harness.py), against
directional finite differences of ccpy's own code, and end to end against fd_gradient.

Each exact-reference check receives REFERENCE inputs, so an error in one piece does not
cascade into the next.

    pytest -q tests/scf/test_crcc23_harness.py
    pytest -q tests/scf/test_crcc23_harness.py -k s_lambda
"""
import re
from pathlib import Path

import numpy as np
import pytest

from ccpy.density.ccsd_rdm import calc_ccsd_rdm, calc_rdm2_vvvv_slice
from ccpy.density.crcc23_rdm import calc_crcc23_rdm
from ccpy.gradients import crcc23_gradient_reference as G
from ccpy.gradients.density_gradient import add_densities, build_DY, get_bare_integrals
from ccpy.gradients.finite_difference import fd_gradient
from crcc23_harness import ResponseCase

EXACT = ["rhf_h2o", "rohf_ch2", "uhf_ch2"]
LARGE = ["rhf_h2o_dz"]                     # finite-difference checks only (RHF: the gradient needs RHF)
_cases = {}


def case(name, reference=True):
    if name not in _cases:
        _cases[name] = ResponseCase(name, reference=reference)
    return _cases[name]


def assert_close(c, mine, name, tol=1.0e-10):
    errors = c.compare(mine, name)
    scale = max(np.abs(v).max() for v in c.ref[name].values())
    bad = {k: e for k, e in errors.items() if e > tol * max(scale, 1.0)}
    assert not bad, f"{name}: blocks off by {bad} (largest reference element {scale:.2e})"


def assert_DY_close(c, mine, piece, tol=1.0e-10):
    errors = c.compare_DY(mine, piece)
    assert max(errors.values()) < tol, f"density_{piece}: {errors}"


# --------------------------------------------------------------------------------------
# the harness itself: the exact reference passes its own finite-difference checks
# --------------------------------------------------------------------------------------
def test_harness_self_consistency():
    c, r = case("rhf_h2o"), case("rhf_h2o").ref
    assert c.check_s_lambda(r["s_lambda"]) < 1.0e-8
    assert c.check_s_t(r["s_t"]) < 1.0e-5
    assert c.check_BTw(r["BTw"], r["w"]) < 1.0e-6
    assert c.check_w(r["w"], r["s_lambda"]) < 1.0e-10
    assert c.check_z(r["z"], r["s_t"], r["BTw"]) < 1.0e-10
    assert np.allclose([r["s_t_R3"][k] for k in ("aa", "ab", "bb")], 0.0)   # R3 part is singles-only


def test_harness_exact_density_gradient():
    """Factorized CCSD densities + exact reference triples/z/w densities -> orbital relaxation
    (with the occupied-occupied / virtual-virtual multipliers) reproduce finite differences
    of E_CCSD + delta_A."""
    c = case("rhf_h2o")
    analytic = c.exact_gradient()
    numerical = fd_gradient(c.energy_function(), c.mol, step=1.0e-3, npoints=4, verbose=False)
    assert np.abs(analytic - numerical).max() < 1.0e-8


# --------------------------------------------------------------------------------------
# pieces against the exact reference
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", EXACT)
def test_1_r3_l3(name):
    c = case(name)
    H, H0, _ = c.transformed
    R3, L3 = G.calc_r3_l3(c.T, c.L, H, H0, c.system)
    assert_close(c, R3, "R3")
    assert_close(c, L3, "L3")


@pytest.mark.parametrize("name", EXACT)
def test_2_s_lambda(name):
    c = case(name)
    H, _, _ = c.transformed
    assert_close(c, G.calc_s_lambda(_triples(c.ref["R3"]), H, c.system), "s_lambda")


@pytest.mark.parametrize("name", EXACT)
def test_3_s_t(name):
    c = case(name)
    H, _, reference = c.transformed
    s_t = G.calc_s_t(_triples(c.ref["R3"]), _triples(c.ref["L3"]), c.T, c.L, H, reference)
    assert_close(c, s_t, "s_t")


@pytest.mark.parametrize("name", EXACT)
def test_4_BTw(name):
    c = case(name)
    H, _, reference = c.transformed
    assert_close(c, G.calc_BTw(c.ref["w"], c.T, c.L, H, reference), "BTw")


@pytest.mark.parametrize("name", EXACT)
def test_5a_solve_w(name):
    c = case(name)
    assert_close(c, G.solve_w(c.ref["s_lambda"], c.T, c.Hbar, c.system), "w", tol=1.0e-8)


@pytest.mark.parametrize("name", EXACT)
def test_5b_solve_z(name):
    c = case(name)
    rhs = {k: -(c.ref["s_t"][k] + c.ref["BTw"][k]) for k in ("a", "b", "aa", "ab", "bb")}
    assert_close(c, G.solve_z(rhs, c.T, c.L, c.Hbar, c.system), "z", tol=1.0e-8)


@pytest.mark.parametrize("name", EXACT)
def test_6_full_chain(name):
    """calc_crcc23A_response end to end (no reference inputs)."""
    c = case(name)
    H, H0, reference = c.transformed
    out = G.calc_crcc23A_response(c.T, c.L, c.Hbar, H, H0, reference)
    for key in ("s_lambda", "s_t", "w", "BTw", "z"):
        assert_close(c, out[key], key, tol=1.0e-8)


@pytest.mark.parametrize("name", EXACT)
def test_7a_density_triples(name):
    c = case(name)
    H, _, reference = c.transformed
    R3, L3 = _triples(c.ref["R3"]), _triples(c.ref["L3"])
    D1, Y = build_DY(*calc_crcc23_rdm(R3, L3, c.T, c.L, c.system), None, H.chol, c.T,
                     get_bare_integrals(reference)[1], reference)
    D1d, Yd = G.calc_denominator_density(R3, L3, H, reference)
    assert_DY_close(c, add_densities((D1, Y), (D1d, Yd)), "triples")


@pytest.mark.parametrize("name", EXACT)
def test_7b_density_z(name):
    c = case(name)
    H, _, reference = c.transformed
    z = _amplitudes(c.ref["z"])
    DY = build_DY(*calc_ccsd_rdm(c.T, z, c.system, energy=False), lambda a, s: calc_rdm2_vvvv_slice(z, c.T, a, s),
                  H.chol, c.T, get_bare_integrals(reference)[1], reference)
    assert_DY_close(c, DY, "z")


@pytest.mark.parametrize("name", EXACT)
def test_7c_density_w(name):
    c = case(name)
    H, _, reference = c.transformed
    assert_DY_close(c, G.calc_w_density(c.ref["w"], c.T, c.L, H, reference), "w")


def _reference_gradient(driver):
    """CR-CC(2,3)_A gradient of the reference implementation for a legacy Driver."""
    from ccpy.methods.reference import Reference

    reference = Reference(driver.scf, driver.system.nfrozen, driver.system.ndelete)
    reference.system = driver.system
    return G.calc_crcc23_gradient(driver.T, driver.L[0], driver.hamiltonian, reference, verbose=False)


def test_8_gradient_vs_finite_difference():
    """The complete analytic CR-CC(2,3)_A gradient against fd_gradient."""
    c = case("rhf_h2o")
    analytic = _reference_gradient(c.driver)
    assert np.abs(analytic - c.exact_gradient()).max() < 1.0e-9
    numerical = fd_gradient(c.energy_function(), c.mol, step=1.0e-3, npoints=4, verbose=False)
    assert np.abs(analytic - numerical).max() < 1.0e-8


def test_8b_gradient_cholesky_vs_finite_difference():
    """The same gradient with Cholesky vectors (pivots held fixed across the displacements)."""
    from ccpy import Driver, Molecule
    import ccpy.scf
    mol = Molecule("O 0 0.05 0.12; H 0 0.76 -0.47; H 0.1 -0.76 -0.47", "6-31g", cholesky_tei=True,
                   cholesky_tol=1.0e-8, verbose=False)
    mol.cholesky_pivots = mol.df.pivots

    def run(m):
        mf = ccpy.scf.RHF(m, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
        mf.kernel()
        d = Driver.from_scf(mf, nfrozen=1)
        d.options.update({"energy_convergence": 1.0e-13, "amp_convergence": 1.0e-11})
        d.run_cc(method="ccsd")
        d.run_hbar(method="ccsd")
        d.run_leftcc(method="left_ccsd")
        return d

    def energy(m):
        d = run(m)
        d.run_ccp3(method="crcc23")
        return d.system.reference_energy + d.correlation_energy + d.deltap3[0]["A"]

    analytic = _reference_gradient(run(mol))
    numerical = fd_gradient(energy, mol, step=1.0e-3, npoints=4, verbose=False)
    assert np.abs(analytic - numerical).max() < 1.0e-8


# --------------------------------------------------------------------------------------
# larger molecules: directional finite differences and equation residuals
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", LARGE)
def test_7_finite_difference_checks(name):
    c = case(name, reference=False)
    H, H0, reference = c.transformed
    out = G.calc_crcc23A_response(c.T, c.L, c.Hbar, H, H0, reference)
    assert c.check_s_lambda(out["s_lambda"]) < 1.0e-7
    assert c.check_s_t(out["s_t"]) < 1.0e-5
    assert c.check_BTw(out["BTw"], out["w"]) < 1.0e-6
    assert c.check_w(out["w"], out["s_lambda"]) < 1.0e-8
    assert c.check_z(out["z"], out["s_t"], out["BTw"]) < 1.0e-8


@pytest.mark.parametrize("name", LARGE)
def test_9_gradient_vs_finite_difference_large(name):
    """Analytic CR-CC(2,3)_A gradient of cc-pVDZ water against fd_gradient."""
    c = case(name, reference=False)
    analytic = _reference_gradient(c.driver)
    numerical = fd_gradient(c.energy_function(), c.mol, step=1.0e-3, npoints=4, verbose=False)
    assert np.abs(analytic - numerical).max() < 1.0e-7


# --------------------------------------------------------------------------------------
# factorization: no contraction forms an intermediate with four virtual indices
# --------------------------------------------------------------------------------------
VIRTUAL = set("abcdefgh")


def _intermediates(spec, o=6, v=30):
    ins, out = spec.split("->")
    ins = ins.split(",")
    ops = [np.empty([v if c in VIRTUAL else o for c in s]) for s in ins]
    path, _ = np.einsum_path(spec, *ops, optimize="greedy")
    cur, res = list(ins), []
    for step in path[1:]:
        picked = [cur[i] for i in sorted(step, reverse=True)]
        for i in sorted(step, reverse=True):
            cur.pop(i)
        idx = "".join(sorted(set("".join(picked)) & set("".join(cur) + out)))
        res.append(idx)
        cur.append(idx)
    return res


@pytest.mark.parametrize("module", ["ccpy/density/ccsd_rdm.py", "ccpy/density/crcc23_rdm.py",
                                    "ccpy/gradients/crcc23_gradient_reference.py", "ccpy/hbar/hbar_ccs_chol.py"])
def test_no_four_virtual_intermediates(module):
    source = (Path(__file__).parents[2] / module).read_text()
    specs = re.findall(r'np\.einsum\("([a-z,]+->[a-z]*)"', source)
    assert specs
    for spec in specs:
        if spec.count(",") < 2:
            continue
        for idx in _intermediates(spec):
            assert sum(c in VIRTUAL for c in idx) < 4 or idx == spec.split("->")[1], (spec, idx)


def _triples(d):
    from types import SimpleNamespace
    return SimpleNamespace(**d)


def _amplitudes(d):
    from types import SimpleNamespace
    return SimpleNamespace(**d)
