"""CC(P) / CC(P;Q) analytic gradients of ccpy.gradients.ccpq (separate stationary functionals,
four-center integrals, exact determinant-space amplitude stage) against finite differences
and against the existing CCSD and CR-CC(2,3)_A gradients."""
import numpy as np
import pytest

from ccpy import CCSD, CRCC23, Molecule, RHF
from ccpy.gradients.ccpq import CCPQGradient, spaces
from ccpy.gradients.ccpq.scf import run_rhf
from ccpy.gradients.finite_difference import fd_gradient

DISTORTED_WATER = "O 0.0 0.05 0.1173; H 0.0 0.7572 -0.4692; H 0.1 -0.7572 -0.4692"


@pytest.fixture(scope="module")
def water():
    return Molecule(DISTORTED_WATER, "sto-3g", cholesky_tei=True, cholesky_tol=1.0e-13, verbose=False)


def sd_plus_triples(S):
    return spaces.union(spaces.ranks(S, (1, 2)), spaces.ranks(S, (3,))[::7])


def nonclosed(S):
    # all singles and doubles but one, plus three quadruples: Lambda != Lambda_grad (Eq. 2.9)
    SD = spaces.ranks(S, (1, 2))
    return spaces.union(np.setdiff1d(SD, SD[:1]), spaces.ranks(S, (4,))[:3])


CASES = {
    "ccsd": ((1, 2), None, "mp"),
    "ccsdt": ((1, 2, 3), None, "mp"),
    "crcc23_A": ((1, 2), (3,), "mp"),
    "crcc23_full": ((1, 2), (3,), "full"),
    "ccp": (sd_plus_triples, None, "mp"),
    "ccp3_A": (sd_plus_triples, (3,), "mp"),
    "ccp3_full": (sd_plus_triples, (3,), "full"),
    "ccpq_tq_A": (sd_plus_triples, (3, 4), "mp"),
    "nonclosed_full": (nonclosed, (1, 2, 3, 4), "full"),
    "nonclosed_mp": (nonclosed, (1, 2, 3, 4), "mp"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_gradient_vs_finite_difference(water, name):
    p, q, den = CASES[name]
    calc = CCPQGradient(water, pspace=p, qspace=q, denominator=den, nfrozen=1)
    grad = calc.gradient()
    checks = calc.stages["amplitudes"].checks
    for key, value in checks.items():
        if key not in ("lambda_minus_lambda_grad", "eq_2_9_defect"):
            assert value < 1.0e-10, (key, value)
    if name.startswith("nonclosed"):
        assert checks["lambda_minus_lambda_grad"] > 1.0e-7
    fd = fd_gradient(calc.energy, water, step=1.0e-3, npoints=4, verbose=False)
    assert np.allclose(grad, fd, atol=1.0e-8)
    assert np.allclose(grad.sum(axis=0), 0.0, atol=1.0e-10)


def test_fixed_orbital_density(water):
    """Sec. 9.2, test 3: reconverge the amplitudes at h + eta dh, (pq|rs) + eta d(pq|rs) and
    compare with D.dh + 1/2 G.d(pr|qs), separately for the base and the correction."""
    calc = CCPQGradient(water, pspace=nonclosed, qspace=(3, 4), denominator="mp", nfrozen=1)
    h, eri = run_rhf(water).mo_integrals()
    rng = np.random.default_rng(71)
    dh = rng.standard_normal(h.shape)
    dh = 0.01 * (dh + dh.T)
    dv = 0.01 * rng.standard_normal(eri.shape)
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        dv = dv + dv.transpose(perm)
    amp = calc.backend.run(h, eri)
    eta = 1.0e-4
    plus = calc.backend.run(h + eta * dh, eri + eta * dv)
    minus = calc.backend.run(h - eta * dh, eri - eta * dv)
    for attr, rho in (("energy_base", amp.rho_base), ("delta", amp.rho_corr)):
        fd = (getattr(plus, attr) - getattr(minus, attr)) / (2 * eta)
        assert np.isclose(fd, rho.energy(dh, dv), atol=1.0e-9)


@pytest.mark.parametrize("variant", ["ccsd", "crcc23_A"])
def test_against_existing_gradients(water, variant):
    mf = RHF(water, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    opts = dict(amp_print_threshold=1.0, energy_convergence=1.0e-12, amp_convergence=1.0e-10)
    method = CCSD(mf, nfrozen=1, **opts) if variant == "ccsd" else CRCC23(mf, variant="A", nfrozen=1, **opts)
    energy, grad = method.energy_and_gradient()
    p, q, den = CASES[variant]
    calc = CCPQGradient(water, pspace=p, qspace=q, denominator=den, nfrozen=1)
    assert np.allclose(calc.gradient(), grad, atol=1.0e-10)
    assert np.isclose(calc.e_tot, energy, atol=1.0e-10)
