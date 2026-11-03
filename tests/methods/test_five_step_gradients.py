"""Five-step DF analytic gradients (ccpq_analytical_gradients.pdf): CR-CC(2,3)_A
(ccpy/gradients/crcc23_gradient.py), CR-EOMCC(2,3)_A (ccpy/gradients/creomcc23_gradient.py) and
EOMCCSD (ccpy/gradients/eomccsd_gradient.py), through the method API (gradient()) and the kernels.

CR-CC(2,3)_A (RHF and UHF) is compared with the verified crcc23_gradient.py; CR-EOMCC(2,3)_A with central
differences of ccpy's own EOMCCSD and CREOMCC23 energies, base and correction separately, for
two roots of a nonplanar molecule (nonzero r0). The 2.5e-4 bohr step keeps the four-point
truncation error, which grows near the second root, below 1e-9 Eh/bohr.
"""
import contextlib
import io

import numpy as np
import pytest
from conftest import AUX, OPTS, converged_scf

from ccpy import CREOMCC23, CRCC23, EOMCCSD, Molecule, RHF, UHF
from ccpy.gradients import crcc23_gradient as v2
from ccpy.gradients import crcc23_gradient_reference as v1
from ccpy.gradients.creomcc23_gradient import calc_creomcc23_gradient
from ccpy.gradients.finite_difference import finite_difference
from ccpy.density.ccsd_rdm import calc_ccsd_rdm, calc_rdm2_vvvv_slice
from ccpy.gradients.density_gradient import build_DY, get_bare_integrals
from ccpy.hbar.hbar_ccs_chol import build_hbar_ccs_chol, build_hbar_ccs_chol_derivative

TIGHT = dict(OPTS, energy_convergence=1.0e-13, amp_convergence=1.0e-12)
WATER = "O 0.0 0.05 0.1173; H 0.0 0.7572 -0.4692; H 0.1 -0.7572 -0.4692"
NH3 = "N 0.0 0.0 0.12; H 0.0 0.94 -0.27; H 0.81 -0.47 -0.29; H -0.80 -0.45 -0.25"


def _quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


def _density_energy(DY, reference):
    """sum D1 . h + 1/2 sum Y . B (Euler's identity for amplitude-relaxed densities)."""
    D1, Y = DY
    h, B = get_bare_integrals(reference)
    return float(_pair(D1.a, h.a) + _pair(D1.b, h.b) + 0.5 * (_pair(Y.a, B.a) + _pair(Y.b, B.b)))


def _intermediates(T, H, constant=True):
    return {"aaa": v2.build_triples_intermediates_aaa(T, H, constant=constant),
            "aab": v2.build_triples_intermediates_aab(T, H, constant=constant),
            "abb": v2.build_triples_intermediates_abb(T, H, constant=constant),
            "bbb": v2.build_triples_intermediates_bbb(T, H, constant=constant)}


def _pair(X, M):
    return np.sum(X.oo * M.oo) + np.sum(X.ov * M.ov) + np.sum(X.vo * M.vo) + np.sum(X.vv * M.vv)


# ---------------------------------------------------------------------- CR-CC(2,3)
@pytest.fixture(scope="module", params=["rhf", "uhf"])
def crcc23(request):
    if request.param == "rhf":
        mf = converged_scf(RHF, Molecule(WATER, "6-31g", auxiliary_basis=AUX, verbose=False))
    else:
        mf = converged_scf(UHF, Molecule(WATER, "6-31g", auxiliary_basis=AUX, charge=1, multiplicity=2,
                                         verbose=False))
    method = CRCC23(mf, variant="A", nfrozen=1, **TIGHT)
    _quiet(method.energy)
    return mf, method


def test_crcc23_five_steps_reproduce_verified_gradient(crcc23):
    mf, m = crcc23
    g1 = v1.calc_crcc23_gradient(m.cc.T, m.left.L, m.hbar.H, m.reference, verbose=False)
    g2 = v2.calc_crcc23_gradient(m.cc.T, m.left.L, m.hbar.H, m.reference, delta=m.deltas["A"], components=True,
                                 verbose=False)
    assert np.allclose(g2["total"], g1, atol=1.0e-12)
    assert np.allclose(_quiet(m.gradient), g1, atol=1.0e-12)


def test_crcc23_triples_and_lagrangian_density(crcc23):
    mf, m = crcc23
    T, L, system = m.cc.T, m.left.L, m.system
    H = build_hbar_ccs_chol(T, m.reference.H, system)
    tr = v2.calc_triples(_intermediates(T, H), T, L, H, m.reference, *v2.zero_triples_density(system))
    assert abs(tr["delta"] - m.deltas["A"]) < 1.0e-12
    _, B = get_bare_integrals(m.reference)
    base = build_DY(*calc_ccsd_rdm(T, L, system), lambda a, s: calc_rdm2_vvvv_slice(L, T, a, s), H.chol, T, B,
                    m.reference, ref_weight=True)
    assert abs(_density_energy(base, m.reference) - (m.cc.energy() - mf.mol.nuclear_repulsion)) < 1.0e-10


# ---------------------------------------------------------------------- CR-EOMCC(2,3)
def _creomcc23(mol, nroots=2):
    mf = converged_scf(RHF, mol)
    eom = EOMCCSD(mf, nfrozen=1, **TIGHT)
    eom.initial_guess(method="cisd", multiplicity=1, nacto=2, nactu=2, roots_per_irrep={"A": nroots})
    method = CREOMCC23(eom, variant="A")
    _quiet(method.ensure_solved)
    return mf, eom, method


@pytest.fixture(scope="module")
def nh3_eom():
    mol = Molecule(NH3, "sto-3g", auxiliary_basis=AUX, verbose=False)
    mf, eom, m = _creomcc23(mol)

    def energies(X):
        _, e, mm = _creomcc23(mol.with_geometry(X, unit="bohr"))
        return np.array([[_quiet(e.energy)[k], mm.deltas[k]["A"]] for k in range(2)])

    X0 = np.array([a[1] for a in mol.atoms], dtype=float)
    fd = finite_difference(energies, X0, step=2.5e-4, npoints=4)          # (natoms, 3, root, base/corr)
    return mf, eom, m, fd


@pytest.mark.parametrize("root", [0, 1])
def test_creomcc23_moments(nh3_eom, root):
    mf, eom, m, _ = nh3_eom
    T, system, r0 = eom.cc.T, m.system, eom.r0[root]
    R, L = eom.R[root], m.left.L[root]
    assert abs(r0) > 1.0e-4
    H = build_hbar_ccs_chol(T, m.reference.H, system)
    H1 = build_hbar_ccs_chol_derivative(R, H, system)
    X = _intermediates(T, H)
    XH1 = _intermediates(T, H1)
    XR = _intermediates(R, H, constant=False)
    J = {c: [r0 * x + y + z for x, y, z in zip(X[c], XH1[c], XR[c])] for c in X}
    tr = v2.calc_triples(X, T, L, H, m.reference, *v2.zero_triples_density(system), omega=eom.omega[root], R=R,
                         r0=r0, J=J)
    assert abs(tr["delta"] - m.deltas[root]["A"]) < 1.0e-12


@pytest.mark.parametrize("root", [0, 1])
def test_creomcc23_gradient_vs_finite_difference(nh3_eom, root):
    _, eom, m, fd = nh3_eom
    g = _quiet(calc_creomcc23_gradient, eom.cc.T, eom.R[root], eom.r0[root], m.left.L[root], eom.omega[root],
               eom.hbar.H, m.reference, delta=m.deltas[root]["A"], components=True, verbose=False)
    assert np.allclose(g["base"], fd[:, :, root, 0], atol=1.0e-9)
    assert np.allclose(g["correction"], fd[:, :, root, 1], atol=1.0e-10)
    assert np.allclose(_quiet(m.gradient, root=root), fd[:, :, root].sum(axis=-1), atol=1.0e-9)


@pytest.mark.parametrize("root", [0, 1])
def test_eomccsd_gradient_vs_finite_difference(nh3_eom, root):
    _, eom, _, fd = nh3_eom
    assert eom.has_analytic_gradient(root=root)
    assert np.allclose(_quiet(eom.gradient, root=root), fd[:, :, root, 0], atol=1.0e-9)
