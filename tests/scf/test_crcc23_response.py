"""Response theory of CR-CC(2,3)_A (CRCC23 Gradient Method.pdf, Eqs. 4-8) in an exact
determinant-space reference: d delta_A / dx from the w and z solves equals finite
differences with CCSD and Lambda reconverged."""
import numpy as np

from ccpy import Driver, Molecule, RHF
from crcc23_triples import SpinOrbitalSpace, spin_orbital_quantities
from determinant_space import (
    CCSDReference,
    DeterminantSpace,
    amplitudes_from_spin_orbital,
)


def test_crcc23A_correction_derivative_vs_finite_difference():
    mol = Molecule("O 0 0 0.12; H 0 0.76 -0.47; H 0 -0.76 -0.47", "sto-3g",
                   cholesky_tei=True, cholesky_tol=1.0e-12, verbose=False)
    mf = RHF(mol, verbose=0, e_tol=1.0e-12, d_tol=1.0e-10)
    mf.kernel()
    driver = Driver.from_scf(mf, nfrozen=1)
    driver.options.update({"energy_convergence": 1.0e-13, "amp_convergence": 1.0e-11})
    space = SpinOrbitalSpace(driver.system)
    f, v = space.one_body(driver.hamiltonian), space.two_body(driver.hamiltonian)
    driver.run_cc(method="ccsd")
    driver.run_hbar(method="ccsd")
    driver.run_leftcc(method="left_ccsd")
    driver.run_ccp3(method="crcc23")

    dets = DeterminantSpace(space.nocc, space.nvir)
    ref = CCSDReference(dets, f, v)
    q = spin_orbital_quantities(driver)
    t = ref.solve_ccsd(amplitudes_from_spin_orbital(dets, q["t1"], q["t2"]))
    lam = ref.solve_lambda(t)
    assert np.allclose(lam, amplitudes_from_spin_orbital(dets, q["l1"], q["l2"]), atol=1.0e-9)
    assert np.isclose(ref.crcc23_A(t, lam)[0], driver.deltap3[0]["A"], atol=1.0e-13)

    # random real perturbation with the permutational and spin symmetry of H
    rng = np.random.default_rng(7)
    n = dets.norb
    fV = rng.standard_normal((n, n))
    fV = 0.05 * (fV + fV.T) * ((np.abs(f) > 0) | np.eye(n, dtype=bool))
    g = rng.standard_normal((n,) * 4)
    g = g + g.transpose(1, 0, 3, 2)
    g = g + g.transpose(2, 3, 0, 1)
    vV = 0.02 * (g - g.transpose(0, 1, 3, 2)) * ((np.abs(v) > 0) | (np.abs(v.transpose(0, 1, 3, 2)) > 0))

    analytic, _ = ref.correction_derivative(t, lam, fV, vV)
    numerical = ref.correction_finite_difference(t, fV, vV, h=1.0e-4)
    assert np.isclose(analytic, numerical, atol=1.0e-10, rtol=0.0)
