"""Native density-fitted Kohn-Sham DFT (RKS/UKS) and the AO gradients on grid points."""
import numpy as np
import pytest

from ccpy.integrals import Molecule, integrals
from ccpy.scf import RKS, UKS

H2O = "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692"
CH2 = "C 0 0 0.1; H 0 0.98 -0.6; H 0 -0.98 -0.6"
AUX = "def2-universal-jkfit"


def test_ao_derivatives_match_finite_differences():
    mol = Molecule(H2O, "cc-pvqz", auxiliary_basis="cc-pvqz-jkfit", verbose=False)  # up to g
    points = np.random.default_rng(1).normal(scale=1.5, size=(40, 3))
    ao1 = integrals.basis_at_points(mol, points, deriv=1)
    ao2 = integrals.basis_at_points(mol, points, deriv=2)
    assert np.allclose(ao1[0], integrals.basis_at_points(mol, points), atol=1.0e-14)
    assert np.allclose(ao2[:4], ao1, atol=1.0e-14)
    h = 1.0e-5
    hessian = {(0, 0): 4, (0, 1): 5, (0, 2): 6, (1, 1): 7, (1, 2): 8, (2, 2): 9}
    for k, step in enumerate(h * np.eye(3)):
        plus = integrals.basis_at_points(mol, points + step, deriv=1)
        minus = integrals.basis_at_points(mol, points - step, deriv=1)
        assert np.allclose(ao1[k + 1], (plus[0] - minus[0]) / (2 * h), atol=1.0e-8)
        for i in range(k + 1):
            assert np.allclose(ao2[hessian[i, k]], (plus[1 + i] - minus[1 + i]) / (2 * h), atol=1.0e-7)


# references: PySCF 2.14 dft.RKS/UKS(mol).density_fit(auxbasis="def2-universal-jkfit")
# with its default grids (level 3, NWChem pruning, as ccpy)
@pytest.mark.parametrize("xc, energy", [
    ("svwn", -75.8547177066),     # LDA
    ("pbe", -76.3334718356),      # GGA
    ("b3lyp", -76.4203858909),    # global hybrid
    ("r2scan", -76.3813954050),   # meta-GGA
    ("wb97x", -76.4003922440),    # long-range corrected (erf) hybrid
    ("hse06", -76.3452138379),    # screened (erfc) hybrid
])
def test_rks_h2o(xc, energy):
    mol = Molecule(H2O, "cc-pvdz", auxiliary_basis=AUX, verbose=False)
    assert np.isclose(RKS(mol, xc=xc, verbose=0).kernel(), energy, atol=1.0e-08)


@pytest.mark.parametrize("xc, energy", [("pbe", -39.0713419700), ("b3lyp", -39.1330905450)])
def test_uks_triplet_ch2(xc, energy):
    mol = Molecule(CH2, "cc-pvdz", auxiliary_basis=AUX, multiplicity=3, symmetry=True, verbose=False)
    mf = UKS(mol, xc=xc, verbose=0)
    assert np.isclose(mf.kernel(), energy, atol=1.0e-08)
    assert mf.determinant_symmetry == "b1"
    assert type(mf.with_molecule(mol)) is UKS


# references: PySCF 2.14 .nuc_grad_method() of the density-fitted RKS/UKS above (no grid
# response, as ccpy), at distorted geometries
H2O_DISTORTED = "O 0.0 0.05 0.1173; H 0.0 0.7572 -0.4692; H 0.1 -0.7572 -0.4692"
CH2_DISTORTED = "C 0 0 0.1; H 0 0.98 -0.6; H 0.05 -0.98 -0.62"


@pytest.mark.parametrize("method, geometry, multiplicity, xc, gradient", [
    (RKS, H2O_DISTORTED, 1, "b3lyp", [[-0.0028648066, 0.0696707289, -0.0218182731],
                                      [-0.0002140795, -0.0433169002, 0.0386117364],
                                      [0.0030773038, -0.0263533021, -0.0167925438]]),
    (RKS, H2O_DISTORTED, 1, "wb97x", [[-0.0034152637, 0.0712387074, -0.0161953736],
                                      [-0.0002010658, -0.0406783094, 0.0362120286],
                                      [0.0036192385, -0.0305859315, -0.0200225846]]),
    (UKS, CH2_DISTORTED, 3, "r2scan", [[-0.0033256710, 0.0029474446, 0.0921873352],
                                       [0.0008048913, 0.0305207573, -0.0446745826],
                                       [0.0025137267, -0.0334691805, -0.0474646686]]),
])
def test_ks_gradient(method, geometry, multiplicity, xc, gradient):
    mol = Molecule(geometry, "cc-pvdz", auxiliary_basis=AUX, multiplicity=multiplicity, verbose=False)
    mf = method(mol, xc=xc, e_tol=1.0e-12, d_tol=1.0e-9, verbose=0)
    mf.kernel()
    assert np.allclose(mf.gradient(), gradient, atol=1.0e-7)


@pytest.mark.parametrize("method, geometry, multiplicity, options", [
    (RKS, H2O_DISTORTED, 1, dict(xc="wb97x")),
    (UKS, CH2_DISTORTED, 3, dict(xc="b3lyp", trah_after=0)),
])
def test_direct_density_fitting_matches_stored(method, geometry, multiplicity, options):
    results = []
    for direct in (False, True):
        mol = Molecule(geometry, "cc-pvdz", auxiliary_basis=AUX, multiplicity=multiplicity, df_direct=direct, verbose=False)
        mf = method(mol, e_tol=1.0e-12, d_tol=1.0e-9, verbose=0, **options)
        results.append((mf.kernel(), mf.gradient()))
        assert mol.df.direct == direct
        if direct:
            assert mol.df._B_Pmn is None  # the B tensor is never built
    (e_stored, g_stored), (e_direct, g_direct) = results
    assert np.isclose(e_direct, e_stored, atol=1.0e-10)
    assert np.allclose(g_direct, g_stored, atol=1.0e-8)
