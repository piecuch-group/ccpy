"""Second-order (TRAH) SCF, the orbital Hessian, and stability analysis."""
import numpy as np
import pytest

from ccpy.integrals import Molecule
from ccpy.scf import RHF, RKS, UHF, UKS
from ccpy.scf.trah import OrbitalHessian

H2O = "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692"
CH2 = "C 0 0 0.1; H 0 0.98 -0.6; H 0 -0.98 -0.6"
AUX = "def2-universal-jkfit"


@pytest.mark.parametrize("method, geometry, multiplicity, options", [
    (RHF, H2O, 1, {}),
    (UHF, CH2, 3, {}),
    (RKS, H2O, 1, dict(xc="b3lyp")),
    (UKS, CH2, 3, dict(xc="wb97x")),
])
def test_trah_matches_diis(method, geometry, multiplicity, options):
    mol = Molecule(geometry, "cc-pvdz", auxiliary_basis=AUX, multiplicity=multiplicity, symmetry=True, verbose=False)
    diis = method(mol, trah_after=None, verbose=0, **options)
    trah = method(mol, trah_after=0, verbose=0, **options)
    assert np.isclose(trah.kernel(), diis.kernel(), atol=1.0e-9)
    assert trah.determinant_symmetry == diis.determinant_symmetry
    for e_trah, e_diis in zip(trah.eps, diis.eps):
        assert np.allclose(e_trah, e_diis, atol=1.0e-6)


@pytest.mark.parametrize("method, geometry, multiplicity, options", [
    (RHF, H2O, 1, {}), (UKS, CH2, 3, dict(xc="b3lyp")), (RKS, H2O, 1, dict(xc="tpss")),
])
def test_orbital_hessian_finite_differences(method, geometry, multiplicity, options):
    mol = Molecule(geometry, "cc-pvdz", auxiliary_basis=AUX, multiplicity=multiplicity, verbose=False)
    mf = method(mol, verbose=0, **options)
    mf.kernel()
    H, S, df = mol.ints_hcore(), mol.ints_overlap(), mol.df
    mf.C = OrbitalHessian(mf, H, df, S).rotate(0.05 * np.random.default_rng(2).normal(
        size=len(OrbitalHessian(mf, H, df, S).gradient)))  # away from the minimum
    hess = OrbitalHessian(mf, H, df, S)
    v = np.random.default_rng(3).normal(size=len(hess.gradient))
    v /= np.linalg.norm(v)
    h = 3.0e-4
    energy = {}
    for t in (-1, 1):
        mf.C = hess.rotate(t * h * v)
        energy[t] = OrbitalHessian(mf, H, df, S).energy
    assert np.isclose(hess.gradient @ v, (energy[1] - energy[-1]) / (2 * h), rtol=1.0e-6, atol=1.0e-8)
    assert np.isclose(v @ hess(v), (energy[1] - 2 * hess.energy + energy[-1]) / h**2, rtol=1.0e-5)


# references: PySCF 2.14 density-fitted UHF reached by following its RHF external instability
@pytest.mark.parametrize("geometry, e_uhf", [("H 0 0 0; H 0 0 3.0", -0.99874173), ("N 0 0 0; N 0 0 2.4", -108.77830540)])
def test_rhf_to_uhf_instability(geometry, e_uhf):
    mol = Molecule(geometry, "cc-pvdz", auxiliary_basis=AUX, verbose=False)
    mf = RHF(mol, verbose=0)
    mf.kernel()
    w_external, uhf = mf.stability(external=True, follow=True)
    assert w_external[0] < -0.1
    assert type(uhf) is UHF
    assert np.isclose(uhf.e_tot, e_uhf, atol=1.0e-7)
    assert uhf.stability()[0][0] > 0.0
