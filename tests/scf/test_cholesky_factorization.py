"""Independent full-integral checks of the packed-pair Cholesky builder."""
import numpy as np
import pytest

from ccpy.integrals import Molecule, integrals
from ccpy.integrals.df import DensityFitting


def molecule(**kwargs):
    return Molecule('O 0 0 .1; H 0 .76 -.47; H .03 -.78 -.45', 'cc-pvdz',
                    cholesky_tei=True, verbose=False, **kwargs)


@pytest.mark.parametrize('tol', [1.e-5, 1.e-9])
def test_cholesky_full_eri_and_fixed_pivots(tol):
    mol = molecule(cholesky_tol=tol)
    df = DensityFitting(mol)
    B = df.B_Pmn
    assert B.flags.c_contiguous
    eri = integrals.coulomb_4c(mol)
    approx = np.einsum('Pmn,Prs->mnrs', B, B)
    # PSD residual plus a diagonal stopping criterion bounds every entry.
    assert np.max(np.abs(eri - approx)) <= tol * 1.01
    np.testing.assert_array_equal(B, B.transpose(0, 2, 1))
    assert mol.naux == len(df.pivots) == B.shape[0]
    # The inverse metric factor must map the raw pivot columns back to B.
    A = np.array([eri[:, :, r, s] for r, s in df.pivots])
    np.testing.assert_allclose((df.Mm12 @ A.reshape(len(A), -1)).reshape(B.shape),
                               B, atol=2.e-10, rtol=2.e-9)

    # Fixed pivots may use either orientation, including across different shells.
    mol.cholesky_pivots = [(s, r) for r, s in df.pivots]
    fixed = DensityFitting(mol)
    np.testing.assert_allclose(fixed.B_Pmn, B, atol=2.e-10, rtol=2.e-9)
    assert fixed.pivots == mol.cholesky_pivots


def test_shell_columns_against_full_eri():
    mol = molecule()
    eri = integrals.coulomb_4c(mol)
    shells = mol.basis.shell_first_and_size
    for s3, s4 in [(0, 1), (1, 0), (2, 2), (3, 4)]:
        r, nr = shells[s3]
        s, ns = shells[s4]
        col = integrals.coulomb_4c_shell_column(mol, s3, s4)
        np.testing.assert_allclose(col, eri[:, :, r:r+nr, s:s+ns], atol=2.e-14, rtol=2.e-13)


def test_empty_decomposition():
    mol = molecule(cholesky_tol=1.e10)
    df = DensityFitting(mol)
    assert df.B_Pmn.shape == (0, mol.nbf, mol.nbf)
    assert df.Mm12.shape == (0, 0)
    assert df.pivots == []
