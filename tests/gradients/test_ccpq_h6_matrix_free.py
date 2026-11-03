"""H6/6-31G regression with fixed selected P/Q and no orbital truncation.

The full 18-coordinate, two-step demonstration is
examples/ccpq_h6_631g_matrix_free.py. Here two independent Cartesian components
are checked for each denominator/state combination.
"""
import os
import numpy as np
import pytest

from ccpy import Molecule
from ccpy.gradients.ccpq import CCPQGradient
from ccpy.gradients.finite_difference import finite_difference
from ccpy.integrals import integrals

GEOMETRY = '; '.join(f'H {0.035*((i%3)-1):.6f} {0.025*((i%2)*2-1):.6f} {i*1.0:.6f}' for i in range(6))
P = np.array([1, 4, 9, 10, 13, 18, 219, 275, 334, 573, 740, 919, 1196, 2165])
Q = np.array([1192, 1237, 1240, 1300, 1307, 1408, 1624, 1732, 1735, 1737,
              2247, 2526, 2529, 4371, 5259, 6301, 6304])


def test_screened_four_center_derivative():
    """A screened quartet leaves stale higher derivative pointers in libint2.

    Tight and diffuse shells on separated atoms trigger screening in 6-31G;
    contraction must skip the entire quartet when the first target is null.
    This check does not require FCIpy or any coupled-cluster calculation.
    """
    mol = Molecule(GEOMETRY, '6-31g', verbose=False)
    xyz = np.array([a[1] for a in mol.atoms])
    rng = np.random.default_rng(414)
    weights = rng.normal(size=(mol.nbf,) * 4)
    analytic = integrals.coulomb_4c_deriv(mol, weights).reshape(6, 3)
    indices = [(1, 2), (2, 2), (3, 0)]
    def energy(coords):
        m = mol.with_geometry(coords, unit='bohr')
        return np.sum(weights * integrals.coulomb_4c(m))
    numerical = finite_difference(energy, xyz, step=1e-3, npoints=4, components=indices)
    np.testing.assert_allclose([analytic[i] for i in indices], numerical, atol=2e-9, rtol=0.)
    np.testing.assert_allclose(analytic.sum(axis=0), 0., atol=1e-12)


@pytest.mark.skipif(not os.environ.get('CCPY_FCIPY_EXTENSION'), reason='Optional FCIpy connector not configured')
@pytest.mark.parametrize('state', [0, 1, 2])
@pytest.mark.parametrize('denominator', ['mp', 'full'])
def test_h6_correction_nuclear_gradient(state, denominator):
    mol = Molecule(GEOMETRY, '6-31g', verbose=False)
    calc = CCPQGradient(mol, P, Q, denominator, backend='matrix_free', state=state)
    assert calc.backend.space.ndet == 48400
    assert calc.backend.space.nfrozen == 0
    checks = calc.check_gradient(components=[(1, 2), (3, 0)])
    amp = calc.stages['amplitudes']
    assert abs(amp.delta) > 1e-6  # Exclude a vacuous zero-correction test.
    assert np.any(calc.backend.space.rank[P] == 3)
    assert np.all(calc.backend.space.rank[Q] == 3)
    for key, result in checks.items():
        assert result['max_error'] < 2e-8, (key, result)
    for gradient in calc.gradients.values():
        np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-10)
