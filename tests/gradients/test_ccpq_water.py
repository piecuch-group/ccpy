"""Water/6-31G nuclear derivative regression with an explicitly reduced core.

These quick checks freeze three occupied spatial orbitals (six electrons).
They exercise the same p-function AO derivatives and frozen/active orbital
response as the primary O(1s)-frozen validation, without its 245025 determinants.
The primary one-frozen-orbital results are in examples/results/h2o_631g_frozen_core.json.
"""
import os
import numpy as np
import pytest
from ccpy import Molecule
from ccpy.gradients.ccpq import CCPQGradient, spaces

pytestmark=pytest.mark.skipif(not os.environ.get('CCPY_FCIPY_EXTENSION'),reason='Optional FCIpy connector not configured')
GEOMETRY='O 0 0 0; H .05 -.75 .59; H -.03 .78 .62'


@pytest.mark.parametrize('state',[0,1,2])
@pytest.mark.parametrize('denominator',['mp','full'])
def test_water_reduced_correlation_gradient(state,denominator):
    mol=Molecule(GEOMETRY,'6-31g',verbose=False)
    def selected(s):
        return spaces.union(spaces.ranks(s,(1,2)), spaces.active_excitations(s,(3,4),[3,4],[5,6,7,8,9]))
    calc=CCPQGradient(mol,selected,(3,4),denominator,nfrozen=3,state=state,backend='matrix_free',sparse_cache=True)
    assert calc.backend.space.ndet==2025
    values=calc.check_gradient()  # All nine Cartesian components.
    for key,result in values.items():
        assert result['analytic'].shape == result['numerical'].shape == (9,)
        assert result['max_error']<2e-8,(key,result)
    assert abs(calc.stages['amplitudes'].delta)>1e-8
    for g in calc.gradients.values():
        np.testing.assert_allclose(g.sum(axis=0),0.,atol=1e-10)
