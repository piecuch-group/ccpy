"""Independent tensor-code checks of determinant DF/Cholesky derivatives."""
import contextlib
import copy
import io
import os
import numpy as np
import pytest
from ccpy import Molecule, RHF, CRCC23, CCSDT_P, CCP3, get_active_triples_pspace
from ccpy.gradients.ccpq import CCPQGradient, spaces
from ccpy.gradients.finite_difference import finite_difference

pytestmark=pytest.mark.skipif(not os.environ.get('CCPY_FCIPY_EXTENSION'),reason='Optional FCIpy connector not configured')
GEOMETRY='O 0 0 0; H .05 -.75 .59; H -.03 .78 .62'
OPTIONS=dict(energy_convergence=1e-13,amp_convergence=1e-11,amp_print_threshold=1.)


def meanfield(mol):
    mf=RHF(mol,verbose=0,e_tol=1e-13,d_tol=1e-11)
    mf.kernel()
    return mf


@pytest.fixture(params=[False,True],ids=['df','cholesky'])
def mol(request):
    return Molecule(GEOMETRY,'sto-3g',auxiliary_basis='def2-universal-jkfit',
        cholesky_tei=request.param,cholesky_tol=1e-13,verbose=False)


def test_crcc23_matches_native_analytic_gradient(mol):
    with contextlib.redirect_stdout(io.StringIO()):
        method=CRCC23(meanfield(mol),nfrozen=1,variant='A',**OPTIONS)
        energy,gradient=method.energy_and_gradient()
    calc=CCPQGradient(mol,(1,2),(3,),'mp',nfrozen=1,backend='matrix_free',
        sparse_cache=True,integral_model='factorized')
    g=calc.gradient_components()
    assert calc.e_tot==pytest.approx(energy,abs=1e-10)
    np.testing.assert_allclose(g['total'],gradient,atol=1e-10,rtol=0.)
    assert calc.stages['amplitudes'].delta==pytest.approx(method.deltas['A'],abs=1e-12)
    # The finite-difference API must preserve the factorized integral model.
    assert calc.energy()==pytest.approx(energy,abs=1e-10)


def test_full_selected_triples_matches_independent_optimized_finite_difference(mol):
    with contextlib.redirect_stdout(io.StringIO()):
        cr=CRCC23(meanfield(mol),nfrozen=1,**OPTIONS)
        cr.cc.ensure_solved()
        cr.system.set_active_space(1,1)
        triples=get_active_triples_pspace(cr.system)
    def optimized(coords):
        with contextlib.redirect_stdout(io.StringIO()):
            mf=meanfield(mol.with_geometry(coords,unit='bohr'))
            cc=CCSDT_P(mf,copy.deepcopy(triples),nfrozen=1,**OPTIONS)
            base=cc.energy()
            cp=CCP3(cc,two_body_approx=False,**OPTIONS)
            cp.energy()
            return np.array([base,cp.deltas['A'],cp.deltas['D']])
    xyz=np.array([a[1] for a in mol.atoms])
    reference=optimized(xyz)
    fd=finite_difference(optimized,xyz,step=1e-3,npoints=4,components=[(2,1)])[0]
    P=lambda s:spaces.union(spaces.ranks(s,(1,2)),spaces.from_t3_excitations(s,triples))
    calc=CCPQGradient(mol,P,(3,),'mp',nfrozen=1,backend='matrix_free',sparse_cache=True,integral_model='factorized')
    for column,denominator in [(1,'mp'),(2,'full')]:
        calc.backend.denominator=denominator
        g=calc.gradient_components()
        amp=calc.stages['amplitudes']
        assert mol.nuclear_repulsion+amp.energy_base==pytest.approx(reference[0],abs=1e-10)
        assert amp.delta==pytest.approx(reference[column],abs=1e-12)
        assert g['base'][2,1]==pytest.approx(fd[0],abs=1e-8)
        assert g['correction'][2,1]==pytest.approx(fd[column],abs=1e-9)
        assert g['total'][2,1]==pytest.approx(fd[0]+fd[column],abs=1e-8)


def test_factorized_excited_state_against_its_fitted_energy():
    mol=Molecule(GEOMETRY,'sto-3g',auxiliary_basis='def2-universal-jkfit',verbose=False)
    calc=CCPQGradient(mol,(1,2),(3,),'full',nfrozen=1,state=1,backend='matrix_free',
        sparse_cache=True,integral_model='factorized')
    results=calc.check_gradient(components=[(1,2),(2,1)])
    assert all(result['max_error']<2e-8 for result in results.values())
