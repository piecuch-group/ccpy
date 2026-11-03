"""Reuse fixed Slater connectivity, refreshing every geometry's Hamiltonian."""
import os
import numpy as np
import pytest
from ccpy.gradients.ccpq import MatrixFreeBackend
from ccpy.gradients.ccpq.fcipy_adapter import FCIpyOperator
from ccpy.gradients.ccpq.determinants import DeterminantSpace,spin_orbital_integrals,hamiltonian_matrix

pytestmark = pytest.mark.skipif(not os.environ.get('CCPY_FCIPY_EXTENSION'), reason='Optional FCIpy connector not configured')


def integrals(n, seed=83):
    rng=np.random.default_rng(seed)
    h=np.diag(np.linspace(-1.6,.8,n));a=rng.normal(size=(n,n))*.01;h+=(a+a.T)/2
    L=rng.normal(size=(5,n,n))*.1;L=(L+L.transpose(0,2,1))/2
    return h,np.einsum('kpq,krs->pqrs',L,L)


@pytest.mark.parametrize('nfrozen',[0,1])
def test_refresh_numeric_values_retains_graph_and_structural_zeros(nfrozen):
    s=DeterminantSpace(4,2,2,nfrozen)
    h,eri=integrals(4)
    H=FCIpyOperator(s,np.diag(np.diag(h)),np.zeros_like(eri),sparse_cache=True)
    if not hasattr(H.lib.ccpq_fcipy,'update_connected_values'):
        pytest.skip('Rebuild the connector for numeric graph updates')
    indices,indptr,data=H.sparse.indices,H.sparse.indptr,H.sparse.data
    rng=np.random.default_rng(77);left,right=rng.normal(size=(2,s.ndet))
    before=H.transition_density(left,right)
    assert H.update_integrals(h,eri)
    assert H.sparse.indices is indices and H.sparse.indptr is indptr and H.sparse.data is data
    hs,vs=spin_orbital_integrals(h,eri)
    expected=hamiltonian_matrix(s,hs,vs)
    np.testing.assert_allclose(H.sparse.toarray(),expected,atol=1e-12)
    np.testing.assert_allclose(H(right),expected@right,atol=1e-12)
    after=H.transition_density(left,right)
    np.testing.assert_allclose(after.D,before.D,atol=1e-12)
    np.testing.assert_allclose(after.G,before.G,atol=1e-12)


def test_backend_refresh_detects_inplace_integral_changes(monkeypatch):
    h,eri=integrals(3)
    args=(3,1,1,0,(1,),(2,),'full')
    backend=MatrixFreeBackend(*args,state=1,sparse_cache=True)
    first=backend.run(h,eri)
    operator=backend._operator
    if not hasattr(operator.lib.ccpq_fcipy,'update_connected_values'):
        pytest.skip('Rebuild the connector for numeric graph updates')
    D=first.rho_base.D.copy();G=first.rho_corr.G.copy()
    h[:]+=1e-4*np.array([[.1,.2,-.1],[.2,.3,.05],[-.1,.05,-.1]])
    eri[:]*=1.01
    changed=backend.run(h,eri)
    assert backend._operator is operator
    independent=MatrixFreeBackend(*args,state=1,sparse_cache=True).run(h,eri)
    assert changed.energy_base==pytest.approx(independent.energy_base,abs=2e-10)
    assert changed.delta==pytest.approx(independent.delta,abs=2e-10)
    for a,b in [(changed.rho_base,independent.rho_base),(changed.rho_corr,independent.rho_corr)]:
        np.testing.assert_allclose(a.D,b.D,atol=2e-9)
        np.testing.assert_allclose(a.G,b.G,atol=2e-9)
    np.testing.assert_array_equal(first.rho_base.D,D)
    np.testing.assert_array_equal(first.rho_corr.G,G)
    def forbidden(*args,**kwargs):
        raise AssertionError('Identical integrals must not refresh numeric values')
    monkeypatch.setattr(operator,'update_integrals',forbidden)
    same=backend.run(h.copy(),eri.copy())
    assert same.energy_base==pytest.approx(changed.energy_base,abs=2e-10)
