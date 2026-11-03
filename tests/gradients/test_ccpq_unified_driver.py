"""Regression: one driver, published ground/excited moments, MP/EN, RDM, FD.
Adapted from ccpq_audit/test_unified_driver.py; run with pytest.
"""
import numpy as np
from ccpy.gradients.ccpq.driver import DeterminantBackend
from ccpy.gradients.ccpq.determinants import (DeterminantSpace, spin_orbital_integrals,
                               hamiltonian_matrix, fock_diagonal)
from ccpy.gradients.ccpq.spaces import ranks, complement
from ccpy.gradients.ccpq.amplitudes import SelectedCC


def make_eri(n,rng,scale=.13):
    L=rng.normal(size=(5,n,n))*scale
    L=(L+L.transpose(0,2,1))/2
    return np.einsum('kpq,krs->pqrs',L,L)


def check(norb,na,nb,seed,Pcase,variant):
    rng=np.random.default_rng(seed)
    s=DeterminantSpace(norb,na,nb)
    h=np.diag(np.linspace(-1.3,.75,norb))+0.01*rng.normal(size=(norb,norb))
    h=(h+h.T)/2
    eri=make_eri(norb,rng)
    dh=rng.normal(size=h.shape);dh=.3*(dh+dh.T)/2
    L=rng.normal(size=(3,norb,norb))*.05;L=(L+L.transpose(0,2,1))/2
    d_eri=np.einsum('kpq,krs->pqrs',L,L)
    P=ranks(s,(1,)) if Pcase=='singles' else np.r_[ranks(s,(1,))[:1],ranks(s,(2,))]
    Q=complement(s,P,(1,2))
    if len(P)>7: P=P[:7];Q=complement(s,P,(1,2))
    hso,vso=spin_orbital_integrals(h,eri)
    H=hamiltonian_matrix(s,hso,vso)
    dhso,dvso=spin_orbital_integrals(dh,d_eri)
    dH=hamiltonian_matrix(s,dhso,dvso)
    for state in (0,1,2):
        backend=DeterminantBackend(norb,na,nb,0,P,Q,variant,state=state)
        base=backend.run(h,eri)
        assert backend.state == state
        predicted=(base.rho_base+base.rho_corr).energy(dh,d_eri)
        delta_pred=base.rho_corr.energy(dh,d_eri)
        eps=1e-5
        pp=DeterminantBackend(norb,na,nb,0,P,Q,variant,state=state).run(h+eps*dh,eri+eps*d_eri)
        mm=DeterminantBackend(norb,na,nb,0,P,Q,variant,state=state).run(h-eps*dh,eri-eps*d_eri)
        fd=(pp.energy_base+pp.delta-mm.energy_base-mm.delta)/(2*eps)
        fd_delta=(pp.delta-mm.delta)/(2*eps)
        err=abs(fd-predicted);err_delta=abs(fd_delta-delta_pred)
        print(f'{Pcase:9s} {variant:4s} state={state} delta={base.delta:+.7e} '
              f'gradient_error={err:.2e} delta_error={err_delta:.2e}')
        assert err<5e-7 and err_delta<5e-7
        if state==0:
            z=SelectedCC(s,H,P,Q,fock_diag=fock_diagonal(hso,vso,s.ref_occ),denominator=variant).run()
            assert abs(base.delta-z.delta)<1e-10
            assert abs(base.energy_base-z.energy)<1e-10
        else:
            assert base.checks['constraint']<1e-8
            assert base.checks['stationarity_corr']<1e-8
    return True


def test_no_q():
    rng=np.random.default_rng(45)
    s=DeterminantSpace(3,1,1)
    h=np.diag([-1.1,.1,.4]);eri=make_eri(3,rng)
    P=ranks(s,(1,2));Q=np.zeros(0,dtype=int)
    hs,vs=spin_orbital_integrals(h,eri);H=hamiltonian_matrix(s,hs,vs)
    energies=np.linalg.eigvalsh(H)
    for state in (0,1,2):
        stage=DeterminantBackend(3,1,1,0,P,Q,'full',state=state).run(h,eri)
        print('no Q, state',state,'E',stage.energy_base,'exact',energies[state])
        assert abs(stage.energy_base-energies[state])<1e-8
        assert stage.delta==0


import pytest

@pytest.mark.parametrize("variant", ["full", "mp"])
@pytest.mark.parametrize("selection,seed", [("singles", 51), ("nonclosed", 73)])
def test_selected_states(variant, selection, seed):
    check(3, 1, 1, seed, selection, variant)


@pytest.mark.parametrize("variant", ["full", "mp"])
@pytest.mark.parametrize("nonclosed", [False, True])
def test_independent_ground_adjoint(variant, nonclosed):
    from ccpy.gradients.ccpq import StateOracle
    from ccpy.gradients.ccpq.densities import SpinOrbitalDensities, spin_trace
    s = DeterminantSpace(3, 1, 1)
    h = np.diag([-1.2, .1, .55])
    eri = make_eri(3, np.random.default_rng(73))
    P = np.r_[ranks(s, (1,))[:1], ranks(s, (2,))] if nonclosed else ranks(s, (1,))
    Q = complement(s, P)
    hs, vs = spin_orbital_integrals(h, eri)
    H = hamiltonian_matrix(s, hs, vs)
    result = StateOracle(s, P, Q, denominator=variant).solve(H, fock_diagonal(hs, vs, s.ref_occ))
    stage = DeterminantBackend(3, 1, 1, 0, P, Q, variant).run(h, eri)
    assert result.E_base == pytest.approx(stage.energy_base, abs=1e-10)
    assert result.delta == pytest.approx(stage.delta, abs=1e-10)
    for gamma, Gamma, expected in ((result.gamma_base, result.Gamma_base, stage.rho_base),
                                   (result.gamma_corr, result.Gamma_corr, stage.rho_corr)):
        actual = spin_trace(SpinOrbitalDensities(gamma, Gamma), s.norb)
        np.testing.assert_allclose(actual.D, expected.D, atol=1e-10)
        np.testing.assert_allclose(actual.G, expected.G, atol=1e-10)
