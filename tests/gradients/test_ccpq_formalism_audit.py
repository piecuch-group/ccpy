"""Independent audit of gauge invariance, moments, denominators and orbital perturbations.

Adapted from ccpq_audit/test_formalism_audit.py; run with pytest.
This does NOT test SCF orbital response to nuclear displacements.
"""
import numpy as np
from scipy.linalg import expm
from ccpy.gradients.ccpq.state_oracle import StateOracle
from ccpy.gradients.ccpq.determinants import DeterminantSpace, spin_orbital_integrals, hamiltonian_matrix, fock_diagonal
from ccpy.gradients.ccpq.spaces import ranks, complement


def integrals(n,seed=65,scale=.20):
    rng=np.random.default_rng(seed)
    h=np.diag(np.linspace(-1.6,.8,n))
    a=rng.normal(size=(n,n))*.015
    h+=a+a.T
    v=rng.normal(size=(4,n,n))*scale
    v=(v+v.transpose(0,2,1))/2
    eri=np.einsum('kpq,krs->pqrs',v,v)
    return h,eri


def make_H(s,h,eri):
    hs,vs=spin_orbital_integrals(h,eri)
    return hamiltonian_matrix(s,hs,vs),fock_diagonal(hs,vs,s.ref_occ)


import pytest

@pytest.mark.parametrize("variant", ["full", "mp"])
@pytest.mark.parametrize("nonclosed", [False, True])
@pytest.mark.parametrize("state", [0, 1, 2])
def test_one(variant,nonclosed,state):
    s=DeterminantSpace(3,1,1)
    h,eri=integrals(3,seed=61 if nonclosed else 63)
    P=np.r_[ranks(s,(1,))[:1],ranks(s,(2,))] if nonclosed else ranks(s,(1,))
    Q=complement(s,P,(1,2))
    H,f=make_H(s,h,eri)
    def calc(c):
        return StateOracle(s,P,Q,state,variant,gauge_scale=c).solve(H,f)
    reference=calc(1)
    for gauge in (.2, 2., -3.0):
        r=calc(gauge)
        errE=abs(r.E-reference.E)
        errGamma=np.max(np.abs(r.gamma-reference.gamma))
        errG=np.max(np.abs(r.Gamma-reference.Gamma))
        errX=np.max(np.abs(r.gradient_H-reference.gradient_H))
        print('gauge',variant,'nonclosed',nonclosed,'state',state,'c',gauge,
              'dE',errE,'dRDM',max(errGamma,errG),'dgrad',errX)
        assert errE<2e-9 and max(errGamma,errG,errX)<2e-7
    # direct moment computation without the oracle's array contractions
    o=StateOracle(s,P,Q,state,variant)
    t,_,_,_,pivot=o.initial(H)
    r=reference.r;l=reference.l
    _,_,B=o.hbar(H,t)
    Ef=reference.E_base
    # full projected eigenvector is necessary, particularly for the right reference coefficient
    rhs=np.zeros(s.ndet);lhs=np.zeros(s.ndet)
    rhs[o.full]=r;lhs[o.full]=l
    moments=np.array([B[k,:]@rhs for k in Q]); left=np.array([lhs@B[:,k] for k in Q])
    if variant=='full': D=np.array([Ef-B[k,k] for k in Q])
    else: D=np.array([Ef-B[0,0]+o.chi[i]@f for i in range(len(Q))])
    explicit=np.sum(left*moments/D)
    print('moment',variant,'nonclosed',nonclosed,'state',state,'delta',reference.delta,'direct',explicit)
    assert abs(explicit-reference.delta)<1e-9
    # chain rule with the Hamiltonian perturbation obtained by rotating spatial orbitals
    rng=np.random.default_rng(100)
    k=rng.normal(size=(s.norb,s.norb));k=k-k.T;k*=.12
    def rotated(x):
        U=expm(x*k)
        hr=U.T@h@U
        vr=np.einsum('pa,qb,rc,sd,pqrs->abcd',U,U,U,U,eri,optimize=True)
        return hr,vr
    ep=1e-5
    hp,vp=rotated(ep);hm,vm=rotated(-ep)
    Hp,fp=make_H(s,hp,vp);Hm,fm=make_H(s,hm,vm)
    dp=(Hp-Hm)/(2*ep)
    # The MP denominator has explicit Fock dependence; use RDM integral contractions
    from ccpy.gradients.ccpq.densities import SpinOrbitalDensities,spin_trace
    rho=spin_trace(SpinOrbitalDensities(reference.gamma,reference.Gamma),s.norb)
    dh=(hp-hm)/(2*ep);dv=(vp-vm)/(2*ep)
    pred=rho.energy(dh,dv)
    plus=StateOracle(s,P,Q,state,variant).solve(Hp,fp).E
    minus=StateOracle(s,P,Q,state,variant).solve(Hm,fm).E
    fd=(plus-minus)/(2*ep)
    err=abs(pred-fd)
    print('orbital-integrals-chain-rule',variant,'nonclosed',nonclosed,'state',state,'fd',fd,'adj',pred,'error',err)
    assert err<3e-6


