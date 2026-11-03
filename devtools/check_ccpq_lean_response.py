"""Algebraic prototype of the lean total excited-state response.

Checks projected left/right responses and a three-factor total Hbar covector
against the independent dense constrained oracle. This is not a production
performance implementation or benchmark.
"""
import json
from pathlib import Path
import numpy as np
from ccpy.gradients.ccpq.determinants import DeterminantSpace,spin_orbital_integrals,hamiltonian_matrix,fock_diagonal
from ccpy.gradients.ccpq.spaces import ranks,complement
from ccpy.gradients.ccpq.state_oracle import StateOracle
s=DeterminantSpace(3,1,1);rng=np.random.default_rng(191)
h=np.diag([-1.6,-.3,.9]);a=rng.normal(size=(3,3))*.025;h+=(a+a.T)/2
v=rng.normal(size=(4,3,3))*.14;v=(v+v.transpose(0,2,1))/2;eri=np.einsum('xpq,xrs->pqrs',v,v)
hs,vs=spin_orbital_integrals(h,eri);H=hamiltonian_matrix(s,hs,vs);f=fock_diagonal(hs,vs,s.ref_occ)
maximum=0
records=[]
for closed in (True,False):
 low=np.array([k for k in ranks(s,(1,)) if s.holes_particles(k)[1][0]%s.norb==1])
 P=ranks(s,(1,)) if closed else np.r_[low,ranks(s,(2,))];Q=complement(s,P,(1,2))
 for den in ('full','mp'):
  for state in (1,2):
   o=StateOracle(s,P,Q,state,den);r=o.solve(H,f);_,_,_,_,pivot=o.initial(H)
   packed=o.pack(r.t,r.r,r.l,r.E_base,pivot)
   _,delta,_,(U,V,HB,M,N,D,rr,ll,E)=o.evaluate(H,packed,pivot,f)
   A=HB[np.ix_(o.full,o.full)];B=A-E*np.eye(o.k);u=M/D;v=N/D;w=u*v;W=sum(w)
   gr=HB[np.ix_(Q,o.full)].T@v;gl=HB[np.ix_(o.full,Q)]@u
   border=np.block([[B,rr[:,None]],[ll[None,:],np.zeros((1,1))]])
   x=np.linalg.solve(border,np.r_[-(gl-delta*rr),0.])[:-1]
   y=np.linalg.solve(border.T,np.r_[-(gr-delta*ll),0.])[:-1]
   assert abs(ll@x)<1e-10 and abs(rr@y)<1e-10
   RF=np.zeros(s.ndet);LF=RF.copy();UF=RF.copy();VF=RF.copy();XF=RF.copy();YF=RF.copy();e0=RF.copy();e0[0]=1
   RF[o.full]=rr;LF[o.full]=ll;UF[Q]=u;VF[Q]=v;XF[o.full]=x;YF[o.full]=y
   A0=np.outer((1-W)*LF+VF+YF,RF)+np.outer(LF,UF+XF)
   if den=='full':A0[Q,Q]+=w
   else:A0[0,0]+=W
   HI=[];J=np.zeros((o.m,o.m));g=np.zeros(o.m)
   for i in range(o.m):
    tau=o.exc.matrix(np.eye(o.m)[i],s.ndet).toarray();X=HB@tau-tau@HB
    J[:,i]=X[P,0];g[i]=np.sum(A0*X)
   z=np.linalg.solve(J.T,-g);ZF=np.zeros(s.ndet);ZF[P]=z
   C=A0+np.outer(ZF,e0);K=U.T@C@V.T
   error=float(abs(K-r.gradient_H).max());maximum=max(maximum,error)
   assert error<1e-10,(closed,den,state,error)
   records.append(dict(closed_P=closed,denominator=den,state=state,coefficient_error=error))
   print(closed,den,state,'coefficient error',error)
print('PASS maximum',maximum)

Path(__file__).resolve().parents[1].joinpath("docs/theory/ccpq_lean_response_checks.json").write_text(json.dumps(dict(passed=True,configurations=len(records),max_error=maximum,scope="Fixed-Fock bare-Hamiltonian total-gradient coefficient; retain the unchanged explicit MP Fock density",records=records),indent=2)+"\n")
