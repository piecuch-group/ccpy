"""Check the expanded equations in docs/theory against independent complex steps.

Small physical Hamiltonians exercise closed and nonclosed P, ground/excited
states, both denominators, gauge fixing, and the stationary Hbar pullback.
No optional FCIpy extension or molecular SCF is required.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from ccpy.gradients.ccpq.determinants import (DeterminantSpace,spin_orbital_integrals,
    hamiltonian_matrix,fock_diagonal)
from ccpy.gradients.ccpq.spaces import ranks,complement
from ccpy.gradients.ccpq.state_oracle import StateOracle
from ccpy.gradients.ccpq.amplitudes import SelectedCC
from ccpy.gradients.ccpq.eom_oracle import csgrad


def check():
    s=DeterminantSpace(3,1,1)
    rng=np.random.default_rng(191)
    h=np.diag([-1.6,-.3,.9]);a=rng.normal(size=(3,3))*.025;h+=(a+a.T)/2
    B=rng.normal(size=(4,3,3))*.14;B=(B+B.transpose(0,2,1))/2
    eri=np.einsum('xpq,xrs->pqrs',B,B)
    hs,vs=spin_orbital_integrals(h,eri);H=hamiltonian_matrix(s,hs,vs);f=fock_diagonal(hs,vs,s.ref_occ)
    records=[]
    for closed in (True,False):
        # Keep both spin partners of the low-virtual single in the nonclosed
        # case; omitting just one creates a near-resonant Q partner and makes
        # an absolute machine-precision algebra check poorly conditioned.
        low_singles=np.array([k for k in ranks(s,(1,)) if s.holes_particles(k)[1][0]%s.norb==1])
        P=ranks(s,(1,)) if closed else np.r_[low_singles,ranks(s,(2,))]
        Q=complement(s,P,(1,2))
        for denominator in ('full','mp'):
            for state in (0,1,2):
                o=StateOracle(s,P,Q,state=state,denominator=denominator,gauge_scale=-1.7)
                r=o.solve(H,f)
                t,_,_,_,pivot=o.initial(H)
                y=o.pack(r.t,r.r,r.l,r.E_base,pivot)
                _,_,_,(U,V,HB,M,N,D,R,L,E)=o.evaluate(H,y,pivot,f)
                A=HB[np.ix_(o.full,o.full)];E0=HB[0,0];m=len(P)
                er=np.zeros(s.ndet);er[0]=1
                rr=np.zeros(s.ndet);ll=rr.copy();rr[o.full]=R;ll[o.full]=L
                u=M/D;v=N/D;w=u*v;W=sum(w)
                uf=np.zeros(s.ndet);vf=uf.copy();uf[Q]=u;vf[Q]=v
                tau=[o.exc.matrix(np.eye(m)[i],s.ndet).toarray() for i in range(m)]
                HI=[HB@x-x@HB for x in tau]
                gt=[]
                for X in HI:
                    dD=(X[0,0]-np.diag(X)[Q]) if state==0 and denominator=='full' else (
                         np.zeros(len(Q)) if state==0 else (
                         -np.diag(X)[Q] if denominator=='full' else np.full(len(Q),-X[0,0])))
                    gt.append(v@(X@rr)[Q]+u@(ll@X[:,Q])-w@dD)
                if state==0:
                    manual=np.r_[gt,HB[np.ix_(P,Q)]@u]
                else:
                    gr=HB[np.ix_(Q,o.full)].T@v;gl=HB[np.ix_(o.full,Q)]@u
                    manual=np.r_[gt,np.delete(gr,pivot),gl,-W]
                reference=csgrad(lambda yy:o.evaluate(H,yy,pivot,f)[1],y)
                source_error=float(np.max(abs(manual-reference),initial=0))
                errors=dict(expanded_correction_source=source_error)
                if state:
                    errors['normalization_multiplier_base']=abs(r.arrays['adj_base'][-1]+r.E_base)
                    errors['normalization_multiplier_correction']=abs(r.arrays['adj_corr'][-1]+r.delta)
                for part in (0,1):
                    if part==0:
                        C=np.outer(er,er) if state==0 else np.outer(ll,rr)
                    else:
                        C=np.outer(vf,rr)+np.outer(ll,uf)
                        if denominator=='full':
                            C[Q,Q]+=w
                            if state==0:C[0,0]-=W
                        elif state:C[0,0]+=W
                    z=r.arrays['adj_base' if part==0 else 'adj_corr']
                    zt=np.zeros(s.ndet);zt[P]=z[:m];C+=np.outer(zt,er)
                    if state==0:
                        sigma=np.zeros(s.ndet);sigma[P]=z[m:]
                        C+=np.outer(ll,sigma);C[0,0]-=L[1:]@z[m:]
                    else:
                        aa=np.zeros(s.ndet);bb=aa.copy()
                        aa[o.full]=z[m:m+o.k]
                        bb[o.full]=np.insert(z[m+o.k:-1],pivot,0.)
                        C+=np.outer(aa,rr)+np.outer(ll,bb)
                    K=U.T@C@V.T
                    expected=r.gradient_H_base if part==0 else r.gradient_H_corr
                    errors['base_pullback' if part==0 else 'correction_pullback']=float(np.max(abs(K-expected)))
                    df=-(w@o.chi) if part==1 and denominator=='mp' else np.zeros(s.nso)
                    target=r.E_base if part==0 else r.delta
                    errors['base_euler' if part==0 else 'correction_euler']=abs(np.sum(K*H)+df@f-target)
                if state==0:
                    cc=SelectedCC(s,H,P,Q,fock_diag=f,denominator=denominator).run()
                    errors['ground_lambda_grad']=float(np.max(abs(r.arrays['adj_base'][:m]-cc.lambda_grad)))
                    errors['ground_sigma']=float(np.max(abs(r.arrays['adj_corr'][m:]-cc.arrays['sigma'])))
                    errors['ground_pi']=float(np.max(abs(r.arrays['adj_corr'][:m]-cc.arrays['pi'])))
                    J=SelectedCC(s,H,P,Q).jacobian(cc.Hbar)
                    BB=cc.Hbar[np.ix_(P,P)]-cc.energy*np.eye(m)
                    errors['lambda_defect_identity']=float(np.max(abs((J-BB).T@cc.lam-(cc.Hbar[0,P]+J.T@cc.lam))))
                    if not closed:
                        assert np.linalg.norm(J-BB)>1e-5
                        assert np.linalg.norm(cc.lam-cc.lambda_grad)>1e-6
                if denominator=='mp':assert abs(sum(-(w@o.chi)))<1e-12
                maximum=max(errors.values())
                assert maximum<1e-9,(closed,denominator,state,errors)
                records.append(dict(closed_P=closed,state=state,denominator=denominator,errors={k:float(v) for k,v in errors.items()}))
    return dict(passed=True,configurations=len(records),max_error=max(max(r['errors'].values()) for r in records),records=records)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('docs/theory/ccpq_formula_checks.json'))
    args=p.parse_args();result=check();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f"PASS {result['configurations']} configurations; maximum algebraic/adjoint discrepancy {result['max_error']:.3e}")
