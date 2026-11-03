"""Unified ground/excited CC(P;Q) determinant-space gradient verification.

Published biorthogonal moment correction:
    delta_mu = sum_{K in Q} <L_mu|Hbar|K><K|Hbar|R_mu> /
                            (E_mu - <K|Hbar|K>).
For mu=0, R_0=|0>, L_0=<0|+Lambda; for mu>0 the right vector
may have a reference coefficient, and the left vector has none.
The projected EOM Hamiltonian (including reference) is used, NOT the
amplitude-residual Jacobian. Both coincide only under further assumptions.

The adjoint/density construction is independent of analytical CC contractions.
"""
from dataclasses import dataclass
import operator
import numpy as np
import scipy.linalg as la
from .amplitudes import SelectedCC, similarity_transform
from .densities import transition_densities, mp_denominator_densities
from .eom_oracle import csjac, csgrad
from .spaces import validate_indices

@dataclass
class StateResult:
    """State energies and relaxed densities.

    ``gradient_H`` holds the derivative at fixed Fock diagonal. For MP with
    nonempty Q, use the integral densities or supply ``dfock_diag`` to
    ``StateOracle.directional_gradient`` for the explicit denominator response.
    """
    state: int
    E: float
    E_base: float
    delta: float
    omega: float
    t: np.ndarray
    r: np.ndarray
    l: np.ndarray
    gamma_base: np.ndarray
    Gamma_base: np.ndarray
    gamma_corr: np.ndarray
    Gamma_corr: np.ndarray
    gradient_H_base: np.ndarray
    gradient_H_corr: np.ndarray
    checks: dict
    arrays: dict

    @property
    def gamma(self): return self.gamma_base + self.gamma_corr
    @property
    def Gamma(self): return self.Gamma_base + self.Gamma_corr
    @property
    def gradient_H(self): return self.gradient_H_base + self.gradient_H_corr


class StateOracle:
    """state=0 is CC(P;Q); state>=1 is an excited EOMCC(P;Q) root.

    Supported denominator choices: 'full' and 'mp'. Bare-Fock MP uses
    omega_mu + sum_{occ(ref)} f_pp - sum_{occ(K)} f_pp.
    """
    def __init__(self, space, P, Q, state=0, denominator='full', gauge_scale=1.0):
        self.space=space
        self.P=validate_indices(space, P, "P")
        self.Q=validate_indices(space, Q, "Q")
        self.state=operator.index(state)
        self.denominator=denominator
        self.gauge_scale=float(gauge_scale)
        if not np.isfinite(self.gauge_scale) or self.gauge_scale == 0:
            raise ValueError("gauge_scale must be finite and nonzero")
        if self.state<0 or self.state>len(P): raise ValueError('invalid state')
        if denominator not in ('full','mp'): raise ValueError(denominator)
        if 0 in self.P or 0 in self.Q or np.intersect1d(self.P,self.Q).size:
            raise ValueError('P and Q must be disjoint and exclude reference')
        self.full=np.r_[0,self.P]
        self.m=len(self.P)
        self.k=self.m+1
        self.exc=space.excitation_transitions(self.P)
        self.max_rank=int(space.rank.max())
        self.chi=space.chi(self.Q) if len(self.Q) else np.zeros((0,space.nso))

    def hbar(self,H,t):
        T=self.exc.matrix(t,self.space.ndet).toarray()
        return similarity_transform(H,T,self.max_rank)

    def initial(self,H):
        cc=SelectedCC(self.space,H,self.P,self.Q)
        t,U,V,B=cc.solve_right()
        small=B[np.ix_(self.full,self.full)]
        if self.state==0:
            m=self.m
            C=small[1:,1:]-small[0,0]*np.eye(m)
            lam=la.solve(C.T,-small[0,1:])
            return t,np.r_[1.,np.zeros(m)],np.r_[1.,lam],float(np.real(small[0,0])),None
        # EOM full projected Hamiltonian: right excited vector includes r0.
        vals,vl,vr=la.eig(small,left=True,right=True)
        ground=np.argmin(np.abs(vals-small[0,0]))
        order=sorted((j for j in range(len(vals)) if j!=ground),key=lambda j:vals[j].real)
        i=order[self.state-1]
        if np.abs(vals[i].imag)>1e-8: raise ValueError('complex EOM root')
        r=np.real(vr[:,i]); l=np.real(vl[:,i])
        pivot=int(np.argmax(abs(r[1:])))+1
        r=self.gauge_scale*r/r[pivot];l=l/(l@r)
        return t,r,l,float(vals[i].real),pivot

    def pack(self,t,r,l,E,pivot):
        if self.state==0: return np.r_[t,l[1:]]
        return np.r_[t,np.delete(r,pivot),l,E]

    def unpack(self,y,pivot):
        m=self.m;k=self.k
        t=y[:m]
        if self.state==0:
            dtype=np.result_type(y)
            r=np.zeros(k,dtype=dtype);r[0]=1
            l=np.r_[np.ones(1,dtype=dtype),y[m:]]
            return t,r,l,None
        r=np.insert(y[m:m+k-1],pivot,self.gauge_scale)
        l=y[m+k-1:m+2*k-1]
        E=y[-1]
        return t,r,l,E

    def evaluate(self,H,y,pivot,fock_diag=None):
        t,r,l,E=self.unpack(y,pivot)
        U,V,B=self.hbar(H,t)
        small=B[np.ix_(self.full,self.full)]
        E0=B[0,0]
        if self.state==0: E=E0
        rhs=np.zeros(self.space.ndet,dtype=np.result_type(B,y))
        lhs=np.zeros_like(rhs)
        rhs[self.full]=r;lhs[self.full]=l
        M=B[np.ix_(self.Q,np.arange(self.space.ndet))]@rhs
        N=lhs@B[:,self.Q]
        if self.denominator=='full':
            D=E-np.diag(B)[self.Q]
        else:
            if fock_diag is None and len(self.Q): raise ValueError('MP requires fock_diag')
            D=(E-E0)+self.chi@fock_diag if len(self.Q) else np.zeros(0,dtype=B.dtype)
        if np.any(abs(D)<1e-9): raise ValueError('near-zero Q-space denominator')
        delta=np.sum(N*M/D)
        if self.state==0:
            base=E0
            C=small[1:,1:]-E0*np.eye(self.m)
            constraints=np.r_[B[self.P,0],C.T@l[1:]+small[0,1:]]
        else:
            base=l@small@r
            right=(small-E*np.eye(self.k))@r
            left=(small.T-E*np.eye(self.k))@l
            constraints=np.r_[B[self.P,0],right,np.delete(left,pivot),l@r-1]
        return base,delta,constraints,(U,V,B,M,N,D,r,l,E)

    def solve(self,H,fock_diag=None):
        t,r,l,E,pivot=self.initial(H)
        y=self.pack(t,r,l,E,pivot)
        def evaluate(yy): return self.evaluate(H,yy,pivot,fock_diag)
        def c(yy):return evaluate(yy)[2]
        J=csjac(c,y)
        adj=[];coeff=[];rho=[]
        d_mp=np.zeros(self.space.nso)
        for part in (0,1):
            F=lambda yy: evaluate(yy)[part]
            gy=csgrad(F,y)
            z=la.solve(J.T,-gy)
            adj.append(z)
            _,_,_,(U,V,B,M,N,D,r,l,E)=evaluate(y)
            def fixed_bar(flat):
                X=flat.reshape(self.space.ndet,self.space.ndet)
                small=X[np.ix_(self.full,self.full)]
                E0=X[0,0]
                # At fixed variables, the ground energy is E0, while
                # the excited energy E is an independent response variable.
                Efixed=E0 if self.state==0 else E
                rhs=np.zeros(self.space.ndet,dtype=X.dtype);lhs=np.zeros_like(rhs)
                rhs[self.full]=r;lhs[self.full]=l
                moments=X[self.Q,:]@rhs; numerator=lhs@X[:,self.Q]
                if self.denominator=='full':den=Efixed-np.diag(X)[self.Q]
                else:den=(Efixed-E0)+self.chi@fock_diag if len(self.Q) else np.zeros(0,dtype=X.dtype)
                delta=np.sum(numerator*moments/den)
                if self.state==0:
                    C=small[1:,1:]-E0*np.eye(self.m)
                    constraints=np.r_[X[self.P,0],C.T@l[1:]+small[0,1:]]
                    base=E0
                else:
                    right=(small-E*np.eye(self.k))@r
                    left=(small.T-E*np.eye(self.k))@l
                    constraints=np.r_[X[self.P,0],right,np.delete(left,pivot),l@r-1]
                    base=l@small@r
                return (base if part==0 else delta)+z@constraints
            Abar=csgrad(fixed_bar,B.real.flatten()).reshape(B.shape)
            KH=U.T@Abar@V.T
            rr=transition_densities(KH,self.space,self.space.one_body_transitions(),self.space.two_body_transitions())
            if part==1 and self.denominator=='mp' and len(self.Q):
                d_mp= -(N*M/D**2)@self.chi
                rr=rr+mp_denominator_densities(d_mp,self.space.ref_occ)
            coeff.append(KH);rho.append(rr)
        base,delta,C,details=evaluate(y)
        U,V,B,M,N,D,r,l,E=details
        checks={'constraint':float(np.max(abs(C),initial=0.)),
                'stationarity_base':float(np.max(abs(csgrad(lambda yy:evaluate(yy)[0],y)+J.T@adj[0]),initial=0.)),
                'stationarity_corr':float(np.max(abs(csgrad(lambda yy:evaluate(yy)[1],y)+J.T@adj[1]),initial=0.)),
                'jacobian_cond':float(np.linalg.cond(J)) if J.size else 1.0,
                'left_right':float(l@r)}
        return StateResult(self.state,float(np.real(base+delta)),float(np.real(base)),float(np.real(delta)),
                           float(np.real(E-B[0,0])),t,r,l,rho[0].gamma,rho[0].Gamma,
                           rho[1].gamma,rho[1].Gamma,coeff[0],coeff[1],checks,
                           {'M':M,'N':N,'D':D,'d_mp':d_mp,'adj_base':adj[0],'adj_corr':adj[1]})

    def directional_gradient(self,result,dH,dfock_diag=None):
        """Hamiltonian derivative, including the explicit MP Fock response."""
        value=np.sum(result.gradient_H*dH)
        if self.denominator=='mp' and len(self.Q):
            if dfock_diag is None:
                raise ValueError("MP directional gradients require dfock_diag; alternatively contract the RDMs")
            value+=result.arrays['d_mp']@np.asarray(dfock_diag)
        return value
