"""Independent EOMCCSDt/full EOMCC(P;3) derivative comparison.

Native CCpy tensor energies supply the finite differences. Determinant
amplitudes, EOM roots and analytical derivatives are solved independently.
Root identities are checked against an optimized-HR Arnoldi spectrum and
physical determinant-coordinate eigenvector overlaps.
"""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
from ccpy import (Molecule, RHF, CRCC23, CCSDT_P, EOMCCSDT_P,
                  EOMCCP3, EOMCCT3, get_active_triples_pspace)
from ccpy.gradients.ccpq import CCPQGradient, spaces
from ccpy.gradients.ccpq.determinants import DeterminantSpace
from ccpy.gradients.finite_difference import finite_difference
from ccpy.integrals import integrals

WATER='O 0 0 0; H .05 -.75 .59; H -.03 .78 .62'
H4='H 0 0 0; H .12 -.06 1.0; H -.08 .10 2.1; H .15 .06 3.15'
OPTIONS=dict(energy_convergence=1e-13, amp_convergence=1e-11,
    maximum_iterations=250, amp_print_threshold=1., davidson_max_subspace_size=60)


class PLayout:
    """Unique determinant coefficients <-> redundant CCpy spin blocks."""
    def __init__(self, space, P):
        self.space, self.P=space,np.asarray(P)
        self.entries=[]
        for k in P:
            holes,parts=space.holes_particles(k)
            case=''.join('a' if p<space.norb else 'b' for p in parts)
            idx=tuple(p%space.norb-space.nalpha for p in parts)+tuple(p%space.norb-space.nfrozen for p in holes)
            self.entries.append((case,idx,space.excitation_string(k)[1]))

    def rows(self, triples):
        return {(case,tuple(np.asarray(row,dtype=int)-1)):i
            for case,values in triples.items() for i,row in enumerate(values)}

    def vector(self, operator, triples):
        rows=self.rows(triples)
        return np.array([phase*(getattr(operator,case)[rows[case,idx]] if len(case)==3 else getattr(operator,case)[idx])
                         for case,idx,phase in self.entries])

    def operator(self, eom, vector, triples):
        R=eom._new_operator()
        rows=self.rows(triples)
        for value,(case,idx,phase) in zip(vector,self.entries):
            value=phase*value
            X=getattr(R,case)
            if len(case)==3:
                X[rows[case,idx]]=value
            elif case in ('a','b','ab'):
                X[idx]=value
            else:
                a,b,i,j=idx
                X[a,b,i,j]=X[b,a,j,i]=value
                X[b,a,i,j]=X[a,b,j,i]=-value
        return R


def overlap(a,b):
    return float(abs(np.vdot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)))


def optimized(mol, nfrozen, nacto=1, nactu=1, triples=None, layout=None,
              reference=None, audit=False):
    """Fresh native tensor solves; no determinant amplitudes/roots are supplied."""
    with contextlib.redirect_stdout(io.StringIO()):
        mf=RHF(mol,verbose=0,e_tol=1e-13,d_tol=1e-11);mf.kernel()
        if reference is not None:
            S=integrals.overlap(mol,basis1=reference.mf.mol.basis,basis2=mol.basis)
            d=np.diag(reference.mf.C[0].T@S@mf.C[0])
            if np.min(abs(d))<.5:
                raise RuntimeError('Displaced canonical MOs cannot be matched by signs alone')
            mf.C[0]*=np.where(d<0.,-1.,1.)[None,:]
        if triples is None:
            setup=CRCC23(mf,nfrozen=nfrozen)
            setup.system.set_active_space(nacto,nactu)
            triples=get_active_triples_pspace(setup.system)
        if layout is None:
            s=DeterminantSpace(mol.nmo,mol.nelectrons//2,mol.nelectrons//2,nfrozen)
            P=spaces.union(spaces.ranks(s,(1,2)),spaces.from_t3_excitations(s,triples))
            layout=PLayout(s,P)
        cc=CCSDT_P(mf,copy.deepcopy(triples),nfrozen=nfrozen,**OPTIONS)
        ground=cc.energy()
        roots=[]
        for mult in (3,1):
            # Full spin blocks are retained for both multiplicities.
            eom=EOMCCSDT_P(cc,copy.deepcopy(triples),RHF_symmetry=False,**OPTIONS)
            eom.initial_guess('cisd',multiplicity=mult,nacto=cc.system.noccupied_alpha,
                nactu=cc.system.nunoccupied_alpha,roots_per_irrep={'A':1})
            base=float(eom.energy()[0])
            correction=EOMCCP3(eom,variant='A',**OPTIONS)
            correction.energy()
            if not eom.converged or not correction.left.converged:
                raise RuntimeError('Unconverged native EOM or left-EOM state')
            left_error=float(abs(correction.left.omega[0]-eom.omega[0]))
            if left_error>1e-9:
                raise RuntimeError('Native left/right roots disagree')
            right=np.r_[eom.r0[0],layout.vector(eom.R[0],eom.r3_excitations)]
            left=np.r_[0.,layout.vector(correction.left.L[0],eom.r3_excitations)]
            lr=float(left@right)
            if abs(lr-1)>1e-8:
                raise RuntimeError(f'Physical determinant-coordinate L.R mismatch: {lr}')
            root=SimpleNamespace(multiplicity=mult,base=base,omega=float(eom.omega[0]),
                delta={k:float(v) for k,v in correction.deltas[0].items()},
                right=right,left=left,eom=eom,correction=correction,
                r0=float(eom.r0[0]),left_error=left_error,lr=lr)
            if audit:
                ct=EOMCCT3(eom,two_body_approx=False,nacto=nacto,nactu=nactu,variant='A',**OPTIONS)
                ct.energy()
                root.cct3_flag_false={k:float(v) for k,v in ct.deltas[0].items()}
            roots.append(root)
    return SimpleNamespace(mf=mf,cc=cc,ground=float(ground),triples=triples,layout=layout,roots=roots)


def optimized_spectrum(bundle):
    """Arnoldi on the independent optimized HR action, in unique P coordinates."""
    from ccpy.eomcc.eomccsdt_p import HR
    eom=bundle.roots[0].eom
    layout=bundle.layout
    reference_r3=copy.deepcopy(eom.r3_excitations)
    def action(x):
        r3=copy.deepcopy(reference_r3)
        R=layout.operator(eom,x,r3)
        dR=eom._new_operator()
        HR(dR,R,bundle.cc.T,eom.hbar.H,False,eom.system,bundle.cc.t3_excitations,r3)
        return layout.vector(dR,r3)
    op=LinearOperator((len(layout.P),)*2,matvec=action,dtype=float)
    values,vectors=eigs(op,k=4,which='SR',ncv=32,tol=1e-12,
        v0=np.random.default_rng(8181).normal(size=len(layout.P)))
    order=np.argsort(values.real);values,vectors=values[order],vectors[:,order]
    if np.max(abs(values.imag))>1e-10:
        raise RuntimeError('Complex optimized roots in reference buffer')
    residual=max(np.max(abs(action(vectors[:,i].real)-values[i].real*vectors[:,i].real)) for i in range(4))
    if residual>1e-9:
        raise RuntimeError(f'Optimized Arnoldi residual too large: {residual}')
    return values.real, float(residual)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--molecule',choices=['water','h4'],default='water')
    p.add_argument('--basis',default='6-31g')
    p.add_argument('--nfrozen',type=int,help='default O(1s) for water, none for H4')
    p.add_argument('--components',nargs='+',help='zero-based atom:axis; default all')
    p.add_argument('--step',type=float,default=1e-3,help='finite-difference step in bohr')
    p.add_argument('--output',type=Path,default=Path('examples/results/h2o_631g_excited_optimized.json'))
    args=p.parse_args()
    nfrozen=(1 if args.molecule=='water' else 0) if args.nfrozen is None else args.nfrozen
    mol=Molecule(WATER if args.molecule=='water' else H4,args.basis,
        auxiliary_basis='def2-universal-jkfit',symmetry=False,verbose=False)
    components=list(np.ndindex((mol.natoms,3))) if args.components is None else [tuple(map(int,s.split(':'))) for s in args.components]
    start=time.perf_counter()
    print('START optimized reference',flush=True)
    ref=optimized(mol,nfrozen,audit=True)
    spectrum,residual=optimized_spectrum(ref)
    print('ARNOLDI optimized spectrum',spectrum,'residual',residual,flush=True)
    states=[]
    for r in ref.roots:
        index=int(np.argmin(abs(spectrum-r.omega)))
        error=float(abs(spectrum[index]-r.omega))
        if error>1e-9:
            raise RuntimeError('Native target root absent from independently ordered buffer')
        others=np.delete(spectrum,index)
        r.radius=float(np.min(abs(others-r.omega))/4)
        states.append(index+1)
        print('NATIVE',r.multiplicity,'oracle state',index+1,'base',r.base,'delta',r.delta,'r0',r.r0,flush=True)
    report=dict(molecule=args.molecule,geometry_angstrom=WATER if args.molecule=='water' else H4,
        basis=args.basis,nfrozen=nfrozen,integral_model='df',auxiliary_basis='def2-universal-jkfit',
        nacto=1,nactu=1,num_active=1,P_rank_counts={str(i):int(np.sum(ref.layout.space.rank[ref.layout.P]==i)) for i in (1,2,3)},
        Q_size=len(spaces.complement(ref.layout.space,ref.layout.P,(3,))),ndet=ref.layout.space.ndet,
        t3_excitations={k:np.asarray(v).tolist() for k,v in ref.triples.items()},
        r3_uses_same_fixed_selection=True,components=components,stencil_points=4,step_bohr=args.step,
        independent_optimized_spectrum=dict(method='SciPy Arnoldi on native optimized HR, k=4, which=SR, ncv=32',
            excitation_energies=spectrum.tolist(),max_residual=residual),reference_roots=[],fd_checks=[],cases=[])
    for r,state in zip(ref.roots,states):
        report['reference_roots'].append(dict(multiplicity=r.multiplicity,oracle_state=state,base_energy=r.base,
            omega=r.omega,r0=r.r0,delta=r.delta,lr=r.lr,left_root_error=r.left_error,
            cct3_flag_false_delta=r.cct3_flag_false,
            hbar_class=type(r.eom.hbar).__name__,left_class=type(r.correction.left).__name__))
    evaluations=[0]
    def energies(coords):
        b=optimized(mol.with_geometry(coords,unit='bohr'),nfrozen,triples=ref.triples,layout=ref.layout,reference=ref)
        checks=[]
        for r,rref in zip(b.roots,ref.roots):
            ov=overlap(r.right,rref.right)
            drift=abs(r.omega-rref.omega)
            if ov<.9 or drift>rref.radius:
                raise RuntimeError('Optimized FD root left reference overlap/gap guard')
            checks.append(dict(multiplicity=r.multiplicity,right_overlap=ov,excitation_energy_drift=drift,
                left_root_error=r.left_error,lr=r.lr))
        report['fd_checks'].append(checks)
        evaluations[0]+=1
        print(f'FD {evaluations[0]}/{4*len(components)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        return np.array([[r.base,r.delta['A'],r.delta['D']] for r in b.roots])
    xyz=np.array([a[1] for a in mol.atoms])
    numerical=finite_difference(energies,xyz,step=args.step,npoints=4,components=components)
    report['optimized_numerical']=numerical.tolist()
    report['passed']=False  # Analytical comparisons have not completed yet.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    calc=CCPQGradient(mol,ref.layout.P,(3,),'mp',nfrozen=nfrozen,state=states[0],
        backend='matrix_free',sparse_cache=True,integral_model='factorized')
    passed=True
    for root_index,(r,state) in enumerate(zip(ref.roots,states)):
        calc.state=calc.backend.state=state
        calc.backend.clear_reference()
        for variant,denominator,column in [('A','mp',1),('D','full',2)]:
            print('START determinant',r.multiplicity,state,variant,'elapsed',time.perf_counter()-start,flush=True)
            calc.backend.denominator=denominator
            g=calc.gradient_components();amp=calc.stages['amplitudes']
            if not np.allclose(calc.mf.C[0],ref.mf.C[0],atol=1e-10,rtol=0.):
                raise RuntimeError('Reference canonical phases differ between native and determinant calculations')
            spectrum_error=float(np.max(abs(amp.data.eom_values[:len(spectrum)]-(spectrum+amp.data.E0))))
            ov=overlap(r.right,amp.data.r);lov=overlap(r.left,amp.data.l)
            base_error=abs(mol.nuclear_repulsion+amp.energy_base-r.base)
            correction_error=abs(amp.delta-r.delta[variant])
            case=dict(multiplicity=r.multiplicity,oracle_state=state,variant=variant,
                determinant_base=mol.nuclear_repulsion+amp.energy_base,determinant_delta=amp.delta,
                base_energy_error=base_error,correction_energy_error=correction_error,
                native_right_overlap=ov,native_left_overlap=lov,independent_spectrum_error=spectrum_error,
                solver_checks={k:float(v) for k,v in amp.checks.items()},gradients={})
            passed &= max(base_error,correction_error,spectrum_error)<1e-9 and min(ov,lov)>1-1e-8
            for key,num in [('base',numerical[:,root_index,0]),('correction',numerical[:,root_index,column]),
                            ('total',numerical[:,root_index,0]+numerical[:,root_index,column])]:
                a=np.array([g[key][index] for index in components]);error=float(np.max(abs(a-num)))
                case['gradients'][key]=dict(analytic=a.tolist(),numerical=num.tolist(),full_analytic=g[key].tolist(),max_error=error)
                passed &= error<2e-8
            report['cases'].append(case);report['passed']=bool(passed)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('CHECK',r.multiplicity,variant,'energies',base_error,correction_error,'spectrum',spectrum_error,
                'overlaps',ov,lov,'gradients',{k:v['max_error'] for k,v in case['gradients'].items()},flush=True)
    if not passed:
        raise SystemExit('Excited-state comparison failed; inspect report')
    print('PASS',args.output.resolve(),flush=True)


if __name__=='__main__':
    main()
