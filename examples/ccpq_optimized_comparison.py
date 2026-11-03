"""Compare determinant derivatives with optimized, integral-matched CCpy energies.

CR-CC(2,3) A/D and full CC(P;3) A/D are like-for-like comparisons. Optimized
CC(t;3) is also measured, but uses the CCSD-like two-body/left approximation
and is therefore reported separately from the full selected-space functional.
"""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import time
import numpy as np
from ccpy import Molecule, RHF, CRCC23, CCSDT_P, CCT3, CCP3, get_active_triples_pspace
from ccpy.gradients.ccpq import CCPQGradient, spaces
from ccpy.gradients.finite_difference import finite_difference

OPTIONS = dict(energy_convergence=1e-13, amp_convergence=1e-11, amp_print_threshold=1.)
GEOMETRY = 'O 0 0 0; H .05 -.75 .59; H -.03 .78 .62'


def optimized(mol, nfrozen, triples=None, nacto=1, nactu=1, gradient=False):
    """Solve the independent tensor code, sharing only its own CC/left states."""
    with contextlib.redirect_stdout(io.StringIO()):
        mf = RHF(mol, verbose=0, e_tol=1e-13, d_tol=1e-11)
        mf.kernel()
        cr = CRCC23(mf, nfrozen=nfrozen, variant='A', **OPTIONS)
        cr.energy()
        result = dict(ccsd=cr.cc.energy(), cr_A=cr.deltas['A'], cr_D=cr.deltas['D'])
        native_gradient = cr.gradient() if gradient else None
        cr.system.set_active_space(nacto, nactu)
        if triples is None:
            triples = get_active_triples_pspace(cr.system)
        cc = CCSDT_P(mf, copy.deepcopy(triples), nfrozen=nfrozen, **OPTIONS)
        result['ccsdtp'] = cc.energy()
        ct = CCT3(cc, nacto=nacto, nactu=nactu, variant='A', **OPTIONS)
        ct.energy()
        cp = CCP3(cc, two_body_approx=False, variant='A', **OPTIONS)
        cp.energy()
        for label, method in [('ct', ct), ('cp', cp)]:
            for variant in ('A', 'D'):
                result[f'{label}_{variant}'] = method.deltas[variant]
        if gradient:
            cp2 = CCP3(cc, two_body_approx=True, variant='A', **OPTIONS)
            cp2.energy()
            result['cct3_variants'] = {k:float(v) for k,v in ct.deltas.items()}
            result['ccp3_two_body_approximation'] = {k:float(v) for k,v in cp2.deltas.items()}
    return result, triples, native_gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basis', default='6-31g')
    parser.add_argument('--geometry', default=GEOMETRY, help='angstrom geometry')
    parser.add_argument('--nfrozen', type=int, default=1)
    parser.add_argument('--nacto', type=int, default=1)
    parser.add_argument('--nactu', type=int, default=1)
    parser.add_argument('--integrals', choices=['df', 'cholesky'], default='df')
    parser.add_argument('--auxiliary-basis', default='def2-universal-jkfit')
    parser.add_argument('--components', nargs='+', help='zero-based atom:axis; default all')
    parser.add_argument('--output', type=Path, default=Path('examples/results/h2o_optimized_comparison.json'))
    args = parser.parse_args()
    mol = Molecule(args.geometry, args.basis, auxiliary_basis=args.auxiliary_basis,
        cholesky_tei=args.integrals=='cholesky', cholesky_tol=1e-13, symmetry=False, verbose=False)
    components = list(np.ndindex((mol.natoms, 3))) if args.components is None else [tuple(map(int,s.split(':'))) for s in args.components]
    start = time.perf_counter()
    print('START optimized reference', flush=True)
    reference, triples, native_gradient = optimized(mol, args.nfrozen, nacto=args.nacto, nactu=args.nactu, gradient=True)
    print('REFERENCE', reference, flush=True)
    report = dict(geometry_angstrom=args.geometry, basis=args.basis, nfrozen=args.nfrozen,
        integral_model=args.integrals, auxiliary_basis=args.auxiliary_basis if args.integrals=='df' else None,
        cholesky_tolerance=1e-13 if args.integrals=='cholesky' else None,
        nacto=args.nacto,nactu=args.nactu,num_active=1,
        active_selection='at least one active hole AND at least one active particle (CCSDt)',
        t3_excitations={k:np.asarray(v).tolist() for k,v in triples.items()},
        components=components,step_bohr=1e-3,stencil_points=4,optimized_reference=reference,cases=[])
    keys = ['ccsd','cr_A','cr_D','ccsdtp','ct_A','ct_D','cp_A','cp_D']
    evaluations = [0]
    def evaluate(xyz):
        result, _, _ = optimized(mol.with_geometry(xyz,unit='bohr'), args.nfrozen, triples, args.nacto, args.nactu)
        evaluations[0] += 1
        print(f'FD {evaluations[0]}/{4*len(components)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        return np.array([result[k] for k in keys])
    xyz=np.array([a[1] for a in mol.atoms])
    numerical=finite_difference(evaluate,xyz,step=1e-3,npoints=4,components=components)
    fd={k:numerical[:,i] for i,k in enumerate(keys)}
    passed=True
    for theory in ('crcc23','ccp3'):
        selected=(1,2) if theory=='crcc23' else lambda s:spaces.union(spaces.ranks(s,(1,2)),spaces.from_t3_excitations(s,triples))
        calc=CCPQGradient(mol,selected,(3,),'mp',nfrozen=args.nfrozen,backend='matrix_free',
            sparse_cache=True,integral_model='factorized')
        for variant,denominator in [('A','mp'),('D','full')]:
            print(f'START determinant {theory} {variant} elapsed={time.perf_counter()-start:.1f}s',flush=True)
            calc.backend.denominator=denominator
            analytic=calc.gradient_components()
            amp=calc.stages['amplitudes']
            base='ccsd' if theory=='crcc23' else 'ccsdtp'
            correction=('cr_' if theory=='crcc23' else 'cp_')+variant
            case=dict(theory=theory,variant=variant,denominator=denominator,
                P_rank_counts={str(r):int(np.sum(calc.backend.space.rank[calc.backend.P]==r)) for r in (1,2,3)},
                Q_size=len(calc.backend.Q),ndet=calc.backend.space.ndet,
                determinant_base_energy=mol.nuclear_repulsion+amp.energy_base,determinant_correction=amp.delta,
                base_energy_error=abs(mol.nuclear_repulsion+amp.energy_base-reference[base]),
                correction_energy_error=abs(amp.delta-reference[correction]),
                solver_checks={k:float(v) for k,v in amp.checks.items()},gradients={})
            for part,num in [('base',fd[base]),('correction',fd[correction]),('total',fd[base]+fd[correction])]:
                a=np.array([analytic[part][index] for index in components])
                error=float(abs(a-num).max())
                case['gradients'][part]=dict(full_analytic=analytic[part].tolist(),analytic=a.tolist(),numerical=num.tolist(),max_error=error)
                passed &= error<2e-8
            if theory=='crcc23' and variant=='A':
                case['optimized_analytic_gradient']=native_gradient.tolist()
                case['optimized_analytic_error']=float(abs(native_gradient-analytic['total']).max())
                passed &= case['optimized_analytic_error']<1e-9
            if theory=='ccp3':
                # Do not label a different approximate correction as a failed
                # implementation of the exact selected-space functional.
                ct='ct_'+variant
                case['cct3_comparison']=dict(same_functional=False,
                    reason='CCT3 uses CCSD-like Hbar/moments and left-CCSD, while the determinant functional uses full CC(P) Hbar and left-CC(P)',
                    correction_energy_difference=float(amp.delta-reference[ct]),gradients={})
                for part,num in [('correction',fd[ct]),('total',fd[base]+fd[ct])]:
                    a=np.array([analytic[part][index] for index in components])
                    case['cct3_comparison']['gradients'][part]=dict(numerical=num.tolist(),max_difference=float(abs(a-num).max()))
            passed &= case['base_energy_error']<1e-9 and case['correction_energy_error']<1e-9
            report['cases'].append(case)
            report['matched_functionals_passed']=bool(passed)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('CHECK',theory,variant,'energy errors',case['base_energy_error'],case['correction_energy_error'],
                'gradient errors',{k:v['max_error'] for k,v in case['gradients'].items()},flush=True)
        del calc
    if not passed:
        raise SystemExit('Matched-functional comparison failed; inspect report')
    print(f'PASS matched functionals: {args.output.resolve()}',flush=True)


if __name__=='__main__':
    main()
