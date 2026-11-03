"""Full SD plus active T / selected Q, with complete complementary Q spaces.

All electrons are correlated by default; --nfrozen freezes occupied orbitals.
All basis orbitals are retained. The active occupied/virtual sets
restrict only the selected high-rank cluster amplitudes; singles and doubles
are complete, and every remaining triple (and optionally quadruple) enters Q.
MP and full denominators share displaced CC/EOM solutions to reduce test cost.

Build the FCIpy connector first and set CCPY_FCIPY_EXTENSION. For example:
python examples/ccpq_complete_spaces.py --models sdt --output examples/results/h6_sdt_complete.json
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from ccpy import Molecule
from ccpy.gradients.ccpq import CCPQGradient, spaces
from ccpy.gradients.ccpq.determinants import DeterminantSpace
from ccpy.gradients.ccpq.scf import run_rhf, align_orbital_phases
from ccpy.gradients.finite_difference import finite_difference


def make_spaces(space, model, occupied=None, virtual=None, quadruple_stride=2):
    occupied = list(range(max(space.nfrozen, space.nalpha-2), space.nalpha)) if occupied is None else list(occupied)
    virtual = list(range(space.nalpha, min(space.norb, space.nalpha+3))) if virtual is None else list(virtual)
    if quadruple_stride < 1:
        raise ValueError('quadruple_stride must be positive')
    triples = spaces.active_excitations(space, (3,), occupied, virtual)
    P = spaces.union(spaces.ranks(space, (1, 2)), triples)
    if model == 'sdtq':
        P = spaces.union(P, spaces.active_excitations(space, (4,), occupied, virtual)[::quadruple_stride])
    Q = spaces.complement(space, P, (3, 4) if model == 'sdtq' else (3,))
    assert np.all(np.isin(spaces.ranks(space, (1, 2)), P))
    assert not np.intersect1d(P, Q).size
    expected = spaces.ranks(space, (1, 2, 3, 4) if model == 'sdtq' else (1, 2, 3))
    np.testing.assert_array_equal(spaces.union(P, Q), expected)
    return P, Q, occupied, virtual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=['sdt', 'sdtq'], default=['sdt', 'sdtq'])
    parser.add_argument('--states', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--natoms', type=int, choices=[4, 6], default=6)
    parser.add_argument('--basis', default='6-31g')
    parser.add_argument('--geometry', help='quoted CCpy geometry in angstrom; overrides the H-chain geometry')
    parser.add_argument('--nfrozen', type=int, default=0, help='frozen occupied spatial orbitals')
    parser.add_argument('--verify-roots', action='store_true', help='independently compare buffered reference roots with SciPy Arnoldi')
    parser.add_argument('--active-occupied', nargs='+', type=int, help='zero-based occupied spatial MO indices')
    parser.add_argument('--active-virtual', nargs='+', type=int, help='zero-based virtual spatial MO indices')
    parser.add_argument('--quadruple-stride', type=int, default=2, help='include every Nth active quadruple; 1 includes all')
    parser.add_argument('--align-phases', action=argparse.BooleanOptionalAction, default=True,
                        help='align displaced RHF orbital signs to the reference for continuous warm starts')
    parser.add_argument('--follow-reference', action=argparse.BooleanOptionalAction, default=True,
                        help='follow the verified reference root within gap and overlap guards for FD energies')
    parser.add_argument('--step', type=float, default=1e-3)
    parser.add_argument('--components', nargs='+', default=['1:2', '2:0'])
    parser.add_argument('--tolerance', type=float, default=2e-8)
    parser.add_argument('--output', type=Path, default=Path('examples/results/h6_complete_spaces.json'))
    parser.add_argument('--resume', action='store_true', help='reuse completed passing state/denominator pairs in the output')
    args = parser.parse_args()
    components = [tuple(map(int, s.split(':'))) for s in args.components]
    geometry = args.geometry or '; '.join(f'H {0.035*((i%3)-1):.6f} {0.025*((i%2)*2-1):.6f} {i*1.0:.6f}' for i in range(args.natoms))
    mol = Molecule(geometry, args.basis, symmetry=False, verbose=False)
    nocc = mol.nelectrons//2
    space = DeterminantSpace(mol.nmo, nocc, nocc, args.nfrozen)
    xyz = np.array([a[1] for a in mol.atoms])
    report = dict(geometry_angstrom=geometry, basis=args.basis, nelectrons=mol.nelectrons,
                  nmo=mol.nmo, ndet=space.ndet, nfrozen=args.nfrozen, deleted_orbitals=0,
                  components=components, stencil_points=4, step_bohr=args.step,
                  tolerance=args.tolerance, models={}, cases=[])
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        for key in ('geometry_angstrom', 'basis', 'components', 'stencil_points', 'step_bohr', 'nfrozen'):
            if previous[key] != json.loads(json.dumps(report[key])):
                raise ValueError(f'Cannot resume a report with different {key}')
        report = previous
    passed = True
    for model in args.models:
        P, Q, occupied, virtual = make_spaces(space, model, args.active_occupied, args.active_virtual, args.quadruple_stride)
        definition = dict(P=P.tolist(), Q=Q.tolist(), active_occupied=occupied, active_virtual=virtual,
                          P_rank_counts={str(r): int(np.sum(space.rank[P] == r)) for r in range(1, 5)},
                          Q_rank_counts={str(r): int(np.sum(space.rank[Q] == r)) for r in range(1, 5)})
        definition['P_rank_fractions'] = {str(r): float(np.sum(space.rank[P] == r)/np.sum(space.rank == r))
                                          for r in (3, 4) if np.any(space.rank == r)}
        if args.resume and model in report['models'] and any(
                report['models'][model][key] != definition[key]
                for key in ('P', 'Q', 'active_occupied', 'active_virtual')):
            raise ValueError('Cannot resume a report with different P/Q selections')
        report['models'][model] = definition
        last_t = None
        calc = None
        reference_guesses = None
        verified = None
        for state in args.states:
            old = [c for c in report['cases'] if c['model'] == model and c['state'] == state]
            root_checked = not args.verify_roots or state == 0 or all(c.get('independent_root_check') for c in old)
            if root_checked and {c['denominator'] for c in old} == {'mp', 'full'} and all(
                    v['max_error'] < args.tolerance for c in old for v in c['gradients'].values()):
                print(f'SKIP completed {model} state={state}', flush=True)
                continue
            report['cases'] = [c for c in report['cases'] if not (c['model'] == model and c['state'] == state)]
            start = time.perf_counter()
            print(f'START {model} state={state}, P={len(P)}, Q={len(Q)}, ranks={definition["P_rank_counts"]}', flush=True)
            if not 0 <= state <= len(P):
                raise ValueError('State must be 0..len(P)')
            if calc is None:
                calc = CCPQGradient(mol, P, Q, 'mp', state=state, nfrozen=args.nfrozen, backend='matrix_free', sparse_cache=True)
            else:
                # P, Q and geometry are identical across reference states.
                # Reuse only connectivity and starting vectors; every CC/EOM
                # equation, response and residual is solved/checked afresh.
                calc.state = calc.backend.state = state
                calc.backend.clear_reference()
                calc.backend._eom_guesses = reference_guesses
            if last_t is not None:
                calc.backend.t = last_t.copy()
            gradients, amplitudes = {}, {}
            for denominator in ('mp', 'full'):
                calc.backend.denominator = denominator
                gradients[denominator] = calc.gradient_components()
                amplitudes[denominator] = calc.stages['amplitudes']
                print(f'  ANALYTIC {denominator}: delta={amplitudes[denominator].delta:+.10e} elapsed={time.perf_counter()-start:.1f}s', flush=True)
            last_t = calc.backend.t.copy()
            reference_guesses = calc.backend._eom_guesses
            reference = calc.stages['reference']
            independent = None
            if args.verify_roots and state:
                snapshots = calc.backend._operator_integrals
                same_reference = verified is not None and np.allclose(last_t, verified['t'], atol=1e-11, rtol=0.) and all(
                    np.array_equal(a, b) for a, b in zip(snapshots, verified['integrals']))
                if same_reference:
                    values = verified['values']
                else:
                    from scipy.sparse.linalg import LinearOperator, eigs
                    from ccpy.gradients.ccpq.matrix_free import MatrixFreeState
                    check = MatrixFreeState(space, P, np.zeros(0, dtype=int), calc.backend._operator,
                        np.zeros(space.nso), state, 'full', excitations=calc.backend._exc)
                    check.set_t(amplitudes['full'].data.t)
                    operator = LinearOperator((len(P), len(P)),
                        matvec=lambda x: check.bar(check.embed(x, P))[P], dtype=float)
                    values = eigs(operator, k=4, which='SR', ncv=32, tol=1e-12,
                        v0=np.random.default_rng(4401).normal(size=len(P)), return_eigenvectors=False)
                    values = values[np.argsort(values.real)]
                    verified = dict(t=last_t.copy(), integrals=tuple(a.copy() for a in snapshots), values=values.copy())
                    del operator, check
                error = float(np.max(abs(values-amplitudes['full'].data.eom_values)))
                if error > 1e-8:
                    raise RuntimeError(f'Independent Arnoldi root ordering mismatch: {error:.3e}')
                independent = dict(method='SciPy Arnoldi, k=4, which=SR, ncv=32, tol=1e-12',
                    eigenvalues_electronic=values.real.tolist(), max_error=error,
                    reused_identical_reference_spectrum=bool(same_reference))
                print(f'  ARNOLDI root agreement={error:.3e} elapsed={time.perf_counter()-start:.1f}s', flush=True)
            following = calc.backend.follow_reference(amplitudes['full'].data) if args.follow_reference and state else None
            evaluations = [0]
            def energies(coords):
                displaced = mol.with_geometry(coords, unit='bohr')
                mf = run_rhf(displaced)
                if args.align_phases:
                    align_orbital_phases(reference, mf)
                h, eri = mf.mo_integrals()
                # Both denominators use exactly the same underlying CC/EOM state.
                data = calc.backend.run(h, eri, densities=False)
                base = displaced.nuclear_repulsion + data.E_base
                delta_mp = float(np.sum(data.arrays['M'] * data.arrays['N'] / data.arrays['D_mp']))
                evaluations[0] += 1
                print(f'  FD {evaluations[0]}/{4*len(components)} Ebase={base:.12f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
                return np.array([base, delta_mp, data.delta, base+delta_mp, base+data.delta])
            numerical = finite_difference(energies, xyz, step=args.step, npoints=4, components=components)
            for denominator, correction_column, total_column in [('mp', 1, 3), ('full', 2, 4)]:
                amp = amplitudes[denominator]
                case = dict(model=model, state=state, denominator=denominator,
                    orbital_phase_alignment=args.align_phases,
                    guarded_root_following=following,
                    independent_root_check=independent,
                    delta_hartree=amp.delta, base_energy_hartree=mol.nuclear_repulsion+amp.energy_base,
                    min_abs_denominator=float(np.min(abs(amp.data.arrays['D']))),
                    solver_checks={k: float(v) for k, v in amp.checks.items()}, gradients={})
                case['projected_root_energies_electronic'] = amp.data.eom_values.tolist()
                for key, column in [('base', 0), ('correction', correction_column), ('total', total_column)]:
                    analytic = np.array([gradients[denominator][key][i] for i in components])
                    error = float(np.max(abs(analytic - numerical[:, column])))
                    case['gradients'][key] = dict(analytic=analytic.tolist(), numerical=numerical[:, column].tolist(),
                        max_error=error, full_analytic=gradients[denominator][key].tolist(),
                        net_force=gradients[denominator][key].sum(axis=0).tolist())
                    passed &= error < args.tolerance
                case['elapsed_seconds_for_both_denominators'] = time.perf_counter()-start
                report['cases'].append(case)
                print(f"PASSCHECK {model} state={state} {denominator} delta={amp.delta:+.10e} "
                      + ' '.join(f"{k}_error={case['gradients'][k]['max_error']:.3e}" for k in ('base','correction','total')), flush=True)
            report['passed'] = bool(passed)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
    if not passed:
        raise SystemExit('Gradient tolerance exceeded; inspect the saved report')
    print(f'PASS: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
