"""Reproducible all-electron H6/6-31G selected-P/Q nuclear-gradient test.

Build the FCIpy connector and set CCPY_FCIPY_EXTENSION first. No orbitals or
electrons are frozen/deleted. P and Q are fixed selected determinant lists,
NOT all singles/doubles/triples. Two P triples and 17 Q triples were selected
at the reference geometry by moment screening, then frozen for every FD step.

python examples/ccpq_h6_631g_matrix_free.py --output examples/results/ccpq_h6_631g_matrix_free.json
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from ccpy import Molecule
from ccpy.gradients.ccpq import CCPQGradient
from ccpy.gradients.finite_difference import finite_difference

GEOMETRY = '; '.join(f'H {0.035*((i%3)-1):.6f} {0.025*((i%2)*2-1):.6f} {i*1.0:.6f}' for i in range(6))
P = np.array([1, 4, 9, 10, 13, 18, 219, 275, 334, 573, 740, 919, 1196, 2165])
Q = np.array([1192, 1237, 1240, 1300, 1307, 1408, 1624, 1732, 1735, 1737,
              2247, 2526, 2529, 4371, 5259, 6301, 6304])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('examples/results/ccpq_h6_631g_matrix_free.json'))
    parser.add_argument('--states', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--denominators', nargs='+', choices=['mp', 'full'], default=['mp', 'full'])
    parser.add_argument('--step', type=float, default=1e-3)
    parser.add_argument('--tolerance', type=float, default=2e-8)
    args = parser.parse_args()
    mol = Molecule(GEOMETRY, '6-31g', symmetry=False, verbose=False)
    xyz = np.array([atom[1] for atom in mol.atoms])
    keys = ('base', 'correction', 'total')
    report = dict(geometry_angstrom=GEOMETRY, basis='6-31g', nelectrons=6, nmo=mol.nmo,
                  nfrozen=0, deleted_orbitals=0, P=P.tolist(), Q=Q.tolist(),
                  method='selected CC(P;Q)/EOMCC(P;Q), not complete CCSD/triples spaces',
                  step_bohr=args.step, stencil_points=4, tolerance=args.tolerance, cases=[])
    failed = False
    for denominator in args.denominators:
        for state in args.states:
            start = time.perf_counter()
            calc = CCPQGradient(mol, P, Q, denominator, state=state, backend='matrix_free')
            print(f'Start {denominator} state={state}, ndet={calc.backend.space.ndet}', flush=True)
            checks = calc.check_gradient(step=args.step)
            amp = calc.stages['amplitudes']
            energies = dict(base=mol.nuclear_repulsion + amp.energy_base, correction=amp.delta, total=calc.e_tot)
            case = dict(denominator=denominator, state=state, ndet=calc.backend.space.ndet,
                        energies_hartree=energies,
                        min_abs_denominator=float(np.min(abs(amp.data.arrays['D']))),
                        solver_checks={k: float(v) for k, v in amp.checks.items()},
                        gradients={}, step_checks={})
            for key in keys:
                value = checks[key]
                gradient = calc.gradients[key]
                case['gradients'][key] = dict(analytic=gradient.tolist(),
                    numerical=value['numerical'].reshape(xyz.shape).tolist(),
                    max_error=value['max_error'],
                    net_force=gradient.sum(axis=0).tolist(),
                    net_torque=np.cross(xyz, gradient).sum(axis=0).tolist())
                failed |= value['max_error'] > args.tolerance
            # Independent smaller-step stencil on a longitudinal and transverse component.
            indices = [(1, 2), (3, 0)]
            def energies_at(coords):
                values = calc.energy_components(mol.with_geometry(coords, unit='bohr'))
                return np.array([values[key] for key in keys])
            smaller = finite_difference(energies_at, xyz, step=args.step/2, npoints=4, components=indices)
            for column, key in enumerate(keys):
                actual = np.array([calc.gradients[key][index] for index in indices])
                error = float(np.max(abs(actual - smaller[:, column])))
                case['step_checks'][key] = dict(step_bohr=args.step/2, components=indices,
                                                numerical=smaller[:, column].tolist(), max_error=error)
                failed |= error > args.tolerance
            case['elapsed_seconds'] = time.perf_counter() - start
            report['cases'].append(case)
            print(f"{denominator} state={state} delta={amp.delta:+.10e} "
                  + ' '.join(f"{key}_err={checks[key]['max_error']:.3e}" for key in keys)
                  + f" time={case['elapsed_seconds']:.1f}s", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Results: {args.output.resolve()}', flush=True)
    if failed:
        raise SystemExit('Gradient tolerance exceeded; see the numerical report')


if __name__ == '__main__':
    main()
