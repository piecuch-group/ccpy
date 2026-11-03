"""Small molecular CC(P;Q)/EOMCC(P;Q) gradient demonstration.

Run with the installed CCpy Python, e.g.:
    python examples/ccpq_determinant_gradients.py --state 1 --denominator full --fd
"""
import argparse
import numpy as np

from ccpy import Molecule
from ccpy.gradients.ccpq import CCPQGradient, spaces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default="H 0 0 0; H 0.1 0 1.5; H 0 0.2 3.1; H 0.2 0.1 4.8",
                        help="CCpy molecular geometry in angstrom (quote the string)")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--state", type=int, default=0, help="0 ground; 1, 2, ... excited roots")
    parser.add_argument("--denominator", choices=("mp", "full"), default="full")
    parser.add_argument("--p-ranks", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--q-ranks", type=int, nargs="*", default=[3, 4])
    parser.add_argument("--triples-stride", type=int, default=3,
                        help="add every Nth triple to P; 0 disables selected triples")
    parser.add_argument("--nfrozen", type=int, default=0)
    parser.add_argument("--fd", action="store_true", help="check base/correction/total nuclear gradients")
    parser.add_argument("--step", type=float, default=1e-3, help="FD step in bohr")
    parser.add_argument("--npoints", type=int, choices=(2, 4, 6), default=4)
    parser.add_argument("--tolerance", type=float, default=2e-7, help="FD absolute tolerance in Eh/bohr")
    args = parser.parse_args()
    if args.triples_stride < 0:
        parser.error("--triples-stride must be nonnegative")
    mol = Molecule(args.geometry, args.basis, symmetry=False, verbose=False)

    def selected(space):
        p = spaces.ranks(space, args.p_ranks)
        if args.triples_stride:
            p = spaces.union(p, spaces.ranks(space, (3,))[::args.triples_stride])
        return p

    calc = CCPQGradient(mol, pspace=selected, qspace=tuple(args.q_ranks),
                        denominator=args.denominator, nfrozen=args.nfrozen, state=args.state, verbose=True)
    if args.fd:
        checks = calc.check_gradient(step=args.step, npoints=args.npoints)
        for key, result in checks.items():
            print(f"\n{key}: columns = analytic, finite difference, difference (Eh/bohr)")
            print(np.column_stack((result['analytic'], result['numerical'],
                                   result['analytic'] - result['numerical'])))
            print(f"max error = {result['max_error']:.3e}")
        if any(result['max_error'] > args.tolerance for result in checks.values()):
            raise SystemExit("Finite-difference tolerance exceeded")
    else:
        for key, gradient in calc.gradient_components().items():
            print(f"\n{key} gradient (Eh/bohr):\n{gradient}")


if __name__ == "__main__":
    main()
