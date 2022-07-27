import argparse
import numpy as np
from matplotlib import pyplot as plt

def main(args):

    from pyscf import gto, scf, cc

    mol = gto.Mole()

    Re = 2.66816 # a.u.

    use_cartesian = False
    # for consistency with our papers on F2, cc-pVDZ basis uses Cartesian basis functions
    if args.basis == 'ccpvdz':
        use_cartesian = True

    ngeoms = 60

    re = np.linspace(0.25 * Re, 1.5 * Re, ngeoms)
    geoms = []
    e_rhf = np.zeros(ngeoms)
    e_uhf = np.zeros(ngeoms)

    for i in range(ngeoms):

        print("Geometry - ", i + 1)

        geoms.append([['F', (0, 0, -0.5 * re[i] * Re)], ['F', (0, 0, 0.5 * re[i] * Re)]])

        mol.build(
            atom=geoms[i],
            basis=args.basis,
            charge=0,
            symmetry="D2H",
            cart=use_cartesian,
            unit='Bohr',
            #spin=0,
        )

        mf = scf.UHF(mol)
        mf.kernel()

        try:
            mycc = cc.UCCSD(mf, frozen=2)
            mycc.kernel()

            et = mycc.uccsd_t()
        except:
            continue

        e_rhf[i] = mycc.e_tot + et


        # mf = scf.UHF(mol)
        # mf.kernel()
        # e_uhf[i] = mf.e_tot

    plt.scatter(re, e_rhf)
    #plt.scatter(re, e_uhf)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run F2 calculation at certain separation in units of Re (Re = 2.66816 a.u.).")
    parser.add_argument("-basis", type=str, help="Basis set (default is ccpvdz).", default="ccpvdz")

    args = parser.parse_args()

    main(args)


