"""This is a script for testing out the various components needed
for the optimized CC(P) implementation of triples."""

import numpy as np

from ccpy.utilities.permutations import do_lineup_permutation
from ccpy.models.hilbert import DeterminantalSubspace
from ccpy.utilities.determinants import calculate_excitation_difference

from pyscf import gto, scf
from ccpy.interfaces.pyscf_tools import load_pyscf_integrals

import time


if __name__ == "__main__":

    mol = gto.Mole()
    mol.build(
        atom="""F 0.0 0.0 -2.66816
                F 0.0 0.0  2.66816""",
        basis="ccpvdz",
        charge=1,
        spin=1,
        symmetry="D2H",
        cart=True,
        unit="Bohr",
    )
    mf = scf.ROHF(mol)
    mf.kernel()

    nfrozen = 2
    system, H = load_pyscf_integrals(mf, nfrozen)

    pspace = DeterminantalSubspace(system, 3, fill_level=3)

    # this is way, way, too slow!
    print("Excitation differences aaa")
    excit_levels_aaa = [ [calculate_excitation_difference(f1, f2, 'aaa', 'aaa') for f1 in pspace.aaa] for f2 in pspace.aaa]