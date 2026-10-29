"""libint2-based atomic-orbital integrals for ccpy.

    from ccpy.integrals import Molecule, integrals

    mol = Molecule("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", "cc-pvdz")
    S = integrals.overlap(mol)
    B = mol.df.B_Pmn          # density-fitted (P|mn), metric-orthonormalized

Adapted from forte2 (https://github.com/evangelistalab/forte2, LGPL-3.0); see
THIRD_PARTY_NOTICES.md.
"""
from ccpy.integrals._libint import Basis, Shell
from ccpy.integrals import integrals
from ccpy.integrals.basis import build_basis, build_basis_from_dict, decontract_basis
from ccpy.integrals.df import DensityFitting
from ccpy.integrals.molecule import Molecule
