"""Contraction of orbital-relaxed densities with the derivative integrals,

    dE/dx = V_nn^x + sum_s [ D[s] . h^x - W[s] . S^x ] + (3- and 2-center terms of Y[s]),

with D[s], Y[s], W[s] from ccpy.gradients.orbital_response. The densities of the two spins
are back-transformed with their own orbitals; only then, as AO quantities contracted with the
spin-free AO derivative integrals, are they added.
"""

import numpy as np

from ccpy.gradients.density_gradient import SPINS
from ccpy.integrals import integrals


def contract_derivative_integrals(D, Y, W, reference):
    """Nuclear gradient, shape (natoms, 3), Eh/bohr, of relaxed densities (D, Y, W) per spin.

    B[s] = X (Q|mn) C[s] C[s] with X = L^{-1}, M = (Q|Q') = L L^T, for density fitting and for
    Cholesky vectors alike (Q: auxiliary function or Cholesky pivot pair); only the derivative
    integrals of (Q|mn) and (Q|Q') differ. For a Cholesky decomposition the pivots are held fixed
    (the gradient is exact for that energy)."""
    mol, C, B = reference.molecule, reference.mo_coeff, reference.B_mo
    natoms = mol.natoms
    D_ao = sum(C[s] @ D[s] @ C[s].T for s in SPINS)
    W_ao = sum(C[s] @ W[s] @ C[s].T for s in SPINS)
    Y_ao = sum(np.einsum("mp,xpq,nq->xmn", C[s], Y[s], C[s], optimize=True) for s in SPINS)
    BY = sum(np.einsum("xpq,ypq->xy", B[s], Y[s], optimize=True) for s in SPINS)

    grad = integrals.nuclear_repulsion_deriv(mol)
    grad += integrals.hcore_deriv(mol, D_ao).reshape(natoms, 3)
    grad -= integrals.overlap_deriv(mol, W_ao).reshape(natoms, 3)
    X = mol.df.Mm12
    W3 = np.ascontiguousarray(np.einsum("QP,Qmn->Pmn", X, Y_ao, optimize=True))
    W2 = np.ascontiguousarray(-0.5 * X.T @ BY @ X)
    if mol.cholesky_tei:
        grad += integrals.cholesky_3c_deriv(mol, W3, mol.df.pivots).reshape(natoms, 3)
        grad += integrals.cholesky_2c_deriv(mol, W2, mol.df.pivots).reshape(natoms, 3)
    else:
        grad += integrals.coulomb_3c_deriv(mol, W3).reshape(natoms, 3)
        grad += integrals.coulomb_2c_deriv(mol, W2).reshape(natoms, 3)
    return grad


def print_gradient(mol, grad, title, elapsed):
    from ccpy.integrals.atom_data import Z_TO_ATOM_SYMBOL

    print(f"\n   {title}")
    print("   " + "-" * len(title))
    print(f"   Gradient time: {elapsed:.2f} s")
    print("\n   Atom          dE/dx            dE/dy            dE/dz")
    for (Zn, _), gA in zip(mol.atoms, grad):
        print(f"   {Z_TO_ATOM_SYMBOL[Zn]:<4}{gA[0]:>17.10f}{gA[1]:>17.10f}{gA[2]:>17.10f}")
    print("")
