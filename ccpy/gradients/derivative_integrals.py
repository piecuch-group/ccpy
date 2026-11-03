"""Contraction of orbital-relaxed densities with the derivative integrals,

    dE/dx = V_nn^x + sum_s [ D[s] . h^x - W[s] . S^x ] + (3- and 2-center terms of Y[s]),

with D[s], Y[s], W[s] from ccpy.gradients.orbital_response. The densities of the two spins
are back-transformed with their own orbitals; only then, as AO quantities contracted with the
spin-free AO derivative integrals, are they added.
"""

import numpy as np

from ccpy.gradients.density_gradient import get_bare_integrals
from ccpy.integrals import integrals


def contract_derivative_integrals(D, Y, W, reference):
    """Nuclear gradient, shape (natoms, 3), Eh/bohr, of relaxed densities (D, Y, W) (blocks oo,
    ov, vo, vv of each spin; Y with a leading auxiliary index).

    B[s] = X (Q|mn) C[s] C[s] with X = L^{-1}, M = (Q|Q') = L L^T, for density fitting and for
    Cholesky vectors alike (Q: auxiliary function or Cholesky pivot pair); only the derivative
    integrals of (Q|mn) and (Q|Q') differ. For a Cholesky decomposition the pivots are held fixed
    (the gradient is exact for that energy)."""
    mol, nfz = reference.molecule, reference.nfrozen
    _, B = get_bare_integrals(reference)
    noa, nob = nfz + reference.system.noccupied_alpha, nfz + reference.system.noccupied_beta
    Coa, Cva = reference.mo_coeff["a"][:, :noa], reference.mo_coeff["a"][:, noa:]
    Cob, Cvb = reference.mo_coeff["b"][:, :nob], reference.mo_coeff["b"][:, nob:]
    natoms = mol.natoms
    D_ao = _to_ao(D.a, Coa, Cva) + _to_ao(D.b, Cob, Cvb)
    W_ao = _to_ao(W.a, Coa, Cva) + _to_ao(W.b, Cob, Cvb)
    Y_ao = _to_ao(Y.a, Coa, Cva) + _to_ao(Y.b, Cob, Cvb)
    BY = _overlap(B.a, Y.a) + _overlap(B.b, Y.b)

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


def _to_ao(M, Co, Cv):
    """sum_pq C_mp M_pq C_nq over the blocks (M may carry a leading auxiliary index)."""
    return (Co @ M.oo @ Co.T + Co @ M.ov @ Cv.T + Cv @ M.vo @ Co.T + Cv @ M.vv @ Cv.T)


def _overlap(B, Y):
    """sum_pq B_x[p, q] Y_y[p, q] over the blocks."""
    return (np.einsum("xpq,ypq->xy", B.oo, Y.oo, optimize=True) + np.einsum("xpq,ypq->xy", B.ov, Y.ov, optimize=True)
            + np.einsum("xpq,ypq->xy", B.vo, Y.vo, optimize=True) + np.einsum("xpq,ypq->xy", B.vv, Y.vv, optimize=True))


def print_gradient(mol, grad, title, elapsed):
    from ccpy.integrals.atom_data import Z_TO_ATOM_SYMBOL

    print(f"\n   {title}")
    print("   " + "-" * len(title))
    print(f"   Gradient time: {elapsed:.2f} s")
    print("\n   Atom          dE/dx            dE/dy            dE/dz")
    for (Zn, _), gA in zip(mol.atoms, grad):
        print(f"   {Z_TO_ATOM_SYMBOL[Zn]:<4}{gA[0]:>17.10f}{gA[1]:>17.10f}{gA[2]:>17.10f}")
    print("")
