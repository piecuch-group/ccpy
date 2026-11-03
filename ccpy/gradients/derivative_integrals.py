"""Contraction of orbital-relaxed densities with the derivative integrals,

    dE/dx = V_nn^x + sum_s [ D[s] . h^x - W[s] . S^x ] + (3- and 2-center terms of Y[s]),

with D[s], Y[s], W[s] from ccpy.gradients.orbital_response. The densities of the two spins
are back-transformed with their own orbitals; only then, as AO quantities contracted with the
spin-free AO derivative integrals, are they added.

The same densities give the first derivative with respect to the strength of any one-electron
perturbation, h -> h + lambda V, whose operator does not move the basis functions or the nuclei
(electric multipoles, ...): S, (Q|mn), (Q|Q') and V_nn do not depend on lambda, so

    dE/dlambda = sum_s D[s] . V,

the orbital response of the reference being in D[s] through the Z-vector. With the electronic
multipole integrals V = sum_i x_i^a y_i^b z_i^c (multipole_integrals), the moments of the state
are sum_A Z_A X_A^a Y_A^b Z_A^c - dE/dlambda (multipole_moments).
"""

import numpy as np

from ccpy.gradients.density_gradient import get_bare_integrals
from ccpy.integrals import integrals


def contract_derivative_integrals(D, Y, W, reference, perturbation=None):
    """Nuclear gradient, shape (natoms, 3), Eh/bohr, of relaxed densities (D, Y, W) (blocks oo,
    ov, vo, vv of each spin; Y with a leading auxiliary index).

    perturbation : AO integrals V (nao, nao), or a stack (ncomp, nao, nao), of one-electron
        operators added to h with strengths lambda; the derivatives dE/dlambda = D . V, shape
        (ncomp,), are returned instead of the nuclear gradient.

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
    if perturbation is not None:
        return np.einsum("mn,kmn->k", D_ao, np.reshape(perturbation, (-1,) + D_ao.shape), optimize=True)
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


def multipole_integrals(mol, order=1, origin=None):
    """Labels ("x", "xx", "xyz", ...) and AO integrals <m|(x - O_x)^a (y - O_y)^b (z - O_z)^c|n>,
    shape (ncomp, nao, nao), of the Cartesian electronic moments of orders 1..order (<= 3) about
    origin (bohr; default the coordinate origin), in the libint2 order."""
    labels = MULTIPOLE_LABELS[:{1: 3, 2: 9, 3: 19}[order]]
    V = {1: integrals.emultipole1, 2: integrals.emultipole2, 3: integrals.emultipole3}[order](mol, origin=origin)
    return labels, np.asarray(V)[1:]


def multipole_moments(dE, mol, order=1, origin=None):
    """Cartesian multipole moments (atomic units; not traceless) of the state with derivatives dE
    with respect to the multipole_integrals(mol, order, origin): nuclear minus electronic part."""
    labels, _ = multipole_integrals(mol, order)
    O = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    Z = np.array([atom[0] for atom in mol.atoms], dtype=float)
    R = np.array([atom[1] for atom in mol.atoms], dtype=float) - O
    nuc = np.array([np.sum(Z * np.prod([R[:, "xyz".index(c)] for c in label], axis=0)) for label in labels])
    return dict(zip(labels, nuc - np.asarray(dE)))


MULTIPOLE_LABELS = ("x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz",
                    "xxx", "xxy", "xxz", "xyy", "xyz", "xzz", "yyy", "yyz", "yzz", "zzz")


def print_gradient(mol, grad, title, elapsed):
    from ccpy.integrals.atom_data import Z_TO_ATOM_SYMBOL

    print(f"\n   {title}")
    print("   " + "-" * len(title))
    print(f"   Gradient time: {elapsed:.2f} s")
    if np.ndim(grad) == 1:
        print("\n   Perturbation      dE/dlambda")
        for k, gk in enumerate(grad):
            print(f"   {k:<12}{gk:>17.10f}")
        print("")
        return
    print("\n   Atom          dE/dx            dE/dy            dE/dz")
    for (Zn, _), gA in zip(mol.atoms, grad):
        print(f"   {Z_TO_ATOM_SYMBOL[Zn]:<4}{gA[0]:>17.10f}{gA[1]:>17.10f}{gA[2]:>17.10f}")
    print("")
