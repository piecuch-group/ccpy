"""Four-center nuclear gradient from relaxed densities (Sec. 8.6 of ccpq_gradients_revised.pdf):

    E^x = V_nn^x + sum D^AO h^x - sum W^AO S^x + 1/2 sum G^AO_{mu nu rho sigma} (mu rho|nu sigma)^x  (8.21)
"""

import numpy as np

from ccpy.integrals import integrals


def contract_derivative_integrals(mol, C, rel, include_nuclear=True):
    """dE/dR, shape (natoms, 3), Eh/bohr, from RelaxedDensities `rel` in the MO basis C."""
    natoms = mol.natoms
    D_ao = C @ rel.D @ C.T
    W_ao = C @ rel.W @ C.T
    G_ao = _ao_two_body(C, rel.G)
    grad = integrals.nuclear_repulsion_deriv(mol) if include_nuclear else np.zeros((natoms, 3))
    grad += integrals.hcore_deriv(mol, D_ao).reshape(natoms, 3)
    grad -= integrals.overlap_deriv(mol, W_ao).reshape(natoms, 3)
    # 1/2 G^AO_{mn rs} (mr|ns)^x = sum W4_{mrns} (mr|ns)^x
    W4 = 0.5 * G_ao.transpose(0, 2, 1, 3)
    grad += integrals.coulomb_4c_deriv(mol, np.ascontiguousarray(W4)).reshape(natoms, 3)
    return grad


def _ao_two_body(C, G):
    """G^AO_{mn rs} = sum_pqrs C_mp C_nq C_rr' C_ss' G_pqr's' (Eq. 8.19)."""
    X = np.einsum("mp,pqrs->mqrs", C, G, optimize=True)
    X = np.einsum("nq,mqrs->mnrs", C, X, optimize=True)
    X = np.einsum("Rr,mnrs->mnRs", C, X, optimize=True)
    return np.einsum("Ss,mnRs->mnRS", C, X, optimize=True)
