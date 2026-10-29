"""Density-fitted analytic RHF/UHF nuclear gradients built on the libint2 derivative
integrals (ported from forte2/scf/hf_grad.py and forte2/gradients/utils.py).

    E_x = V_nn^x + D_{mn} h_{mn}^x - W_{mn} S_{mn}^x
          + W^P_{mn} (P|mn)^x + W_{PQ} (P|Q)^x
"""

import numpy as np

from ccpy.integrals import integrals


def _check_df(mf, method):
    mol = mf.mol
    if mol.cholesky_tei:
        raise NotImplementedError(f"{method} gradients require density fitting, not cholesky_tei.")
    max_l = max(mol.basis.max_l, mol.auxiliary_basis.max_l)
    if max_l > integrals.LIBINT2_MAX_AM:
        raise NotImplementedError(
            f"{method} gradients need derivative integrals with l = {max_l} "
            f"(libint2 supports {integrals.LIBINT2_MAX_AM})."
        )
    if not mf.converged:
        mf.run()


def build_hf_df_deriv_weights(df, occupied_coefficients, D1):
    r"""Two-electron derivative weights of a single-determinant energy.

    With Z^{P,s}_{ij} = C^s_{mi} Z^P_{mn} C^s_{nj} and rho^P = D_{mn} Z^P_{mn}
    (Z = metric-inverted three-center integrals),

        W_{PQ}    = -1/2 rho^P rho^Q + 1/2 \sum_s Z^{P,s}_{ij} Z^{Q,s}_{ji}
        W^P_{mn}  = D_{mn} rho^P - \sum_s C^s_{mi} C^s_{nj} Z^{P,s}_{ji}

    RHF passes its occupied orbitals twice (once per spin), UHF the alpha and beta ones.
    """
    rho = df.build_metric_inverted_density_contraction(D1)
    W2 = -0.5 * np.outer(rho, rho)
    W3 = np.einsum("mn,P->Pmn", D1, rho, optimize=True)
    for Cocc in occupied_coefficients:
        Z_oo = df.build_metric_inverted_mo_block((Cocc, Cocc))
        W2 += 0.5 * np.einsum("Pij,Qji->PQ", Z_oo, Z_oo, optimize=True)
        W3 -= np.einsum("mi,nj,Pji->Pmn", Cocc, Cocc, Z_oo, optimize=True)
    return W2, W3


def assemble_gradient(mol, D1, W1, W2, W3):
    """Sum the nuclear, one-electron, overlap, and DF two-electron contributions,
    returning an array of shape (natoms, 3) in Eh/bohr."""
    natoms = mol.natoms
    grad = integrals.nuclear_repulsion_deriv(mol)
    grad += integrals.hcore_deriv(mol, D1).reshape(natoms, 3)
    grad -= integrals.overlap_deriv(mol, W1).reshape(natoms, 3)
    grad += integrals.coulomb_3c_deriv(mol, W3).reshape(natoms, 3)
    grad += integrals.coulomb_2c_deriv(mol, W2).reshape(natoms, 3)
    return grad


def rhf_gradient(mf):
    _check_df(mf, "RHF")
    Cocc = mf.C[0][:, : mf.na]
    D1 = 2.0 * mf.D[0]
    W1 = 2.0 * (Cocc * mf.eps[0][: mf.na]) @ Cocc.T
    W2, W3 = build_hf_df_deriv_weights(mf.mol.df, (Cocc, Cocc), D1)
    return assemble_gradient(mf.mol, D1, W1, W2, W3)


def uhf_gradient(mf):
    _check_df(mf, "UHF")
    Ca = mf.C[0][:, : mf.na]
    Cb = mf.C[1][:, : mf.nb]
    D1 = mf.D[0] + mf.D[1]
    W1 = (Ca * mf.eps[0][: mf.na]) @ Ca.T + (Cb * mf.eps[1][: mf.nb]) @ Cb.T
    W2, W3 = build_hf_df_deriv_weights(mf.mol.df, (Ca, Cb), D1)
    return assemble_gradient(mf.mol, D1, W1, W2, W3)
