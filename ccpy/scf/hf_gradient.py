"""Density-fitted analytic RHF/UHF nuclear gradients built on the libint2 derivative
integrals,

    E_x = V_nn^x + D_{mn} h_{mn}^x - W_{mn} S_{mn}^x
          + W^P_{mn} (P|mn)^x + W_{PQ} (P|Q)^x

Adapted from forte2 (https://github.com/evangelistalab/forte2, LGPL-3.0); see
THIRD_PARTY_NOTICES.md.
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


def build_hf_df_deriv_weights(df, occupied_coefficients, D1, k_scale=1.0):
    r"""Two-electron derivative weights of a single-determinant energy.

    With Z^{P,s}_{ij} = C^s_{mi} Z^P_{mn} C^s_{nj} and rho^P = D_{mn} Z^P_{mn}
    (Z = metric-inverted three-center integrals),

        W_{PQ}    = -1/2 rho^P rho^Q + 1/2 \sum_s Z^{P,s}_{ij} Z^{Q,s}_{ji}
        W^P_{mn}  = D_{mn} rho^P - \sum_s C^s_{mi} C^s_{nj} Z^{P,s}_{ji}

    RHF passes its occupied orbitals twice (once per spin), UHF the alpha and beta ones.
    ``k_scale`` multiplies the exchange terms (the exact-exchange fraction of hybrid DFT;
    0 skips them).
    """
    rho = df.build_metric_inverted_density_contraction(D1)
    W2 = -0.5 * np.outer(rho, rho)
    W3 = np.einsum("mn,P->Pmn", D1, rho, optimize=True)
    for Cocc in occupied_coefficients if k_scale != 0.0 else ():
        Z_oo = df.build_metric_inverted_mo_block((Cocc, Cocc))
        W2 += 0.5 * k_scale * np.einsum("Pij,Qji->PQ", Z_oo, Z_oo, optimize=True)
        W3 -= k_scale * np.einsum("mi,nj,Pji->Pmn", Cocc, Cocc, Z_oo, optimize=True)
    return W2, W3


def assemble_gradient(mol, D1, W1, W2, W3, two_electron=True):
    """Sum the nuclear, one-electron, overlap, and (unless two_electron=False) DF
    two-electron contributions, returning an array of shape (natoms, 3) in Eh/bohr."""
    natoms = mol.natoms
    grad = integrals.nuclear_repulsion_deriv(mol)
    grad += integrals.hcore_deriv(mol, D1).reshape(natoms, 3)
    grad -= integrals.overlap_deriv(mol, W1).reshape(natoms, 3)
    if two_electron:
        grad += integrals.coulomb_3c_deriv(mol, W3).reshape(natoms, 3)
        grad += integrals.coulomb_2c_deriv(mol, W2).reshape(natoms, 3)
    return grad


def scf_gradient(mol, occupied, eps, k_scale=1.0):
    """Nuclear, one-electron, overlap, and density-fitted Coulomb + k_scale * exchange
    gradient of a single determinant with the occupied orbitals of each spin (a closed
    shell passes its orbitals twice) and their orbital energies."""
    D1 = sum(C @ C.T for C in occupied)
    W1 = sum((C * e) @ C.T for C, e in zip(occupied, eps))
    df = mol.df
    if not df.direct:
        W2, W3 = build_hf_df_deriv_weights(df, occupied, D1, k_scale)
        return assemble_gradient(mol, D1, W1, W2, W3)
    # integral-direct: the three-center weights are formed block by block
    grad = assemble_gradient(mol, D1, W1, None, None, two_electron=False)
    rho = df.fitting.fit([D1])[0]
    return grad + df.fitting.exchange_gradient(occupied, D1, rho, k_scale)


def rhf_gradient(mf):
    _check_df(mf, "RHF")
    Cocc = mf.C[0][:, : mf.na]
    e = mf.eps[0][: mf.na]
    return scf_gradient(mf.mol, (Cocc, Cocc), (e, e))


def uhf_gradient(mf):
    _check_df(mf, "UHF")
    return scf_gradient(mf.mol, (mf.C[0][:, : mf.na], mf.C[1][:, : mf.nb]),
                        (mf.eps[0][: mf.na], mf.eps[1][: mf.nb]))
