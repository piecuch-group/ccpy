import numpy as np

from ccpy.integrals import integrals
from ccpy.integrals.basis import build_sap_potential_basis


def sap_guess_hamiltonian(system, H):
    """
    Return the superposition of atomic potentials (SAP) guess Hamiltonian.

    S. Lehtola, J. Chem. Theory Comput. 15, 1593-1604 (2019), arXiv:1810.11659.
    For details, see https://doi.org/10.1063/5.0004046

    Parameters
    ----------
    system : ccpy.integrals.Molecule
        The system object containing the atoms and basis set.
    H : NDArray
        The core Hamiltonian matrix.

    Returns
    -------
    NDArray
        The core Hamiltonian plus the SAP potential, whose orbitals are the initial guess.
    """

    # Convert the SAP potential coefficients to a normalized charge-density basis.
    scaled_sap_basis = build_sap_potential_basis(
        "sap_helfem_large",
        system.geom_helper,
    )

    # generate the SAP integrals (P|mn)
    SAP_ints = integrals.coulomb_3c(system, scaled_sap_basis)

    # generate the SAP potential V_mn = sum_P (P|mn)
    SAP_V = np.einsum("Pmn->mn", SAP_ints)

    return H + SAP_V


def smallest_gap_same_irrep(eps, irreps, nocc):
    """
    Find two orbitals that belong to the same irrep with the smallest gap.

    Parameters
    ----------
    eps : NDArray
        Orbital energies, in ascending order within the occupied orbitals and within
        the virtual ones.
    irreps : NDArray
        The irrep index of each orbital.
    nocc : int
        The number of occupied orbitals, which come first.

    Returns
    -------
    tuple[int, int] | None
        The HOMO and LUMO if they share an irrep. Otherwise, of the occupied and
        virtual orbitals that share an irrep, the pair with the smallest energy gap, or
        None if no such pair exists.
    """
    homo, lumo = nocc - 1, nocc
    if irreps[homo] == irreps[lumo]:
        return homo, lumo
    best = None
    for irrep in np.unique(irreps[:nocc]):
        virtuals = nocc + np.flatnonzero(irreps[nocc:] == irrep)
        if len(virtuals) == 0:
            continue
        pair = (int(np.flatnonzero(irreps[:nocc] == irrep)[-1]), int(virtuals[0]))
        if best is None or eps[pair[1]] - eps[pair[0]] < eps[best[1]] - eps[best[0]]:
            best = pair
    return best


def guess_mix(C, occ_idx, vir_idx, mixing_parameter=np.pi / 4):
    """
    Induce the breaking of S^2 symmetry for UHF ms=0.0 calculations.
    This is helpful for obtaining proxies for open-shell singlets, for example.

    Parameters
    ----------
    C : NDArray
        The MO coefficients.
    occ_idx : int
        The index of the occupied orbital to mix, usually the HOMO.
    vir_idx : int
        The index of the virtual orbital to mix, usually the LUMO.
    mixing_parameter : float, optional
        The mixing parameter for the Givens rotation.

    Returns
    -------
    NDArray
        The modified MO coefficients.

    Notes
    -----
    See Szabo and Ostlund Ch. 3.8.7.

    """
    cosq = np.cos(mixing_parameter)
    sinq = np.sin(mixing_parameter)
    Ca = givens_rotation(C, cosq, sinq, occ_idx, vir_idx)
    Cb = givens_rotation(C, cosq, -sinq, occ_idx, vir_idx)
    return [Ca, Cb]


def givens_rotation(A, c, s, i, j, column=True):
    """Apply a (real) Givens rotation to columns (or rows) i and j of A, as in forte2."""
    A = A.copy()
    if column:
        ai, aj = A[:, i].copy(), A[:, j].copy()
        A[:, i] = c * ai + s * aj
        A[:, j] = -s * ai + c * aj
    else:
        ai, aj = A[i, :].copy(), A[j, :].copy()
        A[i, :] = c * ai - s * aj
        A[j, :] = s * ai + c * aj
    return A
