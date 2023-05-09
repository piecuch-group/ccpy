import numpy as np
from ccpy.energy.hf_energy import calc_hf_energy_unsorted

def shift_normal_order(H, occ_a, occ_b, occ_a_prev, occ_b_prev):

    # de-normal order Z = F - G
    if occ_a_prev: # if occ_a_prev /= []
        H.a -= np.einsum("piqi->pq", H.aa[:, occ_a_prev, :, occ_a_prev], optimize=True)
        H.b -= np.einsum("ipiq->pq", H.ab[occ_a_prev, :, occ_a_prev, :], optimize=True)
    if occ_b_prev: # if occ_b_prev /= []
        H.a -= np.einsum("piqi->pq", H.ab[:, occ_b_prev, :, occ_b_prev], optimize=True)
        H.b -= np.einsum("piqi->pq", H.bb[:, occ_b_prev, :, occ_b_prev], optimize=True)

    # compute reference energy with bare Hamiltonian
    reference_energy = calc_hf_energy_unsorted(H, occ_a, occ_b)

    # normal order F = Z + G
    H.a += np.einsum("piqi->pq", H.aa[:, occ_a, :, occ_a], optimize=True)
    H.b += np.einsum("ipiq->pq", H.ab[occ_a, :, occ_a, :], optimize=True)
    H.a += np.einsum("piqi->pq", H.ab[:, occ_b, :, occ_b], optimize=True)
    H.b += np.einsum("piqi->pq", H.bb[:, occ_b, :, occ_b], optimize=True)

    return H, reference_energy


if __name__ == "__main__":

    from ccpy.models.hilbert import Determinant

    model_space = [
        Determinant(),
    ]

    d = len(model_space)

    occ_prev_a = []
    occ_prev_b = []

    for q in range(d):

        occ_a, unocc_a, occ_b, unocc_b = model_space[q].get_orbital_partitioning(system)

        H, e_ref = shift_normal_order(H, occ_a, occ_b, occ_prev_a, occ_prev_b)

        occ_prev_a = occ_a.copy()
        occ_prev_b = occ_b.copy()