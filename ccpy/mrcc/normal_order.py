import numpy as np

def shift_normal_order(H, occ_a, occ_b, occ_a_prev, occ_b_prev):

    # de-normal order Z = F - G
    if occ_a_prev: # if occ_a_prev /= []
        H.a -= np.einsum("piqi->pq", H.aa[:, occ_a_prev, :, occ_a_prev], optimize=True)
        H.b -= np.einsum("ipiq->pq", H.ab[occ_a_prev, :, occ_a_prev, :], optimize=True)
    if occ_b_prev: # if occ_b_prev /= []
        H.a -= np.einsum("piqi->pq", H.ab[:, occ_b_prev, :, occ_b_prev], optimize=True)
        H.b -= np.einsum("piqi->pq", H.bb[:, occ_b_prev, :, occ_b_prev], optimize=True)

    # normal order F = Z + G
    H.a += np.einsum("piqi->pq", H.aa[:, occ_a, :, occ_a], optimize=True)
    H.b += np.einsum("ipiq->pq", H.ab[occ_a, :, occ_a, :], optimize=True)
    H.a += np.einsum("piqi->pq", H.ab[:, occ_b, :, occ_b], optimize=True)
    H.b += np.einsum("piqi->pq", H.bb[:, occ_b, :, occ_b], optimize=True)

    return H