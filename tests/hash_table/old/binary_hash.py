import numpy as np
from numba import njit
from utilities import sub2ind, binary_search

@njit
def get_bucket(key, nbits):
   mask = (1 << nbits) - 1
   return key & mask - 1

@njit
def get_number_hash_bits(num_keys):
    nbits_max = int(np.ceil(np.log2(num_keys + 1)))
    return nbits_max

@njit
def make_pspace_hash(no, nu, pspace, nbits=0):

    if nbits == 0:
        nbits = get_number_hash_bits(pspace.shape[0])

    n_p = pspace.shape[0]
    n_buckets = 2**nbits - 1

    # Need a dry run to determine the maximum bucket depth for P space hash table
    counts_per_bucket = np.zeros(n_buckets, dtype=np.int32)
    for idet in range(n_p):
        index = sub2ind(pspace[idet, :], (nu, nu, nu, no, no, no))
        i_bucket = get_bucket(index, nbits)
        counts_per_bucket[i_bucket] += 1

    # Allocate P space hash table with determined maximum bucket depth
    hash_table = np.zeros((n_buckets, max(counts_per_bucket), 2))
    counts_per_bucket = np.zeros(n_buckets, dtype=np.int32)
    for idet in range(n_p):
        index = sub2ind(pspace[idet, :], (nu, nu, nu, no, no, no))
        i_bucket = get_bucket(index, nbits)

        hash_table[i_bucket, counts_per_bucket[i_bucket], 0] = index # full key (always > 0)
        hash_table[i_bucket, counts_per_bucket[i_bucket], 1] = idet  # value corresponding to indical position in T3 vector

        counts_per_bucket[i_bucket] += 1

    # Sort the hash table by key so that each bucket can be binary searched
    first_nonzero_index = np.zeros(hash_table.shape[0])
    for i in range(hash_table.shape[0]):
        idx = np.argsort(hash_table[i, :, 0])
        temp1 = hash_table[i, :, 0].copy()
        temp2 = hash_table[i, :, 1].copy()
        for j in range(len(idx)):
            temp1[j] = hash_table[i, idx[j], 0]
            temp2[j] = hash_table[i, idx[j], 1]
        hash_table[i, :, 0] = temp1
        hash_table[i, :, 1] = temp2

        first_nonzero_index[i] = hash_table.shape[1] - counts_per_bucket[i]

    return hash_table, nbits, counts_per_bucket, sum(counts_per_bucket) / len(counts_per_bucket)

@njit
def get_from_hashtable(a, b, c, i, j, k, no, nu, nbits, hash_table, counts_per_bucket):

    index = sub2ind( (a,b,c,i,j,k), (nu,nu,nu,no,no,no) )
    i_bucket = get_bucket(index, nbits)

    n = counts_per_bucket[i_bucket]
    m = hash_table.shape[1]

    if n == 0:   # no population in bucket
        return -1
    elif n == 1: # only one element in bucket
        return hash_table[i_bucket, -1, 1]
    elif n < m:  # collision in bucket (not full)
        temp = hash_table[i_bucket, m-n-1:, 0].copy()
        i = binary_search(temp, index)
        if i == -1:
            return i
        else:
            return hash_table[i_bucket, m-n-1+i, 1]
    else:        # collision in full bucket
        temp = hash_table[i_bucket, :, 0].copy()
        i = binary_search(temp, index)
        if i == -1:
            return i
        else:
            return hash_table[i_bucket, i, 1]
