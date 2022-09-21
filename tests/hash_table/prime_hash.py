import numpy as np
from numba import njit
from utilities import binary_search, sub2ind

@njit
def get_bucket(key, prime):
    return key % prime

@njit
def get_large_prime(num_keys):
    lwr = np.arange(5, 31)
    upr = np.arange(6, 32)

    primes = np.array([53, 97, 193, 389, 769, 1543,
                       3079, 6151, 12289, 24593, 49157,
                       98317, 196613, 393241, 786433, 1572869,
                       3145739, 6291469, 12582917, 25165843, 50331653,
                       100663319, 201326611, 402653189, 805306457, 1610612741])

    for i in range(len(lwr)):
        if num_keys > 2**lwr[i] and num_keys < 2**upr[i]:
            return primes[i]
    else:
        raise RuntimeError("No suitable prime found for number of keys!")

@njit
def make_pspace_hash(no, nu, pspace, prime=0):

    if prime == 0:
        prime = get_large_prime(pspace.shape[0])

    n_p = pspace.shape[0]
    n_buckets = prime

    # Need a dry run to determine the maximum bucket depth for P space hash table
    counts_per_bucket = np.zeros(n_buckets, dtype=np.int32)
    for idet in range(n_p):
        index = sub2ind(pspace[idet, :], (nu, nu, nu, no, no, no))
        i_bucket = get_bucket(index, prime)
        counts_per_bucket[i_bucket] += 1

    # Allocate P space hash table with determined maximum bucket depth
    hash_table = np.zeros((n_buckets, max(counts_per_bucket), 2))
    counts_per_bucket = np.zeros(n_buckets, dtype=np.int32)
    for idet in range(n_p):
        index = sub2ind(pspace[idet, :], (nu, nu, nu, no, no, no))
        i_bucket = get_bucket(index, prime)

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

    return hash_table, prime, counts_per_bucket, sum(counts_per_bucket) / len(counts_per_bucket)

@njit
def get_from_hashtable(a, b, c, i, j, k, no, nu, prime, hash_table, counts_per_bucket):

    index = sub2ind( (a,b,c,i,j,k), (nu,nu,nu,no,no,no) )
    i_bucket = get_bucket(index, prime)

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