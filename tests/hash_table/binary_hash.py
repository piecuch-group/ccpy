import numpy as np
from numba import njit
from utilities import sub2ind, binary_search

import sys

@njit
def get_bucket(index):
   #mask = (1 << 16) - 1
   return (index - 1) & ((1 << 16) - 1) - 1

# key and bucket codes need to be written in Fortran as well

# check that the keys and buckets actually return only 2 byte integers
# bucket and key cannot coincide for 2 different determinants. So if bucket is the same, key must be different.
@njit
def get_key(index):
    return index & 0xFFFF0000 + 1 # does this need to be > 0???
    #return (-(~x))

# @njit
# def get_bucket(index):
#     return index & 0x0000FFFF - 1
#     #x = index & 0x0000FFFF
#     #return ((x << 1) + (~x))

# hash table should be a 2D array of 8-bit integers (2 bytes)
@njit
def make_pspace_hash(no, nu, pspace):

    n_p = pspace.shape[0]
    n_buckets = 2**16 - 1

    counts_per_bucket = np.zeros(n_buckets, dtype=np.int32)
    for idet in range(n_p):
        index = sub2ind(pspace[idet, :], (nu, nu, nu, no, no, no))
        i_bucket = get_bucket(index)
        counts_per_bucket[i_bucket] += 1

    id_0 = []
    for i in range(n_buckets):
        if counts_per_bucket[i] == 0:
            id_0.append(i)
            counts_per_bucket[i] = 1

    bucket_address = np.zeros(n_buckets + 1, dtype=np.int32)
    for i in range(1, n_buckets + 1):
        bucket_address[i] = sum(counts_per_bucket[:i])
        
    for i in id_0:
        counts_per_bucket[i] = 0

    hash_table = np.zeros((bucket_address[-1], 2))
    counts_per_bucket = np.zeros(n_buckets, dtype=np.int32)
    for idet in range(n_p):
        index = sub2ind(pspace[idet, :], (nu, nu, nu, no, no, no))
        i_bucket = get_bucket(index)
        key = get_key(index)

        hash_table[bucket_address[i_bucket] + counts_per_bucket[i_bucket], 0] = key   # full key (always > 0)
        hash_table[bucket_address[i_bucket] + counts_per_bucket[i_bucket], 1] = idet  # value corresponding to indical position in T3 vector

        counts_per_bucket[i_bucket] += 1

    # sort the keys in each bucket in the table to enable fast binary searching
    for i in range(n_buckets):
        temp = hash_table[bucket_address[i]:bucket_address[i + 1], 0].copy()
        idx = np.argsort(temp)
        temp = temp[idx]
        hash_table[bucket_address[i]:bucket_address[i + 1], 0] = temp

        temp = hash_table[bucket_address[i]:bucket_address[i + 1], 1].copy()
        hash_table[bucket_address[i]:bucket_address[i + 1], 1] = temp[idx]

    return hash_table, bucket_address

# this code would be written in Fortran and called within the CC(P) update loops
@njit
def get_from_hashtable(a, b, c, i, j, k, no, nu, hash_table, bucket_address):

    index = sub2ind( (a,b,c,i,j,k), (nu,nu,nu,no,no,no) ) # slowest line
    i_bucket = get_bucket(index)

    idx = binary_search(hash_table[bucket_address[i_bucket]:bucket_address[i_bucket + 1], 0],
                        get_key(index)) # this is not bad actually
    if idx == -1:
        return -1
    else:
        return hash_table[bucket_address[i_bucket] + idx, 1]



