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
    print("making bucket address array")
    counts_per_bucket = np.zeros(n_buckets, dtype=np.int32)
    bucket_address = np.arange(n_buckets, dtype=np.int32)
    for idet in range(n_p):
        index = sub2ind(pspace[idet, :], (nu, nu, nu, no, no, no))
        i_bucket = get_bucket(index, nbits)
        if i_bucket == n_buckets - 1:
            bucket_address[-1] += 1
        else:
            for i in range(i_bucket + 1, n_buckets):
                bucket_address[i] += 1

        counts_per_bucket[i_bucket] += 1
    print("finished bucket address array")

    # # This is slow!
    # print("making bucket address")
    # numel = 0
    # for i in range(n_buckets):
    #     #print(i)
    #     if counts_per_bucket[i] == 0:
    #         numel += 1
    #     else:
    #         numel += counts_per_bucket[i]
    #
    #     for j in range(i):
    #         if counts_per_bucket[j] == 0:
    #             bucket_address[i] += 1
    #         else:
    #             bucket_address[i] += counts_per_bucket[j]
    # print("finished bucket address")

    print("making hash table")
    hash_table = np.zeros((numel, 2))
    counts_per_bucket = np.zeros(n_buckets, dtype=np.int32)
    for idet in range(n_p):
        index = sub2ind(pspace[idet, :], (nu, nu, nu, no, no, no))
        i_bucket = get_bucket(index, nbits)

        hash_table[bucket_address[i_bucket] + counts_per_bucket[i_bucket], 0] = index # full key (always > 0)
        hash_table[bucket_address[i_bucket] + counts_per_bucket[i_bucket], 1] = idet  # value corresponding to indical position in T3 vector

        counts_per_bucket[i_bucket] += 1
    print("finished hash table")

    print("sorting hash table")
    for i in range(n_buckets - 1):
        temp = hash_table[bucket_address[i]:bucket_address[i + 1], 0].copy()
        idx = np.argsort(temp)
        temp = temp[idx]
        hash_table[bucket_address[i]:bucket_address[i + 1], 0] = temp

        temp = hash_table[bucket_address[i]:bucket_address[i + 1], 1].copy()
        hash_table[bucket_address[i]:bucket_address[i + 1], 1] = temp[idx]
    print("finished sorting")

    return hash_table, nbits, bucket_address, sum(counts_per_bucket) / len(counts_per_bucket)

@njit
def get_from_hashtable(a, b, c, i, j, k, no, nu, nbits, hash_table, bucket_address):

    index = sub2ind( (a,b,c,i,j,k), (nu,nu,nu,no,no,no) )
    i_bucket = get_bucket(index, nbits)

    if i_bucket == len(bucket_address):
        keys = hash_table[bucket_address[i_bucket]:, 0]
        values = hash_table[bucket_address[i_bucket]:, 1]
    else:
        keys = hash_table[bucket_address[i_bucket]:bucket_address[i_bucket + 1], 0]
        values = hash_table[bucket_address[i_bucket]:bucket_address[i_bucket + 1], 1]

    if len(keys) == 1:
        if keys[0] == index:
            return values[0]
        else:
            return -1
    else:
        idx = binary_search(keys, index)
        if idx == -1:
            return -1
        else:
            return values[idx]


