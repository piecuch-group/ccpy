import numpy as np
from numba import njit
from utilities import sub2ind, binary_search, linear_search

# @njit
# def get_bucket(key):
#    mask = (1 << 16) - 1
#    return (key-1) & mask - 1
#
# @njit
# def get_key(key):

@njit
def get_key(index):
    return index & 0xFFFF0000 + 1

@njit
def get_bucket(index):
    return index & 0x0000FFFF - 1

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

    bucket_address = np.zeros(n_buckets, dtype=np.int32)
    for i in range(1, n_buckets):
        bucket_address[i] = sum(counts_per_bucket[:i])
        
    for i in id_0:
        counts_per_bucket[i] = 0

    numel = bucket_address[-1] + counts_per_bucket[-1]
    #print(counts_per_bucket)
    #print(bucket_address)

    # This is slow!
   # print("making bucket address")
   # numel = 0
   # bucket_address = np.zeros(n_buckets, dtype=np.int32)
   # for i in range(n_buckets):
   #      if counts_per_bucket[i] == 0:
   #          numel += 1
   #      else:
   #          numel += counts_per_bucket[i]
   # 
    #     for j in range(i):
    #         if counts_per_bucket[j] == 0:
    #             bucket_address[i] += 1
    #         else:
    #             bucket_address[i] += counts_per_bucket[j]
    #print(counts_per_bucket)
    #print(bucket_address)
    #print("finished bucket address")
    
    
    #numel = 0
    #for i in range(n_buckets):
    #    if counts_per_bucket[i] == 0:
    #        numel += 1
    #    else:
    #        numel += counts_per_bucket[i]
    #assert(numel == bucket_address[-1] + counts_per_bucket[-1])

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
        i_bucket = get_bucket(index)
        key = get_key(index)

        hash_table[bucket_address[i_bucket] + counts_per_bucket[i_bucket], 0] = key   # full key (always > 0)
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
    # sort the last bucket manually
    temp = hash_table[bucket_address[-1]:, 0].copy()
    idx = np.argsort(temp)
    temp = temp[idx]
    hash_table[bucket_address[-1]:, 0] = temp

    temp = hash_table[bucket_address[-1]:, 1].copy()
    hash_table[bucket_address[-1]:, 1] = temp[idx]
    print("finished sorting")

    return hash_table, bucket_address

@njit
def get_from_hashtable(a, b, c, i, j, k, no, nu, hash_table, bucket_address):

    index = sub2ind( (a,b,c,i,j,k), (nu,nu,nu,no,no,no) )
    i_bucket = get_bucket(index)
    key = get_key(index) # key we want, always > 0

    if i_bucket == len(bucket_address) - 1:
        keys = hash_table[bucket_address[i_bucket]:, 0]
        values = hash_table[bucket_address[i_bucket]:, 1]
    else:
        keys = hash_table[bucket_address[i_bucket]:bucket_address[i_bucket + 1], 0]
        values = hash_table[bucket_address[i_bucket]:bucket_address[i_bucket + 1], 1]

    if len(keys) == 1:
        if keys[0] == key:
            return values[0]
        else:
            return -1
    else:
        idx = binary_search(keys, key)
        if idx == -1:
            return -1
        else:
            return values[idx]


