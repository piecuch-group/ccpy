import numpy as np
from numba import njit

@njit
def sub2ind(x, shape):

    return (x[0]
            + shape[0] * x[1]
            + shape[0] * shape[1] * x[2]
            + shape[0] * shape[1] * shape[2] * x[3]
            + shape[0] * shape[1] * shape[2] * shape[3] * x[4]
            + shape[0] * shape[1] * shape[2] * shape[3] * shape[4] * x[5])

@njit
def get_bucket(key, nbits):
    mask = (1 << nbits) - 1
    return key & mask - 1

@njit
def get_number_hash_bits(num_keys):
    nbits_max = int(np.ceil(np.log2(num_keys + 1)))
    #nbits_min = 15
    return nbits_max
    #return int(0.5* (nbits_max + nbits_min))

# @njit
# def hash2(x):
#     x = ((x >> 16) ^ x) * 0x45d9f3b
#     x = ((x >> 16) ^ x) * 0x45d9f3b
#     x = (x >> 16) ^ x
#     return x

# @njit
# def bithash(index):
#     key = index & 0x0000FFFF # 16 least significant bits
#     bucket = index & 0xFFFF0000 # 16 most significant bits
#     return key, bucket

@njit
def get_pspace(no, nu, p_rand):
    n_p = 0
    pspace = np.zeros((nu, nu, nu, no, no, no), dtype=np.int32)
    for a in range(nu):
        for b in range(a + 1, nu):
            for c in range(b + 1, nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        for k in range(j + 1, no):

                            if np.random.rand() < p_rand:
                                n_p += 1
                                pspace[a, b, c, i, j, k] = 1
    return pspace, n_p

@njit
def allocate_hash_table(no, nu, pspace, nbits):

    n_buckets = 2**nbits - 1

    counts_per_bucket = np.zeros(n_buckets, dtype=np.int32)
    for a in range(nu):
        for b in range(a + 1, nu):
            for c in range(b + 1, nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        for k in range(j + 1, no):

                            if pspace[a, b, c, i, j, k] == 1:
                                index = sub2ind((a,b,c,i,j,k), (nu,nu,nu,no,no,no))
                                i_bucket = get_bucket(index, nbits=15)
                                #i_bucket = hash2(index)
                                #key, i_bucket = bithash(index)
                                counts_per_bucket[i_bucket] += 1

    max_depth = max(counts_per_bucket)
    average_depth = sum(counts_per_bucket) / len(counts_per_bucket)

    hash_table = np.zeros((n_buckets, max_depth, 2))

    return hash_table, average_depth

@njit
def fill_hash_table(no, nu, pspace, hash_table, nbits):

    counts_per_bucket = np.zeros(hash_table.shape[0], dtype=np.int32)
    value = 0
    for a in range(nu):
        for b in range(a + 1, nu):
            for c in range(b + 1, nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        for k in range(j + 1, no):

                            if pspace[a, b, c, i, j, k] == 1:
                                index = sub2ind((a,b,c,i,j,k), (nu,nu,nu,no,no,no))
                                i_bucket = get_bucket(index, nbits)
                                #key, i_bucket = bithash(index)
                                #i_bucket = hash2(index)

                                hash_table[i_bucket, counts_per_bucket[i_bucket], 0] = index
                                hash_table[i_bucket, counts_per_bucket[i_bucket], 1] = value

                                counts_per_bucket[i_bucket] += 1

                                value += 1
    return hash_table

def main(no, nu, p_rand):

    pspace, n_p = get_pspace(no, nu, p_rand)
    print("P space generated")
    print("-----------------")
    print("Number of triples in P space = ", n_p, "Memory ~ ", n_p * 8 / 1e9, "GB")
    print("")

    nbits = get_number_hash_bits(n_p)
    hash_table, average_depth = allocate_hash_table(no, nu, pspace, nbits)
    print("Hash table allocated (nbits =", nbits, "):")
    print("---------------------")
    print("Size of hash table = ", hash_table.shape, "Average bucket depth = ", round(average_depth), "Memory ~ ", np.prod(hash_table.shape) * 4 / 1e9, "GB")
    print("")

    hash_table = fill_hash_table(no, nu, pspace, hash_table, nbits)
    print("Hash table filled", "Load factor = ", round(n_p / np.prod(hash_table.shape) * 100), "%")

if __name__ == "__main__":

    main(20, 60, 0.3)