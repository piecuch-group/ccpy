import numpy as np
from numba import njit

def get_key(a, b, c, i, j, k):
    key = np.ravel_multi_index((a,b,c,i,j,k), (nu,nu,nu,no,no,no))
    return key

def get_bucket(key, sze):
    return key % sze

@njit
def get_pspace(no, nu, p_rand):
    n_p = 0
    pspace = np.zeros((nu, nu, nu, no, no, no))
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

def allocate_hash_table(no, nu, pspace, sze):

    max_bucket = 0
    counts_per_bucket = np.zeros(sze, dtype=np.int32)
    for a in range(nu):
        for b in range(a + 1, nu):
            for c in range(b + 1, nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        for k in range(j + 1, no):

                            if pspace[a, b, c, i, j, k] == 1:
                                key = np.ravel_multi_index( (a,b,c,i,j,k), (nu,nu,nu,no,no,no) )
                                i_bucket = key % sze
                                counts_per_bucket[i_bucket] += 1
                                max_bucket = max([i_bucket + 1, max_bucket])

    max_depth = max(counts_per_bucket)
    average_depth = sum(counts_per_bucket) / len(counts_per_bucket)
    hash_table = np.zeros((max_bucket, max_depth, 2))

    return hash_table, average_depth

def fill_hash_table(no, nu, pspace, sze, hash_table):

    counts_per_bucket = np.zeros(hash_table.shape[0], dtype=np.int32)
    value = 0
    for a in range(nu):
        for b in range(a + 1, nu):
            for c in range(b + 1, nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        for k in range(j + 1, no):

                            if pspace[a, b, c, i, j, k] == 1:
                                key = np.ravel_multi_index( (a,b,c,i,j,k), (nu,nu,nu,no,no,no) )
                                i_bucket = key % sze

                                hash_table[i_bucket, counts_per_bucket[i_bucket], 0] = key
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

    large_prime_number = 393241
    hash_table, average_depth = allocate_hash_table(no, nu, pspace, large_prime_number)
    print("Hash table allocated:")
    print("---------------------")
    print("Size of hash table = ", hash_table.shape, "Average bucket depth = ", round(average_depth), "Memory ~ ", np.prod(hash_table.shape) * 4 / 1e9, "GB")
    print("")

    hash_table = fill_hash_table(no, nu, pspace, large_prime_number, hash_table)
    print("Hash table filled", "Load factor = ", round(n_p / np.prod(hash_table.shape) * 100), "%")

if __name__ == "__main__":

    main(20, 40, 0.01)


