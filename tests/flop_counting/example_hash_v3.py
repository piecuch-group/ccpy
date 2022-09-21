import numpy as np
from numba import njit

def get_pspace(no, nu, p_rand):

    pspace = []
    for a in range(nu):
        for b in range(a + 1, nu):
            for c in range(b + 1, nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        for k in range(j + 1, no):
                            if p_rand >= np.random.rand():
                                pspace.append([a, b, c, i, j, k])

    return np.asarray(pspace, dtype=np.int32)

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

# @njit
# def get_bucket(key, nbits):
#     key = ((key >> nbits) ^ key) * 0x45d9f3b
#     key = ((key >> nbits) ^ key) * 0x45d9f3b
#     key = (key >> nbits) ^ key
#
#     mask = (1 << nbits) - 1
#
#     return key & mask - 1

# @njit
# def get_bucket(a, nbits):
#
#     a = (a+0x7ed55d16) + (a<<12)
#     a = (a^0xc761c23c) ^ (a>>19)
#     a = (a+0x165667b1) + (a<<5)
#     a = (a+0xd3a2646c) ^ (a<<9)
#     a = (a+0xfd7046c5) + (a<<3)
#     a = (a^0xb55a4f09) ^ (a>>16)
#
#     mask = (1 << nbits) - 1
#
#     return a & mask - 1

@njit
def get_number_hash_bits(num_keys):
    nbits_max = int(np.ceil(np.log2(num_keys + 1)))
    return nbits_max

@njit
def make_pspace_hash(no, nu, pspace, nbits=15):

    n_p = pspace.shape[0]
    n_buckets = 2**nbits - 1

    # Need a dry run to determine the maximum bucket depth for P space hash table
    counts_per_bucket = np.zeros(n_buckets, dtype=np.int32)
    for idet in range(n_p):
        index = sub2ind(pspace[idet, :], (nu, nu, nu, no, no, no))
        i_bucket = get_bucket(index, nbits)
        counts_per_bucket[i_bucket] += 1

    # Allocate P space hash table with determined maximum bucket depth
    hash_table = np.zeros((n_buckets, max(counts_per_bucket), 2), dtype=np.int32)
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

    return hash_table, counts_per_bucket, sum(counts_per_bucket) / len(counts_per_bucket)

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
        for i in range(len(temp)):
            if temp[i] == index:
                return hash_table[i_bucket, m-n-1+i, 1]
        return -1
    else:        # collision in full bucket
        temp = hash_table[i_bucket, :, 0].copy()
        for i in range(len(temp)):
            if temp[i] == index:
                return hash_table[i_bucket, i, 1]
        return -1

def main(no, nu, p_rand, nbits=0):

    # To appreciate size of problem, show size of T3 vector
    num_triples = no**3 * nu**3
    print("Total number of triples = ", num_triples)
    print("Size of associated T3 array = ", num_triples * 8 / 1e9, "GB")

    # Generate a ficticious P space with a certain fraction of triples
    pspace = get_pspace(no, nu, p_rand)
    print("P space generated")
    print("-----------------")
    print("Number of triples in P space = ", pspace.shape[0], "Memory ~ ", pspace.shape[0] * 8 / 1e9, "GB")
    print("")

    # Get the number of bits used to construct the hash table (2**n - 1) buckets
    if nbits == 0:
        nbits = get_number_hash_bits(pspace.shape[0])
    print("Hash table has", 2**nbits - 1, "buckets")
    print("")

    # Obtain the hash table storing (key, value) pairs, where key uniquely identifies (a,b,c,i,j,k) in the P space
    # and value identifies the index of this triple in the T3 vector
    hash_table, counts_per_bucket, average_depth = make_pspace_hash(no, nu, pspace, nbits)
    print("Hash table allocated (nbits =", nbits, "):")
    print("---------------------")
    print("Size of hash table = ", hash_table.shape, "Average bucket depth = ", round(average_depth), "Memory ~ ", np.prod(hash_table.shape) * 4 / 1e9, "GB")
    print("")

    return hash_table, counts_per_bucket, pspace, nbits

if __name__ == "__main__":

    no = 30
    nu = 100
    fraction = 0.1
    nbits = 0 # default to autoomatic ceil(log2(n_p))

    hash_table, counts_per_bucket, pspace, nbits = main(no, nu, fraction, nbits=nbits)

    for m in range(pspace.shape[0]):
        idx = get_from_hashtable(pspace[m, 0],pspace[m, 1],pspace[m, 2],
                                 pspace[m, 3],pspace[m, 4],pspace[m, 5],
                                 no,nu,nbits,hash_table,counts_per_bucket)

        try:
            assert(idx == m)
        except AssertionError:
                print(idx,m)
                print(pspace[m, :])
                index = sub2ind(pspace[m, :], (nu,nu,nu,no,no,no))
                bucket = get_bucket(index, nbits)
                print("index = ", index)
                print("bucket = ", bucket)
                print(hash_table[bucket, :, 0])
                print(hash_table[bucket, :, 1])
                break