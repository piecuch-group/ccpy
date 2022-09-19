"""
In this module, we are trying to come up with good hashing functions to
represent the T3A, T3B, T3C, and T3D operators that respect the required
permutational symmetry of indices. The goal is to have the hashing function
simply return the index in the associated T vector of the tuple (a,b,c,i,j,k).
"""

# There is a snag:
# How to we account for non-contiguous nature of the P space? The relevant indices
# resulting from the hash will need to be re-indexed through another array (out of RAM),
# thus we are again very expensive...
#
# The only solution seems to be brute force... we are asking whether a given tuple (a,b,c,i,j,k)
# lies in the P space or not without storing the whole P space. Thus, the P pspace can be stored
# using 6 arrays, each of length NT_p.
# i_p = (...), j_p = (...), k_p = (...), a_p = (...), b_p = (...), and c_p = (...), where
# each triple is indexed by (a_p(idx), b_p(idx), c_p(idx), i_p(idx), j_p(idx), k_p(idx)), for all
# idx = 1, ..., NT_p.
#
# What we can do is the cache/brute approach. Pick an amount of no and nu that you are willing to
# allocate in memory (around Fermi level) to create the RAM p_space(nu,nu,nu,no,no,no). Identify
# the list of P space triples that exist in this space and a separate list that are out of this space.
# Have a function called
#
# idx = get_pspace_idx(a, b, c, i, j, k, p_space_RAM,p_space_complementary)
# if (idx /= 0) then
#    res += H(p,q,r,s) * t(idx)
#
# function get_pspace_idx(a,b,c,i,j,k,p_space_RAM,p_space_complementary)
# if (a,b,c,i,j,k) in p_space_RAM: # cheap exit if the tuple lies in the P space array
#     return idx = p_space_RAM(a, b, c, i, j, k)
# else: # brute force linear search if you are not in array, you cannot do better than linear for unsorted arrays
#     for (a',b',c',i',j',k') in p_space_complementary:
#         if a,b,c,i,j,k  == (a',b',c',i',j',k')
#            return idx
#
#
# We could define p_space_complementary_sorted = sort(hash3(pspace_complementary)). Then look for hash3(a,b,c,i,j,k)
# in this list using binary search (N * log(N))

import numpy as np

def hash2(i, j, k, l):

    # p = min([i, j])
    # r = max([i, j])
    #
    # q = min([k, l])
    # s = max([k, l])
    #
    # p += (r * r - r) >> 1
    # q += (s * s - s) >> 1
    #
    # v = min([p, q])
    # w = max([p, q])
    #
    # v += (w * w - w) >> 1
    p = min([i, j])
    q = max([i, j])

    r = min([k, l])
    s = max([k, l])

    p += (q * q - q) >> 1
    r += (s * s - s) >> 1

    v = min([p, r])
    w = max([p, r])

    v += (w * w - w) >> 1

    return v
    #return (v >> 15), v
    #return v & (~0 << 1), v

def hash3(i, j, k, l, m, n):
    p = min([i, j, k])
    q = max([i, j, k])

    r = min([l, m, n])
    s = max([l, m, n])

    p += (q * q - q) >> 1
    r += (s * s - s) >> 1

    v = min([p, r])
    w = max([p, r])

    v += (w * w - w) >> 1

    return v


if __name__ == "__main__":

    nmo = 20

    #print(hash2(2, 2, 1, 1))

    arr_hash = []
    ijkl = []
    ct = 1
    for i in range(1, nmo + 1):
        print(i)
        for j in range(1, i + 1):
            ij = i * (i - 1)/2 + j
            for k in range(1, nmo + 1):
                for l in range(1, k + 1):
                    kl = k * (k - 1)/2 + l
                    if ij >= kl:

                        #print("ijkl = ", i,j,k,l)
                        hash_ijkl = hash2(i, j, k, l)

                        if (ct != hash_ijkl):
                            print("index wrong")
                            break

                        # these permutations are respected
                        hash_jikl = hash2(j, i, k, l)
                        hash_ijlk = hash2(i, j, l, k)
                        hash_jilk = hash2(j, i, l, k)

                        # complex conjugation symmetry is not respected (good!)
                        #hash_klij = hash2(k + 1, l + 1, i + 1, j + 1)

                        if (hash_jikl != hash_ijkl):
                            print("permutation wrong")
                            break
                        if (hash_ijlk != hash_ijkl):
                            print("permutation wrong")
                            break
                        if (hash_jilk != hash_ijkl):
                            print("permutation wrong")
                            break



                        #arr_hash.append(hash_val)
                        #ijkl.append((i, j, k, l))
                        ct += 1

    # idx = np.argsort(arr_hash)
    # for i in idx:
    #     print(ijkl[i])

    # for i in range(1, nmo + 1):
    #     for j in range(1, nmo + 1):
    #         for k in range(1, nmo + 1):
    #             for l in range(1, nmo + 1):
    #                 print(hash2(i, k, j, l))