import numpy as np
#from numba import njit
import argparse
import time
from utilities import get_pspace

from binary_hash import make_pspace_hash as get_binary_hash
from binary_hash import get_from_hashtable as find_hash_binary

# [TODO]:
# Change key to first 15 bits of index and store that. Thus, bucket is last 15 bits, key is first 15 bits, so
# uniqueness is guaranteed. Each bucket holds 2**15 - 1 keys and each key is 2 bytes (1 byte = 8 bits). Thus, each bucket
# is ~65 KiB, which fits into L1 cache (or something close by), making binary search very fast.

#@njit
def test_hash_key(hash_table, bucket_address, no, nu):

    for a in range(nu):
        for b in range(nu):
            for c in range(nu):
                for i in range(no):
                    for j in range(no):
                        for k in range(no):
                            idx = find_hash_binary(a, b, c, i, j, k, no, nu, hash_table, bucket_address)
    return

#@njit
def test_hits(pspace, hash_table, bucket_address, no, nu):

    for m in range(pspace.shape[0]):
        idx = find_hash_binary(pspace[m, 0], pspace[m, 1], pspace[m, 2], pspace[m, 3], pspace[m, 4], pspace[m, 5], no, nu, hash_table, bucket_address)
        assert(idx == m)
    return


def main(args):

    no = args.no
    nu = args.nu
    fraction = args.fraction

    # To appreciate size of problem, show size of T3 vector
    num_triples = no**3 * nu**3
    print("Total number of triples = ", num_triples, "Memory ~", num_triples * 8 / 1e9, "GB")

    # Generate a ficticious P space with a certain fraction of triples
    pspace, num_p_unique = get_pspace(no, nu, fraction)
    print("P space generated")
    print("-----------------")
    print("Number of triples in P space = ", pspace.shape[0], "Memory ~", pspace.shape[0] * 8 / 1e9, "GB")
    print("")

    t1 = time.time()
    hash_table, bucket_address = get_binary_hash(no, nu, pspace)
    t2 = time.time()
    print("Hash table allocated")
    print("---------------------")
    print("Size of hash table = ", hash_table.shape, "Memory ~", np.prod(hash_table.shape) * 8 / 1e9, "GB")
    print("Completed in", t2 - t1, "seconds")

    print("Testing key fetch...")
    t1 = time.time()
    test_hits(pspace, hash_table, bucket_address, no, nu)
    t2 = time.time()
    print("Total time = ", t2 - t1)
    print("Average time per key (hits) = ", (t2 - t1) / (pspace.shape[0]))

    t1 = time.time()
    test_hash_key(hash_table, bucket_address, no, nu)
    t2 = time.time()
    print("Total time = ", t2 - t1)
    print("Average time per key (overall) = ", (t2 - t1) / (no**3 * nu**3))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for testing binary hash table.")
    parser.add_argument("no", type=int, help="Number of occupied spinorbitals")
    parser.add_argument("nu", type=int, help="Number of unoccupied spinorbitals")
    parser.add_argument("fraction", type=float, help="Fraction of triples space in P space")
    args = parser.parse_args()

    main(args)

