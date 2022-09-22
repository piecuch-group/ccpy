import numpy as np
from numba import njit
import argparse
import time
from utilities import get_pspace, binary_search, sub2ind, get_list_of_hashes


# [TODO]:
# Change key to first 15 bits of index and store that. Thus, bucket is last 15 bits, key is first 15 bits, so
# uniqueness is guaranteed. Each bucket holds 2**15 - 1 keys and each key is 2 bytes (1 byte = 8 bits). Thus, each bucket
# is ~65 KiB, which fits into L1 cache (or something close by), making binary search very fast.

@njit
def find_hash_binary_search(a, b, c, i, j, k, no, nu, list_of_hashes):
    index = sub2ind( (a, b, c, i, j, k), (nu, nu, nu, no, no, no) )
    return binary_search(list_of_hashes, index)

@njit
def test_binary_search(list_of_hashes, no, nu):

    for a in range(nu):
        for b in range(nu):
            for c in range(nu):
                for i in range(no):
                    for j in range(no):
                        for k in range(no):
                            idx = find_hash_binary_search(a, b, c, i, j, k, no, nu, list_of_hashes)
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

    # Get an ordered list of all keys (index) in the pspace
    print("Getting list of keys")
    list_of_hashes = get_list_of_hashes(pspace, no, nu)
    print("")

    # Test key fetch using fully binary search
    print("Testing key fetch...")
    t1 = time.time()
    test_binary_search(list_of_hashes, no, nu)
    t2 = time.time()
    print("Took", t2 - t1)
    print("Average time per lookup = ", (t2 - t1)/(no**3 * nu**3))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for testing binary hash table.")
    parser.add_argument("no", type=int, help="Number of occupied spinorbitals")
    parser.add_argument("nu", type=int, help="Number of unoccupied spinorbitals")
    parser.add_argument("fraction", type=float, help="Fraction of triples space in P space")
    args = parser.parse_args()

    main(args)

