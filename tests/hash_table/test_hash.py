import numpy as np
from numba import njit
import argparse
import time
from utilities import get_pspace, sub2ind, binary_search

from binary_hash import make_pspace_hash as get_binary_hash
from binary_hash import get_from_hashtable as find_hash_binary
from binary_hash import get_bucket

@njit
def test_hash_key(hash_table, bucket_address, no, nu, nbits):

    for a in range(nu):
        for b in range(nu):
            for c in range(nu):
                for i in range(no):
                    for j in range(no):
                        for k in range(no):
                            idx = find_hash_binary(a, b, c, i, j, k, no, nu, nbits, hash_table, bucket_address)
    return

def main(args):

    no = args.no
    nu = args.nu
    fraction = args.fraction
    nbits = args.nbits

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
    hash_table, nbits, bucket_address = get_binary_hash(no, nu, pspace, nbits=nbits)
    t2 = time.time()
    print("Hash table allocated")
    print("---------------------")
    print("Size of hash table = ", hash_table.shape, "Memory ~", np.prod(hash_table.shape) * 8 / 1e9, "GB")
    print("Completed in", t2 - t1, "seconds")

    print("Testing key fetch...")
    t1 = time.time()
    test_hash_key(hash_table, bucket_address, no, nu, nbits)
    t2 = time.time()
    print("Total time = ", t2 - t1)
    print("Average time per key = ", (t2 - t1) / (no**3 * nu**3))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parser for testing binary hash table.")
    parser.add_argument("no", type=int, help="Number of occupied spinorbitals")
    parser.add_argument("nu", type=int, help="Number of unoccupied spinorbitals")
    parser.add_argument("fraction", type=float, help="Fraction of triples space in P space")
    parser.add_argument("-n", "--nbits", type=int, default=0, required=False, help="Number of bits to be used")
    args = parser.parse_args()

    main(args)

