import numpy as np
import time
from utilities import get_pspace

from binary_hash import make_pspace_hash as get_binary_hash
from binary_hash import get_from_hashtable as find_hash_binary

from prime_hash import make_pspace_hash as get_prime_hash
from prime_hash import get_from_hashtable as find_hash_prime

if __name__ == "__main__":

    no = 20
    nu = 60
    fraction = 0.1

    # To appreciate size of problem, show size of T3 vector
    num_triples = no**3 * nu**3
    print("Total number of triples = ", num_triples)
    print("Size of associated T3 array = ", num_triples * 8 / 1e9, "GB")

    # Generate a ficticious P space with a certain fraction of triples
    pspace, num_p_unique = get_pspace(no, nu, fraction)
    print("P space generated")
    print("-----------------")
    print("Number of triples in P space = ", pspace.shape[0], "Memory ~ ", pspace.shape[0] * 8 / 1e9, "GB")
    print("")

    print("Testing Binary hash:")
    t1 = time.time()
    hash_table, nbits, counts_per_bucket, average_depth = get_binary_hash(no, nu, pspace)
    t2 = time.time()
    print("Hash table allocated")
    print("---------------------")
    print("Size of hash table = ", hash_table.shape, "Average bucket depth = ", round(average_depth), "Memory ~ ", np.prod(hash_table.shape) * 8 / 1e9, "GB")
    print("Completed in", t2 - t1, "seconds")
    t1 = time.time()
    for m in range(pspace.shape[0]):
        idx = find_hash_binary(pspace[m, 0],pspace[m, 1],pspace[m, 2],
                                 pspace[m, 3],pspace[m, 4],pspace[m, 5],
                                 no,nu,nbits,hash_table,counts_per_bucket)

        assert(idx == m)
    t2 = time.time()
    print("Tested all keys in", t2 - t1, "seconds")
    print("")

    print("Testing Prime hash:")
    t1 = time.time()
    hash_table, prime, counts_per_bucket, average_depth = get_prime_hash(no, nu, pspace)
    t2 = time.time()
    print("Hash table allocated")
    print("---------------------")
    print("Size of hash table = ", hash_table.shape, "Average bucket depth = ", round(average_depth), "Memory ~ ", np.prod(hash_table.shape) * 8 / 1e9, "GB")
    print("Completed in", t2 - t1, "seconds")
    t1 = time.time()
    for m in range(pspace.shape[0]):
        idx = find_hash_prime(pspace[m, 0],pspace[m, 1],pspace[m, 2],
                                 pspace[m, 3],pspace[m, 4],pspace[m, 5],
                                 no,nu,prime,hash_table,counts_per_bucket)

        assert(idx == m)
    t2 = time.time()
    print("Tested all keys in ", t2 - t1, "seconds")
    print("")