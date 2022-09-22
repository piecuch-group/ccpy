import numpy as np
from matplotlib import pyplot as plt

# For 32- or 64-bit integers:
#   ADD - 1
#   MUL - 3
#   BIT - 1
# For 64-bit floating points:
#   ADD - 3
#   MUL - 5

# Hash function lookups should have O(1) time:
#   - for us, the index function is 15 integer multiplications : 15 * 3 = 45
#   - key and bucket evaluation are bit operations : 2 * 1 = 2
#   - search within bucket : hard to predict, but assume fast : 15
#   - table lookup is a memory access from RAM : 250
#   - if-statement evaluation: 25 - 30 at most (branch misprediction)
#   - total : 250 + 45 + 30 + + 15 + 2 = 342
#
# FLOP computation: res = res + h(...) * t3(...)
#   - indexing of H and t3 vector are memory accesses : 250 * 2 = 500
#   - FLOP execution : 5 (MUL) + 3 (ADD) = 8
#   - total : 500 + 8 = 508 cycles

def hash_scheme(f, no, nu, cost_hash=342, cost_flop=508, cost_memory=250, g=0.0):

    num_triples = no**3 * nu**3
    num_triples_p = f * num_triples
    iters_per_triple = 3 * no + 3 * nu + 9 * no * nu + 3 * no**2 + 3 * nu**2 + no*nu**2 + nu*no**2

    c_hash = num_triples_p * iters_per_triple * cost_hash
    c_flop = num_triples_p * iters_per_triple * cost_flop * f

    cost = c_flop + c_hash
    return cost


def cost_original(no, nu, cost_flop=508):

    num_triples = no**3 * nu**3
    iters_per_triple = 3 * no + 3 * nu + 9 * no * nu + 3 * no ** 2 + 3 * nu ** 2 + no * nu ** 2 + nu * no ** 2

    c_flop = num_triples * iters_per_triple * cost_flop

    return c_flop


if __name__ == "__main__":

    no = 100
    nu = 500

    m = 1000

    fraction = np.linspace(0.0, 0.2, m)
    my_speedup_ccsdt = np.zeros(m - 1)
    my_speedup_slater = np.zeros(m - 1)

    for i in range(m - 1):
        my_speedup_ccsdt[i] = cost_original(no, nu) / hash_scheme(fraction[i + 1], no, nu)
        #my_speedup_slater[i] = cost_slater(fraction[i + 1], no, nu) / hash_scheme(fraction[i + 1], no, nu)

    plt.plot(fraction[1:] * 100, my_speedup_ccsdt, label="Expacted speedup")
    plt.plot(fraction[1:] * 100, np.ones(m - 1), color='black')
    plt.plot(fraction[1:] * 100, 1.0/ (fraction[1:]), color='red', label="Linear speedup")
    plt.xlabel("% triples in P space")
    plt.ylabel("Speedup")
    plt.show()