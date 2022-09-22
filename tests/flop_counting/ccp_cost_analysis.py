import numpy as np
from matplotlib import pyplot as plt


def cost_slater(f, no, nu, cost_hash=10, cost_flop=20, cost_memory=250):

    num_triples = no**3 * nu**3
    num_triples_p = f * num_triples

    # realistic scenario, perform bitwise analysis on every pair of determinants in P space

    # this accounts for a quick bitwise comparison between determinants
    c_hash = num_triples_p**2 * cost_hash
    # this is the flop cost per pair of determinant, assuming that a small fraction of pairs (0.01% actually result in computations)
    c_flop = num_triples_p ** 2 * cost_flop * 0.0001

    return c_hash + c_flop

def hash_scheme(f, no, nu, cost_hash=120, cost_flop=20, cost_memory=250, g=0.0):

    num_triples = no**3 * nu**3
    num_triples_p = f * num_triples
    iters_per_triple = 3 * no + 3 * nu + 9 * no * nu + 3 * no**2 + 3 * nu**2 + no*nu**2 + nu*no**2

    # binary search has time scaling as log2(N)
    #cost_search = ( np.log2(num_triples_p) + cost_memory) * (1 - g)

    #c_hash = num_triples_p * iters_per_triple * (g * cost_memory + (1 - g) * (cost_hash + cost_search))
    c_hash = num_triples_p * iters_per_triple * cost_hash
    c_flop = num_triples_p * iters_per_triple * cost_flop * f

    cost = c_flop + c_hash
    return cost

def search_scheme(f, no, nu, cost_hash=120, cost_flop=20, cost_memory=250, g=0.0):

    num_triples = no**3 * nu**3
    num_triples_p = f * num_triples
    iters_per_triple = 3 * no + 3 * nu + 9 * no * nu + 3 * no**2 + 3 * nu**2 + no*nu**2 + nu*no**2

    # binary search has time scaling as log2(N)
    cost_search = ( np.log2(num_triples_p) + cost_memory) * (1 - g)

    c_hash = num_triples_p * iters_per_triple * (g * cost_memory + (1 - g) * (cost_hash + cost_search))
    c_flop = num_triples_p * iters_per_triple * cost_flop * f

    cost = c_flop + c_hash
    return cost


def cost_original(no, nu, cost_flop=20):

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
        my_speedup_slater[i] = cost_slater(fraction[i + 1], no, nu) / hash_scheme(fraction[i + 1], no, nu)

    plt.plot(fraction[1:] * 100, my_speedup_ccsdt, label="CCSDT")
    plt.plot(fraction[1:] * 100, my_speedup_slater, label="Slater")
    plt.xlabel("% triples in P space")
    plt.ylabel("Speedup")
    plt.show()