import numpy as np
from utilities import antisymmetrize_aaaa, antisymmetrize_aaa, antisymmetrize_aa

def get_random_t(nocc, nunocc):

    t1 = np.random.rand(nunocc, nocc)
    t2 = np.random.rand(nunocc, nunocc, nocc, nocc)
    t3 = np.random.rand(nunocc, nunocc, nunocc, nocc, nocc, nocc)
    t4 = np.random.rand(nunocc, nunocc, nunocc, nunocc, nocc, nocc, nocc, nocc)

    return t1, antisymmetrize_aa(t2), antisymmetrize_aaa(t3), antisymmetrize_aaaa(t4)
