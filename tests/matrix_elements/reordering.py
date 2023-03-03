import numpy as np

def fseq(s1, s2):

    n = 0
    for p in range(len(s1)):
        if s1[p] == s2[p]: n += 1
    return n

if __name__ == "__main__":

    i = 1
    j = 3
    k = 5

    l = 3
    m = 4
    n = 5

    # test out each permutation

    # (1)
    f1 = fseq([i, j], [l, m])
    print(f1)
    # (ij)
    f2 = fseq([j, i], [l, m])
    print(f2)
    # (lm)
    f3 = fseq([i, j], [m, l])
    print(f3)
    # (ij)(lm)
    f4 = fseq([j, i], [m, l])
    print(f4)

    imax = max([f1, f2, f3, f4])
    print(imax)
