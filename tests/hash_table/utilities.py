import numpy as np
#from numba import njit
from itertools import permutations

def get_pspace(no, nu, p_rand):

    pspace = []
    num_p_unique = 0
    for a in range(nu):
        for b in range(a + 1, nu):
            for c in range(b + 1, nu):
                for i in range(no):
                    for j in range(i + 1, no):
                        for k in range(j + 1, no):

                            if p_rand >= np.random.rand():
                                num_p_unique += 1
                                for perms_unocc in permutations((a, b, c)):
                                    for perms_occ in permutations((i, j, k)):
                                        pspace.append([perms_unocc[0], perms_unocc[1], perms_unocc[2], perms_occ[0], perms_occ[1], perms_occ[2]])

    return np.asarray(pspace, dtype=np.int32), num_p_unique


def get_list_of_hashes(pspace, no, nu):

    list_of_hashes = []
    for m in range(pspace.shape[0]):
        index = sub2ind((pspace[m, 0], pspace[m, 1], pspace[m, 2], pspace[m, 3], pspace[m, 4], pspace[m, 5]), (nu, nu, nu, no, no, no))
        list_of_hashes.append(index)

    list_of_hashes = np.array(list_of_hashes)
    idx = np.argsort(list_of_hashes)

    return list_of_hashes[idx]

#@njit
def binary_search(arr, x):

    if len(arr) == 1:
        if arr[0] == x:
            return 0
        else:
            return -1

    low = 0
    high = len(arr) - 1
    mid = 0
    while low <= high:
        mid = (high + low) >> 1
        # If x is greater, ignore left half
        if arr[mid] < x:
            low = mid + 1
        # If x is smaller, ignore right half
        elif arr[mid] > x:
            high = mid - 1
        # means x is present at mid
        else:
            return mid
    # If we reach here, then the element was not present
    return -1

#@njit
def linear_search(arr, x):
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    else:
        return -1


#@njit
def sub2ind(x, shape):
    return (x[0]
            + shape[0] * x[1]
            + shape[0] * shape[1] * x[2]
            + shape[0] * shape[1] * shape[2] * x[3]
            + shape[0] * shape[1] * shape[2] * shape[3] * x[4]
            + shape[0] * shape[1] * shape[2] * shape[3] * shape[4] * x[5])

# @njit
# def sub2ind(x, shape):
#     return 31*(31*(31*(31*(31*x[0]+x[1])+x[2])+x[3])+x[4])+x[5]

#@njit
def cantor(i, j):
    x = (i + j) * (i + j + 1)
    x = x >> 1
    return x + j

# @njit
# def sub2ind(x, shape):
#     y = cantor(x[0], x[1])
#     y = cantor(y, x[2])
#     y = cantor(y, x[3])
#     y = cantor(y, x[4])
#     return cantor(y, x[5]) + 1

    # return (x[0]
    #         + shape[0] * x[1]
    #         + shape[0] * shape[1] * x[2]
    #         + shape[0] * shape[1] * shape[2] * x[3]
    #         + shape[0] * shape[1] * shape[2] * shape[3] * x[4]
    #         + shape[0] * shape[1] * shape[2] * shape[3] * shape[4] * x[5]) + 1
