
def calculate_permutation_parity(lst):
    """Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd."""
    parity = 1.0
    for i in range(0, len(lst) - 1):
        if lst[i] != i:
            parity *= -1.0
            mn = min(range(i, len(lst)), key=lst.__getitem__)
            lst[i], lst[mn] = lst[mn], lst[i]
    return parity

def do_lineup_permutation(f1, f2):
    """Given two lists f1 and f2, returns their sorted counterparts in
    order of maximum coincidence along with the associated phase of permutation
    for each list."""

    f1 = list(sorted(f1))
    f2 = list(sorted(f2))

    # find the positions of the indices common to both lists
    i, j = 0, 0
    idx_common_1 = []
    idx_common_2 = []
    # merge sort O( n*log(m) ), where n
    while i < len(f1) and j < len(f2):
        if f1[i] == f2[j]:
            idx_common_1.append(i)
            idx_common_2.append(j)
            i += 1
            j += 1
        elif f1[i] < f2[j]:
            i += 1
        else:
            j += 1
    idx_different_1 = [i for i in range(len(f1)) if i not in idx_common_1]
    idx_different_2 = [i for i in range(len(f2)) if i not in idx_common_2]

    p1 = idx_common_1 + idx_different_1
    p2 = idx_common_2 + idx_different_2

    return [f1[i] for i in p1], [f2[i] for i in p2], calculate_permutation_parity(p1), calculate_permutation_parity(p2)
