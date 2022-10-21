
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

def get_excit_rank(D, D0):
    return len(set(D) - set(D0))


def get_excits_from(D, D0):
    return list(set(D0) - set(D))


def get_excits_to(D, D0):
    return list(set(D) - set(D0))


def spatial_orb_idx(x):
    if x % 2 == 1:
        return int((x + 1) / 2)
    else:
        return int(x / 2)


def get_spincase(excits_from, excits_to):
    assert (len(excits_from) == len(excits_to))

    num_alpha_occ = sum([i % 2 for i in excits_from])
    num_alpha_unocc = sum([i % 2 for i in excits_to])

    assert (num_alpha_occ == num_alpha_unocc)

    spincase = 'a' * num_alpha_occ + 'b' * (len(excits_from) - num_alpha_occ)

    return spincase

def calculate_excitation_difference(f1, f2, spintype1, spintype2):

    if spintype1 == 'aaa':
        if spintype2 == 'aaa':
            return get_number_difference(f1[:3], f2[:3]) + get_number_difference(f1[3:], f2[3:])
        if spintype2 == 'aab':
            return get_number_difference(f1[:3], f2[:2]) + get_number_difference(f1[3:], f2[3:5])
        if spintype2 == 'abb':
            return 4
        if spintype2 == 'bbb':
            return 6

    if spintype1 == 'aab':
        if spintype2 == 'aaa':
            return get_number_difference(f1[:2], f2[:3]) + get_number_difference(f1[3:5], f2[3:])
        if spintype2 == 'aab':
            return get_number_difference(f1[:2], f2[:2]) + get_number_difference(f1[3], f2[3])\
                  +get_number_difference(f1[3:5], f2[3:5]) + get_number_difference(f1[5], f2[5])
        if spintype2 == 'abb':
            return get_number_difference(f1[:2], f2[3]) + get_number_difference(f1[3], f2[1:3])\
                  +get_number_difference(f1[3:5], f2[3]) + get_number_difference(f1[5], f2[4:])
        if spintype2 == 'bbb':
            return get_number_difference(f1[3], f2[:3]) + get_number_difference(f1[5], f2[3:])

def get_number_difference(f1, f2):
    return len( set(f1) ^ set(f2) )
