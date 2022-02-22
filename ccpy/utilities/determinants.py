
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
