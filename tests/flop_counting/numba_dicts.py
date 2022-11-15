import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict


float_array = types.float64[:]
@njit
def test():

    result_dict = Dict.empty(
                key_type=types.unicode_type,
                value_type=float_array,
    )

    value = np.array([1, 2, 3])
    for i, key in enumerate(['dog', 'cat', 'bird']):
        result_dict[key] = value[i]
        print(key,"->",result_dict[key])

if __name__ == "__main__":
    test()
