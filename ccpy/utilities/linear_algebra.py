import numpy as np


def ccpy_einsum(contraction_string, *args, **kwargs):
    optimize = kwargs.pop("optimize", True)
    return np.einsum(contraction_string, *args, **kwargs, optimize=optimize)
