import numpy as np


def contract(contraction_string, *args, **kwargs):
    kwargs["optimize"] = True
    return np.einsum(contraction_string, *args, **kwargs)
