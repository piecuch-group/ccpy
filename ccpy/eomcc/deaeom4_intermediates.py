import numpy as np

from ccpy.models.integrals import Integral


def get_deaeom4_intermediates(H, R, T, system):


    # Create new 2-body integral object
    X = Integral.from_empty(system, 2, data_type=H.a.oo.dtype)

    return X
