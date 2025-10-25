import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum


def contract_vt3_singles(H, T_ext, system):

    x1a = np.zeros((system.nunoccupied_alpha, system.noccupied_alpha))
    x1b = np.zeros((system.nunoccupied_beta, system.noccupied_beta))

    x1a = 0.25 * ccpy_einsum("mnef,aefimn->ai", H.aa.oovv, T_ext.aaa)
    x1a += ccpy_einsum("mnef,aefimn->ai", H.ab.oovv, T_ext.aab)
    x1a += 0.25 * ccpy_einsum("mnef,aefimn->ai", H.bb.oovv, T_ext.abb)

    x1b = 0.25 * ccpy_einsum("mnef,aefimn->ai", H.bb.oovv, T_ext.bbb)
    x1b += 0.25 * ccpy_einsum("mnef,efamni->ai", H.aa.oovv, T_ext.aab)
    x1b += ccpy_einsum("mnef,efamni->ai", H.ab.oovv, T_ext.abb)

    return x1a, x1b


