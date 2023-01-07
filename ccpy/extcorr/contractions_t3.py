import numpy as np


def contract_vt3_singles(H, T_ext, system):

    x1a = np.zeros((system.nunoccupied_alpha, system.noccupied_alpha))
    x1b = np.zeros((system.nunoccupied_beta, system.noccupied_beta))

    x1a = 0.25 * np.einsum("mnef,aefimn->ai", H.aa.oovv, T_ext.aaa, optimize=True)
    x1a += np.einsum("mnef,aefimn->ai", H.ab.oovv, T_ext.aab, optimize=True)
    x1a += 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, T_ext.abb, optimize=True)

    x1b = 0.25 * np.einsum("mnef,aefimn->ai", H.bb.oovv, T_ext.bbb, optimize=True)
    x1b += 0.25 * np.einsum("mnef,efamni->ai", H.aa.oovv, T_ext.aab, optimize=True)
    x1b += np.einsum("mnef,efamni->ai", H.ab.oovv, T_ext.abb, optimize=True)

    return x1a, x1b


