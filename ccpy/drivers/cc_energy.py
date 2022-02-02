"""Module containing function to calculate the CC correlation energy."""
import numpy as np


def get_cc_energy(T, H0):
    """Calculate the CC correlation energy <0|(H_N e^T)_C|0>.

    Parameters
    ----------
    cc_t : dict
        Cluster amplitudes T1, T2
    ints : dict
        Sliced integrals F_N and V_N that define the bare Hamiltonian H_N

    Returns
    -------
    Ecorr : float
        CC correlation energy
    """
    e1a = np.einsum("me,em->", H0.a.ov, T.a, optimize=True)
    e1b = np.einsum("me,em->", H0.b.ov, T.b, optimize=True)
    e2aa = 0.25 * np.einsum("mnef,efmn->", H0.aa.oovv, T.aa, optimize=True)
    e2ab = np.einsum("mnef,efmn->", H0.ab.oovv, T.ab, optimize=True)
    e2bb = 0.25 * np.einsum("mnef,efmn->", H0.bb.oovv, T.bb, optimize=True)
    e1a1a = 0.5 * np.einsum("mnef,fn,em->", H0.aa.oovv, T.a, T.a, optimize=True)
    e1b1b = 0.5 * np.einsum("mnef,fn,em->", H0.bb.oovv, T.b, T.b, optimize=True)
    e1a1b = np.einsum("mnef,em,fn->", H0.ab.oovv, T.a, T.b, optimize=True)

    Ecorr = e1a + e1b + e2aa + e2ab + e2bb + e1a1a + e1a1b + e1b1b
    return Ecorr
