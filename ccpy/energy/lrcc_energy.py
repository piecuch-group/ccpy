import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

def get_lrcc_energy(T1, W, T, H):
    """Calculate the CC correlation energy <0|(H_N e^T)_C|0>."""
    w1a = ccpy_einsum("me,em->", W.a.ov, T.a)
    w1b = ccpy_einsum("me,em->", W.b.ov, T.b)
    e1a = ccpy_einsum("me,em->", H.a.ov, T1.a)
    e1b = ccpy_einsum("me,em->", H.b.ov, T1.b)
    e2aa = 0.25 * ccpy_einsum("mnef,efmn->", H.aa.oovv, T1.aa)
    e2ab = ccpy_einsum("mnef,efmn->", H.ab.oovv, T1.ab)
    e2bb = 0.25 * ccpy_einsum("mnef,efmn->", H.bb.oovv, T1.bb)
    Ecorr = w1a + w1b + e1a + e1b + e2aa + e2ab + e2bb
    return Ecorr