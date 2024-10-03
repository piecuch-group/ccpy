import numpy as np

def get_lrcc_energy(T1, W, T, H):
    """Calculate the CC correlation energy <0|(H_N e^T)_C|0>."""
    w1a = np.einsum("me,em->", W.a.ov, T.a, optimize=True)
    w1b = np.einsum("me,em->", W.b.ov, T.b, optimize=True)
    e1a = np.einsum("me,em->", H.a.ov, T1.a, optimize=True)
    e1b = np.einsum("me,em->", H.b.ov, T1.b, optimize=True)
    e2aa = 0.25 * np.einsum("mnef,efmn->", H.aa.oovv, T1.aa, optimize=True)
    e2ab = np.einsum("mnef,efmn->", H.ab.oovv, T1.ab, optimize=True)
    e2bb = 0.25 * np.einsum("mnef,efmn->", H.bb.oovv, T1.bb, optimize=True)
    Ecorr = w1a + w1b + e1a + e1b + e2aa + e2ab + e2bb
    return Ecorr