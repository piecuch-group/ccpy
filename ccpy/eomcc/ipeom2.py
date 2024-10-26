import numpy as np
from ccpy.lib.core import cc_loops2

# R.a -> (noa) -> (i)
# R.aa -> (noa,nua,noa) -> (ibj)
# R.ab -> (noa,nub,nob) -> (ib~j~)

def update(R, omega, H, RHF_symmetry, system):

    R.a, R.aa, R.ab = cc_loops2.update_r_2h1p(
        R.a,
        R.aa,
        R.ab,
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
        0.0,
    )
    return R

def HR(dR, R, T, H, flag_RHF, system):
    # update R1
    dR.a = build_HR_1A(R, T, H)
    # update R2
    dR.aa = build_HR_2A(R, T, H)
    dR.ab = build_HR_2B(R, T, H)
    return dR.flatten()

def build_HR_1A(R, T, H):
    """Calculate the projection <i|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    X1A = 0.0
    X1A -= np.einsum("mi,m->i", H.a.oo, R.a, optimize=True)
    X1A -= 0.5 * np.einsum("mnif,mfn->i", H.aa.ooov, R.aa, optimize=True)
    X1A -= np.einsum("mnif,mfn->i", H.ab.ooov, R.ab, optimize=True)
    X1A += np.einsum("me,iem->i", H.a.ov, R.aa, optimize=True)
    X1A += np.einsum("me,iem->i", H.b.ov, R.ab, optimize=True)
    return X1A

def build_HR_2A(R, T, H):
    """Calculate the projection <ijb|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    X2A = -0.5 * np.einsum("bmji,m->ibj", H.aa.vooo, R.a, optimize=True)
    X2A += 0.5 * np.einsum("be,iej->ibj", H.a.vv, R.aa, optimize=True)
    X2A += 0.25 * np.einsum("mnij,mbn->ibj", H.aa.oooo, R.aa, optimize=True)
    I1 = (
        -0.5 * np.einsum("mnef,mfn->e", H.aa.oovv, R.aa, optimize=True)
        - np.einsum("mnef,mfn->e", H.ab.oovv, R.ab, optimize=True)
    )
    X2A += 0.5 * np.einsum("e,ebij->ibj", I1, T.aa, optimize=True)
    X2A -= np.einsum("mi,mbj->ibj", H.a.oo, R.aa, optimize=True)
    X2A += np.einsum("bmje,iem->ibj", H.aa.voov, R.aa, optimize=True)
    X2A += np.einsum("bmje,iem->ibj", H.ab.voov, R.ab, optimize=True)
    X2A -= np.transpose(X2A, (2, 1, 0))
    return X2A

def build_HR_2B(R, T, H):
    """Calculate the projection <ij~b~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    X2B = -1.0 * np.einsum("mbij,m->ibj", H.ab.ovoo, R.a, optimize=True)
    X2B -= np.einsum("mi,mbj->ibj", H.a.oo, R.ab, optimize=True)
    X2B -= np.einsum("mj,ibm->ibj", H.b.oo, R.ab, optimize=True)
    X2B += np.einsum("be,iej->ibj", H.b.vv, R.ab, optimize=True)
    X2B += np.einsum("mnij,mbn->ibj", H.ab.oooo, R.ab, optimize=True)
    X2B += np.einsum("mbej,iem->ibj", H.ab.ovvo, R.aa, optimize=True)
    X2B += np.einsum("bmje,iem->ibj", H.bb.voov, R.ab, optimize=True)
    X2B -= np.einsum("mbie,mej->ibj", H.ab.ovov, R.ab, optimize=True)
    I1 = (
        -0.5 * np.einsum("mnef,mfn->e", H.aa.oovv, R.aa, optimize=True)
        - np.einsum("mnef,mfn->e", H.ab.oovv, R.ab, optimize=True)
    )
    X2B += np.einsum("e,ebij->ibj", I1, T.ab, optimize=True)
    return X2B

