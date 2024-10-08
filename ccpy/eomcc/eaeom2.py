import numpy as np
from ccpy.lib.core import cc_loops2

# R.a -> (nua) -> (a)
# R.aa -> (nua,nua,noa) -> (abj)
# R.ab -> (nua,nub,nob) -> (ab~j~)

def update(R, omega, H, RHF_symmetry, system):

    R.a, R.aa, R.ab = cc_loops2.update_r_2p1h(
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
    """Calculate the projection <a|[ (H_N e^(T1+T2))_C*(R1h+R2p1h) ]_C|0>."""
    X1A = np.einsum("ae,e->a", H.a.vv, R.a, optimize=True)
    X1A += 0.5 * np.einsum("anef,efn->a", H.aa.vovv, R.aa, optimize=True)
    X1A += np.einsum("anef,efn->a", H.ab.vovv, R.ab, optimize=True)
    X1A += np.einsum("me,aem->a", H.a.ov, R.aa, optimize=True)
    X1A += np.einsum("me,aem->a", H.b.ov, R.ab, optimize=True)
    return X1A

def build_HR_2A(R, T, H):
    """Calculate the projection <ajb|[ (H_N e^(T1+T2))_C*(R1h+R2p1h) ]_C|0>."""
    X2A = 0.5 * np.einsum("baje,e->abj", H.aa.vvov, R.a, optimize=True)
    X2A -= 0.5 * np.einsum("mj,abm->abj", H.a.oo, R.aa, optimize=True)
    X2A += 0.25 * np.einsum("abef,efj->abj", H.aa.vvvv, R.aa, optimize=True)
    I1 = (
        0.5 * np.einsum("mnef,efn->m", H.aa.oovv, R.aa, optimize=True)
        + np.einsum("mnef,efn->m", H.ab.oovv, R.ab, optimize=True)
    )
    X2A -= 0.5 * np.einsum("m,abmj->abj", I1, T.aa, optimize=True)
    X2A += np.einsum("ae,ebj->abj", H.a.vv, R.aa, optimize=True)
    X2A += np.einsum("bmje,aem->abj", H.aa.voov, R.aa, optimize=True)
    X2A += np.einsum("bmje,aem->abj", H.ab.voov, R.ab, optimize=True)
    X2A -= np.transpose(X2A, (1, 0, 2))
    return X2A

def build_HR_2B(R, T, H):
    """Calculate the projection <aj~b~|[ (H_N e^(T1+T2))_C*(R1h+R2p1h) ]_C|0>."""
    X2B = np.einsum("abej,e->abj", H.ab.vvvo, R.a, optimize=True)
    X2B += np.einsum("ae,ebj->abj", H.a.vv, R.ab, optimize=True)
    X2B += np.einsum("be,aej->abj", H.b.vv, R.ab, optimize=True)
    X2B -= np.einsum("mj,abm->abj", H.b.oo, R.ab, optimize=True)
    X2B += np.einsum("mbej,aem->abj", H.ab.ovvo, R.aa, optimize=True)
    X2B += np.einsum("bmje,aem->abj", H.bb.voov, R.ab, optimize=True)
    X2B -= np.einsum("amej,ebm->abj", H.ab.vovo, R.ab, optimize=True)
    X2B += np.einsum("abef,efj->abj", H.ab.vvvv, R.ab, optimize=True)
    I1 = (
        0.5 * np.einsum("mnef,efn->m", H.aa.oovv, R.aa, optimize=True)
        + np.einsum("mnef,efn->m", H.ab.oovv, R.ab, optimize=True)
    )
    X2B -= np.einsum("m,abmj->abj", I1, T.ab, optimize=True)
    return X2B

