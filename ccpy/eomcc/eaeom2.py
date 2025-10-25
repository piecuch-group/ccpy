'''
Electron Attachment Equation-of-Motion Coupled-Cluster
Method with 1p and 2p-1h Excitations on top of CCSD [EA-EOMCCSD(2p-1h)]
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import cc_loops2

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
    X1A = ccpy_einsum("ae,e->a", H.a.vv, R.a)
    X1A += 0.5 * ccpy_einsum("anef,efn->a", H.aa.vovv, R.aa)
    X1A += ccpy_einsum("anef,efn->a", H.ab.vovv, R.ab)
    X1A += ccpy_einsum("me,aem->a", H.a.ov, R.aa)
    X1A += ccpy_einsum("me,aem->a", H.b.ov, R.ab)
    return X1A

def build_HR_2A(R, T, H):
    """Calculate the projection <ajb|[ (H_N e^(T1+T2))_C*(R1h+R2p1h) ]_C|0>."""
    X2A = 0.5 * ccpy_einsum("baje,e->abj", H.aa.vvov, R.a)
    X2A -= 0.5 * ccpy_einsum("mj,abm->abj", H.a.oo, R.aa)
    X2A += 0.25 * ccpy_einsum("abef,efj->abj", H.aa.vvvv, R.aa)
    I1 = (
        0.5 * ccpy_einsum("mnef,efn->m", H.aa.oovv, R.aa)
        + ccpy_einsum("mnef,efn->m", H.ab.oovv, R.ab)
    )
    X2A -= 0.5 * ccpy_einsum("m,abmj->abj", I1, T.aa)
    X2A += ccpy_einsum("ae,ebj->abj", H.a.vv, R.aa)
    X2A += ccpy_einsum("bmje,aem->abj", H.aa.voov, R.aa)
    X2A += ccpy_einsum("bmje,aem->abj", H.ab.voov, R.ab)
    X2A -= np.transpose(X2A, (1, 0, 2))
    return X2A

def build_HR_2B(R, T, H):
    """Calculate the projection <aj~b~|[ (H_N e^(T1+T2))_C*(R1h+R2p1h) ]_C|0>."""
    X2B = ccpy_einsum("abej,e->abj", H.ab.vvvo, R.a)
    X2B += ccpy_einsum("ae,ebj->abj", H.a.vv, R.ab)
    X2B += ccpy_einsum("be,aej->abj", H.b.vv, R.ab)
    X2B -= ccpy_einsum("mj,abm->abj", H.b.oo, R.ab)
    X2B += ccpy_einsum("mbej,aem->abj", H.ab.ovvo, R.aa)
    X2B += ccpy_einsum("bmje,aem->abj", H.bb.voov, R.ab)
    X2B -= ccpy_einsum("amej,ebm->abj", H.ab.vovo, R.ab)
    X2B += ccpy_einsum("abef,efj->abj", H.ab.vvvv, R.ab)
    I1 = (
        0.5 * ccpy_einsum("mnef,efn->m", H.aa.oovv, R.aa)
        + ccpy_einsum("mnef,efn->m", H.ab.oovv, R.ab)
    )
    X2B -= ccpy_einsum("m,abmj->abj", I1, T.ab)
    return X2B

