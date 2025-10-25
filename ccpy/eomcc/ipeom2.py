"""Ionization Potential Equation-of-Motion Coupled-Cluster
Method with 1h and 2h-1p Excitations on top of CCSD [IP-EOMCCSD(2h-1p)]"""
import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
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
    X1A -= ccpy_einsum("mi,m->i", H.a.oo, R.a)
    X1A -= 0.5 * ccpy_einsum("mnif,mfn->i", H.aa.ooov, R.aa)
    X1A -= ccpy_einsum("mnif,mfn->i", H.ab.ooov, R.ab)
    X1A += ccpy_einsum("me,iem->i", H.a.ov, R.aa)
    X1A += ccpy_einsum("me,iem->i", H.b.ov, R.ab)
    return X1A

def build_HR_2A(R, T, H):
    """Calculate the projection <ijb|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    X2A = -0.5 * ccpy_einsum("bmji,m->ibj", H.aa.vooo, R.a)
    X2A += 0.5 * ccpy_einsum("be,iej->ibj", H.a.vv, R.aa)
    X2A += 0.25 * ccpy_einsum("mnij,mbn->ibj", H.aa.oooo, R.aa)
    I1 = (
        -0.5 * ccpy_einsum("mnef,mfn->e", H.aa.oovv, R.aa)
        - ccpy_einsum("mnef,mfn->e", H.ab.oovv, R.ab)
    )
    X2A += 0.5 * ccpy_einsum("e,ebij->ibj", I1, T.aa)
    X2A -= ccpy_einsum("mi,mbj->ibj", H.a.oo, R.aa)
    X2A += ccpy_einsum("bmje,iem->ibj", H.aa.voov, R.aa)
    X2A += ccpy_einsum("bmje,iem->ibj", H.ab.voov, R.ab)
    X2A -= np.transpose(X2A, (2, 1, 0))
    return X2A

def build_HR_2B(R, T, H):
    """Calculate the projection <ij~b~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    X2B = -1.0 * ccpy_einsum("mbij,m->ibj", H.ab.ovoo, R.a)
    X2B -= ccpy_einsum("mi,mbj->ibj", H.a.oo, R.ab)
    X2B -= ccpy_einsum("mj,ibm->ibj", H.b.oo, R.ab)
    X2B += ccpy_einsum("be,iej->ibj", H.b.vv, R.ab)
    X2B += ccpy_einsum("mnij,mbn->ibj", H.ab.oooo, R.ab)
    X2B += ccpy_einsum("mbej,iem->ibj", H.ab.ovvo, R.aa)
    X2B += ccpy_einsum("bmje,iem->ibj", H.bb.voov, R.ab)
    X2B -= ccpy_einsum("mbie,mej->ibj", H.ab.ovov, R.ab)
    I1 = (
        -0.5 * ccpy_einsum("mnef,mfn->e", H.aa.oovv, R.aa)
        - ccpy_einsum("mnef,mfn->e", H.ab.oovv, R.ab)
    )
    X2B += ccpy_einsum("e,ebij->ibj", I1, T.ab)
    return X2B

