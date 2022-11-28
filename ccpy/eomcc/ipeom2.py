import numpy as np
from ccpy.utilities.updates import cc_loops

from ccpy.models.operators import FockOperator

# R.a -> R.a (m)
# R.b -> R.b (m)
# R.aa -> R.aa (fnm)
# R.ab -> R.ab (fnm~)
# R.ba -> R.ba (f~n~m)
# R.bb -> R.bb (f~n~m~)

def update(R, omega, H, system):

    R.a, R.b, R.aa, R.ab, R.ba, R.bb = cc_loops.cc_loops.update_r_2h1p(
        R.a,
        R.b,
        R.aa,
        R.ab,
        R.ba,
        R.bb,
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
    if flag_RHF:
        dR.b = dR.a.copy()
    else:
        dR.b = build_HR_1B(R, T, H)

    # update R2
    dR.aa = build_HR_2A(R, T, H)
    dR.ab = build_HR_2B(R, T, H)
    if flag_RHF:
        dR.ba = dR.ab.copy()
        dR.bb = dR.aa.copy()
    else:
        dR.ba = build_HR_2C(R, T, H)
        dR.bb = build_HR_2D(R, T, H)

    return dR.flatten()


def build_HR_1A(R, T, H):
    """Calculate the projection <i|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    X1A = 0.0
    X1A -= np.einsum("mi,m->i", H.a.oo, R.a, optimize=True)
    X1A -= 0.5 * np.einsum("mnif,fnm->i", H.aa.ooov, R.aa, optimize=True)
    X1A -= np.einsum("mnif,fnm->i", H.ab.ooov, R.ba, optimize=True)
    X1A += np.einsum("me,emi->i", H.a.ov, R.aa, optimize=True)
    X1A += np.einsum("me,emi->i", H.b.ov, R.ba, optimize=True)

    return X1A


def build_HR_1B(R, T, H):
    """Calculate the projection <i~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    X1B = 0.0
    X1B -= np.einsum("mi,m->i", H.b.oo, R.b, optimize=True)
    X1B -= np.einsum("nmfi,fnm->i", H.ab.oovo, R.ab, optimize=True)
    X1B -= 0.5 * np.einsum("mnif,fnm->i", H.bb.ooov, R.bb, optimize=True)
    X1B += np.einsum("me,emi->i", H.a.ov, R.ab, optimize=True)
    X1B += np.einsum("me,emi->i", H.b.ov, R.bb, optimize=True)

    return X1B


def build_HR_2A(R, T, H):
    """Calculate the projection <ijb|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""

    X2A = -1.0 * np.einsum("bmji,m->bji", H.aa.vooo, R.a, optimize=True)
    X2A += np.einsum("be,eji->bji", H.a.vv, R.aa, optimize=True)
    X2A += 0.5 * np.einsum("mnij,bnm->bji", H.aa.oooo, R.aa, optimize=True)
    I1 = (
        -0.5 * np.einsum("mnef,fnm->e", H.aa.oovv, R.aa, optimize=True)
        - np.einsum("mnef,fnm->e", H.ab.oovv, R.ba, optimize=True)
    )
    X2A += np.einsum("e,ebij->bji", I1, T.aa, optimize=True)

    D_ij = 0.0
    D_ij -= np.einsum("mi,bjm->bji", H.a.oo, R.aa, optimize=True)
    D_ij += np.einsum("bmje,emi->bji", H.aa.voov, R.aa, optimize=True)
    D_ij += np.einsum("bmje,emi->bji", H.ab.voov, R.ba, optimize=True)
    D_ij -= np.transpose(D_ij, (0, 2, 1))

    X2A += D_ij

    return X2A


def build_HR_2B(R, T, H):
    """Calculate the projection <i~jb|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""

    X2B = -1.0 * np.einsum("bmji,m->bji", H.ab.vooo, R.b, optimize=True)
    X2B -= np.einsum("mi,bjm->bji", H.b.oo, R.ab, optimize=True)
    X2B -= np.einsum("mj,bmi->bji", H.a.oo, R.ab, optimize=True)
    X2B += np.einsum("be,eji->bji", H.a.vv, R.ab, optimize=True)
    X2B += np.einsum("nmji,bnm->bji", H.ab.oooo, R.ab, optimize=True)
    X2B += np.einsum("bmje,emi->bji", H.aa.voov, R.ab, optimize=True)
    X2B += np.einsum("bmje,emi->bji", H.ab.voov, R.bb, optimize=True)
    X2B -= np.einsum("bmei,ejm->bji", H.ab.vovo, R.ab, optimize=True)
    I1 = (
        -np.einsum("nmfe,fnm->e", H.ab.oovv, R.ab, optimize=True) 
        - 0.5 * np.einsum("mnef,fnm->e", H.bb.oovv, R.bb, optimize=True)
    )
    X2B += np.einsum("e,beji->bji", I1, T.ab, optimize=True)

    return X2B


def build_HR_2C(R, T, H):
    """Calculate the projection <ij~b~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""

    X2C = -1.0 * np.einsum("mbij,m->bji", H.ab.ovoo, R.a, optimize=True)
    X2C -= np.einsum("mi,bjm->bji", H.a.oo, R.ba, optimize=True)
    X2C -= np.einsum("mj,bmi->bji", H.b.oo, R.ba, optimize=True)
    X2C += np.einsum("be,eji->bji", H.b.vv, R.ba, optimize=True)
    X2C += np.einsum("mnij,bnm->bji", H.ab.oooo, R.ba, optimize=True)
    X2C += np.einsum("mbej,emi->bji", H.ab.ovvo, R.aa, optimize=True)
    X2C += np.einsum("bmje,emi->bji", H.bb.voov, R.ba, optimize=True)
    X2C -= np.einsum("mbie,ejm->bji", H.ab.ovov, R.ba, optimize=True)
    I1 = (
        -0.5 * np.einsum("mnef,fnm->e", H.aa.oovv, R.aa, optimize=True)
        - np.einsum("mnef,fnm->e", H.ab.oovv, R.ba, optimize=True)
    )
    X2C += np.einsum("e,ebij->bji", I1, T.ab, optimize=True)

    return X2C


def build_HR_2D(R, T, H):
    """Calculate the projection <i~j~b~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""

    X2D = -1.0 * np.einsum("bmji,m->bji", H.bb.vooo, R.b, optimize=True)
    X2D += np.einsum("be,eji->bji", H.b.vv, R.bb, optimize=True)
    X2D += 0.5 * np.einsum("mnij,bnm->bji", H.bb.oooo, R.bb, optimize=True)
    I1 = (
        -0.5 * np.einsum("mnef,fnm->e", H.bb.oovv, R.bb, optimize=True)
        - np.einsum("nmfe,fnm->e", H.ab.oovv, R.ab, optimize=True)
    )
    X2D += np.einsum("e,ebij->bji", I1, T.bb, optimize=True)

    D_ij = 0.0
    D_ij -= np.einsum("mi,bjm->bji", H.b.oo, R.bb, optimize=True)
    D_ij += np.einsum("bmje,emi->bji", H.bb.voov, R.bb, optimize=True)
    D_ij += np.einsum("mbej,emi->bji", H.ab.ovvo, R.ab, optimize=True)
    D_ij -= np.transpose(D_ij, (0, 2, 1))

    X2D += D_ij

    return X2D
