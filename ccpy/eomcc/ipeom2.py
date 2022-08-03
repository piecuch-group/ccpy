import numpy as np
from ccpy.utilities.updates import cc_loops

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

    X2A = 0.0
    X2A -= np.einsum("bmji,m->bji", H.aa.vooo, R.a, optimize=True)
    X2A += np.einsum("be,eji->bji", H.a.vv, R.aa, optimize=True)
    X2A += 0.5 * np.einsum("mnij,bnm->bji", H.aa.oooo, R.aa, optimize=True)
    I1 = (
        -0.5 * np.einsum("mnef,fnm->e", H.aa.oovv, R.aa, optimize=True)
        - np.einsum("mnef,fnm->e", H.ab.oovv, R.ba, optimize=True)
    )
    X2A += np.einsum("e,ebij->bji", I1, T.aa, optimize=True)

    D_ij = 0.0
    D_ij -= np.einsum("mi,bmj->bji", H.a.oo, R.aa, optimize=True)
    D_ij += np.einsum("bmje,emi->bji", H.aa.voov, R.aa, optimize=True)
    D_ij += np.einsum("bmje,emi->bji", H.ab.voov, R.ba, optimize=True)
    D_ij -= np.transpose(D_ij, (0, 2, 1))

    X2A += D_ij

    return X2A


def build_HR_2B(R, T, H):
    """Calculate the projection <i~jb|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""

    X2B = 0.0
    X2B -= np.einsum("bmji,m->bij", H2B["vooo"], r1b, optimize=True)
    X2B -= np.einsum("mi,bmj->bij", H1B["oo"], r2b, optimize=True)
    X2B -= np.einsum("mj,bim->bij", H1A["oo"], r2b, optimize=True)
    X2B += np.einsum("be,eij->bij", H1A["vv"], r2b, optimize=True)
    X2B += np.einsum("nmji,bmn->bij", H2B["oooo"], r2b, optimize=True)
    X2B += np.einsum("bmje,eim->bij", H2A["voov"], r2b, optimize=True)
    X2B += np.einsum("bmje,eim->bij", H2B["voov"], r2d, optimize=True)
    X2B -= np.einsum("bmei,emj->bij", H2B["vovo"], r2b, optimize=True)
    I1 = -np.einsum("nmfe,fmn->e", vB["oovv"], r2b, optimize=True) - 0.5 * np.einsum(
        "mnef,fmn->e", vC["oovv"], r2d, optimize=True
    )
    X2B += np.einsum("e,beji->bij", I1, t2b, optimize=True)

    return X2B


def build_HR_2C(R, T, H):
    """Calculate the projection <ij~b~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""

    X2C = 0.0
    X2C -= np.einsum("mbij,m->bij", H2B["ovoo"], r1a, optimize=True)
    X2C -= np.einsum("mi,bmj->bij", H1A["oo"], r2c, optimize=True)
    X2C -= np.einsum("mj,bim->bij", H1B["oo"], r2c, optimize=True)
    X2C += np.einsum("be,eij->bij", H1B["vv"], r2c, optimize=True)
    X2C += np.einsum("mnij,bmn->bij", H2B["oooo"], r2c, optimize=True)
    X2C += np.einsum("mbej,eim->bij", H2B["ovvo"], r2a, optimize=True)
    X2C += np.einsum("bmje,eim->bij", H2C["voov"], r2c, optimize=True)
    X2C -= np.einsum("mbie,emj->bij", H2B["ovov"], r2c, optimize=True)
    I1 = -0.5 * np.einsum("mnef,fmn->e", vA["oovv"], r2a, optimize=True) - np.einsum(
        "mnef,fmn->e", vB["oovv"], r2c, optimize=True
    )
    X2C += np.einsum("e,ebij->bij", I1, t2b, optimize=True)

    return X2C


def build_HR_2D(R, T, H):
    """Calculate the projection <i~j~b~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""

    X2D = 0.0
    X2D -= np.einsum("bmji,m->bij", H2C["vooo"], r1b, optimize=True)
    X2D += np.einsum("be,eij->bij", H1B["vv"], r2d, optimize=True)
    X2D += 0.5 * np.einsum("mnij,bmn->bij", H2C["oooo"], r2d, optimize=True)
    I1 = -0.5 * np.einsum("mnef,fmn->e", vC["oovv"], r2d, optimize=True) - np.einsum(
        "nmfe,fmn->e", vB["oovv"], r2b, optimize=True
    )
    X2D += np.einsum("e,ebij->bij", I1, t2c, optimize=True)

    D_ij = 0.0
    D_ij -= np.einsum("mi,bmj->bij", H1B["oo"], r2d, optimize=True)
    D_ij += np.einsum("bmje,eim->bij", H2C["voov"], r2d, optimize=True)
    D_ij += np.einsum("mbej,eim->bij", H2B["ovvo"], r2b, optimize=True)
    D_ij -= np.transpose(D_ij, (0, 2, 1))

    X2D += D_ij

    return X2D


def guess_1h(ints, sys):
    """Build and diagonalize the Hamiltonian in the space of 1h excitations."""
    fA = ints["fA"]
    fB = ints["fB"]

    n1a = sys["Nocc_a"]
    n1b = sys["Nocc_b"]

    HAA = np.zeros((n1a, n1a))
    HAB = np.zeros((n1a, n1b))
    HBA = np.zeros((n1b, n1a))
    HBB = np.zeros((n1b, n1b))

    ct1 = 0
    for i in range(sys["Nocc_a"]):
        ct2 = 0
        for j in range(sys["Nocc_a"]):
            HAA[ct1, ct2] = fA["oo"][i, j]
            ct2 += 1
        ct1 += 1

    ct1 = 0
    for i in range(sys["Nocc_b"]):
        ct2 = 0
        for j in range(sys["Nocc_b"]):
            HBB[ct1, ct2] = fB["oo"][i, j]
            ct2 += 1
        ct1 += 1

    H = np.hstack((np.vstack((HAA, HBA)), np.vstack((HAB, HBB))))

    E_1h, C = np.linalg.eigh(H)
    idx = np.argsort(E_1h)
    E_1h = E_1h[idx]
    C = C[:, idx]

    return C, E_1h
