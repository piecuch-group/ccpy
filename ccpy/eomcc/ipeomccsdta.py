import numpy as np
from ccpy.eomcc.ipeom3_intermediates import get_ipeom3_intermediates
from ccpy.lib.core import cc_loops2

# R.a -> (noa) -> (i)
# R.aa -> (noa,nua,noa) -> (ibj)
# R.ab -> (noa,nub,nob) -> (ib~j~)
# R.aaa -> (noa,nua,nua,noa,noa) -> (ibcjk)
# R.aab -> (noa,nua,nub,noa,nob) -> (ibc~jk~)
# R.abb -> (noa,nub,nub,nob,nob) -> (ib~c~j~k~)

def update(R, omega, H, RHF_symmetry, system):

    R.a, R.aa, R.ab, R.aaa, R.aab, R.abb = cc_loops2.update_r_3h2p(
        R.a,
        R.aa,
        R.ab,
        R.aaa,
        R.aab,
        R.abb,
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
        0.0,
    )
    return R

def HR(dR, R, T, H, flag_RHF, system):
    # Get intermediates
    X = get_ipeom3_intermediates(H, R)
    # update R1
    dR.a = build_HR_1A(R, T, H)
    # update R2
    dR.aa = build_HR_2A(R, T, H)
    dR.ab = build_HR_2B(R, T, H)
    # update R3
    dR.aaa = build_HR_3A(R, T, X, H)
    dR.aab = build_HR_3B(R, T, X, H)
    dR.abb = build_HR_3C(R, T, X, H)
    return dR.flatten()

def build_HR_1A(R, T, H):
    """Calculate the projection <i|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>."""
    X1A = 0.0
    X1A -= np.einsum("mi,m->i", H.a.oo, R.a, optimize=True)
    X1A -= 0.5 * np.einsum("mnif,mfn->i", H.aa.ooov, R.aa, optimize=True)
    X1A -= np.einsum("mnif,mfn->i", H.ab.ooov, R.ab, optimize=True)
    X1A += np.einsum("me,iem->i", H.a.ov, R.aa, optimize=True)
    X1A += np.einsum("me,iem->i", H.b.ov, R.ab, optimize=True)
    # additional terms with R3
    X1A += 0.25 * np.einsum("mnef,iefmn->i", H.aa.oovv, R.aaa, optimize=True)
    X1A += np.einsum("mnef,iefmn->i", H.ab.oovv, R.aab, optimize=True)
    X1A += 0.25 * np.einsum("mnef,iefmn->i", H.bb.oovv, R.abb, optimize=True)
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
    # additional terms with R3
    X2A += 0.5 * np.einsum("me,ibejm->ibj", H.a.ov, R.aaa, optimize=True)
    X2A += 0.5 * np.einsum("me,ibejm->ibj", H.b.ov, R.aab, optimize=True)
    X2A += 0.25 * np.einsum("bnef,iefjn->ibj", H.aa.vovv, R.aaa, optimize=True)
    X2A += 0.5 * np.einsum("bnef,iefjn->ibj", H.ab.vovv, R.aab, optimize=True)
    X2A -= 0.5 * np.einsum("mnjf,ibfmn->ibj", H.aa.ooov, R.aaa, optimize=True)
    X2A -= np.einsum("mnjf,ibfmn->ibj", H.ab.ooov, R.aab, optimize=True)
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
    # additional terms with R3
    X2B += np.einsum("me,iebmj->ibj", H.a.ov, R.aab, optimize=True)
    X2B += np.einsum("me,ibejm->ibj", H.b.ov, R.abb, optimize=True)
    X2B += np.einsum("nbfe,ifenj->ibj", H.ab.ovvv, R.aab, optimize=True)
    X2B += 0.5 * np.einsum("bnef,iefjn->ibj", H.bb.vovv, R.abb, optimize=True)
    X2B -= 0.5 * np.einsum("mnif,mfbnj->ibj", H.aa.ooov, R.aab, optimize=True)
    X2B -= np.einsum("mnif,mfbnj->ibj", H.ab.ooov, R.abb, optimize=True)
    X2B -= np.einsum("nmfj,ifbnm->ibj", H.ab.oovo, R.aab, optimize=True)
    X2B -= 0.5 * np.einsum("mnjf,ifbnm->ibj", H.bb.ooov, R.abb, optimize=True)
    return X2B

def build_HR_3A(R, T, X, H):
    """Calculate the projection <ijkbc|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>."""
    X3A = -(3.0 / 12.0) * np.einsum("mj,ibcmk->ibcjk", H.a.oo, R.aaa, optimize=True)
    X3A += (2.0 / 12.0) * np.einsum("be,iecjk->ibcjk", H.a.vv, R.aaa, optimize=True)
    X3A += (3.0 / 24.0) * np.einsum("mnjk,ibcmn->ibcjk", H.aa.oooo, R.aaa, optimize=True)
    X3A += (1.0 / 24.0) * np.einsum("bcef,iefjk->ibcjk", H.aa.vvvv, R.aaa, optimize=True)
    X3A += (6.0 / 12.0) * np.einsum("bmje,iecmk->ibcjk", H.aa.voov, R.aaa, optimize=True)
    X3A += (6.0 / 12.0) * np.einsum("bmje,icekm->ibcjk", H.ab.voov, R.aab, optimize=True)
    # moment-like terms
    X3A -= (6.0 / 12.0) * np.einsum("cmkj,ibm->ibcjk", H.aa.vooo, R.aa, optimize=True)
    X3A += (3.0 / 12.0) * np.einsum("cbke,iej->ibcjk", H.aa.vvov, R.aa, optimize=True)
    # 3-body Hbar terms factorized using intermediates
    X3A -= (3.0 / 12.0) * np.einsum("imj,bcmk->ibcjk", X["aa"]["ooo"], T.aa, optimize=True)
    X3A += (6.0 / 12.0) * np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovv"], T.aa, optimize=True)
    X3A -= np.transpose(X3A, (3, 1, 2, 0, 4)) + np.transpose(X3A, (4, 1, 2, 3, 0)) # antisymmetrize A(i/jk)
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    return X3A

def build_HR_3B(R, T, X, H):
    """Calculate the projection <ijk~bc~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>."""
    X3B = -np.einsum("mj,ibcmk->ibcjk", H.a.oo, R.aab, optimize=True) # (1)
    X3B -= 0.5 * np.einsum("mk,ibcjm->ibcjk", H.b.oo, R.aab, optimize=True) # (2)
    X3B += 0.5 * np.einsum("be,iecjk->ibcjk", H.a.vv, R.aab, optimize=True) # (3)
    X3B += 0.5 * np.einsum("ce,ibejk->ibcjk", H.b.vv, R.aab, optimize=True) # (4)
    X3B += np.einsum("mnjk,ibcmn->ibcjk", H.ab.oooo, R.aab, optimize=True) # (5)
    X3B += 0.25 * np.einsum("mnij,mbcnk->ibcjk", H.aa.oooo, R.aab, optimize=True) # (6)
    X3B += 0.5 * np.einsum("bcef,iefjk->ibcjk", H.ab.vvvv, R.aab, optimize=True) # (7)
    X3B += 0.5 * np.einsum("mcek,ibejm->ibcjk", H.ab.ovvo, R.aaa, optimize=True) # (8)
    X3B += 0.5 * np.einsum("cmke,ibejm->ibcjk", H.bb.voov, R.aab, optimize=True) # (9)
    X3B += np.einsum("bmje,iecmk->ibcjk", H.aa.voov, R.aab, optimize=True) # (10)
    X3B += np.einsum("bmje,iecmk->ibcjk", H.ab.voov, R.abb, optimize=True) # (11)
    X3B -= np.einsum("mcje,ibemk->ibcjk", H.ab.ovov, R.aab, optimize=True) # (12)
    X3B -= 0.5 * np.einsum("bmek,iecjm->ibcjk", H.ab.vovo, R.aab, optimize=True) # (13)
    # moment-like terms
    X3B -= np.einsum("mcjk,ibm->ibcjk", H.ab.ovoo, R.aa, optimize=True) # (14)
    X3B -= 0.5 * np.einsum("bmji,mck->ibcjk", H.aa.vooo, R.ab, optimize=True) # (15)
    X3B -= np.einsum("bmjk,icm->ibcjk", H.ab.vooo, R.ab, optimize=True) # (16)
    X3B += np.einsum("bcje,iek->ibcjk", H.ab.vvov, R.ab, optimize=True) # (17)
    X3B += 0.5 * np.einsum("bcek,iej->ibcjk", H.ab.vvvo, R.aa, optimize=True) # (18)
    # 3-body Hbar terms factorized using intermediates
    X3B += 0.5 * np.einsum("eck,ebij->ibcjk", X["ab"]["vvo"], T.aa, optimize=True) # (19)
    X3B -= 0.5 * np.einsum("imj,bcmk->ibcjk", X["aa"]["ooo"], T.ab, optimize=True) # (20)
    X3B -= np.einsum("imk,bcjm->ibcjk", X["ab"]["ooo"], T.ab, optimize=True) # (21)
    X3B += np.einsum("ice,bejk->ibcjk", X["ab"]["ovv"], T.ab, optimize=True) # (22)
    X3B += np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovv"], T.ab, optimize=True) # (23)
    X3B -= np.transpose(X3B, (3, 1, 2, 0, 4)) # antisymmetrize (ij)
    return X3B

def build_HR_3C(R, T, X, H):
    """Calculate the projection <ij~k~b~c~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>."""
    X3C = -(2.0 / 4.0) * np.einsum("mj,ibcmk->ibcjk", H.b.oo, R.abb, optimize=True) # (1)
    X3C -= (1.0 / 4.0) * np.einsum("mi,mbcjk->ibcjk", H.a.oo, R.abb, optimize=True) # (2)
    X3C += (2.0 / 4.0) * np.einsum("be,iecjk->ibcjk", H.b.vv, R.abb, optimize=True) # (3)
    X3C += (1.0 / 8.0) * np.einsum("mnjk,ibcmn->ibcjk", H.bb.oooo, R.abb, optimize=True) # (4)
    X3C += (2.0 / 4.0) * np.einsum("mnij,mbcnk->ibcjk", H.ab.oooo, R.abb, optimize=True) # (5)
    X3C += (1.0 / 8.0) * np.einsum("bcef,iefjk->ibcjk", H.bb.vvvv, R.abb, optimize=True) # (6)
    X3C += np.einsum("mbej,iecmk->ibcjk", H.ab.ovvo, R.aab, optimize=True) # (7)
    X3C += np.einsum("bmje,iecmk->ibcjk", H.bb.voov, R.abb, optimize=True) # (8)
    X3C -= (2.0 / 4.0) * np.einsum("mbie,mecjk->ibcjk", H.ab.ovov, R.abb, optimize=True) # (9)
    # moment-like terms
    X3C -= np.einsum("mcik,mbj->ibcjk", H.ab.ovoo, R.ab, optimize=True) # (10)
    X3C -= (2.0 / 4.0) * np.einsum("cmkj,ibm->ibcjk", H.bb.vooo, R.ab, optimize=True) # (11)
    X3C += (2.0 / 4.0) * np.einsum("cbke,iej->ibcjk", H.bb.vvov, R.ab, optimize=True) # (12)
    # 3-body Hbar terms factorized using intermediates
    X3C -= (2.0 / 4.0) * np.einsum("imj,bcmk->ibcjk", X["ab"]["ooo"], T.bb, optimize=True) # (13)
    X3C += (2.0 / 4.0) * np.einsum("ibe,ecjk->ibcjk", X["ab"]["ovv"], T.bb, optimize=True) # (14)
    X3C += np.einsum("ebj,ecik->ibcjk", X["ab"]["vvo"], T.ab, optimize=True) # (15)
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3C -= np.transpose(X3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3C
