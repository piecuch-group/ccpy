import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.eomcc.ipeom3_intermediates import get_ipeomccsdta_intermediates
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
    X = get_ipeomccsdta_intermediates(H, R, T)
    # update R1
    dR.a = build_HR_1A(R, T, H)
    # update R2
    dR.aa = build_HR_2A(R, T, H)
    dR.ab = build_HR_2B(R, T, H)
    # update R3
    # X = add_v_term(X, H, R) # this contribution is 0 for canonical orbitals, since F_N = 0
    dR.aaa = build_HR_3A(R, T, X, H)
    dR.aab = build_HR_3B(R, T, X, H)
    dR.abb = build_HR_3C(R, T, X, H)
    return dR.flatten()

def build_HR_1A(R, T, H):
    """Calculate the projection <i|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>."""
    X1A = 0.0
    X1A -= ccpy_einsum("mi,m->i", H.a.oo, R.a)
    X1A -= 0.5 * ccpy_einsum("mnif,mfn->i", H.aa.ooov, R.aa)
    X1A -= ccpy_einsum("mnif,mfn->i", H.ab.ooov, R.ab)
    X1A += ccpy_einsum("me,iem->i", H.a.ov, R.aa)
    X1A += ccpy_einsum("me,iem->i", H.b.ov, R.ab)
    # additional terms with R3
    X1A += 0.25 * ccpy_einsum("mnef,iefmn->i", H.aa.oovv, R.aaa)
    X1A += ccpy_einsum("mnef,iefmn->i", H.ab.oovv, R.aab)
    X1A += 0.25 * ccpy_einsum("mnef,iefmn->i", H.bb.oovv, R.abb)
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
    # additional terms with R3
    X2A += 0.5 * ccpy_einsum("me,ibejm->ibj", H.a.ov, R.aaa)
    X2A += 0.5 * ccpy_einsum("me,ibejm->ibj", H.b.ov, R.aab)
    X2A += 0.25 * ccpy_einsum("bnef,iefjn->ibj", H.aa.vovv, R.aaa)
    X2A += 0.5 * ccpy_einsum("bnef,iefjn->ibj", H.ab.vovv, R.aab)
    X2A -= 0.5 * ccpy_einsum("mnjf,ibfmn->ibj", H.aa.ooov, R.aaa)
    X2A -= ccpy_einsum("mnjf,ibfmn->ibj", H.ab.ooov, R.aab)
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
    # additional terms with R3
    X2B += ccpy_einsum("me,iebmj->ibj", H.a.ov, R.aab)
    X2B += ccpy_einsum("me,ibejm->ibj", H.b.ov, R.abb)
    X2B += ccpy_einsum("nbfe,ifenj->ibj", H.ab.ovvv, R.aab)
    X2B += 0.5 * ccpy_einsum("bnef,iefjn->ibj", H.bb.vovv, R.abb)
    X2B -= 0.5 * ccpy_einsum("mnif,mfbnj->ibj", H.aa.ooov, R.aab)
    X2B -= ccpy_einsum("mnif,mfbnj->ibj", H.ab.ooov, R.abb)
    X2B -= ccpy_einsum("nmfj,ifbnm->ibj", H.ab.oovo, R.aab)
    X2B -= 0.5 * ccpy_einsum("mnjf,ifbnm->ibj", H.bb.ooov, R.abb)
    return X2B

def build_HR_3A(R, T, X, H):
    """Calculate the projection <ijkbc|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>."""
    X3A = -(3.0 / 12.0) * ccpy_einsum("mj,ibcmk->ibcjk", H.a.oo, R.aaa)
    X3A += (2.0 / 12.0) * ccpy_einsum("be,iecjk->ibcjk", H.a.vv, R.aaa)
    X3A += (3.0 / 24.0) * ccpy_einsum("mnjk,ibcmn->ibcjk", H.aa.oooo, R.aaa)
    X3A += (1.0 / 24.0) * ccpy_einsum("bcef,iefjk->ibcjk", H.aa.vvvv, R.aaa)
    X3A += (6.0 / 12.0) * ccpy_einsum("bmje,iecmk->ibcjk", H.aa.voov, R.aaa)
    X3A += (6.0 / 12.0) * ccpy_einsum("bmje,icekm->ibcjk", H.ab.voov, R.aab)
    # moment-like terms
    X3A -= (6.0 / 12.0) * ccpy_einsum("cmkj,ibm->ibcjk", H.aa.vooo, R.aa)
    X3A += (3.0 / 12.0) * ccpy_einsum("cbke,iej->ibcjk", H.aa.vvov, R.aa)
    # 3-body Hbar terms factorized using intermediates
    X3A -= (3.0 / 12.0) * ccpy_einsum("imj,bcmk->ibcjk", X["aa"]["ooo"], T.aa)
    X3A += (6.0 / 12.0) * ccpy_einsum("ibe,ecjk->ibcjk", X["aa"]["ovv"], T.aa)
    # parts with T3 (these should be all bare integrals)
    X3A += (3.0 / 12.0) * ccpy_einsum("iem,ebcmjk->ibcjk", X["aa"]["ovo"], T.aaa) # [1]
    X3A += (3.0 / 12.0) * ccpy_einsum("iem,bcejkm->ibcjk", X["ab"]["ovo"], T.aab) # [2]
    X3A += (2.0 / 24.0) * ccpy_einsum("bef,fecijk->ibcjk", X["aa"]["vvv"], T.aaa) # [3]
    X3A += (1.0 / 12.0) * ccpy_einsum("e,ebcijk->ibcjk", X["a"]["v"], T.aaa)      # [4]
    X3A -= np.transpose(X3A, (3, 1, 2, 0, 4)) + np.transpose(X3A, (4, 1, 2, 3, 0)) # antisymmetrize A(i/jk)
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    return X3A

def build_HR_3B(R, T, X, H):
    """Calculate the projection <ijk~bc~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>."""
    X3B = -ccpy_einsum("mj,ibcmk->ibcjk", H.a.oo, R.aab) # (1)
    X3B -= 0.5 * ccpy_einsum("mk,ibcjm->ibcjk", H.b.oo, R.aab) # (2)
    X3B += 0.5 * ccpy_einsum("be,iecjk->ibcjk", H.a.vv, R.aab) # (3)
    X3B += 0.5 * ccpy_einsum("ce,ibejk->ibcjk", H.b.vv, R.aab) # (4)
    X3B += ccpy_einsum("mnjk,ibcmn->ibcjk", H.ab.oooo, R.aab) # (5)
    X3B += 0.25 * ccpy_einsum("mnij,mbcnk->ibcjk", H.aa.oooo, R.aab) # (6)
    X3B += 0.5 * ccpy_einsum("bcef,iefjk->ibcjk", H.ab.vvvv, R.aab) # (7)
    X3B += 0.5 * ccpy_einsum("mcek,ibejm->ibcjk", H.ab.ovvo, R.aaa) # (8)
    X3B += 0.5 * ccpy_einsum("cmke,ibejm->ibcjk", H.bb.voov, R.aab) # (9)
    X3B += ccpy_einsum("bmje,iecmk->ibcjk", H.aa.voov, R.aab) # (10)
    X3B += ccpy_einsum("bmje,iecmk->ibcjk", H.ab.voov, R.abb) # (11)
    X3B -= ccpy_einsum("mcje,ibemk->ibcjk", H.ab.ovov, R.aab) # (12)
    X3B -= 0.5 * ccpy_einsum("bmek,iecjm->ibcjk", H.ab.vovo, R.aab) # (13)
    # moment-like terms
    X3B -= ccpy_einsum("mcjk,ibm->ibcjk", H.ab.ovoo, R.aa) # (14)
    X3B -= 0.5 * ccpy_einsum("bmji,mck->ibcjk", H.aa.vooo, R.ab) # (15)
    X3B -= ccpy_einsum("bmjk,icm->ibcjk", H.ab.vooo, R.ab) # (16)
    X3B += ccpy_einsum("bcje,iek->ibcjk", H.ab.vvov, R.ab) # (17)
    X3B += 0.5 * ccpy_einsum("bcek,iej->ibcjk", H.ab.vvvo, R.aa) # (18)
    # 3-body Hbar terms factorized using intermediates
    X3B += 0.5 * ccpy_einsum("eck,ebij->ibcjk", X["ab"]["vvo"], T.aa) # (19)
    X3B -= 0.5 * ccpy_einsum("imj,bcmk->ibcjk", X["aa"]["ooo"], T.ab) # (20)
    X3B -= ccpy_einsum("imk,bcjm->ibcjk", X["ab"]["ooo"], T.ab) # (21)
    X3B += ccpy_einsum("ice,bejk->ibcjk", X["ab"]["ovv"], T.ab) # (22)
    X3B += ccpy_einsum("ibe,ecjk->ibcjk", X["aa"]["ovv"], T.ab) # (23)
    # parts with T3 (these should be all bare integrals)
    X3B += ccpy_einsum("iem,ebcmjk->ibcjk", X["aa"]["ovo"], T.aab) # [1]
    X3B += ccpy_einsum("iem,becjmk->ibcjk", X["ab"]["ovo"], T.abb) # [2]
    X3B -= (1.0 / 2.0) * ccpy_einsum("emk,ebcijm->ibcjk", X["ab"]["voo"], T.aab) # [3]
    X3B += (1.0 / 4.0) * ccpy_einsum("bfe,efcijk->ibcjk", X["aa"]["vvv"], T.aab) # [4]
    X3B += (1.0 / 2.0) * ccpy_einsum("ecf,ebfijk->ibcjk", X["ab"]["vvv"], T.aab) # [5]
    X3B += (1.0 / 2.0) * ccpy_einsum("e,ebcijk->ibcjk", X["a"]["v"], T.aab)      # [6]
    X3B -= np.transpose(X3B, (3, 1, 2, 0, 4)) # antisymmetrize (ij)
    return X3B

def build_HR_3C(R, T, X, H):
    """Calculate the projection <ij~k~b~c~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>."""
    X3C = -(2.0 / 4.0) * ccpy_einsum("mj,ibcmk->ibcjk", H.b.oo, R.abb) # (1)
    X3C -= (1.0 / 4.0) * ccpy_einsum("mi,mbcjk->ibcjk", H.a.oo, R.abb) # (2)
    X3C += (2.0 / 4.0) * ccpy_einsum("be,iecjk->ibcjk", H.b.vv, R.abb) # (3)
    X3C += (1.0 / 8.0) * ccpy_einsum("mnjk,ibcmn->ibcjk", H.bb.oooo, R.abb) # (4)
    X3C += (2.0 / 4.0) * ccpy_einsum("mnij,mbcnk->ibcjk", H.ab.oooo, R.abb) # (5)
    X3C += (1.0 / 8.0) * ccpy_einsum("bcef,iefjk->ibcjk", H.bb.vvvv, R.abb) # (6)
    X3C += ccpy_einsum("mbej,iecmk->ibcjk", H.ab.ovvo, R.aab) # (7)
    X3C += ccpy_einsum("bmje,iecmk->ibcjk", H.bb.voov, R.abb) # (8)
    X3C -= (2.0 / 4.0) * ccpy_einsum("mbie,mecjk->ibcjk", H.ab.ovov, R.abb) # (9)
    # moment-like terms
    X3C -= ccpy_einsum("mcik,mbj->ibcjk", H.ab.ovoo, R.ab) # (10)
    X3C -= (2.0 / 4.0) * ccpy_einsum("cmkj,ibm->ibcjk", H.bb.vooo, R.ab) # (11)
    X3C += (2.0 / 4.0) * ccpy_einsum("cbke,iej->ibcjk", H.bb.vvov, R.ab) # (12)
    # 3-body Hbar terms factorized using intermediates
    X3C -= (2.0 / 4.0) * ccpy_einsum("imj,bcmk->ibcjk", X["ab"]["ooo"], T.bb) # (13)
    X3C += (2.0 / 4.0) * ccpy_einsum("ibe,ecjk->ibcjk", X["ab"]["ovv"], T.bb) # (14)
    X3C += ccpy_einsum("ebj,ecik->ibcjk", X["ab"]["vvo"], T.ab) # (15)
    # parts with T3 (these should be all bare integrals)
    X3C += (1.0 / 4.0) * ccpy_einsum("iem,ebcmjk->ibcjk", X["aa"]["ovo"], T.abb) # [1]
    X3C += (1.0 / 4.0) * ccpy_einsum("iem,ebcmjk->ibcjk", X["ab"]["ovo"], T.bbb) # [2]
    X3C -= (2.0 / 4.0) * ccpy_einsum("emj,ebcimk->ibcjk", X["ab"]["voo"], T.abb) # [3]
    X3C += (2.0 / 4.0) * ccpy_einsum("ebf,efcijk->ibcjk", X["ab"]["vvv"], T.abb) # [4]
    X3C += (1.0 / 4.0) * ccpy_einsum("e,ebcijk->ibcjk", X["a"]["v"], T.abb)      # [5]
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3C -= np.transpose(X3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3C
