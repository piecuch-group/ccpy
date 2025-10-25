'''
Electron Attachment Equation-of-Motion Coupled-Cluster
Method with 1p, 2p-1h, and 3p-2h Excitations on top of CCSDT [EA-EOMCCSDT(3p-2h)]
'''

import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.eomcc.eaeom3_intermediates import get_eaeomccsdt_intermediates, add_o_term
from ccpy.lib.core import cc_loops2

def update(R, omega, H, RHF_symmetry, system):

    R.a, R.aa, R.ab, R.aaa, R.aab, R.abb = cc_loops2.update_r_3p2h(
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
    X = get_eaeomccsdt_intermediates(H, R)
    # update R1
    dR.a = build_HR_1A(R, T, H)
    # update R2
    dR.aa = build_HR_2A(R, T, X, H)
    dR.ab = build_HR_2B(R, T, X, H)
    # update R3
    X = add_o_term(X, H, R)
    dR.aaa = build_HR_3A(R, T, X, H)
    dR.aab = build_HR_3B(R, T, X, H)
    dR.abb = build_HR_3C(R, T, X, H)
    return dR.flatten()

def build_HR_1A(R, T, H):
    """Calculate the projection <a|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    X1A = ccpy_einsum("ae,e->a", H.a.vv, R.a)
    X1A += 0.5 * ccpy_einsum("anef,efn->a", H.aa.vovv, R.aa)
    X1A += ccpy_einsum("anef,efn->a", H.ab.vovv, R.ab)
    X1A += ccpy_einsum("me,aem->a", H.a.ov, R.aa)
    X1A += ccpy_einsum("me,aem->a", H.b.ov, R.ab)
    # terms with R3
    X1A += 0.25 * ccpy_einsum("mnef,aefmn->a", H.aa.oovv, R.aaa)
    X1A += ccpy_einsum("mnef,aefmn->a", H.ab.oovv, R.aab)
    X1A += 0.25 * ccpy_einsum("mnef,aefmn->a", H.bb.oovv, R.abb)
    return X1A

def build_HR_2A(R, T, X, H):
    """Calculate the projection <ajb|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    X2A = 0.5 * ccpy_einsum("baje,e->abj", H.aa.vvov, R.a)
    X2A -= 0.5 * ccpy_einsum("mj,abm->abj", H.a.oo, R.aa)
    X2A += 0.25 * ccpy_einsum("abef,efj->abj", H.aa.vvvv, R.aa)
    X2A -= 0.5 * ccpy_einsum("m,abmj->abj", X["a"]["o"], T.aa)
    X2A += ccpy_einsum("ae,ebj->abj", H.a.vv, R.aa)
    X2A += ccpy_einsum("bmje,aem->abj", H.aa.voov, R.aa)
    X2A += ccpy_einsum("bmje,aem->abj", H.ab.voov, R.ab)
    # terms with R3
    X2A += 0.5 * ccpy_einsum("me,abejm->abj", H.a.ov, R.aaa)
    X2A += 0.5 * ccpy_einsum("me,abejm->abj", H.b.ov, R.aab)
    X2A -= 0.25 * ccpy_einsum("mnjf,abfmn->abj", H.aa.ooov, R.aaa)
    X2A -= 0.5 * ccpy_einsum("mnjf,abfmn->abj", H.ab.ooov, R.aab)
    X2A += 0.5 * ccpy_einsum("bnef,aefjn->abj", H.aa.vovv, R.aaa)
    X2A += ccpy_einsum("bnef,aefjn->abj", H.ab.vovv, R.aab)
    X2A -= np.transpose(X2A, (1, 0, 2))
    return X2A

def build_HR_2B(R, T, X, H):
    """Calculate the projection <aj~b~|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    X2B = ccpy_einsum("abej,e->abj", H.ab.vvvo, R.a)
    X2B += ccpy_einsum("ae,ebj->abj", H.a.vv, R.ab)
    X2B += ccpy_einsum("be,aej->abj", H.b.vv, R.ab)
    X2B -= ccpy_einsum("mj,abm->abj", H.b.oo, R.ab)
    X2B += ccpy_einsum("mbej,aem->abj", H.ab.ovvo, R.aa)
    X2B += ccpy_einsum("bmje,aem->abj", H.bb.voov, R.ab)
    X2B -= ccpy_einsum("amej,ebm->abj", H.ab.vovo, R.ab)
    X2B += ccpy_einsum("abef,efj->abj", H.ab.vvvv, R.ab)
    X2B -= ccpy_einsum("m,abmj->abj", X["a"]["o"], T.ab)
    # terms wtih R3
    X2B += ccpy_einsum("me,aebmj->abj", H.a.ov, R.aab)
    X2B += ccpy_einsum("me,aebmj->abj", H.b.ov, R.abb)
    X2B -= ccpy_einsum("nmfj,afbnm->abj", H.ab.oovo, R.aab)
    X2B -= 0.5 * ccpy_einsum("mnjf,abfmn->abj", H.bb.ooov, R.abb)
    X2B += ccpy_einsum("nbfe,afenj->abj", H.ab.ovvv, R.aab)
    X2B += 0.5 * ccpy_einsum("bnef,aefjn->abj", H.bb.vovv, R.abb)
    X2B += 0.5 * ccpy_einsum("anef,efbnj->abj", H.aa.vovv, R.aab)
    X2B += ccpy_einsum("anef,ebfjn->abj", H.ab.vovv, R.abb)
    return X2B

def build_HR_3A(R, T, X, H):
    """Calculate the projection <abcjk|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    X3A = -(2.0 / 12.0) * ccpy_einsum("mj,abcmk->abcjk", H.a.oo, R.aaa)       # (1)
    X3A += (3.0 / 12.0) * ccpy_einsum("be,aecjk->abcjk", H.a.vv, R.aaa)       # (2)
    X3A += (3.0 / 24.0) * ccpy_einsum("abef,efcjk->abcjk", H.aa.vvvv, R.aaa)  # (3)
    X3A += (1.0 / 24.0) * ccpy_einsum("mnjk,abcmn->abcjk", H.aa.oooo, R.aaa)  # (4)
    X3A += (6.0 / 12.0) * ccpy_einsum("cmke,abejm->abcjk", H.aa.voov, R.aaa)  # (5)
    X3A += (6.0 / 12.0) * ccpy_einsum("cmke,abejm->abcjk", H.ab.voov, R.aab)  # (6)
    # moment-like terms
    X3A -= (3.0 / 12.0) * ccpy_einsum("cmkj,abm->abcjk", H.aa.vooo, R.aa)     # (7)
    X3A += (6.0 / 12.0) * ccpy_einsum("cbke,aej->abcjk", H.aa.vvov, R.aa)     # (8)
    # 3-body Hbar terms factorized using intermediates
    X3A -= (6.0 / 12.0) * ccpy_einsum("amj,bcmk->abcjk", X["aa"]["voo"], T.aa) # (9)
    X3A += (3.0 / 12.0) * ccpy_einsum("abe,ecjk->abcjk", X["aa"]["vvv"], T.aa) # (10)
    # add T3 terms
    X3A += (3.0 / 12.0) * ccpy_einsum("aem,ebcmjk->abcjk", X["aa"]["vvo"], T.aaa) # [1]
    X3A += (3.0 / 12.0) * ccpy_einsum("aem,bcejkm->abcjk", X["ab"]["vvo"], T.aab) # [2]
    X3A += (2.0 / 24.0) * ccpy_einsum("mnj,abcmnk->abcjk", X["aa"]["ooo"], T.aaa) # [3]
    X3A -= (1.0 / 12.0) * ccpy_einsum("m,abcmjk->abcjk", X["a"]["o"], T.aaa)      # [4]
    X3A -= np.transpose(X3A, (1, 0, 2, 3, 4)) + np.transpose(X3A, (2, 1, 0, 3, 4)) # antisymmetrize A(a/bc)
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3A

def build_HR_3B(R, T, X, H):
    """Calculate the projection <abc~jk~|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    X3B = -(1.0 / 2.0) * ccpy_einsum("mj,abcmk->abcjk", H.a.oo, R.aab) # (1)
    X3B -= (1.0 / 2.0) * ccpy_einsum("mk,abcjm->abcjk", H.b.oo, R.aab) # (2)
    X3B += (1.0 / 2.0) * ccpy_einsum("ce,abejk->abcjk", H.b.vv, R.aab) # (3)
    X3B += ccpy_einsum("be,aecjk->abcjk", H.a.vv, R.aab) # (4)
    X3B += (1.0 / 2.0) * ccpy_einsum("mnjk,abcmn->abcjk", H.ab.oooo, R.aab) # (5)
    X3B += (1.0 / 2.0) * ccpy_einsum("mcek,abejm->abcjk", H.ab.ovvo, R.aaa) # (6)
    X3B += (1.0 / 2.0) * ccpy_einsum("cmke,abejm->abcjk", H.bb.voov, R.aab) # (7)
    X3B += ccpy_einsum("bmje,aecmk->abcjk", H.aa.voov, R.aab) # (8)
    X3B -= ccpy_einsum("bmek,aecjm->abcjk", H.ab.vovo, R.aab) # (10)
    X3B -= (1.0 / 2.0) * ccpy_einsum("mcje,abemk->abcjk", H.ab.ovov, R.aab) # (11)
    X3B += ccpy_einsum("bmje,aecmk->abcjk", H.ab.voov, R.abb) # (9)
    X3B += (1.0 / 4.0) * ccpy_einsum("abef,efcjk->abcjk", H.aa.vvvv, R.aab) # (12)
    X3B += ccpy_einsum("bcef,aefjk->abcjk", H.ab.vvvv, R.aab) # (13)
    # moment-like terms
    X3B -= (1.0 / 2.0) * ccpy_einsum("mcjk,abm->abcjk", H.ab.ovoo, R.aa) # (14)
    X3B -= ccpy_einsum("bmjk,acm->abcjk", H.ab.vooo, R.ab) # (15)
    X3B += ccpy_einsum("bcek,aej->abcjk", H.ab.vvvo, R.aa) # (16)
    X3B += ccpy_einsum("bcje,aek->abcjk", H.ab.vvov, R.ab) # (17)
    X3B += (1.0 / 2.0) * ccpy_einsum("baje,eck->abcjk", H.aa.vvov, R.ab) # (23)
    # 3-body Hbar terms factorized using intermediates
    X3B -= (1.0 / 2.0) * ccpy_einsum("mck,abmj->abcjk", X["ab"]["ovo"], T.aa) # (18)
    X3B -= ccpy_einsum("amj,bcmk->abcjk", X["aa"]["voo"], T.ab) # (19)
    X3B -= ccpy_einsum("amk,bcjm->abcjk", X["ab"]["voo"], T.ab) # (20)
    X3B += (1.0 / 2.0) * ccpy_einsum("abe,ecjk->abcjk", X["aa"]["vvv"], T.ab) # (21)
    X3B += ccpy_einsum("ace,bejk->abcjk", X["ab"]["vvv"], T.ab) # (22)
    # add T3 terms
    X3B += ccpy_einsum("aem,ebcmjk->abcjk", X["aa"]["vvo"], T.aab) # [1]
    X3B += ccpy_einsum("aem,bcejkm->abcjk", X["ab"]["vvo"], T.abb) # [2]
    X3B -= (1.0 / 2.0) * ccpy_einsum("mcf,abfmjk->abcjk", X["ab"]["ovv"], T.aab) # [3]
    X3B += (1.0 / 4.0) * ccpy_einsum("mnj,abcmnk->abcjk", X["aa"]["ooo"], T.aab) # [4]
    X3B += (1.0 / 2.0) * ccpy_einsum("mnk,abcmjn->abcjk", X["ab"]["ooo"], T.aab) # [5]
    X3B -= (1.0 / 2.0) * ccpy_einsum("m,abcmjk->abcjk", X["a"]["o"], T.aab) # [6]
    X3B -= np.transpose(X3B, (1, 0, 2, 3, 4)) # antisymmetrize A(ab)
    return X3B

def build_HR_3C(R, T, X, H):
    """Calculate the projection <ab~c~j~k~|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    X3C = -(2.0 / 4.0) * ccpy_einsum("mj,abcmk->abcjk", H.b.oo, R.abb) # (1)
    X3C += (2.0 / 4.0) * ccpy_einsum("be,aecjk->abcjk", H.b.vv, R.abb) # (2)
    X3C += (1.0 / 4.0) * ccpy_einsum("ae,ebcjk->abcjk", H.a.vv, R.abb) # (3)
    X3C += (1.0 / 8.0) * ccpy_einsum("mnjk,abcmn->abcjk", H.bb.oooo, R.abb) # (4)
    X3C += ccpy_einsum("mbej,aecmk->abcjk", H.ab.ovvo, R.aab) # (5)
    X3C += ccpy_einsum("bmje,aecmk->abcjk", H.bb.voov, R.abb) # (6)
    X3C -= (2.0 / 4.0) * ccpy_einsum("amej,ebcmk->abcjk", H.ab.vovo, R.abb) # (7)
    X3C += (2.0 / 4.0) * ccpy_einsum("abef,efcjk->abcjk", H.ab.vvvv, R.abb) # (8)
    X3C += (1.0 / 8.0) * ccpy_einsum("bcef,aefjk->abcjk", H.bb.vvvv, R.abb) # (9)
    # moment-like terms
    X3C -= (2.0 / 4.0) * ccpy_einsum("cmkj,abm->abcjk", H.bb.vooo, R.ab) # (10)
    X3C += (2.0 / 4.0) * ccpy_einsum("cbke,aej->abcjk", H.bb.vvov, R.ab) # (11)
    X3C += ccpy_einsum("acek,ebj->abcjk", H.ab.vvvo, R.ab) # (12)
    # 3-body Hbar terms factorized using intermediates
    X3C -= (2.0 / 4.0) * ccpy_einsum("amj,bcmk->abcjk", X["ab"]["voo"], T.bb) # (13)
    X3C -= ccpy_einsum("mck,abmj->abcjk", X["ab"]["ovo"], T.ab) # (14)
    X3C += (2.0 / 4.0) * ccpy_einsum("abe,ecjk->abcjk", X["ab"]["vvv"], T.bb) # (15)
    # add T3 terms
    X3C += (1.0 / 4.0) * ccpy_einsum("aem,ebcmjk->abcjk", X["aa"]["vvo"], T.abb) # [1]
    X3C += (1.0 / 4.0) * ccpy_einsum("aem,ebcmjk->abcjk", X["ab"]["vvo"], T.bbb) # [2]
    X3C -= (2.0 / 4.0) * ccpy_einsum("mbf,afcmjk->abcjk", X["ab"]["ovv"], T.abb) # [3]
    X3C += (2.0 / 4.0) * ccpy_einsum("mnk,abcmjn->abcjk", X["ab"]["ooo"], T.abb) # [4]
    X3C -= (1.0 / 4.0) * ccpy_einsum("m,abcmjk->abcjk", X["a"]["o"], T.abb) # [5]
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3C -= np.transpose(X3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3C
