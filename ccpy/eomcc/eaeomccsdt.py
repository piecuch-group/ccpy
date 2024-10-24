import numpy as np
from ccpy.eomcc.eaeom3_intermediates import get_eaeomccsdt_intermediates, add_o_term
from ccpy.lib.core import cc_loops2

# R.a -> (nua) -> (a)
# R.aa -> (nua,nua,noa) -> (abj)
# R.ab -> (nua,nub,nob) -> (ab~j~)
# R.aaa -> (nua,nua,nua,noa,noa) -> (abcjk)
# R.aab -> (nua,nua,nub,noa,nob) -> (abc~jk~)
# R.abb -> (nua,nub,nub,nob,nob) -> (ab~c~j~k~)

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
    X1A = np.einsum("ae,e->a", H.a.vv, R.a, optimize=True)
    X1A += 0.5 * np.einsum("anef,efn->a", H.aa.vovv, R.aa, optimize=True)
    X1A += np.einsum("anef,efn->a", H.ab.vovv, R.ab, optimize=True)
    X1A += np.einsum("me,aem->a", H.a.ov, R.aa, optimize=True)
    X1A += np.einsum("me,aem->a", H.b.ov, R.ab, optimize=True)
    # terms with R3
    X1A += 0.25 * np.einsum("mnef,aefmn->a", H.aa.oovv, R.aaa, optimize=True)
    X1A += np.einsum("mnef,aefmn->a", H.ab.oovv, R.aab, optimize=True)
    X1A += 0.25 * np.einsum("mnef,aefmn->a", H.bb.oovv, R.abb, optimize=True)
    return X1A

def build_HR_2A(R, T, X, H):
    """Calculate the projection <ajb|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    X2A = 0.5 * np.einsum("baje,e->abj", H.aa.vvov, R.a, optimize=True)
    X2A -= 0.5 * np.einsum("mj,abm->abj", H.a.oo, R.aa, optimize=True)
    X2A += 0.25 * np.einsum("abef,efj->abj", H.aa.vvvv, R.aa, optimize=True)
    X2A -= 0.5 * np.einsum("m,abmj->abj", X["a"]["o"], T.aa, optimize=True)
    X2A += np.einsum("ae,ebj->abj", H.a.vv, R.aa, optimize=True)
    X2A += np.einsum("bmje,aem->abj", H.aa.voov, R.aa, optimize=True)
    X2A += np.einsum("bmje,aem->abj", H.ab.voov, R.ab, optimize=True)
    # terms with R3
    X2A += 0.5 * np.einsum("me,abejm->abj", H.a.ov, R.aaa, optimize=True)
    X2A += 0.5 * np.einsum("me,abejm->abj", H.b.ov, R.aab, optimize=True)
    X2A -= 0.25 * np.einsum("mnjf,abfmn->abj", H.aa.ooov, R.aaa, optimize=True)
    X2A -= 0.5 * np.einsum("mnjf,abfmn->abj", H.ab.ooov, R.aab, optimize=True)
    X2A += 0.5 * np.einsum("bnef,aefjn->abj", H.aa.vovv, R.aaa, optimize=True)
    X2A += np.einsum("bnef,aefjn->abj", H.ab.vovv, R.aab, optimize=True)
    X2A -= np.transpose(X2A, (1, 0, 2))
    return X2A

def build_HR_2B(R, T, X, H):
    """Calculate the projection <aj~b~|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    X2B = np.einsum("abej,e->abj", H.ab.vvvo, R.a, optimize=True)
    X2B += np.einsum("ae,ebj->abj", H.a.vv, R.ab, optimize=True)
    X2B += np.einsum("be,aej->abj", H.b.vv, R.ab, optimize=True)
    X2B -= np.einsum("mj,abm->abj", H.b.oo, R.ab, optimize=True)
    X2B += np.einsum("mbej,aem->abj", H.ab.ovvo, R.aa, optimize=True)
    X2B += np.einsum("bmje,aem->abj", H.bb.voov, R.ab, optimize=True)
    X2B -= np.einsum("amej,ebm->abj", H.ab.vovo, R.ab, optimize=True)
    X2B += np.einsum("abef,efj->abj", H.ab.vvvv, R.ab, optimize=True)
    X2B -= np.einsum("m,abmj->abj", X["a"]["o"], T.ab, optimize=True)
    # terms wtih R3
    X2B += np.einsum("me,aebmj->abj", H.a.ov, R.aab, optimize=True)
    X2B += np.einsum("me,aebmj->abj", H.b.ov, R.abb, optimize=True)
    X2B -= np.einsum("nmfj,afbnm->abj", H.ab.oovo, R.aab, optimize=True)
    X2B -= 0.5 * np.einsum("mnjf,abfmn->abj", H.bb.ooov, R.abb, optimize=True)
    X2B += np.einsum("nbfe,afenj->abj", H.ab.ovvv, R.aab, optimize=True)
    X2B += 0.5 * np.einsum("bnef,aefjn->abj", H.bb.vovv, R.abb, optimize=True)
    X2B += 0.5 * np.einsum("anef,efbnj->abj", H.aa.vovv, R.aab, optimize=True)
    X2B += np.einsum("anef,ebfjn->abj", H.ab.vovv, R.abb, optimize=True)
    return X2B

def build_HR_3A(R, T, X, H):
    """Calculate the projection <abcjk|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    X3A = -(2.0 / 12.0) * np.einsum("mj,abcmk->abcjk", H.a.oo, R.aaa, optimize=True)       # (1)
    X3A += (3.0 / 12.0) * np.einsum("be,aecjk->abcjk", H.a.vv, R.aaa, optimize=True)       # (2)
    X3A += (3.0 / 24.0) * np.einsum("abef,efcjk->abcjk", H.aa.vvvv, R.aaa, optimize=True)  # (3)
    X3A += (1.0 / 24.0) * np.einsum("mnjk,abcmn->abcjk", H.aa.oooo, R.aaa, optimize=True)  # (4)
    X3A += (6.0 / 12.0) * np.einsum("cmke,abejm->abcjk", H.aa.voov, R.aaa, optimize=True)  # (5)
    X3A += (6.0 / 12.0) * np.einsum("cmke,abejm->abcjk", H.ab.voov, R.aab, optimize=True)  # (6)
    # moment-like terms
    X3A -= (3.0 / 12.0) * np.einsum("cmkj,abm->abcjk", H.aa.vooo, R.aa, optimize=True)     # (7)
    X3A += (6.0 / 12.0) * np.einsum("cbke,aej->abcjk", H.aa.vvov, R.aa, optimize=True)     # (8)
    # 3-body Hbar terms factorized using intermediates
    X3A -= (6.0 / 12.0) * np.einsum("amj,bcmk->abcjk", X["aa"]["voo"], T.aa, optimize=True) # (9)
    X3A += (3.0 / 12.0) * np.einsum("abe,ecjk->abcjk", X["aa"]["vvv"], T.aa, optimize=True) # (10)
    # add T3 terms
    X3A += (3.0 / 12.0) * np.einsum("aem,ebcmjk->abcjk", X["aa"]["vvo"], T.aaa, optimize=True) # [1]
    X3A += (3.0 / 12.0) * np.einsum("aem,bcejkm->abcjk", X["ab"]["vvo"], T.aab, optimize=True) # [2]
    X3A += (2.0 / 24.0) * np.einsum("mnj,abcmnk->abcjk", X["aa"]["ooo"], T.aaa, optimize=True) # [3]
    X3A -= (1.0 / 12.0) * np.einsum("m,abcmjk->abcjk", X["a"]["o"], T.aaa, optimize=True)      # [4]
    X3A -= np.transpose(X3A, (1, 0, 2, 3, 4)) + np.transpose(X3A, (2, 1, 0, 3, 4)) # antisymmetrize A(a/bc)
    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3A

def build_HR_3B(R, T, X, H):
    """Calculate the projection <abc~jk~|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    X3B = -(1.0 / 2.0) * np.einsum("mj,abcmk->abcjk", H.a.oo, R.aab, optimize=True) # (1)
    X3B -= (1.0 / 2.0) * np.einsum("mk,abcjm->abcjk", H.b.oo, R.aab, optimize=True) # (2)
    X3B += (1.0 / 2.0) * np.einsum("ce,abejk->abcjk", H.b.vv, R.aab, optimize=True) # (3)
    X3B += np.einsum("be,aecjk->abcjk", H.a.vv, R.aab, optimize=True) # (4)
    X3B += (1.0 / 2.0) * np.einsum("mnjk,abcmn->abcjk", H.ab.oooo, R.aab, optimize=True) # (5)
    X3B += (1.0 / 2.0) * np.einsum("mcek,abejm->abcjk", H.ab.ovvo, R.aaa, optimize=True) # (6)
    X3B += (1.0 / 2.0) * np.einsum("cmke,abejm->abcjk", H.bb.voov, R.aab, optimize=True) # (7)
    X3B += np.einsum("bmje,aecmk->abcjk", H.aa.voov, R.aab, optimize=True) # (8)
    X3B -= np.einsum("bmek,aecjm->abcjk", H.ab.vovo, R.aab, optimize=True) # (10)
    X3B -= (1.0 / 2.0) * np.einsum("mcje,abemk->abcjk", H.ab.ovov, R.aab, optimize=True) # (11)
    X3B += np.einsum("bmje,aecmk->abcjk", H.ab.voov, R.abb, optimize=True) # (9)
    X3B += (1.0 / 4.0) * np.einsum("abef,efcjk->abcjk", H.aa.vvvv, R.aab, optimize=True) # (12)
    X3B += np.einsum("bcef,aefjk->abcjk", H.ab.vvvv, R.aab, optimize=True) # (13)
    # moment-like terms
    X3B -= (1.0 / 2.0) * np.einsum("mcjk,abm->abcjk", H.ab.ovoo, R.aa, optimize=True) # (14)
    X3B -= np.einsum("bmjk,acm->abcjk", H.ab.vooo, R.ab, optimize=True) # (15)
    X3B += np.einsum("bcek,aej->abcjk", H.ab.vvvo, R.aa, optimize=True) # (16)
    X3B += np.einsum("bcje,aek->abcjk", H.ab.vvov, R.ab, optimize=True) # (17)
    X3B += (1.0 / 2.0) * np.einsum("baje,eck->abcjk", H.aa.vvov, R.ab, optimize=True) # (23)
    # 3-body Hbar terms factorized using intermediates
    X3B -= (1.0 / 2.0) * np.einsum("mck,abmj->abcjk", X["ab"]["ovo"], T.aa, optimize=True) # (18)
    X3B -= np.einsum("amj,bcmk->abcjk", X["aa"]["voo"], T.ab, optimize=True) # (19)
    X3B -= np.einsum("amk,bcjm->abcjk", X["ab"]["voo"], T.ab, optimize=True) # (20)
    X3B += (1.0 / 2.0) * np.einsum("abe,ecjk->abcjk", X["aa"]["vvv"], T.ab, optimize=True) # (21)
    X3B += np.einsum("ace,bejk->abcjk", X["ab"]["vvv"], T.ab, optimize=True) # (22)
    # add T3 terms
    X3B += np.einsum("aem,ebcmjk->abcjk", X["aa"]["vvo"], T.aab, optimize=True) # [1]
    X3B += np.einsum("aem,bcejkm->abcjk", X["ab"]["vvo"], T.abb, optimize=True) # [2]
    X3B -= (1.0 / 2.0) * np.einsum("mcf,abfmjk->abcjk", X["ab"]["ovv"], T.aab, optimize=True) # [3]
    X3B += (1.0 / 4.0) * np.einsum("mnj,abcmnk->abcjk", X["aa"]["ooo"], T.aab, optimize=True) # [4]
    X3B += (1.0 / 2.0) * np.einsum("mnk,abcmjn->abcjk", X["ab"]["ooo"], T.aab, optimize=True) # [5]
    X3B -= (1.0 / 2.0) * np.einsum("m,abcmjk->abcjk", X["a"]["o"], T.aab, optimize=True) # [6]
    X3B -= np.transpose(X3B, (1, 0, 2, 3, 4)) # antisymmetrize A(ab)
    return X3B

def build_HR_3C(R, T, X, H):
    """Calculate the projection <ab~c~j~k~|[ (H_N e^(T1+T2))_C*(R1h+R2p1h+R3p2h) ]_C|0>."""
    X3C = -(2.0 / 4.0) * np.einsum("mj,abcmk->abcjk", H.b.oo, R.abb, optimize=True) # (1)
    X3C += (2.0 / 4.0) * np.einsum("be,aecjk->abcjk", H.b.vv, R.abb, optimize=True) # (2)
    X3C += (1.0 / 4.0) * np.einsum("ae,ebcjk->abcjk", H.a.vv, R.abb, optimize=True) # (3)
    X3C += (1.0 / 8.0) * np.einsum("mnjk,abcmn->abcjk", H.bb.oooo, R.abb, optimize=True) # (4)
    X3C += np.einsum("mbej,aecmk->abcjk", H.ab.ovvo, R.aab, optimize=True) # (5)
    X3C += np.einsum("bmje,aecmk->abcjk", H.bb.voov, R.abb, optimize=True) # (6)
    X3C -= (2.0 / 4.0) * np.einsum("amej,ebcmk->abcjk", H.ab.vovo, R.abb, optimize=True) # (7)
    X3C += (2.0 / 4.0) * np.einsum("abef,efcjk->abcjk", H.ab.vvvv, R.abb, optimize=True) # (8)
    X3C += (1.0 / 8.0) * np.einsum("bcef,aefjk->abcjk", H.bb.vvvv, R.abb, optimize=True) # (9)
    # moment-like terms
    X3C -= (2.0 / 4.0) * np.einsum("cmkj,abm->abcjk", H.bb.vooo, R.ab, optimize=True) # (10)
    X3C += (2.0 / 4.0) * np.einsum("cbke,aej->abcjk", H.bb.vvov, R.ab, optimize=True) # (11)
    X3C += np.einsum("acek,ebj->abcjk", H.ab.vvvo, R.ab, optimize=True) # (12)
    # 3-body Hbar terms factorized using intermediates
    X3C -= (2.0 / 4.0) * np.einsum("amj,bcmk->abcjk", X["ab"]["voo"], T.bb, optimize=True) # (13)
    X3C -= np.einsum("mck,abmj->abcjk", X["ab"]["ovo"], T.ab, optimize=True) # (14)
    X3C += (2.0 / 4.0) * np.einsum("abe,ecjk->abcjk", X["ab"]["vvv"], T.bb, optimize=True) # (15)
    # add T3 terms
    X3C += (1.0 / 4.0) * np.einsum("aem,ebcmjk->abcjk", X["aa"]["vvo"], T.abb, optimize=True) # [1]
    X3C += (1.0 / 4.0) * np.einsum("aem,ebcmjk->abcjk", X["ab"]["vvo"], T.bbb, optimize=True) # [2]
    X3C -= (2.0 / 4.0) * np.einsum("mbf,afcmjk->abcjk", X["ab"]["ovv"], T.abb, optimize=True) # [3]
    X3C += (2.0 / 4.0) * np.einsum("mnk,abcmjn->abcjk", X["ab"]["ooo"], T.abb, optimize=True) # [4]
    X3C -= (1.0 / 4.0) * np.einsum("m,abcmjk->abcjk", X["a"]["o"], T.abb, optimize=True) # [5]
    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3C -= np.transpose(X3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    return X3C
