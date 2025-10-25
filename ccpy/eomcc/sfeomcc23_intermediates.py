import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

def get_sfeomcc23_intermediates(H, R, T, system):

    X = {"ab": {"ovoo": 0.0, "vooo": 0.0, "vvvo": 0.0},
         "bb": {"ovoo": 0.0, "vvvo": 0.0, "vvov": 0.0},
    }

    # x(m~e) intermediate
    x_ov = -ccpy_einsum("mnef,fm->ne", H.ab.oovv, R.b)

    # I've chosen to remove the x_ov * T2 term in each of the 3p-1h intermediates
    # because I'm assuming that this x_ov * T2**2 final term is double counted
    # if we include it both the 3h-1p and 3p-1h intermediates.

    # [1] x(mc~jk)
    X["ab"]["ovoo"] = (
        0.5 * ccpy_einsum("me,ecjk->mcjk", H.a.ov, R.ab)
        - 0.5 * ccpy_einsum("mnjk,cn->mcjk", H.aa.oooo, R.b)
        + ccpy_einsum("mcje,ek->mcjk", H.ab.ovov, R.b)
        + ccpy_einsum("mnjf,fcnk->mcjk", H.aa.ooov, R.ab)
        + ccpy_einsum("mnjf,fcnk->mcjk", H.ab.ooov, R.bb)
        + 0.25 * ccpy_einsum("mnef,efcjnk->mcjk", H.aa.oovv, R.aab)
        + 0.5 * ccpy_einsum("mnef,efcjnk->mcjk", H.ab.oovv, R.abb)
    )
    # antisymmetrize (jk)
    X["ab"]["ovoo"] -= np.transpose(X["ab"]["ovoo"], (0, 1, 3, 2))

    # [2] x(am~ik)
    X["ab"]["vooo"] = (
        0.5 * ccpy_einsum("me,aeik->amik", H.b.ov, R.ab)
        + 0.5 * ccpy_einsum("me,aeik->amik", x_ov, T.aa) # (!)
        + ccpy_einsum("amie,ek->amik", H.ab.voov, R.b)
        - ccpy_einsum("nmie,aenk->amik", H.ab.ooov, R.ab)
        + 0.5 * ccpy_einsum("amfe,feik->amik", H.ab.vovv, R.ab)
        + 0.5 * ccpy_einsum("nmfe,afeink->amik", H.ab.oovv, R.aab)
        + 0.25 * ccpy_einsum("nmfe,afeink->amik", H.bb.oovv, R.abb)
    )
    # antisymmetrize (ik)
    X["ab"]["vooo"] -= np.transpose(X["ab"]["vooo"], (0, 1, 3, 2))

    # [4] x(ac~ek)
    X["ab"]["vvvo"] = (
        -ccpy_einsum("me,acmk->acek", H.a.ov, R.ab)
        + ccpy_einsum("acef,fk->acek", H.ab.vvvv, R.b)
        + ccpy_einsum("amke,cm->acek", H.aa.voov, R.b) # flip sign h2a(amek) -> h2a(amke)
        + ccpy_einsum("anef,fcnk->acek", H.aa.vovv, R.ab)
        + ccpy_einsum("anef,fcnk->acek", H.ab.vovv, R.bb)
        - 0.5 * ccpy_einsum("mnef,afcmnk->acek", H.aa.oovv, R.aab)
        - ccpy_einsum("mnef,afcmnk->acek", H.ab.oovv, R.abb)
        #
        #+ ccpy_einsum("me,ackm->acek", x_ov, T.ab) # (!) flip sign to rearrange path k -> c~, e -> a ## block
        + 0.5 * ccpy_einsum("mnke,acnm->acek", H.aa.ooov, R.ab)
    )

    # [3] x(m~c~j~k)
    X["bb"]["ovoo"] = (
        ccpy_einsum("me,ecjk->mcjk", H.b.ov, R.bb)
        - ccpy_einsum("cmje,ek->mcjk", H.bb.voov, R.b) # flip sign h2c(mcje) -> -h2c(cmje)
        - ccpy_einsum("nmkj,cn->mcjk", H.ab.oooo, R.b)
        + ccpy_einsum("nmfj,fcnk->mcjk", H.ab.oovo, R.ab)
        + ccpy_einsum("mnjf,fcnk->mcjk", H.bb.ooov, R.bb)
        + ccpy_einsum("nmfe,fecnjk->mcjk", H.ab.oovv, R.abb)
        + 0.5 * ccpy_einsum("nmfe,fecnjk->mcjk", H.bb.oovv, R.bbb)
        #
        - ccpy_einsum("me,eckj->mcjk ", x_ov, T.ab) # (!) flip sign to rearrange path k -> c~, j~ -> m~
        + 0.5 * ccpy_einsum("cmfe,efjk->mcjk", H.bb.vovv, R.bb)
    )

    # [5] x(b~c~e~k)
    X["bb"]["vvvo"] = (
        -0.5 * ccpy_einsum("me,bcmk->bcek", H.b.ov, R.bb)
        - ccpy_einsum("mbke,cm->bcek", H.ab.ovov, R.b)
        + 0.5 * ccpy_einsum("bcef,fk->bcek", H.bb.vvvv, R.b)
        + ccpy_einsum("nbfe,fcnk->bcek", H.ab.ovvv, R.ab)
        + ccpy_einsum("bnef,fcnk->bcek", H.bb.vovv, R.bb)
        - 0.5 * ccpy_einsum("nmfe,fbcnmk->bcek", H.ab.oovv, R.abb)
        - 0.25 * ccpy_einsum("nmfe,fbcnmk->bcek", H.bb.oovv, R.bbb)
    )
    # antisymmetrize (bc)
    X["bb"]["vvvo"] -= np.transpose(X["bb"]["vvvo"], (1, 0, 2, 3))

    # [6] x(b~c~j~e)
    X["bb"]["vvov"] = (
        -0.5 * ccpy_einsum("me,bcjm->bcje", H.a.ov, R.bb)
        #- 0.5 * ccpy_einsum("me,bcjm->bcje", x_ov, T.bb) # (!) ## block
        - ccpy_einsum("mbej,cm->bcje", H.ab.ovvo, R.b)
        - ccpy_einsum("mbef,fcjm->bcje", H.ab.ovvv, R.bb)
        + 0.5 * ccpy_einsum("mnej,bcnm->bcje", H.ab.oovo, R.bb)
        - 0.25 * ccpy_einsum("mnef,fbcnjm->bcje", H.aa.oovv, R.abb)
        - 0.5 * ccpy_einsum("mnef,fbcnjm->bcje", H.ab.oovv, R.bbb)
    )
    # antisymmetrize (bc)
    X["bb"]["vvov"] -= np.transpose(X["bb"]["vvov"], (1, 0, 2, 3))

    return X