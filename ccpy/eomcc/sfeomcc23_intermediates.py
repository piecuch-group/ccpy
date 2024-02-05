import numpy as np

def get_sfeomcc23_intermediates(H, R, T, system):

    X = {"ab": {"ovoo": 0.0, "vooo": 0.0, "vvvo": 0.0},
         "bb": {"ovoo": 0.0, "vvvo": 0.0, "vvov": 0.0},
    }

    # x(m~e) intermediate
    x_ov = -np.einsum("mnef,fm->ne", H.ab.oovv, R.b)

    # I've chosen to remove the x_ov * T2 term in each of the 3p-1h intermediates
    # because I'm assuming that this x_ov * T2**2 final term is double counted
    # if we include it both the 3h-1p and 3p-1h intermediates.

    # [1] x(mc~jk)
    X["ab"]["ovoo"] = (
        0.5 * np.einsum("me,ecjk->mcjk", H.a.ov, R.ab, optimize=True)
        - 0.5 * np.einsum("mnjk,cn->mcjk", H.aa.oooo, R.b, optimize=True)
        + np.einsum("mcje,ek->mcjk", H.ab.ovov, R.b, optimize=True)
        + np.einsum("mnjf,fcnk->mcjk", H.aa.ooov, R.ab, optimize=True)
        + np.einsum("mnjf,fcnk->mcjk", H.ab.ooov, R.bb, optimize=True)
        + 0.25 * np.einsum("mnef,efcjnk->mcjk", H.aa.oovv, R.aab, optimize=True)
        + 0.5 * np.einsum("mnef,efcjnk->mcjk", H.ab.oovv, R.abb, optimize=True)
    )
    # antisymmetrize (jk)
    X["ab"]["ovoo"] -= np.transpose(X["ab"]["ovoo"], (0, 1, 3, 2))

    # [2] x(am~ik)
    X["ab"]["vooo"] = (
        0.5 * np.einsum("me,aeik->amik", H.b.ov, R.ab, optimize=True)
        + 0.5 * np.einsum("me,aeik->amik", x_ov, T.aa, optimize=True) # (!)
        + np.einsum("amie,ek->amik", H.ab.voov, R.b, optimize=True)
        - np.einsum("nmie,aenk->amik", H.ab.ooov, R.ab, optimize=True)
        + 0.5 * np.einsum("amfe,feik->amik", H.ab.vovv, R.ab, optimize=True)
        + 0.5 * np.einsum("nmfe,afeink->amik", H.ab.oovv, R.aab, optimize=True)
        + 0.25 * np.einsum("nmfe,afeink->amik", H.bb.oovv, R.abb, optimize=True)
    )
    # antisymmetrize (ik)
    X["ab"]["vooo"] -= np.transpose(X["ab"]["vooo"], (0, 1, 3, 2))

    # [4] x(ac~ek)
    X["ab"]["vvvo"] = (
        -np.einsum("me,acmk->acek", H.a.ov, R.ab, optimize=True)
        + np.einsum("acef,fk->acek", H.ab.vvvv, R.b, optimize=True)
        + np.einsum("amke,cm->acek", H.aa.voov, R.b, optimize=True) # flip sign h2a(amek) -> h2a(amke)
        + np.einsum("anef,fcnk->acek", H.aa.vovv, R.ab, optimize=True)
        + np.einsum("anef,fcnk->acek", H.ab.vovv, R.bb, optimize=True)
        - 0.5 * np.einsum("mnef,afcmnk->acek", H.aa.oovv, R.aab, optimize=True)
        - np.einsum("mnef,afcmnk->acek", H.ab.oovv, R.abb, optimize=True)
        #
        #+ np.einsum("me,ackm->acek", x_ov, T.ab, optimize=True) # (!) flip sign to rearrange path k -> c~, e -> a ## block
        + 0.5 * np.einsum("mnke,acnm->acek", H.aa.ooov, R.ab, optimize=True)
    )

    # [3] x(m~c~j~k)
    X["bb"]["ovoo"] = (
        np.einsum("me,ecjk->mcjk", H.b.ov, R.bb, optimize=True)
        - np.einsum("cmje,ek->mcjk", H.bb.voov, R.b, optimize=True) # flip sign h2c(mcje) -> -h2c(cmje)
        - np.einsum("nmkj,cn->mcjk", H.ab.oooo, R.b, optimize=True)
        + np.einsum("nmfj,fcnk->mcjk", H.ab.oovo, R.ab, optimize=True)
        + np.einsum("mnjf,fcnk->mcjk", H.bb.ooov, R.bb, optimize=True)
        + np.einsum("nmfe,fecnjk->mcjk", H.ab.oovv, R.abb, optimize=True)
        + 0.5 * np.einsum("nmfe,fecnjk->mcjk", H.bb.oovv, R.bbb, optimize=True)
        #
        - np.einsum("me,eckj->mcjk ", x_ov, T.ab, optimize=True) # (!) flip sign to rearrange path k -> c~, j~ -> m~
        + 0.5 * np.einsum("cmfe,efjk->mcjk", H.bb.vovv, R.bb, optimize=True)
    )

    # [5] x(b~c~e~k)
    X["bb"]["vvvo"] = (
        -0.5 * np.einsum("me,bcmk->bcek", H.b.ov, R.bb, optimize=True)
        - np.einsum("mbke,cm->bcek", H.ab.ovov, R.b, optimize=True)
        + 0.5 * np.einsum("bcef,fk->bcek", H.bb.vvvv, R.b, optimize=True)
        + np.einsum("nbfe,fcnk->bcek", H.ab.ovvv, R.ab, optimize=True)
        + np.einsum("bnef,fcnk->bcek", H.bb.vovv, R.bb, optimize=True)
        - 0.5 * np.einsum("nmfe,fbcnmk->bcek", H.ab.oovv, R.abb, optimize=True)
        - 0.25 * np.einsum("nmfe,fbcnmk->bcek", H.bb.oovv, R.bbb, optimize=True)
    )
    # antisymmetrize (bc)
    X["bb"]["vvvo"] -= np.transpose(X["bb"]["vvvo"], (1, 0, 2, 3))

    # [6] x(b~c~j~e)
    X["bb"]["vvov"] = (
        -0.5 * np.einsum("me,bcjm->bcje", H.a.ov, R.bb, optimize=True)
        #- 0.5 * np.einsum("me,bcjm->bcje", x_ov, T.bb, optimize=True) # (!) ## block
        - np.einsum("mbej,cm->bcje", H.ab.ovvo, R.b, optimize=True)
        - np.einsum("mbef,fcjm->bcje", H.ab.ovvv, R.bb, optimize=True)
        + 0.5 * np.einsum("mnej,bcnm->bcje", H.ab.oovo, R.bb, optimize=True)
        - 0.25 * np.einsum("mnef,fbcnjm->bcje", H.aa.oovv, R.abb, optimize=True)
        - 0.5 * np.einsum("mnef,fbcnjm->bcje", H.ab.oovv, R.bbb, optimize=True)
    )
    # antisymmetrize (bc)
    X["bb"]["vvov"] -= np.transpose(X["bb"]["vvov"], (1, 0, 2, 3))

    return X