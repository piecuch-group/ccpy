import numpy as np

from ccpy.models.integrals import Integral

def get_eomccsd_intermediates(H, R, system):
    """Calculate the H*(R1+R2) intermediates for EOMCCSD."""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 1, data_type=H.a.oo.dtype)

    X.a.ov = (
        np.einsum("mnef,fn->me", H.aa.oovv, R.a, optimize=True)
        + np.einsum("mnef,fn->me", H.ab.oovv, R.b, optimize=True)
    )

    X.b.ov = (
        np.einsum("nmfe,fn->me", H.ab.oovv, R.a, optimize=True)
        + np.einsum("nmfe,fn->me", H.bb.oovv, R.b, optimize=True)
    )

    X.a.oo = (
            + np.einsum("mnjf,fn->mj", H.aa.ooov, R.a, optimize=True)
            + np.einsum("mnjf,fn->mj", H.ab.ooov, R.b, optimize=True)
            + 0.5 * np.einsum("mnef,efjn->mj", H.aa.oovv, R.aa, optimize=True)
            + np.einsum("mnef,efjn->mj", H.ab.oovv, R.ab, optimize=True)
    )

    X.a.vv = (
            + np.einsum("bnef,fn->be", H.aa.vovv, R.a, optimize=True)
            + np.einsum("bnef,fn->be", H.ab.vovv, R.b, optimize=True)
            - 0.5 * np.einsum("mnef,bfmn->be", H.aa.oovv, R.aa, optimize=True)
            - np.einsum("mnef,bfmn->be", H.ab.oovv, R.ab, optimize=True)
    )

    X.b.oo = (
            + np.einsum("nmfk,fn->mk", H.ab.oovo, R.a, optimize=True)
            + np.einsum("mnkf,fn->mk", H.bb.ooov, R.b, optimize=True)
            + np.einsum("nmfe,fenk->mk", H.ab.oovv, R.ab, optimize=True)
            + 0.5 * np.einsum("mnef,efkn->mk", H.bb.oovv, R.bb, optimize=True)
    )

    X.b.vv = (
            + np.einsum("ncfe,fn->ce", H.ab.ovvv, R.a, optimize=True)
            + np.einsum("cnef,fn->ce", H.bb.vovv, R.b, optimize=True)
            -1.0 * np.einsum("nmfe,fcnm->ce", H.ab.oovv, R.ab, optimize=True)
            - 0.5 * np.einsum("mnef,fcnm->ce", H.bb.oovv, R.bb, optimize=True)
    )
    return X


def get_eomccsdt1_intermediates(H, R, T, system):
    """Calculate the CCSD-like intermediates for CCSDT. This routine
    should only calculate terms with T2 and any remaining terms outside of the CCS intermediate
    routine."""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 2, data_type=H.a.oo.dtype)

    X.a.ov = (
        np.einsum("mnef,fn->me", H.aa.oovv, R.a, optimize=True)
        + np.einsum("mnef,fn->me", H.ab.oovv, R.b, optimize=True)
    )
    #X.a.ov = X_eomccsd.a.ov

    X.b.ov = (
        np.einsum("nmfe,fn->me", H.ab.oovv, R.a, optimize=True)
        + np.einsum("nmfe,fn->me", H.bb.oovv, R.b, optimize=True)
    )
    #X.b.ov = X_eomccsd.b.ov

    X.a.oo = (
            np.einsum("me,ej->mj", H.a.ov, R.a, optimize=True)
            #+ X_eomccsd.a.oo
            + np.einsum("mnjf,fn->mj", H.aa.ooov, R.a, optimize=True)
            + np.einsum("mnjf,fn->mj", H.ab.ooov, R.b, optimize=True)
            + 0.5 * np.einsum("mnef,efjn->mj", H.aa.oovv, R.aa, optimize=True)
            + np.einsum("mnef,efjn->mj", H.ab.oovv, R.ab, optimize=True)
    )

    X.a.vv = (
            -1.0 * np.einsum("me,bm->be", H.a.ov, R.a, optimize=True)
            #+ X_eomccsd.a.vv
            + np.einsum("bnef,fn->be", H.aa.vovv, R.a, optimize=True)
            + np.einsum("bnef,fn->be", H.ab.vovv, R.b, optimize=True)
            - 0.5 * np.einsum("mnef,bfmn->be", H.aa.oovv, R.aa, optimize=True)
            - np.einsum("mnef,bfmn->be", H.ab.oovv, R.ab, optimize=True)
    )

    X.b.oo = (
            np.einsum("me,ek->mk", H.b.ov, R.b, optimize=True)
            #+ X_eomccsd.b.oo
            + np.einsum("nmfk,fn->mk", H.ab.oovo, R.a, optimize=True)
            + np.einsum("mnkf,fn->mk", H.bb.ooov, R.b, optimize=True)
            + np.einsum("nmfe,fenk->mk", H.ab.oovv, R.ab, optimize=True)
            + 0.5 * np.einsum("mnef,efkn->mk", H.bb.oovv, R.bb, optimize=True)
    )

    X.b.vv = (
            -1.0 * np.einsum("me,cm->ce", H.b.ov, R.b, optimize=True)
            #+ X_eomccsd.b.vv
            + np.einsum("ncfe,fn->ce", H.ab.ovvv, R.a, optimize=True)
            + np.einsum("cnef,fn->ce", H.bb.vovv, R.b, optimize=True)
            -1.0 * np.einsum("nmfe,fcnm->ce", H.ab.oovv, R.ab, optimize=True)
            - 0.5 * np.einsum("mnef,fcnm->ce", H.bb.oovv, R.bb, optimize=True)
    )

    X.aa.oooo = (
        np.einsum("nmje,ei->mnij", H.aa.ooov, R.a, optimize=True)
        + 0.25 * np.einsum("mnef,efij->mnij", H.aa.oovv, R.aa, optimize=True)
    )
    X.aa.oooo -= np.transpose(X.aa.oooo, (0, 1, 3, 2))

    X.ab.oooo = (
        np.einsum("nmje,ek->nmjk", H.ab.ooov, R.b, optimize=True)
        + np.einsum("nmek,ej->nmjk", H.ab.oovo, R.a, optimize=True)
        + np.einsum("mnef,efjk->mnjk", H.ab.oovv, R.ab, optimize=True)
    )

    X.bb.oooo = (
        np.einsum("mnie,ej->mnij", H.bb.ooov, R.b, optimize=True)
        + 0.25 * np.einsum("mnef,efij->mnij", H.bb.oovv, R.bb, optimize=True)
    )
    X.bb.oooo -= np.transpose(X.bb.oooo, (0, 1, 3, 2))

    X.aa.vvvv = (
        -1.0 * np.einsum("amef,bm->abef", H.aa.vovv, R.a, optimize=True)
        + 0.25 * np.einsum("mnef,abmn->abef", H.aa.oovv, R.aa, optimize=True)
    )
    X.aa.vvvv -= np.transpose(X.aa.vvvv, (1, 0, 2, 3))

    X.ab.vvvv = (
        - np.einsum("bmfe,cm->bcfe", H.ab.vovv, R.b, optimize=True)
        - np.einsum("mcfe,bm->bcfe", H.ab.ovvv, R.a, optimize=True)
        + np.einsum("mnef,bcmn->bcef", H.ab.oovv, R.ab, optimize=True)
    )

    X.bb.vvvv = (
        - np.einsum("amef,bm->abef", H.bb.vovv, R.b, optimize=True)
        + 0.25 * np.einsum("mnef,abmn->abef", H.bb.oovv, R.bb, optimize=True)
    )
    X.bb.vvvv -= np.transpose(X.bb.vvvv, (1, 0, 2, 3))

    X.aa.voov = (
        -1.0 * np.einsum("nmje,bn->bmje", H.aa.ooov, R.a, optimize=True)
        + np.einsum("bmfe,fj->bmje", H.aa.vovv, R.a, optimize=True)
        + np.einsum("mnef,fcnk->cmke", H.aa.oovv, R.aa, optimize=True)
        + np.einsum("mnef,cfkn->cmke", H.ab.oovv, R.ab, optimize=True)
    )

    X.ab.voov = (
        -1.0 * np.einsum("nmje,bn->bmje", H.ab.ooov, R.a, optimize=True)
        + np.einsum("bmfe,fj->bmje", H.ab.vovv, R.a, optimize=True)
        + np.einsum("nmfe,fcnk->cmke", H.ab.oovv, R.aa, optimize=True)
        + np.einsum("mnef,cfkn->cmke", H.bb.oovv, R.ab, optimize=True)
    )

    X.ab.ovvo = (
        - np.einsum("nmfk,cm->ncfk", H.ab.oovo, R.b, optimize=True)
        + np.einsum("ncfe,ek->ncfk", H.ab.ovvv, R.b, optimize=True)
        + np.einsum("mnef,ecmk->ncfk", H.aa.oovv, R.ab, optimize=True)
        + np.einsum("nmfe,ecmk->ncfk", H.ab.oovv, R.bb, optimize=True)
    )

    X.ab.vovo = (
        np.einsum("bmfe,ek->bmfk", H.ab.vovv, R.b, optimize=True)
        - np.einsum("nmfk,bn->bmfk", H.ab.oovo, R.a, optimize=True)
        - np.einsum("mnef,bfmk->bnek", H.ab.oovv, R.ab, optimize=True)
    )

    X.ab.ovov = (
        - np.einsum("nmje,cm->ncje", H.ab.ooov, R.b, optimize=True)
        + np.einsum("ncfe,fj->ncje", H.ab.ovvv, R.a, optimize=True)
        - np.einsum("mnef,ecjn->mcjf", H.ab.oovv, R.ab, optimize=True)
    )

    X.bb.voov = (
            - np.einsum("mnkf,cm->cnkf", H.bb.ooov, R.b, optimize=True)
            + np.einsum("cnef,ek->cnkf", H.bb.vovv, R.b, optimize=True)
            + np.einsum("mnef,ecmk->cnkf", H.ab.oovv, R.ab, optimize=True)
            + np.einsum("mnef,ecmk->cnkf", H.bb.oovv, R.bb, optimize=True)
    )

    X.aa.vvov =(
        np.einsum("amje,bm->baje", H.aa.voov, R.a, optimize=True)
        + np.einsum("amfe,bejm->bajf", H.aa.vovv, R.aa, optimize=True)
        + np.einsum("amfe,bejm->bajf", H.ab.vovv, R.ab, optimize=True)
        + 0.5 * np.einsum("abfe,ej->bajf", H.aa.vvvv, R.a, optimize=True)
        + 0.25 * np.einsum("nmje,abmn->baje", H.aa.ooov, R.aa, optimize=True)
        - 0.5 * np.einsum("me,abmj->baje", X.a.ov, T.aa, optimize=True) # counterterm, similar to CR-CC(2,3)
    )
    X.aa.vvov -= np.transpose(X.aa.vvov, (1, 0, 2, 3))

    X.aa.vooo = (
        -np.einsum("bmie,ej->bmji", H.aa.voov, R.a, optimize=True)
        +np.einsum("nmie,bejm->bnji", H.aa.ooov, R.aa, optimize=True)
        +np.einsum("nmie,bejm->bnji", H.ab.ooov, R.ab, optimize=True)
        - 0.5 * np.einsum("nmij,bm->bnji", H.aa.oooo, R.a, optimize=True)
        + 0.25 * np.einsum("bmfe,efij->bmji", H.aa.vovv, R.aa, optimize=True)
    )
    X.aa.vooo -= np.transpose(X.aa.vooo, (0, 1, 3, 2))

    X.ab.vvvo = (
        - np.einsum("mcek,bm->bcek", H.ab.ovvo, R.a, optimize=True)
        - np.einsum("bmek,cm->bcek", H.ab.vovo, R.b, optimize=True)
        + np.einsum("bcfe,ek->bcfk", H.ab.vvvv, R.b, optimize=True)
        + np.einsum("mnek,bcmn->bcek", H.ab.oovo, R.ab, optimize=True)
        + np.einsum("bmfe,ecmk->bcfk", H.aa.vovv, R.ab, optimize=True)
        + np.einsum("bmfe,ecmk->bcfk", H.ab.vovv, R.bb, optimize=True)
        - np.einsum("mcfe,bemk->bcfk", H.ab.ovvv, R.ab, optimize=True)
        - np.einsum("me,bcmk->bcek", X.a.ov, T.ab, optimize=True) # counterterm, similar to CR-CC(2,3)
    )

    X.ab.ovoo = (
        - np.einsum("nmjk,cm->ncjk", H.ab.oooo, R.b, optimize=True)
        + np.einsum("mcje,ek->mcjk", H.ab.ovov, R.b, optimize=True)
        + np.einsum("mcek,ej->mcjk", H.ab.ovvo, R.a, optimize=True)
        + np.einsum("mcef,efjk->mcjk", H.ab.ovvv, R.ab, optimize=True)
        + np.einsum("nmje,ecmk->ncjk", H.aa.ooov, R.ab, optimize=True)
        + np.einsum("nmje,ecmk->ncjk", H.ab.ooov, R.bb, optimize=True)
        - np.einsum("nmek,ecjm->ncjk", H.ab.oovo, R.ab, optimize=True)
    )

    X.ab.vvov = (
        - np.einsum("mcje,bm->bcje", H.ab.ovov, R.a, optimize=True)
        - np.einsum("bmje,cm->bcje", H.ab.voov, R.b, optimize=True)
        + np.einsum("bcef,ej->bcjf", H.ab.vvvv, R.a, optimize=True)
        + np.einsum("mnjf,bcmn->bcjf", H.ab.ooov, R.ab, optimize=True)
        + np.einsum("mcef,bejm->bcjf", H.ab.ovvv, R.aa, optimize=True)
        + np.einsum("cmfe,bejm->bcjf", H.bb.vovv, R.ab, optimize=True)
        - np.einsum("bmef,ecjm->bcjf", H.ab.vovv, R.ab, optimize=True)
        - np.einsum("me,bcjm->bcje", X.b.ov, T.ab, optimize=True) # counterterm, similar to CR-CC(2,3)
    )

    X.ab.vooo = (
        - np.einsum("mnjk,bm->bnjk", H.ab.oooo, R.a, optimize=True)
        + np.einsum("bmje,ek->bmjk", H.ab.voov, R.b, optimize=True)
        + np.einsum("bmek,ej->bmjk", H.ab.vovo, R.a, optimize=True)
        + np.einsum("bnef,efjk->bnjk", H.ab.vovv, R.ab, optimize=True)
        + np.einsum("mnek,bejm->bnjk", H.ab.oovo, R.aa, optimize=True)
        + np.einsum("nmke,bejm->bnjk", H.bb.ooov, R.ab, optimize=True)
        - np.einsum("nmje,benk->bmjk", H.ab.ooov, R.ab, optimize=True)
    )

    X.bb.vvov = (
        np.einsum("amje,bm->baje", H.bb.voov, R.b, optimize=True)
        + 0.5 * np.einsum("abfe,ej->bajf", H.bb.vvvv, R.b, optimize=True)
        + 0.25 * np.einsum("nmje,abmn->baje", H.bb.ooov, R.bb, optimize=True)
        + np.einsum("amfe,bejm->bajf", H.bb.vovv, R.bb, optimize=True)
        + np.einsum("maef,ebmj->bajf", H.ab.ovvv, R.ab, optimize=True)
        - 0.5 * np.einsum("me,abmj->baje", X.b.ov, T.bb, optimize=True) # counterterm, similar to CR-CC(2,3)
    )
    X.bb.vvov -= np.transpose(X.bb.vvov, (1, 0, 2, 3))

    X.bb.vooo = (
        -0.5 * np.einsum("nmij,bm->bnji", H.bb.oooo, R.b, optimize=True)
        - np.einsum("bmie,ej->bmji", H.bb.voov, R.b, optimize=True)
        + 0.25 * np.einsum("bmfe,efij->bmji", H.bb.vovv, R.bb, optimize=True)
        + np.einsum("nmie,bejm->bnji", H.bb.ooov, R.bb, optimize=True)
        + np.einsum("mnei,ebmj->bnji", H.ab.oovo, R.ab, optimize=True)
    )
    X.bb.vooo -= np.transpose(X.bb.vooo, (0, 1, 3, 2))



    return X

def add_R3_terms(X, H, R):

    X.aa.vvov += (
            -0.5 * np.einsum("mnef,abfimn->abie", H.aa.oovv, R.aaa, optimize=True)
            - np.einsum("mnef,abfimn->abie", H.ab.oovv, R.aab, optimize=True)
    )

    X.aa.vooo += (
            0.5 * np.einsum("mnef,efcjnk->cmkj", H.aa.oovv, R.aaa, optimize=True)
            + np.einsum("mnef,ecfjkn->cmkj", H.ab.oovv, R.aab, optimize=True)
    )

    X.ab.vvvo += (
            -0.5 * np.einsum("mnef,bfcmnk->bcek", H.aa.oovv, R.aab, optimize=True)
            - np.einsum("mnef,bfcmnk->bcek", H.ab.oovv, R.abb, optimize=True)
    )

    X.ab.ovoo += (
            0.5 * np.einsum("mnef,efcjnk->mcjk", H.aa.oovv, R.aab, optimize=True)
            + np.einsum("mnef,efcjnk->mcjk", H.ab.oovv, R.abb, optimize=True)
    )

    X.ab.vvov += (
            -np.einsum("nmfe,bfcjnm->bcje", H.ab.oovv, R.aab, optimize=True)
            - 0.5 * np.einsum("mnef,bfcjnm->bcje", H.bb.oovv, R.abb, optimize=True)
    )

    X.ab.vooo += (
            np.einsum("nmfe,bfejnk->bmjk", H.ab.oovv, R.aab, optimize=True)
            + 0.5 * np.einsum("mnef,befjkn->bmjk", H.bb.oovv, R.abb, optimize=True)
    )

    X.bb.vvov += (
            -0.5 * np.einsum("mnef,abfmjn->baje", H.bb.oovv, R.bbb, optimize=True)
            - np.einsum("nmfe,fbanjm->baje", H.ab.oovv, R.abb, optimize=True)
    )

    X.bb.vooo += (
            0.5 * np.einsum("mnef,aefijn->amij", H.bb.oovv, R.bbb, optimize=True)
            + np.einsum("nmfe,feanji->amij", H.ab.oovv, R.abb, optimize=True)
    )

    return X