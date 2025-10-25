import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
import time
from ccpy.models.integrals import Integral
from ccpy.lib.core import hbar_ccsdt_p

def get_eomccsd_intermediates(H, R, system):
    """Calculate the H*(R1+R2) intermediates for EOMCCSD."""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 1, data_type=H.a.oo.dtype)

    X.a.ov = (
        ccpy_einsum("mnef,fn->me", H.aa.oovv, R.a)
        + ccpy_einsum("mnef,fn->me", H.ab.oovv, R.b)
    )

    X.b.ov = (
        ccpy_einsum("nmfe,fn->me", H.ab.oovv, R.a)
        + ccpy_einsum("nmfe,fn->me", H.bb.oovv, R.b)
    )

    X.a.oo = (
            + ccpy_einsum("mnjf,fn->mj", H.aa.ooov, R.a)
            + ccpy_einsum("mnjf,fn->mj", H.ab.ooov, R.b)
            + 0.5 * ccpy_einsum("mnef,efjn->mj", H.aa.oovv, R.aa)
            + ccpy_einsum("mnef,efjn->mj", H.ab.oovv, R.ab)
    )

    X.a.vv = (
            + ccpy_einsum("bnef,fn->be", H.aa.vovv, R.a)
            + ccpy_einsum("bnef,fn->be", H.ab.vovv, R.b)
            - 0.5 * ccpy_einsum("mnef,bfmn->be", H.aa.oovv, R.aa)
            - ccpy_einsum("mnef,bfmn->be", H.ab.oovv, R.ab)
    )

    X.b.oo = (
            + ccpy_einsum("nmfk,fn->mk", H.ab.oovo, R.a)
            + ccpy_einsum("mnkf,fn->mk", H.bb.ooov, R.b)
            + ccpy_einsum("nmfe,fenk->mk", H.ab.oovv, R.ab)
            + 0.5 * ccpy_einsum("mnef,efkn->mk", H.bb.oovv, R.bb)
    )

    X.b.vv = (
            + ccpy_einsum("ncfe,fn->ce", H.ab.ovvv, R.a)
            + ccpy_einsum("cnef,fn->ce", H.bb.vovv, R.b)
            -1.0 * ccpy_einsum("nmfe,fcnm->ce", H.ab.oovv, R.ab)
            - 0.5 * ccpy_einsum("mnef,fcnm->ce", H.bb.oovv, R.bb)
    )
    return X


def get_eomccsdt_intermediates(H, R, T, X_eomccsd, system):
    """Calculate the CCSD-like intermediates for CCSDT. This routine
    should only calculate terms with T2 and any remaining terms outside of the CCS intermediate
    routine."""

    # Create new 2-body integral object
    X = Integral.from_empty(system, 2, data_type=H.a.oo.dtype, use_none=True)

    # these should be removed
    X.aa.oovv = np.zeros_like(H.aa.oovv)
    X.ab.oovv = np.zeros_like(H.ab.oovv)
    X.bb.oovv = np.zeros_like(H.bb.oovv)

    # X.a.ov = (
    #     ccpy_einsum("mnef,fn->me", H.aa.oovv, R.a)
    #     + ccpy_einsum("mnef,fn->me", H.ab.oovv, R.b)
    # )
    X.a.ov = X_eomccsd.a.ov

    # X.b.ov = (
    #     ccpy_einsum("nmfe,fn->me", H.ab.oovv, R.a)
    #     + ccpy_einsum("nmfe,fn->me", H.bb.oovv, R.b)
    # )
    X.b.ov = X_eomccsd.b.ov
    
    #tic = time.time()
    X.a.oo = (
            ccpy_einsum("me,ej->mj", H.a.ov, R.a)
            + X_eomccsd.a.oo
            # + ccpy_einsum("mnjf,fn->mj", H.aa.ooov, R.a)
            # + ccpy_einsum("mnjf,fn->mj", H.ab.ooov, R.b)
            # + 0.5 * ccpy_einsum("mnef,efjn->mj", H.aa.oovv, R.aa)
            # + ccpy_einsum("mnef,efjn->mj", H.ab.oovv, R.ab)
    )
    X.b.oo = (
            ccpy_einsum("me,ek->mk", H.b.ov, R.b)
            + X_eomccsd.b.oo
            # + ccpy_einsum("nmfk,fn->mk", H.ab.oovo, R.a)
            # + ccpy_einsum("mnkf,fn->mk", H.bb.ooov, R.b)
            # + ccpy_einsum("nmfe,fenk->mk", H.ab.oovv, R.ab)
            # + 0.5 * ccpy_einsum("mnef,efkn->mk", H.bb.oovv, R.bb)
    )
    #toc = time.time()
    #print("time for oo = ", tic - toc, "s")

    #tic = time.time()
    X.a.vv = (
            -1.0 * ccpy_einsum("me,bm->be", H.a.ov, R.a)
            + X_eomccsd.a.vv
            # + ccpy_einsum("bnef,fn->be", H.aa.vovv, R.a)
            # + ccpy_einsum("bnef,fn->be", H.ab.vovv, R.b)
            # - 0.5 * ccpy_einsum("mnef,bfmn->be", H.aa.oovv, R.aa)
            # - ccpy_einsum("mnef,bfmn->be", H.ab.oovv, R.ab)
    )
    X.b.vv = (
            -1.0 * ccpy_einsum("me,cm->ce", H.b.ov, R.b)
            + X_eomccsd.b.vv
            # + ccpy_einsum("ncfe,fn->ce", H.ab.ovvv, R.a)
            # + ccpy_einsum("cnef,fn->ce", H.bb.vovv, R.b)
            # -1.0 * ccpy_einsum("nmfe,fcnm->ce", H.ab.oovv, R.ab)
            # - 0.5 * ccpy_einsum("mnef,fcnm->ce", H.bb.oovv, R.bb)
    )
    #toc = time.time()
    #print("time for vv = ", tic - toc, "s")

    #tic = time.time()
    X.aa.oooo = (
        ccpy_einsum("nmje,ei->mnij", H.aa.ooov, R.a)
        + 0.25 * ccpy_einsum("mnef,efij->mnij", H.aa.oovv, R.aa)
    )
    X.aa.oooo -= np.transpose(X.aa.oooo, (0, 1, 3, 2))

    X.ab.oooo = (
        ccpy_einsum("nmje,ek->nmjk", H.ab.ooov, R.b)
        + ccpy_einsum("nmek,ej->nmjk", H.ab.oovo, R.a)
        + ccpy_einsum("mnef,efjk->mnjk", H.ab.oovv, R.ab)
    )

    X.bb.oooo = (
        ccpy_einsum("mnie,ej->mnij", H.bb.ooov, R.b)
        + 0.25 * ccpy_einsum("mnef,efij->mnij", H.bb.oovv, R.bb)
    )
    X.bb.oooo -= np.transpose(X.bb.oooo, (0, 1, 3, 2))
    #toc = time.time()
    #print("time for oooo = ", tic - toc, "s")

    #tic = time.time()
    X.aa.vvvv = (
        -1.0 * ccpy_einsum("amef,bm->abef", H.aa.vovv, R.a)
        + 0.25 * ccpy_einsum("mnef,abmn->abef", H.aa.oovv, R.aa)
    )
    X.aa.vvvv -= np.transpose(X.aa.vvvv, (1, 0, 2, 3))

    X.ab.vvvv = (
        - ccpy_einsum("bmfe,cm->bcfe", H.ab.vovv, R.b)
        - ccpy_einsum("mcfe,bm->bcfe", H.ab.ovvv, R.a)
        + ccpy_einsum("mnef,bcmn->bcef", H.ab.oovv, R.ab)
    )

    X.bb.vvvv = (
        - ccpy_einsum("amef,bm->abef", H.bb.vovv, R.b)
        + 0.25 * ccpy_einsum("mnef,abmn->abef", H.bb.oovv, R.bb)
    )
    X.bb.vvvv -= np.transpose(X.bb.vvvv, (1, 0, 2, 3))
    #toc = time.time()
    #print("time for vvvv = ", tic - toc, "s")

    #tic = time.time()
    X.aa.voov = (
        -1.0 * ccpy_einsum("nmje,bn->bmje", H.aa.ooov, R.a)
        + ccpy_einsum("bmfe,fj->bmje", H.aa.vovv, R.a)
        + ccpy_einsum("mnef,fcnk->cmke", H.aa.oovv, R.aa)
        + ccpy_einsum("mnef,cfkn->cmke", H.ab.oovv, R.ab)
    )

    X.ab.voov = (
        -1.0 * ccpy_einsum("nmje,bn->bmje", H.ab.ooov, R.a)
        + ccpy_einsum("bmfe,fj->bmje", H.ab.vovv, R.a)
        + ccpy_einsum("nmfe,fcnk->cmke", H.ab.oovv, R.aa)
        + ccpy_einsum("mnef,cfkn->cmke", H.bb.oovv, R.ab)
    )

    X.ab.ovvo = (
        - ccpy_einsum("nmfk,cm->ncfk", H.ab.oovo, R.b)
        + ccpy_einsum("ncfe,ek->ncfk", H.ab.ovvv, R.b)
        + ccpy_einsum("mnef,ecmk->ncfk", H.aa.oovv, R.ab)
        + ccpy_einsum("nmfe,ecmk->ncfk", H.ab.oovv, R.bb)
    )

    X.ab.vovo = (
        ccpy_einsum("bmfe,ek->bmfk", H.ab.vovv, R.b)
        - ccpy_einsum("nmfk,bn->bmfk", H.ab.oovo, R.a)
        - ccpy_einsum("mnef,bfmk->bnek", H.ab.oovv, R.ab)
    )

    X.ab.ovov = (
        - ccpy_einsum("nmje,cm->ncje", H.ab.ooov, R.b)
        + ccpy_einsum("ncfe,fj->ncje", H.ab.ovvv, R.a)
        - ccpy_einsum("mnef,ecjn->mcjf", H.ab.oovv, R.ab)
    )

    X.bb.voov = (
            - ccpy_einsum("mnkf,cm->cnkf", H.bb.ooov, R.b)
            + ccpy_einsum("cnef,ek->cnkf", H.bb.vovv, R.b)
            + ccpy_einsum("mnef,ecmk->cnkf", H.ab.oovv, R.ab)
            + ccpy_einsum("mnef,ecmk->cnkf", H.bb.oovv, R.bb)
    )
    #toc = time.time()
    #print("time for voov = ", tic - toc, "s")

    X.aa.vvov =(
        ccpy_einsum("amje,bm->baje", H.aa.voov, R.a)
        + ccpy_einsum("amfe,bejm->bajf", H.aa.vovv, R.aa)
        + ccpy_einsum("amfe,bejm->bajf", H.ab.vovv, R.ab)
        + 0.5 * ccpy_einsum("abfe,ej->bajf", H.aa.vvvv, R.a)
        + 0.25 * ccpy_einsum("nmje,abmn->baje", H.aa.ooov, R.aa)
        - 0.5 * ccpy_einsum("me,abmj->baje", X.a.ov, T.aa) # counterterm, similar to CR-CC(2,3)
    )
    X.aa.vvov -= np.transpose(X.aa.vvov, (1, 0, 2, 3))

    X.aa.vooo = (
        -ccpy_einsum("bmie,ej->bmji", H.aa.voov, R.a)
        +ccpy_einsum("nmie,bejm->bnji", H.aa.ooov, R.aa)
        +ccpy_einsum("nmie,bejm->bnji", H.ab.ooov, R.ab)
        - 0.5 * ccpy_einsum("nmij,bm->bnji", H.aa.oooo, R.a)
        + 0.25 * ccpy_einsum("bmfe,efij->bmji", H.aa.vovv, R.aa)
    )
    X.aa.vooo -= np.transpose(X.aa.vooo, (0, 1, 3, 2))

    X.ab.vvvo = (
        - ccpy_einsum("mcek,bm->bcek", H.ab.ovvo, R.a)
        - ccpy_einsum("bmek,cm->bcek", H.ab.vovo, R.b)
        + ccpy_einsum("bcfe,ek->bcfk", H.ab.vvvv, R.b)
        + ccpy_einsum("mnek,bcmn->bcek", H.ab.oovo, R.ab)
        + ccpy_einsum("bmfe,ecmk->bcfk", H.aa.vovv, R.ab)
        + ccpy_einsum("bmfe,ecmk->bcfk", H.ab.vovv, R.bb)
        - ccpy_einsum("mcfe,bemk->bcfk", H.ab.ovvv, R.ab)
        - ccpy_einsum("me,bcmk->bcek", X.a.ov, T.ab) # counterterm, similar to CR-CC(2,3)
    )

    X.ab.ovoo = (
        - ccpy_einsum("nmjk,cm->ncjk", H.ab.oooo, R.b)
        + ccpy_einsum("mcje,ek->mcjk", H.ab.ovov, R.b)
        + ccpy_einsum("mcek,ej->mcjk", H.ab.ovvo, R.a)
        + ccpy_einsum("mcef,efjk->mcjk", H.ab.ovvv, R.ab)
        + ccpy_einsum("nmje,ecmk->ncjk", H.aa.ooov, R.ab)
        + ccpy_einsum("nmje,ecmk->ncjk", H.ab.ooov, R.bb)
        - ccpy_einsum("nmek,ecjm->ncjk", H.ab.oovo, R.ab)
    )

    X.ab.vvov = (
        - ccpy_einsum("mcje,bm->bcje", H.ab.ovov, R.a)
        - ccpy_einsum("bmje,cm->bcje", H.ab.voov, R.b)
        + ccpy_einsum("bcef,ej->bcjf", H.ab.vvvv, R.a)
        + ccpy_einsum("mnjf,bcmn->bcjf", H.ab.ooov, R.ab)
        + ccpy_einsum("mcef,bejm->bcjf", H.ab.ovvv, R.aa)
        + ccpy_einsum("cmfe,bejm->bcjf", H.bb.vovv, R.ab)
        - ccpy_einsum("bmef,ecjm->bcjf", H.ab.vovv, R.ab)
        - ccpy_einsum("me,bcjm->bcje", X.b.ov, T.ab) # counterterm, similar to CR-CC(2,3)
    )

    X.ab.vooo = (
        - ccpy_einsum("mnjk,bm->bnjk", H.ab.oooo, R.a)
        + ccpy_einsum("bmje,ek->bmjk", H.ab.voov, R.b)
        + ccpy_einsum("bmek,ej->bmjk", H.ab.vovo, R.a)
        + ccpy_einsum("bnef,efjk->bnjk", H.ab.vovv, R.ab)
        + ccpy_einsum("mnek,bejm->bnjk", H.ab.oovo, R.aa)
        + ccpy_einsum("nmke,bejm->bnjk", H.bb.ooov, R.ab)
        - ccpy_einsum("nmje,benk->bmjk", H.ab.ooov, R.ab)
    )

    X.bb.vvov = (
        ccpy_einsum("amje,bm->baje", H.bb.voov, R.b)
        + 0.5 * ccpy_einsum("abfe,ej->bajf", H.bb.vvvv, R.b)
        + 0.25 * ccpy_einsum("nmje,abmn->baje", H.bb.ooov, R.bb)
        + ccpy_einsum("amfe,bejm->bajf", H.bb.vovv, R.bb)
        + ccpy_einsum("maef,ebmj->bajf", H.ab.ovvv, R.ab)
        - 0.5 * ccpy_einsum("me,abmj->baje", X.b.ov, T.bb) # counterterm, similar to CR-CC(2,3)
    )
    X.bb.vvov -= np.transpose(X.bb.vvov, (1, 0, 2, 3))

    X.bb.vooo = (
        -0.5 * ccpy_einsum("nmij,bm->bnji", H.bb.oooo, R.b)
        - ccpy_einsum("bmie,ej->bmji", H.bb.voov, R.b)
        + 0.25 * ccpy_einsum("bmfe,efij->bmji", H.bb.vovv, R.bb)
        + ccpy_einsum("nmie,bejm->bnji", H.bb.ooov, R.bb)
        + ccpy_einsum("mnei,ebmj->bnji", H.ab.oovo, R.ab)
    )
    X.bb.vooo -= np.transpose(X.bb.vooo, (0, 1, 3, 2))
    return X

def add_R3_terms(X, H, R):

    X.aa.vvov += (
            -0.5 * ccpy_einsum("mnef,abfimn->abie", H.aa.oovv, R.aaa)
            - ccpy_einsum("mnef,abfimn->abie", H.ab.oovv, R.aab)
    )

    X.aa.vooo += (
            0.5 * ccpy_einsum("mnef,efcjnk->cmkj", H.aa.oovv, R.aaa)
            + ccpy_einsum("mnef,ecfjkn->cmkj", H.ab.oovv, R.aab)
    )

    X.ab.vvvo += (
            -0.5 * ccpy_einsum("mnef,bfcmnk->bcek", H.aa.oovv, R.aab)
            - ccpy_einsum("mnef,bfcmnk->bcek", H.ab.oovv, R.abb)
    )

    X.ab.ovoo += (
            0.5 * ccpy_einsum("mnef,efcjnk->mcjk", H.aa.oovv, R.aab)
            + ccpy_einsum("mnef,efcjnk->mcjk", H.ab.oovv, R.abb)
    )

    X.ab.vvov += (
            -ccpy_einsum("nmfe,bfcjnm->bcje", H.ab.oovv, R.aab)
            - 0.5 * ccpy_einsum("mnef,bfcjnm->bcje", H.bb.oovv, R.abb)
    )

    X.ab.vooo += (
            ccpy_einsum("nmfe,bfejnk->bmjk", H.ab.oovv, R.aab)
            + 0.5 * ccpy_einsum("mnef,befjkn->bmjk", H.bb.oovv, R.abb)
    )

    X.bb.vvov += (
            -0.5 * ccpy_einsum("mnef,abfmjn->baje", H.bb.oovv, R.bbb)
            - ccpy_einsum("nmfe,fbanjm->baje", H.ab.oovv, R.abb)
    )

    X.bb.vooo += (
            0.5 * ccpy_einsum("mnef,aefijn->amij", H.bb.oovv, R.bbb)
            + ccpy_einsum("nmfe,feanji->amij", H.ab.oovv, R.abb)
    )

    return X

def add_R3_p_terms(X, H, R, R3_excitations):

    X.aa.vooo = hbar_ccsdt_p.add_t3_h2a_vooo(X.aa.vooo,
                                                          R.aaa, R3_excitations["aaa"],
                                                          R.aab, R3_excitations["aab"],
                                                          H.aa.oovv, H.ab.oovv, phase=1.0,
    )
    X.aa.vvov = hbar_ccsdt_p.add_t3_h2a_vvov(X.aa.vvov,
                                                          R.aaa, R3_excitations["aaa"],
                                                          R.aab, R3_excitations["aab"],
                                                          H.aa.oovv, H.ab.oovv, phase=1.0,
    )
    X.ab.vooo = hbar_ccsdt_p.add_t3_h2b_vooo(X.ab.vooo,
                                                          R.aab, R3_excitations["aab"],
                                                          R.abb, R3_excitations["abb"],
                                                          H.ab.oovv, H.bb.oovv, phase=1.0,
    )
    X.ab.ovoo = hbar_ccsdt_p.add_t3_h2b_ovoo(X.ab.ovoo,
                                                          R.aab, R3_excitations["aab"],
                                                          R.abb, R3_excitations["abb"],
                                                          H.aa.oovv, H.ab.oovv, phase=1.0,
    )
    X.ab.vvov = hbar_ccsdt_p.add_t3_h2b_vvov(X.ab.vvov,
                                                          R.aab, R3_excitations["aab"],
                                                          R.abb, R3_excitations["abb"],
                                                          H.ab.oovv, H.bb.oovv, phase=1.0,
    )
    X.ab.vvvo = hbar_ccsdt_p.add_t3_h2b_vvvo(X.ab.vvvo,
                                                          R.aab, R3_excitations["aab"],
                                                          R.abb, R3_excitations["abb"],
                                                          H.aa.oovv, H.ab.oovv, phase=1.0,
    )
    X.bb.vooo = hbar_ccsdt_p.add_t3_h2c_vooo(X.bb.vooo,
                                                          R.abb, R3_excitations["abb"],
                                                          R.bbb, R3_excitations["bbb"],
                                                          H.ab.oovv, H.bb.oovv, phase=1.0,
    )
    X.bb.vvov = hbar_ccsdt_p.add_t3_h2c_vvov(X.bb.vvov,
                                                          R.abb, R3_excitations["abb"],
                                                          R.bbb, R3_excitations["bbb"],
                                                          H.ab.oovv, H.bb.oovv, phase=1.0,
    )
    return X
