import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import ipeom3_p_intermediates

def get_ipeom3_intermediates(H, R):

    # These intermediates will be 3-index quantities, which are not
    # set up in the models at the moment. We will just use a dictionary
    # as a workaround for now.

    X = {"aa" : {}, "ab" : {}}

    # x2a(ibe)
    X["aa"]["ovv"] = (
            -0.5 * ccpy_einsum("mnef,ibfmn->ibe", H.aa.oovv, R.aaa)
            -ccpy_einsum("mnef,ibfmn->ibe", H.ab.oovv, R.aab)
            +ccpy_einsum("bnef,ifn->ibe", H.aa.vovv, R.aa)
            +ccpy_einsum("bnef,ifn->ibe", H.ab.vovv, R.ab)
            +0.5 * ccpy_einsum("nmie,nbm->ibe", H.aa.ooov, R.aa)
            +ccpy_einsum("bmie,m->ibe", H.aa.voov, R.a)
    )
    # x2b(eb~j~)
    X["ab"]["vvo"] = (
            -0.5 * ccpy_einsum("mnef,mfbnj->ebj", H.aa.oovv, R.aab)
            -ccpy_einsum("mnef,mbfjn->ebj", H.ab.oovv, R.abb)
            -ccpy_einsum("mbef,mfj->ebj", H.ab.ovvv, R.ab)
            +ccpy_einsum("mnej,mbn->ebj", H.ab.oovo, R.ab)
            -ccpy_einsum("mbej,m->ebj", H.ab.ovvo, R.a)
    )
    # x2b(ib~e~)
    X["ab"]["ovv"] = (
            -ccpy_einsum("nmfe,ifbnm->ibe", H.ab.oovv, R.aab)
            -0.5 * ccpy_einsum("mnef,ibfmn->ibe", H.bb.oovv, R.abb)
            +ccpy_einsum("nbfe,ifn->ibe", H.ab.ovvv, R.aa)
            +ccpy_einsum("bnef,ifn->ibe", H.bb.vovv, R.ab)
            +ccpy_einsum("nmie,nbm->ibe", H.ab.ooov, R.ab)
            -ccpy_einsum("mbie,m->ibe", H.ab.ovov, R.a)
    )

    # x2a(imj)
    X["aa"]["ooo"] = (
             0.25 * ccpy_einsum("mnef,iefjn->imj", H.aa.oovv, R.aaa)
            +0.5 * ccpy_einsum("mnef,iefjn->imj", H.ab.oovv, R.aab)
            +ccpy_einsum("mnjf,ifn->imj", H.aa.ooov, R.aa)
            +ccpy_einsum("mnjf,ifn->imj", H.ab.ooov, R.ab)
            -0.5 * ccpy_einsum("mnji,n->imj", H.aa.oooo, R.a)
    )
    X["aa"]["ooo"] -= np.transpose(X["aa"]["ooo"], (2, 1, 0))
    # x2b(im~j~)
    X["ab"]["ooo"] = (
             ccpy_einsum("nmfe,ifenj->imj", H.ab.oovv, R.aab)
            +0.5 * ccpy_einsum("mnef,iefjn->imj", H.bb.oovv, R.abb)
            +ccpy_einsum("nmfj,ifn->imj", H.ab.oovo, R.aa)
            +ccpy_einsum("mnjf,ifn->imj", H.bb.ooov, R.ab)
            -ccpy_einsum("nmie,nej->imj", H.ab.ooov, R.ab)
            -ccpy_einsum("nmij,n->imj", H.ab.oooo, R.a)
    )

    return X

def get_ipeom3_p_intermediates(H, R, R3_excitations):

    # These intermediates will be 3-index quantities, which are not
    # set up in the models at the moment. We will just use a dictionary
    # as a workaround for now.

    X = {"aa" : {}, "ab" : {}}

    # x2a(ibe)
    X["aa"]["ovv"] = (
            +ccpy_einsum("bnef,ifn->ibe", H.aa.vovv, R.aa)
            +ccpy_einsum("bnef,ifn->ibe", H.ab.vovv, R.ab)
            +0.5 * ccpy_einsum("nmie,nbm->ibe", H.aa.ooov, R.aa)
            +ccpy_einsum("bmie,m->ibe", H.aa.voov, R.a)
    )
    X["aa"]["ovv"] = ipeom3_p_intermediates.add_r3_x2a_ovv(X["aa"]["ovv"],
                                                           R.aaa, R3_excitations["aaa"],
                                                           R.aab, R3_excitations["aab"],
                                                           H.aa.oovv, H.ab.oovv)
    # x2a(imj)
    X["aa"]["ooo"] = (
            +ccpy_einsum("mnjf,ifn->imj", H.aa.ooov, R.aa)
            +ccpy_einsum("mnjf,ifn->imj", H.ab.ooov, R.ab)
            -0.5 * ccpy_einsum("mnji,n->imj", H.aa.oooo, R.a)
    )
    X["aa"]["ooo"] -= np.transpose(X["aa"]["ooo"], (2, 1, 0))
    X["aa"]["ooo"] = ipeom3_p_intermediates.add_r3_x2a_ooo(X["aa"]["ooo"],
                                                           R.aaa, R3_excitations["aaa"],
                                                           R.aab, R3_excitations["aab"],
                                                           H.aa.oovv, H.ab.oovv)
    # x2b(eb~j~)
    X["ab"]["vvo"] = (
            -ccpy_einsum("mbef,mfj->ebj", H.ab.ovvv, R.ab)
            +ccpy_einsum("mnej,mbn->ebj", H.ab.oovo, R.ab)
            -ccpy_einsum("mbej,m->ebj", H.ab.ovvo, R.a)
    )
    X["ab"]["vvo"] = ipeom3_p_intermediates.add_r3_x2b_vvo(X["ab"]["vvo"],
                                                           R.aab, R3_excitations["aab"],
                                                           R.abb, R3_excitations["abb"],
                                                           H.aa.oovv, H.ab.oovv)
    # x2b(ib~e~)
    X["ab"]["ovv"] = (
            +ccpy_einsum("nbfe,ifn->ibe", H.ab.ovvv, R.aa)
            +ccpy_einsum("bnef,ifn->ibe", H.bb.vovv, R.ab)
            +ccpy_einsum("nmie,nbm->ibe", H.ab.ooov, R.ab)
            -ccpy_einsum("mbie,m->ibe", H.ab.ovov, R.a)
    )
    X["ab"]["ovv"] = ipeom3_p_intermediates.add_r3_x2b_ovv(X["ab"]["ovv"],
                                                           R.aab, R3_excitations["aab"],
                                                           R.abb, R3_excitations["abb"],
                                                           H.ab.oovv, H.bb.oovv)
    # x2b(im~j~)
    X["ab"]["ooo"] = (
            +ccpy_einsum("nmfj,ifn->imj", H.ab.oovo, R.aa)
            +ccpy_einsum("mnjf,ifn->imj", H.bb.ooov, R.ab)
            -ccpy_einsum("nmie,nej->imj", H.ab.ooov, R.ab)
            -ccpy_einsum("nmij,n->imj", H.ab.oooo, R.a)
    )
    X["ab"]["ooo"] = ipeom3_p_intermediates.add_r3_x2b_ooo(X["ab"]["ooo"],
                                                           R.aab, R3_excitations["aab"],
                                                           R.abb, R3_excitations["abb"],
                                                           H.ab.oovv, H.bb.oovv)
    return X

def get_ipeomccsdt_intermediates(H, R):

    # These intermediates will be 3-index quantities, which are not
    # set up in the models at the moment. We will just use a dictionary
    # as a workaround for now.

    X = {"a": {}, "aa": {}, "ab": {}}

    # x1a(e)
    X["a"]["v"] = (
        -0.5 * ccpy_einsum("mnef,mfn->e", H.aa.oovv, R.aa)
        - ccpy_einsum("mnef,mfn->e", H.ab.oovv, R.ab)
    )

    # x2a(ibe)
    X["aa"]["ovv"] = (
            -0.5 * ccpy_einsum("mnef,ibfmn->ibe", H.aa.oovv, R.aaa)
            -ccpy_einsum("mnef,ibfmn->ibe", H.ab.oovv, R.aab)
            +ccpy_einsum("bnef,ifn->ibe", H.aa.vovv, R.aa)
            +ccpy_einsum("bnef,ifn->ibe", H.ab.vovv, R.ab)
            +0.5 * ccpy_einsum("nmie,nbm->ibe", H.aa.ooov, R.aa)
            +ccpy_einsum("bmie,m->ibe", H.aa.voov, R.a)
    )
    # x2b(eb~j~)
    X["ab"]["vvo"] = (
            -0.5 * ccpy_einsum("mnef,mfbnj->ebj", H.aa.oovv, R.aab)
            -ccpy_einsum("mnef,mbfjn->ebj", H.ab.oovv, R.abb)
            -ccpy_einsum("mbef,mfj->ebj", H.ab.ovvv, R.ab)
            +ccpy_einsum("mnej,mbn->ebj", H.ab.oovo, R.ab)
            -ccpy_einsum("mbej,m->ebj", H.ab.ovvo, R.a)
    )
    # x2b(ib~e~)
    X["ab"]["ovv"] = (
            -ccpy_einsum("nmfe,ifbnm->ibe", H.ab.oovv, R.aab)
            -0.5 * ccpy_einsum("mnef,ibfmn->ibe", H.bb.oovv, R.abb)
            +ccpy_einsum("nbfe,ifn->ibe", H.ab.ovvv, R.aa)
            +ccpy_einsum("bnef,ifn->ibe", H.bb.vovv, R.ab)
            +ccpy_einsum("nmie,nbm->ibe", H.ab.ooov, R.ab)
            -ccpy_einsum("mbie,m->ibe", H.ab.ovov, R.a)
    )

    # x2a(imj)
    X["aa"]["ooo"] = (
             0.25 * ccpy_einsum("mnef,iefjn->imj", H.aa.oovv, R.aaa)
            +0.5 * ccpy_einsum("mnef,iefjn->imj", H.ab.oovv, R.aab)
            +ccpy_einsum("mnjf,ifn->imj", H.aa.ooov, R.aa)
            +ccpy_einsum("mnjf,ifn->imj", H.ab.ooov, R.ab)
            -0.5 * ccpy_einsum("mnji,n->imj", H.aa.oooo, R.a)
    )
    X["aa"]["ooo"] -= np.transpose(X["aa"]["ooo"], (2, 1, 0))
    # x2b(im~j~)
    X["ab"]["ooo"] = (
             ccpy_einsum("nmfe,ifenj->imj", H.ab.oovv, R.aab)
            +0.5 * ccpy_einsum("mnef,iefjn->imj", H.bb.oovv, R.abb)
            +ccpy_einsum("nmfj,ifn->imj", H.ab.oovo, R.aa)
            +ccpy_einsum("mnjf,ifn->imj", H.bb.ooov, R.ab)
            -ccpy_einsum("nmie,nej->imj", H.ab.ooov, R.ab)
            -ccpy_einsum("nmij,n->imj", H.ab.oooo, R.a)
    )

    # additional intermediates for T3

    # These are redunant intermediates. They are part of h(vvov)*R1 and h(vooo)*R1 in R2 update,
    # which are taken care of by CCSDT Hbar T3 terms
    # # x2a(fne)
    # X["aa"]["vov"] = -ccpy_einsum("mnef,m->fne", H.aa.oovv, R.a)
    # # x2b(f~n~e)
    # X["ab"]["vov"] = -ccpy_einsum("mnef,m->fne", H.ab.oovv, R.a)
    # x2a(iem)
    X["aa"]["ovo"] = (
        -ccpy_einsum("nmie,n->iem", H.aa.ooov, R.a)
        + ccpy_einsum("mnef,ifn->iem", H.aa.oovv, R.aa)
        + ccpy_einsum("mnef,ifn->iem", H.ab.oovv, R.ab)
    )
    # x2b(ie~m~)
    X["ab"]["ovo"] = (
        -ccpy_einsum("nmie,n->iem", H.ab.ooov, R.a)
        + ccpy_einsum("nmfe,ifn->iem", H.ab.oovv, R.aa)
        + ccpy_einsum("mnef,ifn->iem", H.bb.oovv, R.ab)
    )
    # x2b (fm~j~)
    X["ab"]["voo"] = (
        -ccpy_einsum("nmfj,n->fmj", H.ab.oovo, R.a)
        -ccpy_einsum("nmfe,nej->fmj", H.ab.oovv, R.ab)
    )
    # x2a(aef)
    X["aa"]["vvv"] = (
        -ccpy_einsum("anef,n->aef", H.aa.vovv, R.a)
        +0.5 * ccpy_einsum("mnef,nam->aef", H.aa.oovv, R.aa)
    )
    # x2b(fa~e~)
    X["ab"]["vvv"] = (
        -ccpy_einsum("nafe,n->fae", H.ab.ovvv, R.a)
        +ccpy_einsum("nmfe,nam->fae", H.ab.oovv, R.ab)
    )
    return X

def get_ipeomt_p_intermediates(H, R, R3_excitations):

    # These intermediates will be 3-index quantities, which are not
    # set up in the models at the moment. We will just use a dictionary
    # as a workaround for now.

    X = {"a": {}, "aa": {}, "ab": {}}

    # x1a(e)
    X["a"]["v"] = (
        -0.5 * ccpy_einsum("mnef,mfn->e", H.aa.oovv, R.aa)
        - ccpy_einsum("mnef,mfn->e", H.ab.oovv, R.ab)
    )

    # x2a(ibe)
    X["aa"]["ovv"] = (
            +ccpy_einsum("bnef,ifn->ibe", H.aa.vovv, R.aa)
            +ccpy_einsum("bnef,ifn->ibe", H.ab.vovv, R.ab)
            +0.5 * ccpy_einsum("nmie,nbm->ibe", H.aa.ooov, R.aa)
            +ccpy_einsum("bmie,m->ibe", H.aa.voov, R.a)
    )
    X["aa"]["ovv"] = ipeom3_p_intermediates.add_r3_x2a_ovv(X["aa"]["ovv"],
                                                           R.aaa, R3_excitations["aaa"],
                                                           R.aab, R3_excitations["aab"],
                                                           H.aa.oovv, H.ab.oovv)
    # x2a(imj)
    X["aa"]["ooo"] = (
            +ccpy_einsum("mnjf,ifn->imj", H.aa.ooov, R.aa)
            +ccpy_einsum("mnjf,ifn->imj", H.ab.ooov, R.ab)
            -0.5 * ccpy_einsum("mnji,n->imj", H.aa.oooo, R.a)
    )
    X["aa"]["ooo"] -= np.transpose(X["aa"]["ooo"], (2, 1, 0))
    X["aa"]["ooo"] = ipeom3_p_intermediates.add_r3_x2a_ooo(X["aa"]["ooo"],
                                                           R.aaa, R3_excitations["aaa"],
                                                           R.aab, R3_excitations["aab"],
                                                           H.aa.oovv, H.ab.oovv)
    # x2b(eb~j~)
    X["ab"]["vvo"] = (
            -ccpy_einsum("mbef,mfj->ebj", H.ab.ovvv, R.ab)
            +ccpy_einsum("mnej,mbn->ebj", H.ab.oovo, R.ab)
            -ccpy_einsum("mbej,m->ebj", H.ab.ovvo, R.a)
    )
    X["ab"]["vvo"] = ipeom3_p_intermediates.add_r3_x2b_vvo(X["ab"]["vvo"],
                                                           R.aab, R3_excitations["aab"],
                                                           R.abb, R3_excitations["abb"],
                                                           H.aa.oovv, H.ab.oovv)
    # x2b(ib~e~)
    X["ab"]["ovv"] = (
            +ccpy_einsum("nbfe,ifn->ibe", H.ab.ovvv, R.aa)
            +ccpy_einsum("bnef,ifn->ibe", H.bb.vovv, R.ab)
            +ccpy_einsum("nmie,nbm->ibe", H.ab.ooov, R.ab)
            -ccpy_einsum("mbie,m->ibe", H.ab.ovov, R.a)
    )
    X["ab"]["ovv"] = ipeom3_p_intermediates.add_r3_x2b_ovv(X["ab"]["ovv"],
                                                           R.aab, R3_excitations["aab"],
                                                           R.abb, R3_excitations["abb"],
                                                           H.ab.oovv, H.bb.oovv)
    # x2b(im~j~)
    X["ab"]["ooo"] = (
            +ccpy_einsum("nmfj,ifn->imj", H.ab.oovo, R.aa)
            +ccpy_einsum("mnjf,ifn->imj", H.bb.ooov, R.ab)
            -ccpy_einsum("nmie,nej->imj", H.ab.ooov, R.ab)
            -ccpy_einsum("nmij,n->imj", H.ab.oooo, R.a)
    )
    X["ab"]["ooo"] = ipeom3_p_intermediates.add_r3_x2b_ooo(X["ab"]["ooo"],
                                                           R.aab, R3_excitations["aab"],
                                                           R.abb, R3_excitations["abb"],
                                                           H.ab.oovv, H.bb.oovv)
    # additional intermediates for T3

    # These are redunant intermediates. They are part of h(vvov)*R1 and h(vooo)*R1 in R2 update,
    # which are taken care of by CCSDT Hbar T3 terms
    # # x2a(fne)
    # X["aa"]["vov"] = -ccpy_einsum("mnef,m->fne", H.aa.oovv, R.a)
    # # x2b(f~n~e)
    # X["ab"]["vov"] = -ccpy_einsum("mnef,m->fne", H.ab.oovv, R.a)
    # x2a(iem)
    X["aa"]["ovo"] = (
        -ccpy_einsum("nmie,n->iem", H.aa.ooov, R.a)
        + ccpy_einsum("mnef,ifn->iem", H.aa.oovv, R.aa)
        + ccpy_einsum("mnef,ifn->iem", H.ab.oovv, R.ab)
    )
    # x2b(ie~m~)
    X["ab"]["ovo"] = (
        -ccpy_einsum("nmie,n->iem", H.ab.ooov, R.a)
        + ccpy_einsum("nmfe,ifn->iem", H.ab.oovv, R.aa)
        + ccpy_einsum("mnef,ifn->iem", H.bb.oovv, R.ab)
    )
    # x2b (fm~j~)
    X["ab"]["voo"] = (
        -ccpy_einsum("nmfj,n->fmj", H.ab.oovo, R.a)
        -ccpy_einsum("nmfe,nej->fmj", H.ab.oovv, R.ab)
    )
    # x2a(aef)
    X["aa"]["vvv"] = (
        -ccpy_einsum("anef,n->aef", H.aa.vovv, R.a)
        +0.5 * ccpy_einsum("mnef,nam->aef", H.aa.oovv, R.aa)
    )
    # x2b(fa~e~)
    X["ab"]["vvv"] = (
        -ccpy_einsum("nafe,n->fae", H.ab.ovvv, R.a)
        +ccpy_einsum("nmfe,nam->fae", H.ab.oovv, R.ab)
    )
    return X

def get_ipeomccsdta_intermediates(H, R, T):

    # IP-EOMCCSD(T)(a) intermediates should use bare integrals in all
    # places that contract with T3

    # These intermediates will be 3-index quantities, which are not
    # set up in the models at the moment. We will just use a dictionary
    # as a workaround for now.

    X = {"a": {}, "aa": {}, "ab": {}}

    # x1a(e)
    X["a"]["v"] = (
        -0.5 * ccpy_einsum("mnef,mfn->e", H.aa.oovv, R.aa)
        - ccpy_einsum("mnef,mfn->e", H.ab.oovv, R.ab)
    )

    # x2a(ibe)
    X["aa"]["ovv"] = (
            -0.5 * ccpy_einsum("mnef,ibfmn->ibe", H.aa.oovv, R.aaa)
            -ccpy_einsum("mnef,ibfmn->ibe", H.ab.oovv, R.aab)
            +ccpy_einsum("bnef,ifn->ibe", H.aa.vovv, R.aa)
            +ccpy_einsum("bnef,ifn->ibe", H.ab.vovv, R.ab)
            +0.5 * ccpy_einsum("nmie,nbm->ibe", H.aa.ooov, R.aa)
            +ccpy_einsum("bmie,m->ibe", H.aa.voov, R.a)
    )
    # x2b(eb~j~)
    X["ab"]["vvo"] = (
            -0.5 * ccpy_einsum("mnef,mfbnj->ebj", H.aa.oovv, R.aab)
            -ccpy_einsum("mnef,mbfjn->ebj", H.ab.oovv, R.abb)
            -ccpy_einsum("mbef,mfj->ebj", H.ab.ovvv, R.ab)
            +ccpy_einsum("mnej,mbn->ebj", H.ab.oovo, R.ab)
            -ccpy_einsum("mbej,m->ebj", H.ab.ovvo, R.a)
    )
    # x2b(ib~e~)
    X["ab"]["ovv"] = (
            -ccpy_einsum("nmfe,ifbnm->ibe", H.ab.oovv, R.aab)
            -0.5 * ccpy_einsum("mnef,ibfmn->ibe", H.bb.oovv, R.abb)
            +ccpy_einsum("nbfe,ifn->ibe", H.ab.ovvv, R.aa)
            +ccpy_einsum("bnef,ifn->ibe", H.bb.vovv, R.ab)
            +ccpy_einsum("nmie,nbm->ibe", H.ab.ooov, R.ab)
            -ccpy_einsum("mbie,m->ibe", H.ab.ovov, R.a)
    )

    # x2a(imj)
    X["aa"]["ooo"] = (
             0.25 * ccpy_einsum("mnef,iefjn->imj", H.aa.oovv, R.aaa)
            +0.5 * ccpy_einsum("mnef,iefjn->imj", H.ab.oovv, R.aab)
            +ccpy_einsum("mnjf,ifn->imj", H.aa.ooov, R.aa)
            +ccpy_einsum("mnjf,ifn->imj", H.ab.ooov, R.ab)
            -0.5 * ccpy_einsum("mnji,n->imj", H.aa.oooo, R.a)
    )
    X["aa"]["ooo"] -= np.transpose(X["aa"]["ooo"], (2, 1, 0))
    # x2b(im~j~)
    X["ab"]["ooo"] = (
             ccpy_einsum("nmfe,ifenj->imj", H.ab.oovv, R.aab)
            +0.5 * ccpy_einsum("mnef,iefjn->imj", H.bb.oovv, R.abb)
            +ccpy_einsum("nmfj,ifn->imj", H.ab.oovo, R.aa)
            +ccpy_einsum("mnjf,ifn->imj", H.bb.ooov, R.ab)
            -ccpy_einsum("nmie,nej->imj", H.ab.ooov, R.ab)
            -ccpy_einsum("nmij,n->imj", H.ab.oooo, R.a)
    )

    # additional intermediates for T3 [should be bare integrals for CCSD(T)(a)]
    # here, we only have oovv, ooov, and vovv. For convenience, just remove
    # the T1 contribution to ooov and vovv and contract to intermediate here
    h0_aa_ooov = H.aa.ooov - ccpy_einsum("mnfe,fi->mnie", H.aa.oovv, T.a)
    h0_ab_ooov = H.ab.ooov - ccpy_einsum("mnfe,fi->mnie", H.ab.oovv, T.a)
    h0_ab_oovo = H.ab.oovo - ccpy_einsum("mnef,fi->mnei", H.ab.oovv, T.b)
    h0_aa_vovv = H.aa.vovv + ccpy_einsum("mnfe,an->amef", H.aa.oovv, T.a)
    h0_ab_ovvv = H.ab.ovvv + ccpy_einsum("mnef,an->maef", H.ab.oovv, T.b)
    # x2a(iem)
    X["aa"]["ovo"] = (
        -ccpy_einsum("nmie,n->iem", h0_aa_ooov, R.a)
        + ccpy_einsum("mnef,ifn->iem", H.aa.oovv, R.aa)
        + ccpy_einsum("mnef,ifn->iem", H.ab.oovv, R.ab)
    )
    # x2b(ie~m~)
    X["ab"]["ovo"] = (
        -ccpy_einsum("nmie,n->iem", h0_ab_ooov, R.a)
        + ccpy_einsum("nmfe,ifn->iem", H.ab.oovv, R.aa)
        + ccpy_einsum("mnef,ifn->iem", H.bb.oovv, R.ab)
    )
    # x2b (fm~j~)
    X["ab"]["voo"] = (
        -ccpy_einsum("nmfj,n->fmj", h0_ab_oovo, R.a)
        -ccpy_einsum("nmfe,nej->fmj", H.ab.oovv, R.ab)
    )
    # x2a(aef)
    X["aa"]["vvv"] = (
        -ccpy_einsum("anef,n->aef", h0_aa_vovv, R.a)
        +0.5 * ccpy_einsum("mnef,nam->aef", H.aa.oovv, R.aa)
    )
    # x2b(fa~e~)
    X["ab"]["vvv"] = (
        -ccpy_einsum("nafe,n->fae", h0_ab_ovvv, R.a)
        +ccpy_einsum("nmfe,nam->fae", H.ab.oovv, R.ab)
    )
    return X

def add_v_term(X, H, R):
    # add h(ov) * R1 term to X["a"]["v"] intermediate
    X["a"]["v"] -= ccpy_einsum("me,m->e", H.a.ov, R.a)
    return X

def add_v_term_Ta(X, H, R, T):
    # obtain f(ov) element from h(ov) [this is useless for RHF orbitals, since f(ov) = 0]
    f_a_ov = (
        H.a.ov
        - ccpy_einsum("mnef,fn->me", H.aa.oovv, T.a)
        - ccpy_einsum("mnef,fn->me", H.ab.oovv, T.b)
    )
    # add f(ov) * R1 term to X["a"]["v"] intermediate
    X["a"]["v"] -= ccpy_einsum("me,m->e", f_a_ov, R.a)
    return X