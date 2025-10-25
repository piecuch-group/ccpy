import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import eaeom3_p_intermediates

def get_eaeom3_intermediates(H, R):

    # These intermediates will be 3-index quantities, which are not
    # set up in the models at the moment. We will just use a dictionary
    # as a workaround for now.

    X = {"aa" : {}, "ab" : {}}

    # x2a(amj)
    X["aa"]["voo"] = (
                    0.5*ccpy_einsum("mnef,aefjn->amj", H.aa.oovv, R.aaa)
                    +ccpy_einsum("mnef,aefjn->amj", H.ab.oovv, R.aab)
                    +ccpy_einsum("mnjf,afn->amj", H.aa.ooov, R.aa)
                    +ccpy_einsum("mnjf,afn->amj", H.ab.ooov, R.ab)
                    +0.5*ccpy_einsum("amef,efj->amj", H.aa.vovv, R.aa)
                    -ccpy_einsum("amje,e->amj", H.aa.voov, R.a) # CAREFUL: this is a minus sign
    )
    # x2b(mb~j~)
    X["ab"]["ovo"] = (
                    0.5*ccpy_einsum("mnef,efbnj->mbj", H.aa.oovv, R.aab)
                    +ccpy_einsum("mnef,efbnj->mbj", H.ab.oovv, R.abb)
                    -ccpy_einsum("mnej,ebn->mbj", H.ab.oovo, R.ab)
                    +ccpy_einsum("mbef,efj->mbj", H.ab.ovvv, R.ab)
                    +ccpy_einsum("mbfj,f->mbj", H.ab.ovvo, R.a)
    )
    # x2b(am~j~)
    X["ab"]["voo"] = (
                    ccpy_einsum("nmfe,afenj->amj", H.ab.oovv, R.aab)
                    +0.5*ccpy_einsum("mnef,aefjn->amj", H.bb.oovv, R.abb)
                    +ccpy_einsum("nmfj,afn->amj", H.ab.oovo, R.aa)
                    +ccpy_einsum("mnjf,afn->amj", H.bb.ooov, R.ab)
                    +ccpy_einsum("amef,efj->amj", H.ab.vovv, R.ab)
                    +ccpy_einsum("amej,e->amj", H.ab.vovo, R.a)
    )

    # x2a(abe)
    X["aa"]["vvv"] = (
                    -0.25*ccpy_einsum("mnef,abfmn->abe", H.aa.oovv, R.aaa)
                    -0.5*ccpy_einsum("mnef,abfmn->abe", H.ab.oovv, R.aab)
                    +ccpy_einsum("bnef,afn->abe", H.aa.vovv, R.aa)
                    +ccpy_einsum("bnef,afn->abe", H.ab.vovv, R.ab)
                    +0.5*ccpy_einsum("abfe,f->abe", H.aa.vvvv, R.a)
    )
    X["aa"]["vvv"] -= np.transpose(X["aa"]["vvv"], (1, 0, 2))
    # x2b(ab~e~)
    X["ab"]["vvv"] = (
                    -ccpy_einsum("nmfe,afbnm->abe", H.ab.oovv, R.aab)
                    -0.5*ccpy_einsum("mnef,abfmn->abe", H.bb.oovv, R.abb)
                    +ccpy_einsum("nbfe,afn->abe", H.ab.ovvv, R.aa)
                    +ccpy_einsum("bnef,afn->abe", H.bb.vovv, R.ab)
                    -ccpy_einsum("amfe,fbm->abe", H.ab.vovv, R.ab)
                    +ccpy_einsum("abfe,f->abe", H.ab.vvvv, R.a)
    )

    return X

def get_eaeom3_p_intermediates(H, R, R3_excitations):

    # These intermediates will be 3-index quantities, which are not
    # set up in the models at the moment. We will just use a dictionary
    # as a workaround for now.

    X = {"aa": {}, "ab": {}}

    # x2a(amj)
    X["aa"]["voo"] = (
                    +ccpy_einsum("mnjf,afn->amj", H.aa.ooov, R.aa)
                    +ccpy_einsum("mnjf,afn->amj", H.ab.ooov, R.ab)
                    +0.5*ccpy_einsum("amef,efj->amj", H.aa.vovv, R.aa)
                    -ccpy_einsum("amje,e->amj", H.aa.voov, R.a) # CAREFUL: this is a minus sign
    )
    X["aa"]["voo"] = eaeom3_p_intermediates.add_r3_x2a_voo(X["aa"]["voo"],
                                                                                  R.aaa, R3_excitations["aaa"],
                                                                                  R.aab, R3_excitations["aab"],
                                                                                  H.aa.oovv, H.ab.oovv)
    # x2b(mb~j~)
    X["ab"]["ovo"] = (
                    -ccpy_einsum("mnej,ebn->mbj", H.ab.oovo, R.ab)
                    +ccpy_einsum("mbef,efj->mbj", H.ab.ovvv, R.ab)
                    +ccpy_einsum("mbfj,f->mbj", H.ab.ovvo, R.a)
    )
    X["ab"]["ovo"] = eaeom3_p_intermediates.add_r3_x2b_ovo(X["ab"]["ovo"],
                                                                                  R.aab, R3_excitations["aab"],
                                                                                  R.abb, R3_excitations["abb"],
                                                                                  H.aa.oovv, H.ab.oovv)
    # x2b(am~j~)
    X["ab"]["voo"] = (
                    +ccpy_einsum("nmfj,afn->amj", H.ab.oovo, R.aa)
                    +ccpy_einsum("mnjf,afn->amj", H.bb.ooov, R.ab)
                    +ccpy_einsum("amef,efj->amj", H.ab.vovv, R.ab)
                    +ccpy_einsum("amej,e->amj", H.ab.vovo, R.a)
    )
    X["ab"]["voo"] = eaeom3_p_intermediates.add_r3_x2b_voo(X["ab"]["voo"],
                                                                                  R.aab, R3_excitations["aab"],
                                                                                  R.abb, R3_excitations["abb"],
                                                                                  H.ab.oovv, H.bb.oovv)

    # x2a(abe)
    X["aa"]["vvv"] = (
                    +ccpy_einsum("bnef,afn->abe", H.aa.vovv, R.aa)
                    +ccpy_einsum("bnef,afn->abe", H.ab.vovv, R.ab)
                    +0.5*ccpy_einsum("abfe,f->abe", H.aa.vvvv, R.a)
    )
    X["aa"]["vvv"] -= np.transpose(X["aa"]["vvv"], (1, 0, 2))
    X["aa"]["vvv"] = eaeom3_p_intermediates.add_r3_x2a_vvv(X["aa"]["vvv"],
                                                                                  R.aaa, R3_excitations["aaa"],
                                                                                  R.aab, R3_excitations["aab"],
                                                                                  H.aa.oovv, H.ab.oovv)
    # x2b(ab~e~)
    X["ab"]["vvv"] = (
                    +ccpy_einsum("nbfe,afn->abe", H.ab.ovvv, R.aa)
                    +ccpy_einsum("bnef,afn->abe", H.bb.vovv, R.ab)
                    -ccpy_einsum("amfe,fbm->abe", H.ab.vovv, R.ab)
                    +ccpy_einsum("abfe,f->abe", H.ab.vvvv, R.a)
    )
    X["ab"]["vvv"] = eaeom3_p_intermediates.add_r3_x2b_vvv(X["ab"]["vvv"],
                                                                                  R.aab, R3_excitations["aab"],
                                                                                  R.abb, R3_excitations["abb"],
                                                                                  H.ab.oovv, H.bb.oovv)

    return X

def get_eaeomccsdt_intermediates(H, R):

    # These intermediates will be 3-index quantities, which are not
    # set up in the models at the moment. We will just use a dictionary
    # as a workaround for now.

    X = {"a": {}, "aa": {}, "ab": {}}

    # x1a(m)
    X["a"]["o"] = (
        0.5 * ccpy_einsum("mnef,efn->m", H.aa.oovv, R.aa)
        + ccpy_einsum("mnef,efn->m", H.ab.oovv, R.ab)
    )

    # x2a(amj)
    X["aa"]["voo"] = (
                    0.5*ccpy_einsum("mnef,aefjn->amj", H.aa.oovv, R.aaa)
                    +ccpy_einsum("mnef,aefjn->amj", H.ab.oovv, R.aab)
                    +ccpy_einsum("mnjf,afn->amj", H.aa.ooov, R.aa)
                    +ccpy_einsum("mnjf,afn->amj", H.ab.ooov, R.ab)
                    +0.5*ccpy_einsum("amef,efj->amj", H.aa.vovv, R.aa)
                    -ccpy_einsum("amje,e->amj", H.aa.voov, R.a) # CAREFUL: this is a minus sign
    )
    # x2b(mb~j~)
    X["ab"]["ovo"] = (
                    0.5*ccpy_einsum("mnef,efbnj->mbj", H.aa.oovv, R.aab)
                    +ccpy_einsum("mnef,efbnj->mbj", H.ab.oovv, R.abb)
                    -ccpy_einsum("mnej,ebn->mbj", H.ab.oovo, R.ab)
                    +ccpy_einsum("mbef,efj->mbj", H.ab.ovvv, R.ab)
                    +ccpy_einsum("mbfj,f->mbj", H.ab.ovvo, R.a)
    )
    # x2b(am~j~)
    X["ab"]["voo"] = (
                    ccpy_einsum("nmfe,afenj->amj", H.ab.oovv, R.aab)
                    +0.5*ccpy_einsum("mnef,aefjn->amj", H.bb.oovv, R.abb)
                    +ccpy_einsum("nmfj,afn->amj", H.ab.oovo, R.aa)
                    +ccpy_einsum("mnjf,afn->amj", H.bb.ooov, R.ab)
                    +ccpy_einsum("amef,efj->amj", H.ab.vovv, R.ab)
                    +ccpy_einsum("amej,e->amj", H.ab.vovo, R.a)
    )

    # x2a(abe)
    X["aa"]["vvv"] = (
                    -0.25*ccpy_einsum("mnef,abfmn->abe", H.aa.oovv, R.aaa)
                    -0.5*ccpy_einsum("mnef,abfmn->abe", H.ab.oovv, R.aab)
                    +ccpy_einsum("bnef,afn->abe", H.aa.vovv, R.aa)
                    +ccpy_einsum("bnef,afn->abe", H.ab.vovv, R.ab)
                    +0.5*ccpy_einsum("abfe,f->abe", H.aa.vvvv, R.a)
    )
    X["aa"]["vvv"] -= np.transpose(X["aa"]["vvv"], (1, 0, 2))
    # x2b(ab~e~)
    X["ab"]["vvv"] = (
                    -ccpy_einsum("nmfe,afbnm->abe", H.ab.oovv, R.aab)
                    -0.5*ccpy_einsum("mnef,abfmn->abe", H.bb.oovv, R.abb)
                    +ccpy_einsum("nbfe,afn->abe", H.ab.ovvv, R.aa)
                    +ccpy_einsum("bnef,afn->abe", H.bb.vovv, R.ab)
                    -ccpy_einsum("amfe,fbm->abe", H.ab.vovv, R.ab)
                    +ccpy_einsum("abfe,f->abe", H.ab.vvvv, R.a)
    )

    # additional intermediates for T3
    X["aa"]["vvo"] = (
            ccpy_einsum("amfe,f->aem", H.aa.vovv, R.a)
            +ccpy_einsum("mnef,afn->aem", H.aa.oovv, R.aa)
            +ccpy_einsum("mnef,afn->aem", H.ab.oovv, R.ab)
    )
    X["ab"]["vvo"] = (
            ccpy_einsum("amfe,f->aem", H.ab.vovv, R.a)
            +ccpy_einsum("nmfe,afn->aem", H.ab.oovv, R.aa)
            +ccpy_einsum("mnef,afn->aem", H.bb.oovv, R.ab)
    )
    X["ab"]["ovv"] = (
            ccpy_einsum("mbef,e->mbf", H.ab.ovvv, R.a)
            -ccpy_einsum("mnef,ebn->mbf", H.ab.oovv, R.ab)
    )
    X["aa"]["ooo"] = (
            ccpy_einsum("nmje,e->mnj", H.aa.ooov, R.a)
            +0.5 * ccpy_einsum("mnef,efj->mnj", H.aa.oovv, R.aa)
    )
    X["ab"]["ooo"] = (
            ccpy_einsum("mnej,e->mnj", H.ab.oovo, R.a)
            +ccpy_einsum("mnef,efj->mnj", H.ab.oovv, R.ab)
    )

    return X

def add_o_term(X, H, R):
    # add h(ov) * R1 term to X["a"]["o"] intermediates
    X["a"]["o"] += ccpy_einsum("me,e->m", H.a.ov, R.a)
    return X
