import numpy as np
from ccpy.lib.core import eaeom3_p_intermediates

def get_eaeom3_intermediates(H, R):

    # These intermediates will be 3-index quantities, which are not
    # set up in the models at the moment. We will just use a dictionary
    # as a workaround for now.

    X = {"aa" : {}, "ab" : {}}

    # x2a(amj)
    X["aa"]["voo"] = (
                    0.5*np.einsum("mnef,aefjn->amj", H.aa.oovv, R.aaa, optimize=True)
                    +np.einsum("mnef,aefjn->amj", H.ab.oovv, R.aab, optimize=True)
                    +np.einsum("mnjf,afn->amj", H.aa.ooov, R.aa, optimize=True)
                    +np.einsum("mnjf,afn->amj", H.ab.ooov, R.ab, optimize=True)
                    +0.5*np.einsum("amef,efj->amj", H.aa.vovv, R.aa, optimize=True)
                    -np.einsum("amje,e->amj", H.aa.voov, R.a, optimize=True) # CAREFUL: this is a minus sign
    )
    # x2b(mb~j~)
    X["ab"]["ovo"] = (
                    0.5*np.einsum("mnef,efbnj->mbj", H.aa.oovv, R.aab, optimize=True)
                    +np.einsum("mnef,efbnj->mbj", H.ab.oovv, R.abb, optimize=True)
                    -np.einsum("mnej,ebn->mbj", H.ab.oovo, R.ab, optimize=True)
                    +np.einsum("mbef,efj->mbj", H.ab.ovvv, R.ab, optimize=True)
                    +np.einsum("mbfj,f->mbj", H.ab.ovvo, R.a, optimize=True)
    )
    # x2b(am~j~)
    X["ab"]["voo"] = (
                    np.einsum("nmfe,afenj->amj", H.ab.oovv, R.aab, optimize=True)
                    +0.5*np.einsum("mnef,aefjn->amj", H.bb.oovv, R.abb, optimize=True)
                    +np.einsum("nmfj,afn->amj", H.ab.oovo, R.aa, optimize=True)
                    +np.einsum("mnjf,afn->amj", H.bb.ooov, R.ab, optimize=True)
                    +np.einsum("amef,efj->amj", H.ab.vovv, R.ab, optimize=True)
                    +np.einsum("amej,e->amj", H.ab.vovo, R.a, optimize=True)
    )

    # x2a(abe)
    X["aa"]["vvv"] = (
                    -0.25*np.einsum("mnef,abfmn->abe", H.aa.oovv, R.aaa, optimize=True)
                    -0.5*np.einsum("mnef,abfmn->abe", H.ab.oovv, R.aab, optimize=True)
                    +np.einsum("bnef,afn->abe", H.aa.vovv, R.aa, optimize=True)
                    +np.einsum("bnef,afn->abe", H.ab.vovv, R.ab, optimize=True)
                    +0.5*np.einsum("abfe,f->abe", H.aa.vvvv, R.a, optimize=True)
    )
    X["aa"]["vvv"] -= np.transpose(X["aa"]["vvv"], (1, 0, 2))
    # x2b(ab~e~)
    X["ab"]["vvv"] = (
                    -np.einsum("nmfe,afbnm->abe", H.ab.oovv, R.aab, optimize=True)
                    -0.5*np.einsum("mnef,abfmn->abe", H.bb.oovv, R.abb, optimize=True)
                    +np.einsum("nbfe,afn->abe", H.ab.ovvv, R.aa, optimize=True)
                    +np.einsum("bnef,afn->abe", H.bb.vovv, R.ab, optimize=True)
                    -np.einsum("amfe,fbm->abe", H.ab.vovv, R.ab, optimize=True)
                    +np.einsum("abfe,f->abe", H.ab.vvvv, R.a, optimize=True)
    )

    return X

def get_eaeom3_p_intermediates(H, R, R3_excitations):

    # These intermediates will be 3-index quantities, which are not
    # set up in the models at the moment. We will just use a dictionary
    # as a workaround for now.

    X = {"aa": {}, "ab": {}}

    # x2a(amj)
    X["aa"]["voo"] = (
                    +np.einsum("mnjf,afn->amj", H.aa.ooov, R.aa, optimize=True)
                    +np.einsum("mnjf,afn->amj", H.ab.ooov, R.ab, optimize=True)
                    +0.5*np.einsum("amef,efj->amj", H.aa.vovv, R.aa, optimize=True)
                    -np.einsum("amje,e->amj", H.aa.voov, R.a, optimize=True) # CAREFUL: this is a minus sign
    )
    X["aa"]["voo"] = eaeom3_p_intermediates.add_r3_x2a_voo(X["aa"]["voo"],
                                                                                  R.aaa, R3_excitations["aaa"],
                                                                                  R.aab, R3_excitations["aab"],
                                                                                  H.aa.oovv, H.ab.oovv)
    # x2b(mb~j~)
    X["ab"]["ovo"] = (
                    -np.einsum("mnej,ebn->mbj", H.ab.oovo, R.ab, optimize=True)
                    +np.einsum("mbef,efj->mbj", H.ab.ovvv, R.ab, optimize=True)
                    +np.einsum("mbfj,f->mbj", H.ab.ovvo, R.a, optimize=True)
    )
    X["ab"]["ovo"] = eaeom3_p_intermediates.add_r3_x2b_ovo(X["ab"]["ovo"],
                                                                                  R.aab, R3_excitations["aab"],
                                                                                  R.abb, R3_excitations["abb"],
                                                                                  H.aa.oovv, H.ab.oovv)
    # x2b(am~j~)
    X["ab"]["voo"] = (
                    +np.einsum("nmfj,afn->amj", H.ab.oovo, R.aa, optimize=True)
                    +np.einsum("mnjf,afn->amj", H.bb.ooov, R.ab, optimize=True)
                    +np.einsum("amef,efj->amj", H.ab.vovv, R.ab, optimize=True)
                    +np.einsum("amej,e->amj", H.ab.vovo, R.a, optimize=True)
    )
    X["ab"]["voo"] = eaeom3_p_intermediates.add_r3_x2b_voo(X["ab"]["voo"],
                                                                                  R.aab, R3_excitations["aab"],
                                                                                  R.abb, R3_excitations["abb"],
                                                                                  H.ab.oovv, H.bb.oovv)

    # x2a(abe)
    X["aa"]["vvv"] = (
                    +np.einsum("bnef,afn->abe", H.aa.vovv, R.aa, optimize=True)
                    +np.einsum("bnef,afn->abe", H.ab.vovv, R.ab, optimize=True)
                    +0.5*np.einsum("abfe,f->abe", H.aa.vvvv, R.a, optimize=True)
    )
    X["aa"]["vvv"] -= np.transpose(X["aa"]["vvv"], (1, 0, 2))
    X["aa"]["vvv"] = eaeom3_p_intermediates.add_r3_x2a_vvv(X["aa"]["vvv"],
                                                                                  R.aaa, R3_excitations["aaa"],
                                                                                  R.aab, R3_excitations["aab"],
                                                                                  H.aa.oovv, H.ab.oovv)
    # x2b(ab~e~)
    X["ab"]["vvv"] = (
                    +np.einsum("nbfe,afn->abe", H.ab.ovvv, R.aa, optimize=True)
                    +np.einsum("bnef,afn->abe", H.bb.vovv, R.ab, optimize=True)
                    -np.einsum("amfe,fbm->abe", H.ab.vovv, R.ab, optimize=True)
                    +np.einsum("abfe,f->abe", H.ab.vvvv, R.a, optimize=True)
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
        0.5 * np.einsum("mnef,efn->m", H.aa.oovv, R.aa, optimize=True)
        + np.einsum("mnef,efn->m", H.ab.oovv, R.ab, optimize=True)
    )

    # x2a(amj)
    X["aa"]["voo"] = (
                    0.5*np.einsum("mnef,aefjn->amj", H.aa.oovv, R.aaa, optimize=True)
                    +np.einsum("mnef,aefjn->amj", H.ab.oovv, R.aab, optimize=True)
                    +np.einsum("mnjf,afn->amj", H.aa.ooov, R.aa, optimize=True)
                    +np.einsum("mnjf,afn->amj", H.ab.ooov, R.ab, optimize=True)
                    +0.5*np.einsum("amef,efj->amj", H.aa.vovv, R.aa, optimize=True)
                    -np.einsum("amje,e->amj", H.aa.voov, R.a, optimize=True) # CAREFUL: this is a minus sign
    )
    # x2b(mb~j~)
    X["ab"]["ovo"] = (
                    0.5*np.einsum("mnef,efbnj->mbj", H.aa.oovv, R.aab, optimize=True)
                    +np.einsum("mnef,efbnj->mbj", H.ab.oovv, R.abb, optimize=True)
                    -np.einsum("mnej,ebn->mbj", H.ab.oovo, R.ab, optimize=True)
                    +np.einsum("mbef,efj->mbj", H.ab.ovvv, R.ab, optimize=True)
                    +np.einsum("mbfj,f->mbj", H.ab.ovvo, R.a, optimize=True)
    )
    # x2b(am~j~)
    X["ab"]["voo"] = (
                    np.einsum("nmfe,afenj->amj", H.ab.oovv, R.aab, optimize=True)
                    +0.5*np.einsum("mnef,aefjn->amj", H.bb.oovv, R.abb, optimize=True)
                    +np.einsum("nmfj,afn->amj", H.ab.oovo, R.aa, optimize=True)
                    +np.einsum("mnjf,afn->amj", H.bb.ooov, R.ab, optimize=True)
                    +np.einsum("amef,efj->amj", H.ab.vovv, R.ab, optimize=True)
                    +np.einsum("amej,e->amj", H.ab.vovo, R.a, optimize=True)
    )

    # x2a(abe)
    X["aa"]["vvv"] = (
                    -0.25*np.einsum("mnef,abfmn->abe", H.aa.oovv, R.aaa, optimize=True)
                    -0.5*np.einsum("mnef,abfmn->abe", H.ab.oovv, R.aab, optimize=True)
                    +np.einsum("bnef,afn->abe", H.aa.vovv, R.aa, optimize=True)
                    +np.einsum("bnef,afn->abe", H.ab.vovv, R.ab, optimize=True)
                    +0.5*np.einsum("abfe,f->abe", H.aa.vvvv, R.a, optimize=True)
    )
    X["aa"]["vvv"] -= np.transpose(X["aa"]["vvv"], (1, 0, 2))
    # x2b(ab~e~)
    X["ab"]["vvv"] = (
                    -np.einsum("nmfe,afbnm->abe", H.ab.oovv, R.aab, optimize=True)
                    -0.5*np.einsum("mnef,abfmn->abe", H.bb.oovv, R.abb, optimize=True)
                    +np.einsum("nbfe,afn->abe", H.ab.ovvv, R.aa, optimize=True)
                    +np.einsum("bnef,afn->abe", H.bb.vovv, R.ab, optimize=True)
                    -np.einsum("amfe,fbm->abe", H.ab.vovv, R.ab, optimize=True)
                    +np.einsum("abfe,f->abe", H.ab.vvvv, R.a, optimize=True)
    )

    # additional intermediates for T3
    X["aa"]["vvo"] = (
            np.einsum("amfe,f->aem", H.aa.vovv, R.a, optimize=True)
            +np.einsum("mnef,afn->aem", H.aa.oovv, R.aa, optimize=True)
            +np.einsum("mnef,afn->aem", H.ab.oovv, R.ab, optimize=True)
    )
    X["ab"]["vvo"] = (
            np.einsum("amfe,f->aem", H.ab.vovv, R.a, optimize=True)
            +np.einsum("nmfe,afn->aem", H.ab.oovv, R.aa, optimize=True)
            +np.einsum("mnef,afn->aem", H.bb.oovv, R.ab, optimize=True)
    )
    X["ab"]["ovv"] = (
            np.einsum("mbef,e->mbf", H.ab.ovvv, R.a, optimize=True)
            -np.einsum("mnef,ebn->mbf", H.ab.oovv, R.ab, optimize=True)
    )
    X["aa"]["ooo"] = (
            np.einsum("nmje,e->mnj", H.aa.ooov, R.a, optimize=True)
            +0.5 * np.einsum("mnef,efj->mnj", H.aa.oovv, R.aa, optimize=True)
    )
    X["ab"]["ooo"] = (
            np.einsum("mnej,e->mnj", H.ab.oovo, R.a, optimize=True)
            +np.einsum("mnef,efj->mnj", H.ab.oovv, R.ab, optimize=True)
    )

    return X

def add_o_term(X, H, R):
    # add h(ov) * R1 term to X["a"]["o"] intermediates
    X["a"]["o"] += np.einsum("me,e->m", H.a.ov, R.a, optimize=True)
    return X
