import numpy as np
from ccpy.utilities.updates import ipeom3_p_intermediates

def get_ipeom3_intermediates(H, R):

    # These intermediates will be 3-index quantities, which are not
    # set up in the models at the moment. We will just use a dictionary
    # as a workaround for now.

    X = {"aa" : {}, "ab" : {}}

    # x2a(ibe)
    X["aa"]["ovv"] = (
            -0.5 * np.einsum("mnef,ibfmn->ibe", H.aa.oovv, R.aaa, optimize=True)
            -np.einsum("mnef,ibfmn->ibe", H.ab.oovv, R.aab, optimize=True)
            +np.einsum("bnef,ifn->ibe", H.aa.vovv, R.aa, optimize=True)
            +np.einsum("bnef,ifn->ibe", H.ab.vovv, R.ab, optimize=True)
            +0.5 * np.einsum("nmie,nbm->ibe", H.aa.ooov, R.aa, optimize=True)
            +np.einsum("bmie,m->ibe", H.aa.voov, R.a, optimize=True)
    )
    # x2b(eb~j~)
    X["ab"]["vvo"] = (
            -0.5 * np.einsum("mnef,mfbnj->ebj", H.aa.oovv, R.aab, optimize=True)
            -np.einsum("mnef,mbfjn->ebj", H.ab.oovv, R.abb, optimize=True)
            -np.einsum("mbef,mfj->ebj", H.ab.ovvv, R.ab, optimize=True)
            +np.einsum("mnej,mbn->ebj", H.ab.oovo, R.ab, optimize=True)
            -np.einsum("mbej,m->ebj", H.ab.ovvo, R.a, optimize=True)
    )
    # x2b(ib~e~)
    X["ab"]["ovv"] = (
            -np.einsum("nmfe,ifbnm->ibe", H.ab.oovv, R.aab, optimize=True)
            -0.5 * np.einsum("mnef,ibfmn->ibe", H.bb.oovv, R.abb, optimize=True)
            +np.einsum("nbfe,ifn->ibe", H.ab.ovvv, R.aa, optimize=True)
            +np.einsum("bnef,ifn->ibe", H.bb.vovv, R.ab, optimize=True)
            +np.einsum("nmie,nbm->ibe", H.ab.ooov, R.ab, optimize=True)
            -np.einsum("mbie,m->ibe", H.ab.ovov, R.a, optimize=True)
    )

    # x2a(imj)
    X["aa"]["ooo"] = (
             0.25 * np.einsum("mnef,iefjn->imj", H.aa.oovv, R.aaa, optimize=True)
            +0.5 * np.einsum("mnef,iefjn->imj", H.ab.oovv, R.aab, optimize=True)
            +np.einsum("mnjf,ifn->imj", H.aa.ooov, R.aa, optimize=True)
            +np.einsum("mnjf,ifn->imj", H.ab.ooov, R.ab, optimize=True)
            -0.5 * np.einsum("mnji,n->imj", H.aa.oooo, R.a, optimize=True)
    )
    X["aa"]["ooo"] -= np.transpose(X["aa"]["ooo"], (2, 1, 0))
    # x2b(im~j~)
    X["ab"]["ooo"] = (
             np.einsum("nmfe,ifenj->imj", H.ab.oovv, R.aab, optimize=True)
            +0.5 * np.einsum("mnef,iefjn->imj", H.bb.oovv, R.abb, optimize=True)
            +np.einsum("nmfj,ifn->imj", H.ab.oovo, R.aa, optimize=True)
            +np.einsum("mnjf,ifn->imj", H.bb.ooov, R.ab, optimize=True)
            -np.einsum("nmie,nej->imj", H.ab.ooov, R.ab, optimize=True)
            -np.einsum("nmij,n->imj", H.ab.oooo, R.a, optimize=True)
    )

    return X

def get_ipeom3_p_intermediates(H, R, R3_excitations):

    # These intermediates will be 3-index quantities, which are not
    # set up in the models at the moment. We will just use a dictionary
    # as a workaround for now.

    X = {"aa" : {}, "ab" : {}}

    # x2a(ibe)
    X["aa"]["ovv"] = (
            +np.einsum("bnef,ifn->ibe", H.aa.vovv, R.aa, optimize=True)
            +np.einsum("bnef,ifn->ibe", H.ab.vovv, R.ab, optimize=True)
            +0.5 * np.einsum("nmie,nbm->ibe", H.aa.ooov, R.aa, optimize=True)
            +np.einsum("bmie,m->ibe", H.aa.voov, R.a, optimize=True)
    )
    X["aa"]["ovv"] = ipeom3_p_intermediates.ipeom3_p_intermediates.add_r3_x2a_ovv(X["aa"]["ovv"],
                                                                                  R.aaa, R3_excitations["aaa"],
                                                                                  R.aab, R3_excitations["aab"],
                                                                                  H.aa.oovv, H.ab.oovv)
    # x2a(imj)
    X["aa"]["ooo"] = (
            +np.einsum("mnjf,ifn->imj", H.aa.ooov, R.aa, optimize=True)
            +np.einsum("mnjf,ifn->imj", H.ab.ooov, R.ab, optimize=True)
            -0.5 * np.einsum("mnji,n->imj", H.aa.oooo, R.a, optimize=True)
    )
    X["aa"]["ooo"] -= np.transpose(X["aa"]["ooo"], (2, 1, 0))
    X["aa"]["ooo"] = ipeom3_p_intermediates.ipeom3_p_intermediates.add_r3_x2a_ooo(X["aa"]["ooo"],
                                                                                  R.aaa, R3_excitations["aaa"],
                                                                                  R.aab, R3_excitations["aab"],
                                                                                  H.aa.oovv, H.ab.oovv)
    # x2b(eb~j~)
    X["ab"]["vvo"] = (
            -np.einsum("mbef,mfj->ebj", H.ab.ovvv, R.ab, optimize=True)
            +np.einsum("mnej,mbn->ebj", H.ab.oovo, R.ab, optimize=True)
            -np.einsum("mbej,m->ebj", H.ab.ovvo, R.a, optimize=True)
    )
    X["ab"]["vvo"] = ipeom3_p_intermediates.ipeom3_p_intermediates.add_r3_x2b_vvo(X["ab"]["vvo"],
                                                                                  R.aab, R3_excitations["aab"],
                                                                                  R.abb, R3_excitations["abb"],
                                                                                  H.aa.oovv, H.ab.oovv)
    # x2b(ib~e~)
    X["ab"]["ovv"] = (
            +np.einsum("nbfe,ifn->ibe", H.ab.ovvv, R.aa, optimize=True)
            +np.einsum("bnef,ifn->ibe", H.bb.vovv, R.ab, optimize=True)
            +np.einsum("nmie,nbm->ibe", H.ab.ooov, R.ab, optimize=True)
            -np.einsum("mbie,m->ibe", H.ab.ovov, R.a, optimize=True)
    )
    X["ab"]["ovv"] = ipeom3_p_intermediates.ipeom3_p_intermediates.add_r3_x2b_ovv(X["ab"]["ovv"],
                                                                                  R.aab, R3_excitations["aab"],
                                                                                  R.abb, R3_excitations["abb"],
                                                                                  H.ab.oovv, H.bb.oovv)
    # x2b(im~j~)
    X["ab"]["ooo"] = (
            +np.einsum("nmfj,ifn->imj", H.ab.oovo, R.aa, optimize=True)
            +np.einsum("mnjf,ifn->imj", H.bb.ooov, R.ab, optimize=True)
            -np.einsum("nmie,nej->imj", H.ab.ooov, R.ab, optimize=True)
            -np.einsum("nmij,n->imj", H.ab.oooo, R.a, optimize=True)
    )
    X["ab"]["ooo"] = ipeom3_p_intermediates.ipeom3_p_intermediates.add_r3_x2b_ooo(X["ab"]["ooo"],
                                                                                  R.aab, R3_excitations["aab"],
                                                                                  R.abb, R3_excitations["abb"],
                                                                                  H.ab.oovv, H.bb.oovv)
    return X
