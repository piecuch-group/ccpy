import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum
from ccpy.lib.core import deaeom4_p_intermediates

def get_deaeom4_intermediates(H, R):

    # Create dictionary to store intermediates, which have spincases that resemble those of the DEA R operator itself
    X = {"ab": {"vo": np.array([0.0]), "ov": np.array([0.0]), "oo": np.array([0.0])},
         "aba": {"vvvv": np.array([0.0]), "vvoo": np.array([0.0]), "vovo": np.array([0.0])},
         "abb": {"vvvv": np.array([0.0]), "vvoo": np.array([0.0]), "ovvo": np.array([0.0])}}

    # x(mb~)
    X["ab"]["ov"] = (
            ccpy_einsum("mbef,ef->mb", H.ab.ovvv, R.ab)
            + 0.5 * ccpy_einsum("mnef,ebfn->mb", H.aa.oovv, R.aba)
            + ccpy_einsum("mnef,ebfn->mb", H.ab.oovv, R.abb)
    )
    # x(am~)
    X["ab"]["vo"] = (
            ccpy_einsum("amef,ef->am", H.ab.vovv, R.ab)
            + 0.5 * ccpy_einsum("nmfe,aefn->am", H.bb.oovv, R.abb)
            + ccpy_einsum("nmfe,aefn->am", H.ab.oovv, R.aba)
    )
    # x(mn~)
    X["ab"]["oo"] = ccpy_einsum("mnef,ef->mn", H.ab.oovv, R.ab)

    # x(ab~mk) [1]
    X["aba"]["vvoo"] = (
        # h2a(mnkf) r_aba(ab~fn)
        ccpy_einsum("mnkf,abfn->abmk", H.aa.ooov, R.aba)
        # h2b(mn~kf~) r_abb(ab~f~n~)
        + ccpy_einsum("mnkf,abfn->abmk", H.ab.ooov, R.abb)
        # 1/2 h2a(amef) r_aba(eb~fk)
        + 0.5 * ccpy_einsum("amef,ebfk->abmk", H.aa.vovv, R.aba)
        # h2b(mb~ef~) r_aba(af~ek)
        + ccpy_einsum("mbef,afek->abmk", H.ab.ovvv, R.aba)
        # h2a(amek) r_ab(eb~) -> -h2a(amke) r_ab(eb~)
        - ccpy_einsum("amke,eb->abmk", H.aa.voov, R.ab)
        # h2b(mb~ke~) r_ab(ae~)
        + ccpy_einsum("mbke,ae->abmk", H.ab.ovov, R.ab)
        # 1/2 h2a(mnef) r_abaa(ab~efkn)
        + 0.5 * ccpy_einsum("mnef,abefkn->abmk", H.aa.oovv, R.abaa)
        # h2b(mn~ef~) r_abab(ab~ef~kn~)
        + ccpy_einsum("mnef,abefkn->abmk", H.ab.oovv, R.abab)
    )
    # x(ab~ce) [2]
    X["aba"]["vvvv"] = (
            # A(ac) h2a(cnef) r_aba(ab~fn)
            ccpy_einsum("cnef,abfn->abce", H.aa.vovv, R.aba)
            # A(ac) h2b(cn~ef~) r_abb(ab~f~n~)
            + ccpy_einsum("cnef,abfn->abce", H.ab.vovv, R.abb)
            # -h2b(mb~ef~) r_aba(af~cm)
            - 0.5 * ccpy_einsum("mbef,afcm->abce", H.ab.ovvv, R.aba)
            # A(ac) h2b(cb~ef~) r_ab(af~)
            + ccpy_einsum("cbef,af->abce", H.ab.vvvv, R.ab)
            # h2a(acfe) r_ab(fb~)
            + 0.5 * ccpy_einsum("acfe,fb->abce", H.aa.vvvv, R.ab)
            # -1/2 h2a(mnef) r_abaa(ab~cfmn)
            - 0.25 * ccpy_einsum("mnef,abcfmn->abce", H.aa.oovv, R.abaa)
            # -h2b(mn~ef~) r_abab(ab~cf~mn~)
            - 0.5 * ccpy_einsum("mnef,abcfmn->abce", H.ab.oovv, R.abab)
    )
    X["aba"]["vvvv"] -= np.transpose(X["aba"]["vvvv"], (2, 1, 0, 3)) # antisymmetrize A(ac)
    # x_aba(am~ck) [3]
    X["aba"]["vovo"] = (
        # A(ac) h2b(am~ef~) r_aba(ef~ck)
        ccpy_einsum("amef,efck->amck", H.ab.vovv, R.aba)
        # -h2b(nm~ke~) r_aba(ae~cn)
        - 0.5 * ccpy_einsum("nmke,aecn->amck", H.ab.ooov, R.aba)
        # A(ac) h2b(cm~kf~) r_ab(af~)
        + ccpy_einsum("cmkf,af->amck", H.ab.voov, R.ab)
        # 1/2 h2c(m~n~e~f~) r_abab(ae~cf~kn~)
        + 0.25 * ccpy_einsum("mnef,aecfkn->amck", H.bb.oovv, R.abab)
        # h2b(nm~fe~) r_abaa(ae~cfkn)
        + 0.5 * ccpy_einsum("nmfe,aecfkn->amck", H.ab.oovv, R.abaa)
    )
    X["aba"]["vovo"] -= np.transpose(X["aba"]["vovo"], (2, 1, 0, 3)) # antisymmetrize A(ac)
    # x(ab~m~k~) [4]
    X["abb"]["vvoo"] = (
        # h2b(nm~fk~) r_aba(ab~fn)
        ccpy_einsum("nmfk,abfn->abmk", H.ab.oovo, R.aba)
        # h2c(m~n~k~f~) r_abb(ab~f~n~)
        + ccpy_einsum("mnkf,abfn->abmk", H.bb.ooov, R.abb)
        # 1/2 h2c(b~m~e~f~) r_abb(ae~f~k~)
        + 0.5 * ccpy_einsum("bmef,aefk->abmk", H.bb.vovv, R.abb)
        # h2b(am~ef~) r_abb(eb~f~k~)
        + ccpy_einsum("amef,ebfk->abmk", H.ab.vovv, R.abb)
        # h2b(am~fk~) r_ab(fb~)
        + ccpy_einsum("amfk,fb->abmk", H.ab.vovo, R.ab)
        # h2c(b~m~f~k~) r_ab(af~) -> -h2c(b~m~k~f~) r_ab(af~)
        - ccpy_einsum("bmkf,af->abmk", H.bb.voov, R.ab)
        # 1/2 h2c(m~n~e~f~) r_abbb(ab~e~f~k~n~)
        + 0.5 * ccpy_einsum("mnef,abefkn->abmk", H.bb.oovv, R.abbb)
        # h2b(nm~fe~) r_abab(ab~fe~nk~)
        + ccpy_einsum("nmfe,abfenk->abmk", H.ab.oovv, R.abab)
    )
    # x(ab~c~e~) [5]
    X["abb"]["vvvv"] = (
            # A(bc) h2b(nc~fe~) r_aba(ab~fn)
            ccpy_einsum("ncfe,abfn->abce", H.ab.ovvv, R.aba)
            # A(bc) h2c(c~n~e~f~) r_abb(ab~f~n~)
            + ccpy_einsum("cnef,abfn->abce", H.bb.vovv, R.abb)
            # -h2b(an~fe~) r_abb(fb~c~n~)
            - 0.5 * ccpy_einsum("anfe,fbcn->abce", H.ab.vovv, R.abb)
            # h2c(b~c~f~e~) r_ab(af~)
            + 0.5 * ccpy_einsum("bcfe,af->abce", H.bb.vvvv, R.ab)
            # A(bc) h2b(ac~fe~) r_ab(fb~)
            + ccpy_einsum("acfe,fb->abce", H.ab.vvvv, R.ab)
            # -1/2 h2c(mnef) r_abbb(ab~c~f~m~n~)
            - 0.25 * ccpy_einsum("mnef,abcfmn->abce", H.bb.oovv, R.abbb)
            # -h2b(nm~fe~) r_abab(ab~fc~nm~)
            - 0.5 * ccpy_einsum("nmfe,abfcnm->abce", H.ab.oovv, R.abab)
    )
    X["abb"]["vvvv"] -= np.transpose(X["abb"]["vvvv"], (0, 2, 1, 3)) # antisymmetrize A(bc)
    # x_abb(mb~d~l~) [6]
    X["abb"]["ovvo"] = (
        # -h2b(mn~el~) r_abb(eb~d~n)
        - 0.5 * ccpy_einsum("mnel,ebdn->mbdl", H.ab.oovo, R.abb)
        # A(bd) h2b(mb~ef~) r_abb(ef~d~l~)
        + ccpy_einsum("mbef,efdl->mbdl", H.ab.ovvv, R.abb)
        # A(bd) h2b(md~el~) r_ab(eb~)
        + ccpy_einsum("mdel,eb->mbdl", H.ab.ovvo, R.ab)
        # 1/2 h2a(mnef) r_abab(eb~fd~nl~)
        + 0.25 * ccpy_einsum("mnef,ebfdnl->mbdl", H.aa.oovv, R.abab)
        # h2b(mn~ef~) r_abbb(eb~f~d~n~l~)
        + 0.5 * ccpy_einsum("mnef,ebfdnl->mbdl", H.ab.oovv, R.abbb)
    )
    X["abb"]["ovvo"] -= np.transpose(X["abb"]["ovvo"], (0, 2, 1, 3))  # antisymmetrize A(bd)

    return X

def get_deaeom4_p_intermediates(H, T, R, r3_excitations):

    # Create dictionary to store intermediates, which have spincases that resemble those of the DEA R operator itself
    X = {"ab": {"vo": np.array([0.0]), "ov": np.array([0.0]), "oo": np.array([0.0])},
         "aba": {"vvvv": np.array([0.0]), "vvoo": np.array([0.0]), "vovo": np.array([0.0])},
         "abb": {"vvvv": np.array([0.0]), "vvoo": np.array([0.0]), "ovvo": np.array([0.0])}}

    # x(mb~)
    X["ab"]["ov"] = (
            ccpy_einsum("mbef,ef->mb", H.ab.ovvv, R.ab)
            + 0.5 * ccpy_einsum("mnef,ebfn->mb", H.aa.oovv, R.aba)
            + ccpy_einsum("mnef,ebfn->mb", H.ab.oovv, R.abb)
    )
    # x(am~)
    X["ab"]["vo"] = (
            ccpy_einsum("amef,ef->am", H.ab.vovv, R.ab)
            + 0.5 * ccpy_einsum("nmfe,aefn->am", H.bb.oovv, R.abb)
            + ccpy_einsum("nmfe,aefn->am", H.ab.oovv, R.aba)
    )
    # x(mn~)
    X["ab"]["oo"] = ccpy_einsum("mnef,ef->mn", H.ab.oovv, R.ab)

    # x(ab~mk) [1]
    X["aba"]["vvoo"] = (
        # h2a(mnkf) r_aba(ab~fn)
        ccpy_einsum("mnkf,abfn->abmk", H.aa.ooov, R.aba)
        # h2b(mn~kf~) r_abb(ab~f~n~)
        + ccpy_einsum("mnkf,abfn->abmk", H.ab.ooov, R.abb)
        # 1/2 h2a(amef) r_aba(eb~fk)
        + 0.5 * ccpy_einsum("amef,ebfk->abmk", H.aa.vovv, R.aba)
        # h2b(mb~ef~) r_aba(af~ek)
        + ccpy_einsum("mbef,afek->abmk", H.ab.ovvv, R.aba)
        # h2a(amek) r_ab(eb~) -> -h2a(amke) r_ab(eb~)
        - ccpy_einsum("amke,eb->abmk", H.aa.voov, R.ab)
        # h2b(mb~ke~) r_ab(ae~)
        + ccpy_einsum("mbke,ae->abmk", H.ab.ovov, R.ab)
    )
    # X["aba"]["vvoo"] -= 0.5 * ccpy_einsum("mn,cbkn->cbkm", X["ab"]["oo"], T.ab)
    X["aba"]["vvoo"] = deaeom4_p_intermediates.add_r4_x3b_vvoo(X["aba"]["vvoo"],
                                                               R.abaa, r3_excitations["abaa"],
                                                               R.abab, r3_excitations["abab"],
                                                               H.aa.oovv, H.ab.oovv)


    # x(ab~ce) [2]
    X["aba"]["vvvv"] = (
            # A(ac) h2a(cnef) r_aba(ab~fn)
            ccpy_einsum("cnef,abfn->abce", H.aa.vovv, R.aba)
            # A(ac) h2b(cn~ef~) r_abb(ab~f~n~)
            + ccpy_einsum("cnef,abfn->abce", H.ab.vovv, R.abb)
            # -h2b(mb~ef~) r_aba(af~cm)
            - 0.5 * ccpy_einsum("mbef,afcm->abce", H.ab.ovvv, R.aba)
            # A(ac) h2b(cb~ef~) r_ab(af~)
            + ccpy_einsum("cbef,af->abce", H.ab.vvvv, R.ab)
            # h2a(acfe) r_ab(fb~)
            + 0.5 * ccpy_einsum("acfe,fb->abce", H.aa.vvvv, R.ab)
    )
    X["aba"]["vvvv"] -= np.transpose(X["aba"]["vvvv"], (2, 1, 0, 3)) # antisymmetrize A(ac)
    X["aba"]["vvvv"] = deaeom4_p_intermediates.add_r4_x3b_vvvv(X["aba"]["vvvv"],
                                                               R.abaa, r3_excitations["abaa"],
                                                               R.abab, r3_excitations["abab"],
                                                               H.aa.oovv, H.ab.oovv)

    # x_aba(am~ck) [3]
    X["aba"]["vovo"] = (
        # A(ac) h2b(am~ef~) r_aba(ef~ck)
        ccpy_einsum("amef,efck->amck", H.ab.vovv, R.aba)
        # -h2b(nm~ke~) r_aba(ae~cn)
        - 0.5 * ccpy_einsum("nmke,aecn->amck", H.ab.ooov, R.aba)
        # A(ac) h2b(cm~kf~) r_ab(af~)
        + ccpy_einsum("cmkf,af->amck", H.ab.voov, R.ab)
    )
    X["aba"]["vovo"] -= np.transpose(X["aba"]["vovo"], (2, 1, 0, 3)) # antisymmetrize A(ac)
    X["aba"]["vovo"] = deaeom4_p_intermediates.add_r4_x3b_vovo(X["aba"]["vovo"],
                                                               R.abaa, r3_excitations["abaa"],
                                                               R.abab, r3_excitations["abab"],
                                                               H.ab.oovv, H.bb.oovv)

    # x(ab~m~k~) [4]
    X["abb"]["vvoo"] = (
        # h2b(nm~fk~) r_aba(ab~fn)
        ccpy_einsum("nmfk,abfn->abmk", H.ab.oovo, R.aba)
        # h2c(m~n~k~f~) r_abb(ab~f~n~)
        + ccpy_einsum("mnkf,abfn->abmk", H.bb.ooov, R.abb)
        # 1/2 h2c(b~m~e~f~) r_abb(ae~f~k~)
        + 0.5 * ccpy_einsum("bmef,aefk->abmk", H.bb.vovv, R.abb)
        # h2b(am~ef~) r_abb(eb~f~k~)
        + ccpy_einsum("amef,ebfk->abmk", H.ab.vovv, R.abb)
        # h2b(am~fk~) r_ab(fb~)
        + ccpy_einsum("amfk,fb->abmk", H.ab.vovo, R.ab)
        # h2c(b~m~f~k~) r_ab(af~) -> -h2c(b~m~k~f~) r_ab(af~)
        - ccpy_einsum("bmkf,af->abmk", H.bb.voov, R.ab)
    )
    # X["abb"]["vvoo"] -= 0.5 * ccpy_einsum("mn,bdnl->bdml", X["ab"]["oo"], T.bb)
    X["abb"]["vvoo"] = deaeom4_p_intermediates.add_r4_x3c_vvoo(X["abb"]["vvoo"],
                                                               R.abab, r3_excitations["abab"],
                                                               R.abbb, r3_excitations["abbb"],
                                                               H.ab.oovv, H.bb.oovv)

    # x(ab~c~e~) [5]
    X["abb"]["vvvv"] = (
            # A(bc) h2b(nc~fe~) r_aba(ab~fn)
            ccpy_einsum("ncfe,abfn->abce", H.ab.ovvv, R.aba)
            # A(bc) h2c(c~n~e~f~) r_abb(ab~f~n~)
            + ccpy_einsum("cnef,abfn->abce", H.bb.vovv, R.abb)
            # -h2b(an~fe~) r_abb(fb~c~n~)
            - 0.5 * ccpy_einsum("anfe,fbcn->abce", H.ab.vovv, R.abb)
            # h2c(b~c~f~e~) r_ab(af~)
            + 0.5 * ccpy_einsum("bcfe,af->abce", H.bb.vvvv, R.ab)
            # A(bc) h2b(ac~fe~) r_ab(fb~)
            + ccpy_einsum("acfe,fb->abce", H.ab.vvvv, R.ab)
    )
    X["abb"]["vvvv"] -= np.transpose(X["abb"]["vvvv"], (0, 2, 1, 3)) # antisymmetrize A(bc)
    X["abb"]["vvvv"] = deaeom4_p_intermediates.add_r4_x3c_vvvv(X["abb"]["vvvv"],
                                                               R.abab, r3_excitations["abab"],
                                                               R.abbb, r3_excitations["abbb"],
                                                               H.ab.oovv, H.bb.oovv)

    # x_abb(mb~d~l~) [6]
    X["abb"]["ovvo"] = (
        # -h2b(mn~el~) r_abb(eb~d~n)
        - 0.5 * ccpy_einsum("mnel,ebdn->mbdl", H.ab.oovo, R.abb)
        # A(bd) h2b(mb~ef~) r_abb(ef~d~l~)
        + ccpy_einsum("mbef,efdl->mbdl", H.ab.ovvv, R.abb)
        # A(bd) h2b(md~el~) r_ab(eb~)
        + ccpy_einsum("mdel,eb->mbdl", H.ab.ovvo, R.ab)
    )
    X["abb"]["ovvo"] -= np.transpose(X["abb"]["ovvo"], (0, 2, 1, 3))  # antisymmetrize A(bd)
    X["abb"]["ovvo"] = deaeom4_p_intermediates.add_r4_x3c_ovvo(X["abb"]["ovvo"],
                                                               R.abab, r3_excitations["abab"],
                                                               R.abbb, r3_excitations["abbb"],
                                                               H.aa.oovv, H.ab.oovv)

    return X
