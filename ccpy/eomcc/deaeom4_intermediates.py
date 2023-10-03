import numpy as np

def get_deaeom4_intermediates(H, R):

    # Create dictionary to store intermediates, which have spincases that resemble those of the DEA R operator itself
    X = {"ab": {"vo": np.array([0.0]), "ov": np.array([0.0]), "oo": np.array([0.0])},
         "aba": {"vvvv": np.array([0.0]), "vvoo": np.array([0.0]), "vovo": np.array([0.0])},
         "abb": {"vvvv": np.array([0.0]), "vvoo": np.array([0.0]), "ovvo": np.array([0.0])}}

    # x(mb~)
    X["ab"]["ov"] = (
            np.einsum("mbef,ef->mb", H.ab.ovvv, R.ab, optimize=True)
            + 0.5 * np.einsum("mnef,ebfn->mb", H.aa.oovv, R.aba, optimize=True)
            + np.einsum("mnef,ebfn->mb", H.ab.oovv, R.abb, optimize=True)
    )
    # x(am~)
    X["ab"]["vo"] = (
            np.einsum("amef,ef->am", H.ab.vovv, R.ab, optimize=True)
            + 0.5 * np.einsum("nmfe,aefn->am", H.bb.oovv, R.abb, optimize=True)
            + np.einsum("nmfe,aefn->am", H.ab.oovv, R.aba, optimize=True)
    )
    # x(mn~)
    X["ab"]["oo"] = np.einsum("mnef,ef->mn", H.ab.oovv, R.ab, optimize=True)

    # x(ab~mk) [1]
    X["aba"]["vvoo"] = (
        # h2a(mnkf) r_aba(ab~fn)
        np.einsum("mnkf,abfn->abmk", H.aa.ooov, R.aba, optimize=True)
        # h2b(mn~kf~) r_abb(ab~f~n~)
        + np.einsum("mnkf,abfn->abmk", H.ab.ooov, R.abb, optimize=True)
        # 1/2 h2a(amef) r_aba(eb~fk)
        + 0.5 * np.einsum("amef,ebfk->abmk", H.aa.vovv, R.aba, optimize=True)
        # h2b(mb~ef~) r_aba(af~ek)
        + np.einsum("mbef,afek->abmk", H.ab.ovvv, R.aba, optimize=True)
        # h2a(amek) r_ab(eb~) -> -h2a(amke) r_ab(eb~)
        - np.einsum("amke,eb->abmk", H.aa.voov, R.ab, optimize=True)
        # h2b(mb~ke~) r_ab(ae~)
        + np.einsum("mbke,ae->abmk", H.ab.ovov, R.ab, optimize=True)
        # 1/2 h2a(mnef) r_abaa(ab~efkn)
        + 0.5 * np.einsum("mnef,abefkn->abmk", H.aa.oovv, R.abaa, optimize=True)
        # h2b(mn~ef~) r_abab(ab~ef~kn~)
        + np.einsum("mnef,abefkn->abmk", H.ab.oovv, R.abab, optimize=True)
    )
    # x(ab~ce) [2]
    X["aba"]["vvvv"] = (
            # A(ac) h2a(cnef) r_aba(ab~fn)
            np.einsum("cnef,abfn->abce", H.aa.vovv, R.aba, optimize=True)
            # A(ac) h2b(cn~ef~) r_abb(ab~f~n~)
            + np.einsum("cnef,abfn->abce", H.ab.vovv, R.abb, optimize=True)
            # -h2b(mb~ef~) r_aba(af~cm)
            - 0.5 * np.einsum("mbef,afcm->abce", H.ab.ovvv, R.aba, optimize=True)
            # A(ac) h2b(cb~ef~) r_ab(af~)
            + np.einsum("cbef,af->abce", H.ab.vvvv, R.ab, optimize=True)
            # h2a(acfe) r_ab(fb~)
            + 0.5 * np.einsum("acfe,fb->abce", H.aa.vvvv, R.ab, optimize=True)
            # -1/2 h2a(mnef) r_abaa(ab~cfmn)
            - 0.25 * np.einsum("mnef,abcfmn->abce", H.aa.oovv, R.abaa, optimize=True)
            # -h2b(mn~ef~) r_abab(ab~cf~mn~)
            - 0.5 * np.einsum("mnef,abcfmn->abce", H.ab.oovv, R.abab, optimize=True)
    )
    X["aba"]["vvvv"] -= np.transpose(X["aba"]["vvvv"], (2, 1, 0, 3)) # antisymmetrize A(ac)
    # x_aba(am~ck) [3]
    X["aba"]["vovo"] = (
        # A(ac) h2b(am~ef~) r_aba(ef~ck)
        np.einsum("amef,efck->amck", H.ab.vovv, R.aba, optimize=True)
        # -h2b(nm~ke~) r_aba(ae~cn)
        - 0.5 * np.einsum("nmke,aecn->amck", H.ab.ooov, R.aba, optimize=True)
        # A(ac) h2b(cm~kf~) r_ab(af~)
        + np.einsum("cmkf,af->amck", H.ab.voov, R.ab, optimize=True)
        # 1/2 h2c(m~n~e~f~) r_abab(ae~cf~kn~)
        + 0.25 * np.einsum("mnef,aecfkn->amck", H.bb.oovv, R.abab, optimize=True)
        # h2b(nm~fe~) r_abaa(ae~cfkn)
        + 0.5 * np.einsum("nmfe,aecfkn->amck", H.ab.oovv, R.abaa, optimize=True)
    )
    X["aba"]["vovo"] -= np.transpose(X["aba"]["vovo"], (2, 1, 0, 3)) # antisymmetrize A(ac)
    # x(ab~m~k~) [4]
    X["abb"]["vvoo"] = (
        # h2b(nm~fk~) r_aba(ab~fn)
        np.einsum("nmfk,abfn->abmk", H.ab.oovo, R.aba, optimize=True)
        # h2c(m~n~k~f~) r_abb(ab~f~n~)
        + np.einsum("mnkf,abfn->abmk", H.bb.ooov, R.abb, optimize=True)
        # 1/2 h2c(b~m~e~f~) r_abb(ae~f~k~)
        + 0.5 * np.einsum("bmef,aefk->abmk", H.bb.vovv, R.abb, optimize=True)
        # h2b(am~ef~) r_abb(eb~f~k~)
        + np.einsum("amef,ebfk->abmk", H.ab.vovv, R.abb, optimize=True)
        # h2b(am~fk~) r_ab(fb~)
        + np.einsum("amfk,fb->abmk", H.ab.vovo, R.ab, optimize=True)
        # h2c(b~m~f~k~) r_ab(af~) -> -h2c(b~m~k~f~) r_ab(af~)
        - np.einsum("bmkf,af->abmk", H.bb.voov, R.ab, optimize=True)
        # 1/2 h2c(m~n~e~f~) r_abbb(ab~e~f~k~n~)
        + 0.5 * np.einsum("mnef,abefkn->abmk", H.bb.oovv, R.abbb, optimize=True)
        # h2b(nm~fe~) r_abab(ab~fe~nk~)
        + np.einsum("nmfe,abfenk->abmk", H.ab.oovv, R.abab, optimize=True)
    )
    # x(ab~c~e~) [5]
    X["abb"]["vvvv"] = (
            # A(bc) h2b(nc~fe~) r_aba(ab~fn)
            np.einsum("ncfe,abfn->abce", H.ab.ovvv, R.aba, optimize=True)
            # A(bc) h2c(c~n~e~f~) r_abb(ab~f~n~)
            + np.einsum("cnef,abfn->abce", H.bb.vovv, R.abb, optimize=True)
            # -h2b(an~fe~) r_abb(fb~c~n~)
            - 0.5 * np.einsum("anfe,fbcn->abce", H.ab.vovv, R.abb, optimize=True)
            # h2c(b~c~f~e~) r_ab(af~)
            + 0.5 * np.einsum("bcfe,af->abce", H.bb.vvvv, R.ab, optimize=True)
            # A(bc) h2b(ac~fe~) r_ab(fb~)
            + np.einsum("acfe,fb->abce", H.ab.vvvv, R.ab, optimize=True)
            # -1/2 h2c(mnef) r_abbb(ab~c~f~m~n~)
            - 0.25 * np.einsum("mnef,abcfmn->abce", H.bb.oovv, R.abbb, optimize=True)
            # -h2b(nm~fe~) r_abab(ab~fc~nm~)
            - 0.5 * np.einsum("nmfe,abfcnm->abce", H.ab.oovv, R.abab, optimize=True)
    )
    X["abb"]["vvvv"] -= np.transpose(X["abb"]["vvvv"], (0, 2, 1, 3)) # antisymmetrize A(bc)
    # x_abb(mb~d~l~) [6]
    X["abb"]["ovvo"] = (
        # -h2b(mn~el~) r_abb(eb~d~n)
        - 0.5 * np.einsum("mnel,ebdn->mbdl", H.ab.oovo, R.abb, optimize=True)
        # A(bd) h2b(mb~ef~) r_abb(ef~d~l~)
        + np.einsum("mbef,efdl->mbdl", H.ab.ovvv, R.abb, optimize=True)
        # A(bd) h2b(md~el~) r_ab(eb~)
        + np.einsum("mdel,eb->mbdl", H.ab.ovvo, R.ab, optimize=True)
        # 1/2 h2a(mnef) r_abab(eb~fd~nl~)
        + 0.25 * np.einsum("mnef,ebfdnl->mbdl", H.aa.oovv, R.abab, optimize=True)
        # h2b(mn~ef~) r_abbb(eb~f~d~n~l~)
        + 0.5 * np.einsum("mnef,ebfdnl->mbdl", H.ab.oovv, R.abbb, optimize=True)
    )
    X["abb"]["ovvo"] -= np.transpose(X["abb"]["ovvo"], (0, 2, 1, 3))  # antisymmetrize A(bd)

    return X
