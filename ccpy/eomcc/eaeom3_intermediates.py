import numpy as np

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
                    +np.einsum("amje,e->amj", H.aa.voov, R.a, optimize=True) # what do we do about this one???
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
