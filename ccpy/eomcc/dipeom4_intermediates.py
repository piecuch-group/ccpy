import numpy as np

def get_dipeom4_intermediates(H, R):

    # Create dictionary to store intermediates, which have spincases that resemble those of the DIP R operator itself
    X = {"ab": {"vo": np.array([0.0]), "ov": np.array([0.0]), "vv": np.array([0.0])},
         "aba": {"oooo": np.array([0.0]), "oovv": np.array([0.0]), "ovvo": np.array([0.0])},
         "abb": {"oooo": np.array([0.0]), "oovv": np.array([0.0]), "vovo": np.array([0.0])}}

    ### one-body intermediates ###
    # x(ie~)
    X["ab"]["ov"] = (
            np.einsum("mnie,mn->ie", H.ab.ooov, R.ab, optimize=True)
            - np.einsum("nmfe,imfn->ie", H.ab.oovv, R.aba, optimize=True)
            - 0.5 * np.einsum("nmfe,imfn->ie", H.bb.oovv, R.abb, optimize=True)
    )
    # x(ej~)
    X["ab"]["vo"] = (
            np.einsum("mnej,mn->ej", H.ab.oovo, R.ab, optimize=True)
            - 0.5 * np.einsum("mnef,mjfn->ej", H.aa.oovv, R.aba, optimize=True)
            - np.einsum("mnef,mjfn->ej", H.ab.oovv, R.abb, optimize=True)
    )
    # x(ef~)
    X["ab"]["vv"] = np.einsum("mnef,mn->ef", H.ab.oovv, R.ab, optimize=True)

    ### two-body intermediates ###
    # x(ij~ce) [1]
    X["aba"]["oovv"] = (
            np.einsum("cnef,ijfn->ijce", H.aa.vovv, R.aba, optimize=True)
            + np.einsum("cnef,ijfn->ijce", H.ab.vovv, R.abb, optimize=True)
            + np.einsum("cmie,mj->ijce", H.aa.voov, R.ab, optimize=True) # flip sign, h2a(vovo) -> -h2a(voov)
            + np.einsum("mnej,incm->ijce", H.ab.oovo, R.aba, optimize=True)
            + 0.5 * np.einsum("mnie,mjcn->ijce", H.aa.ooov, R.aba, optimize=True)
            - np.einsum("cmej,im->ijce", H.ab.vovo, R.ab, optimize=True)
            - 0.5 * np.einsum("mnef,ijcfmn->ijce", H.aa.oovv, R.abaa, optimize=True)
            - np.einsum("mnef,ijcfmn->ijce", H.ab.oovv, R.abab, optimize=True)
    )

    # x(ij~mk) [2]
    X["aba"]["oooo"] = (
            np.einsum("mnkf,ijfn->ijmk", H.aa.ooov, R.aba, optimize=True)
            + np.einsum("mnkf,ijfn->ijmk", H.ab.ooov, R.abb, optimize=True)
            - 0.5 * np.einsum("mnej,inek->ijmk", H.ab.oovo, R.aba, optimize=True)
            - 0.5 * np.einsum("nmik,nj->ijmk", H.aa.oooo, R.ab, optimize=True)
            - np.einsum("mnkj,in->ijmk", H.ab.oooo, R.ab, optimize=True)
            + 0.25 * np.einsum("mnef,ijefkn->ijmk", H.aa.oovv, R.abaa, optimize=True)
            + 0.5 * np.einsum("mnef,ijefkn->ijmk", H.ab.oovv, R.abab, optimize=True)
    )
    # antisymmetrize A(ik)
    X["aba"]["oooo"] -= np.transpose(X["aba"]["oooo"], (3, 1, 2, 0))

    # x(ieck) [3]
    X["aba"]["ovvo"] = (
            np.einsum("nmie,nmck->ieck", H.ab.ooov, R.aba, optimize=True)
            - 0.5 * np.einsum("cmfe,imfk->ieck", H.ab.vovv, R.aba, optimize=True)
            - np.einsum("cmke,im->ieck", H.ab.voov, R.ab, optimize=True)
            - 0.5 * np.einsum("nmfe,imcfkn->ieck", H.ab.oovv, R.abaa, optimize=True)
            - 0.25 * np.einsum("mnef,imcfkn->ieck", H.bb.oovv, R.abab, optimize=True)
    )
    # antisymmetrize A(ik)
    X["aba"]["ovvo"] -= np.transpose(X["aba"]["ovvo"], (3, 1, 2, 0))

    # x(ijde) [4]
    X["abb"]["oovv"] = (
            np.einsum("ndfe,ijfn->ijde", H.ab.ovvv, R.aba, optimize=True)
            + np.einsum("dnef,ijfn->ijde", H.bb.vovv, R.abb, optimize=True)
            + np.einsum("dmje,im->ijde", H.bb.voov, R.ab, optimize=True) # flip sign, h2c(vovo) -> -h2c(voov)
            + 0.5 * np.einsum("mnje,imdn->ijde", H.bb.ooov, R.abb, optimize=True)
            + np.einsum("nmie,njdm->ijde", H.ab.ooov, R.abb, optimize=True)
            - np.einsum("mdie,mj->ijde", H.ab.ovov, R.ab, optimize=True)
            - np.einsum("nmfe,ijfdnm->ijde", H.ab.oovv, R.abab, optimize=True)
            - 0.5 * np.einsum("mnef,ijfdnm->ijde", H.bb.oovv, R.abbb, optimize=True)
    )

    # x(ij~m~k~) [5]
    X["abb"]["oooo"] = (
            np.einsum("nmfk,ijfn->ijmk", H.ab.oovo, R.aba, optimize=True)
            + np.einsum("mnkf,ijfn->ijmk", H.bb.ooov, R.abb, optimize=True)
            - np.einsum("nmik,nj->ijmk", H.ab.oooo, R.ab, optimize=True)
            - 0.5 * np.einsum("nmie,njek->ijmk", H.ab.ooov, R.abb, optimize=True)
            - 0.5 * np.einsum("nmjk,in->ijmk", H.bb.oooo, R.ab, optimize=True)
            + 0.5 * np.einsum("nmfe,ijfenk->ijmk", H.ab.oovv, R.abab, optimize=True)
            + 0.25 * np.einsum("mnef,ijfenk->ijmk", H.bb.oovv, R.abbb, optimize=True)
    )
    # antisymmetrize A(jk)
    X["abb"]["oooo"] -= np.transpose(X["abb"]["oooo"], (0, 3, 2, 1))

    # x(ejdl) [6]
    X["abb"]["vovo"] = (
            - 0.5 * np.einsum("mdef,mjfl->ejdl", H.ab.ovvv, R.abb, optimize=True)
            + np.einsum("mnej,mndl->ejdl", H.ab.oovo, R.abb, optimize=True)
            - np.einsum("mdel,mj->ejdl", H.ab.ovvo, R.ab, optimize=True)
            - 0.25 * np.einsum("mnef,mjfdnl->ejdl", H.aa.oovv, R.abab, optimize=True)
            - 0.5 * np.einsum("mnef,mjfdnl->ejdl", H.ab.oovv, R.abbb, optimize=True)
    )
    # antisymmetrize A(jl)
    X["abb"]["vovo"] -= np.transpose(X["abb"]["vovo"], (0, 3, 2, 1))

    return X

def get_dipeomccsdt_intermediates(H, R):

    # Create dictionary to store intermediates, which have spincases that resemble those of the DIP R operator itself
    X = {"ab": {"vo": np.array([0.0]), "ov": np.array([0.0]), "vv": np.array([0.0])},
         "aba": {"oooo": np.array([0.0]), "oovv": np.array([0.0]), "ovvo": np.array([0.0]),
                 "oovo": np.array([0.0]), "ovoo": np.array([0.0]),
                 "vovv": np.array([0.0]), "vvvo": np.array([0.0])},
         "abb": {"oooo": np.array([0.0]), "oovv": np.array([0.0]), "vovo": np.array([0.0]),
                 "oovo": np.array([0.0]), "vooo": np.array([0.0]),
                 "vovv": np.array([0.0]), "vvvo": np.array([0.0])}}

    ### one-body intermediates ###
    # x(ie~)
    X["ab"]["ov"] = (
            np.einsum("mnie,mn->ie", H.ab.ooov, R.ab, optimize=True)
            - np.einsum("nmfe,imfn->ie", H.ab.oovv, R.aba, optimize=True)
            - 0.5 * np.einsum("nmfe,imfn->ie", H.bb.oovv, R.abb, optimize=True)
    )
    # x(ej~)
    X["ab"]["vo"] = (
            np.einsum("mnej,mn->ej", H.ab.oovo, R.ab, optimize=True)
            - 0.5 * np.einsum("mnef,mjfn->ej", H.aa.oovv, R.aba, optimize=True)
            - np.einsum("mnef,mjfn->ej", H.ab.oovv, R.abb, optimize=True)
    )
    # x(ef~)
    X["ab"]["vv"] = np.einsum("mnef,mn->ef", H.ab.oovv, R.ab, optimize=True)

    ### two-body intermediates ###
    # x(ij~ce) [1]
    X["aba"]["oovv"] = (
            np.einsum("cnef,ijfn->ijce", H.aa.vovv, R.aba, optimize=True)
            + np.einsum("cnef,ijfn->ijce", H.ab.vovv, R.abb, optimize=True)
            + np.einsum("cmie,mj->ijce", H.aa.voov, R.ab, optimize=True) # flip sign, h2a(vovo) -> -h2a(voov)
            + np.einsum("mnej,incm->ijce", H.ab.oovo, R.aba, optimize=True)
            + 0.5 * np.einsum("mnie,mjcn->ijce", H.aa.ooov, R.aba, optimize=True)
            - np.einsum("cmej,im->ijce", H.ab.vovo, R.ab, optimize=True)
            - 0.5 * np.einsum("mnef,ijcfmn->ijce", H.aa.oovv, R.abaa, optimize=True)
            - np.einsum("mnef,ijcfmn->ijce", H.ab.oovv, R.abab, optimize=True)
    )

    # x(ij~mk) [2]
    X["aba"]["oooo"] = (
            np.einsum("mnkf,ijfn->ijmk", H.aa.ooov, R.aba, optimize=True)
            + np.einsum("mnkf,ijfn->ijmk", H.ab.ooov, R.abb, optimize=True)
            - 0.5 * np.einsum("mnej,inek->ijmk", H.ab.oovo, R.aba, optimize=True)
            - 0.5 * np.einsum("nmik,nj->ijmk", H.aa.oooo, R.ab, optimize=True)
            - np.einsum("mnkj,in->ijmk", H.ab.oooo, R.ab, optimize=True)
            + 0.25 * np.einsum("mnef,ijefkn->ijmk", H.aa.oovv, R.abaa, optimize=True)
            + 0.5 * np.einsum("mnef,ijefkn->ijmk", H.ab.oovv, R.abab, optimize=True)
    )
    # antisymmetrize A(ik)
    X["aba"]["oooo"] -= np.transpose(X["aba"]["oooo"], (3, 1, 2, 0))

    # x(ieck) [3]
    X["aba"]["ovvo"] = (
            np.einsum("nmie,nmck->ieck", H.ab.ooov, R.aba, optimize=True)
            - 0.5 * np.einsum("cmfe,imfk->ieck", H.ab.vovv, R.aba, optimize=True)
            - np.einsum("cmke,im->ieck", H.ab.voov, R.ab, optimize=True)
            - 0.5 * np.einsum("nmfe,imcfkn->ieck", H.ab.oovv, R.abaa, optimize=True)
            - 0.25 * np.einsum("mnef,imcfkn->ieck", H.bb.oovv, R.abab, optimize=True)
    )
    # antisymmetrize A(ik)
    X["aba"]["ovvo"] -= np.transpose(X["aba"]["ovvo"], (3, 1, 2, 0))

    # x(ijde) [4]
    X["abb"]["oovv"] = (
            np.einsum("ndfe,ijfn->ijde", H.ab.ovvv, R.aba, optimize=True)
            + np.einsum("dnef,ijfn->ijde", H.bb.vovv, R.abb, optimize=True)
            + np.einsum("dmje,im->ijde", H.bb.voov, R.ab, optimize=True) # flip sign, h2c(vovo) -> -h2c(voov)
            + 0.5 * np.einsum("mnje,imdn->ijde", H.bb.ooov, R.abb, optimize=True)
            + np.einsum("nmie,njdm->ijde", H.ab.ooov, R.abb, optimize=True)
            - np.einsum("mdie,mj->ijde", H.ab.ovov, R.ab, optimize=True)
            - np.einsum("nmfe,ijfdnm->ijde", H.ab.oovv, R.abab, optimize=True)
            - 0.5 * np.einsum("mnef,ijfdnm->ijde", H.bb.oovv, R.abbb, optimize=True)
    )

    # x(ij~m~k~) [5]
    X["abb"]["oooo"] = (
            np.einsum("nmfk,ijfn->ijmk", H.ab.oovo, R.aba, optimize=True)
            + np.einsum("mnkf,ijfn->ijmk", H.bb.ooov, R.abb, optimize=True)
            - np.einsum("nmik,nj->ijmk", H.ab.oooo, R.ab, optimize=True)
            - 0.5 * np.einsum("nmie,njek->ijmk", H.ab.ooov, R.abb, optimize=True)
            - 0.5 * np.einsum("nmjk,in->ijmk", H.bb.oooo, R.ab, optimize=True)
            + 0.5 * np.einsum("nmfe,ijfenk->ijmk", H.ab.oovv, R.abab, optimize=True)
            + 0.25 * np.einsum("mnef,ijfenk->ijmk", H.bb.oovv, R.abbb, optimize=True)
    )
    # antisymmetrize A(jk)
    X["abb"]["oooo"] -= np.transpose(X["abb"]["oooo"], (0, 3, 2, 1))

    # x(ejdl) [6]
    X["abb"]["vovo"] = (
            - 0.5 * np.einsum("mdef,mjfl->ejdl", H.ab.ovvv, R.abb, optimize=True)
            + np.einsum("mnej,mndl->ejdl", H.ab.oovo, R.abb, optimize=True)
            - np.einsum("mdel,mj->ejdl", H.ab.ovvo, R.ab, optimize=True)
            - 0.25 * np.einsum("mnef,mjfdnl->ejdl", H.aa.oovv, R.abab, optimize=True)
            - 0.5 * np.einsum("mnef,mjfdnl->ejdl", H.ab.oovv, R.abbb, optimize=True)
    )
    # antisymmetrize A(jl)
    X["abb"]["vovo"] -= np.transpose(X["abb"]["vovo"], (0, 3, 2, 1))

    # Intermediates [7] - [14] are only needed in DIP-EOMCCSDT to contract with T3

    # x(ij~em) [7]
    X["aba"]["oovo"] = (
        -np.einsum("mnej,in->ijem", H.ab.oovo, R.ab, optimize=True)
        -np.einsum("nmie,nj->ijem", H.aa.ooov, R.ab, optimize=True)
        +np.einsum("mnef,ijfn->ijem", H.aa.oovv, R.aba, optimize=True)
        +np.einsum("mnef,ijfn->ijem", H.ab.oovv, R.abb, optimize=True)
    )

    # x(ij~e~m~) [8]
    X["abb"]["oovo"] = (
        -np.einsum("nmje,in->ijem", H.bb.ooov, R.ab, optimize=True)
        -np.einsum("nmie,nj->ijem", H.ab.ooov, R.ab, optimize=True)
        +np.einsum("nmfe,ijfn->ijem", H.ab.oovv, R.aba, optimize=True)
        +np.einsum("mnef,ijfn->ijem", H.bb.oovv, R.abb, optimize=True)
    )

    # x(ie~mk) [9]; i ->, e~ -> (j~), k -> m
    X["aba"]["ovoo"] = (
        -0.5 * np.einsum("mnfe,infk->iemk", H.ab.oovv, R.aba, optimize=True)
        -np.einsum("mnke,in->iemk", H.ab.ooov, R.ab, optimize=True)
    )
    # antisymmetrize (ik)
    X["aba"]["ovoo"] -= np.transpose(X["aba"]["ovoo"], (3, 1, 2, 0))

    # x(ej~m~k~) [10]; j~ ->, e -> (i), k~ -> m~
    X["abb"]["vooo"] = (
        -0.5 * np.einsum("nmef,njfk->ejmk", H.ab.oovv, R.abb, optimize=True)
        -np.einsum("nmek,nj->ejmk", H.ab.oovo, R.ab, optimize=True)
    )
    # antisymmetrize A(j~k~)
    X["abb"]["vooo"] -= np.transpose(X["abb"]["vooo"], (0, 3, 2, 1))

    # x(ckef~) [11]; c <-> k, e -> (i), f~ -> (j~)
    X["aba"]["vovv"] = (
        np.einsum("mnef,mnck->ckef", H.ab.oovv, R.aba, optimize=True)
        + np.einsum("cmef,km->ckef", H.ab.vovv, R.ab, optimize=True)
    )

    # x(c~k~e~f) [12]; c~ <-> k~, e~ -> (j~), f -> (i)
    X["abb"]["vovv"] = (
        np.einsum("nmfe,nmck->ckef", H.ab.oovv, R.abb, optimize=True)
        + np.einsum("ncfe,nk->ckef", H.ab.ovvv, R.ab, optimize=True)
    )

    # x(cfej~) [13]: c <-> f, e <-> j~
    X["aba"]["vvvo"] = (
        0.5 * np.einsum("mnef,mjcn->cfej", H.aa.oovv, R.aba, optimize=True)
        - np.einsum("cnfe,nj->cfej", H.aa.vovv, R.ab, optimize=True)
    )

    # x(c~f~e~i) [14]; c~ <-> f~, e~ <-> i
    X["abb"]["vvvo"] = (
        0.5 * np.einsum("mnef,imcn->cfei", H.bb.oovv, R.abb, optimize=True)
        - np.einsum("cnfe,in->cfei", H.bb.vovv, R.ab, optimize=True)
    )

    return X

def add_ov_intermediates(X, R, H):
    # These terms are required to contract with T3 in the 4h-2p updates in DIP-EOMCCSDT,
    # but they should not be included in the 3h-1p updates (since H1(ov)*R.ab is implicilty
    # included in H2(vooo)*R.ab already).
    # x(ie~)
    X["ab"]["ov"] -= np.einsum("me,im->ie", H.b.ov, R.ab, optimize=True)
    # x(ej~)
    X["ab"]["vo"] -= np.einsum("me,mj->ej", H.a.ov, R.ab, optimize=True)
    return X