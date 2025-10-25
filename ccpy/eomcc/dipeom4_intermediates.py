import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

def get_dipeom4_intermediates(H, R):

    # Create dictionary to store intermediates, which have spincases that resemble those of the DIP R operator itself
    X = {"ab": {"vo": np.array([0.0]), "ov": np.array([0.0]), "vv": np.array([0.0])},
         "aba": {"oooo": np.array([0.0]), "oovv": np.array([0.0]), "ovvo": np.array([0.0])},
         "abb": {"oooo": np.array([0.0]), "oovv": np.array([0.0]), "vovo": np.array([0.0])}}

    ### one-body intermediates ###
    # x(ie~)
    X["ab"]["ov"] = (
            ccpy_einsum("mnie,mn->ie", H.ab.ooov, R.ab)
            - ccpy_einsum("nmfe,imfn->ie", H.ab.oovv, R.aba)
            - 0.5 * ccpy_einsum("nmfe,imfn->ie", H.bb.oovv, R.abb)
    )
    # x(ej~)
    X["ab"]["vo"] = (
            ccpy_einsum("mnej,mn->ej", H.ab.oovo, R.ab)
            - 0.5 * ccpy_einsum("mnef,mjfn->ej", H.aa.oovv, R.aba)
            - ccpy_einsum("mnef,mjfn->ej", H.ab.oovv, R.abb)
    )
    # x(ef~)
    X["ab"]["vv"] = ccpy_einsum("mnef,mn->ef", H.ab.oovv, R.ab)

    ### two-body intermediates ###
    # x(ij~ce) [1]
    X["aba"]["oovv"] = (
            ccpy_einsum("cnef,ijfn->ijce", H.aa.vovv, R.aba)
            + ccpy_einsum("cnef,ijfn->ijce", H.ab.vovv, R.abb)
            + ccpy_einsum("cmie,mj->ijce", H.aa.voov, R.ab) # flip sign, h2a(vovo) -> -h2a(voov)
            + ccpy_einsum("mnej,incm->ijce", H.ab.oovo, R.aba)
            + 0.5 * ccpy_einsum("mnie,mjcn->ijce", H.aa.ooov, R.aba)
            - ccpy_einsum("cmej,im->ijce", H.ab.vovo, R.ab)
            - 0.5 * ccpy_einsum("mnef,ijcfmn->ijce", H.aa.oovv, R.abaa)
            - ccpy_einsum("mnef,ijcfmn->ijce", H.ab.oovv, R.abab)
    )

    # x(ij~mk) [2]
    X["aba"]["oooo"] = (
            ccpy_einsum("mnkf,ijfn->ijmk", H.aa.ooov, R.aba)
            + ccpy_einsum("mnkf,ijfn->ijmk", H.ab.ooov, R.abb)
            - 0.5 * ccpy_einsum("mnej,inek->ijmk", H.ab.oovo, R.aba)
            - 0.5 * ccpy_einsum("nmik,nj->ijmk", H.aa.oooo, R.ab)
            - ccpy_einsum("mnkj,in->ijmk", H.ab.oooo, R.ab)
            + 0.25 * ccpy_einsum("mnef,ijefkn->ijmk", H.aa.oovv, R.abaa)
            + 0.5 * ccpy_einsum("mnef,ijefkn->ijmk", H.ab.oovv, R.abab)
    )
    # antisymmetrize A(ik)
    X["aba"]["oooo"] -= np.transpose(X["aba"]["oooo"], (3, 1, 2, 0))

    # x(ieck) [3]
    X["aba"]["ovvo"] = (
            ccpy_einsum("nmie,nmck->ieck", H.ab.ooov, R.aba)
            - 0.5 * ccpy_einsum("cmfe,imfk->ieck", H.ab.vovv, R.aba)
            - ccpy_einsum("cmke,im->ieck", H.ab.voov, R.ab)
            - 0.5 * ccpy_einsum("nmfe,imcfkn->ieck", H.ab.oovv, R.abaa)
            - 0.25 * ccpy_einsum("mnef,imcfkn->ieck", H.bb.oovv, R.abab)
    )
    # antisymmetrize A(ik)
    X["aba"]["ovvo"] -= np.transpose(X["aba"]["ovvo"], (3, 1, 2, 0))

    # x(ijde) [4]
    X["abb"]["oovv"] = (
            ccpy_einsum("ndfe,ijfn->ijde", H.ab.ovvv, R.aba)
            + ccpy_einsum("dnef,ijfn->ijde", H.bb.vovv, R.abb)
            + ccpy_einsum("dmje,im->ijde", H.bb.voov, R.ab) # flip sign, h2c(vovo) -> -h2c(voov)
            + 0.5 * ccpy_einsum("mnje,imdn->ijde", H.bb.ooov, R.abb)
            + ccpy_einsum("nmie,njdm->ijde", H.ab.ooov, R.abb)
            - ccpy_einsum("mdie,mj->ijde", H.ab.ovov, R.ab)
            - ccpy_einsum("nmfe,ijfdnm->ijde", H.ab.oovv, R.abab)
            - 0.5 * ccpy_einsum("mnef,ijfdnm->ijde", H.bb.oovv, R.abbb)
    )

    # x(ij~m~k~) [5]
    X["abb"]["oooo"] = (
            ccpy_einsum("nmfk,ijfn->ijmk", H.ab.oovo, R.aba)
            + ccpy_einsum("mnkf,ijfn->ijmk", H.bb.ooov, R.abb)
            - ccpy_einsum("nmik,nj->ijmk", H.ab.oooo, R.ab)
            - 0.5 * ccpy_einsum("nmie,njek->ijmk", H.ab.ooov, R.abb)
            - 0.5 * ccpy_einsum("nmjk,in->ijmk", H.bb.oooo, R.ab)
            + 0.5 * ccpy_einsum("nmfe,ijfenk->ijmk", H.ab.oovv, R.abab)
            + 0.25 * ccpy_einsum("mnef,ijfenk->ijmk", H.bb.oovv, R.abbb)
    )
    # antisymmetrize A(jk)
    X["abb"]["oooo"] -= np.transpose(X["abb"]["oooo"], (0, 3, 2, 1))

    # x(ejdl) [6]
    X["abb"]["vovo"] = (
            - 0.5 * ccpy_einsum("mdef,mjfl->ejdl", H.ab.ovvv, R.abb)
            + ccpy_einsum("mnej,mndl->ejdl", H.ab.oovo, R.abb)
            - ccpy_einsum("mdel,mj->ejdl", H.ab.ovvo, R.ab)
            - 0.25 * ccpy_einsum("mnef,mjfdnl->ejdl", H.aa.oovv, R.abab)
            - 0.5 * ccpy_einsum("mnef,mjfdnl->ejdl", H.ab.oovv, R.abbb)
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
            ccpy_einsum("mnie,mn->ie", H.ab.ooov, R.ab)
            - ccpy_einsum("nmfe,imfn->ie", H.ab.oovv, R.aba)
            - 0.5 * ccpy_einsum("nmfe,imfn->ie", H.bb.oovv, R.abb)
    )
    # x(ej~)
    X["ab"]["vo"] = (
            ccpy_einsum("mnej,mn->ej", H.ab.oovo, R.ab)
            - 0.5 * ccpy_einsum("mnef,mjfn->ej", H.aa.oovv, R.aba)
            - ccpy_einsum("mnef,mjfn->ej", H.ab.oovv, R.abb)
    )
    # x(ef~)
    X["ab"]["vv"] = ccpy_einsum("mnef,mn->ef", H.ab.oovv, R.ab)

    ### two-body intermediates ###
    # x(ij~ce) [1]
    X["aba"]["oovv"] = (
            ccpy_einsum("cnef,ijfn->ijce", H.aa.vovv, R.aba)
            + ccpy_einsum("cnef,ijfn->ijce", H.ab.vovv, R.abb)
            + ccpy_einsum("cmie,mj->ijce", H.aa.voov, R.ab) # flip sign, h2a(vovo) -> -h2a(voov)
            + ccpy_einsum("mnej,incm->ijce", H.ab.oovo, R.aba)
            + 0.5 * ccpy_einsum("mnie,mjcn->ijce", H.aa.ooov, R.aba)
            - ccpy_einsum("cmej,im->ijce", H.ab.vovo, R.ab)
            - 0.5 * ccpy_einsum("mnef,ijcfmn->ijce", H.aa.oovv, R.abaa)
            - ccpy_einsum("mnef,ijcfmn->ijce", H.ab.oovv, R.abab)
    )

    # x(ij~mk) [2]
    X["aba"]["oooo"] = (
            ccpy_einsum("mnkf,ijfn->ijmk", H.aa.ooov, R.aba)
            + ccpy_einsum("mnkf,ijfn->ijmk", H.ab.ooov, R.abb)
            - 0.5 * ccpy_einsum("mnej,inek->ijmk", H.ab.oovo, R.aba)
            - 0.5 * ccpy_einsum("nmik,nj->ijmk", H.aa.oooo, R.ab)
            - ccpy_einsum("mnkj,in->ijmk", H.ab.oooo, R.ab)
            + 0.25 * ccpy_einsum("mnef,ijefkn->ijmk", H.aa.oovv, R.abaa)
            + 0.5 * ccpy_einsum("mnef,ijefkn->ijmk", H.ab.oovv, R.abab)
    )
    # antisymmetrize A(ik)
    X["aba"]["oooo"] -= np.transpose(X["aba"]["oooo"], (3, 1, 2, 0))

    # x(ieck) [3]
    X["aba"]["ovvo"] = (
            ccpy_einsum("nmie,nmck->ieck", H.ab.ooov, R.aba)
            - 0.5 * ccpy_einsum("cmfe,imfk->ieck", H.ab.vovv, R.aba)
            - ccpy_einsum("cmke,im->ieck", H.ab.voov, R.ab)
            - 0.5 * ccpy_einsum("nmfe,imcfkn->ieck", H.ab.oovv, R.abaa)
            - 0.25 * ccpy_einsum("mnef,imcfkn->ieck", H.bb.oovv, R.abab)
    )
    # antisymmetrize A(ik)
    X["aba"]["ovvo"] -= np.transpose(X["aba"]["ovvo"], (3, 1, 2, 0))

    # x(ijde) [4]
    X["abb"]["oovv"] = (
            ccpy_einsum("ndfe,ijfn->ijde", H.ab.ovvv, R.aba)
            + ccpy_einsum("dnef,ijfn->ijde", H.bb.vovv, R.abb)
            + ccpy_einsum("dmje,im->ijde", H.bb.voov, R.ab) # flip sign, h2c(vovo) -> -h2c(voov)
            + 0.5 * ccpy_einsum("mnje,imdn->ijde", H.bb.ooov, R.abb)
            + ccpy_einsum("nmie,njdm->ijde", H.ab.ooov, R.abb)
            - ccpy_einsum("mdie,mj->ijde", H.ab.ovov, R.ab)
            - ccpy_einsum("nmfe,ijfdnm->ijde", H.ab.oovv, R.abab)
            - 0.5 * ccpy_einsum("mnef,ijfdnm->ijde", H.bb.oovv, R.abbb)
    )

    # x(ij~m~k~) [5]
    X["abb"]["oooo"] = (
            ccpy_einsum("nmfk,ijfn->ijmk", H.ab.oovo, R.aba)
            + ccpy_einsum("mnkf,ijfn->ijmk", H.bb.ooov, R.abb)
            - ccpy_einsum("nmik,nj->ijmk", H.ab.oooo, R.ab)
            - 0.5 * ccpy_einsum("nmie,njek->ijmk", H.ab.ooov, R.abb)
            - 0.5 * ccpy_einsum("nmjk,in->ijmk", H.bb.oooo, R.ab)
            + 0.5 * ccpy_einsum("nmfe,ijfenk->ijmk", H.ab.oovv, R.abab)
            + 0.25 * ccpy_einsum("mnef,ijfenk->ijmk", H.bb.oovv, R.abbb)
    )
    # antisymmetrize A(jk)
    X["abb"]["oooo"] -= np.transpose(X["abb"]["oooo"], (0, 3, 2, 1))

    # x(ejdl) [6]
    X["abb"]["vovo"] = (
            - 0.5 * ccpy_einsum("mdef,mjfl->ejdl", H.ab.ovvv, R.abb)
            + ccpy_einsum("mnej,mndl->ejdl", H.ab.oovo, R.abb)
            - ccpy_einsum("mdel,mj->ejdl", H.ab.ovvo, R.ab)
            - 0.25 * ccpy_einsum("mnef,mjfdnl->ejdl", H.aa.oovv, R.abab)
            - 0.5 * ccpy_einsum("mnef,mjfdnl->ejdl", H.ab.oovv, R.abbb)
    )
    # antisymmetrize A(jl)
    X["abb"]["vovo"] -= np.transpose(X["abb"]["vovo"], (0, 3, 2, 1))

    # Intermediates [7] - [14] are only needed in DIP-EOMCCSDT to contract with T3

    # x(ij~em) [7]
    X["aba"]["oovo"] = (
        -ccpy_einsum("mnej,in->ijem", H.ab.oovo, R.ab)
        -ccpy_einsum("nmie,nj->ijem", H.aa.ooov, R.ab)
        +ccpy_einsum("mnef,ijfn->ijem", H.aa.oovv, R.aba)
        +ccpy_einsum("mnef,ijfn->ijem", H.ab.oovv, R.abb)
    )

    # x(ij~e~m~) [8]
    X["abb"]["oovo"] = (
        -ccpy_einsum("nmje,in->ijem", H.bb.ooov, R.ab)
        -ccpy_einsum("nmie,nj->ijem", H.ab.ooov, R.ab)
        +ccpy_einsum("nmfe,ijfn->ijem", H.ab.oovv, R.aba)
        +ccpy_einsum("mnef,ijfn->ijem", H.bb.oovv, R.abb)
    )

    # x(ie~mk) [9]; i ->, e~ -> (j~), k -> m
    X["aba"]["ovoo"] = (
        -0.5 * ccpy_einsum("mnfe,infk->iemk", H.ab.oovv, R.aba)
        -ccpy_einsum("mnke,in->iemk", H.ab.ooov, R.ab)
    )
    # antisymmetrize (ik)
    X["aba"]["ovoo"] -= np.transpose(X["aba"]["ovoo"], (3, 1, 2, 0))

    # x(ej~m~k~) [10]; j~ ->, e -> (i), k~ -> m~
    X["abb"]["vooo"] = (
        -0.5 * ccpy_einsum("nmef,njfk->ejmk", H.ab.oovv, R.abb)
        -ccpy_einsum("nmek,nj->ejmk", H.ab.oovo, R.ab)
    )
    # antisymmetrize A(j~k~)
    X["abb"]["vooo"] -= np.transpose(X["abb"]["vooo"], (0, 3, 2, 1))

    # x(ckef~) [11]; c <-> k, e -> (i), f~ -> (j~)
    X["aba"]["vovv"] = (
        ccpy_einsum("mnef,mnck->ckef", H.ab.oovv, R.aba)
        + ccpy_einsum("cmef,km->ckef", H.ab.vovv, R.ab)
    )

    # x(c~k~e~f) [12]; c~ <-> k~, e~ -> (j~), f -> (i)
    X["abb"]["vovv"] = (
        ccpy_einsum("nmfe,nmck->ckef", H.ab.oovv, R.abb)
        + ccpy_einsum("ncfe,nk->ckef", H.ab.ovvv, R.ab)
    )

    # x(cfej~) [13]: c <-> f, e <-> j~
    X["aba"]["vvvo"] = (
        0.5 * ccpy_einsum("mnef,mjcn->cfej", H.aa.oovv, R.aba)
        - ccpy_einsum("cnfe,nj->cfej", H.aa.vovv, R.ab)
    )

    # x(c~f~e~i) [14]; c~ <-> f~, e~ <-> i
    X["abb"]["vvvo"] = (
        0.5 * ccpy_einsum("mnef,imcn->cfei", H.bb.oovv, R.abb)
        - ccpy_einsum("cnfe,in->cfei", H.bb.vovv, R.ab)
    )

    return X

def get_dipeomccsdta_intermediates(H, H0, R):

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
            ccpy_einsum("mnie,mn->ie", H.ab.ooov, R.ab)
            - ccpy_einsum("nmfe,imfn->ie", H.ab.oovv, R.aba)
            - 0.5 * ccpy_einsum("nmfe,imfn->ie", H.bb.oovv, R.abb)
    )
    # x(ej~)
    X["ab"]["vo"] = (
            ccpy_einsum("mnej,mn->ej", H.ab.oovo, R.ab)
            - 0.5 * ccpy_einsum("mnef,mjfn->ej", H.aa.oovv, R.aba)
            - ccpy_einsum("mnef,mjfn->ej", H.ab.oovv, R.abb)
    )
    # x(ef~)
    X["ab"]["vv"] = ccpy_einsum("mnef,mn->ef", H.ab.oovv, R.ab)

    ### two-body intermediates ###
    # x(ij~ce) [1]
    X["aba"]["oovv"] = (
            ccpy_einsum("cnef,ijfn->ijce", H.aa.vovv, R.aba)
            + ccpy_einsum("cnef,ijfn->ijce", H.ab.vovv, R.abb)
            + ccpy_einsum("cmie,mj->ijce", H.aa.voov, R.ab) # flip sign, h2a(vovo) -> -h2a(voov)
            + ccpy_einsum("mnej,incm->ijce", H.ab.oovo, R.aba)
            + 0.5 * ccpy_einsum("mnie,mjcn->ijce", H.aa.ooov, R.aba)
            - ccpy_einsum("cmej,im->ijce", H.ab.vovo, R.ab)
            - 0.5 * ccpy_einsum("mnef,ijcfmn->ijce", H.aa.oovv, R.abaa)
            - ccpy_einsum("mnef,ijcfmn->ijce", H.ab.oovv, R.abab)
    )

    # x(ij~mk) [2]
    X["aba"]["oooo"] = (
            ccpy_einsum("mnkf,ijfn->ijmk", H.aa.ooov, R.aba)
            + ccpy_einsum("mnkf,ijfn->ijmk", H.ab.ooov, R.abb)
            - 0.5 * ccpy_einsum("mnej,inek->ijmk", H.ab.oovo, R.aba)
            - 0.5 * ccpy_einsum("nmik,nj->ijmk", H.aa.oooo, R.ab)
            - ccpy_einsum("mnkj,in->ijmk", H.ab.oooo, R.ab)
            + 0.25 * ccpy_einsum("mnef,ijefkn->ijmk", H.aa.oovv, R.abaa)
            + 0.5 * ccpy_einsum("mnef,ijefkn->ijmk", H.ab.oovv, R.abab)
    )
    # antisymmetrize A(ik)
    X["aba"]["oooo"] -= np.transpose(X["aba"]["oooo"], (3, 1, 2, 0))

    # x(ieck) [3]
    X["aba"]["ovvo"] = (
            ccpy_einsum("nmie,nmck->ieck", H.ab.ooov, R.aba)
            - 0.5 * ccpy_einsum("cmfe,imfk->ieck", H.ab.vovv, R.aba)
            - ccpy_einsum("cmke,im->ieck", H.ab.voov, R.ab)
            - 0.5 * ccpy_einsum("nmfe,imcfkn->ieck", H.ab.oovv, R.abaa)
            - 0.25 * ccpy_einsum("mnef,imcfkn->ieck", H.bb.oovv, R.abab)
    )
    # antisymmetrize A(ik)
    X["aba"]["ovvo"] -= np.transpose(X["aba"]["ovvo"], (3, 1, 2, 0))

    # x(ijde) [4]
    X["abb"]["oovv"] = (
            ccpy_einsum("ndfe,ijfn->ijde", H.ab.ovvv, R.aba)
            + ccpy_einsum("dnef,ijfn->ijde", H.bb.vovv, R.abb)
            + ccpy_einsum("dmje,im->ijde", H.bb.voov, R.ab) # flip sign, h2c(vovo) -> -h2c(voov)
            + 0.5 * ccpy_einsum("mnje,imdn->ijde", H.bb.ooov, R.abb)
            + ccpy_einsum("nmie,njdm->ijde", H.ab.ooov, R.abb)
            - ccpy_einsum("mdie,mj->ijde", H.ab.ovov, R.ab)
            - ccpy_einsum("nmfe,ijfdnm->ijde", H.ab.oovv, R.abab)
            - 0.5 * ccpy_einsum("mnef,ijfdnm->ijde", H.bb.oovv, R.abbb)
    )

    # x(ij~m~k~) [5]
    X["abb"]["oooo"] = (
            ccpy_einsum("nmfk,ijfn->ijmk", H.ab.oovo, R.aba)
            + ccpy_einsum("mnkf,ijfn->ijmk", H.bb.ooov, R.abb)
            - ccpy_einsum("nmik,nj->ijmk", H.ab.oooo, R.ab)
            - 0.5 * ccpy_einsum("nmie,njek->ijmk", H.ab.ooov, R.abb)
            - 0.5 * ccpy_einsum("nmjk,in->ijmk", H.bb.oooo, R.ab)
            + 0.5 * ccpy_einsum("nmfe,ijfenk->ijmk", H.ab.oovv, R.abab)
            + 0.25 * ccpy_einsum("mnef,ijfenk->ijmk", H.bb.oovv, R.abbb)
    )
    # antisymmetrize A(jk)
    X["abb"]["oooo"] -= np.transpose(X["abb"]["oooo"], (0, 3, 2, 1))

    # x(ejdl) [6]
    X["abb"]["vovo"] = (
            - 0.5 * ccpy_einsum("mdef,mjfl->ejdl", H.ab.ovvv, R.abb)
            + ccpy_einsum("mnej,mndl->ejdl", H.ab.oovo, R.abb)
            - ccpy_einsum("mdel,mj->ejdl", H.ab.ovvo, R.ab)
            - 0.25 * ccpy_einsum("mnef,mjfdnl->ejdl", H.aa.oovv, R.abab)
            - 0.5 * ccpy_einsum("mnef,mjfdnl->ejdl", H.ab.oovv, R.abbb)
    )
    # antisymmetrize A(jl)
    X["abb"]["vovo"] -= np.transpose(X["abb"]["vovo"], (0, 3, 2, 1))

    # Intermediates [7] - [14] are only needed in DIP-EOMCCSDT to contract with T3

    # x(ij~em) [7]
    X["aba"]["oovo"] = (
        -ccpy_einsum("mnej,in->ijem", H0["ab"]["oovo"], R.ab)
        -ccpy_einsum("nmie,nj->ijem", H0["aa"]["ooov"], R.ab)
        +ccpy_einsum("mnef,ijfn->ijem", H.aa.oovv, R.aba)
        +ccpy_einsum("mnef,ijfn->ijem", H.ab.oovv, R.abb)
    )

    # x(ij~e~m~) [8]
    X["abb"]["oovo"] = (
        -ccpy_einsum("nmje,in->ijem", H0["bb"]["ooov"], R.ab)
        -ccpy_einsum("nmie,nj->ijem", H0["ab"]["ooov"], R.ab)
        +ccpy_einsum("nmfe,ijfn->ijem", H.ab.oovv, R.aba)
        +ccpy_einsum("mnef,ijfn->ijem", H.bb.oovv, R.abb)
    )

    # x(ie~mk) [9]; i ->, e~ -> (j~), k -> m
    X["aba"]["ovoo"] = (
        -0.5 * ccpy_einsum("mnfe,infk->iemk", H.ab.oovv, R.aba)
        -ccpy_einsum("mnke,in->iemk", H0["ab"]["ooov"], R.ab)
    )
    # antisymmetrize (ik)
    X["aba"]["ovoo"] -= np.transpose(X["aba"]["ovoo"], (3, 1, 2, 0))

    # x(ej~m~k~) [10]; j~ ->, e -> (i), k~ -> m~
    X["abb"]["vooo"] = (
        -0.5 * ccpy_einsum("nmef,njfk->ejmk", H.ab.oovv, R.abb)
        -ccpy_einsum("nmek,nj->ejmk", H0["ab"]["oovo"], R.ab)
    )
    # antisymmetrize A(j~k~)
    X["abb"]["vooo"] -= np.transpose(X["abb"]["vooo"], (0, 3, 2, 1))

    # x(ckef~) [11]; c <-> k, e -> (i), f~ -> (j~)
    X["aba"]["vovv"] = (
        ccpy_einsum("mnef,mnck->ckef", H.ab.oovv, R.aba)
        + ccpy_einsum("cmef,km->ckef", H0["ab"]["vovv"], R.ab)
    )

    # x(c~k~e~f) [12]; c~ <-> k~, e~ -> (j~), f -> (i)
    X["abb"]["vovv"] = (
        ccpy_einsum("nmfe,nmck->ckef", H.ab.oovv, R.abb)
        + ccpy_einsum("ncfe,nk->ckef", H0["ab"]["ovvv"], R.ab)
    )

    # x(cfej~) [13]: c <-> f, e <-> j~
    X["aba"]["vvvo"] = (
        0.5 * ccpy_einsum("mnef,mjcn->cfej", H.aa.oovv, R.aba)
        - ccpy_einsum("cnfe,nj->cfej", H0["aa"]["vovv"], R.ab)
    )

    # x(c~f~e~i) [14]; c~ <-> f~, e~ <-> i
    X["abb"]["vvvo"] = (
        0.5 * ccpy_einsum("mnef,imcn->cfei", H.bb.oovv, R.abb)
        - ccpy_einsum("cnfe,in->cfei", H0["bb"]["vovv"], R.ab)
    )

    return X

def add_ov_intermediates(X, R, H):
    # These terms are required to contract with T3 in the 4h-2p updates in DIP-EOMCCSDT,
    # but they should not be included in the 3h-1p updates (since H1(ov)*R.ab is implicilty
    # included in H2(vooo)*R.ab already).
    # x(ie~)
    X["ab"]["ov"] -= ccpy_einsum("me,im->ie", H.b.ov, R.ab)
    # x(ej~)
    X["ab"]["vo"] -= ccpy_einsum("me,mj->ej", H.a.ov, R.ab)
    return X
