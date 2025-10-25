import numpy as np
from ccpy.utilities.linear_algebra import ccpy_einsum

def get_cc4_intermediates(T, H0):
    """Calculate the CCS-like intermediates for CC4."""

    X = {"aa": {"vooo": 0.0, "vvov": 0.0, "vvvv": 0.0, "oooo": 0.0, "voov": 0.0},
         "ab": {"vooo": 0.0, "ovoo": 0.0, "vvov": 0.0, "vvvo": 0.0,
                "vvvv": 0.0, "oooo": 0.0,
                "voov": 0.0, "ovvo": 0.0, "ovov": 0.0, "vovo": 0.0},
         "bb": {"vooo": 0.0, "vvov": 0.0, "vvvv": 0.0, "oooo": 0.0, "voov": 0.0}}

    Q1 = -ccpy_einsum("mnfe,an->amef", H0.aa.oovv, T.a)
    I2A_vovv = H0.aa.vovv + 0.5 * Q1

    Q1 = ccpy_einsum("mnfe,fi->mnie", H0.aa.oovv, T.a)
    I2A_ooov = H0.aa.ooov + 0.5 * Q1

    Q1 = -ccpy_einsum("nmef,an->amef", H0.ab.oovv, T.a)
    I2B_vovv = H0.ab.vovv + 0.5 * Q1

    Q1 = ccpy_einsum("mnfe,fi->mnie", H0.ab.oovv, T.a)
    I2B_ooov = H0.ab.ooov + 0.5 * Q1

    Q1 = -ccpy_einsum("mnef,an->maef", H0.ab.oovv, T.b)
    I2B_ovvv = H0.ab.ovvv + 0.5 * Q1

    Q1 = ccpy_einsum("nmef,fi->nmei", H0.ab.oovv, T.b)
    I2B_oovo = H0.ab.oovo + 0.5 * Q1

    Q1 = -ccpy_einsum("nmef,an->amef", H0.bb.oovv, T.b)
    I2C_vovv = H0.bb.vovv + 0.5 * Q1

    Q1 = ccpy_einsum("mnfe,fi->mnie", H0.bb.oovv, T.b)
    I2C_ooov = H0.bb.ooov + 0.5 * Q1

    Q1 = -ccpy_einsum("bmfe,am->abef", I2A_vovv, T.a)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    X["aa"]["vvvv"] = H0.aa.vvvv + Q1

    X["ab"]["vvvv"] = H0.ab.vvvv + (
            - ccpy_einsum("mbef,am->abef", I2B_ovvv, T.a)
            - ccpy_einsum("amef,bm->abef", I2B_vovv, T.b)
    )

    Q1 = -ccpy_einsum("bmfe,am->abef", I2C_vovv, T.b)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    X["bb"]["vvvv"] = H0.bb.vvvv + Q1

    Q1 = +ccpy_einsum("nmje,ei->mnij", I2A_ooov, T.a)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    X["aa"]["oooo"] = H0.aa.oooo + Q1

    X["ab"]["oooo"] = H0.ab.oooo + (
            ccpy_einsum("mnej,ei->mnij", I2B_oovo, T.a)
            + ccpy_einsum("mnie,ej->mnij", I2B_ooov, T.b)
    )

    Q1 = +ccpy_einsum("nmje,ei->mnij", I2C_ooov, T.b)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    X["bb"]["oooo"] = H0.bb.oooo + Q1

    X["aa"]["voov"] = H0.aa.voov + (
            ccpy_einsum("amfe,fi->amie", I2A_vovv, T.a)
            - ccpy_einsum("nmie,an->amie", I2A_ooov, T.a)
    )

    X["ab"]["voov"] = H0.ab.voov + (
            ccpy_einsum("amfe,fi->amie", I2B_vovv, T.a)
            - ccpy_einsum("nmie,an->amie", I2B_ooov, T.a)
    )

    X["ab"]["ovvo"] = H0.ab.ovvo + (
            ccpy_einsum("maef,fi->maei", I2B_ovvv, T.b)
            - ccpy_einsum("mnei,an->maei", I2B_oovo, T.b)
    )

    X["ab"]["ovov"] = H0.ab.ovov + (
            ccpy_einsum("mafe,fi->maie", I2B_ovvv, T.a)
            - ccpy_einsum("mnie,an->maie", I2B_ooov, T.b)
    )

    X["ab"]["vovo"] = H0.ab.vovo + (
            - ccpy_einsum("nmei,an->amei", I2B_oovo, T.a)
            + ccpy_einsum("amef,fi->amei", I2B_vovv, T.b)
    )

    X["bb"]["voov"] = H0.bb.voov + (
            ccpy_einsum("amfe,fi->amie", I2C_vovv, T.b)
            - ccpy_einsum("nmie,an->amie", I2C_ooov, T.b)
    )

    Q1 = H0.aa.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.aa.vovv, T.a)
    Q1 = ccpy_einsum("amif,fj->amij", Q1, T.a)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    X["aa"]["vooo"] = H0.aa.vooo + Q1 - ccpy_einsum("nmij,an->amij", X["aa"]["oooo"], T.a)
    # added in for ROHF
    # X["aa"]["vooo"] += ccpy_einsum("me,aeij->amij", H0.a.ov, T.aa)

    Q1 = H0.ab.voov + ccpy_einsum("amfe,fi->amie", H0.ab.vovv, T.a)
    X["ab"]["vooo"] = H0.ab.vooo + (
            - ccpy_einsum("nmij,an->amij", X["ab"]["oooo"], T.a)
            + ccpy_einsum("amej,ei->amij", H0.ab.vovo, T.a)
            + ccpy_einsum("amie,ej->amij", Q1, T.b)
    )
    # added in for ROHF
    # X["ab"]["vooo"] += ccpy_einsum("me,aeik->amik", H0.b.ov, T.ab)

    Q1 = H0.ab.ovov + ccpy_einsum("mafe,fj->maje", H0.ab.ovvv, T.a)
    X["ab"]["ovoo"] = H0.ab.ovoo + (
            - ccpy_einsum("mnji,an->maji", X["ab"]["oooo"], T.b)
            + ccpy_einsum("maje,ei->maji", Q1, T.b)
            + ccpy_einsum("maei,ej->maji", H0.ab.ovvo, T.a)
    )
    # added in for ROHF
    # X["ab"]["ovoo"] += ccpy_einsum("me,ecjk->mcjk", H0.a.ov, T.ab)

    Q1 = H0.bb.voov + 0.5 * ccpy_einsum("amef,ei->amif", H0.bb.vovv, T.b)
    Q1 = ccpy_einsum("amif,fj->amij", Q1, T.b)
    Q1 -= np.transpose(Q1, (0, 1, 3, 2))
    X["bb"]["vooo"] = H0.bb.vooo + Q1 - ccpy_einsum("nmij,an->amij", X["bb"]["oooo"], T.b)
    # added in for ROHF
    # X["bb"]["vooo"] += ccpy_einsum("me,aeij->amij", H0.b.ov, T.bb)

    Q1 = H0.aa.ovov - 0.5 * ccpy_einsum("mnie,bn->mbie", H0.aa.ooov, T.a)
    Q1 = -ccpy_einsum("mbie,am->abie", Q1, T.a)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    X["aa"]["vvov"] = H0.aa.vvov + Q1 + ccpy_einsum("abfe,fi->abie", X["aa"]["vvvv"], T.a)

    Q1 = H0.ab.ovov - ccpy_einsum("mnie,bn->mbie", H0.ab.ooov, T.b)
    Q1 = -ccpy_einsum("mbie,am->abie", Q1, T.a)
    X["ab"]["vvov"] = H0.ab.vvov + Q1 + (
            + ccpy_einsum("abfe,fi->abie", X["ab"]["vvvv"], T.a)
            - ccpy_einsum("amie,bm->abie", H0.ab.voov, T.b)
    )

    Q1 = H0.ab.vovo - ccpy_einsum("nmei,bn->bmei", H0.ab.oovo, T.a)
    Q1 = -ccpy_einsum("bmei,am->baei", Q1, T.b)
    X["ab"]["vvvo"] = H0.ab.vvvo + Q1 + (
            + ccpy_einsum("baef,fi->baei", X["ab"]["vvvv"], T.b)
            - ccpy_einsum("naei,bn->baei", H0.ab.ovvo, T.a)
    )

    Q1 = H0.bb.ovov - 0.5 * ccpy_einsum("mnie,bn->mbie", H0.bb.ooov, T.b)
    Q1 = -ccpy_einsum("mbie,am->abie", Q1, T.b)
    Q1 -= np.transpose(Q1, (1, 0, 2, 3))
    X["bb"]["vvov"] = H0.bb.vvov + Q1 + ccpy_einsum("abfe,fi->abie", X["bb"]["vvvv"], T.b)

    return X
